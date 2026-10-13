"""Train a small dog-bark classifier with optional Wav2Vec2 distillation.

Expected data layout:
    data_dir/
        class_a/*.wav
        class_b/*.wav

The student uses log-mel features and depthwise-separable convolutions.  It is
small enough that the FP32 checkpoint is normally below 5 MB; the optional
dynamic INT8 export is smaller still.
"""

from __future__ import annotations

import project_paths as paths

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SR = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clip(path: str, train: bool) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) > CLIP_SAMPLES:
        start = np.random.randint(0, len(audio) - CLIP_SAMPLES + 1) if train else 0
        audio = audio[start : start + CLIP_SAMPLES]
    elif len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))

    if train:
        if np.random.rand() < 0.7:
            audio *= 10.0 ** (np.random.uniform(-10.0, 6.0) / 20.0)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-int(0.08 * SR), int(0.08 * SR) + 1)
            audio = np.roll(audio, shift)
        if np.random.rand() < 0.7:
            signal_power = max(float(np.mean(audio * audio)), 1e-8)
            snr_db = np.random.uniform(8.0, 25.0)
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak * 0.99
    return audio.astype(np.float32)


def mix_real_background(audio: np.ndarray, noise_files: List[str]) -> np.ndarray:
    """Mix a non-dog recording as a low/high SNR background when available."""
    if not noise_files or np.random.rand() >= 0.55:
        return audio
    try:
        noise = load_clip(random.choice(noise_files), train=False)
        signal_rms = np.sqrt(np.mean(audio * audio) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise * noise) + 1e-8)
        if noise_rms < 1e-6 or signal_rms < 1e-6:
            return audio
        snr_db = np.random.uniform(0.0, 18.0)
        noise = noise * (signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms))
        mixed = audio + noise
        peak = np.max(np.abs(mixed))
        return (mixed / peak * 0.99).astype(np.float32) if peak > 1.0 else mixed.astype(np.float32)
    except Exception:
        return audio


def make_features(audio: np.ndarray, train: bool = False) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SR // 2,
        power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if train:
        # SpecAugment helps the student ignore narrow-band and intermittent noise.
        if np.random.rand() < 0.5:
            width = np.random.randint(2, 9)
            start = np.random.randint(0, max(1, mel.shape[0] - width + 1))
            mel[start : start + width, :] = 0.0
        if np.random.rand() < 0.5:
            width = np.random.randint(3, 16)
            start = np.random.randint(0, max(1, mel.shape[1] - width + 1))
            mel[:, start : start + width] = 0.0
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)


class BarkDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], train: bool, noise_files: Optional[List[str]] = None):
        self.files = files
        self.labels = labels
        self.train = train
        self.noise_files = noise_files or []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        try:
            audio = load_clip(self.files[index], self.train)
            if self.train:
                audio = mix_real_background(audio, self.noise_files)
            return make_features(audio, self.train), torch.from_numpy(audio), torch.tensor(self.labels[index], dtype=torch.long)
        except Exception:
            # A few source files may be truncated or have a wrong extension.
            # Returning None lets the collate function skip them without killing a worker.
            return None


def skip_bad_audio_collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    features, waveforms, labels = zip(*valid)
    return torch.stack(features), torch.stack(waveforms), torch.stack(labels)


class DSConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyBarkNet(nn.Module):
    def __init__(self, num_classes: int, width_multiplier: float = 1.0):
        super().__init__()
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive")
        def channels(value: int) -> int:
            return max(8, int(round(value * width_multiplier / 8.0)) * 8)

        c1, c2, c3, c4, c5 = [channels(value) for value in (32, 64, 128, 192, 256)]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DSConvBlock(c1, c2, 2),
            DSConvBlock(c2, c3, 2),
            DSConvBlock(c3, c4, 2),
            DSConvBlock(c4, c5, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(c5, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


class TeacherWav2Vec2(nn.Module):
    """Wav2Vec2 teacher with the checkpoint-compatible classifier head."""

    def __init__(self, model_name: str, checkpoint: str, num_classes: int, device: torch.device):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.device = device
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        checkpoint_data = load_legacy_teacher_checkpoint(checkpoint)
        self.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values).last_hidden_state
        return self.classifier(hidden.mean(dim=1))


def load_legacy_teacher_checkpoint(checkpoint: str):
    """Load old checkpoints that saved a transformers processor alongside weights.

    The old processor pickle references transformers.tokenization_utils.Trie,
    which recent transformers releases no longer expose.  The processor is not
    used here; this compatibility placeholder only lets us read model_state_dict.
    """
    from transformers import tokenization_utils

    if not hasattr(tokenization_utils, "Trie"):
        class LegacyTrie:
            pass
        LegacyTrie.__name__ = "Trie"
        LegacyTrie.__qualname__ = "Trie"
        LegacyTrie.__module__ = "transformers.tokenization_utils"
        tokenization_utils.Trie = LegacyTrie
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def collect_files(data_dir: str, positive_name: str = "dog", negative_names: str = "") -> Tuple[List[str], List[int], List[str]]:
    root = Path(data_dir)
    directories = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    if positive_name.lower() not in directories:
        raise ValueError(f"找不到正类目录: {root / positive_name}")
    positive_dir = directories[positive_name.lower()]

    # 猫叫被明确排除。若未指定负类目录，则合并除 dog/cat 外的所有目录。
    if negative_names.strip():
        requested = [x.strip().lower() for x in negative_names.split(",") if x.strip()]
        negative_dirs = [directories[x] for x in requested if x in directories and x != positive_name.lower()]
        if not negative_dirs:
            raise ValueError(f"找不到负类目录，已请求: {requested}")
    else:
        negative_dirs = [
            path for name, path in directories.items()
            if name not in {positive_name.lower(), "cat", "猫", "cats", "猫叫"}
        ]
    if not negative_dirs:
        raise ValueError("没有找到非狗叫目录。请用 --negative-names 指定，例如 non_dog")

    class_names = ["dog", "non_dog"]
    files, labels = [], []
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    for path in sorted(positive_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(str(path))
            labels.append(0)
    for negative_dir in negative_dirs:
        for path in sorted(negative_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(str(path))
                labels.append(1)
    if len(class_names) < 2 or not files:
        raise ValueError("data_dir must contain at least two non-empty class folders")
    print(f"dog files: {labels.count(0)}, non-dog files: {labels.count(1)}")
    print(f"ignored directories: cat/猫 and any folders not selected as negative")
    return files, labels, class_names


def filter_unreadable_audio(files: List[str], labels: List[int], report_file: Path):
    """Decode every source once and permanently report files that cannot be used."""
    valid_files, valid_labels, failures = [], [], []
    total = len(files)
    for index, (path, label) in enumerate(zip(files, labels), start=1):
        try:
            load_clip(path, train=False)
            valid_files.append(path)
            valid_labels.append(label)
        except Exception as error:
            failures.append(f"{path}\t{type(error).__name__}: {error}")
        if index % 1000 == 0 or index == total:
            print(f"audio check: {index}/{total}, unreadable={len(failures)}")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"audio check complete: valid={len(valid_files)}, unreadable={len(failures)}")
    print(f"unreadable-file report: {report_file}")
    return valid_files, valid_labels


def make_splits(files, labels, seed: int, split_file: Path):
    """Create a 80/10/10 split once, then reuse its exact file list forever."""
    if split_file.exists():
        saved = json.loads(split_file.read_text(encoding="utf-8"))
        expected = {str(Path(path).resolve()): int(label) for path, label in zip(files, labels)}
        splits = {}
        for split_name in ("train", "val", "test"):
            entries = saved.get(split_name, [])
            if not entries:
                raise ValueError(f"固定划分文件缺少 {split_name}: {split_file}")
            kept_entries = []
            skipped = 0
            for entry in entries:
                path = str(Path(entry["path"]).resolve())
                label = int(entry["label"])
                if expected.get(path) == label:
                    kept_entries.append({"path": path, "label": label})
                else:
                    skipped += 1
            if skipped:
                print(f"removed {skipped} unavailable/unreadable files from fixed {split_name} split")
                saved[split_name] = kept_entries
            split_files = [entry["path"] for entry in kept_entries]
            split_labels = [entry["label"] for entry in kept_entries]
            if not split_files:
                raise ValueError(f"固定划分 {split_name} 没有可用音频: {split_file}")
            splits[split_name] = (split_files, split_labels)
        split_file.write_text(json.dumps(saved, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Reusing fixed split: {split_file}")
    else:
        train_files, test_files, train_labels, test_labels = train_test_split(
            files, labels, test_size=0.1, random_state=seed, stratify=labels
        )
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels, test_size=0.111111, random_state=seed, stratify=train_labels
        )
        splits = {
            "train": (train_files, train_labels),
            "val": (val_files, val_labels),
            "test": (test_files, test_labels),
        }
        payload = {
            split_name: [
                {"path": str(Path(path).resolve()), "label": int(label)}
                for path, label in zip(split_files, split_labels)
            ]
            for split_name, (split_files, split_labels) in splits.items()
        }
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Created fixed 80/10/10 split: {split_file}")

    train_files, train_labels = splits["train"]
    val_files, val_labels = splits["val"]
    test_files, test_labels = splits["test"]
    print(f"split counts: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    return (
        BarkDataset(train_files, train_labels, True),
        BarkDataset(val_files, val_labels, False),
        BarkDataset(test_files, test_labels, False),
    )


def teacher_inputs(processor, waveforms: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = [waveform for waveform in waveforms.cpu().numpy()]
    values = processor(
        batch, sampling_rate=SR, return_tensors="pt", padding=True
    ).input_values
    return values.to(device)


def run_epoch(student, teacher, loader, optimizer, processor, device, class_weights=None, alpha=0.3, temperature=3.0, description="train"):
    training = optimizer is not None
    student.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for batch in tqdm(loader, desc=description, leave=False):
        if batch is None:
            continue
        features, waveforms, labels = batch
        features, waveforms, labels = features.to(device), waveforms.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(teacher_inputs(processor, waveforms, device)) if teacher else None
        student_logits = student(features)
        ce = F.cross_entropy(student_logits, labels, weight=class_weights)
        if teacher_logits is not None:
            kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature * temperature
            loss = (1.0 - alpha) * ce + alpha * kd
        else:
            loss = ce
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((student_logits.argmax(1) == labels).sum().item())
        count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def model_size_mb(model) -> float:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values()) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--positive-name", default="dog")
    parser.add_argument("--negative-names", default="", help="负类目录名，多个用逗号分隔；留空则自动合并除 dog/cat 外的目录")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--output-dir", default=str(paths.OUTPUTS_ROOT / "train_bark_classifier"))
    parser.add_argument("--split-file", default="", help="固定数据划分 JSON；首次运行创建，后续运行复用")
    parser.add_argument("--check-audio", action="store_true", help="逐个解码并从固定划分中剔除损坏音频；首次训练建议执行一次")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width-multiplier", type=float, default=1.0,
                        help="学生模型宽度；2.0 通常仍远小于 5 MB，适合进一步提升精度")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = Path(args.split_file) if args.split_file else output_dir / "fixed_split.json"
    files, labels, class_names = collect_files(args.data_dir, args.positive_name, args.negative_names)
    if args.check_audio:
        files, labels = filter_unreadable_audio(files, labels, output_dir / "unreadable_audio_files.txt")
    train_set, val_set, test_set = make_splits(files, labels, args.seed, split_file)
    # Use only training-set non-dog recordings as background noise, so validation
    # and test metrics remain clean and the fixed split is not contaminated.
    train_set.noise_files = [path for path, label in zip(train_set.files, train_set.labels) if label == 1]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=skip_bad_audio_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=skip_bad_audio_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=skip_bad_audio_collate)

    student = TinyBarkNet(len(class_names), width_multiplier=args.width_multiplier).to(device)
    counts = np.bincount(np.asarray(train_set.labels), minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    print(f"class weights: dog={class_weights[0].item():.3f}, non_dog={class_weights[1].item():.3f}")
    teacher = None
    processor = None
    if args.teacher_model and args.teacher_checkpoint:
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(args.teacher_model)
        teacher = TeacherWav2Vec2(args.teacher_model, args.teacher_checkpoint, len(class_names), device)
        print("Teacher loaded; distillation enabled")
    else:
        print("No teacher supplied; training with hard labels only")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = -1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            student, teacher, train_loader, optimizer, processor, device, class_weights,
            description=f"epoch {epoch + 1}/{args.epochs} train",
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                student, None, val_loader, None, None, device, class_weights,
                description=f"epoch {epoch + 1}/{args.epochs} val",
            )
        scheduler.step()
        print(f"epoch {epoch + 1:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": student.state_dict(), "class_names": class_names}, output_dir / "student_fp32.pth")

    checkpoint = torch.load(output_dir / "student_fp32.pth", map_location=device, weights_only=False)
    student.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        _, test_acc = run_epoch(student, None, test_loader, None, None, device, class_weights, description="test")
    print(f"test_acc={test_acc:.4f}; width_multiplier={args.width_multiplier}; fp32_state_size={model_size_mb(student):.2f} MB")

    quantized = torch.quantization.quantize_dynamic(student.cpu(), {nn.Linear}, dtype=torch.qint8)
    torch.save({"model": quantized, "class_names": class_names}, output_dir / "student_int8.pth")
    print(f"int8_file_size={(output_dir / 'student_int8.pth').stat().st_size / (1024 * 1024):.2f} MB")
    (output_dir / "config.json").write_text(json.dumps({
        "sample_rate": SR, "clip_seconds": CLIP_SECONDS, "n_mels": N_MELS,
        "n_fft": N_FFT, "hop_length": HOP_LENGTH, "class_names": class_names,
        "width_multiplier": args.width_multiplier,
    }, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
