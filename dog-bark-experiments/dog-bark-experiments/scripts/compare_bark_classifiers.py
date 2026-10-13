"""Compare two dog/non-dog student models on a fixed, configurable audio sample.

The selected file manifest is saved and reused. Since the supplied directory
contains dog recordings only, the main metric is dog recall (positive rate),
not full two-class accuracy. Adding a non-dog directory with --negative-dir
also enables accuracy, precision, specificity, and F1 reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn

SR = 16000
CLIP_SAMPLES = 2 * SR
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) > CLIP_SAMPLES:
        audio = audio[:CLIP_SAMPLES]
    elif len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    return audio.astype(np.float32)


def make_features(audio: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=20, fmax=SR // 2, power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)


class DSConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyBarkNet(nn.Module):
    def __init__(self, num_classes: int, width_multiplier: float = 1.0):
        super().__init__()
        def channels(value: int) -> int:
            return max(8, int(round(value * width_multiplier / 8.0)) * 8)
        c1, c2, c3, c4, c5 = [channels(v) for v in (32, 64, 128, 192, 256)]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU6(inplace=True)
        )
        self.blocks = nn.Sequential(
            DSConvBlock(c1, c2, 2), DSConvBlock(c2, c3, 2),
            DSConvBlock(c3, c4, 2), DSConvBlock(c4, c5, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(c5, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.classifier(self.dropout(self.pool(x).flatten(1)))


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def choose_manifest(data_dir, manifest: Path, count: int, seed: int):
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        if len(files) != count:
            raise ValueError(f"现有清单包含 {len(files)} 条，不是要求的 {count} 条: {manifest}")
        print(f"Reusing fixed manifest: {manifest}")
        return files

    roots = [Path(data_dir)] if isinstance(data_dir, str) else [Path(item) for item in data_dir]
    candidates = sorted(
        str(p.resolve()) for root in roots for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    valid = []
    for index, path in enumerate(candidates, 1):
        try:
            load_audio(path)
            valid.append(path)
        except Exception:
            pass
        if index % 1000 == 0:
            print(f"checking audio: {index}/{len(candidates)}, valid={len(valid)}")
    if len(valid) < count:
        raise ValueError(f"可解码音频只有 {len(valid)} 条，少于要求的 {count} 条")
    rng = random.Random(seed)
    selected = sorted(rng.sample(valid, count))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "seed": seed, "count": count,
        "source_dir": [str(root.resolve()) for root in roots],
        "files": selected,
        "file_sha256_16": {path: file_digest(path) for path in selected},
    }, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Created fixed manifest: {manifest}")
    return selected


def load_model(path: str, device: torch.device):
    # The INT8 checkpoints contain the complete quantized TinyBarkNet object.
    # These class definitions above make old __main__ pickles loadable.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    # Dynamic INT8 Linear layers are CPU operators; keep the whole tiny model
    # on CPU for reliable loading and comparable timing.
    return model.to(device).eval()


def evaluate(model, files: List[str], device: torch.device, name: str, negative_files: Optional[List[str]] = None):
    rows = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for path in files:
            try:
                features = make_features(load_audio(path)).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(features), dim=1)[0]
                predicted = int(probabilities.argmax().item())
                rows.append({
                    "path": path, "true_label": 0, "predicted": predicted,
                    "dog_probability": float(probabilities[0].item()), "ok": True,
                })
            except Exception as error:
                rows.append({"path": path, "true_label": 0, "predicted": -1, "dog_probability": 0.0,
                             "ok": False, "error": f"{type(error).__name__}: {error}"})
    elapsed = time.perf_counter() - start_time
    valid = [row for row in rows if row["ok"]]
    dog_recall = sum(row["predicted"] == 0 for row in valid) / max(len(valid), 1)
    result = {
        "model": name, "files": len(files), "valid": len(valid), "decode_failed": len(rows) - len(valid),
        "dog_recall": dog_recall, "mean_dog_probability": float(np.mean([r["dog_probability"] for r in valid])) if valid else 0.0,
        "seconds": elapsed, "milliseconds_per_file": elapsed * 1000 / max(len(valid), 1),
    }
    if negative_files:
        negative_rows = []
        with torch.inference_mode():
            for path in negative_files:
                try:
                    features = make_features(load_audio(path)).unsqueeze(0).to(device)
                    probabilities = torch.softmax(model(features), dim=1)[0]
                    negative_rows.append(int(probabilities.argmax().item()) == 1)
                except Exception:
                    pass
        result["negative_count"] = len(negative_rows)
        result["specificity"] = sum(negative_rows) / max(len(negative_rows), 1)
        true_positive = sum(row["predicted"] == 0 for row in valid)
        true_negative = sum(negative_rows)
        false_positive = len(negative_rows) - true_negative
        result["accuracy"] = (true_positive + true_negative) / max(len(valid) + len(negative_rows), 1)
        result["dog_precision"] = true_positive / max(true_positive + false_positive, 1)
        result["dog_f1"] = 2.0 * result["dog_recall"] * result["dog_precision"] / max(
            result["dog_recall"] + result["dog_precision"], 1e-12
        )
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog-dir", required=True)
    parser.add_argument("--old-model", required=True)
    parser.add_argument("--new-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--negative-dir", nargs="+", default=[],
                        help="一个或多个非狗叫目录；每个目录列表合并后固定抽取 count 条")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20250822)
    args = parser.parse_args()
    # The exported checkpoints use dynamic INT8 Linear layers, which are CPU
    # operators. Evaluation is intentionally CPU-based and reproducible.
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest) if args.manifest else output_dir / f"dog_test_{args.count}_seed{args.seed}.json"
    files = choose_manifest(args.dog_dir, manifest, args.count, args.seed)
    negative_files = None
    negative_manifest = None
    if args.negative_dir:
        negative_manifest = output_dir / f"non_dog_test_{args.count}_seed{args.seed}.json"
        negative_files = choose_manifest(args.negative_dir, negative_manifest, args.count, args.seed + 1)
    summary, details = [], {}
    for name, path in (("old", args.old_model), ("new", args.new_model)):
        model = load_model(path, device)
        result, rows = evaluate(model, files, device, name, negative_files)
        summary.append(result)
        details[name] = rows
        print(json.dumps(result, ensure_ascii=False, indent=2))
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "comparison_details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "old_predicted", "old_dog_probability", "new_predicted", "new_dog_probability"])
        for index, path in enumerate(files):
            old_row, new_row = details["old"][index], details["new"][index]
            writer.writerow([path, old_row["predicted"], old_row["dog_probability"], new_row["predicted"], new_row["dog_probability"]])
    print(f"manifest: {manifest}")
    if negative_manifest:
        print(f"negative manifest: {negative_manifest}")
    print(f"summary: {output_dir / 'comparison_summary.json'}")
    print(f"details: {output_dir / 'comparison_details.csv'}")


if __name__ == "__main__":
    main()
