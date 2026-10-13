"""Evaluate Qwen2.5-Omni dog-bark detection with configurable sample counts."""

from __future__ import annotations

import project_paths as paths

import argparse
import csv
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
import os

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


# Model, dataset, and output locations.
MODEL_PATH = Path(str(paths.QWEN_MODEL))
DOG_BARK_DIR = Path(
    str(paths.DOG_AUDIO)
)
ESC50_DIR = Path(
    str(paths.ESC50_AUDIO)
)
URBANSOUND8K_DIR = Path(
    str(paths.URBANSOUND8K_AUDIO)
)
OUTPUT_DIR = Path(str(paths.OUTPUTS_ROOT / "evaluate_bark_detection_qwen"))
RESULTS_CSV = OUTPUT_DIR / "results.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
FAILURE_TXT = OUTPUT_DIR / "failure.txt"

DOG_COUNT = 300
ESC_COUNT = 150
URBAN_COUNT = 150
NON_DOG_COUNT = ESC_COUNT + URBAN_COUNT
SEED = 20260819
MAX_NEW_TOKENS = 12
REQUIRED_ACCURACY = 0.85
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))

PROMPT = (
    "Classify the main sound in this audio. "
    "Reply with exactly one label: dog_bark or non_dog_bark."
)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def choose_files(files: list[Path], count: int, seed: int) -> list[Path]:
    if len(files) < count:
        raise ValueError(f"Need {count} files, found {len(files)}")
    # Deterministic spacing avoids changing the test set between runs.
    step = len(files) / count
    indices = [min(int(i * step), len(files) - 1) for i in range(count)]
    selected = [files[i] for i in indices]
    if len(set(selected)) != count:
        raise ValueError("Could not select distinct audio files")
    return selected


def build_test_set() -> list[dict]:
    dog_files = choose_files(audio_files(DOG_BARK_DIR), DOG_COUNT, SEED)
    esc_files = choose_files(audio_files(ESC50_DIR), ESC_COUNT, SEED + 1)
    urban_files = choose_files(audio_files(URBANSOUND8K_DIR), URBAN_COUNT, SEED + 2)

    rows = [
        {
            "sample_id": f"dog_{i:04d}",
            "audio_path": str(path),
            "label": "dog_bark",
            "source": "DogSpeak",
        }
        for i, path in enumerate(dog_files, 1)
    ]
    rows += [
        {
            "sample_id": f"non_dog_esc_{i:04d}",
            "audio_path": str(path),
            "label": "non_dog_bark",
            "source": "ESC-50",
        }
        for i, path in enumerate(esc_files, 1)
    ]
    rows += [
        {
            "sample_id": f"non_dog_urban_{i:04d}",
            "audio_path": str(path),
            "label": "non_dog_bark",
            "source": "UrbanSound8K",
        }
        for i, path in enumerate(urban_files, 1)
    ]
    return rows


def normalize(text: str) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def parse_prediction(response: str) -> str:
    text = normalize(response)
    if "non dog bark" in text or "not a dog" in text or "no dog" in text:
        return "non_dog_bark"
    if "dog bark" in text or "dog barking" in text or "dog" in text or "bark" in text:
        return "dog_bark"
    if any(term in text for term in ("狗叫", "犬吠", "吠叫", "狗", "犬", "汪")):
        return "dog_bark"
    return "non_dog_bark"


def move_inputs_to_model(inputs, model):
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(model.device)
            if value.is_floating_point():
                value = value.to(model.dtype)
            inputs[key] = value
    return inputs


def load_model():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-Omni inference")
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    kwargs = {
        "torch_dtype": torch.bfloat16,
        # Keep the whole model on one GPU. Qwen2.5-Omni generation currently
        # can fail when device_map="auto" splits thinker state across GPUs.
        "device_map": {"": device.index},
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), attn_implementation="flash_attention_2", **kwargs
        )
    except Exception as error:
        print(f"FlashAttention 2 unavailable; using SDPA: {error}")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), attn_implementation="sdpa", **kwargs
        )
    model.disable_talker()
    model.eval()
    return processor, model


def predict(processor, model, row: dict) -> dict:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=False
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    input_length = inputs["input_ids"].shape[1]
    inputs = move_inputs_to_model(inputs, model)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_audio=False,
            use_audio_in_video=False,
        )
    if isinstance(output, tuple):
        output = output[0]
    response = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    prediction = parse_prediction(response)
    return {
        **row,
        "response": response,
        "prediction": prediction,
        "correct": prediction == row["label"],
        "inference_seconds": round(time.perf_counter() - started, 4),
        "run_time": datetime.now().isoformat(timespec="seconds"),
    }


def summarize(results: list[dict]) -> dict:
    tp = sum(r["label"] == "dog_bark" and r["prediction"] == "dog_bark" for r in results)
    fn = sum(r["label"] == "dog_bark" and r["prediction"] == "non_dog_bark" for r in results)
    tn = sum(r["label"] == "non_dog_bark" and r["prediction"] == "non_dog_bark" for r in results)
    fp = sum(r["label"] == "non_dog_bark" and r["prediction"] == "dog_bark" for r in results)
    accuracy = (tp + tn) / len(results) if results else 0.0
    return {
        "total": len(results),
        "dog_bark": sum(r["label"] == "dog_bark" for r in results),
        "non_dog_bark": sum(r["label"] == "non_dog_bark" for r in results),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "accuracy": accuracy,
        "required_accuracy": REQUIRED_ACCURACY,
        "pass_85_percent": accuracy >= REQUIRED_ACCURACY,
    }



def positive_count(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni dog-bark detection with configurable sample counts."
    )
    parser.add_argument("--dog-dir", type=Path, default=paths.DOG_AUDIO)
    parser.add_argument("--esc50-dir", type=Path, default=paths.ESC50_AUDIO)
    parser.add_argument("--urban-dir", type=Path, default=paths.URBANSOUND8K_AUDIO)
    parser.add_argument("--model-path", type=Path, default=paths.QWEN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=paths.OUTPUTS_ROOT / "evaluate_bark_detection_qwen")
    parser.add_argument("--dog-count", type=positive_count, default=300)
    parser.add_argument("--esc-count", type=positive_count, default=150)
    parser.add_argument("--urban-count", type=positive_count, default=150)
    return parser.parse_args(argv)


def configure(args: argparse.Namespace) -> None:
    """Apply one command-line configuration before constructing the test set."""
    global MODEL_PATH, DOG_BARK_DIR, ESC50_DIR, URBANSOUND8K_DIR
    global OUTPUT_DIR, RESULTS_CSV, SUMMARY_JSON, FAILURE_TXT
    global DOG_COUNT, ESC_COUNT, URBAN_COUNT
    global NON_DOG_COUNT
    MODEL_PATH = args.model_path
    DOG_BARK_DIR = args.dog_dir
    ESC50_DIR = args.esc50_dir
    URBANSOUND8K_DIR = args.urban_dir
    OUTPUT_DIR = args.output_dir
    RESULTS_CSV = OUTPUT_DIR / "results.csv"
    SUMMARY_JSON = OUTPUT_DIR / "summary.json"
    FAILURE_TXT = OUTPUT_DIR / "failure.txt"
    DOG_COUNT = args.dog_count
    ESC_COUNT = args.esc_count
    URBAN_COUNT = args.urban_count
    NON_DOG_COUNT = ESC_COUNT + URBAN_COUNT


def main(argv: list[str] | None = None):
    configure(parse_args(argv))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_set = build_test_set()
    expected_total = DOG_COUNT + NON_DOG_COUNT
    if len(test_set) != expected_total:
        raise ValueError(f"Expected {expected_total} test files, got {len(test_set)}")
    processor, model = load_model()
    results = []
    for index, row in enumerate(test_set, 1):
        result = predict(processor, model, row)
        results.append(result)
        print(
            f"[{index:03d}/{len(test_set)}] "
            f"{row['sample_id']} gt={row['label']} "
            f"pred={result['prediction']} response={result['response']!r}"
        )

    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    summary = summarize(results)
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "PASS: accuracy >= 85%" if summary["pass_85_percent"]
        else "FAIL: accuracy < 85%"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FAILURE_TXT.write_text(traceback.format_exc(), encoding="utf-8")
        raise
