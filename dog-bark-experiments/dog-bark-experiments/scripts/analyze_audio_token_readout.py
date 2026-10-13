"""Audio-token concept readout for Qwen2.5-Omni (paper stage 4B-02).

This script keeps the 4A lens frozen.  It first verifies, for every audio
sample, that the audio tower output is exactly the block of embeddings that
the thinker text model receives at ``audio_token_id`` positions.  Only after
that check does it read sound/animal/dog/bark and competing concepts from the
pre-fitted J-lens at the source layers used by 4A. Each position is ranked
against the vocabulary before taking the median over positions in each stage.

The default run is the full 300-sample readout set (150 dog and
150 non-dog).  Set ``QWEN_4B02_LIMIT=1`` for a one-sample smoke test first.
The 300-sample set is constructed to be disjoint from the existing 600-sample
behavior gate by reproducing its deterministic selection and sampling from
the remaining files.
"""

from __future__ import annotations

import project_paths as paths

import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info

try:
    import jlens
    from jlens.hf import Layout
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "jlens is not installed. Activate the experiment environment and install the "
        "official anthropics/jacobian-lens package."
    ) from exc


# Frozen paths and experiment configuration.
MODEL_PATH = Path(str(paths.QWEN_MODEL))
LENS_PATH = Path(
    str(paths.LENS)
)
OUTPUT_DIR = Path(
    str(paths.OUTPUTS_ROOT / "analyze_audio_token_readout")
)
INTERNAL_DIR = OUTPUT_DIR / "internal"
CACHE_DIR = INTERNAL_DIR / "sample_cache"

DOG_BARK_DIR = Path(
    str(paths.DOG_AUDIO)
)
ESC50_DIR = Path(
    str(paths.ESC50_AUDIO)
)
URBANSOUND8K_DIR = Path(
    str(paths.URBANSOUND8K_AUDIO)
)

EXPECTED_TOTAL = 300
EXPECTED_DOG = 150
EXPECTED_NON_DOG = 150
BEHAVIOR_DOG = 300
BEHAVIOR_ESC = 150
BEHAVIOR_URBAN = 150
SEED = 20260819
READOUT_SEED_OFFSET = 100
SAMPLE_LIMIT = int(os.environ.get("QWEN_4B02_LIMIT", str(EXPECTED_TOTAL)))
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))

# These are exactly the source layers in the frozen 4A main Lens.  Layer 26
# was the 4A target layer and is deliberately not treated as a source layer.
SOURCE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24]
TARGET_LAYER = 26
EXPECTED_N_PROMPTS = 1000
EXPECTED_D_MODEL = 3584

# This prompt contains no target animal or bark words.  It is used unchanged
# for every audio sample so that differences are attributable to the audio.
PROMPT = (
    "Identify the primary source in this audio. "
    "Answer with one category only, without explanation."
)
CONCEPTS = [
    "sound",
    "animal",
    "dog",
    "bark",
    "cat",
    "bird",
    "cow",
    "noise",
    "woof",
    "canine",
    "meow",
]
# The primary analysis uses single-token concepts only.  Phrase-level scores
# for multi-token words require a separate sequence-scoring definition and
# must not be approximated by silently selecting one subtoken.
REQUIRED_CONCEPTS = [
    "sound",
    "animal",
    "dog",
    "bark",
    "cat",
    "bird",
    "cow",
    "noise",
    "canine",
]
OPTIONAL_MULTI_TOKEN_CONCEPTS = ["woof", "meow"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALIGNMENT_MAX_ABS_TOL = 1e-3
ALIGNMENT_MIN_COSINE = 0.999
BOOTSTRAP_REPEATS = 1000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def evenly_select(files: list[Path], count: int, seed: int) -> list[Path]:
    """Match the deterministic spacing used by the frozen 600-file gate."""
    if len(files) < count:
        raise ValueError(f"Need {count} files, found {len(files)}")
    step = len(files) / count
    selected = [files[min(int(i * step), len(files) - 1)] for i in range(count)]
    if len(set(selected)) != count:
        raise ValueError("Deterministic selection produced duplicate files")
    # Keep the seed in the signature even though spacing is deterministic; it
    # makes accidental changes to the selection rule visible in the manifest.
    random.Random(seed).random()
    return selected


def choose_remaining(files: list[Path], excluded: set[Path], count: int, seed: int) -> list[Path]:
    remaining = [path for path in files if path not in excluded]
    return evenly_select(remaining, count, seed)


def build_manifest() -> list[dict[str, Any]]:
    dog_all = audio_files(DOG_BARK_DIR)
    esc_all = audio_files(ESC50_DIR)
    urban_all = audio_files(URBANSOUND8K_DIR)

    behavior_dog = set(evenly_select(dog_all, BEHAVIOR_DOG, SEED))
    behavior_esc = set(evenly_select(esc_all, BEHAVIOR_ESC, SEED + 1))
    behavior_urban = set(evenly_select(urban_all, BEHAVIOR_URBAN, SEED + 2))

    dog = choose_remaining(dog_all, behavior_dog, EXPECTED_DOG, SEED + READOUT_SEED_OFFSET)
    esc = choose_remaining(esc_all, behavior_esc, EXPECTED_NON_DOG // 2, SEED + 1 + READOUT_SEED_OFFSET)
    urban = choose_remaining(urban_all, behavior_urban, EXPECTED_NON_DOG // 2, SEED + 2 + READOUT_SEED_OFFSET)

    rows = [
        {
            "sample_id": f"readout_dog_{index:04d}",
            "audio_path": str(path),
            "label": "dog_bark",
            "source": "DogSpeak",
            "behavior_gate_overlap": False,
        }
        for index, path in enumerate(dog, 1)
    ]
    rows.extend(
        {
            "sample_id": f"readout_non_dog_esc_{index:04d}",
            "audio_path": str(path),
            "label": "non_dog_bark",
            "source": "ESC-50",
            "behavior_gate_overlap": False,
        }
        for index, path in enumerate(esc, 1)
    )
    rows.extend(
        {
            "sample_id": f"readout_non_dog_urban_{index:04d}",
            "audio_path": str(path),
            "label": "non_dog_bark",
            "source": "UrbanSound8K",
            "behavior_gate_overlap": False,
        }
        for index, path in enumerate(urban, 1)
    )
    random.Random(SEED + READOUT_SEED_OFFSET).shuffle(rows)
    if len(rows) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} readout samples, got {len(rows)}")
    if sum(row["label"] == "dog_bark" for row in rows) != EXPECTED_DOG:
        raise AssertionError("Readout dog count is not 150")
    if len({row["audio_path"] for row in rows}) != EXPECTED_TOTAL:
        raise AssertionError("Readout manifest contains duplicate paths")
    return rows


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        moved[key] = value
    return moved


def output_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot extract hidden state from {type(output).__name__}")


def model_config_value(model: Any, name: str) -> Any:
    for candidate in (model.config, getattr(model.config, "thinker_config", None)):
        if candidate is not None and hasattr(candidate, name):
            return getattr(candidate, name)
    thinker_config = getattr(model, "config", None)
    if thinker_config is not None and hasattr(thinker_config, "get_text_config"):
        text_config = thinker_config.get_text_config()
        if hasattr(text_config, name):
            return getattr(text_config, name)
    raise AttributeError(f"Cannot find {name} in model config")


def resolve_special_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Special token is not present in tokenizer: {token}")
    return int(token_id)


def resolve_concepts(tokenizer: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    token_ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for concept in CONCEPTS:
        variants = [f" {concept}", concept, concept.capitalize()]
        chosen: dict[str, Any] | None = None
        for variant in variants:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                chosen = {"concept": concept, "text": variant, "token_id": int(encoded[0]), "token_count": 1, "status": "PASS"}
                break
        if chosen is None:
            encoded_variants = []
            for variant in variants:
                encoded_variants.append({
                    "text": variant,
                    "token_ids": [int(item) for item in tokenizer(variant, add_special_tokens=False)["input_ids"]],
                })
            status = "MULTI_TOKEN_SKIPPED" if concept in OPTIONAL_MULTI_TOKEN_CONCEPTS else "FAIL"
            rows.append({
                "concept": concept,
                "text": "",
                "token_id": "",
                "token_count": max((len(item["token_ids"]) for item in encoded_variants), default=0),
                "token_ids": json.dumps(encoded_variants, ensure_ascii=False),
                "status": status,
            })
            continue
        token_ids[concept] = int(chosen["token_id"])
        rows.append(chosen)
    missing = sorted(set(REQUIRED_CONCEPTS) - set(token_ids))
    if missing:
        raise RuntimeError(f"Concepts are not single-token vocabulary items: {missing}")
    return token_ids, rows


def load_model_and_lens() -> tuple[Any, Any, Any, Any, Any, torch.device, dict[str, int], list[dict[str, Any]]]:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not LENS_PATH.is_file():
        raise FileNotFoundError(f"Frozen main Lens not found: {LENS_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4B-02")

    device = torch.device(f"cuda:{DEVICE_INDEX}")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    container = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": device.index},
        low_cpu_mem_usage=True,
        local_files_only=True,
        # 4A was fitted with SDPA; keep the forward implementation identical.
        attn_implementation="sdpa",
    )
    container.disable_talker()
    container.eval()
    thinker = container.thinker
    thinker.eval()
    lens_model = jlens.from_hf(
        thinker,
        processor.tokenizer,
        layout=Layout(path="model"),
        force_bos=True,
    )
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    if lens.source_layers != SOURCE_LAYERS:
        raise ValueError(f"Lens layers {lens.source_layers} != frozen {SOURCE_LAYERS}")
    if lens.n_prompts != EXPECTED_N_PROMPTS or lens.d_model != EXPECTED_D_MODEL:
        raise ValueError(
            f"Unexpected Lens metadata: prompts={lens.n_prompts}, d_model={lens.d_model}"
        )
    concept_ids, concept_rows = resolve_concepts(processor.tokenizer)
    return processor, container, thinker, lens_model, lens, device, concept_ids, concept_rows


def prepare_audio_inputs(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
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
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return {key: value for key, value in inputs.items()}


def contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current != previous + 1:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))
    return runs


def stage_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor, audio_token_id: int, audio_bos_id: int, audio_eos_id: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    valid_len = int(attention_mask.long().sum().item())
    ids = input_ids[0, :valid_len].detach().cpu().tolist()
    audio_positions = [index for index, token_id in enumerate(ids) if token_id == audio_token_id]
    if not audio_positions:
        raise ValueError("No audio_token_id positions were found in input_ids")
    runs = contiguous_runs(audio_positions)
    if len(runs) != 1:
        raise ValueError(f"Audio token positions are not one contiguous block: {runs}")
    audio_start, audio_stop = runs[0]
    audio_bos_index = audio_start - 1
    if audio_bos_index < 0 or ids[audio_bos_index] != audio_bos_id:
        raise ValueError(
            f"Audio block is not preceded by audio_bos: index={audio_bos_index}, "
            f"token={ids[audio_bos_index] if audio_bos_index >= 0 else None}, expected={audio_bos_id}"
        )
    audio_eos_positions = [index for index in range(audio_stop, valid_len) if ids[index] == audio_eos_id]
    if not audio_eos_positions:
        raise ValueError("Could not find audio_eos immediately after audio token block")
    audio_eos_index = audio_eos_positions[0]
    decision_index = valid_len - 1
    audio_count = audio_stop - audio_start
    audio_end_count = max(1, math.ceil(audio_count * 0.10))
    audio_end = list(range(audio_stop - audio_end_count, audio_stop))
    post_audio = list(range(audio_eos_index + 1, decision_index))
    if not post_audio:
        post_audio = [audio_eos_index]
    stages = {
        "audio_end": audio_end,
        "post_audio": post_audio,
        "decision": [decision_index],
    }
    metadata = {
        "valid_sequence_length": valid_len,
        "audio_start": audio_start,
        "audio_stop_exclusive": audio_stop,
        "audio_count": audio_count,
        "audio_bos_index": audio_bos_index,
        "audio_eos_index": audio_eos_index,
        "decision_index": decision_index,
        "audio_end_count": audio_end_count,
        "stage_counts": {key: len(value) for key, value in stages.items()},
        "audio_runs": runs,
    }
    return stages, metadata


def capture_forward(
    thinker: Any,
    inputs: dict[str, Any],
    source_layers: list[int],
    audio_token_id: int,
    audio_bos_id: int,
    audio_eos_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    captures: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def audio_hook(_module: Any, _module_inputs: Any, output: Any) -> Any:
        # Keep the tower dtype so the comparison below can reproduce the
        # model's exact ``audio_features.to(inputs_embeds.dtype)`` cast.
        captures["audio_features"] = output_hidden(output).detach().cpu()
        return output

    def text_model_pre_hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        merged = kwargs.get("inputs_embeds")
        if merged is None:
            raise RuntimeError("Thinker text model did not receive inputs_embeds")
        captures["merged_inputs"] = merged.detach().cpu()

    def layer_hook(layer_index: int):
        def hook(_module: Any, _module_inputs: Any, output: Any) -> Any:
            captures[f"layer_{layer_index}"] = output_hidden(output).detach().float().cpu()
            return output
        return hook

    handles.append(thinker.audio_tower.register_forward_hook(audio_hook))
    handles.append(thinker.model.register_forward_pre_hook(text_model_pre_hook, with_kwargs=True))
    for layer_index in source_layers:
        handles.append(thinker.model.layers[layer_index].register_forward_hook(layer_hook(layer_index)))

    try:
        with torch.inference_mode():
            outputs = thinker(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                use_audio_in_video=False,
            )
        if "audio_features" not in captures or "merged_inputs" not in captures:
            raise RuntimeError("Audio or merged embedding hook did not fire")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        stages, position_metadata = stage_positions(input_ids, attention_mask, audio_token_id, audio_bos_id, audio_eos_id)
        audio_positions = list(range(position_metadata["audio_start"], position_metadata["audio_stop_exclusive"]))
        audio_features = captures["audio_features"]
        merged_inputs = captures["merged_inputs"][0]
        if audio_features.ndim != 2:
            raise ValueError(f"Unexpected audio feature shape: {tuple(audio_features.shape)}")
        projected_in_llm = merged_inputs[audio_positions]
        audio_features = audio_features.to(projected_in_llm.dtype)
        if projected_in_llm.shape != audio_features.shape:
            raise ValueError(
                f"Audio feature/input shape mismatch: tower={tuple(audio_features.shape)}, "
                f"llm={tuple(projected_in_llm.shape)}"
            )
        difference = (audio_features.float() - projected_in_llm.float()).abs()
        cosine = F.cosine_similarity(audio_features.float(), projected_in_llm.float(), dim=-1).mean()
        alignment = {
            **position_metadata,
            "audio_feature_shape": list(audio_features.shape),
            "llm_audio_shape": list(projected_in_llm.shape),
            "audio_feature_dim": int(audio_features.shape[-1]),
            "audio_feature_dtype": str(captures["audio_features"].dtype),
            "llm_embedding_dtype": str(projected_in_llm.dtype),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "mean_cosine": float(cosine.item()),
            "audio_token_id": int(audio_token_id),
            "audio_bos_token_id": int(audio_bos_id),
            "audio_eos_token_id": int(audio_eos_id),
            "alignment_status": "PASS"
            if float(difference.max().item()) <= ALIGNMENT_MAX_ABS_TOL
            and float(cosine.item()) >= ALIGNMENT_MIN_COSINE
            else "FAIL",
        }
        final_logits = outputs.logits[0, position_metadata["decision_index"]].detach().float().cpu()
        top_ids = torch.topk(final_logits, k=5).indices.tolist()
        model_output = {
            "decision_top1_id": int(top_ids[0]),
            "decision_top1_token": str(top_ids[0]),
            "decision_top5_ids": [int(token_id) for token_id in top_ids],
        }
        del outputs
        return captures, alignment, {"stages": stages, "model_output": model_output}
    finally:
        for handle in handles:
            handle.remove()


def ranks_and_scores(logits: torch.Tensor, token_ids: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Return median rank and median -log10(rank) for each concept."""
    if logits.ndim == 3:
        logits = logits[0]
    result: dict[str, tuple[float, float]] = {}
    for concept, token_id in token_ids.items():
        target = logits[:, token_id : token_id + 1]
        rank = 1 + (logits > target).sum(dim=-1)
        rank_float = rank.float()
        median_rank = float(torch.median(rank_float).item())
        median_score = float((-torch.log10(rank_float)).median().item())
        result[concept] = (median_rank, median_score)
    return result


def readout_sample(
    row: dict[str, Any],
    processor: Any,
    thinker: Any,
    lens_model: Any,
    lens: Any,
    device: torch.device,
    token_ids: dict[str, int],
    audio_token_id: int,
    audio_bos_id: int,
    audio_eos_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_inputs = prepare_audio_inputs(processor, row)
    dtype = next(thinker.parameters()).dtype
    inputs = move_inputs_to_device(raw_inputs, device, dtype)
    captures, alignment, extra = capture_forward(
        thinker, inputs, SOURCE_LAYERS, audio_token_id, audio_bos_id, audio_eos_id
    )
    if alignment["alignment_status"] != "PASS":
        raise RuntimeError(f"Audio embedding alignment failed: {alignment}")

    stage_indices: dict[str, list[int]] = extra["stages"]
    readout_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for layer in SOURCE_LAYERS:
            hidden = captures[f"layer_{layer}"]
            unique_positions = sorted({position for positions in stage_indices.values() for position in positions})
            selected = hidden[:, unique_positions, :].to(device)
            transported = lens.transport(selected, layer)
            logits = lens_model.unembed(transported).float()
            position_to_offset = {position: offset for offset, position in enumerate(unique_positions)}
            for stage, positions in stage_indices.items():
                offsets = [position_to_offset[position] for position in positions]
                stage_logits = logits[:, offsets, :]
                values = ranks_and_scores(stage_logits, token_ids)
                for concept, (median_rank, median_score) in values.items():
                    readout_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "label": row["label"],
                            "source": row["source"],
                            "layer": layer,
                            "stage": stage,
                            "concept": concept,
                            "n_positions": len(positions),
                            "median_rank": median_rank,
                            "median_score": median_score,
                            "finite": True,
                            "status": "PASS",
                        }
                    )
            del selected, transported, logits
            torch.cuda.empty_cache()

    return {
        "sample_id": row["sample_id"],
        "label": row["label"],
        "source": row["source"],
        "audio_path": row["audio_path"],
        "alignment": alignment,
        "stages": {
            stage: {"positions": positions, "count": len(positions)}
            for stage, positions in stage_indices.items()
        },
        "model_output": extra["model_output"],
        "readout_rows": readout_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": now_iso(),
    }


def exact_auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return float("nan")
    sorted_negative = sorted(negative)
    total = 0.0
    import bisect
    for value in positive:
        less = bisect.bisect_left(sorted_negative, value)
        equal = bisect.bisect_right(sorted_negative, value) - less
        total += less + 0.5 * equal
    return total / (len(positive) * len(negative))


def bootstrap_ci(positive: list[float], negative: list[float], seed: int) -> tuple[float, float]:
    if len(positive) < 2 or len(negative) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        estimates[index] = exact_auc(
            pos[rng.integers(0, len(pos), len(pos))].tolist(),
            neg[rng.integers(0, len(neg), len(neg))].tolist(),
        )
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def scientific_summary(readout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(lambda: {"dog_bark": [], "non_dog_bark": []})
    for row in readout_rows:
        grouped[(int(row["layer"]), row["stage"], row["concept"])][row["label"]].append(float(row["median_score"]))
    rows: list[dict[str, Any]] = []
    for (layer, stage, concept), groups in sorted(grouped.items()):
        positive = groups["dog_bark"]
        negative = groups["non_dog_bark"]
        auc = exact_auc(positive, negative)
        ci_low, ci_high = bootstrap_ci(positive, negative, SEED + layer * 100 + len(stage) + len(concept))
        rows.append(
            {
                "layer": layer,
                "stage": stage,
                "concept": concept,
                "dog_n": len(positive),
                "non_dog_n": len(negative),
                "dog_median_score": float(np.median(positive)) if positive else float("nan"),
                "non_dog_median_score": float(np.median(negative)) if negative else float("nan"),
                "median_score_delta": float(np.median(positive) - np.median(negative)) if positive and negative else float("nan"),
                "auc": auc,
                "auc_ci95_low": ci_low,
                "auc_ci95_high": ci_high,
                "status": "PASS" if len(positive) == EXPECTED_DOG and len(negative) == EXPECTED_NON_DOG else "INCOMPLETE",
            }
        )
    return rows


def cache_signature(model_config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(model_config, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> None:
    if SAMPLE_LIMIT < 1 or SAMPLE_LIMIT > EXPECTED_TOTAL:
        raise ValueError(f"QWEN_4B02_LIMIT must be in [1, {EXPECTED_TOTAL}]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    write_csv(INTERNAL_DIR / "4B02_manifest.csv", manifest)
    processor, container, thinker, lens_model, lens, device, concept_ids, concept_rows = load_model_and_lens()

    audio_token_id = int(model_config_value(thinker, "audio_token_id"))
    tokenizer = lens_model.tokenizer
    audio_bos_id = resolve_special_id(tokenizer, "<|audio_bos|>")
    audio_eos_id = resolve_special_id(tokenizer, "<|audio_eos|>")
    config = {
        "model_path": str(MODEL_PATH),
        "lens_path": str(LENS_PATH),
        "lens_sha256": sha256_file(LENS_PATH),
        "source_layers": SOURCE_LAYERS,
        "target_layer": TARGET_LAYER,
        "lens_metadata_n_prompts": lens.n_prompts,
        "lens_d_model": lens.d_model,
        "expected_total": EXPECTED_TOTAL,
        "sample_limit": SAMPLE_LIMIT,
        "prompt": PROMPT,
        "concepts": CONCEPTS,
        "audio_token_id": audio_token_id,
        "audio_bos_token_id": audio_bos_id,
        "audio_eos_token_id": audio_eos_id,
        "alignment_max_abs_tol": ALIGNMENT_MAX_ABS_TOL,
        "alignment_min_cosine": ALIGNMENT_MIN_COSINE,
        "selection_seed": SEED,
        "created_at": now_iso(),
    }
    # Do not include the smoke/full sample limit or timestamp in the cache
    # signature.  This lets a successful one-sample smoke test be reused when
    # the same frozen configuration is later expanded to all 300 samples.
    signature_payload = {
        key: value for key, value in config.items()
        if key not in {"sample_limit", "created_at"}
    }
    signature = cache_signature(signature_payload)
    config["run_signature"] = signature
    write_json(INTERNAL_DIR / "frozen_4b02_config.json", config)
    write_csv(INTERNAL_DIR / "concept_tokenization.csv", concept_rows)

    technical: list[dict[str, Any]] = []
    technical.append({
        "check_name": "manifest",
        "result": f"total={len(manifest)}, dog={sum(row['label'] == 'dog_bark' for row in manifest)}, non_dog={sum(row['label'] == 'non_dog_bark' for row in manifest)}",
        "status": "PASS" if len(manifest) == EXPECTED_TOTAL else "FAIL",
    })
    technical.append({
        "check_name": "lens_frozen_config",
        "result": f"prompts={config['lens_metadata_n_prompts']}, d_model={config['lens_d_model']}, layers={SOURCE_LAYERS}",
        "status": "PASS" if config["lens_metadata_n_prompts"] == EXPECTED_N_PROMPTS and config["lens_d_model"] == EXPECTED_D_MODEL else "FAIL",
    })
    technical.append({
        "check_name": "concept_tokenization",
        "result": (
            f"{len(concept_ids)}/{len(REQUIRED_CONCEPTS)} required single-token concepts; "
            f"optional multi-token skipped={len(set(OPTIONAL_MULTI_TOKEN_CONCEPTS) - set(concept_ids))}"
        ),
        "status": "PASS" if set(REQUIRED_CONCEPTS).issubset(concept_ids) else "FAIL",
    })
    if any(row["status"] == "FAIL" for row in technical):
        write_csv(OUTPUT_DIR / "4B02_technical_check.csv", technical)
        raise RuntimeError("4B-02 preflight checks failed")

    all_results: list[dict[str, Any]] = []
    selected_manifest = manifest[:SAMPLE_LIMIT]
    for index, row in enumerate(selected_manifest, 1):
        cache_path = CACHE_DIR / f"{row['sample_id']}.json"
        cached: dict[str, Any] | None = None
        if cache_path.exists():
            try:
                candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                if candidate.get("run_signature") == signature:
                    cached = candidate["result"]
            except Exception:
                cached = None
        if cached is None:
            try:
                result = readout_sample(
                    row,
                    processor=processor,
                    thinker=thinker,
                    lens_model=lens_model,
                    lens=lens,
                    device=device,
                    token_ids=concept_ids,
                    audio_token_id=audio_token_id,
                    audio_bos_id=audio_bos_id,
                    audio_eos_id=audio_eos_id,
                )
                write_json(cache_path, {"run_signature": signature, "result": result})
            except Exception as exc:
                failure = {"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failed_at": now_iso()}
                write_json(INTERNAL_DIR / "4B02_failure.json", failure)
                raise
        else:
            result = cached
        all_results.append(result)
        print(
            f"[{index:03d}/{SAMPLE_LIMIT}] {row['sample_id']} "
            f"label={row['label']} audio_tokens={result['alignment']['audio_count']} "
            f"align_max={result['alignment']['max_abs_error']:.3g} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )

    alignment_rows = []
    readout_rows: list[dict[str, Any]] = []
    for result in all_results:
        alignment_rows.append({"sample_id": result["sample_id"], "label": result["label"], **result["alignment"]})
        readout_rows.extend(result["readout_rows"])
    write_csv(INTERNAL_DIR / "position_alignment_checks.csv", alignment_rows)
    write_csv(INTERNAL_DIR / "stage_level_readout.csv", readout_rows)

    alignment_pass = all(row["alignment_status"] == "PASS" for row in alignment_rows)
    finite_pass = all(row.get("finite") is True and row.get("status") == "PASS" for row in readout_rows)
    full_run = SAMPLE_LIMIT == EXPECTED_TOTAL and len(all_results) == EXPECTED_TOTAL
    technical.extend([
        {"check_name": "audio_position_alignment", "result": f"{sum(row['alignment_status'] == 'PASS' for row in alignment_rows)}/{len(alignment_rows)} samples aligned", "status": "PASS" if alignment_pass and full_run else ("SMOKE_ONLY" if alignment_pass else "FAIL")},
        {"check_name": "stage_lens_readout", "result": f"{len(readout_rows)} finite readout rows", "status": "PASS" if finite_pass and full_run else ("SMOKE_ONLY" if finite_pass else "FAIL")},
    ])
    scientific_rows = scientific_summary(readout_rows)
    write_csv(OUTPUT_DIR / "4B02_scientific_result.csv", scientific_rows)
    overall = all(row["status"] == "PASS" for row in technical)
    technical.append({
        "check_name": "technical_overall",
        "result": "all checks passed" if overall else ("smoke test passed; full run pending" if all(row["status"] in {"PASS", "SMOKE_ONLY"} for row in technical) else "one or more checks failed"),
        "status": "PASS" if overall else ("SMOKE_ONLY" if all(row["status"] in {"PASS", "SMOKE_ONLY"} for row in technical) else "FAIL"),
    })
    write_csv(OUTPUT_DIR / "4B02_technical_check.csv", technical)
    write_json(OUTPUT_DIR / "4B02_summary.json", {
        "sample_limit": SAMPLE_LIMIT,
        "processed": len(all_results),
        "expected_total": EXPECTED_TOTAL,
        "alignment_pass": alignment_pass,
        "readout_rows": len(readout_rows),
        "technical_status": technical[-1]["status"],
        "completed_at": now_iso(),
    })
    print(json.dumps(technical, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(OUTPUT_DIR / "failure.txt", traceback.format_exc())
        raise
