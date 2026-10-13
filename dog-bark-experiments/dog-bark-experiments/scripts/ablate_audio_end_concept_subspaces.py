"""Audio-end concept-subspace ablation for dog/bark recognition (paper stage 4D-01).

The frozen 4A J-lens maps a source-layer residual h_l into the final residual
basis J_l h_l.  For an output concept direction u, this script uses J_l.T u
as the corresponding source-layer direction.  It removes that component only
at the verified Audio-end positions, restores the original residual norm, and
then observes the model's unmodified downstream logits and generated answer.

Main test: L24 Audio-end dog/bark joint ablation.
Supporting replication: L20 Audio-end.
Controls: clean, sham hook, and deterministic matched-strength random
subspaces.  The random dog/bark controls are rank-1 and the random joint
control is rank-2, each matched to its corresponding real ablation.  This
script never modifies model weights, the audio data, or the frozen 4A lens
file.

Set QWEN_4D_LIMIT=2 for a one-dog/one-non-dog smoke test.  The
default (300) runs the frozen 4B-02 manifest and resumes from per-condition
caches.  Use CUDA_VISIBLE_DEVICES=<physical GPU> and QWEN_OMNI_DEVICE=0.
"""

from __future__ import annotations

import project_paths as paths

import bisect
import csv
import hashlib
import json
import math
import os
import re
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
    raise RuntimeError("jlens is required; activate the experiment environment.") from exc


MODEL_PATH = Path(str(paths.QWEN_MODEL))
LENS_PATH = Path(str(paths.LENS))
MANIFEST_PATH = Path(
    str(paths.OUTPUTS_ROOT / "analyze_audio_token_readout/internal/4B02_manifest.csv")
)
OUTPUT_DIR = Path(str(paths.OUTPUTS_ROOT / "ablate_audio_end_concept_subspaces"))
INTERNAL_DIR = OUTPUT_DIR / "internal"
CACHE_DIR = INTERNAL_DIR / "sample_condition_cache"

SOURCE_LAYERS = [20, 24]
EXPECTED_LENS_SOURCE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24]
EXPECTED_N_PROMPTS = 1000
EXPECTED_D_MODEL = 3584
EXPECTED_TOTAL = 300
EXPECTED_DOG = 150
EXPECTED_NON_DOG = 150
SEED = 20260824
SAMPLE_LIMIT = int(os.environ.get("QWEN_4D_LIMIT", str(EXPECTED_TOTAL)))
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))
MAX_NEW_TOKENS = 8

# Keep the neutral 4B-02 prompt.  It does not name dog/bark or expose the
# intervention targets in the text context.
PROMPT = "Identify the primary source in this audio. Answer with one category only, without explanation."
CONCEPTS = ["sound", "animal", "dog", "bark", "cat", "bird", "cow", "noise", "canine"]
DOG_NEGATIVES = ["cat", "bird", "cow", "animal"]
BARK_NEGATIVES = ["noise", "sound"]
CONDITIONS = [
    "clean",
    "sham",
    "dog_ablate",
    "bark_ablate",
    "joint_ablate",
    "random_dog",
    "random_bark",
    "random_joint",
]

ALIGNMENT_MAX_ABS_TOL = 1e-3
ALIGNMENT_MIN_COSINE = 0.999
NORM_REL_TOL = 1e-4
RANDOM_MATCH_REL_TOL = 0.10
BOOTSTRAP_REPEATS = 2000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text_atomic(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("_", " ").replace("-", " ")).strip()


def parse_prediction(response: str) -> str:
    text = normalize(response)
    if "non dog bark" in text or "not a dog" in text or "no dog" in text:
        return "non_dog_bark"
    if "dog bark" in text or "dog barking" in text or "dog" in text or "bark" in text:
        return "dog_bark"
    if any(word in text for word in ("狗叫", "犬吠", "吠叫", "狗", "犬", "汪")):
        return "dog_bark"
    return "non_dog_bark"


def output_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot extract hidden state from {type(output).__name__}")


def replace_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Cannot replace hidden state in {type(output).__name__}")


def move_inputs(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        result[name] = value
    return result


def resolve_special_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Cannot resolve special token {token}")
    return int(token_id)


def resolve_single_token_concepts(tokenizer: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for concept in CONCEPTS:
        choice: tuple[str, int] | None = None
        for text in (f" {concept}", concept, concept.capitalize()):
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                choice = (text, int(encoded[0]))
                break
        if choice is None:
            rows.append({"concept": concept, "status": "FAIL"})
            continue
        ids[concept] = choice[1]
        rows.append({"concept": concept, "text": choice[0], "token_id": choice[1], "status": "PASS"})
    missing = sorted(set(CONCEPTS) - set(ids))
    if missing:
        raise RuntimeError(f"Required causal concepts are multi-token or missing: {missing}")
    return ids, rows


def model_config_value(model: Any, name: str) -> Any:
    for config in (model.config, getattr(model.config, "thinker_config", None)):
        if config is not None and hasattr(config, name):
            return getattr(config, name)
    raise AttributeError(f"Could not find {name} in thinker config")


def resolve_lm_head(container: Any, thinker: Any) -> Any:
    """Find the output projection used by the frozen thinker logits."""
    candidates = (
        getattr(thinker, "lm_head", None),
        getattr(getattr(thinker, "model", None), "lm_head", None),
        getattr(container, "lm_head", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "weight"):
            return candidate
    raise AttributeError("Could not locate the thinker's LM head")


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"4B-02 frozen manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} manifest rows, found {len(rows)}")
    if sum(row["label"] == "dog_bark" for row in rows) != EXPECTED_DOG:
        raise ValueError("Manifest does not contain 150 dog-bark samples")
    if sum(row["label"] == "non_dog_bark" for row in rows) != EXPECTED_NON_DOG:
        raise ValueError("Manifest does not contain 150 non-dog samples")
    if len({row["audio_path"] for row in rows}) != EXPECTED_TOTAL:
        raise ValueError("Manifest has duplicate audio paths")
    for row in rows:
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(f"Missing audio file: {row['audio_path']}")
    return rows


def choose_run_rows(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    if SAMPLE_LIMIT < 1 or SAMPLE_LIMIT > EXPECTED_TOTAL:
        raise ValueError(f"QWEN_4D_LIMIT must be in [1, {EXPECTED_TOTAL}]")
    if SAMPLE_LIMIT == EXPECTED_TOTAL:
        return manifest
    # A smoke test must contain both labels as soon as it has room for two.
    dog = next(row for row in manifest if row["label"] == "dog_bark")
    non_dog = next(row for row in manifest if row["label"] == "non_dog_bark")
    chosen = [dog] if SAMPLE_LIMIT == 1 else [dog, non_dog]
    for row in manifest:
        if len(chosen) >= SAMPLE_LIMIT:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def prepare_inputs(processor: Any, row: dict[str, str]) -> dict[str, Any]:
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": row["audio_path"]},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    return dict(processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    ))


def contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    result: list[tuple[int, int]] = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current != previous + 1:
            result.append((start, previous + 1))
            start = current
        previous = current
    result.append((start, previous + 1))
    return result


def find_audio_end_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor, audio_token_id: int, audio_bos_id: int, audio_eos_id: int) -> tuple[list[int], dict[str, Any]]:
    valid_length = int(attention_mask.long().sum().item())
    ids = input_ids[0, :valid_length].detach().cpu().tolist()
    positions = [index for index, token_id in enumerate(ids) if token_id == audio_token_id]
    runs = contiguous_runs(positions)
    if len(runs) != 1:
        raise ValueError(f"Expected one contiguous audio block, found {runs}")
    start, stop = runs[0]
    if start == 0 or ids[start - 1] != audio_bos_id:
        raise ValueError("Audio token block is not preceded by audio_bos")
    eos_candidates = [index for index in range(stop, valid_length) if ids[index] == audio_eos_id]
    if not eos_candidates:
        raise ValueError("Could not find audio_eos after the audio token block")
    audio_eos_index = eos_candidates[0]
    count = stop - start
    end_count = max(1, math.ceil(count * 0.10))
    return list(range(stop - end_count, stop)), {
        "valid_sequence_length": valid_length,
        "audio_start": start,
        "audio_stop_exclusive": stop,
        "audio_count": count,
        "audio_bos_index": start - 1,
        "audio_eos_index": audio_eos_index,
        "audio_end_count": end_count,
        "decision_index": valid_length - 1,
    }


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def orthonormal(columns: list[torch.Tensor]) -> torch.Tensor:
    matrix = torch.stack([column.float() for column in columns], dim=1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    keep = torch.diagonal(r).abs() > 1e-6
    if not bool(keep.any()):
        raise RuntimeError("Causal direction matrix has rank zero")
    return q[:, keep]


def build_source_bases(lens: Any, lm_head: Any, token_ids: dict[str, int], device: torch.device) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    weights = lm_head.weight.detach().float().to(device)

    dog_out = unit(weights[token_ids["dog"]] - weights[[token_ids[name] for name in DOG_NEGATIVES]].mean(dim=0))
    bark_out = unit(weights[token_ids["bark"]] - weights[[token_ids[name] for name in BARK_NEGATIVES]].mean(dim=0))
    bases: dict[int, dict[str, torch.Tensor]] = {}
    details: list[dict[str, Any]] = []
    for layer in SOURCE_LAYERS:
        jacobian = lens.jacobians[layer].to(device=device, dtype=torch.float32)
        dog = unit(jacobian.T @ dog_out)
        bark = unit(jacobian.T @ bark_out)
        joint = orthonormal([dog, bark])
        generator = torch.Generator(device=device)
        generator.manual_seed(SEED + layer)
        raw = torch.randn((joint.shape[0], joint.shape[1]), generator=generator, device=device, dtype=torch.float32)
        raw = raw - joint @ (joint.T @ raw)
        random_joint = orthonormal([raw[:, index] for index in range(raw.shape[1])])
        random_dog = random_joint[:, :1]
        random_bark = random_joint[:, 1:2]
        bases[layer] = {
            "dog": dog[:, None],
            "bark": bark[:, None],
            "joint": joint,
            "random_dog": random_dog,
            "random_bark": random_bark,
            "random_joint": random_joint,
        }
        details.append({
            "layer": layer,
            "dog_bark_cosine": float(torch.dot(dog, bark).item()),
            "joint_rank": int(joint.shape[1]),
            "random_dog_rank": int(random_dog.shape[1]),
            "random_bark_rank": int(random_bark.shape[1]),
            "random_joint_rank": int(random_joint.shape[1]),
            "dog_source_norm": float(dog.norm().item()),
            "bark_source_norm": float(bark.norm().item()),
            "status": "PASS"
            if joint.shape[1] == 2
            and random_dog.shape == dog[:, None].shape
            and random_bark.shape == bark[:, None].shape
            and random_joint.shape == joint.shape
            else "FAIL",
        })
    return bases, details


def condition_basis(condition: str, bases: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if condition == "dog_ablate":
        return bases["dog"]
    if condition == "bark_ablate":
        return bases["bark"]
    if condition == "joint_ablate":
        return bases["joint"]
    if condition == "random_dog":
        return bases["random_dog"]
    if condition == "random_bark":
        return bases["random_bark"]
    if condition == "random_joint":
        return bases["random_joint"]
    return None


def apply_ablation(hidden: torch.Tensor, basis: torch.Tensor, random_match_basis: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove a subspace and restore every residual vector's original norm."""
    original_norm = hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = (hidden.float() @ basis) @ basis.T
    if random_match_basis is not None:
        semantic_projection = (hidden.float() @ random_match_basis) @ random_match_basis.T
        wanted = semantic_projection.norm(dim=-1, keepdim=True)
        observed = projection.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        projection = projection * (wanted / observed)
    altered = hidden.float() - projection
    altered_norm = altered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    restored = altered * (original_norm / altered_norm)
    final_delta = (hidden.float() - restored).norm(dim=-1)
    raw_delta = projection.norm(dim=-1)
    return restored.to(hidden.dtype), {
        "mean_original_norm": float(original_norm.mean().item()),
        "mean_restored_norm": float(restored.norm(dim=-1).mean().item()),
        "max_norm_relative_error": float(((restored.norm(dim=-1, keepdim=True) - original_norm).abs() / original_norm).max().item()),
        "mean_raw_delta_norm": float(raw_delta.mean().item()),
        "mean_final_delta_norm": float(final_delta.mean().item()),
    }


class InterventionHook:
    def __init__(self, *, layer: int, positions: list[int], condition: str, bases: dict[str, torch.Tensor]):
        self.layer = layer
        self.positions = positions
        self.condition = condition
        self.bases = bases
        self.calls = 0
        self.applied = False
        self.audit: dict[str, Any] = {"condition": condition, "source_layer": layer, "positions": positions}

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.calls += 1
        hidden = output_hidden(output)
        # Generation re-enters the layer on one newly decoded token.  The
        # Audio-end positions only exist during the initial full forward.
        if hidden.shape[1] <= max(self.positions):
            return output
        selected = hidden[:, self.positions, :]
        if self.condition == "clean":
            self.applied = True
            self.audit.update({"operation": "no_hook_change", "max_abs_change": 0.0})
            return output
        if self.condition == "sham":
            self.applied = True
            self.audit.update({"operation": "hook_no_change", "max_abs_change": 0.0})
            return output
        basis = condition_basis(self.condition, self.bases)
        if basis is None:
            raise ValueError(f"Unknown condition {self.condition}")
        random_match = None
        if self.condition == "random_dog":
            random_match = self.bases["dog"]
        elif self.condition == "random_bark":
            random_match = self.bases["bark"]
        elif self.condition == "random_joint":
            random_match = self.bases["joint"]
        changed, diagnostics = apply_ablation(selected, basis, random_match)
        replacement = hidden.clone()
        replacement[:, self.positions, :] = changed
        self.applied = True
        self.audit.update({
            "operation": self.condition,
            "max_abs_change": float((changed.float() - selected.float()).abs().max().item()),
            **diagnostics,
        })
        return replace_output_hidden(output, replacement)


def model_forward_with_intervention(thinker: Any, inputs: dict[str, Any], source_layer: int, positions: list[int], condition: str, bases: dict[str, torch.Tensor]) -> tuple[Any, dict[str, Any]]:
    hook = InterventionHook(layer=source_layer, positions=positions, condition=condition, bases=bases)
    handle = thinker.model.layers[source_layer].register_forward_hook(hook)
    try:
        thinker.rope_deltas = None
        with torch.inference_mode():
            output = thinker(**inputs, use_cache=False, return_dict=True, use_audio_in_video=False)
        if not hook.applied:
            raise RuntimeError(f"Intervention hook did not see Audio-end positions for {condition} at L{source_layer}")
        return output, hook.audit
    finally:
        handle.remove()


def generate_with_intervention(container: Any, thinker: Any, processor: Any, inputs: dict[str, Any], source_layer: int, positions: list[int], condition: str, bases: dict[str, torch.Tensor]) -> str:
    hook = InterventionHook(layer=source_layer, positions=positions, condition=condition, bases=bases)
    handle = thinker.model.layers[source_layer].register_forward_hook(hook)
    try:
        thinker.rope_deltas = None
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = container.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_audio=False,
                use_audio_in_video=False,
            )
        if isinstance(generated, tuple):
            generated = generated[0]
        if not hook.applied:
            raise RuntimeError("Generation intervention hook did not fire")
        return processor.batch_decode(generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    finally:
        handle.remove()


def decision_metrics(logits: torch.Tensor, concept_ids: dict[str, int]) -> dict[str, Any]:
    values = logits.float()
    top_ids = torch.topk(values, k=5).indices.tolist()
    log_probs = F.log_softmax(values, dim=-1)
    return {
        "dog_logit": float(values[concept_ids["dog"]].item()),
        "bark_logit": float(values[concept_ids["bark"]].item()),
        "dog_bark_logit_sum": float((values[concept_ids["dog"]] + values[concept_ids["bark"]]).item()),
        "dog_logprob": float(log_probs[concept_ids["dog"]].item()),
        "bark_logprob": float(log_probs[concept_ids["bark"]].item()),
        "entropy": float((-(log_probs.exp() * log_probs).sum()).item()),
        "top1_token_id": int(top_ids[0]),
        "top5_token_ids": [int(item) for item in top_ids],
    }


def kl_clean_to_condition(clean_logits: torch.Tensor, condition_logits: torch.Tensor) -> float:
    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
    condition_log_probs = F.log_softmax(condition_logits.float(), dim=-1)
    return float((clean_log_probs.exp() * (clean_log_probs - condition_log_probs)).sum().item())


def run_condition(
    row: dict[str, str],
    processor: Any,
    container: Any,
    thinker: Any,
    device: torch.device,
    source_layer: int,
    condition: str,
    bases: dict[str, torch.Tensor],
    concept_ids: dict[str, int],
    special_ids: dict[str, int],
    clean_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    started = time.perf_counter()
    inputs = move_inputs(prepare_inputs(processor, row), device, next(thinker.parameters()).dtype)
    positions, location = find_audio_end_positions(inputs["input_ids"], inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])), special_ids["audio"], special_ids["audio_bos"], special_ids["audio_eos"])
    output, audit = model_forward_with_intervention(thinker, inputs, source_layer, positions, condition, bases)
    decision_logits = output.logits[0, location["decision_index"]].detach().float().cpu()
    metrics = decision_metrics(decision_logits, concept_ids)
    response = generate_with_intervention(container, thinker, processor, inputs, source_layer, positions, condition, bases)
    result = {
        "sample_id": row["sample_id"],
        "audio_path": row["audio_path"],
        "label": row["label"],
        "source": row["source"],
        "source_layer": source_layer,
        "condition": condition,
        **location,
        **audit,
        **metrics,
        "response": response,
        "prediction": parse_prediction(response),
        "correct": parse_prediction(response) == row["label"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": now_iso(),
    }
    if clean_logits is not None:
        result["kl_clean_to_condition"] = kl_clean_to_condition(clean_logits, decision_logits)
    else:
        result["kl_clean_to_condition"] = 0.0
    return result, decision_logits


def cache_signature(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def bootstrap_mean_ci(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        means[index] = array[rng.integers(0, len(array), len(array))].mean()
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def paired_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["sample_id"], int(row["source_layer"]))][row["condition"]] = row
    output: list[dict[str, Any]] = []
    for layer in SOURCE_LAYERS:
        for label in ("dog_bark", "non_dog_bark"):
            for condition in CONDITIONS:
                if condition == "clean":
                    continue
                pairs = [entry for (sample, current_layer), entry in by_key.items() if current_layer == layer and entry.get("clean", {}).get("label") == label and condition in entry]
                for metric in ("dog_logit", "bark_logit", "dog_bark_logit_sum", "entropy", "kl_clean_to_condition"):
                    deltas = [float(item[condition][metric]) - float(item["clean"][metric]) for item in pairs]
                    low, high = bootstrap_mean_ci(deltas, SEED + layer * 1000 + len(label) * 100 + len(condition) * 10 + len(metric))
                    output.append({
                        "source_layer": layer,
                        "label": label,
                        "condition": condition,
                        "metric": metric,
                        "n": len(deltas),
                        "mean_delta_vs_clean": float(np.mean(deltas)) if deltas else float("nan"),
                        "median_delta_vs_clean": float(np.median(deltas)) if deltas else float("nan"),
                        "ci95_low": low,
                        "ci95_high": high,
                        "status": "PASS" if len(deltas) == (EXPECTED_DOG if label == "dog_bark" else EXPECTED_NON_DOG) else "INCOMPLETE",
                    })
    return output


def technical_report(rows: list[dict[str, Any]], full_run: bool, direction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervention_rows = [row for row in rows if row["condition"] not in {"clean", "sham"}]
    sham_rows = [row for row in rows if row["condition"] == "sham"]
    alignment = all(row["audio_end_count"] >= 1 for row in rows)
    norm = all(float(row.get("max_norm_relative_error", 0.0)) <= NORM_REL_TOL for row in intervention_rows)
    sham = all(float(row.get("max_abs_change", 0.0)) == 0.0 for row in sham_rows)
    finite = all(math.isfinite(float(row["dog_bark_logit_sum"])) and math.isfinite(float(row["entropy"])) for row in rows)
    random_pairs = {
        "random_dog": "dog_ablate",
        "random_bark": "bark_ablate",
        "random_joint": "joint_ablate",
    }
    random_rows = [row for row in rows if row["condition"] in random_pairs]
    target_by_key = {
        (row["sample_id"], row["source_layer"], row["condition"]): row
        for row in rows
        if row["condition"] in {"dog_ablate", "bark_ablate", "joint_ablate"}
    }
    matching_errors = []
    for row in random_rows:
        target_condition = random_pairs[row["condition"]]
        paired = target_by_key.get((row["sample_id"], row["source_layer"], target_condition))
        if paired is not None:
            denominator = max(float(paired.get("mean_final_delta_norm", 0.0)), 1e-8)
            matching_errors.append(abs(float(row.get("mean_final_delta_norm", 0.0)) - denominator) / denominator)
    random_match = bool(matching_errors) and max(matching_errors) <= RANDOM_MATCH_REL_TOL
    return [
        {"check_name": "frozen_lens", "result": f"source_layers={EXPECTED_LENS_SOURCE_LAYERS}; target uses L20/L24", "status": "PASS"},
        {"check_name": "audio_end_positions", "result": f"{len(rows)}/{len(rows)} condition forwards have verified Audio-end positions", "status": "PASS" if alignment else "FAIL"},
        {"check_name": "sham_transparency", "result": f"{len(sham_rows)}/{len(sham_rows)} sham interventions changed no activation", "status": "PASS" if sham else "FAIL"},
        {"check_name": "norm_restoration", "result": f"{len(intervention_rows)}/{len(intervention_rows)} interventions within relative tolerance {NORM_REL_TOL}", "status": "PASS" if norm else "FAIL"},
        {"check_name": "random_strength_match", "result": f"max relative mismatch={max(matching_errors) if matching_errors else float('nan'):.4g}", "status": "PASS" if random_match else "FAIL"},
        {"check_name": "finite_downstream_outputs", "result": f"{len(rows)}/{len(rows)} condition outputs finite", "status": "PASS" if finite else "FAIL"},
        {"check_name": "direction_geometry", "result": f"{sum(row['status'] == 'PASS' for row in direction_rows)}/{len(direction_rows)} layer bases valid", "status": "PASS" if all(row["status"] == "PASS" for row in direction_rows) else "FAIL"},
        {"check_name": "technical_overall", "result": "full run passed" if full_run else "smoke test passed; full run pending", "status": "PASS" if full_run else "SMOKE_ONLY"},
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not MODEL_PATH.is_dir() or not LENS_PATH.is_file():
        raise FileNotFoundError("Qwen model or frozen 4A Lens is missing")

    manifest = load_manifest()
    run_rows = choose_run_rows(manifest)
    if not LENS_PATH.is_file():
        raise FileNotFoundError(LENS_PATH)
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    container = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map={"": device.index},
        low_cpu_mem_usage=True, local_files_only=True, attn_implementation="sdpa",
    )
    container.disable_talker()
    container.eval()
    thinker = container.thinker
    thinker.eval()
    lens_model = jlens.from_hf(thinker, processor.tokenizer, layout=Layout(path="model"), force_bos=True)
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    if lens.source_layers != EXPECTED_LENS_SOURCE_LAYERS or lens.n_prompts != EXPECTED_N_PROMPTS or lens.d_model != EXPECTED_D_MODEL:
        raise ValueError("Frozen 4A Lens metadata does not match the registered 4A configuration")
    concept_ids, concept_rows = resolve_single_token_concepts(processor.tokenizer)
    bases, direction_rows = build_source_bases(
        lens,
        resolve_lm_head(container, thinker),
        concept_ids,
        device,
    )
    special_ids = {
        "audio": int(model_config_value(thinker, "audio_token_id")),
        "audio_bos": resolve_special_id(processor.tokenizer, "<|audio_bos|>"),
        "audio_eos": resolve_special_id(processor.tokenizer, "<|audio_eos|>"),
    }
    config = {
        "model_path": str(MODEL_PATH), "lens_path": str(LENS_PATH), "lens_sha256": sha256_file(LENS_PATH),
        "manifest_path": str(MANIFEST_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_layers": SOURCE_LAYERS, "conditions": CONDITIONS, "prompt": PROMPT,
        "dog_negatives": DOG_NEGATIVES, "bark_negatives": BARK_NEGATIVES,
        "norm_restoration": True,
        "random_matched_to": {
            "random_dog": "dog_ablate",
            "random_bark": "bark_ablate",
            "random_joint": "joint_ablate",
        },
        "seed": SEED,
        "sample_limit": SAMPLE_LIMIT, "created_at": now_iso(),
    }
    signature = cache_signature({key: value for key, value in config.items() if key not in {"sample_limit", "created_at"}})
    config["run_signature"] = signature
    write_json(INTERNAL_DIR / "frozen_4d01_config.json", config)
    write_csv(INTERNAL_DIR / "concept_tokenization.csv", concept_rows)
    write_csv(INTERNAL_DIR / "source_direction_geometry.csv", direction_rows)
    write_csv(INTERNAL_DIR / "4D01_manifest.csv", run_rows)

    result_rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(run_rows, 1):
        for source_layer in SOURCE_LAYERS:
            clean_logits: torch.Tensor | None = None
            for condition in CONDITIONS:
                cache_path = CACHE_DIR / f"{row['sample_id']}_L{source_layer}_{condition}.json"
                cached = None
                if cache_path.exists():
                    try:
                        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                        if candidate.get("run_signature") == signature:
                            cached = candidate["result"]
                    except Exception:
                        cached = None
                if cached is None:
                    result, logits = run_condition(row, processor, container, thinker, device, source_layer, condition, bases[source_layer], concept_ids, special_ids, clean_logits)
                    if condition == "clean":
                        clean_logits = logits
                    write_json(cache_path, {"run_signature": signature, "result": result})
                else:
                    result = cached
                    if condition == "clean":
                        # Cached clean logits are intentionally not persisted: rerun only the
                        # clean direct pass for paired KL if the full cache is resumed.
                        _, clean_logits = run_condition(row, processor, container, thinker, device, source_layer, "clean", bases[source_layer], concept_ids, special_ids)
                result_rows.append(result)
                print(f"[{sample_index:03d}/{len(run_rows)}] L{source_layer} {condition} {row['sample_id']} pred={result['prediction']} delta={result.get('mean_final_delta_norm', 0.0):.4f}", flush=True)

    full_run = SAMPLE_LIMIT == EXPECTED_TOTAL and len(run_rows) == EXPECTED_TOTAL
    write_csv(INTERNAL_DIR / "condition_level_results.csv", result_rows)
    statistics = paired_statistics(result_rows)
    write_csv(OUTPUT_DIR / "4D01_paired_statistics.csv", statistics)
    technical = technical_report(result_rows, full_run, direction_rows)
    if any(row["status"] == "FAIL" for row in technical):
        technical[-1] = {"check_name": "technical_overall", "result": "one or more technical checks failed", "status": "FAIL"}
    write_csv(OUTPUT_DIR / "4D01_technical_check.csv", technical)
    write_json(OUTPUT_DIR / "4D01_summary.json", {
        "processed_samples": len(run_rows), "condition_rows": len(result_rows),
        "full_run": full_run, "technical_status": technical[-1]["status"], "completed_at": now_iso(),
    })
    print(json.dumps(technical, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_text_atomic(OUTPUT_DIR / "failure.txt", traceback.format_exc())
        raise
