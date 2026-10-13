"""Hierarchical behavior with exact single-token scoring (original 6D protocol).

This script turns the latent-logit effects from experiments 5A/5B/5C into a
behavioral test.  A frozen Jacobian Lens supplies J-space token directions at
Thinker layer 24.  The native Thinker then generates an answer after one of
the following prefill-only residual interventions:

* projection ablation (necessity),
* norm-scaled steering/injection (sufficiency),
* pseudoinverse coordinate swap (dog <-> cat),
* strictly norm-matched semantic and random controls.

The intervention is applied only during the prompt prefill.  Decode steps are
never edited.  Outcomes are native final-layer logits and greedy generated
text, not J-lens readouts.

Expected manifest columns:
    sample_id,audio_path,label,source

Supported labels:
    dog_bark, cat_vocalization, non_dog_bark

Recommended workflow:
    --phase d0       baseline prompt gate
    --phase smoke    small sample set, primary conditions
    --phase primary  all samples, preregistered main conditions
    --phase full     all prompts and robustness conditions
"""

from __future__ import annotations

import project_paths as paths

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info

try:
    import jlens
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "jlens is not installed. Activate the experiment environment and install the "
        "official anthropics/jacobian-lens package."
    ) from exc


SCRIPT_VERSION = "6d-behavior-1.0"
DEFAULT_MODEL = Path(str(paths.QWEN_MODEL))
DEFAULT_LENS = Path(
    str(paths.LENS)
)
DEFAULT_MANIFEST = Path(
    str(paths.MANIFEST)
)
DEFAULT_OUTPUT = Path(
    str(paths.OUTPUTS_ROOT / "evaluate_single_token_behavior")
)

SOURCE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24]
PRIMARY_LAYER = 24
LENS_TARGET_LAYER = 26
CANONICAL_FINAL_LAYER = 27
EXPECTED_D_MODEL = 3584
EXPECTED_LENS_PROMPTS = 1000
SUPPORTED_LABELS = ("dog_bark", "cat_vocalization", "non_dog_bark")

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    candidates: tuple[str, ...]
    truth_by_label: dict[str, str]


PROMPTS: dict[str, PromptSpec] = {
    "species_a": PromptSpec(
        name="species_a",
        text=(
            "Classify the primary sound source. Answer with exactly one label "
            "from: dog, cat, other. Do not provide any explanation."
        ),
        candidates=("dog", "cat", "other"),
        truth_by_label={
            "dog_bark": "dog",
            "cat_vocalization": "cat",
            "non_dog_bark": "other",
        },
    ),
    "species_b": PromptSpec(
        name="species_b",
        text=(
            "Classify the primary sound source. Answer with exactly one label "
            "from: cat, dog, other. Do not provide any explanation."
        ),
        candidates=("cat", "dog", "other"),
        truth_by_label={
            "dog_bark": "dog",
            "cat_vocalization": "cat",
            "non_dog_bark": "other",
        },
    ),
    "superclass": PromptSpec(
        name="superclass",
        text=(
            "Does the primary sound source come from an animal? Answer with "
            "exactly one label: animal or other. Do not provide any explanation."
        ),
        candidates=("animal", "other"),
        # Non-dog contains sheep/pig/human sounds, so it has no uniform truth.
        truth_by_label={"dog_bark": "animal", "cat_vocalization": "animal"},
    ),
    "vocalization": PromptSpec(
        name="vocalization",
        text=(
            "Classify the primary vocalization in this audio. Answer with "
            "exactly one label from: bark, meow, other. Do not provide any "
            "explanation."
        ),
        candidates=("bark", "meow", "other"),
        truth_by_label={
            "dog_bark": "bark",
            "cat_vocalization": "meow",
            "non_dog_bark": "other",
        },
    ),
    "open": PromptSpec(
        name="open",
        text=(
            "Identify the primary source in this audio. Answer with one category "
            "only, without explanation."
        ),
        candidates=(),
        truth_by_label={},
    ),
}

TRACKED_CONCEPTS = (
    "dog",
    "cat",
    "other",
    "animal",
    "bark",
    "meow",
    "sound",
    "noise",
    "vehicle",
    "bird",
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
BOOTSTRAP_REPEATS = 10_000
SIGN_PERMUTATION_REPEATS = 9_999
MAX_ALLOWED_DELTA_RATIO = 0.25


@dataclass(frozen=True)
class Condition:
    name: str
    operation: str
    strength: float = 0.0
    position_mode: str = "decision"
    source_concept: str = ""
    target_concept: str = ""
    expected_behavior: str = "neutral"


BASELINE = Condition("baseline", "none", position_mode="none")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
    )


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
        writer.writerows([{k: json_safe(v) for k, v in row.items()} for row in rows])
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Required columns are "
            "sample_id,audio_path,label,source."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "audio_path", "label", "source"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        if row["label"] not in SUPPORTED_LABELS:
            raise ValueError(f"Unsupported label {row['label']!r} in {path}")
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio extension: {audio_path}")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        canonical = str(audio_path.resolve())
        if canonical in seen_paths:
            raise ValueError(f"Duplicate audio path: {audio_path}")
        seen_ids.add(row["sample_id"])
        seen_paths.add(canonical)
    return rows


def select_manifest(
    rows: list[dict[str, Any]], phase: str, per_label: int, seed: int
) -> list[dict[str, Any]]:
    grouped = {label: [row for row in rows if row["label"] == label] for label in SUPPORTED_LABELS}
    for label, values in grouped.items():
        if len(values) < per_label:
            raise ValueError(f"Need {per_label} {label} samples, found {len(values)}")
    if phase == "smoke":
        selected = [row for label in SUPPORTED_LABELS for row in grouped[label][:per_label]]
    else:
        rng = random.Random(seed)
        selected = []
        for label in SUPPORTED_LABELS:
            values = list(grouped[label])
            rng.shuffle(values)
            selected.extend(values[:per_label])
        rng.shuffle(selected)
    return selected


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
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = hidden
        return output
    raise TypeError(f"Cannot replace hidden state in {type(output).__name__}")


def model_config_value(model: Any, name: str) -> Any:
    for candidate in (model.config, getattr(model.config, "thinker_config", None)):
        if candidate is not None and hasattr(candidate, name):
            return getattr(candidate, name)
    if hasattr(model.config, "get_text_config"):
        text_config = model.config.get_text_config()
        if hasattr(text_config, name):
            return getattr(text_config, name)
    raise AttributeError(f"Cannot find {name} in Thinker configuration")


def resolve_special_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Special token is not present in tokenizer: {token}")
    return int(token_id)


def move_inputs_to_device(
    inputs: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        moved[key] = value
    return moved


def resolve_single_token(tokenizer: Any, word: str) -> tuple[int | None, list[dict[str, Any]]]:
    variants = [f" {word}", word, word.capitalize()]
    tested: list[dict[str, Any]] = []
    for text in variants:
        ids = [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]
        tested.append({"text": text, "token_ids": ids})
        if len(ids) == 1:
            return ids[0], tested
    return None, tested


def resolve_concepts(tokenizer: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    token_ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for concept in TRACKED_CONCEPTS:
        token_id, variants = resolve_single_token(tokenizer, concept)
        if token_id is not None:
            token_ids[concept] = token_id
        rows.append(
            {
                "concept": concept,
                "chosen_token_id": token_id if token_id is not None else "",
                "status": "PASS" if token_id is not None else "MULTI_TOKEN",
                "variants": json.dumps(variants, ensure_ascii=False),
            }
        )
    required = {"dog", "cat", "other", "animal", "bark", "sound", "vehicle"}
    missing = sorted(required - set(token_ids))
    if missing:
        raise RuntimeError(f"Required labels are not single tokens: {missing}")
    return token_ids, rows


def load_model_and_lens(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, torch.device, dict[str, int], list[dict[str, Any]]]:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.lens_path.is_file():
        raise FileNotFoundError(f"Lens not found: {args.lens_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.device_index}")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(args.model_path), local_files_only=True
    )
    container = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device.index},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    container.disable_talker()
    container.eval()
    thinker = container.thinker
    thinker.eval()
    lens = jlens.JacobianLens.load(str(args.lens_path))
    if lens.source_layers != SOURCE_LAYERS:
        raise ValueError(f"Lens layers {lens.source_layers} != {SOURCE_LAYERS}")
    if lens.n_prompts != EXPECTED_LENS_PROMPTS or lens.d_model != EXPECTED_D_MODEL:
        raise ValueError(
            f"Unexpected lens metadata: prompts={lens.n_prompts}, d_model={lens.d_model}"
        )
    token_ids, token_rows = resolve_concepts(processor.tokenizer)
    return processor, container, thinker, lens, device, token_ids, token_rows


def build_jspace_directions(
    thinker: Any,
    lens: Any,
    layer: int,
    token_ids: dict[str, int],
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Return normalized v_c = J_l^T w_c directions on CPU float32."""
    if layer not in lens.jacobians:
        raise ValueError(f"Layer {layer} is not in fitted lens layers {lens.source_layers}")
    J = lens.jacobians[layer].float().cpu()
    W = thinker.lm_head.weight.detach().float().cpu()
    directions: dict[str, torch.Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    for concept, token_id in token_ids.items():
        raw = J.T @ W[token_id]
        norm = raw.norm().clamp_min(1e-12)
        directions[concept] = raw / norm
        diagnostics.append(
            {
                "layer": layer,
                "concept": concept,
                "token_id": token_id,
                "raw_norm": float(norm.item()),
                "finite": bool(torch.isfinite(raw).all().item()),
            }
        )
    required_for_random = [
        directions[name]
        for name in ("dog", "cat", "bark", "animal", "sound", "vehicle")
        if name in directions
    ]
    D = torch.stack(required_for_random, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vector = torch.randn(D.shape[0], generator=generator)
    random_vector = random_vector - D @ (torch.linalg.pinv(D) @ random_vector)
    random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
    directions["random_orthogonal"] = random_vector
    diagnostics.append(
        {
            "layer": layer,
            "concept": "random_orthogonal",
            "token_id": "",
            "raw_norm": 1.0,
            "finite": bool(torch.isfinite(random_vector).all().item()),
        }
    )
    for left, right in (("dog", "cat"), ("dog", "bark"), ("dog", "vehicle")):
        pair = torch.stack([directions[left], directions[right]], dim=1)
        diagnostics.append(
            {
                "layer": layer,
                "concept": f"pair:{left}:{right}",
                "cosine": float(F.cosine_similarity(pair[:, 0], pair[:, 1], dim=0).item()),
                "pair_condition_number": float(torch.linalg.cond(pair).item()),
                "finite": True,
            }
        )
    return directions, diagnostics


def prepare_audio_inputs(
    processor: Any, row: dict[str, Any], prompt: PromptSpec
) -> dict[str, Any]:
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": prompt.text},
            ],
        },
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
    result: list[tuple[int, int]] = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current != previous + 1:
            result.append((start, previous + 1))
            start = current
        previous = current
    result.append((start, previous + 1))
    return result


def locate_positions(
    inputs: dict[str, Any], tokenizer: Any, audio_token_id: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    valid_len = int(attention_mask[0].long().sum().item())
    ids = input_ids[0, :valid_len].detach().cpu().tolist()
    audio_bos_id = resolve_special_id(tokenizer, "<|audio_bos|>")
    audio_eos_id = resolve_special_id(tokenizer, "<|audio_eos|>")
    positions = [i for i, token_id in enumerate(ids) if token_id == audio_token_id]
    runs = contiguous_runs(positions)
    if len(runs) != 1:
        raise ValueError(f"Expected one contiguous audio token block, got {runs}")
    audio_start, audio_stop = runs[0]
    bos_index = audio_start - 1
    if bos_index < 0 or ids[bos_index] != audio_bos_id:
        raise ValueError("Audio block is not preceded by <|audio_bos|>")
    eos_candidates = [i for i in range(audio_stop, valid_len) if ids[i] == audio_eos_id]
    if not eos_candidates:
        raise ValueError("Could not find <|audio_eos|>")
    eos_index = eos_candidates[0]
    decision = valid_len - 1
    count = audio_stop - audio_start
    middle_start = audio_start + count // 3
    middle_stop = audio_start + (2 * count) // 3
    if middle_stop <= middle_start:
        middle_stop = min(audio_stop, middle_start + 1)
    post_audio = list(range(eos_index + 1, decision)) or [eos_index]
    modes = {
        "decision": [decision],
        "audio_middle": list(range(middle_start, middle_stop)),
        "post_audio": post_audio,
        "none": [],
    }
    metadata = {
        "valid_sequence_length": valid_len,
        "audio_start": audio_start,
        "audio_stop_exclusive": audio_stop,
        "audio_count": count,
        "audio_bos_index": bos_index,
        "audio_eos_index": eos_index,
        "decision_index": decision,
        "audio_middle_count": len(modes["audio_middle"]),
        "post_audio_count": len(post_audio),
    }
    return modes, metadata


def projection_delta(h: torch.Tensor, direction: torch.Tensor, beta: float) -> torch.Tensor:
    direction = direction.to(h.device, h.dtype)
    coefficient = torch.dot(h.float(), direction.float()).to(h.dtype)
    return -beta * coefficient * direction


def injection_delta(h: torch.Tensor, direction: torch.Tensor, ratio: float) -> torch.Tensor:
    direction = direction.to(h.device, h.dtype)
    return ratio * h.float().norm().to(h.dtype) * direction


def span_projection(h: torch.Tensor, vectors: list[torch.Tensor]) -> torch.Tensor:
    D = torch.stack([vector.to(h.device, torch.float32) for vector in vectors], dim=1)
    coefficients = torch.linalg.pinv(D) @ h.float()
    return (D @ coefficients).to(h.dtype)


def swap_delta(
    h: torch.Tensor, source: torch.Tensor, target: torch.Tensor, strength: float
) -> torch.Tensor:
    D = torch.stack(
        [source.to(h.device, torch.float32), target.to(h.device, torch.float32)], dim=1
    )
    coefficients = torch.linalg.pinv(D) @ h.float()
    swapped = coefficients.flip(0)
    return (strength * D @ (swapped - coefficients)).to(h.dtype)


def matched_direction_delta(
    h: torch.Tensor, direction: torch.Tensor, target_norm: torch.Tensor, reduce_coordinate: bool
) -> torch.Tensor:
    direction = direction.to(h.device, h.dtype)
    if reduce_coordinate:
        coefficient = torch.dot(h.float(), direction.float())
        sign = 1.0 if float(coefficient.item()) >= 0.0 else -1.0
        return -sign * target_norm.to(h.dtype) * direction
    return target_norm.to(h.dtype) * direction


def condition_delta(
    h: torch.Tensor,
    condition: Condition,
    directions: dict[str, torch.Tensor],
) -> torch.Tensor:
    op = condition.operation
    if op in {"none", "zero_hook"}:
        return torch.zeros_like(h)
    if op == "ablate":
        return projection_delta(h, directions[condition.source_concept], condition.strength)
    if op == "joint_ablate":
        projection = span_projection(h, [directions["dog"], directions["bark"]])
        return -condition.strength * projection
    if op == "inject":
        return injection_delta(h, directions[condition.source_concept], condition.strength)
    if op == "joint_inject":
        joint = directions["dog"] + directions["bark"]
        joint = joint / joint.norm().clamp_min(1e-12)
        return injection_delta(h, joint, condition.strength)
    if op == "swap":
        return swap_delta(
            h,
            directions[condition.source_concept],
            directions[condition.target_concept],
            condition.strength,
        )
    if op in {"matched_random_ablate", "matched_vehicle_ablate"}:
        reference = projection_delta(h, directions["dog"], condition.strength)
        name = "random_orthogonal" if op == "matched_random_ablate" else "vehicle"
        return matched_direction_delta(
            h, directions[name], reference.float().norm(), reduce_coordinate=(name == "vehicle")
        )
    if op in {"random_inject", "vehicle_inject"}:
        name = "random_orthogonal" if op == "random_inject" else "vehicle"
        return injection_delta(h, directions[name], condition.strength)
    if op == "matched_control_swap":
        reference = swap_delta(
            h, directions["dog"], directions["cat"], condition.strength
        )
        control = swap_delta(
            h, directions["sound"], directions["vehicle"], condition.strength
        )
        control_norm = control.float().norm()
        if float(control_norm.item()) <= 1e-12:
            return torch.zeros_like(h)
        return control * (reference.float().norm() / control_norm).to(control.dtype)
    raise ValueError(f"Unknown operation: {op}")


class PrefillIntervention:
    def __init__(
        self,
        layer_module: Any,
        condition: Condition,
        directions: dict[str, torch.Tensor],
        positions: list[int],
        expected_prefill_length: int,
    ) -> None:
        self.layer_module = layer_module
        self.condition = condition
        self.directions = directions
        self.positions = positions
        self.expected_prefill_length = expected_prefill_length
        self.handle: Any = None
        self.applied = False
        self.hidden_norms: list[float] = []
        self.delta_norms: list[float] = []
        self.delta_ratios: list[float] = []

    def __enter__(self) -> "PrefillIntervention":
        def hook(_module: Any, _module_inputs: Any, output: Any) -> Any:
            hidden = output_hidden(output)
            if self.applied or hidden.shape[1] != self.expected_prefill_length:
                return output
            edited = hidden.clone()
            for position in self.positions:
                original = hidden[0, position]
                delta = condition_delta(original, self.condition, self.directions)
                edited[0, position] = original + delta
                hidden_norm = float(original.float().norm().item())
                delta_norm = float(delta.float().norm().item())
                self.hidden_norms.append(hidden_norm)
                self.delta_norms.append(delta_norm)
                self.delta_ratios.append(delta_norm / max(hidden_norm, 1e-12))
            self.applied = True
            return replace_output_hidden(output, edited)

        self.handle = self.layer_module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "hook_applied": self.applied,
            "selected_positions": len(self.positions),
            "mean_hidden_norm": float(np.mean(self.hidden_norms)) if self.hidden_norms else 0.0,
            "mean_delta_norm": float(np.mean(self.delta_norms)) if self.delta_norms else 0.0,
            "mean_actual_delta_norm_ratio": float(np.mean(self.delta_ratios)) if self.delta_ratios else 0.0,
            "max_actual_delta_norm_ratio": float(np.max(self.delta_ratios)) if self.delta_ratios else 0.0,
        }


def decode_generated(
    tokenizer: Any, sequences: torch.Tensor, prompt_length: int
) -> tuple[str, list[int]]:
    sequence = sequences[0].detach().cpu()
    generated = sequence[prompt_length:] if sequence.numel() > prompt_length else sequence
    ids = [int(x) for x in generated.tolist()]
    text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return text, ids


def run_generation(
    thinker: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    condition: Condition,
    directions: dict[str, torch.Tensor],
    positions: list[int],
    layer: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, str, list[int], dict[str, Any]]:
    prompt_length = int(inputs["input_ids"].shape[1])
    layer_module = thinker.model.layers[layer]
    context = PrefillIntervention(
        layer_module, condition, directions, positions, prompt_length
    )
    generation_kwargs = {
        **inputs,
        "use_audio_in_video": False,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }
    if hasattr(thinker.generation_config, "output_logits"):
        generation_kwargs["output_logits"] = True
    hook_context = context if condition.operation != "none" else nullcontext()
    with hook_context, torch.inference_mode():
        generated = thinker.generate(**generation_kwargs)
    if not hasattr(generated, "scores") or not generated.scores:
        raise RuntimeError(
            "Thinker.generate did not return scores. The installed transformers "
            "version must support return_dict_in_generate=True and output_scores=True."
        )
    raw_logits = getattr(generated, "logits", None)
    first_logits = (
        raw_logits[0][0]
        if raw_logits is not None and len(raw_logits) > 0
        else generated.scores[0][0]
    ).detach().float().cpu()
    generated_text, generated_ids = decode_generated(
        tokenizer, generated.sequences, prompt_length
    )
    diagnostics = context.diagnostics()
    if condition.operation != "none" and not diagnostics["hook_applied"]:
        raise RuntimeError("Intervention hook did not fire during prefill")
    return first_logits, generated_text, generated_ids, diagnostics


def zero_strength_transparency_check(
    row: dict[str, Any],
    processor: Any,
    thinker: Any,
    device: torch.device,
    directions: dict[str, torch.Tensor],
    layer: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Compare a native generation with a registered zero-delta prefill hook."""
    prompt = PROMPTS["species_a"]
    raw_inputs = prepare_audio_inputs(processor, row, prompt)
    position_modes, metadata = locate_positions(
        raw_inputs,
        processor.tokenizer,
        int(model_config_value(thinker, "audio_token_id")),
    )
    inputs = move_inputs_to_device(
        raw_inputs, device, next(thinker.parameters()).dtype
    )
    native_logits, native_text, native_ids, _ = run_generation(
        thinker,
        processor.tokenizer,
        inputs,
        BASELINE,
        directions,
        [],
        layer,
        max_new_tokens,
    )
    zero_condition = Condition(
        "zero_strength_hook",
        "zero_hook",
        0.0,
        "decision",
        "dog",
    )
    hook_logits, hook_text, hook_ids, hook_diagnostics = run_generation(
        thinker,
        processor.tokenizer,
        inputs,
        zero_condition,
        directions,
        position_modes["decision"],
        layer,
        max_new_tokens,
    )
    max_error = float((native_logits - hook_logits).abs().max().item())
    return {
        "sample_id": row["sample_id"],
        "audio_token_count": metadata["audio_count"],
        "max_abs_first_step_logit_error": max_error,
        "generated_token_ids_equal": native_ids == hook_ids,
        "generated_text_equal": native_text == hook_text,
        **hook_diagnostics,
        "status": (
            "PASS"
            if max_error == 0.0
            and native_ids == hook_ids
            and native_text == hook_text
            and hook_diagnostics["hook_applied"]
            else "FAIL"
        ),
    }


def normalize_label(text: str, candidates: tuple[str, ...]) -> str:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    words = normalized.split()
    aliases = {
        "dog": {"dog", "canine", "puppy"},
        "cat": {"cat", "feline", "kitten"},
        "other": {"other", "none", "unknown"},
        "animal": {"animal", "yes"},
        "bark": {"bark", "barking", "woof"},
        "meow": {"meow", "meowing", "mew"},
    }
    for candidate in candidates:
        if any(word in aliases.get(candidate, {candidate}) for word in words):
            return candidate
    return "unmapped"


def score_logits(
    logits: torch.Tensor,
    token_ids: dict[str, int],
    prompt: PromptSpec,
    tokenizer: Any,
) -> dict[str, Any]:
    top_values, top_ids = torch.topk(logits, k=min(10, logits.numel()))
    row: dict[str, Any] = {
        "global_top1_id": int(top_ids[0].item()),
        "global_top1_token": tokenizer.decode([int(top_ids[0].item())], skip_special_tokens=False),
        "global_top10_tokens": json.dumps(
            [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "logit": float(value),
                }
                for value, token_id in zip(top_values.tolist(), top_ids.tolist())
            ],
            ensure_ascii=False,
        ),
    }
    for concept, token_id in token_ids.items():
        value = logits[token_id]
        row[f"{concept}_logit"] = float(value.item())
        row[f"{concept}_rank"] = 1 + int((logits > value).sum().item())
    if "dog" in token_ids and "cat" in token_ids:
        row["dog_minus_cat_logit"] = row["dog_logit"] - row["cat_logit"]
    if "dog" in token_ids and "sound" in token_ids and "noise" in token_ids:
        row["dog_minus_sound_noise_logit"] = row["dog_logit"] - max(
            row["sound_logit"], row["noise_logit"]
        )
    if prompt.candidates and all(candidate in token_ids for candidate in prompt.candidates):
        candidate_logits = torch.tensor(
            [float(logits[token_ids[candidate]].item()) for candidate in prompt.candidates]
        )
        probabilities = torch.softmax(candidate_logits, dim=0)
        for candidate, probability in zip(prompt.candidates, probabilities.tolist()):
            row[f"closed_{candidate}_probability"] = float(probability)
        row["closed_logit_top1"] = prompt.candidates[int(candidate_logits.argmax().item())]
        row["closed_probability_status"] = "EXACT_SINGLE_TOKEN"
    elif prompt.candidates:
        row["closed_logit_top1"] = ""
        row["closed_probability_status"] = "MULTI_TOKEN_CANDIDATE"
    else:
        row["closed_logit_top1"] = ""
        row["closed_probability_status"] = "NOT_APPLICABLE"
    return row


def desired_label(label: str, prompt_name: str, condition: Condition) -> str:
    if prompt_name.startswith("species"):
        if condition.name == "dog_cat_swap":
            if label == "dog_bark":
                return "cat"
            if label == "cat_vocalization":
                return "dog"
        if condition.operation in {"inject", "joint_inject"} and label == "non_dog_bark":
            return "dog"
        if condition.operation in {"ablate", "joint_ablate"} and label == "dog_bark":
            return "not_dog"
    if prompt_name == "superclass" and label in {"dog_bark", "cat_vocalization"}:
        return "animal"
    if prompt_name == "vocalization":
        if condition.source_concept == "bark" and condition.operation == "inject":
            return "bark"
        if condition.source_concept == "bark" and condition.operation == "ablate":
            return "not_bark"
    return ""


def condition_catalog(label: str, prompt_name: str, phase: str) -> list[Condition]:
    conditions: list[Condition] = [BASELINE]
    if phase == "d0":
        return conditions
    primary_only = phase == "smoke"
    if prompt_name.startswith("species"):
        if label == "dog_bark":
            betas = [1.0] if primary_only else [0.5, 1.0, 2.0]
            conditions += [
                Condition("dog_ablate", "ablate", beta, source_concept="dog", expected_behavior="not_dog")
                for beta in betas
            ]
            conditions += [
                Condition("matched_vehicle_ablate", "matched_vehicle_ablate", 1.0, source_concept="vehicle"),
                Condition("matched_random_ablate", "matched_random_ablate", 1.0, source_concept="random_orthogonal"),
            ]
            swaps = [1.0] if primary_only else [0.5, 1.0, 1.5]
            conditions += [
                Condition("dog_cat_swap", "swap", gamma, source_concept="dog", target_concept="cat", expected_behavior="cat")
                for gamma in swaps
            ]
            conditions.append(
                Condition("matched_sound_vehicle_swap", "matched_control_swap", 1.0, source_concept="sound", target_concept="vehicle")
            )
            if phase == "full":
                conditions += [
                    Condition("bark_ablate", "ablate", 1.0, source_concept="bark"),
                    Condition("dog_bark_joint_ablate", "joint_ablate", 1.0, source_concept="dog", target_concept="bark"),
                    Condition("dog_ablate_audio_middle", "ablate", 1.0, "audio_middle", "dog"),
                    Condition("dog_ablate_post_audio", "ablate", 1.0, "post_audio", "dog"),
                ]
        elif label == "cat_vocalization":
            swaps = [1.0] if primary_only else [0.5, 1.0, 1.5]
            conditions += [
                Condition("dog_cat_swap", "swap", gamma, source_concept="dog", target_concept="cat", expected_behavior="dog")
                for gamma in swaps
            ]
            conditions.append(
                Condition("matched_sound_vehicle_swap", "matched_control_swap", 1.0, source_concept="sound", target_concept="vehicle")
            )
        elif label == "non_dog_bark":
            ratios = [0.03, 0.10] if primary_only else [0.03, 0.06, 0.10]
            conditions += [
                Condition("dog_inject", "inject", ratio, source_concept="dog", expected_behavior="dog")
                for ratio in ratios
            ]
            conditions += [
                Condition("vehicle_inject", "vehicle_inject", 0.06, source_concept="vehicle"),
                Condition("random_orthogonal_inject", "random_inject", 0.06, source_concept="random_orthogonal"),
            ]
            if phase == "full":
                conditions += [
                    Condition("bark_inject", "inject", ratio, source_concept="bark")
                    for ratio in (0.03, 0.06, 0.10)
                ]
                conditions += [
                    Condition("dog_bark_joint_inject", "joint_inject", ratio, source_concept="dog", target_concept="bark")
                    for ratio in (0.03, 0.06, 0.10)
                ]
                conditions += [
                    Condition("dog_inject_audio_middle", "inject", 0.06, "audio_middle", "dog"),
                    Condition("dog_inject_post_audio", "inject", 0.06, "post_audio", "dog"),
                ]
    elif prompt_name == "superclass" and label in {"dog_bark", "cat_vocalization"}:
        conditions += [
            Condition("dog_cat_swap", "swap", 1.0, source_concept="dog", target_concept="cat", expected_behavior="animal"),
            Condition("matched_sound_vehicle_swap", "matched_control_swap", 1.0, source_concept="sound", target_concept="vehicle"),
        ]
        if phase == "full" and label == "dog_bark":
            conditions.append(Condition("dog_ablate", "ablate", 1.0, source_concept="dog"))
    elif prompt_name == "vocalization":
        if label == "dog_bark":
            conditions += [
                Condition("dog_ablate", "ablate", 1.0, source_concept="dog"),
                Condition("bark_ablate", "ablate", 1.0, source_concept="bark", expected_behavior="not_bark"),
                Condition("dog_bark_joint_ablate", "joint_ablate", 1.0, source_concept="dog", target_concept="bark"),
                Condition("dog_cat_swap", "swap", 1.0, source_concept="dog", target_concept="cat"),
            ]
        elif label == "cat_vocalization":
            conditions.append(Condition("dog_cat_swap", "swap", 1.0, source_concept="dog", target_concept="cat"))
        elif label == "non_dog_bark":
            conditions += [
                Condition("dog_inject", "inject", 0.06, source_concept="dog"),
                Condition("bark_inject", "inject", 0.06, source_concept="bark", expected_behavior="bark"),
                Condition("dog_bark_joint_inject", "joint_inject", 0.06, source_concept="dog", target_concept="bark"),
            ]
    elif prompt_name == "open":
        if label == "dog_bark":
            conditions += [
                Condition("dog_ablate", "ablate", 1.0, source_concept="dog"),
                Condition("dog_cat_swap", "swap", 1.0, source_concept="dog", target_concept="cat"),
            ]
        elif label == "cat_vocalization":
            conditions.append(Condition("dog_cat_swap", "swap", 1.0, source_concept="dog", target_concept="cat"))
        else:
            conditions.append(Condition("dog_inject", "inject", 0.06, source_concept="dog"))
    return conditions


def prompt_names_for_phase(phase: str) -> list[str]:
    if phase == "smoke":
        return ["species_a", "superclass"]
    if phase == "d0":
        return ["species_a", "species_b", "superclass", "vocalization", "open"]
    if phase == "primary":
        return ["species_a", "superclass", "open"]
    return ["species_a", "species_b", "superclass", "vocalization", "open"]


def run_one(
    row: dict[str, Any],
    prompt: PromptSpec,
    condition: Condition,
    processor: Any,
    thinker: Any,
    device: torch.device,
    token_ids: dict[str, int],
    directions: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_inputs = prepare_audio_inputs(processor, row, prompt)
    position_modes, position_metadata = locate_positions(
        raw_inputs,
        processor.tokenizer,
        int(model_config_value(thinker, "audio_token_id")),
    )
    positions = position_modes[condition.position_mode]
    inputs = move_inputs_to_device(raw_inputs, device, next(thinker.parameters()).dtype)
    logits, text, generated_ids, intervention_diagnostics = run_generation(
        thinker=thinker,
        tokenizer=processor.tokenizer,
        inputs=inputs,
        condition=condition,
        directions=directions,
        positions=positions,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )
    generated_label = normalize_label(text, prompt.candidates) if prompt.candidates else ""
    truth = prompt.truth_by_label.get(row["label"], "")
    desired = desired_label(row["label"], prompt.name, condition)
    desired_success: bool | str = ""
    if desired:
        if desired.startswith("not_"):
            desired_success = generated_label != desired[4:] and generated_label != "unmapped"
        else:
            desired_success = generated_label == desired
    scored = score_logits(logits, token_ids, prompt, processor.tokenizer)
    return {
        **row,
        "phase": args.phase,
        "prompt_name": prompt.name,
        "prompt_text": prompt.text,
        "condition": condition.name,
        "operation": condition.operation,
        "layer": args.layer,
        "position_mode": condition.position_mode,
        "source_concept": condition.source_concept,
        "target_concept": condition.target_concept,
        "strength": condition.strength,
        "expected_behavior": condition.expected_behavior,
        **position_metadata,
        **intervention_diagnostics,
        "generated_text": text,
        "generated_token_ids": json.dumps(generated_ids),
        "generated_label": generated_label,
        "truth_label": truth,
        "is_correct": (generated_label == truth) if truth else "",
        "desired_label": desired,
        "desired_success": desired_success,
        **scored,
        "finite_first_step_logits": bool(torch.isfinite(logits).all().item()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": now_iso(),
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_REPEATS, values.size)
    )
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def sign_permutation_p(values: np.ndarray, seed: int) -> float:
    if values.size < 2 or np.allclose(values, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(SIGN_PERMUTATION_REPEATS, values.size),
    )
    null_means = np.abs((signs * values[None, :]).mean(axis=1))
    exceed = int(np.sum(null_means >= observed))
    return (exceed + 1) / (SIGN_PERMUTATION_REPEATS + 1)


def bh_fdr(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    sorted_p = np.asarray(p_values, dtype=np.float64)[order]
    sorted_q = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)
    result = np.empty(m, dtype=np.float64)
    result[order] = sorted_q
    return result.tolist()


def summarize_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (row["sample_id"], row["prompt_name"]): row
        for row in rows
        if row["condition"] == "baseline"
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["label"], row["prompt_name"], row["condition"], row["operation"],
            row["strength"], row["position_mode"],
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=str):
        label, prompt_name, condition, operation, strength, position_mode = key
        paired = [(baselines.get((row["sample_id"], prompt_name)), row) for row in values]
        paired = [(base, row) for base, row in paired if base is not None]
        n = len(paired)
        baseline_correct = [bool(base["is_correct"]) for base, _ in paired if base["is_correct"] != ""]
        intervention_correct = [bool(row["is_correct"]) for _, row in paired if row["is_correct"] != ""]
        baseline_labels = [base["generated_label"] for base, _ in paired]
        intervention_labels = [row["generated_label"] for _, row in paired]
        desired_values = [bool(row["desired_success"]) for _, row in paired if row["desired_success"] != ""]
        mapped = [label_value != "unmapped" for label_value in intervention_labels if label_value]
        b = c = 0
        if baseline_correct and len(baseline_correct) == len(intervention_correct):
            b = sum(before and not after for before, after in zip(baseline_correct, intervention_correct))
            c = sum((not before) and after for before, after in zip(baseline_correct, intervention_correct))
        conditional_pairs = [
            (base, row) for base, row in paired
            if base["is_correct"] is True and base["generated_label"] != "unmapped"
        ]
        conditional_changed = sum(
            base["generated_label"] != row["generated_label"]
            for base, row in conditional_pairs
        )
        desired_successes = sum(desired_values)
        desired_ci = wilson_interval(desired_successes, len(desired_values))
        summaries.append(
            {
                "label": label,
                "prompt_name": prompt_name,
                "condition": condition,
                "operation": operation,
                "strength": strength,
                "position_mode": position_mode,
                "n": n,
                "baseline_accuracy": float(np.mean(baseline_correct)) if baseline_correct else "",
                "intervention_accuracy": float(np.mean(intervention_correct)) if intervention_correct else "",
                "baseline_generated_distribution": json.dumps(dict(sorted({x: baseline_labels.count(x) for x in set(baseline_labels)}.items())), ensure_ascii=False),
                "intervention_generated_distribution": json.dumps(dict(sorted({x: intervention_labels.count(x) for x in set(intervention_labels)}.items())), ensure_ascii=False),
                "mapped_output_rate": float(np.mean(mapped)) if mapped else "",
                "desired_success_n": desired_successes if desired_values else "",
                "desired_success_rate": float(np.mean(desired_values)) if desired_values else "",
                "desired_success_ci95_low": desired_ci[0] if desired_values else "",
                "desired_success_ci95_high": desired_ci[1] if desired_values else "",
                "baseline_correct_n": len(conditional_pairs),
                "conditional_any_flip_rate": conditional_changed / len(conditional_pairs) if conditional_pairs else "",
                "mcnemar_b_correct_to_wrong": b if baseline_correct else "",
                "mcnemar_c_wrong_to_correct": c if baseline_correct else "",
                "mcnemar_exact_p": exact_mcnemar_p(b, c) if baseline_correct else "",
            }
        )
    return summaries


def summarize_paired_numeric(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    baselines = {
        (row["sample_id"], row["prompt_name"]): row
        for row in rows if row["condition"] == "baseline"
    }
    metrics = [
        "dog_logit", "cat_logit", "animal_logit", "bark_logit",
        "dog_minus_cat_logit", "dog_minus_sound_noise_logit",
        "closed_dog_probability", "closed_cat_probability",
        "closed_other_probability", "closed_animal_probability",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] == "baseline":
            continue
        key = (
            row["label"], row["prompt_name"], row["condition"], row["strength"],
            row["position_mode"],
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    p_locations: list[int] = []
    p_values: list[float] = []
    for key, values in sorted(grouped.items(), key=str):
        label, prompt_name, condition, strength, position_mode = key
        for metric in metrics:
            changes: list[float] = []
            baseline_values: list[float] = []
            intervention_values: list[float] = []
            for row in values:
                base = baselines.get((row["sample_id"], prompt_name))
                if base is None or metric not in row or metric not in base:
                    continue
                try:
                    before = float(base[metric])
                    after = float(row[metric])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(before) and math.isfinite(after):
                    baseline_values.append(before)
                    intervention_values.append(after)
                    changes.append(after - before)
            if not changes:
                continue
            array = np.asarray(changes, dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(array, seed + len(result) * 17)
            p_value = sign_permutation_p(array, seed + len(result) * 31)
            result.append(
                {
                    "label": label,
                    "prompt_name": prompt_name,
                    "condition": condition,
                    "strength": strength,
                    "position_mode": position_mode,
                    "metric": metric,
                    "n": len(changes),
                    "baseline_mean": float(np.mean(baseline_values)),
                    "intervention_mean": float(np.mean(intervention_values)),
                    "mean_paired_change": float(array.mean()),
                    "median_paired_change": float(np.median(array)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "two_sided_sign_permutation_p": p_value,
                    "positive_change_consistency": float(np.mean(array > 0)),
                }
            )
            p_locations.append(len(result) - 1)
            p_values.append(p_value)
    for index, q_value in zip(p_locations, bh_fdr(p_values)):
        result[index]["bh_fdr_q_all_numeric_tests"] = q_value
    return result


def baseline_gate(rows: list[dict[str, Any]], per_label: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for prompt_name in ("species_a", "species_b", "superclass", "vocalization"):
        prompt_rows = [
            row for row in rows
            if row["prompt_name"] == prompt_name and row["condition"] == "baseline"
            and row["truth_label"]
        ]
        for label in SUPPORTED_LABELS:
            values = [row for row in prompt_rows if row["label"] == label]
            if not values:
                continue
            accuracy = float(np.mean([bool(row["is_correct"]) for row in values]))
            threshold = 0.90 if prompt_name == "superclass" else 0.70
            status = "PASS" if len(values) == per_label and accuracy >= threshold else (
                "PENDING" if len(values) < per_label else "FAIL"
            )
            result.append(
                {
                    "prompt_name": prompt_name,
                    "label": label,
                    "n": len(values),
                    "accuracy": accuracy,
                    "threshold": threshold,
                    "status": status,
                }
            )
    return result


def technical_checks(
    rows: list[dict[str, Any]], selected: list[dict[str, Any]], lens: Any,
    token_rows: list[dict[str, Any]], phase: str, per_label: int,
    zero_hook_check: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = {label: sum(row["label"] == label for row in selected) for label in SUPPORTED_LABELS}
    finite = all(bool(row["finite_first_step_logits"]) for row in rows)
    generated = all(bool(str(row["generated_text"]).strip()) for row in rows)
    max_ratio = max((float(row["max_actual_delta_norm_ratio"]) for row in rows), default=0.0)
    expected_counts = per_label * len(SUPPORTED_LABELS)
    required_tokens = {"dog", "cat", "other", "animal", "bark", "sound", "vehicle"}
    passed_tokens = {row["concept"] for row in token_rows if row["status"] == "PASS"}
    audio_aligned = all(int(row["audio_count"]) > 0 for row in rows)
    return [
        {"check_name": "lens_metadata", "result": f"prompts={lens.n_prompts}, d_model={lens.d_model}", "status": "PASS" if lens.n_prompts == EXPECTED_LENS_PROMPTS and lens.d_model == EXPECTED_D_MODEL else "FAIL"},
        {"check_name": "canonical_final_layer_target", "result": f"lens target={LENS_TARGET_LAYER}; canonical final block={CANONICAL_FINAL_LAYER}", "status": "PENDING"},
        {"check_name": "manifest_labels", "result": json.dumps(labels, sort_keys=True), "status": "PASS" if len(selected) == expected_counts else "FAIL"},
        {"check_name": "required_single_tokens", "result": json.dumps(sorted(required_tokens & passed_tokens)), "status": "PASS" if required_tokens <= passed_tokens else "FAIL"},
        {"check_name": "zero_strength_hook_transparency", "result": f"max_abs_first_step_logit_error={zero_hook_check['max_abs_first_step_logit_error']}", "status": zero_hook_check["status"]},
        {"check_name": "audio_token_alignment", "result": f"rows={len(rows)}", "status": "PASS" if audio_aligned else "FAIL"},
        {"check_name": "finite_native_generation_logits", "result": f"rows={len(rows)}", "status": "PASS" if finite else "FAIL"},
        {"check_name": "nonempty_greedy_generation", "result": f"rows={len(rows)}", "status": "PASS" if generated else "FAIL"},
        {"check_name": "perturbation_magnitude", "result": f"maximum_actual_delta_norm_ratio={max_ratio:.6f}", "status": "PASS" if max_ratio <= MAX_ALLOWED_DELTA_RATIO else "FAIL"},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("d0", "smoke", "primary", "full"), default="smoke")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lens-path", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-label", type=int, default=None,
                        help="Samples per label; default: smoke=1, other phases=35.")
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("QWEN_OMNI_DEVICE", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_label is None:
        args.per_label = 1 if args.phase == "smoke" else 35
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")
    if args.layer not in SOURCE_LAYERS:
        raise ValueError(f"--layer must be one of {SOURCE_LAYERS}")
    phase_dir = args.output_dir / args.phase
    internal_dir = phase_dir / "internal"
    cache_dir = internal_dir / "cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    selected = select_manifest(manifest, args.phase, args.per_label, args.seed)
    write_csv(internal_dir / "selected_manifest.csv", selected)

    processor, container, thinker, lens, device, token_ids, token_rows = load_model_and_lens(args)
    write_csv(internal_dir / "concept_tokenization.csv", token_rows)
    directions, direction_rows = build_jspace_directions(
        thinker, lens, args.layer, token_ids, args.seed
    )
    write_csv(internal_dir / "direction_diagnostics.csv", direction_rows)

    config = {
        "script_version": SCRIPT_VERSION,
        "phase": args.phase,
        "model_path": str(args.model_path),
        "lens_path": str(args.lens_path),
        "lens_sha256": sha256_file(args.lens_path),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "output_dir": str(phase_dir),
        "system_prompt": SYSTEM_PROMPT,
        "prompts": {name: asdict(spec) for name, spec in PROMPTS.items()},
        "prompt_names_in_phase": prompt_names_for_phase(args.phase),
        "source_layers": SOURCE_LAYERS,
        "primary_layer": args.layer,
        "lens_target_layer": LENS_TARGET_LAYER,
        "canonical_final_layer": CANONICAL_FINAL_LAYER,
        "per_label": args.per_label,
        "selected_counts": {label: sum(row["label"] == label for row in selected) for label in SUPPORTED_LABELS},
        "max_new_tokens": args.max_new_tokens,
        "greedy_generation": True,
        "prefill_only_intervention": True,
        "seed": args.seed,
        "created_at": now_iso(),
    }
    config["run_signature"] = stable_hash(config)
    write_json(internal_dir / "frozen_config.json", config)

    zero_hook_check = zero_strength_transparency_check(
        selected[0],
        processor,
        thinker,
        device,
        directions,
        args.layer,
        args.max_new_tokens,
    )
    write_json(internal_dir / "zero_strength_hook_check.json", zero_hook_check)
    if zero_hook_check["status"] != "PASS":
        raise RuntimeError(
            "Zero-strength hook transparency check failed: "
            + json.dumps(zero_hook_check, ensure_ascii=False)
        )

    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jobs: list[tuple[dict[str, Any], PromptSpec, Condition]] = []
    for row in selected:
        for prompt_name in prompt_names_for_phase(args.phase):
            if prompt_name == "superclass" and row["label"] == "non_dog_bark":
                continue
            prompt = PROMPTS[prompt_name]
            for condition in condition_catalog(row["label"], prompt_name, args.phase):
                jobs.append((row, prompt, condition))

    for index, (row, prompt, condition) in enumerate(jobs, 1):
        cache_key = stable_hash(
            {
                "run_signature": config["run_signature"],
                "sample_id": row["sample_id"],
                "prompt_name": prompt.name,
                "condition": asdict(condition),
            }
        )
        cache_path = cache_dir / f"{cache_key}.json"
        print(
            f"[{index:04d}/{len(jobs):04d}] {row['sample_id']} "
            f"{prompt.name} {condition.name} strength={condition.strength:g}",
            flush=True,
        )
        try:
            if cache_path.is_file() and not args.no_resume:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("run_signature") == config["run_signature"]:
                    result = cached["result"]
                else:
                    result = run_one(
                        row, prompt, condition, processor, thinker, device,
                        token_ids, directions, args,
                    )
            else:
                result = run_one(
                    row, prompt, condition, processor, thinker, device,
                    token_ids, directions, args,
                )
            write_json(cache_path, {"run_signature": config["run_signature"], "result": result})
            all_results.append(result)
        except Exception as exc:
            failure = {
                "sample_id": row["sample_id"],
                "prompt_name": prompt.name,
                "condition": condition.name,
                "strength": condition.strength,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": now_iso(),
            }
            failures.append(failure)
            write_csv(phase_dir / "6D_failures.csv", failures)
            if not args.continue_on_error:
                raise
        finally:
            torch.cuda.empty_cache()

    write_csv(phase_dir / "6D_behavior_results.csv", all_results)
    behavior_summary = summarize_behavior(all_results)
    paired_summary = summarize_paired_numeric(all_results, args.seed)
    gate_rows = baseline_gate(all_results, args.per_label)
    checks = technical_checks(
        all_results,
        selected,
        lens,
        token_rows,
        args.phase,
        args.per_label,
        zero_hook_check,
    )
    required_failed = any(row["status"] == "FAIL" for row in checks)
    write_csv(phase_dir / "6D_behavior_summary.csv", behavior_summary)
    write_csv(phase_dir / "6D_paired_numeric_summary.csv", paired_summary)
    write_csv(phase_dir / "6D_baseline_gate.csv", gate_rows)
    write_csv(phase_dir / "6D_technical_check.csv", checks)
    write_json(
        phase_dir / "6D_summary.json",
        {
            "script_version": SCRIPT_VERSION,
            "phase": args.phase,
            "samples": len(selected),
            "jobs": len(jobs),
            "result_rows": len(all_results),
            "failure_rows": len(failures),
            "behavior_summary_rows": len(behavior_summary),
            "paired_numeric_summary_rows": len(paired_summary),
            "technical_status": "FAIL" if required_failed or failures else "PASS",
            "zero_strength_hook_check": zero_hook_check,
            "baseline_gate": gate_rows,
            "completed_at": now_iso(),
        },
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
