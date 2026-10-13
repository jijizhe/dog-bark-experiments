"""Experiments 5A/5B/5C: causal J-space interventions for Qwen2.5-Omni-7B.

The script uses the frozen Jacobian Lens fitted in experiment 4A and intervenes
on the Thinker residual stream after a selected decoder block:

  A. ablate a dog/bark J-lens direction on dog-bark audio;
  B. inject a dog/bark J-lens direction on non-dog audio;
  C. swap dog and cat J-lens coordinates while preserving the orthogonal
     residual component.

It records *actual model decision logits* after the remaining decoder blocks,
not J-lens readouts.  Thus a successful, concept-specific and layer-specific
effect is causal evidence that downstream computation reads these directions.

Required manifest columns:
    sample_id,audio_path,label,source

Recognised labels are ``dog_bark``, ``cat_vocalization`` and
``non_dog_bark``.  Experiment C can run dog->cat without cat examples, but a
reciprocal cat->dog result requires cat_vocalization rows.

Typical commands:

    # Two/three-sample engineering test.
    python intervene_bark_concepts.py --phase smoke

    # Dose-response at the preregistered primary layer (24).
    python intervene_bark_concepts.py --phase dose

    # Layer-specificity using the fixed primary intervention strengths.
    python intervene_bark_concepts.py --phase layer

Paths and limits may also be overridden with QWEN_CAUSAL_* environment
variables.  Run ``--help`` for details.
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
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info

try:
    import jlens
    from jlens.hf import Layout
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "jlens is not installed. Activate the same environment used by 4A/4B "
        "and install anthropics/jacobian-lens."
    ) from exc


# Frozen model/lens configuration from 4A.
MODEL_PATH = Path(os.environ.get(
    "QWEN_CAUSAL_MODEL_PATH",
    str(paths.QWEN_MODEL),
))
LENS_PATH = Path(os.environ.get(
    "QWEN_CAUSAL_LENS_PATH",
    str(paths.LENS),
))
DEFAULT_MANIFEST = Path(os.environ.get(
    "QWEN_CAUSAL_MANIFEST",
    str(paths.MANIFEST),
))
DEFAULT_OUTPUT = Path(os.environ.get(
    "QWEN_CAUSAL_OUTPUT_DIR",
    str(paths.OUTPUTS_ROOT / "intervene_bark_concepts"),
))

SOURCE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24]
TARGET_LAYER = 26
PRIMARY_LAYER = 24
EXPECTED_N_PROMPTS = 1000
EXPECTED_D_MODEL = 3584
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))
SEED = 20260826

PROMPT = (
    "Identify the primary source in this audio. "
    "Answer with one category only, without explanation."
)
REQUIRED_CONCEPTS = [
    "dog", "bark", "cat", "animal", "sound", "noise", "vehicle", "bird",
]
ANSWER_CONCEPTS = REQUIRED_CONCEPTS

# The primary paper analysis is deliberately small.  Dose selection must be
# completed at L24 before the layer phase is interpreted.
ABLATION_BETAS = [0.5, 1.0, 2.0]
INJECTION_NORM_RATIOS = [0.01, 0.03, 0.10]
SWAP_MULTIPLIERS = [0.5, 1.0, 1.5]
PRIMARY_ABLATION_BETA = 1.0
PRIMARY_INJECTION_RATIO = 0.03
PRIMARY_SWAP_MULTIPLIER = 1.0
LAYER_PHASE_LAYERS = [0, 8, 16, 20, 22, 24]

ALLOWED_LABELS = {"dog_bark", "cat_vocalization", "non_dog_bark"}
ALIGNMENT_MAX_ABS_TOL = 1e-3
ALIGNMENT_MIN_COSINE = 0.999
BOOTSTRAP_REPEATS = 2000
PERMUTATION_REPEATS = 10000


@dataclass(frozen=True)
class Intervention:
    experiment: str
    condition: str
    layer: int
    position_mode: str
    operation: str
    source_concept: str
    target_concept: str
    strength: float
    expected_direction: int


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


def model_config_value(model: Any, name: str) -> Any:
    for candidate in (model.config, getattr(model.config, "thinker_config", None)):
        if candidate is not None and hasattr(candidate, name):
            return getattr(candidate, name)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
        if hasattr(text_config, name):
            return getattr(text_config, name)
    raise AttributeError(f"Cannot find {name} in model config")


def resolve_special_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not contain required special token {token!r}")
    return int(token_id)


def resolve_concepts(tokenizer: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    token_ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for concept in REQUIRED_CONCEPTS:
        variants = [f" {concept}", concept, concept.capitalize()]
        chosen: tuple[str, int] | None = None
        encoded_rows = []
        for variant in variants:
            encoded = [int(item) for item in tokenizer(
                variant, add_special_tokens=False
            )["input_ids"]]
            encoded_rows.append({"text": variant, "token_ids": encoded})
            if chosen is None and len(encoded) == 1:
                chosen = (variant, encoded[0])
        if chosen is None:
            rows.append({
                "concept": concept,
                "status": "FAIL",
                "variants": json.dumps(encoded_rows, ensure_ascii=False),
            })
        else:
            token_ids[concept] = chosen[1]
            rows.append({
                "concept": concept,
                "chosen_text": chosen[0],
                "token_id": chosen[1],
                "status": "PASS",
                "variants": json.dumps(encoded_rows, ensure_ascii=False),
            })
    missing = sorted(set(REQUIRED_CONCEPTS) - set(token_ids))
    if missing:
        raise RuntimeError(f"Required concepts are not single tokens: {missing}")
    return token_ids, rows


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Required columns are "
            "sample_id,audio_path,label,source."
        )
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "audio_path", "label", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    rows = frame.to_dict("records")
    if not rows:
        raise ValueError("Manifest is empty")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest sample_id values are not unique")
    unexpected = sorted({row["label"] for row in rows} - ALLOWED_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected manifest labels: {unexpected}")
    missing_audio = [row["audio_path"] for row in rows if not Path(row["audio_path"]).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Missing audio files, first five: {missing_audio[:5]}")
    return rows


def balanced_limit(
    rows: list[dict[str, str]], per_label: int | None
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(SEED)
    selected: list[dict[str, str]] = []
    for label in sorted(grouped):
        values = sorted(grouped[label], key=lambda row: row["sample_id"])
        rng.shuffle(values)
        selected.extend(values if per_label is None else values[:per_label])
    rng.shuffle(selected)
    return selected


def load_model_and_lens() -> tuple[Any, Any, Any, Any, Any, torch.device, dict[str, int], list[dict[str, Any]]]:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not LENS_PATH.is_file():
        raise FileNotFoundError(f"Frozen Jacobian Lens not found: {LENS_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    container = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
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
    thinker.config.use_cache = False
    lens_model = jlens.from_hf(
        thinker, processor.tokenizer, layout=Layout(path="model"), force_bos=True
    )
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    if lens.source_layers != SOURCE_LAYERS:
        raise ValueError(f"Lens source layers {lens.source_layers} != {SOURCE_LAYERS}")
    if lens.n_prompts != EXPECTED_N_PROMPTS or lens.d_model != EXPECTED_D_MODEL:
        raise ValueError(
            f"Unexpected lens metadata: n_prompts={lens.n_prompts}, "
            f"d_model={lens.d_model}"
        )
    concept_ids, concept_rows = resolve_concepts(processor.tokenizer)
    return processor, container, thinker, lens_model, lens, device, concept_ids, concept_rows


def prepare_audio_inputs(processor: Any, audio_path: str) -> dict[str, Any]:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": PROMPT},
        ],
    }]
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
    runs = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current != previous + 1:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))
    return runs


def position_sets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_token_id: int,
    audio_bos_id: int,
    audio_eos_id: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    valid_len = int(attention_mask.long().sum().item())
    ids = input_ids[0, :valid_len].detach().cpu().tolist()
    audio_positions = [i for i, token_id in enumerate(ids) if token_id == audio_token_id]
    runs = contiguous_runs(audio_positions)
    if len(runs) != 1:
        raise ValueError(f"Expected one contiguous audio block, found {runs}")
    audio_start, audio_stop = runs[0]
    if audio_start == 0 or ids[audio_start - 1] != audio_bos_id:
        raise ValueError("Audio block is not preceded by audio_bos")
    eos_candidates = [i for i in range(audio_stop, valid_len) if ids[i] == audio_eos_id]
    if not eos_candidates:
        raise ValueError("Could not find audio_eos after the audio block")
    audio_eos_index = eos_candidates[0]
    decision_index = valid_len - 1
    post_audio = list(range(audio_eos_index + 1, decision_index)) or [audio_eos_index]
    middle_start = audio_start + (audio_stop - audio_start) // 3
    middle_stop = audio_start + 2 * (audio_stop - audio_start) // 3
    positions = {
        "decision": [decision_index],
        "audio_middle": list(range(middle_start, max(middle_start + 1, middle_stop))),
        "post_audio": post_audio,
        "audio_all": list(range(audio_start, audio_stop)),
    }
    return positions, {
        "valid_sequence_length": valid_len,
        "audio_start": audio_start,
        "audio_stop_exclusive": audio_stop,
        "audio_count": audio_stop - audio_start,
        "audio_bos_index": audio_start - 1,
        "audio_eos_index": audio_eos_index,
        "decision_index": decision_index,
    }


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
        raise ValueError("Direction has zero or non-finite norm")
    return vector / norm


def build_directions(
    lens_model: Any,
    lens: Any,
    token_ids: dict[str, int],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Build source-layer J-lens vectors v_t = J_l.T @ W_U[t]."""
    unembedding = lens_model._lm_head.weight.detach().float().cpu()
    directions: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for layer in SOURCE_LAYERS:
        J = lens.jacobians[layer].detach().float().cpu()
        layer_directions: dict[str, torch.Tensor] = {}
        for concept, token_id in token_ids.items():
            raw_vector = J.T @ unembedding[token_id]
            vector = unit(raw_vector)
            # Unit vectors make ablation and norm-ratio injection comparable
            # across concepts.  Coordinate swaps use the unnormalised J-lens
            # vectors, matching V=[v_source, v_target] in the paper.
            layer_directions[concept] = vector
            layer_directions[f"raw::{concept}"] = raw_vector
            rows.append({
                "layer": layer,
                "concept": concept,
                "token_id": token_id,
                "raw_norm": float(raw_vector.norm().item()),
                "finite": bool(torch.isfinite(vector).all().item()),
            })
        joint = unit(layer_directions["dog"] + layer_directions["bark"])
        layer_directions["dog_bark_joint"] = joint

        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED + layer)
        random_vector = torch.randn(EXPECTED_D_MODEL, generator=generator)
        # A random control orthogonal to all named directions avoids accidental
        # overlap while keeping exactly the same perturbation rule and norm.
        basis_names = [*token_ids.keys(), "dog_bark_joint"]
        basis = torch.stack([layer_directions[name] for name in basis_names], dim=1)
        random_vector = random_vector - basis @ torch.linalg.lstsq(
            basis, random_vector
        ).solution
        layer_directions["random_orthogonal"] = unit(random_vector)
        directions[layer] = layer_directions

        for left, right in (("dog", "cat"), ("dog", "bark"), ("dog", "vehicle")):
            raw_pair = torch.stack([
                layer_directions[f"raw::{left}"],
                layer_directions[f"raw::{right}"],
            ], dim=1)
            singular_values = torch.linalg.svdvals(raw_pair)
            rows.append({
                "layer": layer,
                "concept": f"cosine:{left}:{right}",
                "token_id": "",
                "raw_norm": "",
                "finite": True,
                "cosine": float(torch.dot(
                    layer_directions[left], layer_directions[right]
                ).item()),
                "pair_condition_number": float(
                    (singular_values.max() / singular_values.min().clamp_min(1e-12)).item()
                ),
            })
    return directions, rows


def interventions_for(phase: str, label: str) -> list[Intervention]:
    layers = [PRIMARY_LAYER] if phase in {"smoke", "dose"} else LAYER_PHASE_LAYERS
    values: list[Intervention] = []
    for layer in layers:
        if label == "dog_bark":
            ablation_betas = ABLATION_BETAS if phase in {"smoke", "dose"} else [PRIMARY_ABLATION_BETA]
            for beta in ablation_betas:
                values.append(Intervention(
                    "A_ablation", "dog_ablate", layer, "decision", "ablate",
                    "dog", "", beta, -1,
                ))
            if phase in {"smoke", "dose"}:
                for concept in ("bark", "dog_bark_joint", "vehicle", "random_orthogonal"):
                    values.append(Intervention(
                        "A_ablation", f"{concept}_ablate", layer, "decision", "ablate",
                        concept, "", PRIMARY_ABLATION_BETA,
                        -1 if concept in {"dog", "bark", "dog_bark_joint"} else 0,
                    ))
            else:
                for concept in ("vehicle", "random_orthogonal"):
                    values.append(Intervention(
                        "A_ablation", f"{concept}_ablate", layer, "decision", "ablate",
                        concept, "", PRIMARY_ABLATION_BETA, 0,
                    ))
            if phase in {"smoke", "dose"}:
                for position_mode in ("audio_middle", "post_audio"):
                    values.append(Intervention(
                        "A_ablation", f"dog_ablate_{position_mode}", layer,
                        position_mode, "ablate", "dog", "",
                        PRIMARY_ABLATION_BETA, -1,
                    ))

        if label == "non_dog_bark":
            ratios = INJECTION_NORM_RATIOS if phase in {"smoke", "dose"} else [PRIMARY_INJECTION_RATIO]
            for ratio in ratios:
                values.append(Intervention(
                    "B_injection", "dog_inject", layer, "decision", "inject",
                    "dog", "", ratio, 1,
                ))
            if phase in {"smoke", "dose"}:
                for concept in ("bark", "dog_bark_joint", "vehicle", "random_orthogonal"):
                    values.append(Intervention(
                        "B_injection", f"{concept}_inject", layer, "decision", "inject",
                        concept, "", PRIMARY_INJECTION_RATIO,
                        1 if concept in {"dog", "bark", "dog_bark_joint"} else 0,
                    ))
            else:
                for concept in ("vehicle", "random_orthogonal"):
                    values.append(Intervention(
                        "B_injection", f"{concept}_inject", layer, "decision", "inject",
                        concept, "", PRIMARY_INJECTION_RATIO, 0,
                    ))
            if phase in {"smoke", "dose"}:
                for position_mode in ("audio_middle", "post_audio"):
                    values.append(Intervention(
                        "B_injection", f"dog_inject_{position_mode}", layer,
                        position_mode, "inject", "dog", "",
                        PRIMARY_INJECTION_RATIO, 1,
                    ))

        if label in {"dog_bark", "cat_vocalization"}:
            multipliers = SWAP_MULTIPLIERS if phase in {"smoke", "dose"} else [PRIMARY_SWAP_MULTIPLIER]
            for multiplier in multipliers:
                values.append(Intervention(
                    "C_swap", "dog_cat_swap", layer, "decision", "swap",
                    "dog", "cat", multiplier,
                    -1 if label == "dog_bark" else 1,
                ))
            if phase in {"smoke", "dose"}:
                values.append(Intervention(
                    "C_swap", "sound_vehicle_swap", layer, "decision", "swap",
                    "sound", "vehicle", PRIMARY_SWAP_MULTIPLIER, 0,
                ))
    return values


def apply_intervention(
    hidden: torch.Tensor,
    positions: list[int],
    specification: Intervention,
    directions: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    changed = hidden.clone()
    selected = changed[0, positions, :].float()
    before_norm = selected.norm(dim=-1)
    if specification.operation == "ablate":
        direction = directions[specification.source_concept].to(selected.device)
        coefficients = selected @ direction
        delta = -specification.strength * coefficients[:, None] * direction[None, :]
    elif specification.operation == "inject":
        direction = directions[specification.source_concept].to(selected.device)
        # ``strength`` is the perturbation norm divided by each residual norm.
        delta = specification.strength * before_norm[:, None] * direction[None, :]
    elif specification.operation == "swap":
        source = directions[f"raw::{specification.source_concept}"].to(selected.device)
        target = directions[f"raw::{specification.target_concept}"].to(selected.device)
        V = torch.stack([source, target], dim=1)  # [d_model, 2]
        pinv = torch.linalg.pinv(V)               # [2, d_model]
        coordinates = selected @ pinv.T
        swapped = coordinates[:, [1, 0]]
        delta = specification.strength * ((swapped - coordinates) @ V.T)
    else:
        raise ValueError(f"Unknown operation {specification.operation}")
    after = selected + delta
    changed[0, positions, :] = after.to(changed.dtype)
    actual_delta = changed[0, positions, :].float() - hidden[0, positions, :].float()
    ratio = actual_delta.norm(dim=-1) / before_norm.clamp_min(1e-12)
    return changed, {
        "selected_positions": float(len(positions)),
        "mean_hidden_norm": float(before_norm.mean().item()),
        "mean_delta_norm": float(actual_delta.norm(dim=-1).mean().item()),
        "mean_actual_delta_norm_ratio": float(ratio.mean().item()),
        "max_actual_delta_norm_ratio": float(ratio.max().item()),
    }


def score_logits(
    logits: torch.Tensor,
    token_ids: dict[str, int],
    tokenizer: Any,
) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    top_ids = torch.topk(logits, k=10).indices.tolist()
    candidate_values = torch.stack([logits[token_ids[c]] for c in ANSWER_CONCEPTS])
    candidate_probabilities = torch.softmax(candidate_values, dim=0)
    result: dict[str, Any] = {
        "global_top1_id": int(top_ids[0]),
        "global_top1_token": tokenizer.decode([int(top_ids[0])]),
        "global_top10_tokens": json.dumps(
            [tokenizer.decode([int(token_id)]) for token_id in top_ids],
            ensure_ascii=False,
        ),
    }
    for index, concept in enumerate(ANSWER_CONCEPTS):
        value = float(logits[token_ids[concept]].item())
        result[f"{concept}_logit"] = value
        result[f"{concept}_rank"] = 1 + int((logits > value).sum().item())
        result[f"{concept}_candidate_probability"] = float(candidate_probabilities[index].item())
    result["dog_minus_cat_logit"] = result["dog_logit"] - result["cat_logit"]
    result["dog_minus_sound_noise_logit"] = result["dog_logit"] - max(
        result["sound_logit"], result["noise_logit"]
    )
    result["candidate_top1"] = ANSWER_CONCEPTS[int(candidate_values.argmax().item())]
    return result


def run_forward(
    thinker: Any,
    inputs: dict[str, Any],
    decision_index: int,
    positions_by_mode: dict[str, list[int]],
    specification: Intervention | None,
    layer_directions: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    handle = None
    diagnostics: dict[str, float] = {
        "selected_positions": 0.0,
        "mean_hidden_norm": 0.0,
        "mean_delta_norm": 0.0,
        "mean_actual_delta_norm_ratio": 0.0,
        "max_actual_delta_norm_ratio": 0.0,
    }
    if specification is not None:
        if layer_directions is None:
            raise ValueError("Directions are required for an intervention")
        positions = positions_by_mode[specification.position_mode]

        def intervention_hook(_module: Any, _module_inputs: Any, output: Any) -> Any:
            nonlocal diagnostics
            hidden = output_hidden(output)
            changed, diagnostics = apply_intervention(
                hidden, positions, specification, layer_directions
            )
            return replace_output_hidden(output, changed)

        handle = thinker.model.layers[specification.layer].register_forward_hook(
            intervention_hook
        )
    try:
        with torch.inference_mode():
            outputs = thinker(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                use_audio_in_video=False,
            )
        logits = outputs.logits[0, decision_index].detach()
        del outputs
        return logits, diagnostics
    finally:
        if handle is not None:
            handle.remove()


def check_audio_alignment(
    thinker: Any,
    inputs: dict[str, Any],
    positions: dict[str, list[int]],
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}

    def audio_hook(_module: Any, _args: Any, output: Any) -> Any:
        captured["audio"] = output_hidden(output).detach()
        return output

    def model_pre_hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        captured["merged"] = kwargs["inputs_embeds"].detach()

    handles = [
        thinker.audio_tower.register_forward_hook(audio_hook),
        thinker.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            thinker(
                **inputs, use_cache=False, return_dict=True,
                output_hidden_states=False, use_audio_in_video=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    audio = captured["audio"]
    merged = captured["merged"][0, positions["audio_all"], :]
    if audio.ndim != 2 or audio.shape != merged.shape:
        raise ValueError(f"Audio alignment shape mismatch: {audio.shape} vs {merged.shape}")
    difference = (audio.to(merged.dtype).float() - merged.float()).abs()
    cosine = F.cosine_similarity(audio.float(), merged.float(), dim=-1).mean()
    return {
        "alignment_max_abs_error": float(difference.max().item()),
        "alignment_mean_abs_error": float(difference.mean().item()),
        "alignment_mean_cosine": float(cosine.item()),
        "alignment_status": "PASS" if (
            float(difference.max().item()) <= ALIGNMENT_MAX_ABS_TOL
            and float(cosine.item()) >= ALIGNMENT_MIN_COSINE
        ) else "FAIL",
    }


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    intervention_frame = frame[frame["experiment"] != "baseline"].copy()
    baseline = frame[frame["experiment"] == "baseline"].set_index("sample_id")
    metrics = [
        "dog_logit", "bark_logit", "cat_logit", "dog_minus_cat_logit",
        "dog_minus_sound_noise_logit", "dog_candidate_probability",
    ]
    rng = np.random.default_rng(SEED + 5000)
    output: list[dict[str, Any]] = []
    group_columns = [
        "label", "experiment", "condition", "layer", "position_mode",
        "operation", "source_concept", "target_concept", "strength",
        "expected_direction",
    ]
    for keys, group in intervention_frame.groupby(group_columns, dropna=False):
        common = [sample_id for sample_id in group["sample_id"] if sample_id in baseline.index]
        if not common:
            continue
        group_by_sample = group.set_index("sample_id")
        base_row = dict(zip(group_columns, keys))
        for metric in metrics:
            delta = np.asarray([
                float(group_by_sample.loc[sample_id, metric])
                - float(baseline.loc[sample_id, metric])
                for sample_id in common
            ], dtype=np.float64)
            if len(delta) >= 2:
                bootstrap = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
                for index in range(BOOTSTRAP_REPEATS):
                    bootstrap[index] = delta[
                        rng.integers(0, len(delta), len(delta))
                    ].mean()
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                # Paired random-sign permutation test for a zero mean effect.
                signs = rng.choice(
                    [-1.0, 1.0], size=(PERMUTATION_REPEATS, len(delta))
                )
                permuted = (signs * delta[None, :]).mean(axis=1)
                p_value = (1.0 + np.sum(np.abs(permuted) >= abs(delta.mean()))) / (
                    PERMUTATION_REPEATS + 1.0
                )
            else:
                low = high = p_value = float("nan")
            # Directional consistency is reported only where the intervention
            # makes a preregistered prediction for this exact metric.  It would
            # be misleading, for example, to expect cat_logit itself to fall
            # merely because dog was ablated.
            expected = 0
            experiment = str(base_row["experiment"])
            source = str(base_row["source_concept"])
            if experiment == "A_ablation":
                if source == "dog" and metric in {
                    "dog_logit", "dog_minus_cat_logit",
                    "dog_minus_sound_noise_logit", "dog_candidate_probability",
                }:
                    expected = -1
                elif source == "bark" and metric == "bark_logit":
                    expected = -1
                elif source == "dog_bark_joint" and metric in {
                    "dog_logit", "bark_logit", "dog_minus_cat_logit",
                    "dog_minus_sound_noise_logit", "dog_candidate_probability",
                }:
                    expected = -1
            elif experiment == "B_injection":
                if source == "dog" and metric in {
                    "dog_logit", "dog_minus_cat_logit",
                    "dog_minus_sound_noise_logit", "dog_candidate_probability",
                }:
                    expected = 1
                elif source == "bark" and metric == "bark_logit":
                    expected = 1
                elif source == "dog_bark_joint" and metric in {
                    "dog_logit", "bark_logit", "dog_minus_cat_logit",
                    "dog_minus_sound_noise_logit", "dog_candidate_probability",
                }:
                    expected = 1
            elif experiment == "C_swap" and str(base_row["condition"]) == "dog_cat_swap":
                if metric == "dog_minus_cat_logit":
                    expected = -1 if str(base_row["label"]) == "dog_bark" else 1
            if expected > 0:
                consistency = float(np.mean(delta > 0))
            elif expected < 0:
                consistency = float(np.mean(delta < 0))
            else:
                consistency = float("nan")
            output.append({
                **base_row,
                "metric": metric,
                "metric_expected_direction": expected,
                "n": len(delta),
                "baseline_mean": float(np.mean([
                    baseline.loc[sample_id, metric] for sample_id in common
                ])),
                "intervention_mean": float(np.mean([
                    group_by_sample.loc[sample_id, metric] for sample_id in common
                ])),
                "mean_paired_change": float(delta.mean()),
                "median_paired_change": float(np.median(delta)),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "two_sided_sign_permutation_p": float(p_value),
                "expected_direction_consistency": consistency,
            })
    # Benjamini-Hochberg correction over every finite confirmatory comparison
    # emitted by this run.  The raw p-value remains available for transparency.
    finite_indices = [
        index for index, row in enumerate(output)
        if math.isfinite(float(row["two_sided_sign_permutation_p"]))
    ]
    if finite_indices:
        ordered = sorted(
            finite_indices,
            key=lambda index: float(output[index]["two_sided_sign_permutation_p"]),
        )
        adjusted = [1.0] * len(ordered)
        running = 1.0
        total = len(ordered)
        for reverse_index in range(total - 1, -1, -1):
            output_index = ordered[reverse_index]
            rank = reverse_index + 1
            candidate = min(
                1.0,
                float(output[output_index]["two_sided_sign_permutation_p"])
                * total / rank,
            )
            running = min(running, candidate)
            adjusted[reverse_index] = running
        for output_index, q_value in zip(ordered, adjusted):
            output[output_index]["bh_fdr_q_all_reported_tests"] = q_value
    for row in output:
        row.setdefault("bh_fdr_q_all_reported_tests", float("nan"))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "dose", "layer"], default="smoke")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-label",
        type=int,
        default=None,
        help="Maximum samples per label; defaults to 1 for smoke, 20 for dose, all for layer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_label = args.per_label
    if per_label is None:
        per_label = {"smoke": 1, "dose": 20, "layer": None}[args.phase]
    if per_label is not None and per_label < 1:
        raise ValueError("--per-label must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    output_dir = args.output_dir / args.phase
    internal_dir = output_dir / "internal"
    cache_dir = internal_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    selected = balanced_limit(manifest, per_label)
    write_csv(internal_dir / "selected_manifest.csv", selected)

    processor, container, thinker, lens_model, lens, device, concept_ids, concept_rows = load_model_and_lens()
    tokenizer = lens_model.tokenizer
    directions, direction_rows = build_directions(lens_model, lens, concept_ids)
    write_csv(internal_dir / "concept_tokenization.csv", concept_rows)
    write_csv(internal_dir / "direction_diagnostics.csv", direction_rows)

    audio_token_id = int(model_config_value(thinker, "audio_token_id"))
    audio_bos_id = resolve_special_id(tokenizer, "<|audio_bos|>")
    audio_eos_id = resolve_special_id(tokenizer, "<|audio_eos|>")
    selected_counts = {
        str(label): int(count)
        for label, count in pd.Series(
            [row["label"] for row in selected], dtype="object"
        ).value_counts().items()
    }
    config = {
        "script_version": "5abc-1.1",
        "phase": args.phase,
        "model_path": str(MODEL_PATH),
        "lens_path": str(LENS_PATH),
        "lens_sha256": sha256_file(LENS_PATH),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "output_dir": str(output_dir),
        "source_layers": SOURCE_LAYERS,
        "target_layer": TARGET_LAYER,
        "canonical_final_decoder_layer": 27,
        "primary_layer": PRIMARY_LAYER,
        "prompt": PROMPT,
        "per_label": per_label,
        # Convert pandas/numpy scalar counts to builtin int so the frozen run
        # signature is portable across Python JSON implementations.
        "selected_counts": selected_counts,
        "ablation_betas": ABLATION_BETAS,
        "injection_norm_ratios": INJECTION_NORM_RATIOS,
        "swap_multipliers": SWAP_MULTIPLIERS,
        "seed": SEED,
        "created_at": now_iso(),
    }
    signature_payload = {key: value for key, value in config.items() if key != "created_at"}
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic) else str(value),
        ).encode("utf-8")
    ).hexdigest()
    config["run_signature"] = signature
    write_json(internal_dir / "frozen_config.json", config)

    technical: list[dict[str, Any]] = [
        {
            "check_name": "lens_metadata",
            "result": f"prompts={lens.n_prompts}, d_model={lens.d_model}",
            "status": "PASS",
        },
        {
            "check_name": "canonical_final_layer_target",
            "result": (
                f"4A was fitted to decoder layer {TARGET_LAYER}; the canonical "
                "J-lens target for a 28-block Thinker is layer 27"
            ),
            "status": "PENDING" if TARGET_LAYER != 27 else "PASS",
        },
        {
            "check_name": "manifest_labels",
            "result": json.dumps(config["selected_counts"], ensure_ascii=False),
            "status": "PASS",
        },
        {
            "check_name": "reciprocal_cat_swap_available",
            "result": f"cat_samples={config['selected_counts'].get('cat_vocalization', 0)}",
            "status": "PASS" if config["selected_counts"].get("cat_vocalization", 0) else "PENDING",
        },
    ]

    result_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    hook_transparency_error: float | None = None
    dtype = next(thinker.parameters()).dtype
    for sample_index, sample in enumerate(selected, start=1):
        raw_inputs = prepare_audio_inputs(processor, sample["audio_path"])
        inputs = move_inputs_to_device(raw_inputs, device, dtype)
        attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        positions, position_metadata = position_sets(
            inputs["input_ids"], attention_mask,
            audio_token_id, audio_bos_id, audio_eos_id,
        )
        alignment = check_audio_alignment(thinker, inputs, positions)
        alignment_rows.append({
            "sample_id": sample["sample_id"], "label": sample["label"],
            **position_metadata, **alignment,
        })
        if alignment["alignment_status"] != "PASS":
            raise RuntimeError(f"Audio alignment failed for {sample['sample_id']}")

        baseline_logits, baseline_diagnostics = run_forward(
            thinker, inputs, position_metadata["decision_index"], positions, None, None
        )
        baseline_scores = score_logits(baseline_logits, concept_ids, tokenizer)
        result_rows.append({
            **sample,
            "phase": args.phase,
            "experiment": "baseline",
            "condition": "baseline",
            "layer": -1,
            "position_mode": "none",
            "operation": "none",
            "source_concept": "",
            "target_concept": "",
            "strength": 0.0,
            "expected_direction": 0,
            **baseline_diagnostics,
            **baseline_scores,
        })

        if hook_transparency_error is None:
            zero_specification = Intervention(
                "technical", "zero_strength_hook", PRIMARY_LAYER, "decision",
                "inject", "dog", "", 0.0, 0,
            )
            zero_logits, _ = run_forward(
                thinker,
                inputs,
                position_metadata["decision_index"],
                positions,
                zero_specification,
                directions[PRIMARY_LAYER],
            )
            hook_transparency_error = float(
                (baseline_logits.float() - zero_logits.float()).abs().max().item()
            )
            del zero_logits

        sample_interventions = interventions_for(args.phase, sample["label"])
        for intervention_index, specification in enumerate(sample_interventions, start=1):
            cache_payload = {
                "signature": signature,
                "sample_id": sample["sample_id"],
                "intervention": asdict(specification),
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            cache_path = cache_dir / f"{sample['sample_id']}__{cache_key}.json"
            cached = None
            if cache_path.is_file():
                candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                if candidate.get("cache_payload") == cache_payload:
                    cached = candidate["result"]
            if cached is None:
                logits, diagnostics = run_forward(
                    thinker,
                    inputs,
                    position_metadata["decision_index"],
                    positions,
                    specification,
                    directions[specification.layer],
                )
                scores = score_logits(logits, concept_ids, tokenizer)
                cached = {
                    **sample,
                    "phase": args.phase,
                    **asdict(specification),
                    **diagnostics,
                    **scores,
                }
                write_json(cache_path, {"cache_payload": cache_payload, "result": cached})
            result_rows.append(cached)
            print(
                f"[{sample_index:03d}/{len(selected):03d}] {sample['sample_id']} "
                f"{intervention_index:02d}/{len(sample_interventions):02d} "
                f"{specification.condition} L{specification.layer} "
                f"strength={specification.strength:g}",
                flush=True,
            )
        del inputs, raw_inputs, baseline_logits
        torch.cuda.empty_cache()

    write_csv(internal_dir / "audio_alignment.csv", alignment_rows)
    write_csv(output_dir / "5ABC_intervention_results.csv", result_rows)
    summary_rows = paired_summary(result_rows)
    write_csv(output_dir / "5ABC_paired_effect_summary.csv", summary_rows)

    alignment_pass = bool(alignment_rows) and all(
        row["alignment_status"] == "PASS" for row in alignment_rows
    )
    finite_columns = [column for column in pd.DataFrame(result_rows).columns if column.endswith("_logit")]
    finite_pass = all(
        math.isfinite(float(row[column]))
        for row in result_rows for column in finite_columns
    )
    max_ratio = max(float(row["max_actual_delta_norm_ratio"]) for row in result_rows)
    technical.extend([
        {
            "check_name": "zero_strength_hook_transparency",
            "result": f"max_abs_logit_error={hook_transparency_error}",
            "status": "PASS" if hook_transparency_error == 0.0 else "FAIL",
        },
        {
            "check_name": "audio_alignment",
            "result": f"{sum(row['alignment_status'] == 'PASS' for row in alignment_rows)}/{len(alignment_rows)}",
            "status": "PASS" if alignment_pass else "FAIL",
        },
        {
            "check_name": "finite_decision_logits",
            "result": f"rows={len(result_rows)}, columns={len(finite_columns)}",
            "status": "PASS" if finite_pass else "FAIL",
        },
        {
            "check_name": "perturbation_magnitude",
            "result": f"maximum_actual_delta_norm_ratio={max_ratio:.6g}",
            "status": "PASS" if math.isfinite(max_ratio) and max_ratio <= 0.50 else "FAIL",
        },
    ])
    failed = [row for row in technical if row["status"] == "FAIL"]
    technical.append({
        "check_name": "technical_overall",
        "result": "all required checks passed" if not failed else f"failed={len(failed)}",
        "status": "PASS" if not failed else "FAIL",
    })
    write_csv(output_dir / "5ABC_technical_check.csv", technical)
    write_json(output_dir / "5ABC_summary.json", {
        "phase": args.phase,
        "samples": len(selected),
        "result_rows": len(result_rows),
        "paired_summary_rows": len(summary_rows),
        "labels": config["selected_counts"],
        "reciprocal_cat_swap_available": bool(config["selected_counts"].get("cat_vocalization", 0)),
        "technical_status": technical[-1]["status"],
        "completed_at": now_iso(),
    })
    print(json.dumps(technical, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure_dir = DEFAULT_OUTPUT
        # Use the parsed output directory when available; otherwise use
        # the environment-configured default.
        try:
            failure_dir = parse_args().output_dir
        except Exception:
            pass
        failure_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(failure_dir / "failure.txt", traceback.format_exc())
        raise
