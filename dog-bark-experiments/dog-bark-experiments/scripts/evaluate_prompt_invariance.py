"""Experiment 7E: prompt/task invariance of the dog J-space concept.

Question
--------
Does a dog-bark waveform evoke the internal ``dog`` direction even when the
prompt asks an unrelated question, or is the apparent concept mainly induced
by dog-related words/tasks in the prompt?

The experiment keeps the audio, model, L26-target J-lens, source layer, and
decision-token readout fixed while changing only the prompt.  It records:

1. the L24 decision residual's coordinate along the J-space dog direction;
2. dog-vs-cat coordinates in the joint dog/cat span;
3. native final-layer first-step dog logits/ranks;
4. the model's greedy answer.

Prompt families
---------------
* target-free unrelated tasks: acknowledge, loudness, event count;
* relevant species task: dog/cat/other (positive control);
* matched lexical controls: dog-barking, cat-meowing, vehicle-sound topic
  labels followed by the identical loudness task;
* optional text-only versions of the matched topic-label prompts.

The main scientific contrasts are:

* spontaneous audio evidence: dog audio versus all non-dog audio under each
  target-free unrelated task;
* lexical induction: within the same audio, dog-topic prompt minus the matched
  vehicle-topic prompt;
* text-only induction: dog-topic text minus vehicle-topic text without audio.

Expected manifest columns:
    sample_id,audio_path,label,source

Supported labels:
    dog_bark, cat_vocalization, non_dog_bark

This script imports the shared model/J-lens utilities from
``evaluate_hierarchical_behavior.py``.  Place both scripts in the
same code directory.
"""

from __future__ import annotations

import project_paths as paths

import argparse
import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_hierarchical_behavior as base
from qwen_omni_utils import process_mm_info


SCRIPT_VERSION = "7e-prompt-task-invariance-1.0.0"
DEFAULT_OUTPUT = Path(
    str(paths.OUTPUTS_ROOT / "evaluate_prompt_invariance")
)
PRIMARY_LAYER = 24
DEFAULT_LENS_TARGET_LAYER = 26
BOOTSTRAP_REPEATS = 10_000
PERMUTATION_REPEATS = 9_999


@dataclass(frozen=True)
class ProbePrompt:
    name: str
    text: str
    candidates: tuple[str, ...]
    family: str
    contains_target_lexeme: bool
    use_audio: bool = True


# The three topic-label prompts have exactly the same task suffix.  This makes
# dog-topic versus vehicle-topic a substantially cleaner lexical-induction
# comparison than comparing prompts with different answer formats.
LOUDNESS_SUFFIX = (
    "Judge only the overall loudness of the audio. Answer with exactly one "
    "label: quiet or loud. Do not explain."
)
ACK_SUFFIX = "There is no audio classification task. Answer exactly: ready."

PROMPTS: dict[str, ProbePrompt] = {
    "neutral_ack": ProbePrompt(
        "neutral_ack",
        "Listen to the audio without naming its source. Answer exactly: ready.",
        ("ready",),
        "target_free_unrelated",
        False,
    ),
    "plain_loudness": ProbePrompt(
        "plain_loudness",
        LOUDNESS_SUFFIX,
        ("quiet", "loud"),
        "target_free_unrelated",
        False,
    ),
    "event_count": ProbePrompt(
        "event_count",
        (
            "Judge only whether the audio contains one continuous sound event "
            "or multiple distinct sound events. Answer with exactly one label: "
            "one or multiple. Do not name the source and do not explain."
        ),
        ("one", "multiple"),
        "target_free_unrelated",
        False,
    ),
    "species_control": ProbePrompt(
        "species_control",
        (
            "Classify the primary sound source. Answer with exactly one label "
            "from: dog, cat, other. Do not provide any explanation."
        ),
        ("dog", "cat", "other"),
        "relevant_positive_control",
        True,
    ),
    "dog_topic_loudness": ProbePrompt(
        "dog_topic_loudness",
        (
            "Unrelated topic label: dog barking. The topic label does not "
            "describe the audio and must not affect the judgment. " + LOUDNESS_SUFFIX
        ),
        ("quiet", "loud"),
        "matched_lexical_induction",
        True,
    ),
    "cat_topic_loudness": ProbePrompt(
        "cat_topic_loudness",
        (
            "Unrelated topic label: cat meowing. The topic label does not "
            "describe the audio and must not affect the judgment. " + LOUDNESS_SUFFIX
        ),
        ("quiet", "loud"),
        "matched_lexical_induction",
        False,
    ),
    "vehicle_topic_loudness": ProbePrompt(
        "vehicle_topic_loudness",
        (
            "Unrelated topic label: vehicle sound. The topic label does not "
            "describe the audio and must not affect the judgment. " + LOUDNESS_SUFFIX
        ),
        ("quiet", "loud"),
        "matched_lexical_induction",
        False,
    ),
    "text_only_neutral": ProbePrompt(
        "text_only_neutral",
        "Unrelated topic label: table surface. " + ACK_SUFFIX,
        ("ready",),
        "text_only_control",
        False,
        False,
    ),
    "text_only_dog": ProbePrompt(
        "text_only_dog",
        "Unrelated topic label: dog barking. " + ACK_SUFFIX,
        ("ready",),
        "text_only_control",
        True,
        False,
    ),
    "text_only_cat": ProbePrompt(
        "text_only_cat",
        "Unrelated topic label: cat meowing. " + ACK_SUFFIX,
        ("ready",),
        "text_only_control",
        False,
        False,
    ),
    "text_only_vehicle": ProbePrompt(
        "text_only_vehicle",
        "Unrelated topic label: vehicle sound. " + ACK_SUFFIX,
        ("ready",),
        "text_only_control",
        False,
        False,
    ),
}

AUDIO_PROMPT_NAMES = (
    "neutral_ack",
    "plain_loudness",
    "event_count",
    "species_control",
    "dog_topic_loudness",
    "cat_topic_loudness",
    "vehicle_topic_loudness",
)
TEXT_ONLY_PROMPT_NAMES = (
    "text_only_neutral",
    "text_only_dog",
    "text_only_cat",
    "text_only_vehicle",
)
TARGET_FREE_PROMPT_NAMES = ("neutral_ack", "plain_loudness", "event_count")


def prepare_probe_inputs(
    processor: Any,
    prompt: ProbePrompt,
    audio_path: str | None,
) -> dict[str, Any]:
    content: list[dict[str, str]] = []
    if prompt.use_audio:
        if not audio_path:
            raise ValueError(f"Prompt {prompt.name} requires an audio path")
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": prompt.text})
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": base.SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    if prompt.use_audio:
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False
        )
        inputs = processor(
            text=rendered,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
    else:
        inputs = processor(
            text=rendered,
            return_tensors="pt",
            padding=True,
        )
    return {key: value for key, value in inputs.items()}


class DecisionResidualCapture:
    """Capture the selected layer output only on the generation prefill pass."""

    def __init__(self, layer_module: Any, decision_index: int, prompt_length: int):
        self.layer_module = layer_module
        self.decision_index = decision_index
        self.prompt_length = prompt_length
        self.handle: Any = None
        self.vector: torch.Tensor | None = None
        self.prefill_calls = 0

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = base.output_hidden(output)
        if (
            self.vector is None
            and hidden.ndim == 3
            and hidden.shape[1] == self.prompt_length
            and self.decision_index < hidden.shape[1]
        ):
            self.vector = hidden[0, self.decision_index].detach().float().cpu()
            self.prefill_calls += 1
        return output

    def __enter__(self) -> "DecisionResidualCapture":
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()


def internal_scores(
    hidden: torch.Tensor,
    directions: dict[str, torch.Tensor],
) -> dict[str, float]:
    h = hidden.float().cpu()
    h_norm = float(h.norm().clamp_min(1e-12).item())
    result: dict[str, float] = {
        "internal_residual_norm": h_norm,
    }
    for concept in ("dog", "cat", "animal", "bark", "sound", "vehicle"):
        if concept not in directions:
            continue
        coordinate = float(torch.dot(h, directions[concept].float().cpu()).item())
        result[f"internal_{concept}_unit_coordinate"] = coordinate
        result[f"internal_{concept}_cosine"] = coordinate / h_norm
        result[f"internal_{concept}_scaled_cosine"] = (
            coordinate / h_norm * math.sqrt(h.numel())
        )

    dog_cat = torch.stack(
        [directions["dog"].float().cpu(), directions["cat"].float().cpu()], dim=1
    )
    coordinates = torch.linalg.pinv(dog_cat) @ h
    result["internal_dog_span_coordinate"] = float(coordinates[0].item())
    result["internal_cat_span_coordinate"] = float(coordinates[1].item())
    result["internal_dog_minus_cat_span_coordinate"] = float(
        (coordinates[0] - coordinates[1]).item()
    )
    result["internal_dog_minus_cat_scaled_cosine"] = (
        result["internal_dog_scaled_cosine"]
        - result["internal_cat_scaled_cosine"]
    )
    return result


def run_probe(
    row: dict[str, Any],
    prompt: ProbePrompt,
    processor: Any,
    thinker: Any,
    device: torch.device,
    token_ids: dict[str, int],
    directions: dict[str, torch.Tensor],
    layer: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_inputs = prepare_probe_inputs(
        processor,
        prompt,
        str(row.get("audio_path", "")) if prompt.use_audio else None,
    )
    attention_mask = raw_inputs.get(
        "attention_mask", torch.ones_like(raw_inputs["input_ids"])
    )
    valid_len = int(attention_mask[0].long().sum().item())
    decision_index = valid_len - 1
    if prompt.use_audio:
        _, position_metadata = base.locate_positions(
            raw_inputs,
            processor.tokenizer,
            int(base.model_config_value(thinker, "audio_token_id")),
        )
    else:
        position_metadata = {
            "valid_sequence_length": valid_len,
            "audio_start": "",
            "audio_stop_exclusive": "",
            "audio_count": 0,
            "audio_bos_index": "",
            "audio_eos_index": "",
            "decision_index": decision_index,
            "audio_middle_count": 0,
            "post_audio_count": 0,
        }

    inputs = base.move_inputs_to_device(
        raw_inputs, device, next(thinker.parameters()).dtype
    )
    prompt_length = int(inputs["input_ids"].shape[1])
    capture = DecisionResidualCapture(
        thinker.model.layers[layer], decision_index, prompt_length
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
    with capture, torch.inference_mode():
        generated = thinker.generate(**generation_kwargs)
    if capture.vector is None:
        raise RuntimeError(
            f"Did not capture L{layer} decision residual during prefill"
        )
    raw_logits = getattr(generated, "logits", None)
    first_logits = (
        raw_logits[0][0]
        if raw_logits is not None and len(raw_logits) > 0
        else generated.scores[0][0]
    ).detach().float().cpu()
    text, generated_ids = base.decode_generated(
        processor.tokenizer, generated.sequences, prompt_length
    )
    prompt_spec = base.PromptSpec(
        name=prompt.name,
        text=prompt.text,
        candidates=prompt.candidates,
        truth_by_label={},
    )
    final_scores = base.score_logits(
        first_logits, token_ids, prompt_spec, processor.tokenizer
    )
    generated_label = base.normalize_label(text, prompt.candidates)
    target_words = set(re.findall(r"[a-z]+", text.lower()))
    return {
        **row,
        "prompt_name": prompt.name,
        "prompt_family": prompt.family,
        "prompt_text": prompt.text,
        "prompt_contains_target_lexeme": prompt.contains_target_lexeme,
        "use_audio": prompt.use_audio,
        "layer": layer,
        "readout_position": "decision",
        **position_metadata,
        "capture_prefill_calls": capture.prefill_calls,
        **internal_scores(capture.vector, directions),
        "generated_text": text,
        "generated_token_ids": json.dumps(generated_ids),
        "generated_label": generated_label,
        "generated_mentions_dog_or_bark": bool(
            target_words & {"dog", "dogs", "bark", "barking", "canine", "puppy"}
        ),
        **final_scores,
        "finite_first_step_logits": bool(torch.isfinite(first_logits).all().item()),
        "finite_internal_residual": bool(torch.isfinite(capture.vector).all().item()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": base.now_iso(),
    }


def midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = midranks(combined)
    n_pos = positive.size
    rank_sum = float(ranks[:n_pos].sum())
    return (
        rank_sum - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * negative.size)


def independent_mean_contrast(
    positive: np.ndarray,
    negative: np.ndarray,
    seed: int,
) -> tuple[float, float, float, float]:
    observed = float(positive.mean() - negative.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        boot[index] = (
            rng.choice(positive, size=positive.size, replace=True).mean()
            - rng.choice(negative, size=negative.size, replace=True).mean()
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    pooled = np.concatenate([positive, negative])
    exceed = 0
    for _ in range(PERMUTATION_REPEATS):
        permuted = rng.permutation(pooled)
        null_value = float(
            permuted[: positive.size].mean()
            - permuted[positive.size :].mean()
        )
        exceed += abs(null_value) >= abs(observed)
    p_value = (exceed + 1) / (PERMUTATION_REPEATS + 1)
    return observed, float(low), float(high), float(p_value)


def paired_contrast(
    left: dict[str, float],
    right: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    values = np.asarray([left[key] - right[key] for key in ids], dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    if values.size == 1:
        low = high = float("nan")
        p_value = float("nan")
    else:
        low, high = base.bootstrap_mean_ci(values, seed)
        p_value = base.sign_permutation_p(values, seed + 1)
    return {
        "n": int(values.size),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "two_sided_sign_permutation_p": p_value,
        "positive_difference_consistency": float(np.mean(values > 0)),
    }


SUMMARY_METRICS = (
    "internal_dog_scaled_cosine",
    "internal_bark_scaled_cosine",
    "internal_dog_minus_cat_span_coordinate",
    "internal_dog_minus_cat_scaled_cosine",
    "concept_dog_logit",
    "concept_bark_logit",
    "concept_dog_minus_cat_logit",
    "concept_dog_rank",
    "concept_bark_rank",
)

CONTRAST_METRICS = (
    "internal_dog_scaled_cosine",
    "internal_bark_scaled_cosine",
    "internal_dog_minus_cat_span_coordinate",
    "internal_dog_minus_cat_scaled_cosine",
    "concept_dog_logit",
    "concept_bark_logit",
    "concept_dog_minus_cat_logit",
)


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["prompt_name"]), str(row["label"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (prompt_name, label), values in sorted(grouped.items()):
        out: dict[str, Any] = {
            "prompt_name": prompt_name,
            "prompt_family": values[0]["prompt_family"],
            "label": label,
            "n": len(values),
            "generated_dog_or_bark_rate": float(
                np.mean([bool(row["generated_mentions_dog_or_bark"]) for row in values])
            ),
            "generated_distribution": json.dumps(
                {
                    item: sum(row["generated_label"] == item for row in values)
                    for item in sorted({str(row["generated_label"]) for row in values})
                },
                sort_keys=True,
            ),
        }
        for metric in SUMMARY_METRICS:
            numbers = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            out[f"{metric}_mean"] = float(numbers.mean())
            out[f"{metric}_median"] = float(np.median(numbers))
            out[f"{metric}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        result.append(out)
    return result


def audio_separation_contrasts(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    audio_rows = [row for row in rows if bool(row["use_audio"])]
    for prompt_name in AUDIO_PROMPT_NAMES:
        prompt_rows = [row for row in audio_rows if row["prompt_name"] == prompt_name]
        for metric in CONTRAST_METRICS:
            positives = np.asarray(
                [float(row[metric]) for row in prompt_rows if row["label"] == "dog_bark"],
                dtype=np.float64,
            )
            comparisons = {
                "dog_vs_all_non_dog": np.asarray(
                    [float(row[metric]) for row in prompt_rows if row["label"] != "dog_bark"],
                    dtype=np.float64,
                ),
                "dog_vs_cat": np.asarray(
                    [float(row[metric]) for row in prompt_rows if row["label"] == "cat_vocalization"],
                    dtype=np.float64,
                ),
                "dog_vs_environment": np.asarray(
                    [float(row[metric]) for row in prompt_rows if row["label"] == "non_dog_bark"],
                    dtype=np.float64,
                ),
            }
            for comparison, negatives in comparisons.items():
                if positives.size == 0 or negatives.size == 0:
                    continue
                effect, low, high, p_value = independent_mean_contrast(
                    positives, negatives, seed + len(result) * 101
                )
                result.append(
                    {
                        "contrast_type": "audio_label_separation",
                        "prompt_name": prompt_name,
                        "comparison": comparison,
                        "metric": metric,
                        "n_positive": int(positives.size),
                        "n_negative": int(negatives.size),
                        "positive_mean": float(positives.mean()),
                        "negative_mean": float(negatives.mean()),
                        "mean_difference": effect,
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "two_sided_permutation_p": p_value,
                        "roc_auc": roc_auc(positives, negatives),
                    }
                )
    return result


def prompt_induction_contrasts(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    prompt_pairs = (
        ("dog_topic_loudness", "vehicle_topic_loudness", "dog_minus_vehicle"),
        ("dog_topic_loudness", "cat_topic_loudness", "dog_minus_cat_topic"),
        ("dog_topic_loudness", "plain_loudness", "dog_topic_minus_plain"),
    )
    labels = ("dog_bark", "cat_vocalization", "non_dog_bark", "all_non_dog")
    for left_name, right_name, comparison in prompt_pairs:
        for label in labels:
            selected = [
                row for row in rows
                if bool(row["use_audio"])
                and (row["label"] != "dog_bark" if label == "all_non_dog" else row["label"] == label)
            ]
            for metric in CONTRAST_METRICS:
                left = {
                    str(row["sample_id"]): float(row[metric])
                    for row in selected if row["prompt_name"] == left_name
                }
                right = {
                    str(row["sample_id"]): float(row[metric])
                    for row in selected if row["prompt_name"] == right_name
                }
                contrast = paired_contrast(
                    left, right, seed + len(result) * 131
                )
                result.append(
                    {
                        "contrast_type": "paired_prompt_induction",
                        "prompt_name": left_name,
                        "comparison": comparison,
                        "label": label,
                        "metric": metric,
                        **contrast,
                    }
                )

    text_rows = [row for row in rows if not bool(row["use_audio"])]
    for left_name, right_name, comparison in (
        ("text_only_dog", "text_only_vehicle", "text_only_dog_minus_vehicle"),
        ("text_only_dog", "text_only_cat", "text_only_dog_minus_cat"),
        ("text_only_dog", "text_only_neutral", "text_only_dog_minus_neutral"),
    ):
        for metric in CONTRAST_METRICS:
            left_values = [float(row[metric]) for row in text_rows if row["prompt_name"] == left_name]
            right_values = [float(row[metric]) for row in text_rows if row["prompt_name"] == right_name]
            if not left_values or not right_values:
                continue
            result.append(
                {
                    "contrast_type": "text_only_prompt_induction",
                    "prompt_name": left_name,
                    "comparison": comparison,
                    "label": "text_only",
                    "metric": metric,
                    "n": 1,
                    "mean_difference": left_values[0] - right_values[0],
                    "bootstrap_ci95_low": "",
                    "bootstrap_ci95_high": "",
                    "two_sided_sign_permutation_p": "",
                    "positive_difference_consistency": float(left_values[0] > right_values[0]),
                }
            )
    return result


def get_contrast(
    rows: list[dict[str, Any]],
    contrast_type: str,
    prompt_name: str,
    comparison: str,
    metric: str,
    label: str | None = None,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("contrast_type") == contrast_type
            and row.get("prompt_name") == prompt_name
            and row.get("comparison") == comparison
            and row.get("metric") == metric
            and (label is None or row.get("label") == label)
        ):
            return row
    return None


def scientific_interpretation(
    contrasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for concept, metric in (
        ("dog", "internal_dog_scaled_cosine"),
        ("bark", "internal_bark_scaled_cosine"),
    ):
        unrelated: list[dict[str, Any]] = []
        for prompt_name in TARGET_FREE_PROMPT_NAMES:
            row = get_contrast(
                contrasts,
                "audio_label_separation",
                prompt_name,
                "dog_vs_all_non_dog",
                metric,
            )
            if row is not None:
                unrelated.append(row)
        spontaneous_passes = sum(
            float(row["mean_difference"]) > 0
            and float(row["bootstrap_ci95_low"]) > 0
            and float(row["roc_auc"]) >= 0.70
            for row in unrelated
        )
        spontaneous_status = (
            "SUPPORTED" if unrelated and spontaneous_passes == len(unrelated)
            else "MIXED" if spontaneous_passes > 0
            else "NOT_SUPPORTED"
        )

        lexical = get_contrast(
            contrasts,
            "paired_prompt_induction",
            "dog_topic_loudness",
            "dog_minus_vehicle",
            metric,
            "all_non_dog",
        )
        plain_audio = get_contrast(
            contrasts,
            "audio_label_separation",
            "plain_loudness",
            "dog_vs_all_non_dog",
            metric,
        )
        ratio = float("nan")
        if lexical is not None and plain_audio is not None:
            denominator = abs(float(plain_audio["mean_difference"]))
            if denominator > 1e-12:
                ratio = abs(float(lexical["mean_difference"])) / denominator
        if math.isfinite(ratio):
            lexical_status = (
                "LOW" if ratio < 0.25 else "MODERATE" if ratio < 0.75 else "HIGH"
            )
        else:
            lexical_status = "UNAVAILABLE"
        result.extend([
            {
                "question": f"{concept}_audio_separates_non_dog_under_target_free_tasks",
                "criterion": "positive CI and ROC-AUC >= 0.70 for all three unrelated prompts",
                "result": f"{spontaneous_passes}/{len(unrelated)} prompts meet criterion",
                "status": spontaneous_status,
            },
            {
                "question": f"dog_barking_words_induce_{concept}_coordinate_on_non_dog_audio",
                "criterion": "absolute paired prompt shift divided by plain-task audio separation",
                "result": f"lexical_to_audio_effect_ratio={ratio:.6f}" if math.isfinite(ratio) else "unavailable",
                "status": lexical_status,
            },
        ])
    result.append(
        {
            "question": "interpretation_rule",
            "criterion": "scientific outcomes do not affect technical validity",
            "result": (
                "SUPPORTED+LOW favors spontaneous task-invariant audio concept; "
                "NOT_SUPPORTED suggests task-gated readout; HIGH indicates strong prompt confounding"
            ),
            "status": "INFO",
        }
    )
    return result


def prompt_lexical_audit() -> list[dict[str, Any]]:
    target_pattern = re.compile(r"\b(dog|dogs|canine|puppy|bark|barking)\b", re.I)
    rows: list[dict[str, Any]] = []
    for prompt in PROMPTS.values():
        found = sorted(set(match.group(0).lower() for match in target_pattern.finditer(prompt.text)))
        expected = prompt.contains_target_lexeme
        rows.append(
            {
                "prompt_name": prompt.name,
                "family": prompt.family,
                "declared_contains_target_lexeme": expected,
                "detected_target_lexemes": json.dumps(found),
                "status": "PASS" if bool(found) == expected else "FAIL",
            }
        )
    return rows


def technical_checks(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    lens: Any,
    args: argparse.Namespace,
    detected_target: int | None,
    lexical_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_audio_rows = len(selected) * len(AUDIO_PROMPT_NAMES)
    actual_audio_rows = sum(bool(row["use_audio"]) for row in rows)
    expected_text_rows = len(TEXT_ONLY_PROMPT_NAMES) if args.include_text_only else 0
    actual_text_rows = sum(not bool(row["use_audio"]) for row in rows)
    selected_counts = {
        label: sum(row["label"] == label for row in selected)
        for label in base.SUPPORTED_LABELS
    }
    required_tokens = {"dog", "cat", "other", "animal", "bark", "sound", "vehicle"}
    passed_tokens = {row["concept"] for row in token_rows if row["status"] == "PASS"}
    audio_aligned = all(
        int(row["audio_count"]) > 0 for row in rows if bool(row["use_audio"])
    )
    return [
        {
            "check_name": "lens_metadata",
            "result": f"prompts={lens.n_prompts}, d_model={lens.d_model}",
            "status": "PASS" if lens.n_prompts == base.EXPECTED_LENS_PROMPTS and lens.d_model == base.EXPECTED_D_MODEL else "FAIL",
        },
        {
            "check_name": "l26_penultimate_target",
            "result": f"declared={args.lens_target_layer}; detected={detected_target}",
            "status": "PASS" if args.lens_target_layer == 26 and detected_target in {None, 26} else "FAIL",
        },
        {
            "check_name": "manifest_counts",
            "result": json.dumps(selected_counts, sort_keys=True),
            "status": "PASS" if len(selected) == (args.per_label * len(base.SUPPORTED_LABELS)) else "FAIL",
        },
        {
            "check_name": "required_single_tokens",
            "result": json.dumps(sorted(required_tokens & passed_tokens)),
            "status": "PASS" if required_tokens <= passed_tokens else "FAIL",
        },
        {
            "check_name": "prompt_lexical_audit",
            "result": f"rows={len(lexical_audit)}",
            "status": "PASS" if all(row["status"] == "PASS" for row in lexical_audit) else "FAIL",
        },
        {
            "check_name": "complete_audio_jobs",
            "result": f"actual={actual_audio_rows}; expected={expected_audio_rows}",
            "status": "PASS" if actual_audio_rows == expected_audio_rows else "FAIL",
        },
        {
            "check_name": "complete_text_only_jobs",
            "result": f"actual={actual_text_rows}; expected={expected_text_rows}",
            "status": "PASS" if actual_text_rows == expected_text_rows else "FAIL",
        },
        {
            "check_name": "audio_alignment",
            "result": f"audio_rows={actual_audio_rows}",
            "status": "PASS" if audio_aligned else "FAIL",
        },
        {
            "check_name": "decision_residual_capture",
            "result": f"rows={len(rows)}",
            "status": "PASS" if rows and all(int(row["capture_prefill_calls"]) == 1 for row in rows) else "FAIL",
        },
        {
            "check_name": "finite_outputs",
            "result": f"rows={len(rows)}",
            "status": "PASS" if rows and all(bool(row["finite_first_step_logits"]) and bool(row["finite_internal_residual"]) for row in rows) else "FAIL",
        },
        {
            "check_name": "nonempty_generation",
            "result": f"rows={len(rows)}",
            "status": "PASS" if rows and all(str(row["generated_text"]).strip() for row in rows) else "FAIL",
        },
        {
            "check_name": "failures",
            "result": f"rows={len(failures)}",
            "status": "PASS" if not failures else "FAIL",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "primary"), default="smoke")
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--lens-path", type=Path, default=base.DEFAULT_LENS)
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-label", type=int, default=None,
                        help="Samples per label; defaults to 1 for smoke, 35 for primary.")
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--lens-target-layer", type=int, default=DEFAULT_LENS_TARGET_LAYER)
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("QWEN_OMNI_DEVICE", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--include-text-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_label is None:
        args.per_label = 1 if args.phase == "smoke" else 35
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")
    if args.layer not in base.SOURCE_LAYERS:
        raise ValueError(f"--layer must be one of {base.SOURCE_LAYERS}")
    if args.lens_target_layer != 26:
        raise ValueError("Experiment 7E primary protocol requires the L26-target lens")

    detected_target, target_metadata_path = base.detect_lens_target_layer(args.lens_path)
    if detected_target is not None and detected_target != args.lens_target_layer:
        raise ValueError(
            f"Declared L{args.lens_target_layer} lens, but metadata records L{detected_target}"
        )

    phase_dir = args.output_dir / args.phase
    internal_dir = phase_dir / "internal"
    cache_dir = internal_dir / "cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = base.load_manifest(args.manifest)
    selected = base.select_manifest(manifest, args.phase, args.per_label, args.seed)
    base.write_csv(internal_dir / "selected_manifest.csv", selected)
    lexical_audit = prompt_lexical_audit()
    base.write_csv(internal_dir / "prompt_lexical_audit.csv", lexical_audit)

    processor, container, thinker, lens, device, token_ids, token_rows = (
        base.load_model_and_lens(args)
    )
    del container
    base.write_csv(internal_dir / "concept_tokenization.csv", token_rows)
    directions, direction_rows = base.build_jspace_directions(
        thinker, lens, args.layer, token_ids, args.seed
    )
    base.write_csv(internal_dir / "direction_diagnostics.csv", direction_rows)

    config = {
        "script_version": SCRIPT_VERSION,
        "phase": args.phase,
        "model_path": str(args.model_path),
        "lens_path": str(args.lens_path),
        "lens_sha256": base.sha256_file(args.lens_path),
        "lens_target_layer": args.lens_target_layer,
        "detected_lens_target_layer": detected_target,
        "lens_target_metadata_path": target_metadata_path,
        "manifest_path": str(args.manifest),
        "manifest_sha256": base.sha256_file(args.manifest),
        "output_dir": str(phase_dir),
        "source_layer": args.layer,
        "readout_position": "decision",
        "system_prompt": base.SYSTEM_PROMPT,
        "prompts": {name: asdict(prompt) for name, prompt in PROMPTS.items()},
        "audio_prompt_names": list(AUDIO_PROMPT_NAMES),
        "text_only_prompt_names": list(TEXT_ONLY_PROMPT_NAMES) if args.include_text_only else [],
        "per_label": args.per_label,
        "selected_counts": {
            label: sum(row["label"] == label for row in selected)
            for label in base.SUPPORTED_LABELS
        },
        "include_text_only": args.include_text_only,
        "max_new_tokens": args.max_new_tokens,
        "greedy_generation": True,
        "seed": args.seed,
        "created_at": base.now_iso(),
    }
    config["run_signature"] = base.stable_hash(config)
    base.write_json(internal_dir / "frozen_config.json", config)

    jobs: list[tuple[dict[str, Any], ProbePrompt]] = []
    for row in selected:
        for prompt_name in AUDIO_PROMPT_NAMES:
            jobs.append((row, PROMPTS[prompt_name]))
    if args.include_text_only:
        text_row = {
            "sample_id": "text_only_once",
            "audio_path": "",
            "label": "text_only",
            "source": "synthetic_text_only_control",
            "semantic_subtype": "text_only",
        }
        for prompt_name in TEXT_ONLY_PROMPT_NAMES:
            jobs.append((text_row, PROMPTS[prompt_name]))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (row, prompt) in enumerate(jobs, 1):
        cache_key = base.stable_hash(
            {
                "run_signature": config["run_signature"],
                "sample_id": row["sample_id"],
                "prompt_name": prompt.name,
            }
        )
        cache_path = cache_dir / f"{cache_key}.json"
        print(
            f"[{index:04d}/{len(jobs):04d}] {row['sample_id']} {prompt.name}",
            flush=True,
        )
        try:
            if cache_path.is_file() and not args.no_resume:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("run_signature") == config["run_signature"]:
                    result = payload["result"]
                else:
                    result = run_probe(
                        row, prompt, processor, thinker, device, token_ids,
                        directions, args.layer, args.max_new_tokens,
                    )
            else:
                result = run_probe(
                    row, prompt, processor, thinker, device, token_ids,
                    directions, args.layer, args.max_new_tokens,
                )
            base.write_json(
                cache_path,
                {"run_signature": config["run_signature"], "result": result},
            )
            results.append(result)
        except Exception as exc:
            failure = {
                "sample_id": row["sample_id"],
                "prompt_name": prompt.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": base.now_iso(),
            }
            failures.append(failure)
            base.write_csv(phase_dir / "7E_failures.csv", failures)
            if not args.continue_on_error:
                raise
        finally:
            torch.cuda.empty_cache()

    group_summary = summarize_groups(results)
    contrasts = audio_separation_contrasts(results, args.seed)
    contrasts.extend(prompt_induction_contrasts(results, args.seed + 700_000))
    interpretation = scientific_interpretation(contrasts)
    checks = technical_checks(
        results, failures, selected, token_rows, lens, args,
        detected_target, lexical_audit,
    )
    technical_status = (
        "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    )

    base.write_csv(phase_dir / "7E_prompt_task_results.csv", results)
    base.write_csv(phase_dir / "7E_group_summary.csv", group_summary)
    base.write_csv(phase_dir / "7E_scientific_contrasts.csv", contrasts)
    base.write_csv(phase_dir / "7E_scientific_interpretation.csv", interpretation)
    base.write_csv(phase_dir / "7E_technical_check.csv", checks)
    base.write_json(
        phase_dir / "7E_summary.json",
        {
            "script_version": SCRIPT_VERSION,
            "phase": args.phase,
            "samples": len(selected),
            "jobs": len(jobs),
            "result_rows": len(results),
            "failure_rows": len(failures),
            "technical_status": technical_status,
            "scientific_interpretation": interpretation,
            "completed_at": base.now_iso(),
        },
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(interpretation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
