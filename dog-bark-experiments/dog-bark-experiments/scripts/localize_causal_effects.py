"""Experiment 8F: causal localization across layer, position, and task.

This experiment asks where a Qwen2.5-Omni dog concept becomes causally
available and whether the same J-space direction generalizes across tasks.
For every selected layer and one-token position, it applies the same:

* dog projection ablation to dog-bark samples (necessity),
* norm-scaled dog injection to task-negative samples (sufficiency),
* norm-matched vehicle and random-orthogonal controls.

The four primary tasks are species, superclass, vocalization, and open
specific generation.  Interventions are prefill-only; decoding is untouched.
The L26-target J-lens supplies a layer-specific dog direction
v_d(l) = normalize(J_l^T w_d).

Position definitions deliberately use exactly one token each so that position
effects are not confounded by editing different numbers of tokens:

* audio_middle: central audio-content token,
* audio_end: final audio-content token before <|audio_eos|>,
* post_audio: first token after <|audio_eos|> (audio-to-text bridge),
* decision: final valid prefill token immediately before generation.

Expected manifest columns:
    sample_id,audio_path,label,source

Supported labels:
    dog_bark, cat_vocalization, non_dog_bark

Place this script beside evaluate_hierarchical_behavior.py.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_hierarchical_behavior as base


SCRIPT_VERSION = "8f-causal-localization-1.0.0"
DEFAULT_OUTPUT = Path(
    str(paths.OUTPUTS_ROOT / "localize_causal_effects")
)
DEFAULT_LENS_TARGET_LAYER = 26
POSITION_NAMES = ("audio_middle", "audio_end", "post_audio", "decision")
TASK_NAMES = ("species_a", "superclass", "vocalization", "open_specific")
PHASE_LAYERS = {
    "smoke": [24],
    "screen": [12, 16, 20, 22, 24],
    "primary": [16, 20, 22, 24],
    "full": list(base.SOURCE_LAYERS),
}
PHASE_PER_LABEL = {
    "smoke": 1,
    "screen": 10,
    "primary": 35,
    "full": 35,
}
PRIMARY_METRIC_NAME = "task_primary_metric"
MAX_ALLOWED_DELTA_RATIO = 0.25
MATCHED_NORM_RELATIVE_TOLERANCE = 2e-3


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_layers(value: str) -> list[int]:
    result = [int(item) for item in parse_csv_list(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate layers: {result}")
    invalid = sorted(set(result) - set(base.SOURCE_LAYERS))
    if invalid:
        raise ValueError(
            f"Layers {invalid} are not fitted J-lens source layers; available="
            f"{base.SOURCE_LAYERS}"
        )
    return result


def selected_layers(args: argparse.Namespace) -> list[int]:
    return parse_layers(args.layers) if args.layers else list(PHASE_LAYERS[args.phase])


def selected_tasks(args: argparse.Namespace) -> list[str]:
    result = parse_csv_list(args.tasks) if args.tasks else list(TASK_NAMES)
    invalid = sorted(set(result) - set(TASK_NAMES))
    if invalid:
        raise ValueError(f"Unsupported tasks {invalid}; available={list(TASK_NAMES)}")
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate tasks: {result}")
    return result


def selected_positions(args: argparse.Namespace) -> list[str]:
    result = parse_csv_list(args.positions) if args.positions else list(POSITION_NAMES)
    invalid = sorted(set(result) - set(POSITION_NAMES))
    if invalid:
        raise ValueError(
            f"Unsupported positions {invalid}; available={list(POSITION_NAMES)}"
        )
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate positions: {result}")
    return result


def one_token_positions(
    inputs: dict[str, Any], tokenizer: Any, audio_token_id: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Return four one-token sites plus an auditable position definition."""
    _, metadata = base.locate_positions(inputs, tokenizer, audio_token_id)
    audio_start = int(metadata["audio_start"])
    audio_stop = int(metadata["audio_stop_exclusive"])
    audio_eos = int(metadata["audio_eos_index"])
    decision = int(metadata["decision_index"])
    if audio_stop <= audio_start:
        raise ValueError("Empty audio token span")
    text_positions = list(range(audio_eos + 1, decision))
    if not text_positions:
        raise ValueError("No post-audio task-text tokens before decision position")
    positions = {
        "audio_middle": [audio_start + (audio_stop - audio_start) // 2],
        "audio_end": [audio_stop - 1],
        # The first post-audio token is chosen instead of a task-text midpoint.
        # This avoids accidentally selecting a task-specific label word such as
        # "dog" or "animal" and cleanly probes the audio-to-text bridge.
        "post_audio": [text_positions[0]],
        "decision": [decision],
    }
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    audit: dict[str, Any] = dict(metadata)
    audit.update(
        {
            "position_policy": "one_token_per_site",
            "post_audio_candidate_count": len(text_positions),
        }
    )
    for name, indices in positions.items():
        index = indices[0]
        audit[f"{name}_index"] = index
        audit[f"{name}_token_id"] = int(input_ids[index])
        audit[f"{name}_token_text"] = tokenizer.decode(
            [int(input_ids[index])], skip_special_tokens=False
        )
    return positions, audit


def intervention_conditions(
    role: str, position_name: str, ablation_strength: float, injection_ratio: float
) -> list[base.Condition]:
    if role == "ablation_dog":
        return [
            base.Condition(
                "dog_ablate", "ablate", ablation_strength,
                position_name, "dog", "", "decrease_task_target",
            ),
            base.Condition(
                "matched_vehicle_ablate", "matched_vehicle_ablate",
                ablation_strength, position_name, "vehicle", "",
                "neutral_control",
            ),
            base.Condition(
                "matched_random_ablate", "matched_random_ablate",
                ablation_strength, position_name, "random_orthogonal", "",
                "neutral_control",
            ),
        ]
    if role == "injection_negative":
        return [
            base.Condition(
                "dog_inject", "inject", injection_ratio,
                position_name, "dog", "", "increase_task_target",
            ),
            base.Condition(
                "vehicle_inject", "vehicle_inject", injection_ratio,
                position_name, "vehicle", "", "neutral_control",
            ),
            base.Condition(
                "random_orthogonal_inject", "random_inject", injection_ratio,
                position_name, "random_orthogonal", "", "neutral_control",
            ),
        ]
    raise ValueError(f"Unknown role {role!r}")


def eligible_role(row: dict[str, Any], task_name: str) -> str | None:
    if row["label"] == "dog_bark":
        return "ablation_dog"
    # Sufficiency should be tested on task-negative examples.  In superclass,
    # cat and other-animal samples are already positive (animal), so only
    # superclass_truth=other is eligible for dog injection.
    if task_name == "superclass" and row["superclass_truth"] != "other":
        return None
    return "injection_negative"


def logsumexp_numbers(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def task_primary_metric(task_name: str, scored: dict[str, Any]) -> tuple[str, float]:
    if task_name == "species_a":
        dog = float(scored["behavior_dog_first_token_score"])
        rest = logsumexp_numbers(
            [
                float(scored["behavior_cat_first_token_score"]),
                float(scored["behavior_other_first_token_score"]),
            ]
        )
        return "dog_vs_cat_other_log_odds", dog - rest
    if task_name == "superclass":
        return (
            "animal_vs_other_log_odds",
            float(scored["behavior_animal_first_token_score"])
            - float(scored["behavior_other_first_token_score"]),
        )
    if task_name == "vocalization":
        bark = float(scored["behavior_bark_first_token_score"])
        rest = logsumexp_numbers(
            [
                float(scored["behavior_meow_first_token_score"]),
                float(scored["behavior_other_vocalization_first_token_score"]),
                float(scored["behavior_no_vocalization_first_token_score"]),
            ]
        )
        return "bark_vs_other_vocalization_log_odds", bark - rest
    if task_name == "open_specific":
        return "native_first_step_dog_logit", float(scored["concept_dog_logit"])
    raise ValueError(f"No primary metric for task {task_name!r}")


def generated_dog_mention(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & {"dog", "dogs", "canine", "puppy", "bark", "barking"})


def task_target_indicator(
    task_name: str, generated_label: str, generated_text: str
) -> bool:
    if task_name == "species_a":
        return generated_label == "dog"
    if task_name == "superclass":
        return generated_label == "animal"
    if task_name == "vocalization":
        return generated_label == "bark"
    if task_name == "open_specific":
        return generated_label == "dog" or generated_dog_mention(generated_text)
    raise ValueError(task_name)


def run_prepared(
    row: dict[str, Any],
    prompt: base.PromptSpec,
    condition: base.Condition,
    role: str,
    positions_by_name: dict[str, list[int]],
    position_audit: dict[str, Any],
    inputs: dict[str, Any],
    processor: Any,
    thinker: Any,
    token_ids: dict[str, int],
    directions: dict[str, torch.Tensor],
    layer: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    positions = (
        [] if condition.operation == "none"
        else positions_by_name[condition.position_mode]
    )
    logits, text, generated_ids, diagnostics = base.run_generation(
        thinker=thinker,
        tokenizer=processor.tokenizer,
        inputs=inputs,
        condition=condition,
        directions=directions,
        positions=positions,
        layer=layer,
        max_new_tokens=max_new_tokens,
    )
    generated_label = (
        base.normalize_label(text, prompt.candidates)
        if prompt.candidates
        else base.normalize_open_label(text)
    )
    truth = base.truth_for_row(row, prompt)
    scored = base.score_logits(logits, token_ids, prompt, processor.tokenizer)
    metric_name, metric_value = task_primary_metric(prompt.name, scored)
    return {
        **row,
        "task_name": prompt.name,
        "prompt_name": prompt.name,
        "prompt_text": prompt.text,
        "sample_role": role,
        "condition": condition.name,
        "operation": condition.operation,
        "layer": layer,
        "position_name": condition.position_mode,
        "selected_position_index": positions[0] if positions else "",
        "source_concept": condition.source_concept,
        "strength": condition.strength,
        "expected_behavior": condition.expected_behavior,
        **position_audit,
        **diagnostics,
        "generated_text": text,
        "generated_token_ids": json.dumps(generated_ids),
        "generated_label": generated_label,
        "generated_mentions_dog_or_bark": generated_dog_mention(text),
        "truth_label": truth,
        "is_correct": (generated_label == truth) if truth else "",
        "task_primary_metric_name": metric_name,
        PRIMARY_METRIC_NAME: metric_value,
        "task_target_indicator": task_target_indicator(prompt.name, generated_label, text),
        **scored,
        "finite_first_step_logits": bool(torch.isfinite(logits).all().item()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": base.now_iso(),
    }


def baseline_condition() -> base.Condition:
    return base.Condition("baseline", "none", 0.0, "none")


def cache_result(
    cache_dir: Path,
    run_signature: str,
    identity: dict[str, Any],
    compute: Any,
    no_resume: bool,
) -> dict[str, Any]:
    key = base.stable_hash({"run_signature": run_signature, **identity})
    path = cache_dir / f"{key}.json"
    if path.is_file() and not no_resume:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_signature") == run_signature:
            return payload["result"]
    result = compute()
    base.write_json(path, {"run_signature": run_signature, "result": result})
    return result


def paired_values_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    if values.size < 2:
        low = high = p_value = float("nan")
    else:
        low, high = base.bootstrap_mean_ci(values, seed)
        p_value = base.sign_permutation_p(values, seed + 1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "two_sided_sign_permutation_p": p_value,
        "positive_consistency": float(np.mean(values > 0)),
    }


def site_effects(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    baselines = {
        (row["sample_id"], row["task_name"]): row
        for row in rows if row["condition"] == "baseline"
    }
    site_rows = [row for row in rows if row["condition"] != "baseline"]
    metrics = (
        PRIMARY_METRIC_NAME,
        "concept_dog_logit",
        "concept_bark_logit",
        "task_target_indicator",
    )
    groups: list[tuple[str, Any]] = [
        ("dog_bark", lambda row: row["sample_role"] == "ablation_dog"),
        (
            "all_injection_negative",
            lambda row: row["sample_role"] == "injection_negative",
        ),
        (
            "cat_vocalization",
            lambda row: row["sample_role"] == "injection_negative"
            and row["label"] == "cat_vocalization",
        ),
        (
            "non_dog_bark",
            lambda row: row["sample_role"] == "injection_negative"
            and row["label"] == "non_dog_bark",
        ),
    ]
    output: list[dict[str, Any]] = []
    p_indices: list[int] = []
    p_values: list[float] = []
    site_keys = sorted(
        {
            (
                int(row["layer"]), str(row["position_name"]),
                str(row["task_name"]), str(row["sample_role"]),
            )
            for row in site_rows
        }
    )
    for layer, position, task_name, role in site_keys:
        target_name = "dog_ablate" if role == "ablation_dog" else "dog_inject"
        control_names = (
            ("matched_vehicle_ablate", "matched_random_ablate")
            if role == "ablation_dog"
            else ("vehicle_inject", "random_orthogonal_inject")
        )
        expected_sign = -1.0 if role == "ablation_dog" else 1.0
        at_site = [
            row for row in site_rows
            if int(row["layer"]) == layer
            and row["position_name"] == position
            and row["task_name"] == task_name
            and row["sample_role"] == role
        ]
        for group_name, predicate in groups:
            if (role == "ablation_dog") != (group_name == "dog_bark"):
                continue
            selected = [row for row in at_site if predicate(row)]
            by_sample: dict[str, dict[str, dict[str, Any]]] = {}
            for row in selected:
                by_sample.setdefault(str(row["sample_id"]), {})[str(row["condition"])] = row
            for metric in metrics:
                target_deltas: list[float] = []
                control_deltas: list[float] = []
                net_specificity: list[float] = []
                for sample_id, conditions in by_sample.items():
                    if target_name not in conditions or not all(name in conditions for name in control_names):
                        continue
                    baseline = baselines.get((sample_id, task_name))
                    if baseline is None:
                        continue
                    baseline_value = float(baseline[metric])
                    target_delta = float(conditions[target_name][metric]) - baseline_value
                    control_delta = float(
                        np.mean(
                            [float(conditions[name][metric]) - baseline_value for name in control_names]
                        )
                    )
                    target_deltas.append(target_delta)
                    control_deltas.append(control_delta)
                    net_specificity.append(expected_sign * (target_delta - control_delta))
                net_array = np.asarray(net_specificity, dtype=np.float64)
                if net_array.size == 0:
                    continue
                stats = paired_values_summary(
                    net_array, seed + len(output) * 97
                )
                output.append(
                    {
                        "layer": layer,
                        "position_name": position,
                        "task_name": task_name,
                        "sample_role": role,
                        "label_group": group_name,
                        "metric": metric,
                        "expected_raw_direction": "decrease" if expected_sign < 0 else "increase",
                        "n": stats["n"],
                        "mean_target_change": float(np.mean(target_deltas)),
                        "mean_matched_control_change": float(np.mean(control_deltas)),
                        "mean_signed_net_specificity": stats["mean"],
                        "median_signed_net_specificity": stats["median"],
                        "bootstrap_ci95_low": stats["bootstrap_ci95_low"],
                        "bootstrap_ci95_high": stats["bootstrap_ci95_high"],
                        "two_sided_sign_permutation_p": stats["two_sided_sign_permutation_p"],
                        "correct_direction_consistency": stats["positive_consistency"],
                    }
                )
                if metric == PRIMARY_METRIC_NAME and stats["n"] >= 2:
                    p_indices.append(len(output) - 1)
                    p_values.append(float(stats["two_sided_sign_permutation_p"]))
    for index, q_value in zip(p_indices, base.bh_fdr(p_values)):
        output[index]["bh_fdr_q_all_primary_site_tests"] = q_value
    return output


def raw_condition_summary(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    baselines = {
        (row["sample_id"], row["task_name"]): row
        for row in rows if row["condition"] == "baseline"
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] == "baseline":
            continue
        key = (
            row["label"], row["task_name"], int(row["layer"]),
            row["position_name"], row["condition"], row["sample_role"],
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=str):
        label, task_name, layer, position, condition, role = key
        metric_changes: list[float] = []
        dog_logit_changes: list[float] = []
        target_indicator_changes: list[float] = []
        generated_labels: list[str] = []
        for row in values:
            baseline = baselines.get((row["sample_id"], task_name))
            if baseline is None:
                continue
            metric_changes.append(
                float(row[PRIMARY_METRIC_NAME]) - float(baseline[PRIMARY_METRIC_NAME])
            )
            dog_logit_changes.append(
                float(row["concept_dog_logit"]) - float(baseline["concept_dog_logit"])
            )
            target_indicator_changes.append(
                float(bool(row["task_target_indicator"]))
                - float(bool(baseline["task_target_indicator"]))
            )
            generated_labels.append(str(row["generated_label"]))
        array = np.asarray(metric_changes, dtype=np.float64)
        if array.size == 0:
            continue
        low, high = (
            base.bootstrap_mean_ci(array, seed + len(output) * 13)
            if array.size >= 2 else (float("nan"), float("nan"))
        )
        output.append(
            {
                "label": label,
                "task_name": task_name,
                "layer": layer,
                "position_name": position,
                "condition": condition,
                "sample_role": role,
                "n": int(array.size),
                "mean_primary_metric_change": float(array.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "mean_dog_logit_change": float(np.mean(dog_logit_changes)),
                "mean_task_target_indicator_change": float(np.mean(target_indicator_changes)),
                "generated_distribution": json.dumps(
                    {
                        item: generated_labels.count(item)
                        for item in sorted(set(generated_labels))
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return output


def cross_task_consistency(
    effects: list[dict[str, Any]], tasks: list[str], layers: list[int],
    positions: list[str]
) -> list[dict[str, Any]]:
    primary = [row for row in effects if row["metric"] == PRIMARY_METRIC_NAME]
    output: list[dict[str, Any]] = []
    for layer in layers:
        for position in positions:
            total_supported = 0
            total_directional = 0
            role_details: dict[str, Any] = {}
            for role, group_name in (
                ("ablation_dog", "dog_bark"),
                ("injection_negative", "all_injection_negative"),
            ):
                cells = [
                    row for row in primary
                    if int(row["layer"]) == layer
                    and row["position_name"] == position
                    and row["sample_role"] == role
                    and row["label_group"] == group_name
                    and row["task_name"] in tasks
                ]
                by_task = {row["task_name"]: row for row in cells}
                supported_tasks: list[str] = []
                directional_tasks: list[str] = []
                for task_name in tasks:
                    cell = by_task.get(task_name)
                    if cell is None:
                        continue
                    directional = float(cell["mean_signed_net_specificity"]) > 0
                    q_raw = cell.get("bh_fdr_q_all_primary_site_tests", "")
                    q_value = float(q_raw) if q_raw not in {"", None} else 1.0
                    supported = (
                        directional
                        and float(cell["bootstrap_ci95_low"]) > 0
                        and q_value <= 0.05
                    )
                    if directional:
                        directional_tasks.append(task_name)
                    if supported:
                        supported_tasks.append(task_name)
                role_details[f"{role}_directional_tasks"] = json.dumps(directional_tasks)
                role_details[f"{role}_supported_tasks"] = json.dumps(supported_tasks)
                role_details[f"{role}_directional_count"] = len(directional_tasks)
                role_details[f"{role}_supported_count"] = len(supported_tasks)
                total_directional += len(directional_tasks)
                total_supported += len(supported_tasks)
            expected = len(tasks) * 2
            output.append(
                {
                    "layer": layer,
                    "position_name": position,
                    "expected_task_role_cells": expected,
                    "directional_cells": total_directional,
                    "supported_cells": total_supported,
                    "directional_fraction": total_directional / expected,
                    "supported_fraction": total_supported / expected,
                    "all_tasks_both_roles_supported": total_supported == expected,
                    **role_details,
                }
            )
    output.sort(
        key=lambda row: (
            -int(row["supported_cells"]), -int(row["directional_cells"]),
            int(row["layer"]), str(row["position_name"]),
        )
    )
    for rank, row in enumerate(output, 1):
        row["localization_rank"] = rank
    return output


def matched_norm_check(rows: list[dict[str, Any]]) -> tuple[float, int]:
    intervention_rows = [row for row in rows if row["condition"] != "baseline"]
    grouped: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in intervention_rows:
        key = (
            row["sample_id"], row["task_name"], int(row["layer"]),
            row["position_name"], row["sample_role"],
        )
        grouped.setdefault(key, {})[str(row["condition"])] = float(row["mean_delta_norm"])
    errors: list[float] = []
    incomplete = 0
    for key, values in grouped.items():
        role = key[-1]
        names = (
            ("dog_ablate", "matched_vehicle_ablate", "matched_random_ablate")
            if role == "ablation_dog"
            else ("dog_inject", "vehicle_inject", "random_orthogonal_inject")
        )
        if not all(name in values for name in names):
            incomplete += 1
            continue
        target = values[names[0]]
        for name in names[1:]:
            errors.append(abs(values[name] - target) / max(abs(target), 1e-12))
    return max(errors, default=0.0), incomplete


def technical_checks(
    rows: list[dict[str, Any]], failures: list[dict[str, Any]],
    selected: list[dict[str, Any]], token_rows: list[dict[str, Any]], lens: Any,
    args: argparse.Namespace, layers: list[int], tasks: list[str],
    positions: list[str], detected_target: int | None,
    zero_checks: list[dict[str, Any]], expected_intervention_jobs: int,
) -> list[dict[str, Any]]:
    intervention_rows = [row for row in rows if row["condition"] != "baseline"]
    baseline_rows = [row for row in rows if row["condition"] == "baseline"]
    expected_baselines = len(selected) * len(tasks)
    required_tokens = {"dog", "cat", "other", "animal", "bark", "sound", "vehicle"}
    passed_tokens = {row["concept"] for row in token_rows if row["status"] == "PASS"}
    max_delta_ratio = max(
        (float(row["max_actual_delta_norm_ratio"]) for row in intervention_rows),
        default=0.0,
    )
    max_norm_error, incomplete_norm_groups = matched_norm_check(rows)
    expected_selected = args.per_label * len(base.SUPPORTED_LABELS)
    return [
        {
            "check_name": "lens_metadata",
            "result": f"prompts={lens.n_prompts}, d_model={lens.d_model}",
            "status": "PASS" if lens.n_prompts == base.EXPECTED_LENS_PROMPTS and lens.d_model == base.EXPECTED_D_MODEL else "FAIL",
        },
        {
            "check_name": "l26_penultimate_target",
            "result": f"declared={args.lens_target_layer}; detected={detected_target}",
            "status": "PASS" if args.lens_target_layer == 26 and detected_target == 26 else "FAIL",
        },
        {
            "check_name": "manifest_counts",
            "result": json.dumps({label: sum(row["label"] == label for row in selected) for label in base.SUPPORTED_LABELS}, sort_keys=True),
            "status": "PASS" if len(selected) == expected_selected else "FAIL",
        },
        {
            "check_name": "required_single_tokens",
            "result": json.dumps(sorted(required_tokens & passed_tokens)),
            "status": "PASS" if required_tokens <= passed_tokens else "FAIL",
        },
        {
            "check_name": "configured_layers",
            "result": json.dumps(layers),
            "status": "PASS" if layers and set(layers) <= set(base.SOURCE_LAYERS) else "FAIL",
        },
        {
            "check_name": "four_one_token_positions",
            "result": json.dumps(positions),
            "status": "PASS" if set(positions) == set(POSITION_NAMES) and all(int(row["selected_positions"]) == 1 for row in intervention_rows) else "FAIL",
        },
        {
            "check_name": "four_cross_tasks",
            "result": json.dumps(tasks),
            "status": "PASS" if set(tasks) == set(TASK_NAMES) else "FAIL",
        },
        {
            "check_name": "baseline_completeness",
            "result": f"actual={len(baseline_rows)}; expected={expected_baselines}",
            "status": "PASS" if len(baseline_rows) == expected_baselines else "FAIL",
        },
        {
            "check_name": "intervention_completeness",
            "result": f"actual={len(intervention_rows)}; expected={expected_intervention_jobs}",
            "status": "PASS" if len(intervention_rows) == expected_intervention_jobs else "FAIL",
        },
        {
            "check_name": "zero_strength_hook_transparency_all_layers",
            "result": f"layers={len(zero_checks)}; max_error={max((float(row['max_abs_first_step_logit_error']) for row in zero_checks), default=float('nan'))}",
            "status": "PASS" if len(zero_checks) == len(layers) and all(row["status"] == "PASS" for row in zero_checks) else "FAIL",
        },
        {
            "check_name": "matched_control_delta_norm",
            "result": f"max_relative_error={max_norm_error:.8f}; incomplete_groups={incomplete_norm_groups}",
            "status": "PASS" if max_norm_error <= MATCHED_NORM_RELATIVE_TOLERANCE and incomplete_norm_groups == 0 else "FAIL",
        },
        {
            "check_name": "perturbation_magnitude",
            "result": f"maximum_actual_delta_norm_ratio={max_delta_ratio:.6f}",
            "status": "PASS" if max_delta_ratio <= MAX_ALLOWED_DELTA_RATIO else "FAIL",
        },
        {
            "check_name": "hook_application",
            "result": f"intervention_rows={len(intervention_rows)}",
            "status": "PASS" if intervention_rows and all(bool(row["hook_applied"]) for row in intervention_rows) else "FAIL",
        },
        {
            "check_name": "finite_native_logits",
            "result": f"rows={len(rows)}",
            "status": "PASS" if rows and all(bool(row["finite_first_step_logits"]) for row in rows) else "FAIL",
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


def expected_intervention_count(
    selected: list[dict[str, Any]], tasks: list[str], layers: list[int],
    positions: list[str]
) -> int:
    eligible_sample_tasks = sum(
        eligible_role(row, task_name) is not None
        for row in selected for task_name in tasks
    )
    return eligible_sample_tasks * len(layers) * len(positions) * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("smoke", "screen", "primary", "full"),
        default="smoke",
    )
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--lens-path", type=Path, default=base.DEFAULT_LENS)
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-label", type=int, default=None,
        help="Default: smoke=1, screen=10, primary/full=35.",
    )
    parser.add_argument(
        "--layers", type=str, default="",
        help="Comma-separated override. Primary default: 16,20,22,24.",
    )
    parser.add_argument(
        "--positions", type=str, default="",
        help="Comma-separated override; primary protocol uses all four sites.",
    )
    parser.add_argument(
        "--tasks", type=str, default="",
        help="Comma-separated override; primary protocol uses all four tasks.",
    )
    parser.add_argument("--lens-target-layer", type=int, default=DEFAULT_LENS_TARGET_LAYER)
    parser.add_argument("--ablation-strength", type=float, default=1.0)
    parser.add_argument("--injection-ratio", type=float, default=0.06)
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("QWEN_OMNI_DEVICE", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_label is None:
        args.per_label = PHASE_PER_LABEL[args.phase]
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")
    if not (0 < args.injection_ratio <= MAX_ALLOWED_DELTA_RATIO):
        raise ValueError("--injection-ratio must be in (0, 0.25]")
    if args.ablation_strength <= 0:
        raise ValueError("--ablation-strength must be positive")
    layers = selected_layers(args)
    tasks = selected_tasks(args)
    positions = selected_positions(args)
    if args.phase in {"primary", "full"}:
        if set(tasks) != set(TASK_NAMES):
            raise ValueError("Primary/full protocol requires all four tasks")
        if set(positions) != set(POSITION_NAMES):
            raise ValueError("Primary/full protocol requires all four positions")
    if args.lens_target_layer != 26:
        raise ValueError("Experiment 8F primary protocol requires the L26-target lens")
    detected_target, target_metadata_path = base.detect_lens_target_layer(args.lens_path)
    if detected_target != 26:
        raise ValueError(
            f"Could not verify an L26-target lens; detected={detected_target}, "
            f"metadata={target_metadata_path or 'not found'}"
        )

    phase_dir = args.output_dir / args.phase
    internal_dir = phase_dir / "internal"
    cache_dir = internal_dir / "cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = base.load_manifest(args.manifest)
    selected = base.select_manifest(manifest, args.phase, args.per_label, args.seed)
    base.write_csv(internal_dir / "selected_manifest.csv", selected)
    processor, container, thinker, lens, device, token_ids, token_rows = (
        base.load_model_and_lens(args)
    )
    del container
    base.write_csv(internal_dir / "concept_tokenization.csv", token_rows)

    directions_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    direction_rows: list[dict[str, Any]] = []
    for layer in layers:
        directions, diagnostics = base.build_jspace_directions(
            thinker, lens, layer, token_ids, args.seed + layer * 1009
        )
        directions_by_layer[layer] = directions
        direction_rows.extend(diagnostics)
    base.write_csv(internal_dir / "direction_diagnostics.csv", direction_rows)

    expected_jobs = expected_intervention_count(selected, tasks, layers, positions)
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
        "layers": layers,
        "positions": positions,
        "position_policy": "one_token_per_site",
        "tasks": tasks,
        "prompts": {name: asdict(base.PROMPTS[name]) for name in tasks},
        "per_label": args.per_label,
        "selected_counts": {label: sum(row["label"] == label for row in selected) for label in base.SUPPORTED_LABELS},
        "ablation_strength": args.ablation_strength,
        "injection_ratio": args.injection_ratio,
        "matched_controls": ["vehicle", "random_orthogonal"],
        "expected_intervention_jobs": expected_jobs,
        "expected_baseline_jobs": len(selected) * len(tasks),
        "max_new_tokens": args.max_new_tokens,
        "prefill_only_intervention": True,
        "greedy_generation": True,
        "seed": args.seed,
        "created_at": base.now_iso(),
    }
    config["run_signature"] = base.stable_hash(config)
    base.write_json(internal_dir / "frozen_config.json", config)

    zero_checks: list[dict[str, Any]] = []
    for layer in layers:
        check = base.zero_strength_transparency_check(
            selected[0], processor, thinker, device,
            directions_by_layer[layer], layer, args.max_new_tokens,
        )
        check["layer"] = layer
        zero_checks.append(check)
    base.write_csv(internal_dir / "zero_strength_hook_checks.csv", zero_checks)
    if not all(row["status"] == "PASS" for row in zero_checks):
        raise RuntimeError("Zero-strength transparency failed on at least one layer")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    position_audits: list[dict[str, Any]] = []
    total_jobs = len(selected) * len(tasks) + expected_jobs
    completed_jobs = 0

    for sample_index, row in enumerate(selected, 1):
        for task_name in tasks:
            prompt = base.PROMPTS[task_name]
            raw_inputs = base.prepare_audio_inputs(processor, row, prompt)
            positions_by_name, position_audit = one_token_positions(
                raw_inputs,
                processor.tokenizer,
                int(base.model_config_value(thinker, "audio_token_id")),
            )
            position_audits.append(
                {"sample_id": row["sample_id"], "task_name": task_name, **position_audit}
            )
            inputs = base.move_inputs_to_device(
                raw_inputs, device, next(thinker.parameters()).dtype
            )
            role = eligible_role(row, task_name)

            baseline = baseline_condition()
            completed_jobs += 1
            print(
                f"[{completed_jobs:06d}/{total_jobs:06d}] {row['sample_id']} "
                f"{task_name} baseline",
                flush=True,
            )
            try:
                result = cache_result(
                    cache_dir,
                    config["run_signature"],
                    {
                        "sample_id": row["sample_id"], "task_name": task_name,
                        "condition": "baseline",
                    },
                    lambda: run_prepared(
                        row, prompt, baseline, role or "ineligible_baseline",
                        positions_by_name, position_audit, inputs, processor,
                        thinker, token_ids, directions_by_layer[layers[0]],
                        layers[0], args.max_new_tokens,
                    ),
                    args.no_resume,
                )
                results.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": row["sample_id"], "task_name": task_name,
                        "layer": "", "position_name": "baseline",
                        "condition": "baseline", "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc(),
                        "failed_at": base.now_iso(),
                    }
                )
                base.write_csv(phase_dir / "8F_failures.csv", failures)
                if not args.continue_on_error:
                    raise

            if role is not None:
                for layer in layers:
                    directions = directions_by_layer[layer]
                    for position_name in positions:
                        for condition in intervention_conditions(
                            role, position_name, args.ablation_strength,
                            args.injection_ratio,
                        ):
                            completed_jobs += 1
                            print(
                                f"[{completed_jobs:06d}/{total_jobs:06d}] "
                                f"{row['sample_id']} {task_name} L{layer} "
                                f"{position_name} {condition.name}",
                                flush=True,
                            )
                            try:
                                result = cache_result(
                                    cache_dir,
                                    config["run_signature"],
                                    {
                                        "sample_id": row["sample_id"],
                                        "task_name": task_name,
                                        "layer": layer,
                                        "position_name": position_name,
                                        "condition": asdict(condition),
                                    },
                                    lambda c=condition, d=directions, l=layer: run_prepared(
                                        row, prompt, c, role, positions_by_name,
                                        position_audit, inputs, processor, thinker,
                                        token_ids, d, l, args.max_new_tokens,
                                    ),
                                    args.no_resume,
                                )
                                results.append(result)
                            except Exception as exc:
                                failures.append(
                                    {
                                        "sample_id": row["sample_id"],
                                        "task_name": task_name, "layer": layer,
                                        "position_name": position_name,
                                        "condition": condition.name,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                        "traceback": traceback.format_exc(),
                                        "failed_at": base.now_iso(),
                                    }
                                )
                                base.write_csv(phase_dir / "8F_failures.csv", failures)
                                if not args.continue_on_error:
                                    raise
                            if completed_jobs % args.checkpoint_every == 0:
                                base.write_csv(phase_dir / "8F_results.partial.csv", results)
            del inputs, raw_inputs
            torch.cuda.empty_cache()

    base.write_csv(internal_dir / "position_audit.csv", position_audits)
    base.write_csv(phase_dir / "8F_results.csv", results)
    raw_summary = raw_condition_summary(results, args.seed)
    effects = site_effects(results, args.seed + 1_000_000)
    consistency = cross_task_consistency(effects, tasks, layers, positions)
    baseline_rows = [row for row in results if row["condition"] == "baseline"]
    gate_phase = "smoke" if args.phase == "smoke" else "primary"
    baseline_gate = base.baseline_gate(baseline_rows, selected, gate_phase)
    checks = technical_checks(
        results, failures, selected, token_rows, lens, args, layers, tasks,
        positions, detected_target, zero_checks, expected_jobs,
    )
    technical_status = (
        "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    )
    base.write_csv(phase_dir / "8F_raw_condition_summary.csv", raw_summary)
    base.write_csv(phase_dir / "8F_site_effects.csv", effects)
    base.write_csv(phase_dir / "8F_cross_task_consistency.csv", consistency)
    base.write_csv(phase_dir / "8F_baseline_gate.csv", baseline_gate)
    base.write_csv(phase_dir / "8F_technical_check.csv", checks)
    base.write_json(
        phase_dir / "8F_summary.json",
        {
            "script_version": SCRIPT_VERSION,
            "phase": args.phase,
            "samples": len(selected),
            "layers": layers,
            "positions": positions,
            "tasks": tasks,
            "jobs": total_jobs,
            "result_rows": len(results),
            "failure_rows": len(failures),
            "technical_status": technical_status,
            "top_localization_sites": consistency[:10],
            "completed_at": base.now_iso(),
        },
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(consistency[:10], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
