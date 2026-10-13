"""Experiment 9G: validate a text-fitted J-lens on audio-conditioned states.

The existing 4A lens is fitted on text prompts.  This experiment does not fit
or refit a lens.  It runs the audio pipeline, captures source-layer and L26
hidden states during the same prefill pass, and evaluates the frozen linear
Jacobian map ``J_l @ h_l`` against the observed L26 state.

The output is deliberately a validation artifact, not a claim that a good
reconstruction proves semantic validity.  It reports raw and centered cosine,
relative L2 error, raw lm-head agreement, and layer/prompt/label breakdowns.
"""

from __future__ import annotations

import project_paths as paths

import argparse
import csv
import json
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_prompt_invariance as probe
import evaluate_hierarchical_behavior as base
from qwen_omni_utils import process_mm_info


SCRIPT_VERSION = "9g-audio-jlens-validation-1.0.0"
SOURCE_LAYERS = tuple(base.SOURCE_LAYERS)
TARGET_LAYER = 26
DEFAULT_OUTPUT = Path(
    str(paths.OUTPUTS_ROOT / "validate_audio_jacobian_lens")
)
DEFAULT_PROMPTS = ("neutral_ack", "plain_loudness", "event_count", "species_control")


class LayerCapture:
    """Capture one residual-stream vector on the first prefill call."""

    def __init__(self, layer_module: Any, prompt_length: int, position: int):
        self.layer_module = layer_module
        self.prompt_length = prompt_length
        self.position = position
        self.vector: torch.Tensor | None = None
        self.calls = 0
        self.handle: Any = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = base.output_hidden(output)
        if (
            self.vector is None
            and hidden.ndim == 3
            and hidden.shape[1] == self.prompt_length
            and self.position < hidden.shape[1]
        ):
            self.vector = hidden[0, self.position].detach().float().cpu()
            self.calls += 1
        return output

    def __enter__(self) -> "LayerCapture":
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "primary"), default="smoke")
    parser.add_argument("--model-path", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument("--lens-path", type=Path, default=base.DEFAULT_LENS)
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-label", type=int, default=None,
                        help="Samples per label; defaults to 1 for smoke, 3 for primary.")
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("QWEN_OMNI_DEVICE", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a[None].float(), b[None].float(), dim=-1).item())


def metrics(pred: torch.Tensor, actual: torch.Tensor, lm_head: Any) -> dict[str, Any]:
    pred = pred.float().cpu()
    actual = actual.float().cpu()
    error = pred - actual
    actual_norm = actual.norm().clamp_min(1e-12)
    pred_norm = pred.norm().clamp_min(1e-12)
    pred_centered = pred - pred.mean()
    actual_centered = actual - actual.mean()
    centered_error = pred_centered - actual_centered
    lm_weight = lm_head.weight
    lm_device = lm_weight.device
    lm_dtype = lm_weight.dtype
    pred_logits = lm_head(pred[None].to(device=lm_device, dtype=lm_dtype)).float().cpu()[0]
    actual_logits = lm_head(actual[None].to(device=lm_device, dtype=lm_dtype)).float().cpu()[0]
    top1_pred = int(pred_logits.argmax().item())
    top1_actual = int(actual_logits.argmax().item())
    top10_pred = set(torch.topk(pred_logits, k=10).indices.tolist())
    top10_actual = set(torch.topk(actual_logits, k=10).indices.tolist())
    return {
        "relative_l2_error": float(error.norm().item() / actual_norm.item()),
        "cosine_similarity": cosine(pred, actual),
        "centered_relative_l2_error": float(centered_error.norm().item() / actual_centered.norm().clamp_min(1e-12).item()),
        "centered_cosine_similarity": cosine(pred_centered, actual_centered),
        "raw_lm_head_mae": float((pred_logits - actual_logits).abs().mean().item()),
        "raw_lm_head_rmse": float(torch.sqrt(((pred_logits - actual_logits) ** 2).mean()).item()),
        "raw_lm_head_top1_match": bool(top1_pred == top1_actual),
        "raw_lm_head_top10_jaccard": float(len(top10_pred & top10_actual) / max(1, len(top10_pred | top10_actual))),
        "pred_norm": float(pred_norm.item()),
        "actual_norm": float(actual_norm.item()),
        "finite": bool(torch.isfinite(pred).all() and torch.isfinite(actual).all()),
    }


def prepare_audio_inputs(processor: Any, prompt: probe.ProbePrompt, audio_path: str) -> dict[str, Any]:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": base.SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt.text},
        ]},
    ]
    rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=rendered,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return {key: value for key, value in inputs.items()}


def run_one(
    row: dict[str, Any],
    prompt: probe.ProbePrompt,
    processor: Any,
    thinker: Any,
    lens: Any,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    raw_inputs = prepare_audio_inputs(processor, prompt, str(row["audio_path"]))
    attention_mask = raw_inputs.get("attention_mask", torch.ones_like(raw_inputs["input_ids"]))
    valid_len = int(attention_mask[0].long().sum().item())
    decision_index = valid_len - 1
    _, position_metadata = base.locate_positions(
        raw_inputs, processor.tokenizer, int(base.model_config_value(thinker, "audio_token_id"))
    )
    inputs = base.move_inputs_to_device(raw_inputs, device, next(thinker.parameters()).dtype)
    prompt_length = int(inputs["input_ids"].shape[1])
    captures = {
        layer: LayerCapture(thinker.model.layers[layer], prompt_length, decision_index)
        for layer in (*SOURCE_LAYERS, TARGET_LAYER)
    }
    contexts = []
    try:
        for capture in captures.values():
            contexts.append(capture.__enter__())
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
        with torch.inference_mode():
            thinker.generate(**generation_kwargs)
    finally:
        for capture in reversed(contexts):
            capture.__exit__(None, None, None)
    if any(captures[layer].vector is None for layer in (*SOURCE_LAYERS, TARGET_LAYER)):
        missing = [layer for layer, capture in captures.items() if capture.vector is None]
        raise RuntimeError(f"Missing prefill capture for layers {missing}")
    target = captures[TARGET_LAYER].vector
    assert target is not None
    rows: list[dict[str, Any]] = []
    lm_head = thinker.lm_head
    for layer in SOURCE_LAYERS:
        source = captures[layer].vector
        assert source is not None
        matrix = lens.jacobians[layer].detach().float().cpu()
        if matrix.ndim != 2 or tuple(matrix.shape) != (target.numel(), source.numel()):
            raise ValueError(f"Unexpected J{layer} shape {tuple(matrix.shape)} for d={source.numel()}")
        predicted = matrix @ source
        rows.append({
            "sample_id": row["sample_id"],
            "label": row["label"],
            "source": row.get("source", ""),
            "audio_path": row["audio_path"],
            "prompt_name": prompt.name,
            "prompt_family": prompt.family,
            "layer": layer,
            "target_layer": TARGET_LAYER,
            "readout_position": "decision",
            **position_metadata,
            **metrics(predicted, target, lm_head),
        })
    return rows


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)
    metrics_names = (
        "relative_l2_error", "cosine_similarity", "centered_relative_l2_error",
        "centered_cosine_similarity", "raw_lm_head_mae", "raw_lm_head_rmse",
        "raw_lm_head_top1_match", "raw_lm_head_top10_jaccard",
    )
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        out = {name: value for name, value in zip(group_keys, key)}
        out["n"] = len(items)
        for metric_name in metrics_names:
            values = np.asarray([float(item[metric_name]) for item in items], dtype=float)
            out[f"{metric_name}_mean"] = float(values.mean())
            out[f"{metric_name}_median"] = float(np.median(values))
            out[f"{metric_name}_p05"] = float(np.quantile(values, 0.05))
            out[f"{metric_name}_p95"] = float(np.quantile(values, 0.95))
        output.append(out)
    return output


def main() -> None:
    args = parse_args()
    if args.per_label is None:
        args.per_label = 1 if args.phase == "smoke" else 3
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")
    prompt_names = tuple(name.strip() for name in args.prompts.split(",") if name.strip())
    unknown = sorted(set(prompt_names) - set(probe.AUDIO_PROMPT_NAMES))
    if unknown:
        raise ValueError(f"Unknown audio prompts: {unknown}")
    phase_dir = args.output_dir / args.phase
    internal_dir = phase_dir / "internal"
    cache_dir = internal_dir / "cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.load_manifest(args.manifest)
    selected = base.select_manifest(manifest, args.phase, args.per_label, args.seed)
    base.write_csv(internal_dir / "selected_manifest.csv", selected)
    fake_args = argparse.Namespace(
        model_path=args.model_path,
        lens_path=args.lens_path,
        device_index=args.device_index,
    )
    detected_target, target_metadata_path = base.detect_lens_target_layer(args.lens_path)
    if detected_target is not None and detected_target != TARGET_LAYER:
        raise ValueError(f"Lens metadata target is L{detected_target}, expected L{TARGET_LAYER}")
    processor, container, thinker, lens, device, token_ids, token_rows = base.load_model_and_lens(fake_args)
    del token_ids, token_rows
    config = {
        "script_version": SCRIPT_VERSION,
        "phase": args.phase,
        "model_path": str(args.model_path),
        "lens_path": str(args.lens_path),
        "lens_sha256": base.sha256_file(args.lens_path),
        "lens_source_layers": list(SOURCE_LAYERS),
        "lens_target_layer": TARGET_LAYER,
        "detected_lens_target_layer": detected_target,
        "target_metadata_path": target_metadata_path,
        "manifest_path": str(args.manifest),
        "manifest_sha256": base.sha256_file(args.manifest),
        "selected_counts": {label: sum(row["label"] == label for row in selected) for label in base.SUPPORTED_LABELS},
        "prompt_names": list(prompt_names),
        "per_label": args.per_label,
        "readout_position": "decision",
        "mapping": "predicted_L26 = J_l @ source_hidden_Ll; no audio refit or affine correction",
        "created_at": base.now_iso(),
    }
    config["run_signature"] = base.stable_hash(config)
    write_json(internal_dir / "frozen_config.json", config)
    # One generation captures all source layers and L26.  The output is
    # expanded to one row per layer below.
    jobs = [(row, probe.PROMPTS[name]) for row in selected for name in prompt_names]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (row, prompt) in enumerate(jobs, 1):
        cache_key = base.stable_hash({"run_signature": config["run_signature"], "sample_id": row["sample_id"], "prompt_name": prompt.name})
        cache_path = cache_dir / f"{cache_key}.json"
        print(f"[{index:05d}/{len(jobs):05d}] {row['sample_id']} {prompt.name} (L{SOURCE_LAYERS[0]}..L{SOURCE_LAYERS[-1]} -> L{TARGET_LAYER})", flush=True)
        try:
            if cache_path.is_file() and not args.no_resume:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("run_signature") == config["run_signature"]:
                    one_rows = payload["rows"]
                else:
                    one_rows = run_one(row, prompt, processor, thinker, lens, device, args.max_new_tokens)
            else:
                one_rows = run_one(row, prompt, processor, thinker, lens, device, args.max_new_tokens)
            base.write_json(cache_path, {"run_signature": config["run_signature"], "rows": one_rows})
            # run_one captures all layers in one pass.
            results.extend(one_rows)
        except Exception as exc:
            failure = {"sample_id": row["sample_id"], "prompt_name": prompt.name, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "failed_at": base.now_iso()}
            failures.append(failure)
            write_csv(phase_dir / "9G_failures.csv", failures)
            if not args.continue_on_error:
                raise
        finally:
            torch.cuda.empty_cache()
    by_layer_prompt = summarize(results, ("layer", "prompt_name"))
    by_layer_label = summarize(results, ("layer", "label"))
    by_layer = summarize(results, ("layer",))
    checks = [
        {"check_name": "manifest_counts", "result": json.dumps(config["selected_counts"], ensure_ascii=False), "status": "PASS" if len(selected) == args.per_label * len(base.SUPPORTED_LABELS) else "FAIL"},
        {"check_name": "lens_target", "result": f"detected={detected_target}; expected={TARGET_LAYER}", "status": "PASS" if detected_target in (None, TARGET_LAYER) else "FAIL"},
        {"check_name": "source_layers", "result": str(list(SOURCE_LAYERS)), "status": "PASS" if len(results) == len(selected) * len(prompt_names) * len(SOURCE_LAYERS) else "FAIL"},
        {"check_name": "finite_metrics", "result": f"rows={len(results)}", "status": "PASS" if results and all(row["finite"] for row in results) else "FAIL"},
        {"check_name": "technical_overall", "result": "audio hidden-state validation completed", "status": "PASS" if not failures and results else "FAIL"},
    ]
    write_csv(phase_dir / "9G_audio_jlens_results.csv", results)
    write_csv(phase_dir / "9G_summary_by_layer_prompt.csv", by_layer_prompt)
    write_csv(phase_dir / "9G_summary_by_layer_label.csv", by_layer_label)
    write_csv(phase_dir / "9G_summary_by_layer.csv", by_layer)
    write_csv(phase_dir / "9G_technical_check.csv", checks)
    write_json(phase_dir / "9G_summary.json", {
        "script_version": SCRIPT_VERSION,
        "phase": args.phase,
        "samples": len(selected),
        "prompts": list(prompt_names),
        "source_layers": list(SOURCE_LAYERS),
        "target_layer": TARGET_LAYER,
        "jobs": len(selected) * len(prompt_names),
        "result_rows": len(results),
        "failure_rows": len(failures),
        "technical_status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "interpretation": "Metrics validate frozen text-fitted J_l @ h_l on audio-conditioned states; they do not fit an audio lens or prove semantic causality.",
        "completed_at": base.now_iso(),
    })
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
