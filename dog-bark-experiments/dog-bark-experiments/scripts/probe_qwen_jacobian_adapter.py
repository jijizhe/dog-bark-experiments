"""Check whether Qwen2.5-Omni's text thinker can be adapted by jlens.

The probe does not fit a Lens and does not process audio. It tests candidate
objects with a short text-only forward pass and with ``jlens.from_hf``.
"""

from __future__ import annotations

import project_paths as paths

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


MODEL_PATH = Path(str(paths.QWEN_MODEL))
OUTPUT_DIR = Path(str(paths.OUTPUTS_ROOT / "probe_qwen_jacobian_adapter"))
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))
PROBE_TEXT = "This is a short calibration sentence for a language model."


def one_line_signature(function: Any) -> str:
    try:
        return str(inspect.signature(function))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def direct_forward(candidate: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    with torch.inference_mode():
        output = candidate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = getattr(output, "last_hidden_state", None)
    hidden_states = getattr(output, "hidden_states", None)
    result: dict[str, Any] = {
        "output_class": type(output).__name__,
        "has_logits": getattr(output, "logits", None) is not None,
        "hidden_state_count": len(hidden_states) if hidden_states is not None else None,
    }
    if torch.is_tensor(hidden):
        result["last_hidden_state_shape"] = list(hidden.shape)
    if hidden_states:
        result["first_hidden_state_shape"] = list(hidden_states[0].shape)
        result["last_hidden_state_from_tuple_shape"] = list(hidden_states[-1].shape)
    return result


def try_jlens(candidate: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    try:
        import jlens
    except Exception as exc:
        return {
            "status": "jlens_import_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        lens_model = jlens.from_hf(candidate, tokenizer, force_bos=True)
        return {
            "status": "adapter_created",
            "adapter_class": type(lens_model).__name__,
            "n_layers": getattr(lens_model, "n_layers", None),
            "d_model": getattr(lens_model, "d_model", None),
            "source_layers": getattr(lens_model, "source_layers", None),
        }
    except Exception as exc:
        return {
            "status": "adapter_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> None:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the probe")

    device = torch.device(f"cuda:{DEVICE_INDEX}")
    print(f"Loading {MODEL_PATH} on {device}")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": device.index},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.disable_talker()
    model.eval()

    encoded = processor.tokenizer(
        PROBE_TEXT,
        return_tensors="pt",
        add_special_tokens=True,
    )
    text_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }

    candidates = {
        "model.thinker": model.thinker,
        "model.thinker.model": getattr(model.thinker, "model", None),
    }
    report: dict[str, Any] = {
        "model_class": type(model).__name__,
        "thinker_class": type(model.thinker).__name__,
        "device": str(device),
        "probe_text": PROBE_TEXT,
        "token_count": int(text_inputs["input_ids"].shape[1]),
        "candidates": {},
    }

    for name, candidate in candidates.items():
        if candidate is None:
            report["candidates"][name] = {"status": "missing"}
            continue
        item: dict[str, Any] = {
            "class": type(candidate).__name__,
            "forward_signature": one_line_signature(candidate.forward),
            "layer_count": len(candidate.model.layers)
            if hasattr(candidate, "model") and hasattr(candidate.model, "layers")
            else (len(candidate.layers) if hasattr(candidate, "layers") else None),
        }
        try:
            item["text_forward"] = direct_forward(candidate, text_inputs)
        except Exception as exc:
            item["text_forward"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        item["jlens"] = try_jlens(candidate, processor.tokenizer)
        report["candidates"][name] = item
        print(f"\n{name}: {json.dumps(item, ensure_ascii=False, indent=2, default=str)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "jlens_probe.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nProbe report: {output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
