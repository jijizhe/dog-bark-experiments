"""Inspect the Qwen2.5-Omni module tree needed for J-lens adaptation.

This script loads the model on one GPU, but does not run audio inference or
generate text. It records the model hierarchy and likely Transformer layer
stacks so the J-lens hook points can be chosen from the actual checkpoint.
"""

from __future__ import annotations

import project_paths as paths

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5OmniForConditionalGeneration


MODEL_PATH = Path(str(paths.QWEN_MODEL))
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))
OUTPUT_DIR = Path(str(paths.OUTPUTS_ROOT / "inspect_qwen_architecture"))


def module_summary(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.parameters(recurse=False))
    return {
        "class": type(module).__name__,
        "parameter_count_direct": sum(p.numel() for p in parameters),
        "children": [name for name, _ in module.named_children()],
    }


def find_layer_stacks(model: torch.nn.Module) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 4:
            continue
        first = module[0]
        candidates.append(
            {
                "module_name": name or "<root>",
                "container_class": type(module).__name__,
                "length": len(module),
                "block_class": type(first).__name__,
                "block_children": [child for child, _ in first.named_children()],
                "first_block_summary": module_summary(first),
            }
        )
    return candidates


def choose_language_stack(model: torch.nn.Module, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Prefer stacks under thinker and stacks whose block class looks like a
    # decoder layer. This is only a report heuristic; it never changes the
    # model or silently selects a J-lens target layer.
    ranked: list[tuple[int, dict[str, Any]]] = []
    for candidate in candidates:
        name = candidate["module_name"].lower()
        block = candidate["block_class"].lower()
        score = 0
        if "thinker" in name:
            score += 100
        if "layer" in name:
            score += 10
        if "decoder" in block or "qwen" in block:
            score += 5
        ranked.append((score, candidate))
    return max(ranked, key=lambda item: (item[0], item[1]["length"]))[1] if ranked else None


def inspect_model() -> dict[str, Any]:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for structure inspection")

    device = torch.device(f"cuda:{DEVICE_INDEX}")
    print(f"Loading model from {MODEL_PATH}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": device.index},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()

    candidates = find_layer_stacks(model)
    selected = choose_language_stack(model, candidates)
    direct_children = {
        "<root>": [name for name, _ in model.named_children()],
    }
    for name, child in model.named_children():
        direct_children[name] = [child_name for child_name, _ in child.named_children()]

    report: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "model_class": type(model).__name__,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config_class": type(model.config).__name__,
        "config": model.config.to_dict(),
        "direct_children": direct_children,
        "module_count": sum(1 for _ in model.named_modules()),
        "layer_stack_candidates": candidates,
        "heuristic_language_stack": selected,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "structure.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        f"model_class: {report['model_class']}",
        f"config_class: {report['config_class']}",
        f"device: {report['device']}",
        f"dtype: {report['dtype']}",
        f"parameter_count: {report['parameter_count']:,}",
        f"module_count: {report['module_count']:,}",
        "",
        "Direct children:",
    ]
    for name, children in direct_children.items():
        lines.append(f"  {name}: {children}")
    lines.extend(["", "ModuleList layer-stack candidates:"])
    for candidate in candidates:
        lines.append(
            f"  {candidate['module_name']} | length={candidate['length']} | "
            f"block={candidate['block_class']} | children={candidate['block_children']}"
        )
    lines.extend(["", f"Heuristic language stack: {selected}", "", f"JSON report: {json_path}"])
    text_path = OUTPUT_DIR / "structure.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    del model
    torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    try:
        inspect_model()
    except Exception as exc:
        print(f"Structure inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
