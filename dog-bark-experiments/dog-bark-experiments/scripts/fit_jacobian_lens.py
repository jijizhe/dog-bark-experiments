"""Jacobian-lens fitting for Qwen2.5-Omni-7B (paper stage 4A).

This stage uses only text calibration data. It fits two document-disjoint
500-prompt lenses, reserves an independent 200-prompt text validation set,
and runs adapter, layer, gradient, numerical, reload, and validation checks.
The audio experiments belong to 4B and must use this Lens without refitting.
"""

from __future__ import annotations

import project_paths as paths

import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

try:
    import jlens
    from jlens.hf import Layout
    from jlens.hooks import ActivationRecorder
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "jlens is not installed. Run:\n"
        "python -m pip install --no-deps "
        "git+https://github.com/anthropics/jacobian-lens.git"
    ) from exc


# Frozen 4A configuration.
MODEL_PATH = str(paths.QWEN_MODEL)
CALIBRATION_INPUT_PATHS = [
    str(paths.CALIBRATION_EN),
    str(paths.CALIBRATION_ZH),
]
OUTPUT_DIR = str(paths.OUTPUTS_ROOT / "fit_jacobian_lens")
PROMPTS_PER_HALF = 500
VALIDATION_PER_LANGUAGE = 100
LANGUAGE_TARGETS_PER_HALF = {"zh": 250, "en": 250}
SEQUENCE_LENGTH = 128
VALIDATION_MAX_SEQ_LEN = 128
RANDOM_SEED = 42

# thinker.model.layers has 28 decoder blocks indexed 0..27.
SOURCE_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24]
TARGET_LAYER = 26
DIM_BATCH = 4
SKIP_FIRST_POSITIONS = 16
CHECKPOINT_EVERY = 5
RESUME = True
FIT_DTYPE = torch.bfloat16
DEVICE_INDEX = int(os.environ.get("QWEN_OMNI_DEVICE", "0"))

LAYER_ALIGNMENT_TOL = 1e-3
ADAPTER_MAX_LOGIT_ERROR = 5e-2
GRADIENT_REL_ERROR_TOL = 0.35
# When the analytic directional derivative is very close to zero, the sign
# of a BF16 central difference is not meaningful.  Use an explicit absolute
# error criterion in that regime instead of forcing sign agreement.
GRADIENT_NEAR_ZERO_DERIVATIVE = 0.02
GRADIENT_NEAR_ZERO_ABS_ERROR = 0.02
# Qwen2.5-Omni is loaded in BF16 for the fit.  A 1% activation perturbation
# is sometimes smaller than one BF16 quantization step, so the central
# difference can be dominated by rounding.  This fixed scale
# is large enough to be measurable while remaining a local perturbation.
GRADIENT_FD_SCALE_FRACTION = 0.10


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    language: str
    source_path: str


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    document_id: str
    language: str
    prompt: str
    token_count: int
    prompt_sha256: str
    source_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk >= max(2, latin // 2):
        return "zh"
    if latin >= 2:
        return "en"
    return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_documents(paths: Iterable[str]) -> list[Document]:
    documents: list[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration corpus not found: {path}")
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Calibration input must be JSONL: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = normalize_text(str(item.get("text", "")))
                if not text:
                    continue
                language = str(item.get("language", "")).lower().strip()
                if language not in {"zh", "en"}:
                    language = detect_language(text)
                documents.append(
                    Document(
                        document_id=str(item.get("document_id", f"{path.name}:line_{line_no}")),
                        text=text,
                        language=language,
                        source_path=str(path),
                    )
                )

    unique: dict[str, Document] = {}
    for document in documents:
        unique.setdefault(sha256_text(document.text), document)
    return list(unique.values())


def build_records(
    tokenizer: Any, documents: list[Document]
) -> tuple[list[PromptRecord], list[dict[str, Any]]]:
    records: list[PromptRecord] = []
    rejected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for document in documents:
        if document.language not in {"zh", "en"}:
            rejected.append({"document_id": document.document_id, "reason": "language_unknown"})
            continue
        # Truncate at the protocol length during tokenization. Encoding a very
        # long FineWeb document in one call only creates a misleading HF warning
        # (and needlessly materializes tens of thousands of token ids).
        token_ids = tokenizer(
            document.text,
            add_special_tokens=False,
            truncation=True,
            max_length=SEQUENCE_LENGTH,
        )["input_ids"]
        if len(token_ids) < SEQUENCE_LENGTH:
            rejected.append({"document_id": document.document_id, "reason": "too_short", "token_count": len(token_ids)})
            continue
        prompt = normalize_text(tokenizer.decode(
            token_ids[:SEQUENCE_LENGTH],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))
        reencoded = tokenizer.encode(prompt, add_special_tokens=True)
        if not prompt or len(reencoded) < SEQUENCE_LENGTH:
            rejected.append({"document_id": document.document_id, "reason": "decode_reencode_too_short", "token_count": len(reencoded)})
            continue
        prompt_hash = sha256_text(prompt)
        if prompt_hash in seen_prompts:
            rejected.append({"document_id": document.document_id, "reason": "duplicate_prompt"})
            continue
        seen_prompts.add(prompt_hash)
        records.append(PromptRecord(
            prompt_id=f"cal_{len(records) + 1:06d}",
            document_id=document.document_id,
            language=document.language,
            prompt=prompt,
            token_count=min(len(reencoded), SEQUENCE_LENGTH),
            prompt_sha256=prompt_hash,
            source_path=document.source_path,
        ))
    return records, rejected


def choose_document_disjoint_records(
    records: list[PromptRecord], language: str, rng: random.Random
) -> tuple[list[PromptRecord], list[PromptRecord], list[PromptRecord]]:
    by_document: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in records:
        if record.language == language:
            by_document[record.document_id].append(record)
    multi_record = {key: len(value) for key, value in by_document.items() if len(value) != 1}
    if multi_record:
        raise ValueError(f"Expected one prompt per document for {language}; got {list(multi_record.items())[:5]}")

    document_ids = list(by_document)
    rng.shuffle(document_ids)
    required_train = PROMPTS_PER_HALF
    required_total = required_train + VALIDATION_PER_LANGUAGE
    if len(document_ids) < required_total:
        raise ValueError(f"{language} needs {required_total} usable documents, found {len(document_ids)}")

    validation_ids = document_ids[:VALIDATION_PER_LANGUAGE]
    train_ids = document_ids[VALIDATION_PER_LANGUAGE:VALIDATION_PER_LANGUAGE + required_train]
    validation = [by_document[key][0] for key in validation_ids]
    train = [by_document[key][0] for key in train_ids]
    cut = LANGUAGE_TARGETS_PER_HALF[language]
    half_a, half_b = train[:cut], train[cut:]
    if len(half_a) != cut or len(half_b) != cut:
        raise AssertionError(f"Invalid {language} split sizes")
    return half_a, half_b, validation


def split_fit_and_validation(
    records: list[PromptRecord],
) -> tuple[list[PromptRecord], list[PromptRecord], list[PromptRecord]]:
    rng = random.Random(RANDOM_SEED)
    half_a: list[PromptRecord] = []
    half_b: list[PromptRecord] = []
    validation: list[PromptRecord] = []
    for language in ("zh", "en"):
        selected_a, selected_b, selected_validation = choose_document_disjoint_records(records, language, rng)
        half_a.extend(selected_a)
        half_b.extend(selected_b)
        validation.extend(selected_validation)
    rng.shuffle(half_a)
    rng.shuffle(half_b)
    rng.shuffle(validation)
    docs_a = {record.document_id for record in half_a}
    docs_b = {record.document_id for record in half_b}
    docs_validation = {record.document_id for record in validation}
    if docs_a & docs_b or (docs_a | docs_b) & docs_validation:
        raise AssertionError("Fit/validation document overlap detected")
    if len(half_a) != PROMPTS_PER_HALF or len(half_b) != PROMPTS_PER_HALF:
        raise AssertionError(f"Expected 500/500 fit prompts, got {len(half_a)}/{len(half_b)}")
    if len(validation) != 2 * VALIDATION_PER_LANGUAGE:
        raise AssertionError(f"Expected 200 validation prompts, got {len(validation)}")
    return half_a, half_b, validation


def records_to_rows(records: list[PromptRecord], group: str) -> list[dict[str, Any]]:
    return [{
        "group": group,
        "prompt_id": record.prompt_id,
        "document_id": record.document_id,
        "language": record.language,
        "token_count": record.token_count,
        "prompt_sha256": record.prompt_sha256,
        "source_path": record.source_path,
        "prompt": record.prompt,
    } for record in records]


def load_model_and_lens(processor: Any) -> tuple[Any, Any]:
    device = torch.device(f"cuda:{DEVICE_INDEX}")
    container = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=FIT_DTYPE,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map={"": device.index},
        attn_implementation="sdpa",
    )
    container.disable_talker()
    thinker = container.thinker
    thinker.eval()
    thinker.config.use_cache = False
    for parameter in thinker.parameters():
        parameter.requires_grad_(False)
    lens_model = jlens.from_hf(thinker, processor.tokenizer, layout=Layout(path="model"), force_bos=True)
    if lens_model.n_layers != 28 or lens_model.d_model != 3584:
        raise ValueError(f"Unexpected thinker geometry: layers={lens_model.n_layers}, d_model={lens_model.d_model}")
    return container, lens_model


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


def check_layer_alignment(container: Any, lens_model: Any, prompt: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layers = [*SOURCE_LAYERS, TARGET_LAYER]
    input_ids = lens_model.encode(prompt, max_length=40)
    with torch.no_grad():
        with ActivationRecorder(lens_model.layers, at=layers) as recorder:
            outputs = container.thinker(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    details: list[dict[str, Any]] = []
    for layer in layers:
        hook_hidden = recorder.activations[layer].float()
        native_hidden = outputs.hidden_states[layer + 1].float()
        difference = (hook_hidden - native_hidden).abs()
        max_error = float(difference.max().item())
        cosine = float(F.cosine_similarity(hook_hidden.reshape(-1, hook_hidden.shape[-1]), native_hidden.reshape(-1, native_hidden.shape[-1]), dim=-1).mean().item())
        details.append({"layer": layer, "max_abs_error": max_error, "mean_cosine": cosine, "status": "PASS" if max_error <= LAYER_ALIGNMENT_TOL else "FAIL"})
    passed = all(row["status"] == "PASS" for row in details)
    return ({"check_name": "layer_alignment", "result": f"{sum(row['status'] == 'PASS' for row in details)}/{len(details)} layers aligned", "status": "PASS" if passed else "FAIL"}, details)


def check_adapter_transparency(container: Any, lens_model: Any, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    input_ids = lens_model.encode(prompt, max_length=40)
    final_layer = lens_model.n_layers - 1
    with torch.no_grad():
        raw_outputs = container.thinker(input_ids=input_ids, use_cache=False, return_dict=True)
        with ActivationRecorder(lens_model.layers, at=[final_layer]) as recorder:
            lens_model.forward(input_ids)
        wrapped_logits = lens_model.unembed(recorder.activations[final_layer]).float()
        raw_logits = raw_outputs.logits.float()
    difference = (raw_logits - wrapped_logits).abs()
    top_k = min(10, raw_logits.shape[-1])
    top1_same = bool(torch.equal(raw_logits.argmax(-1), wrapped_logits.argmax(-1)))
    top10_same = bool(torch.equal(torch.topk(raw_logits, top_k, dim=-1).indices, torch.topk(wrapped_logits, top_k, dim=-1).indices))
    details = {"max_abs_logit_error": float(difference.max().item()), "mean_abs_logit_error": float(difference.mean().item()), "top1_same": top1_same, "top10_same": top10_same, "raw_shape": list(raw_logits.shape), "wrapped_shape": list(wrapped_logits.shape)}
    passed = details["raw_shape"] == details["wrapped_shape"] and details["max_abs_logit_error"] <= ADAPTER_MAX_LOGIT_ERROR and top1_same and top10_same
    return ({"check_name": "adapter_transparency", "result": "raw and adapter logits agree" if passed else "logits differ", "status": "PASS" if passed else "FAIL"}, details)


def target_objective(target: torch.Tensor, position: int, direction: torch.Tensor) -> torch.Tensor:
    end = max(position + 1, target.shape[1] - 1)
    return (target[0, position:end, :].float() @ direction.float()).sum()


def autograd_derivative(container: Any, lens_model: Any, input_ids: torch.Tensor, source_layer: int, target_layer: int, position: int, target_direction: torch.Tensor, perturb_direction: torch.Tensor) -> tuple[float, float]:
    captured: dict[str, torch.Tensor] = {}

    def source_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        leaf = output_hidden(output).detach().clone().requires_grad_(True)
        captured["source"] = leaf
        return replace_output_hidden(output, leaf)

    def target_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        captured["target"] = output_hidden(output)
        return output

    handles = [lens_model.layers[source_layer].register_forward_hook(source_hook), lens_model.layers[target_layer].register_forward_hook(target_hook)]
    try:
        with torch.enable_grad():
            container.thinker(input_ids=input_ids, use_cache=False, return_dict=True)
            objective = target_objective(captured["target"], position, target_direction)
            gradient = torch.autograd.grad(objective, captured["source"])[0]
        # Scale the finite-difference perturbation using the source activation
        # magnitude, not the
        # the derivative magnitude. The latter can be tiny and makes BF16
        # finite differences dominated by quantization.
        source_norm = float(captured["source"][0, position].float().norm().item())
        predicted = float((gradient[0, position].float() @ perturb_direction.float()).item())
        return predicted, source_norm
    finally:
        for handle in handles:
            handle.remove()


def finite_difference(container: Any, lens_model: Any, input_ids: torch.Tensor, source_layer: int, target_layer: int, position: int, target_direction: torch.Tensor, perturb_direction: torch.Tensor, scale: float) -> float:
    captured: dict[str, torch.Tensor] = {}

    def source_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output_hidden(output).clone()
        hidden[0, position, :] += scale * perturb_direction.to(hidden)
        return replace_output_hidden(output, hidden)

    def target_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        captured["target"] = output_hidden(output)
        return output

    handles = [lens_model.layers[source_layer].register_forward_hook(source_hook), lens_model.layers[target_layer].register_forward_hook(target_hook)]
    try:
        with torch.no_grad():
            container.thinker(input_ids=input_ids, use_cache=False, return_dict=True)
        return float(target_objective(captured["target"], position, target_direction).item())
    finally:
        for handle in handles:
            handle.remove()


def check_gradients(container: Any, lens_model: Any, prompt: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_ids = lens_model.encode(prompt, max_length=40)
    position = min(SKIP_FIRST_POSITIONS, int(input_ids.shape[1]) - 2)
    source_layer = SOURCE_LAYERS[len(SOURCE_LAYERS) // 2]
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(RANDOM_SEED + 1001)
    details: list[dict[str, Any]] = []
    for direction_index in range(3):
        target_direction = torch.randn(lens_model.d_model, device=input_ids.device, dtype=torch.float32, generator=generator)
        perturb_direction = torch.randn(lens_model.d_model, device=input_ids.device, dtype=torch.float32, generator=generator)
        target_direction /= target_direction.norm().clamp_min(1e-12)
        perturb_direction /= perturb_direction.norm().clamp_min(1e-12)
        predicted, source_norm = autograd_derivative(container, lens_model, input_ids, source_layer, TARGET_LAYER, position, target_direction, perturb_direction)
        scale = max(1e-3, GRADIENT_FD_SCALE_FRACTION * max(source_norm, 1.0))
        plus = finite_difference(container, lens_model, input_ids, source_layer, TARGET_LAYER, position, target_direction, perturb_direction, scale)
        minus = finite_difference(container, lens_model, input_ids, source_layer, TARGET_LAYER, position, target_direction, perturb_direction, -scale)
        numerical = (plus - minus) / (2.0 * scale)
        absolute_error = abs(predicted - numerical)
        denominator = max(abs(predicted), abs(numerical), 1e-6)
        relative_error = abs(predicted - numerical) / denominator
        sign_match = abs(predicted) < 1e-6 or abs(numerical) < 1e-6 or math.copysign(1.0, predicted) == math.copysign(1.0, numerical)
        near_zero = abs(predicted) <= GRADIENT_NEAR_ZERO_DERIVATIVE
        passed = (absolute_error <= GRADIENT_NEAR_ZERO_ABS_ERROR) if near_zero else (relative_error <= GRADIENT_REL_ERROR_TOL and sign_match)
        details.append({"direction": direction_index, "source_layer": source_layer, "target_layer": TARGET_LAYER, "position": position, "source_norm": source_norm, "scale": scale, "autograd": predicted, "finite_difference": numerical, "absolute_error": absolute_error, "relative_error": relative_error, "near_zero": near_zero, "sign_match": sign_match, "status": "PASS" if passed else "FAIL"})
    passed = all(row["status"] == "PASS" for row in details)
    return ({"check_name": "gradient_finite_difference", "result": f"{sum(row['status'] == 'PASS' for row in details)}/{len(details)} directions agree", "status": "PASS" if passed else "FAIL"}, details)


def check_lens_numerics(lens: Any, expected_layers: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    passed = True
    for layer in expected_layers:
        matrix = lens.jacobians.get(layer)
        if matrix is None:
            details.append({"layer": layer, "status": "FAIL", "reason": "missing"})
            passed = False
            continue
        finite = bool(torch.isfinite(matrix).all().item())
        nonzero = bool(torch.count_nonzero(matrix).item() > 0)
        valid = finite and nonzero and tuple(matrix.shape) == (lens.d_model, lens.d_model)
        passed = passed and valid
        details.append({"layer": layer, "shape": list(matrix.shape), "mean_abs": float(matrix.float().abs().mean().item()), "frobenius_norm": float(matrix.float().norm().item()), "finite": finite, "nonzero": nonzero, "status": "PASS" if valid else "FAIL"})
    return ({"check_name": "lens_numerics", "result": f"{sum(row['status'] == 'PASS' for row in details)}/{len(details)} matrices valid", "status": "PASS" if passed else "FAIL"}, details)


def validate_on_text(lens_model: Any, lens: Any, records: list[PromptRecord], output_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failed = 0
    for index, record in enumerate(records, start=1):
        try:
            lens_logits, model_logits, input_ids = lens.apply(lens_model, record.prompt, layers=SOURCE_LAYERS, positions=[-2], max_seq_len=VALIDATION_MAX_SEQ_LEN)
            if not torch.isfinite(model_logits).all():
                raise ValueError("model logits contain non-finite values")
            for layer in SOURCE_LAYERS:
                logits = lens_logits[layer]
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Lens logits contain non-finite values at L{layer}")
                top_id = int(logits[0].argmax().item())
                rows.append({"index": index, "prompt_id": record.prompt_id, "language": record.language, "layer": layer, "sequence_length": int(input_ids.shape[1]), "top1_token_id": top_id, "top1_token": lens_model.tokenizer.decode([top_id]), "finite": True, "status": "PASS"})
        except Exception as exc:
            failed += 1
            rows.append({"index": index, "prompt_id": record.prompt_id, "language": record.language, "layer": "all", "finite": False, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    write_csv(output_path, rows)
    languages = Counter(record.language for record in records)
    passed = failed == 0 and len(records) == 2 * VALIDATION_PER_LANGUAGE
    return ({"check_name": "independent_text_validation", "result": f"{len(records) - failed}/{len(records)} prompts completed; zh={languages['zh']}, en={languages['en']}", "status": "PASS" if passed else "FAIL"}, rows)


def fit_signature(records: list[PromptRecord], group: str) -> dict[str, Any]:
    return {"group": group, "prompt_sha256": sha256_text("\n".join(record.prompt_sha256 for record in records)), "prompt_count": len(records), "source_layers": SOURCE_LAYERS, "target_layer": TARGET_LAYER, "sequence_length": SEQUENCE_LENGTH, "skip_first_positions": SKIP_FIRST_POSITIONS, "dim_batch": DIM_BATCH, "model_path": MODEL_PATH}


def fit_half(lens_model: Any, records: list[PromptRecord], group: str, internal: Path) -> Any:
    lens_path = internal / f"jlens_half_{group}.pt"
    checkpoint_path = internal / f"fit_checkpoint_{group}.pt"
    signature_path = internal / f"fit_signature_{group}.json"
    signature = fit_signature(records, group)
    if signature_path.exists():
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(f"Fit signature changed for group {group}; refuse to resume")
    elif lens_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"Existing Lens/checkpoint for group {group} has no signature; move old files before starting")
    write_json(signature_path, signature)
    if lens_path.exists():
        loaded = jlens.JacobianLens.load(str(lens_path))
        if loaded.n_prompts != len(records) or loaded.source_layers != SOURCE_LAYERS:
            raise RuntimeError(f"Existing Lens {lens_path} does not match current fit")
        return loaded
    started = time.perf_counter()
    lens = jlens.fit(lens_model, [record.prompt for record in records], source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER, dim_batch=DIM_BATCH, max_seq_len=SEQUENCE_LENGTH, skip_first=SKIP_FIRST_POSITIONS, checkpoint_path=str(checkpoint_path), checkpoint_every=CHECKPOINT_EVERY, resume=RESUME)
    lens.save(str(lens_path), dtype=torch.float32)
    write_json(internal / f"fit_runtime_{group}.json", {"group": group, "n_prompts": lens.n_prompts, "elapsed_seconds": time.perf_counter() - started, "lens_path": str(lens_path)})
    return lens


def main() -> None:
    set_seed(RANDOM_SEED)
    output = Path(OUTPUT_DIR)
    internal = output / "internal"
    output.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 4A fit")
    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    documents = load_documents(CALIBRATION_INPUT_PATHS)
    records, rejected = build_records(processor.tokenizer, documents)
    half_a, half_b, validation = split_fit_and_validation(records)
    write_csv(internal / "fit_manifest.csv", records_to_rows(half_a, "A") + records_to_rows(half_b, "B"))
    write_csv(internal / "validation_manifest.csv", records_to_rows(validation, "validation"))
    write_csv(internal / "rejected_calibration_items.csv", rejected)
    write_json(internal / "data_summary.json", {"document_count": len(documents), "valid_prompt_count": len(records), "rejected_count": len(rejected), "fit_A_count": len(half_a), "fit_B_count": len(half_b), "validation_count": len(validation), "fit_A_languages": dict(Counter(record.language for record in half_a)), "fit_B_languages": dict(Counter(record.language for record in half_b)), "validation_languages": dict(Counter(record.language for record in validation))})

    technical: list[dict[str, Any]] = [{"check_name": "data_split", "result": f"A={len(half_a)}, B={len(half_b)}, validation={len(validation)}", "status": "PASS"}]
    container, lens_model = load_model_and_lens(processor)
    layer_summary, layer_details = check_layer_alignment(container, lens_model, half_a[0].prompt)
    technical.append(layer_summary)
    write_csv(internal / "layer_alignment_details.csv", layer_details)
    adapter_summary, adapter_details = check_adapter_transparency(container, lens_model, half_a[0].prompt)
    technical.append(adapter_summary)
    write_json(internal / "adapter_transparency.json", adapter_details)
    gradient_summary, gradient_details = check_gradients(container, lens_model, half_a[0].prompt)
    technical.append(gradient_summary)
    write_csv(internal / "gradient_check_details.csv", gradient_details)
    if any(row["status"] != "PASS" for row in technical):
        for row in technical:
            logging.error("4A pre-fit check: %s", row)
        write_csv(output / "4A_technical_check.csv", technical)
        raise RuntimeError("Pre-fit technical checks failed; fit was not started")

    logging.info("Fitting Lens A (%d prompts)", len(half_a))
    lens_a = fit_half(lens_model, half_a, "A", internal)
    logging.info("Fitting Lens B (%d prompts)", len(half_b))
    lens_b = fit_half(lens_model, half_b, "B", internal)
    numerical_a, details_a = check_lens_numerics(lens_a, SOURCE_LAYERS)
    numerical_b, details_b = check_lens_numerics(lens_b, SOURCE_LAYERS)
    technical.extend([numerical_a, numerical_b])
    write_csv(internal / "lens_A_layer_statistics.csv", details_a)
    write_csv(internal / "lens_B_layer_statistics.csv", details_b)
    if numerical_a["status"] != "PASS" or numerical_b["status"] != "PASS":
        write_csv(output / "4A_technical_check.csv", technical)
        raise RuntimeError("Lens A/B numerical checks failed")

    main_lens = jlens.JacobianLens.merge([lens_a, lens_b])
    main_path = output / "jlens_main.pt"
    main_lens.save(str(main_path), dtype=torch.float32)
    loaded_main = jlens.JacobianLens.load(str(main_path))
    matrix_reload = all(torch.equal(main_lens.jacobians[layer], loaded_main.jacobians[layer]) for layer in SOURCE_LAYERS)
    technical.append({"check_name": "lens_reload", "result": f"matrix_equal={matrix_reload}", "status": "PASS" if matrix_reload else "FAIL"})
    validation_summary, _ = validate_on_text(lens_model, loaded_main, validation, internal / "validation_readout.csv")
    technical.append(validation_summary)
    overall_pass = all(row["status"] == "PASS" for row in technical) and matrix_reload
    technical.append({"check_name": "technical_overall", "result": "all checks passed" if overall_pass else "one or more checks failed", "status": "PASS" if overall_pass else "FAIL"})
    write_csv(output / "4A_technical_check.csv", technical)
    write_json(internal / "frozen_fit_config.json", {"model_path": MODEL_PATH, "calibration_input_paths": CALIBRATION_INPUT_PATHS, "calibration_sha256": {path: sha256_file(Path(path)) for path in CALIBRATION_INPUT_PATHS}, "prompts_per_half": PROMPTS_PER_HALF, "validation_per_language": VALIDATION_PER_LANGUAGE, "language_targets_per_half": LANGUAGE_TARGETS_PER_HALF, "source_layers": SOURCE_LAYERS, "target_layer": TARGET_LAYER, "sequence_length": SEQUENCE_LENGTH, "skip_first_positions": SKIP_FIRST_POSITIONS, "dim_batch": DIM_BATCH, "random_seed": RANDOM_SEED, "fit_dtype": str(FIT_DTYPE), "main_lens_path": str(main_path), "main_lens_n_prompts": main_lens.n_prompts, "main_lens_d_model": main_lens.d_model, "torch": torch.__version__, "transformers": __import__("transformers").__version__, "platform": platform.platform(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if not overall_pass:
        raise RuntimeError("4A completed with failed technical checks")
    print("\n4A Qwen2.5-Omni J-lens fit complete: all checks PASS")
    print("Main Lens:", main_path)
    print("Technical report:", output / "4A_technical_check.csv")
    print("Internal records:", internal)
    del container, lens_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"4A failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
