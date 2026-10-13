"""Exercise experiment CLI and manifest selection without importing GPU libraries.

The actual parser, initial count resolution, and selection functions are
compiled from each script's AST. Model loading and inference are not executed.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EXPERIMENTS = (
    "intervene_bark_concepts",
    "evaluate_single_token_behavior",
    "evaluate_hierarchical_behavior",
    "evaluate_prompt_invariance",
    "localize_causal_effects",
    "validate_audio_jacobian_lens",
)
LABELS = ("dog_bark", "cat_vocalization", "non_dog_bark")
PATHS = SimpleNamespace(
    MANIFEST=Path("outputs/manifests/audio_manifest.csv"),
    OUTPUTS_ROOT=Path("outputs"),
    QWEN_MODEL=Path("models/Qwen2.5-Omni-7B"),
    LENS=Path("outputs/fit_jacobian_lens/jlens_main.pt"),
)


def load_interface(stem: str) -> dict[str, Any]:
    tree = ast.parse((SCRIPTS / f"{stem}.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    base = SimpleNamespace(
        DEFAULT_MODEL=PATHS.QWEN_MODEL,
        DEFAULT_LENS=PATHS.LENS,
        SUPPORTED_LABELS=LABELS,
        SOURCE_LAYERS=[0, 4, 8, 12, 16, 20, 22, 24],
    )
    namespace = {
        "__doc__": "Experiment interface test",
        "argparse": argparse,
        "Path": Path,
        "os": os,
        "random": random,
        "defaultdict": defaultdict,
        "json": json,
        "Any": Any,
        "paths": PATHS,
        "base": base,
    }
    constant_names = {
        "DEFAULT_MODEL", "DEFAULT_LENS", "DEFAULT_MANIFEST", "DEFAULT_OUTPUT",
        "PRIMARY_LAYER", "DEFAULT_LENS_TARGET_LAYER", "DEFAULT_LENS_TARGET_POLICY",
        "PHASE_PER_LABEL", "DEFAULT_PROMPTS", "SUPPORTED_LABELS",
        "PENULTIMATE_TARGET_LAYER", "FINAL_TARGET_LAYER", "EXPECTED_LENS_PROMPTS",
        "EXPECTED_D_MODEL", "MAX_ALLOWED_DELTA_RATIO",
        "LENS_TARGET_LAYER", "CANONICAL_FINAL_LAYER", "SEED",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constant_names for target in node.targets)
    ]
    nodes.append(functions["parse_args"])
    startup = []
    for node in functions["main"].body:
        startup.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) in {
            "args.per_label < 1", "per_label is not None and per_label < 1"
        }:
            break
    else:
        raise AssertionError(f"Cannot locate sample-count validation in {stem}")
    count_name = "per_label" if stem == "intervene_bark_concepts" else "args.per_label"
    startup.append(ast.parse(f"return args, {count_name}").body[0])
    resolve = ast.FunctionDef(
        name="resolve_interface",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=startup,
        decorator_list=[],
    )
    nodes.append(resolve)
    for name in ("select_manifest", "balanced_limit"):
        if name in functions:
            nodes.append(functions[name])
    if stem in {"evaluate_single_token_behavior", "evaluate_hierarchical_behavior"}:
        nodes.append(functions["technical_checks"])
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    with patch.dict(os.environ, {}, clear=True):
        exec(compile(module, str(SCRIPTS / f"{stem}.py"), "exec"), namespace)
    return namespace


def resolve(stem: str, *arguments: str) -> tuple[argparse.Namespace, int | None]:
    interface = load_interface(stem)
    with patch.object(sys, "argv", [f"{stem}.py", *arguments]):
        return interface["resolve_interface"]()


def rows_per_label(count: int) -> list[dict[str, str]]:
    return [
        {"sample_id": f"{label}-{index}", "label": label}
        for label in LABELS for index in range(count)
    ]


def select_samples(stem: str, rows: list[dict[str, str]], phase: str, count: int | None) -> list[dict[str, str]]:
    if stem == "intervene_bark_concepts":
        return load_interface(stem)["balanced_limit"](rows, count)
    owner = stem if stem == "evaluate_single_token_behavior" else "evaluate_hierarchical_behavior"
    return load_interface(owner)["select_manifest"](rows, phase, count, 17)


class ExperimentInterfaceTests(unittest.TestCase):
    def test_default_smoke_accepts_one_sample_per_class(self) -> None:
        small_manifest = rows_per_label(1)
        for stem in EXPERIMENTS:
            with self.subTest(script=stem):
                args, count = resolve(stem)
                self.assertEqual(args.phase, "smoke")
                self.assertEqual(count, 1)
                selected = select_samples(stem, small_manifest, args.phase, count)
                self.assertCountEqual(selected, small_manifest)

    def test_phase_defaults(self) -> None:
        cases = (
            ("intervene_bark_concepts", "dose", 20),
            ("intervene_bark_concepts", "layer", None),
            ("evaluate_single_token_behavior", "d0", 35),
            ("evaluate_single_token_behavior", "primary", 35),
            ("evaluate_single_token_behavior", "full", 35),
            ("evaluate_hierarchical_behavior", "annotate", 35),
            ("evaluate_hierarchical_behavior", "d0", 35),
            ("evaluate_hierarchical_behavior", "primary", 35),
            ("evaluate_hierarchical_behavior", "full", 35),
            ("evaluate_prompt_invariance", "primary", 35),
            ("localize_causal_effects", "screen", 10),
            ("localize_causal_effects", "primary", 35),
            ("localize_causal_effects", "full", 35),
            ("validate_audio_jacobian_lens", "primary", 3),
        )
        for stem, phase, expected in cases:
            with self.subTest(script=stem, phase=phase):
                self.assertEqual(resolve(stem, "--phase", phase)[1], expected)

    def test_explicit_smoke_count_selects_requested_samples(self) -> None:
        manifest = rows_per_label(4)
        for stem in EXPERIMENTS:
            with self.subTest(script=stem):
                args, count = resolve(stem, "--phase", "smoke", "--per-label", "2")
                self.assertEqual(count, 2)
                selected = select_samples(stem, manifest, args.phase, count)
                self.assertEqual(len(selected), 6)
                for label in LABELS:
                    self.assertEqual(sum(row["label"] == label for row in selected), 2)

    def test_large_count_is_configurable(self) -> None:
        for stem in EXPERIMENTS:
            with self.subTest(script=stem):
                phase = "layer" if stem == "intervene_bark_concepts" else "primary"
                args, count = resolve(stem, "--phase", phase, "--per-label", "200")
                self.assertEqual(count, 200)
                self.assertEqual(len(select_samples(stem, rows_per_label(200), args.phase, count)), 600)

    def test_invalid_count_and_insufficient_manifest_are_rejected(self) -> None:
        for stem in EXPERIMENTS:
            for count in ("0", "-1"):
                with self.subTest(script=stem, count=count):
                    with self.assertRaisesRegex(ValueError, "--per-label must be positive"):
                        resolve(stem, "--per-label", count)
        select = load_interface("evaluate_hierarchical_behavior")["select_manifest"]
        with self.assertRaisesRegex(ValueError, "Need 2 dog_bark samples, found 1"):
            select(rows_per_label(1), "smoke", 2, 17)

    def test_manifest_and_output_cli_overrides(self) -> None:
        for stem in EXPERIMENTS:
            with self.subTest(script=stem):
                args, _ = resolve(stem)
                self.assertEqual(args.manifest, PATHS.MANIFEST)
                self.assertEqual(args.output_dir, PATHS.OUTPUTS_ROOT / stem)
                args, _ = resolve(stem, "--manifest", "custom/manifest.csv", "--output-dir", "custom/results")
                self.assertEqual(args.manifest, Path("custom/manifest.csv"))
                self.assertEqual(args.output_dir, Path("custom/results"))

    def test_smoke_technical_check_accepts_custom_count(self) -> None:
        for stem in ("evaluate_single_token_behavior", "evaluate_hierarchical_behavior"):
            with self.subTest(script=stem):
                interface = load_interface(stem)
                extra = {} if stem == "evaluate_single_token_behavior" else {
                    "lens_target_policy": "penultimate", "lens_target_layer": 26,
                    "detected_lens_target_layer": 26, "lens_metadata_path": "frozen_fit_config.json",
                }
                checks = interface["technical_checks"](
                    rows=[], selected=rows_per_label(2),
                    lens=SimpleNamespace(n_prompts=1000, d_model=3584), token_rows=[],
                    phase="smoke", per_label=2,
                    zero_hook_check={"max_abs_first_step_logit_error": 0, "status": "PASS"},
                    **extra,
                )
                manifest_check = next(row for row in checks if row["check_name"] == "manifest_labels")
                self.assertEqual(manifest_check["status"], "PASS")

    def test_canonical_local_imports_exist(self) -> None:
        expected = {
            "evaluate_prompt_invariance": {"evaluate_hierarchical_behavior"},
            "localize_causal_effects": {"evaluate_hierarchical_behavior"},
            "validate_audio_jacobian_lens": {"evaluate_hierarchical_behavior", "evaluate_prompt_invariance"},
        }
        for stem, dependencies in expected.items():
            with self.subTest(script=stem):
                tree = ast.parse((SCRIPTS / f"{stem}.py").read_text(encoding="utf-8"))
                imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
                self.assertTrue(dependencies <= imports)
                self.assertFalse(any(name.startswith("qwen25_omni_") for name in imports))
                for name in dependencies:
                    self.assertTrue((SCRIPTS / f"{name}.py").is_file())


if __name__ == "__main__":
    unittest.main()
