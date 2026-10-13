"""Exercise Qwen baseline selection and reporting without model dependencies."""

import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EXTERNAL_IMPORTS = {"torch", "transformers", "qwen_omni_utils", "project_paths"}


def load_baseline(paths: types.SimpleNamespace):
    """Run source with external imports removed; inference is stubbed by tests."""
    source = SCRIPTS / "evaluate_bark_detection_qwen.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] in EXTERNAL_IMPORTS for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] in EXTERNAL_IMPORTS
        )
    ]
    module = types.ModuleType(source.stem)
    module.paths = paths
    exec(compile(tree, str(source), "exec"), module.__dict__)
    return module, tree.body[-1]


class BarkEvaluationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = types.SimpleNamespace(
            DOG_AUDIO=self.root / "dog",
            ESC50_AUDIO=self.root / "esc",
            URBANSOUND8K_AUDIO=self.root / "urban",
            QWEN_MODEL=self.root / "qwen",
            OUTPUTS_ROOT=self.root / "outputs",
        )
        for directory in (self.paths.DOG_AUDIO, self.paths.ESC50_AUDIO, self.paths.URBANSOUND8K_AUDIO):
            directory.mkdir()
            # Selection needs paths only; the tests do not decode these files.
            for index in range(10):
                (directory / f"clip_{index:02d}.wav").touch()

    def test_defaults_and_invalid_counts(self):
        module, _ = load_baseline(self.paths)
        defaults = module.parse_args([])
        self.assertEqual((defaults.dog_count, defaults.esc_count, defaults.urban_count), (300, 150, 150))
        self.assertEqual(defaults.output_dir, self.paths.OUTPUTS_ROOT / "evaluate_bark_detection_qwen")
        for flag in ("--dog-count", "--esc-count", "--urban-count"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args([flag, value])
                    self.assertEqual(error.exception.code, 2)

    def test_count_overrides_control_selection_summary_and_paths(self):
        module, _ = load_baseline(self.paths)
        output = self.root / "qwen_small"
        arguments = [
            "--dog-dir", str(self.paths.DOG_AUDIO),
            "--esc50-dir", str(self.paths.ESC50_AUDIO),
            "--urban-dir", str(self.paths.URBANSOUND8K_AUDIO),
            "--model-path", str(self.root / "custom_model"),
            "--output-dir", str(output),
            "--dog-count", "3", "--esc-count", "2", "--urban-count", "1",
        ]
        predict = lambda *args: {**args[-1], "prediction": args[-1]["label"], "response": args[-1]["label"]}
        with mock.patch.object(module, "load_model", return_value=(None, None)):
            with mock.patch.object(module, "predict", side_effect=predict), contextlib.redirect_stdout(io.StringIO()) as stdout:
                module.main(arguments)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["total"], summary["dog_bark"], summary["non_dog_bark"]), (6, 3, 3))
        self.assertEqual((summary["TP"], summary["TN"]), (3, 3))
        self.assertIn("[006/6]", stdout.getvalue())
        self.assertEqual(module.MODEL_PATH, self.root / "custom_model")
        selected = module.build_test_set()
        for source, count in [("DogSpeak", 3), ("ESC-50", 2), ("UrbanSound8K", 1)]:
            self.assertEqual(sum(row["source"] == source for row in selected), count)
        self.assertEqual(len({row["audio_path"] for row in selected}), 6)
        self.assertEqual(module.NON_DOG_COUNT, 3)

    def test_failure_report_uses_cli_output_directory(self):
        module, entry_point = load_baseline(self.paths)
        output = self.root / "qwen_failed"
        arguments = ["qwen", "--output-dir", str(output), "--dog-count", "1", "--esc-count", "1", "--urban-count", "1"]
        module.__name__ = "__main__"
        with mock.patch.object(sys, "argv", arguments):
            with mock.patch.object(module, "load_model", side_effect=RuntimeError("test model unavailable")):
                with self.assertRaisesRegex(RuntimeError, "test model unavailable"):
                    exec(compile(ast.Module(body=[entry_point], type_ignores=[]), "qwen", "exec"), module.__dict__)
        self.assertIn("test model unavailable", (output / "failure.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
