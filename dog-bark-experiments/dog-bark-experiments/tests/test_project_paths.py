"""Test configuration precedence and portability without ML dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/project_paths.py"


class ProjectPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        (self.root / "scripts").mkdir(parents=True)
        shutil.copyfile(SOURCE, self.root / "scripts/project_paths.py")
        self.cwd = Path(self.temp.name) / "elsewhere"
        self.cwd.mkdir()

    def run_paths(self, extra_env=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith("BARK_")}
        env.update(extra_env or {})
        code = (
            "import runpy, json; "
            f"d = runpy.run_path({str(self.root / 'scripts/project_paths.py')!r}); "
            "print(json.dumps({k: str(d[k]) for k in "
            "['QWEN_MODEL', 'DOG_AUDIO', 'OUTPUTS_ROOT', 'LENS', 'MANIFEST']}))"
        )
        return subprocess.run(
            [sys.executable, "-c", code], env=env, cwd=self.cwd,
            capture_output=True, text=True, check=False,
        )

    def test_defaults_are_relative_to_repository_not_cwd(self):
        result = self.run_paths()
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)
        self.assertEqual(Path(paths["DOG_AUDIO"]), self.root / "data/dogspeak")
        self.assertEqual(Path(paths["LENS"]), self.root / "outputs/fit_jacobian_lens/jlens_main.pt")
        self.assertEqual(Path(paths["MANIFEST"]), self.root / "outputs/manifests/audio_manifest.csv")

    def test_environment_overrides_toml_and_lens_follows_output_root(self):
        (self.root / "config.local.toml").write_text(
            '[paths]\ndog_audio = "corpus/dog"\noutputs_root = "runs"\n', encoding="utf-8"
        )
        result = self.run_paths({"BARK_DOG_DIR": str(self.cwd / "dogs")})
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)
        self.assertEqual(Path(paths["DOG_AUDIO"]), self.cwd / "dogs")
        self.assertEqual(Path(paths["LENS"]), self.root / "runs/fit_jacobian_lens/jlens_main.pt")
        self.assertEqual(Path(paths["MANIFEST"]), self.root / "runs/manifests/audio_manifest.csv")

    def test_explicit_config_and_lens_override(self):
        config = self.cwd / "custom.toml"
        config.write_text('[paths]\nqwen_model = "weights/qwen"\nlens = "weights/lens.pt"\nmanifest = "lists/audio.csv"\n', encoding="utf-8")
        result = self.run_paths({"BARK_CONFIG": str(config), "BARK_MANIFEST": "lists/custom.csv"})
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)
        self.assertEqual(Path(paths["QWEN_MODEL"]), self.root / "weights/qwen")
        self.assertEqual(Path(paths["LENS"]), self.root / "weights/lens.pt")
        self.assertEqual(Path(paths["MANIFEST"]), self.root / "lists/custom.csv")

    def test_explicit_missing_config_fails(self):
        result = self.run_paths({"BARK_CONFIG": "missing.toml"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BARK_CONFIG file not found", result.stderr)

    def test_invalid_config_fails_with_diagnostic(self):
        for content, diagnostic in [
            ('[paths]\nmodel_typo = "x"\n', "Unknown path settings"),
            ('[paths]\ndog_audio = 42\n', "must be a non-empty string"),
            ('[wrong_table]\ndog_audio = "x"\n', "Only the [paths] table"),
        ]:
            with self.subTest(content=content):
                (self.root / "config.local.toml").write_text(content, encoding="utf-8")
                result = self.run_paths()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)


if __name__ == "__main__":
    unittest.main()
