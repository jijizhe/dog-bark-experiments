"""Resolve portable experiment paths from environment, TOML, or defaults."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config() -> dict[str, str]:
    explicit = os.environ.get("BARK_CONFIG")
    config = _absolute(explicit or "config.local.toml")
    if not config.is_file():
        if explicit:
            raise FileNotFoundError(f"BARK_CONFIG file not found: {config}")
        return {}
    with config.open("rb") as handle:
        content = tomllib.load(handle)
    if set(content) - {"paths"}:
        raise ValueError(f"Only the [paths] table is supported in {config}")
    values = content.get("paths", {})
    if not isinstance(values, dict):
        raise ValueError(f"[paths] must be a table in {config}")
    allowed = {
        "qwen_model", "dog_audio", "cat_audio",
        "esc50_audio", "urbansound8k_audio", "calibration_en", "calibration_zh",
        "outputs_root", "lens", "manifest",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown path settings in {config}: {sorted(unknown)}")
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path {key!r} must be a non-empty string in {config}")
    return values


_CONFIG = _load_config()


def _path(key: str, env: str, default: str | Path) -> Path:
    return _absolute(os.environ.get(env) or _CONFIG.get(key, str(default)))


QWEN_MODEL = _path("qwen_model", "BARK_MODEL_PATH", "models/Qwen2.5-Omni-7B")
DOG_AUDIO = _path("dog_audio", "BARK_DOG_DIR", "data/dogspeak")
CAT_AUDIO = _path("cat_audio", "BARK_CAT_DIR", "data/cat")
ESC50_AUDIO = _path("esc50_audio", "BARK_ESC50_DIR", "data/esc50_negative")
URBANSOUND8K_AUDIO = _path("urbansound8k_audio", "BARK_URBANSOUND8K_DIR", "data/urbansound8k_negative")
CALIBRATION_EN = _path("calibration_en", "BARK_CALIBRATION_EN", "data/calibration/fineweb_english.jsonl")
CALIBRATION_ZH = _path("calibration_zh", "BARK_CALIBRATION_ZH", "data/calibration/fineweb2_chinese_simplified.jsonl")
OUTPUTS_ROOT = _path("outputs_root", "BARK_OUTPUTS_ROOT", "outputs")
LENS = _path("lens", "BARK_LENS_PATH", OUTPUTS_ROOT / "fit_jacobian_lens/jlens_main.pt")
MANIFEST = _path("manifest", "BARK_MANIFEST", OUTPUTS_ROOT / "manifests/audio_manifest.csv")
