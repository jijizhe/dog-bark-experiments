#!/usr/bin/env python3
"""Build a balanced audio manifest with a configurable number of clips per class."""

from __future__ import annotations

import project_paths as paths

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SEED = 20260829

DEFAULT_DOG_ROOT = Path(
    str(paths.DOG_AUDIO)
)
DEFAULT_CAT_ROOT = Path(
    str(paths.CAT_AUDIO)
)
DEFAULT_ESC_ROOT = Path(
    str(paths.ESC50_AUDIO)
)
DEFAULT_URBAN_ROOT = Path(
    str(paths.URBANSOUND8K_AUDIO)
)
DEFAULT_OUTPUT = paths.MANIFEST


def audio_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {root}")
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def deterministic_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    values = list(dict.fromkeys(paths))
    rng = random.Random(seed)
    rng.shuffle(values)
    if len(values) < count:
        raise ValueError(f"Need {count} audio files, found {len(values)}")
    return sorted(values[:count], key=str)


def stratified_sample(
    items: list[tuple[Path, str]], count: int, seed: int
) -> list[tuple[Path, str]]:
    """Round-robin across source categories to maximize acoustic diversity."""
    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, category in items:
        groups[category].append((path, category))
    rng = random.Random(seed)
    categories = sorted(groups)
    for category in categories:
        rng.shuffle(groups[category])
    selected: list[tuple[Path, str]] = []
    while len(selected) < count:
        progressed = False
        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) < count:
        raise ValueError(
            f"Need {count} stratified audio files, found {len(selected)} "
            f"across {len(categories)} categories"
        )
    return sorted(selected, key=lambda item: str(item[0]))


def select_dogs(root: Path, count: int) -> list[Path]:
    paths = audio_files(root)
    # Prefer one clip per dog_id, then fill remaining slots if necessary.
    by_dog: dict[str, list[Path]] = {}
    for path in paths:
        dog_id = next((part for part in path.parts if part.startswith("dog_")), path.stem)
        by_dog.setdefault(dog_id, []).append(path)
    rng = random.Random(SEED + 1)
    dog_ids = sorted(by_dog)
    rng.shuffle(dog_ids)
    selected: list[Path] = []
    remaining: list[Path] = []
    for dog_id in dog_ids:
        clips = sorted(by_dog[dog_id], key=str)
        rng.shuffle(clips)
        selected.append(clips[0])
        remaining.extend(clips[1:])
    rng.shuffle(remaining)
    selected = selected[:count]
    if len(selected) < count:
        selected.extend(remaining[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Need {count} dog clips, found {len(selected)}")
    return sorted(selected, key=str)


def select_non_dog(
    esc_root: Path, urban_root: Path, count: int
) -> list[tuple[Path, str, str]]:
    esc = [
        (path, path.parent.name.lower())
        for path in audio_files(esc_root)
        if path.parent.name.lower() not in {"cat", "dog"}
    ]
    urban = [(path, path.parent.name.lower()) for path in audio_files(urban_root)]
    # Keep the two established negative sources balanced and stratified.
    esc_selected = stratified_sample(esc, count // 2, SEED + 2)
    urban_selected = stratified_sample(urban, count - count // 2, SEED + 3)
    selected = [
        (path, "esc50", category) for path, category in esc_selected
    ] + [
        (path, "urbansound8k", category) for path, category in urban_selected
    ]
    return sorted(selected, key=lambda item: str(item[0]))


def build(args: argparse.Namespace) -> None:
    dog = deterministic_sample(select_dogs(args.dog_root, args.per_class), args.per_class, SEED + 4)
    cat = deterministic_sample(audio_files(args.cat_root), args.per_class, SEED + 5)
    other = select_non_dog(args.esc_root, args.urban_root, args.per_class)
    if len(other) != args.per_class:
        raise ValueError(f"Expected {args.per_class} non-dog clips, found {len(other)}")

    rows: list[dict[str, str]] = []
    for index, path in enumerate(dog, 1):
        rows.append({"sample_id": f"dog_bark_{index:04d}", "audio_path": str(path),
                     "label": "dog_bark", "source": "dogspeak",
                     "selection_stratum": path.parent.name})
    for index, path in enumerate(cat, 1):
        rows.append({"sample_id": f"cat_vocalization_{index:04d}", "audio_path": str(path),
                     "label": "cat_vocalization", "source": "bark_translate_cat",
                     "selection_stratum": path.parent.name})
    for index, (path, source, stratum) in enumerate(other, 1):
        subtype = "other_animal_vocalization" if any(
            word in stratum for word in ("animal", "bird", "cow", "frog", "sheep", "pig")
        ) else "non_vocalization"
        rows.append({"sample_id": f"non_dog_bark_{index:04d}", "audio_path": str(path),
                     "label": "non_dog_bark", "source": source,
                     "selection_stratum": stratum, "semantic_subtype": subtype})

    paths = [row["audio_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate audio paths detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "audio_path", "label", "source", "selection_stratum",
            "semantic_subtype"
        ])
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Wrote {len(rows)} rows: dog={len(dog)}, cat={len(cat)}, other={len(other)}")
    print("Non-dog source counts:", dict(Counter(row["source"] for row in rows if row["label"] == "non_dog_bark")))
    print("Non-dog strata:", dict(Counter(row["selection_stratum"] for row in rows if row["label"] == "non_dog_bark")))
    print(f"Manifest: {args.output}")
    print(f"SHA256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog-root", type=Path, default=DEFAULT_DOG_ROOT)
    parser.add_argument("--cat-root", type=Path, default=DEFAULT_CAT_ROOT)
    parser.add_argument("--esc-root", type=Path, default=DEFAULT_ESC_ROOT)
    parser.add_argument("--urban-root", type=Path, default=DEFAULT_URBAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=200)
    args = parser.parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be positive")
    build(args)


if __name__ == "__main__":
    main()
