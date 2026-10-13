"""Semantic vocabulary summarization by exact aliases (paper stage 4B-05).

The 4B-04 vocabulary file contains BPE tokens, punctuation and code fragments.
This script does not rerun Qwen.  It maps only an explicit, auditable semantic
lexicon to canonical concepts, aggregates aliases, and writes the remaining
human-readable but unmapped tokens separately for manual review.
"""

from __future__ import annotations

import project_paths as paths

import csv
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


INPUT_PATH = Path(os.environ.get(
    "QWEN_4B05_INPUT",
    str(paths.OUTPUTS_ROOT / "analyze_diverse_concept_trajectories/internal/top_vocabulary_tokens.csv"),
))
OUTPUT_DIR = Path(os.environ.get(
    "QWEN_4B05_OUTPUT",
    str(paths.OUTPUTS_ROOT / "summarize_semantic_vocabulary"),
))


# Explicit aliases are deliberately conservative.  A token is included only
# when it is an exact normalized alias; arbitrary substring matching would turn
# BPE fragments such as ``mamm`` or ``ocrat`` into false concepts.
SEMANTIC_LEXICON: dict[str, dict[str, Any]] = {
    "dog": {"group": "dog_and_vocalization", "aliases": ["dog", "dogs", "doggy", "狗", "犬"]},
    "canine": {"group": "dog_and_vocalization", "aliases": ["canine", "canines"]},
    "puppy": {"group": "dog_and_vocalization", "aliases": ["puppy", "puppies"]},
    "bark": {"group": "dog_and_vocalization", "aliases": ["bark", "barks", "barking", "barked", "woof", "吠", "吠叫", "狗叫"]},
    "howl": {"group": "dog_and_vocalization", "aliases": ["howl", "howls", "howling"]},
    "growl": {"group": "dog_and_vocalization", "aliases": ["growl", "growls", "growling"]},
    "animal": {"group": "other_animals", "aliases": ["animal", "animals", "动物", "動物", "兽", "獸"]},
    "mammal": {"group": "other_animals", "aliases": ["mammal", "mammals", "哺乳动物", "哺乳動物"]},
    "livestock": {"group": "other_animals", "aliases": ["livestock", "牲畜"]},
    "wildlife": {"group": "other_animals", "aliases": ["wildlife", "野生动物", "野生動物"]},
    "cat": {"group": "other_animals", "aliases": ["cat", "cats", "kitten", "kittens", "猫", "貓"]},
    "meow": {"group": "other_animals", "aliases": ["meow", "meows", "meowing"]},
    "bird": {"group": "other_animals", "aliases": ["bird", "birds", "鸟", "鳥"]},
    "chirp": {"group": "other_animals", "aliases": ["chirp", "chirps", "chirping"]},
    "cow": {"group": "other_animals", "aliases": ["cow", "cows", "cattle", "牛"]},
    "moo": {"group": "other_animals", "aliases": ["moo", "moos", "mooing"]},
    "horse": {"group": "other_animals", "aliases": ["horse", "horses", "马", "馬"]},
    "sheep": {"group": "other_animals", "aliases": ["sheep"]},
    "frog": {"group": "other_animals", "aliases": ["frog", "frogs", "青蛙"]},
    "sound": {"group": "acoustic_scene", "aliases": ["sound", "sounds", "声音", "聲音"]},
    "noise": {"group": "acoustic_scene", "aliases": ["noise", "noises", "noisy", "噪声", "噪聲"]},
    "audio": {"group": "acoustic_scene", "aliases": ["audio", "音频", "音訊"]},
    "silence": {"group": "acoustic_scene", "aliases": ["silence", "silent", "安静", "安靜"]},
    "music": {"group": "acoustic_scene", "aliases": ["music", "音乐", "音樂"]},
    "speech": {"group": "acoustic_scene", "aliases": ["speech", "speeches", "说话", "說話"]},
    "voice": {"group": "acoustic_scene", "aliases": ["voice", "voices", "人声", "人聲"]},
    "alarm": {"group": "acoustic_scene", "aliases": ["alarm", "alarms", "警报", "警報"]},
    "siren": {"group": "acoustic_scene", "aliases": ["siren", "sirens"]},
    "bell": {"group": "acoustic_scene", "aliases": ["bell", "bells", "铃", "鈴"]},
    "whistle": {"group": "acoustic_scene", "aliases": ["whistle", "whistles", "whistling"]},
    "beep": {"group": "acoustic_scene", "aliases": ["beep", "beeps", "beeping"]},
    "rain": {"group": "environment_and_objects", "aliases": ["rain", "raining", "雨"]},
    "wind": {"group": "environment_and_objects", "aliases": ["wind", "风", "風"]},
    "water": {"group": "environment_and_objects", "aliases": ["water", "水"]},
    "fire": {"group": "environment_and_objects", "aliases": ["fire", "火"]},
    "door": {"group": "environment_and_objects", "aliases": ["door", "doors", "门", "門"]},
    "engine": {"group": "environment_and_objects", "aliases": ["engine", "engines"]},
    "vehicle": {"group": "environment_and_objects", "aliases": ["vehicle", "vehicles"]},
    "machine": {"group": "environment_and_objects", "aliases": ["machine", "machines"]},
    "footsteps": {"group": "environment_and_objects", "aliases": ["footsteps", "footstep"]},
    "clap": {"group": "environment_and_objects", "aliases": ["clap", "claps", "clapping"]},
    "laugh": {"group": "environment_and_objects", "aliases": ["laugh", "laughs", "laughing", "laughter"]},
    "cry": {"group": "environment_and_objects", "aliases": ["cry", "cries", "crying"]},
    "human": {"group": "other_entities", "aliases": ["human", "humans", "person", "people", "人类", "人類"]},
    "pet": {"group": "other_entities", "aliases": ["pet", "pets"]},
    "nature": {"group": "other_entities", "aliases": ["nature", "natural", "自然"]},
    "category": {"group": "prompt_and_output", "aliases": ["category", "categories", "类别", "類別"]},
    "candidate": {"group": "prompt_and_output", "aliases": ["candidate", "candidates", "候选", "候選"]},
}


def normalize_token_text(value: str) -> str:
    text = value.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2581", " ").replace("\u0120", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


ALIAS_TO_CONCEPT: dict[str, str] = {}
for concept_name, metadata in SEMANTIC_LEXICON.items():
    for alias in metadata["aliases"]:
        ALIAS_TO_CONCEPT[normalize_token_text(alias).casefold()] = concept_name


def map_semantic_concept(token_text: str) -> str | None:
    normalized = normalize_token_text(token_text)
    if not normalized or len(normalized) > 40:
        return None
    return ALIAS_TO_CONCEPT.get(normalized.casefold())


def is_unmapped_readable_token(token_text: str) -> bool:
    normalized = normalize_token_text(token_text)
    if not re.fullmatch(r"[A-Za-z][A-Za-z' -]{2,}", normalized):
        return False
    if any(marker in normalized for marker in ("BackgroundImage", "Component", "ResponseBody", "Connection", "MessageBox")):
        return False
    return normalized.casefold() not in ALIAS_TO_CONCEPT


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"4B-04 vocabulary file not found: {INPUT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("The vocabulary file is empty")

    groups: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    unmapped: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    label_sample_ids: dict[str, set[str]] = defaultdict(set)
    raw_row_format = "vocab_rank" in raw_rows[0]
    for row in raw_rows:
        label = row.get("label", "")
        layer = int(row["layer"])
        stage = row["stage"]
        sample_id = row.get("sample_id", "")
        if sample_id:
            label_sample_ids[label].add(sample_id)
        token_text = normalize_token_text(row.get("token_text", ""))
        if "vocab_rank" in row:
            # Raw top_vocabulary_tokens.csv: each row is one sample/layer/stage
            # token, so one matching row contributes one appearance.
            appearances = 1
            mean_rank = float(row["vocab_rank"])
            mean_logit = float(row["mean_logit"])
        else:
            # Also accept the already aggregated 4B04 vocabulary summary.
            appearances = int(float(row.get("top_k_appearances", 0)))
            mean_rank = float(row.get("mean_rank_when_present", "nan"))
            mean_logit = float(row.get("mean_logit_when_present", "nan"))
        concept = map_semantic_concept(token_text)
        target = groups if concept else unmapped
        key = (label, layer, stage, concept or token_text)
        item = target.setdefault(key, {
            "label": label,
            "layer": layer,
            "stage": stage,
            "concept": concept or "",
            "token_text": token_text,
            "token_variants": set(),
            "token_ids": set(),
            "alias_rows": 0,
            "total_top_k_appearances": 0,
            "weighted_rank_sum": 0.0,
            "weighted_logit_sum": 0.0,
            "best_top200_rank": float("inf"),
            "sample_ids": set(),
        })
        item["token_variants"].add(token_text)
        item["token_ids"].add(row.get("token_id", ""))
        item["alias_rows"] += 1
        item["total_top_k_appearances"] += appearances
        item["weighted_rank_sum"] += appearances * mean_rank
        item["weighted_logit_sum"] += appearances * mean_logit
        if sample_id:
            item["sample_ids"].add(sample_id)
        if mean_rank == mean_rank:
            item["best_top200_rank"] = min(item["best_top200_rank"], mean_rank)

    summary_rows: list[dict[str, Any]] = []
    for item in groups.values():
        total = item["total_top_k_appearances"]
        summary_rows.append({
            "label": item["label"],
            "layer": item["layer"],
            "stage": item["stage"],
            "group": SEMANTIC_LEXICON[item["concept"]]["group"],
            "concept": item["concept"],
            "token_variants": " | ".join(sorted(item["token_variants"])),
            "distinct_token_ids": len(item["token_ids"]),
            "alias_rows": item["alias_rows"],
            "total_top_k_appearances": total,
            "sample_coverage_count": len(item["sample_ids"]) if raw_row_format else "",
            "sample_coverage_rate": (len(item["sample_ids"]) / len(label_sample_ids[item["label"]])) if raw_row_format and label_sample_ids[item["label"]] else "",
            "minimum_observed_top200_rank": item["best_top200_rank"] if item["best_top200_rank"] != float("inf") else "",
            "conditional_mean_top200_rank": item["weighted_rank_sum"] / total if total else float("nan"),
            "weighted_mean_logit": item["weighted_logit_sum"] / total if total else float("nan"),
        })
    summary_rows.sort(key=lambda row: (
        row["label"], int(row["layer"]), row["stage"],
        float(row["minimum_observed_top200_rank"] or "inf"),
    ))

    unmapped_rows: list[dict[str, Any]] = []
    for item in unmapped.values():
        if not is_unmapped_readable_token(item["token_text"]):
            continue
        total = item["total_top_k_appearances"]
        unmapped_rows.append({
            "label": item["label"],
            "layer": item["layer"],
            "stage": item["stage"],
            "token_text": item["token_text"],
            "total_top_k_appearances": total,
            "sample_coverage_count": len(item["sample_ids"]) if raw_row_format else "",
            "sample_coverage_rate": (len(item["sample_ids"]) / len(label_sample_ids[item["label"]])) if raw_row_format and label_sample_ids[item["label"]] else "",
            "minimum_observed_top200_rank": item["best_top200_rank"] if item["best_top200_rank"] != float("inf") else "",
            "conditional_mean_top200_rank": item["weighted_rank_sum"] / total if total else float("nan"),
        })
    unmapped_rows.sort(key=lambda row: (
        row["label"], int(row["layer"]), row["stage"],
        float(row["minimum_observed_top200_rank"] or "inf"),
    ))

    fields = [
        "label", "layer", "stage", "group", "concept", "token_variants",
        "distinct_token_ids", "alias_rows", "total_top_k_appearances",
        "sample_coverage_count", "sample_coverage_rate", "minimum_observed_top200_rank",
        "conditional_mean_top200_rank", "weighted_mean_logit",
    ]
    write_csv(OUTPUT_DIR / "4B05_semantic_concept_summary.csv", summary_rows, fields)
    ranking_rows: list[dict[str, Any]] = []
    ranking_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        # Prompt/output words are retained in the full summary but excluded
        # from the semantic ranking intended for interpretation.
        if row["group"] == "prompt_and_output":
            continue
        ranking_groups[(row["label"], int(row["layer"]), row["stage"])].append(row)
    for group_key, values in ranking_groups.items():
        values.sort(key=lambda row: (
            -(float(row["sample_coverage_rate"]) if row["sample_coverage_rate"] else 0.0),
            float(row["conditional_mean_top200_rank"] or "inf"),
        ))
        for position, row in enumerate(values, 1):
            ranking_rows.append({
                "label": group_key[0],
                "layer": group_key[1],
                "stage": group_key[2],
                "semantic_rank_position": position,
                "group": row["group"],
                "concept": row["concept"],
                "token_variants": row["token_variants"],
                "sample_coverage_rate": row["sample_coverage_rate"],
                "conditional_mean_top200_rank": row["conditional_mean_top200_rank"],
                "minimum_observed_top200_rank": row["minimum_observed_top200_rank"],
            })
    write_csv(OUTPUT_DIR / "4B05_semantic_stage_ranking.csv", ranking_rows, [
        "label", "layer", "stage", "semantic_rank_position", "group", "concept",
        "token_variants", "sample_coverage_rate", "conditional_mean_top200_rank",
        "minimum_observed_top200_rank",
    ])
    write_csv(OUTPUT_DIR / "4B05_unmapped_readable_tokens.csv", unmapped_rows, [
        "label", "layer", "stage", "token_text", "total_top_k_appearances",
        "sample_coverage_count", "sample_coverage_rate", "minimum_observed_top200_rank",
        "conditional_mean_top200_rank",
    ])
    write_csv(OUTPUT_DIR / "4B05_semantic_lexicon.csv", [
        {"concept": concept, "group": metadata["group"], "aliases": " | ".join(metadata["aliases"])}
        for concept, metadata in SEMANTIC_LEXICON.items()
    ], ["concept", "group", "aliases"])
    summary = {
        "input": str(INPUT_PATH),
        "raw_rows": len(raw_rows),
        "semantic_rows": len(summary_rows),
        "semantic_stage_ranking_rows": len(ranking_rows),
        "unmapped_readable_rows": len(unmapped_rows),
        "canonical_concepts": len(SEMANTIC_LEXICON),
        "input_format": "raw_top_vocabulary_tokens" if raw_row_format else "aggregated_vocabulary_summary",
        "method": "exact normalized alias matching; no substring matching",
    }
    (OUTPUT_DIR / "4B05_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
