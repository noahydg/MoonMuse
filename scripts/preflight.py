#!/usr/bin/env python3
"""Audit MoonMuse training assets before starting a paid GPU instance."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "stage1_hooktheory": REPO_ROOT / "stage1_chord/config/hooktheory_pretrain.yaml",
    "stage1_emopia": REPO_ROOT / "stage1_chord/config/emopia_finetune.yaml",
    "stage2_pop1k7": REPO_ROOT / "stage2_melody/config/pop1k7_pretrain_new_gpt2.yaml",
    "stage2_emopia": REPO_ROOT / "stage2_melody/config/emopia_finetune_new_gpt2.yaml",
    "stage3_pop1k7": REPO_ROOT / "stage3_performance/config/pop1k7_pretrain_new_gpt2.yaml",
    "stage3_emopia": REPO_ROOT / "stage3_performance/config/emopia_finetune_new_gpt2.yaml",
}


def resolve_config_path(
    raw: str,
    events_root: Path,
    representation: str = "functional",
) -> Path:
    normalized = raw.format(representation).replace("\\", "/")
    if normalized == "events":
        return events_root
    if normalized.startswith("events/"):
        return events_root / normalized.removeprefix("events/")
    return REPO_ROOT / normalized


def inspect_config(name: str, config_path: Path, events_root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if name.startswith("stage1"):
        values = config["data"]
        paths = {
            "events": resolve_config_path(values["data_dir"], events_root),
            "train_split": resolve_config_path(values["train_split"], events_root),
            "valid_split": resolve_config_path(values["val_split"], events_root),
            "vocab": resolve_config_path(values["vocab_path"], events_root),
        }
    else:
        values = config["data_loader"]
        paths = {
            "events": resolve_config_path(values["data_path"], events_root),
            "train_split": resolve_config_path(values["train_split"], events_root),
            "valid_split": resolve_config_path(values["val_split"], events_root),
            "vocab": resolve_config_path(values["vocab_path"], events_root),
        }

    checks = {key: path.exists() for key, path in paths.items()}
    counts = {}
    if checks["events"]:
        event_files = list(paths["events"].glob("*.pkl"))
        appledouble_files = [path for path in event_files if path.name.startswith("._")]
        counts["events"] = len(event_files) - len(appledouble_files)
        counts["appledouble_events"] = len(appledouble_files)
    for split_name in ("train_split", "valid_split"):
        if checks[split_name]:
            with paths[split_name].open("rb") as handle:
                split = list(pickle.load(handle))
            counts[split_name] = len(split)
            counts[f"{split_name}_missing"] = sum(
                not (paths["events"] / filename).is_file() for filename in split
            )

    return {
        "ready": all(checks.values())
        and counts.get("train_split_missing", 1) == 0
        and counts.get("valid_split_missing", 1) == 0
        and counts.get("appledouble_events", 1) == 0,
        "paths": {key: str(path) for key, path in paths.items()},
        "checks": checks,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--events-root", type=Path, default=REPO_ROOT / "events")
    args = parser.parse_args()

    events_root = args.events_root.expanduser().resolve()
    report = {
        name: inspect_config(name, path, events_root)
        for name, path in CONFIGS.items()
    }
    report["all_training_data_ready"] = all(
        item["ready"] for item in report.values() if isinstance(item, dict)
    )

    if args.as_json:
        print(json.dumps(report, indent=2))
        return

    for name, item in report.items():
        if not isinstance(item, dict):
            continue
        state = "PASS" if item["ready"] else "BLOCKED"
        counts = ", ".join(f"{key}={value}" for key, value in item["counts"].items())
        print(f"[{state}] {name}: {counts or 'required assets missing'}")
        if not item["ready"]:
            for key, exists in item["checks"].items():
                if not exists:
                    print(f"  missing {key}: {item['paths'][key]}")

    if not report["all_training_data_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
