#!/usr/bin/env python3
"""Prepare independent Stage 2 phrase-plan data using train-only thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_phrase_planning import (  # noqa: E402
    PHRASE_PLAN_TOKENS,
    append_phrase_vocabulary,
    fit_plan_thresholds,
    piece_bar_features,
    transform_piece,
)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path):
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(source_root, output_root, limit=None, force=False):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root:
        raise ValueError("output root must differ from source root")
    source_events = source_root / "events"
    source_splits = source_root / "data_splits"
    source_dictionary = source_root / "dictionary.pkl"
    for path in (source_events, source_splits, source_dictionary):
        if not path.exists():
            raise FileNotFoundError(path)

    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError("{} is not empty; pass --force".format(output_root))
        shutil.rmtree(output_root)
    output_events = output_root / "events"
    output_splits = output_root / "data_splits"
    output_events.mkdir(parents=True, exist_ok=True)
    output_splits.mkdir(parents=True, exist_ok=True)

    old_event2idx, old_idx2event = load_pickle(source_dictionary)[:2]
    new_event2idx, new_idx2event = append_phrase_vocabulary(old_event2idx, old_idx2event)
    dump_pickle((new_event2idx, new_idx2event), output_root / "dictionary.pkl")

    split_names = {
        path.name: list(load_pickle(path)) for path in sorted(source_splits.glob("*.pkl"))
    }
    if "train.pkl" not in split_names or "valid.pkl" not in split_names:
        raise ValueError("train.pkl and valid.pkl are required")
    selected_order = []
    seen = set()
    for names in split_names.values():
        for name in names:
            if name not in seen:
                selected_order.append(name)
                seen.add(name)
    if limit is not None:
        selected_order = selected_order[:limit]
        selected = set(selected_order)
        split_names = {
            split: [name for name in names if name in selected]
            for split, names in split_names.items()
        }
        if not split_names["train.pkl"]:
            raise ValueError("limited conversion selected no training files")

    train_features = []
    for name in split_names["train.pkl"]:
        train_features.extend(piece_bar_features(load_pickle(source_events / name)))
    thresholds = fit_plan_thresholds(train_features)

    token_counts = Counter()
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    converted_count = 0
    for name in selected_order:
        source_path = source_events / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_piece = load_pickle(source_path)
        converted = transform_piece(source_piece, thresholds)
        for event in converted[2]:
            token = "{}_{}".format(event.get("name"), event.get("value"))
            if token in PHRASE_PLAN_TOKENS:
                token_counts[token] += 1
        output_path = output_events / name
        dump_pickle(converted, output_path)
        source_digest.update(name.encode("utf-8") + b"\0" + bytes.fromhex(sha256_file(source_path)))
        output_digest.update(name.encode("utf-8") + b"\0" + bytes.fromhex(sha256_file(output_path)))
        converted_count += 1

    split_counts = {}
    for split, names in split_names.items():
        dump_pickle(names, output_splits / split)
        split_counts[split] = len(names)

    report = {
        "schema": "moonmuse-stage2-phrase-plan-data-v1",
        "conditioning_mode": "bar_level_phrase_plan",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_dictionary_sha256": sha256_file(source_dictionary),
        "output_dictionary_sha256": sha256_file(output_root / "dictionary.pkl"),
        "legacy_vocab_size": len(old_event2idx),
        "phrase_vocab_size": len(new_event2idx),
        "legacy_pad_index": len(old_event2idx),
        "phrase_pad_index": len(new_event2idx),
        "appended_tokens": list(PHRASE_PLAN_TOKENS),
        "threshold_source": "train_split_only",
        "thresholds": thresholds,
        "converted_files": converted_count,
        "split_counts": split_counts,
        "plan_token_counts": dict(sorted(token_counts.items())),
        "source_population_sha256": source_digest.hexdigest(),
        "output_population_sha256": output_digest.hexdigest(),
        "limited_conversion": limit is not None,
    }
    with open(output_root / "phrase_plan_data_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    prepare_dataset(args.source_root, args.output_root, args.limit, args.force)


if __name__ == "__main__":
    main()

