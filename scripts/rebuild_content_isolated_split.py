#!/usr/bin/env python3
"""Build train/validation splits with exact-content isolation.

The source split files are never modified. Event files with the same SHA-256
digest are treated as one group and assigned to a single output split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_split(path: Path) -> list[str]:
    with path.open("rb") as handle:
        return list(pickle.load(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pickle_atomic(path: Path, value: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def content_groups(event_dir: Path, filenames: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for filename in filenames:
        groups[file_sha256(event_dir / filename)].append(filename)
    return dict(groups)


def overlap_count(
    groups: dict[str, list[str]], train: set[str], valid: set[str]
) -> int:
    return sum(
        1
        for names in groups.values()
        if any(name in train for name in names)
        and any(name in valid for name in names)
    )


def rebuild_split(
    event_dir: Path, train: list[str], valid: list[str]
) -> tuple[list[str], list[str], dict]:
    if len(train) != len(set(train)):
        raise ValueError("train split contains duplicate filenames")
    if len(valid) != len(set(valid)):
        raise ValueError("valid split contains duplicate filenames")

    train_set = set(train)
    valid_set = set(valid)
    filename_overlap = train_set & valid_set
    if filename_overlap:
        raise ValueError(
            f"train and valid share {len(filename_overlap)} filenames"
        )

    selected = train + valid
    missing = sorted(name for name in selected if not (event_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split files are missing; first: {missing[0]}"
        )

    groups = content_groups(event_dir, selected)
    assignment = {name: "train" for name in train}
    assignment.update({name: "valid" for name in valid})
    repaired_groups = []

    for digest, names in sorted(groups.items()):
        train_names = [name for name in names if name in train_set]
        valid_names = [name for name in names if name in valid_set]
        if not train_names or not valid_names:
            continue

        destination = "train" if len(train_names) >= len(valid_names) else "valid"
        for name in names:
            assignment[name] = destination
        repaired_groups.append(
            {
                "sha256": digest,
                "destination": destination,
                "train_before": train_names,
                "valid_before": valid_names,
            }
        )

    new_train = [name for name in train if assignment[name] == "train"]
    new_train.extend(name for name in valid if assignment[name] == "train")
    new_valid = [name for name in valid if assignment[name] == "valid"]
    new_valid.extend(name for name in train if assignment[name] == "valid")

    new_train_set = set(new_train)
    new_valid_set = set(new_valid)
    if new_train_set & new_valid_set:
        raise AssertionError("rebuilt splits share filenames")
    if new_train_set | new_valid_set != train_set | valid_set:
        raise AssertionError("rebuilt splits changed the selected file set")

    report = {
        "policy": "content-group majority; ties go to train",
        "content_groups": len(groups),
        "cross_split_content_groups_before": overlap_count(
            groups, train_set, valid_set
        ),
        "cross_split_content_groups_after": overlap_count(
            groups, new_train_set, new_valid_set
        ),
        "train_before": len(train),
        "valid_before": len(valid),
        "train_after": len(new_train),
        "valid_after": len(new_valid),
        "moved_to_train": [name for name in valid if name in new_train_set],
        "moved_to_valid": [name for name in train if name in new_valid_set],
        "repaired_groups": repaired_groups,
    }
    return new_train, new_valid, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-split-dir",
        default="data_splits",
        help="split directory inside the dataset root",
    )
    parser.add_argument(
        "--output-split-dir",
        default="data_splits_content_isolated",
        help="new split directory inside the dataset root",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    event_dir = dataset_root / "events"
    source_dir = dataset_root / args.source_split_dir
    output_dir = dataset_root / args.output_split_dir
    train_path = source_dir / "train.pkl"
    valid_path = source_dir / "valid.pkl"

    if not event_dir.is_dir():
        raise FileNotFoundError(event_dir)
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(source_dir)
    output_files = (output_dir / "train.pkl", output_dir / "valid.pkl")
    if any(path.exists() for path in output_files) and not args.force:
        raise FileExistsError(
            f"output split already exists at {output_dir}; use --force to replace it"
        )

    train = load_split(train_path)
    valid = load_split(valid_path)
    new_train, new_valid, report = rebuild_split(event_dir, train, valid)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_pickle_atomic(output_dir / "train.pkl", new_train)
    write_pickle_atomic(output_dir / "valid.pkl", new_valid)
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "source_split_dir": args.source_split_dir,
            "output_split_dir": args.output_split_dir,
            "source_train_sha256": file_sha256(train_path),
            "source_valid_sha256": file_sha256(valid_path),
            "output_train_sha256": file_sha256(output_dir / "train.pkl"),
            "output_valid_sha256": file_sha256(output_dir / "valid.pkl"),
        }
    )
    (output_dir / "content_isolation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(
        "CONTENT_ISOLATED_SPLIT_READY "
        f"train={len(new_train)} valid={len(new_valid)} "
        f"overlap_before={report['cross_split_content_groups_before']} "
        f"overlap_after={report['cross_split_content_groups_after']}"
    )
    print(f"Evidence: {output_dir / 'content_isolation_report.json'}")


if __name__ == "__main__":
    main()
