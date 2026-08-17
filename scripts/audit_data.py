#!/usr/bin/env python3
"""Audit MoonMuse event splits for label balance and piece-level leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(path: Path) -> list[str]:
    with path.open("rb") as handle:
        return list(pickle.load(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_name(dataset: Path) -> str:
    name = dataset.name.lower()
    for source in ("emopia", "pop1k7", "hooktheory"):
        if source in name:
            return source
    return "unknown"


def piece_id(filename: str, source: str) -> str:
    stem = Path(filename).stem
    if source == "emopia":
        without_label = re.sub(r"^Q[1-4]_", "", stem)
        piece, separator, segment = without_label.rpartition("_")
        return piece if separator and segment.isdigit() else without_label
    return stem


def emotion_label(filename: str) -> str:
    match = re.match(r"^(Q[1-4])_", Path(filename).stem)
    return match.group(1) if match else "unlabeled"


def repeated_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def audit_dataset(dataset: Path, hash_files: bool, split_dir_name: str) -> dict:
    event_dir = dataset / "events"
    split_dir = dataset / split_dir_name
    train = load_split(split_dir / "train.pkl")
    valid = load_split(split_dir / "valid.pkl")
    source = source_name(dataset)
    available = {path.name for path in event_dir.glob("*.pkl")}
    train_set = set(train)
    valid_set = set(valid)
    train_pieces = {piece_id(name, source) for name in train_set}
    valid_pieces = {piece_id(name, source) for name in valid_set}

    report = {
        "source": source,
        "split_dir": split_dir_name,
        "event_files": len(available),
        "train_files": len(train),
        "valid_files": len(valid),
        "train_duplicate_entries": repeated_count(train),
        "valid_duplicate_entries": repeated_count(valid),
        "missing_train_files": sorted(train_set - available),
        "missing_valid_files": sorted(valid_set - available),
        "unassigned_event_files": len(available - train_set - valid_set),
        "filename_overlap_count": len(train_set & valid_set),
        "piece_overlap_count": len(train_pieces & valid_pieces),
        "piece_overlap_examples": sorted(train_pieces & valid_pieces)[:20],
        "train_labels": dict(sorted(Counter(emotion_label(name) for name in train).items())),
        "valid_labels": dict(sorted(Counter(emotion_label(name) for name in valid).items())),
    }

    if hash_files:
        hashes: dict[str, dict[str, list[str]]] = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
        }
        for split_name, names in (("train", train_set), ("valid", valid_set)):
            for name in sorted(names & available):
                hashes[split_name][file_sha256(event_dir / name)].append(name)
        overlap = set(hashes["train"]) & set(hashes["valid"])
        report["content_hash_overlap_count"] = len(overlap)
        report["content_hash_overlap_examples"] = [
            {
                "train": hashes["train"][digest][:3],
                "valid": hashes["valid"][digest][:3],
            }
            for digest in sorted(overlap)[:20]
        ]

    report["passed_file_integrity"] = not any(
        (
            report["train_duplicate_entries"],
            report["valid_duplicate_entries"],
            report["missing_train_files"],
            report["missing_valid_files"],
            report["filename_overlap_count"],
            report["unassigned_event_files"],
        )
    )
    report["passed_piece_isolation"] = report["piece_overlap_count"] == 0
    report["passed_content_isolation"] = (
        report.get("content_hash_overlap_count", 0) == 0
    )
    return report


def select_split_dir(
    dataset: Path, preferred: str, fallback: str | None
) -> str | None:
    if (dataset / preferred).is_dir():
        return preferred
    if fallback and (dataset / fallback).is_dir():
        return fallback
    return None


def markdown_report(report: dict) -> str:
    lines = [
        "# MoonMuse Data Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "| Dataset | Train | Valid | Labels (train / valid) | Exact overlap | Piece overlap | Unassigned | Status |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for name, item in report["datasets"].items():
        status = "PASS" if item["passed_for_formal_evaluation"] else "REVIEW"
        labels = f"{item['train_labels']} / {item['valid_labels']}"
        lines.append(
            f"| {name} | {item['train_files']} | {item['valid_files']} | "
            f"{labels} | {item.get('content_hash_overlap_count', 0)} | "
            f"{item['piece_overlap_count']} | {item['unassigned_event_files']} | "
            f"{status} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- File integrity: **{'PASS' if report['summary']['file_integrity_passed'] else 'BLOCKED'}**",
            f"- Piece-level isolation: **{'PASS' if report['summary']['piece_isolation_passed'] else 'BLOCKED'}**",
            f"- Exact-content isolation: **{'PASS' if report['summary']['content_isolation_passed'] else 'REVIEW'}**",
            f"- Ready for smoke training: **{'YES' if report['summary']['ready_for_smoke'] else 'NO'}**",
            f"- Ready for formal evaluation: **{'YES' if report['summary']['ready_for_formal_evaluation'] else 'NO'}**",
            "",
            "Smoke training only requires complete, non-overlapping split file lists. "
            "Formal evaluation also requires source-piece and exact-content isolation.",
        ]
    )
    blocked = {
        name: item["piece_overlap_examples"]
        for name, item in report["datasets"].items()
        if item["piece_overlap_count"]
    }
    if blocked:
        lines.extend(["", "## Piece-level overlap examples", ""])
        for name, examples in blocked.items():
            lines.append(f"- `{name}`: {', '.join(examples)}")
    content_blocked = {
        name: item.get("content_hash_overlap_examples", [])
        for name, item in report["datasets"].items()
        if item.get("content_hash_overlap_count", 0)
    }
    if content_blocked:
        lines.extend(["", "## Exact-content overlap examples", ""])
        for name, examples in content_blocked.items():
            for example in examples[:5]:
                lines.append(
                    f"- `{name}`: train={example['train']}; valid={example['valid']}"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-root", type=Path, default=REPO_ROOT / "events")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--split-dir-name",
        default="data_splits",
        help="split directory name inside each dataset root",
    )
    parser.add_argument(
        "--fallback-split-dir-name",
        help="use this split directory when the preferred one is absent",
    )
    parser.add_argument("--skip-content-hash", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    events_root = args.events_root.expanduser().resolve()
    datasets = sorted(
        (path, split_dir_name)
        for path in events_root.glob("*/*")
        if (path / "events").is_dir()
        and (
            split_dir_name := select_split_dir(
                path,
                args.split_dir_name,
                args.fallback_split_dir_name,
            )
        )
    )
    if not datasets:
        raise SystemExit(f"No datasets found under {events_root}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_root": str(events_root),
        "datasets": {
            str(dataset.relative_to(events_root)): audit_dataset(
                dataset,
                hash_files=not args.skip_content_hash,
                split_dir_name=split_dir_name,
            )
            for dataset, split_dir_name in datasets
        },
    }
    values = list(report["datasets"].values())
    report["summary"] = {
        "dataset_count": len(values),
        "file_integrity_passed": all(item["passed_file_integrity"] for item in values),
        "piece_isolation_passed": all(item["passed_piece_isolation"] for item in values),
        "content_isolation_passed": all(item["passed_content_isolation"] for item in values),
    }
    report["summary"]["ready_for_smoke"] = report["summary"]["file_integrity_passed"]
    report["summary"]["ready_for_formal_evaluation"] = all(
        (
            report["summary"]["file_integrity_passed"],
            report["summary"]["piece_isolation_passed"],
            report["summary"]["content_isolation_passed"],
        )
    )
    for item in values:
        item["passed_for_formal_evaluation"] = all(
            (
                item["passed_file_integrity"],
                item["passed_piece_isolation"],
                item["passed_content_isolation"],
            )
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else REPO_ROOT.parent / "results" / "data-audit-20260803"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "data_audit.md").write_text(markdown_report(report), encoding="utf-8")

    print(markdown_report(report), end="")
    print(f"Evidence: {output_dir}")
    if args.strict and not report["summary"]["ready_for_formal_evaluation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
