#!/usr/bin/env python3
"""Validate exact Stage 2 note-skeleton preservation in a Stage 3 run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Direct execution (``python scripts/validate_...py``) puts ``scripts/`` rather
# than the repository root on sys.path.  The smoke runner uses that supported
# entry point, so make the sibling ``scripts`` package importable there too.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_stage2_stage3_note_flow import (
    onset_summary,
    parse_track_note_counter,
    sha256_file,
)


SCHEMA = "moonmuse-stage3-skeleton-lock-validation-v1"
SOURCE_PATTERN = re.compile(r"^(?P<source>.+_melody)_Q[1-4]_performance\.txt$")


def validate(root: Path, expected_files: int) -> dict:
    paths = sorted(root.glob("**/stage3_performance/*.txt"))
    if len(paths) != expected_files:
        raise ValueError(f"expected {expected_files} Stage 3 files, found {len(paths)}")
    rows = []
    for performance_path in paths:
        match = SOURCE_PATTERN.match(performance_path.name)
        if match is None:
            raise ValueError(f"unexpected Stage 3 filename: {performance_path.name}")
        source_path = (
            performance_path.parent.parent
            / "stage2_melody_conn"
            / f"{match.group('source')}.txt"
        )
        if not source_path.is_file():
            raise ValueError(f"missing Stage 2 condition: {source_path}")
        stage2 = parse_track_note_counter(source_path, "Melody")
        stage3 = parse_track_note_counter(performance_path, "Performance")
        stage2_stats = onset_summary(stage2)
        stage3_stats = onset_summary(stage3)
        rows.append(
            {
                "performance_path": str(performance_path.relative_to(root)),
                "condition_path": str(source_path.relative_to(root)),
                "performance_sha256": sha256_file(performance_path),
                "condition_sha256": sha256_file(source_path),
                "counter_equal": stage2 == stage3,
                "extra_note_group_count": sum((stage3 - stage2).values()),
                "missing_note_group_count": sum((stage2 - stage3).values()),
                "stage2_note_group_count": stage2_stats["note_group_count"],
                "stage3_note_group_count": stage3_stats["note_group_count"],
            }
        )
    passed = all(
        row["counter_equal"]
        and row["extra_note_group_count"] == 0
        and row["missing_note_group_count"] == 0
        for row in rows
    )
    return {
        "schema": SCHEMA,
        "formalStudyData": False,
        "passed": passed,
        "status": "skeleton_lock_validation_pass" if passed else "skeleton_lock_validation_fail",
        "file_count": len(rows),
        "counter_equal_count": sum(row["counter_equal"] for row in rows),
        "total_extra_note_group_count": sum(
            row["extra_note_group_count"] for row in rows
        ),
        "total_missing_note_group_count": sum(
            row["missing_note_group_count"] for row in rows
        ),
        "rows": rows,
        "interpretation_boundary": (
            "Passing proves note-skeleton preservation for this run only. It does not "
            "prove musical quality, emotion control, or listener preference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.root, args.expected_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
