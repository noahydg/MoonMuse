#!/usr/bin/env python3
"""Verify that v81r2 leaves every successfully generated v81r1 file unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files(root: Path, pattern: str) -> dict[Path, Path]:
    return {path.relative_to(root): path for path in root.glob(pattern)}


def compare_kind(
    r1_root: Path, r2_root: Path, pattern: str, expected_r1: int, expected_r2: int
) -> dict:
    r1 = relative_files(r1_root, pattern)
    r2 = relative_files(r2_root, pattern)
    overlap = sorted(set(r1) & set(r2))
    rows = []
    for relative in overlap:
        r1_sha = sha256_file(r1[relative])
        r2_sha = sha256_file(r2[relative])
        rows.append(
            {
                "relative_path": str(relative),
                "v81r1_sha256": r1_sha,
                "v81r2_sha256": r2_sha,
                "byte_identical": r1_sha == r2_sha,
            }
        )
    criteria = {
        "v81r1_count_expected": len(r1) == expected_r1,
        "v81r2_count_expected": len(r2) == expected_r2,
        "every_v81r1_file_overlaps": len(overlap) == len(r1),
        "all_overlap_files_byte_identical": bool(rows)
        and all(row["byte_identical"] for row in rows),
    }
    return {
        "pattern": pattern,
        "v81r1_count": len(r1),
        "v81r2_count": len(r2),
        "overlap_count": len(overlap),
        "byte_identical_count": sum(row["byte_identical"] for row in rows),
        "criteria": criteria,
        "passed": all(criteria.values()),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81r1-root", type=Path, required=True)
    parser.add_argument("--v81r2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-r1", type=int, default=204)
    parser.add_argument("--expected-r2", type=int, default=300)
    args = parser.parse_args()
    events = compare_kind(
        args.v81r1_root,
        args.v81r2_root,
        "seed*/Q*/stage3_performance/*.txt",
        args.expected_r1,
        args.expected_r2,
    )
    midis = compare_kind(
        args.v81r1_root,
        args.v81r2_root,
        "seed*/Q*/midi/*.mid",
        args.expected_r1,
        args.expected_r2,
    )
    report = {
        "schema": "moonmuse-v81r1-v81r2-overlap-validation-v1",
        "formalStudyData": False,
        "passed": events["passed"] and midis["passed"],
        "v81r1_root": str(args.v81r1_root.resolve()),
        "v81r2_root": str(args.v81r2_root.resolve()),
        "events": events,
        "midis": midis,
        "interpretation_boundary": (
            "Passing proves byte identity only for files v81r1 successfully "
            "produced. It does not establish quality or affective superiority."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "event_overlap": events["overlap_count"],
                "event_identical": events["byte_identical_count"],
                "midi_overlap": midis["overlap_count"],
                "midi_identical": midis["byte_identical_count"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
