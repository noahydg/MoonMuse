#!/usr/bin/env python3
"""Audit deterministic incomplete identity prefixes in frozen v81 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from performance_skeleton import discard_incomplete_identity_prefixes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_pair_count(events: list[str], prefix: str, suffix: str) -> int:
    return sum(
        event.startswith(prefix)
        and index + 1 < len(events)
        and events[index + 1].startswith(suffix)
        for index, event in enumerate(events)
    )


def audit(source_root: Path) -> dict:
    paths = sorted(
        path
        for path in source_root.glob("seed*/Q*/*.txt")
        if not path.name.startswith("._")
    )
    rows = []
    note_pairs_before = 0
    note_pairs_after = 0
    chord_pairs_before = 0
    chord_pairs_after = 0
    for path in paths:
        events = path.read_text(encoding="utf-8").splitlines()
        cleaned, discarded = discard_incomplete_identity_prefixes(events)
        note_pairs_before += complete_pair_count(
            events, "Note_Octave_", "Note_Degree_"
        )
        note_pairs_after += complete_pair_count(
            cleaned, "Note_Octave_", "Note_Degree_"
        )
        chord_pairs_before += complete_pair_count(
            events, "Chord_Root_", "Chord_Quality_"
        )
        chord_pairs_after += complete_pair_count(
            cleaned, "Chord_Root_", "Chord_Quality_"
        )
        if discarded:
            rows.append(
                {
                    "source": str(path.relative_to(source_root)),
                    "source_sha256": sha256_file(path),
                    "discarded": [
                        {
                            "line": index + 1,
                            "event": event,
                            "following_event": following,
                        }
                        for index, event, following in discarded
                    ],
                }
            )

    discarded_count = sum(len(row["discarded"]) for row in rows)
    criteria = {
        "source_file_count_300": len(paths) == 300,
        "complete_note_pairs_unchanged": note_pairs_before == note_pairs_after,
        "complete_chord_pairs_unchanged": chord_pairs_before == chord_pairs_after,
    }
    return {
        "schema": "moonmuse-v81-condition-grammar-audit-v1",
        "formalStudyData": False,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "source_root": str(source_root.resolve()),
        "source_file_count": len(paths),
        "issue_file_count": len(rows),
        "discarded_prefix_count": discarded_count,
        "complete_note_pairs_before": note_pairs_before,
        "complete_note_pairs_after": note_pairs_after,
        "complete_chord_pairs_before": chord_pairs_before,
        "complete_chord_pairs_after": chord_pairs_after,
        "rows": rows,
        "interpretation_boundary": (
            "The canonicalizer removes only unpaired Note_Octave or Chord_Root "
            "prefixes. It does not replace a source, add a note, change a complete "
            "onset-pitch identity, or establish affective quality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, default=300)
    parser.add_argument("--expected-issue-files", type=int)
    parser.add_argument("--expected-discarded-prefixes", type=int)
    args = parser.parse_args()
    report = audit(args.source_root)
    expectations = {
        "expected_files": report["source_file_count"] == args.expected_files,
        "expected_issue_files": (
            args.expected_issue_files is None
            or report["issue_file_count"] == args.expected_issue_files
        ),
        "expected_discarded_prefixes": (
            args.expected_discarded_prefixes is None
            or report["discarded_prefix_count"]
            == args.expected_discarded_prefixes
        ),
    }
    report["expectations"] = expectations
    report["passed"] = report["passed"] and all(expectations.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "source_file_count",
                    "issue_file_count",
                    "discarded_prefix_count",
                    "complete_note_pairs_before",
                    "complete_note_pairs_after",
                    "complete_chord_pairs_before",
                    "complete_chord_pairs_after",
                )
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
