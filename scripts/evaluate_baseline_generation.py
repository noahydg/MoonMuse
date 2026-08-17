#!/usr/bin/env python3
"""Validate the formal EMO-Disentangler author-checkpoint sample set."""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from validate_midi import inspect_midi


QUADRANT_PATTERN = re.compile(r"_(Q[1-4])_full$")


def quadrant_from_stem(stem):
    match = QUADRANT_PATTERN.search(stem)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen EMO-Disentangler generation gate"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-candidates", type=int, default=100)
    parser.add_argument("--min-valid-midi", type=int, default=90)
    parser.add_argument("--min-valid-per-quadrant", type=int, default=20)
    parser.add_argument("--min-median-notes", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    midi_paths = sorted(output_dir.glob("samp_*_Q[1-4]_full.mid"))
    midi_files = [inspect_midi(path) for path in midi_paths]
    valid_midi = [item for item in midi_files if item["valid"]]
    valid_by_quadrant = Counter(
        quadrant
        for quadrant in (
            quadrant_from_stem(Path(item["file"]).stem) for item in valid_midi
        )
        if quadrant
    )
    median_notes = (
        statistics.median(item["notes"] for item in valid_midi)
        if valid_midi
        else 0
    )
    all_quadrants_valid = all(
        valid_by_quadrant[quadrant] >= args.min_valid_per_quadrant
        for quadrant in ("Q1", "Q2", "Q3", "Q4")
    )
    passed = (
        len(midi_paths) == args.expected_candidates
        and len(valid_midi) >= args.min_valid_midi
        and all_quadrants_valid
        and median_notes >= args.min_median_notes
    )
    report = {
        "output_dir": str(output_dir),
        "passed": passed,
        "thresholds": {
            "expected_candidates": args.expected_candidates,
            "min_valid_midi": args.min_valid_midi,
            "min_valid_per_quadrant": args.min_valid_per_quadrant,
            "min_median_notes": args.min_median_notes,
        },
        "candidate_count": len(midi_paths),
        "valid_midi_count": len(valid_midi),
        "valid_by_quadrant": dict(sorted(valid_by_quadrant.items())),
        "median_valid_notes": median_notes,
        "midi_files": midi_files,
    }
    report_path = args.output or output_dir / "generation_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)
    print(f"BASELINE_GENERATION_GATE_PASS output={report_path}")


if __name__ == "__main__":
    main()
