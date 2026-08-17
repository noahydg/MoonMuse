#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from validate_midi import inspect_midi


QUADRANT_PATTERN = re.compile(r"_(Q[1-4])_performance$")


def quadrant_from_stem(stem):
    match = QUADRANT_PATTERN.search(stem)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate the frozen MoonMuse pilot generation gate")
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--min-valid-midi", type=int, default=6)
    parser.add_argument("--min-valid-per-quadrant", type=int, default=1)
    parser.add_argument("--min-median-notes", type=int, default=16)
    parser.add_argument(
        "--event-dir-name",
        choices=("stage2_performance", "stage3_performance"),
        default="stage3_performance",
        help=(
            "performance-event directory under inference_dir; full MoonMuse uses "
            "stage3_performance and the C2P ablation uses stage2_performance"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    inference_dir = args.inference_dir.resolve()
    event_dir = inference_dir / args.event_dir_name
    midi_dir = inference_dir / "midi"
    event_paths = sorted(event_dir.glob("*.txt"))
    midi_paths = sorted(midi_dir.glob("*.mid"))
    midi_files = [inspect_midi(path) for path in midi_paths]
    valid_midi = [item for item in midi_files if item["valid"]]

    generated_by_quadrant = Counter(
        quadrant for quadrant in (quadrant_from_stem(path.stem) for path in event_paths)
        if quadrant
    )
    valid_by_quadrant = Counter(
        quadrant for quadrant in (quadrant_from_stem(Path(item["file"]).stem) for item in valid_midi)
        if quadrant
    )
    median_notes = statistics.median([item["notes"] for item in valid_midi]) if valid_midi else 0
    all_quadrants_valid = all(
        valid_by_quadrant[q] >= args.min_valid_per_quadrant
        for q in ("Q1", "Q2", "Q3", "Q4")
    )
    passed = (
        len(event_paths) == args.expected_candidates
        and len(valid_midi) >= args.min_valid_midi
        and all_quadrants_valid
        and median_notes >= args.min_median_notes
    )
    report = {
        "inference_dir": str(inference_dir),
        "event_dir_name": args.event_dir_name,
        "event_dir": str(event_dir),
        "passed": passed,
        "thresholds": {
            "expected_candidates": args.expected_candidates,
            "min_valid_midi": args.min_valid_midi,
            "min_valid_per_quadrant": args.min_valid_per_quadrant,
            "min_median_notes": args.min_median_notes,
            "required_quadrants": ["Q1", "Q2", "Q3", "Q4"],
        },
        "candidate_count": len(event_paths),
        "generated_by_quadrant": dict(sorted(generated_by_quadrant.items())),
        "valid_midi_count": len(valid_midi),
        "valid_by_quadrant": dict(sorted(valid_by_quadrant.items())),
        "median_valid_notes": median_notes,
        "midi_files": midi_files,
    }
    output = args.output or inference_dir / "generation_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)
    print(f"PILOT_GENERATION_GATE_PASS output={output}")


if __name__ == "__main__":
    main()
