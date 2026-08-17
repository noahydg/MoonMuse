#!/usr/bin/env python3
"""Audit whether Stage 3 changes Stage 2 onset-pitch identities in frozen v63."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path


SCHEMA = "moonmuse-stage2-stage3-note-flow-audit-v1"
SOURCE_PATTERN = re.compile(r"^(?P<source>.+_melody)_Q[1-4]_performance\.txt$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_track_note_counter(path: Path, track: str) -> Counter:
    """Return (bar, beat, octave, degree) multiplicities for one track type."""
    marker = f"Track_{track}"
    counter: Counter = Counter()
    active = False
    bar = -1
    beat = None
    pending_octave = None
    for event in path.read_text(encoding="utf-8").splitlines():
        if event.startswith("Track_"):
            active = event == marker
            if active:
                bar += 1
                beat = None
                pending_octave = None
            continue
        if not active:
            continue
        if event.startswith("Beat_"):
            beat = int(event.rsplit("_", 1)[1])
            pending_octave = None
        elif event.startswith("Note_Octave_"):
            pending_octave = int(event.rsplit("_", 1)[1])
        elif event.startswith("Note_Degree_"):
            if beat is not None and pending_octave is not None:
                degree = event.removeprefix("Note_Degree_")
                counter[(bar, beat, pending_octave, degree)] += 1
            pending_octave = None
        elif event.startswith("Note_"):
            pending_octave = None
    return counter


def onset_summary(counter: Counter) -> dict:
    onset_counts: Counter = Counter()
    for (bar, beat, _octave, _degree), multiplicity in counter.items():
        onset_counts[(bar, beat)] += multiplicity
    values = list(onset_counts.values())
    return {
        "note_group_count": sum(counter.values()),
        "unique_onset_pitch_count": len(counter),
        "duplicate_note_group_count": sum(max(0, count - 1) for count in counter.values()),
        "onset_count": len(onset_counts),
        "max_notes_per_onset": max(values, default=0),
        "onsets_with_at_least_4_notes": sum(value >= 4 for value in values),
    }


def verify_provenance(root: Path, pilot_comparison: Path) -> dict:
    pilot = json.loads(pilot_comparison.read_text(encoding="utf-8"))
    expected = pilot["inputs"]["baseline_quality_sha256"]
    quality = root / "gate_c_analysis" / "quality_file_summary.csv"
    actual = sha256_file(quality)
    if actual != expected:
        raise ValueError(
            "v63 quality-summary provenance mismatch: "
            f"expected {expected}, got {actual} from {quality}"
        )
    return {
        "pilot_comparison": str(pilot_comparison.resolve()),
        "quality_summary": str(quality.resolve()),
        "quality_summary_sha256": actual,
        "quality_summary_hash_matches": True,
    }


def audit(root: Path, pilot_comparison: Path, expected_files: int | None = 24) -> dict:
    provenance = verify_provenance(root, pilot_comparison)
    performance_paths = sorted(root.glob("seed*/*/stage3_performance/*.txt"))
    if expected_files is not None and len(performance_paths) != expected_files:
        raise ValueError(
            f"expected {expected_files} Stage 3 files, found {len(performance_paths)}"
        )
    rows = []
    source_hashes = {}
    for performance_path in performance_paths:
        relative = performance_path.relative_to(root)
        match = SOURCE_PATTERN.match(performance_path.name)
        if match is None:
            raise ValueError(f"unexpected Stage 3 filename: {performance_path.name}")
        source_path = (
            performance_path.parent.parent
            / "stage2_melody_conn"
            / f"{match.group('source')}.txt"
        )
        if not source_path.is_file():
            raise ValueError(f"missing Stage 2 source: {source_path}")
        source_relative = str(source_path.relative_to(root))
        source_hashes[source_relative] = sha256_file(source_path)
        stage2 = parse_track_note_counter(source_path, "Melody")
        stage3 = parse_track_note_counter(performance_path, "Performance")
        stage2_stats = onset_summary(stage2)
        stage3_stats = onset_summary(stage3)
        extra = stage3 - stage2
        missing = stage2 - stage3
        if stage2 == stage3:
            flow_class = "exact_copy"
        elif set(stage2) != set(stage3):
            flow_class = "onset_pitch_identity_changed"
        elif extra and not missing:
            flow_class = "multiplicity_extra_only"
        elif missing and not extra:
            flow_class = "multiplicity_missing_only"
        else:
            flow_class = "multiplicity_redistributed"
        target_match = re.search(r"_(Q[1-4])_performance\.txt$", performance_path.name)
        rows.append(
            {
                "relative_path": str(relative),
                "seed": relative.parts[0].removeprefix("seed"),
                "source_valence": relative.parts[1],
                "target_quadrant": target_match.group(1) if target_match else "unknown",
                "source_relative_path": source_relative,
                "source_sha256": source_hashes[source_relative],
                "performance_sha256": sha256_file(performance_path),
                "onset_pitch_set_equal": set(stage2) == set(stage3),
                "onset_pitch_counter_equal": stage2 == stage3,
                "flow_class": flow_class,
                "extra_note_group_count": sum(extra.values()),
                "missing_note_group_count": sum(missing.values()),
                **{f"stage2_{key}": value for key, value in stage2_stats.items()},
                **{f"stage3_{key}": value for key, value in stage3_stats.items()},
            }
        )
    duplicate_source_count = len(
        {
            row["source_relative_path"]
            for row in rows
            if row["stage2_duplicate_note_group_count"] > 0
        }
    )
    flow_classes = Counter(row["flow_class"] for row in rows)
    return {
        "schema": SCHEMA,
        "formalStudyData": False,
        "generation_root": str(root.resolve()),
        "provenance": provenance,
        "file_count": len(rows),
        "unique_stage2_source_count": len(source_hashes),
        "onset_pitch_set_equal_count": sum(row["onset_pitch_set_equal"] for row in rows),
        "onset_pitch_counter_equal_count": sum(
            row["onset_pitch_counter_equal"] for row in rows
        ),
        "total_extra_note_group_count": sum(row["extra_note_group_count"] for row in rows),
        "total_missing_note_group_count": sum(
            row["missing_note_group_count"] for row in rows
        ),
        "stage2_duplicate_file_count": sum(
            row["stage2_duplicate_note_group_count"] > 0 for row in rows
        ),
        "unique_stage2_source_with_duplicates_count": duplicate_source_count,
        "stage3_duplicate_file_count": sum(
            row["stage3_duplicate_note_group_count"] > 0 for row in rows
        ),
        "flow_class_counts": dict(sorted(flow_classes.items())),
        "median_stage2_max_notes_per_onset": statistics.median(
            row["stage2_max_notes_per_onset"] for row in rows
        ),
        "median_stage3_max_notes_per_onset": statistics.median(
            row["stage3_max_notes_per_onset"] for row in rows
        ),
        "rows": rows,
        "interpretation_boundary": (
            "This audit tests discrete bar/beat/octave/degree note identities only. "
            "Stage 3 still controls durations, velocities, and tempo, so audible overlap "
            "or heaviness cannot be assigned to Stage 2 from identity equality alone."
        ),
    }


def write_outputs(report: dict, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    markdown = [
        "# Stage 2 to Stage 3 note-flow audit",
        "",
        f"- Stage 3 files: `{report['file_count']}`",
        f"- Unique Stage 2 sources: `{report['unique_stage2_source_count']}`",
        f"- Onset-pitch set matches: `{report['onset_pitch_set_equal_count']}/{report['file_count']}`",
        f"- Exact multiplicity matches: `{report['onset_pitch_counter_equal_count']}/{report['file_count']}`",
        f"- Stage 3 extra note groups: `{report['total_extra_note_group_count']}`",
        f"- Stage 3 missing note groups: `{report['total_missing_note_group_count']}`",
        f"- Files with Stage 2 / Stage 3 exact duplicates: "
        f"`{report['stage2_duplicate_file_count']} / {report['stage3_duplicate_file_count']}`",
        f"- Unique Stage 2 sources with exact duplicates: "
        f"`{report['unique_stage2_source_with_duplicates_count']}`",
        f"- Note-flow classes: `{json.dumps(report['flow_class_counts'], sort_keys=True)}`",
        f"- Median maximum notes per onset, Stage 2 / Stage 3: "
        f"`{report['median_stage2_max_notes_per_onset']} / "
        f"{report['median_stage3_max_notes_per_onset']}`",
        "- Formal study data: `false`",
        "",
        "## Boundary",
        "",
        report["interpretation_boundary"],
        "",
    ]
    md_path.write_text("\n".join(markdown), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--pilot-comparison", required=True, type=Path)
    parser.add_argument("--expected-files", type=int, default=24)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.generation_root, args.pilot_comparison, args.expected_files)
    write_outputs(report, args.output_prefix)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
