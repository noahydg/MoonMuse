#!/usr/bin/env python3
"""Evaluate corrected MoonMuse inference for continuity and structural diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import mido


BAR_TICKS = 1920
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


def quadrant_from_stem(stem: str) -> str:
    matches = [part for part in stem.split("_") if part in QUADRANTS]
    return matches[-1] if matches else "unknown"


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def read_notes(path: Path) -> list[dict]:
    midi = mido.MidiFile(path)
    tempo = 500000
    absolute_tick = 0
    absolute_seconds = 0.0
    active: dict[tuple[int, int], list[tuple[float, int, int]]] = defaultdict(list)
    notes = []

    for message in mido.merge_tracks(midi.tracks):
        absolute_seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        absolute_tick += message.time
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        channel = getattr(message, "channel", 0)
        key = (channel, getattr(message, "note", -1))
        if message.type == "note_on" and message.velocity > 0:
            active[key].append((absolute_seconds, absolute_tick, message.velocity))
            continue
        if message.type not in {"note_off", "note_on"} or not active[key]:
            continue
        start_seconds, start_tick, velocity = active[key].pop(0)
        notes.append(
            {
                "channel": channel,
                "pitch": message.note,
                "velocity": velocity,
                "start_seconds": start_seconds,
                "end_seconds": max(start_seconds, absolute_seconds),
                "start_tick": start_tick,
                "end_tick": max(start_tick, absolute_tick),
            }
        )

    for (channel, pitch), pending in active.items():
        for start_seconds, start_tick, velocity in pending:
            notes.append(
                {
                    "channel": channel,
                    "pitch": pitch,
                    "velocity": velocity,
                    "start_seconds": start_seconds,
                    "end_seconds": max(start_seconds, absolute_seconds),
                    "start_tick": start_tick,
                    "end_tick": max(start_tick, absolute_tick),
                }
            )
    return sorted(notes, key=lambda note: (note["start_seconds"], note["pitch"]))


def onset_pathology_metrics(notes: list[dict]) -> dict:
    onset_pitches: dict[tuple[int, int], list[int]] = defaultdict(list)
    for note in notes:
        onset_pitches[(note["channel"], note["start_tick"])].append(note["pitch"])

    maximum_onset_notes = 0
    maximum_same_pitch = 0
    duplicate_onset_pitch_groups = 0
    for pitches in onset_pitches.values():
        maximum_onset_notes = max(maximum_onset_notes, len(pitches))
        pitch_counts: dict[int, int] = defaultdict(int)
        for pitch in pitches:
            pitch_counts[pitch] += 1
        maximum_same_pitch = max(maximum_same_pitch, max(pitch_counts.values()))
        duplicate_onset_pitch_groups += sum(
            count > 1 for count in pitch_counts.values()
        )
    return {
        "max_onset_note_count": maximum_onset_notes,
        "max_same_pitch_onset_count": maximum_same_pitch,
        "duplicate_onset_pitch_group_count": duplicate_onset_pitch_groups,
    }


def recurrence_score(notes: list[dict], expected_bars: int) -> float:
    bar_vectors = [[0.0] * 12 for _ in range(expected_bars)]
    for note in notes:
        bar = int(note["start_tick"] // BAR_TICKS)
        if 0 <= bar < expected_bars:
            bar_vectors[bar][note["pitch"] % 12] += 1.0
    similarities = [
        cosine(bar_vectors[left], bar_vectors[right])
        for left in range(expected_bars)
        for right in range(left + 2, expected_bars)
        if any(bar_vectors[left]) and any(bar_vectors[right])
    ]
    return max(similarities, default=0.0)


def event_semantics(path: Path, condition_track: str) -> dict:
    events = path.read_text(encoding="utf-8").splitlines()
    performance_positions = [
        index for index, event in enumerate(events) if event == "Track_Performance"
    ]
    well_formed = sum(
        index + 1 < len(events) and events[index + 1] == "Bar_None"
        for index in performance_positions
    )
    return {
        "condition_track": condition_track,
        "condition_track_count": events.count(f"Track_{condition_track}"),
        "performance_bar_count": len(performance_positions),
        "well_formed_performance_bar_count": well_formed,
    }


def inspect_candidate(
    midi_path: Path,
    event_path: Path,
    *,
    condition_track: str,
    expected_bars: int,
    window_seconds: float,
) -> dict:
    notes = read_notes(midi_path)
    intervals = merge_intervals(
        [(note["start_seconds"], note["end_seconds"]) for note in notes]
    )
    if intervals:
        first_start = intervals[0][0]
        last_end = intervals[-1][1]
        note_span = last_end - first_start
        gaps = [
            current[0] - previous[1]
            for previous, current in zip(intervals, intervals[1:])
        ]
        window_end = first_start + window_seconds
        active_seconds = sum(
            max(0.0, min(end, window_end) - max(start, first_start))
            for start, end in intervals
        )
    else:
        first_start = 0.0
        last_end = 0.0
        note_span = 0.0
        gaps = []
        active_seconds = 0.0

    semantic = event_semantics(event_path, condition_track)
    pitches = [note["pitch"] for note in notes]
    return {
        "file": midi_path.name,
        "quadrant": quadrant_from_stem(midi_path.stem),
        "note_count": len(notes),
        "first_note_seconds": first_start,
        "last_note_seconds": last_end,
        "note_span_seconds": note_span,
        "active_seconds_in_window": active_seconds,
        "active_ratio": active_seconds / window_seconds if window_seconds else 0.0,
        "max_internal_gap_seconds": max(gaps, default=0.0),
        "internal_gap_count_ge_1s": sum(gap >= 1.0 for gap in gaps),
        "pitch_range_semitones": max(pitches) - min(pitches) if pitches else 0,
        "bar_recurrence_max": recurrence_score(notes, expected_bars),
        **onset_pathology_metrics(notes),
        **semantic,
    }


def quality_score(row: dict, min_note_span: float, max_internal_gap: float) -> float:
    span_score = min(1.0, row["note_span_seconds"] / min_note_span) if min_note_span else 1.0
    gap_score = max(0.0, 1.0 - row["max_internal_gap_seconds"] / max(max_internal_gap * 4, 1e-6))
    return (
        0.35 * min(1.0, row["active_ratio"])
        + 0.25 * span_score
        + 0.20 * gap_score
        + 0.20 * row["bar_recurrence_max"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--event-dir-name", default="stage3_performance")
    parser.add_argument("--condition-track", choices=("Melody", "Chord"), default="Melody")
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--expected-bars", type=int, default=12)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--min-note-span", type=float, default=20.0)
    parser.add_argument("--max-internal-gap", type=float, default=1.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.75)
    parser.add_argument("--min-notes", type=int, default=32)
    parser.add_argument("--max-same-pitch-onset-count", type=int, default=1)
    parser.add_argument("--min-pass-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    inference_dir = args.inference_dir.resolve()
    event_dir = inference_dir / args.event_dir_name
    midi_dir = inference_dir / "midi"
    midi_paths = sorted(midi_dir.glob("*.mid"))
    rows = []
    missing_event_files = []
    for midi_path in midi_paths:
        event_path = event_dir / f"{midi_path.stem}.txt"
        if not event_path.is_file():
            missing_event_files.append(event_path.name)
            continue
        row = inspect_candidate(
            midi_path,
            event_path,
            condition_track=args.condition_track,
            expected_bars=args.expected_bars,
            window_seconds=args.window_seconds,
        )
        row["quality_pass"] = (
            row["performance_bar_count"] == args.expected_bars
            and row["well_formed_performance_bar_count"] == args.expected_bars
            and row["note_count"] >= args.min_notes
            and row["note_span_seconds"] >= args.min_note_span
            and row["max_internal_gap_seconds"] <= args.max_internal_gap
            and row["active_ratio"] >= args.min_active_ratio
            and row["max_same_pitch_onset_count"]
            <= args.max_same_pitch_onset_count
        )
        row["quality_score"] = quality_score(
            row, args.min_note_span, args.max_internal_gap
        )
        rows.append(row)

    passing = [row for row in rows if row["quality_pass"]]
    pass_ratio = len(passing) / args.expected_candidates if args.expected_candidates else 0.0
    passed = (
        len(midi_paths) == args.expected_candidates
        and len(rows) == args.expected_candidates
        and not missing_event_files
        and pass_ratio >= args.min_pass_ratio
    )
    ranked = sorted(rows, key=lambda row: (-row["quality_score"], row["file"]))
    report = {
        "status": "quality_pilot_pass" if passed else "quality_pilot_fail",
        "passed": passed,
        "inference_dir": str(inference_dir),
        "event_dir_name": args.event_dir_name,
        "condition_track": args.condition_track,
        "thresholds": {
            "expected_candidates": args.expected_candidates,
            "expected_bars": args.expected_bars,
            "window_seconds": args.window_seconds,
            "min_note_span": args.min_note_span,
            "max_internal_gap": args.max_internal_gap,
            "min_active_ratio": args.min_active_ratio,
            "min_notes": args.min_notes,
            "max_same_pitch_onset_count": args.max_same_pitch_onset_count,
            "min_pass_ratio": args.min_pass_ratio,
        },
        "midi_count": len(midi_paths),
        "analysed_count": len(rows),
        "passing_count": len(passing),
        "pass_ratio": pass_ratio,
        "missing_event_files": missing_event_files,
        "aggregate": {
            "median_note_span_seconds": statistics.median(
                [row["note_span_seconds"] for row in rows]
            ) if rows else 0.0,
            "median_active_ratio": statistics.median(
                [row["active_ratio"] for row in rows]
            ) if rows else 0.0,
            "median_max_internal_gap_seconds": statistics.median(
                [row["max_internal_gap_seconds"] for row in rows]
            ) if rows else 0.0,
            "median_bar_recurrence_max": statistics.median(
                [row["bar_recurrence_max"] for row in rows]
            ) if rows else 0.0,
            "median_quality_score": statistics.median(
                [row["quality_score"] for row in rows]
            ) if rows else 0.0,
        },
        "ranked_files": [row["file"] for row in ranked],
        "files": rows,
        "interpretation": {
            "bar_recurrence_max": "descriptive ranking signal, not a validated musicality metric",
            "quality_score": "predeclared pilot ranking only, not formal paper evidence",
            "duplicate_onset_pitch_group_count": (
                "post-pilot diagnostic: exact same-channel, same-onset, same-pitch "
                "duplicates are invalid for the single-piano output"
            ),
        },
    }
    output = args.output or inference_dir / "quality_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_csv
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
