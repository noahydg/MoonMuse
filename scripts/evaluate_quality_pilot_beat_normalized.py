#!/usr/bin/env python3
"""Evaluate fixed-bar MIDI structure with tempo-invariant tick metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mido


SCHEMA = "moonmuse-quality-pilot-beat-normalized-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def read_notes_in_ticks(path: Path) -> tuple[int, list[dict]]:
    midi = mido.MidiFile(path)
    absolute_tick = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    notes = []
    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += message.time
        channel = getattr(message, "channel", 0)
        key = (channel, getattr(message, "note", -1))
        if message.type == "note_on" and message.velocity > 0:
            active[key].append((absolute_tick, message.velocity))
            continue
        if message.type not in {"note_off", "note_on"} or not active[key]:
            continue
        start_tick, velocity = active[key].pop(0)
        notes.append(
            {
                "channel": channel,
                "pitch": message.note,
                "velocity": velocity,
                "start_tick": start_tick,
                "end_tick": max(start_tick, absolute_tick),
            }
        )
    for (channel, pitch), pending in active.items():
        for start_tick, velocity in pending:
            notes.append(
                {
                    "channel": channel,
                    "pitch": pitch,
                    "velocity": velocity,
                    "start_tick": start_tick,
                    "end_tick": max(start_tick, absolute_tick),
                }
            )
    return midi.ticks_per_beat, sorted(
        notes, key=lambda note: (note["start_tick"], note["pitch"])
    )


def event_semantics(path: Path, condition_track: str) -> dict:
    events = path.read_text(encoding="utf-8").splitlines()
    positions = [
        index for index, event in enumerate(events) if event == "Track_Performance"
    ]
    return {
        "condition_track": condition_track,
        "condition_track_count": events.count(f"Track_{condition_track}"),
        "performance_bar_count": len(positions),
        "well_formed_performance_bar_count": sum(
            index + 1 < len(events) and events[index + 1] == "Bar_None"
            for index in positions
        ),
    }


def inspect_candidate(
    midi_path: Path,
    event_path: Path,
    *,
    condition_track: str,
) -> dict:
    ticks_per_beat, notes = read_notes_in_ticks(midi_path)
    intervals = merge_intervals(
        [(note["start_tick"], note["end_tick"]) for note in notes]
    )
    if intervals:
        first_tick = intervals[0][0]
        last_tick = intervals[-1][1]
        span_ticks = last_tick - first_tick
        gaps = [
            current[0] - previous[1]
            for previous, current in zip(intervals, intervals[1:])
        ]
        active_ticks = sum(end - start for start, end in intervals)
    else:
        first_tick = last_tick = span_ticks = active_ticks = 0
        gaps = []

    onset_pitches: dict[tuple[int, int], list[int]] = defaultdict(list)
    for note in notes:
        onset_pitches[(note["channel"], note["start_tick"])].append(note["pitch"])
    same_pitch_counts = []
    for pitches in onset_pitches.values():
        counts: dict[int, int] = defaultdict(int)
        for pitch in pitches:
            counts[pitch] += 1
        same_pitch_counts.extend(counts.values())

    return {
        "file": midi_path.name,
        "midi_sha256": sha256_file(midi_path),
        "event_sha256": sha256_file(event_path),
        "ticks_per_beat": ticks_per_beat,
        "note_count": len(notes),
        "first_note_tick": first_tick,
        "last_note_tick": last_tick,
        "note_span_beats": span_ticks / ticks_per_beat if ticks_per_beat else 0.0,
        "max_internal_gap_beats": (
            max(gaps, default=0) / ticks_per_beat if ticks_per_beat else 0.0
        ),
        "own_span_active_ratio": active_ticks / span_ticks if span_ticks else 0.0,
        "max_same_pitch_onset_count": max(same_pitch_counts, default=0),
        **event_semantics(event_path, condition_track),
    }


def evaluate_rows(
    rows: list[dict],
    *,
    expected_bars: int,
    min_notes: int,
    min_note_span_beats: float,
    max_internal_gap_beats: float,
    min_own_span_active_ratio: float,
    max_same_pitch_onset_count: int,
) -> list[dict]:
    for row in rows:
        row["quality_pass"] = (
            row["performance_bar_count"] == expected_bars
            and row["well_formed_performance_bar_count"] == expected_bars
            and row["note_count"] >= min_notes
            and row["note_span_beats"] >= min_note_span_beats
            and row["max_internal_gap_beats"] <= max_internal_gap_beats
            and row["own_span_active_ratio"] >= min_own_span_active_ratio
            and row["max_same_pitch_onset_count"]
            <= max_same_pitch_onset_count
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--event-dir-name", default="stage3_performance")
    parser.add_argument("--condition-track", choices=("Melody", "Chord"), default="Melody")
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--expected-bars", type=int, default=8)
    parser.add_argument("--min-notes", type=int, default=32)
    parser.add_argument("--min-note-span-beats", type=float, default=28.0)
    parser.add_argument("--max-internal-gap-beats", type=float, default=2.0)
    parser.add_argument("--min-own-span-active-ratio", type=float, default=0.75)
    parser.add_argument("--max-same-pitch-onset-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    inference_dir = args.inference_dir.resolve()
    midi_paths = sorted((inference_dir / "midi").glob("*.mid"))
    event_dir = inference_dir / args.event_dir_name
    rows = []
    missing_events = []
    for midi_path in midi_paths:
        event_path = event_dir / f"{midi_path.stem}.txt"
        if not event_path.is_file():
            missing_events.append(event_path.name)
            continue
        rows.append(
            inspect_candidate(
                midi_path, event_path, condition_track=args.condition_track
            )
        )
    evaluate_rows(
        rows,
        expected_bars=args.expected_bars,
        min_notes=args.min_notes,
        min_note_span_beats=args.min_note_span_beats,
        max_internal_gap_beats=args.max_internal_gap_beats,
        min_own_span_active_ratio=args.min_own_span_active_ratio,
        max_same_pitch_onset_count=args.max_same_pitch_onset_count,
    )
    execution_complete = (
        len(midi_paths) == args.expected_candidates
        and len(rows) == args.expected_candidates
        and not missing_events
    )
    report = {
        "schema": SCHEMA,
        "execution_complete": execution_complete,
        "inference_dir": str(inference_dir),
        "thresholds": {
            "expected_candidates": args.expected_candidates,
            "expected_bars": args.expected_bars,
            "min_notes": args.min_notes,
            "min_note_span_beats": args.min_note_span_beats,
            "max_internal_gap_beats": args.max_internal_gap_beats,
            "min_own_span_active_ratio": args.min_own_span_active_ratio,
            "max_same_pitch_onset_count": args.max_same_pitch_onset_count,
        },
        "individual_file_count": len(rows),
        "individual_quality_pass_count": sum(row["quality_pass"] for row in rows),
        "individual_quality_pass_rate": (
            sum(row["quality_pass"] for row in rows) / len(rows) if rows else 0.0
        ),
        "missing_event_files": missing_events,
        "rows": rows,
        "evidence_boundary": (
            "technical_tick_structure_only_not_listenability_affect_or_preference"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(json.dumps(report, indent=2))
    return 0 if execution_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

