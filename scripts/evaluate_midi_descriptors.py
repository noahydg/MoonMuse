#!/usr/bin/env python3
"""Extract transparent per-file descriptors from generated MIDI sets."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import mido


METRICS = (
    "duration_seconds",
    "note_count",
    "note_density_per_second",
    "onset_density_per_second",
    "pitch_class_entropy_bits",
    "pitch_range_semitones",
    "major_mode_indicator",
    "key_profile_correlation",
    "polyphonic_onset_rate",
    "mean_note_duration_seconds",
    "mean_velocity",
    "velocity_std",
    "time_weighted_tempo_bpm",
    "time_weighted_tempo_std",
)

MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def quadrant_from_stem(stem):
    quadrants = [part for part in stem.split("_") if part in {"Q1", "Q2", "Q3", "Q4"}]
    return quadrants[-1] if quadrants else "unknown"


def entropy_bits(values):
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def pearson_correlation(left, right):
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def estimate_key_mode(notes):
    pitch_profile = [0.0] * 12
    for note in notes:
        pitch_profile[note["pitch"] % 12] += max(note["duration"], 1e-6)
    if not any(pitch_profile):
        return {"tonic_pc": None, "mode": "unknown", "correlation": 0.0}

    candidates = []
    for tonic in range(12):
        relative_profile = [pitch_profile[(tonic + interval) % 12] for interval in range(12)]
        candidates.append(
            (pearson_correlation(relative_profile, MAJOR_PROFILE), tonic, "major")
        )
        candidates.append(
            (pearson_correlation(relative_profile, MINOR_PROFILE), tonic, "minor")
        )
    correlation, tonic, mode = max(candidates)
    return {"tonic_pc": tonic, "mode": mode, "correlation": correlation}


def inspect_midi(path):
    midi = mido.MidiFile(path)
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000
    absolute_tick = 0
    absolute_seconds = 0.0
    tempo_weighted_sum = 0.0
    tempo_squared_weighted_sum = 0.0
    tempo_duration = 0.0
    active = defaultdict(list)
    notes = []
    onsets = Counter()

    for message in mido.merge_tracks(midi.tracks):
        delta_seconds = mido.tick2second(message.time, ticks_per_beat, tempo)
        bpm = mido.tempo2bpm(tempo)
        tempo_weighted_sum += bpm * delta_seconds
        tempo_squared_weighted_sum += bpm * bpm * delta_seconds
        tempo_duration += delta_seconds
        absolute_seconds += delta_seconds
        absolute_tick += message.time

        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if message.type == "note_on" and message.velocity > 0:
            channel = getattr(message, "channel", 0)
            active[(channel, message.note)].append((absolute_seconds, message.velocity))
            onsets[absolute_tick] += 1
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        channel = getattr(message, "channel", 0)
        key = (channel, message.note)
        if not active[key]:
            continue
        start, velocity = active[key].pop(0)
        notes.append(
            {
                "pitch": message.note,
                "velocity": velocity,
                "duration": max(0.0, absolute_seconds - start),
            }
        )

    for (_, pitch), starts in active.items():
        for start, velocity in starts:
            notes.append(
                {
                    "pitch": pitch,
                    "velocity": velocity,
                    "duration": max(0.0, absolute_seconds - start),
                }
            )

    pitches = [note["pitch"] for note in notes]
    velocities = [note["velocity"] for note in notes]
    durations = [note["duration"] for note in notes]
    duration_seconds = absolute_seconds
    onset_count = len(onsets)
    polyphonic_onsets = sum(count >= 2 for count in onsets.values())
    tempo_mean = (
        tempo_weighted_sum / tempo_duration
        if tempo_duration > 0
        else mido.tempo2bpm(tempo)
    )
    tempo_variance = (
        max(0.0, tempo_squared_weighted_sum / tempo_duration - tempo_mean**2)
        if tempo_duration > 0
        else 0.0
    )
    key_mode = estimate_key_mode(notes)

    return {
        "file": path.name,
        "quadrant": quadrant_from_stem(path.stem),
        "duration_seconds": duration_seconds,
        "note_count": len(notes),
        "note_density_per_second": len(notes) / duration_seconds if duration_seconds else 0.0,
        "onset_density_per_second": onset_count / duration_seconds if duration_seconds else 0.0,
        "pitch_class_entropy_bits": entropy_bits([pitch % 12 for pitch in pitches]),
        "pitch_range_semitones": max(pitches) - min(pitches) if pitches else 0,
        "estimated_key_pc": key_mode["tonic_pc"],
        "estimated_mode": key_mode["mode"],
        "major_mode_indicator": 1.0 if key_mode["mode"] == "major" else 0.0,
        "key_profile_correlation": key_mode["correlation"],
        "polyphonic_onset_rate": polyphonic_onsets / onset_count if onset_count else 0.0,
        "mean_note_duration_seconds": statistics.fmean(durations) if durations else 0.0,
        "mean_velocity": statistics.fmean(velocities) if velocities else 0.0,
        "velocity_std": statistics.pstdev(velocities) if len(velocities) > 1 else 0.0,
        "time_weighted_tempo_bpm": tempo_mean,
        "time_weighted_tempo_std": math.sqrt(tempo_variance),
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["quadrant"]].append(row)
        grouped["all"].append(row)

    output = {}
    for group, group_rows in sorted(grouped.items()):
        output[group] = {"file_count": len(group_rows), "metrics": {}}
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            output[group]["metrics"][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
            }
    return output


def resolve_validation_path(input_dir, explicit):
    if explicit:
        return explicit
    candidates = (
        input_dir / "midi_validation.json",
        input_dir.parent / "midi_validation.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--pattern", default="*.mid")
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    all_paths = sorted(args.input_dir.glob(args.pattern))
    validation_path = resolve_validation_path(args.input_dir, args.validation_json)
    excluded_files = []
    paths = all_paths
    if validation_path and not args.include_invalid:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        valid_names = {
            item["file"] for item in validation.get("files", []) if item.get("valid")
        }
        paths = [path for path in all_paths if path.name in valid_names]
        excluded_files = [path.name for path in all_paths if path.name not in valid_names]
    if not paths:
        raise SystemExit(f"no MIDI files matched {args.pattern!r} in {args.input_dir}")
    rows = [inspect_midi(path) for path in paths]
    report = {
        "input_dir": str(args.input_dir.resolve()),
        "pattern": args.pattern,
        "validation_json": str(validation_path.resolve()) if validation_path else None,
        "population": (
            "all_matched_midi" if args.include_invalid or not validation_path
            else "validation_passed_midi_only"
        ),
        "matched_file_count": len(all_paths),
        "analysed_file_count": len(paths),
        "excluded_files": excluded_files,
        "definitions": {
            "pitch_class_entropy_bits": "Shannon entropy over note pitch classes",
            "polyphonic_onset_rate": "fraction of unique onset ticks containing two or more notes",
            "tempo": "time-weighted over the rendered MIDI duration",
            "estimated_key_mode": (
                "Krumhansl-Schmuckler duration-weighted pitch-class profile; "
                "a music-theory proxy rather than an emotion classifier"
            ),
        },
        "aggregate": aggregate(rows),
        "files": rows,
    }

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "file",
                    "quadrant",
                    "estimated_key_pc",
                    "estimated_mode",
                    *METRICS,
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
