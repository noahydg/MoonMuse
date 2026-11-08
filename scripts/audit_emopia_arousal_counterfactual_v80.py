#!/usr/bin/env python3
"""Run the frozen v80 velocity x tempo counterfactual classifier audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import mido
import numpy as np

try:
    from scripts.build_corrected_three_system_population import sha256_file
    from scripts.run_official_emopia_classifier_v78 import OfficialRemiClassifier
except ImportError:
    from build_corrected_three_system_population import sha256_file
    from run_official_emopia_classifier_v78 import OfficialRemiClassifier


VELOCITY_SHIFTS = (-16, 0, 16)
TEMPO_MULTIPLIERS = (0.80, 1.00, 1.25)


def load_sources(prediction_dir: Path) -> list[dict]:
    specifications = (
        ("generated_eight_bar_arousal.csv", 900),
        ("emopia_test_eight_bar_arousal.csv", 86),
    )
    sources = []
    for filename, expected in specifications:
        with (prediction_dir / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"{filename}: expected {expected}, found {len(rows)}")
        for row in rows:
            path = Path(row["midi_path"])
            if sha256_file(path) != row["midi_sha256"]:
                raise ValueError(f"source hash mismatch: {path}")
            sources.append(
                {
                    "population": row["population"],
                    "system": row["system"],
                    "seed": row["seed"],
                    "quadrant": row["quadrant"],
                    "sample_id": row["sample_id"],
                    "midi_path": path,
                    "midi_sha256": row["midi_sha256"],
                }
            )
    if len(sources) != 986:
        raise ValueError(f"expected 986 sources, found {len(sources)}")
    return sources


def message_payload(message: mido.Message) -> dict:
    payload = message.dict()
    payload.pop("time", None)
    if message.type == "note_on" and message.velocity > 0:
        payload["velocity"] = "<VELOCITY>"
    if message.type == "set_tempo":
        payload["tempo"] = "<TEMPO>"
    return payload


def signatures(midi: mido.MidiFile) -> dict:
    invariant_events = []
    note_events = []
    velocities = []
    tempos = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            invariant_events.append((track_index, order, tick, message_payload(message)))
            if not message.is_meta and message.type in {"note_on", "note_off"}:
                note_events.append(
                    (
                        track_index,
                        order,
                        tick,
                        message.type,
                        getattr(message, "channel", 0),
                        message.note,
                        1 if message.type == "note_on" and message.velocity > 0 else 0,
                    )
                )
            if not message.is_meta and message.type == "note_on" and message.velocity > 0:
                velocities.append(message.velocity)
            if message.type == "set_tempo":
                tempos.append(message.tempo)
    if not velocities or not tempos:
        raise ValueError("MIDI must contain positive note-on velocities and explicit tempo")
    encoded = json.dumps(invariant_events, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "invariant_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "note_events": note_events,
        "velocities": velocities,
        "tempos": tempos,
        "mean_velocity": float(statistics.mean(velocities)),
        "median_bpm": float(mido.tempo2bpm(statistics.median(tempos))),
    }


def transform_midi(
    source: Path,
    destination: Path,
    velocity_shift: int,
    tempo_multiplier: float,
) -> dict:
    midi = mido.MidiFile(source)
    before = signatures(midi)
    transformed = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    for track in midi.tracks:
        output_track = mido.MidiTrack()
        for message in track:
            copied = message.copy()
            if not message.is_meta and message.type == "note_on" and message.velocity > 0:
                copied.velocity = max(1, min(127, message.velocity + velocity_shift))
            elif message.type == "set_tempo":
                copied.tempo = max(1, min(16777215, round(message.tempo / tempo_multiplier)))
            output_track.append(copied)
        transformed.tracks.append(output_track)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.save(destination)
    after = signatures(mido.MidiFile(destination))
    expected_velocities = [max(1, min(127, value + velocity_shift)) for value in before["velocities"]]
    expected_tempos = [max(1, min(16777215, round(value / tempo_multiplier))) for value in before["tempos"]]
    if before["invariant_sha256"] != after["invariant_sha256"]:
        raise ValueError(f"non-target payload changed: {destination}")
    if before["note_events"] != after["note_events"]:
        raise ValueError(f"note event signature changed: {destination}")
    if after["velocities"] != expected_velocities:
        raise ValueError(f"velocity transform mismatch: {destination}")
    if after["tempos"] != expected_tempos:
        raise ValueError(f"tempo transform mismatch: {destination}")
    return {
        "source_mean_velocity": before["mean_velocity"],
        "source_median_bpm": before["median_bpm"],
        "variant_mean_velocity": after["mean_velocity"],
        "variant_median_bpm": after["median_bpm"],
        "invariant_sha256": after["invariant_sha256"],
        "note_event_count": len(after["note_events"]),
    }


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def contrast_summary(rows: list[dict]) -> dict:
    by_source = defaultdict(dict)
    for row in rows:
        by_source[row["source_key"]][(int(row["velocity_shift"]), float(row["tempo_multiplier"]))] = row
    records = []
    for source_key, variants in by_source.items():
        velocity_low = variants[(-16, 1.0)]
        velocity_high = variants[(16, 1.0)]
        tempo_low = variants[(0, 0.8)]
        tempo_high = variants[(0, 1.25)]
        records.append(
            {
                "source_key": source_key,
                "system": velocity_low["system"],
                "velocity_logit_contrast": float(velocity_high["logit_margin"] - velocity_low["logit_margin"]),
                "velocity_probability_contrast": float(velocity_high["probability_HA"] - velocity_low["probability_HA"]),
                "velocity_class_flip": f"{velocity_low['prediction']}->{velocity_high['prediction']}",
                "tempo_logit_contrast": float(tempo_high["logit_margin"] - tempo_low["logit_margin"]),
                "tempo_probability_contrast": float(tempo_high["probability_HA"] - tempo_low["probability_HA"]),
                "tempo_class_flip": f"{tempo_low['prediction']}->{tempo_high['prediction']}",
            }
        )

    def summarize_group(group: list[dict]) -> dict:
        velocity_logits = [row["velocity_logit_contrast"] for row in group]
        velocity_probabilities = [row["velocity_probability_contrast"] for row in group]
        tempo_logits = [row["tempo_logit_contrast"] for row in group]
        tempo_probabilities = [row["tempo_probability_contrast"] for row in group]
        median_velocity_abs = float(np.median(np.abs(velocity_logits)))
        median_tempo_abs = float(np.median(np.abs(tempo_logits)))
        return {
            "velocity_logit_contrast": quantiles(velocity_logits),
            "velocity_probability_contrast": quantiles(velocity_probabilities),
            "velocity_positive_count": sum(value > 0 for value in velocity_logits),
            "velocity_class_flips": dict(sorted(Counter(row["velocity_class_flip"] for row in group).items())),
            "tempo_logit_contrast": quantiles(tempo_logits),
            "tempo_probability_contrast": quantiles(tempo_probabilities),
            "tempo_positive_count": sum(value > 0 for value in tempo_logits),
            "tempo_class_flips": dict(sorted(Counter(row["tempo_class_flip"] for row in group).items())),
            "median_absolute_logit_ratio_velocity_over_tempo": (
                median_velocity_abs / median_tempo_abs if median_tempo_abs else None
            ),
        }

    groups = {"pooled": records}
    for system in sorted({row["system"] for row in records}):
        groups[system] = [row for row in records if row["system"] == system]
    summary = {name: summarize_group(group) for name, group in groups.items()}
    pooled = summary["pooled"]
    n = pooled["velocity_logit_contrast"]["n"]
    velocity_dominant = (
        pooled["velocity_positive_count"] / n >= 0.90
        and (pooled["median_absolute_logit_ratio_velocity_over_tempo"] or 0.0) >= 3.0
        and pooled["velocity_probability_contrast"]["median"] >= 0.25
    )
    tempo_small = abs(pooled["tempo_probability_contrast"]["median"]) <= 0.10
    return {
        "groups": summary,
        "diagnostic_labels": {
            "velocity_dominant_sensitivity": velocity_dominant,
            "tempo_probability_effect_small": tempo_small,
        },
        "per_source": records,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    sources = load_sources(args.prediction_dir)
    classifier = OfficialRemiClassifier(args.upstream_root, "arousal", "cpu")
    manifest_rows = []
    prediction_rows = []
    total = len(sources) * len(VELOCITY_SHIFTS) * len(TEMPO_MULTIPLIERS)
    completed = 0
    for source_index, source in enumerate(sources):
        source_key = f"{source['population']}::{source['system']}::{source['sample_id']}"
        for velocity_shift in VELOCITY_SHIFTS:
            for tempo_multiplier in TEMPO_MULTIPLIERS:
                variant_name = f"v{velocity_shift:+03d}_t{tempo_multiplier:.2f}.mid"
                destination = (
                    args.output_dir
                    / "counterfactual_midi"
                    / f"source_{source_index:04d}"
                    / variant_name
                )
                validation = transform_midi(
                    source["midi_path"], destination, velocity_shift, tempo_multiplier
                )
                prediction, logits, probabilities, token_count = classifier.predict(destination)
                manifest = {
                    "source_key": source_key,
                    "population": source["population"],
                    "system": source["system"],
                    "seed": source["seed"],
                    "quadrant": source["quadrant"],
                    "sample_id": source["sample_id"],
                    "source_midi": str(source["midi_path"]),
                    "source_sha256": source["midi_sha256"],
                    "velocity_shift": velocity_shift,
                    "tempo_multiplier": f"{tempo_multiplier:.2f}",
                    "variant_midi": str(destination),
                    "variant_sha256": sha256_file(destination),
                    **validation,
                }
                manifest_rows.append(manifest)
                prediction_rows.append(
                    {
                        "source_key": source_key,
                        "population": source["population"],
                        "system": source["system"],
                        "seed": source["seed"],
                        "quadrant": source["quadrant"],
                        "sample_id": source["sample_id"],
                        "velocity_shift": velocity_shift,
                        "tempo_multiplier": tempo_multiplier,
                        "prediction": prediction,
                        "logit_HA": logits[0],
                        "logit_LA": logits[1],
                        "logit_margin": logits[0] - logits[1],
                        "probability_HA": probabilities[0],
                        "probability_LA": probabilities[1],
                        "token_count": token_count,
                    }
                )
                completed += 1
                if completed == 1 or completed % 250 == 0 or completed == total:
                    print(f"[v80] {completed}/{total}", flush=True)

    for path, rows in (
        (args.output_dir / "counterfactual_manifest.csv", manifest_rows),
        (args.output_dir / "predictions.csv", prediction_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    source_velocity = [row["source_mean_velocity"] for row in manifest_rows[::9]]
    source_bpm = [row["source_median_bpm"] for row in manifest_rows[::9]]
    velocity_limits = tuple(np.quantile(source_velocity, [0.01, 0.99]))
    bpm_limits = tuple(np.quantile(source_bpm, [0.01, 0.99]))
    overlap = {}
    for velocity_shift in VELOCITY_SHIFTS:
        group = [row for row in manifest_rows if row["velocity_shift"] == velocity_shift]
        overlap[f"velocity_shift_{velocity_shift:+d}"] = sum(
            velocity_limits[0] <= row["variant_mean_velocity"] <= velocity_limits[1]
            for row in group
        ) / len(group)
    for tempo_multiplier in TEMPO_MULTIPLIERS:
        group = [row for row in manifest_rows if row["tempo_multiplier"] == f"{tempo_multiplier:.2f}"]
        overlap[f"tempo_multiplier_{tempo_multiplier:.2f}"] = sum(
            bpm_limits[0] <= row["variant_median_bpm"] <= bpm_limits[1]
            for row in group
        ) / len(group)

    contrasts = contrast_summary(prediction_rows)
    per_source = contrasts.pop("per_source")
    with (args.output_dir / "paired_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_source[0]))
        writer.writeheader()
        writer.writerows(per_source)
    report = {
        "schema": "moonmuse-emopia-arousal-classifier-factorial-counterfactual-v80",
        "status": "complete",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "source_count": len(sources),
        "variant_count": len(prediction_rows),
        "classifier": classifier.provenance(),
        "intervention": {
            "velocity_shifts": VELOCITY_SHIFTS,
            "tempo_multipliers": TEMPO_MULTIPLIERS,
        },
        "source_feature_1st_99th_percentile": {
            "mean_velocity": velocity_limits,
            "median_bpm": bpm_limits,
        },
        "distribution_overlap_fraction": overlap,
        "contrasts": contrasts,
        "artifact_sha256": {
            "counterfactual_manifest.csv": sha256_file(args.output_dir / "counterfactual_manifest.csv"),
            "predictions.csv": sha256_file(args.output_dir / "predictions.csv"),
            "paired_contrasts.csv": sha256_file(args.output_dir / "paired_contrasts.csv"),
        },
    }
    status_path = args.output_dir / "status.json"
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "CLASSIFIER_V80_COMPLETE").write_text(
        sha256_file(status_path) + "  status.json\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
