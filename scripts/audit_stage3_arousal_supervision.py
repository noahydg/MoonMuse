#!/usr/bin/env python3
"""Audit Stage 3 arousal labels and performance-side supervision signals.

The audit is deliberately descriptive.  It checks whether High/Low labels are
balanced and whether the real EMOPIA performance events contain consistent
arousal-correlated signals before a new loss or sampling policy is declared.
"""

import argparse
import json
import math
import pickle
import statistics
from collections import Counter
from pathlib import Path


QUADRANT_FACTORS = {
    "Q1": ("Positive", "High"),
    "Q2": ("Negative", "High"),
    "Q3": ("Negative", "Low"),
    "Q4": ("Positive", "Low"),
}

METRIC_DIRECTIONS = {
    "mean_velocity": 1,
    "mean_tempo": 1,
    "notes_per_bar": 1,
    "onsets_per_bar": 1,
    "mean_note_duration": -1,
}


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def numeric_values(events, name):
    values = []
    for event in events:
        if event.get("name") != name:
            continue
        try:
            value = float(event.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def extract_piece_metrics(piece):
    if not isinstance(piece, (list, tuple)) or len(piece) < 3:
        raise ValueError("expected a three-part Stage 3 pickle")
    performance_spans = piece[1]
    events = piece[2]
    if not performance_spans:
        raise ValueError("piece has no Performance spans")

    performance_events = []
    for start, end in performance_spans:
        if not (0 <= start < end <= len(events)):
            raise ValueError("invalid Performance span: ({}, {})".format(start, end))
        segment = events[start:end]
        if not segment or segment[0].get("name") != "Track" or segment[0].get("value") != "Performance":
            raise ValueError("Performance span does not start at Track_Performance")
        performance_events.extend(segment)

    velocities = numeric_values(performance_events, "Note_Velocity")
    tempos = numeric_values(performance_events, "Tempo")
    durations = numeric_values(performance_events, "Note_Duration")
    bars = sum(event.get("name") == "Bar" for event in performance_events)
    notes = len(velocities)
    onsets = sum(event.get("name") == "Beat" for event in performance_events)
    if bars < 1 or notes < 1:
        raise ValueError("Performance track must contain at least one bar and note")

    return {
        "mean_velocity": mean_or_none(velocities),
        "mean_tempo": mean_or_none(tempos),
        "notes_per_bar": notes / bars,
        "onsets_per_bar": onsets / bars,
        "mean_note_duration": mean_or_none(durations),
        "bars": bars,
        "notes": notes,
    }


def labels_from_piece(filename, events):
    quadrant = filename.split("_", 1)[0]
    if quadrant not in QUADRANT_FACTORS:
        raise ValueError("filename has no recognized quadrant: {}".format(filename))
    expected_valence, expected_arousal = QUADRANT_FACTORS[quadrant]
    actual_valence = None
    actual_arousal = None
    if len(events) >= 2:
        if events[0].get("name") == "Valence":
            actual_valence = str(events[0].get("value"))
        if events[1].get("name") == "Arousal":
            actual_arousal = str(events[1].get("value"))
    valid = actual_valence == expected_valence and actual_arousal == expected_arousal
    return {
        "quadrant": quadrant,
        "valence": expected_valence,
        "arousal": expected_arousal,
        "actual_valence": actual_valence,
        "actual_arousal": actual_arousal,
        "valid": valid,
    }


def summarize(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def cohen_d(high_values, low_values):
    high = [value for value in high_values if value is not None and math.isfinite(value)]
    low = [value for value in low_values if value is not None and math.isfinite(value)]
    if len(high) < 2 or len(low) < 2:
        return None
    high_var = statistics.variance(high)
    low_var = statistics.variance(low)
    pooled = math.sqrt(
        ((len(high) - 1) * high_var + (len(low) - 1) * low_var)
        / (len(high) + len(low) - 2)
    )
    if pooled == 0:
        return 0.0
    return (statistics.fmean(high) - statistics.fmean(low)) / pooled


def summarize_split(rows):
    quadrant_counts = Counter(row["quadrant"] for row in rows)
    arousal_counts = Counter(row["arousal"] for row in rows)
    high_count = arousal_counts.get("High", 0)
    low_count = arousal_counts.get("Low", 0)
    smaller = min(high_count, low_count)
    balance_ratio = max(high_count, low_count) / smaller if smaller else None

    metric_summary = {}
    for metric, expected_direction in METRIC_DIRECTIONS.items():
        high = [row[metric] for row in rows if row["arousal"] == "High"]
        low = [row[metric] for row in rows if row["arousal"] == "Low"]
        high_summary = summarize(high)
        low_summary = summarize(low)
        difference = None
        if high_summary["mean"] is not None and low_summary["mean"] is not None:
            difference = high_summary["mean"] - low_summary["mean"]
        effect = cohen_d(high, low)
        direction_passed = (
            difference is not None and difference * expected_direction > 0
        )
        metric_summary[metric] = {
            "expected_high_minus_low_direction": expected_direction,
            "high": high_summary,
            "low": low_summary,
            "high_minus_low_mean": difference,
            "cohen_d": effect,
            "direction_passed": direction_passed,
        }

    return {
        "pieces": len(rows),
        "quadrant_counts": dict(sorted(quadrant_counts.items())),
        "arousal_counts": dict(sorted(arousal_counts.items())),
        "max_to_min_arousal_count_ratio": balance_ratio,
        "metrics": metric_summary,
    }


def audit_dataset(data_root):
    data_root = Path(data_root).resolve()
    event_root = data_root / "events"
    split_root = data_root / "data_splits"
    if not event_root.is_dir() or not split_root.is_dir():
        raise FileNotFoundError("expected events/ and data_splits/ under {}".format(data_root))

    split_names = {}
    for split in ("train", "valid"):
        split_path = split_root / "{}.pkl".format(split)
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        names = list(load_pickle(split_path))
        if len(names) != len(set(names)):
            raise ValueError("{} split contains duplicate filenames".format(split))
        split_names[split] = names
    overlap = sorted(set(split_names["train"]) & set(split_names["valid"]))
    if overlap:
        raise ValueError("train/valid overlap: {}".format(overlap[:5]))

    split_reports = {}
    label_errors = []
    referenced = set()
    for split, names in split_names.items():
        rows = []
        for name in names:
            path = event_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            referenced.add(name)
            piece = load_pickle(path)
            labels = labels_from_piece(name, piece[2])
            if not labels["valid"]:
                label_errors.append({"split": split, "filename": name, **labels})
            metrics = extract_piece_metrics(piece)
            rows.append({"filename": name, **labels, **metrics})
        split_reports[split] = summarize_split(rows)

    event_files = {path.name for path in event_root.glob("*.pkl")}
    unused = sorted(event_files - referenced)
    balance_ok = all(
        report["max_to_min_arousal_count_ratio"] is not None
        and report["max_to_min_arousal_count_ratio"] <= 1.25
        for report in split_reports.values()
    )
    signal_metrics = []
    for metric in METRIC_DIRECTIONS:
        train_metric = split_reports["train"]["metrics"][metric]
        valid_metric = split_reports["valid"]["metrics"][metric]
        if (
            train_metric["direction_passed"]
            and valid_metric["direction_passed"]
            and abs(train_metric["cohen_d"] or 0.0) >= 0.5
            and abs(valid_metric["cohen_d"] or 0.0) >= 0.3
        ):
            signal_metrics.append(metric)

    label_integrity_passed = not label_errors
    empirical_signal_passed = len(signal_metrics) >= 3
    if label_integrity_passed and balance_ok and empirical_signal_passed:
        recommendation = "explicit_counterfactual_arousal_alignment"
    elif not balance_ok:
        recommendation = "arousal_balanced_sampling_or_weighting"
    else:
        recommendation = "repair_labels_or_reassess_stage3_supervision"

    return {
        "schema": "moonmuse-stage3-arousal-supervision-audit-v1",
        "data_root": str(data_root),
        "split_reports": split_reports,
        "train_valid_overlap_count": 0,
        "label_errors": label_errors,
        "unused_event_files": unused,
        "gates": {
            "label_integrity_passed": label_integrity_passed,
            "arousal_balance_passed": balance_ok,
            "empirical_arousal_signal_passed": empirical_signal_passed,
        },
        "consistent_signal_metrics": signal_metrics,
        "recommended_intervention": recommendation,
        "safeguards": [
            "retain ordinary next-token cross-entropy on the observed sequence",
            "apply directional alignment only to Performance-side attributes supported by both splits",
            "keep the existing duplicate-onset and continuity quality gates",
            "reject gains achieved only by pathological loudness, density, or tempo inflation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_dataset(args.data_root)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[stage3-arousal-audit] report:", output)
    print("[stage3-arousal-audit] gates:", report["gates"])
    print("[stage3-arousal-audit] recommendation:", report["recommended_intervention"])
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
