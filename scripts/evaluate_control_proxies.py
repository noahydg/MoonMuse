#!/usr/bin/env python3
"""Compute seed-paired valence/arousal contrasts from transparent MIDI proxies."""

import argparse
import csv
import json
import statistics
from pathlib import Path


CONTRASTS = {
    "valence_positive_minus_negative": {"Q1": 0.5, "Q4": 0.5, "Q2": -0.5, "Q3": -0.5},
    "valence_at_high_arousal": {"Q1": 1.0, "Q2": -1.0},
    "valence_at_low_arousal": {"Q4": 1.0, "Q3": -1.0},
    "arousal_high_minus_low": {"Q1": 0.5, "Q2": 0.5, "Q3": -0.5, "Q4": -0.5},
    "arousal_at_positive_valence": {"Q1": 1.0, "Q4": -1.0},
    "arousal_at_negative_valence": {"Q2": 1.0, "Q3": -1.0},
}

PRIMARY_PROXIES = {
    ("major_mode_indicator", "valence_positive_minus_negative"): (
        "valence_target",
        "positive",
    ),
    ("note_density_per_second", "arousal_high_minus_low"): (
        "arousal_target",
        "positive",
    ),
    ("onset_density_per_second", "arousal_high_minus_low"): (
        "arousal_target",
        "positive",
    ),
    ("mean_velocity", "arousal_high_minus_low"): (
        "arousal_target",
        "positive",
    ),
    ("major_mode_indicator", "arousal_at_positive_valence"): (
        "valence_non_target_drift",
        "near_zero",
    ),
    ("major_mode_indicator", "arousal_at_negative_valence"): (
        "valence_non_target_drift",
        "near_zero",
    ),
    ("note_density_per_second", "valence_at_high_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
    ("note_density_per_second", "valence_at_low_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
    ("onset_density_per_second", "valence_at_high_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
    ("onset_density_per_second", "valence_at_low_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
    ("mean_velocity", "valence_at_high_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
    ("mean_velocity", "valence_at_low_arousal"): (
        "arousal_non_target_drift",
        "near_zero",
    ),
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptors_by_seed", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.descriptors_by_seed.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    values = {}
    models = set()
    for row in source_rows:
        if row["quadrant"] not in {"Q1", "Q2", "Q3", "Q4"}:
            continue
        models.add(row["model"])
        key = (int(row["seed"]), row["metric"], row["quadrant"])
        values[key] = float(row["within_seed_mean"])
    if len(models) != 1:
        raise SystemExit(f"expected one model in input, found {sorted(models)}")
    model = next(iter(models))
    seeds = sorted({key[0] for key in values})
    metrics = sorted({key[1] for key in values})

    seed_rows = []
    for seed in seeds:
        for metric in metrics:
            quadrants = {
                quadrant: values.get((seed, metric, quadrant))
                for quadrant in ("Q1", "Q2", "Q3", "Q4")
            }
            if any(value is None for value in quadrants.values()):
                continue
            for contrast, weights in CONTRASTS.items():
                value = sum(quadrants[quadrant] * weight for quadrant, weight in weights.items())
                role, expected = PRIMARY_PROXIES.get(
                    (metric, contrast), ("descriptive_only", "none")
                )
                seed_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "contrast": contrast,
                        "value": value,
                        "role": role,
                        "expected_direction": expected,
                    }
                )

    if not seed_rows:
        raise SystemExit("no complete Q1-Q4 seed/metric groups found")
    grouped = {}
    for row in seed_rows:
        key = (row["metric"], row["contrast"], row["role"], row["expected_direction"])
        grouped.setdefault(key, []).append(float(row["value"]))
    summary_rows = []
    for (metric, contrast, role, expected), group_values in sorted(grouped.items()):
        mean = statistics.fmean(group_values)
        std = sample_std(group_values)
        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                "contrast": contrast,
                "role": role,
                "expected_direction": expected,
                "seed_count": len(group_values),
                "mean_across_seeds": mean,
                "std_across_seeds": std,
                "report_value": f"{mean:.4f} +/- {std:.4f}",
            }
        )

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "control_proxy_effects_by_seed.csv", seed_rows)
    write_csv(output_dir / "control_proxy_summary.csv", summary_rows)
    report = {
        "model": model,
        "seeds": seeds,
        "contrasts": CONTRASTS,
        "primary_proxy_summary": [
            row for row in summary_rows if row["role"] != "descriptive_only"
        ],
        "interpretation_boundary": (
            "These are seed-paired music-theory and performance proxies. They do "
            "not establish perceived emotion, causal disentanglement, or superiority."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "control_proxy_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        f"# {model} control-proxy summary",
        "",
        "| Role | Metric | Contrast | Mean +/- SD across seeds | Expected |",
        "|---|---|---|---:|---|",
    ]
    for row in report["primary_proxy_summary"]:
        markdown.append(
            f"| {row['role']} | {row['metric']} | {row['contrast']} | "
            f"{row['report_value']} | {row['expected_direction']} |"
        )
    markdown.extend(["", report["interpretation_boundary"]])
    (output_dir / "control_proxy_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
