#!/usr/bin/env python3
"""Create compact manuscript-facing tables from the frozen v82 joined ratings."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy import stats


SYSTEMS = ("moonmuse_v70", "emo_disentangler")
METRICS = {
    "arousal_accuracy": "arousal_correct",
    "valence_accuracy": "valence_correct",
    "four_quadrant_accuracy": "quadrant_correct",
    "quality": "quality",
    "coherence": "coherence",
    "any_defect_rate": "has_defect",
}
DEFECTS = (
    "repetition",
    "overlap_noise",
    "abrupt_ending",
    "long_pause",
    "overdense_fatiguing",
    "incoherent",
    "other",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def ci95(values: list[float]) -> tuple[float, float, float]:
    center = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    half_width = stats.t.ppf(0.975, len(values) - 1) * standard_error
    return center, center - half_width, center + half_width


def paired_differences(rows: list[dict], unit_field: str, metric_field: str) -> list[float]:
    values = []
    for unit in sorted({row[unit_field] for row in rows}):
        unit_rows = [row for row in rows if row[unit_field] == unit]
        by_system = {
            system: [row for row in unit_rows if row["system_private"] == system]
            for system in SYSTEMS
        }
        if any(not by_system[system] for system in SYSTEMS):
            raise ValueError(f"{unit_field}={unit} does not contain both systems")
        values.append(
            mean(by_system["moonmuse_v70"], metric_field)
            - mean(by_system["emo_disentangler"], metric_field)
        )
    return values


def summarize(input_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = read_csv(input_path)
    if not rows:
        raise ValueError("joined-ratings input is empty")
    output_dir.mkdir(parents=True)

    overall_rows = []
    for metric_name, metric_field in METRICS.items():
        system_means = {
            system: mean(
                [row for row in rows if row["system_private"] == system], metric_field
            )
            for system in SYSTEMS
        }
        for unit, unit_field in (
            ("listener", "participant_code"),
            ("stimulus_slot", "slot_id_private"),
        ):
            differences = paired_differences(rows, unit_field, metric_field)
            difference, lower, upper = ci95(differences)
            overall_rows.append(
                {
                    "metric": metric_name,
                    "moonmuse_mean": system_means["moonmuse_v70"],
                    "baseline_mean": system_means["emo_disentangler"],
                    "moonmuse_minus_baseline": difference,
                    "analysis_unit": unit,
                    "n_units": len(differences),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    write_csv(output_dir / "overall_paired_effects.csv", overall_rows)

    quadrant_rows = []
    for quadrant in ("Q1", "Q2", "Q3", "Q4"):
        quadrant_data = [row for row in rows if row["quadrant_private"] == quadrant]
        for metric_name, metric_field in METRICS.items():
            candidate = mean(
                [row for row in quadrant_data if row["system_private"] == "moonmuse_v70"],
                metric_field,
            )
            baseline = mean(
                [row for row in quadrant_data if row["system_private"] == "emo_disentangler"],
                metric_field,
            )
            quadrant_rows.append(
                {
                    "quadrant": quadrant,
                    "metric": metric_name,
                    "moonmuse_mean": candidate,
                    "baseline_mean": baseline,
                    "moonmuse_minus_baseline": candidate - baseline,
                }
            )
    write_csv(output_dir / "quadrant_descriptive_effects.csv", quadrant_rows)

    defect_rows = []
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system_private"] == system]
        for defect in DEFECTS:
            count = sum(
                defect in ast.literal_eval(row["defects"])
                for row in system_rows
            )
            defect_rows.append(
                {
                    "system": system,
                    "defect": defect,
                    "count": count,
                    "ratings": len(system_rows),
                    "rate": count / len(system_rows),
                }
            )
    write_csv(output_dir / "defect_rates.csv", defect_rows)

    rating_distribution = []
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system_private"] == system]
        for metric in ("quality", "coherence"):
            counts = Counter(int(row[metric]) for row in system_rows)
            for score in range(1, 6):
                rating_distribution.append(
                    {
                        "system": system,
                        "metric": metric,
                        "score": score,
                        "count": counts[score],
                        "proportion": counts[score] / len(system_rows),
                    }
                )
    write_csv(output_dir / "rating_distributions.csv", rating_distribution)

    summary = {
        "participants": len({row["participant_code"] for row in rows}),
        "participants_by_form": dict(Counter(row["form"] for row in rows)),
        "stimulus_slots": len({row["slot_id_private"] for row in rows}),
        "ratings": len(rows),
        "overall_paired_effects": overall_rows,
    }
    (output_dir / "manuscript_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(summarize(args.input, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
