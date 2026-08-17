#!/usr/bin/env python3
"""Apply the predeclared Gate C thresholds to a fixed-source analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_PAIRS = 12
EXPECTED_OUTPUTS = 24
EXPECTED_GROUPS = 6
SECONDARY_METRICS = (
    "note_density_per_second",
    "onset_density_per_second",
    "time_weighted_tempo_bpm",
)


def effect_by_metric(report: dict) -> dict[str, dict]:
    effects = report.get("effect_summary")
    if not isinstance(effects, list):
        raise ValueError("analysis has no effect_summary list")
    rows = {row.get("metric"): row for row in effects if isinstance(row, dict)}
    required = {"mean_velocity", *SECONDARY_METRICS}
    missing = sorted(required - rows.keys())
    if missing:
        raise ValueError(f"analysis is missing metrics: {missing}")
    return rows


def evaluate_report(report: dict) -> dict:
    if report.get("schema") != "moonmuse-stage3-arousal-counterfactual-analysis-v1":
        raise ValueError(f"unexpected analysis schema: {report.get('schema')!r}")

    execution = report.get("execution", {})
    quality = report.get("quality", {})
    effects = effect_by_metric(report)
    group_reports = quality.get("group_reports", [])

    execution_pass = (
        report.get("pair_count") == EXPECTED_PAIRS
        and report.get("expected_output_count") == EXPECTED_OUTPUTS
        and execution.get("all_critical_stages_passed") is True
        and execution.get("all_midi_valid") is True
        and execution.get("valid_midi_count") == EXPECTED_OUTPUTS
    )
    individual_quality_pass = (
        quality.get("individual_file_count") == EXPECTED_OUTPUTS
        and quality.get("individual_quality_pass_count", -1) >= 18
    )
    group_quality_pass = (
        isinstance(group_reports, list)
        and len(group_reports) == EXPECTED_GROUPS
        and all(
            row.get("analysed_count") == 4 and row.get("passing_count", -1) >= 3
            for row in group_reports
        )
    )

    velocity = effects["mean_velocity"]
    velocity_pass = (
        velocity.get("pair_count") == EXPECTED_PAIRS
        and velocity.get("expected_win_count", -1) >= 8
        and velocity.get("mean_high_minus_low", 0) > 0
    )
    secondary_details = {}
    for metric in SECONDARY_METRICS:
        row = effects[metric]
        secondary_details[metric] = {
            "passed": (
                row.get("pair_count") == EXPECTED_PAIRS
                and row.get("expected_win_count", -1) >= 7
                and row.get("mean_high_minus_low", 0) > 0
            ),
            "expected_win_count": row.get("expected_win_count"),
            "mean_high_minus_low": row.get("mean_high_minus_low"),
        }
    secondary_pass_count = sum(
        item["passed"] for item in secondary_details.values()
    )
    secondary_pass = secondary_pass_count >= 2

    gates = {
        "execution_and_24_valid_midi": {
            "passed": execution_pass,
            "valid_midi_count": execution.get("valid_midi_count"),
        },
        "individual_quality_at_least_18_of_24": {
            "passed": individual_quality_pass,
            "passing_count": quality.get("individual_quality_pass_count"),
        },
        "every_group_quality_at_least_3_of_4": {
            "passed": group_quality_pass,
            "group_count": len(group_reports) if isinstance(group_reports, list) else None,
        },
        "velocity_direction_at_least_8_of_12_and_positive_mean": {
            "passed": velocity_pass,
            "expected_win_count": velocity.get("expected_win_count"),
            "mean_high_minus_low": velocity.get("mean_high_minus_low"),
        },
        "two_of_three_secondary_arousal_proxies": {
            "passed": secondary_pass,
            "passing_metric_count": secondary_pass_count,
            "metrics": secondary_details,
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema": "moonmuse-stage3-factorized-gate-c-evaluation-v1",
        "passed": passed,
        "status": "gate_c_pass" if passed else "gate_c_fail_preserved",
        "gates": gates,
        "interpretation_boundary": (
            "A pass is a technical fixed-source acceptance result, not perceived-"
            "emotion, musical-quality, therapeutic-effect, or superiority evidence. "
            "A failure must be preserved and must not be repaired by replacing samples."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.analysis_json.read_text(encoding="utf-8"))
        result = evaluate_report(report)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"STAGE3_FACTORIZED_GATE_C_EVALUATION_ERROR {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "STAGE3_FACTORIZED_GATE_C_PASS"
        if result["passed"]
        else "STAGE3_FACTORIZED_GATE_C_FAIL_PRESERVED"
    )
    raise SystemExit(0 if result["passed"] else 3)


if __name__ == "__main__":
    main()
