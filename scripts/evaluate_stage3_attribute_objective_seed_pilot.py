#!/usr/bin/env python3
"""Evaluate the prospective single-seed v67 generation pilot."""

import argparse
import json
from pathlib import Path


def metric(report, name):
    rows = [row for row in report["effect_summary"] if row["metric"] == name]
    if len(rows) != 1:
        raise ValueError("expected one {} row".format(name))
    return rows[0]


def evaluate(skeleton, objective):
    velocity = metric(objective, "mean_velocity")
    tempo = metric(objective, "time_weighted_tempo_bpm")
    criteria = {
        "execution_passed": objective["execution"]["all_critical_stages_passed"],
        "midi_8_of_8_valid": objective["execution"]["all_midi_valid"] and objective["execution"]["valid_midi_count"] == 8,
        "skeleton_8_of_8_exact": skeleton["passed"] and skeleton["counter_equal_count"] == 8 and skeleton["total_extra_note_group_count"] == 0 and skeleton["total_missing_note_group_count"] == 0,
        "velocity_direction_at_least_3_of_4": velocity["pair_count"] == 4 and velocity["expected_win_count"] >= 3,
        "tempo_direction_at_least_3_of_4": tempo["pair_count"] == 4 and tempo["expected_win_count"] >= 3,
        "strict_quality_at_least_3_of_8": objective["quality"]["individual_file_count"] == 8 and objective["quality"]["individual_quality_pass_count"] >= 3,
    }
    passed = all(criteria.values())
    return {
        "schema": "moonmuse-stage3-attribute-objective-seed-pilot-gate-v1",
        "passed": passed,
        "status": "stage3_attribute_objective_seed_pilot_pass" if passed else "stage3_attribute_objective_seed_pilot_fail_preserved",
        "criteria": criteria,
        "observed": {
            "valid_midi_count": objective["execution"]["valid_midi_count"],
            "exact_skeleton_count": skeleton["counter_equal_count"],
            "velocity_expected_win_count": velocity["expected_win_count"],
            "tempo_expected_win_count": tempo["expected_win_count"],
            "strict_quality_pass_count": objective["quality"]["individual_quality_pass_count"],
        },
        "decision": "expand_to_seeds_43_44" if passed else "stop_and_preserve_negative_pilot",
        "interpretation_boundary": "A passing seed-42 pilot authorizes replication only; it is not a quality or controllability claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skeleton", type=Path)
    parser.add_argument("objective", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.skeleton.read_text()), json.loads(args.objective.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
