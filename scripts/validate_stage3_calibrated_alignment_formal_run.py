#!/usr/bin/env python3
"""Validate one v69 calibrated-alignment formal fine-tuning run."""

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_stage3_attribute_objective_formal_run import validate_run


EXPECTED_ATTRIBUTE_WEIGHTS = {
    "tempo": 0.500,
    "duration": 0.875,
    "velocity": 0.875,
}


def validate(run_dir, seed, require_status=True):
    report = validate_run(
        run_dir,
        seed,
        require_status,
        expected_protocol="stage3-performance-calibrated-alignment-formal-v1",
        expected_alignment_weight=2.25,
        expected_attribute_weights=EXPECTED_ATTRIBUTE_WEIGHTS,
        checkpoint_dir_name="stage3_calibrated_alignment_finetune",
        status_prefix="STAGE3_CALIBRATED_ALIGNMENT_FORMAL_PASS",
        schema="moonmuse-stage3-calibrated-alignment-formal-validation-v1",
    )
    report["configured_attribute_coefficients"] = EXPECTED_ATTRIBUTE_WEIGHTS
    report["evidence_boundary"] = "training_and_checkpoint_integrity_only_not_music_quality"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--allow-missing-status", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.run_dir, args.seed, not args.allow_missing_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
