#!/usr/bin/env python3
"""Validate the bounded Stage 2/3 PAD-loss correction smoke."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_training_log(path: Path, expected_steps: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    steps = [int(value) for value in re.findall(r"\bstep = (\d+)", text)]
    maximum_step = max(steps, default=0)
    losses = [float(value) for value in re.findall(r"\* loss = ([-+0-9.eE]+)", text)]
    if maximum_step != expected_steps:
        raise ValueError(
            f"{path} expected exactly {expected_steps} steps, found {maximum_step}"
        )
    if f"[smoke] reached {expected_steps} steps; checkpoint saved" not in text:
        raise ValueError(f"{path} is missing the smoke checkpoint marker")
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError(f"{path} has no finite logged loss")
    return {
        "observed_max_step": maximum_step,
        "last_logged_loss": losses[-1],
    }


def validate_run(run_root: Path, expected_steps: int = 100) -> dict:
    run_root = run_root.resolve()
    required = [
        run_root / "pad_loss_tensor_validation.json",
        run_root / "stage2_source_checkpoint_integrity.json",
        run_root / "stage2_smoke_checkpoint_integrity.json",
        run_root / "stage3_source_checkpoint_integrity.json",
        run_root / "stage3_smoke_checkpoint_integrity.json",
        run_root / "logs/stage2_training.log",
        run_root / "logs/stage3_training.log",
        run_root / "checkpoints/stage2/params/smoke_params.pt",
        run_root / "checkpoints/stage2/optim/smoke_optim.pt",
        run_root / "checkpoints/stage3/params/smoke_params.pt",
        run_root / "checkpoints/stage3/optim/smoke_optim.pt",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing smoke artifacts: {missing}")
    tensor_report = read_json(run_root / "pad_loss_tensor_validation.json")
    integrity_names = (
        "stage2_source_checkpoint_integrity.json",
        "stage2_smoke_checkpoint_integrity.json",
        "stage3_source_checkpoint_integrity.json",
        "stage3_smoke_checkpoint_integrity.json",
    )
    integrity = {name: read_json(run_root / name) for name in integrity_names}
    if not tensor_report.get("passed"):
        raise ValueError("tensor PAD-loss validation failed")
    if not all(report.get("passed") for report in integrity.values()):
        raise ValueError("checkpoint integrity validation failed")
    stage2 = validate_training_log(
        run_root / "logs/stage2_training.log", expected_steps
    )
    stage3 = validate_training_log(
        run_root / "logs/stage3_training.log", expected_steps
    )
    return {
        "schema": "moonmuse-training-pad-loss-smoke-validation-v1",
        "passed": True,
        "expected_steps_per_stage": expected_steps,
        "tensor_pad_loss_validation_passed": True,
        "stage2": stage2,
        "stage3": stage3,
        "checkpoint_integrity_pass_count": len(integrity),
        "evidence_boundary": "objective_runnability_only_not_music_quality",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_run(args.run_root, args.expected_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"TRAINING_PAD_LOSS_SMOKE_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
