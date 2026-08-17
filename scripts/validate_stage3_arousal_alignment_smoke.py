#!/usr/bin/env python3
"""Validate the Stage 3 counterfactual arousal-alignment smoke run."""

import argparse
import json
import math
import re
from pathlib import Path

import yaml


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def validate_run(run_root, expected_steps=100):
    run_root = Path(run_root).resolve()
    train_root = run_root / "checkpoints/stage3_arousal_alignment_smoke"
    log_path = run_root / "logs/training.log"
    required = [
        log_path,
        run_root / "migrated_checkpoint_integrity.json",
        run_root / "smoke_checkpoint_integrity.json",
        train_root / "params/smoke_params.pt",
        train_root / "optim/smoke_optim.pt",
        train_root / "config.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("missing smoke artifacts: {}".format(missing))

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    steps = [int(value) for value in re.findall(r"\bstep = (\d+)", log_text)]
    maximum_step = max(steps, default=0)
    if maximum_step != expected_steps:
        raise ValueError("expected exactly {} steps, found {}".format(expected_steps, maximum_step))
    if "[arousal-alignment] enabled" not in log_text:
        raise ValueError("training log does not confirm arousal alignment")
    total_losses = [float(value) for value in re.findall(r"\* loss = ([-+0-9.eE]+)", log_text)]
    alignment_losses = [
        float(value) for value in re.findall(r"alignment_loss = ([-+0-9.eE]+)", log_text)
    ]
    if not total_losses or not all(math.isfinite(value) for value in total_losses):
        raise ValueError("training log has no finite total losses")
    if not alignment_losses or not all(
        math.isfinite(value) and value >= 0 for value in alignment_losses
    ):
        raise ValueError("training log has no finite non-negative alignment losses")

    migrated = read_json(run_root / "migrated_checkpoint_integrity.json")
    final = read_json(run_root / "smoke_checkpoint_integrity.json")
    if not migrated.get("passed") or not final.get("passed"):
        raise ValueError("checkpoint integrity gate failed")

    config = yaml.safe_load((train_root / "config.yaml").read_text("utf-8"))
    alignment = config.get("arousal_alignment", {})
    if not alignment.get("enabled") or float(alignment.get("weight", 0)) <= 0:
        raise ValueError("saved config does not enable weighted arousal alignment")

    return {
        "schema": "moonmuse-stage3-arousal-alignment-smoke-validation-v1",
        "passed": True,
        "run_root": str(run_root),
        "expected_steps": expected_steps,
        "observed_max_step": maximum_step,
        "optimizer_updates": (maximum_step + 1) // 2,
        "alignment_weight": float(alignment["weight"]),
        "observed_total_losses": len(total_losses),
        "observed_alignment_losses": len(alignment_losses),
        "last_total_loss": total_losses[-1],
        "last_alignment_loss": alignment_losses[-1],
        "migrated_checkpoint_passed": True,
        "final_checkpoint_passed": True,
        "evidence_boundary": "runnability_only_not_music_quality",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    report = validate_run(args.run_root, args.expected_steps)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[stage3-arousal-alignment-smoke] PASS", output)


if __name__ == "__main__":
    main()
