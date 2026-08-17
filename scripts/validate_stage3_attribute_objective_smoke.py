#!/usr/bin/env python3
"""Validate a Stage 3 performance-attribute-only 100-batch smoke run."""

import argparse
import json
import math
import re
from pathlib import Path

import yaml


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def validate_run(run_root, expected_steps=100):
    run_root = Path(run_root).resolve()
    train_root = run_root / "checkpoints/stage3_attribute_objective_smoke"
    log_path = run_root / "logs/training.log"
    required = [
        log_path,
        run_root / "source_checkpoint_integrity.json",
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
    if "[reconstruction-objective] mode=performance_attributes_only" not in log_text:
        raise ValueError("training log does not confirm the attribute-only objective")
    if "[arousal-alignment] enabled" not in log_text:
        raise ValueError("training log does not confirm arousal alignment")
    losses = [float(value) for value in re.findall(r"\* loss = ([-+0-9.eE]+)", log_text)]
    alignment = [float(value) for value in re.findall(r"alignment_loss = ([-+0-9.eE]+)", log_text)]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("training log has no finite total losses")
    if not alignment or not all(math.isfinite(value) and value >= 0 for value in alignment):
        raise ValueError("training log has no finite non-negative alignment losses")

    source = read_json(run_root / "source_checkpoint_integrity.json")
    final = read_json(run_root / "smoke_checkpoint_integrity.json")
    if not source.get("passed") or not final.get("passed"):
        raise ValueError("checkpoint integrity gate failed")
    config = yaml.safe_load((train_root / "config.yaml").read_text("utf-8"))
    objective = config.get("reconstruction_objective", {})
    attributes = objective.get("attributes", [])
    if objective.get("mode") != "performance_attributes_only":
        raise ValueError("saved config does not enable the attribute-only objective")
    if set(attributes) != {"tempo", "duration", "velocity"}:
        raise ValueError("saved config has the wrong performance attributes")

    return {
        "schema": "moonmuse-stage3-attribute-objective-smoke-validation-v1",
        "passed": True,
        "expected_steps": expected_steps,
        "observed_max_step": maximum_step,
        "optimizer_updates": (maximum_step + 1) // 2,
        "reconstruction_mode": objective["mode"],
        "attributes": attributes,
        "last_total_loss": losses[-1],
        "last_alignment_loss": alignment[-1],
        "source_checkpoint_passed": True,
        "final_checkpoint_passed": True,
        "evidence_boundary": "runnability_only_not_music_quality",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_run(args.run_root, args.expected_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[stage3-attribute-objective-smoke] PASS", args.output)


if __name__ == "__main__":
    main()
