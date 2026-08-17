#!/usr/bin/env python3
"""Validate a completed Stage 3 arousal-alignment formal fine-tuning run."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import yaml


def load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read valid JSON: {}".format(path)) from exc
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def parse_manifest(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("cannot read manifest: {}".format(path)) from exc
    return {
        key.strip(): value.strip()
        for line in lines if "=" in line
        for key, value in [line.split("=", 1)]
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(run_dir, seed, require_status=True):
    run_dir = Path(run_dir).resolve()
    manifest = parse_manifest(run_dir / "training_manifest.txt")
    expected = {
        "protocol": "stage3-arousal-alignment-formal-v1",
        "seed": str(seed),
        "conditioning": "factorized_va",
        "optimizer_transfer": "reset_after_checkpoint_migration",
        "selection": "lowest_total_validation_objective",
        "objective": "token_cross_entropy_plus_paired_counterfactual_directional_hinge_v1",
        "alignment_weight": "0.20",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                "manifest {} is {!r}, expected {!r}".format(key, manifest.get(key), value)
            )

    if require_status:
        status_path = run_dir / "status.txt"
        try:
            status = status_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError("cannot read status: {}".format(status_path)) from exc
        if not status.startswith("STAGE3_AROUSAL_ALIGNMENT_FORMAL_PASS seed={} ".format(seed)):
            raise ValueError("formal completion marker is missing: {}".format(status_path))

    audit = load_json(run_dir / "arousal_supervision_audit.json")
    if not all(audit.get("gates", {}).values()):
        raise ValueError("arousal supervision audit gates did not all pass")
    if audit.get("recommended_intervention") != "explicit_counterfactual_arousal_alignment":
        raise ValueError("audit does not recommend counterfactual arousal alignment")

    migrated_gate = load_json(run_dir / "migrated_checkpoint_integrity.json")
    final_gate = load_json(run_dir / "formal_checkpoint_integrity.json")
    for name, gate in (("migrated", migrated_gate), ("formal", final_gate)):
        if not gate.get("passed") or gate.get("nonfinite_keys"):
            raise ValueError("{} checkpoint integrity failed".format(name))
    compatibility = final_gate.get("reference_compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("formal checkpoint has no reference compatibility report")
    for key in ("missing_keys", "unexpected_keys", "mismatched_shapes"):
        if compatibility.get(key):
            raise ValueError("formal checkpoint compatibility has {}".format(key))

    checkpoint_root = run_dir / "checkpoints/stage3_arousal_alignment_finetune"
    params = checkpoint_root / "params/best_params.pt"
    optimizer = checkpoint_root / "optim/best_optim.pt"
    for path in (params, optimizer):
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise ValueError("checkpoint missing or implausibly small: {}".format(path))
    if final_gate.get("params_sha256") != sha256(params):
        raise ValueError("parameter checkpoint hash does not match gate")
    if final_gate.get("optimizer_sha256") != sha256(optimizer):
        raise ValueError("optimizer checkpoint hash does not match gate")

    config = yaml.safe_load((checkpoint_root / "config.yaml").read_text("utf-8"))
    alignment = config.get("arousal_alignment", {})
    if (
        not alignment.get("enabled")
        or alignment.get("objective") != "paired_counterfactual_directional_hinge_v1"
        or float(alignment.get("weight", 0)) != 0.20
    ):
        raise ValueError("saved config does not match declared alignment objective")

    valloss_path = checkpoint_root / "valloss.txt"
    try:
        valloss_text = valloss_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError("cannot read validation log: {}".format(valloss_path)) from exc
    epochs = [int(value) for value in re.findall(r"ep(\d+)\s+\|", valloss_text)]
    total_losses = [float(value) for value in re.findall(r"valloss:\s*([-+0-9.eE]+)", valloss_text)]
    recons_losses = [float(value) for value in re.findall(r"val_recons:\s*([-+0-9.eE]+)", valloss_text)]
    alignment_losses = [float(value) for value in re.findall(r"val_alignment:\s*([-+0-9.eE]+)", valloss_text)]
    if not epochs or not total_losses or not recons_losses or not alignment_losses:
        raise ValueError("validation log lacks total/reconstruction/alignment losses")
    all_losses = total_losses + recons_losses + alignment_losses
    if not all(math.isfinite(value) for value in all_losses):
        raise ValueError("validation log contains non-finite losses")
    if any(value < 0 for value in alignment_losses):
        raise ValueError("alignment loss cannot be negative")

    return {
        "schema": "moonmuse-stage3-arousal-alignment-formal-validation-v1",
        "passed": True,
        "run_dir": str(run_dir),
        "seed": seed,
        "completed_epochs": max(epochs),
        "best_total_validation_objective": min(total_losses),
        "best_reconstruction_validation_loss": min(recons_losses),
        "best_alignment_validation_loss": min(alignment_losses),
        "alignment_weight": 0.20,
        "params_sha256": final_gate["params_sha256"],
        "optimizer_sha256": final_gate["optimizer_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--allow-missing-status", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_run(
            args.run_dir, args.seed, require_status=not args.allow_missing_status
        )
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise SystemExit("STAGE3_AROUSAL_ALIGNMENT_FORMAL_VALIDATION_FAIL {}".format(exc)) from exc
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("STAGE3_AROUSAL_ALIGNMENT_FORMAL_VALIDATION_PASS run_dir={} seed={}".format(
        report["run_dir"], report["seed"]
    ))


if __name__ == "__main__":
    main()
