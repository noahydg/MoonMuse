#!/usr/bin/env python3
"""Validate one formal Stage 3 attribute-objective fine-tuning run."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def parse_manifest(path):
    return {
        key.strip(): value.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def validate_run(
    run_dir,
    seed,
    require_status=True,
    *,
    expected_protocol="stage3-performance-attribute-only-objective-formal-v1",
    expected_alignment_weight=0.20,
    expected_attribute_weights=None,
    checkpoint_dir_name="stage3_attribute_objective_finetune",
    status_prefix="STAGE3_ATTRIBUTE_OBJECTIVE_FORMAL_PASS",
    schema="moonmuse-stage3-attribute-objective-formal-validation-v1",
):
    run_dir = Path(run_dir).resolve()
    manifest = parse_manifest(run_dir / "training_manifest.txt")
    expected = {
        "protocol": expected_protocol,
        "seed": str(seed),
        "conditioning": "factorized_va",
        "optimizer_transfer": "reset_from_v63_alignment_checkpoint",
        "selection": "lowest_total_validation_objective",
        "reconstruction_objective": "performance_attributes_only",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("manifest {} is {!r}, expected {!r}".format(key, manifest.get(key), value))
    try:
        manifest_alignment_weight = float(manifest["alignment_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest alignment_weight is missing or invalid") from exc
    if manifest_alignment_weight != float(expected_alignment_weight):
        raise ValueError(
            "manifest alignment_weight is {!r}, expected {!r}".format(
                manifest_alignment_weight, float(expected_alignment_weight)
            )
        )
    if require_status:
        status = (run_dir / "status.txt").read_text(encoding="utf-8").strip()
        if not status.startswith("{} seed={} ".format(status_prefix, seed)):
            raise ValueError("formal completion marker is missing")

    source_gate = load_json(run_dir / "source_checkpoint_integrity.json")
    final_gate = load_json(run_dir / "formal_checkpoint_integrity.json")
    if not source_gate.get("passed") or not final_gate.get("passed"):
        raise ValueError("checkpoint integrity gate failed")
    compatibility = final_gate.get("reference_compatibility")
    if not isinstance(compatibility, dict) or any(
        compatibility.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_shapes")
    ):
        raise ValueError("formal checkpoint is not compatible with its source")

    checkpoint_root = run_dir / "checkpoints" / checkpoint_dir_name
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
    objective = config.get("reconstruction_objective", {})
    if objective.get("mode") != "performance_attributes_only" or set(
        objective.get("attributes", [])
    ) != {"tempo", "duration", "velocity"}:
        raise ValueError("saved config does not match the attribute-only objective")
    alignment = config.get("arousal_alignment", {})
    if not alignment.get("enabled") or float(alignment.get("weight", 0)) != float(
        expected_alignment_weight
    ):
        raise ValueError("saved config does not retain weighted arousal alignment")
    if expected_attribute_weights is not None:
        actual_attribute_weights = {
            name: float(alignment.get("attributes", {}).get(name, {}).get("weight", float("nan")))
            for name in expected_attribute_weights
        }
        if actual_attribute_weights != {
            name: float(value) for name, value in expected_attribute_weights.items()
        }:
            raise ValueError(
                "saved config attribute coefficients are {!r}, expected {!r}".format(
                    actual_attribute_weights, expected_attribute_weights
                )
            )

    valloss = (checkpoint_root / "valloss.txt").read_text(encoding="utf-8", errors="replace")
    epochs = [int(value) for value in re.findall(r"ep(\d+)\s+\|", valloss)]
    total = [float(value) for value in re.findall(r"valloss:\s*([-+0-9.eE]+)", valloss)]
    recon = [float(value) for value in re.findall(r"val_recons:\s*([-+0-9.eE]+)", valloss)]
    alignment_losses = [float(value) for value in re.findall(r"val_alignment:\s*([-+0-9.eE]+)", valloss)]
    if not epochs or not total or not recon or not alignment_losses:
        raise ValueError("validation log lacks required loss series")
    if not all(math.isfinite(value) for value in total + recon + alignment_losses):
        raise ValueError("validation log contains non-finite losses")
    return {
        "schema": schema,
        "passed": True,
        "seed": seed,
        "completed_epochs": max(epochs),
        "best_total_validation_objective": min(total),
        "best_reconstruction_validation_loss": min(recon),
        "best_alignment_validation_loss": min(alignment_losses),
        "params_sha256": final_gate["params_sha256"],
        "optimizer_sha256": final_gate["optimizer_sha256"],
        "configured_alignment_weight": float(alignment["weight"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--allow-missing-status", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_run(args.run_dir, args.seed, not args.allow_missing_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
