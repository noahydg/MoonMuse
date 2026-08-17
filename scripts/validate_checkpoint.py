#!/usr/bin/env python3
"""Validate that a training checkpoint is loadable, finite, and structurally complete."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path):
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping in {path}, got {type(value).__name__}")
    return value


def comparable_tensors(mapping):
    return {
        key: value
        for key, value in mapping.items()
        if torch.is_tensor(value) and "feature_map.omega" not in key
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params", type=Path)
    parser.add_argument("--optimizer", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--min-numel", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "params": str(args.params.resolve()),
        "optimizer": str(args.optimizer.resolve()) if args.optimizer else None,
        "reference": str(args.reference.resolve()) if args.reference else None,
        "min_numel": args.min_numel,
        "passed": False,
        "errors": [],
    }
    try:
        params = load_mapping(args.params)
        tensors = comparable_tensors(params)
        tensor_numel = sum(value.numel() for value in tensors.values())
        nonfinite_keys = sorted(
            key
            for key, value in tensors.items()
            if (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        )
        report.update(
            {
                "params_size_bytes": args.params.stat().st_size,
                "params_sha256": sha256(args.params),
                "tensor_key_count": len(tensors),
                "tensor_numel": tensor_numel,
                "nonfinite_keys": nonfinite_keys,
            }
        )
        if tensor_numel < args.min_numel:
            report["errors"].append(
                f"tensor_numel {tensor_numel} is below {args.min_numel}"
            )
        if nonfinite_keys:
            report["errors"].append("checkpoint contains non-finite tensors")

        if args.optimizer:
            optimizer = load_mapping(args.optimizer)
            optimizer_valid = "state" in optimizer and "param_groups" in optimizer
            report.update(
                {
                    "optimizer_size_bytes": args.optimizer.stat().st_size,
                    "optimizer_sha256": sha256(args.optimizer),
                    "optimizer_has_state_and_param_groups": optimizer_valid,
                }
            )
            if not optimizer_valid:
                report["errors"].append("optimizer state is missing required keys")

        if args.reference:
            reference = comparable_tensors(load_mapping(args.reference))
            missing = sorted(set(reference) - set(tensors))
            unexpected = sorted(set(tensors) - set(reference))
            mismatched = sorted(
                key
                for key in set(reference) & set(tensors)
                if reference[key].shape != tensors[key].shape
            )
            report["reference_compatibility"] = {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "mismatched_shapes": mismatched,
            }
            if missing or unexpected or mismatched:
                report["errors"].append("checkpoint is incompatible with reference")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")

    report["passed"] = not report["errors"]
    output = args.output or args.params.with_name("checkpoint_integrity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    print(f"CHECKPOINT_INTEGRITY_PASS output={output}")


if __name__ == "__main__":
    main()
