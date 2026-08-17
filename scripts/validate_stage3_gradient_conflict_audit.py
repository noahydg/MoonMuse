#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def validate(report_path, expected_batches):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    errors = []
    if report.get("schema") != "moonmuse-stage3-gradient-conflict-audit-v1":
        errors.append("unexpected schema")
    if report.get("training_performed") is not False:
        errors.append("audit must declare training_performed=false")
    if report.get("checkpoint_modified") is not False:
        errors.append("audit must declare checkpoint_modified=false")
    if report.get("gradient_summary", {}).get("batch_count") != expected_batches:
        errors.append("unexpected batch count")
    required_scopes = {"full_model", "shared_backbone", "input_conditioning", "output_head"}
    scopes = report.get("gradient_summary", {}).get("scopes", {})
    if set(scopes) != required_scopes:
        errors.append("missing or extra gradient scopes")
    for scope_name in required_scopes & set(scopes):
        scope = scopes[scope_name]
        pair = scope.get("cosines", {}).get("total_tempo__total_velocity", {})
        if pair.get("count") != expected_batches:
            errors.append("{} primary cosine count mismatch".format(scope_name))
        for key in ("median", "negative_fraction", "q25", "q75"):
            value = pair.get(key)
            if value is None or not math.isfinite(float(value)):
                errors.append("{} primary cosine {} is not finite".format(scope_name, key))
        for objective in ("total_tempo", "total_velocity", "total_duration"):
            norm = scope.get("norms", {}).get(objective, {})
            if norm.get("count") != expected_batches or not math.isfinite(float(norm.get("median", float("nan")))):
                errors.append("{} {} norm is incomplete".format(scope_name, objective))
    if report.get("diagnostic_route", {}).get("label") not in {
        "shared_gradient_conflict_supported",
        "mixed_shared_gradient_conflict",
        "tempo_gradient_underpowered",
        "conflict_and_underpower_not_supported",
    }:
        errors.append("unexpected diagnostic route")
    return {"passed": not errors, "errors": errors, "report": str(report_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--expected-batches", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate(args.report, args.expected_batches)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
