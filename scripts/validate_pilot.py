#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


PHASES = (
    "stage1_pretrain",
    "stage1_finetune",
    "stage2_pretrain",
    "stage2_finetune",
    "stage3_pretrain",
    "stage3_finetune",
)
VALLOSS_PATTERN = re.compile(
    r"ep(?P<epoch>\d+).*?loss:\s*(?P<train>[-+0-9.eE]+).*?"
    r"valloss:\s*(?P<valid>[-+0-9.eE]+)"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_valloss(path):
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VALLOSS_PATTERN.search(line)
        if not match:
            continue
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train")),
                "valid_loss": float(match.group("valid")),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Validate a bounded MoonMuse training pilot")
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-checkpoint-integrity", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(PHASES),
        help="checkpoint phases to validate (default: the six full-model phases)",
    )
    args = parser.parse_args()

    if len(args.phases) != len(set(args.phases)):
        parser.error("--phases must not contain duplicates")
    invalid_phases = [
        phase
        for phase in args.phases
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", phase)
    ]
    if invalid_phases:
        parser.error(f"invalid phase names: {', '.join(invalid_phases)}")

    pilot_dir = args.pilot_dir.resolve()
    output = args.output or pilot_dir / "pilot_validation.json"
    summary = {
        "pilot_dir": str(pilot_dir),
        "requested_phases": args.phases,
        "phases": {},
        "passed": True,
    }

    for phase in args.phases:
        phase_dir = pilot_dir / "checkpoints" / phase
        params = phase_dir / "params" / "best_params.pt"
        optim = phase_dir / "optim" / "best_optim.pt"
        integrity_path = phase_dir / "checkpoint_integrity.json"
        integrity = None
        if integrity_path.is_file():
            try:
                integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                integrity = {"passed": False, "errors": ["invalid integrity report"]}
        records = parse_valloss(phase_dir / "valloss.txt")
        finite = bool(records) and all(
            math.isfinite(item["train_loss"]) and math.isfinite(item["valid_loss"])
            for item in records
        )
        integrity_passed = bool(integrity and integrity.get("passed"))
        passed = (
            params.is_file()
            and optim.is_file()
            and finite
            and (integrity_passed or not args.require_checkpoint_integrity)
        )
        summary["phases"][phase] = {
            "passed": passed,
            "validation_epochs": len(records),
            "first_valid_loss": records[0]["valid_loss"] if records else None,
            "final_valid_loss": records[-1]["valid_loss"] if records else None,
            "best_valid_loss": min((item["valid_loss"] for item in records), default=None),
            "best_params_sha256": sha256(params) if params.is_file() else None,
            "best_optim_sha256": sha256(optim) if optim.is_file() else None,
            "checkpoint_integrity_required": args.require_checkpoint_integrity,
            "checkpoint_integrity_passed": integrity_passed,
            "checkpoint_integrity_report": str(integrity_path) if integrity else None,
        }
        summary["passed"] = summary["passed"] and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {phase}: validation_epochs={len(records)}, "
            f"first={summary['phases'][phase]['first_valid_loss']}, "
            f"best={summary['phases'][phase]['best_valid_loss']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit(1)
    print(f"PILOT_TRAINING_VALIDATED output={output}")


if __name__ == "__main__":
    main()
