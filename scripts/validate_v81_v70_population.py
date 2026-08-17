#!/usr/bin/env python3
"""Validate and freeze the 300-output v81 v70 candidate population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_quality_pilot_beat_normalized import SCHEMA as V74_SCHEMA
from scripts.validate_midi import inspect_midi
from scripts.validate_stage3_skeleton_lock import validate as validate_skeleton


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise ValueError(f"expected 300 source rows, found {len(rows)}")
    return rows


def expected_paths(root: Path, source: dict) -> tuple[Path, Path, Path]:
    seed = int(source["seed"])
    quadrant = source["quadrant"]
    source_stem = Path(source["source_name"]).stem
    group = root / f"seed{seed}" / quadrant
    event = group / "stage3_performance" / f"{source_stem}_{quadrant}_performance.txt"
    midi = group / "midi" / f"{source_stem}_{quadrant}_performance.mid"
    condition = group / "stage2_melody_conn" / source["source_name"]
    return condition, event, midi


def validate_population(args: argparse.Namespace) -> dict:
    root = args.run_root.resolve()
    source_rows = load_sources(args.source_manifest)
    expected = []
    missing = []
    for source in source_rows:
        condition, event, midi = expected_paths(root, source)
        for role, path in (("condition", condition), ("event", event), ("midi", midi)):
            if not path.is_file():
                missing.append({"role": role, "path": str(path)})
        expected.append((source, condition, event, midi))

    actual_events = sorted(root.glob("seed*/Q*/stage3_performance/*.txt"))
    actual_midis = sorted(root.glob("seed*/Q*/midi/*.mid"))
    expected_event_paths = {event for _, _, event, _ in expected}
    expected_midi_paths = {midi for _, _, _, midi in expected}
    unexpected_events = sorted(str(path) for path in set(actual_events) - expected_event_paths)
    unexpected_midis = sorted(str(path) for path in set(actual_midis) - expected_midi_paths)

    skeleton = None
    skeleton_error = None
    if not missing and len(actual_events) == 300:
        try:
            skeleton = validate_skeleton(root, 300)
        except Exception as exc:  # Preserve a readable failed gate instead of losing evidence.
            skeleton_error = f"{type(exc).__name__}: {exc}"

    midi_rows = []
    if not missing:
        for source, condition, event, midi in expected:
            inspection = inspect_midi(midi)
            midi_rows.append(
                {
                    "system": "moonmuse_v70",
                    "seed": int(source["seed"]),
                    "quadrant": source["quadrant"],
                    "sample_index": int(source["sample_index"]),
                    "source_frozen_path": source["frozen_path"],
                    "source_sha256": source["source_sha256"],
                    "condition_path": str(condition.relative_to(root)),
                    "condition_sha256": sha256_file(condition),
                    "event_path": str(event.relative_to(root)),
                    "event_sha256": sha256_file(event),
                    "midi_path": str(midi.relative_to(root)),
                    "midi_sha256": sha256_file(midi),
                    "midi_valid": inspection["valid"],
                    "note_count": inspection["notes"],
                    "max_tick": inspection["max_tick"],
                }
            )

    v74_reports = []
    v74_errors = []
    for seed in (42, 43, 44):
        for quadrant in ("Q1", "Q2", "Q3", "Q4"):
            path = root / f"seed{seed}" / quadrant / "quality_gate_v74.json"
            if not path.is_file():
                v74_errors.append(f"missing {path}")
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("schema") != V74_SCHEMA:
                v74_errors.append(f"unexpected schema {path}")
                continue
            if not report.get("execution_complete") or report.get("individual_file_count") != 25:
                v74_errors.append(f"incomplete {path}")
            v74_reports.append({"seed": seed, "quadrant": quadrant, "report": report})

    v74_by_seed = defaultdict(lambda: {"files": 0, "passes": 0})
    for item in v74_reports:
        seed = item["seed"]
        report = item["report"]
        v74_by_seed[seed]["files"] += int(report["individual_file_count"])
        v74_by_seed[seed]["passes"] += int(report["individual_quality_pass_count"])
    for values in v74_by_seed.values():
        values["rate"] = values["passes"] / values["files"] if values["files"] else 0.0
    v74_total_files = sum(item["report"]["individual_file_count"] for item in v74_reports)
    v74_total_passes = sum(item["report"]["individual_quality_pass_count"] for item in v74_reports)
    v74_rate = v74_total_passes / v74_total_files if v74_total_files else 0.0

    counts = Counter((row["seed"], row["quadrant"]) for row in midi_rows)
    execution_criteria = {
        "no_missing_expected_files": not missing,
        "no_unexpected_event_files": not unexpected_events,
        "no_unexpected_midi_files": not unexpected_midis,
        "event_count_300": len(actual_events) == 300,
        "midi_count_300": len(actual_midis) == 300,
        "balanced_25_per_seed_quadrant": all(
            counts[(seed, quadrant)] == 25
            for seed in (42, 43, 44)
            for quadrant in ("Q1", "Q2", "Q3", "Q4")
        ),
        "all_midi_valid": len(midi_rows) == 300 and all(row["midi_valid"] for row in midi_rows),
        "skeleton_300_of_300_exact": bool(
            skeleton
            and skeleton["passed"]
            and skeleton["counter_equal_count"] == 300
            and skeleton["total_extra_note_group_count"] == 0
            and skeleton["total_missing_note_group_count"] == 0
        ),
        "v74_reports_complete": len(v74_reports) == 12 and not v74_errors,
    }
    execution_passed = all(execution_criteria.values())
    technical_handoff_criteria = {
        "execution_passed": execution_passed,
        "v74_aggregate_at_least_70_percent": v74_rate >= 0.70,
        "v74_each_seed_at_least_65_percent": all(
            v74_by_seed[seed]["files"] == 100 and v74_by_seed[seed]["rate"] >= 0.65
            for seed in (42, 43, 44)
        ),
    }
    technical_handoff_passed = all(technical_handoff_criteria.values())

    if midi_rows:
        manifest_path = root / "population_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(midi_rows[0]))
            writer.writeheader()
            writer.writerows(midi_rows)
        manifest_hash = sha256_file(manifest_path)
    else:
        manifest_path = root / "population_manifest.csv"
        manifest_hash = None

    report = {
        "schema": "moonmuse-v81-v70-population-validation-v1",
        "formalStudyData": False,
        "status": (
            "execution_and_technical_handoff_pass"
            if technical_handoff_passed
            else "execution_pass_technical_handoff_fail"
            if execution_passed
            else "execution_fail_preserved"
        ),
        "run_root": str(root),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "execution_passed": execution_passed,
        "execution_criteria": execution_criteria,
        "technical_handoff_passed": technical_handoff_passed,
        "technical_handoff_criteria": technical_handoff_criteria,
        "observed": {
            "event_count": len(actual_events),
            "midi_count": len(actual_midis),
            "valid_midi_count": sum(row["midi_valid"] for row in midi_rows),
            "exact_skeleton_count": skeleton["counter_equal_count"] if skeleton else 0,
            "v74_pass_count": v74_total_passes,
            "v74_file_count": v74_total_files,
            "v74_pass_rate": v74_rate,
            "v74_by_seed": {str(seed): v74_by_seed[seed] for seed in (42, 43, 44)},
        },
        "missing": missing,
        "unexpected_events": unexpected_events,
        "unexpected_midis": unexpected_midis,
        "skeleton_error": skeleton_error,
        "v74_errors": v74_errors,
        "population_manifest": str(manifest_path),
        "population_manifest_sha256": manifest_hash,
        "interpretation_boundary": (
            "This gate establishes execution, exact Stage 2 skeleton preservation, "
            "and tick-normalized technical eligibility. It does not establish "
            "affective accuracy, listenability, or baseline superiority."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    final = validate_population(parse_args())
    print(json.dumps({key: final[key] for key in ("status", "execution_passed", "technical_handoff_passed", "observed")}, indent=2))
    raise SystemExit(0 if final["execution_passed"] else 3)
