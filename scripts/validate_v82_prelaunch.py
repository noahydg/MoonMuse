#!/usr/bin/env python3
"""Validate all automated v82 prelaunch gates and expose the human blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PROTOCOL_SHA256 = (
    "d19374c9718cd0ddbfb1addc253c23a587396973c22faab14d0c6811448b8687"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_manifest(root: Path, manifest: Path) -> dict:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch: {path}")
        rows.append(relative)
    return {"files": len(rows), "all_passed": True}


def evaluate(
    study_manifest: dict,
    audio_report: dict,
    questionnaire_validation: dict,
    pass_analysis: dict,
    fail_analysis: dict,
) -> dict:
    automated = {
        "population_is_24_slots_48_stimuli": (
            study_manifest.get("selection", {}).get("selected_slots") == 24
            and study_manifest.get("selection", {}).get("selected_stimuli") == 48
        ),
        "population_selection_is_outcome_blind": all(
            study_manifest.get("selection", {}).get(key) is False
            for key in (
                "selection_uses_classifier",
                "selection_uses_v74",
                "selection_uses_human_judgment",
            )
        ),
        "audio_uses_one_global_gain": (
            audio_report.get("settings", {}).get("per_clip_loudness_normalization") is False
            and isinstance(audio_report.get("settings", {}).get("global_gain_db"), (int, float))
        ),
        "audio_peak_and_relative_loudness_pass": all(
            audio_report.get("verification", {}).get(key) is True
            for key in (
                "all_true_peaks_pass",
                "all_relative_loudness_offsets_preserved_within_0_35_db",
                "all_files_hash_verified",
            )
        ),
        "questionnaires_static_validation_pass": (
            questionnaire_validation.get("status") == "prelaunch_static_validation_passed"
            and questionnaire_validation.get("formalStudyData") is False
            and questionnaire_validation.get("cross_form_stimulus_overlap") == 0
        ),
        "synthetic_positive_control_passes": (
            pass_analysis.get("prelaunchSyntheticData") is True
            and pass_analysis.get("confirmatory", {}).get(
                "overall_bounded_baseline_outperformance_claim_passed"
            ) is True
        ),
        "synthetic_negative_control_fails": (
            fail_analysis.get("prelaunchSyntheticData") is True
            and fail_analysis.get("confirmatory", {}).get(
                "overall_bounded_baseline_outperformance_claim_passed"
            ) is False
        ),
    }
    return {
        "automated_checks": automated,
        "automated_prelaunch_passed": all(automated.values()),
        "manual_browser_qa_passed": False,
        "ethics_approval_or_exemption_documented": False,
        "formal_launch_ready": False,
        "human_result_available": False,
        "baseline_outperformance_claim_supported": False,
    }


def validate(args: argparse.Namespace) -> dict:
    protocol_hash = sha256_file(args.protocol)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash mismatch")
    study_manifest = json.loads(
        (args.study_dir / "study_manifest.json").read_text(encoding="utf-8")
    )
    audio_report = json.loads(
        (args.study_dir / "audio_prelaunch" / "render_report.json").read_text(
            encoding="utf-8"
        )
    )
    questionnaire_validation = json.loads(
        (
            args.study_dir
            / "questionnaires_prelaunch"
            / "static_validation.json"
        ).read_text(encoding="utf-8")
    )
    pass_analysis = json.loads(
        (args.study_dir / "prelaunch_qa" / "synthetic_pass_analysis" / "analysis.json").read_text(encoding="utf-8")
    )
    fail_analysis = json.loads(
        (args.study_dir / "prelaunch_qa" / "synthetic_fail_analysis" / "analysis.json").read_text(encoding="utf-8")
    )
    checks = evaluate(
        study_manifest,
        audio_report,
        questionnaire_validation,
        pass_analysis,
        fail_analysis,
    )
    if not checks["automated_prelaunch_passed"]:
        raise ValueError(f"automated v82 prelaunch gate failed: {checks}")
    if list(args.study_dir.glob("questionnaires_prelaunch/*-launch.html")):
        raise ValueError("formal launch HTML exists before the ethics gate")
    required_docs = (
        "ethics-record-template.json",
        "advisor-ethics-review-pack-draft.md",
        "data-management-prelaunch.md",
        "prelaunch-manual-checklist.md",
    )
    missing = [name for name in required_docs if not (args.study_dir / name).is_file()]
    if missing:
        raise ValueError(f"prelaunch documentation is missing: {missing}")
    study_checksums = verify_checksum_manifest(
        args.study_dir, args.study_dir / "checksums.sha256"
    )
    audio_checksums = verify_checksum_manifest(
        args.study_dir / "audio_prelaunch",
        args.study_dir / "audio_prelaunch" / "checksums.sha256",
    )
    report = {
        "schema": "moonmuse-v82-prelaunch-validation-v1",
        "status": "automated_prelaunch_complete_manual_and_ethics_required",
        "formalStudyData": False,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_hash,
        **checks,
        "checksum_gates": {
            "study_population": study_checksums,
            "audio": audio_checksums,
        },
        "questionnaire_forms": {
            form: questionnaire_validation["forms"][form]["html"]
            for form in ("F1", "F2")
        },
        "remaining_human_gates": [
            "Researcher manually opens and checks both prelaunch HTML forms.",
            "Supervisor or responsible institution documents ethics approval or exemption and approves consent wording.",
            "Only then build and validate launch-mode HTML before recruiting participants.",
        ],
        "claim_boundary": (
            "No participant data exist. Automated v81 superiority is not a human "
            "baseline-outperformance result, and v82 prelaunch readiness supports no "
            "paper result by itself."
        ),
        "artifact_hashes": {
            "study_manifest": sha256_file(args.study_dir / "study_manifest.json"),
            "audio_render_report": sha256_file(
                args.study_dir / "audio_prelaunch" / "render_report.json"
            ),
            "questionnaire_static_validation": sha256_file(
                args.study_dir
                / "questionnaires_prelaunch"
                / "static_validation.json"
            ),
            "synthetic_pass_analysis": sha256_file(
                args.study_dir / "prelaunch_qa" / "synthetic_pass_analysis" / "analysis.json"
            ),
            "synthetic_fail_analysis": sha256_file(
                args.study_dir / "prelaunch_qa" / "synthetic_fail_analysis" / "analysis.json"
            ),
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    report = validate(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
