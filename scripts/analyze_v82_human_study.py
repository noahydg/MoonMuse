#!/usr/bin/env python3
"""Freeze v82 responses before unblinding and run the confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from scipy import stats


EXPORT_SCHEMA = "moonmuse-v82-listener-export-v1"
ANALYSIS_SCHEMA = "moonmuse-v82-human-study-analysis-v1"
FORMS = ("F1", "F2")
SYSTEMS = ("moonmuse_v70", "emo_disentangler")
VALENCE_CHOICES = {"positive", "negative", "uncertain"}
AROUSAL_CHOICES = {"high", "low", "uncertain"}
DEFECT_CHOICES = {
    "repetition",
    "overlap_noise",
    "abrupt_ending",
    "long_pause",
    "overdense_fatiguing",
    "incoherent",
    "other",
}
ALPHA_ONE_SIDED = 0.025
QUALITY_NI_MARGIN = -0.30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def public_form_ids(study_dir: Path) -> dict[str, set[str]]:
    result = {}
    for form in FORMS:
        rows = read_csv(study_dir / "forms_public" / f"{form}.csv")
        ids = {row["stimulus_id"] for row in rows}
        if len(rows) != 24 or len(ids) != 24:
            raise ValueError(f"public form {form} does not contain 24 unique stimuli")
        result[form] = ids
    if result["F1"] & result["F2"]:
        raise ValueError("public form stimuli unexpectedly overlap")
    return result


def allocation_index(study_dir: Path) -> dict[str, str]:
    rows = read_csv(study_dir / "participant_code_allocation_private.csv")
    result = {}
    for row in rows:
        code = row["participant_code"]
        if code in result or row["form"] not in FORMS:
            raise ValueError("invalid private participant-code allocation")
        result[code] = row["form"]
    if Counter(result.values()) != Counter({"F1": 42, "F2": 42}):
        raise ValueError("participant-code allocation is not 42/42")
    return result


def validate_response(
    path: Path,
    form_ids: dict[str, set[str]],
    expected_formal: bool,
    allowed_codes: dict[str, str] | None,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"unexpected response schema: {payload.get('schema')}")
    if payload.get("formalStudyData") is not expected_formal:
        raise ValueError("formalStudyData flag does not match analysis mode")
    form = payload.get("form")
    code = str(payload.get("participant_code", "")).strip().upper()
    if form not in FORMS or not code:
        raise ValueError("response is missing a valid form or participant code")
    if allowed_codes is not None and allowed_codes.get(code) != form:
        raise ValueError("participant code is absent or assigned to another form")
    eligibility = payload.get("eligibility")
    if not isinstance(eligibility, dict) or not all(
        eligibility.get(key) is True
        for key in ("adult", "consent", "headphones_or_quiet")
    ):
        raise ValueError("eligibility confirmations are incomplete")
    completed_at = str(payload.get("completed_at", ""))
    completed_time = parse_timestamp(completed_at)
    responses = payload.get("responses")
    if not isinstance(responses, list) or len(responses) != 24:
        raise ValueError("response must contain exactly 24 item rows")
    ids = [str(row.get("stimulus_id", "")) for row in responses]
    if len(set(ids)) != 24 or set(ids) != form_ids[form]:
        raise ValueError("response stimulus IDs do not match the assigned form")
    positions = sorted(int(row.get("presentation_position", 0)) for row in responses)
    if positions != list(range(1, 25)):
        raise ValueError("presentation positions must be 1 through 24")

    normalized_rows = []
    for row in responses:
        stimulus_id = row["stimulus_id"]
        valence = row.get("valence")
        arousal = row.get("arousal")
        if valence not in VALENCE_CHOICES or arousal not in AROUSAL_CHOICES:
            raise ValueError(f"invalid emotion response: {code}/{stimulus_id}")
        try:
            quality = int(row["quality"])
            coherence = int(row["coherence"])
            played_seconds = float(row["played_seconds"])
            duration_seconds = float(row["duration_seconds"])
            played_fraction = float(row["played_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric response: {code}/{stimulus_id}") from exc
        if quality not in range(1, 6) or coherence not in range(1, 6):
            raise ValueError(f"ratings must be integers 1-5: {code}/{stimulus_id}")
        if (
            not all(math.isfinite(value) for value in (played_seconds, duration_seconds, played_fraction))
            or duration_seconds <= 0
            or played_seconds < 0
            or played_fraction < 0.80
            or played_fraction > 1.001
        ):
            raise ValueError(f"playback gate failed: {code}/{stimulus_id}")
        defects = row.get("defects", [])
        if not isinstance(defects, list) or not set(defects).issubset(DEFECT_CHOICES):
            raise ValueError(f"invalid defect choices: {code}/{stimulus_id}")
        normalized_rows.append(
            {
                "participant_code": code,
                "form": form,
                "musical_experience": str(payload.get("musical_experience", "")),
                "stimulus_id": stimulus_id,
                "presentation_position": int(row["presentation_position"]),
                "valence": valence,
                "arousal": arousal,
                "quality": quality,
                "coherence": coherence,
                "defects": defects,
                "notes_private": str(row.get("notes", ""))[:500],
                "played_seconds": played_seconds,
                "duration_seconds": duration_seconds,
                "played_fraction": played_fraction,
            }
        )
    return {
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "participant_code": code,
        "form": form,
        "completed_at": completed_at,
        "completed_time": completed_time,
        "rows": normalized_rows,
    }


def freeze_responses(
    response_paths: list[Path],
    study_dir: Path,
    output_dir: Path,
    expected_formal: bool,
    eligible_per_form: int,
    enforce_allocation: bool,
) -> tuple[list[dict], dict]:
    if len(response_paths) > 84:
        raise ValueError("submission cap of 84 responses was exceeded")
    form_ids = public_form_ids(study_dir)
    codes = allocation_index(study_dir) if enforce_allocation else None
    frozen_dir = output_dir / "frozen_raw_responses"
    frozen_dir.mkdir(parents=True)
    valid = []
    records = []
    for path in sorted(path.resolve() for path in response_paths):
        digest = sha256_file(path)
        destination = frozen_dir / f"{digest[:12]}-{path.name}"
        shutil.copy2(path, destination)
        if sha256_file(destination) != digest:
            raise ValueError(f"frozen raw response hash mismatch: {path}")
        try:
            item = validate_response(destination, form_ids, expected_formal, codes)
            valid.append(item)
            records.append(
                {
                    "source_path": str(path),
                    "source_sha256": digest,
                    "frozen_path": str(destination),
                    "validation": "valid_complete",
                    "reason": "",
                    "participant_code": item["participant_code"],
                    "form": item["form"],
                    "completed_at": item["completed_at"],
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            records.append(
                {
                    "source_path": str(path),
                    "source_sha256": digest,
                    "frozen_path": str(destination),
                    "validation": "excluded_invalid_or_ineligible",
                    "reason": str(exc),
                    "participant_code": "",
                    "form": "",
                    "completed_at": "",
                }
            )

    by_code = defaultdict(list)
    for item in valid:
        by_code[item["participant_code"]].append(item)
    deduplicated = []
    for code, items in by_code.items():
        ordered = sorted(items, key=lambda item: (item["completed_time"], item["sha256"]))
        deduplicated.append(ordered[0])
        for duplicate in ordered[1:]:
            for record in records:
                if record["source_sha256"] == duplicate["sha256"]:
                    record["validation"] = "excluded_duplicate_after_first_complete"
                    record["reason"] = f"first complete export retained for {code}"
                    break

    selected = []
    counts = {}
    for form in FORMS:
        ordered = sorted(
            (item for item in deduplicated if item["form"] == form),
            key=lambda item: (item["completed_time"], item["sha256"]),
        )
        selected.extend(ordered[:eligible_per_form])
        for extra in ordered[eligible_per_form:]:
            for record in records:
                if record["source_sha256"] == extra["sha256"]:
                    record["validation"] = "excluded_after_form_quota_reached"
                    record["reason"] = f"earliest {eligible_per_form} eligible {form} exports retained"
                    break
        counts[form] = len(ordered[:eligible_per_form])

    selected_hashes = {item["sha256"] for item in selected}
    for record in records:
        if record["source_sha256"] in selected_hashes:
            record["validation"] = "included_confirmatory"
    freeze_manifest = {
        "schema": "moonmuse-v82-response-freeze-v1",
        "status": (
            "complete_and_frozen_before_unblinding"
            if all(counts[form] == eligible_per_form for form in FORMS)
            else "incomplete_frozen_no_unblinding"
        ),
        "formalStudyData": expected_formal,
        "raw_submission_count": len(response_paths),
        "eligible_target_per_form": eligible_per_form,
        "included_by_form": counts,
        "private_mapping_opened": False,
        "records": records,
    }
    freeze_path = output_dir / "response_freeze_manifest.json"
    freeze_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, freeze_manifest


def lower_t_bound(values: list[float], alpha: float = ALPHA_ONE_SIDED) -> dict:
    if len(values) < 2:
        raise ValueError("at least two values are required for a t bound")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    lower = mean - stats.t.ppf(1 - alpha, len(values) - 1) * standard_error
    return {
        "n": len(values),
        "mean_difference": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "one_sided_confidence_level": 1 - alpha,
        "lower_confidence_bound": float(lower),
    }


def load_private_mapping(study_dir: Path) -> dict[str, dict]:
    rows = read_csv(study_dir / "master_manifest_private.csv")
    if len(rows) != 48:
        raise ValueError("private mapping must contain 48 stimuli")
    mapping = {}
    for row in rows:
        stimulus_id = row["stimulus_id"]
        if stimulus_id in mapping or row["system_private"] not in SYSTEMS:
            raise ValueError("invalid private stimulus mapping")
        mapping[stimulus_id] = row
    return mapping


def join_rows(selected: list[dict], mapping: dict[str, dict]) -> list[dict]:
    joined = []
    for participant in selected:
        for response in participant["rows"]:
            source = mapping[response["stimulus_id"]]
            target_valence = source["target_valence_private"]
            target_arousal = source["target_arousal_private"]
            valence_correct = response["valence"] == target_valence
            arousal_correct = response["arousal"] == target_arousal
            joined.append(
                {
                    **response,
                    "system_private": source["system_private"],
                    "slot_id_private": source["slot_id"],
                    "seed_private": source["seed_private"],
                    "quadrant_private": source["quadrant_private"],
                    "target_valence_private": target_valence,
                    "target_arousal_private": target_arousal,
                    "valence_correct": int(valence_correct),
                    "arousal_correct": int(arousal_correct),
                    "quadrant_correct": int(valence_correct and arousal_correct),
                    "valence_uncertain": int(response["valence"] == "uncertain"),
                    "arousal_uncertain": int(response["arousal"] == "uncertain"),
                    "has_defect": int(bool(response["defects"])),
                }
            )
    return joined


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def summarize(rows: list[dict]) -> dict:
    defects = Counter(defect for row in rows for defect in row["defects"])
    return {
        "ratings": len(rows),
        "arousal_accuracy": mean(rows, "arousal_correct"),
        "valence_accuracy": mean(rows, "valence_correct"),
        "four_quadrant_accuracy": mean(rows, "quadrant_correct"),
        "arousal_uncertain_rate": mean(rows, "arousal_uncertain"),
        "valence_uncertain_rate": mean(rows, "valence_uncertain"),
        "mean_quality": mean(rows, "quality"),
        "mean_coherence": mean(rows, "coherence"),
        "any_defect_rate": mean(rows, "has_defect"),
        "defect_counts": dict(sorted(defects.items())),
    }


def difference_rows(joined: list[dict]) -> tuple[list[dict], list[dict]]:
    listener_rows = []
    for participant_code in sorted({row["participant_code"] for row in joined}):
        group = [row for row in joined if row["participant_code"] == participant_code]
        values = {
            system: [row for row in group if row["system_private"] == system]
            for system in SYSTEMS
        }
        if any(len(values[system]) != 12 for system in SYSTEMS):
            raise ValueError(f"participant is not 12/12 balanced: {participant_code}")
        listener_rows.append(
            {
                "participant_code_private": participant_code,
                "form": group[0]["form"],
                "arousal_candidate_minus_baseline": mean(values["moonmuse_v70"], "arousal_correct") - mean(values["emo_disentangler"], "arousal_correct"),
                "quality_candidate_minus_baseline": mean(values["moonmuse_v70"], "quality") - mean(values["emo_disentangler"], "quality"),
                "coherence_candidate_minus_baseline": mean(values["moonmuse_v70"], "coherence") - mean(values["emo_disentangler"], "coherence"),
            }
        )

    slot_rows = []
    for slot_id in sorted({row["slot_id_private"] for row in joined}):
        group = [row for row in joined if row["slot_id_private"] == slot_id]
        values = {
            system: [row for row in group if row["system_private"] == system]
            for system in SYSTEMS
        }
        if any(not values[system] for system in SYSTEMS):
            raise ValueError(f"slot does not contain both systems: {slot_id}")
        slot_rows.append(
            {
                "slot_id_private": slot_id,
                "seed_private": group[0]["seed_private"],
                "quadrant_private": group[0]["quadrant_private"],
                "arousal_candidate_minus_baseline": mean(values["moonmuse_v70"], "arousal_correct") - mean(values["emo_disentangler"], "arousal_correct"),
                "quality_candidate_minus_baseline": mean(values["moonmuse_v70"], "quality") - mean(values["emo_disentangler"], "quality"),
                "coherence_candidate_minus_baseline": mean(values["moonmuse_v70"], "coherence") - mean(values["emo_disentangler"], "coherence"),
            }
        )
    if len(slot_rows) != 24:
        raise ValueError(f"expected 24 stimulus slots, found {len(slot_rows)}")
    return listener_rows, slot_rows


def confirmatory(listener_rows: list[dict], slot_rows: list[dict]) -> dict:
    arousal_listener = lower_t_bound(
        [row["arousal_candidate_minus_baseline"] for row in listener_rows]
    )
    arousal_stimulus = lower_t_bound(
        [row["arousal_candidate_minus_baseline"] for row in slot_rows]
    )
    quality_listener = lower_t_bound(
        [row["quality_candidate_minus_baseline"] for row in listener_rows]
    )
    quality_stimulus = lower_t_bound(
        [row["quality_candidate_minus_baseline"] for row in slot_rows]
    )
    arousal_pass = (
        arousal_listener["lower_confidence_bound"] > 0
        and arousal_stimulus["lower_confidence_bound"] > 0
    )
    quality_pass = (
        quality_listener["lower_confidence_bound"] > QUALITY_NI_MARGIN
        and quality_stimulus["lower_confidence_bound"] > QUALITY_NI_MARGIN
    )
    return {
        "alpha_one_sided": ALPHA_ONE_SIDED,
        "arousal_superiority": {
            "margin": 0.0,
            "listener_analysis": arousal_listener,
            "stimulus_analysis": arousal_stimulus,
            "passed": arousal_pass,
        },
        "quality_noninferiority": {
            "margin": QUALITY_NI_MARGIN,
            "listener_analysis": quality_listener,
            "stimulus_analysis": quality_stimulus,
            "passed": quality_pass,
        },
        "overall_bounded_baseline_outperformance_claim_passed": arousal_pass and quality_pass,
    }


def deidentified_rows(joined: list[dict]) -> list[dict]:
    participant_ids = {
        code: f"P{index:03d}"
        for index, code in enumerate(
            sorted({row["participant_code"] for row in joined}), start=1
        )
    }
    return [
        {
            "participant_id": participant_ids[row["participant_code"]],
            "form": row["form"],
            "stimulus_id": row["stimulus_id"],
            "slot_id": row["slot_id_private"],
            "system": row["system_private"],
            "seed": row["seed_private"],
            "quadrant": row["quadrant_private"],
            "target_valence": row["target_valence_private"],
            "target_arousal": row["target_arousal_private"],
            "perceived_valence": row["valence"],
            "perceived_arousal": row["arousal"],
            "quality_1_to_5": row["quality"],
            "coherence_1_to_5": row["coherence"],
            "defects": ";".join(row["defects"]),
            "arousal_correct": row["arousal_correct"],
            "valence_correct": row["valence_correct"],
            "quadrant_correct": row["quadrant_correct"],
            "played_fraction": row["played_fraction"],
        }
        for row in joined
    ]


def analyze(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    response_paths = list(args.response or [])
    if args.response_dir:
        response_paths.extend(sorted(args.response_dir.glob("*.json")))
    response_paths = sorted(set(path.resolve() for path in response_paths))
    if not response_paths:
        raise ValueError("no JSON responses were provided")

    selected, freeze_manifest = freeze_responses(
        response_paths,
        args.study_dir.resolve(),
        args.output_dir,
        expected_formal=not args.prelaunch_test,
        eligible_per_form=args.eligible_per_form,
        enforce_allocation=not args.prelaunch_test,
    )
    freeze_path = args.output_dir / "response_freeze_manifest.json"
    if freeze_manifest["status"] != "complete_and_frozen_before_unblinding":
        report = {
            "schema": ANALYSIS_SCHEMA,
            "status": "incomplete_no_unblinding_no_endpoint_analysis",
            "formalStudyData": not args.prelaunch_test,
            "response_freeze_manifest_sha256": sha256_file(freeze_path),
            "included_by_form": freeze_manifest["included_by_form"],
            "private_mapping_opened": False,
        }
        (args.output_dir / "analysis_status.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    mapping = load_private_mapping(args.study_dir)
    joined = join_rows(selected, mapping)
    listener_rows, slot_rows = difference_rows(joined)
    decision = confirmatory(listener_rows, slot_rows)
    by_system = {
        system: summarize([row for row in joined if row["system_private"] == system])
        for system in SYSTEMS
    }
    by_quadrant = {
        quadrant: {
            system: summarize(
                [
                    row for row in joined
                    if row["quadrant_private"] == quadrant
                    and row["system_private"] == system
                ]
            )
            for system in SYSTEMS
        }
        for quadrant in ("Q1", "Q2", "Q3", "Q4")
    }
    report = {
        "schema": ANALYSIS_SCHEMA,
        "status": (
            "bounded_baseline_outperformance_supported"
            if decision["overall_bounded_baseline_outperformance_claim_passed"]
            else "bounded_baseline_outperformance_not_supported"
        ),
        "formalStudyData": not args.prelaunch_test,
        "prelaunchSyntheticData": args.prelaunch_test,
        "participants": len(selected),
        "participants_by_form": Counter(item["form"] for item in selected),
        "ratings": len(joined),
        "response_freeze_manifest_sha256": sha256_file(freeze_path),
        "private_mapping_sha256": sha256_file(
            args.study_dir / "master_manifest_private.csv"
        ),
        "unblinding_order": "raw_exports_validated_and_hash_frozen_before_private_mapping_open",
        "confirmatory": decision,
        "by_system": by_system,
        "by_quadrant": by_quadrant,
        "claim_boundary": (
            "A pass supports only better human-recognized arousal control with "
            "noninferior listenability for these eight-bar symbolic-piano populations. "
            "It does not establish valence superiority, general musical superiority, "
            "commercial-system parity, or therapeutic efficacy."
        ),
    }
    report["participants_by_form"] = dict(report["participants_by_form"])
    write_csv(args.output_dir / "joined_ratings_private.csv", joined)
    write_csv(args.output_dir / "listener_differences_private.csv", listener_rows)
    write_csv(args.output_dir / "stimulus_slot_differences_private.csv", slot_rows)
    write_csv(args.output_dir / "source_data_deidentified.csv", deidentified_rows(joined))
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            (
                "# MoonMuse v82 analysis",
                "",
                f"Status: `{report['status']}`",
                f"Participants: `{report['participants']}`; ratings: `{report['ratings']}`.",
                f"Arousal superiority passed: `{str(decision['arousal_superiority']['passed']).lower()}`.",
                f"Quality noninferiority passed: `{str(decision['quality_noninferiority']['passed']).lower()}`.",
                f"Overall bounded claim passed: `{str(decision['overall_bounded_baseline_outperformance_claim_passed']).lower()}`.",
                "",
                "`joined_ratings_private.csv` contains private mappings and free text. Do not release it.",
                "`source_data_deidentified.csv` omits participant codes and free text, but release still requires the approved consent/data plan.",
                "",
                report["claim_boundary"],
                "",
            )
        ),
        encoding="utf-8",
    )
    dictionary_rows = [
        {"field": "participant_id", "definition": "De-identified sequential participant identifier", "allowed_values": "P001...", "unit": "none"},
        {"field": "system", "definition": "Frozen generation system", "allowed_values": ";".join(SYSTEMS), "unit": "none"},
        {"field": "target_arousal", "definition": "Target arousal derived from frozen quadrant", "allowed_values": "high;low", "unit": "none"},
        {"field": "perceived_arousal", "definition": "Participant arousal response", "allowed_values": "high;low;uncertain", "unit": "none"},
        {"field": "quality_1_to_5", "definition": "Overall subjective listenability", "allowed_values": "1;2;3;4;5", "unit": "ordinal point"},
        {"field": "coherence_1_to_5", "definition": "Subjective musical coherence", "allowed_values": "1;2;3;4;5", "unit": "ordinal point"},
        {"field": "played_fraction", "definition": "Maximum verified playback fraction", "allowed_values": "0.80 to 1.00", "unit": "proportion"},
    ]
    write_csv(args.output_dir / "data_dictionary.csv", dictionary_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--response", action="append", type=Path)
    parser.add_argument("--response-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eligible-per-form", type=int, default=36)
    parser.add_argument("--prelaunch-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = analyze(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
