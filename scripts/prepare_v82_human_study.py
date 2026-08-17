#!/usr/bin/env python3
"""Prepare the frozen v82 two-form human listening-study population."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "moonmuse-v82-human-study-prelaunch-v1"
SELECTION_NAMESPACE = "moonmuse-v82-selection-20260809-v1"
BLIND_NAMESPACE = "moonmuse-v82-blind-id-20260809-v1"
CODE_NAMESPACE = "moonmuse-v82-participant-code-20260809-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "d19374c9718cd0ddbfb1addc253c23a587396973c22faab14d0c6811448b8687"
)
SEEDS = (42, 43, 44)
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
SYSTEMS = ("moonmuse_v70", "emo_disentangler")
FORMS = ("F1", "F2")
EMOTION_TARGETS = {
    "Q1": ("positive", "high"),
    "Q2": ("negative", "high"),
    "Q3": ("negative", "low"),
    "Q4": ("positive", "low"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(namespace: str, *parts: object) -> str:
    material = "|".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_reason_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    normalized = value.strip()
    if not normalized.startswith("["):
        return [item for item in normalized.split(";") if item]
    parsed = ast.literal_eval(normalized)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"unexpected failure-reason value: {value!r}")
    return parsed


def load_v74_index(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    result = {}
    for row in rows:
        midi_hash = row["midi_sha256"]
        if midi_hash in result:
            raise ValueError(f"duplicate v74 MIDI hash: {midi_hash}")
        locked = parse_reason_list(row.get("locked_failure_reasons"))
        midi_only = parse_reason_list(row.get("midi_failure_reasons"))
        without_duplicate = parse_reason_list(
            row.get("without_duplicate_failure_reasons")
        )
        result[midi_hash] = {
            "v74_locked_failure_reasons": ";".join(locked),
            "v74_locked_pass": not locked if row["system"] == "moonmuse_v70" else "",
            "v74_midi_failure_reasons": ";".join(midi_only),
            "v74_midi_pass": not midi_only,
            "v74_without_duplicate_failure_reasons": ";".join(without_duplicate),
            "v74_without_duplicate_pass": not without_duplicate,
            "v74_note_count": row["note_count"],
            "v74_note_span_beats": row["note_span_beats"],
            "v74_max_internal_gap_beats": row["max_internal_gap_beats"],
            "v74_max_same_pitch_onset_count": row[
                "max_same_pitch_onset_count"
            ],
        }
    return result


def load_candidate(manifest_path: Path) -> dict[tuple[int, str, int], dict]:
    rows = read_csv(manifest_path)
    if len(rows) != 300:
        raise ValueError(f"expected 300 candidate rows, found {len(rows)}")
    root = manifest_path.parent
    result = {}
    for row in rows:
        if row["system"] != "moonmuse_v70":
            raise ValueError(f"unexpected candidate system: {row['system']}")
        key = (int(row["seed"]), row["quadrant"], int(row["sample_index"]))
        if key in result:
            raise ValueError(f"duplicate candidate key: {key}")
        midi_path = root / row["midi_path"]
        actual_hash = sha256_file(midi_path)
        if actual_hash != row["midi_sha256"]:
            raise ValueError(f"candidate MIDI hash mismatch: {midi_path}")
        result[key] = {
            "candidate_midi_path": midi_path.resolve(),
            "candidate_midi_sha256": actual_hash,
            "candidate_event_path": (root / row["event_path"]).resolve(),
            "candidate_event_sha256": row["event_sha256"],
            "candidate_note_count_manifest": row["note_count"],
        }
    return result


def load_baseline(manifest_path: Path) -> dict[tuple[int, str, int], dict]:
    rows = [
        row for row in read_csv(manifest_path)
        if row["system"] == "emo_disentangler"
    ]
    if len(rows) != 300:
        raise ValueError(f"expected 300 baseline rows, found {len(rows)}")
    root = manifest_path.parent
    result = {}
    for row in rows:
        key = (int(row["seed"]), row["quadrant"], int(row["sample_index"]))
        if key in result:
            raise ValueError(f"duplicate baseline key: {key}")
        midi_path = root / row["standardized_midi"]
        actual_hash = sha256_file(midi_path)
        if actual_hash != row["standardized_sha256"]:
            raise ValueError(f"baseline MIDI hash mismatch: {midi_path}")
        result[key] = {
            "baseline_midi_path": midi_path.resolve(),
            "baseline_midi_sha256": actual_hash,
            "baseline_source_locator": row["source_locator"],
            "baseline_source_sha256": row["source_sha256"],
            "baseline_note_count_manifest": row["standardized_note_count"],
        }
    return result


def join_population(
    candidate: dict[tuple[int, str, int], dict],
    baseline: dict[tuple[int, str, int], dict],
) -> list[dict]:
    expected = {
        (seed, quadrant, sample_index)
        for seed in SEEDS
        for quadrant in QUADRANTS
        for sample_index in range(25)
    }
    if set(candidate) != expected or set(baseline) != expected:
        raise ValueError(
            "candidate/baseline join keys do not equal the frozen 3x4x25 population"
        )
    rows = []
    for seed in SEEDS:
        for quadrant in QUADRANTS:
            for sample_index in range(25):
                key = (seed, quadrant, sample_index)
                candidate_row = candidate[key]
                baseline_row = baseline[key]
                digest = stable_digest(
                    SELECTION_NAMESPACE,
                    seed,
                    quadrant,
                    sample_index,
                    candidate_row["candidate_midi_sha256"],
                    baseline_row["baseline_midi_sha256"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "quadrant": quadrant,
                        "sample_index": sample_index,
                        "selection_digest": digest,
                        **candidate_row,
                        **baseline_row,
                    }
                )
    return rows


def select_slots(joined: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in joined:
        grouped[(int(row["seed"]), row["quadrant"])].append(row)
    if set(grouped) != {(seed, quadrant) for seed in SEEDS for quadrant in QUADRANTS}:
        raise ValueError("joined population is missing seed-by-quadrant cells")

    selected = []
    audit = []
    slot_number = 0
    for seed in SEEDS:
        for quadrant in QUADRANTS:
            ordered = sorted(grouped[(seed, quadrant)], key=lambda row: row["selection_digest"])
            if len(ordered) != 25:
                raise ValueError(f"expected 25 rows for {(seed, quadrant)}")
            selected_digests = {row["selection_digest"] for row in ordered[:2]}
            rank_by_digest = {
                row["selection_digest"]: rank
                for rank, row in enumerate(ordered, start=1)
            }
            for row in ordered:
                audit.append(
                    {
                        "seed": seed,
                        "quadrant": quadrant,
                        "sample_index": row["sample_index"],
                        "selection_digest": row["selection_digest"],
                        "selection_rank_within_cell": rank_by_digest[row["selection_digest"]],
                        "selected": row["selection_digest"] in selected_digests,
                        "candidate_midi_sha256": row["candidate_midi_sha256"],
                        "baseline_midi_sha256": row["baseline_midi_sha256"],
                    }
                )
            for cell_rank, row in enumerate(ordered[:2], start=1):
                slot_number += 1
                selected.append(
                    {
                        **row,
                        "cell_rank": cell_rank,
                        "slot_id": f"SL{slot_number:02d}",
                    }
                )
    if len(selected) != 24 or len(audit) != 300:
        raise AssertionError("selection must produce 24 slots and 300 audit rows")
    return selected, audit


def assigned_form(cell_rank: int, system: str) -> str:
    if cell_rank not in {1, 2} or system not in SYSTEMS:
        raise ValueError("invalid form-assignment input")
    candidate_form = "F1" if cell_rank == 1 else "F2"
    if system == "moonmuse_v70":
        return candidate_form
    return "F2" if candidate_form == "F1" else "F1"


def build_variants(slots: list[dict], v74_indexes: dict[str, dict]) -> list[dict]:
    variants = []
    for slot in slots:
        target_valence, target_arousal = EMOTION_TARGETS[slot["quadrant"]]
        for system in SYSTEMS:
            prefix = "candidate" if system == "moonmuse_v70" else "baseline"
            midi_hash = slot[f"{prefix}_midi_sha256"]
            if midi_hash not in v74_indexes:
                raise ValueError(f"selected MIDI missing from v74 audit: {midi_hash}")
            variants.append(
                {
                    **slot,
                    "system_private": system,
                    "form": assigned_form(int(slot["cell_rank"]), system),
                    "target_valence_private": target_valence,
                    "target_arousal_private": target_arousal,
                    "source_midi_path": slot[f"{prefix}_midi_path"],
                    "source_midi_sha256": midi_hash,
                    "blind_digest": stable_digest(
                        BLIND_NAMESPACE,
                        slot["selection_digest"],
                        system,
                        midi_hash,
                    ),
                    **v74_indexes[midi_hash],
                }
            )
    ordered = sorted(variants, key=lambda row: row["blind_digest"])
    return [
        {**row, "stimulus_id": f"V82{index:03d}"}
        for index, row in enumerate(ordered, start=1)
    ]


def validate_design(rows: list[dict]) -> dict:
    if len(rows) != 48 or len({row["stimulus_id"] for row in rows}) != 48:
        raise ValueError("v82 must contain 48 unique stimuli")
    if Counter(row["system_private"] for row in rows) != Counter(
        {"moonmuse_v70": 24, "emo_disentangler": 24}
    ):
        raise ValueError("system totals are not 24/24")
    slot_groups = defaultdict(list)
    for row in rows:
        slot_groups[row["slot_id"]].append(row)
    for slot_id, group in slot_groups.items():
        if len(group) != 2:
            raise ValueError(f"slot {slot_id} does not contain two variants")
        if {row["system_private"] for row in group} != set(SYSTEMS):
            raise ValueError(f"slot {slot_id} does not cover both systems")
        if {row["form"] for row in group} != set(FORMS):
            raise ValueError(f"slot {slot_id} does not reverse across forms")

    by_form = {}
    for form in FORMS:
        group = [row for row in rows if row["form"] == form]
        counts = Counter(row["system_private"] for row in group)
        strata = Counter(
            (row["seed"], row["quadrant"], row["system_private"])
            for row in group
        )
        if len(group) != 24 or any(counts[system] != 12 for system in SYSTEMS):
            raise ValueError(f"form {form} is not 12/12 balanced")
        expected_strata = {
            (seed, quadrant, system)
            for seed in SEEDS
            for quadrant in QUADRANTS
            for system in SYSTEMS
        }
        if set(strata) != expected_strata or set(strata.values()) != {1}:
            raise ValueError(f"form {form} is not balanced in every stratum")
        by_form[form] = {
            "stimuli": len(group),
            "by_system": dict(sorted(counts.items())),
            "strata_count": len(strata),
        }
    return {
        "stimuli": len(rows),
        "slots": len(slot_groups),
        "by_form": by_form,
        "slot_system_and_form_reversal": True,
    }


def participant_codes() -> list[dict]:
    rows = []
    for allocation_order in range(1, 85):
        form = FORMS[(allocation_order - 1) % 2]
        digest = stable_digest(CODE_NAMESPACE, allocation_order, form)
        rows.append(
            {
                "allocation_order": allocation_order,
                "participant_code": f"MM-{digest[:8].upper()}",
                "form": form,
                "target_or_reserve": (
                    "target" if allocation_order <= 72 else "reserve"
                ),
                "issued": False,
                "eligible_complete": "",
            }
        )
    if Counter(row["form"] for row in rows) != Counter({"F1": 42, "F2": 42}):
        raise AssertionError("participant-code allocation must be 42/42")
    return rows


def public_rows(private_rows: list[dict], form: str) -> list[dict]:
    selected = sorted(
        (row for row in private_rows if row["form"] == form),
        key=lambda row: row["stimulus_id"],
    )
    return [
        {
            "stimulus_id": row["stimulus_id"],
            "midi_file": f"stimuli_midi/{row['stimulus_id']}.mid",
            "midi_sha256": row["source_midi_sha256"],
        }
        for row in selected
    ]


def checksum_manifest(root: Path, relative_paths: list[Path]) -> None:
    lines = [
        f"{sha256_file(root / path)}  {path.as_posix()}"
        for path in sorted(relative_paths, key=lambda value: value.as_posix())
    ]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol_hash = sha256_file(args.protocol)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            f"protocol hash mismatch: expected {EXPECTED_PROTOCOL_SHA256}, found {protocol_hash}"
        )

    candidate = load_candidate(args.candidate_manifest)
    baseline = load_baseline(args.baseline_manifest)
    joined = join_population(candidate, baseline)
    slots, audit_rows = select_slots(joined)
    v74_index = {
        **load_v74_index(args.candidate_v74_rows),
        **load_v74_index(args.baseline_v74_rows),
    }
    variants = build_variants(slots, v74_index)
    design = validate_design(variants)

    output = args.output_dir
    midi_dir = output / "stimuli_midi"
    midi_dir.mkdir(parents=True)
    master_rows = []
    for row in sorted(variants, key=lambda item: item["stimulus_id"]):
        destination = midi_dir / f"{row['stimulus_id']}.mid"
        shutil.copyfile(row["source_midi_path"], destination)
        copied_hash = sha256_file(destination)
        if copied_hash != row["source_midi_sha256"]:
            raise ValueError(f"copied MIDI hash mismatch: {destination}")
        master_rows.append(
            {
                "stimulus_id": row["stimulus_id"],
                "slot_id": row["slot_id"],
                "form": row["form"],
                "system_private": row["system_private"],
                "seed_private": row["seed"],
                "quadrant_private": row["quadrant"],
                "sample_index_private": row["sample_index"],
                "cell_rank_private": row["cell_rank"],
                "target_valence_private": row["target_valence_private"],
                "target_arousal_private": row["target_arousal_private"],
                "selection_digest_private": row["selection_digest"],
                "blind_digest_private": row["blind_digest"],
                "source_midi_path_private": str(row["source_midi_path"]),
                "source_midi_sha256": row["source_midi_sha256"],
                "study_midi_file": f"stimuli_midi/{destination.name}",
                "study_midi_sha256": copied_hash,
                "v74_locked_pass_private": row["v74_locked_pass"],
                "v74_locked_failure_reasons_private": row[
                    "v74_locked_failure_reasons"
                ],
                "v74_midi_pass_private": row["v74_midi_pass"],
                "v74_midi_failure_reasons_private": row[
                    "v74_midi_failure_reasons"
                ],
                "v74_without_duplicate_pass_private": row[
                    "v74_without_duplicate_pass"
                ],
                "v74_without_duplicate_failure_reasons_private": row[
                    "v74_without_duplicate_failure_reasons"
                ],
                "v74_note_count_private": row["v74_note_count"],
                "v74_note_span_beats_private": row["v74_note_span_beats"],
                "v74_max_internal_gap_beats_private": row[
                    "v74_max_internal_gap_beats"
                ],
                "v74_max_same_pitch_onset_count_private": row[
                    "v74_max_same_pitch_onset_count"
                ],
            }
        )

    write_csv(output / "master_manifest_private.csv", master_rows)
    write_csv(output / "selection_audit_private.csv", audit_rows)
    for form in FORMS:
        write_csv(
            output / "forms_public" / f"{form}.csv",
            public_rows(master_rows, form),
        )
        write_csv(
            output / "forms_private" / f"{form}.csv",
            [row for row in master_rows if row["form"] == form],
        )
    code_rows = participant_codes()
    write_csv(output / "participant_code_allocation_private.csv", code_rows)

    report = {
        "schema": SCHEMA,
        "status": "prelaunch_population_ready_ethics_and_audio_pending",
        "formalStudyData": False,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_hash,
        "input_manifests": {
            "candidate": str(args.candidate_manifest.resolve()),
            "candidate_sha256": sha256_file(args.candidate_manifest),
            "baseline": str(args.baseline_manifest.resolve()),
            "baseline_sha256": sha256_file(args.baseline_manifest),
            "candidate_v74_rows_sha256": sha256_file(args.candidate_v74_rows),
            "baseline_v74_rows_sha256": sha256_file(args.baseline_v74_rows),
        },
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "population_rows_audited": len(audit_rows),
            "selected_slots": len(slots),
            "selected_stimuli": len(master_rows),
            "selection_uses_classifier": False,
            "selection_uses_v74": False,
            "selection_uses_human_judgment": False,
        },
        "design": design,
        "recruitment": {
            "eligible_target": 72,
            "eligible_target_per_form": 36,
            "submission_cap": 84,
            "available_codes_per_form": 42,
        },
        "private_files": [
            "master_manifest_private.csv",
            "selection_audit_private.csv",
            "forms_private/F1.csv",
            "forms_private/F2.csv",
            "participant_code_allocation_private.csv",
        ],
        "public_prelaunch_files": ["forms_public/F1.csv", "forms_public/F2.csv"],
        "remaining_gate": (
            "Render and validate audio, build prelaunch questionnaires and analysis; "
            "do not recruit before documented ethics approval or exemption."
        ),
    }
    manifest_path = output / "study_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = "\n".join(
        (
            "# MoonMuse v82 prelaunch study population",
            "",
            "Status: population frozen; audio, questionnaires, analyzer, and ethics gate pending.",
            "",
            "- 300 joined rows were ranked only by the frozen SHA-256 rule.",
            "- 24 slots / 48 stimuli were selected.",
            "- F1 and F2 each contain 12 v70 and 12 EMO-Disentangler stimuli.",
            "- `master_manifest_private.csv`, `forms_private/`, the selection audit, and participant codes must remain private.",
            "- `forms_public/` contains no system, seed, quadrant, target, source path, classifier, or v74 label.",
            "- Do not distribute participant materials before the ethics/consent gate in the frozen protocol is satisfied.",
            "",
        )
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    critical = [
        Path("README.md"),
        Path("study_manifest.json"),
        Path("master_manifest_private.csv"),
        Path("selection_audit_private.csv"),
        Path("forms_public/F1.csv"),
        Path("forms_public/F2.csv"),
        Path("forms_private/F1.csv"),
        Path("forms_private/F2.csv"),
        Path("participant_code_allocation_private.csv"),
        *[Path("stimuli_midi") / f"{row['stimulus_id']}.mid" for row in master_rows],
    ]
    checksum_manifest(output, critical)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--candidate-v74-rows", required=True, type=Path)
    parser.add_argument("--baseline-v74-rows", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    report = prepare(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
