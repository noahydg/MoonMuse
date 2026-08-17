#!/usr/bin/env python3
"""Recover and freeze the 300 Stage 2 sources used by the v81 v70 screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


SOURCE_RE = re.compile(
    r"(?:^|/)stage2_melody/"
    r"samp_(?P<index>\d{2})_(?P<valence>Positive|Negative)_"
    r"(?P<quadrant>Q[1-4])_melody\.txt$"
)
EXPECTED_VALENCE = {"Q1": "Positive", "Q2": "Negative", "Q3": "Negative", "Q4": "Positive"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_member_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"not a regular archive member: {member_name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot read archive member: {member_name}")
    return handle.read()


def collect_stage2_sources(archive_bytes: bytes, seed: int) -> list[dict]:
    rows = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            match = SOURCE_RE.search(member.name)
            if match is None:
                continue
            if not member.isfile():
                raise ValueError(f"source is not a regular file: {member.name}")
            data = read_member_bytes(archive, member.name)
            quadrant = match.group("quadrant")
            valence = match.group("valence")
            if valence != EXPECTED_VALENCE[quadrant]:
                raise ValueError(f"quadrant/valence mismatch: {member.name}")
            rows.append(
                {
                    "seed": seed,
                    "quadrant": quadrant,
                    "sample_index": int(match.group("index")),
                    "valence": valence.lower(),
                    "source_member": member.name,
                    "source_name": Path(member.name).name,
                    "source_sha256": sha256_bytes(data),
                    "data": data,
                }
            )
    rows.sort(key=lambda row: (row["quadrant"], row["sample_index"]))
    expected = {(quadrant, index) for quadrant in EXPECTED_VALENCE for index in range(25)}
    observed = {(row["quadrant"], row["sample_index"]) for row in rows}
    if len(rows) != 100 or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"seed {seed}: expected 100 balanced sources; missing={missing}, extra={extra}")
    return rows


def load_inputs(args: argparse.Namespace, pins: dict) -> tuple[dict[int, bytes], dict]:
    seed42_bytes = args.seed42_archive.read_bytes()
    seed42_hash = sha256_bytes(seed42_bytes)
    if seed42_hash != pins["seed42"]["sha256"]:
        raise ValueError(f"seed42 archive hash mismatch: {seed42_hash}")

    container_hash = sha256_file(args.container_archive)
    if container_hash != pins["container"]["sha256"]:
        raise ValueError(f"container archive hash mismatch: {container_hash}")

    archive_bytes = {42: seed42_bytes}
    nested_provenance = {}
    with tarfile.open(args.container_archive, mode="r:gz") as outer:
        for seed in (43, 44):
            spec = pins["nested"][str(seed)]
            data = read_member_bytes(outer, spec["member"])
            actual_hash = sha256_bytes(data)
            if actual_hash != spec["sha256"]:
                raise ValueError(f"seed{seed} nested archive hash mismatch: {actual_hash}")
            archive_bytes[seed] = data
            nested_provenance[str(seed)] = {
                "member": spec["member"],
                "sha256": actual_hash,
            }
    return archive_bytes, {
        "seed42_archive": str(args.seed42_archive.resolve()),
        "seed42_archive_sha256": seed42_hash,
        "container_archive": str(args.container_archive.resolve()),
        "container_archive_sha256": container_hash,
        "nested": nested_provenance,
    }


def prepare(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    pins = json.loads(args.pin_file.read_text(encoding="utf-8"))
    archive_bytes, provenance = load_inputs(args, pins)
    protocol_hash = sha256_file(args.protocol)

    all_rows = []
    args.output_dir.mkdir(parents=True)
    for seed in (42, 43, 44):
        nested_hash = sha256_bytes(archive_bytes[seed])
        for row in collect_stage2_sources(archive_bytes[seed], seed):
            relative = Path("sources") / f"seed{seed}" / row["quadrant"] / row["source_name"]
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(row.pop("data"))
            if sha256_file(destination) != row["source_sha256"]:
                raise RuntimeError(f"written source hash mismatch: {destination}")
            row.update(
                {
                    "source_archive_sha256": nested_hash,
                    "frozen_path": str(relative),
                }
            )
            all_rows.append(row)

    counts = Counter((row["seed"], row["quadrant"]) for row in all_rows)
    if len(all_rows) != 300 or any(counts[(seed, q)] != 25 for seed in (42, 43, 44) for q in EXPECTED_VALENCE):
        raise RuntimeError("final source population is not 3 x 4 x 25")

    fieldnames = list(all_rows[0])
    with (args.output_dir / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    hash_counts = Counter(row["source_sha256"] for row in all_rows)
    report = {
        "schema": "moonmuse-v81-v70-source-freeze-v1",
        "status": "source_freeze_complete",
        "formalStudyData": False,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_hash,
        "pin_file": str(args.pin_file.resolve()),
        "pin_file_sha256": sha256_file(args.pin_file),
        "provenance": provenance,
        "source_count": len(all_rows),
        "counts_by_seed_quadrant": {
            str(seed): {q: counts[(seed, q)] for q in EXPECTED_VALENCE}
            for seed in (42, 43, 44)
        },
        "unique_source_hash_count": len(hash_counts),
        "duplicate_content_group_count": sum(count > 1 for count in hash_counts.values()),
        "source_manifest_sha256": sha256_file(args.output_dir / "source_manifest.csv"),
        "interpretation_boundary": (
            "This freezes the original Full Stage 2 event texts only; it does not "
            "generate or evaluate v70 outputs."
        ),
    }
    (args.output_dir / "source_freeze.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SOURCE_FREEZE_COMPLETE").write_text(
        sha256_file(args.output_dir / "source_freeze.json") + "  source_freeze.json\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-archive", type=Path, required=True)
    parser.add_argument("--container-archive", type=Path, required=True)
    parser.add_argument("--pin-file", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
