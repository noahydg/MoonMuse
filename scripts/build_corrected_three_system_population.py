#!/usr/bin/env python3
"""Build the v78 corrected, eight-bar, three-system MIDI population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import mido


SYSTEM_MAP = {"full": "moonmuse_full", "c2p": "moonmuse_c2p"}
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_notes_and_last_tick(midi: mido.MidiFile) -> tuple[int, int]:
    notes = 0
    last_tick = 0
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            last_tick = max(last_tick, absolute)
            if message.type == "note_on" and message.velocity > 0:
                notes += 1
    return notes, last_tick


def _crop_track(track: mido.MidiTrack, boundary: int) -> mido.MidiTrack:
    retained: list[tuple[int, int, mido.Message]] = []
    active: Counter[tuple[int, int]] = Counter()
    absolute = 0
    order = 0
    for message in track:
        absolute += message.time
        if absolute > boundary:
            break
        if message.type == "end_of_track":
            continue
        copied = message.copy(time=0)
        retained.append((absolute, order, copied))
        order += 1
        if not message.is_meta and message.type == "note_on":
            key = (getattr(message, "channel", 0), message.note)
            if message.velocity > 0:
                active[key] += 1
            elif active[key]:
                active[key] -= 1
        elif not message.is_meta and message.type == "note_off":
            key = (getattr(message, "channel", 0), message.note)
            if active[key]:
                active[key] -= 1

    for (channel, note), count in sorted(active.items()):
        for _ in range(count):
            retained.append(
                (boundary, order, mido.Message("note_off", channel=channel, note=note, velocity=0))
            )
            order += 1
    retained.append((boundary, order, mido.MetaMessage("end_of_track")))
    retained.sort(key=lambda item: (item[0], item[1]))

    output = mido.MidiTrack()
    previous = 0
    for tick, _, message in retained:
        output.append(message.copy(time=max(0, tick - previous)))
        previous = tick
    return output


def crop_midi(source: Path, destination: Path, bars: int) -> dict:
    midi = mido.MidiFile(source)
    if midi.ticks_per_beat <= 0:
        raise ValueError(f"invalid ticks_per_beat: {source}")
    boundary = bars * 4 * midi.ticks_per_beat
    source_notes, source_last_tick = count_notes_and_last_tick(midi)
    cropped = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    for track in midi.tracks:
        cropped.tracks.append(_crop_track(track, boundary))
    standardized_notes, standardized_last_tick = count_notes_and_last_tick(cropped)
    if standardized_notes <= 0:
        raise ValueError(f"crop contains no notes: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination)
    return {
        "ticks_per_beat": midi.ticks_per_beat,
        "crop_boundary_tick": boundary,
        "source_note_count": source_notes,
        "standardized_note_count": standardized_notes,
        "source_last_tick": source_last_tick,
        "standardized_last_tick": standardized_last_tick,
    }


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        members.append(member)
    return members


def load_corrected_rows(root: Path) -> list[dict]:
    manifest = root / "population_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600:
        raise ValueError(f"expected 600 corrected Full/C2P rows, found {len(rows)}")
    return rows


def baseline_archives(v49_archive: Path, workspace: Path) -> list[tuple[int, Path, str, str]]:
    outer_hash = sha256_file(v49_archive)
    results = []
    with tarfile.open(v49_archive, "r:gz") as outer:
        members = {member.name: member for member in safe_members(outer)}
        for seed in EXPECTED_SEEDS:
            suffix = (
                "formal-artifacts/formal-author-checkpoint-v41-20260804-231109/"
                f"seed{seed}-samples.tar.gz"
            )
            member = members.get(suffix)
            if member is None:
                raise ValueError(f"missing baseline samples: {suffix}")
            extracted = outer.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read baseline samples: {suffix}")
            nested_path = workspace / f"seed{seed}-samples.tar.gz"
            nested_path.write_bytes(extracted.read())
            results.append((seed, nested_path, suffix, outer_hash))
    return results


def sample_index(name: str) -> int:
    match = re.search(r"samp_(\d+)", name)
    if not match:
        raise ValueError(f"cannot parse sample index: {name}")
    return int(match.group(1))


def quadrant(name: str) -> str:
    match = re.search(r"_(Q[1-4])_", name)
    if not match:
        raise ValueError(f"cannot parse quadrant: {name}")
    return match.group(1)


def validate_balance(rows: list[dict]) -> None:
    counts = Counter((row["system"], int(row["seed"]), row["quadrant"]) for row in rows)
    expected = {
        (system, seed, quadrant_name): 25
        for system in ("moonmuse_full", "moonmuse_c2p", "emo_disentangler")
        for seed in EXPECTED_SEEDS
        for quadrant_name in EXPECTED_QUADRANTS
    }
    if counts != expected:
        missing = {key: (counts.get(key, 0), value) for key, value in expected.items() if counts.get(key, 0) != value}
        raise ValueError(f"population is not balanced: {missing}")
    if len(rows) != 900:
        raise ValueError(f"expected 900 rows, found {len(rows)}")


def build(args: argparse.Namespace) -> dict:
    corrected_root = args.corrected_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "population_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    if args.expected_corrected_manifest_sha256:
        actual = sha256_file(corrected_root / "population_manifest.csv")
        if actual != args.expected_corrected_manifest_sha256:
            raise ValueError(f"corrected manifest SHA-256 mismatch: {actual}")
    if args.expected_v49_sha256:
        actual = sha256_file(args.v49_archive)
        if actual != args.expected_v49_sha256:
            raise ValueError(f"v49 SHA-256 mismatch: {actual}")

    rows_out: list[dict] = []
    for row in load_corrected_rows(corrected_root):
        system = SYSTEM_MAP[row["system"]]
        seed = int(row["seed"])
        source = corrected_root / row["midi_file"]
        if sha256_file(source) != row["midi_sha256"]:
            raise ValueError(f"corrected MIDI hash mismatch: {source}")
        destination_rel = Path("midi") / system / f"seed{seed}" / source.name
        destination = output_dir / destination_rel
        audit = crop_midi(source, destination, args.bars)
        rows_out.append(
            {
                "system": system,
                "seed": seed,
                "quadrant": row["quadrant"],
                "sample_index": sample_index(source.name),
                "source_locator": row["midi_file"],
                "source_sha256": row["midi_sha256"],
                "source_container": row["nested_archive_member"],
                "source_container_sha256": row["nested_archive_sha256"],
                "standardized_midi": destination_rel.as_posix(),
                "standardized_sha256": sha256_file(destination),
                "crop_bars": args.bars,
                **audit,
            }
        )

    with tempfile.TemporaryDirectory(prefix="moonmuse-v78-baseline-") as temp:
        workspace = Path(temp)
        for seed, nested_path, outer_member, outer_hash in baseline_archives(args.v49_archive, workspace):
            nested_hash = sha256_file(nested_path)
            with tarfile.open(nested_path, "r:gz") as nested:
                members = [member for member in safe_members(nested) if member.isfile() and member.name.lower().endswith((".mid", ".midi"))]
                if len(members) != 100:
                    raise ValueError(f"expected 100 baseline MIDI for seed {seed}, found {len(members)}")
                for member in sorted(members, key=lambda item: item.name):
                    payload = nested.extractfile(member)
                    if payload is None:
                        raise ValueError(f"cannot extract {member.name}")
                    raw = workspace / f"seed{seed}" / Path(member.name).name
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(payload.read())
                    destination_rel = Path("midi") / "emo_disentangler" / f"seed{seed}" / raw.name
                    destination = output_dir / destination_rel
                    audit = crop_midi(raw, destination, args.bars)
                    rows_out.append(
                        {
                            "system": "emo_disentangler",
                            "seed": seed,
                            "quadrant": quadrant(raw.name),
                            "sample_index": sample_index(raw.name),
                            "source_locator": member.name,
                            "source_sha256": sha256_file(raw),
                            "source_container": outer_member,
                            "source_container_sha256": nested_hash,
                            "standardized_midi": destination_rel.as_posix(),
                            "standardized_sha256": sha256_file(destination),
                            "crop_bars": args.bars,
                            **audit,
                        }
                    )

    validate_balance(rows_out)
    rows_out.sort(key=lambda row: (row["system"], int(row["seed"]), row["quadrant"], int(row["sample_index"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "schema": "moonmuse-corrected-three-system-population-v1",
        "status": "population_frozen_and_validated",
        "protocol": "v78",
        "bars": args.bars,
        "row_count": len(rows_out),
        "counts": {
            system: {
                str(seed): Counter(row["quadrant"] for row in rows_out if row["system"] == system and int(row["seed"]) == seed)
                for seed in EXPECTED_SEEDS
            }
            for system in ("moonmuse_full", "moonmuse_c2p", "emo_disentangler")
        },
        "input_hashes": {
            "corrected_manifest_sha256": sha256_file(corrected_root / "population_manifest.csv"),
            "v49_archive_sha256": sha256_file(args.v49_archive),
        },
        "population_manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "build_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--v49-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--expected-corrected-manifest-sha256")
    parser.add_argument("--expected-v49-sha256")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))

