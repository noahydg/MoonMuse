#!/usr/bin/env python3
"""Build MoonMuse chord, chord-to-melody, and melody-to-performance events.

The source archive is the processed ``events.zip`` published with
EMO-Disentangler. The script preserves the official train/validation splits
and records source hashes and output counts for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = REPO_ROOT / "representations"
sys.path.insert(0, str(REPRESENTATIONS))

HOOKTHEORY_PROVENANCE = {
    "upstream_dataset": "Sheet Sage HookTheory dataset",
    "upstream_url": "https://github.com/chrisdonahue/sheetsage#hooktheory-dataset",
    "upstream_license": "CC BY-NC-SA 3.0",
    "upstream_citation": "Donahue, Thickstun, and Liang, ISMIR 2022",
    "processing_reference": "EMO-Disentangler HookTheory representation pipeline",
}

from events2words import events2dictionary  # noqa: E402
from pkl_event import modify_from_pkl_dir  # noqa: E402


@dataclass(frozen=True)
class Source:
    name: str
    event_prefix: str
    train_split: str
    valid_split: str
    event_pos: int


@dataclass(frozen=True)
class Target:
    source: str
    relative_root: str
    mode: str


SOURCES = {
    "hooktheory": Source(
        name="hooktheory",
        event_prefix=(
            "events/stage1/hooktheory_events/"
            "lead_sheet_chord11_functional/events/"
        ),
        train_split="events/stage1/hooktheory_events/data_splits/train.pkl",
        valid_split="events/stage1/hooktheory_events/data_splits/valid.pkl",
        event_pos=1,
    ),
    "emopia": Source(
        name="emopia",
        event_prefix=(
            "events/stage2/emopia_events/"
            "full_song_chord11_functional/events/"
        ),
        train_split="events/stage2/emopia_events/data_splits/train.pkl",
        valid_split="events/stage2/emopia_events/data_splits/valid.pkl",
        event_pos=2,
    ),
    "pop1k7": Source(
        name="pop1k7",
        event_prefix=(
            "events/stage2/pop1k7_events/"
            "full_song_chorder_functional/events/"
        ),
        train_split="events/stage2/pop1k7_events/data_splits/train.pkl",
        valid_split="events/stage2/pop1k7_events/data_splits/valid.pkl",
        event_pos=2,
    ),
}


TARGETS = (
    Target("hooktheory", "chord/hooktheory_chord_functional", "chord"),
    Target("emopia", "chord/emopia_chord_functional", "chord"),
    Target("pop1k7", "chord/pop1k7_chord_functional", "chord"),
    Target("emopia", "c2m/emopia_c2m_functional", "c2m"),
    Target("pop1k7", "c2m/pop1k7_c2m_functional", "c2m"),
    Target("emopia", "m2p/emopia_m2p_functional", "m2p"),
    Target("pop1k7", "m2p/pop1k7_m2p_functional", "m2p"),
)


DICTIONARY_OPTIONS = {
    "chord": dict(
        add_chord=True,
        check=True,
        relative=None,
        event_pos=1,
        num_emotion=2,
        add_emotion=True,
        add_tempo=False,
        add_velocity=False,
    ),
    "c2m": dict(
        add_chord=False,
        check=False,
        relative=True,
        event_pos=2,
        num_emotion=4,
        add_emotion=True,
        add_tempo=False,
        add_velocity=False,
    ),
    "m2p": dict(
        add_chord=False,
        check=False,
        relative=True,
        event_pos=2,
        num_emotion=4,
        add_emotion=True,
        add_tempo=True,
        add_velocity=True,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(archive: zipfile.ZipFile, member: str) -> list[str]:
    return list(pickle.loads(archive.read(member)))


def selected_split(
    archive: zipfile.ZipFile,
    member: str,
    available: set[str],
    limit_per_split: int | None,
) -> list[str]:
    names = [name for name in load_split(archive, member) if name in available]
    if limit_per_split is not None:
        names = names[:limit_per_split]
    return names


def extract_selected(
    archive: zipfile.ZipFile,
    source: Source,
    destination: Path,
    selected: set[str],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    members = {
        Path(name).name: name
        for name in archive.namelist()
        if name.startswith(source.event_prefix) and name.endswith(".pkl")
    }
    missing = sorted(selected - members.keys())
    if missing:
        raise RuntimeError(
            f"{source.name}: {len(missing)} split files are absent from the archive; "
            f"first missing file: {missing[0]}"
        )
    for filename in sorted(selected):
        with archive.open(members[filename]) as src, (destination / filename).open("wb") as dst:
            shutil.copyfileobj(src, dst)


def write_splits(root: Path, train: list[str], valid: list[str]) -> None:
    split_dir = root / "data_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "train.pkl").open("wb") as handle:
        pickle.dump(train, handle)
    with (split_dir / "valid.pkl").open("wb") as handle:
        pickle.dump(valid, handle)


def ensure_clean_targets(output_root: Path, force: bool) -> None:
    populated = []
    for target in TARGETS:
        root = output_root / target.relative_root
        if root.exists() and any(root.rglob("*")):
            populated.append(root)
    if populated and not force:
        paths = "\n  ".join(str(path) for path in populated)
        raise RuntimeError(
            "Refusing to overwrite existing prepared data. Use --force after "
            f"reviewing these paths:\n  {paths}"
        )
    if force:
        for target in TARGETS:
            shutil.rmtree(output_root / target.relative_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-events-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "events")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="prepare a deterministic subset for local validation",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive_path = args.official_events_zip.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")

    ensure_clean_targets(output_root, args.force)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_archive": str(archive_path),
        "source_sha256": sha256(archive_path),
        "data_license_notice": "See DATA_LICENSES.md; data is not MIT-licensed.",
        "hooktheory_provenance": HOOKTHEORY_PROVENANCE,
        "limit_per_split": args.limit_per_split,
        "targets": {},
    }

    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        source_state = {}
        for source in SOURCES.values():
            available = {
                Path(name).name
                for name in archive.namelist()
                if name.startswith(source.event_prefix) and name.endswith(".pkl")
            }
            train = selected_split(
                archive, source.train_split, available, args.limit_per_split
            )
            valid = selected_split(
                archive, source.valid_split, available, args.limit_per_split
            )
            selected = set(train) | set(valid)
            source_dir = temp_root / source.name
            extract_selected(archive, source, source_dir, selected)
            source_state[source.name] = (source_dir, train, valid)

        for target in TARGETS:
            source = SOURCES[target.source]
            source_dir, train, valid = source_state[target.source]
            target_root = output_root / target.relative_root
            event_dir = target_root / "events"
            modify_from_pkl_dir(
                str(source_dir),
                str(event_dir),
                track_name="Full",
                mode=target.mode,
                pos=source.event_pos,
            )
            write_splits(target_root, train, valid)
            events2dictionary(str(target_root), **DICTIONARY_OPTIONS[target.mode])
            count = len(list(event_dir.glob("*.pkl")))
            manifest["targets"][target.relative_root] = {
                "source": target.source,
                "mode": target.mode,
                "events": count,
                "train": len(train),
                "valid": len(valid),
            }
            print(
                f"[ready] {target.relative_root}: "
                f"events={count}, train={len(train)}, valid={len(valid)}"
            )

    manifest_path = output_root / "moonmuse_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
