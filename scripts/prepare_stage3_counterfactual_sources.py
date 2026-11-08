#!/usr/bin/env python3
"""Freeze Stage 2 melody sources for a Stage 3 arousal counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


QUADRANT_RE = re.compile(r"_(Q[1-4])_melody\.txt$")
QUADRANT_ATTRIBUTES = {
    "Q1": ("positive", "high"),
    "Q2": ("negative", "high"),
    "Q3": ("negative", "low"),
    "Q4": ("positive", "low"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_source_argument(value: str) -> tuple[int, Path]:
    try:
        seed_text, directory = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "source must be written as SEED=/path/to/stage2_melody"
        ) from exc
    return seed, Path(directory)


def discover_candidates(seed: int, source_dir: Path) -> list[dict]:
    candidates = []
    for path in sorted(source_dir.glob("*.txt")):
        match = QUADRANT_RE.search(path.name)
        if not match:
            continue
        quadrant = match.group(1)
        payload = path.read_bytes()
        valence, arousal = QUADRANT_ATTRIBUTES[quadrant]
        candidates.append(
            {
                "seed": seed,
                "source_quadrant": quadrant,
                "source_valence": valence,
                "source_arousal": arousal,
                "source_path": path,
                "source_sha256": sha256_bytes(payload),
                "payload": payload,
            }
        )
    return candidates


def select_sources(
    sources: list[tuple[int, Path]], expected_per_quadrant: int
) -> list[dict]:
    selected = []
    seen_seeds = set()
    for seed, source_dir in sources:
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seen_seeds.add(seed)
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        candidates = discover_candidates(seed, source_dir)
        for quadrant in QUADRANT_ATTRIBUTES:
            group = [row for row in candidates if row["source_quadrant"] == quadrant]
            if len(group) != expected_per_quadrant:
                raise ValueError(
                    f"seed {seed} {quadrant}: expected {expected_per_quadrant} "
                    f"candidates, found {len(group)}"
                )
            selected.append(
                min(group, key=lambda row: (row["source_sha256"], row["source_path"].name))
            )
    return sorted(selected, key=lambda row: (row["seed"], row["source_quadrant"]))


def freeze_sources(selected: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for row in selected:
        valence = row["source_valence"]
        destination_dir = output_dir / "sources" / f"seed{row['seed']}" / valence
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_name = (
            f"source_{row['source_quadrant']}_{row['source_sha256'][:12]}_melody.txt"
        )
        destination = destination_dir / destination_name
        destination.write_bytes(row["payload"])
        if sha256_bytes(destination.read_bytes()) != row["source_sha256"]:
            raise RuntimeError(f"copied source hash mismatch: {destination}")
        manifest_rows.append(
            {
                "seed": row["seed"],
                "source_quadrant": row["source_quadrant"],
                "source_valence": valence,
                "source_arousal": row["source_arousal"],
                "source_filename": row["source_path"].name,
                "source_sha256": row["source_sha256"],
                "frozen_path": str(destination.relative_to(output_dir)),
                "target_quadrants": (
                    ["Q1", "Q4"] if valence == "positive" else ["Q2", "Q3"]
                ),
            }
        )

    seeds = sorted({row["seed"] for row in manifest_rows})
    manifest = {
        "schema": "moonmuse-stage3-arousal-counterfactual-sources-v1",
        "status": "sources_frozen_before_counterfactual_generation",
        "formalStudyData": False,
        "selection_rule": "lowest_sha256_within_model_seed_and_source_quadrant",
        "source_count": len(manifest_rows),
        "expected_output_count": len(manifest_rows) * 2,
        "seeds": seeds,
        "sources": manifest_rows,
        "interpretation": (
            "Each source is rendered under both arousal targets sharing its valence. "
            "No source was selected by listening quality."
        ),
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source_argument,
        metavar="SEED=DIR",
        help="repeat for each model seed; DIR must contain raw Stage 2 melody txt files",
    )
    parser.add_argument("--expected-per-quadrant", type=int, default=25)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.expected_per_quadrant <= 0:
        parser.error("--expected-per-quadrant must be positive")
    try:
        selected = select_sources(args.source, args.expected_per_quadrant)
        manifest = freeze_sources(selected, args.output_dir)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
