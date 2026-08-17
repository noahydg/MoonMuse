#!/usr/bin/env python3
"""Audit how much of the Stage 2/3 training target is PAD-masked.

The Stage 2 and Stage 3 dataloaders build a 768-token target in which only the
generated track is supervised. Every other position uses the vocabulary's PAD
token and must be ignored by cross entropy. This script mirrors the dataloader's
window construction without importing PyTorch so the masking ratio can be
audited locally and on the training server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "moonmuse-training-target-mask-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values")
    return float(ordered[round((len(ordered) - 1) * probability)])


def admissible_start_bars(
    event_count: int,
    condition_positions: list[tuple[int, int]],
    max_len: int,
) -> list[int]:
    if event_count <= max_len:
        return [0]
    starts = []
    for bar, (start, _end) in enumerate(condition_positions):
        if event_count - start >= 0.5 * max_len:
            starts.append(bar)
        else:
            break
    if not starts:
        raise ValueError("long piece has no admissible start bar")
    return starts


def count_target_positions(
    target_positions: list[tuple[int, int]],
    start_bar: int,
    offset: int,
    max_len: int,
) -> int:
    total = 0
    for start, end in target_positions[start_bar:]:
        clipped_start = max(0, start + offset)
        clipped_end = min(max_len, end + offset)
        if clipped_end > clipped_start:
            total += clipped_end - clipped_start
    return total


def audit_piece(payload, filename: str, max_len: int) -> list[dict]:
    if not isinstance(payload, (list, tuple)) or len(payload) < 3:
        raise ValueError(f"unexpected event payload: {filename}")
    condition_positions, target_positions, events = payload[:3]
    if not condition_positions or len(condition_positions) != len(target_positions):
        raise ValueError(f"invalid track positions: {filename}")
    prefix_length = int(condition_positions[0][0])
    rows = []
    starts = admissible_start_bars(len(events), condition_positions, max_len)
    for start_bar in starts:
        offset = -int(condition_positions[start_bar][0]) + prefix_length
        target_count = count_target_positions(
            target_positions, start_bar, offset, max_len
        )
        pad_target_count = max_len - target_count
        input_length = prefix_length + len(events) - int(condition_positions[start_bar][0])
        rows.append(
            {
                "filename": filename,
                "quadrant": filename.split("_", 1)[0],
                "bar_count": len(condition_positions),
                "start_bar": start_bar,
                "admissible_start_count": len(starts),
                "raw_event_count": len(events),
                "input_token_count": min(input_length, max_len),
                "target_token_count": target_count,
                "pad_target_count": pad_target_count,
                "target_token_ratio": target_count / max_len,
                "pad_target_ratio": pad_target_count / max_len,
            }
        )
    return rows


def summarize(rows: list[dict], piece_count: int) -> dict:
    target_ratios = [float(row["target_token_ratio"]) for row in rows]
    pad_ratios = [float(row["pad_target_ratio"]) for row in rows]
    windows_per_piece = Counter(row["filename"] for row in rows)
    by_quadrant = defaultdict(list)
    for row in rows:
        by_quadrant[row["quadrant"]].append(float(row["pad_target_ratio"]))
    return {
        "piece_count": piece_count,
        "admissible_window_count": len(rows),
        "pieces_with_multiple_admissible_windows": sum(
            count > 1 for count in windows_per_piece.values()
        ),
        "target_token_ratio": {
            "mean": statistics.fmean(target_ratios),
            "p10": observed_quantile(target_ratios, 0.10),
            "p50": observed_quantile(target_ratios, 0.50),
            "p90": observed_quantile(target_ratios, 0.90),
        },
        "pad_target_ratio": {
            "mean": statistics.fmean(pad_ratios),
            "p10": observed_quantile(pad_ratios, 0.10),
            "p50": observed_quantile(pad_ratios, 0.50),
            "p90": observed_quantile(pad_ratios, 0.90),
        },
        "mean_pad_target_ratio_by_quadrant": {
            quadrant: statistics.fmean(values)
            for quadrant, values in sorted(by_quadrant.items())
        },
    }


def audit(event_dir: Path, split_path: Path, max_len: int) -> dict:
    filenames = pickle.loads(split_path.read_bytes())
    if not isinstance(filenames, list) or not filenames:
        raise ValueError("split must be a non-empty filename list")
    rows = []
    for filename in filenames:
        path = event_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(audit_piece(pickle.loads(path.read_bytes()), filename, max_len))
    return {
        "schema": SCHEMA,
        "formalStudyData": False,
        "descriptiveOnly": True,
        "event_dir": str(event_dir.resolve()),
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "max_len": max_len,
        "summary": summarize(rows, len(filenames)),
        "rows": rows,
        "interpretation_boundary": (
            "PAD-target prevalence quantifies the exposure of an unmasked loss. "
            "It does not by itself establish audible improvement after correction; "
            "the corrected objective requires a prospective training and generation gate."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["filename"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_len < 1:
        parser.error("--max-len must be positive")
    report = audit(args.event_dir, args.split, args.max_len)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_mask_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "target_mask_windows.csv", report["rows"])
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
