"""Pure contracts for the frozen V103 causal sequential-generation gate."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .conditioning import INTERVAL_TO_ID, MAX_RHYTHM_GAP, encode_realizer_condition
from .motif_memory import (
    CellMotif,
    build_cell_motif,
    extract_cell_motifs,
    parse_melody_onsets,
)
from .online_memory import PUBLIC_FIELDS
from .planner import build_eight_bar_plan


GENERATION_SCHEMA = "moonmuse-stage2-v2-v103-sequential-generation-v1"
CONDITION_SCHEMA = "moonmuse-stage2-v2-realizer-condition-v1"
VARIANTS = ("full_memory", "zero_memory", "backbone_only")
TARGET_BARS = 8
CELL_BARS = 2
MIN_FIRST8_ONSETS = 16
MIN_CELL_ONSETS = 3
SEED_NAMESPACE = "v103-generation"
REPLAY_NAMESPACE = "v103-replay"

SOURCE_HASHES = {
    "samp_00_Negative.txt": "3c2e423071636b9314982386c29f3f8e48028de2fc4f717e5a96f5fa95d684cc",
    "samp_00_Positive.txt": "a91cec7e590ddb93232fef45b98b61c92944253688902ce69f487e85c8259058",
    "samp_01_Negative.txt": "31412a0c813405bf4ca1603b265085e5e564a2a013c7b3c6e3f1a379bf36b5cc",
    "samp_01_Positive.txt": "b251850505e69b351709a4cfed4fd4c1de74ec1f51af86d6565c6f5c3b1c0bdb",
}

TOKEN_PREFIXES = (
    "Chord_Quality_",
    "Chord_Root_",
    "Note_Octave_",
    "Note_Degree_",
    "Emotion_",
    "Track_",
    "Beat_",
    "Bar_",
    "Key_",
    "EOS_",
    "PAD_",
    "Pad_",
)
INTEGER_EVENT_NAMES = frozenset(("Beat", "Note_Octave"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenChordSource:
    path: Path
    filename: str
    source_sha256: str
    key_token: str
    chord_bars: tuple[tuple[str, ...], ...]


def quadrants_for_source(filename: str) -> tuple[str, str]:
    if "Negative" in filename:
        return ("Q2", "Q3")
    if "Positive" in filename:
        return ("Q1", "Q4")
    raise ValueError("V103 source lacks frozen Positive/Negative stratum")


def generation_seed(filename: str, quadrant: str, *, base_seed: int = 103) -> int:
    if quadrant not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("unsupported V103 quadrant")
    payload = f"{SEED_NAMESPACE}|{base_seed}|{filename}|{quadrant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def read_frozen_chord_source(
    path: Path,
    *,
    expected_sha256: str,
    target_bars: int = TARGET_BARS,
) -> FrozenChordSource:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing V103 chord source: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"V103 chord-source hash drift: {path.name}")
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens or not tokens[0].startswith("Key_"):
        raise ValueError(f"V103 chord source lacks a leading key: {path.name}")
    bar_starts = [index for index, token in enumerate(tokens) if token == "Bar_None"]
    if len(bar_starts) < target_bars:
        raise ValueError(f"V103 chord source has fewer than {target_bars} bars")
    bar_starts.append(len(tokens))
    bars = tuple(
        tuple(tokens[bar_starts[index] : bar_starts[index + 1]])
        for index in range(target_bars)
    )
    if any(not bar or bar[0] != "Bar_None" for bar in bars):
        raise ValueError("malformed V103 chord bar")
    return FrozenChordSource(
        path=path,
        filename=path.name,
        source_sha256=actual,
        key_token=tokens[0],
        chord_bars=bars,
    )


def empty_online_condition(quadrant: str) -> dict[str, Any]:
    motifs = extract_cell_motifs(())
    encoded = encode_realizer_condition(build_eight_bar_plan(quadrant), motifs).to_dict()
    condition = {field: encoded[field] for field in PUBLIC_FIELDS}
    condition = json.loads(canonical_json(condition))
    if set(condition) != set(PUBLIC_FIELDS):
        raise AssertionError("V103 condition fields drifted")
    if condition["schema"] != CONDITION_SCHEMA:
        raise AssertionError("V103 condition schema drifted")
    return condition


def parse_event_token(token: str) -> dict[str, Any]:
    for prefix in TOKEN_PREFIXES:
        if not token.startswith(prefix):
            continue
        name = prefix[:-1]
        raw = token[len(prefix) :]
        if name in INTEGER_EVENT_NAMES:
            value: Any = int(raw)
        elif raw == "None" and name in {"Bar", "EOS", "PAD", "Pad"}:
            value = None
        else:
            value = raw
        return {"name": name, "value": value}
    raise ValueError(f"unsupported Stage 2 token: {token!r}")


def tokens_to_events(tokens: Iterable[str]) -> list[dict[str, Any]]:
    return [parse_event_token(token) for token in tokens]


def melody_bar_token_groups(tokens: Sequence[str]) -> list[list[str]]:
    track: str | None = None
    bars: list[list[str]] = []
    current: list[str] | None = None
    for token in tokens:
        if token.startswith("Track_"):
            track = token[len("Track_") :]
            if track != "Melody":
                current = None
            continue
        if track != "Melody":
            continue
        if token == "Bar_None":
            current = [token]
            bars.append(current)
        elif current is not None:
            current.append(token)
    return bars


def update_observed_motif(
    condition: Mapping[str, Any],
    *,
    decoded_tokens: Sequence[str],
    completed_slot: int,
    apply_to_condition: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract one completed generated cell and optionally expose it online."""

    if completed_slot not in (0, 1, 2):
        raise ValueError("V103 updates memory only after A, A-prime and B")
    output = copy.deepcopy(dict(condition))
    events = tokens_to_events(decoded_tokens)
    key, onsets, bar_count = parse_melody_onsets(events)
    expected_bar_count = (completed_slot + 1) * CELL_BARS
    if bar_count != expected_bar_count:
        raise ValueError(
            f"memory update expected {expected_bar_count} completed melody bars, got {bar_count}"
        )
    motif: CellMotif = build_cell_motif(
        onsets,
        slot_index=completed_slot,
        start_bar=completed_slot * CELL_BARS,
        end_bar=(completed_slot + 1) * CELL_BARS,
    )
    interval_ids = [INTERVAL_TO_ID[token] for token in motif.interval_tokens]
    rhythm_ids = [min(MAX_RHYTHM_GAP, max(1, gap)) for gap in motif.rhythm_gaps]
    if apply_to_condition:
        output["motif_interval_ids"][completed_slot] = interval_ids
        output["motif_rhythm_ids"][completed_slot] = rhythm_ids
        output["motif_lengths"][completed_slot] = len(interval_ids)
    trace = {
        "completed_slot": completed_slot,
        "completed_bars": expected_bar_count,
        "key": key,
        "onset_count": motif.onset_count,
        "interval_ids": interval_ids,
        "rhythm_ids": rhythm_ids,
        "motif_length": len(interval_ids),
        "motif_fingerprint": motif.fingerprint(),
        "applied_to_condition": apply_to_condition,
        "condition_sha256_after_update": canonical_sha256(output),
    }
    return output, trace


def validate_generated_tokens(tokens: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "parse_error": "",
        "bars": 0,
        "first8_onset_count": 0,
        "cell_onset_counts": [],
        "onset_monophonic": False,
        "passed": False,
    }
    try:
        events = tokens_to_events(tokens)
        _, onsets, bars = parse_melody_onsets(events)
        first8 = [onset for onset in onsets if onset.bar < TARGET_BARS]
        motifs = extract_cell_motifs(first8)
        counts = [motif.onset_count for motif in motifs]
        report.update(
            {
                "bars": bars,
                "first8_onset_count": len(first8),
                "cell_onset_counts": counts,
                "onset_monophonic": True,
                "passed": (
                    bars == TARGET_BARS
                    and len(first8) >= MIN_FIRST8_ONSETS
                    and min(counts) >= MIN_CELL_ONSETS
                ),
            }
        )
    except Exception as exc:  # The report preserves the only attempted output.
        report["parse_error"] = f"{type(exc).__name__}: {exc}"
    return report


def first_cell_tokens(tokens: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    bars = melody_bar_token_groups(tokens)
    if len(bars) < CELL_BARS:
        raise ValueError("generated output lacks the first two melody bars")
    return tuple(bars[0]), tuple(bars[1])


def later_cells_differ(left: Sequence[str], right: Sequence[str]) -> bool:
    left_bars = melody_bar_token_groups(left)
    right_bars = melody_bar_token_groups(right)
    if len(left_bars) != TARGET_BARS or len(right_bars) != TARGET_BARS:
        return False
    return left_bars[CELL_BARS:] != right_bars[CELL_BARS:]


def select_replay_keys(relative_paths: Iterable[str], *, count: int = 2) -> list[str]:
    ranked = sorted(
        relative_paths,
        key=lambda path: hashlib.sha256(
            f"{REPLAY_NAMESPACE}|{path}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError("not enough V103 outputs for replay selection")
    return ranked[:count]
