"""Transposition-invariant motif memory for onset-monophonic functional events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import MotifRelation


BEATS_PER_BAR = 16
CELL_BARS = 2
MAJOR_DEGREES = {
    "I": 0,
    "I#": 1,
    "II": 2,
    "II#": 3,
    "III": 4,
    "IV": 5,
    "IV#": 6,
    "V": 7,
    "V#": 8,
    "VI": 9,
    "VI#": 10,
    "VII": 11,
}
MINOR_DEGREES = {
    "I": 0,
    "I#": 1,
    "II": 2,
    "II#": 3,
    "III": 3,
    "IV": 5,
    "IV#": 6,
    "V": 7,
    "V#": 8,
    "VI": 8,
    "VI#": 9,
    "VII": 10,
}


@dataclass(frozen=True)
class MelodyOnset:
    bar: int
    beat: int
    pitch: int
    degree: str

    @property
    def position(self) -> int:
        return self.bar * BEATS_PER_BAR + self.beat


@dataclass(frozen=True)
class CellMotif:
    slot_index: int
    start_bar: int
    end_bar: int
    onsets: tuple[MelodyOnset, ...]
    interval_tokens: tuple[str, ...]
    rhythm_gaps: tuple[int, ...]

    @property
    def onset_count(self) -> int:
        return len(self.onsets)

    def fingerprint(self) -> str:
        payload = {
            "onset_count": self.onset_count,
            "interval_tokens": self.interval_tokens,
            "rhythm_gaps": self.rhythm_gaps,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def transposed(self, semitones: int) -> "CellMotif":
        shifted = tuple(
            MelodyOnset(
                bar=onset.bar,
                beat=onset.beat,
                pitch=onset.pitch + semitones,
                degree=onset.degree,
            )
            for onset in self.onsets
        )
        return build_cell_motif(
            shifted,
            slot_index=self.slot_index,
            start_bar=self.start_bar,
            end_bar=self.end_bar,
        )


@dataclass(frozen=True)
class MotifComparison:
    source_slot: int
    target_slot: int
    interval_similarity: float
    rhythm_similarity: float
    combined_similarity: float
    relation: MotifRelation

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "source_slot": self.source_slot,
            "target_slot": self.target_slot,
            "interval_similarity": self.interval_similarity,
            "rhythm_similarity": self.rhythm_similarity,
            "combined_similarity": self.combined_similarity,
            "relation": self.relation.value,
        }


def scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def functional_pitch(key: str, octave: int, degree: str) -> int:
    mapping = MAJOR_DEGREES if key == key.upper() else MINOR_DEGREES
    if degree not in mapping:
        raise ValueError(f"unsupported functional degree: {degree!r}")
    return octave * 12 + mapping[degree]


def parse_melody_onsets(events: Iterable[dict]) -> tuple[str, list[MelodyOnset], int]:
    key = ""
    track: str | None = None
    melody_bar = -1
    beat = 0
    pending_octave: int | None = None
    by_position: dict[tuple[int, int], MelodyOnset] = {}
    for raw in events:
        name = str(raw.get("name"))
        value = scalar(raw.get("value"))
        if name == "Key":
            key = str(value)
        if name == "Track":
            track = str(value)
            pending_octave = None
            continue
        if track != "Melody":
            continue
        if name == "Bar":
            melody_bar += 1
            beat = 0
            pending_octave = None
        elif name == "Beat":
            beat = int(value)
            if not 0 <= beat < BEATS_PER_BAR:
                raise ValueError(f"beat outside 0..15: {beat}")
            pending_octave = None
        elif name == "Note_Octave":
            pending_octave = int(value)
        elif name == "Note_Degree":
            if melody_bar < 0:
                raise ValueError("note before first Melody Bar")
            if pending_octave is None:
                raise ValueError("Note_Degree without Note_Octave")
            if not key:
                raise ValueError("missing Key")
            degree = str(value)
            onset = MelodyOnset(
                bar=melody_bar,
                beat=beat,
                pitch=functional_pitch(key, pending_octave, degree),
                degree=degree,
            )
            position = (melody_bar, beat)
            if position in by_position:
                raise ValueError(f"non-monophonic Melody onset: {position}")
            by_position[position] = onset
            pending_octave = None
    if pending_octave is not None:
        raise ValueError("terminal Note_Octave without Note_Degree")
    if not key:
        raise ValueError("missing Key")
    return key, [by_position[key] for key in sorted(by_position)], melody_bar + 1


def interval_token(interval: int) -> str:
    if interval == 0:
        return "S0"
    direction = "U" if interval > 0 else "D"
    magnitude = abs(interval)
    if magnitude <= 2:
        band = "1"
    elif magnitude <= 5:
        band = "2"
    elif magnitude <= 11:
        band = "3"
    else:
        band = "4"
    return direction + band


def build_cell_motif(
    onsets: Sequence[MelodyOnset], *, slot_index: int, start_bar: int, end_bar: int
) -> CellMotif:
    selected = tuple(
        onset for onset in onsets if start_bar <= onset.bar < end_bar
    )
    pitches = [onset.pitch for onset in selected]
    positions = [onset.position for onset in selected]
    return CellMotif(
        slot_index=slot_index,
        start_bar=start_bar,
        end_bar=end_bar,
        onsets=selected,
        interval_tokens=tuple(
            interval_token(right - left) for left, right in zip(pitches, pitches[1:])
        ),
        rhythm_gaps=tuple(
            right - left for left, right in zip(positions, positions[1:])
        ),
    )


def extract_cell_motifs(
    onsets: Sequence[MelodyOnset], *, total_bars: int = 8
) -> tuple[CellMotif, ...]:
    if total_bars != 8:
        raise ValueError("the V97 motif audit is fixed to eight bars")
    return tuple(
        build_cell_motif(
            onsets,
            slot_index=slot,
            start_bar=slot * CELL_BARS,
            end_bar=(slot + 1) * CELL_BARS,
        )
        for slot in range(total_bars // CELL_BARS)
    )


def levenshtein_distance(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def sequence_similarity(left: Sequence[object], right: Sequence[object]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - levenshtein_distance(left, right) / denominator


def relation_from_score(
    score: float, *, variation_boundary: float = 0.55, return_boundary: float = 0.80
) -> MotifRelation:
    if not 0.0 <= variation_boundary <= return_boundary <= 1.0:
        raise ValueError("invalid motif relation boundaries")
    if score >= return_boundary:
        return MotifRelation.RETURN
    if score >= variation_boundary:
        return MotifRelation.VARIATE
    return MotifRelation.CONTRAST


def compare_motifs(
    source: CellMotif,
    target: CellMotif,
    *,
    variation_boundary: float = 0.55,
    return_boundary: float = 0.80,
) -> MotifComparison:
    interval_similarity = sequence_similarity(
        source.interval_tokens, target.interval_tokens
    )
    rhythm_similarity = sequence_similarity(source.rhythm_gaps, target.rhythm_gaps)
    combined = 0.65 * interval_similarity + 0.35 * rhythm_similarity
    return MotifComparison(
        source_slot=source.slot_index,
        target_slot=target.slot_index,
        interval_similarity=interval_similarity,
        rhythm_similarity=rhythm_similarity,
        combined_similarity=combined,
        relation=relation_from_score(
            combined,
            variation_boundary=variation_boundary,
            return_boundary=return_boundary,
        ),
    )


def cadence_target(onsets: Sequence[MelodyOnset], *, total_bars: int = 8) -> str | None:
    retained = [onset for onset in onsets if onset.bar < total_bars]
    if not retained:
        return None
    degree = retained[-1].degree
    if degree == "I":
        return "TONIC"
    if degree in {"V", "V#"}:
        return "DOMINANT"
    return "OTHER"
