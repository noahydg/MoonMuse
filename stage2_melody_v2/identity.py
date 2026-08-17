"""Deterministic motif-identity views and synthetic transformations for V105.

The functions in this module are deliberately symbolic and CPU-only.  They do not
claim that two motifs are perceptually identical; they construct a falsifiable data
and representation gate before a learned identity encoder is considered.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Any, Iterable, Sequence

from .motif_memory import CellMotif, interval_token, sequence_similarity


CELL_STEPS = 32
OPERATIONS = ("transpose", "invert", "time_scale", "rhythm_shift", "fragment")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_index(material: str, size: int) -> int:
    if size <= 0:
        raise ValueError("hash choice requires a non-empty population")
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


@dataclass(frozen=True)
class IdentityMotif:
    """A two-bar onset motif in absolute pitch and cell-relative time."""

    pitches: tuple[int, ...]
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.pitches) != len(self.positions):
            raise ValueError("motif pitch/position lengths differ")
        if len(self.pitches) < 3:
            raise ValueError("identity motif requires at least three onsets")
        if any(not 0 <= pitch <= 127 for pitch in self.pitches):
            raise ValueError("motif pitch is outside MIDI range")
        if any(not 0 <= position < CELL_STEPS for position in self.positions):
            raise ValueError("motif position is outside the two-bar cell")
        if any(right <= left for left, right in zip(self.positions, self.positions[1:])):
            raise ValueError("motif positions must be strictly increasing")

    @property
    def onset_count(self) -> int:
        return len(self.pitches)

    @property
    def intervals(self) -> tuple[int, ...]:
        return tuple(
            right - left for left, right in zip(self.pitches, self.pitches[1:])
        )

    @property
    def gaps(self) -> tuple[int, ...]:
        return tuple(
            right - left for left, right in zip(self.positions, self.positions[1:])
        )

    @property
    def relative_pitches(self) -> tuple[int, ...]:
        first = self.pitches[0]
        return tuple(pitch - first for pitch in self.pitches)

    @property
    def relative_positions(self) -> tuple[int, ...]:
        first = self.positions[0]
        return tuple(position - first for position in self.positions)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "pitches": list(self.pitches),
            "positions": list(self.positions),
            "relative_pitches": list(self.relative_pitches),
            "relative_positions": list(self.relative_positions),
        }


@dataclass(frozen=True)
class TransformResult:
    operation: str
    motif: IdentityMotif
    parameters: dict[str, Any]


def identity_motif_from_cell(cell: CellMotif) -> IdentityMotif:
    cell_start = cell.start_bar * 16
    return IdentityMotif(
        pitches=tuple(onset.pitch for onset in cell.onsets),
        positions=tuple(onset.position - cell_start for onset in cell.onsets),
    )


def normalized_gaps(motif: IdentityMotif) -> tuple[int, ...]:
    gaps = motif.gaps
    divisor = reduce(gcd, gaps) if gaps else 1
    return tuple(gap // divisor for gap in gaps)


def canonical_identity_signature(motif: IdentityMotif) -> str:
    """Hash a full-motif family invariant to the four exact V105 transforms.

    Fragmentation intentionally changes this full-motif signature.  The signature is
    used only to prevent exact full-family duplicates from becoming hard negatives.
    """

    direct = motif.intervals
    inverted = tuple(-value for value in direct)
    canonical_pitch = min(direct, inverted)
    return sha256_json(
        {
            "intervals_up_to_inversion": canonical_pitch,
            "gaps_up_to_integer_scale": normalized_gaps(motif),
        }
    )


def _transpose(motif: IdentityMotif, material: str) -> TransformResult | None:
    shifts = [-5, -2, 2, 5]
    start = _hash_index(material, len(shifts))
    ordered = shifts[start:] + shifts[:start]
    for shift in ordered:
        pitches = tuple(pitch + shift for pitch in motif.pitches)
        if all(0 <= pitch <= 127 for pitch in pitches):
            return TransformResult(
                operation="transpose",
                motif=IdentityMotif(pitches=pitches, positions=motif.positions),
                parameters={"semitones": shift},
            )
    return None


def _invert(motif: IdentityMotif) -> TransformResult | None:
    axis = motif.pitches[0]
    pitches = tuple(2 * axis - pitch for pitch in motif.pitches)
    if pitches == motif.pitches or any(not 0 <= pitch <= 127 for pitch in pitches):
        return None
    return TransformResult(
        operation="invert",
        motif=IdentityMotif(pitches=pitches, positions=motif.positions),
        parameters={"axis_pitch": axis},
    )


def v105_inversion_status(motif: IdentityMotif) -> str:
    """Classify why the original V105 absolute-register inversion is eligible."""

    axis = motif.pitches[0]
    pitches = tuple(2 * axis - pitch for pitch in motif.pitches)
    if pitches == motif.pitches:
        return "unchanged_all_same_pitch"
    if any(not 0 <= pitch <= 127 for pitch in pitches):
        return "out_of_midi_range"
    return "eligible"


def register_normalized_inversion(motif: IdentityMotif) -> TransformResult | None:
    """Invert, then apply one global shift if absolute register leaves MIDI range.

    This V106 correction preserves the exact inverted interval sequence.  It never
    clips or folds individual notes.
    """

    axis = motif.pitches[0]
    reflected = tuple(2 * axis - pitch for pitch in motif.pitches)
    if reflected == motif.pitches:
        return None
    minimum_shift = -min(reflected)
    maximum_shift = 127 - max(reflected)
    if minimum_shift > maximum_shift:
        return None
    shifts = range(minimum_shift, maximum_shift + 1)
    shift = min(shifts, key=lambda value: (abs(value), value))
    pitches = tuple(pitch + shift for pitch in reflected)
    return TransformResult(
        operation="invert",
        motif=IdentityMotif(pitches=pitches, positions=motif.positions),
        parameters={
            "axis_pitch": axis,
            "register_shift_semitones": shift,
            "register_normalized": shift != 0,
        },
    )


def _time_scale(motif: IdentityMotif, material: str) -> TransformResult | None:
    anchor = motif.positions[0]
    deltas = tuple(position - anchor for position in motif.positions)
    candidates: list[tuple[str, tuple[int, ...], dict[str, int]]] = []

    augmented = tuple(anchor + 2 * delta for delta in deltas)
    if augmented[-1] < CELL_STEPS and augmented != motif.positions:
        candidates.append(
            (
                "augmentation",
                augmented,
                {"factor_numerator": 2, "factor_denominator": 1},
            )
        )

    if all(delta % 2 == 0 for delta in deltas):
        diminished = tuple(anchor + delta // 2 for delta in deltas)
        if len(set(diminished)) == len(diminished) and diminished != motif.positions:
            candidates.append(
                (
                    "diminution",
                    diminished,
                    {"factor_numerator": 1, "factor_denominator": 2},
                )
            )

    if not candidates:
        return None
    mode, positions, parameters = candidates[_hash_index(material, len(candidates))]
    return TransformResult(
        operation="time_scale",
        motif=IdentityMotif(pitches=motif.pitches, positions=positions),
        parameters={"mode": mode, **parameters},
    )


def _rhythm_shift(motif: IdentityMotif, material: str) -> TransformResult | None:
    minimum_delta = -motif.positions[0]
    maximum_delta = (CELL_STEPS - 1) - motif.positions[-1]
    shifts = [
        delta
        for delta in range(minimum_delta, maximum_delta + 1)
        if delta != 0
    ]
    if not shifts:
        return None
    delta = shifts[_hash_index(material, len(shifts))]
    positions = tuple(position + delta for position in motif.positions)
    return TransformResult(
        operation="rhythm_shift",
        motif=IdentityMotif(pitches=motif.pitches, positions=positions),
        parameters={"sixteenth_steps": delta},
    )


def _fragment(motif: IdentityMotif, material: str) -> TransformResult | None:
    retained = max(3, math.ceil(0.67 * motif.onset_count))
    if retained >= motif.onset_count:
        return None
    use_suffix = bool(_hash_index(material, 2))
    if use_suffix:
        pitches = motif.pitches[-retained:]
        positions = motif.positions[-retained:]
        side = "suffix"
    else:
        pitches = motif.pitches[:retained]
        positions = motif.positions[:retained]
        side = "prefix"
    return TransformResult(
        operation="fragment",
        motif=IdentityMotif(pitches=pitches, positions=positions),
        parameters={"side": side, "retained_onsets": retained},
    )


def transform_motif(
    motif: IdentityMotif, *, operation: str, material: str
) -> TransformResult | None:
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported V105 operation: {operation}")
    salted = f"{material}|{operation}"
    if operation == "transpose":
        return _transpose(motif, salted)
    if operation == "invert":
        return _invert(motif)
    if operation == "time_scale":
        return _time_scale(motif, salted)
    if operation == "rhythm_shift":
        return _rhythm_shift(motif, salted)
    return _fragment(motif, salted)


def lcs_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def containment_similarity(left: Sequence[Any], right: Sequence[Any]) -> float:
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 1.0 if len(left) == len(right) else 0.0
    return lcs_length(left, right) / denominator


def legacy_similarity(left: IdentityMotif, right: IdentityMotif) -> float:
    left_bands = tuple(interval_token(value) for value in left.intervals)
    right_bands = tuple(interval_token(value) for value in right.intervals)
    return 0.65 * sequence_similarity(left_bands, right_bands) + 0.35 * sequence_similarity(
        left.gaps, right.gaps
    )


def identity_reference_similarity(left: IdentityMotif, right: IdentityMotif) -> float:
    direct_pitch = containment_similarity(left.intervals, right.intervals)
    inverted_pitch = containment_similarity(
        left.intervals, tuple(-value for value in right.intervals)
    )
    pitch_score = max(direct_pitch, inverted_pitch)
    rhythm_score = containment_similarity(normalized_gaps(left), normalized_gaps(right))
    return 0.65 * pitch_score + 0.35 * rhythm_score


def median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def transition_features_from_view(
    view: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    """Convert a serialized V105/V106 motif view to the frozen V107 features."""

    pitches = tuple(int(value) for value in view["relative_pitches"])
    positions = tuple(int(value) for value in view["relative_positions"])
    if len(pitches) != len(positions) or len(pitches) < 3:
        raise ValueError("V107 motif view requires at least three aligned onsets")
    if positions[0] != 0:
        raise ValueError("V107 positions must be relative to the first onset")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("V107 relative positions must be strictly increasing")
    intervals = tuple(
        right - left for left, right in zip(pitches, pitches[1:])
    )
    gaps = tuple(right - left for left, right in zip(positions, positions[1:]))
    divisor = reduce(gcd, gaps)
    normalized = tuple(gap // divisor for gap in gaps)
    return [
        (
            max(-48, min(48, interval)) / 48.0,
            min(48, abs(interval)) / 48.0,
            gap / 31.0,
            normalized_gap / 31.0,
        )
        for interval, gap, normalized_gap in zip(intervals, gaps, normalized)
    ]
