"""Operation-aware canonicalization for directional motif containment.

This module is an experimental V107-C2 engineering primitive.  It has no Stage 2
call site and performs no learned scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OperationTransform:
    """Known operation applied to a partial motif query."""

    invert: bool = False
    time_scale_mode: str | None = None

    def __post_init__(self) -> None:
        if self.time_scale_mode not in {None, "double", "half"}:
            raise ValueError(
                "time_scale_mode must be None, 'double', or 'half'"
            )


def _integer_sequence(values: Sequence[Any], *, name: str) -> list[int]:
    output = []
    for value in values:
        integer = int(value)
        if integer != value:
            raise ValueError(f"{name} must contain exact integers")
        output.append(integer)
    return output


def validate_view(view: Mapping[str, Any]) -> dict[str, list[int]]:
    """Validate and copy the two fields consumed by motif identity features."""

    pitches = _integer_sequence(view["relative_pitches"], name="relative_pitches")
    positions = _integer_sequence(
        view["relative_positions"], name="relative_positions"
    )
    if len(pitches) != len(positions) or len(pitches) < 3:
        raise ValueError("motif view requires at least three aligned onsets")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("relative positions must be strictly increasing")
    return {
        "relative_pitches": pitches,
        "relative_positions": positions,
    }


def rebase_view(view: Mapping[str, Any]) -> dict[str, list[int]]:
    """Move the first pitch and onset to zero without changing intervals/gaps."""

    validated = validate_view(view)
    pitch_origin = validated["relative_pitches"][0]
    position_origin = validated["relative_positions"][0]
    return {
        "relative_pitches": [
            value - pitch_origin for value in validated["relative_pitches"]
        ],
        "relative_positions": [
            value - position_origin for value in validated["relative_positions"]
        ],
    }


def contiguous_windows(
    source: Mapping[str, Any], query_onsets: int
) -> list[dict[str, list[int]]]:
    """Enumerate every rebased contiguous source window of a fixed onset count."""

    validated = validate_view(source)
    if query_onsets < 3:
        raise ValueError("query_onsets must be at least three")
    if len(validated["relative_pitches"]) < query_onsets:
        return []
    windows = []
    for start in range(len(validated["relative_pitches"]) - query_onsets + 1):
        windows.append(
            rebase_view(
                {
                    "relative_pitches": validated["relative_pitches"][
                        start : start + query_onsets
                    ],
                    "relative_positions": validated["relative_positions"][
                        start : start + query_onsets
                    ],
                }
            )
        )
    return windows


def canonicalize_query(
    query: Mapping[str, Any], operation: OperationTransform
) -> dict[str, list[int]]:
    """Apply the known inverse operation before containment matching."""

    canonical = rebase_view(query)
    pitches = list(canonical["relative_pitches"])
    positions = list(canonical["relative_positions"])
    if operation.invert:
        pitches = [-value for value in pitches]
    if operation.time_scale_mode == "double":
        if any(position % 2 != 0 for position in positions):
            raise ValueError("double-scaled query cannot be inverted exactly")
        positions = [position // 2 for position in positions]
    elif operation.time_scale_mode == "half":
        positions = [position * 2 for position in positions]
    return rebase_view(
        {
            "relative_pitches": pitches,
            "relative_positions": positions,
        }
    )


def matching_window_indices(
    query: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    operation: OperationTransform,
) -> list[int]:
    """Return exact source-window indices after inverse-operation canonicalization."""

    canonical_query = canonicalize_query(query, operation)
    windows = contiguous_windows(
        source, len(canonical_query["relative_pitches"])
    )
    return [index for index, window in enumerate(windows) if window == canonical_query]


def exact_operation_aware_containment(
    query: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    operation: OperationTransform,
) -> bool:
    """Whether a canonicalized partial query is an exact contiguous source part."""

    return bool(matching_window_indices(query, source, operation=operation))
