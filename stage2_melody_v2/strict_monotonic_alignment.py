"""Validation-blind strict monotonic alignment primitive for V107-C3D2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .operation_aware_containment import rebase_view


PITCH_WEIGHT = 0.65
TIME_WEIGHT = 0.35
SOURCE_SKIP_COST = 0.25


@dataclass(frozen=True)
class MonotonicAlignment:
    matched_pairs: tuple[tuple[int, int], ...]
    skipped_source_indices: tuple[int, ...]
    total_cost: float
    similarity: float

    def __post_init__(self) -> None:
        query_indices = [pair[0] for pair in self.matched_pairs]
        source_indices = [pair[1] for pair in self.matched_pairs]
        if query_indices != list(range(len(query_indices))):
            raise ValueError("alignment must match every query onset exactly once")
        if any(right <= left for left, right in zip(source_indices, source_indices[1:])):
            raise ValueError("alignment source indices must be strictly increasing")
        if len(source_indices) != len(set(source_indices)):
            raise ValueError("alignment source indices must be unique")
        if self.total_cost < 0.0:
            raise ValueError("alignment cost cannot be negative")
        if not 0.0 < self.similarity <= 1.0:
            raise ValueError("alignment similarity must lie in (0, 1]")


def _normalized_times(view: Mapping[str, Any]) -> list[float]:
    positions = rebase_view(view)["relative_positions"]
    final = positions[-1]
    if final <= 0:
        raise ValueError("alignment view must have positive duration")
    return [position / final for position in positions]


def local_alignment_cost(
    query_pitch: int,
    query_time: float,
    source_pitch: int,
    source_time: float,
) -> float:
    pitch = min(abs(query_pitch - source_pitch) / 12.0, 1.0)
    timing = min(abs(query_time - source_time) / 0.25, 1.0)
    return PITCH_WEIGHT * pitch + TIME_WEIGHT * timing


def strict_monotonic_alignment(
    query: Mapping[str, Any], source_window: Mapping[str, Any]
) -> MonotonicAlignment | None:
    """Align every query onset once while allowing only source-side skips.

    The caller supplies one already selected/rebased candidate source window.  Query
    skips, source-index reuse and non-monotonic paths are impossible by construction.
    """

    canonical_query = rebase_view(query)
    canonical_source = rebase_view(source_window)
    query_pitches = canonical_query["relative_pitches"]
    source_pitches = canonical_source["relative_pitches"]
    if len(query_pitches) > len(source_pitches):
        return None
    query_times = _normalized_times(canonical_query)
    source_times = _normalized_times(canonical_source)
    query_count = len(query_pitches)
    source_count = len(source_pitches)
    infinity = float("inf")
    costs = [
        [infinity for _ in range(source_count + 1)]
        for _ in range(query_count + 1)
    ]
    back: list[list[tuple[str, int, int] | None]] = [
        [None for _ in range(source_count + 1)]
        for _ in range(query_count + 1)
    ]
    costs[0][0] = 0.0
    for source_index in range(1, source_count + 1):
        costs[0][source_index] = costs[0][source_index - 1] + SOURCE_SKIP_COST
        back[0][source_index] = ("skip", 0, source_index - 1)

    for query_index in range(1, query_count + 1):
        for source_index in range(1, source_count + 1):
            match_cost = costs[query_index - 1][source_index - 1]
            if math.isfinite(match_cost):
                match_cost += local_alignment_cost(
                    query_pitches[query_index - 1],
                    query_times[query_index - 1],
                    source_pitches[source_index - 1],
                    source_times[source_index - 1],
                )
            skip_cost = costs[query_index][source_index - 1]
            if math.isfinite(skip_cost):
                skip_cost += SOURCE_SKIP_COST
            if match_cost <= skip_cost:
                costs[query_index][source_index] = match_cost
                back[query_index][source_index] = (
                    "match",
                    query_index - 1,
                    source_index - 1,
                )
            else:
                costs[query_index][source_index] = skip_cost
                back[query_index][source_index] = (
                    "skip",
                    query_index,
                    source_index - 1,
                )

    total_cost = costs[query_count][source_count]
    if not math.isfinite(total_cost):
        return None
    matched: list[tuple[int, int]] = []
    skipped: list[int] = []
    query_index = query_count
    source_index = source_count
    while query_index > 0 or source_index > 0:
        step = back[query_index][source_index]
        if step is None:
            raise AssertionError("strict monotonic alignment backtrack is incomplete")
        kind, previous_query, previous_source = step
        if kind == "match":
            matched.append((query_index - 1, source_index - 1))
        elif kind == "skip":
            skipped.append(source_index - 1)
        else:
            raise AssertionError(f"unexpected alignment step: {kind}")
        query_index, source_index = previous_query, previous_source
    matched.reverse()
    skipped.reverse()
    return MonotonicAlignment(
        matched_pairs=tuple(matched),
        skipped_source_indices=tuple(skipped),
        total_cost=total_cost,
        similarity=math.exp(-total_cost / query_count),
    )
