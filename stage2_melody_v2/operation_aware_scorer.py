"""Bounded operation-aware cross-attention primitives for V107-C3.

The module is intentionally isolated from the Stage 2 generator.  It constructs
deterministic synthetic robustness queries, canonicalizes the declared operation,
and scores query/window pairs with a small monotonic-biased attention network.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .operation_aware_containment import (
    OperationTransform,
    canonicalize_query,
    contiguous_windows,
    rebase_view,
    validate_view,
)


@dataclass(frozen=True)
class ApproximateVariation:
    """Synthetic local edits applied in a fixed delete-pitch-timing order."""

    pitch: bool = False
    timing: bool = False
    delete: bool = False

    def __post_init__(self) -> None:
        if not (self.pitch or self.timing or self.delete):
            raise ValueError("an approximate variation must enable at least one edit")


def _hash_index(material: str, size: int) -> int:
    if size <= 0:
        raise ValueError("hash selection requires a non-empty population")
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _delete_one(view: Mapping[str, Any], *, material: str) -> dict[str, list[int]] | None:
    current = rebase_view(view)
    if len(current["relative_pitches"]) < 4:
        return None
    candidates = list(range(1, len(current["relative_pitches"]) - 1))
    index = candidates[_hash_index(f"{material}|delete", len(candidates))]
    return rebase_view(
        {
            "relative_pitches": [
                value
                for item, value in enumerate(current["relative_pitches"])
                if item != index
            ],
            "relative_positions": [
                value
                for item, value in enumerate(current["relative_positions"])
                if item != index
            ],
        }
    )


def _change_one_pitch(
    view: Mapping[str, Any], *, material: str
) -> dict[str, list[int]] | None:
    current = rebase_view(view)
    candidates = list(range(1, len(current["relative_pitches"]) - 1))
    if not candidates:
        return None
    index = candidates[_hash_index(f"{material}|pitch-index", len(candidates))]
    delta = (-1, 1)[_hash_index(f"{material}|pitch-delta", 2)]
    pitches = list(current["relative_pitches"])
    pitches[index] += delta
    return rebase_view(
        {
            "relative_pitches": pitches,
            "relative_positions": current["relative_positions"],
        }
    )


def _jitter_one_onset(
    view: Mapping[str, Any], *, material: str
) -> dict[str, list[int]] | None:
    current = rebase_view(view)
    positions = list(current["relative_positions"])
    candidates: list[tuple[int, int]] = []
    for index in range(1, len(positions) - 1):
        for delta in (-1, 1):
            shifted = positions[index] + delta
            if positions[index - 1] < shifted < positions[index + 1]:
                candidates.append((index, delta))
    if not candidates:
        return None
    index, delta = candidates[_hash_index(f"{material}|timing", len(candidates))]
    positions[index] += delta
    return rebase_view(
        {
            "relative_pitches": current["relative_pitches"],
            "relative_positions": positions,
        }
    )


def apply_approximate_variation(
    view: Mapping[str, Any],
    variation: ApproximateVariation,
    *,
    material: str,
) -> dict[str, list[int]] | None:
    """Apply a deterministic variation, returning None only if structurally illegal."""

    current: dict[str, list[int]] | None = rebase_view(view)
    if variation.delete:
        current = _delete_one(current, material=material)
    if current is not None and variation.pitch:
        current = _change_one_pitch(current, material=material)
    if current is not None and variation.timing:
        current = _jitter_one_onset(current, material=material)
    return current


def apply_declared_operation(view: Mapping[str, Any]) -> dict[str, list[int]]:
    """Apply the fixed C3 inversion plus double-time operation."""

    canonical = rebase_view(view)
    return rebase_view(
        {
            "relative_pitches": [
                -value for value in canonical["relative_pitches"]
            ],
            "relative_positions": [
                2 * value for value in canonical["relative_positions"]
            ],
        }
    )


def canonicalize_declared_operation(
    view: Mapping[str, Any],
) -> dict[str, list[int]]:
    return canonicalize_query(
        view,
        OperationTransform(invert=True, time_scale_mode="double"),
    )


def onset_features_from_view(view: Mapping[str, Any]) -> list[tuple[float, ...]]:
    """Return the four frozen C3 onset features."""

    canonical = rebase_view(view)
    pitches = canonical["relative_pitches"]
    positions = canonical["relative_positions"]
    final_position = positions[-1]
    final_ordinal = len(pitches) - 1
    if final_position <= 0 or final_ordinal <= 0:
        raise ValueError("C3 view requires positive duration and multiple onsets")
    return [
        (
            max(-48, min(48, pitch)) / 48.0,
            min(48, abs(pitch)) / 48.0,
            position / final_position,
            index / final_ordinal,
        )
        for index, (pitch, position) in enumerate(zip(pitches, positions))
    ]


def pair_views_to_tensors(
    queries: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(queries) != len(sources) or not queries:
        raise ValueError("C3 query/source batches must be non-empty and aligned")
    query_sequences = [onset_features_from_view(view) for view in queries]
    source_sequences = [onset_features_from_view(view) for view in sources]
    query_max = max(len(sequence) for sequence in query_sequences)
    source_max = max(len(sequence) for sequence in source_sequences)
    query_features = torch.zeros((len(queries), query_max, 4), dtype=torch.float32)
    source_features = torch.zeros((len(sources), source_max, 4), dtype=torch.float32)
    query_mask = torch.zeros((len(queries), query_max), dtype=torch.bool)
    source_mask = torch.zeros((len(sources), source_max), dtype=torch.bool)
    length_features = torch.zeros((len(queries), 2), dtype=torch.float32)
    for index, (query, source) in enumerate(zip(query_sequences, source_sequences)):
        query_length = len(query)
        source_length = len(source)
        query_features[index, :query_length] = torch.tensor(query)
        source_features[index, :source_length] = torch.tensor(source)
        query_mask[index, :query_length] = True
        source_mask[index, :source_length] = True
        length_features[index, 0] = abs(query_length - source_length) / 31.0
        length_features[index, 1] = query_length / source_length
    return (
        query_features.to(device),
        query_mask.to(device),
        source_features.to(device),
        source_mask.to(device),
        length_features.to(device),
    )


class OperationAwareCrossAttentionScorer(nn.Module):
    """Small query-to-window scorer with a learned non-negative diagonal bias."""

    def __init__(self) -> None:
        super().__init__()
        self.onset_projection = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.raw_monotonic_penalty = nn.Parameter(torch.tensor(0.0))
        self.local_match = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.length_head = nn.Linear(2, 1)

    def forward(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        length_features: torch.Tensor,
    ) -> torch.Tensor:
        if query_features.ndim != 3 or query_features.shape[-1] != 4:
            raise ValueError("C3 query features must have shape [batch, onsets, 4]")
        if source_features.ndim != 3 or source_features.shape[-1] != 4:
            raise ValueError("C3 source features must have shape [batch, onsets, 4]")
        if query_features.shape[0] != source_features.shape[0]:
            raise ValueError("C3 query/source batch sizes differ")
        if query_mask.shape != query_features.shape[:2]:
            raise ValueError("C3 query mask shape mismatch")
        if source_mask.shape != source_features.shape[:2]:
            raise ValueError("C3 source mask shape mismatch")
        if length_features.shape != (query_features.shape[0], 2):
            raise ValueError("C3 length features must have shape [batch, 2]")
        if not torch.all(query_mask.any(dim=1)) or not torch.all(source_mask.any(dim=1)):
            raise ValueError("C3 every pair must have non-empty query and source")

        query_hidden = self.onset_projection(query_features)
        source_hidden = self.onset_projection(source_features)
        logits = torch.einsum("bqd,bsd->bqs", query_hidden, source_hidden)
        logits = logits / math.sqrt(query_hidden.shape[-1])
        query_ordinal = query_features[..., 3].unsqueeze(-1)
        source_ordinal = source_features[..., 3].unsqueeze(1)
        penalty = F.softplus(self.raw_monotonic_penalty)
        logits = logits - penalty * (query_ordinal - source_ordinal).abs()
        logits = logits.masked_fill(~source_mask.unsqueeze(1), -1e9)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bqs,bsd->bqd", attention, source_hidden)
        local = torch.cat(
            [
                query_hidden,
                attended,
                (query_hidden - attended).abs(),
                query_hidden * attended,
            ],
            dim=-1,
        )
        local_logits = self.local_match(local).squeeze(-1)
        weights = query_mask.to(local_logits.dtype)
        pooled = (local_logits * weights).sum(dim=1) / weights.sum(dim=1)
        return pooled + self.length_head(length_features).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def candidate_windows(
    source: Mapping[str, Any], query_onsets: int
) -> list[dict[str, list[int]]]:
    """Enumerate the frozen Q and Q+1 candidate-window population."""

    validate_view(source)
    if query_onsets < 3:
        raise ValueError("C3 query must contain at least three onsets")
    windows: list[dict[str, list[int]]] = []
    for length in (query_onsets, query_onsets + 1):
        windows.extend(contiguous_windows(source, length))
    return windows


def top_two_mean(scores: Sequence[float]) -> float:
    if not scores:
        return float("-inf")
    selected = sorted((float(value) for value in scores), reverse=True)[:2]
    return sum(selected) / len(selected)
