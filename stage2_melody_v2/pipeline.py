"""Runnable sidecar pipeline joining V2 planning and observed motif memory."""

from __future__ import annotations

from typing import Any, Iterable

from .motif_memory import (
    cadence_target,
    compare_motifs,
    extract_cell_motifs,
    parse_melody_onsets,
)
from .planner import build_eight_bar_plan


def build_architecture_sidecar(
    events: Iterable[dict], *, source_id: str, emotion_quadrant: str | None = None
) -> dict[str, Any]:
    """Build a model-independent JSON-ready V2 development sidecar."""

    event_list = list(events)
    key, onsets, bar_count = parse_melody_onsets(event_list)
    plan = build_eight_bar_plan(emotion_quadrant)
    motifs = extract_cell_motifs(onsets)
    comparisons = [compare_motifs(motifs[0], motif) for motif in motifs[1:]]
    return {
        "schema": "moonmuse-stage2-v2-architecture-sidecar-v1",
        "formalStudyData": False,
        "source_id": source_id,
        "key": key,
        "available_melody_bars": bar_count,
        "first8_onset_count": sum(onset.bar < 8 for onset in onsets),
        "plan": plan.to_dict(),
        "plan_sha256": plan.fingerprint(),
        "observed_cells": [
            {
                "slot_index": motif.slot_index,
                "onset_count": motif.onset_count,
                "motif_sha256": motif.fingerprint(),
                "interval_tokens": list(motif.interval_tokens),
                "rhythm_gaps": list(motif.rhythm_gaps),
            }
            for motif in motifs
        ],
        "observed_relations_to_A": [item.to_dict() for item in comparisons],
        "observed_final_target": cadence_target(onsets),
        "claim_boundary": (
            "Architecture/data sidecar only; not a human phrase annotation, trained "
            "planner, generated V2 melody, or quality result."
        ),
    }
