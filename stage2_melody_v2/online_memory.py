"""Causal online motif-memory snapshots for Stage 2 V2 inference."""

from __future__ import annotations

from typing import Any, Mapping


CELL_COUNT = 4
PUBLIC_FIELDS = (
    "schema",
    "role_ids",
    "source_slot_ids",
    "bar_to_slot_ids",
    "motif_interval_ids",
    "motif_rhythm_ids",
    "motif_lengths",
)


def mask_condition_for_online_inference(
    payload: Mapping[str, Any], *, target_slot: int
) -> dict[str, Any]:
    """Expose motif memory only from cells completed before ``target_slot``.

    Relation and cadence fields are deliberately omitted. Generation is sequential,
    so completed slots are exactly ``0 .. target_slot - 1``.
    """

    if payload.get("schema") != "moonmuse-stage2-v2-realizer-condition-v1":
        raise ValueError("unexpected realizer-condition schema")
    if not 0 <= target_slot < CELL_COUNT:
        raise ValueError("target_slot must be in 0..3")
    missing = [field for field in PUBLIC_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"condition is missing fields: {missing}")
    four_cell_fields = (
        "role_ids",
        "source_slot_ids",
        "motif_interval_ids",
        "motif_rhythm_ids",
        "motif_lengths",
    )
    if any(len(payload[field]) != CELL_COUNT for field in four_cell_fields):
        raise ValueError("online condition must contain four cells")

    source_slots = [int(value) for value in payload["source_slot_ids"]]
    for slot, source in enumerate(source_slots):
        if source < -1 or source >= slot:
            raise ValueError("motif source must be absent or strictly earlier")
    current_source = source_slots[target_slot]
    if current_source >= target_slot:
        raise ValueError("target motif source is not available online")

    interval_ids = []
    rhythm_ids = []
    motif_lengths = []
    for slot in range(CELL_COUNT):
        intervals = [int(value) for value in payload["motif_interval_ids"][slot]]
        rhythms = [int(value) for value in payload["motif_rhythm_ids"][slot]]
        declared_length = int(payload["motif_lengths"][slot])
        if len(intervals) != len(rhythms) or len(intervals) != declared_length:
            raise ValueError(f"motif feature length mismatch in cell {slot}")
        if slot < target_slot:
            interval_ids.append(intervals)
            rhythm_ids.append(rhythms)
            motif_lengths.append(declared_length)
        else:
            interval_ids.append([])
            rhythm_ids.append([])
            motif_lengths.append(0)

    return {
        "schema": payload["schema"],
        "role_ids": [int(value) for value in payload["role_ids"]],
        "source_slot_ids": source_slots,
        "bar_to_slot_ids": [int(value) for value in payload["bar_to_slot_ids"]],
        "motif_interval_ids": interval_ids,
        "motif_rhythm_ids": rhythm_ids,
        "motif_lengths": motif_lengths,
    }


def build_online_condition_history(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the four sequential inference snapshots A through cadence cell."""

    return [
        mask_condition_for_online_inference(payload, target_slot=slot)
        for slot in range(CELL_COUNT)
    ]
