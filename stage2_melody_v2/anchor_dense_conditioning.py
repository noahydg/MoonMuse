"""Pure contracts for feeding predicted anchor plans to the dense Stage 2 model.

The neural planner finishes all 32 anchor decisions before dense event
generation starts.  The dense realizer may consume only a serialized
``planner_predictions`` payload.  This module deliberately contains no Torch
dependency so the data and leakage contracts can be checked locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .anchor_conditioning import (
    ANCHOR_SLOT_COUNT,
    DEGREE_LABELS,
    DEGREE_TO_ID,
    MISSING_CLASS_ID,
    REALIZER_SCHEMA,
)


@dataclass(frozen=True)
class DenseAnchorAlignment:
    """Per-input-token routing from the dense stream to one anchor slot."""

    anchor_slot_ids: tuple[int, ...]
    melody_condition_mask: tuple[int, ...]
    melody_bar_count: int

    def validate(self, token_count: int, *, require_complete: bool = True) -> None:
        if len(self.anchor_slot_ids) != token_count:
            raise ValueError("anchor-slot alignment length differs from token count")
        if len(self.melody_condition_mask) != token_count:
            raise ValueError("melody-condition mask length differs from token count")
        if require_complete and self.melody_bar_count != 8:
            raise ValueError("dense anchor conditioning requires exactly eight melody bars")
        if not require_complete and not 1 <= self.melody_bar_count <= 8:
            raise ValueError("partial dense anchor conditioning requires 1..8 melody bars")
        if any(not 0 <= slot < ANCHOR_SLOT_COUNT for slot in self.anchor_slot_ids):
            raise ValueError("dense token references an anchor slot outside 0..31")
        if any(value not in (0, 1) for value in self.melody_condition_mask):
            raise ValueError("melody-condition mask must be binary")
        if not any(self.melody_condition_mask):
            raise ValueError("dense sequence contains no conditioned Melody tokens")


@dataclass(frozen=True)
class PredictedAnchorArrays:
    """Validated numeric representation of one inference-only anchor payload."""

    active_ids: tuple[int, ...]
    degree_ids: tuple[int, ...]
    octave_ids: tuple[int, ...]

    def validate(self) -> None:
        fields = (self.active_ids, self.degree_ids, self.octave_ids)
        if not all(len(field) == ANCHOR_SLOT_COUNT for field in fields):
            raise ValueError("predicted anchor arrays must each contain 32 slots")
        for active, degree, octave in zip(*fields):
            if active not in (0, 1):
                raise ValueError("predicted anchor activity must be binary")
            if active:
                if not 0 <= degree < len(DEGREE_LABELS):
                    raise ValueError("active anchor has an invalid degree ID")
                if not 0 <= octave <= 8:
                    raise ValueError("active anchor has an invalid octave ID")
            elif degree != MISSING_CLASS_ID or octave != MISSING_CLASS_ID:
                raise ValueError("rest anchor must not carry pitch labels")


def validate_realizer_anchor_condition(
    payload: Mapping[str, Any],
) -> PredictedAnchorArrays:
    """Validate that a dense-realizer condition contains predictions, not labels."""

    if payload.get("schema") != REALIZER_SCHEMA:
        raise ValueError("unexpected anchor-realizer condition schema")
    if payload.get("inference_source") != "planner_predictions":
        raise ValueError("dense realizer requires planner_predictions")
    if payload.get("ground_truth_anchor_fields_present") is not False:
        raise ValueError("ground-truth anchor fields are forbidden at inference")
    if payload.get("relation_labels_present") is not False:
        raise ValueError("relation labels are forbidden in the anchor condition")
    if payload.get("cadence_labels_present") is not False:
        raise ValueError("cadence labels are forbidden in the anchor condition")
    anchors = payload.get("anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        raise ValueError("anchor condition must contain an ordered anchor list")
    if len(anchors) != ANCHOR_SLOT_COUNT:
        raise ValueError("anchor condition must contain exactly 32 predictions")

    active_ids: list[int] = []
    degree_ids: list[int] = []
    octave_ids: list[int] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise ValueError("each anchor prediction must be an object")
        if int(anchor.get("slot", -1)) != index:
            raise ValueError("anchor predictions must be ordered by slot 0..31")
        if int(anchor.get("bar", -1)) != index // 4:
            raise ValueError("anchor bar does not match its slot")
        if int(anchor.get("beat", -1)) != (index % 4) * 4:
            raise ValueError("anchor beat does not match its slot")
        if any(
            "target" in str(key).lower() or "ground_truth" in str(key).lower()
            for key in anchor
        ):
            raise ValueError("anchor prediction contains a forbidden target field")
        active = bool(anchor.get("active"))
        degree = anchor.get("degree")
        octave = anchor.get("octave")
        if active:
            if degree not in DEGREE_TO_ID:
                raise ValueError("active anchor degree is outside the fixed vocabulary")
            if isinstance(octave, bool) or not isinstance(octave, int):
                raise ValueError("active anchor octave must be an integer")
            if not 0 <= octave <= 8:
                raise ValueError("active anchor octave must be in 0..8")
            active_ids.append(1)
            degree_ids.append(DEGREE_TO_ID[str(degree)])
            octave_ids.append(int(octave))
        else:
            if degree is not None or octave is not None:
                raise ValueError("rest anchor must serialize null degree and octave")
            active_ids.append(0)
            degree_ids.append(MISSING_CLASS_ID)
            octave_ids.append(MISSING_CLASS_ID)

    arrays = PredictedAnchorArrays(
        active_ids=tuple(active_ids),
        degree_ids=tuple(degree_ids),
        octave_ids=tuple(octave_ids),
    )
    arrays.validate()
    return arrays


def align_dense_tokens_to_anchor_slots(
    tokens: Sequence[str], *, require_complete: bool = True
) -> DenseAnchorAlignment:
    """Route the first eight Melody bars to their four quarter-note anchors.

    A condition vector is added only at Melody input positions.  Each selected
    slot representation is globally contextualized by the full 32-slot plan in
    the neural encoder, so this local routing does not hide later planned form.
    """

    slot_ids: list[int] = []
    condition_mask: list[int] = []
    current_track: str | None = None
    melody_bar = -1
    current_beat = 0

    for raw_token in tokens:
        token = str(raw_token)
        if token.startswith("Track_"):
            current_track = token.removeprefix("Track_")
            if current_track == "Melody":
                melody_bar += 1
                current_beat = 0
        elif current_track == "Melody" and token.startswith("Beat_"):
            current_beat = int(token.removeprefix("Beat_"))
            if not 0 <= current_beat < 16:
                raise ValueError("Melody beat is outside 0..15")

        in_conditioned_melody = current_track == "Melody" and 0 <= melody_bar < 8
        if in_conditioned_melody:
            slot = melody_bar * 4 + min(3, current_beat // 4)
            slot_ids.append(slot)
            condition_mask.append(1)
        else:
            # The value is ignored when the mask is zero. Keeping it in range
            # makes tensor gathering total and avoids sentinel-index behavior.
            slot_ids.append(0)
            condition_mask.append(0)

    alignment = DenseAnchorAlignment(
        anchor_slot_ids=tuple(slot_ids),
        melody_condition_mask=tuple(condition_mask),
        melody_bar_count=melody_bar + 1,
    )
    alignment.validate(len(tokens), require_complete=require_complete)
    return alignment
