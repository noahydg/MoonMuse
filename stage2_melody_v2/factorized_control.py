"""Factorized Stage 2 control contract after the V111 rhythm diagnosis.

This module is deliberately model-independent.  It separates five quantities that
the earlier V97/V98 sidecar partially conflated:

``z_id``
    Which earlier motif identity, if any, must be preserved.
``o_t``
    The explicit compositional operation path requested for the cell.
``d_pitch`` / ``d_rhythm``
    Independent ordinal design targets.  They are not perceptually calibrated
    distances and must not be interpreted as such.
``e_t``
    A valence/arousal target for each cell, kept independent of structural edits.

The legacy ``source_slot`` remains useful as a comparison anchor.  It is *not* an
identity pointer for a contrast cell: B can be compared with A while introducing a
new motif identity.  Keeping those pointers separate is the main executable change
in this contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .schema import MotifRelation, PhrasePlan


CONTROL_SCHEMA = "moonmuse-stage2-v2-factorized-control-plan-v1"
ENCODED_CONTROL_SCHEMA = "moonmuse-stage2-v2-factorized-control-condition-v1"


class MotifOperation(str, Enum):
    INTRODUCE = "INTRODUCE"
    REPEAT = "REPEAT"
    RETURN = "RETURN"
    PITCH_VARIATION = "PITCH_VARIATION"
    RHYTHM_VARIATION = "RHYTHM_VARIATION"
    TRANSPOSE = "TRANSPOSE"
    INVERT = "INVERT"
    TIME_SCALE = "TIME_SCALE"
    FRAGMENT = "FRAGMENT"
    CONTRAST = "CONTRAST"
    CADENTIAL_CLOSE = "CADENTIAL_CLOSE"


class DevelopmentDegree(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ValenceTarget(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"


class ArousalTarget(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    LOW = "LOW"
    HIGH = "HIGH"


PITCH_OPERATIONS = frozenset(
    {
        MotifOperation.PITCH_VARIATION,
        MotifOperation.TRANSPOSE,
        MotifOperation.INVERT,
        MotifOperation.FRAGMENT,
        MotifOperation.CADENTIAL_CLOSE,
    }
)
RHYTHM_OPERATIONS = frozenset(
    {
        MotifOperation.RHYTHM_VARIATION,
        MotifOperation.TIME_SCALE,
        MotifOperation.FRAGMENT,
        MotifOperation.CADENTIAL_CLOSE,
    }
)

OPERATION_TOKENS = ("PAD",) + tuple(value.value for value in MotifOperation)
OPERATION_TO_ID = {value: index for index, value in enumerate(OPERATION_TOKENS)}
DEGREE_TO_ID = {value: index for index, value in enumerate(DevelopmentDegree)}
VALENCE_TO_ID = {value: index for index, value in enumerate(ValenceTarget)}
AROUSAL_TO_ID = {value: index for index, value in enumerate(ArousalTarget)}


@dataclass(frozen=True)
class AffectTarget:
    valence: ValenceTarget
    arousal: ArousalTarget

    @classmethod
    def from_quadrant(cls, quadrant: Optional[str]) -> "AffectTarget":
        mapping = {
            None: (ValenceTarget.UNSPECIFIED, ArousalTarget.UNSPECIFIED),
            "Q1": (ValenceTarget.POSITIVE, ArousalTarget.HIGH),
            "Q2": (ValenceTarget.NEGATIVE, ArousalTarget.HIGH),
            "Q3": (ValenceTarget.NEGATIVE, ArousalTarget.LOW),
            "Q4": (ValenceTarget.POSITIVE, ArousalTarget.LOW),
        }
        try:
            valence, arousal = mapping[quadrant]
        except KeyError as exc:
            raise ValueError(f"unsupported emotion quadrant: {quadrant}") from exc
        return cls(valence=valence, arousal=arousal)

    def to_dict(self) -> dict[str, str]:
        return {"valence": self.valence.value, "arousal": self.arousal.value}


@dataclass(frozen=True)
class FactorizedControlSlot:
    slot_index: int
    label: str
    start_bar: int
    end_bar: int
    identity_label: str
    identity_source_slot: Optional[int]
    comparison_source_slot: Optional[int]
    operation_path: tuple[MotifOperation, ...]
    pitch_degree: DevelopmentDegree
    rhythm_degree: DevelopmentDegree
    affect_target: AffectTarget

    def validate_local(self) -> None:
        if not self.label or not self.identity_label:
            raise ValueError("control slot labels must be non-empty")
        if self.start_bar < 0 or self.end_bar - self.start_bar != 2:
            raise ValueError("factorized control slots must span two bars")
        if not self.operation_path:
            raise ValueError("operation path must contain at least one operation")
        if len(set(self.operation_path)) != len(self.operation_path):
            raise ValueError("operation path may not repeat an operation token")
        if self.identity_source_slot is not None and self.identity_source_slot >= self.slot_index:
            raise ValueError("identity source must be strictly earlier")
        if self.comparison_source_slot is not None and self.comparison_source_slot >= self.slot_index:
            raise ValueError("comparison source must be strictly earlier")

        path = set(self.operation_path)
        introduces_identity = bool(
            path.intersection({MotifOperation.INTRODUCE, MotifOperation.CONTRAST})
        )
        if introduces_identity and self.identity_source_slot is not None:
            raise ValueError("introduce/contrast must not reuse an earlier identity")
        if not introduces_identity and self.identity_source_slot is None:
            raise ValueError("identity-preserving operations require an identity source")

        if introduces_identity and (
            self.pitch_degree is not DevelopmentDegree.NONE
            or self.rhythm_degree is not DevelopmentDegree.NONE
        ):
            raise ValueError("within-identity degrees are undefined for a new identity")
        if (
            self.pitch_degree is not DevelopmentDegree.NONE
            and not path.intersection(PITCH_OPERATIONS)
        ):
            raise ValueError("nonzero pitch degree requires a pitch operation")
        if (
            self.rhythm_degree is not DevelopmentDegree.NONE
            and not path.intersection(RHYTHM_OPERATIONS)
        ):
            raise ValueError("nonzero rhythm degree requires a rhythm operation")

    def to_dict(self) -> dict[str, Any]:
        self.validate_local()
        return {
            "slot_index": self.slot_index,
            "label": self.label,
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "z_id": {
                "identity_label": self.identity_label,
                "source_slot": self.identity_source_slot,
            },
            "comparison_source_slot": self.comparison_source_slot,
            "o_t": [value.value for value in self.operation_path],
            "d_pitch": self.pitch_degree.value,
            "d_rhythm": self.rhythm_degree.value,
            "e_t": self.affect_target.to_dict(),
        }


@dataclass(frozen=True)
class FactorizedControlPlan:
    total_bars: int
    slots: tuple[FactorizedControlSlot, ...]
    schema: str = CONTROL_SCHEMA
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema != CONTROL_SCHEMA:
            raise ValueError(f"unsupported factorized control schema: {self.schema}")
        if self.total_bars != 8 or len(self.slots) != 4:
            raise ValueError("factorized V2 control requires four cells over eight bars")
        cursor = 0
        for expected_index, slot in enumerate(self.slots):
            slot.validate_local()
            if slot.slot_index != expected_index:
                raise ValueError("control slot indices must be contiguous")
            if slot.start_bar != cursor:
                raise ValueError("control slots must cover contiguous bars")
            if slot.identity_source_slot is not None:
                source = self.slots[slot.identity_source_slot]
                if source.identity_label != slot.identity_label:
                    raise ValueError("z_id source and target identity labels differ")
            cursor = slot.end_bar
        if cursor != self.total_bars:
            raise ValueError("control slots do not cover the full plan")
        if self.slots[0].operation_path != (MotifOperation.INTRODUCE,):
            raise ValueError("the first cell must introduce the first identity")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "total_bars": self.total_bars,
            "slots": [slot.to_dict() for slot in self.slots],
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EncodedFactorizedControl:
    identity_family_ids: tuple[int, ...]
    identity_source_slot_ids: tuple[int, ...]
    comparison_source_slot_ids: tuple[int, ...]
    operation_path_ids: tuple[tuple[int, ...], ...]
    operation_lengths: tuple[int, ...]
    pitch_degree_ids: tuple[int, ...]
    rhythm_degree_ids: tuple[int, ...]
    valence_ids: tuple[int, ...]
    arousal_ids: tuple[int, ...]
    bar_to_slot_ids: tuple[int, ...]
    identity_map: Mapping[str, int]

    def validate(self) -> None:
        four_cell_fields = (
            self.identity_family_ids,
            self.identity_source_slot_ids,
            self.comparison_source_slot_ids,
            self.operation_path_ids,
            self.operation_lengths,
            self.pitch_degree_ids,
            self.rhythm_degree_ids,
            self.valence_ids,
            self.arousal_ids,
        )
        if not all(len(values) == 4 for values in four_cell_fields):
            raise ValueError("encoded factorized control requires four cells")
        if self.bar_to_slot_ids != (0, 0, 1, 1, 2, 2, 3, 3):
            raise ValueError("encoded bar-to-slot mapping drifted")
        if any(length < 1 for length in self.operation_lengths):
            raise ValueError("every cell must retain an operation token")
        width = len(self.operation_path_ids[0])
        if width < 1 or any(len(row) != width for row in self.operation_path_ids):
            raise ValueError("operation paths must use one padded width")
        for row, length in zip(self.operation_path_ids, self.operation_lengths):
            if any(value == 0 for value in row[:length]):
                raise ValueError("PAD may not occur inside an operation path")
            if any(value != 0 for value in row[length:]):
                raise ValueError("operation padding must be zero")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": ENCODED_CONTROL_SCHEMA,
            "identity_family_ids": self.identity_family_ids,
            "identity_source_slot_ids": self.identity_source_slot_ids,
            "comparison_source_slot_ids": self.comparison_source_slot_ids,
            "operation_path_ids": self.operation_path_ids,
            "operation_lengths": self.operation_lengths,
            "pitch_degree_ids": self.pitch_degree_ids,
            "rhythm_degree_ids": self.rhythm_degree_ids,
            "valence_ids": self.valence_ids,
            "arousal_ids": self.arousal_ids,
            "bar_to_slot_ids": self.bar_to_slot_ids,
            "id_maps": {
                "identity_families": dict(self.identity_map),
                "operations": OPERATION_TO_ID,
                "degrees": {key.value: value for key, value in DEGREE_TO_ID.items()},
                "valence": {key.value: value for key, value in VALENCE_TO_ID.items()},
                "arousal": {key.value: value for key, value in AROUSAL_TO_ID.items()},
            },
            "axis_boundary": (
                "d_pitch and d_rhythm are ordinal design targets, not perceptually "
                "linear or V109-derived scores; e_t is an independent target."
            ),
        }


def build_factorized_eight_bar_control(
    emotion_quadrant: Optional[str] = None,
    *,
    affect_trajectory: Optional[Sequence[AffectTarget]] = None,
) -> FactorizedControlPlan:
    """Build the A-A'-B-A-cadence scaffold with five explicit control axes."""

    if affect_trajectory is None:
        target = AffectTarget.from_quadrant(emotion_quadrant)
        affect_targets = (target,) * 4
    else:
        if len(affect_trajectory) != 4:
            raise ValueError("affect trajectory must contain one target per cell")
        affect_targets = tuple(affect_trajectory)

    slots = (
        FactorizedControlSlot(
            slot_index=0,
            label="A",
            start_bar=0,
            end_bar=2,
            identity_label="A",
            identity_source_slot=None,
            comparison_source_slot=None,
            operation_path=(MotifOperation.INTRODUCE,),
            pitch_degree=DevelopmentDegree.NONE,
            rhythm_degree=DevelopmentDegree.NONE,
            affect_target=affect_targets[0],
        ),
        FactorizedControlSlot(
            slot_index=1,
            label="A_PRIME",
            start_bar=2,
            end_bar=4,
            identity_label="A",
            identity_source_slot=0,
            comparison_source_slot=0,
            operation_path=(
                MotifOperation.PITCH_VARIATION,
                MotifOperation.RHYTHM_VARIATION,
            ),
            pitch_degree=DevelopmentDegree.LIGHT,
            rhythm_degree=DevelopmentDegree.LIGHT,
            affect_target=affect_targets[1],
        ),
        FactorizedControlSlot(
            slot_index=2,
            label="B",
            start_bar=4,
            end_bar=6,
            identity_label="B",
            identity_source_slot=None,
            comparison_source_slot=0,
            operation_path=(MotifOperation.CONTRAST,),
            pitch_degree=DevelopmentDegree.NONE,
            rhythm_degree=DevelopmentDegree.NONE,
            affect_target=affect_targets[2],
        ),
        FactorizedControlSlot(
            slot_index=3,
            label="A_CAD",
            start_bar=6,
            end_bar=8,
            identity_label="A",
            identity_source_slot=0,
            comparison_source_slot=0,
            operation_path=(MotifOperation.RETURN, MotifOperation.CADENTIAL_CLOSE),
            pitch_degree=DevelopmentDegree.LIGHT,
            rhythm_degree=DevelopmentDegree.LIGHT,
            affect_target=affect_targets[3],
        ),
    )
    plan = FactorizedControlPlan(
        total_bars=8,
        slots=slots,
        metadata={
            "planner_kind": "deterministic_factorized_control_contract",
            "learned": False,
            "formalStudyData": False,
            "source_decision": "D-071",
            "claim_boundary": (
                "Architecture contract only; not supervision, a trained controller, "
                "generated V2 music, perceptual calibration or a quality result."
            ),
        },
    )
    plan.validate()
    return plan


def factorized_control_from_phrase_plan(plan: PhrasePlan) -> FactorizedControlPlan:
    """Compile the frozen V97 scaffold without reusing relation as identity."""

    plan.validate()
    expected_labels = ("A", "A_PRIME", "B", "A_CAD")
    expected_relations = (
        MotifRelation.INTRODUCE,
        MotifRelation.VARIATE,
        MotifRelation.CONTRAST,
        MotifRelation.CADENCE_RETURN,
    )
    if tuple(slot.label for slot in plan.slots) != expected_labels:
        raise ValueError("factorized compiler only accepts the frozen V97 slot labels")
    if tuple(slot.relation for slot in plan.slots) != expected_relations:
        raise ValueError("factorized compiler only accepts the frozen V97 relations")
    compiled = build_factorized_eight_bar_control(plan.emotion_quadrant)
    for legacy, current in zip(plan.slots, compiled.slots):
        if (legacy.start_bar, legacy.end_bar) != (current.start_bar, current.end_bar):
            raise ValueError("legacy and factorized cell boundaries differ")
        if legacy.source_slot != current.comparison_source_slot:
            raise ValueError("legacy source must remain the comparison anchor")
    return compiled


def encode_factorized_control(plan: FactorizedControlPlan) -> EncodedFactorizedControl:
    plan.validate()
    identity_map: dict[str, int] = {}
    for slot in plan.slots:
        if slot.identity_label not in identity_map:
            identity_map[slot.identity_label] = len(identity_map)
    maximum_operations = max(len(slot.operation_path) for slot in plan.slots)
    operation_rows = []
    operation_lengths = []
    for slot in plan.slots:
        row = [OPERATION_TO_ID[value.value] for value in slot.operation_path]
        operation_lengths.append(len(row))
        row.extend([0] * (maximum_operations - len(row)))
        operation_rows.append(tuple(row))
    encoded = EncodedFactorizedControl(
        identity_family_ids=tuple(
            identity_map[slot.identity_label] for slot in plan.slots
        ),
        identity_source_slot_ids=tuple(
            -1 if slot.identity_source_slot is None else slot.identity_source_slot
            for slot in plan.slots
        ),
        comparison_source_slot_ids=tuple(
            -1
            if slot.comparison_source_slot is None
            else slot.comparison_source_slot
            for slot in plan.slots
        ),
        operation_path_ids=tuple(operation_rows),
        operation_lengths=tuple(operation_lengths),
        pitch_degree_ids=tuple(DEGREE_TO_ID[slot.pitch_degree] for slot in plan.slots),
        rhythm_degree_ids=tuple(
            DEGREE_TO_ID[slot.rhythm_degree] for slot in plan.slots
        ),
        valence_ids=tuple(
            VALENCE_TO_ID[slot.affect_target.valence] for slot in plan.slots
        ),
        arousal_ids=tuple(
            AROUSAL_TO_ID[slot.affect_target.arousal] for slot in plan.slots
        ),
        bar_to_slot_ids=tuple(bar // 2 for bar in range(plan.total_bars)),
        identity_map=identity_map,
    )
    encoded.validate()
    return encoded


def phrase_plan_from_payload(payload: Mapping[str, Any]) -> PhrasePlan:
    """Reject drift by rebuilding and matching the frozen deterministic plan."""

    if payload.get("schema") != "moonmuse-stage2-v2-phrase-plan-v1":
        raise ValueError("unexpected legacy phrase-plan schema")
    from .planner import build_eight_bar_plan

    rebuilt = build_eight_bar_plan(payload.get("emotion_quadrant"))
    if rebuilt.to_dict() != dict(payload):
        raise ValueError("legacy phrase-plan payload differs from its deterministic rebuild")
    return rebuilt
