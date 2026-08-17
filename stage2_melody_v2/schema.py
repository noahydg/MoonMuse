"""Versioned data contract for a phrase-planned MoonMuse V2 melody stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "moonmuse-stage2-v2-phrase-plan-v1"


class PhraseRole(str, Enum):
    OPENING = "OPENING"
    RESPONSE = "RESPONSE"
    DEVELOPMENT = "DEVELOPMENT"
    CADENCE = "CADENCE"


class MotifRelation(str, Enum):
    INTRODUCE = "INTRODUCE"
    RETURN = "RETURN"
    VARIATE = "VARIATE"
    CONTRAST = "CONTRAST"
    CADENCE_RETURN = "CADENCE_RETURN"


class CadenceGoal(str, Enum):
    OPEN = "OPEN"
    HALF = "HALF"
    TONIC = "TONIC"


@dataclass(frozen=True)
class PhraseSlot:
    slot_index: int
    label: str
    start_bar: int
    end_bar: int
    role: PhraseRole
    relation: MotifRelation
    source_slot: int | None
    contour_target: str
    register_band: str
    density_band: str
    cadence_goal: CadenceGoal
    contains_climax: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "label": self.label,
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "role": self.role.value,
            "relation": self.relation.value,
            "source_slot": self.source_slot,
            "contour_target": self.contour_target,
            "register_band": self.register_band,
            "density_band": self.density_band,
            "cadence_goal": self.cadence_goal.value,
            "contains_climax": self.contains_climax,
        }

    def control_tokens(self) -> list[str]:
        return [
            f"V2_PhraseRole_{self.role.value}",
            f"V2_MotifRelation_{self.relation.value}",
            f"V2_Contour_{self.contour_target}",
            f"V2_Register_{self.register_band}",
            f"V2_Density_{self.density_band}",
            f"V2_Cadence_{self.cadence_goal.value}",
        ]


@dataclass(frozen=True)
class PhrasePlan:
    total_bars: int
    slots: tuple[PhraseSlot, ...]
    climax_bar: int
    emotion_quadrant: str | None = None
    schema: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported plan schema: {self.schema}")
        if self.total_bars != 8:
            raise ValueError("the V97 architecture contract is fixed to eight bars")
        if len(self.slots) != 4:
            raise ValueError("the V97 architecture contract requires four cells")
        if not 0 <= self.climax_bar < self.total_bars:
            raise ValueError("climax bar is outside the plan")
        cursor = 0
        for expected_index, slot in enumerate(self.slots):
            if slot.slot_index != expected_index:
                raise ValueError("slot indices must be contiguous")
            if slot.start_bar != cursor or slot.end_bar - slot.start_bar != 2:
                raise ValueError("slots must cover contiguous two-bar cells")
            if slot.source_slot is not None and slot.source_slot >= slot.slot_index:
                raise ValueError("motif relations may only reference an earlier cell")
            cursor = slot.end_bar
        if cursor != self.total_bars:
            raise ValueError("phrase cells do not cover all bars")
        if self.slots[0].relation is not MotifRelation.INTRODUCE:
            raise ValueError("the first cell must introduce motif A")
        if sum(slot.role is PhraseRole.CADENCE for slot in self.slots) != 1:
            raise ValueError("the plan must contain exactly one cadence cell")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "total_bars": self.total_bars,
            "climax_bar": self.climax_bar,
            "emotion_quadrant": self.emotion_quadrant,
            "slots": [slot.to_dict() for slot in self.slots],
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def control_tokens(self) -> list[list[str]]:
        self.validate()
        return [slot.control_tokens() for slot in self.slots]
