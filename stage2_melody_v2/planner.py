"""Deterministic planner contract used before a learned V2 planner exists."""

from __future__ import annotations

from .schema import (
    CadenceGoal,
    MotifRelation,
    PhrasePlan,
    PhraseRole,
    PhraseSlot,
)


def build_eight_bar_plan(emotion_quadrant: str | None = None) -> PhrasePlan:
    """Return the frozen V97 A-A'-B-A-cadence architecture scaffold.

    This plan is deliberately deterministic.  It validates interfaces and data
    feasibility; it is not presented as a learned or human-annotated phrase plan.
    """

    if emotion_quadrant is not None and emotion_quadrant not in {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }:
        raise ValueError(f"unsupported emotion quadrant: {emotion_quadrant}")
    slots = (
        PhraseSlot(
            slot_index=0,
            label="A",
            start_bar=0,
            end_bar=2,
            role=PhraseRole.OPENING,
            relation=MotifRelation.INTRODUCE,
            source_slot=None,
            contour_target="ARCH",
            register_band="MID",
            density_band="MODERATE",
            cadence_goal=CadenceGoal.OPEN,
        ),
        PhraseSlot(
            slot_index=1,
            label="A_PRIME",
            start_bar=2,
            end_bar=4,
            role=PhraseRole.RESPONSE,
            relation=MotifRelation.VARIATE,
            source_slot=0,
            contour_target="ANSWER",
            register_band="MID",
            density_band="MODERATE",
            cadence_goal=CadenceGoal.HALF,
        ),
        PhraseSlot(
            slot_index=2,
            label="B",
            start_bar=4,
            end_bar=6,
            role=PhraseRole.DEVELOPMENT,
            relation=MotifRelation.CONTRAST,
            source_slot=0,
            contour_target="RISE_TO_CLIMAX",
            register_band="UPPER_MID",
            density_band="ACTIVE",
            cadence_goal=CadenceGoal.OPEN,
            contains_climax=True,
        ),
        PhraseSlot(
            slot_index=3,
            label="A_CAD",
            start_bar=6,
            end_bar=8,
            role=PhraseRole.CADENCE,
            relation=MotifRelation.CADENCE_RETURN,
            source_slot=0,
            contour_target="RESOLVE",
            register_band="MID",
            density_band="RELEASE",
            cadence_goal=CadenceGoal.TONIC,
        ),
    )
    plan = PhrasePlan(
        total_bars=8,
        slots=slots,
        climax_bar=5,
        emotion_quadrant=emotion_quadrant,
        metadata={
            "planner_kind": "deterministic_architecture_contract",
            "learned": False,
            "formalStudyData": False,
        },
    )
    plan.validate()
    return plan
