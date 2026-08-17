from dataclasses import replace

import pytest

from stage2_melody_v2.factorized_control import (
    AffectTarget,
    ArousalTarget,
    DevelopmentDegree,
    FactorizedControlPlan,
    MotifOperation,
    ValenceTarget,
    build_factorized_eight_bar_control,
    encode_factorized_control,
    factorized_control_from_phrase_plan,
)
from stage2_melody_v2.planner import build_eight_bar_plan


def test_contrast_keeps_comparison_anchor_but_introduces_new_identity():
    control = build_factorized_eight_bar_control("Q1")
    encoded = encode_factorized_control(control)

    assert encoded.comparison_source_slot_ids == (-1, 0, 0, 0)
    assert encoded.identity_source_slot_ids == (-1, 0, -1, 0)
    assert encoded.identity_family_ids == (0, 0, 1, 0)
    assert control.slots[2].operation_path == (MotifOperation.CONTRAST,)


def test_pitch_rhythm_and_affect_axes_change_independently():
    baseline = build_factorized_eight_bar_control("Q1")
    response = baseline.slots[1]
    alternatives = {
        "d_pitch": replace(response, pitch_degree=DevelopmentDegree.MODERATE),
        "d_rhythm": replace(response, rhythm_degree=DevelopmentDegree.MODERATE),
        "e_t": replace(
            response,
            affect_target=AffectTarget(
                valence=ValenceTarget.NEGATIVE,
                arousal=ArousalTarget.LOW,
            ),
        ),
    }
    original = response.to_dict()
    for expected_key, alternative in alternatives.items():
        changed = alternative.to_dict()
        assert {
            key for key in original if original[key] != changed[key]
        } == {expected_key}


def test_nonzero_degree_requires_matching_operation_axis():
    control = build_factorized_eight_bar_control("Q1")
    invalid = replace(
        control.slots[1],
        operation_path=(MotifOperation.REPEAT,),
        pitch_degree=DevelopmentDegree.LIGHT,
        rhythm_degree=DevelopmentDegree.NONE,
    )
    with pytest.raises(ValueError, match="pitch operation"):
        invalid.validate_local()


def test_identity_source_must_point_to_the_same_identity_family():
    control = build_factorized_eight_bar_control("Q1")
    invalid_cadence = replace(control.slots[3], identity_label="B")
    invalid_plan = replace(
        control,
        slots=control.slots[:3] + (invalid_cadence,),
    )
    with pytest.raises(ValueError, match="identity labels differ"):
        invalid_plan.validate()


def test_affect_trajectory_is_per_cell_and_does_not_change_structure():
    trajectory = (
        AffectTarget(ValenceTarget.NEGATIVE, ArousalTarget.LOW),
        AffectTarget(ValenceTarget.NEGATIVE, ArousalTarget.HIGH),
        AffectTarget(ValenceTarget.POSITIVE, ArousalTarget.HIGH),
        AffectTarget(ValenceTarget.POSITIVE, ArousalTarget.LOW),
    )
    baseline = build_factorized_eight_bar_control("Q1")
    changed = build_factorized_eight_bar_control(affect_trajectory=trajectory)
    for left, right in zip(baseline.slots, changed.slots):
        left_payload = left.to_dict()
        right_payload = right.to_dict()
        assert {
            key for key in left_payload if left_payload[key] != right_payload[key]
        } <= {"e_t"}
    encoded = encode_factorized_control(changed)
    assert len(set(encoded.valence_ids)) == 2
    assert len(set(encoded.arousal_ids)) == 2


def test_legacy_plan_compiles_without_treating_contrast_as_identity_reuse():
    legacy = build_eight_bar_plan("Q4")
    legacy_fingerprint = legacy.fingerprint()
    compiled = factorized_control_from_phrase_plan(legacy)
    assert legacy.fingerprint() == legacy_fingerprint
    assert compiled.slots[2].comparison_source_slot == 0
    assert compiled.slots[2].identity_source_slot is None
    assert compiled.fingerprint() == factorized_control_from_phrase_plan(
        legacy
    ).fingerprint()


def test_encoded_contract_uses_independent_axis_arrays_and_stable_padding():
    encoded = encode_factorized_control(build_factorized_eight_bar_control("Q3"))
    payload = encoded.to_dict()
    assert payload["schema"] == (
        "moonmuse-stage2-v2-factorized-control-condition-v1"
    )
    assert payload["bar_to_slot_ids"] == (0, 0, 1, 1, 2, 2, 3, 3)
    assert payload["operation_lengths"] == (1, 2, 1, 2)
    assert payload["operation_path_ids"][0][-1] == 0
    assert payload["operation_path_ids"][1][-1] != 0
    assert "pitch_degree_ids" in payload
    assert "rhythm_degree_ids" in payload
    assert "valence_ids" in payload
    assert "arousal_ids" in payload


def test_new_identity_cannot_claim_within_identity_development_degree():
    control = build_factorized_eight_bar_control("Q2")
    invalid_b = replace(control.slots[2], pitch_degree=DevelopmentDegree.STRONG)
    with pytest.raises(ValueError, match="within-identity degrees"):
        invalid_b.validate_local()
