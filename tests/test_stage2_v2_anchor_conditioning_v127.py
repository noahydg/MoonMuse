from stage2_melody_v2.anchor_conditioning import (
    MISSING_CLASS_ID,
    PREVIOUS_BOS,
    compile_anchor_planner_context,
    compile_anchor_planner_example,
    compile_realizer_anchor_condition,
    planner_context_from_example,
)


def piece_tokens() -> list[str]:
    output = ["Emotion_Q2", "Key_d"]
    for bar in range(8):
        root = "I" if bar < 4 else "V"
        output.extend(
            [
                "Track_Chord",
                "Bar_None",
                "Beat_0",
                f"Chord_Root_{root}",
                "Chord_Quality_m",
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_4",
                f"Note_Degree_{root}",
                "Beat_8",
                "Note_Octave_5",
                f"Note_Degree_{root}",
            ]
        )
    return output


def test_planner_contract_is_exactly_32_and_causally_shifted() -> None:
    example = compile_anchor_planner_example(piece_tokens())
    assert len(example.slot_ids) == 32
    assert example.previous_active_ids[0] == PREVIOUS_BOS
    assert example.previous_degree_ids[1] == example.target_degree_ids[0]
    assert example.previous_octave_ids[1] == example.target_octave_ids[0]
    example.validate()


def test_future_target_change_cannot_change_earlier_planner_inputs() -> None:
    original_tokens = piece_tokens()
    changed_tokens = piece_tokens()
    late_index = max(
        index for index, value in enumerate(changed_tokens) if value == "Note_Octave_5"
    )
    changed_tokens[late_index] = "Note_Octave_6"
    original = compile_anchor_planner_example(original_tokens)
    changed = compile_anchor_planner_example(changed_tokens)
    changed_target_slot = 30
    assert original.target_octave_ids[changed_target_slot] != changed.target_octave_ids[changed_target_slot]
    assert original.previous_active_ids[: changed_target_slot + 1] == changed.previous_active_ids[: changed_target_slot + 1]
    assert original.previous_degree_ids[: changed_target_slot + 1] == changed.previous_degree_ids[: changed_target_slot + 1]
    assert original.previous_octave_ids[: changed_target_slot + 1] == changed.previous_octave_ids[: changed_target_slot + 1]


def test_chord_counterfactual_changes_context_not_targets() -> None:
    original_tokens = piece_tokens()
    changed_tokens = [
        "Chord_Root_III" if token == "Chord_Root_V" else token
        for token in original_tokens
    ]
    original = compile_anchor_planner_example(original_tokens)
    changed = compile_anchor_planner_example(changed_tokens)
    assert original.chord_root_ids != changed.chord_root_ids
    assert original.target_active_ids == changed.target_active_ids
    assert original.target_degree_ids == changed.target_degree_ids
    assert original.target_octave_ids == changed.target_octave_ids


def test_realizer_requires_explicit_predictions_and_exposes_no_labels() -> None:
    example = compile_anchor_planner_example(piece_tokens())
    payload = compile_realizer_anchor_condition(
        example,
        predicted_active_ids=example.target_active_ids,
        predicted_degree_ids=example.target_degree_ids,
        predicted_octave_ids=example.target_octave_ids,
    )
    assert payload["inference_source"] == "planner_predictions"
    assert payload["ground_truth_anchor_fields_present"] is False
    assert payload["relation_labels_present"] is False
    assert payload["cadence_labels_present"] is False
    assert len(payload["anchors"]) == 32


def test_rest_prediction_cannot_hide_pitch_labels() -> None:
    example = compile_anchor_planner_example(piece_tokens())
    active = list(example.target_active_ids)
    degrees = list(example.target_degree_ids)
    octaves = list(example.target_octave_ids)
    rest_slot = active.index(0)
    assert degrees[rest_slot] == octaves[rest_slot] == MISSING_CLASS_ID
    degrees[rest_slot] = 0
    try:
        compile_realizer_anchor_condition(
            example,
            predicted_active_ids=active,
            predicted_degree_ids=degrees,
            predicted_octave_ids=octaves,
        )
    except ValueError as error:
        assert "rest prediction" in str(error)
    else:
        raise AssertionError("rest prediction with a pitch label was accepted")


def test_inference_context_compiles_from_chords_without_melody_labels() -> None:
    tokens = piece_tokens()
    chord_only = []
    track = None
    for token in tokens:
        if token.startswith("Track_"):
            track = token.removeprefix("Track_")
        if track != "Melody":
            chord_only.append(token)
    direct = compile_anchor_planner_context(chord_only)
    from_training = planner_context_from_example(
        compile_anchor_planner_example(tokens)
    )
    assert direct == from_training
    assert not any("target" in key for key in direct.to_dict())
    assert not any("previous" in key for key in direct.to_dict())
