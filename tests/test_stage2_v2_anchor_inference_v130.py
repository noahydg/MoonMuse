import numpy as np
import pytest


torch = pytest.importorskip("torch")

from stage2_melody_v2.anchor_dense_conditioning import PredictedAnchorArrays
from stage2_melody_v2.anchor_inference import (
    _nucleus_sample,
    decoder_window_for_prefix,
    dense_melody_event_is_admissible,
    non_event_output_ids,
    predict_plan_from_stage1_tokens,
    strip_terminal_dense_boundary,
    token_routes_for_prefix,
)
from stage2_melody_v2.anchor_conditioning import compile_anchor_planner_context
from scripts.run_stage2_v2_anchor_fresh_generation_v130 import stage1_tokens


IDX = {
    0: "Emotion_Q1",
    1: "Key_C",
    2: "Track_Chord",
    3: "Bar_None",
    4: "Beat_0",
    5: "Chord_Root_I",
    6: "Chord_Quality_M",
    7: "Track_Melody",
    8: "Beat_8",
    9: "Note_Octave_4",
    10: "Note_Degree_III",
}


def test_partial_token_routing_matches_dense_v1_segments():
    tokens = list(IDX)
    segments, slots, mask = token_routes_for_prefix(tokens, idx2event=IDX)
    assert segments == mask
    assert segments[:7] == [0] * 7
    assert segments[7:] == [1] * 4
    assert slots[7] == 0
    assert slots[8:] == [2, 2, 2]


def test_nucleus_sampling_is_repeatable_and_respects_mask():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    first = _nucleus_sample(
        logits,
        temperature=1.0,
        top_p=0.95,
        inadmissibles=[0],
        rng=np.random.RandomState(130),
    )
    second = _nucleus_sample(
        logits,
        temperature=1.0,
        top_p=0.95,
        inadmissibles=[0],
        rng=np.random.RandomState(130),
    )
    assert first == second
    assert first != 0


def test_loss_only_pad_output_is_masked_before_dictionary_lookup():
    idx = {0: "Beat_0", 1: "Track_Chord", 2: "Track_Melody"}
    masked = non_event_output_ids(4, idx)
    assert masked == [3]
    sampled = _nucleus_sample(
        np.array([1.0, 1.0, 1.0, 100.0]),
        temperature=1.0,
        top_p=0.95,
        inadmissibles=masked,
        rng=np.random.RandomState(130),
    )
    assert sampled in idx


def test_terminal_dense_boundary_cleanup_is_exact():
    idx = {0: "Note_Degree_I", 1: "Track_Chord", 2: "EOS_None"}
    assert strip_terminal_dense_boundary([0, 1], idx) == [0]
    assert strip_terminal_dense_boundary([0, 1, 2], idx) == [0]
    assert strip_terminal_dense_boundary([0], idx) == [0]


def test_dense_melody_grammar_rejects_only_malformed_pair_events():
    idx = {
        0: "Track_Melody",
        1: "Bar_None",
        2: "Beat_0",
        3: "Note_Octave_4",
        4: "Note_Degree_I",
        5: "Chord_Root_I",
        6: "Chord_Quality_M",
        7: "Track_Chord",
        8: "Emotion_Q1",
    }
    assert dense_melody_event_is_admissible(
        "Bar_None", generated=[0], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Note_Degree_I", generated=[0], idx2event=idx
    )
    assert dense_melody_event_is_admissible(
        "Beat_0", generated=[0, 1], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Track_Chord", generated=[0, 1], idx2event=idx
    )
    assert dense_melody_event_is_admissible(
        "Note_Degree_I", generated=[0, 1, 2, 3], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Track_Chord", generated=[0, 1, 2, 3], idx2event=idx
    )
    assert dense_melody_event_is_admissible(
        "Chord_Quality_M", generated=[0, 1, 2, 5], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Note_Octave_4", generated=[0, 1, 2, 5], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Note_Degree_I", generated=[0, 1, 2, 4], idx2event=idx
    )
    assert not dense_melody_event_is_admissible(
        "Emotion_Q1", generated=[0, 1, 2, 4], idx2event=idx
    )
    assert dense_melody_event_is_admissible(
        "Track_Chord", generated=[0, 1, 2, 4], idx2event=idx
    )


def test_decoder_window_preserves_global_melody_route_after_truncation():
    idx = {
        0: "Emotion_Q1",
        1: "Key_C",
        2: "Track_Chord",
        3: "Bar_None",
        4: "Track_Melody",
        5: "Beat_12",
        6: "Note_Octave_4",
        7: "Note_Degree_I",
    }
    prefix = [0, 1, 2, 3, 4, 5] + [6, 7] * 400
    window, segments, slots, mask = decoder_window_for_prefix(
        prefix, idx2event=idx
    )
    assert len(window) == len(segments) == len(slots) == len(mask) == 768
    assert set(segments) == {1}
    assert set(mask) == {1}
    assert set(slots) == {3}


def test_production_plan_helper_has_no_target_argument(monkeypatch):
    expected = {
        "active_ids": tuple([1] * 32),
        "degree_ids": tuple([0] * 32),
        "octave_ids": tuple([4] * 32),
    }
    monkeypatch.setattr(
        "stage2_melody_v2.anchor_inference.compile_anchor_planner_context",
        lambda tokens: object(),
    )
    monkeypatch.setattr(
        "stage2_melody_v2.anchor_inference.predict_anchor_plan_autoregressive",
        lambda planner, context, device: expected,
    )
    planner = torch.nn.Linear(1, 1)
    plan = predict_plan_from_stage1_tokens(
        planner, stage1_tokens=["no-melody-label"], device=torch.device("cpu")
    )
    assert isinstance(plan, PredictedAnchorArrays)
    assert plan.active_ids == expected["active_ids"]


def test_raw_stage1_export_is_adapted_to_eight_chord_segments(tmp_path):
    source = tmp_path / "samp_00_Positive.txt"
    chord_bar = [
        "Bar_None",
        "Beat_0",
        "Chord_Root_I",
        "Chord_Quality_M",
        "Beat_4",
        "Chord_Root_Conti",
        "Chord_Quality_Conti",
        "Beat_8",
        "Chord_Root_Conti",
        "Chord_Quality_Conti",
        "Beat_12",
        "Chord_Root_Conti",
        "Chord_Quality_Conti",
    ]
    source.write_text("\n".join(["Key_C", *(chord_bar * 12)]) + "\n", encoding="utf-8")
    tokens = stage1_tokens(source, "Q1")
    assert tokens[:3] == ["Emotion_Q1", "Key_C", "Track_Chord"]
    assert tokens.count("Track_Chord") == 8
    assert tokens.count("Bar_None") == 8
    context = compile_anchor_planner_context(tokens)
    assert len(context.chord_root_ids) == 32
    assert set(context.chord_root_ids) == {0}
