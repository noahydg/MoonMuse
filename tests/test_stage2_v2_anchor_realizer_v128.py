import pytest


torch = pytest.importorskip("torch")

from stage2_melody.model.music_gpt2 import ImprovedMusicGPT2
from stage2_melody_v2.anchor_conditioning import compile_anchor_planner_example
from stage2_melody_v2.anchor_neural import (
    AnchorPlannerConfig,
    CausalAnchorPlanner,
    predict_anchor_plan_autoregressive,
    predict_anchor_plans_autoregressive,
    sample_anchor_plans_autoregressive,
)
from stage2_melody_v2.anchor_realizer import (
    AnchorConditionedDenseMusicGPT2,
    AnchorDenseRealizerConfig,
)


def planner_tokens(degree: str = "I") -> list[str]:
    output = ["Emotion_Q1", "Key_C"]
    for _ in range(8):
        output.extend(
            [
                "Track_Chord",
                "Bar_None",
                "Beat_0",
                "Chord_Root_I",
                "Chord_Quality_M",
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_4",
                f"Note_Degree_{degree}",
            ]
        )
    return output


def small_dense_models():
    torch.manual_seed(128)
    config = AnchorDenseRealizerConfig(
        vocab_size=23,
        pad_token=22,
        n_layer=1,
        n_head=2,
        d_model=16,
        d_ff=32,
        d_embed=16,
        max_input_len=32,
        backbone_max_positions=64,
        attention_window=16,
        dropout=0.0,
        anchor_encoder_layers=1,
    )
    base = ImprovedMusicGPT2(
        config.vocab_size,
        config.n_layer,
        config.n_head,
        config.d_model,
        config.d_ff,
        config.d_embed,
        dropout=0.0,
        max_seq_len=config.backbone_max_positions,
        attention_window=config.attention_window,
    ).eval()
    model = AnchorConditionedDenseMusicGPT2(config).eval()
    loaded = model.load_v1_backbone_state_dict(base.state_dict())
    return base, model, loaded


def condition(batch: int = 1, *, degree: int = 0):
    return {
        "active": torch.ones(batch, 32, dtype=torch.long),
        "degree": torch.full((batch, 32), degree, dtype=torch.long),
        "octave": torch.full((batch, 32), 4, dtype=torch.long),
    }


def test_zero_initialized_anchor_path_preserves_v1_logits_exactly() -> None:
    base, model, loaded = small_dense_models()
    assert loaded["loaded_v1_backbone_key_count"] == len(base.state_dict())
    tokens = torch.randint(0, 22, (1, 12))
    segments = torch.zeros_like(tokens)
    slots = torch.arange(12).unsqueeze(0)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        expected = base(tokens, seg_inp=segments)
        actual = model(
            tokens,
            seg_inp=segments,
            anchor_condition=condition(),
            token_anchor_slot_ids=slots,
            melody_condition_mask=mask,
        )
    assert torch.equal(expected, actual)


def test_direct_anchor_residual_is_zero_outside_melody_mask() -> None:
    _, model, _ = small_dense_models()
    with torch.no_grad():
        torch.nn.init.eye_(model.anchor_condition_encoder.output_projection.weight)
    slots = torch.arange(6).unsqueeze(0)
    mask = torch.tensor([[0, 0, 1, 1, 0, 1]])
    residual = model.anchor_token_residual(
        anchor_condition=condition(),
        token_anchor_slot_ids=slots,
        melody_condition_mask=mask,
    )
    assert torch.count_nonzero(residual[:, mask[0] == 0]) == 0
    assert torch.count_nonzero(residual[:, mask[0] == 1]) > 0


def test_content_contrast_removes_position_only_shortcut_exactly() -> None:
    _, model, _ = small_dense_models()
    with torch.no_grad():
        torch.nn.init.eye_(model.anchor_condition_encoder.output_projection.weight)
        model.anchor_condition_encoder.active_embedding.weight.zero_()
        model.anchor_condition_encoder.degree_embedding.weight.zero_()
        model.anchor_condition_encoder.octave_embedding.weight.zero_()
    slots = torch.arange(6).unsqueeze(0)
    mask = torch.ones_like(slots)
    residual = model.anchor_token_residual(
        anchor_condition=condition(),
        token_anchor_slot_ids=slots,
        melody_condition_mask=mask,
    )
    assert torch.equal(residual, torch.zeros_like(residual))


def test_full_plan_can_change_early_conditioned_token() -> None:
    _, model, _ = small_dense_models()
    with torch.no_grad():
        torch.nn.init.eye_(model.anchor_condition_encoder.output_projection.weight)
        left = condition()
        right = {name: value.clone() for name, value in left.items()}
        right["degree"][:, 31] = 4
        slots = torch.zeros((1, 1), dtype=torch.long)
        mask = torch.ones((1, 1), dtype=torch.long)
        left_residual = model.anchor_token_residual(
            anchor_condition=left,
            token_anchor_slot_ids=slots,
            melody_condition_mask=mask,
        )
        right_residual = model.anchor_token_residual(
            anchor_condition=right,
            token_anchor_slot_ids=slots,
            melody_condition_mask=mask,
        )
    assert not torch.equal(left_residual, right_residual)


def test_autoregressive_planner_does_not_use_true_previous_anchors() -> None:
    torch.manual_seed(128)
    planner = CausalAnchorPlanner(
        AnchorPlannerConfig(d_model=16, n_head=4, n_layer=1, d_ff=32)
    ).eval()
    left = compile_anchor_planner_example(planner_tokens("I"))
    right = compile_anchor_planner_example(planner_tokens("V"))
    assert left.target_degree_ids != right.target_degree_ids
    left_prediction = predict_anchor_plan_autoregressive(
        planner, left, device=torch.device("cpu")
    )
    right_prediction = predict_anchor_plan_autoregressive(
        planner, right, device=torch.device("cpu")
    )
    assert left_prediction["used_ground_truth_previous_anchors"] is False
    assert right_prediction["used_ground_truth_previous_anchors"] is False
    assert left_prediction["active_ids"] == right_prediction["active_ids"]
    assert left_prediction["degree_ids"] == right_prediction["degree_ids"]
    assert left_prediction["octave_ids"] == right_prediction["octave_ids"]


def test_batched_autoregressive_decode_matches_single_decode() -> None:
    torch.manual_seed(128)
    planner = CausalAnchorPlanner(
        AnchorPlannerConfig(d_model=16, n_head=4, n_layer=1, d_ff=32)
    ).eval()
    examples = [
        compile_anchor_planner_example(planner_tokens("I")),
        compile_anchor_planner_example(planner_tokens("V")),
    ]
    batched = predict_anchor_plans_autoregressive(
        planner, examples, device=torch.device("cpu")
    )
    singles = [
        predict_anchor_plan_autoregressive(
            planner, example, device=torch.device("cpu")
        )
        for example in examples
    ]
    for left, right in zip(batched, singles):
        assert left["active_ids"] == right["active_ids"]
        assert left["degree_ids"] == right["degree_ids"]
        assert left["octave_ids"] == right["octave_ids"]
        assert left["sequence_log_probability"] == pytest.approx(
            right["sequence_log_probability"], abs=1e-6
        )


def test_sampled_autoregression_is_reproducible_and_label_free() -> None:
    torch.manual_seed(128)
    planner = CausalAnchorPlanner(
        AnchorPlannerConfig(d_model=16, n_head=4, n_layer=1, d_ff=32)
    ).eval()
    examples = [
        compile_anchor_planner_example(planner_tokens("I")),
        compile_anchor_planner_example(planner_tokens("V")),
    ]
    left = sample_anchor_plans_autoregressive(
        planner,
        examples,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3128),
    )
    right = sample_anchor_plans_autoregressive(
        planner,
        examples,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3128),
    )
    assert left == right
    assert all(not item["used_ground_truth_previous_anchors"] for item in left)
    assert all(len(item["active_ids"]) == 32 for item in left)
