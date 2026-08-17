import pytest


torch = pytest.importorskip("torch")

from stage2_melody_v2.anchor_conditioning import compile_anchor_planner_example
from stage2_melody_v2.anchor_neural import (
    AnchorPlannerConfig,
    CausalAnchorPlanner,
    tensorize_anchor_examples,
)


def piece_tokens() -> list[str]:
    output = ["Emotion_Q1", "Key_C"]
    for bar in range(8):
        root = "I" if bar % 2 == 0 else "V"
        output.extend(
            [
                "Track_Chord",
                "Bar_None",
                "Beat_0",
                f"Chord_Root_{root}",
                "Chord_Quality_M",
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


def model_and_batch():
    torch.manual_seed(127)
    example = compile_anchor_planner_example(piece_tokens())
    batch = tensorize_anchor_examples([example], device=torch.device("cpu"))
    model = CausalAnchorPlanner(
        AnchorPlannerConfig(d_model=16, n_head=4, n_layer=1, d_ff=32)
    )
    return model, batch


def test_forward_backward_reaches_all_three_heads() -> None:
    model, batch = model_and_batch()
    logits = model(batch)
    losses = model.compute_loss(logits, batch)
    losses["total_loss"].backward()
    for name in ("active_head", "degree_head", "octave_head"):
        assert sum(
            float(parameter.grad.abs().sum())
            for key, parameter in model.named_parameters()
            if key.startswith(name) and parameter.grad is not None
        ) > 0


def test_future_input_change_cannot_change_earlier_logits() -> None:
    model, batch = model_and_batch()
    model.eval()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["previous_degree"][:, 20:] = (
        changed["previous_degree"][:, 20:] + 1
    ) % model.previous_degree_embedding.num_embeddings
    with torch.no_grad():
        left = model(batch)
        right = model(changed)
    for name in left:
        assert torch.equal(left[name][:, :20], right[name][:, :20])


def test_future_chord_counterfactual_reaches_earlier_logits() -> None:
    model, batch = model_and_batch()
    model.eval()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["chord_root"][:, 20:] = (
        changed["chord_root"][:, 20:] + 1
    ) % model.chord_root_embedding.num_embeddings
    with torch.no_grad():
        left = model(batch)
        right = model(changed)
    assert any(
        not torch.equal(left[name][:, :20], right[name][:, :20])
        for name in left
    )
