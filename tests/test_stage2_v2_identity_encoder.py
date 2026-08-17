import pytest


torch = pytest.importorskip("torch")

from stage2_melody_v2.identity_encoder import (  # noqa: E402
    MotifIdentityEncoder,
    symmetric_nt_xent_loss,
    views_to_tensors,
)


def view(offset=0):
    return {
        "relative_pitches": [0, 2 + offset, 5 + offset, 7 + offset],
        "relative_positions": [0, 2, 6, 10],
    }


def test_encoder_shapes_and_normalization():
    model = MotifIdentityEncoder()
    source_features, source_mask = views_to_tensors(
        [view(0), view(1), view(2)], device=torch.device("cpu")
    )
    transformed_features, transformed_mask = views_to_tensors(
        [view(2), view(3), view(4)], device=torch.device("cpu")
    )
    outputs = model(
        source_features,
        source_mask,
        transformed_features,
        transformed_mask,
    )
    assert outputs["source_raw"].shape == (3, 64)
    assert outputs["source_identity"].shape == (3, 32)
    assert outputs["operation_logits"].shape == (3, 5)
    assert torch.allclose(
        outputs["source_identity"].norm(dim=-1), torch.ones(3), atol=1e-6
    )


def test_nt_xent_is_finite_and_has_gradients():
    source = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    transformed = source.detach().clone().requires_grad_(True)
    loss = symmetric_nt_xent_loss(source, transformed, temperature=0.10)
    assert torch.isfinite(loss)
    loss.backward()
    assert transformed.grad is not None
    assert float(transformed.grad.norm()) > 0.0
