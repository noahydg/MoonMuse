"""SS26 Stage 2 PAD diagnostic: the two training objectives differ only in PAD handling."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]


def load_model_class():
    sys.path.insert(0, str(ROOT / "stage2_melody"))
    from model.music_gpt2 import ImprovedMusicGPT2  # noqa: E402
    return ImprovedMusicGPT2


def test_count_and_ignore_objectives_differ_only_through_pad():
    Model = load_model_class()
    torch.manual_seed(0)
    n_token = 12  # PAD = n_token - 1 = 11, as built by the Stage 2 dataset
    model = Model(n_token, 1, 2, 16, 32, 16)
    logits = torch.randn(2, 6, n_token)
    targets = torch.tensor([[1, 2, 3, 11, 11, 11], [4, 5, 11, 11, 11, 11]])
    ignore = model.compute_loss(logits, targets)["recons_loss"]
    count = model.compute_loss(logits, targets, ignore_index=-100)["recons_loss"]
    nonpad = torch.nn.functional.cross_entropy(
        logits.view(-1, n_token), targets.view(-1), ignore_index=11)
    assert torch.isclose(ignore, nonpad)
    assert not torch.isclose(ignore, count)
    no_pad = torch.tensor([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]])
    assert torch.isclose(model.compute_loss(logits, no_pad)["recons_loss"],
                         model.compute_loss(logits, no_pad, ignore_index=-100)["recons_loss"])


def test_train_cli_exposes_pad_switch_with_backward_compatible_defaults():
    src = (ROOT / "stage2_melody" / "train.py").read_text()
    assert "choices=['ignore', 'count'], default='ignore'" in src
    assert "choices=['own', 'nonpad_ce'], default='own'" in src
    assert "PAD_LOSS_KWARGS = {'ignore_index': -100} if args.pad_loss == 'count' else {}" in src
    assert src.count("**PAD_LOSS_KWARGS") == 2
