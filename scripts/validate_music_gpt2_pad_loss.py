#!/usr/bin/env python3
"""Prove that ImprovedMusicGPT2 ignores vocabulary PAD targets by default."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_MODULES = (
    "stage2_melody.model.music_gpt2",
    "stage3_performance.model.music_gpt2",
)


def validate_model(module_name: str) -> dict:
    model_class = importlib.import_module(module_name).ImprovedMusicGPT2
    vocabulary_size = 7
    pad_token = vocabulary_size - 1
    dummy_model = SimpleNamespace(n_token=vocabulary_size)
    base_logits = torch.tensor(
        [
            [
                [0.2, 1.1, -0.4, 0.0, 0.3, -0.1, 0.5],
                [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7],
                [-0.3, 0.2, 1.4, 0.1, -0.5, 0.0, 0.3],
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[1, pad_token, 2, pad_token]], dtype=torch.long)
    altered_logits = base_logits.clone()
    altered_logits[:, 1, :] = torch.tensor(
        [-90.0, -80.0, -70.0, -60.0, -50.0, 100.0, -40.0]
    )
    altered_logits[:, 3, :] = torch.tensor(
        [100.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0]
    )

    default_before = model_class.compute_loss(
        dummy_model, base_logits, targets
    )["total_loss"]
    default_after = model_class.compute_loss(
        dummy_model, altered_logits, targets
    )["total_loss"]
    expected = F.cross_entropy(
        base_logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
        ignore_index=pad_token,
    )
    explicit_before = model_class.compute_loss(
        dummy_model, base_logits, targets, ignore_index=-100
    )["total_loss"]
    explicit_after = model_class.compute_loss(
        dummy_model, altered_logits, targets, ignore_index=-100
    )["total_loss"]

    gradient_logits = base_logits.clone().requires_grad_(True)
    gradient_loss = model_class.compute_loss(
        dummy_model, gradient_logits, targets
    )["total_loss"]
    gradient_loss.backward()
    pad_gradient_max = max(
        float(gradient_logits.grad[:, index, :].abs().max().item())
        for index in (1, 3)
    )
    supervised_gradient_max = max(
        float(gradient_logits.grad[:, index, :].abs().max().item())
        for index in (0, 2)
    )
    default_invariant = torch.equal(default_before, default_after)
    matches_expected = torch.equal(default_before, expected)
    explicit_sensitive = abs(float(explicit_after - explicit_before)) > 1e-3
    passed = (
        default_invariant
        and matches_expected
        and explicit_sensitive
        and pad_gradient_max == 0.0
        and supervised_gradient_max > 0.0
    )
    return {
        "module": module_name,
        "passed": passed,
        "pad_token": pad_token,
        "default_loss_before": float(default_before.item()),
        "default_loss_after_pad_logit_change": float(default_after.item()),
        "explicit_minus100_loss_before": float(explicit_before.item()),
        "explicit_minus100_loss_after_pad_logit_change": float(explicit_after.item()),
        "default_invariant": default_invariant,
        "matches_expected_pad_ignored_loss": matches_expected,
        "explicit_override_is_sensitive": explicit_sensitive,
        "pad_gradient_max": pad_gradient_max,
        "supervised_gradient_max": supervised_gradient_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [validate_model(module_name) for module_name in MODEL_MODULES]
    report = {
        "schema": "moonmuse-improved-music-gpt2-pad-loss-validation-v1",
        "passed": all(result["passed"] for result in results),
        "results": results,
        "evidence_boundary": "tensor_objective_correctness_not_music_quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        return 1
    print(f"PAD_LOSS_TENSOR_VALIDATION_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
