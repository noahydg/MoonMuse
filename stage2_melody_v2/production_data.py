"""Production-backbone tensors derived from the frozen V101 real examples."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .neural_data import Stage2NeuralExample
from .segment_contract import segment_ids_for_tokens


def segment_ids_for_example(
    example: Stage2NeuralExample, event2idx: Mapping[str, int]
) -> tuple[int, ...]:
    return segment_ids_for_tokens(example.input_ids, event2idx)


def collate_production_examples(
    examples: Sequence[Stage2NeuralExample],
    *,
    event2idx: Mapping[str, int],
    pad_token: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("at least one production example is required")
    maximum = max(example.sequence_length for example in examples)
    inputs = torch.full(
        (len(examples), maximum), pad_token, dtype=torch.long, device=device
    )
    targets = torch.full_like(inputs, pad_token)
    token_slots = torch.zeros_like(inputs)
    segment_ids = torch.zeros_like(inputs)
    attention_mask = torch.zeros_like(inputs)
    for index, example in enumerate(examples):
        length = example.sequence_length
        inputs[index, :length] = torch.tensor(
            example.input_ids, dtype=torch.long, device=device
        )
        targets[index, :length] = torch.tensor(
            example.targets, dtype=torch.long, device=device
        )
        token_slots[index, :length] = torch.tensor(
            example.token_slot_ids, dtype=torch.long, device=device
        )
        segment_ids[index, :length] = torch.tensor(
            segment_ids_for_example(example, event2idx),
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, :length] = 1
    return {
        "inputs": inputs,
        "targets": targets,
        "token_slot_ids": token_slots,
        "segment_ids": segment_ids,
        "attention_mask": attention_mask,
    }
