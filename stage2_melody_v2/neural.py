"""Causally safe neural fixture for the MoonMuse V2 realizer path.

This module intentionally ignores the unvalidated V97 relation and cadence IDs.  It
uses only fixed structural roles plus motif memory gathered from an explicitly earlier
source cell.  The native PyTorch backbone keeps the first neural smoke independent of
the production server's optional Transformer dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .conditioning import (
    INTERVAL_TOKENS,
    MAX_RHYTHM_GAP,
    ROLE_TO_ID,
)


NEURAL_SCHEMA = "moonmuse-stage2-v2-causal-neural-fixture-v1"
CELL_COUNT = 4


@dataclass(frozen=True)
class V2NeuralConfig:
    vocab_size: int
    pad_token: int
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    d_ff: int = 128
    max_seq_len: int = 512
    dropout: float = 0.0
    schema: str = NEURAL_SCHEMA

    def validate(self) -> None:
        if self.schema != NEURAL_SCHEMA:
            raise ValueError(f"unsupported neural schema: {self.schema}")
        if self.vocab_size < 2:
            raise ValueError("vocabulary must contain at least two tokens")
        if not 0 <= self.pad_token < self.vocab_size:
            raise ValueError("PAD token is outside the vocabulary")
        if self.d_model < 4 or self.d_model % self.n_head:
            raise ValueError("d_model must be positive and divisible by n_head")
        if self.n_layer < 1 or self.d_ff < self.d_model:
            raise ValueError("invalid Transformer depth or feed-forward width")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len is too small")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def tensorize_realizer_conditions(
    payloads: Sequence[Mapping[str, Any]], *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Convert V98 JSON conditions without exposing relation/cadence semantics.

    The returned dictionary deliberately has no ``relation_ids`` or ``cadence_ids``.
    Motifs remain separated by source cell; causal gathering happens inside the
    condition encoder.
    """

    if not payloads:
        raise ValueError("at least one realizer condition is required")
    max_motif_len = max(
        1,
        max(
            int(length)
            for payload in payloads
            for length in payload["motif_lengths"]
        ),
    )
    batch_size = len(payloads)
    role_ids = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    source_slot_ids = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    motif_lengths = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    motif_interval_ids = torch.zeros(
        (batch_size, CELL_COUNT, max_motif_len), dtype=torch.long
    )
    motif_rhythm_ids = torch.zeros_like(motif_interval_ids)

    for batch_index, payload in enumerate(payloads):
        if payload.get("schema") != "moonmuse-stage2-v2-realizer-condition-v1":
            raise ValueError("unexpected realizer-condition schema")
        fields = (
            payload["role_ids"],
            payload["source_slot_ids"],
            payload["motif_lengths"],
            payload["motif_interval_ids"],
            payload["motif_rhythm_ids"],
        )
        if not all(len(field) == CELL_COUNT for field in fields):
            raise ValueError("realizer condition must contain four cells")
        role_ids[batch_index] = torch.tensor(payload["role_ids"], dtype=torch.long)
        source_slot_ids[batch_index] = torch.tensor(
            payload["source_slot_ids"], dtype=torch.long
        )
        motif_lengths[batch_index] = torch.tensor(
            payload["motif_lengths"], dtype=torch.long
        )
        for slot in range(CELL_COUNT):
            interval_values = tuple(int(x) for x in payload["motif_interval_ids"][slot])
            rhythm_values = tuple(int(x) for x in payload["motif_rhythm_ids"][slot])
            length = int(payload["motif_lengths"][slot])
            if len(interval_values) != len(rhythm_values) or len(interval_values) != length:
                raise ValueError(f"motif feature length mismatch in cell {slot}")
            if any(not 0 <= value < len(INTERVAL_TOKENS) for value in interval_values):
                raise ValueError("motif interval ID is outside the frozen vocabulary")
            if any(not 1 <= value <= MAX_RHYTHM_GAP for value in rhythm_values):
                raise ValueError("motif rhythm gap is outside the frozen range")
            if length:
                motif_interval_ids[batch_index, slot, :length] = torch.tensor(
                    interval_values, dtype=torch.long
                )
                motif_rhythm_ids[batch_index, slot, :length] = torch.tensor(
                    rhythm_values, dtype=torch.long
                )

    return {
        "role_ids": role_ids.to(device),
        "source_slot_ids": source_slot_ids.to(device),
        "motif_lengths": motif_lengths.to(device),
        "motif_interval_ids": motif_interval_ids.to(device),
        "motif_rhythm_ids": motif_rhythm_ids.to(device),
    }


class CausalMotifConditionEncoder(nn.Module):
    """Encode four plan cells while gathering motif memory only from prior cells."""

    semantic_relation_ids_used = False
    semantic_cadence_ids_used = False

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.slot_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.role_embedding = nn.Embedding(len(ROLE_TO_ID), d_model)
        # Index zero represents "no source"; source slots 0..3 map to 1..4.
        self.source_slot_embedding = nn.Embedding(CELL_COUNT + 1, d_model)
        self.interval_embedding = nn.Embedding(
            len(INTERVAL_TOKENS), d_model, padding_idx=0
        )
        self.rhythm_embedding = nn.Embedding(
            MAX_RHYTHM_GAP + 1, d_model, padding_idx=0
        )
        self.motif_projection = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _validate_source_slots(source_slot_ids: torch.Tensor) -> None:
        if source_slot_ids.ndim != 2 or source_slot_ids.size(1) != CELL_COUNT:
            raise ValueError("source_slot_ids must have shape [batch, 4]")
        target_slots = torch.arange(
            CELL_COUNT, device=source_slot_ids.device
        ).unsqueeze(0)
        invalid = (source_slot_ids >= target_slots) | (source_slot_ids < -1)
        if torch.any(invalid):
            raise ValueError("motif source must be absent or strictly earlier")

    def encode_slots(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        role_ids = condition["role_ids"]
        source_slot_ids = condition["source_slot_ids"]
        interval_ids = condition["motif_interval_ids"]
        rhythm_ids = condition["motif_rhythm_ids"]
        lengths = condition["motif_lengths"]
        if role_ids.shape != source_slot_ids.shape or role_ids.shape[1] != CELL_COUNT:
            raise ValueError("role/source tensors must have shape [batch, 4]")
        if interval_ids.shape != rhythm_ids.shape or interval_ids.ndim != 3:
            raise ValueError("motif tensors must have matching [batch, 4, length] shape")
        if interval_ids.shape[:2] != role_ids.shape or lengths.shape != role_ids.shape:
            raise ValueError("motif tensors do not match the four-cell batch")
        self._validate_source_slots(source_slot_ids)

        steps = torch.arange(interval_ids.size(2), device=interval_ids.device)
        motif_mask = steps.view(1, 1, -1) < lengths.unsqueeze(-1)
        motif_events = self.interval_embedding(interval_ids) + self.rhythm_embedding(
            rhythm_ids
        )
        motif_events = motif_events * motif_mask.unsqueeze(-1)
        denominator = lengths.clamp(min=1).unsqueeze(-1).to(motif_events.dtype)
        motif_by_source = motif_events.sum(dim=2) / denominator

        safe_source = source_slot_ids.clamp(min=0)
        gather_index = safe_source.unsqueeze(-1).expand(-1, -1, self.d_model)
        source_memory = torch.gather(motif_by_source, 1, gather_index)
        has_source = source_slot_ids.ge(0).unsqueeze(-1)
        source_memory = self.motif_projection(source_memory) * has_source

        slot_ids = torch.arange(CELL_COUNT, device=role_ids.device).unsqueeze(0)
        slot_ids = slot_ids.expand(role_ids.size(0), -1)
        source_embedding_ids = source_slot_ids + 1
        combined = (
            self.slot_embedding(slot_ids)
            + self.role_embedding(role_ids)
            + self.source_slot_embedding(source_embedding_ids)
            + source_memory
        )
        return self.output_norm(combined)

    def forward(
        self,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if token_slot_ids.ndim != 2:
            raise ValueError("token_slot_ids must have shape [batch, sequence]")
        if torch.any((token_slot_ids < 0) | (token_slot_ids >= CELL_COUNT)):
            raise ValueError("token slot ID is outside 0..3")
        slot_conditions = self.encode_slots(condition)
        if slot_conditions.size(0) != token_slot_ids.size(0):
            raise ValueError("condition batch does not match token batch")
        gather_index = token_slot_ids.unsqueeze(-1).expand(
            -1, -1, self.d_model
        )
        return torch.gather(slot_conditions, 1, gather_index)


class V2CausalTransformer(nn.Module):
    """Small decoder-only Transformer used to prove the V2 neural data path."""

    def __init__(self, config: V2NeuralConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.condition_encoder = CausalMotifConditionEncoder(config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layer)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
        *,
        keep_last_only: bool = False,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape != token_slot_ids.shape:
            raise ValueError("tokens and token_slot_ids must share [batch, sequence]")
        if tokens.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds frozen V101 maximum")
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
            + self.condition_encoder(token_slot_ids, condition)
        )
        hidden = self.input_norm(hidden)
        causal_mask = torch.triu(
            torch.ones(
                tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=tokens.eq(self.config.pad_token),
        )
        logits = self.lm_head(self.output_norm(hidden))
        return logits[:, -1, :] if keep_last_only else logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1).long(),
            ignore_index=self.config.pad_token,
        )

    def one_step_logits(
        self,
        tokens: torch.Tensor,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.forward(
            tokens, token_slot_ids, condition, keep_last_only=True
        )
