"""Factorized, sequence-aware conditioning for the production Stage 2 backbone.

This module is the first trainable bridge between the factorized V2 control
contract and the V92 production model.  It intentionally activates only three
validated axes: the causal identity pointer (``z_id``), the ordered operation path
(``o_t``), and the per-cell affect target (``e_t``).  Pitch/rhythm degree targets
and comparison-only pointers remain outside the model input until they have a
defensible calibration or supervision source.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from .conditioning import INTERVAL_TOKENS, MAX_RHYTHM_GAP
from .factorized_control import (
    AROUSAL_TO_ID,
    OPERATION_TOKENS,
    VALENCE_TO_ID,
    FactorizedControlPlan,
    encode_factorized_control,
)
from .neural import CELL_COUNT
from .production import V2ProductionConfig, V2ProductionMusicGPT2


FACTORIZED_PRODUCTION_SCHEMA = (
    "moonmuse-stage2-v2-factorized-sequence-production-v1"
)
FACTORIZED_SEPARATED_PRODUCTION_SCHEMA = (
    "moonmuse-stage2-v2-factorized-separated-production-v1"
)
ACTIVE_CONDITION_FIELDS = frozenset(
    {
        "identity_source_slot_ids",
        "operation_path_ids",
        "operation_lengths",
        "valence_ids",
        "arousal_ids",
        "motif_interval_ids",
        "motif_rhythm_ids",
        "motif_lengths",
    }
)
DEFAULT_MAX_MOTIF_EVENTS = 32


def tensorize_factorized_sequence_conditions(
    plans: Sequence[FactorizedControlPlan],
    motif_payloads: Sequence[Mapping[str, object]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Join factorized controls to observed motif sequences without axis leakage.

    ``motif_payloads`` use the frozen V103 realizer-condition representation only
    as an observed sequence source.  Its legacy role/source fields are ignored:
    identity pointers come exclusively from the factorized control plan.
    """

    if not plans or len(plans) != len(motif_payloads):
        raise ValueError("plans and motif payloads require one non-empty batch")

    controls = [encode_factorized_control(plan) for plan in plans]
    max_operation_width = max(
        len(row)
        for control in controls
        for row in control.operation_path_ids
    )
    max_motif_length = max(
        1,
        max(
            int(length)
            for payload in motif_payloads
            for length in payload["motif_lengths"]  # type: ignore[index]
        ),
    )
    if max_motif_length > DEFAULT_MAX_MOTIF_EVENTS:
        raise ValueError(
            f"motif sequence exceeds {DEFAULT_MAX_MOTIF_EVENTS} events"
        )

    batch_size = len(plans)
    identity_sources = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    operation_paths = torch.zeros(
        (batch_size, CELL_COUNT, max_operation_width), dtype=torch.long
    )
    operation_lengths = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    valence_ids = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    arousal_ids = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    motif_lengths = torch.empty((batch_size, CELL_COUNT), dtype=torch.long)
    motif_intervals = torch.zeros(
        (batch_size, CELL_COUNT, max_motif_length), dtype=torch.long
    )
    motif_rhythms = torch.zeros_like(motif_intervals)

    for batch_index, (control, payload) in enumerate(zip(controls, motif_payloads)):
        if payload.get("schema") != "moonmuse-stage2-v2-realizer-condition-v1":
            raise ValueError("unexpected motif-payload schema")
        motif_fields = (
            payload["motif_interval_ids"],
            payload["motif_rhythm_ids"],
            payload["motif_lengths"],
        )
        if not all(len(field) == CELL_COUNT for field in motif_fields):
            raise ValueError("motif payload must contain four cells")

        identity_sources[batch_index] = torch.tensor(
            control.identity_source_slot_ids, dtype=torch.long
        )
        operation_lengths[batch_index] = torch.tensor(
            control.operation_lengths, dtype=torch.long
        )
        valence_ids[batch_index] = torch.tensor(
            control.valence_ids, dtype=torch.long
        )
        arousal_ids[batch_index] = torch.tensor(
            control.arousal_ids, dtype=torch.long
        )
        for slot, row in enumerate(control.operation_path_ids):
            operation_paths[batch_index, slot, : len(row)] = torch.tensor(
                row, dtype=torch.long
            )

        for slot in range(CELL_COUNT):
            interval_values = tuple(
                int(value) for value in payload["motif_interval_ids"][slot]  # type: ignore[index]
            )
            rhythm_values = tuple(
                int(value) for value in payload["motif_rhythm_ids"][slot]  # type: ignore[index]
            )
            length = int(payload["motif_lengths"][slot])  # type: ignore[index]
            if length != len(interval_values) or length != len(rhythm_values):
                raise ValueError(f"motif feature length mismatch in cell {slot}")
            if any(
                value <= 0 or value >= len(INTERVAL_TOKENS)
                for value in interval_values
            ):
                raise ValueError("motif interval ID is outside the active vocabulary")
            if any(
                value <= 0 or value > MAX_RHYTHM_GAP
                for value in rhythm_values
            ):
                raise ValueError("motif rhythm gap is outside the active range")
            motif_lengths[batch_index, slot] = length
            if length:
                motif_intervals[batch_index, slot, :length] = torch.tensor(
                    interval_values, dtype=torch.long
                )
                motif_rhythms[batch_index, slot, :length] = torch.tensor(
                    rhythm_values, dtype=torch.long
                )

    tensors = {
        "identity_source_slot_ids": identity_sources.to(device),
        "operation_path_ids": operation_paths.to(device),
        "operation_lengths": operation_lengths.to(device),
        "valence_ids": valence_ids.to(device),
        "arousal_ids": arousal_ids.to(device),
        "motif_interval_ids": motif_intervals.to(device),
        "motif_rhythm_ids": motif_rhythms.to(device),
        "motif_lengths": motif_lengths.to(device),
    }
    if frozenset(tensors) != ACTIVE_CONDITION_FIELDS:
        raise AssertionError("active factorized condition fields drifted")
    return tensors


class FactorizedSequenceMemoryEncoder(nn.Module):
    """Encode four cells with ordered operations and causal motif attention."""

    uses_pitch_degree = False
    uses_rhythm_degree = False
    uses_comparison_source = False
    schema = FACTORIZED_PRODUCTION_SCHEMA

    def __init__(
        self,
        d_model: int,
        n_head: int,
        *,
        dropout: float = 0.0,
        max_motif_events: int = DEFAULT_MAX_MOTIF_EVENTS,
    ):
        super().__init__()
        if d_model < 4 or n_head < 1 or d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        if max_motif_events < 1:
            raise ValueError("max_motif_events must be positive")
        self.d_model = d_model
        self.max_motif_events = max_motif_events
        self.slot_embedding = nn.Embedding(CELL_COUNT, d_model)
        # Zero means no prior identity; source slots 0..3 map to IDs 1..4.
        self.identity_source_embedding = nn.Embedding(CELL_COUNT + 1, d_model)
        self.operation_embedding = nn.Embedding(
            len(OPERATION_TOKENS), d_model, padding_idx=0
        )
        self.operation_cell = nn.GRUCell(d_model, d_model)
        self.valence_embedding = nn.Embedding(len(VALENCE_TO_ID), d_model)
        self.arousal_embedding = nn.Embedding(len(AROUSAL_TO_ID), d_model)
        self.interval_embedding = nn.Embedding(
            len(INTERVAL_TOKENS), d_model, padding_idx=0
        )
        self.rhythm_embedding = nn.Embedding(
            MAX_RHYTHM_GAP + 1, d_model, padding_idx=0
        )
        self.motif_position_embedding = nn.Embedding(max_motif_events, d_model)
        self.memory_query = nn.Linear(d_model, d_model)
        self.motif_attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.motif_projection = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _require_shape(
        name: str, tensor: torch.Tensor, shape: tuple[int | None, ...]
    ) -> None:
        if tensor.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(tensor.shape, shape)
        ):
            rendered = ", ".join("*" if value is None else str(value) for value in shape)
            raise ValueError(f"{name} must have shape [{rendered}]")

    def _validate_condition(self, condition: Mapping[str, torch.Tensor]) -> None:
        if frozenset(condition) != ACTIVE_CONDITION_FIELDS:
            missing = sorted(ACTIVE_CONDITION_FIELDS - frozenset(condition))
            unexpected = sorted(frozenset(condition) - ACTIVE_CONDITION_FIELDS)
            raise ValueError(
                f"factorized condition field mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        sources = condition["identity_source_slot_ids"]
        batch_size = sources.size(0) if sources.ndim else None
        self._require_shape("identity_source_slot_ids", sources, (batch_size, CELL_COUNT))
        for name in ("operation_lengths", "valence_ids", "arousal_ids", "motif_lengths"):
            self._require_shape(name, condition[name], (batch_size, CELL_COUNT))
        operations = condition["operation_path_ids"]
        self._require_shape(
            "operation_path_ids", operations, (batch_size, CELL_COUNT, None)
        )
        intervals = condition["motif_interval_ids"]
        rhythms = condition["motif_rhythm_ids"]
        self._require_shape(
            "motif_interval_ids", intervals, (batch_size, CELL_COUNT, None)
        )
        if intervals.shape != rhythms.shape:
            raise ValueError("motif interval/rhythm tensors must share one shape")
        if intervals.size(2) > self.max_motif_events:
            raise ValueError("motif sequence exceeds encoder position capacity")

        target_slots = torch.arange(CELL_COUNT, device=sources.device).unsqueeze(0)
        if torch.any((sources < -1) | (sources >= target_slots)):
            raise ValueError("identity source must be absent or strictly earlier")
        operation_lengths = condition["operation_lengths"]
        operation_steps = torch.arange(operations.size(2), device=operations.device)
        operation_mask = operation_steps.view(1, 1, -1) < operation_lengths.unsqueeze(-1)
        if torch.any(operation_lengths < 1) or torch.any(
            operation_lengths > operations.size(2)
        ):
            raise ValueError("operation length is outside the padded path width")
        if torch.any(operations[operation_mask] <= 0) or torch.any(
            operations[operation_mask] >= len(OPERATION_TOKENS)
        ):
            raise ValueError("active operation ID is outside the vocabulary")
        if torch.any(operations[~operation_mask] != 0):
            raise ValueError("operation padding must be zero")

        motif_lengths = condition["motif_lengths"]
        if torch.any(motif_lengths < 0) or torch.any(
            motif_lengths > intervals.size(2)
        ):
            raise ValueError("motif length is outside the padded sequence width")
        motif_steps = torch.arange(intervals.size(2), device=intervals.device)
        motif_mask = motif_steps.view(1, 1, -1) < motif_lengths.unsqueeze(-1)
        if torch.any(intervals[motif_mask] <= 0) or torch.any(
            intervals[motif_mask] >= len(INTERVAL_TOKENS)
        ):
            raise ValueError("active motif interval ID is outside the vocabulary")
        if torch.any(rhythms[motif_mask] <= 0) or torch.any(
            rhythms[motif_mask] > MAX_RHYTHM_GAP
        ):
            raise ValueError("active motif rhythm gap is outside the range")
        if torch.any(intervals[~motif_mask] != 0) or torch.any(
            rhythms[~motif_mask] != 0
        ):
            raise ValueError("motif padding must be zero")
        if torch.any(condition["valence_ids"] < 0) or torch.any(
            condition["valence_ids"] >= len(VALENCE_TO_ID)
        ):
            raise ValueError("valence ID is outside the vocabulary")
        if torch.any(condition["arousal_ids"] < 0) or torch.any(
            condition["arousal_ids"] >= len(AROUSAL_TO_ID)
        ):
            raise ValueError("arousal ID is outside the vocabulary")

    def _encode_operation_paths(
        self, operation_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch_size, cell_count, width = operation_ids.shape
        embeddings = self.operation_embedding(operation_ids).reshape(
            batch_size * cell_count, width, self.d_model
        )
        flat_lengths = lengths.reshape(-1)
        hidden = embeddings.new_zeros((batch_size * cell_count, self.d_model))
        for step in range(width):
            candidate = self.operation_cell(embeddings[:, step], hidden)
            active = flat_lengths.gt(step).unsqueeze(-1)
            hidden = torch.where(active, candidate, hidden)
        return hidden.reshape(batch_size, cell_count, self.d_model)

    def _identity_query_hidden(
        self, slot_ids: torch.Tensor, sources: torch.Tensor
    ) -> torch.Tensor:
        return self.slot_embedding(slot_ids) + self.identity_source_embedding(
            sources + 1
        )

    def encode_components(
        self,
        condition: Mapping[str, torch.Tensor],
        *,
        identity_only_query: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return structural and motif-memory paths before their legacy fusion.

        The default query reproduces the original joint adapter exactly: operation
        and affect embeddings can steer the motif-attention query.  The separated
        adapter below opts into ``identity_only_query`` so operation/affect changes
        cannot silently alter the identity-memory representation.
        """

        self._validate_condition(condition)
        sources = condition["identity_source_slot_ids"]
        batch_size = sources.size(0)
        slot_ids = torch.arange(CELL_COUNT, device=sources.device).unsqueeze(0)
        slot_ids = slot_ids.expand(batch_size, -1)
        operation_hidden = self._encode_operation_paths(
            condition["operation_path_ids"], condition["operation_lengths"]
        )
        structural = (
            self.slot_embedding(slot_ids)
            + self.identity_source_embedding(sources + 1)
            + operation_hidden
            + self.valence_embedding(condition["valence_ids"])
            + self.arousal_embedding(condition["arousal_ids"])
        )

        intervals = condition["motif_interval_ids"]
        rhythms = condition["motif_rhythm_ids"]
        motif_lengths = condition["motif_lengths"]
        event_count = intervals.size(2)
        event_positions = torch.arange(event_count, device=intervals.device)
        motif_events = (
            self.interval_embedding(intervals)
            + self.rhythm_embedding(rhythms)
            + self.motif_position_embedding(event_positions).view(
                1, 1, event_count, self.d_model
            )
        )
        own_event_mask = event_positions.view(1, 1, -1) < motif_lengths.unsqueeze(-1)
        motif_events = motif_events * own_event_mask.unsqueeze(-1)

        safe_sources = sources.clamp(min=0)
        gather_index = safe_sources.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, event_count, self.d_model
        )
        source_events = torch.gather(motif_events, 1, gather_index)
        source_lengths = torch.gather(motif_lengths, 1, safe_sources)
        has_memory = sources.ge(0) & source_lengths.gt(0)
        source_events = source_events * has_memory.unsqueeze(-1).unsqueeze(-1)
        key_padding_mask = event_positions.view(1, 1, -1) >= source_lengths.unsqueeze(-1)
        # MultiheadAttention cannot consume an all-masked row.  No-source cells
        # receive one zero key and their final memory output is explicitly zeroed.
        key_padding_mask = torch.where(
            has_memory.unsqueeze(-1),
            key_padding_mask,
            torch.ones_like(key_padding_mask),
        )
        key_padding_mask[..., 0] = torch.where(
            has_memory,
            key_padding_mask[..., 0],
            torch.zeros_like(key_padding_mask[..., 0]),
        )

        if identity_only_query:
            query_hidden = self._identity_query_hidden(slot_ids, sources)
        else:
            query_hidden = structural
        flat_query = self.memory_query(query_hidden).reshape(
            batch_size * CELL_COUNT, 1, self.d_model
        )
        flat_events = source_events.reshape(
            batch_size * CELL_COUNT, event_count, self.d_model
        )
        flat_padding = key_padding_mask.reshape(batch_size * CELL_COUNT, event_count)
        attended, _ = self.motif_attention(
            flat_query,
            flat_events,
            flat_events,
            key_padding_mask=flat_padding,
            need_weights=False,
        )
        memory = self.motif_projection(attended.squeeze(1)).reshape(
            batch_size, CELL_COUNT, self.d_model
        )
        memory = memory * has_memory.unsqueeze(-1)
        return {
            "structural": structural,
            "memory": memory,
            "has_memory": has_memory,
        }

    def encode_slots(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        components = self.encode_components(condition)
        return self.output_norm(
            components["structural"] + components["memory"]
        )

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
        gather_index = token_slot_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        return torch.gather(slot_conditions, 1, gather_index)


class FactorizedV2ProductionMusicGPT2(V2ProductionMusicGPT2):
    """V92 production backbone with the factorized sequence-memory adapter."""

    condition_schema = FACTORIZED_PRODUCTION_SCHEMA

    def __init__(self, config: V2ProductionConfig):
        super().__init__(config)
        self.condition_encoder = FactorizedSequenceMemoryEncoder(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
        )


class SeparatedFactorizedSequenceMemoryEncoder(FactorizedSequenceMemoryEncoder):
    """Keep structural control and source-matched identity memory separable."""

    schema = FACTORIZED_SEPARATED_PRODUCTION_SCHEMA

    def __init__(
        self,
        d_model: int,
        n_head: int,
        *,
        dropout: float = 0.0,
        max_motif_events: int = DEFAULT_MAX_MOTIF_EVENTS,
    ):
        super().__init__(
            d_model,
            n_head,
            dropout=dropout,
            max_motif_events=max_motif_events,
        )
        # The legacy joint LayerNorm remains part of the historical v1 encoder but
        # is deliberately absent here: each causal path receives its own scale.
        del self.output_norm
        self.structural_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.memory_slot_query_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.memory_source_query_embedding = nn.Embedding(CELL_COUNT + 1, d_model)

    def _identity_query_hidden(
        self, slot_ids: torch.Tensor, sources: torch.Tensor
    ) -> torch.Tensor:
        return self.memory_slot_query_embedding(
            slot_ids
        ) + self.memory_source_query_embedding(sources + 1)

    def encode_components(
        self,
        condition: Mapping[str, torch.Tensor],
        *,
        identity_only_query: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not identity_only_query:
            raise ValueError("separated identity memory requires an identity-only query")
        raw = super().encode_components(condition, identity_only_query=True)
        has_memory = raw["has_memory"]
        memory = self.memory_norm(raw["memory"]) * has_memory.unsqueeze(-1)
        return {
            "structural": self.structural_norm(raw["structural"]),
            "memory": memory,
            "has_memory": has_memory,
        }

    def encode_slots(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        components = self.encode_components(condition)
        return components["structural"] + components["memory"]

    def forward_components(
        self,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_slot_ids.ndim != 2:
            raise ValueError("token_slot_ids must have shape [batch, sequence]")
        if torch.any((token_slot_ids < 0) | (token_slot_ids >= CELL_COUNT)):
            raise ValueError("token slot ID is outside 0..3")
        components = self.encode_components(condition)
        if components["structural"].size(0) != token_slot_ids.size(0):
            raise ValueError("condition batch does not match token batch")
        gather_index = token_slot_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        return (
            torch.gather(components["structural"], 1, gather_index),
            torch.gather(components["memory"], 1, gather_index),
        )

    def forward(
        self,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        structural, memory = self.forward_components(token_slot_ids, condition)
        return structural + memory


class SeparatedGateFactorizedV2ProductionMusicGPT2(V2ProductionMusicGPT2):
    """V92 backbone with independent structural and identity-memory gates.

    This is a new diagnostic architecture.  The historical
    :class:`FactorizedV2ProductionMusicGPT2` remains available with byte-compatible
    parameter names and its original single-gate fusion semantics.
    """

    condition_schema = FACTORIZED_SEPARATED_PRODUCTION_SCHEMA

    def __init__(self, config: V2ProductionConfig):
        super().__init__(config)
        del self.condition_encoder
        del self.condition_gate
        self.condition_encoder = SeparatedFactorizedSequenceMemoryEncoder(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
        )
        self.structural_gate = nn.Parameter(
            torch.tensor(float(config.condition_gate_init), dtype=torch.float32)
        )
        self.identity_memory_gate = nn.Parameter(
            torch.tensor(float(config.condition_gate_init), dtype=torch.float32)
        )

    @staticmethod
    def _is_v2_key(name: str) -> bool:
        return name in {"structural_gate", "identity_memory_gate"} or name.startswith(
            "condition_encoder."
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        token_slot_ids: Optional[torch.Tensor] = None,
        condition: Optional[Mapping[str, torch.Tensor]] = None,
        enable_conditioning: bool = True,
        seg_inp: Optional[torch.Tensor] = None,
        chord_inp: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        keep_last_only: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) > self.v2_config.max_input_len:
            raise ValueError("input must be [batch, sequence] within V102 maximum")
        x_emb = self.token_emb(x)
        if seg_inp is not None:
            if seg_inp.shape != x.shape:
                raise ValueError("segment IDs must match token shape")
            x_emb = x_emb + self.token_emb(seg_inp)
        if chord_inp is not None:
            x_emb = x_emb + self.emb_proj(chord_inp.float())
        x_emb = self.emb_proj(x_emb)

        if enable_conditioning:
            if token_slot_ids is None or condition is None:
                raise ValueError("enabled V2 conditioning requires slots and condition")
            if token_slot_ids.shape != x.shape:
                raise ValueError("token slot IDs must match token shape")
            structural_hidden, memory_hidden = (
                self.condition_encoder.forward_components(token_slot_ids, condition)
            )
            x_emb = (
                x_emb
                + self.structural_gate * structural_hidden
                + self.identity_memory_gate * memory_hidden
            )

        x_emb = self.ln_emb(x_emb)
        x_emb = self.emb_dropout(x_emb)
        transformer_out = self.transformer(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = self.head(transformer_out)
        return logits[:, -1, :] if keep_last_only else logits
