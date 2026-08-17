"""Predicted-anchor conditioner for the original dense V1 Stage 2 GPTNeo.

This is not a replacement monophonic generator. It loads the actual dense V1
backbone byte-for-byte, encodes the complete 32-slot predicted plan, and adds a
zero-initialized residual only to Melody input positions. The full plan is
encoded non-causally because Stage 1 and the anchor planner both finish before
dense Stage 2 realization begins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from stage2_melody.model.music_gpt2 import ImprovedMusicGPT2

from .anchor_conditioning import ANCHOR_SLOT_COUNT, DEGREE_LABELS


ANCHOR_DENSE_SCHEMA = "moonmuse-stage2-v2-anchor-conditioned-dense-v128-v1"


@dataclass(frozen=True)
class AnchorDenseRealizerConfig:
    vocab_size: int
    pad_token: int
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 512
    d_embed: int = 256
    max_input_len: int = 768
    backbone_max_positions: int = 4096
    attention_window: int = 512
    dropout: float = 0.1
    anchor_encoder_layers: int = 2
    schema: str = ANCHOR_DENSE_SCHEMA

    def validate(self) -> None:
        if self.schema != ANCHOR_DENSE_SCHEMA:
            raise ValueError("unsupported anchor-dense schema")
        if self.vocab_size < 2 or not 0 <= self.pad_token < self.vocab_size:
            raise ValueError("invalid dense vocabulary or PAD token")
        if self.n_layer < 1 or self.n_head < 1 or self.d_model % self.n_head:
            raise ValueError("invalid dense backbone dimensions")
        if self.d_embed != self.d_model:
            raise ValueError("V1 dense backbone requires d_embed=d_model")
        if self.d_ff < self.d_model or self.anchor_encoder_layers < 1:
            raise ValueError("invalid anchor conditioner dimensions")
        if self.max_input_len > self.backbone_max_positions:
            raise ValueError("dense input exceeds backbone position capacity")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class PredictedAnchorPlanEncoder(nn.Module):
    """Encode all predicted slots before routing contextualized slots to tokens."""

    def __init__(self, config: AnchorDenseRealizerConfig):
        super().__init__()
        d = config.d_model
        self.slot_embedding = nn.Embedding(ANCHOR_SLOT_COUNT, d)
        self.bar_embedding = nn.Embedding(8, d)
        self.beat_embedding = nn.Embedding(4, d)
        self.active_embedding = nn.Embedding(2, d)
        self.degree_embedding = nn.Embedding(len(DEGREE_LABELS) + 1, d)
        self.octave_embedding = nn.Embedding(10, d)
        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.anchor_encoder_layers
        )
        self.output_norm = nn.LayerNorm(d)
        self.output_projection = nn.Linear(d, d)
        # Exact V1 behavior at initialization. Unlike a single scalar gate, the
        # conditioner must learn a structured residual rather than merely turn
        # one global gate up or down.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _encode(
        self,
        *,
        active: torch.Tensor,
        degree: torch.Tensor,
        octave: torch.Tensor,
        include_content: bool,
    ) -> torch.Tensor:
        batch_size = active.size(0)
        device = active.device
        slot = torch.arange(ANCHOR_SLOT_COUNT, device=device).unsqueeze(0)
        slot = slot.expand(batch_size, -1)
        bar = slot // 4
        beat = slot % 4
        hidden = (
            self.slot_embedding(slot)
            + self.bar_embedding(bar)
            + self.beat_embedding(beat)
        )
        if include_content:
            hidden = (
                hidden
                + self.active_embedding(active)
                + self.degree_embedding(degree + 1)
                + self.octave_embedding(octave + 1)
            )
        hidden = self.encoder(self.input_norm(hidden))
        return self.output_projection(self.output_norm(hidden))

    def forward(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = ("active", "degree", "octave")
        if any(name not in condition for name in required):
            raise ValueError("predicted anchor tensors are incomplete")
        shape = condition["active"].shape
        if len(shape) != 2 or shape[1] != ANCHOR_SLOT_COUNT:
            raise ValueError("predicted anchor tensors must be [batch, 32]")
        if any(condition[name].shape != shape for name in required):
            raise ValueError("predicted anchor tensor shapes differ")
        return self._encode(
            active=condition["active"],
            degree=condition["degree"],
            octave=condition["octave"],
            include_content=True,
        )

    def content_contrast(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return plan-content evidence after removing a position-only shortcut.

        The null branch traverses the same positional embeddings, transformer,
        normalization and output projection.  Subtraction therefore guarantees
        exact zero for every position when the content embeddings contribute no
        information; a dense adapter cannot lower loss using slot/bar/beat alone.
        """

        required = ("active", "degree", "octave")
        if any(name not in condition for name in required):
            raise ValueError("predicted anchor tensors are incomplete")
        full = self._encode(
            active=condition["active"],
            degree=condition["degree"],
            octave=condition["octave"],
            include_content=True,
        )
        position_only = self._encode(
            active=condition["active"],
            degree=condition["degree"],
            octave=condition["octave"],
            include_content=False,
        )
        return full - position_only


class AnchorConditionedDenseMusicGPT2(ImprovedMusicGPT2):
    """The original dense V1 Stage 2 plus a predicted-anchor residual path."""

    def __init__(self, config: AnchorDenseRealizerConfig):
        config.validate()
        super().__init__(
            config.vocab_size,
            config.n_layer,
            config.n_head,
            config.d_model,
            config.d_ff,
            config.d_embed,
            dropout=config.dropout,
            max_seq_len=config.backbone_max_positions,
            attention_window=config.attention_window,
            use_rope=True,
            use_local_attn=True,
        )
        self.anchor_dense_config = config
        self.anchor_condition_encoder = PredictedAnchorPlanEncoder(config)

    @staticmethod
    def _is_anchor_key(name: str) -> bool:
        return name.startswith("anchor_condition_encoder.")

    def load_v1_backbone_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, int]:
        """Strictly load the actual dense V1 checkpoint, retaining new keys."""

        current = self.state_dict()
        expected = {name for name in current if not self._is_anchor_key(name)}
        supplied = set(state_dict)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        mismatched = sorted(
            name
            for name in expected & supplied
            if current[name].shape != state_dict[name].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "V1 dense backbone mismatch; "
                f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
            )
        merged = dict(current)
        merged.update(state_dict)
        incompatible = self.load_state_dict(merged, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict V1 merged load failed: {incompatible}")
        return {
            "loaded_v1_backbone_key_count": len(expected),
            "new_anchor_key_count": len(current) - len(expected),
        }

    def anchor_token_residual(
        self,
        *,
        anchor_condition: Mapping[str, torch.Tensor],
        token_anchor_slot_ids: torch.Tensor,
        melody_condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the directly injected residual for audit and forward use."""

        if token_anchor_slot_ids.shape != melody_condition_mask.shape:
            raise ValueError("dense anchor slots and Melody mask must have one shape")
        if torch.any(token_anchor_slot_ids < 0) or torch.any(
            token_anchor_slot_ids >= ANCHOR_SLOT_COUNT
        ):
            raise ValueError("dense anchor slot IDs must be in 0..31")
        anchor_slots = self.anchor_condition_encoder.content_contrast(
            anchor_condition
        )
        if token_anchor_slot_ids.size(0) != anchor_slots.size(0):
            raise ValueError("anchor plan batch differs from dense token batch")
        gather_index = token_anchor_slot_ids.unsqueeze(-1).expand(
            -1, -1, anchor_slots.size(-1)
        )
        token_condition = torch.gather(anchor_slots, 1, gather_index)
        return token_condition * melody_condition_mask.unsqueeze(-1).to(
            token_condition.dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        anchor_condition: Optional[Mapping[str, torch.Tensor]] = None,
        token_anchor_slot_ids: Optional[torch.Tensor] = None,
        melody_condition_mask: Optional[torch.Tensor] = None,
        enable_anchor_conditioning: bool = True,
        seg_inp: Optional[torch.Tensor] = None,
        chord_inp: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        keep_last_only: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) > self.anchor_dense_config.max_input_len:
            raise ValueError("dense input must be [batch, sequence] within V128 maximum")
        x_emb = self.token_emb(x)
        if seg_inp is not None:
            if seg_inp.shape != x.shape:
                raise ValueError("segment IDs must match dense token shape")
            x_emb = x_emb + self.token_emb(seg_inp)
        if chord_inp is not None:
            x_emb = x_emb + self.emb_proj(chord_inp.float())
        x_emb = self.emb_proj(x_emb)

        if enable_anchor_conditioning:
            if (
                anchor_condition is None
                or token_anchor_slot_ids is None
                or melody_condition_mask is None
            ):
                raise ValueError("enabled anchor conditioning requires plan, slots and mask")
            if token_anchor_slot_ids.shape != x.shape:
                raise ValueError("dense anchor slot IDs must match token shape")
            if melody_condition_mask.shape != x.shape:
                raise ValueError("dense melody condition mask must match token shape")
            x_emb = x_emb + self.anchor_token_residual(
                anchor_condition=anchor_condition,
                token_anchor_slot_ids=token_anchor_slot_ids,
                melody_condition_mask=melody_condition_mask,
            )

        x_emb = self.ln_emb(x_emb)
        x_emb = self.emb_dropout(x_emb)
        transformer_out = self.transformer(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = self.head(transformer_out)
        return logits[:, -1, :] if keep_last_only else logits

    def forward_with_prediction_arrays(
        self,
        x: torch.Tensor,
        *,
        predicted_active_ids: torch.Tensor,
        predicted_degree_ids: torch.Tensor,
        predicted_octave_ids: torch.Tensor,
        token_anchor_slot_ids: torch.Tensor,
        melody_condition_mask: torch.Tensor,
        seg_inp: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        keep_last_only: bool = False,
    ) -> torch.Tensor:
        """Inference convenience API that accepts only planner predictions.

        Generation does not have a training example or target anchor object.
        Keeping this wrapper numeric makes that boundary explicit while still
        reusing the exact audited forward path.
        """

        return self(
            x,
            anchor_condition={
                "active": predicted_active_ids,
                "degree": predicted_degree_ids,
                "octave": predicted_octave_ids,
            },
            token_anchor_slot_ids=token_anchor_slot_ids,
            melody_condition_mask=melody_condition_mask,
            seg_inp=seg_inp,
            attention_mask=attention_mask,
            keep_last_only=keep_last_only,
        )
