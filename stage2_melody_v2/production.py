"""V2-only conditioning adapter around the production Stage 2 GPTNeo backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from stage2_melody.model.music_gpt2 import ImprovedMusicGPT2

from .neural import CausalMotifConditionEncoder


PRODUCTION_SCHEMA = "moonmuse-stage2-v2-production-gptneo-v1"


@dataclass(frozen=True)
class V2ProductionConfig:
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
    condition_gate_init: float = 0.1
    schema: str = PRODUCTION_SCHEMA

    def validate(self) -> None:
        if self.schema != PRODUCTION_SCHEMA:
            raise ValueError(f"unsupported production schema: {self.schema}")
        if self.vocab_size < 2 or not 0 <= self.pad_token < self.vocab_size:
            raise ValueError("invalid vocabulary or PAD token")
        if self.n_layer < 1 or self.n_head < 1 or self.d_model % self.n_head:
            raise ValueError("invalid production Transformer dimensions")
        if self.d_embed != self.d_model:
            raise ValueError("V102 requires the V92 d_embed=d_model contract")
        if self.d_ff < self.d_model:
            raise ValueError("feed-forward width is too small")
        if self.max_input_len > self.backbone_max_positions:
            raise ValueError("input length exceeds backbone position capacity")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.condition_gate_init <= 1.0:
            raise ValueError("condition gate initialization must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class V2ProductionMusicGPT2(ImprovedMusicGPT2):
    """Production GPTNeo weights plus a causally safe V2 condition encoder."""

    def __init__(self, config: V2ProductionConfig):
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
        self.v2_config = config
        self.condition_encoder = CausalMotifConditionEncoder(config.d_model)
        self.condition_gate = nn.Parameter(
            torch.tensor(float(config.condition_gate_init), dtype=torch.float32)
        )

    @staticmethod
    def _is_v2_key(name: str) -> bool:
        return name == "condition_gate" or name.startswith("condition_encoder.")

    def load_v92_backbone_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, int]:
        """Strictly load every inherited production key while retaining new V2 keys."""

        current = self.state_dict()
        expected = {name for name in current if not self._is_v2_key(name)}
        supplied = set(state_dict)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing or unexpected:
            raise ValueError(
                f"V92 backbone key mismatch; missing={missing}, unexpected={unexpected}"
            )
        merged = dict(current)
        merged.update(state_dict)
        incompatible = self.load_state_dict(merged, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict merged load failed: {incompatible}")
        return {
            "loaded_backbone_key_count": len(expected),
            "new_v2_key_count": len(current) - len(expected),
        }

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
            condition_hidden = self.condition_encoder(token_slot_ids, condition)
            x_emb = x_emb + self.condition_gate * condition_hidden

        x_emb = self.ln_emb(x_emb)
        x_emb = self.emb_dropout(x_emb)
        transformer_out = self.transformer(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = self.head(transformer_out)
        return logits[:, -1, :] if keep_last_only else logits

    def one_step_logits(
        self,
        tokens: torch.Tensor,
        token_slot_ids: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
        *,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            tokens,
            token_slot_ids=token_slot_ids,
            condition=condition,
            seg_inp=segment_ids,
            attention_mask=attention_mask,
            keep_last_only=True,
        )
