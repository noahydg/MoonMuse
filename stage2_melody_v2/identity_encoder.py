"""Small V107 motif identity encoder and frozen contrastive objective."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .identity import transition_features_from_view


OPERATION_TO_ID = {
    "transpose": 0,
    "invert": 1,
    "time_scale": 2,
    "rhythm_shift": 3,
    "fragment": 4,
}
ID_TO_OPERATION = {value: key for key, value in OPERATION_TO_ID.items()}
MAX_TRANSITIONS = 31


class MotifIdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transition_projection = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.position_embedding = nn.Embedding(MAX_TRANSITIONS, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.identity_projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        self.operation_head = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.GELU(),
            nn.Linear(64, len(OPERATION_TO_ID)),
        )

    def encode(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError("motif features must have shape [batch, steps, 4]")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("motif valid mask shape mismatch")
        if features.shape[1] > MAX_TRANSITIONS:
            raise ValueError("motif exceeds V107 maximum transitions")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("every motif must have at least one valid transition")
        positions = torch.arange(features.shape[1], device=features.device)
        hidden = self.transition_projection(features)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.transformer(hidden, src_key_padding_mask=~valid_mask)
        weights = valid_mask.to(hidden.dtype).unsqueeze(-1)
        raw = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        identity = F.normalize(self.identity_projection(raw), dim=-1)
        return raw, identity

    def operation_logits(
        self, source_raw: torch.Tensor, transformed_raw: torch.Tensor
    ) -> torch.Tensor:
        if source_raw.shape != transformed_raw.shape:
            raise ValueError("source/transformed raw state shapes differ")
        pair = torch.cat(
            [source_raw, transformed_raw, (source_raw - transformed_raw).abs()],
            dim=-1,
        )
        return self.operation_head(pair)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        transformed_features: torch.Tensor,
        transformed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source_raw, source_identity = self.encode(source_features, source_mask)
        transformed_raw, transformed_identity = self.encode(
            transformed_features, transformed_mask
        )
        return {
            "source_raw": source_raw,
            "source_identity": source_identity,
            "transformed_raw": transformed_raw,
            "transformed_identity": transformed_identity,
            "operation_logits": self.operation_logits(source_raw, transformed_raw),
        }


def views_to_tensors(
    views: Sequence[dict[str, Any]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [transition_features_from_view(view) for view in views]
    maximum = max(len(sequence) for sequence in sequences)
    if maximum > MAX_TRANSITIONS:
        raise ValueError("V107 tensor batch exceeds maximum transitions")
    features = torch.zeros((len(sequences), maximum, 4), dtype=torch.float32)
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        features[index, :length] = torch.tensor(sequence, dtype=torch.float32)
        mask[index, :length] = True
    return features.to(device), mask.to(device)


def symmetric_nt_xent_loss(
    source_identity: torch.Tensor,
    transformed_identity: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    if source_identity.shape != transformed_identity.shape:
        raise ValueError("NT-Xent identity tensors differ in shape")
    if source_identity.ndim != 2 or source_identity.shape[0] < 2:
        raise ValueError("NT-Xent requires a batch with at least two identities")
    embeddings = torch.cat([source_identity, transformed_identity], dim=0)
    logits = embeddings @ embeddings.transpose(0, 1) / temperature
    diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    batch = source_identity.shape[0]
    targets = torch.cat(
        [
            torch.arange(batch, 2 * batch, device=logits.device),
            torch.arange(0, batch, device=logits.device),
        ]
    )
    return F.cross_entropy(logits, targets)
