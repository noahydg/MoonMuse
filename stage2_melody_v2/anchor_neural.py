"""Small chord-aware causal anchor-planner fixture for the V127 smoke.

Stage 1 has already produced the complete chord progression when Stage 2 runs.
The planner therefore encodes all 32 chord slots without a causal mask, then
causally decodes the anchor sequence while cross-attending to that complete
harmonic memory.  Future *anchor* inputs remain hidden; future chords do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .anchor_conditioning import (
    ANCHOR_SLOT_COUNT,
    DEGREE_LABELS,
    EMOTION_LABELS,
    MISSING_CLASS_ID,
    PREVIOUS_ACTIVE,
    PREVIOUS_REST,
    QUALITY_LABELS,
    AnchorPlannerContext,
    AnchorPlannerExample,
    compile_realizer_anchor_condition,
    planner_context_from_example,
)


@dataclass(frozen=True)
class AnchorPlannerConfig:
    d_model: int = 48
    n_head: int = 4
    n_layer: int = 2
    d_ff: int = 96
    dropout: float = 0.0
    octave_classes: int = 9

    def validate(self) -> None:
        if self.d_model < 8 or self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_layer < 1 or self.d_ff < self.d_model:
            raise ValueError("invalid Transformer dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.octave_classes < 9:
            raise ValueError("octave head must cover observed classes 0..8")


def _shift_missing(values: torch.Tensor) -> torch.Tensor:
    return values + 1


def tensorize_anchor_examples(
    examples: Sequence[AnchorPlannerExample], *, device: torch.device
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("at least one anchor example is required")
    for example in examples:
        example.validate()

    def tensor(field: str) -> torch.Tensor:
        return torch.tensor(
            [getattr(example, field) for example in examples],
            dtype=torch.long,
            device=device,
        )

    return {
        "key_mode": torch.tensor(
            [example.key_mode_id for example in examples],
            dtype=torch.long,
            device=device,
        ),
        "emotion": torch.tensor(
            [example.emotion_id for example in examples],
            dtype=torch.long,
            device=device,
        ),
        "slot": tensor("slot_ids"),
        "bar": tensor("bar_ids"),
        "beat_index": tensor("beat_ids") // 4,
        "chord_root": tensor("chord_root_ids"),
        "chord_quality": tensor("chord_quality_ids"),
        "previous_active": tensor("previous_active_ids"),
        "previous_degree": _shift_missing(tensor("previous_degree_ids")),
        "previous_octave": _shift_missing(tensor("previous_octave_ids")),
        "target_active": tensor("target_active_ids"),
        "target_degree": tensor("target_degree_ids"),
        "target_octave": tensor("target_octave_ids"),
    }


def tensorize_anchor_contexts(
    contexts: Sequence[AnchorPlannerContext], *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Tensorize label-free inference contexts with BOS/rest decode inputs."""

    if not contexts:
        raise ValueError("at least one anchor planner context is required")
    for context in contexts:
        context.validate()

    def tensor(field: str) -> torch.Tensor:
        return torch.tensor(
            [getattr(context, field) for context in contexts],
            dtype=torch.long,
            device=device,
        )

    batch_size = len(contexts)
    previous_active = torch.full(
        (batch_size, ANCHOR_SLOT_COUNT),
        PREVIOUS_REST,
        dtype=torch.long,
        device=device,
    )
    previous_active[:, 0] = 2
    return {
        "key_mode": torch.tensor(
            [context.key_mode_id for context in contexts],
            dtype=torch.long,
            device=device,
        ),
        "emotion": torch.tensor(
            [context.emotion_id for context in contexts],
            dtype=torch.long,
            device=device,
        ),
        "slot": tensor("slot_ids"),
        "bar": tensor("bar_ids"),
        "beat_index": tensor("beat_ids") // 4,
        "chord_root": tensor("chord_root_ids"),
        "chord_quality": tensor("chord_quality_ids"),
        "previous_active": previous_active,
        "previous_degree": torch.zeros(
            (batch_size, ANCHOR_SLOT_COUNT), dtype=torch.long, device=device
        ),
        "previous_octave": torch.zeros(
            (batch_size, ANCHOR_SLOT_COUNT), dtype=torch.long, device=device
        ),
    }


class CausalAnchorPlanner(nn.Module):
    def __init__(self, config: AnchorPlannerConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.slot_embedding = nn.Embedding(ANCHOR_SLOT_COUNT, d)
        self.bar_embedding = nn.Embedding(8, d)
        self.beat_embedding = nn.Embedding(4, d)
        self.key_mode_embedding = nn.Embedding(2, d)
        self.emotion_embedding = nn.Embedding(len(EMOTION_LABELS) + 1, d)
        self.chord_root_embedding = nn.Embedding(len(DEGREE_LABELS) + 1, d)
        self.chord_quality_embedding = nn.Embedding(len(QUALITY_LABELS) + 1, d)
        self.previous_active_embedding = nn.Embedding(3, d)
        self.previous_degree_embedding = nn.Embedding(len(DEGREE_LABELS) + 1, d)
        self.previous_octave_embedding = nn.Embedding(config.octave_classes + 1, d)
        self.chord_input_norm = nn.LayerNorm(d)
        chord_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chord_encoder = nn.TransformerEncoder(chord_layer, num_layers=1)
        self.anchor_input_norm = nn.LayerNorm(d)
        anchor_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anchor_decoder = nn.TransformerDecoder(
            anchor_layer, num_layers=config.n_layer
        )
        self.output_norm = nn.LayerNorm(d)
        self.active_head = nn.Linear(d, 2)
        self.degree_head = nn.Linear(d, len(DEGREE_LABELS))
        self.octave_head = nn.Linear(d, config.octave_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shape = batch["slot"].shape
        if shape[1] != ANCHOR_SLOT_COUNT:
            raise ValueError("anchor batch must have 32 slots")
        for field in (
            "bar",
            "beat_index",
            "chord_root",
            "chord_quality",
            "previous_active",
            "previous_degree",
            "previous_octave",
        ):
            if batch[field].shape != shape:
                raise ValueError(f"{field} shape does not match slot shape")
        shared_position = (
            self.slot_embedding(batch["slot"])
            + self.bar_embedding(batch["bar"])
            + self.beat_embedding(batch["beat_index"])
            + self.key_mode_embedding(batch["key_mode"]).unsqueeze(1)
            + self.emotion_embedding(batch["emotion"]).unsqueeze(1)
        )
        chord_memory = (
            shared_position
            + self.chord_root_embedding(batch["chord_root"])
            + self.chord_quality_embedding(batch["chord_quality"])
        )
        chord_memory = self.chord_encoder(self.chord_input_norm(chord_memory))
        anchor_hidden = (
            shared_position
            + self.previous_active_embedding(batch["previous_active"])
            + self.previous_degree_embedding(batch["previous_degree"])
            + self.previous_octave_embedding(batch["previous_octave"])
        )
        anchor_hidden = self.anchor_input_norm(anchor_hidden)
        causal_mask = torch.triu(
            torch.ones(
                ANCHOR_SLOT_COUNT,
                ANCHOR_SLOT_COUNT,
                device=anchor_hidden.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        hidden = self.anchor_decoder(
            tgt=anchor_hidden,
            memory=chord_memory,
            tgt_mask=causal_mask,
        )
        hidden = self.output_norm(hidden)
        return {
            "active": self.active_head(hidden),
            "degree": self.degree_head(hidden),
            "octave": self.octave_head(hidden),
        }

    def compute_loss(
        self, logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        active_loss = F.cross_entropy(
            logits["active"].reshape(-1, 2),
            batch["target_active"].reshape(-1),
        )
        active_mask = batch["target_active"].bool()
        if not torch.any(active_mask):
            raise ValueError("anchor batch contains no active pitch targets")
        degree_loss = F.cross_entropy(
            logits["degree"][active_mask],
            batch["target_degree"][active_mask],
        )
        octave_loss = F.cross_entropy(
            logits["octave"][active_mask],
            batch["target_octave"][active_mask],
        )
        total = active_loss + degree_loss + octave_loss
        return {
            "active_loss": active_loss,
            "degree_loss": degree_loss,
            "octave_loss": octave_loss,
            "total_loss": total,
        }


def predict_anchor_plans_autoregressive(
    model: CausalAnchorPlanner,
    examples: Sequence[AnchorPlannerExample | AnchorPlannerContext],
    *,
    device: torch.device,
) -> list[dict]:
    """Greedily predict a batch using only each sequence's own prior outputs.

    ``tensorize_anchor_examples`` includes shifted targets for teacher-forced
    training. This function overwrites every post-BOS previous-anchor input
    before decoding, so validation and generation cannot inherit a true prior
    melody anchor from ``example``.
    """

    if not examples:
        raise ValueError("at least one example is required for anchor decoding")
    contexts = []
    for example in examples:
        example.validate()
        contexts.append(
            planner_context_from_example(example)
            if isinstance(example, AnchorPlannerExample)
            else example
        )
    batch = tensorize_anchor_contexts(contexts, device=device)
    batch_size = len(contexts)
    active_ids = [[] for _ in contexts]
    degree_ids = [[] for _ in contexts]
    octave_ids = [[] for _ in contexts]
    log_probabilities = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        for slot in range(ANCHOR_SLOT_COUNT):
            logits = model(batch)
            active = torch.argmax(logits["active"][:, slot], dim=-1)
            log_probabilities += F.log_softmax(
                logits["active"][:, slot], dim=-1
            ).gather(1, active.unsqueeze(1)).squeeze(1)
            degree = torch.argmax(logits["degree"][:, slot], dim=-1)
            octave = torch.argmax(logits["octave"][:, slot], dim=-1)
            active_mask = active.bool()
            if torch.any(active_mask):
                log_probabilities[active_mask] += F.log_softmax(
                    logits["degree"][active_mask, slot], dim=-1
                ).gather(1, degree[active_mask].unsqueeze(1)).squeeze(1)
                log_probabilities[active_mask] += F.log_softmax(
                    logits["octave"][active_mask, slot], dim=-1
                ).gather(1, octave[active_mask].unsqueeze(1)).squeeze(1)
            degree = torch.where(
                active_mask,
                degree,
                torch.full_like(degree, MISSING_CLASS_ID),
            )
            octave = torch.where(
                active_mask,
                octave,
                torch.full_like(octave, MISSING_CLASS_ID),
            )
            # One host transfer per field and slot avoids a CUDA synchronization
            # for every example while preserving exactly the same greedy decode.
            active_host = active.detach().cpu().tolist()
            degree_host = degree.detach().cpu().tolist()
            octave_host = octave.detach().cpu().tolist()
            for row in range(batch_size):
                active_ids[row].append(int(active_host[row]))
                degree_ids[row].append(int(degree_host[row]))
                octave_ids[row].append(int(octave_host[row]))
            if slot + 1 < ANCHOR_SLOT_COUNT:
                batch["previous_active"][:, slot + 1] = torch.where(
                    active_mask,
                    torch.full_like(active, PREVIOUS_ACTIVE),
                    torch.full_like(active, PREVIOUS_REST),
                )
                batch["previous_degree"][:, slot + 1] = degree + 1
                batch["previous_octave"][:, slot + 1] = octave + 1

    log_probability_host = log_probabilities.detach().cpu().tolist()
    output = []
    for row, context in enumerate(contexts):
        condition = compile_realizer_anchor_condition(
            context,
            predicted_active_ids=active_ids[row],
            predicted_degree_ids=degree_ids[row],
            predicted_octave_ids=octave_ids[row],
        )
        output.append(
            {
                "active_ids": tuple(active_ids[row]),
                "degree_ids": tuple(degree_ids[row]),
                "octave_ids": tuple(octave_ids[row]),
                "sequence_log_probability": float(log_probability_host[row]),
                "realizer_condition": condition,
                "used_ground_truth_previous_anchors": False,
                "decode": "greedy_autoregressive_32_slots",
            }
        )
    return output


def sample_anchor_plans_autoregressive(
    model: CausalAnchorPlanner,
    examples: Sequence[AnchorPlannerExample | AnchorPlannerContext],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> list[dict]:
    """Sample label-free previous anchors for on-policy planner fine-tuning.

    This follows the production autoregressive boundary but draws from each
    categorical head instead of taking argmax.  It is intended for scheduled
    sampling only; the production prediction API remains deterministic greedy.
    """

    if not examples:
        raise ValueError("at least one example is required for anchor sampling")
    contexts = []
    for example in examples:
        example.validate()
        contexts.append(
            planner_context_from_example(example)
            if isinstance(example, AnchorPlannerExample)
            else example
        )
    batch = tensorize_anchor_contexts(contexts, device=device)
    batch_size = len(contexts)
    active_ids = [[] for _ in contexts]
    degree_ids = [[] for _ in contexts]
    octave_ids = [[] for _ in contexts]

    with torch.no_grad():
        for slot in range(ANCHOR_SLOT_COUNT):
            logits = model(batch)
            active = torch.multinomial(
                torch.softmax(logits["active"][:, slot], dim=-1),
                num_samples=1,
                generator=generator,
            ).squeeze(1)
            degree = torch.multinomial(
                torch.softmax(logits["degree"][:, slot], dim=-1),
                num_samples=1,
                generator=generator,
            ).squeeze(1)
            octave = torch.multinomial(
                torch.softmax(logits["octave"][:, slot], dim=-1),
                num_samples=1,
                generator=generator,
            ).squeeze(1)
            active_mask = active.bool()
            degree = torch.where(
                active_mask,
                degree,
                torch.full_like(degree, MISSING_CLASS_ID),
            )
            octave = torch.where(
                active_mask,
                octave,
                torch.full_like(octave, MISSING_CLASS_ID),
            )
            active_host = active.detach().cpu().tolist()
            degree_host = degree.detach().cpu().tolist()
            octave_host = octave.detach().cpu().tolist()
            for row in range(batch_size):
                active_ids[row].append(int(active_host[row]))
                degree_ids[row].append(int(degree_host[row]))
                octave_ids[row].append(int(octave_host[row]))
            if slot + 1 < ANCHOR_SLOT_COUNT:
                batch["previous_active"][:, slot + 1] = torch.where(
                    active_mask,
                    torch.full_like(active, PREVIOUS_ACTIVE),
                    torch.full_like(active, PREVIOUS_REST),
                )
                batch["previous_degree"][:, slot + 1] = degree + 1
                batch["previous_octave"][:, slot + 1] = octave + 1

    return [
        {
            "active_ids": tuple(active_ids[row]),
            "degree_ids": tuple(degree_ids[row]),
            "octave_ids": tuple(octave_ids[row]),
            "used_ground_truth_previous_anchors": False,
            "decode": "sampled_autoregressive_32_slots",
        }
        for row in range(batch_size)
    ]


def predict_anchor_plan_autoregressive(
    model: CausalAnchorPlanner,
    example: AnchorPlannerExample | AnchorPlannerContext,
    *,
    device: torch.device,
) -> dict:
    """Single-example convenience wrapper around batched autoregression."""

    return predict_anchor_plans_autoregressive(
        model, [example], device=device
    )[0]
