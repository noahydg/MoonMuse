"""Production inference helpers for the V129 hierarchical Stage 2 route."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from generation_budget import (
    RejectionBudget,
    SegmentSamplingBudget,
    monotonic_beat_inadmissibles,
    strip_terminal_eos,
)

from .anchor_conditioning import compile_anchor_planner_context
from .anchor_dense_conditioning import PredictedAnchorArrays
from .anchor_neural import CausalAnchorPlanner, predict_anchor_plan_autoregressive
from .anchor_realizer import AnchorConditionedDenseMusicGPT2


TARGET_BARS = 8
MAX_DENSE_TOKENS = 768
MAX_OUTPUT_TOKENS = 1280


@dataclass(frozen=True)
class AnchorGenerationResult:
    tokens: tuple[int, ...]
    generated_bars: int
    total_attempts: int
    rejection_count: int
    elapsed_seconds: float
    anchor_plan: PredictedAnchorArrays
    anchor_plan_sha256: str


def _sha256_plan(plan: PredictedAnchorArrays) -> str:
    payload = "|".join(
        ",".join(str(value) for value in field)
        for field in (plan.active_ids, plan.degree_ids, plan.octave_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_terminal_dense_boundary(
    generated: Sequence[int], idx2event: Mapping[int, str]
) -> list[int]:
    """Remove only the decoder's boundary token after the eighth Melody bar."""

    cleaned = list(strip_terminal_eos(list(generated), idx2event))
    if cleaned and str(idx2event[int(cleaned[-1])]) == "Track_Chord":
        cleaned.pop()
    return cleaned


def predict_plan_from_stage1_tokens(
    planner: CausalAnchorPlanner,
    *,
    stage1_tokens: Sequence[str],
    device: torch.device,
) -> PredictedAnchorArrays:
    context = compile_anchor_planner_context(stage1_tokens)
    prediction = predict_anchor_plan_autoregressive(
        planner, context, device=device
    )
    plan = PredictedAnchorArrays(
        active_ids=tuple(int(value) for value in prediction["active_ids"]),
        degree_ids=tuple(int(value) for value in prediction["degree_ids"]),
        octave_ids=tuple(int(value) for value in prediction["octave_ids"]),
    )
    plan.validate()
    return plan


def token_routes_for_prefix(
    tokens: Sequence[int],
    *,
    idx2event: Mapping[int, str],
) -> tuple[list[int], list[int], list[int]]:
    """Reconstruct V1 segment IDs and V129 slot routing for a partial decode."""

    segment_ids: list[int] = []
    slot_ids: list[int] = []
    melody_mask: list[int] = []
    current_track: str | None = None
    melody_bar = -1
    current_beat = 0
    for token_id in tokens:
        token = str(idx2event[int(token_id)])
        if token.startswith("Track_"):
            current_track = token.removeprefix("Track_")
            if current_track == "Melody":
                melody_bar += 1
                current_beat = 0
        elif current_track == "Melody" and token.startswith("Beat_"):
            current_beat = int(token.removeprefix("Beat_"))
            if not 0 <= current_beat < 16:
                raise ValueError("generated Melody beat is outside 0..15")

        in_melody = current_track == "Melody" and 0 <= melody_bar < TARGET_BARS
        if in_melody:
            slot_ids.append(melody_bar * 4 + min(3, current_beat // 4))
            melody_mask.append(1)
            segment_ids.append(1)
        else:
            slot_ids.append(0)
            melody_mask.append(0)
            segment_ids.append(0)
    if not (len(tokens) == len(segment_ids) == len(slot_ids) == len(melody_mask)):
        raise AssertionError("generated anchor-routing tensors differ in length")
    return segment_ids, slot_ids, melody_mask


def decoder_window_for_prefix(
    tokens: Sequence[int],
    *,
    idx2event: Mapping[int, str],
    max_context: int = MAX_DENSE_TOKENS,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return the V1-style trailing context with globally correct routes."""

    if max_context <= 0 or max_context > MAX_DENSE_TOKENS:
        raise ValueError("invalid dense decoder context length")
    segment_ids, slot_ids, melody_mask = token_routes_for_prefix(
        tokens, idx2event=idx2event
    )
    start = max(0, len(tokens) - max_context)
    return (
        list(tokens[start:]),
        segment_ids[start:],
        slot_ids[start:],
        melody_mask[start:],
    )


def _nucleus_sample(
    logits: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    inadmissibles: Sequence[int] | None,
    rng: np.random.RandomState,
) -> int:
    values = np.asarray(logits, dtype=np.float64).copy()
    if inadmissibles:
        values[np.asarray(inadmissibles, dtype=np.int64)] = -np.inf
    values /= temperature
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError("all Stage 2 logits were masked")
    values -= np.max(values[finite])
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    order = np.argsort(probabilities)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probabilities[order]), top_p)) + 1
    candidates = order[: max(1, min(cutoff, len(order)))]
    candidate_probabilities = probabilities[candidates]
    candidate_probabilities /= candidate_probabilities.sum()
    return int(rng.choice(candidates, p=candidate_probabilities))


def non_event_output_ids(
    output_size: int, idx2event: Mapping[int, str]
) -> list[int]:
    """Return model output classes that are not serialized music events.

    The frozen V1 dense model has one extra output row for the loss-only PAD
    target.  The on-disk event dictionary intentionally does not map that row,
    so it must be excluded before sampling rather than decoded afterwards.
    """

    if output_size <= 0:
        raise ValueError("Stage 2 output size must be positive")
    mapped = {int(token_id) for token_id in idx2event}
    if any(token_id < 0 or token_id >= output_size for token_id in mapped):
        raise ValueError("Stage 2 event ID is outside the model output range")
    return sorted(set(range(output_size)) - mapped)


def dense_melody_event_is_admissible(
    event: str,
    *,
    generated: Sequence[int],
    idx2event: Mapping[int, str],
) -> bool:
    """Reject malformed dense-Melody events before they enter the prefix.

    This is a representation grammar check shared by the V1 and V129 arms,
    not a musical-content constraint.  It preserves every previously sampled
    valid event and only advances the RNG when a malformed event is rejected.
    """

    previous = str(idx2event[int(generated[-1])]) if generated else ""
    if previous == "Track_Melody":
        return event == "Bar_None"
    if previous == "Bar_None":
        return event.startswith("Beat_")
    if previous.startswith("Note_Octave_"):
        return event.startswith("Note_Degree_")
    if previous.startswith("Chord_Root_"):
        return event.startswith("Chord_Quality_")

    if event.startswith("Note_Degree_") or event.startswith("Chord_Quality_"):
        return False
    if event.startswith(("Emotion_", "Key_")):
        return False
    if event in {"Track_Melody", "Bar_None"}:
        return False
    return True


def generate_anchor_conditioned(
    model: AnchorConditionedDenseMusicGPT2,
    planner: CausalAnchorPlanner,
    *,
    event2idx: Mapping[str, int],
    idx2event: Mapping[int, str],
    stage1_tokens: Sequence[str],
    chord_event_ids: Sequence[Sequence[int]],
    primer: Sequence[int],
    seed: int,
    temperature: float,
    top_p: float,
    max_events: int = MAX_OUTPUT_TOKENS,
    max_rejections: int = 512,
    max_segment_attempts: int = 512,
    enable_anchor_conditioning: bool = True,
) -> AnchorGenerationResult:
    """Generate eight Melody bars from a complete predicted anchor plan.

    The planner receives only Stage 1 chords, key and requested emotion. The
    dense model receives only that prediction; no target Melody token or
    ground-truth anchor is available in this function.
    """

    if len(chord_event_ids) < TARGET_BARS:
        raise ValueError("V129 generation requires eight Stage 1 chord bars")
    if max_events <= MAX_DENSE_TOKENS or max_events > MAX_OUTPUT_TOKENS:
        raise ValueError("V129 total output limit must be in 769..1280")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid Stage 2 sampling controls")

    device = next(model.parameters()).device
    planner_device = next(planner.parameters()).device
    if planner_device != device:
        raise ValueError("planner and dense realizer must use one device")
    plan = predict_plan_from_stage1_tokens(
        planner, stage1_tokens=stage1_tokens, device=device
    )
    predicted_active = torch.tensor(
        [plan.active_ids], dtype=torch.long, device=device
    )
    predicted_degree = torch.tensor(
        [plan.degree_ids], dtype=torch.long, device=device
    )
    predicted_octave = torch.tensor(
        [plan.octave_ids], dtype=torch.long, device=device
    )

    track_chord = int(event2idx["Track_Chord"])
    track_melody = int(event2idx["Track_Melody"])
    generated = list(primer) + [track_chord] + list(chord_event_ids[0]) + [track_melody]
    generated_bars = 0
    current_position = 0
    total_attempts = 0
    rejection_budget = RejectionBudget(max_rejections)
    segment_budget = SegmentSamplingBudget(max_segment_attempts)
    rng = np.random.RandomState(seed)
    non_event_ids = non_event_output_ids(
        model.anchor_dense_config.vocab_size, idx2event
    )
    start = time.time()
    model.eval()
    planner.eval()

    with torch.no_grad():
        while generated_bars < TARGET_BARS:
            if len(generated) >= max_events:
                raise RuntimeError("V129 generation exceeded the 1280-token output limit")
            if segment_budget.exhausted:
                raise RuntimeError("V129 per-bar sampling budget exhausted")
            segment_budget.record()
            total_attempts += 1
            decoder_tokens, segment_ids, slot_ids, melody_mask = decoder_window_for_prefix(
                generated, idx2event=idx2event
            )
            tokens = torch.tensor([decoder_tokens], dtype=torch.long, device=device)
            if enable_anchor_conditioning:
                logits = model.forward_with_prediction_arrays(
                    tokens,
                    predicted_active_ids=predicted_active,
                    predicted_degree_ids=predicted_degree,
                    predicted_octave_ids=predicted_octave,
                    token_anchor_slot_ids=torch.tensor(
                        [slot_ids], dtype=torch.long, device=device
                    ),
                    melody_condition_mask=torch.tensor(
                        [melody_mask], dtype=torch.long, device=device
                    ),
                    seg_inp=torch.tensor(
                        [segment_ids], dtype=torch.long, device=device
                    ),
                    attention_mask=torch.ones_like(tokens),
                    keep_last_only=True,
                )
            else:
                logits = model(
                    tokens,
                    seg_inp=torch.tensor(
                        [segment_ids], dtype=torch.long, device=device
                    ),
                    attention_mask=torch.ones_like(tokens),
                    enable_anchor_conditioning=False,
                    keep_last_only=True,
                )
            inadmissibles = monotonic_beat_inadmissibles(
                event2idx, current_position, base=non_event_ids
            )
            word = _nucleus_sample(
                logits[0].detach().float().cpu().numpy(),
                temperature=temperature,
                top_p=top_p,
                inadmissibles=inadmissibles,
                rng=rng,
            )
            event = str(idx2event[word])
            if event in {"PAD_None", "Pad_None", "EOS_None"}:
                if rejection_budget.record():
                    raise RuntimeError("V129 cumulative rejection budget exhausted")
                continue
            if not dense_melody_event_is_admissible(
                event, generated=generated, idx2event=idx2event
            ):
                if rejection_budget.record():
                    raise RuntimeError("V129 cumulative rejection budget exhausted")
                continue
            if event.startswith("Beat_"):
                position = int(event.removeprefix("Beat_"))
                if position < current_position:
                    if rejection_budget.record():
                        raise RuntimeError("V129 cumulative rejection budget exhausted")
                    continue
                current_position = position
            generated.append(word)
            if event == "Track_Chord":
                generated_bars += 1
                segment_budget.reset()
                if generated_bars < TARGET_BARS:
                    generated.extend(chord_event_ids[generated_bars])
                    generated.append(track_melody)
                    current_position = 0

    return AnchorGenerationResult(
        tokens=tuple(strip_terminal_dense_boundary(generated, idx2event)),
        generated_bars=generated_bars,
        total_attempts=total_attempts,
        rejection_count=rejection_budget.rejections,
        elapsed_seconds=time.time() - start,
        anchor_plan=plan,
        anchor_plan_sha256=_sha256_plan(plan),
    )
