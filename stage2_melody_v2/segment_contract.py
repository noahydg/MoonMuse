"""Inherited Stage 2 segment IDs shared by training and generation."""

from __future__ import annotations

from typing import Mapping, Sequence


PRODUCTION_PREFIX_CHORD_SEGMENT_ID = 0
PRODUCTION_MELODY_SEGMENT_ID = 1


def segment_ids_for_tokens(
    input_ids: Sequence[int],
    event2idx: Mapping[str, int],
    *,
    require_both_tracks: bool = True,
) -> tuple[int, ...]:
    """Return the segment IDs used to train the inherited Stage 2 checkpoint.

    V90 pieces are serialized as ``[chord_ranges, melody_ranges, events]``. The
    inherited DataLoader assigns segment 1 to the second range array (Melody), and
    the inherited generator independently uses prefix/Chord=0 and Melody=1.
    """

    try:
        chord_token = int(event2idx["Track_Chord"])
        melody_token = int(event2idx["Track_Melody"])
    except KeyError as exc:
        raise ValueError(f"V90 dictionary lacks a track token: {exc}") from exc
    state = PRODUCTION_PREFIX_CHORD_SEGMENT_ID
    segments = []
    for token in input_ids:
        if token == chord_token:
            state = PRODUCTION_PREFIX_CHORD_SEGMENT_ID
        elif token == melody_token:
            state = PRODUCTION_MELODY_SEGMENT_ID
        segments.append(state)
    if require_both_tracks and (
        PRODUCTION_PREFIX_CHORD_SEGMENT_ID not in segments
        or PRODUCTION_MELODY_SEGMENT_ID not in segments
    ):
        raise ValueError("production example does not contain both track segments")
    return tuple(segments)
