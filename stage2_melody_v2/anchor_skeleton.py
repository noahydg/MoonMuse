"""Deterministic 32-slot anchor skeletons for hierarchical Stage 2 generation.

The extractor retains the highest note at each quarter-note anchor position in
the first eight Melody bars and emits an explicit rest when no note starts at
that position.  It is a self-supervised planning target derived from the
existing dense C2M event stream; it is not a human melody annotation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .hard_return import CHORD_INTERVALS, parse_chord_grids, split_tokens
from .motif_memory import MAJOR_DEGREES, MINOR_DEGREES


SCHEMA = "moonmuse-stage2-v2-anchor-skeleton-v126-v1"
TARGET_BARS = 8
ANCHOR_BEATS = (0, 4, 8, 12)
ANCHOR_SLOT_COUNT = TARGET_BARS * len(ANCHOR_BEATS)
AMBIGUOUS_MINOR_DEGREES = frozenset({"II#", "V#"})


@dataclass(frozen=True)
class AnchorSlot:
    index: int
    bar: int
    beat: int
    active: bool
    degree: str | None
    octave: int | None
    functional_pitch: int | None
    chord_root: str | None
    chord_quality: str | None
    chord_tone: bool | None
    onset_polyphony: int
    ambiguous_minor_degree: bool


@dataclass(frozen=True)
class AnchorSkeleton:
    schema: str
    key: str
    emotion: str | None
    target_bars: int
    anchor_beats: tuple[int, ...]
    source_note_count: int
    source_onset_count: int
    slots: tuple[AnchorSlot, ...]

    @property
    def active_count(self) -> int:
        return sum(slot.active for slot in self.slots)

    @property
    def coverage(self) -> float:
        return self.active_count / len(self.slots)

    @property
    def chord_tone_ratio(self) -> float | None:
        values = [slot.chord_tone for slot in self.slots if slot.chord_tone is not None]
        return sum(values) / len(values) if values else None

    @property
    def cell_active_counts(self) -> tuple[int, int, int, int]:
        return tuple(
            sum(slot.active for slot in self.slots[cell * 8 : (cell + 1) * 8])
            for cell in range(4)
        )  # type: ignore[return-value]

    def fingerprint(self) -> str:
        payload = [
            [slot.active, slot.degree, slot.octave]
            for slot in self.slots
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["active_count"] = self.active_count
        payload["coverage"] = self.coverage
        payload["chord_tone_ratio"] = self.chord_tone_ratio
        payload["cell_active_counts"] = list(self.cell_active_counts)
        payload["fingerprint"] = self.fingerprint()
        return payload


def _degree_map(key: str) -> dict[str, int]:
    return MAJOR_DEGREES if key == key.upper() else MINOR_DEGREES


def _functional_pitch(key: str, octave: int, degree: str) -> int:
    mapping = _degree_map(key)
    if degree not in mapping:
        raise ValueError(f"unsupported functional degree: {degree}")
    return octave * 12 + mapping[degree]


def _parse_melody_bar(
    events: Sequence[str], key: str
) -> dict[int, list[tuple[int, str, int, bool]]]:
    if not events or events[0] != "Bar_None":
        raise ValueError("melody segment does not start with Bar_None")
    beat = 0
    pending_octave: int | None = None
    output: dict[int, list[tuple[int, str, int, bool]]] = {}
    for token in events[1:]:
        if token.startswith("Beat_"):
            beat = int(token.removeprefix("Beat_"))
            if not 0 <= beat < 16:
                raise ValueError(f"beat outside 0..15: {beat}")
            pending_octave = None
        elif token.startswith("Note_Octave_"):
            pending_octave = int(token.removeprefix("Note_Octave_"))
        elif token.startswith("Note_Degree_"):
            if pending_octave is None:
                raise ValueError("Note_Degree without Note_Octave")
            degree = token.removeprefix("Note_Degree_")
            pitch = _functional_pitch(key, pending_octave, degree)
            ambiguous = key != key.upper() and degree in AMBIGUOUS_MINOR_DEGREES
            output.setdefault(beat, []).append(
                (pending_octave, degree, pitch, ambiguous)
            )
            pending_octave = None
    if pending_octave is not None:
        raise ValueError("terminal Note_Octave without Note_Degree")
    return output


def _chord_tone(
    key: str,
    note_degree: str,
    chord_root: str | None,
    chord_quality: str | None,
) -> bool | None:
    if chord_root is None or chord_quality is None:
        return None
    if (
        key != key.upper()
        and (note_degree in AMBIGUOUS_MINOR_DEGREES or chord_root in AMBIGUOUS_MINOR_DEGREES)
    ):
        return None
    mapping = _degree_map(key)
    intervals = CHORD_INTERVALS.get(chord_quality)
    if note_degree not in mapping or chord_root not in mapping or intervals is None:
        return None
    return (mapping[note_degree] - mapping[chord_root]) % 12 in intervals


def extract_anchor_skeleton(
    tokens: Iterable[str], *, target_bars: int = TARGET_BARS
) -> AnchorSkeleton:
    if target_bars != TARGET_BARS:
        raise ValueError("V126 anchor contract is fixed to eight bars")
    preamble, segments = split_tokens(tokens)
    key_values = [
        token.removeprefix("Key_")
        for token in preamble
        if token.startswith("Key_")
    ]
    if len(key_values) != 1:
        raise ValueError(f"expected one Key token, found {len(key_values)}")
    key = key_values[0]
    emotion = next(
        (
            token.removeprefix("Emotion_")
            for token in preamble
            if token.startswith("Emotion_")
        ),
        None,
    )
    chord_bars = [events for track, events in segments if track == "Chord"]
    melody_bars = [events for track, events in segments if track == "Melody"]
    if len(chord_bars) < target_bars or len(melody_bars) < target_bars:
        raise ValueError(
            f"fewer_than_{target_bars}_matched_bars:"
            f"{len(chord_bars)}/{len(melody_bars)}"
        )
    chord_grids = parse_chord_grids(chord_bars[:target_bars])
    parsed_melody = [
        _parse_melody_bar(events, key) for events in melody_bars[:target_bars]
    ]

    slots: list[AnchorSlot] = []
    for bar in range(target_bars):
        for beat in ANCHOR_BEATS:
            index = bar * len(ANCHOR_BEATS) + ANCHOR_BEATS.index(beat)
            candidates = parsed_melody[bar].get(beat, [])
            chord = chord_grids[bar].state_at(beat)
            if not candidates:
                slots.append(
                    AnchorSlot(
                        index=index,
                        bar=bar,
                        beat=beat,
                        active=False,
                        degree=None,
                        octave=None,
                        functional_pitch=None,
                        chord_root=chord.root if chord else None,
                        chord_quality=chord.quality if chord else None,
                        chord_tone=None,
                        onset_polyphony=0,
                        ambiguous_minor_degree=False,
                    )
                )
                continue
            octave, degree, pitch, ambiguous = max(
                candidates,
                key=lambda item: (item[2], -candidates.index(item)),
            )
            slots.append(
                AnchorSlot(
                    index=index,
                    bar=bar,
                    beat=beat,
                    active=True,
                    degree=degree,
                    octave=octave,
                    functional_pitch=pitch,
                    chord_root=chord.root if chord else None,
                    chord_quality=chord.quality if chord else None,
                    chord_tone=_chord_tone(
                        key,
                        degree,
                        chord.root if chord else None,
                        chord.quality if chord else None,
                    ),
                    onset_polyphony=len(candidates),
                    ambiguous_minor_degree=ambiguous,
                )
            )
    if len(slots) != ANCHOR_SLOT_COUNT:
        raise AssertionError("anchor skeleton does not contain exactly 32 slots")
    source_note_count = sum(
        len(notes) for bar in parsed_melody for notes in bar.values()
    )
    source_onset_count = sum(len(bar) for bar in parsed_melody)
    return AnchorSkeleton(
        schema=SCHEMA,
        key=key,
        emotion=emotion,
        target_bars=target_bars,
        anchor_beats=ANCHOR_BEATS,
        source_note_count=source_note_count,
        source_onset_count=source_onset_count,
        slots=tuple(slots),
    )

