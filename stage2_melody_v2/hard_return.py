"""Deterministic hard A-return control for frozen MoonMuse Stage 2 text.

The intervention preserves bars 1-6 and every chord segment.  It copies A into
the A-cadence cell, applies one chord-aware global transposition, and retains the
original final quarter-bar as a cadence tail.  It is a diagnostic control, not a
learned V2 generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .generation import tokens_to_events
from .motif_memory import (
    MAJOR_DEGREES,
    MINOR_DEGREES,
    cadence_target,
    compare_motifs,
    extract_cell_motifs,
    parse_melody_onsets,
)


TARGET_BARS = 8
CADENCE_TAIL_BEAT = 12
MIN_A_ONSETS = 3
MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
SHIFT_CANDIDATES = tuple(range(-6, 7))

TONIC_PITCH_CLASSES = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

CHORD_INTERVALS = {
    "M": (0, 4, 7),
    "M7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
    "m": (0, 3, 7),
    "m7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}


class HardReturnError(ValueError):
    """Base error for malformed or ineligible hard-return inputs."""


class HardReturnIneligible(HardReturnError):
    """A syntactically valid source does not meet the frozen V116 rule."""


@dataclass(frozen=True)
class NoteEvent:
    beat: int
    octave: int
    degree: str


@dataclass(frozen=True)
class ChordState:
    root: str
    quality: str


@dataclass(frozen=True)
class ChordGrid:
    initial: ChordState | None
    changes: tuple[tuple[int, ChordState], ...]

    def state_at(self, beat: int) -> ChordState | None:
        state = self.initial
        for position, candidate in self.changes:
            if position > beat:
                break
            state = candidate
        return state


@dataclass(frozen=True)
class ParsedPiece:
    key: str
    preamble: tuple[str, ...]
    segments: tuple[tuple[str, tuple[str, ...]], ...]
    chord_segment_indices: tuple[int, ...]
    melody_segment_indices: tuple[int, ...]
    chord_grids: tuple[ChordGrid, ...]
    melody_notes: tuple[tuple[NoteEvent, ...], ...]


def split_tokens(
    tokens: Iterable[str],
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    preamble: list[str] = []
    segments: list[tuple[str, list[str]]] = []
    current_track: str | None = None
    current_events: list[str] = []
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if token.startswith("Track_"):
            if current_track is not None:
                segments.append((current_track, current_events))
            current_track = token.removeprefix("Track_")
            current_events = []
        elif current_track is None:
            preamble.append(token)
        else:
            current_events.append(token)
    if current_track is not None:
        segments.append((current_track, current_events))
    return preamble, segments


def serialize_tokens(
    preamble: Sequence[str], segments: Sequence[tuple[str, Sequence[str]]]
) -> list[str]:
    output = list(preamble)
    for track, events in segments:
        output.append(f"Track_{track}")
        output.extend(events)
    return output


def parse_melody_bar(events: Sequence[str]) -> tuple[NoteEvent, ...]:
    if not events or events[0] != "Bar_None":
        raise HardReturnError("melody segment does not start with Bar_None")
    beat = 0
    pending_octave: int | None = None
    notes: dict[int, NoteEvent] = {}
    for token in events[1:]:
        if token.startswith("Beat_"):
            if pending_octave is not None:
                raise HardReturnError("Note_Octave without Note_Degree")
            beat = int(token.removeprefix("Beat_"))
            if not 0 <= beat < 16:
                raise HardReturnError(f"beat outside 0..15: {beat}")
        elif token.startswith("Note_Octave_"):
            if pending_octave is not None:
                raise HardReturnError("consecutive Note_Octave tokens")
            pending_octave = int(token.removeprefix("Note_Octave_"))
        elif token.startswith("Note_Degree_"):
            if pending_octave is None:
                raise HardReturnError("Note_Degree without Note_Octave")
            if beat in notes:
                raise HardReturnError(f"non-monophonic melody onset at beat {beat}")
            notes[beat] = NoteEvent(
                beat=beat,
                octave=pending_octave,
                degree=token.removeprefix("Note_Degree_"),
            )
            pending_octave = None
    if pending_octave is not None:
        raise HardReturnError("terminal Note_Octave without Note_Degree")
    return tuple(notes[beat] for beat in sorted(notes))


def parse_chord_grids(
    chord_bars: Sequence[Sequence[str]],
) -> tuple[ChordGrid, ...]:
    previous_root: str | None = None
    previous_quality: str | None = None
    output: list[ChordGrid] = []
    for events in chord_bars:
        if not events or events[0] != "Bar_None":
            raise HardReturnError("chord segment does not start with Bar_None")
        initial = (
            ChordState(previous_root, previous_quality)
            if previous_root is not None and previous_quality is not None
            else None
        )
        beat = 0
        pending_root: str | None = None
        changes: list[tuple[int, ChordState]] = []
        for token in events[1:]:
            if token.startswith("Beat_"):
                beat = int(token.removeprefix("Beat_"))
                pending_root = None
            elif token.startswith("Chord_Root_"):
                pending_root = token.removeprefix("Chord_Root_")
            elif token.startswith("Chord_Quality_"):
                quality = token.removeprefix("Chord_Quality_")
                if pending_root is not None and pending_root != "Conti":
                    previous_root = pending_root
                if quality != "Conti":
                    previous_quality = quality
                if previous_root is not None and previous_quality is not None:
                    changes.append(
                        (beat, ChordState(previous_root, previous_quality))
                    )
                pending_root = None
        output.append(ChordGrid(initial=initial, changes=tuple(changes)))
    return tuple(output)


def parse_piece(tokens: Sequence[str]) -> ParsedPiece:
    preamble, mutable_segments = split_tokens(tokens)
    key_tokens = [token.removeprefix("Key_") for token in preamble if token.startswith("Key_")]
    if len(key_tokens) != 1:
        raise HardReturnError(f"expected one Key token, got {len(key_tokens)}")
    chord_indices = tuple(
        index for index, (track, _) in enumerate(mutable_segments) if track == "Chord"
    )
    melody_indices = tuple(
        index for index, (track, _) in enumerate(mutable_segments) if track == "Melody"
    )
    if len(chord_indices) < TARGET_BARS or len(melody_indices) < TARGET_BARS:
        raise HardReturnIneligible("fewer_than_eight_matched_chord_melody_bars")
    chord_bars = [mutable_segments[index][1] for index in chord_indices[:TARGET_BARS]]
    melody_bars = [mutable_segments[index][1] for index in melody_indices[:TARGET_BARS]]
    return ParsedPiece(
        key=key_tokens[0],
        preamble=tuple(preamble),
        segments=tuple(
            (track, tuple(events)) for track, events in mutable_segments
        ),
        chord_segment_indices=chord_indices,
        melody_segment_indices=melody_indices,
        chord_grids=parse_chord_grids(chord_bars),
        melody_notes=tuple(parse_melody_bar(events) for events in melody_bars),
    )


def degree_map(key: str) -> dict[str, int]:
    return MAJOR_DEGREES if key == key.upper() else MINOR_DEGREES


def tonic_pitch_class(key: str) -> int:
    name = key.upper()
    if name not in TONIC_PITCH_CLASSES:
        raise HardReturnError(f"unsupported key: {key}")
    return TONIC_PITCH_CLASSES[name]


def note_pitch(key: str, note: NoteEvent) -> int:
    mapping = degree_map(key)
    if note.degree not in mapping:
        raise HardReturnError(f"unsupported functional degree: {note.degree}")
    return note.octave * 12 + tonic_pitch_class(key) + mapping[note.degree]


def inverse_degree_map(key: str) -> dict[int, str]:
    mapping = degree_map(key)
    natural = ("I", "II", "III", "IV", "V", "VI", "VII")
    sharp = ("I#", "II#", "III#", "IV#", "V#", "VI#", "VII#")
    output: dict[int, str] = {}
    for degree in (*natural, *sharp):
        if degree in mapping:
            output.setdefault(mapping[degree], degree)
    return output


def pitch_to_note(key: str, pitch: int, *, beat: int) -> NoteEvent:
    tonic = tonic_pitch_class(key)
    relative = (pitch - tonic) % 12
    inverse = inverse_degree_map(key)
    if relative not in inverse:
        raise HardReturnIneligible(
            f"shifted_pitch_class_not_in_vocabulary:{relative}"
        )
    octave = (pitch - relative) // 12
    return NoteEvent(beat=beat, octave=octave, degree=inverse[relative])


def chord_tone_hit(key: str, pitch: int, chord: ChordState | None) -> bool | None:
    if chord is None:
        return None
    mapping = degree_map(key)
    intervals = CHORD_INTERVALS.get(chord.quality)
    if chord.root not in mapping or intervals is None:
        return None
    root_pc = (tonic_pitch_class(key) + mapping[chord.root]) % 12
    return (pitch - root_pc) % 12 in intervals


def chord_tone_ratio(
    piece: ParsedPiece,
    notes: Sequence[tuple[int, NoteEvent]],
    *,
    shift: int = 0,
) -> float | None:
    hits: list[bool] = []
    for bar_index, note in notes:
        chord = piece.chord_grids[bar_index].state_at(note.beat)
        result = chord_tone_hit(piece.key, note_pitch(piece.key, note) + shift, chord)
        if result is not None:
            hits.append(result)
    return sum(hits) / len(hits) if hits else None


def shift_is_representable(
    key: str, notes: Sequence[NoteEvent], shift: int
) -> bool:
    for note in notes:
        pitch = note_pitch(key, note) + shift
        if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
            return False
        try:
            pitch_to_note(key, pitch, beat=note.beat)
        except HardReturnIneligible:
            return False
    return True


def select_global_shift(
    piece: ParsedPiece, copied: Sequence[tuple[int, NoteEvent]]
) -> tuple[int, float, list[dict[str, int | float]]]:
    candidates: list[dict[str, int | float]] = []
    raw_notes = [note for _, note in copied]
    for shift in SHIFT_CANDIDATES:
        if not shift_is_representable(piece.key, raw_notes, shift):
            continue
        ratio = chord_tone_ratio(piece, copied, shift=shift)
        if ratio is None:
            continue
        candidates.append({"shift": shift, "chord_tone_ratio": ratio})
    if not candidates:
        raise HardReturnIneligible("no_representable_chord_scored_shift")
    selected = max(
        candidates,
        key=lambda row: (
            float(row["chord_tone_ratio"]),
            -abs(int(row["shift"])),
            -int(row["shift"]),
        ),
    )
    return (
        int(selected["shift"]),
        float(selected["chord_tone_ratio"]),
        sorted(candidates, key=lambda row: int(row["shift"])),
    )


def chord_payloads(events: Sequence[str]) -> dict[int, list[str]]:
    if not events or events[0] != "Bar_None":
        raise HardReturnError("chord segment does not start with Bar_None")
    output: dict[int, list[str]] = {}
    beat: int | None = None
    for token in events[1:]:
        if token.startswith("Beat_"):
            beat = int(token.removeprefix("Beat_"))
            output.setdefault(beat, [])
        elif token.startswith(("Chord_Root_", "Chord_Quality_")):
            if beat is None:
                raise HardReturnError("chord token before Beat")
            output[beat].append(token)
        elif token.startswith(("EOS_", "PAD_", "Pad_")):
            continue
        else:
            raise HardReturnError(f"unsupported chord-bar token: {token}")
    return output


def build_melody_bar(
    chord_events: Sequence[str], notes: Sequence[NoteEvent]
) -> list[str]:
    note_map: dict[int, NoteEvent] = {}
    for note in notes:
        if note.beat in note_map:
            raise HardReturnError(f"duplicate rewritten onset at beat {note.beat}")
        note_map[note.beat] = note
    payloads = chord_payloads(chord_events)
    output = ["Bar_None"]
    for beat in sorted(set(payloads) | set(note_map)):
        output.append(f"Beat_{beat}")
        output.extend(payloads.get(beat, ()))
        note = note_map.get(beat)
        if note is not None:
            output.extend(
                (f"Note_Octave_{note.octave}", f"Note_Degree_{note.degree}")
            )
    return output


def shifted_note(key: str, note: NoteEvent, shift: int) -> NoteEvent:
    if shift == 0:
        return note
    return pitch_to_note(key, note_pitch(key, note) + shift, beat=note.beat)


def motif_report(tokens: Sequence[str]) -> dict[str, object]:
    events = tokens_to_events(tokens)
    _, onsets, bar_count = parse_melody_onsets(events)
    motifs = extract_cell_motifs(
        [onset for onset in onsets if onset.bar < TARGET_BARS]
    )
    comparison = compare_motifs(motifs[0], motifs[3])
    return {
        "bar_count": bar_count,
        "first8_onset_count": sum(onset.bar < TARGET_BARS for onset in onsets),
        "cell_onset_counts": [motif.onset_count for motif in motifs],
        "a_to_a_cad": comparison.to_dict(),
        "cadence_target": cadence_target(onsets),
    }


def transform_hard_return(tokens: Sequence[str]) -> tuple[list[str], dict[str, object]]:
    piece = parse_piece(tokens)
    a_notes = (*piece.melody_notes[0], *piece.melody_notes[1])
    if len(a_notes) < MIN_A_ONSETS:
        raise HardReturnIneligible(f"a_cell_has_{len(a_notes)}_onsets")
    cadence_tail = tuple(
        note
        for note in piece.melody_notes[7]
        if note.beat >= CADENCE_TAIL_BEAT
    )
    if not cadence_tail:
        raise HardReturnIneligible("bar8_has_no_cadence_tail_at_or_after_beat12")

    copied = [
        *((6, note) for note in piece.melody_notes[0]),
        *((7, note) for note in piece.melody_notes[1] if note.beat < CADENCE_TAIL_BEAT),
    ]
    if not copied:
        raise HardReturnIneligible("a_cell_has_no_copyable_onsets")
    shift, copied_chord_ratio, shift_candidates = select_global_shift(piece, copied)

    new_bar7 = tuple(shifted_note(piece.key, note, shift) for note in piece.melody_notes[0])
    new_bar8 = tuple(
        sorted(
            (
                *(
                    shifted_note(piece.key, note, shift)
                    for note in piece.melody_notes[1]
                    if note.beat < CADENCE_TAIL_BEAT
                ),
                *cadence_tail,
            ),
            key=lambda note: note.beat,
        )
    )

    mutable_segments = [
        (track, list(events)) for track, events in piece.segments
    ]
    for bar_index, notes in ((6, new_bar7), (7, new_bar8)):
        melody_segment_index = piece.melody_segment_indices[bar_index]
        chord_events = piece.segments[piece.chord_segment_indices[bar_index]][1]
        mutable_segments[melody_segment_index] = (
            "Melody",
            build_melody_bar(chord_events, notes),
        )
    transformed = serialize_tokens(piece.preamble, mutable_segments)
    transformed_piece = parse_piece(transformed)

    modified_indices = {
        piece.melody_segment_indices[6],
        piece.melody_segment_indices[7],
    }
    untouched_segments_equal = all(
        piece.segments[index]
        == (mutable_segments[index][0], tuple(mutable_segments[index][1]))
        for index in range(len(piece.segments))
        if index not in modified_indices
    )
    chord_segments_equal = all(
        piece.segments[index]
        == (mutable_segments[index][0], tuple(mutable_segments[index][1]))
        for index in piece.chord_segment_indices
    )
    cadence_tail_equal = transformed_piece.melody_notes[7][-len(cadence_tail) :] == cadence_tail

    baseline_target_notes = [
        (bar_index, note)
        for bar_index in (6, 7)
        for note in piece.melody_notes[bar_index]
    ]
    transformed_target_notes = [
        (bar_index, note)
        for bar_index in (6, 7)
        for note in transformed_piece.melody_notes[bar_index]
    ]
    copied_bar8 = [note for note in new_bar8 if note.beat < CADENCE_TAIL_BEAT]
    join_leap = None
    if copied_bar8 and cadence_tail:
        join_leap = abs(
            note_pitch(piece.key, cadence_tail[0])
            - note_pitch(piece.key, copied_bar8[-1])
        )

    baseline_motif = motif_report(tokens)
    transformed_motif = motif_report(transformed)
    report: dict[str, object] = {
        "schema": "moonmuse-stage2-v2-hard-return-transform-v1",
        "formalStudyData": False,
        "selected_shift_semitones": shift,
        "selected_copied_region_chord_tone_ratio": copied_chord_ratio,
        "shift_candidates": shift_candidates,
        "source_a_onset_count": len(a_notes),
        "copied_onset_count": len(copied),
        "cadence_tail_onset_count": len(cadence_tail),
        "cadence_join_abs_semitones": join_leap,
        "baseline_bars7_8_chord_tone_ratio": chord_tone_ratio(
            piece, baseline_target_notes
        ),
        "hard_return_bars7_8_chord_tone_ratio": chord_tone_ratio(
            transformed_piece, transformed_target_notes
        ),
        "baseline": baseline_motif,
        "hard_return": transformed_motif,
        "checks": {
            "untouched_segments_equal": untouched_segments_equal,
            "all_chord_segments_equal": chord_segments_equal,
            "cadence_tail_equal": cadence_tail_equal,
            "first_six_melody_bars_equal": (
                piece.melody_notes[:6] == transformed_piece.melody_notes[:6]
            ),
            "first_eight_onset_monophonic": True,
        },
    }
    if not all(bool(value) for value in report["checks"].values()):
        raise AssertionError("hard-return preservation check failed")
    return transformed, report
