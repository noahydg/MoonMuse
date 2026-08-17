"""Validation for the polyphonic dense Stage 2 event representation."""

from __future__ import annotations

from typing import Any, Sequence

from .hard_return import split_tokens


TARGET_BARS = 8
DENSITY_REFERENCE_MIN_FIRST8_ONSETS = 16
DENSITY_REFERENCE_MIN_CELL_ONSETS = 3


def validate_dense_generated_tokens(tokens: Sequence[str]) -> dict[str, Any]:
    """Validate eight dense chord/melody pairs without assuming monophony.

    An onset is one distinct ``(bar, Beat)`` position containing at least one
    complete adjacent ``Note_Octave``/``Note_Degree`` pair. Multiple notes at
    the same position are legal in the frozen V1 dense representation.
    """

    report: dict[str, Any] = {
        "parse_error": "",
        "bars": 0,
        "first8_onset_count": 0,
        "cell_onset_counts": [],
        "note_count": 0,
        "polyphonic_onset_count": 0,
        "density_reference_pass": False,
        "passed": False,
    }
    try:
        _, segments = split_tokens(tokens)
        if len(segments) != TARGET_BARS * 2:
            raise ValueError(
                f"expected {TARGET_BARS * 2} alternating track segments, found {len(segments)}"
            )

        bar_onset_counts: list[int] = []
        note_count = 0
        polyphonic_onset_count = 0
        for bar_index in range(TARGET_BARS):
            chord_track, chord_events = segments[bar_index * 2]
            melody_track, melody_events = segments[bar_index * 2 + 1]
            if (chord_track, melody_track) != ("Chord", "Melody"):
                raise ValueError(
                    f"bar {bar_index} does not alternate Track_Chord/Track_Melody"
                )
            for track, events in ((chord_track, chord_events), (melody_track, melody_events)):
                if not events or events[0] != "Bar_None" or events.count("Bar_None") != 1:
                    raise ValueError(f"bar {bar_index} {track} segment has an invalid boundary")

            current_beat: int | None = None
            previous_beat = -1
            pending_octave: str | None = None
            onset_notes: dict[int, int] = {}
            for token in melody_events[1:]:
                if token.startswith("Beat_"):
                    if pending_octave is not None:
                        raise ValueError("Note_Octave is not followed by Note_Degree")
                    beat = int(token.removeprefix("Beat_"))
                    if not 0 <= beat < 16 or beat < previous_beat:
                        raise ValueError(f"bar {bar_index} contains a non-monotonic beat")
                    current_beat = beat
                    previous_beat = beat
                elif token.startswith("Note_Octave_"):
                    if current_beat is None or pending_octave is not None:
                        raise ValueError("malformed Note_Octave token")
                    pending_octave = token
                elif token.startswith("Note_Degree_"):
                    if current_beat is None or pending_octave is None:
                        raise ValueError("Note_Degree lacks an adjacent Note_Octave")
                    onset_notes[current_beat] = onset_notes.get(current_beat, 0) + 1
                    note_count += 1
                    pending_octave = None
                elif pending_octave is not None:
                    raise ValueError("Note_Octave/Note_Degree pair is not adjacent")
            if pending_octave is not None:
                raise ValueError("trailing Note_Octave lacks Note_Degree")
            bar_onset_counts.append(len(onset_notes))
            polyphonic_onset_count += sum(count > 1 for count in onset_notes.values())

        cell_counts = [
            sum(bar_onset_counts[start : start + 2])
            for start in range(0, TARGET_BARS, 2)
        ]
        onset_count = sum(bar_onset_counts)
        report.update(
            {
                "bars": TARGET_BARS,
                "first8_onset_count": onset_count,
                "cell_onset_counts": cell_counts,
                "note_count": note_count,
                "polyphonic_onset_count": polyphonic_onset_count,
                "density_reference_pass": (
                    onset_count >= DENSITY_REFERENCE_MIN_FIRST8_ONSETS
                    and min(cell_counts) >= DENSITY_REFERENCE_MIN_CELL_ONSETS
                ),
                # Syntax, complete eight-bar structure and non-empty musical
                # content define generation validity.  The older 16/3 onset
                # thresholds were designed to decide whether a training piece
                # could support four two-bar motif pseudo-labels; sparse cells
                # remain valid model outputs and must not be silently dropped.
                "passed": note_count > 0,
            }
        )
    except Exception as exc:
        report["parse_error"] = f"{type(exc).__name__}: {exc}"
    return report
