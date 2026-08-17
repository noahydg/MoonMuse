"""Phrase-break-aware octave realization for deterministic Stage 2 returns.

This module is intentionally independent of the historical V117A implementation.
V117A treated every melody onset as one continuous pitch path. V118 N03 showed
that doing so across a long silence can pull a returning phrase into the
accompaniment register. The functions here preserve pitch class and onset timing,
but reset pitch-continuity scoring after a phrase-sized onset gap and reject any
candidate that introduces melody/accompaniment crossing where the literal copied
note was above the active accompaniment.
"""

from __future__ import annotations

from dataclasses import dataclass


MIN_RENDER_PITCH = 36
MAX_RENDER_PITCH = 96
DEFAULT_PHRASE_BREAK_TICKS = 1920


@dataclass(frozen=True)
class MelodyOnset:
    tick: int
    pitch: int
    accompaniment_top: int | None = None


def contour_flip(original_interval: int, candidate_interval: int) -> int:
    return int(
        original_interval != 0
        and candidate_interval != 0
        and (original_interval > 0) != (candidate_interval > 0)
    )


def _octave_candidates(onset: MelodyOnset) -> list[int]:
    candidates = sorted(
        {
            onset.pitch + 12 * octave
            for octave in range(-6, 7)
            if MIN_RENDER_PITCH
            <= onset.pitch + 12 * octave
            <= MAX_RENDER_PITCH
        }
    )
    if onset.accompaniment_top is not None and onset.pitch > onset.accompaniment_top:
        candidates = [
            pitch for pitch in candidates if pitch > onset.accompaniment_top
        ]
    if not candidates:
        raise ValueError("no octave candidate preserves the literal voice relation")
    return candidates


def choose_register_reset_path(
    previous: MelodyOnset,
    copied: list[MelodyOnset],
    cadence: MelodyOnset,
    *,
    phrase_break_ticks: int = DEFAULT_PHRASE_BREAK_TICKS,
) -> tuple[list[int], dict[str, object]]:
    """Choose a deterministic octave path without smoothing across long silence.

    The score remains lexicographic to retain the V117A mechanism: minimize
    severe active-segment jumps, then the maximum active-segment jump, contour
    flips, total active motion and octave displacement. A transition separated
    by at least ``phrase_break_ticks`` contributes no continuity jump. Literal
    melody-above-accompaniment relations are hard-preserved onset by onset.
    """

    if not copied:
        raise ValueError("copied return contains no melody onsets")
    if phrase_break_ticks <= 0:
        raise ValueError("phrase_break_ticks must be positive")

    states: dict[int, tuple[int, int, int, int, int, tuple[int, ...]]] = {
        previous.pitch: (0, 0, 0, 0, 0, ())
    }
    previous_tick = previous.tick
    previous_original: int | None = None
    reset_before_ticks: list[int] = []

    for onset in copied:
        active_transition = onset.tick - previous_tick < phrase_break_ticks
        if not active_transition:
            reset_before_ticks.append(onset.tick)
        next_states: dict[
            int, tuple[int, int, int, int, int, tuple[int, ...]]
        ] = {}
        for candidate in _octave_candidates(onset):
            best = None
            for last_pitch, state in states.items():
                severe, maximum, flips, total, displacement, path = state
                jump = abs(candidate - last_pitch) if active_transition else 0
                flip = 0
                if active_transition and previous_original is not None:
                    flip = contour_flip(
                        onset.pitch - previous_original,
                        candidate - last_pitch,
                    )
                score = (
                    severe + int(active_transition and jump > 12),
                    max(maximum, jump),
                    flips + flip,
                    total + jump,
                    displacement + abs(candidate - onset.pitch),
                    path + (candidate,),
                )
                if best is None or score < best:
                    best = score
            if best is None:
                raise AssertionError("register-reset path state unexpectedly empty")
            next_states[candidate] = best
        states = next_states
        previous_tick = onset.tick
        previous_original = onset.pitch

    cadence_transition_active = cadence.tick - previous_tick < phrase_break_ticks
    if not cadence_transition_active:
        reset_before_ticks.append(cadence.tick)
    best = None
    for last_pitch, state in states.items():
        severe, maximum, flips, total, displacement, path = state
        jump = abs(cadence.pitch - last_pitch) if cadence_transition_active else 0
        score = (
            severe + int(cadence_transition_active and jump > 12),
            max(maximum, jump),
            flips,
            total + jump,
            displacement,
            path,
        )
        if best is None or score < best:
            best = score
    if best is None:
        raise AssertionError("no register-reset path was selected")

    selected = list(best[-1])
    newly_introduced_crossings = sum(
        onset.accompaniment_top is not None
        and onset.pitch > onset.accompaniment_top
        and pitch <= onset.accompaniment_top
        for onset, pitch in zip(copied, selected)
    )
    if newly_introduced_crossings:
        raise AssertionError("register-reset path introduced voice crossing")
    return selected, {
        "severe_active_jump_count": best[0],
        "maximum_active_jump_semitones": best[1],
        "active_contour_flip_count": best[2],
        "total_active_motion_semitones": best[3],
        "total_octave_displacement": best[4],
        "phrase_break_ticks": phrase_break_ticks,
        "reset_before_ticks": reset_before_ticks,
        "newly_introduced_voice_crossing_count": newly_introduced_crossings,
    }


def has_phrase_break_reset(
    previous: MelodyOnset,
    copied: list[MelodyOnset],
    *,
    phrase_break_ticks: int = DEFAULT_PHRASE_BREAK_TICKS,
) -> bool:
    """Return whether the copied phrase starts after the frozen reset gap."""

    if not copied:
        raise ValueError("copied return contains no melody onsets")
    return copied[0].tick - previous.tick >= phrase_break_ticks
