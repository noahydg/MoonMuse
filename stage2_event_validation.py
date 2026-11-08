"""Strict token/group checks and evidence-aware completion for Stage 2 scores."""
from __future__ import annotations
import math

DEGREES = {'I', 'I#', 'II', 'II#', 'III', 'IV', 'IV#', 'V', 'V#', 'VI', 'VI#', 'VII'}
QUALITIES = {'+', '/o7', '7', 'Conti', 'M', 'M7', 'None', 'm', 'm7', 'o', 'o7', 'sus2', 'sus4'}
KEYS = {k for name in ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B') for k in (name, name.lower())}


def validate_events(tokens, bars=8, generation=None):
    errors = []
    track = None
    counts = {'Chord': 0, 'Melody': 0}
    position = 0
    pending = None
    keys = []
    notes = 0
    for index, token in enumerate(tokens):
        if pending is not None:
            expected = 'Note_Degree_' if pending[0] == 'note' else 'Chord_Quality_'
            if not token.startswith(expected):
                errors.append((index, 'incomplete_' + pending[0] + '_pair'))
                pending = None
        if token.startswith('Key_'):
            key = token[4:]; keys.append(key)
            if key not in KEYS: errors.append((index, 'invalid_key'))
        elif token.startswith('Emotion_'):
            if token not in {'Emotion_Q1', 'Emotion_Q2', 'Emotion_Q3', 'Emotion_Q4', 'Emotion_None'}:
                errors.append((index, 'invalid_emotion'))
        elif token.startswith('Track_'):
            track = token[6:]
            if track not in counts: errors.append((index, 'invalid_track'))
        elif token == 'Bar_None':
            if track in counts: counts[track] += 1
            else: errors.append((index, 'bar_without_track'))
            position = 0
        elif token.startswith('Beat_'):
            try: value = int(token[5:])
            except ValueError: value = -1
            if track not in counts or not 0 <= value < 16 or value < position:
                errors.append((index, 'invalid_beat'))
            position = value
        elif token.startswith('Note_Octave_'):
            try: value = int(token[12:])
            except ValueError: value = -1
            if track != 'Melody' or not 1 <= value <= 9:
                errors.append((index, 'invalid_octave'))
            pending = ('note', value)
        elif token.startswith('Note_Degree_'):
            if pending is None or pending[0] != 'note' or token[12:] not in DEGREES:
                errors.append((index, 'invalid_degree_pair'))
            else: notes += 1
            pending = None
        elif token.startswith('Chord_Root_'):
            value = token[11:]
            if value not in DEGREES | {'Conti', 'None'} or track not in counts:
                errors.append((index, 'invalid_chord_root'))
            pending = ('chord', value)
        elif token.startswith('Chord_Quality_'):
            value = token[14:]
            if pending is None or pending[0] != 'chord' or value not in QUALITIES:
                errors.append((index, 'invalid_chord_pair'))
            elif (pending[1] in {'Conti', 'None'} or value in {'Conti', 'None'}) and pending[1] != value:
                errors.append((index, 'inconsistent_null_chord'))
            pending = None
        elif token == 'EOS_None':
            if index != len(tokens)-1: errors.append((index, 'internal_eos'))
        else:
            errors.append((index, 'unsupported_token'))
    if pending is not None: errors.append((len(tokens), 'incomplete_' + pending[0] + '_pair'))
    if len(keys) != 1: errors.append((0, 'expected_one_key'))
    reached = counts['Melody'] >= bars
    # Historical text strips EOS and generally lacks a stop-reason sidecar.
    # Do not reconstruct a successful stopping reason from entering the final bar.
    completion = None
    if generation is not None:
        completion = bool(generation.get('stop_reason') in {'target_bars', 'eos'} and
                          generation.get('completed_bars') == bars and reached and not errors)
    elif tokens and tokens[-1] == 'EOS_None':
        completion = bool(reached and not errors)
    return dict(valid=not errors, invalid_tokens=len({i for i, _ in errors}),
                errors=errors, reached_target_bar=reached, complete_bars=completion,
                completion_known=completion is not None, melody_bars=counts['Melody'],
                chord_bars=counts['Chord'], note_pairs=notes)
