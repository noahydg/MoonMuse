"""Pure helpers for optional Stage 2 bar-level phrase planning.

The plan tokens are input-only controls inserted immediately before each
``Track_Melody`` segment.  Legacy dictionaries and event sequences are never
mutated in place.
"""

from __future__ import annotations

import statistics


ROLE_TOKENS = (
    "PhraseRole_Opening",
    "PhraseRole_Development",
    "PhraseRole_Climax",
    "PhraseRole_Cadence",
)
DENSITY_TOKENS = (
    "PhraseDensity_Low",
    "PhraseDensity_Mid",
    "PhraseDensity_High",
)
REGISTER_TOKENS = (
    "PhraseRegister_Low",
    "PhraseRegister_Mid",
    "PhraseRegister_High",
)
PHRASE_PLAN_TOKENS = ROLE_TOKENS + DENSITY_TOKENS + REGISTER_TOKENS
PLAN_PREFIXES = ("PhraseRole_", "PhraseDensity_", "PhraseRegister_")

DEGREE_SEMITONES = {
    "I": 0,
    "I#": 1,
    "II": 2,
    "II#": 3,
    "III": 4,
    "III#": 5,
    "IV": 5,
    "IV#": 6,
    "V": 7,
    "V#": 8,
    "VI": 9,
    "VI#": 10,
    "VII": 11,
    "VII#": 0,
}


def append_phrase_vocabulary(event2idx, idx2event):
    """Append plan tokens while preserving every legacy token index."""
    new_event2idx = dict(event2idx)
    new_idx2event = dict(idx2event)
    overlap = sorted(set(new_event2idx).intersection(PHRASE_PLAN_TOKENS))
    if overlap:
        raise ValueError("dictionary already contains phrase tokens: {}".format(overlap))
    expected = set(range(len(new_event2idx)))
    if set(new_event2idx.values()) != expected:
        raise ValueError("legacy dictionary indices must be contiguous from zero")
    for token, index in new_event2idx.items():
        if new_idx2event.get(index) != token:
            raise ValueError("event2idx/idx2event disagree at {}".format(token))
    for token in PHRASE_PLAN_TOKENS:
        index = len(new_event2idx)
        new_event2idx[token] = index
        new_idx2event[index] = token
    return new_event2idx, new_idx2event


def is_plan_token(token):
    return token.startswith(PLAN_PREFIXES)


def plan_token_indices(event2idx):
    return tuple(sorted(index for token, index in event2idx.items() if is_plan_token(token)))


def strip_plan_tokens(tokens):
    """Remove plan tokens from event names before Stage 3 handoff."""
    return [token for token in tokens if not is_plan_token(token)]


def relative_role(bar_index, bar_count):
    if bar_count < 1 or not 0 <= bar_index < bar_count:
        raise ValueError("invalid bar position: {}/{}".format(bar_index, bar_count))
    quartile = min(3, (4 * bar_index) // bar_count)
    return ROLE_TOKENS[quartile]


def eight_bar_plan():
    """Return the single prospective v76 pilot plan."""
    density = ("Mid", "Mid", "Low", "Mid", "High", "High", "Mid", "Low")
    register = ("Mid", "Mid", "Mid", "Mid", "High", "High", "Mid", "Mid")
    return [
        (
            relative_role(index, 8),
            "PhraseDensity_{}".format(density[index]),
            "PhraseRegister_{}".format(register[index]),
        )
        for index in range(8)
    ]


def _event_token(event):
    value = event.get("value")
    return "{}_{}".format(event.get("name"), "None" if value is None else value)


def c2m_track_positions(events):
    """Return Chord/Melody spans under the repository C2M contract."""
    tracks = []
    eos_index = len(events)
    for index, event in enumerate(events):
        if event.get("name") == "Track" and event.get("value") in ("Chord", "Melody"):
            tracks.append((index, event["value"]))
        elif event.get("name") == "EOS":
            eos_index = index
    chord_spans = []
    melody_spans = []
    for position, (index, track) in enumerate(tracks):
        next_index, next_track = (
            tracks[position + 1] if position + 1 < len(tracks) else (eos_index, "EOS")
        )
        if track == "Chord" and next_track == "Melody":
            chord_spans.append((index, next_index))
        elif track == "Melody" and next_track in ("Chord", "EOS"):
            melody_spans.append((index, next_index))
    if not chord_spans or len(chord_spans) != len(melody_spans):
        raise ValueError(
            "invalid C2M track sequence: chord={}, melody={}".format(
                len(chord_spans), len(melody_spans)
            )
        )
    return chord_spans, melody_spans


def melody_bar_features(events):
    """Extract onset count and mean register from one Melody span."""
    onsets = set()
    pitches = []
    beat = None
    octave = None
    for event in events:
        name = event.get("name")
        value = event.get("value")
        if name == "Beat":
            beat = int(value)
            octave = None
        elif name == "Note_Octave" and beat is not None:
            octave = int(value)
        elif name == "Note_Degree" and beat is not None and octave is not None:
            degree = str(value)
            if degree not in DEGREE_SEMITONES:
                raise ValueError("unsupported degree: {}".format(degree))
            onsets.add(beat)
            pitches.append(octave * 12 + DEGREE_SEMITONES[degree])
            octave = None
    return {
        "onset_count": len(onsets),
        "mean_register": statistics.fmean(pitches) if pitches else None,
        "note_count": len(pitches),
    }


def piece_bar_features(piece):
    if not isinstance(piece, (list, tuple)) or len(piece) < 3:
        raise ValueError("expected a three-part Stage 2 pickle")
    events = piece[2]
    _, melody_spans = c2m_track_positions(events)
    return [melody_bar_features(events[start:end]) for start, end in melody_spans]


def observed_quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot fit an empty quantile")
    return ordered[round((len(ordered) - 1) * probability)]


def fit_plan_thresholds(feature_rows):
    densities = [row["onset_count"] for row in feature_rows]
    registers = [row["mean_register"] for row in feature_rows if row["mean_register"] is not None]
    result = {
        "density_low_max": observed_quantile(densities, 1 / 3),
        "density_high_min": observed_quantile(densities, 2 / 3),
        "register_low_max": observed_quantile(registers, 1 / 3),
        "register_high_min": observed_quantile(registers, 2 / 3),
    }
    if result["density_low_max"] >= result["density_high_min"]:
        raise ValueError("density thresholds are degenerate: {}".format(result))
    if result["register_low_max"] >= result["register_high_min"]:
        raise ValueError("register thresholds are degenerate: {}".format(result))
    return result


def _bin_token(value, low_max, high_min, prefix):
    if value is None or value <= low_max:
        label = "Low"
    elif value >= high_min:
        label = "High"
    else:
        label = "Mid"
    return "{}_{}".format(prefix, label)


def plan_tokens_for_features(bar_index, bar_count, features, thresholds):
    return (
        relative_role(bar_index, bar_count),
        _bin_token(
            features["onset_count"],
            thresholds["density_low_max"],
            thresholds["density_high_min"],
            "PhraseDensity",
        ),
        _bin_token(
            features["mean_register"],
            thresholds["register_low_max"],
            thresholds["register_high_min"],
            "PhraseRegister",
        ),
    )


def transform_piece(piece, thresholds):
    """Insert input-only plan events and recompute C2M positions."""
    if not isinstance(piece, (list, tuple)) or len(piece) < 3:
        raise ValueError("expected a three-part Stage 2 pickle")
    source_events = piece[2]
    features = piece_bar_features(piece)
    bar_count = len(features)
    converted = []
    melody_index = 0
    for event in source_events:
        if event.get("name") == "Track" and event.get("value") == "Melody":
            for token in plan_tokens_for_features(
                melody_index, bar_count, features[melody_index], thresholds
            ):
                name, value = token.split("_", 1)
                converted.append({"name": name, "value": value})
            melody_index += 1
        converted.append(dict(event))
    if melody_index != bar_count:
        raise AssertionError("not every melody bar received a plan")
    chord_pos, melody_pos = c2m_track_positions(converted)
    return [chord_pos, melody_pos, converted]

