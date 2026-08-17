from stage2_melody_v2.anchor_production_data import (
    build_production_example_from_payload,
)


def event(name: str, value):
    return {"name": name, "value": value}


def payload():
    events = [event("Emotion", "Q1"), event("Key", "C")]
    chord_ranges = []
    melody_ranges = []
    for bar in range(8):
        chord_start = len(events)
        events.extend(
            [
                event("Track", "Chord"),
                event("Bar", None),
                event("Beat", 0),
                event("Chord_Root", "I"),
                event("Chord_Quality", "M"),
            ]
        )
        chord_ranges.append((chord_start, len(events)))
        melody_start = len(events)
        events.extend(
            [
                event("Track", "Melody"),
                event("Bar", None),
                event("Beat", 0),
                event("Note_Octave", 4),
                event("Note_Degree", "I"),
                event("Beat", 8),
                event("Note_Octave", 4),
                event("Note_Degree", "V"),
            ]
        )
        melody_ranges.append((melody_start, len(events)))
    return [chord_ranges, melody_ranges, events]


def vocabulary(payload):
    tokens = []
    for item in payload[2]:
        value = "None" if item["value"] is None else item["value"]
        token = f"{item['name']}_{value}"
        if token not in tokens:
            tokens.append(token)
    if "EOS_None" not in tokens:
        tokens.append("EOS_None")
    return {token: index for index, token in enumerate(tokens)}


def test_builds_original_dense_target_plus_anchor_alignment() -> None:
    source = payload()
    event2idx = vocabulary(source)
    pad = len(event2idx)
    example = build_production_example_from_payload(
        dataset="fixture",
        split="train",
        source_id="fixture.pkl",
        source_sha256="0" * 64,
        payload=source,
        event2idx=event2idx,
        eos_token=event2idx["EOS_None"],
        pad_token=pad,
    )
    assert example.sequence_length == len(source[2])
    assert example.alignment.melody_bar_count == 8
    assert example.target_count == 8 * 8
    assert example.segment_ids == example.alignment.melody_condition_mask
    first_melody = example.tokens.index("Track_Melody")
    assert example.target_ids[first_melody] == event2idx["Bar_None"]
    assert example.target_ids[-1] == event2idx["EOS_None"]
    assert example.planner_example.target_active_ids[0] == 1


def test_dense_target_mask_is_the_original_v1_melody_track_mask() -> None:
    source = payload()
    event2idx = vocabulary(source)
    pad = len(event2idx)
    example = build_production_example_from_payload(
        dataset="fixture",
        split="train",
        source_id="fixture.pkl",
        source_sha256="0" * 64,
        payload=source,
        event2idx=event2idx,
        eos_token=event2idx["EOS_None"],
        pad_token=pad,
    )
    expected = [0] * len(source[2])
    # The production pickle stores chord-to-melody ranges first and
    # melody-to-next-chord ranges second.  V1 calls the second list
    # ``chord_pos`` internally, but uses it as the melody loss mask.
    for start, stop in source[1]:
        expected[start:stop] = [1] * (stop - start)
    observed = [int(target != pad) for target in example.target_ids]
    assert observed == expected


def test_rejects_first_eight_bars_beyond_frozen_length() -> None:
    source = payload()
    event2idx = vocabulary(source)
    try:
        build_production_example_from_payload(
            dataset="fixture",
            split="train",
            source_id="fixture.pkl",
            source_sha256="0" * 64,
            payload=source,
            event2idx=event2idx,
            eos_token=event2idx["EOS_None"],
            pad_token=len(event2idx),
            max_seq_len=20,
        )
    except ValueError as error:
        assert "exceed_20" in str(error)
    else:
        raise AssertionError("overlength dense example was accepted")


def test_eighth_bar_keeps_v1_track_transition_when_source_continues() -> None:
    source = payload()
    events = source[2]
    chord_start = len(events)
    events.extend(
        [
            event("Track", "Chord"),
            event("Bar", None),
            event("Beat", 0),
            event("Chord_Root", "I"),
            event("Chord_Quality", "M"),
        ]
    )
    source[0].append((chord_start, len(events)))
    melody_start = len(events)
    events.extend(
        [
            event("Track", "Melody"),
            event("Bar", None),
            event("Beat", 0),
            event("Note_Octave", 4),
            event("Note_Degree", "I"),
        ]
    )
    source[1].append((melody_start, len(events)))
    event2idx = vocabulary(source)
    example = build_production_example_from_payload(
        dataset="fixture",
        split="train",
        source_id="fixture.pkl",
        source_sha256="0" * 64,
        payload=source,
        event2idx=event2idx,
        eos_token=event2idx["EOS_None"],
        pad_token=len(event2idx),
    )
    assert example.alignment.melody_bar_count == 8
    assert example.target_ids[-1] == event2idx["Track_Chord"]


def test_eighth_bar_uses_eos_only_when_source_really_ends() -> None:
    source = payload()
    source[2].append(event("EOS", None))
    event2idx = vocabulary(source)
    example = build_production_example_from_payload(
        dataset="fixture",
        split="train",
        source_id="fixture.pkl",
        source_sha256="0" * 64,
        payload=source,
        event2idx=event2idx,
        eos_token=event2idx["EOS_None"],
        pad_token=len(event2idx),
    )
    assert example.target_ids[-1] == event2idx["EOS_None"]
