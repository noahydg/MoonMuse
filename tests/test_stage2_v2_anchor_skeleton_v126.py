from stage2_melody_v2.anchor_skeleton import (
    ANCHOR_SLOT_COUNT,
    extract_anchor_skeleton,
)


def piece_tokens(*, key: str = "C") -> list[str]:
    output = ["Emotion_Q1", f"Key_{key}"]
    for bar in range(8):
        root = "I" if bar % 2 == 0 else "V"
        output.extend(
            [
                "Track_Chord",
                "Bar_None",
                "Beat_0",
                f"Chord_Root_{root}",
                "Chord_Quality_M",
                "Beat_4",
                "Chord_Root_Conti",
                "Chord_Quality_Conti",
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_3",
                f"Note_Degree_{root}",
                "Note_Octave_5",
                f"Note_Degree_{root}",
                "Beat_8",
                "Note_Octave_4",
                f"Note_Degree_{root}",
            ]
        )
    return output


def test_extracts_exactly_32_active_or_rest_slots() -> None:
    skeleton = extract_anchor_skeleton(piece_tokens())
    assert len(skeleton.slots) == ANCHOR_SLOT_COUNT
    assert skeleton.active_count == 16
    assert skeleton.coverage == 0.5
    assert skeleton.cell_active_counts == (4, 4, 4, 4)


def test_polyphonic_anchor_chooses_highest_note() -> None:
    skeleton = extract_anchor_skeleton(piece_tokens())
    first = skeleton.slots[0]
    assert first.active is True
    assert first.octave == 5
    assert first.onset_polyphony == 2
    assert first.chord_tone is True


def test_missing_quarter_onset_is_explicit_rest() -> None:
    skeleton = extract_anchor_skeleton(piece_tokens())
    rest = skeleton.slots[1]
    assert rest.beat == 4
    assert rest.active is False
    assert rest.degree is None
    assert rest.chord_root == "I"


def test_fingerprint_is_deterministic_and_content_sensitive() -> None:
    left = extract_anchor_skeleton(piece_tokens())
    right = extract_anchor_skeleton(piece_tokens())
    changed_tokens = piece_tokens()
    first_high = changed_tokens.index("Note_Octave_5")
    changed_tokens[first_high] = "Note_Octave_6"
    changed = extract_anchor_skeleton(changed_tokens)
    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != changed.fingerprint()


def test_minor_ambiguous_degree_is_exposed_not_hidden() -> None:
    tokens = piece_tokens(key="c")
    first_high_octave = tokens.index("Note_Octave_5")
    tokens[first_high_octave + 1] = "Note_Degree_II#"
    skeleton = extract_anchor_skeleton(tokens)
    assert skeleton.slots[0].ambiguous_minor_degree is True
    assert skeleton.slots[0].chord_tone is None
