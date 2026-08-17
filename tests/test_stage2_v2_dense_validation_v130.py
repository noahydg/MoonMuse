from stage2_melody_v2.dense_validation import validate_dense_generated_tokens


def dense_piece(*, incomplete_pitch: bool = False) -> list[str]:
    tokens = ["Emotion_Q1", "Key_C"]
    for bar in range(8):
        tokens.extend(
            [
                "Track_Chord",
                "Bar_None",
                "Beat_0",
                "Chord_Root_I",
                "Chord_Quality_M",
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_4",
                "Note_Degree_I",
                "Note_Octave_5",
                "Note_Degree_III",
                "Beat_8",
                "Note_Octave_4",
                "Note_Degree_V",
            ]
        )
    if incomplete_pitch:
        tokens[-1] = "Note_Octave_6"
    return tokens


def test_dense_validator_accepts_valid_polyphonic_onsets():
    report = validate_dense_generated_tokens(dense_piece())
    assert report["passed"] is True
    assert report["bars"] == 8
    assert report["first8_onset_count"] == 16
    assert report["cell_onset_counts"] == [4, 4, 4, 4]
    assert report["note_count"] == 24
    assert report["polyphonic_onset_count"] == 8
    assert report["density_reference_pass"] is True


def test_sparse_cell_is_retained_but_reported_as_density_warning():
    tokens = dense_piece()
    first_melody = tokens.index("Track_Melody")
    second_chord = tokens.index("Track_Chord", first_melody)
    # Keep only one onset in the first Melody bar.  The stream is structurally
    # valid but no longer meets the V97 motif-label density reference.
    del tokens[first_melody + 5 : second_chord]
    report = validate_dense_generated_tokens(tokens)
    assert report["passed"] is True
    assert report["parse_error"] == ""
    assert report["density_reference_pass"] is False


def test_dense_validator_rejects_incomplete_pitch_pair():
    report = validate_dense_generated_tokens(dense_piece(incomplete_pitch=True))
    assert report["passed"] is False
    assert "Note_Octave" in report["parse_error"]
    assert report["density_reference_pass"] is False


def test_dense_validator_rejects_wrong_track_order():
    tokens = dense_piece()
    tokens[tokens.index("Track_Melody")] = "Track_Chord"
    report = validate_dense_generated_tokens(tokens)
    assert report["passed"] is False
    assert "alternate" in report["parse_error"]
