from stage2_melody_v2.motif_memory import (
    MelodyOnset,
    build_cell_motif,
    compare_motifs,
    parse_melody_onsets,
    sequence_similarity,
)
from stage2_melody_v2.pipeline import build_architecture_sidecar
from stage2_melody_v2.planner import build_eight_bar_plan
from stage2_melody_v2.schema import MotifRelation


def fixture_events():
    events = [
        {"name": "Emotion", "value": "Q1"},
        {"name": "Key", "value": "C"},
    ]
    degrees = ["I", "II", "III", "II"]
    for bar in range(8):
        events.extend(
            [
                {"name": "Track", "value": "Chord"},
                {"name": "Bar", "value": None},
                {"name": "Beat", "value": 0},
                {"name": "Chord_Root", "value": "I"},
                {"name": "Chord_Quality", "value": "M"},
                {"name": "Track", "value": "Melody"},
                {"name": "Bar", "value": None},
            ]
        )
        for beat, degree in zip((0, 4, 8, 12), degrees):
            events.extend(
                [
                    {"name": "Beat", "value": beat},
                    {"name": "Note_Octave", "value": 5},
                    {"name": "Note_Degree", "value": degree},
                ]
            )
    events.append({"name": "EOS", "value": None})
    return events


def test_eight_bar_plan_covers_four_cells_and_is_deterministic():
    plan = build_eight_bar_plan("Q1")
    assert [slot.label for slot in plan.slots] == ["A", "A_PRIME", "B", "A_CAD"]
    assert plan.slots[-1].relation is MotifRelation.CADENCE_RETURN
    assert plan.fingerprint() == build_eight_bar_plan("Q1").fingerprint()
    assert len(plan.control_tokens()) == 4


def test_parser_reads_only_melody_bars_and_keeps_monophony():
    key, onsets, bar_count = parse_melody_onsets(fixture_events())
    assert key == "C"
    assert bar_count == 8
    assert len(onsets) == 32
    assert onsets[-1].bar == 7
    assert onsets[-1].beat == 12


def test_motif_fingerprint_and_relation_are_transposition_invariant():
    onsets = [
        MelodyOnset(0, 0, 60, "I"),
        MelodyOnset(0, 4, 62, "II"),
        MelodyOnset(0, 8, 64, "III"),
        MelodyOnset(1, 0, 62, "II"),
    ]
    source = build_cell_motif(onsets, slot_index=0, start_bar=0, end_bar=2)
    target = build_cell_motif(onsets, slot_index=1, start_bar=0, end_bar=2)
    shifted_source = source.transposed(5)
    shifted_target = target.transposed(5)
    assert source.fingerprint() == shifted_source.fingerprint()
    assert compare_motifs(source, target).to_dict() == compare_motifs(
        shifted_source, shifted_target
    ).to_dict()
    assert compare_motifs(source, target).relation is MotifRelation.RETURN


def test_sequence_similarity_handles_insertions_without_length_bias_crash():
    assert abs(
        sequence_similarity(["U1", "D1"], ["U1", "S0", "D1"]) - 2 / 3
    ) < 1e-12
    assert sequence_similarity([], []) == 1.0


def test_sidecar_connects_plan_and_observed_motif_memory():
    sidecar = build_architecture_sidecar(
        fixture_events(), source_id="fixture", emotion_quadrant="Q1"
    )
    assert sidecar["available_melody_bars"] == 8
    assert sidecar["first8_onset_count"] == 32
    assert len(sidecar["observed_cells"]) == 4
    assert len(sidecar["observed_relations_to_A"]) == 3
    assert all(
        item["relation"] == "RETURN"
        for item in sidecar["observed_relations_to_A"]
    )
