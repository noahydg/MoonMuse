from stage2_melody_v2.conditioning import encode_realizer_condition
from stage2_melody_v2.motif_memory import extract_cell_motifs, parse_melody_onsets
from stage2_melody_v2.planner import build_eight_bar_plan
from tests.test_stage2_melody_v2 import fixture_events


def test_realizer_condition_maps_eight_bars_to_four_cells():
    _, onsets, _ = parse_melody_onsets(fixture_events())
    condition = encode_realizer_condition(
        build_eight_bar_plan("Q1"), extract_cell_motifs(onsets)
    )
    assert condition.bar_to_slot_ids == (0, 0, 1, 1, 2, 2, 3, 3)
    assert condition.source_slot_ids == (-1, 0, 0, 0)
    assert len(condition.motif_interval_ids) == 4
    condition.validate()
