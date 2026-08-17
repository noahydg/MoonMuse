import math

import pytest

from stage2_melody_v2.operation_aware_containment import contiguous_windows
from stage2_melody_v2.strict_monotonic_alignment import strict_monotonic_alignment


def source_view():
    return {
        "relative_pitches": [0, 7, -5, 12, -8, 4],
        "relative_positions": [0, 3, 7, 11, 15, 20],
    }


def test_exact_suffix_window_has_zero_cost_and_unique_ordered_path():
    suffix = contiguous_windows(source_view(), 4)[-1]
    result = strict_monotonic_alignment(suffix, suffix)
    assert result is not None
    assert result.total_cost == 0.0
    assert result.similarity == 1.0
    assert result.matched_pairs == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert result.skipped_source_indices == ()


def test_deleted_query_maps_around_one_source_onset():
    source = {
        "relative_pitches": [0, 12, -12, 18, 2],
        "relative_positions": [0, 4, 8, 12, 16],
    }
    query = {
        "relative_pitches": [0, 12, 18, 2],
        "relative_positions": [0, 4, 12, 16],
    }
    result = strict_monotonic_alignment(query, source)
    assert result is not None
    assert result.matched_pairs == ((0, 0), (1, 1), (2, 3), (3, 4))
    assert result.skipped_source_indices == (2,)
    assert math.isclose(result.total_cost, 0.25)


def test_local_pitch_and_timing_edit_remains_alignable_with_positive_cost():
    source = {
        "relative_pitches": [0, 4, 7, 11, 14],
        "relative_positions": [0, 4, 8, 12, 16],
    }
    query = {
        "relative_pitches": [0, 4, 8, 11, 14],
        "relative_positions": [0, 4, 9, 12, 16],
    }
    result = strict_monotonic_alignment(query, source)
    assert result is not None
    assert result.total_cost > 0.0
    assert 0.0 < result.similarity < 1.0


def test_reordered_pitch_tail_scores_below_local_edit():
    source = {
        "relative_pitches": [0, 4, 7, 11, 14],
        "relative_positions": [0, 4, 8, 12, 16],
    }
    local = {
        "relative_pitches": [0, 4, 8, 11, 14],
        "relative_positions": [0, 4, 8, 12, 16],
    }
    reordered = {
        "relative_pitches": [0, 11, 4, 14, 7],
        "relative_positions": [0, 4, 8, 12, 16],
    }
    local_result = strict_monotonic_alignment(local, source)
    reordered_result = strict_monotonic_alignment(reordered, source)
    assert local_result is not None and reordered_result is not None
    assert reordered_result.similarity < local_result.similarity


def test_longer_query_has_no_alignment():
    query = source_view()
    short_source = contiguous_windows(source_view(), 4)[0]
    assert strict_monotonic_alignment(query, short_source) is None


def test_malformed_views_fail_through_existing_contract():
    malformed = {
        "relative_pitches": [0, 2, 4],
        "relative_positions": [0, 2, 2],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        strict_monotonic_alignment(malformed, malformed)
