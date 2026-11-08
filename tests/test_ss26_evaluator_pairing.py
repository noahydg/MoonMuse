"""Guard the scientific estimands and missingness errors found in the SS26 audit."""
import math
from types import SimpleNamespace

import pytest

from scripts.analyze_ss26_new_score_cues import tempo_sd
from scripts.pair_ss26_listeners_with_classifier import (
    human_point_summary, describe, source_gaps, spearman, valence_target,
)


@pytest.mark.parametrize('quadrant, expected', [
    ('Q1', 'positive'), ('Q2', 'negative'), ('Q3', 'negative'), ('Q4', 'positive'),
])
def test_valence_is_not_arousal(quadrant, expected):
    assert valence_target(quadrant) == expected


def test_equal_valence_weighting_with_unequal_source_counts():
    result = describe([1, 1, -1], strata=['positive', 'positive', 'negative'], resample=False)
    assert result['mean'] == 0
    assert result['unweighted_source_mean'] == pytest.approx(1/3)
    assert not any('interval' in k or 'ci95' in k for k in result)


def test_missing_stratum_is_not_a_full_population_estimate():
    assert describe([1], strata=['positive'], resample=False)['mean'] is None


def test_missing_arousal_does_not_become_zero_or_a_complete_pair():
    rows = [dict(source_sha256='s', method='m', target_quadrant=q, human_arousal=value)
            for q, value in [('Q1', 5), ('Q4', None)]]
    assert source_gaps(rows, 'human_arousal') == {}


def test_human_summary_does_not_report_source_only_uncertainty():
    result = human_point_summary([{'source_sha256':'s'}], lambda rows:0.5)
    assert result['estimate'] == 0.5
    assert 'ci95_low' not in result and 'ci95_high' not in result


def test_constant_ratings_return_missing_correlation():
    assert spearman([3, 3, 3], [1, 2, 3]) is None
    assert spearman([1, 2, 2, 3], [4, 3, 3, 2]) == pytest.approx(-1)


def test_tempo_sd_weights_elapsed_seconds_not_score_ticks():
    changes = [SimpleNamespace(time=0, tempo=60), SimpleNamespace(time=480, tempo=120)]
    # One beat at 60 BPM lasts twice as long as one beat at 120 BPM.
    assert tempo_sd(changes, 960) == pytest.approx(math.sqrt(800))
    assert tempo_sd([SimpleNamespace(time=0, tempo=90)], 960) == 0
