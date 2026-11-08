from types import SimpleNamespace
import pytest
from scripts.analyze_ss26_comparison import score_seconds, tempo_outside_fraction, paired_gaps, contrast_gaps


def test_tempo_uses_elapsed_time_and_last_event_at_same_tick():
    changes=[SimpleNamespace(time=0,tempo=80),SimpleNamespace(time=0,tempo=60),SimpleNamespace(time=480,tempo=120)]
    assert score_seconds(changes,960,480)==1.5


def test_missing_tempo_uses_midi_default_and_can_violate_target_bounds():
    assert score_seconds([],960,480)==1
    assert tempo_outside_fraction([],960,80,119)==1
    assert tempo_outside_fraction([SimpleNamespace(time=480,tempo=100)],960,80,119)==.5


def test_pairing_uses_identity_not_row_order_and_preserves_difference_in_differences():
    rows=[]
    for source in ('s1','s2'):
        for method,gap in [('R-det',10),('Aplus-bounded',15),('A0-bounded',13),('Aplus-unbounded',8),('A0-unbounded',7)]:
            for q,x in [('Q1',20+gap),('Q4',20)]:
                rows.append(dict(source_sha256=source,method=method,seed=42,valence='positive',target_quadrant=q,x=x))
    gaps=paired_gaps(list(reversed(rows)),('x',))
    contrasts=contrast_gaps(gaps,('x',))
    assert [r['x'] for r in contrasts if r['contrast']=='learned_minus_rule']==[5,5]
    with pytest.raises(ValueError,match='Unpaired'):paired_gaps(rows[1:],('x',))
