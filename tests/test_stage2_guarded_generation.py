"""Exercise the production sampler with controlled model logits, without Torch."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from generation_budget import (RejectionBudget, SegmentSamplingBudget,
                               monotonic_beat_inadmissibles, strip_terminal_eos)
from stage2_event_validation import validate_events
from stage2_sampling_constraints import FunctionalMelodyState, pair_inadmissibles


EVENTS = ['Emotion_Q1', 'Key_C', 'Track_Chord', 'Track_Melody', 'Bar_None',
          'Beat_0', 'Beat_8', 'Note_Octave_5', 'Note_Octave_6', 'Note_Degree_I',
          'Note_Degree_III', 'Chord_Root_I', 'Chord_Root_Conti', 'Chord_Root_None',
          'Chord_Quality_M', 'Chord_Quality_Conti', 'Chord_Quality_None',
          'EOS_None', 'PAD_None']
ENC = {t: i for i, t in enumerate(EVENTS)}
DEC = dict(enumerate(EVENTS))
CHORD = [ENC[t] for t in ['Bar_None', 'Beat_0', 'Chord_Root_I', 'Chord_Quality_M']]


class Tensor:
    def __init__(self, x): self.x = x
    def long(self): return self
    def to(self, device): return self
    def cpu(self): return self
    def detach(self): return self
    def numpy(self): return np.array(self.x)


class Model:
    def __init__(self, outputs): self.outputs = iter(outputs)
    def parameters(self): return iter([SimpleNamespace(device='cpu')])
    def __call__(self, *args, **kwargs):
        spec = next(self.outputs)
        if isinstance(spec, str): spec = {spec: 20.0}
        logits = np.full(len(ENC), -np.inf)
        for token, score in spec.items(): logits[ENC[token]] = score
        return [Tensor(logits)]


@pytest.fixture
def sampler():
    path = Path(__file__).resolve().parents[1] / 'stage2_melody/inference.py'
    names = {'generate_conditional', 'temperature', 'nucleus'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(torch=SimpleNamespace(tensor=Tensor), np=np, time=time,
               RejectionBudget=RejectionBudget, SegmentSamplingBudget=SegmentSamplingBudget,
               strip_terminal_eos=strip_terminal_eos, max_dec_inp_len=2048,
               monotonic_beat_inadmissibles=monotonic_beat_inadmissibles,
               pair_inadmissibles=pair_inadmissibles,
               FunctionalMelodyState=FunctionalMelodyState,
               get_position_idx=lambda x: int(x.split('_')[1]))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    def run(outputs, bars=1, **kwargs):
        meta = {}
        result = env['generate_conditional'](
            Model(outputs), ENC, DEC, [CHORD] * bars, [ENC['Emotion_Q1'], ENC['Key_C']],
            top_p=0.000001, temp=1.0, diagnostics=meta, **kwargs)
        return [DEC[i] for i in result], meta
    return run


def test_pair_guard_rejects_interior_note_breaks_and_orphan_chord_quality(sampler):
    tokens, meta = sampler([
        'Bar_None', 'Beat_0', 'Note_Octave_5',
        {'Beat_8': 30, 'Note_Octave_6': 29, 'Note_Degree_III': 10},
        {'Chord_Quality_M': 30, 'Chord_Root_Conti': 10},
        {'Chord_Quality_M': 30, 'Chord_Quality_Conti': 10}, 'EOS_None',
    ], guarded_generation=True)
    assert tokens[-4:] == ['Note_Octave_5', 'Note_Degree_III',
                          'Chord_Root_Conti', 'Chord_Quality_Conti']
    assert validate_events(tokens, bars=1, generation=meta)['complete_bars']


@pytest.mark.parametrize('root, allowed', [
    ('Chord_Root_I', 'Chord_Quality_M'),
    ('Chord_Root_Conti', 'Chord_Quality_Conti'),
    ('Chord_Root_None', 'Chord_Quality_None'),
])
def test_root_quality_contract_and_caller_mask(root, allowed):
    blocked = pair_inadmissibles(ENC, root, 1)
    assert set(ENC) - {DEC[x] for x in blocked} == {allowed}
    blocked = pair_inadmissibles(ENC, root, 1, base=[ENC[allowed]])
    assert len(blocked) == len(ENC)


def test_total_cap_does_not_cut_a_pair_or_overshoot(sampler):
    tokens, meta = sampler(['Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I'],
                          guarded_generation=True, max_events=12)
    assert len(tokens) == meta['accepted_events'] == 12
    check = validate_events(tokens, bars=1, generation=meta)
    assert meta['stop_reason'] == 'event_limit' and check['valid']
    assert not check['complete_bars']


def test_last_slot_cannot_start_a_new_pair(sampler):
    tokens, meta = sampler(['Bar_None', 'Beat_0',
                           {'Note_Octave_5': 30, 'EOS_None': 10}],
                          guarded_generation=True, max_events=11)
    assert meta['stop_reason'] == 'eos'
    assert validate_events(tokens, bars=1, generation=meta)['complete_bars']


def test_forced_harmony_is_not_partially_inserted_past_cap(sampler):
    tokens, meta = sampler(['Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I',
                           'Track_Chord'], bars=2, guarded_generation=True, max_events=14)
    assert len(tokens) == 13 and tokens[-1] == 'Track_Chord'
    assert meta['completed_bars'] == 1 and meta['stop_reason'] == 'event_limit'
    assert validate_events(tokens, bars=2, generation=meta)['valid']


def test_attempt_budget_reserves_pair_completion(sampler):
    tokens, meta = sampler(['Bar_None', {'Note_Octave_5': 30, 'Beat_0': 10}],
                          guarded_generation=True, max_segment_attempts=2)
    assert tokens[-1] == 'Beat_0' and meta['stop_reason'] == 'segment_attempt_limit'
    assert validate_events(tokens, bars=1, generation=meta)['valid']


def test_conflicting_mask_stops_without_invalid_sampling(sampler):
    tokens, meta = sampler(['Bar_None', 'Note_Octave_5', 'Note_Degree_I'],
                          guarded_generation=True,
                          inadmissibles=[ENC['Note_Degree_I'], ENC['Note_Degree_III']])
    assert meta['stop_reason'] == 'constraint_dead_end'
    assert not validate_events(tokens, bars=1, generation=meta)['complete_bars']


def test_legacy_default_keeps_original_truncation_behavior(sampler):
    outputs = ['Bar_None', 'Note_Octave_5']
    tokens, meta = sampler(outputs, max_events=9)
    explicit, _ = sampler(outputs, max_events=9, guarded_generation=False)
    assert tokens == explicit and len(tokens) == 10
    assert tokens[-1] == 'Note_Octave_5' and meta['sampling_policy'] == 'legacy'


def test_initial_prefix_cannot_exceed_guarded_budget(sampler):
    with pytest.raises(ValueError, match='initial conditioning prefix'):
        sampler([], guarded_generation=True, max_events=7)


def test_structured_blocks_interior_controls_and_early_eos_before_sampling(sampler):
    tokens, meta = sampler([
        {'Key_C': 40, 'Emotion_Q1': 39, 'PAD_None': 38, 'Track_Melody': 37, 'Bar_None': 10},
        {'EOS_None': 40, 'Bar_None': 39, 'Beat_0': 10},
        'Note_Octave_5', 'Note_Degree_I', 'Track_Chord',
        'Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_III',
        {'Key_C': 40, 'PAD_None': 39, 'EOS_None': 10},
    ], bars=2, structured_generation=True)
    assert meta['sampling_policy'] == 'structured-v2'
    assert tokens.count('Key_C') == tokens.count('Emotion_Q1') == 1
    assert meta['skipped_pad'] == meta['skipped_early_eos'] == 0
    assert validate_events(tokens, bars=2, generation=meta)['complete_bars']


def test_structured_repeated_pitch_is_blocked_until_onset_advances(sampler):
    tokens, meta = sampler([
        'Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I',
        'Beat_0', 'Note_Octave_5', {'Note_Degree_I': 40, 'Note_Degree_III': 10},
        'Beat_8', 'Note_Octave_5', 'Note_Degree_I', 'EOS_None',
    ], structured_generation=True)
    assert tokens.count('Note_Degree_I') == 2
    assert tokens.count('Note_Degree_III') == 1
    assert validate_events(tokens, bars=1, generation=meta)['complete_bars']


def test_structured_does_not_enter_exhausted_octave_pair(sampler):
    tokens, meta = sampler([
        'Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I',
        'Note_Octave_5', 'Note_Degree_III',
        {'Note_Octave_5': 40, 'Beat_8': 10}, 'EOS_None',
    ], structured_generation=True)
    assert tokens[-1] == 'Beat_8'
    assert validate_events(tokens, bars=1, generation=meta)['complete_bars']


def test_structured_still_reports_attempt_limit_without_forcing_completion(sampler):
    tokens, meta = sampler(['Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I'],
                          structured_generation=True, max_segment_attempts=4)
    assert meta['sampled_attempts'] == meta['final_segment_attempts'] == 4
    assert meta['stop_reason'] == 'segment_attempt_limit'
    assert validate_events(tokens, bars=1, generation=meta)['valid']
    assert not validate_events(tokens, bars=1, generation=meta)['complete_bars']


def test_legacy_records_skipped_tokens_without_changing_sampling(sampler):
    tokens, meta = sampler(['Bar_None', 'PAD_None', 'EOS_None'],
                          bars=2, max_segment_attempts=3)
    assert meta['skipped_pad'] == meta['skipped_early_eos'] == 1
    assert meta['sampled_attempts'] == 3 and tokens[-1] == 'Bar_None'
