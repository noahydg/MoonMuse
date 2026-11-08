import ast
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import csv
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from stage2_event_validation import validate_events
from generation_budget import RejectionBudget, SegmentSamplingBudget, strip_terminal_eos
from scripts.launch_ss26_stage2_pad import Supervisor
from scripts.evaluate_ss26_stage2_melodies import describe
from scripts.ss26_stage2_runtime_checks import verify_generation
from scripts.partition_ss26_harmonic_families import partition, family


def score():
    return ['Emotion_Q1', 'Key_C'] + [t for _ in range(8) for t in (
        'Track_Chord', 'Bar_None', 'Beat_0', 'Chord_Root_I', 'Chord_Quality_M',
        'Track_Melody', 'Bar_None', 'Beat_0', 'Note_Octave_5', 'Note_Degree_I')]


@pytest.mark.parametrize('old,new', [('Beat_0', 'Beat_999'), ('Beat_0', 'Beat_x'),
    ('Chord_Quality_M', 'Chord_Quality_BAD'), ('Note_Degree_I', 'Note_Degree_BAD')])
def test_invalid_events_cannot_pass(old, new):
    tokens = [new if t == old else t for t in score()]
    assert not validate_events(tokens)['valid']
    assert describe(tokens)['invalid_tokens'] > 0


def test_unfinished_note_and_unverified_end_are_not_complete():
    assert validate_events(score()[:-1])['invalid_tokens'] > 0
    assert validate_events(score())['complete_bars'] is None
    assert validate_events(score())['reached_target_bar']
    assert validate_events(score(), generation={'stop_reason': 'eos', 'completed_bars': 8})['complete_bars']
    assert not validate_events(score(), generation={'stop_reason': 'event_limit', 'completed_bars': 7})['complete_bars']


def supervisor(tmp_path, total=30, reserve=15):
    called = tmp_path / 'shutdown-called'
    s = Supervisor(tmp_path/'out', tmp_path/'backup', tmp_path, total, reserve,
                   [sys.executable, '-c', f'from pathlib import Path; Path({str(called)!r}).write_text("called")'])
    return s, called


def py(code): return [sys.executable, '-c', code]


def test_archive_contains_final_exitcode_and_complete(tmp_path):
    s, called = supervisor(tmp_path)
    assert s.run(py('print("tests")'), py('print("campaign")'), dict(os.environ)) == 0
    assert called.exists()
    with tarfile.open(tmp_path/'backup/out.tar.gz') as f:
        names = f.getnames()
        assert {'out/COMPLETE', 'out/campaign.exitcode', 'out/campaign.log', 'out/pytest.log'} <= set(names)
        assert f.extractfile('out/campaign.exitcode').read().strip() == b'0'
    assert json.loads((tmp_path/'backup/out.receipt.json').read_text())['status'] == 'VERIFIED'


def test_failed_test_prevents_training_but_is_backed_up(tmp_path):
    s, called = supervisor(tmp_path)
    marker = tmp_path/'trained'
    assert s.run(py('raise SystemExit(5)'), py(f'open({str(marker)!r}, "w").close()'), dict(os.environ)) == 5
    assert not marker.exists() and called.exists()
    with tarfile.open(tmp_path/'backup/out.tar.gz') as f:
        assert 'out/INCOMPLETE' in f.getnames()
        assert f.extractfile('out/campaign.exitcode').read().strip() == b'5'


def test_backup_failure_still_shuts_down_and_preserves_outputs(tmp_path, monkeypatch):
    s, called = supervisor(tmp_path)
    original = s.command
    def fail_copy(command, *args, **kwargs):
        if command[0] == 'cp': raise OSError('simulated persistent storage failure')
        return original(command, *args, **kwargs)
    monkeypatch.setattr(s, 'command', fail_copy)
    assert s.run(py('pass'), py('pass'), dict(os.environ)) == 74
    assert called.exists() and (tmp_path/'out.tar.gz').is_file()
    assert (tmp_path/'out/backup_failure.txt').exists()
    assert json.loads((tmp_path/'out/progress.json').read_text())['status'] == 'NEEDS_ATTENTION'


def test_test_timeout_uses_campaign_budget_and_shuts_down(tmp_path):
    s, called = supervisor(tmp_path, total=12, reserve=11.8)
    assert s.run(py('import time; time.sleep(30)'), py('pass'), dict(os.environ)) == 124
    assert called.exists() and not (tmp_path/'out/campaign.log').exists()


def test_generation_stop_metadata_preserves_outputs_and_distinguishes_limits():
    path = Path(__file__).resolve().parents[1]/'stage2_melody/inference.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'generate_conditional')
    class Tensor:
        def __init__(self, x): self.x = x
        def long(self): return self
        def to(self, device): return self
        def cpu(self): return self
        def detach(self): return self
        def numpy(self): return np.array(self.x)
    names = ['Emotion_Q1', 'Key_C', 'Track_Chord', 'Bar_None', 'Track_Melody', 'Note_Octave_5', 'Note_Degree_I', 'EOS_None', 'PAD_None']
    enc = {t:i for i,t in enumerate(names)}; dec = dict(enumerate(names))
    class Model:
        def __init__(self, outputs): self.outputs = iter(outputs)
        def parameters(self): return iter([SimpleNamespace(device='cpu')])
        def __call__(self, *args, **kwargs): return [Tensor([enc[next(self.outputs)]])]
    env = dict(torch=SimpleNamespace(tensor=Tensor), RejectionBudget=RejectionBudget,
        SegmentSamplingBudget=SegmentSamplingBudget, strip_terminal_eos=strip_terminal_eos,
        time=time, max_dec_inp_len=2048, temperature=lambda x,*a,**k:x,
        nucleus=lambda x,*a:int(x[0]), monotonic_beat_inadmissibles=lambda *a,**k:[],
        get_position_idx=lambda x:int(x.split('_')[1]))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    fn = env['generate_conditional']
    for outputs, kwargs, reason in [
        (['Bar_None', 'Note_Octave_5', 'Note_Degree_I', 'EOS_None'], {}, 'eos'),
        (['Bar_None', 'Track_Chord'], {}, 'target_bars'),
        (['PAD_None']*3, {'max_segment_attempts': 2}, 'segment_attempt_limit'),
        (['Bar_None', 'Note_Octave_5'], {'max_events': 5}, 'event_limit'),
    ]:
        args = (enc, dec, [[enc['Bar_None']]], [enc['Emotion_Q1'], enc['Key_C']])
        meta = {}
        a = fn(Model(outputs), *args, **kwargs)
        b = fn(Model(outputs), *args, diagnostics=meta, **kwargs)
        assert a == b and meta['stop_reason'] == reason


def test_full_generation_inventory_includes_failures_and_rejects_missing_or_changed_files(tmp_path):
    plan_root=tmp_path/'plans';plan_root.mkdir();out=tmp_path/'run';out.mkdir()
    plans=[]
    for seed in (42,43,44):
        for valence in ('Positive','Negative'):
            for index in range(25):
                key='Key_C' if valence=='Positive' else 'Key_c'
                name=f'seed{seed}_{valence}_{index}.txt'
                tokens=[key]+['Bar_None','Beat_0','Chord_Root_I','Chord_Quality_M']*8+['Bar_None']
                (plan_root/name).write_text('\n'.join(tokens)+'\n')
                plans.append(dict(seed=seed,valence=valence,key=key,plan_path=name))
                for mode in ('ignore','count'):
                    for target in (('Q1','Q4') if valence=='Positive' else ('Q2','Q3')):
                        p=out/f'melodies/{mode}/seed{seed}/{Path(name).stem}_{target}_melody.txt'
                        p.parent.mkdir(parents=True,exist_ok=True)
                        ev=score();ev[0]='Emotion_'+target;ev[1]=key
                        p.write_text('\n'.join(ev)+'\n')
                        meta=dict(input_file=name,target=target,seed=seed,events_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            stop_reason='event_limit' if mode=='count' else 'eos',completed_bars=7 if mode=='count' else 8)
                        p.with_name(p.name+'.generation.json').write_text(json.dumps(meta))
        for mode in ('ignore','count'):
            p=out/f'checkpoints/seed{seed}/{mode}/pretrain/training_objective.json'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(initial_state_sha256='same-'+str(seed))))
    with (plan_root/'plan_manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(plans[0]));w.writeheader();w.writerows(plans)
    verify_generation(plan_root,out)
    report=json.loads((out/'generation_integrity.json').read_text())
    assert report['files']==600
    assert all(x['completed']==(100 if x['mode']=='ignore' else 0) for x in report['groups'])
    victim=next((out/'melodies').rglob('*_melody.txt'))
    original=victim.read_text()
    sidecar=victim.with_name(victim.name+'.generation.json')
    original_meta=sidecar.read_text()
    victim.write_text(original+'Key_G#\n')
    meta=json.loads(original_meta);meta['events_sha256']=hashlib.sha256(victim.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(AssertionError,match='key contract failures'):verify_generation(plan_root,out)
    report=json.loads((out/'generation_integrity.json').read_text())
    assert report['status']=='FAIL_COLLECTION' and len(report['records'])==600
    assert report['key_contract_failures']==1
    assert verify_generation(plan_root,out,strict_keys=False)['status']=='FAIL_COLLECTION'
    sidecar.write_text(original_meta)
    victim.write_text(original+'BAD\n')
    with pytest.raises(AssertionError):verify_generation(plan_root,out)
    victim.unlink()
    with pytest.raises(FileNotFoundError):verify_generation(plan_root,out)


def test_plan_partition_keeps_entire_calibration_families_out_and_one_melody_per_plan():
    history=[];roles=[]
    for seed in ('42','43','44'):
        for valence,quadrants in [('positive',('Q1','Q4')),('negative',('Q2','Q3'))]:
            for j in range(15):
                for q in quadrants:
                    row=dict(seed=seed,valence=valence,sample_index=str(j),quadrant=q,history_class='H0',
                        ranking_digest=f'{j:02d}_{q}',source_sha256=f'{seed}_{q}_{j}')
                    history.append(row)
                    roles.append(dict(row,role='calibration' if j==0 and q==quadrants[0] else 'held_out_candidate'))
    selected,strict,summary=partition(history,roles)
    assert len(selected)==len({family(r) for r in selected})==66
    assert all(r['sample_index']!='0' for r in selected)
    assert summary['cross_calibration_plans']==0
    assert {family(r) for r in selected if r['pool']=='system'}.isdisjoint({family(r) for r in selected if r['pool']=='within'})
