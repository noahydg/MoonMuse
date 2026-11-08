import json
from pathlib import Path
import pytest
from scripts.run_ss26_new_score_performance import (
    audit_event, campaign, seal, smoke_sources, verify_manifest, loader_dependencies,
)


def tokens():
    bar=['Bar_None','Beat_0','Note_Octave_5','Note_Degree_I']
    return ['Key_c']+[t for _ in range(8) for t in ['Track_Melody']+bar]


def performance():
    bar=['Bar_None','Beat_0','Note_Octave_5','Note_Degree_I']
    result=['Key_c']
    for _ in range(8):
        result+=['Track_Melody']+bar+['Track_Performance']+bar+['Note_Duration_120','Note_Velocity_60']
    return result+['Track_Melody']


@pytest.mark.parametrize('defect',['key','pitch','attribute','bar'])
def test_audit_rejects_changed_key_pitch_missing_attributes_or_bar(tmp_path,defect):
    source=tmp_path/'source';out=tmp_path/'out'
    source.write_text('\n'.join(tokens()))
    events=performance()
    out.write_text('\n'.join(events));assert audit_event(source,out)['complete_bars']==8
    if defect=='key': events[0]='Key_C'
    if defect=='pitch': events[events.index('Track_Performance')+4]='Note_Degree_V'
    if defect=='attribute': events.remove('Note_Duration_120')
    if defect=='bar': events=events[:-12]
    out.write_text('\n'.join(events))
    with pytest.raises(ValueError): audit_event(source,out)


def test_smoke_covers_each_seed_valence_and_hardest_length():
    rows=[dict(seed=s,valence=v,planned_stage3_events=n,source_sha256=f'{s}{v}{n}')
          for s in (42,43,44) for v in ('positive','negative') for n in (400,2200)]
    chosen=smoke_sources(rows)
    assert len(chosen)==6 and all(r['planned_stage3_events']==2200 for r in chosen)
    assert {(r['seed'],r['valence']) for r in chosen}=={(s,v) for s in (42,43,44) for v in ('positive','negative')}


def test_manifest_rejects_extra_or_changed_inputs(tmp_path):
    file=tmp_path/'score';file.write_text('a');seal(tmp_path);verify_manifest(tmp_path)
    file.write_text('b')
    with pytest.raises(ValueError): verify_manifest(tmp_path)
    file.write_text('a');(tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError): verify_manifest(tmp_path)


def test_smoke_failure_prevents_full_generation(tmp_path,monkeypatch):
    from scripts import run_ss26_new_score_performance as module
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'code_assets.json').write_text('{}')
    (inputs/'sources.json').write_text('[]')
    out=tmp_path/'output';out.mkdir();(out/'progress.json').write_text('{}')
    monkeypatch.setattr(module,'preflight',lambda *a,**kw: {'status':'PASS'})
    monkeypatch.setattr(module,'smoke_sources',lambda r: [])
    monkeypatch.setattr(module,'subset',lambda *a: None)
    calls=[]
    def fail(*args):
        calls.append(args[-1].name)
        raise RuntimeError('GPU smoke failed')
    monkeypatch.setattr(module,'generate',fail)
    with pytest.raises(RuntimeError,match='GPU smoke failed'): campaign(inputs,tmp_path/'models',out)
    assert calls==['smoke'] and not (out/'performances').exists()


def test_inference_startup_checks_split_and_every_referenced_piece(tmp_path):
    import pickle,yaml
    config=tmp_path/'stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml'
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({'data_loader':{'val_split':'valid.pkl','vocab_path':'vocab.pkl','data_path':'events'}}))
    with pytest.raises(FileNotFoundError): loader_dependencies(tmp_path)
    (tmp_path/'valid.pkl').write_bytes(pickle.dumps(['a.pkl','b.pkl']))
    (tmp_path/'vocab.pkl').write_bytes(pickle.dumps({}))
    (tmp_path/'events').mkdir();(tmp_path/'events/a.pkl').write_bytes(b'a')
    with pytest.raises(FileNotFoundError,match='b.pkl'): loader_dependencies(tmp_path)
    (tmp_path/'events/b.pkl').write_bytes(b'b')
    assert len(loader_dependencies(tmp_path))==4
