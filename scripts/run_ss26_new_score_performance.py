"""Bounded inference-only preparation on new scores; no model training."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from performance_skeleton import FORCED, build_performance_bar_plan
from source_key import declared_source_key
from scripts.render_rule_performance_ss26 import track_bars
from scripts.audit_ss26_outputs import check_midi
from scripts.run_ss26_comparison import run as generate
from scripts.launch_ss26_stage2_pad import Supervisor

RUN = 'ss26-new-score-performance-20261003-v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def seal(folder):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    (folder/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(folder)}\n' for p in files))


def verify_manifest(folder):
    expected = set()
    for line in (folder/'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError(f'Input identity mismatch: {name}')
        expected.add(name)
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'}
    if actual != expected: raise ValueError('Unlisted or missing input files')


def smoke_sources(rows):
    # Cover all three model seeds and both valences, including the longest score
    # in each cell. These are technical checks, never quality-based selection.
    return [max((r for r in rows if int(r['seed']) == s and r['valence'] == v),
                key=lambda r: (int(r['planned_stage3_events']), r['source_sha256']))
            for s in (42,43,44) for v in ('positive','negative')]


def loader_dependencies(repo):
    """The unchanged inference entry point constructs its validation dataset."""
    import yaml
    config=yaml.safe_load((repo/'stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml').read_text())
    loader=config['data_loader']
    split=repo/loader['val_split']
    with split.open('rb') as f: pieces=pickle.load(f)
    if not pieces or len(pieces)!=len(set(pieces)): raise ValueError('Invalid validation split')
    base=repo/loader['data_path'].format('functional')
    files=[split,repo/loader['vocab_path'].format('functional')]+[base/name for name in pieces]
    for path in files:
        if not path.is_file(): raise FileNotFoundError(f'Inference startup dependency: {path}')
    return files


def preflight(inputs, models, gpu=False):
    verify_manifest(inputs)
    plan = json.loads((inputs/'plan.json').read_text())
    rows = json.loads((inputs/'sources.json').read_text())
    assets = json.loads((inputs/'code_assets.json').read_text())
    for name, digest in assets.items():
        if sha(REPO/name) != digest: raise ValueError(f'Code/asset changed: {name}')
    startup_files=loader_dependencies(REPO)
    if any(str(p.relative_to(REPO)) not in assets for p in startup_files):
        raise ValueError('Unmanifested inference startup dependency')
    if len(rows) != 68 or len({r['plan_id'] for r in rows}) != 68:
        raise ValueError('Expected 68 distinct candidate plans')
    if (plan['branches'],plan['bounds'],plan['expected_outputs'],plan['max_events'],plan['max_bars'],
        plan['temperature'],plan['top_p'],plan['paired_condition_seed']) != (['Aplus'],['bounded'],136,3072,8,1.0,.97,True):
        raise ValueError('Unexpected inference scope/settings')
    for r in rows:
        if sha(inputs/r['frozen_path']) != r['source_sha256'] or sha(inputs/r['condition_path']) != r['condition_sha256']:
            raise ValueError('Score changed')
        tokens = (inputs/r['frozen_path']).read_text().splitlines()
        condition = (inputs/r['condition_path']).read_text().splitlines()
        if declared_source_key(tokens) != declared_source_key(condition): raise ValueError('Key changed')
        if len(track_bars(tokens,'Melody')) != 8 or condition.count('Bar_None') != 8:
            raise ValueError('Not eight complete source bars')
        if int(r['planned_stage3_events']) > plan['max_events']: raise ValueError('Insufficient event budget')
    model_records = []
    for seed in (42,43,44):
        path = models/f'seed{seed}/Aplus/phase2/checkpoints/params/best_params.pt'
        digest = sha(path)
        if digest != plan['model_sha256'][f'Aplus_seed{seed}']: raise ValueError('Model changed')
        model_records.append(dict(seed=seed, path=str(path), sha256=digest))
    report = dict(status='PASS',sources=68,planned_outputs=136,models=model_records,
                  smoke_sources=[r['source_sha256'] for r in smoke_sources(rows)],smoke_outputs=12,
                  training=False,gpu_checked=gpu,startup_validation_records=len(startup_files)-2)
    if gpu:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is not available')
        report.update(torch=torch.__version__,device=torch.cuda.get_device_name(0))
    return report


def subset(inputs, output, rows):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((inputs/'plan.json').read_text())
    plan.update(source_count=len(rows),expected_outputs=2*len(rows),purpose='Technical smoke subset')
    dump(output/'plan.json',plan); dump(output/'sources.json',rows)
    for r in rows:
        for field in ('frozen_path','condition_path'):
            dest=output/r[field];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(inputs/r[field],dest)
    shutil.copy2(inputs/'emopia-train-attribute-bounds-p25-p75.json',output)
    seal(output)


def audit_event(source, event):
    before, after = Path(source).read_text().splitlines(), Path(event).read_text().splitlines()
    if declared_source_key(before) != declared_source_key(after): raise ValueError('Key mismatch')
    melody, performance = track_bars(before,'Melody'), track_bars(after,'Performance')
    if len(melody) != 8 or len(performance) != 8: raise ValueError('Incomplete performance')
    if track_bars(after,'Melody') != melody: raise ValueError('Condition melody changed')
    for condition, actual in zip(melody, performance):
        plan = build_performance_bar_plan(condition)
        if len(actual) != len(plan): raise ValueError('Incomplete performance attribute sequence')
        for (kind, value), token in zip(plan, actual):
            if (kind == FORCED and value != token) or (kind != FORCED and not token.startswith(value)):
                raise ValueError('Performance structure changed')
    return dict(complete_bars=8,key_preserved=True,condition_and_performance_skeleton_equal=True)


def audit(inputs, output):
    sources = json.loads((inputs/'sources.json').read_text())
    lookup = {r['source_sha256']:r for r in sources}
    expected = {(r['source_sha256'],q) for r in sources for q in (('Q1','Q4') if r['valence']=='positive' else ('Q2','Q3'))}
    rows = list(csv.DictReader((output/'population_manifest.csv').open()))
    if Counter((r['source_sha256'],r['target_quadrant']) for r in rows) != Counter(expected):
        raise ValueError('Output identities incomplete or duplicated')
    checked = []
    for r in rows:
        source = inputs/lookup[r['source_sha256']]['frozen_path']
        event, midi = output/r['event_path'],output/r['midi_path']
        if sha(event)!=r['event_sha256'] or sha(midi)!=r['midi_sha256']: raise ValueError('Output hash mismatch')
        structure = audit_event(source,event)
        physical = check_midi(source,midi)
        checked.append(dict(source_sha256=r['source_sha256'],target=r['target_quadrant'],seed=r['seed'],
                            **structure,notes=physical['notes'],tempo_values=physical['tempo_values']))
    report = dict(status='PASS_TECHNICAL_COLLECTION',outputs=len(rows),records=checked,
                  formalStudyData=False,musical_quality_evaluated=False)
    dump(output/'technical_audit.json',report)
    return report


def campaign(inputs, models, out):
    shutil.copytree(inputs,out/'inputs')
    assets=json.loads((inputs/'code_assets.json').read_text())
    for name in assets:
        dest=out/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    def progress(phase,done):
        previous=json.loads((out/'progress.json').read_text())
        previous.update(status='RUNNING',phase=phase,completed=done,total=4,updated_unix=time.time())
        dump(out/'progress.json',previous)
    progress('verify_existing_models_and_inputs',0)
    dump(out/'preflight.json',preflight(inputs,models,gpu=True))
    progress('smoke_12_performances',1)
    subset(inputs,out/'smoke-inputs',smoke_sources(json.loads((inputs/'sources.json').read_text())))
    generate(out/'smoke-inputs',models,out/'smoke-inputs',out/'smoke')
    audit(out/'smoke-inputs',out/'smoke')
    progress('generate_136_performances',2)
    generate(inputs,models,inputs,out/'performances')
    progress('verify_all_performances',3)
    audit(inputs,out/'performances')
    progress('outputs_verified',4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','campaign','launch'])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args();args.inputs=args.inputs.resolve();args.models=args.models.resolve()
    if args.mode=='preflight':
        print(json.dumps(preflight(args.inputs,args.models),indent=2));return 0
    if args.output is None: p.error('--output is required')
    out=args.output.resolve()
    if args.mode=='campaign': campaign(args.inputs,args.models,out);return 0
    if not str(REPO).startswith('/root/autodl-tmp/') or out.name != RUN:
        raise RuntimeError('Launch is only for the explicitly authorized AutoDL run path')
    backup=Path('/root/autodl-fs/moonmuse')/RUN
    supervisor=Supervisor(out,backup,REPO,2700,300,
                          ['bash','/usr/bin/shutdown'],total_phases=4)
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    tests=[sys.executable,'-m','pytest','-q','tests/test_performance_skeleton.py','tests/test_ss26_new_score_performance.py']
    cmd=[sys.executable,str(Path(__file__).resolve()),'campaign','--inputs',str(args.inputs),
         '--models',str(args.models),'--output',str(out)]
    return supervisor.run(tests,cmd,dict(os.environ,PYTHONUNBUFFERED='1'))


if __name__=='__main__': sys.exit(main())
