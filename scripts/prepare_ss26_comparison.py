#!/usr/bin/env python3
"""Prepare canonical score inputs and a small deployment bundle for SS26."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.render_rule_performance_ss26 import track_bars
from performance_skeleton import discard_incomplete_identity_prefixes


def canonical_condition(tokens):
    keys=[t for t in tokens if t.startswith('Key_')]
    if len(set(keys))!=1:raise ValueError('Missing or inconsistent score key')
    bars=track_bars(tokens,'Melody',8)
    if not bars:raise ValueError('No source bars')
    # Existing inference reads key only from line one, and splits bars at
    # Bar_None; it supplies the Track_Melody markers itself during generation.
    output=[keys[0]];dropped=0
    for bar in bars:
        clean,items=discard_incomplete_identity_prefixes(bar)
        output.extend(clean);dropped+=len(items)
    return output,dropped


def prepare(project,output):
    output.mkdir(parents=True,exist_ok=False)
    freeze=project/'results/v81-v70-source-freeze-20260809'
    source_manifest=project/'results/ss26-sources-20260927/eligible_ranked.csv'
    rows=list(csv.DictReader(source_manifest.open()))
    copied=[]
    for row in rows:
        source=freeze/row['frozen_path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['source_sha256']:raise ValueError('Source changed')
        tokens,dropped=canonical_condition(source.read_text().splitlines())
        path=output/'conditions'/f"seed{row['seed']}"/row['quadrant']/row['source_name']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(tokens)+'\n')
        copied.append({**row,'condition_path':str(path.relative_to(output)),
                       'condition_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'discarded_prefixes':dropped,'key':tokens[0]})
    (output/'sources.json').write_text(json.dumps(copied,indent=2)+'\n')
    model_root=project/'results/ss26-matched-alignment-20260927/evidence/ss26-alignment-20260926'
    models={}
    for seed in (42,43,44):
        for branch in ('Aplus','A0'):
            path=model_root/f'seed{seed}'/branch/'phase2/checkpoints/params/best_params.pt'
            models[f'{branch}_seed{seed}']=hashlib.sha256(path.read_bytes()).hexdigest()
    reproduction=json.loads((model_root.parents[1]/'legacy_reproduction.json').read_text())
    if not all(r['byte_identical'] for r in reproduction):raise ValueError('Cannot alias v69 and Aplus')
    scripts=['run_ss26_comparison.py','convert_ss26_events.py','audit_ss26_outputs.py',
             'render_rule_performance_ss26.py','select_ss26_sources.py','show_ss26_comparison_progress.py']
    (output/'scripts').mkdir()
    for name in scripts:shutil.copy2(ROOT/'scripts'/name,output/'scripts'/name)
    plan=dict(schema='SS26-comparison-v1',formalStudyData=False,source_count=len(rows),expected_outputs=len(rows)*8,
              branches=['Aplus','A0'],bounds=['bounded','unbounded'],model_sha256=models,
              v69_alias='All three Aplus phase2 parameter files are byte-identical to historical v69. Generate each output once and retain this alias; do not count as independent replicates.',
              input_fix='Original key first; complete Melody bars only; no redundant inter-bar track markers in inference input.',
              source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
              frozen_inference_sha256=hashlib.sha256((ROOT/'stage3_performance/inference.py').read_bytes()).hexdigest(),
              temperature=1.0,top_p=.97,max_events=3072,max_bars=8,paired_condition_seed=True,
              rendering='Use common fixed-key MIDI converter for every branch and R-det. Retain raw MIDI; share repeated-pitch note-release cap for playback. Independently verify MIDI pitch/onset multiplicities.',
              output_use='technical and objective comparison on history-screened candidates; not final listening selection')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    files=sorted(p for p in output.rglob('*') if p.is_file())
    (output/'MANIFEST.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(output))+'\n' for p in files))
    archive=output.parent/(output.name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as t:t.add(output,arcname=output.name)
    print(json.dumps(dict(inputs=str(output),archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sources=len(rows),learned_outputs=len(rows)*8),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.project.resolve(),a.output.resolve())
