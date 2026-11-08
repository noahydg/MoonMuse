#!/usr/bin/env python3
"""Execute the finite SS26 matched output comparison after local preparation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.audit_ss26_outputs import check_midi


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(inputs,models,source_root,output):
    output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((inputs/'plan.json').read_text());sources=json.loads((inputs/'sources.json').read_text())
    for line in (inputs/'MANIFEST.sha256').read_text().splitlines():
        h,name=line.split('  ',1)
        if sha(inputs/name)!=h:raise ValueError(f'Input changed: {name}')
    if sha(ROOT/'stage3_performance/inference.py')!=plan['frozen_inference_sha256']:raise ValueError('Inference changed')
    for seed in (42,43,44):
        for branch in plan['branches']:
            if sha(models/f'seed{seed}'/branch/'phase2/checkpoints/params/best_params.pt')!=plan['model_sha256'][f'{branch}_seed{seed}']:
                raise ValueError('Checkpoint changed')
    for r in sources:
        if sha(source_root/r['frozen_path'])!=r['source_sha256']:raise ValueError('Source changed')
    shutil=__import__('shutil');shutil.copy2(inputs/'plan.json',output/'plan.json')
    current=output/'progress.json';started=time.time();records=[];commands=[]
    for branch in plan['branches']:
        for mode in plan['bounds']:
            for seed in (42,43,44):
                for q in ('Q1','Q2','Q3','Q4'):
                    selected=[r for r in sources if int(r['seed'])==seed and r['quadrant']==q]
                    if not selected:
                        continue
                    for target in (('Q1','Q4') if q in ('Q1','Q4') else ('Q2','Q3')):
                        group=output/branch/mode/f'seed{seed}'/q/target;group.mkdir(parents=True)
                        state=dict(status='RUNNING',completed=len(records),expected=plan['expected_outputs'],started_unix=started,
                                   branch=branch,bounds=mode,seed=seed,source_quadrant=q,target=target,updated_unix=time.time(),group_expected=len(selected),group_start_completed=len(records),group=str(group))
                        current.write_text(json.dumps(state,indent=2)+'\n')
                        argv=[sys.executable,'stage3_performance/inference.py','-c','stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml',
                              '-m','gpt2','-r','functional','-i',str(models/f'seed{seed}'/branch/'phase2/checkpoints/params/best_params.pt'),
                              '-id',str(inputs/'conditions'/f'seed{seed}'/q),'-od',str(group/'stage3_performance'),
                              '--seed',str(seed),'--emotion-override',target,'--conditioning-mode','factorized_va','--paired-condition-seed',
                              '--condition-track','Melody','--max_bars','8','--max_events','3072','--temperature','1.0','--top-p','0.97','--lock-performance-to-condition-skeleton']
                        if mode=='bounded':argv+=['--attribute-bounds-json',str(source_root/'emopia-train-attribute-bounds-p25-p75.json')]
                        steps=[('generation',argv),('conversion',[sys.executable,'scripts/convert_ss26_events.py','--events',str(group/'stage3_performance'),'--raw-midi',str(group/'raw-midi'),'--playback-midi',str(group/'playback-midi')])]
                        for name,cmd in steps:
                            stamp=time.time()
                            with (group/(name+'.log')).open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                            commands.append(dict(step=name,group=str(group.relative_to(output)),argv=cmd,returncode=result.returncode,elapsed_seconds=time.time()-stamp))
                            (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
                            if result.returncode:raise RuntimeError(f'{name} failed: {group}')
                        expected_names={Path(r['source_name']).stem+f'_{target}_performance.txt' for r in selected}
                        if {p.name for p in (group/'stage3_performance').glob('*.txt')}!=expected_names:raise ValueError('Output count mismatch')
                        for row in selected:
                            name=Path(row['source_name']).stem+f'_{target}_performance'
                            event=group/'stage3_performance'/(name+'.txt')
                            raw=group/'raw-midi'/(name+'.mid');midi=group/'playback-midi'/(name+'.mid')
                            check_midi(source_root/row['frozen_path'],raw)
                            check_midi(source_root/row['frozen_path'],midi)
                            records.append(dict(seed=seed,source_quadrant=q,sample_index=row['sample_index'],source_name=row['source_name'],source_sha256=row['source_sha256'],target_quadrant=target,
                                                branch=branch,bounds=mode,event_path=str(event.relative_to(output)),event_sha256=sha(event),raw_midi_path=str(raw.relative_to(output)),midi_path=str(midi.relative_to(output)),midi_sha256=sha(midi),midi_skeleton_equal=True))
                        with (output/'population_manifest.csv').open('w',newline='') as f:
                            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
                        state.update(completed=len(records),updated_unix=time.time());current.write_text(json.dumps(state,indent=2)+'\n')
                        print(f"PASS {len(records)}/{plan['expected_outputs']} {branch} {mode} seed{seed} {q}->{target}",flush=True)
    if len(records)!=plan['expected_outputs']:raise ValueError('Final population incomplete')
    current.write_text(json.dumps(dict(status='GENERATED',completed=len(records),expected=len(records),started_unix=started,updated_unix=time.time()),indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(status='PASS',outputs=len(records),midi_skeleton_matches=len(records),formalStudyData=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('inputs','models','source-root','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    try:run(a.inputs.resolve(),a.models.resolve(),a.source_root.resolve(),a.output.resolve())
    except Exception as e:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=str(e),time=time.time()),indent=2)+'\n')
        raise
