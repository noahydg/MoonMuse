"""Calibrate a rule's target means on reserved scores; assess held-out drift."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import mido
import miditoolkit
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.analyze_ss26_comparison import score_seconds


def read(p): return list(csv.DictReader(Path(p).open()))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bounded_mean_shift(values,target,lower,upper):
    """Common shift with clipping, then deterministic largest-remainder rounding."""
    x=np.asarray(values,dtype=float)
    if not len(x) or not lower<=target<=upper:raise ValueError('Infeasible velocity mean')
    wanted=int(np.floor(target*len(x)+.5))
    lo,hi=lower-float(x.max())-1,upper-float(x.min())+1
    for _ in range(80):
        mid=(lo+hi)/2
        if np.clip(x+mid,lower,upper).sum()<wanted:lo=mid
        else:hi=mid
    shifted=np.clip(x+(lo+hi)/2,lower,upper)
    integer=np.floor(shifted).astype(int)
    remainder=wanted-int(integer.sum())
    candidates=sorted((i for i in range(len(x)) if integer[i]<upper),key=lambda i:(-(shifted[i]-integer[i]),i))
    if not 0<=remainder<=len(candidates):raise ValueError('Velocity rounding failed')
    for i in candidates[:remainder]:integer[i]+=1
    assert int(integer.sum())==wanted and np.all((integer>=lower)&(integer<=upper))
    return integer.tolist()


def transform(src,dst,bpm,velocity,lower,upper):
    midi=mido.MidiFile(src)
    identities=[]
    for ti,t in enumerate(midi.tracks):
        tick=0
        for mi,msg in enumerate(t):
            tick+=msg.time
            if msg.type=='note_on' and msg.velocity>0:
                identities.append((ti,tick,msg.channel,msg.note,msg.velocity,mi))
    identities.sort()
    values=bounded_mean_shift([r[4] for r in identities],velocity,lower,upper)
    mapped={(r[0],r[5]):v for r,v in zip(identities,values)}
    for ti,t in enumerate(midi.tracks):
        tick=previous=0;new=[]
        for mi,msg in enumerate(t):
            tick+=msg.time
            if msg.type=='set_tempo':continue
            if msg.type=='note_on' and msg.velocity>0:msg=msg.copy(velocity=mapped[ti,mi])
            new.append(msg.copy(time=tick-previous));previous=tick
        t[:]=new
    midi.tracks[0].insert(0,mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bpm),time=0))
    dst.parent.mkdir(parents=True,exist_ok=True);midi.save(dst)


def features(path):
    m=miditoolkit.MidiFile(str(path));notes=[n for i in m.instruments for n in i.notes]
    return dict(effective_bpm=32*60/score_seconds(m.tempo_changes,32*m.ticks_per_beat,m.ticks_per_beat),
        velocity_mean=statistics.mean(n.velocity for n in notes),notes=len(notes),
        velocity_sd=float(np.std([n.velocity for n in notes])))


def skeleton(path,velocity=False):
    m=miditoolkit.MidiFile(str(path))
    return Counter((i,n.pitch,n.start,n.end,*((n.velocity,) if velocity else ())) for i,ins in enumerate(m.instruments) for n in ins.notes)


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    eligible=read(root/'results/ss26-sources-20260927/eligible_ranked.csv')
    for r in eligible:r['role']='calibration' if int(r['rank_within_cell'])==1 or int(r['rank_within_cell'])>10 else 'held_out_candidate'
    write(out/'source_roles.csv',eligible)
    roles={r['source_sha256']:r for r in eligible}
    calibration={h for h,r in roles.items() if r['role']=='calibration'}
    heldout={h for h,r in roles.items() if r['role']=='held_out_candidate'}
    assert len(heldout)==108 and calibration.isdisjoint(heldout)
    dump(out/'fit_specification.json',dict(protocol=str(root/'results/ss26-correction-protocol-20260928.md'),
        protocol_sha256=sha(root/'results/ss26-correction-protocol-20260928.md'),calibration_sources=len(calibration),held_out_sources=len(heldout),
        source_role_sha256=sha(out/'source_roles.csv'),estimator='within seed/source-quadrant means then equal cell weights per target',
        human_responses_used=False,per_stimulus_learning_values_used_for_rule=False,script_sha256=sha(__file__)))
    catalog=read(root/'results/ss26-descriptive-analysis-20260927-r3/catalog.csv')
    corrections={(r['source_sha256'],r['method'],r['target_quadrant']):r for r in read(root/'results/ss26-tempo-sensitivity-20260927-r2/changes.csv')}
    learned={};groups=defaultdict(list);training=[]
    for r in catalog:
        if r['method']!='Aplus-bounded':continue
        h,q=r['source_sha256'],r['target_quadrant'];p=Path(r['midi_path']);assert sha(p)==r['midi_sha256']
        if (h,r['method'],q) in corrections:p=Path(corrections[h,r['method'],q]['corrected_midi'])
        row=dict(r,selected_midi=str(p),selected_midi_sha256=sha(p),**features(p));learned[h,q]=row
        if h in calibration:groups[q,r['seed'],r['source_quadrant']].append(row);training.append(row)
    cellmeans=[dict(target=q,seed=seed,source_quadrant=sq,n=len(v),
        effective_bpm=statistics.mean(r['effective_bpm'] for r in v),velocity_mean=statistics.mean(r['velocity_mean'] for r in v))
        for (q,seed,sq),v in sorted(groups.items())]
    write(out/'calibration_inputs.csv',training);write(out/'calibration_cell_means.csv',cellmeans)
    params={q:{k:statistics.mean(r[k] for r in cellmeans if r['target']==q) for k in ('effective_bpm','velocity_mean')}
        for q in ('Q1','Q2','Q3','Q4')}
    dump(out/'parameters.json',dict(method='R-level-matched',parameters=params,formalStudyData=False,
        calibration_only=True,performance_basis='Training-derived R-det beat accents and note durations; target means learned from reserved scores',
        interpretation='Global symbolic target-level calibration, not per-source or acoustic/perceptual equivalence'))
    bounds=json.loads((root/'results/ss26-rule-fit-20260927/rule_parameters.json').read_text())['bounds']
    rows=[]
    for r in catalog:
        if r['method']!='R-det':continue
        h,q=r['source_sha256'],r['target_quadrant'];p=params[q];b=bounds[q]['Note_Velocity']
        raw=out/'raw-midi'/f'{h}_{q}.mid';play=out/'playback-midi'/raw.name
        for old,new in ((Path(r['raw_midi_path']),raw),(Path(r['midi_path']),play)):
            transform(old,new,p['effective_bpm'],p['velocity_mean'],b['min'],b['max'])
            assert skeleton(old)==skeleton(new),old
        f=features(play)
        assert abs(f['effective_bpm']-p['effective_bpm'])<.001
        assert abs(f['velocity_mean']-p['velocity_mean'])<=.5/f['notes']+1e-9
        assert Counter((i,pitch,start,vel) for i,pitch,start,end,vel in skeleton(raw,True).elements())==Counter((i,pitch,start,vel) for i,pitch,start,end,vel in skeleton(play,True).elements())
        rows.append(dict(source_sha256=h,seed=r['seed'],source_quadrant=r['source_quadrant'],target_quadrant=q,
            role=roles[h]['role'],method='R-level-matched',raw_midi_path=str(raw),midi_path=str(play),midi_sha256=sha(play),
            original_rule_midi=r['midi_path'],original_rule_sha256=r['midi_sha256'],**f,
            learned_midi=learned[h,q]['selected_midi'],learned_midi_sha256=learned[h,q]['selected_midi_sha256'],
            learned_effective_bpm=learned[h,q]['effective_bpm'],learned_velocity_mean=learned[h,q]['velocity_mean']))
    write(out/'population_manifest.csv',rows)
    comparisons=[]
    for subset in ('calibration','held_out_36','held_out_72','held_out_108'):
        count=int(subset.rsplit('_',1)[1]) if subset!='calibration' else None
        selected={h for h,r in roles.items() if (h in calibration if count is None else 2<=int(r['rank_within_cell'])<=1+count//12)}
        for q in ('Q1','Q2','Q3','Q4'):
            rr=[r for r in rows if r['source_sha256'] in selected and r['target_quadrant']==q]
            cells=defaultdict(list)
            for r in rr:cells[r['seed'],r['source_quadrant']].append(r)
            for metric in ('effective_bpm','velocity_mean'):
                rule=statistics.mean(statistics.mean(r[metric] for r in v) for v in cells.values())
                learn=statistics.mean(statistics.mean(r['learned_'+metric] for r in v) for v in cells.values())
                comparisons.append(dict(subset=subset,target=q,sources=len(rr),metric=metric,rule=rule,learned=learn,
                    learned_minus_rule=learn-rule,source_difference_rmse=float(np.sqrt(np.mean([(r['learned_'+metric]-r[metric])**2 for r in rr])))))
    write(out/'level_comparison.csv',comparisons)
    gaps=[]
    for subset in ('calibration','held_out_36','held_out_72','held_out_108'):
        for high,low in (('Q1','Q4'),('Q2','Q3')):
            for metric in ('effective_bpm','velocity_mean'):
                hi=next(r for r in comparisons if (r['subset'],r['target'],r['metric'])==(subset,high,metric))
                lo=next(r for r in comparisons if (r['subset'],r['target'],r['metric'])==(subset,low,metric))
                gaps.append(dict(subset=subset,valence='positive' if high=='Q1' else 'negative',metric=metric,
                    learned_gap=hi['learned']-lo['learned'],rule_gap=hi['rule']-lo['rule'],residual_gap_difference=hi['learned']-lo['learned']-hi['rule']+lo['rule']))
    write(out/'gap_comparison.csv',gaps)
    report=dict(status='PASS',formalStudyData=False,outputs=len(rows),calibration_sources=len(calibration),held_out_sources=len(heldout),
        unchanged_pitch_onset_duration_checks=2*len(rows),human_response_fit=False,formal_audio_frozen=False)
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))


def per_source_diagnostic(root,out):
    """An oracle-style control using each learned clip's summary, not a standalone rule."""
    out.mkdir(parents=True,exist_ok=False)
    base=root/'results/ss26-level-matched-rule-20260928-r2'
    original=read(base/'population_manifest.csv')
    bounds=json.loads((root/'results/ss26-rule-fit-20260927/rule_parameters.json').read_text())['bounds']
    dump(out/'specification_before_render.json',dict(method='R-per-source-matched',formalStudyData=False,
        reason='Held-out target-level drift remained after reserved-source calibration; introduce an explicitly separate within-source diagnostic.',
        conditioning='Uses each paired learned MIDI mean velocity and eight-bar effective BPM; no listener responses.',
        estimand='Remaining temporal/dynamic patterns and articulation jointly, conditional on equal symbolic clip means; not isolated articulation or matched perceived loudness.',
        baseline_role='Model-assisted diagnostic, not an independently deployable train-only rule.',
        parent_manifest_sha256=sha(base/'population_manifest.csv'),script_sha256=sha(__file__)))
    rows=[]
    for r in original:
        h,q=r['source_sha256'],r['target_quadrant'];old=Path(r['original_rule_midi'])
        assert sha(old)==r['original_rule_sha256']
        learned=Path(r['learned_midi']);assert sha(learned)==r['learned_midi_sha256']
        target=features(learned);b=bounds[q]['Note_Velocity'];new=out/'playback-midi'/f'{h}_{q}.mid'
        transform(old,new,target['effective_bpm'],target['velocity_mean'],b['min'],b['max'])
        assert skeleton(old)==skeleton(new)
        checked=features(new)
        assert abs(checked['effective_bpm']-target['effective_bpm'])<.001
        assert abs(checked['velocity_mean']-target['velocity_mean'])<1e-9
        a=Counter((i,p,s) for i,p,s,e in skeleton(new).elements())
        c=Counter((i,p,s) for i,p,s,e in skeleton(learned).elements())
        assert a==c
        rows.append(dict(source_sha256=h,seed=r['seed'],source_quadrant=r['source_quadrant'],target_quadrant=q,
            method='R-per-source-matched',midi_path=str(new),midi_sha256=sha(new),
            original_rule_midi=str(old),original_rule_sha256=sha(old),learned_midi=str(learned),learned_midi_sha256=sha(learned),
            **checked,learned_effective_bpm=target['effective_bpm'],learned_velocity_mean=target['velocity_mean']))
    write(out/'population_manifest.csv',rows)
    report=dict(status='PASS',outputs=len(rows),formalStudyData=False,per_source_matching=True,
        maximum_bpm_error=max(abs(r['effective_bpm']-r['learned_effective_bpm']) for r in rows),
        maximum_velocity_mean_error=max(abs(r['velocity_mean']-r['learned_velocity_mean']) for r in rows),
        rule_pitch_onset_duration_preserved=len(rows),learned_pitch_onset_equal=len(rows),formal_audio_frozen=False)
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REPO.parent);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-source-diagnostic',action='store_true')
    a=p.parse_args();(per_source_diagnostic if a.per_source_diagnostic else run)(a.root.resolve(),a.out.resolve())
