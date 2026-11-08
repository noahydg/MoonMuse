#!/usr/bin/env python3
"""Fit a deterministic train-only baseline and render paired SS26 performances."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import hashlib
import json
import numbers
from pathlib import Path
import pickle
from statistics import mean, median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance_skeleton import build_performance_bar_plan, discard_incomplete_identity_prefixes, FORCED
from scripts.audit_stage2_stage3_note_flow import parse_track_note_counter
from scripts.select_ss26_sources import write_csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track_bars(tokens, track, maximum=None):
    bars, current = [], None
    for token in tokens:
        if token.startswith('Track_'):
            current = [] if token == f'Track_{track}' else None
            if current is not None:
                bars.append(current)
        elif current is not None and token != 'EOS_None':
            current.append(token)
    # Some budget-truncated sources end at a bare Track_Melody marker.
    # The existing inference reader splits at Bar_None, so it creates no
    # additional bar for that marker. Match that behavior without dropping notes.
    while bars and not bars[-1]:
        bars.pop()
    return bars[:maximum] if maximum else bars


def note_records(bars, require_attributes=False):
    notes = []
    for bar, events in enumerate(bars):
        beat = None
        for i, event in enumerate(events):
            if event.startswith('Beat_'):
                beat = int(event[5:])
            if event.startswith('Note_Octave_') and i+1 < len(events) and events[i+1].startswith('Note_Degree_'):
                if beat is None:
                    raise ValueError('Note before beat')
                n = dict(tick=bar*1920+beat*120, beat=beat, bar=bar, octave=event, degree=events[i+1])
                if require_attributes:
                    if i+3 >= len(events) or not events[i+2].startswith('Note_Duration_') or not events[i+3].startswith('Note_Velocity_'):
                        raise ValueError('Incomplete training performance note')
                    n.update(duration=int(events[i+2].rsplit('_',1)[1]), velocity=int(events[i+3].rsplit('_',1)[1]))
                notes.append(n)
    return notes


def intervals(notes):
    starts = sorted({n['tick'] for n in notes})
    if not starts:
        raise ValueError('No complete notes')
    gaps = {a:b-a for a,b in zip(starts, starts[1:])}
    # Final group has no observed next onset; use the immediately preceding
    # positive onset interval, or one quarter note for a single-onset score.
    gaps[starts[-1]] = starts[-1]-starts[-2] if len(starts)>1 else 480
    return gaps, starts[-1]


def fit(data, bounds_path, output):
    output.mkdir(parents=True, exist_ok=False)
    train_path, valid_path = data/'data_splits/train.pkl', data/'data_splits/valid.pkl'
    train, valid = pickle.loads(train_path.read_bytes()), pickle.loads(valid_path.read_bytes())
    if set(train) & set(valid):
        raise ValueError('Train/validation overlap')
    bounds = json.loads(bounds_path.read_text())
    if sha(train_path) != bounds['source']['split_sha256']:
        raise ValueError('Train split differs from the frozen attribute bounds')
    values = {q:defaultdict(list) for q in ('Q1','Q2','Q3','Q4')}
    provenance = []
    for filename in train:
        path=data/'events'/filename
        payload=pickle.loads(path.read_bytes())[-1]
        tokens=[f"{e['name']}_{e['value']}" for e in payload]
        bars=track_bars(tokens,'Performance')
        notes=note_records(bars,True)
        gaps,last=intervals(notes)
        q=filename.split('_')[0]; v=values[q]
        v['tempo'].extend(float(e['value']) for e in payload if e['name']=='Tempo' and isinstance(e['value'],numbers.Real))
        v['velocity'].extend(n['velocity'] for n in notes)
        file_mean=mean(n['velocity'] for n in notes)
        for n in notes:
            v[f"beat{n['beat']}"].append(n['velocity']-file_mean)
            if n['tick'] != last:
                v['duration_ioi_ratio'].append(n['duration']/gaps[n['tick']])
        provenance.append(dict(file=filename,sha256=sha(path),notes=len(notes)))
    parameters={q:dict(tempo_median=median(v['tempo']),velocity_median=median(v['velocity']),
                       duration_ioi_ratio_median=median(v['duration_ioi_ratio']),
                       beat_offsets={str(b):mean(v[f'beat{b}']) if v[f'beat{b}'] else 0 for b in range(16)},
                       observations={k:len(v[k]) for k in ('tempo','velocity','duration_ioi_ratio')}) for q,v in values.items()}
    vocab=pickle.loads((data/'dictionary.pkl').read_bytes())[0]
    vocabulary={prefix:sorted(int(k[len(prefix):]) for k in vocab if k.startswith(prefix) and k[len(prefix):].isdigit())
                for prefix in ('Tempo_','Note_Duration_','Note_Velocity_')}
    report=dict(schema='SS26-R-det-v1',status='FITTED',formalStudyData=False,train_files=len(train),
                train_split_sha256=sha(train_path),valid_split_sha256=sha(valid_path),validation_used_for_fit=False,
                bounds_sha256=sha(bounds_path),bounds=bounds['bounds'],vocabulary=vocabulary,parameters=parameters,
                script_sha256=sha(Path(__file__)),
                duration_definition='duration / next distinct global onset interval; simultaneous notes share IOI; last training group excluded',
                final_group='use preceding positive IOI; single-onset score uses 480 ticks',
                beat_definition='16 within-bar sixteenth-note positions; mean of velocity minus its file mean',
                quantization='nearest allowed token within quadrant bounds, ties toward smaller value',
                repeat_policy='raw event durations obey bounds; all methods share same-pitch release cap in playback MIDI')
    (output/'rule_parameters.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return report


def quantize(value, prefix, q, spec):
    bounds=spec['bounds'][q][prefix.rstrip('_')]
    choices=[x for x in spec['vocabulary'][prefix] if bounds['min']<=x<=bounds['max']]
    if not choices:
        raise ValueError(f'No token allowed for {q} {prefix}')
    return min(choices,key=lambda x:(abs(x-value),x))


def render_tokens(source_tokens, target, spec):
    key=next(t for t in source_tokens if t.startswith('Key_'))
    original=track_bars(source_tokens,'Melody',8)
    cleaned=[]; discarded=[]
    for events in original:
        clean, drops=discard_incomplete_identity_prefixes(events)
        cleaned.append(clean);discarded.extend(drops)
    notes=note_records(cleaned);gaps,_=intervals(notes)
    params=spec['parameters'][target]
    valence='Positive' if target in ('Q1','Q4') else 'Negative'
    arousal='High' if target in ('Q1','Q2') else 'Low'
    output=[f'Valence_{valence}',f'Arousal_{arousal}',key]
    attrs=defaultdict(list)
    for bar, (condition, events) in enumerate(zip(original,cleaned)):
        output.extend(['Track_Melody',*condition,'Track_Performance'])
        beat=0
        for kind,value in build_performance_bar_plan(events):
            if kind==FORCED:
                output.append(value)
                if value.startswith('Beat_'):beat=int(value[5:])
            else:
                if value=='Tempo_':raw=params['tempo_median']
                elif value=='Note_Velocity_':raw=params['velocity_median']+params['beat_offsets'][str(beat)]
                elif value=='Note_Duration_':raw=params['duration_ioi_ratio_median']*gaps[bar*1920+beat*120]
                else:raise ValueError(value)
                n=quantize(raw,value,target,spec);output.append(f'{value}{n}');attrs[value].append(n)
    output.append('EOS_None')
    return output,dict(discarded_incomplete_prefixes=len(discarded),attributes=dict(attrs))


def render(source_root, manifest, parameter_path, output):
    spec=json.loads(parameter_path.read_text())
    rows=list(csv.DictReader(manifest.open()))
    if not rows or any(r.get('history_class')!='H0' for r in rows):
        raise ValueError('Rendering requires a nonempty H0 candidate manifest')
    output.mkdir(parents=True,exist_ok=False)
    records=[];groups=set()
    for row in rows:
        src=source_root/row['frozen_path']
        if sha(src)!=row['source_sha256']:raise ValueError('Source changed')
        targets=('Q1','Q4') if row['quadrant'] in ('Q1','Q4') else ('Q2','Q3')
        for target in targets:
            group=output/f"seed{row['seed']}"/row['quadrant']/target;groups.add(group)
            event_dir=group/'stage3_performance';event_dir.mkdir(parents=True,exist_ok=True)
            tokens,info=render_tokens(src.read_text().splitlines(),target,spec)
            dest=event_dir/(src.stem+f'_{target}_performance.txt')
            dest.write_text('\n'.join(tokens)+'\n')
            before=parse_track_note_counter(src,'Melody')
            before=type(before)({k:v for k,v in before.items() if k[0]<8})
            after=parse_track_note_counter(dest,'Performance')
            if before!=after:raise ValueError(f'Changed source skeleton: {src}')
            records.append(dict(seed=row['seed'],source_quadrant=row['quadrant'],target_quadrant=target,
                                source_sha256=row['source_sha256'],source_name=src.name,renderer='R-det',
                                event_path=str(dest.relative_to(output)),event_sha256=sha(dest),
                                midi_path=str((group/'playback-midi'/(dest.stem+'.mid')).relative_to(output)),
                                skeleton_equal=True,notes=sum(before.values()),
                                discarded_incomplete_prefixes=info['discarded_incomplete_prefixes']))
    for group in sorted(groups):
        argv=[sys.executable,str(ROOT/'scripts/convert_ss26_events.py'),'--events',str(group/'stage3_performance'),
              '--raw-midi',str(group/'raw-midi'),'--playback-midi',str(group/'playback-midi')]
        with (group/'conversion.log').open('w') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Conversion failed: {group}')
    import miditoolkit
    for r in records:
        path=output/r['midi_path'];m=miditoolkit.MidiFile(str(path))
        note_count=sum(len(i.notes) for i in m.instruments)
        if note_count!=r['notes'] or not note_count:raise ValueError(f'Invalid MIDI count: {path}')
        r.update(midi_sha256=sha(path),midi_valid=True)
    write_csv(output/'population_manifest.csv',records,list(records[0]))
    report=dict(status='PASS',formalStudyData=False,source_count=len(rows),outputs=len(records),
                skeleton_matches=len(records),valid_midi=len(records),
                parameters_sha256=sha(parameter_path),source_manifest_sha256=sha(manifest),
                common_converter_sha256=sha(ROOT/'scripts/convert_ss26_events.py'),
                scope='technical generation on history-screened candidates; not final listening stimuli')
    (output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('fit');f.add_argument('--data',type=Path,required=True);f.add_argument('--bounds',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('render');r.add_argument('--source-root',type=Path,required=True);r.add_argument('--manifest',type=Path,required=True);r.add_argument('--parameters',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=fit(a.data.resolve(),a.bounds.resolve(),a.output.resolve()) if a.action=='fit' else render(a.source_root.resolve(),a.manifest.resolve(),a.parameters.resolve(),a.output.resolve())
    print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','vocabulary','bounds')},indent=2))
