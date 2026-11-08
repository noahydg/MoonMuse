#!/usr/bin/env python3
"""Describe the complete SS26 population using source-level paired contrasts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import miditoolkit
import numpy as np

METHODS = ('R-det', 'Aplus-bounded', 'A0-bounded', 'Aplus-unbounded', 'A0-unbounded')
METRICS = ('effective_bpm', 'velocity_mean', 'velocity_sd', 'raw_duration_beats',
           'playback_duration_beats', 'shortened_fraction')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def collect(learned, rules):
    rows = []
    for root, is_rule in ((learned, False), (rules, True)):
        for r in read_csv(root/'population_manifest.csv'):
            event = root/r['event_path']; midi = root/r['midi_path']
            raw = root/r['raw_midi_path'] if not is_rule else midi.parent.parent/'raw-midi'/midi.name
            assert sha(event) == r['event_sha256'] and sha(midi) == r['midi_sha256']
            method = 'R-det' if is_rule else r['branch']+'-'+r['bounds']
            rows.append(dict(source_sha256=r['source_sha256'], seed=int(r['seed']),
                             source_quadrant=r['source_quadrant'], target_quadrant=r['target_quadrant'],
                             valence='positive' if r['source_quadrant'] in ('Q1','Q4') else 'negative',
                             method=method, event_path=str(event), raw_midi_path=str(raw),
                             midi_path=str(midi), midi_sha256=sha(midi)))
    assert len(rows) == 1540
    counts = Counter(r['source_sha256'] for r in rows)
    assert len(counts) == 154 and set(counts.values()) == {10}
    assert len({(r['source_sha256'], r['method'], r['target_quadrant']) for r in rows}) == 1540
    assert Counter(r['method'] for r in rows) == Counter({m:308 for m in METHODS})
    return rows


def score_seconds(tempo_changes, end, ticks_per_beat):
    """Integrate the tempo map over a fixed score window, last event wins at ties."""
    changes = {0:120.0}
    for t in tempo_changes:
        if 0 <= t.time < end:
            changes[t.time] = t.tempo
    points = sorted(changes)
    return sum((b-a)/ticks_per_beat*60/changes[a]
               for a,b in zip(points, points[1:]+[end]))


def tempo_outside_fraction(tempo_changes, end, lower, upper):
    changes={0:120.0}
    for t in tempo_changes:
        if 0<=t.time<end:changes[t.time]=t.tempo
    points=sorted(changes)
    return sum(b-a for a,b in zip(points,points[1:]+[end])
               if changes[a]<lower-1e-3 or changes[a]>upper+1e-3)/end


def attrs(row, bounds):
    raw = miditoolkit.MidiFile(row['raw_midi_path'])
    play = miditoolkit.MidiFile(row['midi_path'])
    assert raw.ticks_per_beat == play.ticks_per_beat == 480
    order = lambda n:(n.start,n.pitch,n.velocity,n.end)
    rn = sorted([n for i in raw.instruments for n in i.notes],key=order)
    pn = sorted([n for i in play.instruments for n in i.notes],key=order)
    assert len(rn) == len(pn) > 0
    assert [(n.start,n.pitch,n.velocity) for n in rn] == [(n.start,n.pitch,n.velocity) for n in pn]
    assert all(0 <= n.start < 8*4*480 and n.end > n.start for n in rn+pn)
    raw_d = np.array([n.end-n.start for n in rn]); play_d = np.array([n.end-n.start for n in pn])
    assert np.all(play_d <= raw_d)
    velocity = np.array([n.velocity for n in pn])
    result = dict(row, notes=len(pn), effective_bpm=32*60/score_seconds(play.tempo_changes,32*480,480),
                  velocity_mean=float(velocity.mean()), velocity_sd=float(velocity.std()),
                  raw_duration_beats=float(raw_d.mean()/480), playback_duration_beats=float(play_d.mean()/480),
                  shortened_fraction=float(np.mean(play_d<raw_d)), score_seconds=score_seconds(play.tempo_changes,32*480,480))
    tokens = Path(row['event_path']).read_text().splitlines()
    for kind in ('Tempo','Note_Velocity','Note_Duration'):
        # Conti carries the previous tempo; it is not a new numeric BPM choice.
        values = np.array([float(t[len(kind)+1:]) for t in tokens
                           if t.startswith(kind+'_') and t != 'Tempo_Conti'])
        lo,hi = (bounds[row['target_quadrant']][kind][k] for k in ('min','max'))
        if not len(values) and kind != 'Tempo':raise ValueError('Missing numeric '+kind)
        result[kind+'_out_of_bounds_fraction'] = float(np.mean((values<lo)|(values>hi))) if len(values) else None
        if kind=='Tempo':
            result['no_numeric_tempo']=int(len(values)==0)
            result['playback_tempo_outside_score_fraction']=tempo_outside_fraction(play.tempo_changes,32*480,lo,hi)
    return result


def describe(values):
    v = np.asarray(values, dtype=float)
    assert len(v) and np.isfinite(v).all()
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p25=float(np.quantile(v,.25)), p75=float(np.quantile(v,.75)),
                minimum=float(v.min()), maximum=float(v.max()),
                positive=int(np.sum(v>1e-8)), zero=int(np.sum(abs(v)<=1e-8)), negative=int(np.sum(v < -1e-8)))


def group_descriptions(rows, group_keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    return [dict(zip(group_keys,key), **{'metric':m,'missing_n':sum(r[m] is None for r in vals)},
                 **describe([r[m] for r in vals if r[m] is not None]))
            for key,vals in sorted(groups.items()) for m in metrics]


def paired_gaps(rows, metrics=METRICS):
    groups = defaultdict(dict)
    for r in rows:
        key = (r['source_sha256'], r['method'])
        level = 'high' if r['target_quadrant'] in ('Q1','Q2') else 'low'
        if level in groups[key]:raise ValueError('Duplicate source/method/target level')
        groups[key][level] = r
    out = []
    for (source,method), pair in groups.items():
        if set(pair) != {'high','low'}:raise ValueError('Unpaired target')
        high,low = pair['high'],pair['low']
        out.append(dict(source_sha256=source, method=method, seed=high['seed'], valence=high['valence'],
                        **{m:high[m]-low[m] for m in metrics}))
    return out


CONTRASTS = (
    ('learned_minus_rule', 'Aplus-bounded', 'R-det'),
    ('alignment_bounded', 'Aplus-bounded', 'A0-bounded'),
    ('alignment_unbounded', 'Aplus-unbounded', 'A0-unbounded'),
    ('bounds_Aplus', 'Aplus-bounded', 'Aplus-unbounded'),
    ('bounds_A0', 'A0-bounded', 'A0-unbounded'),
)


def contrast_gaps(gaps, metrics=METRICS):
    lookup = {(r['source_sha256'],r['method']):r for r in gaps}
    rows=[]
    for source in sorted({r['source_sha256'] for r in gaps}):
        for name,a,b in CONTRASTS:
            left,right=lookup[source,a],lookup[source,b]
            rows.append(dict(source_sha256=source,contrast=name,seed=left['seed'],valence=left['valence'],
                             **{m:left[m]-right[m] for m in metrics}))
    return rows


def run(learned, rules, bounds_path, output):
    output.mkdir(parents=True,exist_ok=False)
    catalog=collect(learned,rules);write_csv(output/'catalog.csv',catalog)
    bounds=json.loads(bounds_path.read_text())['bounds']
    files=[attrs(r,bounds) for r in catalog];write_csv(output/'per_file.csv',files)
    gaps=paired_gaps(files);write_csv(output/'paired_high_minus_low.csv',gaps)
    contrasts=contrast_gaps(gaps);write_csv(output/'paired_gap_contrasts.csv',contrasts)
    summaries={
        'target_levels':group_descriptions(files,('method','target_quadrant'),METRICS),
        'arousal_gaps':group_descriptions(gaps,('method',),METRICS),
        'arousal_gaps_by_valence':group_descriptions(gaps,('method','valence'),METRICS),
        'arousal_gaps_by_seed':group_descriptions(gaps,('method','seed'),METRICS),
        'gap_contrasts':group_descriptions(contrasts,('contrast',),METRICS),
        'gap_contrasts_by_valence':group_descriptions(contrasts,('contrast','valence'),METRICS),
        'gap_contrasts_by_seed':group_descriptions(contrasts,('contrast','seed'),METRICS),
        'postprocessing':group_descriptions(files,('method',),('shortened_fraction','raw_duration_beats','playback_duration_beats')),
        'bounds':group_descriptions(files,('method',),tuple(k+'_out_of_bounds_fraction' for k in ('Tempo','Note_Velocity','Note_Duration'))),
        'tempo_diagnostics':group_descriptions(files,('method','target_quadrant'),('no_numeric_tempo','playback_tempo_outside_score_fraction')),
    }
    for name,data in summaries.items():write_csv(output/(name+'.csv'),data)
    result=dict(status='PASS',sources=154,files=1540,source_method_pairs=770,descriptive_only=True,
                inferential_tests=[],exclusions=[],bounds_sha256=sha(bounds_path),summaries=summaries)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',files=1540,paired_sources=154,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('learned','rules','bounds','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.learned,a.rules,a.bounds,a.output)
