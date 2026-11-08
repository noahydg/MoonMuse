#!/usr/bin/env python3
"""Compare objective arousal cues of MoonMuse and independent rules on the October3 new scores."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import miditoolkit
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from scripts.analyze_ss26_comparison import read_csv, write_csv, sha, score_seconds

PAIRED_METHODS = ('MoonMuse', 'R-det', 'R-level-matched', 'R-per-source-matched')
METRICS = ('effective_bpm', 'tempo_sd_bpm', 'velocity_mean', 'velocity_sd',
           'duration_beats', 'legato_ratio', 'raw_integrated_lufs')
CONTRASTS = (('MoonMuse_minus_R-det', 'MoonMuse', 'R-det'),
             ('MoonMuse_minus_R-level-matched', 'MoonMuse', 'R-level-matched'),
             ('MoonMuse_minus_R-per-source-matched', 'MoonMuse', 'R-per-source-matched'))
WINDOW = 32*480


def tempo_sd(tempo_changes, end):
    """Time-weighted SD of the score-window tempo map in BPM."""
    changes = {0:120.0}
    for t in tempo_changes:
        if 0 <= t.time < end:changes[t.time] = t.tempo
    points = sorted(changes)
    score_lengths = np.array([b-a for a,b in zip(points, points[1:]+[end])], dtype=float)
    values = np.array([changes[p] for p in points])
    # Wall-clock duration is proportional to score ticks / BPM, not ticks alone.
    weights = score_lengths / values
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values-mean)**2, weights=weights)))


def features(row, root, lufs):
    path = root/row['midi_file']
    assert sha(path) == row['midi_sha256']
    midi = miditoolkit.MidiFile(str(path))
    assert midi.ticks_per_beat == 480
    notes = sorted((n for i in midi.instruments for n in i.notes), key=lambda n:(n.start, n.pitch))
    assert len(notes) == int(row['notes'])
    onsets = np.array(sorted({n.start for n in notes}))
    ratios = []
    for n in notes:
        later = onsets[onsets > n.start]
        if len(later):ratios.append((n.end-n.start)/(later[0]-n.start))
    velocity = np.array([n.velocity for n in notes], dtype=float)
    bpm = 32*60/score_seconds(midi.tempo_changes, WINDOW, 480)
    # The catalog BPM was computed by the material builder; disagreement means a different window or file.
    assert abs(bpm-float(row['effective_bpm'])) < 1e-6
    return dict(stimulus_id=row['stimulus_id'], method=row['method'], source_sha256=row['source_sha256'],
                plan_id=row['plan_id'], seed=row['seed'], target_quadrant=row['target_quadrant'],
                valence='positive' if row['target_quadrant'] in ('Q1','Q4') else 'negative',
                level='high' if row['target_quadrant'] in ('Q1','Q2') else 'low',
                strict21=row['strict21'] == 'True', notes=len(notes), effective_bpm=bpm,
                tempo_sd_bpm=tempo_sd(midi.tempo_changes, WINDOW), velocity_mean=float(velocity.mean()),
                velocity_sd=float(velocity.std()), duration_beats=float(np.mean([(n.end-n.start)/480 for n in notes])),
                legato_ratio=float(np.mean(ratios)), raw_integrated_lufs=lufs[row['stimulus_id']])


def bootstrap(values, draws=10000, seed=20261027):
    """Conditional source-resampling interval; no inference over new model seeds."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                ci95_low=float(np.quantile(means, .025)), ci95_high=float(np.quantile(means, .975)),
                positive=int(np.sum(v > 1e-9)), zero=int(np.sum(abs(v) <= 1e-9)), negative=int(np.sum(v < -1e-9)))


def paired_gaps(files):
    groups = defaultdict(dict)
    for r in files:
        if r['method'] in PAIRED_METHODS:
            assert r['level'] not in groups[r['source_sha256'], r['method']]
            groups[r['source_sha256'], r['method']][r['level']] = r
    gaps = []
    for (source, method), pair in sorted(groups.items()):
        assert set(pair) == {'high', 'low'}
        high, low = pair['high'], pair['low']
        gaps.append(dict(source_sha256=source, method=method, seed=high['seed'], valence=high['valence'],
                         strict21=high['strict21'], **{m:high[m]-low[m] for m in METRICS}))
    return gaps


def contrast_gaps(gaps):
    lookup = {(r['source_sha256'], r['method']):r for r in gaps}
    rows = []
    for source in sorted({r['source_sha256'] for r in gaps}):
        for name, a, b in CONTRASTS:
            left, right = lookup[source, a], lookup[source, b]
            rows.append(dict(source_sha256=source, contrast=name, seed=left['seed'], valence=left['valence'],
                             strict21=left['strict21'], **{m:left[m]-right[m] for m in METRICS}))
    return rows


def summarize(rows, key, scopes):
    out = []
    for scope, keep in scopes:
        groups = defaultdict(list)
        for r in rows:
            if keep(r):groups[r[key]].append(r)
        for name, vals in sorted(groups.items()):
            for m in METRICS:out.append(dict(scope=scope, **{key:name}, metric=m, **bootstrap([r[m] for r in vals])))
    return out


def score_adaptivity(files):
    """Across-source spread alone does not identify learned score adaptation."""
    groups = defaultdict(list)
    for r in files:groups[r['method'], r['target_quadrant']].append(r)
    return [dict(method=method, target_quadrant=q, sources=len(vals), metric=m,
                 across_source_sd=float(np.std([r[m] for r in vals], ddof=1)))
            for (method, q), vals in sorted(groups.items()) if len(vals) > 1
            for m in ('effective_bpm', 'velocity_mean', 'legato_ratio')]


def run(materials, output):
    output.mkdir(parents=True, exist_ok=False)
    stimuli = read_csv(materials/'stimuli.csv')
    lufs = {r['stimulus_id']:float(r['raw_integrated_lufs']) for r in read_csv(materials/'render'/'raw_measurements.csv')}
    files = [features(r, materials, lufs) for r in stimuli]
    assert len(files) == 568 and len({r['source_sha256'] for r in files}) == 68
    write_csv(output/'per_file.csv', files)
    gaps = paired_gaps(files);write_csv(output/'paired_high_minus_low.csv', gaps)
    assert len(gaps) == 68*len(PAIRED_METHODS)
    contrasts = contrast_gaps(gaps);write_csv(output/'gap_contrasts.csv', contrasts)
    scopes = (('union68', lambda r:True), ('strict21', lambda r:r['strict21']))
    seed_scopes = tuple(('seed'+s, (lambda s:lambda r:r['seed'] == s)(s)) for s in sorted({r['seed'] for r in files}))
    summaries = dict(gaps=summarize(gaps, 'method', scopes), contrasts=summarize(contrasts, 'contrast', scopes),
                     contrasts_by_seed=summarize(contrasts, 'contrast', seed_scopes),
                     score_adaptivity=score_adaptivity(files))
    for name, data in summaries.items():write_csv(output/(name+'.csv'), data)
    (output/'summary.json').write_text(json.dumps(dict(
        status='PASS', files=len(files), sources=68, paired_methods=PAIRED_METHODS, descriptive_only=True,
        interval='source-resampled 95% percentile interval of the mean, 10000 draws, seed 20261027; conditional on selected sources and fitted seeds; not a human-rating interval',
        weighting='equal sources within scope; auxiliary cue analysis, not the equal-valence listening estimator',
        loudness_basis='raw rendered audio before common gain/encoding, not measured final MP3 perceived loudness',
        tempo_sd_basis='wall-clock-duration weighted BPM standard deviation over 32 beats',
        materials_sha256=sha(materials/'stimuli.csv'), loudness_input_sha256=sha(materials/'render'/'raw_measurements.csv'),
        script_sha256=sha(Path(__file__)), summaries=summaries), indent=2)+'\n')
    print(json.dumps(dict(status='PASS', files=len(files), sources=68, output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('materials', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args();run(a.materials, a.output)
