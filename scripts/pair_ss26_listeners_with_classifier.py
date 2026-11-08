#!/usr/bin/env python3
"""Pair SS26 returnee arousal ratings with classifier scores and objective cues of the same stimuli."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from scripts.analyze_ss26_comparison import read_csv, write_csv, sha

CUES = ('effective_bpm', 'velocity_mean', 'raw_integrated_lufs', 'legato_ratio', 'tempo_sd_bpm', 'velocity_sd')
HIGH = ('Q1', 'Q2')
POSITIVE = ('Q1', 'Q4')
SEED = 20261027


def valence_target(quadrant):
    if quadrant not in ('Q1', 'Q2', 'Q3', 'Q4'):
        raise ValueError('Unknown quadrant')
    return 'positive' if quadrant in POSITIVE else 'negative'


def ranks(v):
    v = np.asarray(v, dtype=float);order = v.argsort();r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for value in np.unique(v):r[v == value] = r[v == value].mean()
    return r


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def human_point_summary(rows, stat, key='source_sha256'):
    """Interim point summary; source-only resampling omits listener dependence."""
    return dict(estimate=stat(rows), stimuli=len(rows), sources=len({r[key] for r in rows}),
                uncertainty='Not estimated: repeated listeners and sources require joint handling.')


def classifier_table(root):
    table = {}
    for task, label in (('arousal', 'HA'), ('valence', 'HV')):
        for r in read_csv(root/(task+'_predictions.csv')):
            other = 'LA' if label == 'HA' else 'LV'
            entry = table.setdefault(r['stimulus_id'], {})
            entry[task+'_score'] = float(r['logit_'+label])-float(r['logit_'+other])
            entry[task+'_correct'] = int(r['correct'])
    for r in read_csv(root/'ar_va_predictions.csv'):table[r['stimulus_id']]['quadrant_correct'] = int(r['correct'])
    return table


def population(files):
    """Classifier view of all 68 sources: accuracy and source-paired high-minus-low arousal score."""
    out = dict(accuracy={}, high_minus_low={}, contrasts={})
    by_method = defaultdict(list)
    for r in files:by_method[r['method']].append(r)
    for method, rows in sorted(by_method.items()):
        out['accuracy'][method] = {k:dict(correct=int(sum(r[k] for r in rows)), n=len(rows))
                                   for k in ('arousal_correct', 'valence_correct', 'quadrant_correct')}
    pairs = defaultdict(dict)
    for r in files:
        if r['method'] != 'EMO-Disentangler':pairs[r['method']][r['source_sha256'], r['target_quadrant'] in HIGH] = r['arousal_score']
    gaps = {m:{s:v[s, True]-v[s, False] for s, _ in v} for m, v in pairs.items()}
    for method, g in sorted(gaps.items()):out['high_minus_low'][method] = describe(list(g.values()))
    for rule in ('R-det', 'R-level-matched', 'R-per-source-matched'):
        out['contrasts']['MoonMuse_minus_'+rule] = describe([gaps['MoonMuse'][s]-gaps[rule][s] for s in gaps['MoonMuse']])
    return out


def describe(values, draws=10000, strata=None, resample=True):
    v = np.asarray(values, dtype=float);rng = np.random.default_rng(SEED)
    groups = None if strata is None else {s:v[np.asarray(strata)==s] for s in ('positive', 'negative')}
    estimate = (float(v.mean()) if len(v) else None) if groups is None else (
        float(np.mean([x.mean() for x in groups.values()])) if all(len(x) for x in groups.values()) else None)
    out = dict(n=len(v), mean=estimate, positive=int(np.sum(v > 0)), negative=int(np.sum(v < 0)),
               unweighted_source_mean=float(v.mean()) if len(v) else None)
    if groups is not None:
        out['stratum_counts'] = {s:len(x) for s, x in groups.items()}
    if resample and len(v):
        if strata is not None:
            raise ValueError('Stratified interim summaries intentionally have no bootstrap interval')
        means = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
        out['source_resampling_interval95'] = [float(np.quantile(means, .025)), float(np.quantile(means, .975))]
    return out


def source_gaps(stimuli_rows, value):
    """Per source and method: mean high-target value minus mean low-target value."""
    cells = defaultdict(list)
    for r in stimuli_rows:
        if r[value] is not None:
            cells[r['source_sha256'], r['method'], r['target_quadrant'] in HIGH].append(r[value])
    out = {}
    for (source, method, high) in cells:
        if high and (source, method, False) in cells:
            out[source, method] = np.mean(cells[source, method, True])-np.mean(cells[source, method, False])
    return out


def run(responses, cues, classifier, output, fixed_cohort=False):
    output.mkdir(parents=True, exist_ok=False)
    files = {r['stimulus_id']:r for r in read_csv(cues/'per_file.csv')}
    scores = classifier_table(classifier)
    if set(scores) != set(files):
        raise ValueError('Classifier and cue stimulus sets differ')
    for task in ('arousal', 'valence', 'ar_va'):
        predictions = read_csv(classifier/(task+'_predictions.csv'))
        if len(predictions) != len(files) or len({r['stimulus_id'] for r in predictions}) != len(files):
            raise ValueError('Missing or duplicate classifier predictions')
        for r in predictions:
            f = files[r['stimulus_id']]
            if any(r[k] != f[k] for k in ('method', 'source_sha256', 'target_quadrant')):
                raise ValueError('Classifier metadata mismatch')
            q = f['target_quadrant']
            target = q if task == 'ar_va' else (('HA' if q in HIGH else 'LA') if task == 'arousal' else ('HV' if q in POSITIVE else 'LV'))
            if r['target'] != target or int(r['correct']) != int(r['prediction']==target):
                raise ValueError('Classifier label mapping mismatch')
    for sid, r in files.items():
        r.update(scores[sid])
        for k in CUES:r[k] = float(r[k])
    lookup = {(r['method'], r['source_sha256'], r['target_quadrant']):r for r in files.values()}
    if len(lookup) != len(files):
        raise ValueError('Duplicate source/method/target identity')
    decoded = json.loads(responses.read_text())['decoded_rows']
    if len({(r['participant'], r['pair_id']) for r in decoded}) != len(decoded):
        raise ValueError('Duplicate participant/pair response')
    ratings = []
    for row in decoded:
        for side, method in (('model', 'MoonMuse'), ('other', row['other_method'])):
            stim = lookup[method, row['source'], row['target']]
            value = None if row[side+'_arousal'] in (None, 'NA', '') else int(row[side+'_arousal'])
            if value is not None and not 1 <= value <= 7:
                raise ValueError('Arousal outside 1–7')
            ratings.append(dict(participant=row['participant'], pair_id=row['pair_id'], block=row['block'],
                                stimulus_id=stim['stimulus_id'], method=method, source_sha256=row['source'],
                                target_quadrant=row['target'], arousal=value,
                                valence_rating=row[side+'_valence'], target_arousal_sign=row['target_sign'],
                                target_valence=valence_target(row['target']),
                                arousal_score=stim['arousal_score'], valence_score=stim['valence_score'],
                                **{k:stim[k] for k in CUES}))
    # Restrict permissions before writing participant identifiers.
    fd = os.open(output/'rating_level.csv', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    write_csv(output/'rating_level.csv', ratings)
    # Stimulus means: each deployed stimulus is one row, however many listeners rated it.
    by_stim = defaultdict(list)
    for r in ratings:by_stim[r['stimulus_id']].append(r)
    stimuli = []
    for rs in by_stim.values():
        if len({r['block'] for r in rs}) != 1:
            raise ValueError('Stimulus appears in both experimental blocks')
        valid = [r['arousal'] for r in rs if r['arousal'] is not None]
        stimuli.append(dict({k:v for k, v in rs[0].items() if k not in ('participant', 'pair_id', 'arousal', 'valence_rating')},
                            human_arousal=float(np.mean(valid)) if valid else None, ratings=len(rs), valid_arousal_ratings=len(valid)))
    write_csv(output/'stimulus_level.csv', stimuli)
    result = dict(status='PASS', analysis_version='r2', descriptive_only=True, interim=not fixed_cohort, exploratory=True,
                  proposal_timing='Human/classifier contrast analysis proposed after viewing 14 forms; declared endpoints unchanged.',
                  script_sha256=sha(Path(__file__)), responses_sha256=sha(responses), cues_sha256=sha(cues/'per_file.csv'),
                  classifier_sha256={t:sha(classifier/(t+'_predictions.csv')) for t in ('arousal','valence','ar_va')},
                  completed_forms=len({r['participant'] for r in decoded}), ratings=len(ratings), rated_stimuli=len(stimuli),
                  interval='This descriptive pairing contains no human-dependent intervals. Machine-only union68 intervals resample sources conditional on fitted seeds/materials.',
                  scale_note='Human rating points and classifier logit differences have different units; do not subtract or compare their raw magnitudes.')
    # A. Does the classifier rank stimuli the way listeners do?
    result['stimulus_spearman_human_vs_classifier'] = {
        scope:human_point_summary([s for s in stimuli if keep(s) and s['human_arousal'] is not None], lambda rs:spearman([r['human_arousal'] for r in rs], [r['arousal_score'] for r in rs]))
        for scope, keep in (('all', lambda s:True), ('within', lambda s:s['block']=='within'), ('system', lambda s:s['block']=='system'))}
    # B. Same-score high-minus-low, human versus classifier, per method and model-minus-rule.
    within = [s for s in stimuli if s['block']=='within']
    human_gap, model_gap = source_gaps(within, 'human_arousal'), source_gaps(within, 'arousal_score')
    sources = sorted({s for s, m in human_gap if (s, 'MoonMuse') in human_gap and (s, 'R-per-source-matched') in human_gap})
    expected = {r['source_sha256']:valence_target(r['target_quadrant']) for r in files.values()
                if r['method']=='MoonMuse' and r['strict21'] in (True, 'True', 'true')}
    if not set(sources) <= set(expected):
        raise ValueError('Unexpected source in within-score block')
    strata = [expected[s] for s in sources]
    result['same_score_high_minus_low'] = {
        'sources_with_both_methods_and_targets':len(sources),
        'expected_sources':len(expected), 'full_prespecified_source_coverage':set(sources)==set(expected),
        'weighting':'equal positive/negative valence, equal sources within each; same sources for human and classifier',
        'incomplete_coverage_policy':'If coverage is incomplete, means describe available complete sources only, not the full planned estimand.',
        **{label+'_'+m:describe([g[s, m] for s in sources], strata=strata, resample=False) for label, g in (('human', human_gap), ('classifier', model_gap))
           for m in ('MoonMuse', 'R-per-source-matched')},
        'human_MoonMuse_minus_rule':describe([human_gap[s, 'MoonMuse']-human_gap[s, 'R-per-source-matched'] for s in sources], strata=strata, resample=False),
        'classifier_MoonMuse_minus_rule':describe([model_gap[s, 'MoonMuse']-model_gap[s, 'R-per-source-matched'] for s in sources], strata=strata, resample=False)}
    if sources:
        write_csv(output/'source_level_contrasts.csv', [dict(source_sha256=s, valence=expected[s],
            weight=0.5/strata.count(expected[s]), human_model=human_gap[s,'MoonMuse'], human_rule=human_gap[s,'R-per-source-matched'],
            human_difference=human_gap[s,'MoonMuse']-human_gap[s,'R-per-source-matched'],
            classifier_model=model_gap[s,'MoonMuse'], classifier_rule=model_gap[s,'R-per-source-matched'],
            classifier_difference=model_gap[s,'MoonMuse']-model_gap[s,'R-per-source-matched']) for s in sources])
    # C. Same-target A/B pairs: do listener and classifier differences agree in sign?
    pairs = defaultdict(dict)
    for r in ratings:
        if r['block'] == 'within' and r['arousal'] is not None: pairs[r['participant'], r['pair_id']][r['method']] = r
    diffs = [dict(source_sha256=p['MoonMuse']['source_sha256'], human=p['MoonMuse']['arousal']-p['R-per-source-matched']['arousal'],
                  classifier=p['MoonMuse']['arousal_score']-p['R-per-source-matched']['arousal_score']) for p in pairs.values() if len(p) == 2]
    result['same_target_pairs'] = dict(
        pairs=len(diffs), human_equal=sum(d['human'] == 0 for d in diffs),
        human_model_higher=sum(d['human'] > 0 for d in diffs), human_rule_higher=sum(d['human'] < 0 for d in diffs),
        classifier_model_higher=sum(d['classifier'] > 0 for d in diffs),
        sign_agreement_among_human_nonzero=sum(int(np.sign(d['human']) == np.sign(d['classifier'])) for d in diffs if d['human'] != 0),
        human_nonzero=sum(d['human'] != 0 for d in diffs),
        mean_human_difference=float(np.mean([d['human'] for d in diffs])) if diffs else None,
        mean_classifier_difference=float(np.mean([d['classifier'] for d in diffs])) if diffs else None,
        interpretation='Descriptive sign count, not a chance-level test; pairs repeat listeners and sources.')
    # D. Which cues track each judgment across the rated stimuli (collinear; descriptive ranks only).
    result['cue_spearman'] = {}
    for scope in ('all','within','system'):
        selected = [s for s in stimuli if s['human_arousal'] is not None and (scope=='all' or s['block']==scope)]
        result['cue_spearman'][scope] = {cue:dict(human=spearman([s['human_arousal'] for s in selected], [s[cue] for s in selected]),
                                               classifier=spearman([s['arousal_score'] for s in selected], [s[cue] for s in selected])) for cue in CUES}
    # E. Valence: listener category versus classifier label, by method, on the rated stimuli.
    valence = {}
    for block, method in sorted({(r['block'], r['method']) for r in ratings}):
        rs = [r for r in ratings if r['method']==method and r['block']==block]
        selected = [s for s in stimuli if s['method']==method and s['block']==block]
        valid = [r for r in rs if r['valence_rating'] not in (None, 'NA', '')]
        match = [r for r in valid if r['valence_rating'] == valence_target(r['target_quadrant'])]
        valence[block+' / '+method] = dict(ratings=len(rs), listener_valid_ratings=len(valid), listener_target_match=len(match),
                               listener_mixed=sum(r['valence_rating']=='mixed' for r in rs), listener_unable=len(rs)-len(valid),
                               classifier_correct_rated_stimuli=sum(files[s['stimulus_id']]['valence_correct'] for s in selected),
                               rated_stimuli=len(selected), unit_note='Human counts are ratings; classifier counts are unique stimuli.')
    result['valence_by_method_and_block'] = valence
    result['classifier_population68'] = population(list(files.values()))
    (output/'summary.json').write_text(json.dumps(result, indent=2, default=float, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', ratings=len(ratings), stimuli=len(stimuli), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('responses', 'cues', 'classifier', 'output'):p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fixed-cohort', action='store_true', help='Label a fixed analysis cohort; pairing remains exploratory.')
    a = p.parse_args();run(a.responses, a.cues, a.classifier, a.output, fixed_cohort=a.fixed_cohort)
