"""Analyse the fixed SS26 cohort using the declared quadrant/source estimand."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.plan_ss26_preference_study import arrays
from scripts.calibrate_ss26_returnee_preference import components


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, private=False):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                           0o600 if private else 0o644), 'w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def numeric(value):
    if value in (None, '', 'NA'):
        return None
    result = int(value)
    assert 1 <= result <= 7
    return result


def estimate(rows, values):
    """Reuse the prospective estimator; transformed outcomes give linear contrasts."""
    a = arrays(rows)
    obs = np.array([[v is not None for v in values]])
    y = np.array([[0. if v is None else v for v in values]])
    point, variances, complete = components(y, obs, a)
    if not complete[0]:
        raise ValueError('A source-target cell has no valid response')
    listeners = len({r['participant'] for r, ok in zip(rows, obs[0]) if ok})
    sources = len({r['source'] for r, ok in zip(rows, obs[0]) if ok})
    df = min(listeners, sources)-1
    result = dict(estimate=float(point[0]), observations=int(obs.sum()),
                  listeners=listeners, sources=sources, df=df)
    for name, (variance, valid) in variances.items():
        if not valid[0]:
            result[name] = dict(valid=False)
            continue
        se = float(np.sqrt(variance[0]))
        half = float(t.ppf(.975, df)*se)
        result[name] = dict(se=se, ci95=[float(point[0]-half), float(point[0]+half)],
                            p_two_sided=float(2*t.sf(abs(point[0])/se, df)))
    return result


def point(rows, values):
    cells = defaultdict(list)
    for r, v in zip(rows, values):
        if v is not None:
            cells[r['source'], r['target']].append(v)
    expected = {(r['source'], r['target']) for r in rows}
    assert set(cells) == expected
    by_q = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        selected = [np.mean(v) for (_, cq), v in cells.items() if cq == q]
        by_q[q] = float(np.mean(selected)) if selected else None
    return dict(mean=float(np.mean(list(by_q.values()))) if all(v is not None for v in by_q.values()) else None,
                by_quadrant=by_q)


def decode(root, snapshot):
    payload = json.loads(snapshot.read_text())
    sessions = payload['payload']['sessions']
    completed = [s for s in sessions if s['status'] == 'completed']
    assert len(completed) == 69 and len(sessions)-len(completed) == 6
    assert all(s['formal_data'] for s in completed)
    assert len({s['participant_code'] for s in completed}) == 69
    assert len({s['form_slot'] for s in completed}) == 69
    forms = json.loads((root/'ss26-listening-study/data/forms.json').read_text())
    allocation = read_csv(root/'results/ss26-returning-cohort-prep-20261019/allocation_strict21_n78_w6_s2.csv')
    lookup = {(int(r['listener_slot'])+1, int(r['position'])):r for r in allocation}
    material = root/'results/ss26-new-score-audio-candidates-20261003-v2'
    stimuli = {r['stimulus_id']:r for r in read_csv(material/'stimuli.csv')}
    hashes = {r['stimulus_id']:r['audio_sha256'] for r in read_csv(material/'render/audio_verification.csv')}
    verified = set()
    rows = []
    for s in completed:
        assert len(s['answers']) == 8
        for pos in range(1, 9):
            pid = f'P{pos:02d}'
            answer = s['answers'][pid]
            alloc = lookup[s['form_slot'], pos]
            methods = {}
            for side in ('A', 'B'):
                sid = alloc['stimulus_'+side.lower()]
                stim = stimuli[sid]
                methods[side] = stim['method']
                assert stim['target_quadrant'] == alloc['target_quadrant']
                # System pairs match target and slot, not composition identity.
                if alloc['block'] == 'within' or stim['method'] == 'MoonMuse':
                    assert stim['source_sha256'] == alloc['source_sha256']
                assert answer['stimulus_'+side.lower()] == pid+side
                audio = root/'ss26-listening-study/public'/forms[s['form_slot']-1]['audio'][pid+side]
                if audio not in verified:
                    assert sha(audio) == hashes[sid]
                    verified.add(audio)
            ms = next(k for k, v in methods.items() if v == 'MoonMuse')
            other = 'B' if ms == 'A' else 'A'
            assert (ms == 'A') == (alloc['learning_first'] == 'True')
            pref = answer['preference']
            assert pref in ('A', 'B', 'tie', 'unable')
            r = dict(participant=s['participant_code'], form_slot=s['form_slot'], pair_id=pid,
                     listener_slot=s['form_slot']-1, plan_id=alloc['plan_id'], seed=int(alloc['seed']),
                     block=alloc['block'], source=alloc['source_sha256'], target=alloc['target_quadrant'],
                     target_quadrant=alloc['target_quadrant'], target_sign=int(alloc['target_sign']),
                     model_side=ms, other_method=methods[other], choice=methods.get(pref, pref),
                     model_arousal=numeric(answer['ratings'][ms+'_arousal']),
                     other_arousal=numeric(answer['ratings'][other+'_arousal']),
                     model_valence=answer['ratings'][ms+'_valence'],
                     other_valence=answer['ratings'][other+'_valence'], note=answer.get('note', ''))
            rows.append(r)
    assert len(rows) == 552
    assert len({(r['participant'], r['source']) for r in rows}) == 552
    old_path = root/'results/ss26-returnee-feedback-review-20261027/descriptive-statistics-2026-10-27T07-08-02-051Z.json'
    old = json.loads(old_path.read_text())['decoded_rows']
    by_id = {(r['participant'], r['pair_id']):r for r in rows}
    for r in old:
        assert all(by_id[r['participant'], r['pair_id']][k] == v for k, v in r.items())
    return rows, completed, dict(audio_files_verified=len(verified), old_rows_reproduced=len(old),
                                partial_records_retained=len(sessions)-len(completed))


def analyse(rows):
    output = {}
    for block in ('within', 'system'):
        part = [r for r in rows if r['block'] == block]
        choices = Counter(r['choice'] for r in part)
        values = [None if r['choice']=='unable' else 0 if r['choice']=='tie'
                  else 1 if r['choice']=='MoonMuse' else -1 for r in part]
        pref = estimate(part, values)
        assert abs(pref['estimate']-point(part, values)['mean']) < 1e-12
        result = dict(comparisons=len(part), choices=dict(choices), preference=pref,
                      preference_by_quadrant=point(part, values)['by_quadrant'],
                      unable_bounds=[point(part, [v if v is not None else fill for v in values])['mean']
                                     for fill in (-1, 1)],
                      equal_arousal=sum(r['model_arousal']==r['other_arousal'] for r in part
                                        if r['model_arousal'] is not None and r['other_arousal'] is not None),
                      arousal={}, valence={})
        for method in ('model', 'other'):
            ratings = [r[method+'_arousal'] for r in part]
            means = point(part, ratings)
            result['arousal'][method] = dict(**means, missing=ratings.count(None))
            if block == 'within':
                contrast = estimate(part, [None if v is None else 2*r['target_sign']*v
                                           for r, v in zip(part, ratings)])
                result['arousal'][method]['high_minus_low'] = contrast
            valid = [r for r in part if r[method+'_valence'] not in ('NA', '', None)]
            correct = lambda r: r[method+'_valence'] == ('positive' if r['target'] in ('Q1','Q4') else 'negative')
            val_values = [int(correct(r)) if r in valid else None for r in part]
            result['valence'][method] = dict(counts=dict(Counter(r[method+'_valence'] for r in part)),
                 correct=sum(correct(r) for r in valid), valid=len(valid), weighted_accuracy=point(part, val_values)['mean'])
        if block == 'within':
            diff = [None if r['model_arousal'] is None or r['other_arousal'] is None else
                    2*r['target_sign']*(r['model_arousal']-r['other_arousal']) for r in part]
            result['arousal']['difference_in_differences'] = estimate(part, diff)
            source_gaps = []
            for source in sorted({r['source'] for r in part}):
                own = [r for r in part if r['source']==source]
                item = dict(source=source, seed=own[0]['seed'], valence='positive' if own[0]['target'] in ('Q1','Q4') else 'negative')
                for method in ('model', 'other'):
                    item[method] = float(np.mean([r[method+'_arousal'] for r in own if r['target_sign']==1 and r[method+'_arousal'] is not None])-np.mean([r[method+'_arousal'] for r in own if r['target_sign']==-1 and r[method+'_arousal'] is not None]))
                item['difference'] = item['model']-item['other']
                source_gaps.append(item)
            result['arousal']['source_gaps'] = source_gaps
        result['preference_by_seed'] = {str(seed): point([r for r in part if r['seed']==seed],
                    [v for r, v in zip(part, values) if r['seed']==seed]) for seed in sorted({r['seed'] for r in part})}
        output[block] = result
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=REPO.parent)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, completed, checks = decode(args.root, args.snapshot)
    results = analyse(rows)
    args.output.mkdir(exist_ok=False, parents=True)
    save(args.output/'decoded-responses.json', dict(snapshot=str(args.snapshot), decoded_rows=rows), True)
    save(args.output/'summary.json', dict(snapshot=str(args.snapshot), snapshot_sha256=sha(args.snapshot),
         script_sha256=sha(Path(__file__)), completed_forms=len(completed), verification=checks,
         method='Equal quadrant and source-cell means; four target nuisance means; listener+source HC1 sum, t(min clusters-1); conventional two-way HC1 as sensitivity.',
         inference='One primary: within preference. Secondary intervals nominal/unadjusted. N69 snapshot selected during collection; no N69 power claim.',
         demographics={k:dict(Counter(s[k] for s in completed)) for k in ('age_band', 'music_training')},
         results=results))
    for block, data in results.items():
        print(block, json.dumps({k:data[k] for k in ('choices','preference','arousal','valence') if k != 'arousal'}, ensure_ascii=False))
        print('arousal', json.dumps({k:v for k,v in data['arousal'].items() if k != 'source_gaps'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
