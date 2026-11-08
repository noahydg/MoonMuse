"""Prospective N78 sensitivity: standard vs conservative two-way covariance.

No participant answers are loaded. The conservative candidate adds listener
and source covariance components without subtracting their intersection.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.plan_ss26_preference_study import arrays, estimate, group_sum, NOISE, quality_location


def components(y, observed, a):
    e, se, good, complete, standard = estimate(y, observed, a)
    counts = group_sum(observed.astype(float), a['ci'])
    weights = observed*a['cw'][a['ci']]/np.maximum(counts[:, a['ci']], 1)
    target = np.stack([(weights*y)[:, a['qi']==q].sum(axis=1)*4 for q in range(4)], axis=1)
    score = weights*(y-target[:, a['qi']])
    m = observed.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        intersection = m/(m-4)*(score**2).sum(axis=1)
    conservative = standard+intersection
    valid = complete & np.isfinite(conservative) & (conservative>0)
    return e, {'two_way_HC1': (standard, good), 'cluster_sum_HC1': (conservative, valid)}, complete


def wilson_upper(p, n):
    z = 1.959963984540054
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def run_case(rows, cfg, reps, seed, critical):
    a = arrays(rows)
    ls, ss, ts = NOISE[cfg['noise']]
    location, threshold = quality_location(cfg['effect'], cfg['tie'], math.sqrt(1+ls*ls+ss*ss+ts*ts))
    rng = np.random.default_rng(seed)
    collected = {m: {'est': [], 'var': [], 'good': []} for m in ('two_way_HC1', 'cluster_sum_HC1')}
    empty = []
    def random_effect(shape, sd):
        if cfg['family'] == 't3':
            return rng.standard_t(3, shape)*sd/math.sqrt(3)
        return rng.normal(0, sd, shape)
    for start in range(0, reps, 250):
        size = min(250, reps-start)
        z = (random_effect((size,78), ls)[:,a['li']] + random_effect((size,a['sources']), ss)[:,a['si']] +
             random_effect((size,a['sources']), ts)[:,a['si']]*a['t'] + rng.normal(0,1,(size,len(rows)))+location)
        y = (z>threshold).astype(float)-(z < -threshold).astype(float)
        returned = rng.random((size,78)) >= cfg['attrition']
        observed = (rng.random(y.shape) >= cfg['unable']) & returned[:,a['li']]
        e, methods, complete = components(y, observed, a)
        empty.append(~complete)
        for name, (v, good) in methods.items():
            collected[name]['est'].append(e)
            collected[name]['var'].append(v)
            collected[name]['good'].append(good)
    results = []
    missing = np.concatenate(empty)
    for name, parts in collected.items():
        e, v, valid = [np.concatenate(parts[k]) for k in ('est','var','good')]
        halfwidth = critical*np.sqrt(np.maximum(v,0))
        rejection = valid & (np.abs(e)>halfwidth)
        positive = valid & (e>halfwidth)
        coverage = valid & (np.abs(e-cfg['effect'])<=halfwidth)
        results.append(dict(**cfg, method=name, reps=reps, seed=seed,
            rejection=float(rejection.mean()), rejection_wilson95_upper=wilson_upper(float(rejection.mean()),reps),
            positive_CI=float(positive.mean()), coverage=float(coverage.mean()),
            missing_cell_fraction=float(missing.mean()), invalid_fraction=float((~valid).mean()),
            mean_estimate=float(e[valid].mean()), median_halfwidth=float(np.median(halfwidth[valid])),
            critical=critical))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replicates', type=int, default=5000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    allocation = REPO.parent/'results/ss26-returning-cohort-prep-20261019/allocation_strict21_n78_w6_s2.csv'
    with allocation.open() as f:
        rows = [r for r in csv.DictReader(f) if r['block']=='within']
    for r in rows:
        r['listener_slot'] = int(r['listener_slot'])
        r['target_sign'] = int(r['target_sign'])
    assert len(rows)==468 and len({r['listener_slot'] for r in rows})==78
    cases = []
    for noise in NOISE:
        for tie in (.1,.3,.5):
            for unable in (0.,.15):
                for attrition in (0.,.1,.2):
                    cases.append(dict(noise=noise,tie=tie,unable=unable,attrition=attrition,effect=0.,family='gaussian'))
    for attrition in (0.,.2):
        cases.append(dict(noise='higher',tie=.3,unable=.05,attrition=attrition,effect=0.,family='t3'))
    for noise in NOISE:
        for effect in (.1,.2,.3):
            cases.append(dict(noise=noise,tie=.3,unable=.05,attrition=0.,effect=effect,family='gaussian'))
    assumptions = dict(date='2026-10-19', formalStudyData=False, human_data_used=False,
        allocation_sha256=hashlib.sha256(allocation.read_bytes()).hexdigest(), cases=cases,replicates=args.replicates,
        noise=NOISE, interval='t(20) with fixed source count21; complete-source-cell estimator',
        candidates=['two_way_HC1','cluster_sum_HC1'],
        rationale='Literature-supported conservative candidate omits the nonnegative intersection covariance; no data-selected scale multiplier.',
        source='https://cameron.econ.ucdavis.edu/research/cameron_miller_slides_SEVS_nov_2022.pdf',
        boundaries=['Gaussian effects are hypothetical, not human estimates.','t3 cases test the zero-effect null only; tie rates need not match Gaussian values.',
                    'Attrition and item missingness are random in these scenarios; informative nonreturn is not resolved.',
                    'The tests concern the new round, not independence from the old cohort.',
                    'No significance-driven extension or real-response-dependent method choice.'])
    (args.out/'assumptions_before_simulation.json').write_text(json.dumps(assumptions,ensure_ascii=False,indent=2)+'\n')
    critical = float(subprocess.check_output(['Rscript','-e','cat(format(qt(.975,20),digits=17))'],text=True))
    results = []
    for i, cfg in enumerate(cases):
        results.extend(run_case(rows,cfg,args.replicates,26101900+i,critical))
        if (i+1)%15==0: print(f'{i+1}/{len(cases)} scenarios complete', flush=True)
    with (args.out/'sensitivity.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader();writer.writerows(results)
    summary = dict(status='COMPLETE_PROSPECTIVE_COMPARISON_NOT_UNIVERSAL_VALIDATION', formalStudyData=False,
        scenarios=len(cases), replicates_per_scenario=args.replicates, intervals_compared=2, methods={})
    for name in ('two_way_HC1','cluster_sum_HC1'):
        null = [r for r in results if r['method']==name and r['effect']==0]
        summary['methods'][name] = dict(null_rejection_range=[min(r['rejection'] for r in null),max(r['rejection'] for r in null)],
            max_null_wilson_upper=max(r['rejection_wilson95_upper'] for r in null),
            max_missing_cells=max(r['missing_cell_fraction'] for r in null),
            net20_detection={r['noise']:r['positive_CI'] for r in results if r['method']==name and r['effect']==.2},
            net10_detection_middle=next(r['positive_CI'] for r in results if r['method']==name and r['effect']==.1 and r['noise']=='middle'))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
