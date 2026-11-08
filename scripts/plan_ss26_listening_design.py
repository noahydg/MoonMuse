#!/usr/bin/env python3
"""Prospective six-pair SS26 allocation and crossed-factor planning sensitivity.

Assumptions are hypothetical, never estimated from the single author response.
Simulation is on paired rating differences (continuous approximation to 1--7
ratings); quality uses a ternary threshold model retaining ties.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import numpy as np

NORMAL = NormalDist()
DESIGNS = [(36, n) for n in (48, 72, 96, 120, 144, 192)] + [
    (72, n) for n in (48, 96, 144, 192)] + [(108, n) for n in (72, 144, 216)]
AROUSAL = {'lower':(.30,.30,.80), 'middle':(.60,.60,1.20), 'higher':(1.00,1.00,1.60)}
DELTAS = (0, .20, .30, .50, .75)


def read_csv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def write_csv(p, rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def dump(p, value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def candidate_sources(root, count):
    pilot=root/'results/ss26-listening-pilot-20260927/private'
    reserved={r['source_sha256'] for r in read_csv(pilot/'pilot_source_reservation.csv')}
    rows=read_csv(root/'results/ss26-sources-20260927/eligible_ranked.csv')
    cells=defaultdict(list)
    for r in rows:
        if r['source_sha256'] not in reserved:
            cells[int(r['seed']),r['quadrant']].append(r)
    chosen=[]
    assert count%12==0 and len(cells)==12
    for key in sorted(cells):
        ranked=sorted(cells[key],key=lambda r:int(r['rank_within_cell']))
        chosen.extend(ranked[:count//12])
    assert len(chosen)==count and not reserved.intersection(r['source_sha256'] for r in chosen)
    return chosen


def allocation(sources, participants, seed=260927):
    """Each listener: six unique scores, 3 high/3 low, 3 learning-first.

    A complete cycle presents every source twice per target, once in each A/B
    orientation. Cyclic permutations cross sources across participant blocks.
    Each listener has one source in each seed x valence cell.
    """
    groups=defaultdict(list)
    for r in sources:groups[int(r['seed']),r['valence']].append(r)
    banks=[groups[k] for k in sorted(groups)]
    g=len(sources)//6;cycle=4*g
    assert len(banks)==6 and all(len(b)==g for b in banks)
    assert participants%cycle==0
    rng=np.random.default_rng(seed);rows=[]
    target_sign=np.array([1,-1,-1,1,1,-1])
    ab_sign=np.array([1,1,-1,-1,1,-1])
    source_map={r['source_sha256']:i for i,r in enumerate(sources)}
    for repeat in range(participants//cycle):
        permutations=[[rng.permutation(g) for _ in range(6)] for _ in range(4)]
        for u in range(4):
            for a in range(g):
                listener=repeat*cycle+u*g+a
                trial=[]
                for group,bank in enumerate(banks):
                    source=bank[int(permutations[u][group][a])]
                    t=int(target_sign[group]*(-1 if u%2 else 1))
                    first=int(ab_sign[group]*(-1 if u//2 else 1))
                    target=('Q1' if t==1 else 'Q4') if source['valence']=='positive' else ('Q2' if t==1 else 'Q3')
                    trial.append(dict(listener_slot=listener,source_index=source_map[source['source_sha256']],
                        source_sha256=source['source_sha256'],seed=source['seed'],source_quadrant=source['quadrant'],
                        valence=source['valence'],target_quadrant=target,target_sign=t,
                        learning_first=first==1))
                rng.shuffle(trial)
                for position,row in enumerate(trial,1):row['position']=position
                rows.extend(trial)
    assert len(rows)==6*participants
    for listener in range(participants):
        r=rows[listener*6:(listener+1)*6]
        assert len({x['source_index'] for x in r})==6
        assert sum(x['target_sign'] for x in r)==0
        assert sum(x['learning_first'] for x in r)==3
        assert Counter(x['seed'] for x in r)=={'42':2,'43':2,'44':2}
        assert Counter(x['valence'] for x in r)=={'positive':3,'negative':3}
    exposure=Counter((r['source_index'],r['target_sign'],r['learning_first']) for r in rows)
    assert len(exposure)==len(sources)*4 and len(set(exposure.values()))==1
    return rows


def crossed_se(y, target, source_ids, participants, source_count, interaction):
    """OLS scalar contrast + two-way clustered CGM variance, finite-G factors.

    Every listener-source intersection is unique. Equal source counts permit
    grouped reshape. Arousal delta is high-minus-low of learned-minus-rule.
    """
    m=y.shape[1];k=2 if interaction else 1
    mean=y.mean(axis=1)
    if interaction:
        estimate=2*(y*target).mean(axis=1)
        residual=y-mean[:,None]-estimate[:,None]*target/2
        score=2*target*residual
    else:
        estimate=mean;score=y-mean[:,None]
    listener_sums=score.reshape(len(y),participants,6).sum(axis=2)
    order=np.argsort(source_ids,kind='stable')
    source_sums=score[:,order].reshape(len(y),source_count,m//source_count).sum(axis=2)
    correction=(m-1)/(m-k)
    variance=correction*(participants/(participants-1)*(listener_sums**2).sum(axis=1)
             +source_count/(source_count-1)*(source_sums**2).sum(axis=1)
             -m/(m-1)*(score**2).sum(axis=1))/m**2
    return estimate,np.sqrt(np.maximum(variance,0)),variance


def tcrit_values():
    dfs=sorted({min(s,n)-1 for s,n in DESIGNS})
    expr='cat(paste(qt(.975,c('+','.join(map(str,dfs))+')),collapse=","))'
    values=subprocess.check_output(['Rscript','-e',expr],text=True).strip().split(',')
    return dict(zip(dfs,map(float,values)))


def quality_location(net_preference,tie_probability,total_sd):
    threshold=total_sd*NORMAL.inv_cdf((1+tie_probability)/2)
    def net(mu):return 1-NORMAL.cdf((threshold-mu)/total_sd)-NORMAL.cdf((-threshold-mu)/total_sd)
    lo,hi=-10.,10.
    for _ in range(70):
        mid=(lo+hi)/2
        if net(mid)<net_preference:lo=mid
        else:hi=mid
    return (lo+hi)/2,threshold


def run(root,out,reps):
    out.mkdir(parents=True,exist_ok=False)
    assumptions=dict(status='prospective_design_sensitivity_not_frozen_protocol',
        independent_author_data_used_for_parameters=False,replicates=reps,seed=26092740,
        design_counts=DESIGNS,trials_per_listener=6,
        arousal_estimand='(learned_high-rule_high)-(learned_low-rule_low), 1--7 rating points',
        arousal_random_slope_sds={k:dict(listener=v[0],source=v[1],paired_residual=v[2]) for k,v in AROUSAL.items()},
        arousal_random_intercept_sds=dict(listener=.4,source=.4),
        arousal_effect_grid=DELTAS,
        arousal_approximation='Gaussian paired differences; actual ordinal bounded scales and missingness require later stress checks',
        quality_estimand='P(learned preferred)-P(rule preferred), ties retained in denominator; unable-to-judge reported separately',
        quality_null_tie_probabilities=[.3,.6,.9],quality_latent_cluster_sds=[.25,.5],
        quality_net_preference_grid=[0,.1,.15],quality_latent_residual_sd=1,
        inference='OLS contrast with listener+source-intersection cluster-robust variance; t(df=min(listener,source)-1); descriptive precision for quality',
        calibration='Report empirical null type-I error and invalid variance; no frozen confirmatory sample size if calibration fails',
        arousal_alpha='two-sided .05; positive lower95% bound for superiority, false-positive positive direction .025',
        source_status='Conditional existing-H0 candidates only; final family awaits separate upstream-source decision; not released stimuli',
        meaningful_effect='No SESOI selected; .20/.30/.50/.75 are sensitivity values, not validated meaningful thresholds')
    dump(out/'assumptions_before_simulation.json',assumptions)
    critical=tcrit_values();arousal=[];quality=[];allocations={}
    for s,n in DESIGNS:
        sources=candidate_sources(root,s)
        path=out/f'sources_{s}.csv'
        if not path.exists():write_csv(path,sources)
        rows=allocation(sources,n)
        write_csv(out/f'allocation_s{s}_n{n}.csv',rows)
        t=np.array([r['target_sign'] for r in rows]);si=np.array([r['source_index'] for r in rows]);li=np.repeat(np.arange(n),6)
        assert np.array_equal(li,np.array([r['listener_slot'] for r in rows]))
        tc=critical[min(s,n)-1];m=6*n
        for k,(pl,ss,resid) in enumerate(AROUSAL.values()):
            label=list(AROUSAL)[k];rng=np.random.default_rng(26092740+s*10000+n*10+k)
            # Random intercepts cancel in the perfectly balanced estimate but
            # remain in observations and finite-sample variance estimation.
            noise=rng.normal(0,.4,(reps,n))[:,li]+rng.normal(0,.4,(reps,s))[:,si]
            noise+=(rng.normal(0,pl,(reps,n))[:,li]+rng.normal(0,ss,(reps,s))[:,si])*t/2
            noise+=rng.normal(0,resid,(reps,m))
            est,se,var=crossed_se(noise,t,si,n,s,True)
            truth_sd=math.sqrt(pl**2/n+ss**2/s+4*resid**2/m)
            for delta in DELTAS:
                estimates=est+delta
                arousal.append(dict(sources=s,listeners=n,pairs_per_listener=6,noise_scenario=label,
                    true_delta_points=delta,simulations=reps,
                    positive_lower95_rate=float(np.mean((estimates-tc*se>0)&(var>0))),
                    two_sided_rejection_rate=float(np.mean((np.abs(estimates)>tc*se)&(var>0))),
                    ci_coverage=float(np.mean((np.abs(est)<=tc*se)&(var>0))),
                    median_ci_halfwidth=float(np.median(tc*se)),mean_estimate=float(np.mean(estimates)),
                    empirical_estimator_sd=float(np.std(est,ddof=1)),known_variance_estimator_sd=truth_sd,
                    median_cluster_se=float(np.median(se)),invalid_variance_fraction=float(np.mean(var<=0)),
                    paired_difference_outside_legal_range=float(np.mean(np.abs(noise+delta*t/2)>6)),
                    known_variance_normal_MDE80_points=(NORMAL.inv_cdf(.975)+NORMAL.inv_cdf(.8))*truth_sd))
        if s in (36,72) and n in (48,96,144,192):
            for q in (.3,.6,.9):
                for sd in (.25,.5):
                    rng=np.random.default_rng(26092840+s*10000+n*10+int(q*100)+int(sd*1000))
                    z=rng.normal(0,sd,(reps,n))[:,li]+rng.normal(0,sd,(reps,s))[:,si]+rng.normal(0,1,(reps,m))
                    total_sd=math.sqrt(1+2*sd**2)
                    for net in (0,.1,.15):
                        mu,threshold=quality_location(net,q,total_sd)
                        y=(z+mu>threshold).astype(float)-(z+mu < -threshold).astype(float)
                        est,se,var=crossed_se(y,t,si,n,s,False)
                        quality.append(dict(sources=s,listeners=n,null_tie_probability=q,latent_listener_source_sd=sd,
                            true_net_preference=net,simulations=reps,observed_tie_fraction=float(np.mean(y==0)),
                            mean_net_preference=float(np.mean(est)),median_ci_halfwidth=float(np.median(tc*se)),
                            ci_coverage=float(np.mean((np.abs(est-net)<=tc*se)&(var>0))),
                            positive_lower95_rate=float(np.mean((est-tc*se>0)&(var>0))),
                            two_sided_rejection_rate=float(np.mean((np.abs(est)>tc*se)&(var>0))),
                            invalid_variance_fraction=float(np.mean(var<=0))))
        print(f'Sources {s}, listeners {n}: allocation and simulations complete',flush=True)
    write_csv(out/'arousal_sensitivity.csv',arousal)
    write_csv(out/'quality_precision.csv',quality)
    null=[r for r in arousal if r['true_delta_points']==0]
    qnull=[r for r in quality if r['true_net_preference']==0]
    verification=dict(status='complete_planning_grid',allocation_count=len(DESIGNS),
        arousal_scenarios=len(arousal),quality_scenarios=len(quality),
        arousal_null_two_sided_range=[min(r['two_sided_rejection_rate'] for r in null),max(r['two_sided_rejection_rate'] for r in null)],
        quality_null_two_sided_range=[min(r['two_sided_rejection_rate'] for r in qnull),max(r['two_sided_rejection_rate'] for r in qnull)],
        maximum_arousal_variance_nonpositive=max(r['invalid_variance_fraction'] for r in arousal),
        maximum_quality_variance_nonpositive=max(r['invalid_variance_fraction'] for r in quality),
        maximum_arousal_analytic_vs_empirical_sd_relative_error=max(abs(r['empirical_estimator_sd']/r['known_variance_estimator_sd']-1) for r in null),
        python=sys.version,numpy=np.__version__,r_version=subprocess.check_output(['Rscript','-e','cat(R.version.string)'],text=True))
    dump(out/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=5000)
    a=p.parse_args();run(a.root,a.output,a.replicates)
