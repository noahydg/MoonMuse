"""Prospective allocation and ternary preference sensitivity on actual SS26 audio.

No human responses are fitted. The simulation is planning evidence, not a
guarantee or a frozen recruitment target. The source alternatives are explicit.
"""
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_ss26_combined_materials import balance_orientation
from scripts.plan_ss26_listening_design import quality_location
from scripts.build_ss26_level_matched_rule import read, write, dump, sha

DESIGNS = [(n,k,2) for n in (36,48,60,72,96) for k in (4,6,8)]
NOISE = {'lower':(.15,.15,.10), 'middle':(.35,.35,.25), 'higher':(.65,.65,.40)}
EFFECTS = (0., .10, .20, .30)


def source_pools(sources, name):
    if name == 'strict21':
        primary = [r for r in sources if r['in_strict21']]
        system = [r for r in sources if r['expanded_pool'] == 'system' and not r['in_strict21']]
        assert len(primary) == 21 and len(system) == 19
    elif name == 'expanded42':
        primary = [r for r in sources if r['expanded_pool'] == 'within']
        system = [r for r in sources if r['expanded_pool'] == 'system']
        assert len(primary) == 42 and len(system) == 24
    else: raise ValueError(name)
    assert not {r['plan_id'] for r in primary} & {r['plan_id'] for r in system}
    return primary, system


def allocate(primary, system, n, within_k, system_k=2, seed=261017):
    rng = random.Random(seed+n*100+within_k*10+len(primary))
    counts, totals = Counter(), Counter(); rows = []
    for listener in range(n):
        used = set(); blocks = {}
        for block, bank, k in [('within',primary,within_k), ('system',system,system_k)]:
            if k == 2: targets = ['Q1','Q3'] if listener%2 == 0 else ['Q2','Q4']
            elif k == 4: targets = ['Q1','Q2','Q3','Q4']
            elif k == 6: targets = ['Q1','Q2','Q3','Q4']+(['Q1','Q3'] if listener%2 == 0 else ['Q2','Q4'])
            elif k == 8: targets = ['Q1','Q2','Q3','Q4']*2
            else: raise ValueError(k)
            rng.shuffle(targets); own = []; seed_counts = Counter()
            for q in targets:
                choices = [r for r in bank if r['plan_id'] not in used and
                           (r['quadrant'] == q if block == 'system' else
                            (r['valence'] == 'positive') == (q in ('Q1','Q4')))]
                ties = {r['plan_id']:rng.random() for r in choices}
                r = min(choices, key=lambda r:(counts[block,r['plan_id'],q],
                        totals[block,r['plan_id']],seed_counts[r['seed']],ties[r['plan_id']]))
                own.append(dict(listener_slot=listener,block=block,source_sha256=r['source_sha256'],
                    plan_id=r['plan_id'],seed=r['seed'],source_quadrant=r['quadrant'],target_quadrant=q,
                    target_sign=1 if q in ('Q1','Q2') else -1,learning_first=False))
                counts[block,r['plan_id'],q] += 1; totals[block,r['plan_id']] += 1
                used.add(r['plan_id']); seed_counts[r['seed']] += 1
            blocks[block] = own
        order = ('within','system') if listener%2 == 0 else ('system','within')
        for position,r in enumerate([r for b in order for r in blocks[b]],1):
            r['position'] = position; rows.append(r)
    balance_orientation(rows, seed+n*100+within_k*10+len(primary))
    check_allocation(rows, primary, system, n, within_k, system_k)
    return rows


def check_allocation(rows, primary, system, n, wk, sk):
    assert len(rows) == n*(wk+sk)
    for listener in range(n):
        own = [r for r in rows if r['listener_slot'] == listener]
        assert len(own) == len({r['plan_id'] for r in own}) == wk+sk
        for block,k in [('within',wk),('system',sk)]:
            part = [r for r in own if r['block'] == block]
            assert len(part) == k and sum(r['target_sign'] for r in part) == 0
            assert sum(r['target_quadrant'] in ('Q1','Q4') for r in part) == k//2
            assert sum(r['learning_first'] for r in part) == k//2
    for block,bank in [('within',primary),('system',system)]:
        part = [r for r in rows if r['block'] == block]
        expected = {(r['source_sha256'],q) for r in bank for q in
                    ([r['quadrant']] if block == 'system' else
                     (['Q1','Q4'] if r['valence'] == 'positive' else ['Q2','Q3']))}
        assert {(r['source_sha256'],r['target_quadrant']) for r in part} == expected
        oc = Counter((r['source_sha256'],r['target_quadrant'],r['learning_first']) for r in part)
        assert all(abs(oc[h,q,True]-oc[h,q,False]) <= 1 for h,q in expected)
        assert Counter(r['target_quadrant'] for r in part).values()
        qs = Counter(r['target_quadrant'] for r in part)
        assert len(set(qs.values())) == 1


def arrays(rows):
    plans = sorted({r['plan_id'] for r in rows}); smap = {p:i for i,p in enumerate(plans)}
    cells = sorted({(r['plan_id'],r['target_quadrant']) for r in rows})
    cmap = {p:i for i,p in enumerate(cells)}
    nq = Counter(q for _,q in cells)
    return dict(li=np.array([int(r['listener_slot']) for r in rows]),
                si=np.array([smap[r['plan_id']] for r in rows]),
                ci=np.array([cmap[r['plan_id'],r['target_quadrant']] for r in rows]),
                qi=np.array([int(r['target_quadrant'][1:])-1 for r in rows]),
                t=np.array([r['target_sign'] for r in rows]),
                cw=np.array([1/(4*nq[q]) for _,q in cells]),
                cells=cells, sources=len(plans))


def group_sum(y, ids):
    order = np.argsort(ids, kind='stable'); ordered = ids[order]
    start = np.r_[0, 1+np.flatnonzero(ordered[1:] != ordered[:-1])]
    return np.add.reduceat(y[:,order], start, axis=1)


def estimate(y, observed, a):
    """Equal target weight, then equal source weight within target.

    Four target means are nuisance-adjusted. Two-way listener + source -
    intersection HC1 sandwich; no repeated listener-source intersections.
    Empty source-target cells or non-positive variances are not accepted.
    """
    counts = group_sum(observed.astype(float), a['ci'])
    complete = np.all(counts > 0, axis=1)
    weights = observed*a['cw'][a['ci']]/np.maximum(counts[:,a['ci']],1)
    weighted = weights*y
    estimate = weighted.sum(axis=1)
    target_mean = np.stack([weighted[:,a['qi']==q].sum(axis=1)*4 for q in range(4)],axis=1)
    score = weights*(y-target_mean[:,a['qi']])
    ls = group_sum(score, a['li']); ss = group_sum(score, a['si'])
    nl = (group_sum(observed.astype(float),a['li']) > 0).sum(axis=1)
    ns = (group_sum(observed.astype(float),a['si']) > 0).sum(axis=1)
    m = observed.sum(axis=1); k = 4
    variance = ((m-1)/(m-k)*(nl/(nl-1)*(ls**2).sum(axis=1)+
                 ns/(ns-1)*(ss**2).sum(axis=1)) - m/(m-k)*(score**2).sum(axis=1))
    return estimate, np.sqrt(np.maximum(variance,0)), complete & (variance>0), complete, variance


def simulate(rows, n, scenario, effect, tie_null, unable, reps, seed, critical):
    a = arrays(rows); s = a['sources']; ls,ss,ts = NOISE[scenario]
    total_sd = math.sqrt(1+ls**2+ss**2+ts**2)
    location,threshold = quality_location(effect,tie_null,total_sd)
    rng = np.random.default_rng(seed)
    ests=[]; ses=[]; goods=[]; empty=[]; variances=[]; tie_sum=0.; all_sum=0
    for start in range(0,reps,250):
        b = min(250,reps-start)
        z = (rng.normal(0,ls,(b,n))[:,a['li']] + rng.normal(0,ss,(b,s))[:,a['si']] +
             rng.normal(0,ts,(b,s))[:,a['si']]*a['t'] + rng.normal(0,1,(b,len(rows))) + location)
        y = (z>threshold).astype(float)-(z < -threshold).astype(float)
        observed = rng.random(y.shape) >= unable
        e,se,good,complete,v = estimate(y,observed,a)
        ests.append(e); ses.append(se); goods.append(good); empty.append(~complete); variances.append(v)
        tie_sum += ((y==0)&observed).sum(); all_sum += observed.sum()
    est=np.concatenate(ests); se=np.concatenate(ses); good=np.concatenate(goods)
    empty=np.concatenate(empty); variance=np.concatenate(variances)
    positive = good & (est-critical*se>0)
    reject = good & (np.abs(est)>critical*se)
    coverage = good & (np.abs(est-effect)<=critical*se)
    return dict(scenario=scenario,true_net_preference=effect,preference_credit=(effect+1)/2,
        null_tie_probability=tie_null,unable_probability=unable,replicates=reps,
        observed_tie_fraction=tie_sum/all_sum,positive_lower95_fraction=float(positive.mean()),
        positive_MCSE=math.sqrt(positive.mean()*(1-positive.mean())/reps),
        two_sided_rejection=float(reject.mean()),coverage95=float(coverage.mean()),
        median_halfwidth=float(np.median(critical*se[good])),mean_estimate=float(est[good].mean()),
        missing_cell_fraction=float(empty.mean()),nonpositive_variance_fraction=float((variance<=0).mean()),
        empirical_sd=float(est[good].std(ddof=1)),median_se=float(np.median(se[good])),critical=critical)


def prepare(root,out,reps):
    out.mkdir(parents=True,exist_ok=False)
    material=root/'results/ss26-new-score-audio-candidates-20261003-v2'
    sources=json.loads((root/'results/ss26-new-score-materials-20261003/sources.json').read_text())
    stimuli=read(material/'stimuli.csv'); audio=read(material/'render/audio_verification.csv')
    au={r['stimulus_id']:r for r in audio}
    lookup={(r['source_sha256'],r['target_quadrant'],r['method']):r for r in stimuli}
    assert all(sha(material/r['audio_file'])==r['audio_sha256'] for r in audio)
    dump(out/'assumptions_before_simulation.json',dict(date='2026-10-17',formalStudyData=False,
        status='DRAFT_DESIGN_NOT_RECRUITMENT_FREEZE',genuine_responses_used=False,replicates=reps,
        sources=['strict21 plus disjoint system19','expanded42 plus system24 (prior sibling exposure retained)'],
        primary='Mean quality score +1 learned win, 0 tie, -1 rule win; unable is missing, not tie.',
        weighting='Each target quadrant 1/4; each plan equally weighted within its target.',
        secondary='System comparison and arousal/valence are secondary; no automatic confirmatory superiority claim.',
        effects=EFFECTS,noise_scenarios=NOISE,latent_residual_sd=1,
        source_target_slope='Independent source random slope times target sign.',
        null_tie_probability=.3,unable_probability=.05,
        alternative_ties='Ternary probit threshold fixed at null; actual tie fraction may change with true effect.',
        effect_calibration='Location solved analytically under marginal Gaussian latent distribution; not fitted to author ratings.',
        analysis='Four nuisance target means; weighted source-cell contrast with listener/source minus intersection HC1 sandwich; t min(listener,source)-1.',
        calibration='Report null rejection, coverage, variance failures, missing cells; failed iterations count as no detection.',
        design_grid=DESIGNS,meaningful_difference='No validated SESOI: net .1/.2/.3 are transparent planning scenarios, not empirically established meaningful margins.',
        sample_size='N is analyzable independent listeners. Whole-listener loss shown separately, not hidden inside N.',
        code_sha256=sha(Path(__file__)),midi_manifest_sha256=sha(material/'stimuli.csv')))
    dfs=sorted({20,35,41})
    vals=subprocess.check_output(['Rscript','-e','cat(paste(qt(.975,c('+','.join(map(str,dfs))+')),collapse=","))'],text=True)
    tc=dict(zip(dfs,map(float,vals.strip().split(','))))
    burden=[]; coverage=[]; simulations=[]; summary_sources=[]
    for pi,pool in enumerate(('strict21','expanded42')):
        primary,system=source_pools(sources,pool)
        write(out/f'{pool}_primary_sources.csv',primary); write(out/f'{pool}_system_sources.csv',system)
        summary_sources.append(dict(pool=pool,primary_plans=len(primary),system_plans=len(system),
            primary_sibling_exposed=sum(r['plan_history']!='H0-only' for r in primary),
            system_sibling_exposed=sum(r['plan_history']!='H0-only' for r in system),source_overlap=0))
        for di,(n,wk,sk) in enumerate(DESIGNS):
            rows=allocate(primary,system,n,wk,sk)
            for r in rows:
                learned=lookup[r['source_sha256'],r['target_quadrant'],'MoonMuse']
                other=lookup[r['source_sha256'],r['target_quadrant'],
                             'R-per-source-matched' if r['block']=='within' else 'EMO-Disentangler']
                left,right=(learned,other) if r['learning_first'] else (other,learned)
                r.update(stimulus_a=left['stimulus_id'],stimulus_b=right['stimulus_id'],
                    audio_a=au[left['stimulus_id']]['audio_file'],audio_b=au[right['stimulus_id']]['audio_file'],
                    audio_seconds=float(au[left['stimulus_id']]['duration_seconds'])+float(au[right['stimulus_id']]['duration_seconds']))
            write(out/f'allocation_{pool}_n{n}_w{wk}_s{sk}.csv',rows)
            times=[sum(r['audio_seconds'] for r in rows if r['listener_slot']==l)/60 for l in range(n)]
            burden.append(dict(pool=pool,listeners=n,total_pairs=wk+sk,within_pairs=wk,system_pairs=sk,
                pure_audio_min=min(times),pure_audio_median=float(np.median(times)),pure_audio_max=max(times),
                answers_per_listener=5*(wk+sk),recruit_if_10pct_loss=math.ceil(n/.9),recruit_if_20pct_loss=math.ceil(n/.8)))
            for block in ('within','system'):
                part=[r for r in rows if r['block']==block]
                counts=Counter((r['plan_id'],r['target_quadrant']) for r in part)
                coverage.append(dict(pool=pool,listeners=n,total_pairs=wk+sk,block=block,
                    source_plans=len({r['plan_id'] for r in part}),target_cells=len(counts),
                    observations=len(part),min_exposures=min(counts.values()),max_exposures=max(counts.values())))
            primary_rows=[r for r in rows if r['block']=='within']
            critical=tc[min(n,len(primary))-1]
            for vi,scenario in enumerate(NOISE):
                for ei,effect in enumerate(EFFECTS):
                    result=simulate(primary_rows,n,scenario,effect,.3,.05,reps,
                        26101700+pi*100000+di*1000+vi*100+ei,critical)
                    simulations.append(dict(pool=pool,listeners=n,total_pairs=wk+sk,within_pairs=wk,
                        source_plans=len(primary),**result))
            print(f'{pool}: {n} listeners x {wk}+{sk} pairs complete',flush=True)
    write(out/'source_policy_comparison.csv',summary_sources)
    write(out/'burden.csv',burden);write(out/'coverage.csv',coverage);write(out/'quality_simulations.csv',simulations)
    sensitivity=[]
    for n in (48,72):
        primary,system=source_pools(sources,'strict21')
        rows=[r for r in allocate(primary,system,n,6) if r['block']=='within']
        for ti,(tie,unable) in enumerate([(.1,.05),(.5,.05),(.3,0.),(.3,.15)]):
            for ei,effect in enumerate((0.,.2)):
                sensitivity.append(dict(pool='strict21',listeners=n,total_pairs=8,within_pairs=6,source_plans=21,
                    **simulate(rows,n,'middle',effect,tie,unable,reps,26101800+n*100+ti*10+ei,tc[20])))
    write(out/'missing_and_tie_sensitivity.csv',sensitivity)
    null=[r for r in simulations+sensitivity if r['true_net_preference']==0]
    result=dict(status='COMPLETE_PLANNING_SIMULATION',formalStudyData=False,allocations=30,
        simulation_cases=len(simulations)+len(sensitivity),replicates_per_case=reps,
        null_rejection_range=[min(r['two_sided_rejection'] for r in null),max(r['two_sided_rejection'] for r in null)],
        max_nonpositive_variance=max(r['nonpositive_variance_fraction'] for r in simulations+sensitivity),
        max_missing_cell_fraction=max(r['missing_cell_fraction'] for r in simulations+sensitivity),
        no_recruitment_or_formal_freeze=True)
    dump(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REPO.parent)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--replicates',type=int,default=3000)
    a=p.parse_args();prepare(a.root.resolve(),a.out.resolve(),a.replicates)
