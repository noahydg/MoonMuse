"""Approximate planning sensitivity for the actual two-block assignments."""
from collections import Counter
import argparse
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.build_ss26_level_matched_rule import read,write,dump
from scripts.prepare_ss26_combined_materials import DESIGNS
from scripts.plan_ss26_listening_design import AROUSAL


def grouped_sum(values,ids):
    order=np.argsort(ids,kind='stable');ordered=ids[order]
    starts=np.r_[0,1+np.flatnonzero(ordered[1:]!=ordered[:-1])]
    return np.add.reduceat(values[:,order],starts,axis=1)


def coefficients(si,t,interaction):
    s=len(set(si));cells=Counter(zip(si,t)) if interaction else Counter(si)
    if interaction:
        assert len(cells)==2*s
        w=np.array([1/cells[a,b] for a,b in zip(si,t)])
        a=t*w/s
    else:
        w=np.array([1/cells[a] for a in si]);a=w/s
    return a,w


def estimate(y,li,si,t,interaction):
    a,w=coefficients(si,t,interaction);est=y@a
    mean=y@w/w.sum()
    residual=y-mean[:,None]-(est[:,None]*t/2 if interaction else 0)
    score=residual*a
    l,s=len(set(li)),len(set(si));m=len(li);k=2 if interaction else 1
    variance=(m-1)/(m-k)*(l/(l-1)*(grouped_sum(score,li)**2).sum(1)+s/(s-1)*(grouped_sum(score,si)**2).sum(1)-m/(m-1)*(score**2).sum(1))
    return est,np.sqrt(np.maximum(variance,0)),variance


def run(folder,reps):
    out=folder/'planning_sensitivity';out.mkdir(exist_ok=False)
    dump(out/'assumptions_before_simulation.json',dict(replicates=reps,seed=26092861,formalStudyData=False,
        genuine_responses_used=False,effects=[0,.2,.3,.5],variance_scenarios=AROUSAL,
        within='Source-equal difference of high-minus-low responses between learned and model-assisted rule; paired difference with listener/source intercepts SD.4 and random target slopes.',
        system='Source-slot-equal mean target-signed arousal rating difference between systems; Gaussian listener/source intercepts and residual. Not legacy binary recognition accuracy.',
        covariance='Two-way listener+source-intersection sandwich; weighted source-target means for unequal exposure; df=min(cluster counts)-1.',
        confidence=[.95,.975],family='97.5% per-comparison intervals illustrate Bonferroni for two arousal comparisons; endpoints and analysis not frozen.',
        boundary='Gaussian planning approximation only; no quality/ordinal/missingness guarantee. Changed comparator effect is unknown.'))
    records=[]
    for di,(n,sk,wk) in enumerate(DESIGNS):
        allocation=read(folder/f'allocation_n{n}_s{sk}_w{wk}.csv')
        for kind in ('system','within'):
            rows=[r for r in allocation if r['block']==kind];interaction=kind=='within'
            labels=sorted({r['source_sha256'] for r in rows});lookup={h:i for i,h in enumerate(labels)}
            li=np.array([int(r['listener_slot']) for r in rows]);si=np.array([lookup[r['source_sha256']] for r in rows]);t=np.array([int(r['target_sign']) for r in rows]);s=len(labels)
            a,w=coefficients(si,t,interaction)
            crit=[float(x) for x in subprocess.check_output(['Rscript','-e',f'cat(paste(qt(c(.975,.9875),{min(n,s)-1}),collapse=","))'],text=True).strip().split(',')]
            for ki,(label,(ls,ss,rs)) in enumerate(AROUSAL.items()):
                rng=np.random.default_rng(26092861+1000*di+100*interaction+ki)
                if interaction:
                    noise=rng.normal(0,.4,(reps,n))[:,li]+rng.normal(0,.4,(reps,s))[:,si]
                    noise+=(rng.normal(0,ls,(reps,n))[:,li]+rng.normal(0,ss,(reps,s))[:,si])*t/2
                    variance=.4**2*np.square(np.bincount(li,weights=a)).sum()+ls**2*np.square(np.bincount(li,weights=a*t/2)).sum()+ss**2/s+rs**2*np.square(a).sum()
                else:
                    noise=rng.normal(0,ls,(reps,n))[:,li]+rng.normal(0,ss,(reps,s))[:,si]
                    variance=ls**2*np.square(np.bincount(li,weights=a)).sum()+ss**2/s+rs**2*np.square(a).sum()
                noise+=rng.normal(0,rs,(reps,len(rows)))
                est,se,v=estimate(noise,li,si,t,interaction)
                for confidence,c in zip((.95,.975),crit):
                    for effect in (0,.2,.3,.5):
                        records.append(dict(listeners=n,total_pairs=sk+wk,system_pairs=sk,within_pairs=wk,block=kind,sources=s,observations=len(rows),scenario=label,
                            confidence=confidence,effect=effect,positive_lower_bound_fraction=float(np.mean((est+effect-c*se>0)&(v>0))),
                            two_sided_rejection=float(np.mean((np.abs(est+effect)>c*se)&(v>0))),median_halfwidth=float(np.median(c*se)),
                            invalid_variance_fraction=float(np.mean(v<=0)),analytic_sd=math.sqrt(variance),empirical_sd=float(np.std(est,ddof=1))))
    write(out/'sensitivity.csv',records)
    null=[r for r in records if r['effect']==0 and r['confidence']==.95]
    report=dict(status='PLANNING_APPROXIMATION_COMPLETE',scenarios=len(records),reps=reps,
        null_95_two_sided_range=[min(r['two_sided_rejection'] for r in null),max(r['two_sided_rejection'] for r in null)],
        maximum_invalid_variance=max(r['invalid_variance_fraction'] for r in records),
        maximum_empirical_sd_relative_error=max(abs(r['empirical_sd']/r['analytic_sd']-1) for r in records))
    dump(out/'verification.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--replicates',type=int,default=5000)
    args=p.parse_args();run(args.folder,args.replicates)
