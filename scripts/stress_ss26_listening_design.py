#!/usr/bin/env python3
"""Ordinal-scale and missing-pair stresses for prospective SS26 planning only."""
from pathlib import Path
import argparse
import subprocess
import sys
import math
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.plan_ss26_listening_design import read_csv, write_csv, dump, AROUSAL


def estimate_incomplete(y,mask,t,si,n,s,crit):
    reps,m=y.shape
    hi=mask*(t==1);lo=mask*(t==-1)
    nh=hi.sum(axis=1);nl=lo.sum(axis=1)
    assert np.all(nh>0)&np.all(nl>0)
    yh=(y*hi).sum(axis=1)/nh;yl=(y*lo).sum(axis=1)/nl
    residual=y-np.where(t==1,yh[:,None],yl[:,None])
    score=residual*(hi/nh[:,None]-lo/nl[:,None])
    order=np.argsort(si,kind='stable')
    ls=score.reshape(reps,n,6).sum(axis=2)
    ss=score[:,order].reshape(reps,s,m//s).sum(axis=2)
    active_l=(mask.reshape(reps,n,6).sum(axis=2)>0).sum(axis=1)
    active_s=(mask[:,order].reshape(reps,s,m//s).sum(axis=2)>0).sum(axis=1)
    used=mask.sum(axis=1)
    var=(used-1)/(used-2)*(active_l/(active_l-1)*(ls**2).sum(axis=1)
        +active_s/(active_s-1)*(ss**2).sum(axis=1)-used/(used-1)*(score**2).sum(axis=1))
    half=crit[np.minimum(active_l,active_s)-1]*np.sqrt(np.maximum(var,0))
    return yh-yl,half,var


def run(directory,reps=5000):
    out=directory/'ordinal_missingness_stress';out.mkdir(exist_ok=False)
    dump(out/'assumptions_before_simulation.json',dict(replicates=reps,
        designs=[{'sources':72,'listeners':n} for n in (96,144)],noise_scenarios=['middle','higher'],
        latent_deltas=[0,.3,.5],target_centers=[[3,5],[2.5,5.5]],
        shared_listener_and_source_sd=.6,shared_pair_sd=.4,
        missing_pair_probabilities=[0,.1,.2],missingness='MCAR per whole same-target pair; not informative missingness or selective dropout',
        bounded_ratings='round to integer then clip each L/R rating to 1--7 before differencing',
        scope='Sensitivity of the candidate estimator to known departures from Gaussian complete-data planning; no measured listener parameters'))
    crit=np.zeros(145)
    q=subprocess.check_output(['Rscript','-e','cat(paste(qt(.975,1:144),collapse=","))'],text=True)
    crit[1:]=[float(x) for x in q.split(',')]
    results=[]
    for n in (96,144):
        s=72;rows=read_csv(directory/f'allocation_s{s}_n{n}.csv')
        t=np.array([int(r['target_sign']) for r in rows]);si=np.array([int(r['source_index']) for r in rows]);li=np.repeat(np.arange(n),6);m=len(t)
        for k,label in enumerate(('middle','higher')):
            pstd,sstd,rest=AROUSAL[label]
            rng=np.random.default_rng(26092770+n*100+k)
            noise=rng.normal(0,.4,(reps,n))[:,li]+rng.normal(0,.4,(reps,s))[:,si]
            noise+=(rng.normal(0,pstd,(reps,n))[:,li]+rng.normal(0,sstd,(reps,s))[:,si])*t/2
            noise+=rng.normal(0,rest,(reps,m))
            shared=rng.normal(0,.6,(reps,n))[:,li]+rng.normal(0,.6,(reps,s))[:,si]+rng.normal(0,.4,(reps,m))
            missing_u=rng.random((reps,m))
            for center_separation in (2.,3.):
                center=4+center_separation*t/2+shared
                for delta in (0,.3,.5):
                    diff=noise+delta*t/2
                    l=np.clip(np.rint(center+diff/2),1,7);r=np.clip(np.rint(center-diff/2),1,7)
                    y=l-r
                    for missing in (0,.1,.2):
                        mask=missing_u>=missing
                        estimate,half,var=estimate_incomplete(y,mask,t,si,n,s,crit)
                        results.append(dict(sources=s,listeners=n,noise_scenario=label,
                            latent_center_high_minus_low=center_separation,latent_delta=delta,missing_pair_probability=missing,
                            observed_scale_mean_delta=float(estimate.mean()),positive_lower95_rate=float(np.mean((estimate>half)&(var>0))),
                            two_sided_rejection_rate=float(np.mean((np.abs(estimate)>half)&(var>0))),
                            median_ci_halfwidth=float(np.median(half)),
                            invalid_variance_fraction=float(np.mean(var<=0)),
                            proportion_scale_endpoints=float(np.mean((l==1)|(l==7)|(r==1)|(r==7))),simulations=reps))
            print(f'Ordinal/missingness: N={n}, noise={label}',flush=True)
    write_csv(out/'sensitivity.csv',results)
    null=[r for r in results if r['latent_delta']==0]
    dump(out/'verification.json',dict(status='sensitivity_complete',scenarios=len(results),
        null_type_I_two_sided_range=[min(r['two_sided_rejection_rate'] for r in null),max(r['two_sided_rejection_rate'] for r in null)],
        maximum_invalid_variance_fraction=max(r['invalid_variance_fraction'] for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--replicates',type=int,default=5000)
    a=p.parse_args();run(a.directory,a.replicates)
