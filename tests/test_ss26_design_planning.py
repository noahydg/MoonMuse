import numpy as np

from scripts.plan_ss26_listening_design import allocation, crossed_se, quality_location, NORMAL
from scripts.stress_ss26_listening_design import estimate_incomplete


def sources(count=36):
    return [dict(seed=str(seed),quadrant=q,source_quadrant=q,
                 valence='positive' if q in ('Q1','Q4') else 'negative',
                 source_sha256=f'{seed}-{q}-{i}')
            for seed in (42,43,44) for q in ('Q1','Q2','Q3','Q4') for i in range(count//12)]


def test_balanced_estimator_cancels_arbitrary_listener_and_score_intercepts():
    rows=allocation(sources(),48)
    t=np.array([r['target_sign'] for r in rows]);si=np.array([r['source_index'] for r in rows]);li=np.array([r['listener_slot'] for r in rows])
    y=(100*li+17*si+.5*t/2)[None,:]
    est,_,_=crossed_se(y,t,si,48,36,True)
    assert np.allclose(est,.5)


def test_crossed_random_slope_variance_has_nonvanishing_source_component():
    rows=allocation(sources(),48);t=np.array([r['target_sign'] for r in rows]);si=np.array([r['source_index'] for r in rows]);li=np.array([r['listener_slot'] for r in rows])
    rng=np.random.default_rng(4401);reps=12000
    y=(rng.normal(0,.6,(reps,48))[:,li]+rng.normal(0,.8,(reps,36))[:,si])*t/2+rng.normal(0,1.2,(reps,len(t)))
    estimate=2*(y*t).mean(axis=1)
    expected=.6**2/48+.8**2/36+4*1.2**2/len(t)
    assert abs(np.var(estimate,ddof=1)/expected-1)<.04


def test_ternary_quality_mean_uses_ties_in_denominator():
    for tie in (.3,.6,.9):
        mu,threshold=quality_location(.1,tie,1.2)
        mean=1-NORMAL.cdf((threshold-mu)/1.2)-NORMAL.cdf((-threshold-mu)/1.2)
        assert abs(mean-.1)<1e-10
        mu,threshold=quality_location(0,tie,1.2)
        assert abs(mu)<1e-10
        assert abs(NORMAL.cdf(threshold/1.2)-NORMAL.cdf(-threshold/1.2)-tie)<1e-10


def test_incomplete_estimator_matches_balanced_formula_when_nothing_missing():
    rows=allocation(sources(),48);t=np.array([r['target_sign'] for r in rows]);si=np.array([r['source_index'] for r in rows])
    y=np.random.default_rng(18).normal(size=(20,len(rows)))
    a,se,v=crossed_se(y,t,si,48,36,True)
    b,half,w=estimate_incomplete(y,np.ones_like(y,dtype=bool),t,si,48,36,np.ones(49))
    assert np.allclose(a,b) and np.allclose(v,w) and np.allclose(se,half)
