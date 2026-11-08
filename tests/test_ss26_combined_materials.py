from collections import Counter
import mido
import numpy as np
from scripts.prepare_ss26_combined_materials import allocate,strict_crop,DESIGNS
from scripts.simulate_ss26_combined_precision import estimate


def sources():
    return [dict(source_sha256=f'{s}_{q}_{j}',seed=str(s),quadrant=q,valence='positive' if q in ('Q1','Q4') else 'negative',
        pool='within' if j<6 else 'system') for s in (42,43,44) for q in ('Q1','Q2','Q3','Q4') for j in range(9)]


def test_allocations_cover_both_targets_and_balance_order():
    for n,sk,wk in DESIGNS:
        rows=allocate(sources(),n,sk,wk)
        for kind,expected in [('system',36),('within',144)]:
            c=Counter((r['source_sha256'],r['target_quadrant']) for r in rows if r['block']==kind)
            assert len(c)==expected
            assert max(c.values())-min(c.values())<=1
            ab=Counter((r['source_sha256'],r['target_quadrant'],r['learning_first']) for r in rows if r['block']==kind)
            assert all(abs(ab[h,q,True]-ab[h,q,False])<=1 for h,q in c)


def test_strict_crop_excludes_next_bar_onset_and_retains_boundary_release(tmp_path):
    m=mido.MidiFile(ticks_per_beat=480);t=mido.MidiTrack();m.tracks.append(t)
    t.extend([mido.Message('note_on',note=60,velocity=70,time=0),
        mido.Message('note_on',note=64,velocity=70,time=15360),
        mido.Message('note_off',note=60,time=480),mido.Message('note_off',note=64,time=0)])
    src=tmp_path/'src.mid';dst=tmp_path/'dst.mid';m.save(src)
    report=strict_crop(src,dst);assert report['excluded_boundary_note_onsets']==1
    tick=0;notes=[];off=[]
    for msg in mido.MidiFile(dst).tracks[0]:
        tick+=msg.time
        if msg.type=='note_on' and msg.velocity>0:notes.append((tick,msg.note))
        if msg.type=='note_off':off.append((tick,msg.note))
    assert notes==[(0,60)] and (15360,60) in off and tick==15360


def test_weighted_source_contrast_cancels_source_intercepts_under_unequal_exposure():
    rows=[r for r in allocate(sources(),48,2,4) if r['block']=='within']
    ids={h:i for i,h in enumerate(sorted({r['source_sha256'] for r in rows}))}
    si=np.array([ids[r['source_sha256']] for r in rows]);li=np.array([r['listener_slot'] for r in rows]);t=np.array([r['target_sign'] for r in rows])
    y=(si*.05+.7*t/2)[None,:]
    est,_,_=estimate(y,li,si,t,True)
    assert abs(est[0]-.7)<1e-12
