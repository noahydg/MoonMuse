"""Inventory and prepare a conditional combined-study candidate MIDI pool."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import miditoolkit
import mido
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.build_ss26_level_matched_rule import read,write,dump,sha,features,skeleton
from scripts.convert_ss26_events import cap_repeated_notes
from scripts.build_corrected_three_system_population import crop_midi
from scripts.render_v82_global_gain_audio import complete_clip_seconds
from scripts.audit_ss26_outputs import check_midi

DESIGNS=((48,2,4),(48,4,4),(48,4,6),(72,4,4),(96,2,4))


def balance_orientation(rows,seed):
    """Alternate an Euler tour: every listener balanced, every cell differs by <=1."""
    rng=random.Random(seed)
    for block in ('system','within'):
        indices=[i for i,r in enumerate(rows) if r['block']==block]
        edges=[(('listener',rows[i]['listener_slot']),('cell',rows[i]['source_sha256'],rows[i]['target_quadrant']),i) for i in indices]
        counts=Counter(v for _,v,_ in edges)
        edges.extend((('dummy',block),cell,None) for cell,n in counts.items() if n%2)
        adjacency=defaultdict(list)
        for ei,(u,v,_) in enumerate(edges):adjacency[u].append(ei);adjacency[v].append(ei)
        for value in adjacency.values():rng.shuffle(value)
        used=set()
        for start in list(adjacency):
            if all(e in used for e in adjacency[start]):continue
            stack=[(start,None)];tour=[]
            while stack:
                u,_=stack[-1]
                while adjacency[u] and adjacency[u][-1] in used:adjacency[u].pop()
                if adjacency[u]:
                    ei=adjacency[u].pop();used.add(ei);a,b,_=edges[ei]
                    stack.append((b if a==u else a,ei))
                else:
                    _,ei=stack.pop()
                    if ei is not None:tour.append(ei)
            assert len(tour)%2==0
            flip=rng.randrange(2)
            for position,ei in enumerate(tour):
                index=edges[ei][2]
                if index is not None:rows[index]['learning_first']=bool((position+flip)%2)
        assert len(used)==len(edges)


def allocate(sources,n,system_k,within_k,seed=260928):
    """Balanced targets/listener; greedily balance source and orientation exposure."""
    rng=random.Random(seed+n*100+system_k*10+within_k)
    counts=Counter();totals=Counter();orient=Counter();rows=[]
    for listener in range(n):
        used=set();blocks={}
        for kind,k in [('system',system_k),('within',within_k)]:
            if k==2:targets=['Q1','Q3'] if listener%2==0 else ['Q2','Q4']
            elif k==4:targets=['Q1','Q2','Q3','Q4']
            elif k==6:targets=['Q1','Q2','Q3','Q4']+(['Q1','Q3'] if listener%2==0 else ['Q2','Q4'])
            else:raise ValueError(k)
            rng.shuffle(targets);first_counts=Counter();seeds=Counter();trial=[]
            for target in targets:
                positive=target in ('Q1','Q4')
                choices=[r for r in sources if r['pool']==kind and r['source_sha256'] not in used and
                    (r['quadrant']==target if kind=='system' else (r['valence']=='positive')==positive)]
                # Stable random tie-breaking, never based on any response or audio.
                ties={r['source_sha256']:rng.random() for r in choices}
                chosen=min(choices,key=lambda r:(counts[r['source_sha256'],target],totals[r['source_sha256']],seeds[r['seed']],ties[r['source_sha256']]))
                h=chosen['source_sha256'];allowed=[v for v in (True,False) if first_counts[v]<k//2]
                rng.shuffle(allowed);first=min(allowed,key=lambda v:orient[h,target,v])
                trial.append(dict(listener_slot=listener,block=kind,source_sha256=h,seed=chosen['seed'],source_quadrant=chosen['quadrant'],
                    target_quadrant=target,target_sign=1 if target in ('Q1','Q2') else -1,learning_first=first))
                used.add(h);counts[h,target]+=1;totals[h]+=1;orient[h,target,first]+=1;first_counts[first]+=1;seeds[chosen['seed']]+=1
            blocks[kind]=trial
        order=('system','within') if listener%2==0 else ('within','system')
        trial=[r for kind in order for r in blocks[kind]]
        for position,r in enumerate(trial,1):r['position']=position
        rows.extend(trial)
    balance_orientation(rows,seed+n*100+system_k*10+within_k)
    assert len(rows)==n*(system_k+within_k)
    for listener in range(n):
        own=[r for r in rows if r['listener_slot']==listener]
        assert len({r['source_sha256'] for r in own})==len(own)
        for kind,k in [('system',system_k),('within',within_k)]:
            part=[r for r in own if r['block']==kind]
            assert len(part)==k and sum(r['target_sign'] for r in part)==0
            assert sum(r['learning_first'] for r in part)==k//2
            assert sum(r['target_quadrant'] in ('Q1','Q4') for r in part)==k//2
    oc=Counter((r['source_sha256'],r['target_quadrant'],r['learning_first']) for r in rows)
    assert all(abs(oc[h,q,True]-oc[h,q,False])<=1 for h,q,_ in oc)
    return rows


def note_onset_velocity(path):
    m=miditoolkit.MidiFile(str(path))
    return Counter((i,n.start,n.pitch,n.velocity) for i,ins in enumerate(m.instruments) for n in ins.notes if n.start<32*m.ticks_per_beat)


def strict_crop(source,dest):
    report=crop_midi(source,dest,8)
    midi=mido.MidiFile(dest);boundary=32*midi.ticks_per_beat;removed=0
    for track in midi.tracks:
        tick=previous=0;keep=[]
        for msg in track:
            tick+=msg.time
            if msg.type=='note_on' and msg.velocity>0 and tick>=boundary:
                removed+=1;continue
            keep.append(msg.copy(time=tick-previous));previous=tick
        track[:]=keep
    midi.save(dest)
    return dict(report,excluded_boundary_note_onsets=removed)


def prepare(root,out):
    out.mkdir(parents=True,exist_ok=False)
    spec=root/'results/ss26-combined-materials-spec-20260928.md'
    shutil.copyfile(spec,out/'preparation_spec.md')
    roles=read(root/'results/ss26-level-matched-rule-20260928-r2/source_roles.csv')
    sources=[]
    for r in roles:
        if r['role']!='held_out_candidate':continue
        assert 2<=int(r['rank_within_cell'])<=10
        sources.append(dict(r,pool='within' if int(r['rank_within_cell'])<=7 else 'system'))
    assert Counter(r['pool'] for r in sources)=={'within':72,'system':36}
    sources.sort(key=lambda r:(r['pool'],r['seed'],r['quadrant'],int(r['rank_within_cell'])))
    write(out/'sources.csv',sources)
    model_root=root/'results/ss26-matched-alignment-20260927/evidence/ss26-alignment-20260926'
    oldplan=json.loads((root/'results/ss26-comparison-20260927/evidence/ss26-comparison-inputs-20260927/plan.json').read_text())
    model_checks=[]
    for seed in (42,43,44):
        p=model_root/f'seed{seed}/Aplus/phase2/checkpoints/params/best_params.pt'
        actual=sha(p);assert actual==oldplan['model_sha256'][f'Aplus_seed{seed}']
        model_checks.append(dict(seed=seed,path=str(p),sha256=actual,exists=True))
    dump(out/'models.json',model_checks)
    paired={(r['source_sha256'],r['target_quadrant']):r for r in read(root/'results/ss26-per-source-matched-rule-20260928/population_manifest.csv')}
    baseroot=root/'results/corrected-three-system-population-v78-20260809'
    baseline={(r['seed'],r['quadrant'],r['sample_index']):r for r in read(baseroot/'population_manifest.csv') if r['system']=='emo_disentangler'}
    old_study=read(root/'results/v82-human-superiority-study-20260809/master_manifest_private.csv')
    oldslots={(r['seed_private'],r['quadrant_private'],r['sample_index_private']) for r in old_study}
    rows=[]
    for source in sources:
        h=source['source_sha256'];native=source['quadrant'];slot=(source['seed'],native,source['sample_index'])
        assert slot not in oldslots
        original=root/'results/v81-v70-source-freeze-20260809'/source['frozen_path'];assert sha(original)==h
        targets=[native] if source['pool']=='system' else (['Q1','Q4'] if source['valence']=='positive' else ['Q2','Q3'])
        for target in targets:
            match=paired[h,target]
            inputs=[('MoonMuse',Path(match['learned_midi']),match['learned_midi_sha256'])]
            if source['pool']=='within':inputs.append(('R-per-source-matched',Path(match['midi_path']),match['midi_sha256']))
            else:
                b=baseline[slot];inputs.append(('EMO-Disentangler',baseroot/b['standardized_midi'],b['standardized_sha256']))
            for method,path,expected in inputs:
                assert sha(path)==expected
                sid=f'C{len(rows)+1:03d}';capped=out/'capped-midi'/f'{sid}.mid';final=out/'midi'/capped.name
                capped.parent.mkdir(exist_ok=True)
                cropped=out/'cropped-midi'/capped.name
                crop=strict_crop(path,cropped)
                m=miditoolkit.MidiFile(str(cropped));n_changed=cap_repeated_notes(m);m.dump(str(capped))
                assert note_onset_velocity(path)==note_onset_velocity(capped)
                strict_crop(capped,final)
                assert note_onset_velocity(path)==note_onset_velocity(final)
                if method!='EMO-Disentangler':check_midi(original,final)
                final_m=miditoolkit.MidiFile(str(final))
                assert all(0<=n.start<n.end<=32*final_m.ticks_per_beat for i in final_m.instruments for n in i.notes)
                assert all(i.program==0 and not i.is_drum for i in final_m.instruments)
                f=features(final)
                rows.append(dict(stimulus_id=sid,pool=source['pool'],source_sha256=h,seed=source['seed'],source_quadrant=native,sample_index=source['sample_index'],
                    target_quadrant=target,method=method,origin=str(path),origin_sha256=expected,midi_file=str(final.relative_to(out)),midi_sha256=sha(final),
                    repeated_release_changes=n_changed,excluded_boundary_note_onsets=crop['excluded_boundary_note_onsets'],clip_seconds=complete_clip_seconds(final),**f))
    assert len(rows)==360
    write(out/'stimuli.csv',rows)
    lookup={(r['source_sha256'],r['target_quadrant'],r['method']):r for r in rows}
    coverage=[];burden=[]
    for n,sk,wk in DESIGNS:
        allocation=allocate(sources,n,sk,wk)
        for row in allocation:
            other='EMO-Disentangler' if row['block']=='system' else 'R-per-source-matched'
            left=lookup[row['source_sha256'],row['target_quadrant'],'MoonMuse'];right=lookup[row['source_sha256'],row['target_quadrant'],other]
            if not row['learning_first']:left,right=right,left
            row.update(stimulus_a=left['stimulus_id'],stimulus_b=right['stimulus_id'],audio_seconds=float(left['clip_seconds'])+float(right['clip_seconds']))
        write(out/f'allocation_n{n}_s{sk}_w{wk}.csv',allocation)
        times=[sum(r['audio_seconds'] for r in allocation if r['listener_slot']==i)/60 for i in range(n)]
        burden.append(dict(listeners=n,total_pairs=sk+wk,system_pairs=sk,within_pairs=wk,total_comparisons=len(allocation),
            minimum_audio_minutes=min(times),median_audio_minutes=float(np.median(times)),maximum_audio_minutes=max(times),
            selections_per_listener=5*(sk+wk),answers_and_replay_included=False))
        for kind in ('system','within'):
            relevant=[r for r in allocation if r['block']==kind]
            counts=Counter((r['source_sha256'],r['target_quadrant']) for r in relevant)
            expectedcells=36 if kind=='system' else 144
            assert len(counts)==expectedcells
            oc=Counter((r['source_sha256'],r['target_quadrant'],r['learning_first']) for r in relevant)
            coverage.append(dict(listeners=n,total_pairs=sk+wk,block=kind,observations=len(relevant),sources=len({r['source_sha256'] for r in relevant}),
                target_cells=len(counts),minimum_ratings_per_source_target=min(counts.values()),maximum_ratings_per_source_target=max(counts.values()),
                orientation_cells=len(oc),possible_orientation_cells=2*expectedcells))
    write(out/'allocation_coverage.csv',coverage);write(out/'burden.csv',burden)
    report=dict(status='PASS_CANDIDATE_MATERIALS',formalStudyData=False,source_choice_conditional_on_PAD_review=True,
        sources=108,system_slots=36,within_sources=72,stimuli=360,methods=dict(Counter(r['method'] for r in rows)),
        verified_model_hashes=3,overlap_with_v82_slots=0,allocations=len(DESIGNS),new_model_inference=False,additional_GPU_cost_yuan=0,
        common_crop_and_repeat_cap=True,script_sha256=sha(__file__),spec_sha256=sha(spec),
        same_score_matching_note='Within block only; system block matches seed,target,index but compositions differ.')
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REPO.parent);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve(),a.out.resolve())
