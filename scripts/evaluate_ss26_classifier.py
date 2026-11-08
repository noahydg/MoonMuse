#!/usr/bin/env python3
"""Apply the preserved official EMOPIA classifier to all SS26 playback outputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.analyze_ss26_comparison import read_csv, write_csv, sha, group_descriptions, paired_gaps, contrast_gaps
from scripts.build_corrected_three_system_population import crop_midi
from scripts.run_official_emopia_classifier_v78 import OfficialRemiClassifier, TASKS, target_for, summarize, official_commit
import torch


def evaluate(catalog_path, legacy, upstream, output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    catalog=read_csv(catalog_path)
    reference=json.loads((legacy/'status.json').read_text())
    assert official_commit(upstream)==reference['classifier_commit']
    assets={}
    for task in TASKS:
        ref=reference['tasks'][task]['classifier']
        assets[task]={}
        for name in ('checkpoint','config','dictionary'):
            path=Path(ref[name]);assert sha(path)==ref[name+'_sha256']
            assets[task][name]=dict(path=str(path),sha256=sha(path))
    (output/'assets.json').write_text(json.dumps(dict(status='PASS',assets=assets,commit=official_commit(upstream),torch=torch.__version__,python=sys.version),indent=2)+'\n')
    standardized=[]
    for i,r in enumerate(catalog):
        src=Path(r['midi_path']);assert sha(src)==r['midi_sha256']
        dst=output/'eight_bar_midi'/r['method']/(r['source_sha256'][:16]+'_'+r['target_quadrant']+'.mid')
        rep=crop_midi(src,dst,bars=8)
        assert rep['source_note_count']==rep['standardized_note_count']
        standardized.append(dict(r,classifier_midi=str(dst),classifier_midi_sha256=sha(dst),**rep))
    write_csv(output/'population_manifest.csv',standardized)
    reproduction=[];report={};started=time.time()
    for task in TASKS:
        classifier=OfficialRemiClassifier(upstream,task,'cpu')
        previous=read_csv(legacy/'predictions'/('emopia_test_eight_bar_'+task+'.csv'))
        # One fixed first-listed real example per quadrant; no new-result selection.
        for q in ('Q1','Q2','Q3','Q4'):
            r=next(r for r in previous if r['quadrant']==q)
            p=Path(r['midi_path']);assert sha(p)==r['midi_sha256']
            predicted,logits,probabilities,count=classifier.predict(p)
            max_error=max(abs(v-float(r['probability_'+label])) for label,v in zip(classifier.labels,probabilities))
            assert predicted==r['prediction'] and count==int(r['token_count']) and max_error<1e-6
            reproduction.append(dict(task=task,quadrant=q,sample_id=r['sample_id'],max_probability_error=max_error,status='PASS'))
        (output/'legacy_reproduction.json').write_text(json.dumps(reproduction,indent=2)+'\n')
        predictions=[]
        for i,r in enumerate(standardized,1):
            p=Path(r['classifier_midi']);assert sha(p)==r['classifier_midi_sha256']
            predicted,logits,probabilities,count=classifier.predict(p)
            target=target_for(task,r['target_quadrant'])
            row={k:r[k] for k in ('source_sha256','seed','valence','method','target_quadrant')}
            row.update(task=task,target=target,prediction=predicted,correct=int(target==predicted),token_count=count,
                       classifier_midi_sha256=r['classifier_midi_sha256'])
            row.update({'probability_'+k:v for k,v in zip(classifier.labels,probabilities)})
            row.update({'logit_'+k:v for k,v in zip(classifier.labels,logits)})
            row['target_probability']=row['probability_'+target]
            predictions.append(row)
            if i%100==0 or i==len(standardized):
                write_csv(output/(task+'_predictions.csv'),predictions)
                (output/'progress.json').write_text(json.dumps(dict(task=task,completed=i,total=len(standardized),elapsed_seconds=time.time()-started))+'\n')
                print(task,i,len(standardized),'elapsed',round(time.time()-started),flush=True)
        groups=defaultdict(list)
        for r in predictions:groups[r['method']].append(r)
        report[task]=dict(methods={method:summarize(rows,classifier.labels) for method,rows in groups.items()},
                          target_levels=group_descriptions(predictions,('method','target_quadrant'),('correct','target_probability')))
        if task=='arousal':
            gaps=paired_gaps(predictions,('probability_HA',))
            contrasts=contrast_gaps(gaps,('probability_HA',))
            write_csv(output/'arousal_source_gaps.csv',gaps)
            write_csv(output/'arousal_source_gap_contrasts.csv',contrasts)
            report[task].update(paired_gaps=group_descriptions(gaps,('method',),('probability_HA',)),
                                paired_gaps_by_valence=group_descriptions(gaps,('method','valence'),('probability_HA',)),
                                gap_contrasts=group_descriptions(contrasts,('contrast',),('probability_HA',)),
                                gap_contrasts_by_seed=group_descriptions(contrasts,('contrast','seed'),('probability_HA',)))
    (output/'summary.json').write_text(json.dumps(dict(status='PASS',files=len(catalog),sources=154,tasks=report,
        elapsed_seconds=time.time()-started,descriptive_only=True,formal_study_data=False),indent=2)+'\n')
    print('CLASSIFIER_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','legacy','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:evaluate(a.catalog,a.legacy,a.upstream,a.output)
    except Exception as e:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=repr(e),time=time.time()),indent=2)+'\n')
        raise
