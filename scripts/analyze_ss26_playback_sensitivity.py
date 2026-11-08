"""Read-only fixed-cohort reanalysis; outputs contain aggregates, never respondent IDs."""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True, help='Original study workspace containing results/')
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
ROOT=args.root.resolve()
OUT=args.output.resolve()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_ss26_returnee_final import estimate

snapshot=ROOT/'results/ss26-returnee-feedback-review-20261029/response-snapshot-2026-10-29T19-55-11+0800.json'
rows=json.loads((ROOT/'results/ss26-questionnaire-analysis-n69-20261029/decoded-responses.json').read_text())['decoded_rows']
sessions={s['participant_code']:s for s in json.loads(snapshot.read_text())['payload']['sessions'] if s['status']=='completed'}
alloc={(int(r['listener_slot'])+1,int(r['position'])):r for r in csv.DictReader((ROOT/'results/ss26-returning-cohort-prep-20261019/allocation_strict21_n78_w6_s2.csv').open())}
durations={r['stimulus_id']:float(r['duration_seconds']) for r in csv.DictReader((ROOT/'results/ss26-new-score-audio-candidates-20261003-v2/render/audio_verification.csv').open())}
for r in rows:
    a=sessions[r['participant']]['answers'][r['pair_id']]
    al=alloc[r['form_slot'],int(r['pair_id'][1:])]
    r['minimum_cumulative_fraction']=min(a['playback'][s]['played_seconds']/durations[al['stimulus_'+s.lower()]] for s in ('A','B'))

def outcomes(r):
    a,b=r['model_arousal'],r['other_arousal'];s=2*r['target_sign']
    return {'model':None if a is None else s*a,'other':None if b is None else s*b,
            'difference':None if a is None or b is None else s*(a-b),
            'preference':None if r['choice']=='unable' else 0 if r['choice']=='tie' else 1 if r['choice']=='MoonMuse' else -1}

results={}
for block in ('within','system'):
    full=[r for r in rows if r['block']==block];expected={(r['source'],r['target']) for r in full}
    results[block]=[]
    for threshold in (0,.5,.8):
        selected=[r for r in full if r['minimum_cumulative_fraction']>=threshold]
        rec={'threshold':threshold,'comparisons':len(selected),'listeners':len({r['participant'] for r in selected}),'source_target_cells':len({(r['source'],r['target']) for r in selected}),'endpoints':{}}
        for name in ('model','other','difference','preference'):
            vals=[outcomes(r)[name] for r in selected]
            covered={(r['source'],r['target']) for r,v in zip(selected,vals) if v is not None}
            missing=len(expected-covered)
            if missing:
                rec['endpoints'][name]={'estimable':False,'missing_source_target_cells':missing,'observations':sum(v is not None for v in vals),'reason':'Original equal-source-target estimand has an empty valid-response cell; no cell dropped or weight changed.'}
            else:rec['endpoints'][name]={'estimable':True,**estimate(selected,vals)}
        results[block].append(rec)

source_rows=[]
full=[r for r in rows if r['block']=='within']
for source in sorted({r['source'] for r in full}):
    row={'source_sha256':source}
    for method in ('model','other'):
        groups=[np.asarray([r[method+'_arousal'] for r in full if r['source']==source and r['target_sign']==sign and r[method+'_arousal'] is not None]) for sign in (1,-1)]
        row[method+'_high_minus_low']=float(groups[0].mean()-groups[1].mean())
        row[method+'_marginal_se']=float(np.sqrt(sum(g.var(ddof=1)/len(g) for g in groups)))
        row[method+'_n_high'],row[method+'_n_low']=[len(g) for g in groups]
    source_rows.append(row)

OUT.mkdir(exist_ok=True)
summary={'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
         'scope':'Post-hoc cumulative-playback-duration sensitivity; fixed 69 complete questionnaires; main estimator and endpoint hierarchy retained.',
         'duration_definition':'Accumulated playback seconds divided by the full rendered duration of each individual clip, including repeats; both clips must meet the threshold. This is not unique temporal coverage.',
         'results':results,'median_marginal_se':{m:float(np.median([r[m+'_marginal_se'] for r in source_rows])) for m in ('model','other')}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'source-estimates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(source_rows[0]));w.writeheader();w.writerows(source_rows)
for block,recs in results.items():
    for rec in recs:
        print(block,rec['threshold'],'comparisons',rec['comparisons'],'listeners',rec['listeners'])
        for name,d in rec['endpoints'].items():
            print(name,([round(d['estimate'],6),d['cluster_sum_HC1']['ci95'],d['observations']] if d['estimable'] else 'not estimable; valid n='+str(d['observations'])))
print(OUT)
