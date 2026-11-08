#!/usr/bin/env python3
"""Apply the preserved official EMOPIA classifiers to the October3 new-score playback MIDI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from scripts.analyze_ss26_comparison import read_csv, write_csv, sha
from scripts.build_corrected_three_system_population import crop_midi
from scripts.run_official_emopia_classifier_v78 import OfficialRemiClassifier, TASKS, target_for, official_commit
import torch


def evaluate(materials, legacy, upstream, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    stimuli = read_csv(materials/'stimuli.csv')
    reference = json.loads((legacy/'status.json').read_text())
    assert official_commit(upstream) == reference['classifier_commit']
    assets = {}
    for task in TASKS:
        ref = reference['tasks'][task]['classifier']
        assets[task] = {name:dict(path=ref[name], sha256=sha(Path(ref[name]))) for name in ('checkpoint', 'config', 'dictionary')}
        assert all(assets[task][name]['sha256'] == ref[name+'_sha256'] for name in assets[task])
    (output/'assets.json').write_text(json.dumps(dict(status='PASS', assets=assets, commit=official_commit(upstream),
                                                      torch=torch.__version__, python=sys.version), indent=2)+'\n')
    standardized = []
    for r in stimuli:
        src = materials/r['midi_file'];assert sha(src) == r['midi_sha256']
        # Same eight-bar cropper as the September SS26 and V78 populations; these files already end at 32 beats.
        rep = crop_midi(src, output/'eight_bar_midi'/(r['stimulus_id']+'.mid'), bars=8)
        assert rep['source_note_count'] == rep['standardized_note_count']
        dst = output/'eight_bar_midi'/(r['stimulus_id']+'.mid')
        standardized.append(dict(stimulus_id=r['stimulus_id'], method=r['method'], source_sha256=r['source_sha256'],
                                 seed=r['seed'], target_quadrant=r['target_quadrant'], classifier_midi=str(dst),
                                 classifier_midi_sha256=sha(dst), **rep))
    write_csv(output/'population_manifest.csv', standardized)
    reproduction, started = [], time.time()
    for task in TASKS:
        classifier = OfficialRemiClassifier(upstream, task, 'cpu')
        previous = read_csv(legacy/'predictions'/('emopia_test_eight_bar_'+task+'.csv'))
        # One fixed first-listed real example per quadrant, as in the September SS26 run.
        for q in ('Q1', 'Q2', 'Q3', 'Q4'):
            r = next(r for r in previous if r['quadrant'] == q)
            p = Path(r['midi_path']);assert sha(p) == r['midi_sha256']
            predicted, logits, probabilities, count = classifier.predict(p)
            max_error = max(abs(v-float(r['probability_'+label])) for label, v in zip(classifier.labels, probabilities))
            assert predicted == r['prediction'] and count == int(r['token_count']) and max_error < 1e-6
            reproduction.append(dict(task=task, quadrant=q, sample_id=r['sample_id'], max_probability_error=max_error, status='PASS'))
        predictions = []
        for r in standardized:
            p = Path(r['classifier_midi']);assert sha(p) == r['classifier_midi_sha256']
            predicted, logits, probabilities, count = classifier.predict(p)
            target = target_for(task, r['target_quadrant'])
            row = {k:r[k] for k in ('stimulus_id', 'method', 'source_sha256', 'seed', 'target_quadrant')}
            row.update(task=task, target=target, prediction=predicted, correct=int(target == predicted), token_count=count)
            row.update({'probability_'+k:v for k, v in zip(classifier.labels, probabilities)})
            row.update({'logit_'+k:v for k, v in zip(classifier.labels, logits)})
            predictions.append(row)
        write_csv(output/(task+'_predictions.csv'), predictions)
        print(task, len(predictions), 'elapsed', round(time.time()-started), flush=True)
    (output/'legacy_reproduction.json').write_text(json.dumps(reproduction, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(status='PASS', files=len(standardized), tasks=TASKS,
        reproduced_examples=len(reproduction), elapsed_seconds=time.time()-started, descriptive_only=True), indent=2)+'\n')
    print('CLASSIFIER_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('materials', 'legacy', 'upstream', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    try:evaluate(a.materials, a.legacy, a.upstream, a.output)
    except Exception as e:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=repr(e), time=time.time()), indent=2)+'\n')
        raise
