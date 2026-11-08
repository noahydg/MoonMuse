"""Runtime contracts for the matched Stage 2 campaign; no paid launch here."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stage2_event_validation import validate_events


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def preflight(plan_root, out, expected_manifest):
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    expected = json.loads(expected_manifest.read_text())
    for name, digest in expected['repo_files'].items():
        assert sha(REPO / name) == digest, ('code/data drift', name)
    plans = list(csv.DictReader((plan_root / 'plan_manifest.csv').open()))
    assert len(plans) == 150
    for row in plans:
        assert sha(plan_root / row['plan_path']) == row['plan_sha256']
    assert sha(plan_root / 'plan_manifest.csv') == expected['plan_manifest_sha256']
    dump(out / 'preflight.json', dict(status='PASS', files=len(expected['repo_files']),
        plans=len(plans), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        asset_manifest_sha256=sha(expected_manifest)))


def model_and_dataset(config, seed=42):
    import torch
    import yaml
    import pickle
    import random
    import numpy as np
    sys.path.insert(0, str(REPO / 'stage2_melody'))
    from dataloader import REMISkylineToMidiTransformerDataset
    from model.music_gpt2 import ImprovedMusicGPT2
    conf = yaml.safe_load(Path(config).read_text())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    d, m = conf['data_loader'], conf['model']
    ds = REMISkylineToMidiTransformerDataset(
        data_dir=d['data_path'].format('functional'), vocab_file=d['vocab_path'].format('functional'),
        model_dec_seqlen=m['max_len'], pieces=pickle.load(open(d['train_split'], 'rb')),
        pad_to_same=True, predict_key=False, appoint_st_bar=0)
    model = ImprovedMusicGPT2(ds.vocab_size, m['n_layer'], m['n_head'], m['d_model'], m['d_ff'], m['d_embed'])
    return model, ds


def objective_probe(out):
    import torch
    from torch.utils.data import DataLoader
    records = []
    for stem in ('pop1k7_pretrain_new_gpt2', 'emopia_finetune_new_gpt2'):
        cfg = REPO / 'stage2_melody/config' / (stem + '.yaml')
        model, ds = model_and_dataset(cfg)
        model = model.cuda().eval()
        for batch in DataLoader(ds, batch_size=4, shuffle=False, num_workers=0):
            targets = batch['dec_target'].cuda()
            pads = int((targets == ds.pad_token).sum())
            if pads and pads < targets.numel(): break
        else: raise AssertionError('No real batch contains both PAD and non-PAD targets')
        with torch.no_grad():
            logits = model(batch['dec_input'].cuda(), seg_inp=batch['track_mask'].cuda())
            ignore = model.compute_loss(logits, targets)['recons_loss']
            count = model.compute_loss(logits, targets, ignore_index=-100)['recons_loss']
            truth = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=ds.pad_token)
            assert torch.isfinite(ignore) and torch.isfinite(count)
            assert torch.allclose(ignore, truth) and not torch.isclose(ignore, count)
        records.append(dict(config=str(cfg.relative_to(REPO)), piece_ids=batch['piece_id'],
            pad_targets=pads, total_targets=targets.numel(), ignore=float(ignore), count=float(count),
            common_nonpad=float(truth), batch_sha256=hashlib.sha256(targets.cpu().numpy().tobytes()).hexdigest()))
        del logits, targets, model
        torch.cuda.empty_cache()
    dump(out / 'actual_objective_probe.json', dict(status='PASS', batches=records))


def checkpoint(run):
    import torch
    model, _ = model_and_dataset(run / 'config.yaml')
    p = run / 'params/best_params.pt'
    state = torch.load(p, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    values = [float(x) for x in re.findall(r'nonpad_ce=([^\s]+)', (run / 'val_nonpad_ce.txt').read_text())]
    assert values and all(math.isfinite(x) for x in values)
    dump(run / 'checkpoint_integrity.json', dict(status='PASS', sha256=sha(p),
        validation_epochs=len(values), best_nonpad_ce=min(values), best_epoch=values.index(min(values))+1))


def chord_segments(tokens):
    segments = []
    mode = None
    for token in tokens:
        if token == 'Track_Chord': segments.append([]); mode = 'chord'
        elif token == 'Track_Melody': mode = 'melody'
        elif mode == 'chord': segments[-1].append(token)
    if segments and not segments[-1]: segments.pop()
    return segments


def verify_generation(plan_root, out, strict_keys=True):
    plans = list(csv.DictReader((plan_root / 'plan_manifest.csv').open()))
    expected, records = set(), []
    for mode in ('ignore', 'count'):
        for row in plans:
            plan = plan_root / row['plan_path']
            tokens = plan.read_text().splitlines()
            starts = [i for i, t in enumerate(tokens) if t == 'Bar_None']
            assert len(starts) == 9
            bars = [tokens[a:b] for a, b in zip(starts, starts[1:])]
            for target in (('Q1', 'Q4') if row['valence'] == 'Positive' else ('Q2', 'Q3')):
                path = out / f'melodies/{mode}/seed{row["seed"]}/{plan.stem}_{target}_melody.txt'
                expected.add(path)
                actual = path.read_text().splitlines()
                meta = json.loads(path.with_name(path.name + '.generation.json').read_text())
                assert sha(path) == meta['events_sha256']
                assert meta['input_file'] == plan.name and meta['target'] == target and meta['seed'] == int(row['seed'])
                keys = [t for t in actual if t.startswith('Key_')]
                key_contract_valid = keys == [row['key']]
                assert [t for t in actual if t.startswith('Emotion_')] == ['Emotion_' + target]
                observed = chord_segments(actual)
                assert observed == bars[:len(observed)], ('forced harmony differs', path)
                check = validate_events(actual, 8, meta)
                records.append(dict(mode=mode, seed=row['seed'], target=target, file=str(path.relative_to(out)),
                    source_plan=row['plan_path'], events_sha256=sha(path),
                    key_contract_valid=key_contract_valid, observed_keys=keys,
                    stop_reason=meta['stop_reason'], **check))
    actual_paths = set((out / 'melodies').rglob('*_melody.txt'))
    assert actual_paths == expected and len(expected) == 600, (len(expected), len(actual_paths))
    summary = []
    for seed in ('42', '43', '44'):
        for mode in ('ignore', 'count'):
            own = [r for r in records if r['seed'] == seed and r['mode'] == mode]
            assert len(own) == 100
            summary.append(dict(seed=seed, mode=mode, files=100,
                syntactically_valid=sum(r['valid'] for r in own),
                key_contract_failures=sum(not r['key_contract_valid'] for r in own),
                completed=sum(r['complete_bars'] is True for r in own),
                reached_target_bar=sum(r['reached_target_bar'] for r in own)))
        a = json.loads((out / f'checkpoints/seed{seed}/ignore/pretrain/training_objective.json').read_text())
        b = json.loads((out / f'checkpoints/seed{seed}/count/pretrain/training_objective.json').read_text())
        assert a['initial_state_sha256'] == b['initial_state_sha256']
    key_failures = sum(not r['key_contract_valid'] for r in records)
    report = dict(status='FAIL_COLLECTION' if key_failures else 'PASS_COLLECTION', files=600,
        boundary='Collection integrity; invalid or incomplete model outputs remain in outcome counts',
        key_contract_failures=key_failures, groups=summary, records=records)
    dump(out / 'generation_integrity.json', report)
    if strict_keys:
        assert key_failures == 0, ('key contract failures', key_failures)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'objectives', 'checkpoint', 'generation'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan-root', type=Path)
    p.add_argument('--asset-manifest', type=Path)
    a = p.parse_args()
    if a.action == 'preflight': preflight(a.plan_root, a.out, a.asset_manifest)
    elif a.action == 'objectives': objective_probe(a.out)
    elif a.action == 'checkpoint': checkpoint(a.out)
    else: verify_generation(a.plan_root, a.out)
