"""Prepared inference-only follow-up; a paid launch needs separate authorization."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.launch_ss26_stage2_pad import Supervisor
from scripts.ss26_stage2_runtime_checks import (dump, model_and_dataset,
                                               preflight, sha, verify_generation)
from stage2_event_validation import validate_events
from generation_budget import OnsetPitchTracker

TOTAL_PHASES = 10
RUN_NAME = 'ss26-stage2-structured-20261002-v1'
PROTOCOL_NAME = 'ss26-stage2-structured-protocol-20261002.md'
ASSET_NAME = 'ss26-stage2-structured-assets-20261002.json'


def shutdown_command():
    # AutoDL's shutdown entry point may be a script without a shebang. Bash
    # supplies normal shell-command semantics for that observed ENOEXEC case.
    # The platform timer remains mandatory; a local test cannot prove power-off.
    return ['/bin/bash', '-c', '/usr/bin/shutdown']


def commands(prior, plans, out):
    """All six conditions, always with the same prospective decoding policy."""
    config = 'stage2_melody/config/emopia_finetune_new_gpt2.yaml'
    return [(seed, mode, [sys.executable, 'stage2_melody/inference.py',
        '-c', config, '-m', 'gpt2', '-r', 'functional',
        '-i', str(prior / f'checkpoints/seed{seed}/{mode}/finetune/params/best_params.pt'),
        '-id', str(plans / f'plans/seed{seed}'),
        '-od', str(out / f'melodies/{mode}/seed{seed}'), '--seed', str(seed),
        '--max_bars', '8', '--max_events', '1536', '--max_rejections', '512',
        '--max_segment_attempts', '512', '--temperature', '1.0', '--top-p', '0.97',
        '--write-generation-metadata', '--structured-generation'])
        for seed in (42, 43, 44) for mode in ('ignore', 'count')]


def smoke_commands(prior, plans, out):
    rows = []
    for seed, mode, command in commands(prior, plans, out):
        if seed != 42: continue
        command[command.index('-id') + 1] = str(out / 'smoke/plans')
        command[command.index('-od') + 1] = str(out / f'smoke/{mode}')
        rows.append((mode, command))
    return rows


def worker():
    assert os.environ.get('SS26_MANAGED') == '1', 'Launch through the supervisor'
    out = Path(os.environ['OUT'])
    prior = Path(os.environ['PRIOR_RUN'])
    plans = Path(os.environ['PLAN_ROOT'])
    asset = Path(os.environ['ASSET_MANIFEST'])
    expected = json.loads(asset.read_text())
    completed = 0
    (out / 'logs').mkdir()
    (out / 'status.tsv').write_text('phase\trc\tseconds\n')

    def progress(name):
        p = out / 'progress.json'; temporary = p.with_suffix('.tmp')
        dump(temporary, dict(status='RUNNING', phase=name, completed=completed,
            total=TOTAL_PHASES, updated_epoch=time.time(),
            deadline_epoch=float(os.environ['SS26_TOTAL_DEADLINE'])))
        temporary.replace(p)

    def phase(name, fn):
        nonlocal completed
        progress(name); start = time.monotonic(); rc = 1
        try:
            fn(); rc = 0
        finally:
            with (out / 'status.tsv').open('a') as f:
                f.write(f'{name}\t{rc}\t{time.monotonic()-start:.3f}\n')
        completed += 1; progress(name)

    def check_inputs():
        assert out.resolve() != prior.resolve()
        assert (prior / 'COMPLETE').is_file()
        assert (prior / 'campaign.exitcode').read_text().strip() == '0'
        preflight(plans, out, asset)
        for name, digest in expected['prior_files'].items():
            assert sha(prior / name) == digest, ('prior-run drift', name)
        assert sha(REPO.parent / PROTOCOL_NAME) == expected['protocol_sha256']
        shutdown = Path('/usr/bin/shutdown')
        assert shutdown.is_file() and os.access(shutdown, os.X_OK)
        assert Path('/bin/bash').is_file()
        dump(out / 'shutdown_preflight.json', dict(command=shutdown_command(),
            entrypoint_sha256=sha(shutdown),
            has_shebang=shutdown.read_bytes().startswith(b'#!'),
            boundary='Read-only inspection, not a shutdown execution test; platform timer required'))
        shutil.copy2(asset, out / 'asset_manifest.json')
        # Small evidence bundle: code/plans/config and model identities, not
        # another copy of the already archived trained weights.
        for name in expected['snapshot_files']:
            destination = out / 'code' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
        shutil.copytree(plans, out / 'plans')
        shutil.copy2(REPO.parent / PROTOCOL_NAME,
                     out / 'protocol.md')

    def check_models():
        import torch
        records = []
        for seed, mode, command in commands(prior, plans, out):
            base = prior / f'checkpoints/seed{seed}/{mode}'
            model, _ = model_and_dataset(base / 'finetune/config.yaml')
            weight = base / 'finetune/params/best_params.pt'
            state = torch.load(weight, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            assert all(torch.isfinite(t).all() for t in state.values())
            assert model.transformer.config.max_position_embeddings >= 1536
            records.append(dict(seed=seed, mode=mode, sha256=sha(weight),
                                status='PASS', command=command))
            objective = out / f'checkpoints/seed{seed}/{mode}/pretrain/training_objective.json'
            objective.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / 'pretrain/training_objective.json', objective)
            del model, state
        dump(out / 'input_checkpoint_checks.json', records)

    def run_command(name, command):
        with (out / 'logs' / (name + '.log')).open('w') as f:
            subprocess.run(command, cwd=REPO, check=True, stdout=f, stderr=subprocess.STDOUT)

    phase('preflight', check_inputs)
    phase('six_checkpoint_checks', check_models)

    def smoke():
        smoke_plans = out / 'smoke/plans'
        smoke_plans.mkdir(parents=True)
        for valence in ('Positive', 'Negative'):
            p = plans / f'plans/seed42/samp_00_{valence}.txt'
            shutil.copy2(p, smoke_plans / p.name)
        checks = []
        for mode, command in smoke_commands(prior, plans, out):
            run_command('smoke_' + mode, command)
            paths = sorted((out / f'smoke/{mode}').glob('*_melody.txt'))
            assert len(paths) == 4
            for p in paths:
                ts = p.read_text().splitlines()
                meta = json.loads(Path(str(p) + '.generation.json').read_text())
                check = validate_events(ts, 8, meta)
                tracker, track, duplicates = OnsetPitchTracker(), None, 0
                for token in ts:
                    if token.startswith('Track_'):
                        track = token[6:]; tracker.reset()
                    elif track == 'Melody':
                        duplicates += bool(tracker.inadmissibles({token: 0}))
                        tracker.observe(token)
                assert sha(p) == meta['events_sha256']
                source = (smoke_plans / meta['input_file']).read_text().splitlines()
                assert [t for t in ts if t.startswith('Key_')] == [source[0]]
                checks.append(dict(mode=mode, file=str(p.relative_to(out)),
                    valid=check['valid'], complete=check['complete_bars'], duplicate_pitch_events=duplicates,
                    sampling_policy=meta['sampling_policy']))
        dump(out / 'smoke_checks.json', checks)
        assert len(checks) == 8 and all(r['valid'] and r['complete'] and
            not r['duplicate_pitch_events'] and r['sampling_policy'] == 'structured-v2' for r in checks), 'smoke failed; no full generation'
    phase('eight_output_smoke', smoke)
    for seed, mode, command in commands(prior, plans, out):
        name = f's{seed}_{mode}_generate'
        phase(name, lambda name=name, command=command: run_command(name, command))

    def evaluate():
        report = verify_generation(plans, out, strict_keys=False)
        for sidecar in (out / 'melodies').rglob('*.generation.json'):
            meta = json.loads(sidecar.read_text())
            assert meta['sampling_policy'] == 'structured-v2' and meta['max_events'] == 1536
            assert meta['accepted_events'] <= 1536
        run_command('evaluate', [sys.executable, 'scripts/evaluate_ss26_stage2_melodies.py',
            '--group', 'stage2_ignore', str(out / 'melodies/ignore'),
            '--group', 'stage2_count', str(out / 'melodies/count'),
            '--emopia-reference', 'events/c2m/emopia_c2m_functional',
            '--output-dir', str(out / 'descriptors')])
        # Finish the outcome inventory and descriptions even when an output
        # violates the key contract, then preserve a nonzero campaign status.
        assert report['status'] == 'PASS_COLLECTION', report['key_contract_failures']
    phase('verify_and_describe', evaluate)
    assert completed == TOTAL_PHASES


def main():
    if sys.argv[1:] == ['--worker']:
        worker(); return 0
    if sys.argv[1:]:
        raise ValueError('No positional launch options supported')
    # An explicit operator acknowledgment prevents accidental local/paid starts.
    if os.environ.get('SS26_REGEN_APPROVED') != '1':
        raise RuntimeError('Not launched: this follow-up needs a new approved GPU budget')
    disk = Path('/root/autodl-tmp')
    if REPO != disk / 'MoonMuse-SS26/MoonMuse-repo':
        raise RuntimeError('This paid launcher is restricted to the retained AutoDL checkout')
    out = disk / RUN_NAME
    env = dict(os.environ, OUT=str(out), PYTHONUNBUFFERED='1', SS26_MANAGED='1',
        PRIOR_RUN=str(disk / 'ss26-stage2-pad-20260928-v2'),
        PLAN_ROOT=str(REPO.parent / 'ss26-stage1-plans-20260928'),
        ASSET_MANIFEST=str(REPO.parent / ASSET_NAME),
        SS26_TOTAL_DEADLINE=str(time.time()+2700))
    supervisor = Supervisor(out, Path('/root/autodl-fs/moonmuse') / RUN_NAME,
        REPO, total_seconds=2700, reserve_seconds=300,
        shutdown_command=shutdown_command(), total_phases=TOTAL_PHASES)
    def interrupted(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    tests = [sys.executable, '-m', 'pytest', '-q',
        'tests/test_stage2_guarded_generation.py', 'tests/test_generation_budget.py',
        'tests/test_ss26_stage2_integrity.py', 'tests/test_ss26_stage2_regeneration.py']
    return supervisor.run(tests, [sys.executable, __file__, '--worker'], env)


if __name__ == '__main__':
    sys.exit(main())
