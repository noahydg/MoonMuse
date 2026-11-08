#!/usr/bin/env python3
"""Prepare or execute the prospectively matched, two-phase SS26 ablation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--smoke-report', type=Path)
    args = ap.parse_args()
    bundle, out = args.bundle_root.resolve(), args.output.resolve()
    repo = bundle / 'MoonMuse-repo'
    out.mkdir(parents=True, exist_ok=False)
    configs_dir = out / 'configs'
    configs_dir.mkdir()
    templates = [repo / 'stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml', repo / 'stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml']
    config_paths = {}
    for phase, template in enumerate(templates, 1):
        original = yaml.safe_load(template.read_text())
        plus, zero = copy.deepcopy(original), copy.deepcopy(original)
        # Explicit false is required by train.py. All other effective settings match.
        zero['arousal_alignment']['enabled'] = False
        for branch, conf in [('Aplus', plus), ('A0', zero)]:
            conf['training'].update(num_epochs=30, patience=8, trained_params=None, trained_optim=None, inference_params=None)
            conf['training']['ckpt_dir'] = 'UNUSED_REQUIRES_COMMAND_LINE_OVERRIDE'
            target = configs_dir / f'{branch}_phase{phase}.yaml'
            target.write_text(yaml.safe_dump(conf, sort_keys=False))
            config_paths[branch, phase] = target
        comparable_zero = copy.deepcopy(zero)
        comparable_zero['arousal_alignment']['enabled'] = True
        if comparable_zero != plus:
            raise ValueError('Unexpected non-alignment difference between branches')
    code_files = [repo / p for p in ['stage3_performance/train.py', 'stage3_performance/dataloader.py', 'stage3_performance/model/music_gpt2.py', 'stage3_arousal_alignment.py', 'stage3_attribute_objective.py', 'scripts/run_ss26_matched_alignment.py']]
    protocol = {
        'protocol': 'SS26-matched-alignment-v1', 'seeds': [42, 43, 44],
        'phases': ['general-token reconstruction, alignment weight 0.20 if enabled', 'performance-attribute reconstruction, calibrated alignment weight 2.25 if enabled'],
        'max_epochs_per_phase': 30, 'early_stopping_patience': 8, 'workers': 4,
        'optimizer': 'reset at the start of every phase',
        'selection': 'lowest total validation objective in each branch; Aplus includes alignment, A0 does not',
        'interpretation': 'ablation of the complete alignment training and selection scheme; not an isolated gradient-only effect',
        'execution_order': 'seed42,43,44; phase1 Aplus,A0 then phase2 Aplus,A0 for each seed',
        'result_dependent_retries': False, 'formal_listening_data': False,
        'code_sha256': {str(p.relative_to(repo)): sha(p) for p in code_files},
        'configuration_sha256': {p.name: sha(p) for p in config_paths.values()},
        'source_M_sha256': {str(s): sha(bundle / 'models/M' / f'seed{s}.pt') for s in [42, 43, 44]},
    }
    protocol_file = out / 'protocol.json'
    protocol_file.write_text(json.dumps(protocol, indent=2) + '\n')
    (out / 'protocol.sha256').write_text(f'{sha(protocol_file)}  protocol.json\n')
    commands = []
    for seed in protocol['seeds']:
        for phase in (1, 2):
            for branch in ('Aplus', 'A0'):
                run_dir = out / f'seed{seed}' / branch / f'phase{phase}'
                source = bundle / 'models/M' / f'seed{seed}.pt' if phase == 1 else out / f'seed{seed}' / branch / 'phase1/checkpoints/params/best_params.pt'
                args_train = [sys.executable, 'stage3_performance/train.py', '-c', str(config_paths[branch, phase]), '-m', 'gpt2', '-r', 'functional', '--seed', str(seed), '--num_workers', '4', '--max_epochs', '30', '--patience', '8', '--ckpt_dir', str(run_dir / 'checkpoints'), '--pretrained_param_path', str(source), '--reset_optimizer']
                commands.append({'seed': seed, 'phase': phase, 'branch': branch, 'run_dir': str(run_dir), 'source': str(source), 'argv': args_train})
    (out / 'planned_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if not args.execute:
        print(json.dumps({'status': 'PREPARED_ONLY', 'protocol_sha256': sha(protocol_file), 'planned_training_runs': len(commands)}))
        return
    if not args.smoke_report or json.loads(args.smoke_report.read_text()).get('status') != 'PASS':
        raise ValueError('A passing SS26 recovery smoke report is required')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    (out / 'environment.json').write_text(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'python': sys.version}, indent=2) + '\n')
    results = []
    for command in commands:
        run_dir, source = Path(command['run_dir']), Path(command['source'])
        run_dir.mkdir(parents=True)
        before = sha(source)
        started = time.time()
        (out / 'current.json').write_text(json.dumps({**command, 'status': 'RUNNING', 'started_unix': started}, indent=2) + '\n')
        with (run_dir / 'training.log').open('w') as log:
            result = subprocess.run(command['argv'], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        entry = {**command, 'returncode': result.returncode, 'elapsed_seconds': time.time() - started, 'source_sha256_before': before, 'source_sha256_after': sha(source)}
        if result.returncode or entry['source_sha256_after'] != before:
            (run_dir / 'status.json').write_text(json.dumps({**entry, 'status': 'FAILED'}, indent=2) + '\n')
            raise RuntimeError(f'Training failed; inspect {run_dir}')
        checkpoint = run_dir / 'checkpoints/params/best_params.pt'
        with (run_dir / 'validation.log').open('w') as log:
            checked = subprocess.run([sys.executable, 'scripts/validate_checkpoint.py', str(checkpoint), '--optimizer', str(run_dir / 'checkpoints/optim/best_optim.pt'), '--reference', str(source), '--output', str(run_dir / 'checkpoint_validation.json')], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        if checked.returncode:
            raise RuntimeError(f'Checkpoint validation failed; inspect {run_dir}')
        entry.update(status='PASS', checkpoint_sha256=sha(checkpoint))
        (run_dir / 'status.json').write_text(json.dumps(entry, indent=2) + '\n')
        results.append(entry)
        (out / 'completed.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f"PASS seed={command['seed']} {command['branch']} phase={command['phase']}", flush=True)
    (out / 'status.json').write_text(json.dumps({'status': 'PASS', 'runs_completed': len(results), 'formal_listening_data': False}, indent=2) + '\n')
    (out / 'current.json').write_text(json.dumps({'status': 'COMPLETE', 'runs_completed': len(results)}) + '\n')


if __name__ == '__main__':
    main()
