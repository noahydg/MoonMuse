#!/usr/bin/env python3
"""Expand a legacy Stage 3 checkpoint for factorized VA conditioning.

Torch is imported only when a checkpoint is actually migrated so that the row
mapping can be unit-tested on machines without the training environment.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factorized_conditioning import FACTORIZED_TOKENS  # noqa: E402


EMBEDDING_SOURCES = {
    'Valence_Positive': ('Emotion_Q1', 'Emotion_Q4'),
    'Valence_Negative': ('Emotion_Q2', 'Emotion_Q3'),
    'Valence_None': ('Emotion_None',),
    'Arousal_High': ('Emotion_Q1', 'Emotion_Q2'),
    'Arousal_Low': ('Emotion_Q3', 'Emotion_Q4'),
    'Arousal_None': ('Emotion_None',),
}

ROW_PARAMETERS = (
    'token_emb.emb_lookup.weight',
    'dec_out_proj.weight',
    'dec_out_proj.bias',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dictionary(path):
    with open(path, 'rb') as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError('{} is not a two-map dictionary pickle'.format(path))
    return value[0], value[1]


def build_checkpoint_plan(old_event2idx, new_event2idx):
    """Validate dictionaries and describe every new checkpoint row."""
    legacy_size = len(old_event2idx)
    factorized_size = len(new_event2idx)
    for token, old_index in old_event2idx.items():
        if new_event2idx.get(token) != old_index:
            raise ValueError('legacy token index changed: {}'.format(token))
    if factorized_size != legacy_size + len(FACTORIZED_TOKENS):
        raise ValueError('factorized dictionary must append exactly six tokens')
    appended = tuple(
        token for token, index in sorted(new_event2idx.items(), key=lambda item: item[1])
        if index >= legacy_size
    )
    if appended != FACTORIZED_TOKENS:
        raise ValueError('unexpected appended token order: {!r}'.format(appended))

    embedding_rows = {}
    for target, sources in EMBEDDING_SOURCES.items():
        missing = [source for source in sources if source not in old_event2idx]
        if missing:
            raise ValueError('missing legacy source tokens: {}'.format(missing))
        embedding_rows[new_event2idx[target]] = tuple(
            old_event2idx[source] for source in sources
        )
    return {
        'legacy_vocab_size': legacy_size,
        'factorized_vocab_size': factorized_size,
        'legacy_pad_index': legacy_size,
        'factorized_pad_index': factorized_size,
        'embedding_rows': embedding_rows,
    }


def expand_row_parameter(tensor, parameter_name, plan, torch):
    old_rows = plan['legacy_vocab_size'] + 1
    new_rows = plan['factorized_vocab_size'] + 1
    if tensor.shape[0] != old_rows:
        raise ValueError(
            '{} has {} rows; expected {}'.format(
                parameter_name, tensor.shape[0], old_rows
            )
        )

    expanded = tensor.new_zeros((new_rows,) + tuple(tensor.shape[1:]))
    expanded[:plan['legacy_vocab_size']] = tensor[:plan['legacy_vocab_size']]
    expanded[plan['factorized_pad_index']] = tensor[plan['legacy_pad_index']]

    if parameter_name == 'token_emb.emb_lookup.weight':
        for target_row, source_rows in plan['embedding_rows'].items():
            source = tensor[list(source_rows)]
            expanded[target_row] = torch.mean(source, dim=0)
    elif parameter_name == 'dec_out_proj.bias':
        for target_row in plan['embedding_rows']:
            expanded[target_row] = -10.0
    return expanded


def migrate_checkpoint(checkpoint, plan, torch):
    missing = [name for name in ROW_PARAMETERS if name not in checkpoint]
    if missing:
        raise KeyError('checkpoint is missing row parameters: {}'.format(missing))
    migrated = dict(checkpoint)
    for name in ROW_PARAMETERS:
        migrated[name] = expand_row_parameter(checkpoint[name], name, plan, torch)
    return migrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--legacy-dictionary', required=True)
    parser.add_argument('--factorized-dictionary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    old_dictionary_path = Path(args.legacy_dictionary).resolve()
    new_dictionary_path = Path(args.factorized_dictionary).resolve()
    output_path = Path(args.output).resolve()
    report_path = output_path.with_suffix(output_path.suffix + '.migration.json')
    if not checkpoint_path.is_file():
        parser.error('checkpoint does not exist: {}'.format(checkpoint_path))
    if output_path.exists() and not args.force:
        parser.error('output exists; pass --force to replace: {}'.format(output_path))

    old_event2idx, _ = load_dictionary(old_dictionary_path)
    new_event2idx, _ = load_dictionary(new_dictionary_path)
    plan = build_checkpoint_plan(old_event2idx, new_event2idx)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is required for checkpoint migration; run this inside the '
            'MoonMuse training environment.'
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise TypeError('expected a state-dict checkpoint')
    migrated = migrate_checkpoint(checkpoint, plan, torch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output_path)

    report = {
        'schema_version': 1,
        'source_checkpoint': str(checkpoint_path),
        'output_checkpoint': str(output_path),
        'source_checkpoint_sha256': sha256_file(checkpoint_path),
        'output_checkpoint_sha256': sha256_file(output_path),
        'legacy_dictionary_sha256': sha256_file(old_dictionary_path),
        'factorized_dictionary_sha256': sha256_file(new_dictionary_path),
        'legacy_vocab_size': plan['legacy_vocab_size'],
        'factorized_vocab_size': plan['factorized_vocab_size'],
        'legacy_pad_index': plan['legacy_pad_index'],
        'factorized_pad_index': plan['factorized_pad_index'],
        'embedding_initialization': EMBEDDING_SOURCES,
        'factor_output_weight_initialization': 'zeros',
        'factor_output_bias_initialization': -10.0,
        'optimizer_migrated': False,
        'optimizer_requirement': 'reset optimizer for factorized fine-tuning',
    }
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('[factorized-checkpoint] output:', output_path)
    print('[factorized-checkpoint] report:', report_path)


if __name__ == '__main__':
    main()
