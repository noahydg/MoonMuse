#!/usr/bin/env python3
"""Create a Stage 3 dataset with separate valence and arousal conditions."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factorized_conditioning import (  # noqa: E402
    FACTORIZED_TOKENS,
    append_factorized_vocabulary,
    factorize_leading_emotion,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def m2p_track_positions(events):
    """Recompute Melody/Performance spans using the repository's M2P contract."""
    track_indices = []
    eos_index = len(events)
    for index, event in enumerate(events):
        if event.get('name') == 'Track' and event.get('value') in ('Melody', 'Performance'):
            track_indices.append((index, event['value']))
        elif event.get('name') == 'EOS':
            eos_index = index

    melody_performance = []
    performance_melody = []
    for position, (index, track) in enumerate(track_indices):
        if position + 1 < len(track_indices):
            next_index, next_track = track_indices[position + 1]
        else:
            next_index, next_track = eos_index, 'EOS'
        if track == 'Melody' and next_track == 'Performance':
            melody_performance.append((index, next_index))
        elif track == 'Performance' and next_track in ('Melody', 'EOS'):
            performance_melody.append((index, next_index))

    if not melody_performance or len(melody_performance) != len(performance_melody):
        raise ValueError(
            'invalid M2P track sequence: melody={}, performance={}'.format(
                len(melody_performance), len(performance_melody)
            )
        )
    return melody_performance, performance_melody


def transform_piece(piece):
    if not isinstance(piece, (list, tuple)) or len(piece) < 3:
        raise ValueError('expected a three-part Stage 3 pickle')
    events = factorize_leading_emotion(piece[2])
    melody_pos, performance_pos = m2p_track_positions(events)
    return [melody_pos, performance_pos, events]


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dump_pickle(value, path):
    with open(path, 'wb') as handle:
        pickle.dump(value, handle)


def prepare_dataset(source_root, output_root, limit=None, force=False):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    source_events = source_root / 'events'
    source_dictionary = source_root / 'dictionary.pkl'
    source_splits = source_root / 'data_splits'

    required = [source_events, source_dictionary, source_splits]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('missing source paths: {}'.format(', '.join(missing)))
    if output_root == source_root:
        raise ValueError('output root must differ from source root')
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(
                '{} is not empty; pass --force to replace it'.format(output_root)
            )
        shutil.rmtree(output_root)

    output_events = output_root / 'events'
    output_splits = output_root / 'data_splits'
    output_events.mkdir(parents=True, exist_ok=True)
    output_splits.mkdir(parents=True, exist_ok=True)

    dictionary_data = load_pickle(source_dictionary)
    if len(dictionary_data) < 2:
        raise ValueError('dictionary pickle must contain event2idx and idx2event')
    old_event2idx, old_idx2event = dictionary_data[:2]
    new_event2idx, new_idx2event = append_factorized_vocabulary(
        old_event2idx, old_idx2event
    )
    dump_pickle((new_event2idx, new_idx2event), output_root / 'dictionary.pkl')

    files = sorted(path for path in source_events.iterdir() if path.suffix == '.pkl')
    if limit is not None:
        files = files[:limit]
    selected = {path.name for path in files}
    labels = Counter()
    for index, source_path in enumerate(files, start=1):
        source_piece = load_pickle(source_path)
        label = str(source_piece[2][0].get('value'))
        transformed = transform_piece(source_piece)
        labels[label] += 1
        dump_pickle(transformed, output_events / source_path.name)
        if index == 1 or index % 200 == 0 or index == len(files):
            print('[factorized-data] {}/{}'.format(index, len(files)))

    split_counts = {}
    for source_split in sorted(source_splits.glob('*.pkl')):
        names = load_pickle(source_split)
        if limit is not None:
            names = [name for name in names if name in selected]
        missing_names = [name for name in names if name not in selected]
        if missing_names:
            raise ValueError(
                '{} references unconverted files: {}'.format(
                    source_split.name, missing_names[:5]
                )
            )
        dump_pickle(names, output_splits / source_split.name)
        split_counts[source_split.name] = len(names)

    report = {
        'schema_version': 1,
        'conditioning_mode': 'factorized_va',
        'source_root': str(source_root),
        'output_root': str(output_root),
        'source_dictionary_sha256': sha256_file(source_dictionary),
        'output_dictionary_sha256': sha256_file(output_root / 'dictionary.pkl'),
        'legacy_vocab_size': len(old_event2idx),
        'factorized_vocab_size': len(new_event2idx),
        'legacy_pad_index': len(old_event2idx),
        'factorized_pad_index': len(new_event2idx),
        'appended_tokens': list(FACTORIZED_TOKENS),
        'converted_files': len(files),
        'quadrant_counts': dict(sorted(labels.items())),
        'split_counts': split_counts,
        'limited_conversion': limit is not None,
    }
    report_path = output_root / 'factorized_data_report.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('[factorized-data] report:', report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--limit', type=int, default=None,
                        help='convert only the first N files for a local smoke test')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    prepare_dataset(
        args.source_root, args.output_root, limit=args.limit, force=args.force
    )


if __name__ == '__main__':
    main()
