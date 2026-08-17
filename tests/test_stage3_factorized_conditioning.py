import json
import pickle
import tempfile
import unittest
from pathlib import Path

from factorized_conditioning import (
    FACTORIZED_TOKENS,
    append_factorized_vocabulary,
    control_token_indices,
    factor_tokens,
    factorize_leading_emotion,
    primer_event_names,
)
from scripts.migrate_stage3_factorized_checkpoint import build_checkpoint_plan
from scripts.prepare_stage3_factorized_data import (
    m2p_track_positions,
    prepare_dataset,
    transform_piece,
)


def sample_events(quadrant='Q1'):
    return [
        {'name': 'Emotion', 'value': quadrant},
        {'name': 'Key', 'value': 'C'},
        {'name': 'Track', 'value': 'Melody'},
        {'name': 'Bar', 'value': None},
        {'name': 'Track', 'value': 'Performance'},
        {'name': 'Bar', 'value': None},
        {'name': 'Track', 'value': 'Melody'},
        {'name': 'Bar', 'value': None},
        {'name': 'Track', 'value': 'Performance'},
        {'name': 'Bar', 'value': None},
        {'name': 'EOS', 'value': None},
    ]


class FactorizedConditioningTests(unittest.TestCase):
    def test_quadrants_map_to_independent_valence_and_arousal(self):
        self.assertEqual(
            ('Valence_Positive', 'Arousal_High'), factor_tokens('Q1')
        )
        self.assertEqual(
            ('Valence_Negative', 'Arousal_Low'), factor_tokens('Q3')
        )
        self.assertEqual(
            ['Valence_Positive', 'Arousal_Low', 'Key_C'],
            primer_event_names('Q4', key='Key_C', mode='factorized_va'),
        )
        self.assertEqual(
            ['Emotion_Q4', 'Key_C'],
            primer_event_names('Q4', key='Key_C'),
        )

    def test_event_conversion_replaces_only_leading_emotion(self):
        source = sample_events('Q2')
        converted = factorize_leading_emotion(source)
        self.assertEqual(
            converted[:3],
            [
                {'name': 'Valence', 'value': 'Negative'},
                {'name': 'Arousal', 'value': 'High'},
                {'name': 'Key', 'value': 'C'},
            ],
        )
        self.assertEqual({'name': 'Emotion', 'value': 'Q2'}, source[0])
        self.assertFalse(any(event['name'] == 'Emotion' for event in converted))

    def test_dictionary_append_preserves_legacy_indices_and_moves_pad(self):
        old_event2idx = {
            'Emotion_None': 0,
            'Emotion_Q1': 1,
            'Emotion_Q2': 2,
            'Emotion_Q3': 3,
            'Emotion_Q4': 4,
            'Key_C': 5,
        }
        old_idx2event = {index: token for token, index in old_event2idx.items()}
        new_event2idx, new_idx2event = append_factorized_vocabulary(
            old_event2idx, old_idx2event
        )
        for token, index in old_event2idx.items():
            self.assertEqual(index, new_event2idx[token])
            self.assertEqual(token, new_idx2event[index])
        self.assertEqual(
            list(FACTORIZED_TOKENS),
            [new_idx2event[index] for index in range(6, 12)],
        )

        plan = build_checkpoint_plan(old_event2idx, new_event2idx)
        self.assertEqual(6, plan['legacy_pad_index'])
        self.assertEqual(12, plan['factorized_pad_index'])
        self.assertEqual((1, 4), plan['embedding_rows'][6])
        self.assertEqual((3, 4), plan['embedding_rows'][10])

    def test_control_tokens_are_output_inadmissible(self):
        event2idx = {
            'Emotion_Q1': 1,
            'Valence_Positive': 5,
            'Arousal_High': 7,
            'Note_Pitch_60': 9,
        }
        self.assertEqual((1, 5, 7), control_token_indices(event2idx))


class FactorizedDataTests(unittest.TestCase):
    def test_track_positions_are_recomputed_after_prefix_expansion(self):
        source_events = sample_events()
        source_positions = m2p_track_positions(source_events)
        converted = transform_piece([
            source_positions[0], source_positions[1], source_events
        ])
        self.assertEqual((3, 5), converted[0][0])
        self.assertEqual((5, 7), converted[1][0])
        self.assertEqual((7, 9), converted[0][1])
        self.assertEqual((9, 11), converted[1][1])

    def test_prepare_dataset_writes_factorized_dictionary_splits_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            output = root / 'output'
            (source / 'events').mkdir(parents=True)
            (source / 'data_splits').mkdir()

            tokens = [
                'Emotion_None', 'Emotion_Q1', 'Emotion_Q2', 'Emotion_Q3',
                'Emotion_Q4', 'Key_C', 'Track_Melody', 'Track_Performance',
                'Bar_None', 'EOS_None',
            ]
            event2idx = {token: index for index, token in enumerate(tokens)}
            idx2event = {index: token for token, index in event2idx.items()}
            with open(source / 'dictionary.pkl', 'wb') as handle:
                pickle.dump((event2idx, idx2event), handle)

            filename = 'Q1_sample.pkl'
            events = sample_events('Q1')
            melody, performance = m2p_track_positions(events)
            with open(source / 'events' / filename, 'wb') as handle:
                pickle.dump([melody, performance, events], handle)
            for split in ('train.pkl', 'valid.pkl'):
                with open(source / 'data_splits' / split, 'wb') as handle:
                    pickle.dump([filename], handle)

            report = prepare_dataset(source, output)
            self.assertEqual(1, report['converted_files'])
            self.assertEqual(16, report['factorized_vocab_size'])
            self.assertEqual({'Q1': 1}, report['quadrant_counts'])
            with open(output / 'events' / filename, 'rb') as handle:
                converted = pickle.load(handle)
            self.assertEqual('Valence', converted[2][0]['name'])
            self.assertEqual('Arousal', converted[2][1]['name'])
            saved_report = json.loads(
                (output / 'factorized_data_report.json').read_text('utf-8')
            )
            self.assertEqual('factorized_va', saved_report['conditioning_mode'])


if __name__ == '__main__':
    unittest.main()
