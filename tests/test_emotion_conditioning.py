import unittest

from emotion_conditioning import (
    emotion_candidates_from_filename,
    paired_sampling_seed,
)


class EmotionConditioningTests(unittest.TestCase):
    def test_filename_resolution_preserves_legacy_behaviour(self):
        self.assertEqual(emotion_candidates_from_filename('piece_Q3.txt'), ['Q3'])
        self.assertEqual(
            emotion_candidates_from_filename('piece_Positive.txt'),
            ['Q1', 'Q4'],
        )
        self.assertEqual(
            emotion_candidates_from_filename('piece_Negative.txt'),
            ['Q2', 'Q3'],
        )

    def test_explicit_overrides_ignore_filename_and_remove_duplicates(self):
        self.assertEqual(
            emotion_candidates_from_filename(
                'piece_Q1.txt', overrides=['Q2', 'Q3', 'Q2']
            ),
            ['Q2', 'Q3'],
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid emotion override'):
            emotion_candidates_from_filename('piece.txt', overrides=['Q5'])

    def test_paired_seed_is_stable_per_source_and_excludes_condition(self):
        seed_a = paired_sampling_seed(42, 'source-a.txt')
        self.assertEqual(seed_a, paired_sampling_seed(42, 'source-a.txt'))
        self.assertNotEqual(seed_a, paired_sampling_seed(42, 'source-b.txt'))
        self.assertNotEqual(seed_a, paired_sampling_seed(43, 'source-a.txt'))
        self.assertGreaterEqual(seed_a, 0)
        self.assertLessEqual(seed_a, 2**32 - 1)


if __name__ == '__main__':
    unittest.main()
