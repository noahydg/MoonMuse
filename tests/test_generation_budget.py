import unittest

from generation_budget import (
    OnsetPitchTracker,
    RejectionBudget,
    SegmentSamplingBudget,
    monotonic_beat_inadmissibles,
    strip_terminal_eos,
)


class RejectionBudgetTest(unittest.TestCase):
    def test_disabled_budget_never_stops(self):
        budget = RejectionBudget()
        self.assertFalse(budget.record())
        self.assertFalse(budget.record())
        self.assertEqual(budget.rejections, 2)

    def test_budget_stops_on_the_limit(self):
        budget = RejectionBudget(3)
        self.assertFalse(budget.record())
        self.assertFalse(budget.record())
        self.assertTrue(budget.record())
        self.assertEqual(budget.rejections, 3)

    def test_non_positive_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            RejectionBudget(0)


class SegmentSamplingBudgetTest(unittest.TestCase):
    def test_attempt_budget_exhausts_at_limit(self):
        budget = SegmentSamplingBudget(2)
        self.assertFalse(budget.exhausted)
        budget.record()
        self.assertFalse(budget.exhausted)
        budget.record()
        self.assertTrue(budget.exhausted)

    def test_reset_starts_a_new_segment(self):
        budget = SegmentSamplingBudget(1)
        budget.record()
        self.assertTrue(budget.exhausted)
        budget.reset()
        self.assertFalse(budget.exhausted)
        self.assertEqual(budget.attempts, 0)

    def test_record_after_exhaustion_is_rejected(self):
        budget = SegmentSamplingBudget(1)
        budget.record()
        with self.assertRaises(RuntimeError):
            budget.record()


class SamplingConstraintTest(unittest.TestCase):
    def test_monotonic_mask_preserves_equal_and_future_beats(self):
        event2idx = {
            "Beat_0": 1,
            "Beat_4": 2,
            "Beat_8": 3,
            "Note_Pitch_60": 4,
        }
        self.assertEqual(
            monotonic_beat_inadmissibles(event2idx, 4, base=[9]),
            [1, 9],
        )

    def test_terminal_eos_is_removed_but_final_bar_is_preserved(self):
        idx2event = {1: "Bar_None", 2: "EOS_None"}
        self.assertEqual(strip_terminal_eos([1], idx2event), [1])
        self.assertEqual(strip_terminal_eos([1, 2], idx2event), [1])

    def test_absolute_pitch_stays_masked_across_equal_beat(self):
        event2idx = {
            "Beat_0": 1,
            "Beat_2": 2,
            "Note_Pitch_60": 3,
            "Note_Pitch_62": 4,
        }
        tracker = OnsetPitchTracker()
        tracker.observe("Beat_0")
        tracker.observe("Note_Pitch_60")
        tracker.observe("Beat_0")
        self.assertEqual(tracker.inadmissibles(event2idx, base=[9]), [3, 9])
        tracker.observe("Beat_2")
        self.assertEqual(tracker.inadmissibles(event2idx), None)

    def test_functional_pitch_masks_only_duplicate_degree_for_octave(self):
        event2idx = {
            "Note_Octave_4": 1,
            "Note_Octave_5": 2,
            "Note_Degree_I": 3,
            "Note_Degree_V": 4,
        }
        tracker = OnsetPitchTracker()
        tracker.observe("Beat_0")
        tracker.observe("Note_Octave_4")
        tracker.observe("Note_Degree_V")

        tracker.observe("Note_Octave_4")
        self.assertEqual(tracker.inadmissibles(event2idx), [4])
        tracker.observe("Note_Degree_I")

        tracker.observe("Note_Octave_5")
        self.assertEqual(tracker.inadmissibles(event2idx), None)

    def test_track_boundary_resets_onset_pitch_state(self):
        event2idx = {"Note_Pitch_60": 1}
        tracker = OnsetPitchTracker()
        tracker.observe("Note_Pitch_60")
        self.assertEqual(tracker.inadmissibles(event2idx), [1])
        tracker.observe("Track_Melody")
        self.assertEqual(tracker.inadmissibles(event2idx), None)


if __name__ == "__main__":
    unittest.main()
