from __future__ import annotations

import unittest

from scripts.render_v82_global_gain_audio import (
    compute_initial_global_gain,
    shared_processing_filter,
)


class V82GlobalGainAudioTest(unittest.TestCase):
    def test_processing_preserves_leading_timeline_and_has_no_normalizer(self):
        value = shared_processing_filter(12.0)
        self.assertIn("apad", value)
        self.assertIn("atrim=duration=12.000", value)
        self.assertNotIn("silenceremove", value)
        self.assertNotIn("loudnorm", value)

    def test_global_gain_uses_median_when_peak_allows(self):
        report = compute_initial_global_gain(
            [-24.0, -22.0, -20.0], [-8.0, -7.0, -6.0]
        )
        self.assertAlmostEqual(report["global_gain_db"], 2.0)
        self.assertFalse(report["peak_constraint_bound_initial_gain"])

    def test_global_gain_uses_single_peak_safe_limit(self):
        report = compute_initial_global_gain(
            [-30.0, -29.0, -28.0], [-2.5, -5.0, -4.0]
        )
        self.assertAlmostEqual(report["peak_safe_gain_db"], 0.6)
        self.assertAlmostEqual(report["global_gain_db"], 0.6)
        self.assertTrue(report["peak_constraint_bound_initial_gain"])


if __name__ == "__main__":
    unittest.main()
