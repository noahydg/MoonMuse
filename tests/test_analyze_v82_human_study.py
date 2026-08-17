from __future__ import annotations

import unittest

from scripts.analyze_v82_human_study import confirmatory, lower_t_bound


class V82HumanStudyAnalysisTest(unittest.TestCase):
    def rows(self, n, arousal, quality):
        return [
            {
                "arousal_candidate_minus_baseline": arousal,
                "quality_candidate_minus_baseline": quality,
            }
            for _ in range(n)
        ]

    def test_constant_shift_has_exact_lower_bound(self):
        report = lower_t_bound([0.2] * 24)
        self.assertAlmostEqual(report["lower_confidence_bound"], 0.2)

    def test_both_facets_and_both_endpoints_are_required(self):
        passing = confirmatory(self.rows(72, 0.25, 0.0), self.rows(24, 0.20, 0.0))
        self.assertTrue(passing["overall_bounded_baseline_outperformance_claim_passed"])
        failed_quality = confirmatory(self.rows(72, 0.25, -0.5), self.rows(24, 0.20, -0.5))
        self.assertFalse(failed_quality["quality_noninferiority"]["passed"])
        self.assertFalse(failed_quality["overall_bounded_baseline_outperformance_claim_passed"])
        failed_stimulus = confirmatory(self.rows(72, 0.25, 0.0), self.rows(24, -0.05, 0.0))
        self.assertFalse(failed_stimulus["arousal_superiority"]["passed"])


if __name__ == "__main__":
    unittest.main()
