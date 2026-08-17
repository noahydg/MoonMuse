from __future__ import annotations

import unittest

from scripts.validate_v82_prelaunch import evaluate


class V82PrelaunchValidationTest(unittest.TestCase):
    def complete_inputs(self):
        study = {
            "selection": {
                "selected_slots": 24,
                "selected_stimuli": 48,
                "selection_uses_classifier": False,
                "selection_uses_v74": False,
                "selection_uses_human_judgment": False,
            }
        }
        audio = {
            "settings": {"per_clip_loudness_normalization": False, "global_gain_db": 1.0},
            "verification": {
                "all_true_peaks_pass": True,
                "all_relative_loudness_offsets_preserved_within_0_35_db": True,
                "all_files_hash_verified": True,
            },
        }
        questionnaire = {
            "status": "prelaunch_static_validation_passed",
            "formalStudyData": False,
            "cross_form_stimulus_overlap": 0,
        }
        positive = {
            "prelaunchSyntheticData": True,
            "confirmatory": {"overall_bounded_baseline_outperformance_claim_passed": True},
        }
        negative = {
            "prelaunchSyntheticData": True,
            "confirmatory": {"overall_bounded_baseline_outperformance_claim_passed": False},
        }
        return study, audio, questionnaire, positive, negative

    def test_complete_automation_still_does_not_authorize_launch(self):
        report = evaluate(*self.complete_inputs())
        self.assertTrue(report["automated_prelaunch_passed"])
        self.assertFalse(report["formal_launch_ready"])
        self.assertFalse(report["baseline_outperformance_claim_supported"])

    def test_positive_control_failure_blocks_automation(self):
        inputs = list(self.complete_inputs())
        inputs[3]["confirmatory"]["overall_bounded_baseline_outperformance_claim_passed"] = False
        report = evaluate(*inputs)
        self.assertFalse(report["automated_prelaunch_passed"])


if __name__ == "__main__":
    unittest.main()
