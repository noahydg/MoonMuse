import unittest
from pathlib import Path

from scripts.evaluate_stage3_attribute_objective_seed_pilot import evaluate
from scripts.filter_stage3_source_manifest import filter_manifest


class Stage3AttributeObjectiveFormalTest(unittest.TestCase):
    def test_manifest_filter_keeps_exact_seed_population(self):
        value = {"sources": [{"seed": 42, "id": i} for i in range(4)] + [{"seed": 43, "id": i} for i in range(4)]}
        result = filter_manifest(value, 42)
        self.assertEqual(4, result["source_count"])
        self.assertEqual(8, result["expected_output_count"])
        self.assertEqual({42}, {row["seed"] for row in result["sources"]})

    def test_pilot_gate_passes_only_at_declared_thresholds(self):
        skeleton = {"passed": True, "counter_equal_count": 8, "total_extra_note_group_count": 0, "total_missing_note_group_count": 0}
        objective = {
            "execution": {"all_critical_stages_passed": True, "all_midi_valid": True, "valid_midi_count": 8},
            "quality": {"individual_file_count": 8, "individual_quality_pass_count": 3},
            "effect_summary": [
                {"metric": "mean_velocity", "pair_count": 4, "expected_win_count": 3},
                {"metric": "time_weighted_tempo_bpm", "pair_count": 4, "expected_win_count": 3},
            ],
        }
        self.assertTrue(evaluate(skeleton, objective)["passed"])
        objective["effect_summary"][0]["expected_win_count"] = 2
        self.assertFalse(evaluate(skeleton, objective)["passed"])

    def test_runners_freeze_objective_and_skeleton_lock(self):
        root = Path(__file__).resolve().parents[1]
        training = (root / "scripts/run_stage3_attribute_objective_formal_seed.sh").read_text("utf-8")
        generation = (root / "scripts/run_stage3_attribute_objective_seed_pilot.sh").read_text("utf-8")
        self.assertIn("emopia_finetune_attribute_only_alignment_gpt2.yaml", training)
        self.assertIn("--reset_optimizer", training)
        self.assertIn("--lock-performance-to-condition-skeleton", generation)
        self.assertIn("--expected-files 8", generation)


if __name__ == "__main__":
    unittest.main()
