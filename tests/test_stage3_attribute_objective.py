import json
import tempfile
import unittest
from pathlib import Path

from stage3_attribute_objective import build_attribute_objective_spec
from scripts.validate_stage3_attribute_objective_smoke import validate_run


class Stage3AttributeObjectiveTest(unittest.TestCase):
    def test_spec_selects_only_declared_performance_attributes(self):
        event2idx = {
            "Tempo_60": 1,
            "Tempo_120": 2,
            "Note_Duration_120": 3,
            "Note_Velocity_80": 4,
            "Beat_0": 5,
            "Note_Degree_I": 6,
        }
        spec = build_attribute_objective_spec(
            event2idx,
            {
                "mode": "performance_attributes_only",
                "attributes": ["tempo", "duration", "velocity"],
            },
        )
        self.assertEqual([1, 2, 3, 4], spec["token_indices"])
        self.assertNotIn("Beat_0", spec["token_names"])

    def test_unknown_attribute_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported performance attributes"):
            build_attribute_objective_spec(
                {"Tempo_60": 1},
                {"mode": "performance_attributes_only", "attributes": ["pitch"]},
            )

    def test_smoke_validator_requires_objective_log_config_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "checkpoints/stage3_attribute_objective_smoke"
            (root / "logs").mkdir(parents=True)
            (train / "params").mkdir(parents=True)
            (train / "optim").mkdir()
            (root / "logs/training.log").write_text(
                "[reconstruction-objective] mode=performance_attributes_only attributes=['tempo', 'duration', 'velocity'] token_count=9\n"
                "[arousal-alignment] enabled weight=0.2\n"
                "* loss = 1.2345, total_acc = 0.1, step = 100\n"
                "* recons_loss = 1.2000, alignment_loss = 0.1725, alignment_weight = 0.200\n",
                encoding="utf-8",
            )
            for name in ("source_checkpoint_integrity.json", "smoke_checkpoint_integrity.json"):
                (root / name).write_text(json.dumps({"passed": True}), encoding="utf-8")
            (train / "params/smoke_params.pt").write_bytes(b"params")
            (train / "optim/smoke_optim.pt").write_bytes(b"optim")
            (train / "config.yaml").write_text(
                "reconstruction_objective:\n"
                "  mode: performance_attributes_only\n"
                "  attributes: [tempo, duration, velocity]\n",
                encoding="utf-8",
            )
            report = validate_run(root, 100)
        self.assertTrue(report["passed"])
        self.assertEqual("performance_attributes_only", report["reconstruction_mode"])

    def test_runner_resets_optimizer_and_validates_before_pass_marker(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_stage3_attribute_objective_smoke.sh"
        ).read_text("utf-8")
        self.assertIn("--reset_optimizer", script)
        self.assertIn("emopia_finetune_attribute_only_alignment_gpt2.yaml", script)
        self.assertLess(
            script.index("validate_stage3_attribute_objective_smoke.py"),
            script.index("STAGE3_ATTRIBUTE_OBJECTIVE_SMOKE_PASS"),
        )


if __name__ == "__main__":
    unittest.main()
