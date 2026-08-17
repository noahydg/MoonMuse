import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_training_pad_loss_smoke import validate_run


class TrainingPadLossSmokeTests(unittest.TestCase):
    def test_validator_accepts_exact_two_stage_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            for stage in ("stage2", "stage3"):
                checkpoint = root / f"checkpoints/{stage}"
                (checkpoint / "params").mkdir(parents=True)
                (checkpoint / "optim").mkdir()
                (checkpoint / "params/smoke_params.pt").write_bytes(b"params")
                (checkpoint / "optim/smoke_optim.pt").write_bytes(b"optim")
                (root / f"logs/{stage}_training.log").write_text(
                    "* loss = 1.2345, total_acc = 0.5, step = 100\n"
                    "[smoke] reached 100 steps; checkpoint saved\n",
                    encoding="utf-8",
                )
                for role in ("source", "smoke"):
                    (root / f"{stage}_{role}_checkpoint_integrity.json").write_text(
                        json.dumps({"passed": True}), encoding="utf-8"
                    )
            (root / "pad_loss_tensor_validation.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            report = validate_run(root, expected_steps=100)
        self.assertTrue(report["passed"])
        self.assertEqual(100, report["stage2"]["observed_max_step"])
        self.assertEqual(100, report["stage3"]["observed_max_step"])

    def test_runner_resets_optimizers_and_validates_before_pass(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/run_training_pad_loss_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(2, script.count("--reset_optimizer"))
        self.assertIn("validate_music_gpt2_pad_loss.py", script)
        self.assertLess(
            script.index("validate_training_pad_loss_smoke.py"),
            script.index("TRAINING_PAD_LOSS_SMOKE_PASS"),
        )


if __name__ == "__main__":
    unittest.main()
