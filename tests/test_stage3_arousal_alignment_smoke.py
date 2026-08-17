import json
import tempfile
import unittest
from pathlib import Path

from scripts.show_stage3_arousal_alignment_progress import build_dashboard, parse_progress
from scripts.validate_stage3_arousal_alignment_smoke import validate_run


class Stage3ArousalAlignmentSmokeTests(unittest.TestCase):
    def test_progress_parser_exposes_batches_and_alignment_loss(self):
        self.assertEqual((100, 0.1234), parse_progress(
            "step = 50\nalignment_loss = 0.2000\nstep = 100\nalignment_loss = 0.1234\n"
        ))

    def test_validator_requires_exact_steps_losses_config_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "checkpoints/stage3_arousal_alignment_smoke"
            (root / "logs").mkdir(parents=True)
            (train / "params").mkdir(parents=True)
            (train / "optim").mkdir()
            (root / "logs/training.log").write_text(
                "[arousal-alignment] enabled weight=0.2\n"
                "* loss = 1.2345, total_acc = 0.1, step = 100\n"
                "* recons_loss = 1.2000, alignment_loss = 0.1725, alignment_weight = 0.200\n",
                encoding="utf-8",
            )
            for name in ("migrated_checkpoint_integrity.json", "smoke_checkpoint_integrity.json"):
                (root / name).write_text(json.dumps({"passed": True}), encoding="utf-8")
            (train / "params/smoke_params.pt").write_bytes(b"params")
            (train / "optim/smoke_optim.pt").write_bytes(b"optim")
            (train / "config.yaml").write_text(
                "arousal_alignment:\n  enabled: true\n  weight: 0.20\n", encoding="utf-8"
            )
            report = validate_run(root, expected_steps=100)
            self.assertTrue(report["passed"])
            self.assertEqual(50, report["optimizer_updates"])
            self.assertEqual("runnability_only_not_music_quality", report["evidence_boundary"])

    def test_dashboard_keeps_smoke_quality_boundary_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "logs/training.log").write_text(
                "step = 40\nalignment_loss = 0.2000\n", encoding="utf-8"
            )
            dashboard = build_dashboard(root)
            self.assertIn("40 / 100 batches", dashboard)
            self.assertIn("20 / 50", dashboard)
            self.assertIn("交叉熵 + 固定源反事实唤醒度对齐", dashboard)

    def test_runner_uses_alignment_config_and_post_training_validator(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/run_stage3_arousal_alignment_smoke.sh").read_text("utf-8")
        self.assertIn("emopia_finetune_factorized_va_alignment_gpt2.yaml", script)
        self.assertIn("validate_stage3_arousal_alignment_smoke.py", script)
        self.assertLess(
            script.index("validate_stage3_arousal_alignment_smoke.py"),
            script.index("STAGE3_AROUSAL_ALIGNMENT_SMOKE_PASS"),
        )


if __name__ == "__main__":
    unittest.main()
