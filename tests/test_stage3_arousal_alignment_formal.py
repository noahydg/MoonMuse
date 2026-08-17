import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.show_stage3_arousal_alignment_gate_c_progress import build_dashboard
from scripts.validate_stage3_arousal_alignment_formal_run import validate_run


ROOT = Path(__file__).resolve().parents[1]


class Stage3ArousalAlignmentFormalTests(unittest.TestCase):
    def make_run(self, root, seed=42, status=True):
        run = root / "run"
        checkpoint = run / "checkpoints/stage3_arousal_alignment_finetune"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "optim").mkdir()
        manifest = "\n".join([
            "protocol=stage3-arousal-alignment-formal-v1",
            "seed={}".format(seed),
            "conditioning=factorized_va",
            "optimizer_transfer=reset_after_checkpoint_migration",
            "selection=lowest_total_validation_objective",
            "objective=token_cross_entropy_plus_paired_counterfactual_directional_hinge_v1",
            "alignment_weight=0.20",
        ]) + "\n"
        (run / "training_manifest.txt").write_text(manifest, encoding="utf-8")
        if status:
            (run / "status.txt").write_text(
                "STAGE3_AROUSAL_ALIGNMENT_FORMAL_PASS seed={} run_dir={}\n".format(seed, run),
                encoding="utf-8",
            )
        (run / "arousal_supervision_audit.json").write_text(json.dumps({
            "gates": {"labels": True, "balance": True, "signal": True},
            "recommended_intervention": "explicit_counterfactual_arousal_alignment",
        }), encoding="utf-8")
        params = checkpoint / "params/best_params.pt"
        optim = checkpoint / "optim/best_optim.pt"
        params.write_bytes(b"p" * 1_000_001)
        optim.write_bytes(b"o" * 1_000_001)
        empty_compat = {"missing_keys": [], "unexpected_keys": [], "mismatched_shapes": []}
        (run / "migrated_checkpoint_integrity.json").write_text(
            json.dumps({"passed": True, "nonfinite_keys": []}), encoding="utf-8"
        )
        (run / "formal_checkpoint_integrity.json").write_text(json.dumps({
            "passed": True,
            "nonfinite_keys": [],
            "reference_compatibility": empty_compat,
            "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
            "optimizer_sha256": hashlib.sha256(optim.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        (checkpoint / "config.yaml").write_text(
            "arousal_alignment:\n  enabled: true\n  objective: paired_counterfactual_directional_hinge_v1\n  weight: 0.20\n",
            encoding="utf-8",
        )
        (checkpoint / "valloss.txt").write_text(
            "ep001 | loss: 0.800 | valloss: 0.700 (±0.01) | val_recons: 0.690 | val_alignment: 0.050 | alignment_weight: 0.200\n",
            encoding="utf-8",
        )
        return run

    def test_complete_formal_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_run(self.make_run(Path(directory)), 42)
            self.assertTrue(report["passed"])
            self.assertEqual(0.7, report["best_total_validation_objective"])
            self.assertEqual(0.05, report["best_alignment_validation_loss"])

    def test_pre_status_validation_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory), status=False)
            with self.assertRaises(ValueError):
                validate_run(run, 42)
            self.assertTrue(validate_run(run, 42, require_status=False)["passed"])

    def test_shell_runners_parse_and_use_alignment_checkpoint(self):
        for name in (
            "run_stage3_arousal_alignment_formal_seed.sh",
            "run_stage3_arousal_alignment_campaign.sh",
            "run_stage3_arousal_alignment_gate_c.sh",
            "run_stage3_arousal_alignment_pipeline.sh",
        ):
            self.assertTrue((ROOT / "scripts" / name).is_file())
        gate = (ROOT / "scripts/run_stage3_arousal_alignment_gate_c.sh").read_text("utf-8")
        self.assertIn("stage3_arousal_alignment_finetune/params/best_params.pt", gate)
        self.assertIn("evaluate_stage3_factorized_gate_c.py", gate)

    def test_dashboard_shows_separate_training_and_gate_c(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            output = root / "output"
            campaign.mkdir()
            output.mkdir()
            dashboard = build_dashboard(campaign, output)
            self.assertIn("三 seed 匹配正式训练", dashboard)
            self.assertIn("固定12源反事实复测", dashboard)
            self.assertIn("0/24", dashboard)


if __name__ == "__main__":
    unittest.main()
