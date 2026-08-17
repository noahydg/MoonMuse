import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_pilot.py"


class ValidatePilotTests(unittest.TestCase):
    def make_phase(self, run_dir, phase, *, integrity_passed=True):
        phase_dir = run_dir / "checkpoints" / phase
        (phase_dir / "params").mkdir(parents=True)
        (phase_dir / "optim").mkdir()
        (phase_dir / "params" / "best_params.pt").write_bytes(b"params")
        (phase_dir / "optim" / "best_optim.pt").write_bytes(b"optimizer")
        (phase_dir / "valloss.txt").write_text(
            "ep1 loss: 1.250 valloss: 1.125\n"
            "ep2 loss: 1.000 valloss: 0.875\n",
            encoding="utf-8",
        )
        (phase_dir / "checkpoint_integrity.json").write_text(
            json.dumps({"passed": integrity_passed}),
            encoding="utf-8",
        )

    def run_validator(self, run_dir, output, *phases):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(run_dir),
                "--output",
                str(output),
                "--require-checkpoint-integrity",
                "--phases",
                *phases,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_custom_two_stage_phases_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.make_phase(run_dir, "c2p_pretrain")
            self.make_phase(run_dir, "c2p_finetune")
            output = run_dir / "training_validation.json"

            result = self.run_validator(
                run_dir, output, "c2p_pretrain", "c2p_finetune"
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["requested_phases"], ["c2p_pretrain", "c2p_finetune"]
            )
            self.assertEqual(set(report["phases"]), set(report["requested_phases"]))

    def test_failed_integrity_fails_custom_phase_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self.make_phase(run_dir, "c2p_pretrain", integrity_passed=False)
            output = run_dir / "training_validation.json"

            result = self.run_validator(run_dir, output, "c2p_pretrain")

            self.assertEqual(result.returncode, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertFalse(report["phases"]["c2p_pretrain"]["passed"])


if __name__ == "__main__":
    unittest.main()
