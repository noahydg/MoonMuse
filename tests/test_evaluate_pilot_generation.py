import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_pilot_generation.py"


class EvaluatePilotGenerationTests(unittest.TestCase):
    def test_explicit_stage2_performance_directory_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            inference_dir = Path(tmp)
            events = inference_dir / "stage2_performance"
            events.mkdir()
            for quadrant in ("Q1", "Q2", "Q3", "Q4"):
                (events / f"samp_00_Positive_{quadrant}_performance.txt").touch()

            output = inference_dir / "generation_gate.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(inference_dir),
                    "--event-dir-name",
                    "stage2_performance",
                    "--expected-candidates",
                    "4",
                    "--min-valid-midi",
                    "0",
                    "--min-valid-per-quadrant",
                    "0",
                    "--min-median-notes",
                    "0",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["event_dir_name"], "stage2_performance")
            self.assertEqual(report["candidate_count"], 4)


if __name__ == "__main__":
    unittest.main()
