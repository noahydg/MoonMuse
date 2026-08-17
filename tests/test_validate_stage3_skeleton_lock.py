import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.validate_stage3_skeleton_lock import validate


def write_tokens(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


class ValidateStage3SkeletonLockTest(unittest.TestCase):
    def test_direct_cli_entrypoint_imports_sibling_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "stage2_melody_conn" / "source_Q1_x_melody.txt"
            performance = (
                root / "stage3_performance" / "source_Q1_x_melody_Q1_performance.txt"
            )
            melody = [
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_4",
                "Note_Degree_I",
            ]
            write_tokens(condition, melody)
            write_tokens(
                performance,
                melody
                + [
                    "Track_Performance",
                    "Bar_None",
                    "Beat_0",
                    "Note_Octave_4",
                    "Note_Degree_I",
                ],
            )
            output = root / "report.json"
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "validate_stage3_skeleton_lock.py"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(root),
                    "--expected-files",
                    "1",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exact_skeleton_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "stage2_melody_conn" / "source_Q1_x_melody.txt"
            performance = (
                root / "stage3_performance" / "source_Q1_x_melody_Q1_performance.txt"
            )
            melody = [
                "Key_C",
                "Track_Melody",
                "Bar_None",
                "Beat_0",
                "Note_Octave_4",
                "Note_Degree_I",
            ]
            write_tokens(condition, melody)
            write_tokens(
                performance,
                melody
                + [
                    "Track_Performance",
                    "Bar_None",
                    "Beat_0",
                    "Note_Octave_4",
                    "Note_Degree_I",
                    "Note_Duration_120",
                    "Note_Velocity_80",
                ],
            )
            report = validate(root, 1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["counter_equal_count"], 1)

    def test_pitch_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "stage2_melody_conn" / "source_Q1_x_melody.txt"
            performance = (
                root / "stage3_performance" / "source_Q1_x_melody_Q1_performance.txt"
            )
            write_tokens(
                condition,
                ["Track_Melody", "Bar_None", "Beat_0", "Note_Octave_4", "Note_Degree_I"],
            )
            write_tokens(
                performance,
                [
                    "Track_Performance",
                    "Bar_None",
                    "Beat_0",
                    "Note_Octave_4",
                    "Note_Degree_V",
                    "Note_Duration_120",
                    "Note_Velocity_80",
                ],
            )
            report = validate(root, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["total_extra_note_group_count"], 1)
        self.assertEqual(report["total_missing_note_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
