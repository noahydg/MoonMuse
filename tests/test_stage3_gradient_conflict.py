import json
import tempfile
import unittest
from pathlib import Path

from stage3_gradient_conflict import (
    classify_tempo_velocity_interaction,
    summarize_gradient_records,
)
from scripts.validate_stage3_gradient_conflict_audit import validate


def record(cosine, tempo_norm=1.0, velocity_norm=1.0):
    scopes = {}
    for scope in ("full_model", "shared_backbone", "input_conditioning", "output_head"):
        scopes[scope] = {
            "norms": {
                "total_tempo": tempo_norm,
                "total_velocity": velocity_norm,
                "total_duration": 0.5,
            },
            "cosines": {"total_tempo__total_velocity": cosine},
        }
    return {"scopes": scopes}


class Stage3GradientConflictTests(unittest.TestCase):
    def test_summary_and_route_detect_declared_shared_conflict(self):
        summary = summarize_gradient_records([record(-0.2) for _ in range(12)] + [record(0.1) for _ in range(4)])
        route = classify_tempo_velocity_interaction(summary)
        self.assertEqual("shared_gradient_conflict_supported", route["label"])
        self.assertEqual(0.75, summary["scopes"]["shared_backbone"]["cosines"]["total_tempo__total_velocity"]["negative_fraction"])

    def test_route_detects_tempo_underpower_without_conflict(self):
        summary = summarize_gradient_records([record(0.2, tempo_norm=0.25, velocity_norm=1.0) for _ in range(16)])
        route = classify_tempo_velocity_interaction(summary)
        self.assertEqual("tempo_gradient_underpowered", route["label"])
        self.assertEqual(0.25, route["tempo_to_velocity_norm_ratio"])

    def test_validator_requires_read_only_complete_report(self):
        records = [record(0.1) for _ in range(16)]
        summary = summarize_gradient_records(records)
        report = {
            "schema": "moonmuse-stage3-gradient-conflict-audit-v1",
            "training_performed": False,
            "checkpoint_modified": False,
            "gradient_summary": summary,
            "diagnostic_route": classify_tempo_velocity_interaction(summary),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = validate(path, 16)
        self.assertTrue(result["passed"], result["errors"])

    def test_runner_is_read_only_and_uses_validation_split(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/run_stage3_gradient_conflict_audit.sh").read_text("utf-8")
        self.assertIn("training_performed=false", script)
        self.assertIn("--split valid", script)
        self.assertIn("checkpoint_before", script)
        self.assertIn("checkpoint_after", script)
        self.assertNotIn("train.py", script)


if __name__ == "__main__":
    unittest.main()
