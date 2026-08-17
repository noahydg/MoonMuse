from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v82_questionnaires import build_form, validate_ethics_record


class V82QuestionnaireBuildTest(unittest.TestCase):
    def test_prelaunch_form_has_no_private_labels_and_is_not_formal(self):
        ids = [f"V82{index:03d}" for index in range(1, 25)]
        audio = {
            stimulus_id: {
                "sha256": hashlib.sha256(stimulus_id.encode()).hexdigest(),
                "data": "data:audio/mpeg;base64,AA==",
            }
            for stimulus_id in ids
        }
        html = build_form("F1", ids, audio, "prelaunch", None)
        self.assertIn('"formalStudyData":false', html)
        self.assertIn("禁止发给参与者", html)
        self.assertNotIn("moonmuse_v70", html)
        self.assertNotIn("emo_disentangler", html)
        self.assertIn("played_fraction<.8", html)

    def test_launch_ethics_record_requires_complete_real_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ethics.json"
            path.write_text(json.dumps({"decision": "approved"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                validate_ethics_record(path)


if __name__ == "__main__":
    unittest.main()
