import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.audit_training_target_mask import (
    admissible_start_bars,
    audit,
    audit_piece,
)


def payload(event_count=20):
    condition_positions = [(2, 5), (10, 13)]
    target_positions = [(5, 10), (13, 20)]
    return [condition_positions, target_positions, [{}] * event_count]


class TrainingTargetMaskAuditTests(unittest.TestCase):
    def test_short_piece_has_one_window_and_counts_target_positions(self):
        rows = audit_piece(payload(), "Q1_piece.pkl", max_len=24)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_token_count"], 12)
        self.assertEqual(rows[0]["pad_target_count"], 12)
        self.assertEqual(rows[0]["pad_target_ratio"], 0.5)

    def test_long_piece_exposes_all_admissible_start_bars(self):
        positions = [(2, 5), (10, 13), (18, 21)]
        self.assertEqual(admissible_start_bars(30, positions, 16), [0, 1, 2])

    def test_split_audit_reports_multiple_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "events"
            event_dir.mkdir()
            filename = "Q2_piece.pkl"
            (event_dir / filename).write_bytes(pickle.dumps(payload()))
            split = root / "split.pkl"
            split.write_bytes(pickle.dumps([filename]))
            report = audit(event_dir, split, max_len=12)
            self.assertEqual(report["summary"]["piece_count"], 1)
            self.assertEqual(report["summary"]["admissible_window_count"], 2)
            self.assertEqual(
                report["summary"]["pieces_with_multiple_admissible_windows"], 1
            )


if __name__ == "__main__":
    unittest.main()
