import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_stage3_counterfactual_sources import (
    freeze_sources,
    parse_source_argument,
    select_sources,
    sha256_bytes,
)


class PrepareStage3CounterfactualSourcesTests(unittest.TestCase):
    def build_source(self, root, seed, per_quadrant=2):
        source = root / f"seed{seed}" / "stage2_melody"
        source.mkdir(parents=True)
        labels = {
            "Q1": "Positive",
            "Q2": "Negative",
            "Q3": "Negative",
            "Q4": "Positive",
        }
        for quadrant, label in labels.items():
            for index in range(per_quadrant):
                path = source / f"samp_{index:02d}_{label}_{quadrant}_melody.txt"
                path.write_text(
                    f"Emotion_{quadrant}\nKey_C\nTrack_Melody\nBar_None\n{seed}-{quadrant}-{index}\n",
                    encoding="utf-8",
                )
        return source

    def test_selects_lowest_hash_per_seed_and_quadrant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.build_source(root, 42)
            selected = select_sources([(42, source)], expected_per_quadrant=2)
            self.assertEqual(len(selected), 4)
            for row in selected:
                quadrant = row["source_quadrant"]
                group = sorted(source.glob(f"*_{quadrant}_melody.txt"))
                expected = min(
                    group,
                    key=lambda path: (sha256_bytes(path.read_bytes()), path.name),
                )
                self.assertEqual(row["source_path"], expected)

    def test_freeze_writes_balanced_manifest_and_hash_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [
                (42, self.build_source(root, 42)),
                (43, self.build_source(root, 43)),
            ]
            selected = select_sources(sources, expected_per_quadrant=2)
            output = root / "frozen"
            manifest = freeze_sources(selected, output)
            self.assertEqual(manifest["source_count"], 8)
            self.assertEqual(manifest["expected_output_count"], 16)
            self.assertEqual(manifest["seeds"], [42, 43])
            written = json.loads((output / "source_manifest.json").read_text())
            self.assertEqual(written["selection_rule"], manifest["selection_rule"])
            for row in manifest["sources"]:
                path = output / row["frozen_path"]
                self.assertEqual(sha256_bytes(path.read_bytes()), row["source_sha256"])

    def test_rejects_missing_quadrant_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.build_source(root, 42)
            next(source.glob("*_Q4_melody.txt")).unlink()
            with self.assertRaisesRegex(ValueError, "seed 42 Q4"):
                select_sources([(42, source)], expected_per_quadrant=2)

    def test_source_argument(self):
        seed, path = parse_source_argument("44=/tmp/stage2")
        self.assertEqual(seed, 44)
        self.assertEqual(path, Path("/tmp/stage2"))


if __name__ == "__main__":
    unittest.main()

