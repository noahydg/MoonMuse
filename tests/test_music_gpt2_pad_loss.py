import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "stage2_melody/model/music_gpt2.py",
    ROOT / "stage3_performance/model/music_gpt2.py",
)
VALIDATOR_PATH = ROOT / "scripts/validate_music_gpt2_pad_loss.py"


class MusicGPT2PadLossTests(unittest.TestCase):
    def test_improved_model_defaults_to_its_pad_token(self):
        for path in MODEL_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            improved = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "ImprovedMusicGPT2"
            )
            compute_loss = next(
                node
                for node in improved.body
                if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
            )
            self.assertIsInstance(compute_loss.args.defaults[-1], ast.Constant, path)
            self.assertIsNone(compute_loss.args.defaults[-1].value, path)
            source = ast.unparse(compute_loss)
            self.assertIn("ignore_index = self.n_token - 1", source, path)
            self.assertIn("ignore_index=ignore_index", source, path)

    def test_tensor_validator_adds_repository_root_to_import_path(self):
        tree = ast.parse(VALIDATOR_PATH.read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        self.assertIn("Path(__file__).resolve().parents[1]", source)
        self.assertIn("sys.path.insert(0, str(ROOT))", source)


if __name__ == "__main__":
    unittest.main()
