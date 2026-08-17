import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_formal_run.sh"


class PackageFormalRunTests(unittest.TestCase):
    def test_two_stage_performance_text_is_packaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "two-stage-c2p-seed42-test"
            inference_dir = run_dir / "inference-seed42"
            artifact_dir = base / "artifacts"

            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "checkpoints" / "c2p_finetune" / "params").mkdir(
                parents=True
            )
            (inference_dir / "midi").mkdir(parents=True)
            (inference_dir / "stage1_chord").mkdir()
            (inference_dir / "stage2_performance").mkdir()
            (run_dir / "training_validation.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            (inference_dir / "generation_gate.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            (run_dir / "logs" / "training.log").write_text("ok\n", encoding="utf-8")
            (
                run_dir
                / "checkpoints"
                / "c2p_finetune"
                / "params"
                / "best_params.pt"
            ).write_bytes(b"checkpoint")
            (inference_dir / "midi" / "sample.mid").write_bytes(b"MThd")
            (inference_dir / "stage1_chord" / "sample.txt").write_text(
                "chord\n", encoding="utf-8"
            )
            performance = inference_dir / "stage2_performance" / "sample.txt"
            performance.write_text("performance\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(SCRIPT), str(run_dir), str(artifact_dir)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            samples = artifact_dir / f"{run_dir.name}-samples.tar.gz"
            with tarfile.open(samples, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("inference-seed42/stage2_performance/sample.txt", names)


if __name__ == "__main__":
    unittest.main()
