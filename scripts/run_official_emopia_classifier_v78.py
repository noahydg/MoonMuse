#!/usr/bin/env python3
"""Run the official EMOPIA REMI classifiers for the frozen v78 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# The 2021 upstream code uses deprecated aliases removed in newer NumPy.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

try:
    import torch
    from omegaconf import OmegaConf
except ImportError:  # Allows lightweight mapping/metric tests without the classifier env.
    torch = None
    OmegaConf = None

try:
    from scripts.build_corrected_three_system_population import crop_midi, sha256_file
except ImportError:
    from build_corrected_three_system_population import crop_midi, sha256_file


TASKS = ("ar_va", "arousal", "valence")
GATE_THRESHOLDS = {
    "full": {"ar_va": 0.55, "arousal": 0.80, "valence": 0.65},
    "eight_bar": {"ar_va": 0.40, "arousal": 0.60, "valence": 0.60},
}


def target_for(task: str, quadrant: str) -> str:
    if task == "ar_va":
        return quadrant
    if task == "arousal":
        return "HA" if quadrant in {"Q1", "Q2"} else "LA"
    if task == "valence":
        return "HV" if quadrant in {"Q1", "Q4"} else "LV"
    raise ValueError(f"unsupported task: {task}")


def macro_f1(targets: list[str], predictions: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def summarize(rows: list[dict], labels: list[str]) -> dict:
    targets = [row["target"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    correct = [target == prediction for target, prediction in zip(targets, predictions)]
    matrix = {
        target: {prediction: 0 for prediction in labels}
        for target in labels
    }
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return {
        "n": len(rows),
        "accuracy": sum(correct) / len(correct) if correct else None,
        "macro_f1": macro_f1(targets, predictions, labels) if rows else None,
        "target_counts": dict(Counter(targets)),
        "prediction_counts": dict(Counter(predictions)),
        "confusion_matrix": matrix,
        "token_count": {
            "min": min(int(row["token_count"]) for row in rows) if rows else None,
            "median": float(np.median([int(row["token_count"]) for row in rows])) if rows else None,
            "max": max(int(row["token_count"]) for row in rows) if rows else None,
        },
    }


def load_upstream_module(root: Path):
    remi_dir = root / "midi_cls" / "midi_helper" / "remi"
    module_path = remi_dir / "midi2event.py"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(remi_dir))
    spec = importlib.util.spec_from_file_location("emopia_official_midi2event", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class OfficialRemiClassifier:
    def __init__(self, upstream_root: Path, task: str, device: str = "cpu"):
        if torch is None or OmegaConf is None:
            raise RuntimeError("torch and omegaconf are required for classifier inference")
        self.upstream_root = upstream_root
        self.task = task
        self.device = torch.device(device)
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        config_path = upstream_root / "best_weight" / "remi" / task / "hparams.yaml"
        checkpoint_path = upstream_root / "best_weight" / "remi" / task / "best.ckpt"
        self.config = OmegaConf.load(config_path)
        self.labels = list(self.config.task.labels)

        from midi_cls.src.model.net import SAN

        self.model = SAN(
            num_of_dim=self.config.task.num_of_dim,
            vocab_size=self.config.midi.pad_idx + 1,
            lstm_hidden_dim=self.config.hparams.lstm_hidden_dim,
            embedding_size=self.config.hparams.embedding_size,
            r=self.config.hparams.r,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state = checkpoint["state_dict"]
        state = {
            key.split("model.", 1)[1]: value
            for key, value in source_state.items()
            if key.startswith("model.")
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch for {task}: missing={missing}, unexpected={unexpected}")
        self.model.eval().to(self.device)
        dictionary_path = upstream_root / "midi_cls" / "midi_helper" / "remi" / "dictionary.pkl"
        with dictionary_path.open("rb") as handle:
            self.event_to_int = pickle.load(handle)[0]
        self.remi = load_upstream_module(upstream_root)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dictionary_path = dictionary_path

    def tokenize(self, midi_path: Path) -> list[int]:
        midi_obj = self.remi.analyzer(str(midi_path))
        song_data = self.remi.corpus(midi_obj)
        event_sequence = self.remi.event(song_data)
        tokens = []
        for item in event_sequence:
            name = str(item["name"])
            value = item["value"]
            key = f"{name}_{value}"
            if key in self.event_to_int:
                tokens.append(self.event_to_int[key])
                continue
            if "Velocity" in name:
                fallback = f"{name}_{value - 2}"
                if fallback in self.event_to_int:
                    tokens.append(self.event_to_int[fallback])
                    continue
            # Match the official public inference path: unknown non-velocity
            # events are skipped rather than mapped to an invented token.
        if not tokens:
            raise ValueError(f"no REMI tokens extracted: {midi_path}")
        return tokens

    def predict(self, midi_path: Path) -> tuple[str, list[float], list[float], int]:
        tokens = self.tokenize(midi_path)
        tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits_tensor = self.model(tensor).squeeze(0).detach().cpu()
            probabilities_tensor = torch.softmax(logits_tensor, dim=0)
        logits = [float(value) for value in logits_tensor]
        probabilities = [float(value) for value in probabilities_tensor]
        return self.labels[int(np.argmax(logits))], logits, probabilities, len(tokens)

    def provenance(self) -> dict:
        return {
            "task": self.task,
            "labels": self.labels,
            "config": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "dictionary": str(self.dictionary_path),
            "dictionary_sha256": sha256_file(self.dictionary_path),
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_rows(
    classifier: OfficialRemiClassifier,
    sources: list[dict],
    output_csv: Path,
    progress_label: str,
) -> tuple[list[dict], dict]:
    predictions = []
    started = time.time()
    for index, source in enumerate(sources, start=1):
        midi_path = Path(source["midi_path"])
        expected_hash = source.get("midi_sha256")
        actual_hash = sha256_file(midi_path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"MIDI hash mismatch: {midi_path}")
        prediction, logits, probabilities, token_count = classifier.predict(midi_path)
        target = target_for(classifier.task, source["quadrant"])
        row = {
            "population": source["population"],
            "system": source.get("system", "emopia_real"),
            "seed": source.get("seed", ""),
            "quadrant": source["quadrant"],
            "sample_id": source["sample_id"],
            "midi_path": str(midi_path),
            "midi_sha256": actual_hash,
            "target": target,
            "prediction": prediction,
            "correct": int(target == prediction),
            "token_count": token_count,
        }
        for label, value in zip(classifier.labels, logits):
            row[f"logit_{label}"] = f"{value:.10f}"
        for label, value in zip(classifier.labels, probabilities):
            row[f"probability_{label}"] = f"{value:.10f}"
        predictions.append(row)
        if index == 1 or index % 25 == 0 or index == len(sources):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = (len(sources) - index) / rate if rate else math.inf
            print(
                f"[{progress_label}] {index}/{len(sources)} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
    write_csv(output_csv, predictions)
    return predictions, summarize(predictions, classifier.labels)


def load_emopia_test_sources(args: argparse.Namespace, crop_root: Path) -> tuple[list[dict], list[dict]]:
    with args.emopia_test_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full_sources = []
    cropped_sources = []
    for row in rows:
        sample_id = row["fname"]
        quadrant = row["label"]
        source = args.emopia_midi_root / f"{sample_id}.mid"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        full_sources.append({
            "population": "emopia_test_full",
            "system": "emopia_real",
            "seed": "",
            "quadrant": quadrant,
            "sample_id": sample_id,
            "midi_path": str(source),
            "midi_sha256": source_hash,
        })
        cropped = crop_root / f"{sample_id}.mid"
        if not cropped.exists():
            crop_midi(source, cropped, bars=8)
        cropped_sources.append({
            "population": "emopia_test_eight_bar",
            "system": "emopia_real",
            "seed": "",
            "quadrant": quadrant,
            "sample_id": sample_id,
            "midi_path": str(cropped),
            "midi_sha256": sha256_file(cropped),
        })
    return full_sources, cropped_sources


def load_generated_sources(args: argparse.Namespace) -> list[dict]:
    with args.generated_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 900:
        raise ValueError(f"expected 900 generated rows, found {len(rows)}")
    root = args.generated_manifest.parent
    sources = []
    for row in rows:
        sources.append({
            "population": "generated_eight_bar",
            "system": row["system"],
            "seed": row["seed"],
            "quadrant": row["quadrant"],
            "sample_id": f"{row['system']}-s{row['seed']}-{row['quadrant']}-{int(row['sample_index']):02d}",
            "midi_path": str(root / row["standardized_midi"]),
            "midi_sha256": row["standardized_sha256"],
        })
    return sources


def generated_summary(rows: list[dict], labels: list[str]) -> dict:
    by_system_seed: dict[str, dict[str, dict]] = defaultdict(dict)
    for system in sorted({row["system"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows if row["system"] == system}):
            group = [row for row in rows if row["system"] == system and row["seed"] == seed]
            by_system_seed[system][str(seed)] = summarize(group, labels)
    by_system = {}
    for system, seed_map in by_system_seed.items():
        accuracy = [seed_map[str(seed)]["accuracy"] for seed in sorted(seed_map, key=int)]
        f1 = [seed_map[str(seed)]["macro_f1"] for seed in sorted(seed_map, key=int)]
        pooled = [row for row in rows if row["system"] == system]
        by_system[system] = {
            "seed_accuracy": accuracy,
            "seed_macro_f1": f1,
            "mean_accuracy": float(np.mean(accuracy)),
            "sample_sd_accuracy": float(np.std(accuracy, ddof=1)),
            "mean_macro_f1": float(np.mean(f1)),
            "sample_sd_macro_f1": float(np.std(f1, ddof=1)),
            "pooled_descriptive": summarize(pooled, labels),
        }
    differences = {}
    for comparator in ("moonmuse_c2p", "emo_disentangler"):
        differences[f"moonmuse_full_minus_{comparator}"] = {
            str(seed): {
                "accuracy": (
                    by_system_seed["moonmuse_full"][str(seed)]["accuracy"]
                    - by_system_seed[comparator][str(seed)]["accuracy"]
                ),
                "macro_f1": (
                    by_system_seed["moonmuse_full"][str(seed)]["macro_f1"]
                    - by_system_seed[comparator][str(seed)]["macro_f1"]
                ),
            }
            for seed in (42, 43, 44)
        }
    return {
        "by_system_seed": by_system_seed,
        "by_system": by_system,
        "paired_seed_differences": differences,
    }


def run(args: argparse.Namespace) -> dict:
    if torch is None:
        raise RuntimeError("torch is required for classifier inference")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    if (args.output_dir / "CLASSIFIER_V78_COMPLETE").exists():
        raise FileExistsError(f"result already frozen: {args.output_dir}")
    torch.set_num_threads(args.torch_threads)
    full_sources, cropped_sources = load_emopia_test_sources(
        args, args.output_dir / "emopia_test_eight_bar_midi"
    )
    generated_sources = load_generated_sources(args)
    report = {
        "schema": "moonmuse-official-emopia-classifier-v78",
        "status": "running_real_data_eligibility_gate",
        "protocol": str(args.protocol),
        "classifier_repository": str(args.upstream_root),
        "classifier_commit": official_commit(args.upstream_root),
        "generated_manifest": str(args.generated_manifest),
        "generated_manifest_sha256": sha256_file(args.generated_manifest),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
            "torch_threads": args.torch_threads,
        },
        "gate_thresholds": GATE_THRESHOLDS,
        "tasks": {},
    }
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    classifiers = {}
    gate_pass = True
    for task in TASKS:
        classifier = OfficialRemiClassifier(args.upstream_root, task, args.device)
        classifiers[task] = classifier
        full_rows, full_summary = predict_rows(
            classifier,
            full_sources,
            args.output_dir / "predictions" / f"emopia_test_full_{task}.csv",
            f"real-full:{task}",
        )
        crop_rows, crop_summary = predict_rows(
            classifier,
            cropped_sources,
            args.output_dir / "predictions" / f"emopia_test_eight_bar_{task}.csv",
            f"real-8bar:{task}",
        )
        full_pass = full_summary["accuracy"] >= GATE_THRESHOLDS["full"][task]
        crop_pass = crop_summary["accuracy"] >= GATE_THRESHOLDS["eight_bar"][task]
        gate_pass = gate_pass and full_pass and crop_pass
        report["tasks"][task] = {
            "classifier": classifier.provenance(),
            "emopia_test_full": full_summary,
            "emopia_test_eight_bar": crop_summary,
            "full_threshold": GATE_THRESHOLDS["full"][task],
            "eight_bar_threshold": GATE_THRESHOLDS["eight_bar"][task],
            "full_pass": full_pass,
            "eight_bar_pass": crop_pass,
        }
        status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report["eligibility_gate_pass"] = gate_pass
    if not gate_pass:
        report["status"] = "classifier_eligibility_gate_failed"
        report["generated_predictions_opened"] = False
        status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "CLASSIFIER_V78_GATE_FAILED").write_text("\n", encoding="utf-8")
        return report

    report["status"] = "running_generated_population"
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for task, classifier in classifiers.items():
        rows, _ = predict_rows(
            classifier,
            generated_sources,
            args.output_dir / "predictions" / f"generated_eight_bar_{task}.csv",
            f"generated:{task}",
        )
        report["tasks"][task]["generated"] = generated_summary(rows, classifier.labels)
        status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report["status"] = "complete"
    report["generated_predictions_opened"] = True
    status_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "CLASSIFIER_V78_COMPLETE").write_text(
        sha256_file(status_path) + "  status.json\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--emopia-midi-root", type=Path, required=True)
    parser.add_argument("--emopia-test-csv", type=Path, required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    final = run(parse_args())
    print(json.dumps({"status": final["status"], "eligibility_gate_pass": final.get("eligibility_gate_pass")}, indent=2))
