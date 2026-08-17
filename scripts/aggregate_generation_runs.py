#!/usr/bin/env python3
"""Aggregate passed generation runs without treating MIDI files as training repeats."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_manifest(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_result_dir(run_dir):
    if (run_dir / "generation_gate.json").is_file():
        return run_dir
    matches = sorted(
        path.parent
        for path in run_dir.glob("inference-seed*/generation_gate.json")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one result directory in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def load_run(run_dir):
    result_dir = resolve_result_dir(run_dir)
    gate_path = result_dir / "generation_gate.json"
    descriptor_path = result_dir / "midi_descriptors.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    descriptors = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise ValueError(f"generation gate did not pass: {gate_path}")

    manifest = parse_manifest(result_dir / "inference_manifest.txt")
    training_manifest = parse_manifest(run_dir / "training_manifest.txt")
    seed = (
        manifest.get("inference_seed")
        or manifest.get("seed")
        or training_manifest.get("seed")
    )
    if seed is None:
        raise ValueError(f"seed is missing from manifests under {run_dir}")

    return {
        "run_dir": str(run_dir.resolve()),
        "result_dir": str(result_dir.resolve()),
        "seed": int(seed),
        "gate": gate,
        "descriptors": descriptors,
    }


def discover_runs(root, seeds):
    selected = []
    for seed in seeds:
        errors = []
        for path in sorted(root.glob(f"formal-full-seed{seed}-*"), reverse=True):
            try:
                selected.append(load_run(path.resolve()))
                break
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {exc}")
        else:
            detail = "; ".join(errors[:3]) or "no matching directories"
            raise ValueError(f"no passed complete run found for seed {seed}: {detail}")
    return selected


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--discover-root", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    if args.discover_root:
        if args.run_dirs or not args.seeds:
            parser.error("--discover-root requires --seeds and cannot use run_dirs")
        runs = discover_runs(args.discover_root.resolve(), args.seeds)
    else:
        if not args.run_dirs or args.seeds:
            parser.error("provide run_dirs, or use --discover-root with --seeds")
        runs = [load_run(path.resolve()) for path in args.run_dirs]
    runs = sorted(runs, key=lambda x: x["seed"])
    seeds = [run["seed"] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"duplicate seeds are not allowed: {seeds}")

    gate_rows = []
    descriptor_seed_rows = []
    for run in runs:
        gate = run["gate"]
        gate_rows.append(
            {
                "model": args.model_name,
                "seed": run["seed"],
                "candidate_count": gate["candidate_count"],
                "valid_midi_count": gate["valid_midi_count"],
                "valid_rate": gate["valid_midi_count"] / gate["candidate_count"],
                "median_valid_notes": gate["median_valid_notes"],
                "passed": gate["passed"],
                "run_dir": run["run_dir"],
            }
        )
        for quadrant, group in sorted(run["descriptors"]["aggregate"].items()):
            for metric, values in sorted(group["metrics"].items()):
                descriptor_seed_rows.append(
                    {
                        "model": args.model_name,
                        "seed": run["seed"],
                        "quadrant": quadrant,
                        "metric": metric,
                        "file_count": group["file_count"],
                        "within_seed_mean": values["mean"],
                        "within_seed_std": values["std"],
                        "within_seed_median": values["median"],
                    }
                )

    grouped = {}
    for row in descriptor_seed_rows:
        grouped.setdefault((row["quadrant"], row["metric"]), []).append(row)
    summary_rows = []
    for (quadrant, metric), rows in sorted(grouped.items()):
        means = [float(row["within_seed_mean"]) for row in rows]
        summary_rows.append(
            {
                "model": args.model_name,
                "quadrant": quadrant,
                "metric": metric,
                "seed_count": len(rows),
                "mean_of_seed_means": statistics.fmean(means),
                "std_of_seed_means": sample_std(means),
                "report_value": f"{statistics.fmean(means):.4f} +/- {sample_std(means):.4f}",
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "generation_gates_by_seed.csv",
        list(gate_rows[0]),
        gate_rows,
    )
    write_csv(
        output_dir / "descriptors_by_seed.csv",
        list(descriptor_seed_rows[0]),
        descriptor_seed_rows,
    )
    write_csv(
        output_dir / "descriptor_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )

    report = {
        "model": args.model_name,
        "seeds": seeds,
        "seed_count": len(seeds),
        "all_generation_gates_passed": all(row["passed"] for row in gate_rows),
        "candidate_count": sum(row["candidate_count"] for row in gate_rows),
        "valid_midi_count": sum(row["valid_midi_count"] for row in gate_rows),
        "gate_rows": gate_rows,
        "descriptor_summary": summary_rows,
        "interpretation_boundary": (
            "Values are descriptive mean +/- sample SD across seed-level means. "
            "They do not establish emotion accuracy, musical quality, or superiority."
        ),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    markdown = [
        f"# {args.model_name} generation summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "| Seed | Candidates | Valid MIDI | Valid rate | Median notes | Gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in gate_rows:
        markdown.append(
            f"| {row['seed']} | {row['candidate_count']} | "
            f"{row['valid_midi_count']} | {row['valid_rate']:.3f} | "
            f"{row['median_valid_notes']} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    markdown.extend(
        [
            "",
            "Descriptor values are reported in `descriptor_summary.csv` as the mean "
            "+/- sample SD across seed-level means. MIDI files are not treated as "
            "independent training repeats.",
            "",
            "These engineering and descriptive results do not establish emotion "
            "accuracy, musical quality, or superiority.",
        ]
    )
    (output_dir / "generation_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
