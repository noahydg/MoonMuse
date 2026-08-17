#!/usr/bin/env python3
"""Analyze fixed-source Stage 3 high/low arousal counterfactuals."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


PRIMARY_METRICS = {
    "note_density_per_second": "positive",
    "onset_density_per_second": "positive",
    "mean_velocity": "positive",
    "time_weighted_tempo_bpm": "positive",
}

DESCRIPTIVE_METRICS = (
    "duration_seconds",
    "mean_note_duration_seconds",
    "velocity_std",
    "time_weighted_tempo_std",
    "polyphonic_onset_rate",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_csv(path, delimiter=None):
    delimiter = delimiter or ("\t" if path.suffix == ".tsv" else ",")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty output: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_source_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    accepted_schemas = {
        "moonmuse-stage3-arousal-counterfactual-sources-v1",
        "moonmuse-stage2-typicality-selected-sources-v1",
    }
    if data.get("schema") not in accepted_schemas:
        raise ValueError(f"unexpected source manifest schema: {data.get('schema')}")
    sources = data.get("sources", [])
    if len(sources) != data.get("source_count"):
        raise ValueError("source manifest count does not match sources")
    if len({item["frozen_path"] for item in sources}) != len(sources):
        raise ValueError("source manifest contains duplicate frozen paths")
    return data


def target_pair(source_valence):
    if source_valence == "positive":
        return "Q1", "Q4"
    if source_valence == "negative":
        return "Q2", "Q3"
    raise ValueError(f"unsupported source valence: {source_valence}")


def load_descriptor_rows(run_root):
    rows = {}
    descriptor_paths = sorted(run_root.glob("seed*/*/midi_descriptors.csv"))
    if not descriptor_paths:
        raise ValueError(f"no MIDI descriptor CSV files found under {run_root}")
    for path in descriptor_paths:
        for row in load_csv(path):
            filename = row["file"]
            if filename in rows:
                raise ValueError(f"duplicate descriptor filename: {filename}")
            rows[filename] = row
    return rows


def load_quality_rows(run_root):
    rows = {}
    group_reports = []
    for path in sorted(run_root.glob("seed*/*/quality_gate.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        group_reports.append(
            {
                "path": str(path.relative_to(run_root)),
                "passed": bool(data["passed"]),
                "passing_count": int(data["passing_count"]),
                "analysed_count": int(data["analysed_count"]),
                "pass_ratio": float(data["pass_ratio"]),
            }
        )
        for row in data["files"]:
            filename = row["file"]
            if filename in rows:
                raise ValueError(f"duplicate quality filename: {filename}")
            rows[filename] = row
    if not group_reports:
        raise ValueError(f"no quality gate JSON files found under {run_root}")
    return rows, group_reports


def validate_execution(run_root):
    status_rows = load_csv(run_root / "status.tsv")
    critical_fields = ("stage3_rc", "conversion_rc", "validation_rc", "descriptor_rc")
    critical_failures = [
        {"seed": row["seed"], "valence": row["valence"], "field": field}
        for row in status_rows
        for field in critical_fields
        if int(row[field]) != 0
    ]
    execution_rc = int((run_root / "execution.rc").read_text().strip())
    validation_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob("seed*/*/midi_validation.json"))
    ]
    return {
        "group_count": len(status_rows),
        "execution_rc": execution_rc,
        "critical_failures": critical_failures,
        "all_critical_stages_passed": execution_rc == 0 and not critical_failures,
        "validation_report_count": len(validation_reports),
        "valid_midi_count": sum(
            int(report["file_count"])
            for report in validation_reports
            if report.get("all_valid")
        ),
        "all_midi_valid": bool(validation_reports)
        and all(report.get("all_valid") for report in validation_reports),
    }


def pair_rows(source_manifest, descriptors, quality):
    rows = []
    quality_rows = []
    metrics = tuple(PRIMARY_METRICS) + DESCRIPTIVE_METRICS
    for pair_index, source in enumerate(source_manifest["sources"], start=1):
        high_quadrant, low_quadrant = target_pair(source["source_valence"])
        stem = Path(source["frozen_path"]).stem
        high_filename = f"{stem}_{high_quadrant}_performance.mid"
        low_filename = f"{stem}_{low_quadrant}_performance.mid"
        try:
            high = descriptors[high_filename]
            low = descriptors[low_filename]
            high_quality = quality[high_filename]
            low_quality = quality[low_filename]
        except KeyError as exc:
            raise ValueError(f"missing generated pair member: {exc.args[0]}") from exc

        common = {
            "pair_id": f"AR{pair_index:02d}",
            "seed": int(source["seed"]),
            "source_quadrant": source["source_quadrant"],
            "source_arousal": source["source_arousal"],
            "source_valence": source["source_valence"],
            "source_sha256": source["source_sha256"],
            "high_target_quadrant": high_quadrant,
            "low_target_quadrant": low_quadrant,
            "high_file": high_filename,
            "low_file": low_filename,
        }
        for metric in metrics:
            high_value = float(high[metric])
            low_value = float(low[metric])
            difference = high_value - low_value
            expected = PRIMARY_METRICS.get(metric, "descriptive")
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "high_value": high_value,
                    "low_value": low_value,
                    "high_minus_low": difference,
                    "role": "primary_arousal_proxy"
                    if metric in PRIMARY_METRICS
                    else "descriptive_only",
                    "expected_direction": expected,
                    "expected_direction_met": (
                        difference > 0 if expected == "positive" else ""
                    ),
                }
            )
        quality_rows.extend(
            (
                {
                    **common,
                    "target_arousal": "high",
                    "target_quadrant": high_quadrant,
                    "file": high_filename,
                    "quality_pass": bool(high_quality["quality_pass"]),
                    "quality_score": float(high_quality["quality_score"]),
                    "note_span_seconds": float(high_quality["note_span_seconds"]),
                    "active_ratio": float(high_quality["active_ratio"]),
                    "max_internal_gap_seconds": float(
                        high_quality["max_internal_gap_seconds"]
                    ),
                    "max_same_pitch_onset_count": int(
                        high_quality["max_same_pitch_onset_count"]
                    ),
                },
                {
                    **common,
                    "target_arousal": "low",
                    "target_quadrant": low_quadrant,
                    "file": low_filename,
                    "quality_pass": bool(low_quality["quality_pass"]),
                    "quality_score": float(low_quality["quality_score"]),
                    "note_span_seconds": float(low_quality["note_span_seconds"]),
                    "active_ratio": float(low_quality["active_ratio"]),
                    "max_internal_gap_seconds": float(
                        low_quality["max_internal_gap_seconds"]
                    ),
                    "max_same_pitch_onset_count": int(
                        low_quality["max_same_pitch_onset_count"]
                    ),
                },
            )
        )
    return rows, quality_rows


def summarize_effects(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["metric"], []).append(row)
    summaries = []
    for metric, group in grouped.items():
        differences = [float(row["high_minus_low"]) for row in group]
        expected = PRIMARY_METRICS.get(metric, "descriptive")
        positive_count = sum(value > 0 for value in differences)
        negative_count = sum(value < 0 for value in differences)
        tie_count = len(differences) - positive_count - negative_count
        expected_win_count = positive_count if expected == "positive" else None
        expected_win_rate = (
            expected_win_count / len(differences)
            if expected_win_count is not None
            else None
        )
        if expected_win_rate is None:
            support = "descriptive_only"
        elif statistics.fmean(differences) > 0 and expected_win_rate >= 0.75:
            support = "directionally_consistent"
        elif statistics.fmean(differences) > 0 and expected_win_rate >= 0.5:
            support = "mixed_positive"
        else:
            support = "not_supported"
        summaries.append(
            {
                "metric": metric,
                "role": "primary_arousal_proxy"
                if metric in PRIMARY_METRICS
                else "descriptive_only",
                "expected_direction": expected,
                "pair_count": len(differences),
                "mean_high_minus_low": statistics.fmean(differences),
                "median_high_minus_low": statistics.median(differences),
                "std_high_minus_low": sample_std(differences),
                "positive_pair_count": positive_count,
                "negative_pair_count": negative_count,
                "tie_pair_count": tie_count,
                "expected_win_count": expected_win_count,
                "expected_win_rate": expected_win_rate,
                "directional_support": support,
            }
        )
    return sorted(summaries, key=lambda row: row["metric"])


def quality_summary(quality_rows, group_reports):
    return {
        "individual_file_count": len(quality_rows),
        "individual_quality_pass_count": sum(
            row["quality_pass"] for row in quality_rows
        ),
        "individual_quality_pass_rate": (
            sum(row["quality_pass"] for row in quality_rows) / len(quality_rows)
        ),
        "group_gate_count": len(group_reports),
        "group_gate_pass_count": sum(report["passed"] for report in group_reports),
        "group_reports": group_reports,
        "interpretation": (
            "The strict pilot gate is diagnostic. Failure does not invalidate MIDI "
            "or the counterfactual pairing, but it blocks any claim that the batch "
            "already meets the listening-quality target."
        ),
    }


def write_markdown(path, report):
    lines = [
        "# Stage 3 fixed-source arousal counterfactual",
        "",
        "## Execution evidence",
        "",
        f"- Critical generation stages passed: "
        f"{report['execution']['all_critical_stages_passed']}",
        f"- Valid MIDI: {report['execution']['valid_midi_count']}/"
        f"{report['quality']['individual_file_count']}",
        f"- Strict individual quality passes: "
        f"{report['quality']['individual_quality_pass_count']}/"
        f"{report['quality']['individual_file_count']}",
        f"- Strict group quality gates passed: "
        f"{report['quality']['group_gate_pass_count']}/"
        f"{report['quality']['group_gate_count']}",
        "",
        "## High-minus-low arousal proxies",
        "",
        "| Metric | Mean difference | Median | Expected wins | Directional reading |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["effect_summary"]:
        if row["role"] != "primary_arousal_proxy":
            continue
        lines.append(
            f"| {row['metric']} | {row['mean_high_minus_low']:.4f} | "
            f"{row['median_high_minus_low']:.4f} | "
            f"{row['expected_win_count']}/{row['pair_count']} | "
            f"{row['directional_support']} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            report["interpretation_boundary"],
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_root, source_manifest_path, output_dir, run_archive=None):
    run_root = run_root.resolve()
    source_manifest_path = source_manifest_path.resolve()
    output_dir = output_dir.resolve()
    source_manifest = load_source_manifest(source_manifest_path)
    descriptors = load_descriptor_rows(run_root)
    quality, group_reports = load_quality_rows(run_root)
    effects, quality_rows = pair_rows(source_manifest, descriptors, quality)
    summaries = summarize_effects(effects)
    report = {
        "schema": "moonmuse-stage3-arousal-counterfactual-analysis-v1",
        "run_root": str(run_root),
        "run_archive_sha256": (
            sha256_file(run_archive.resolve()) if run_archive is not None else None
        ),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "pair_count": len(source_manifest["sources"]),
        "expected_output_count": source_manifest["expected_output_count"],
        "execution": validate_execution(run_root),
        "quality": quality_summary(quality_rows, group_reports),
        "effect_summary": summaries,
        "interpretation_boundary": (
            "This fixed-source comparison isolates the Stage 3 condition at the "
            "implemented sampling interface. Descriptor differences are objective "
            "performance proxies, not evidence that listeners perceive the intended "
            "arousal, that the music is therapeutically effective, or that the model "
            "is superior. The 12-pair result remains a diagnostic until blinded "
            "listening confirms the direction and quality is acceptable."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_arousal_effects.csv", effects)
    write_csv(output_dir / "quality_file_summary.csv", quality_rows)
    write_csv(output_dir / "effect_summary.csv", summaries)
    (output_dir / "objective_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "objective_summary.md", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-archive", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.run_root,
        args.source_manifest,
        args.output_dir,
        run_archive=args.run_archive,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
