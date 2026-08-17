#!/usr/bin/env python3
"""Render v82 stimuli with one shared gain while preserving relative dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import mido

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_listening_study_audio import (
    analyze_loudness,
    fluidsynth_command,
    probe_audio,
    run_checked,
    sha256_file,
    tool_version,
)


SCHEMA = "moonmuse-v82-global-gain-audio-v1"
EXPECTED_STUDY_SCHEMA = "moonmuse-v82-human-study-prelaunch-v1"
EXPECTED_STIMULI = 48
EXPECTED_SOUNDFONT_SHA256 = (
    "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe"
)
FADE_IN_SECONDS = 0.05
FADE_OUT_SECONDS = 0.15
RELEASE_TAIL_SECONDS = 1.2
TARGET_MEDIAN_LUFS = -20.0
TRUE_PEAK_LIMIT_DBTP = -1.5
INITIAL_PEAK_SAFETY_DB = 0.4


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_clip_seconds(midi_path: Path, tail_seconds: float = RELEASE_TAIL_SECONDS) -> float:
    duration = float(mido.MidiFile(midi_path).length) + tail_seconds
    if duration <= FADE_OUT_SECONDS:
        raise ValueError(f"MIDI duration is too short: {midi_path}")
    return math.ceil(duration * 100.0) / 100.0


def shared_processing_filter(duration_seconds: float) -> str:
    if duration_seconds <= FADE_OUT_SECONDS:
        raise ValueError("duration must exceed fade-out duration")
    fade_out_start = duration_seconds - FADE_OUT_SECONDS
    return (
        "apad,"
        f"atrim=duration={duration_seconds:.3f},"
        f"afade=t=in:st=0:d={FADE_IN_SECONDS:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_SECONDS:.3f}"
    )


def compute_initial_global_gain(
    loudness_values: list[float],
    peak_values: list[float],
    target_median_lufs: float = TARGET_MEDIAN_LUFS,
    peak_limit: float = TRUE_PEAK_LIMIT_DBTP,
    safety_db: float = INITIAL_PEAK_SAFETY_DB,
) -> dict:
    if not loudness_values or len(loudness_values) != len(peak_values):
        raise ValueError("paired non-empty loudness and peak values are required")
    if not all(math.isfinite(value) for value in loudness_values + peak_values):
        raise ValueError("all loudness and peak measurements must be finite")
    median_lufs = statistics.median(loudness_values)
    max_peak = max(peak_values)
    loudness_gain = target_median_lufs - median_lufs
    peak_safe_gain = peak_limit - safety_db - max_peak
    chosen = min(loudness_gain, peak_safe_gain)
    return {
        "raw_median_lufs": median_lufs,
        "raw_max_true_peak_dbtp": max_peak,
        "loudness_target_gain_db": loudness_gain,
        "peak_safe_gain_db": peak_safe_gain,
        "global_gain_db": chosen,
        "peak_constraint_bound_initial_gain": peak_safe_gain < loudness_gain,
    }


def load_stimuli(study_dir: Path) -> list[dict]:
    manifest_path = study_dir / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_STUDY_SCHEMA:
        raise ValueError(f"unexpected study schema: {manifest.get('schema')}")
    if manifest.get("formalStudyData") is not False:
        raise ValueError("renderer expects a prelaunch study population")

    rows = read_csv(study_dir / "master_manifest_private.csv")
    if len(rows) != EXPECTED_STIMULI:
        raise ValueError(
            f"expected {EXPECTED_STIMULI} study stimuli, found {len(rows)}"
        )
    result = []
    for row in rows:
        stimulus_id = row["stimulus_id"]
        midi_path = study_dir / row["study_midi_file"]
        if sha256_file(midi_path) != row["study_midi_sha256"]:
            raise ValueError(f"study MIDI hash mismatch: {midi_path}")
        result.append(
            {
                "stimulus_id": stimulus_id,
                "form_private": row["form"],
                "midi_path": midi_path,
                "midi_sha256": row["study_midi_sha256"],
                "clip_seconds": complete_clip_seconds(midi_path),
            }
        )
    if len({row["stimulus_id"] for row in result}) != EXPECTED_STIMULI:
        raise ValueError("stimulus IDs are not unique")
    return sorted(result, key=lambda row: row["stimulus_id"])


def measurement_filter(duration_seconds: float) -> str:
    return (
        f"{shared_processing_filter(duration_seconds)},"
        f"loudnorm=I={TARGET_MEDIAN_LUFS}:TP={TRUE_PEAK_LIMIT_DBTP}:"
        "LRA=11:print_format=json"
    )


def output_measurement(ffmpeg: str, audio_path: Path) -> tuple[float, float]:
    measured = analyze_loudness(
        ffmpeg,
        audio_path,
        f"loudnorm=I={TARGET_MEDIAN_LUFS}:TP={TRUE_PEAK_LIMIT_DBTP}:"
        "LRA=11:print_format=json",
    )
    return float(measured["input_i"]), float(measured["input_tp"])


def encode_all(
    raw_rows: list[dict],
    destination_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    sample_rate: int,
    global_gain_db: float,
) -> list[dict]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    encoded = []
    for index, row in enumerate(raw_rows, start=1):
        output = destination_dir / f"{row['stimulus_id']}.mp3"
        print(
            f"[encode {index:02d}/{len(raw_rows)}] {row['stimulus_id']} "
            f"gain={global_gain_db:+.3f}dB",
            flush=True,
        )
        run_checked(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(row["raw_wav"]),
                "-af",
                f"{shared_processing_filter(row['clip_seconds'])},"
                f"volume={global_gain_db:.6f}dB",
                "-ar",
                str(sample_rate),
                "-ac",
                "2",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                "-map_metadata",
                "-1",
                str(output),
            ]
        )
        loudness, peak = output_measurement(ffmpeg, output)
        probe = probe_audio(ffprobe, output)
        if probe["codec"] != "mp3" or probe["sample_rate"] != sample_rate or probe["channels"] != 2:
            raise ValueError(f"unexpected audio format: {output}: {probe}")
        if abs(probe["duration_seconds"] - row["clip_seconds"]) > 0.12:
            raise ValueError(f"unexpected audio duration: {output}: {probe}")
        encoded.append(
            {
                **row,
                "audio_path": output,
                "output_integrated_lufs": loudness,
                "output_true_peak_dbtp": peak,
                "output_duration_seconds": probe["duration_seconds"],
            }
        )
    return encoded


def render(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.soundfont) != EXPECTED_SOUNDFONT_SHA256:
        raise ValueError("SoundFont hash does not match the frozen protocol")
    stimuli = load_stimuli(args.study_dir.resolve())
    versions = {
        "fluidsynth": tool_version(args.fluidsynth, ("--version",)),
        "ffmpeg": tool_version(args.ffmpeg, ("-version",)),
        "ffprobe": tool_version(args.ffprobe, ("-version",)),
        "mido": getattr(mido, "__version__", "installed-version-not-exposed"),
    }

    with tempfile.TemporaryDirectory(prefix="moonmuse-v82-render-") as temporary:
        temporary_root = Path(temporary)
        raw_rows = []
        for index, row in enumerate(stimuli, start=1):
            print(f"[render {index:02d}/{len(stimuli)}] {row['stimulus_id']}", flush=True)
            raw_wav = temporary_root / f"{row['stimulus_id']}.wav"
            run_checked(
                fluidsynth_command(
                    args.fluidsynth,
                    args.soundfont,
                    row["midi_path"],
                    raw_wav,
                    args.sample_rate,
                )
            )
            measured = analyze_loudness(
                args.ffmpeg,
                raw_wav,
                measurement_filter(row["clip_seconds"]),
            )
            raw_rows.append(
                {
                    **row,
                    "raw_wav": raw_wav,
                    "raw_integrated_lufs": float(measured["input_i"]),
                    "raw_true_peak_dbtp": float(measured["input_tp"]),
                }
            )

        gain_report = compute_initial_global_gain(
            [row["raw_integrated_lufs"] for row in raw_rows],
            [row["raw_true_peak_dbtp"] for row in raw_rows],
        )
        global_gain = gain_report["global_gain_db"]
        encoded_dir = temporary_root / "encoded"
        attempts = []
        for attempt in range(1, 4):
            if encoded_dir.exists():
                shutil.rmtree(encoded_dir)
            encoded = encode_all(
                raw_rows,
                encoded_dir,
                args.ffmpeg,
                args.ffprobe,
                args.sample_rate,
                global_gain,
            )
            maximum_peak = max(row["output_true_peak_dbtp"] for row in encoded)
            attempts.append(
                {
                    "attempt": attempt,
                    "global_gain_db": global_gain,
                    "maximum_output_true_peak_dbtp": maximum_peak,
                }
            )
            if maximum_peak <= TRUE_PEAK_LIMIT_DBTP:
                break
            global_gain -= maximum_peak - TRUE_PEAK_LIMIT_DBTP + 0.10
        else:
            raise ValueError("global-gain encoding did not satisfy true-peak limit")

        output = args.output_dir
        audio_dir = output / "stimuli_audio"
        audio_dir.mkdir(parents=True)
        verification_rows = []
        public_rows = []
        for row in encoded:
            destination = audio_dir / row["audio_path"].name
            shutil.copyfile(row["audio_path"], destination)
            audio_hash = sha256_file(destination)
            gain_delta = row["output_integrated_lufs"] - row["raw_integrated_lufs"]
            if abs(gain_delta - global_gain) > 0.35:
                raise ValueError(
                    f"relative-loudness preservation failed for {row['stimulus_id']}: "
                    f"measured delta {gain_delta:+.3f}, expected {global_gain:+.3f}"
                )
            verification_rows.append(
                {
                    "stimulus_id": row["stimulus_id"],
                    "form_private": row["form_private"],
                    "midi_sha256": row["midi_sha256"],
                    "clip_seconds": f"{row['clip_seconds']:.2f}",
                    "raw_integrated_lufs": f"{row['raw_integrated_lufs']:.2f}",
                    "raw_true_peak_dbtp": f"{row['raw_true_peak_dbtp']:.2f}",
                    "global_gain_db": f"{global_gain:.6f}",
                    "output_integrated_lufs": f"{row['output_integrated_lufs']:.2f}",
                    "output_true_peak_dbtp": f"{row['output_true_peak_dbtp']:.2f}",
                    "output_duration_seconds": f"{row['output_duration_seconds']:.3f}",
                    "audio_file": f"stimuli_audio/{destination.name}",
                    "audio_sha256": audio_hash,
                    "relative_loudness_delta_error_db": f"{gain_delta - global_gain:+.3f}",
                }
            )
            public_rows.append(
                {
                    "stimulus_id": row["stimulus_id"],
                    "audio_file": f"stimuli_audio/{destination.name}",
                    "audio_sha256": audio_hash,
                }
            )

    write_csv(args.output_dir / "audio_verification_private.csv", verification_rows)
    write_csv(args.output_dir / "audio_manifest_public.csv", public_rows)
    output_loudness = [float(row["output_integrated_lufs"]) for row in verification_rows]
    output_peaks = [float(row["output_true_peak_dbtp"]) for row in verification_rows]
    report = {
        "schema": SCHEMA,
        "status": "prelaunch_audio_ready_ethics_and_questionnaire_pending",
        "formalStudyData": False,
        "study_dir": str(args.study_dir.resolve()),
        "study_manifest_sha256": sha256_file(args.study_dir / "study_manifest.json"),
        "soundfont": str(args.soundfont.resolve()),
        "soundfont_sha256": EXPECTED_SOUNDFONT_SHA256,
        "versions": versions,
        "settings": {
            "stimuli": EXPECTED_STIMULI,
            "sample_rate_hz": args.sample_rate,
            "channels": 2,
            "codec": "mp3",
            "bitrate_kbps": 192,
            "release_tail_seconds": RELEASE_TAIL_SECONDS,
            "fade_in_seconds": FADE_IN_SECONDS,
            "fade_out_seconds": FADE_OUT_SECONDS,
            "per_clip_loudness_normalization": False,
            "target_pooled_median_lufs": TARGET_MEDIAN_LUFS,
            "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
            "global_gain_db": global_gain,
        },
        "gain_selection": gain_report,
        "encoding_attempts": attempts,
        "verification": {
            "output_median_integrated_lufs": statistics.median(output_loudness),
            "output_maximum_true_peak_dbtp": max(output_peaks),
            "all_true_peaks_pass": max(output_peaks) <= TRUE_PEAK_LIMIT_DBTP,
            "all_relative_loudness_offsets_preserved_within_0_35_db": True,
            "all_files_hash_verified": True,
        },
        "remaining_gate": (
            "Build and validate prelaunch questionnaires and analysis; do not "
            "distribute before documented ethics approval or exemption."
        ),
    }
    (args.output_dir / "render_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [
        args.output_dir / "audio_manifest_public.csv",
        args.output_dir / "audio_verification_private.csv",
        args.output_dir / "render_report.json",
        *sorted((args.output_dir / "stimuli_audio").glob("*.mp3")),
    ]
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--fluidsynth", default="/opt/homebrew/bin/fluidsynth")
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/opt/homebrew/bin/ffprobe")
    parser.add_argument("--sample-rate", type=int, default=48000)
    return parser.parse_args()


def main() -> int:
    report = render(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
