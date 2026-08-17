#!/usr/bin/env python3
"""Render anonymized listening-study MIDI to fixed, loudness-normalized MP3."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


LOUDNESS_JSON = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)
FADE_IN_SECONDS = 0.05
FADE_OUT_SECONDS = 0.15


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command, timeout=300):
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout} seconds: {' '.join(command)}"
        ) from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed with return code {completed.returncode}: "
            f"{' '.join(command)}\n{detail}"
        )
    return completed


def tool_version(executable, version_args):
    completed = run_checked([executable, *version_args])
    text = completed.stdout.strip() or completed.stderr.strip()
    return text.splitlines()[0]


def parse_loudness_json(stderr):
    matches = LOUDNESS_JSON.findall(stderr)
    if not matches:
        raise ValueError("ffmpeg loudnorm output did not contain measurement JSON")
    data = json.loads(matches[-1])
    required = {
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "target_offset",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"loudnorm measurement is missing fields: {sorted(missing)}")
    return data


def source_filter(clip_seconds):
    if clip_seconds <= FADE_OUT_SECONDS:
        raise ValueError(
            f"clip duration must exceed the {FADE_OUT_SECONDS:.2f}-second fade-out"
        )
    fade_out_start = clip_seconds - FADE_OUT_SECONDS
    return (
        "silenceremove=start_periods=1:start_silence=0.1:"
        "start_threshold=-60dB,apad,"
        f"atrim=duration={clip_seconds:.3f},"
        f"afade=t=in:st=0:d={FADE_IN_SECONDS:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_SECONDS:.3f}"
    )


def analysis_filter(clip_seconds, target_lufs, true_peak, loudness_range):
    return (
        f"{source_filter(clip_seconds)},"
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}:"
        "print_format=json"
    )


def normalization_filter(
    clip_seconds, target_lufs, true_peak, loudness_range, measured
):
    return (
        f"{source_filter(clip_seconds)},"
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )


def final_gain_db(target_lufs, true_peak, measured_lufs, measured_peak):
    gain = target_lufs - measured_lufs
    if gain > 0:
        gain = min(gain, true_peak - measured_peak)
    return gain


def loudness_acceptance(
    target_lufs,
    true_peak,
    measured_lufs,
    measured_peak,
    loudness_tolerance=0.6,
    peak_limit_tolerance=0.3,
):
    if abs(measured_lufs - target_lufs) <= loudness_tolerance:
        return "within_target_tolerance"
    if (
        measured_lufs < target_lufs - loudness_tolerance
        and measured_peak >= true_peak - peak_limit_tolerance
    ):
        return "quieter_because_true_peak_limited"
    raise ValueError(
        f"loudness {measured_lufs} LUFS and true peak {measured_peak} dBTP "
        "do not satisfy the normalization acceptance rule"
    )


def remaining_gate_for_status(status):
    if status == "participant_audio_ready":
        return (
            "Approve consent/questionnaire and perform a manual audio spot-check "
            "before launching participants."
        )
    return (
        "Researcher diagnostic audio only; do not launch participants or treat review "
        "responses as formal study data."
    )


def fluidsynth_command(executable, soundfont, midi_path, wav_path, sample_rate):
    settings = (
        "synth.device-id=16",
        "synth.polyphony=512",
        "synth.reverb.active=1",
        "synth.reverb.damp=0.3",
        "synth.reverb.level=0.7",
        "synth.reverb.room-size=0.5",
        "synth.reverb.width=0.8",
        "synth.chorus.active=1",
        "synth.chorus.depth=3.6",
        "synth.chorus.level=0.55",
        "synth.chorus.nr=4",
        "synth.chorus.speed=0.36",
    )
    command = [
        executable,
        "-ni",
        "-T",
        "wav",
        "-F",
        str(wav_path),
        "-r",
        str(sample_rate),
        "-g",
        "0.5",
    ]
    for setting in settings:
        command.extend(("-o", setting))
    command.extend((str(soundfont), str(midi_path)))
    return command


def load_expected_stimuli(study_dir):
    manifest_path = study_dir / "master_manifest_private.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("master manifest is empty")
    expected = {}
    for row in rows:
        stimulus_id = row["stimulus_id"]
        if stimulus_id in expected:
            raise ValueError(f"duplicate stimulus ID: {stimulus_id}")
        midi_path = study_dir / "stimuli_midi" / f"{stimulus_id}.mid"
        if not midi_path.is_file():
            raise ValueError(f"missing MIDI stimulus: {midi_path}")
        actual_hash = sha256_file(midi_path)
        if actual_hash != row["midi_sha256"]:
            raise ValueError(
                f"MIDI hash mismatch for {stimulus_id}: "
                f"expected {row['midi_sha256']}, found {actual_hash}"
            )
        expected[stimulus_id] = midi_path
    return expected


def load_clip_seconds_map(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"stimulus_id", "clip_seconds"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "clip-seconds CSV must contain stimulus_id and clip_seconds columns"
        )
    durations = {}
    for row in rows:
        stimulus_id = row["stimulus_id"].strip()
        if not stimulus_id:
            raise ValueError("clip-seconds CSV contains an empty stimulus_id")
        if stimulus_id in durations:
            raise ValueError(f"duplicate clip duration for {stimulus_id}")
        duration = float(row["clip_seconds"])
        if duration <= FADE_OUT_SECONDS:
            raise ValueError(
                f"clip duration for {stimulus_id} must exceed "
                f"{FADE_OUT_SECONDS:.2f} seconds"
            )
        durations[stimulus_id] = duration
    return durations


def analyze_loudness(ffmpeg, input_path, audio_filter):
    completed = run_checked(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(input_path),
            "-af",
            audio_filter,
            "-f",
            "null",
            "-",
        ]
    )
    return parse_loudness_json(completed.stderr)


def probe_audio(ffprobe, audio_path):
    completed = run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    return {
        "codec": stream["codec_name"],
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration_seconds": float(data["format"]["duration"]),
    }


def verify_rendered_audio(
    ffmpeg,
    ffprobe,
    audio_path,
    clip_seconds,
    sample_rate,
    target_lufs,
    true_peak,
    loudness_range,
):
    probe = probe_audio(ffprobe, audio_path)
    if probe["codec"] != "mp3":
        raise ValueError(f"unexpected codec for {audio_path.name}: {probe['codec']}")
    if probe["sample_rate"] != sample_rate or probe["channels"] != 2:
        raise ValueError(f"unexpected audio format for {audio_path.name}: {probe}")
    if abs(probe["duration_seconds"] - clip_seconds) > 0.1:
        raise ValueError(f"unexpected duration for {audio_path.name}: {probe}")
    verification = analyze_loudness(
        ffmpeg,
        audio_path,
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}:"
        "print_format=json",
    )
    measured_lufs = float(verification["input_i"])
    measured_peak = float(verification["input_tp"])
    if measured_peak > true_peak + 0.2:
        raise ValueError(
            f"true-peak verification failed for {audio_path.name}: "
            f"{measured_peak} dBTP"
        )
    acceptance = loudness_acceptance(
        target_lufs,
        true_peak,
        measured_lufs,
        measured_peak,
    )
    return {
        **probe,
        "integrated_lufs": measured_lufs,
        "true_peak_dbtp": measured_peak,
        "loudness_acceptance": acceptance,
        "audio_sha256": sha256_file(audio_path),
    }


def render_study(
    study_dir,
    output_dir,
    soundfont,
    expected_soundfont_sha256,
    fluidsynth,
    ffmpeg,
    ffprobe,
    clip_seconds=20.0,
    sample_rate=48000,
    target_lufs=-16.0,
    true_peak=-1.5,
    loudness_range=11.0,
    resume=False,
    clip_seconds_by_stimulus=None,
    status="participant_audio_ready",
):
    study_dir = study_dir.resolve()
    output_dir = output_dir.resolve()
    soundfont = soundfont.resolve()
    if not soundfont.is_file():
        raise ValueError(f"soundfont does not exist: {soundfont}")
    soundfont_sha256 = sha256_file(soundfont)
    if soundfont_sha256 != expected_soundfont_sha256:
        raise ValueError(
            "soundfont hash mismatch: "
            f"expected {expected_soundfont_sha256}, found {soundfont_sha256}"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "stimuli_audio"
    audio_dir.mkdir(exist_ok=True)

    stimuli = load_expected_stimuli(study_dir)
    if clip_seconds_by_stimulus is not None:
        missing = sorted(set(stimuli) - set(clip_seconds_by_stimulus))
        extra = sorted(set(clip_seconds_by_stimulus) - set(stimuli))
        if missing or extra:
            raise ValueError(
                "clip-seconds stimulus mismatch: "
                f"missing={missing}, extra={extra}"
            )
    versions = {
        "fluidsynth": tool_version(fluidsynth, ("--version",)),
        "ffmpeg": tool_version(ffmpeg, ("-version",)),
        "ffprobe": tool_version(ffprobe, ("-version",)),
    }
    public_rows = []
    rendered_rows = []
    total = len(stimuli)
    for index, (stimulus_id, midi_path) in enumerate(sorted(stimuli.items()), start=1):
        stimulus_clip_seconds = (
            clip_seconds_by_stimulus[stimulus_id]
            if clip_seconds_by_stimulus is not None
            else clip_seconds
        )
        audio_path = audio_dir / f"{stimulus_id}.mp3"
        if resume and audio_path.is_file():
            try:
                verified = verify_rendered_audio(
                    ffmpeg,
                    ffprobe,
                    audio_path,
                    stimulus_clip_seconds,
                    sample_rate,
                    target_lufs,
                    true_peak,
                    loudness_range,
                )
            except (RuntimeError, ValueError):
                verified = None
            if verified is not None:
                print(f"[{index:02d}/{total}] reuse {stimulus_id}", flush=True)
                public_rows.append(
                    {
                        "stimulus_id": stimulus_id,
                        "audio_file": f"stimuli_audio/{audio_path.name}",
                        "audio_sha256": verified["audio_sha256"],
                    }
                )
                rendered_rows.append(
                    {"stimulus_id": stimulus_id, **verified, "reused": True}
                )
                continue

        print(f"[{index:02d}/{total}] render {stimulus_id}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"moonmuse-{stimulus_id}-") as tmp:
            raw_wav = Path(tmp) / "raw.wav"
            run_checked(
                fluidsynth_command(
                    fluidsynth, soundfont, midi_path, raw_wav, sample_rate
                )
            )
            measured = analyze_loudness(
                ffmpeg,
                raw_wav,
                analysis_filter(
                    stimulus_clip_seconds,
                    target_lufs,
                    true_peak,
                    loudness_range,
                ),
            )
            normalized_wav = Path(tmp) / "normalized.wav"
            run_checked(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(raw_wav),
                    "-af",
                    normalization_filter(
                        stimulus_clip_seconds,
                        target_lufs,
                        true_peak,
                        loudness_range,
                        measured,
                    ),
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    "-map_metadata",
                    "-1",
                    str(normalized_wav),
                ]
            )
            normalized_measurement = analyze_loudness(
                ffmpeg,
                normalized_wav,
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}:"
                "print_format=json",
            )
            correction_db = final_gain_db(
                target_lufs,
                true_peak,
                float(normalized_measurement["input_i"]),
                float(normalized_measurement["input_tp"]),
            )
            temp_mp3 = Path(tmp) / "normalized.mp3"
            run_checked(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(normalized_wav),
                    "-af",
                    f"volume={correction_db:.4f}dB",
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
                    str(temp_mp3),
                ]
            )
            shutil.copyfile(temp_mp3, audio_path)

        verified = verify_rendered_audio(
            ffmpeg,
            ffprobe,
            audio_path,
            stimulus_clip_seconds,
            sample_rate,
            target_lufs,
            true_peak,
            loudness_range,
        )
        public_rows.append(
            {
                "stimulus_id": stimulus_id,
                "audio_file": f"stimuli_audio/{audio_path.name}",
                "audio_sha256": verified["audio_sha256"],
            }
        )
        rendered_rows.append(
            {
                "stimulus_id": stimulus_id,
                **verified,
                "reused": False,
            }
        )

    with (output_dir / "audio_manifest_public.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("stimulus_id", "audio_file", "audio_sha256")
        )
        writer.writeheader()
        writer.writerows(public_rows)

    report = {
        "status": status,
        "study_launched": False,
        "stimulus_count": len(rendered_rows),
        "protocol": {
            "soundfont_path": str(soundfont),
            "soundfont_sha256": soundfont_sha256,
            "sample_rate_hz": sample_rate,
            "channels": 2,
            "clip_seconds_mode": (
                "per_stimulus_manifest"
                if clip_seconds_by_stimulus is not None
                else "fixed"
            ),
            "clip_seconds_after_leading_silence_trim": (
                None if clip_seconds_by_stimulus is not None else clip_seconds
            ),
            "clip_seconds_by_stimulus": clip_seconds_by_stimulus or {},
            "leading_silence_threshold_db": -60,
            "boundary_fade_in_seconds": FADE_IN_SECONDS,
            "boundary_fade_out_seconds": FADE_OUT_SECONDS,
            "internal_generated_silence_modified": False,
            "target_integrated_lufs": target_lufs,
            "integrated_loudness_tolerance_lu": 0.6,
            "true_peak_dbtp": true_peak,
            "peak_limited_rule": (
                "A quieter clip is accepted only when its measured true peak is "
                "within 0.3 dB of the cap; this preserves peak safety without clipping."
            ),
            "loudness_range_lu": loudness_range,
            "codec": "MP3 192 kbps CBR",
            "synth_gain": 0.5,
            "effects": "GeneralUser GS recommended FluidSynth reverb/chorus settings",
        },
        "tool_versions": versions,
        "stimuli": rendered_rows,
        "remaining_gate": remaining_gate_for_status(status),
    }
    (output_dir / "audio_rendering_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--soundfont-sha256", required=True)
    parser.add_argument("--fluidsynth", default="fluidsynth")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    parser.add_argument("--clip-seconds-csv", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--study-status",
        choices=("participant_audio_ready", "researcher_audio_ready"),
        default="participant_audio_ready",
    )
    args = parser.parse_args()
    report = render_study(
        args.study_dir,
        args.output_dir,
        args.soundfont,
        args.soundfont_sha256,
        args.fluidsynth,
        args.ffmpeg,
        args.ffprobe,
        clip_seconds=args.clip_seconds,
        resume=args.resume,
        clip_seconds_by_stimulus=(
            load_clip_seconds_map(args.clip_seconds_csv)
            if args.clip_seconds_csv is not None
            else None
        ),
        status=args.study_status,
    )
    print(json.dumps({key: report[key] for key in ("status", "stimulus_count")}, indent=2))


if __name__ == "__main__":
    main()
