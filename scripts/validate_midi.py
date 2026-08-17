#!/usr/bin/env python3
"""Parse generated MIDI files and fail on empty or unreadable output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import miditoolkit
except ImportError:
    miditoolkit = None
    import mido


def inspect_midi(path: Path) -> dict:
    if miditoolkit is not None:
        midi = miditoolkit.MidiFile(str(path))
        notes = sum(len(instrument.notes) for instrument in midi.instruments)
        instruments = len(midi.instruments)
        tempo_changes = len(midi.tempo_changes)
        max_tick = midi.max_tick
        backend = "miditoolkit"
    else:
        midi = mido.MidiFile(str(path))
        notes = 0
        tempo_changes = 0
        max_tick = 0
        for track in midi.tracks:
            track_tick = 0
            for message in track:
                track_tick += message.time
                if message.type == "note_on" and message.velocity > 0:
                    notes += 1
                elif message.type == "set_tempo":
                    tempo_changes += 1
            max_tick = max(max_tick, track_tick)
        instruments = len(midi.tracks)
        backend = "mido"
    return {
        "file": path.name,
        "backend": backend,
        "instruments": instruments,
        "notes": notes,
        "tempo_changes": tempo_changes,
        "max_tick": max_tick,
        "valid": notes > 0 and max_tick > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.mid"))
    files = [inspect_midi(path) for path in paths]
    report = {
        "input_dir": str(args.input_dir.resolve()),
        "file_count": len(files),
        "total_notes": sum(item["notes"] for item in files),
        "all_valid": bool(files) and all(item["valid"] for item in files),
        "files": files,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
