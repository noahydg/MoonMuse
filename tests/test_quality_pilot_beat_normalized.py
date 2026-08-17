import tempfile
import unittest
from pathlib import Path

import mido

from scripts.evaluate_quality_pilot_beat_normalized import (
    evaluate_rows,
    inspect_candidate,
)


def write_midi(path: Path, tempo: int) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for index in range(32):
        track.append(
            mido.Message("note_on", note=60 + index % 5, velocity=80, time=0)
        )
        track.append(
            mido.Message("note_off", note=60 + index % 5, velocity=0, time=480)
        )
    midi.save(path)


def write_events(path: Path) -> None:
    events = []
    for _ in range(8):
        events.extend(("Track_Melody", "Bar_None", "Track_Performance", "Bar_None"))
    path.write_text("\n".join(events) + "\n", encoding="utf-8")


class BeatNormalizedQualityTests(unittest.TestCase):
    def test_tick_metrics_are_invariant_to_tempo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "sample.txt"
            slow = root / "slow.mid"
            fast = root / "fast.mid"
            write_events(event)
            write_midi(slow, 1_000_000)
            write_midi(fast, 300_000)
            slow_row = inspect_candidate(slow, event, condition_track="Melody")
            fast_row = inspect_candidate(fast, event, condition_track="Melody")
            for key in (
                "note_span_beats",
                "max_internal_gap_beats",
                "own_span_active_ratio",
                "note_count",
            ):
                self.assertEqual(slow_row[key], fast_row[key], key)

    def test_complete_eight_bar_piece_passes_locked_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "sample.txt"
            midi = root / "sample.mid"
            write_events(event)
            write_midi(midi, 500_000)
            rows = [inspect_candidate(midi, event, condition_track="Melody")]
            evaluate_rows(
                rows,
                expected_bars=8,
                min_notes=32,
                min_note_span_beats=28.0,
                max_internal_gap_beats=2.0,
                min_own_span_active_ratio=0.75,
                max_same_pitch_onset_count=1,
            )
            self.assertTrue(rows[0]["quality_pass"])
            self.assertEqual(rows[0]["note_span_beats"], 32.0)


if __name__ == "__main__":
    unittest.main()

