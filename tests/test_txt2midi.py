import sys
import types
import unittest


if "miditoolkit" not in sys.modules:
    try:
        import miditoolkit  # noqa: F401
    except ModuleNotFoundError:
        miditoolkit_stub = types.ModuleType("miditoolkit")
        midi_stub = types.ModuleType("miditoolkit.midi")
        containers_stub = types.ModuleType("miditoolkit.midi.containers")
        midi_stub.containers = containers_stub
        miditoolkit_stub.midi = midi_stub
        sys.modules["miditoolkit"] = miditoolkit_stub
        sys.modules["miditoolkit.midi"] = midi_stub
        sys.modules["miditoolkit.midi.containers"] = containers_stub

from txt2midi import (
    collapse_exact_onset_pitch_duplicates,
    extract_midi_events_from_generation,
    flatten_midi_bars,
    merge_chord_events,
)


def performance_bar(condition_track, degree):
    return [
        f"Track_{condition_track}",
        "Bar_None",
        "Beat_0",
        "Chord_Root_I",
        "Chord_Quality_M",
        "Track_Performance",
        "Bar_None",
        "Beat_0",
        "Tempo_120",
        "Chord_Root_I",
        "Chord_Quality_M",
        "Note_Octave_4",
        f"Note_Degree_{degree}",
        "Note_Duration_480",
        "Note_Velocity_64",
    ]


class Txt2MidiBarConversionTests(unittest.TestCase):
    @staticmethod
    def note_start_ticks(events):
        bar = -1
        beat = 0
        starts = []
        for event in events:
            if event == "Bar_None":
                bar += 1
            elif event.startswith("Beat_"):
                beat = int(event.rsplit("_", 1)[1])
            elif event.startswith("Note_Pitch_"):
                starts.append(bar * 1920 + beat * 120)
        return starts

    def convert(self, condition_track):
        events = ["Emotion_Q1", "Key_C"]
        events.extend(performance_bar(condition_track, "I"))
        events.extend(performance_bar(condition_track, "III"))
        events = merge_chord_events(events)
        generated_bars = extract_midi_events_from_generation(
            "Key_C",
            events,
            relative_melody=True,
            condition_track=condition_track,
        )
        midi_events = flatten_midi_bars(generated_bars, bars=2)
        return generated_bars, midi_events

    def test_full_conversion_keeps_one_bar_marker_per_generated_bar(self):
        generated_bars, midi_events = self.convert("Melody")

        self.assertEqual(len(generated_bars), 2)
        self.assertEqual(midi_events.count("Bar_None"), 2)
        self.assertEqual(
            self.note_start_ticks(midi_events),
            [0, 1920],
        )

    def test_c2p_conversion_slices_at_chord_markers_without_suffix_duplication(self):
        generated_bars, midi_events = self.convert("Chord")

        self.assertEqual(len(generated_bars), 2)
        self.assertEqual(
            [sum(event.startswith("Note_Duration_") for event in bar)
             for bar in generated_bars],
            [1, 1],
        )
        self.assertEqual(midi_events.count("Bar_None"), 2)
        self.assertEqual(
            self.note_start_ticks(midi_events),
            [0, 1920],
        )

    def test_explicit_duplicate_collapse_keeps_first_note_group_only(self):
        events = [
            "Bar_None",
            "Beat_0",
            "Tempo_100",
            "Note_Pitch_60", "Note_Duration_240", "Note_Velocity_40",
            "Beat_0",
            "Note_Pitch_60", "Note_Duration_480", "Note_Velocity_80",
            "Note_Pitch_64", "Note_Duration_240", "Note_Velocity_50",
            "Beat_4",
            "Note_Pitch_60", "Note_Duration_240", "Note_Velocity_45",
        ]
        transformed, report = collapse_exact_onset_pitch_duplicates(events)
        self.assertEqual(transformed.count("Note_Pitch_60"), 2)
        self.assertIn("Note_Velocity_40", transformed)
        self.assertNotIn("Note_Velocity_80", transformed)
        self.assertIn("Tempo_100", transformed)
        self.assertEqual(report["removed_note_group_count"], 1)
        self.assertEqual(report["removed_token_count"], 3)
        self.assertTrue(report["non_duplicate_tokens_preserved"])

    def test_duplicate_collapse_does_not_cross_bar_boundary(self):
        events = [
            "Bar_None", "Beat_0",
            "Note_Pitch_60", "Note_Duration_240", "Note_Velocity_40",
            "Bar_None", "Beat_0",
            "Note_Pitch_60", "Note_Duration_240", "Note_Velocity_40",
        ]
        transformed, report = collapse_exact_onset_pitch_duplicates(events)
        self.assertEqual(transformed, events)
        self.assertEqual(report["removed_note_group_count"], 0)


if __name__ == "__main__":
    unittest.main()
