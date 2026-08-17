import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_stage2_phrase_plan_data import prepare_dataset
from stage2_phrase_planning import (
    PHRASE_PLAN_TOKENS,
    append_phrase_vocabulary,
    c2m_track_positions,
    eight_bar_plan,
    fit_plan_thresholds,
    piece_bar_features,
    relative_role,
    strip_plan_tokens,
    transform_piece,
)


def sample_piece(bar_count=4, density_offset=0, octave=4):
    events = [
        {"name": "Emotion", "value": "Q1"},
        {"name": "Key", "value": "C"},
    ]
    for bar in range(bar_count):
        events.extend(
            [
                {"name": "Track", "value": "Chord"},
                {"name": "Bar", "value": None},
                {"name": "Beat", "value": 0},
                {"name": "Chord_Root", "value": "I"},
                {"name": "Chord_Quality", "value": "M"},
                {"name": "Track", "value": "Melody"},
                {"name": "Bar", "value": None},
            ]
        )
        for beat in range(bar + 1 + density_offset):
            events.extend(
                [
                    {"name": "Beat", "value": beat},
                    {"name": "Note_Octave", "value": octave + (bar // 2)},
                    {"name": "Note_Degree", "value": "I"},
                ]
            )
    events.append({"name": "EOS", "value": None})
    chord, melody = c2m_track_positions(events)
    return [chord, melody, events]


class PhrasePlanningTests(unittest.TestCase):
    def test_roles_and_fixed_plan_are_deterministic(self):
        self.assertEqual("PhraseRole_Opening", relative_role(0, 8))
        self.assertEqual("PhraseRole_Development", relative_role(2, 8))
        self.assertEqual("PhraseRole_Climax", relative_role(4, 8))
        self.assertEqual("PhraseRole_Cadence", relative_role(7, 8))
        plan = eight_bar_plan()
        self.assertEqual(8, len(plan))
        self.assertEqual("PhraseDensity_Low", plan[-1][1])

    def test_vocabulary_append_and_strip_preserve_legacy(self):
        old = {"Bar_None": 0, "Track_Melody": 1}
        inverse = {index: token for token, index in old.items()}
        new, new_inverse = append_phrase_vocabulary(old, inverse)
        self.assertEqual(old, {token: new[token] for token in old})
        self.assertEqual(list(PHRASE_PLAN_TOKENS), [new_inverse[i] for i in range(2, 12)])
        self.assertEqual(
            ["Bar_None", "Beat_0"],
            strip_plan_tokens(["Bar_None", "PhraseRole_Opening", "Beat_0"]),
        )

    def test_transform_inserts_three_controls_per_bar_and_recomputes_positions(self):
        piece = sample_piece()
        rows = piece_bar_features(piece)
        thresholds = fit_plan_thresholds(rows + [
            {"onset_count": 8, "mean_register": 72.0, "note_count": 8}
        ])
        converted = transform_piece(piece, thresholds)
        tokens = ["{}_{}".format(event["name"], event["value"]) for event in converted[2]]
        self.assertEqual(12, sum(token in PHRASE_PLAN_TOKENS for token in tokens))
        self.assertEqual(4, len(converted[0]))
        self.assertEqual(4, len(converted[1]))
        first_melody = converted[1][0][0]
        self.assertEqual("PhraseRole_Opening", tokens[first_melody - 3])

    def test_prepare_uses_train_thresholds_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            (source / "events").mkdir(parents=True)
            (source / "data_splits").mkdir()
            tokens = [
                "Bar_None", "Beat_0", "Beat_1", "Beat_2", "Beat_3",
                "Chord_Quality_M", "Chord_Root_I", "Emotion_Q1", "EOS_None",
                "Key_C", "Note_Degree_I", "Note_Octave_4", "Track_Chord",
                "Track_Melody",
            ]
            event2idx = {token: index for index, token in enumerate(tokens)}
            idx2event = {index: token for token, index in event2idx.items()}
            (source / "dictionary.pkl").write_bytes(pickle.dumps((event2idx, idx2event)))
            train_names = []
            for index, offset in enumerate((0, 1, 2)):
                name = "train{}.pkl".format(index)
                train_names.append(name)
                (source / "events" / name).write_bytes(pickle.dumps(sample_piece(density_offset=offset, octave=3 + index)))
            valid_name = "valid.pkl"
            (source / "events" / valid_name).write_bytes(pickle.dumps(sample_piece(density_offset=3, octave=7)))
            (source / "data_splits" / "train.pkl").write_bytes(pickle.dumps(train_names))
            (source / "data_splits" / "valid.pkl").write_bytes(pickle.dumps([valid_name]))

            report = prepare_dataset(source, output)
            self.assertEqual("train_split_only", report["threshold_source"])
            self.assertEqual(4, report["converted_files"])
            self.assertEqual(len(event2idx) + len(PHRASE_PLAN_TOKENS), report["phrase_vocab_size"])
            converted_valid = pickle.loads((output / "events" / valid_name).read_bytes())
            self.assertEqual(4, len(converted_valid[0]))


if __name__ == "__main__":
    unittest.main()

