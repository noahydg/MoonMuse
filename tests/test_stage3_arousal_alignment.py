import pickle
import unittest
from pathlib import Path

import yaml

from stage3_arousal_alignment import build_alignment_spec

try:
    import torch
except ImportError:  # Local evidence environment intentionally has no Torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]


class Stage3ArousalAlignmentTests(unittest.TestCase):
    def test_alignment_spec_uses_only_numeric_attribute_tokens(self):
        event2idx = {
            "Beat_0": 0,
            "Beat_1": 1,
            "Note_Duration_120": 2,
            "Note_Duration_240": 3,
            "Note_Octave_4": 4,
            "Note_Octave_5": 5,
            "Note_Velocity_40": 6,
            "Note_Velocity_80": 7,
            "Tempo_90": 8,
            "Tempo_120": 9,
            "Tempo_Conti": 10,
            "Arousal_High": 11,
            "Arousal_Low": 12,
        }
        spec = build_alignment_spec(event2idx, {})
        attributes = {item["name"]: item for item in spec["attributes"]}
        self.assertEqual([8, 9], attributes["tempo"]["token_indices"])
        self.assertEqual([0.0, 1.0], attributes["tempo"]["normalized_values"])
        self.assertEqual(1, attributes["velocity"]["direction"])
        self.assertEqual(-1, attributes["duration"]["direction"])
        self.assertEqual("probability_mass", attributes["note_event_mass"]["kind"])
        self.assertEqual(11, spec["high_token_index"])
        self.assertEqual(12, spec["low_token_index"])

    def test_real_dictionary_and_alignment_config_build_complete_spec(self):
        config_path = ROOT / "stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml"
        config = yaml.safe_load(config_path.read_text("utf-8"))
        dictionary_path = ROOT / "events/m2p/emopia_m2p_factorized_va_functional/dictionary.pkl"
        if not dictionary_path.is_file():
            self.skipTest("external event dictionary is not included in the source release")
        with open(dictionary_path, "rb") as handle:
            event2idx = pickle.load(handle)[0]
        spec = build_alignment_spec(event2idx, config["arousal_alignment"])
        names = {item["name"] for item in spec["attributes"]}
        self.assertEqual(
            {"velocity", "tempo", "duration", "note_event_mass", "onset_event_mass"},
            names,
        )
        tempo = next(item for item in spec["attributes"] if item["name"] == "tempo")
        self.assertNotIn("Tempo_Conti", tempo["token_names"])
        self.assertGreater(len(tempo["token_indices"]), 10)
        self.assertEqual(0.20, config["arousal_alignment"]["weight"])

    def test_invalid_or_empty_alignment_configuration_fails(self):
        with self.assertRaises(ValueError):
            build_alignment_spec({"Arousal_High": 1, "Arousal_Low": 2}, {})
        event2idx = {
            "Arousal_High": 1,
            "Arousal_Low": 2,
            "Note_Velocity_40": 3,
            "Note_Velocity_80": 4,
            "Tempo_90": 5,
            "Tempo_120": 6,
            "Note_Duration_120": 7,
            "Note_Duration_240": 8,
            "Note_Octave_5": 9,
            "Beat_0": 10,
        }
        config = {"attributes": {name: {"enabled": False} for name in (
            "velocity", "tempo", "duration", "note_event_mass", "onset_event_mass"
        )}}
        with self.assertRaises(ValueError):
            build_alignment_spec(event2idx, config)

    @unittest.skipIf(torch is None, "Torch tensor test runs in the AutoDL environment")
    def test_counterfactual_loss_rewards_the_declared_directions(self):
        from stage3_arousal_alignment import compute_counterfactual_alignment_loss

        event2idx = {
            "Note_Velocity_40": 0,
            "Note_Velocity_80": 1,
            "Tempo_90": 2,
            "Tempo_120": 3,
            "Note_Duration_120": 4,
            "Note_Duration_240": 5,
            "Arousal_High": 6,
            "Arousal_Low": 7,
        }
        config = {"attributes": {
            "note_event_mass": {"enabled": False},
            "onset_event_mass": {"enabled": False},
        }}
        spec = build_alignment_spec(event2idx, config)
        pad = 8
        dec_input = torch.tensor([
            [6, 0, 2, 4],
            [7, 0, 2, 4],
        ])
        dec_target = torch.tensor([
            [0, 2, 4, pad],
            [0, 2, 4, pad],
        ])
        shape = (2, 4, 9)
        real = torch.zeros(shape)
        counter = torch.zeros(shape)

        # Sample 0 is really High; sample 1 is really Low.  High prefers the
        # larger velocity/tempo and shorter duration under both pair orders.
        high_logits = torch.zeros((4, 9))
        low_logits = torch.zeros((4, 9))
        high_logits[0, 1] = 8
        high_logits[1, 3] = 8
        high_logits[2, 4] = 8
        low_logits[0, 0] = 8
        low_logits[1, 2] = 8
        low_logits[2, 5] = 8
        real[0], counter[0] = high_logits, low_logits
        real[1], counter[1] = low_logits, high_logits

        aligned = compute_counterfactual_alignment_loss(
            real, counter, dec_input, dec_target, pad, spec
        )["alignment_loss"]
        reversed_loss = compute_counterfactual_alignment_loss(
            counter, real, dec_input, dec_target, pad, spec
        )["alignment_loss"]
        self.assertLess(aligned.item(), 1e-4)
        self.assertGreater(reversed_loss.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
