import copy
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ImportError:  # pragma: no cover - exercised by the base non-Torch test env
    torch = None


if torch is not None:
    from stage2_melody_v2.neural import (
        V2CausalTransformer,
        V2NeuralConfig,
        tensorize_realizer_conditions,
    )
    from stage2_melody_v2.neural_data import build_example_from_payload


def fixture_condition():
    return {
        "schema": "moonmuse-stage2-v2-realizer-condition-v1",
        "role_ids": [0, 1, 2, 3],
        "relation_ids": [0, 2, 3, 4],
        "cadence_ids": [0, 1, 0, 2],
        "source_slot_ids": [-1, 0, 0, 0],
        "bar_to_slot_ids": [0, 0, 1, 1, 2, 2, 3, 3],
        "motif_interval_ids": [[2, 3], [4, 5], [6, 7], [8, 9]],
        "motif_rhythm_ids": [[2, 4], [4, 6], [6, 8], [8, 10]],
        "motif_lengths": [2, 2, 2, 2],
    }


@unittest.skipIf(torch is None, "PyTorch is unavailable in the base test environment")
class V2NeuralFixtureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.device = torch.device("cpu")

    def test_relation_and_cadence_ids_are_not_tensorized(self):
        original = fixture_condition()
        mutated = copy.deepcopy(original)
        mutated["relation_ids"] = [4, 4, 4, 4]
        mutated["cadence_ids"] = [2, 2, 2, 2]
        left = tensorize_realizer_conditions([original], device=self.device)
        right = tensorize_realizer_conditions([mutated], device=self.device)
        self.assertNotIn("relation_ids", left)
        self.assertNotIn("cadence_ids", left)
        self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))

    def test_future_target_motif_cannot_change_earlier_slots(self):
        original = fixture_condition()
        mutated = copy.deepcopy(original)
        mutated["motif_interval_ids"][2] = [9, 9]
        mutated["motif_rhythm_ids"][2] = [31, 31]
        model = V2CausalTransformer(
            V2NeuralConfig(vocab_size=20, pad_token=19, d_model=16, n_head=4,
                           n_layer=1, d_ff=32, max_seq_len=32, dropout=0.0)
        ).eval()
        left = tensorize_realizer_conditions([original], device=self.device)
        right = tensorize_realizer_conditions([mutated], device=self.device)
        with torch.no_grad():
            left_slots = model.condition_encoder.encode_slots(left)
            right_slots = model.condition_encoder.encode_slots(right)
        self.assertTrue(torch.equal(left_slots[:, :2], right_slots[:, :2]))

    def test_noncausal_source_pointer_is_rejected(self):
        payload = fixture_condition()
        payload["source_slot_ids"] = [-1, 1, 0, 0]
        condition = tensorize_realizer_conditions([payload], device=self.device)
        model = V2CausalTransformer(
            V2NeuralConfig(vocab_size=20, pad_token=19, d_model=16, n_head=4,
                           n_layer=1, d_ff=32, max_seq_len=32, dropout=0.0)
        )
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            model.condition_encoder.encode_slots(condition)

    def test_future_tokens_cannot_change_earlier_logits(self):
        config = V2NeuralConfig(
            vocab_size=20, pad_token=19, d_model=16, n_head=4,
            n_layer=1, d_ff=32, max_seq_len=32, dropout=0.0
        )
        model = V2CausalTransformer(config).eval()
        condition = tensorize_realizer_conditions(
            [fixture_condition()], device=self.device
        )
        slots = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long)
        left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        right = left.clone()
        right[:, 5:] = torch.tensor([[10, 11, 12]], dtype=torch.long)
        with torch.no_grad():
            left_logits = model(left, slots, condition)
            right_logits = model(right, slots, condition)
        self.assertTrue(torch.allclose(left_logits[:, :5], right_logits[:, :5]))

    def test_forward_backward_and_checkpoint_reload(self):
        config = V2NeuralConfig(
            vocab_size=20,
            pad_token=19,
            d_model=16,
            n_head=4,
            n_layer=1,
            d_ff=32,
            max_seq_len=32,
            dropout=0.0,
        )
        model = V2CausalTransformer(config)
        condition = tensorize_realizer_conditions(
            [fixture_condition(), fixture_condition()], device=self.device
        )
        tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            dtype=torch.long,
        )
        slots = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [[2, 3, 4, 5, 6, 7, 8, 19], [7, 6, 5, 4, 3, 2, 1, 19]],
            dtype=torch.long,
        )
        logits = model(tokens, slots, condition)
        loss = model.compute_loss(logits, targets)
        loss.backward()
        condition_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith("condition_encoder.") and parameter.grad is not None
        )
        backbone_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if not name.startswith("condition_encoder.") and parameter.grad is not None
        )
        self.assertGreater(condition_grad, 0.0)
        self.assertGreater(backbone_grad, 0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "fixture.pt"
            torch.save({"config": config.to_dict(), "state": model.state_dict()}, checkpoint)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            reloaded = V2CausalTransformer(V2NeuralConfig(**payload["config"]))
            reloaded.load_state_dict(payload["state"], strict=True)
            reloaded.eval()
            model.eval()
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(
                        model(tokens, slots, condition),
                        reloaded(tokens, slots, condition),
                    )
                )

    def test_real_stage2_adapter_marks_only_melody_targets(self):
        events = [
            {"name": "Emotion", "value": None},
            {"name": "Key", "value": "C"},
        ]
        chord_ranges = []
        melody_ranges = []
        for _ in range(8):
            start = len(events)
            events.extend(
                [
                    {"name": "Track", "value": "Chord"},
                    {"name": "Bar", "value": None},
                ]
            )
            chord_ranges.append((start, len(events)))
            start = len(events)
            events.extend(
                [
                    {"name": "Track", "value": "Melody"},
                    {"name": "Bar", "value": None},
                    {"name": "Beat", "value": 0},
                    {"name": "Note_Octave", "value": 5},
                    {"name": "Note_Degree", "value": "I"},
                ]
            )
            melody_ranges.append((start, len(events)))
        keys = sorted({f"{event['name']}_{event['value']}" for event in events} | {"EOS_None"})
        event2idx = {key: index for index, key in enumerate(keys)}
        pad_token = len(event2idx)
        example = build_example_from_payload(
            dataset="fixture",
            source_id="fixture.pkl",
            source_sha256="0" * 64,
            payload=(chord_ranges, melody_ranges, events),
            event2idx=event2idx,
            eos_token=event2idx["EOS_None"],
            pad_token=pad_token,
            condition=fixture_condition(),
            max_seq_len=128,
        )
        self.assertEqual(example.sequence_length, len(events))
        self.assertGreater(sum(value != pad_token for value in example.targets), 0)
        self.assertEqual(example.token_slot_ids[melody_ranges[-1][0]], 3)


if __name__ == "__main__":
    unittest.main()
