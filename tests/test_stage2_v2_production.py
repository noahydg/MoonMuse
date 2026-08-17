import copy
import unittest

from stage2_melody_v2.online_memory import (
    build_online_condition_history,
    mask_condition_for_online_inference,
)
from stage2_melody_v2.segment_contract import (
    PRODUCTION_MELODY_SEGMENT_ID,
    PRODUCTION_PREFIX_CHORD_SEGMENT_ID,
    segment_ids_for_tokens,
)


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


class OnlineMotifMemoryTests(unittest.TestCase):
    def test_segment_contract_matches_serialized_v90_and_legacy_inference(self):
        event2idx = {
            "Emotion_Q1": 0,
            "Key_C": 1,
            "Track_Chord": 2,
            "Chord_Root_I": 3,
            "Track_Melody": 4,
            "Note_Octave_4": 5,
        }
        tokens = (0, 1, 2, 3, 4, 5, 2, 3, 4, 5)
        expected = (0, 0, 0, 0, 1, 1, 0, 0, 1, 1)
        self.assertEqual(PRODUCTION_PREFIX_CHORD_SEGMENT_ID, 0)
        self.assertEqual(PRODUCTION_MELODY_SEGMENT_ID, 1)
        self.assertEqual(segment_ids_for_tokens(tokens, event2idx), expected)

    def test_history_exposes_only_completed_cells(self):
        history = build_online_condition_history(fixture_condition())
        self.assertEqual([row["motif_lengths"] for row in history], [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 2, 0, 0],
            [2, 2, 2, 0],
        ])
        self.assertTrue(all("relation_ids" not in row for row in history))
        self.assertTrue(all("cadence_ids" not in row for row in history))

    def test_noncausal_pointer_is_rejected(self):
        payload = copy.deepcopy(fixture_condition())
        payload["source_slot_ids"][1] = 1
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            mask_condition_for_online_inference(payload, target_slot=1)

    def test_motif_lengths_must_match_features(self):
        payload = copy.deepcopy(fixture_condition())
        payload["motif_lengths"][0] = 3
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            mask_condition_for_online_inference(payload, target_slot=1)


try:
    import torch
    import transformers  # noqa: F401

    from stage2_melody.model.music_gpt2 import ImprovedMusicGPT2
    from stage2_melody_v2.neural import tensorize_realizer_conditions
    from stage2_melody_v2.neural_data import Stage2NeuralExample
    from stage2_melody_v2.production import (
        V2ProductionConfig,
        V2ProductionMusicGPT2,
    )
    from stage2_melody_v2.production_data import segment_ids_for_example
except ImportError:  # pragma: no cover - base environment omits optional GPU stack
    torch = None


@unittest.skipIf(torch is None, "PyTorch/Transformers production stack unavailable")
class ProductionBackboneAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(102)
        self.config = V2ProductionConfig(
            vocab_size=20,
            pad_token=19,
            n_layer=2,
            n_head=4,
            d_model=16,
            d_ff=32,
            d_embed=16,
            max_input_len=32,
            backbone_max_positions=64,
            attention_window=16,
            dropout=0.0,
        )

    def test_disabled_adapter_matches_unmodified_backbone(self):
        base = ImprovedMusicGPT2(
            20, 2, 4, 16, 32, 16,
            dropout=0.0, max_seq_len=64, attention_window=16,
            use_rope=True, use_local_attn=True,
        ).eval()
        model = V2ProductionMusicGPT2(self.config).eval()
        loaded = model.load_v92_backbone_state_dict(base.state_dict())
        self.assertGreater(loaded["loaded_backbone_key_count"], 0)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        segments = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.long)
        attention = torch.ones_like(tokens)
        with torch.no_grad():
            expected = base(tokens, seg_inp=segments, attention_mask=attention)
            actual = model(
                tokens, seg_inp=segments, attention_mask=attention,
                enable_conditioning=False,
            )
        self.assertTrue(torch.equal(expected, actual))

    def test_production_collator_uses_the_frozen_segment_contract(self):
        tokens = (0, 1, 2, 3, 4, 5, 2, 3, 4, 5)
        example = Stage2NeuralExample(
            dataset="synthetic",
            source_id="segment-contract",
            source_sha256="0" * 64,
            input_ids=tokens,
            targets=tokens,
            token_slot_ids=(0,) * len(tokens),
            condition={},
        )
        event2idx = {"Track_Chord": 2, "Track_Melody": 4}
        self.assertEqual(
            segment_ids_for_example(example, event2idx),
            (0, 0, 0, 0, 1, 1, 0, 0, 1, 1),
        )

    def test_condition_path_has_gradient_and_strict_reload(self):
        model = V2ProductionMusicGPT2(self.config)
        condition = tensorize_realizer_conditions(
            [fixture_condition()], device=torch.device("cpu")
        )
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        slots = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long)
        segments = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 19]], dtype=torch.long)
        logits = model(
            tokens, token_slot_ids=slots, condition=condition,
            seg_inp=segments, attention_mask=torch.ones_like(tokens),
        )
        loss = model.compute_loss(logits, targets)["total_loss"]
        loss.backward()
        condition_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if (name.startswith("condition_encoder.") or name == "condition_gate")
            and parameter.grad is not None
        )
        self.assertGreater(condition_gradient, 0.0)
        reloaded = V2ProductionMusicGPT2(self.config)
        incompatible = reloaded.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)


if __name__ == "__main__":
    unittest.main()
