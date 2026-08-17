"""Production data adapter for the V128 hierarchical Stage 2 pilot."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .anchor_conditioning import AnchorPlannerExample, compile_anchor_planner_example
from .anchor_dense_conditioning import (
    DenseAnchorAlignment,
    PredictedAnchorArrays,
    align_dense_tokens_to_anchor_slots,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_token(event: Mapping[str, Any]) -> str:
    value = event.get("value")
    if hasattr(value, "item"):
        value = value.item()
    return f"{event['name']}_{'None' if value is None else value}"


@dataclass(frozen=True)
class AnchorDenseProductionExample:
    dataset: str
    split: str
    source_id: str
    source_sha256: str
    pad_token: int
    tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    segment_ids: tuple[int, ...]
    alignment: DenseAnchorAlignment
    planner_example: AnchorPlannerExample

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def target_count(self) -> int:
        return sum(target != self.pad_token for target in self.target_ids)

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.source_id

    def validate(self, *, pad_token: int) -> None:
        if self.pad_token != pad_token:
            raise ValueError("dense example PAD token differs from the collator")
        length = self.sequence_length
        fields = (self.target_ids, self.segment_ids)
        if not all(len(field) == length for field in fields):
            raise ValueError("dense production fields differ in length")
        self.alignment.validate(length)
        if any(value not in (0, 1) for value in self.segment_ids):
            raise ValueError("dense V1 segment IDs must be binary")
        if tuple(self.segment_ids) != self.alignment.melody_condition_mask:
            raise ValueError("V1 Melody segments and anchor-conditioning mask drifted")
        if any(target < 0 and target != pad_token for target in self.target_ids):
            raise ValueError("dense target contains an invalid sentinel")
        if not any(target != pad_token for target in self.target_ids):
            raise ValueError("dense example contains no Melody targets")
        self.planner_example.validate()


def _track_ranges(tokens: Sequence[str], track_name: str) -> list[tuple[int, int]]:
    starts = [
        index
        for index, token in enumerate(tokens)
        if token == f"Track_{track_name}"
    ]
    all_starts = sorted(
        index for index, token in enumerate(tokens) if token.startswith("Track_")
    )
    output = []
    for start in starts:
        stop = next((value for value in all_starts if value > start), len(tokens))
        output.append((start, stop))
    return output


def build_production_example_from_payload(
    *,
    dataset: str,
    split: str,
    source_id: str,
    source_sha256: str,
    payload: Sequence[Any],
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
    max_seq_len: int = 768,
) -> AnchorDenseProductionExample:
    if len(payload) != 3:
        raise ValueError("unexpected dense C2M pickle payload")
    first_ranges, second_ranges, events = payload
    if len(first_ranges) < 8 or len(second_ranges) < 8:
        raise ValueError("fewer_than_eight_paired_bars")
    end = max(int(first_ranges[7][1]), int(second_ranges[7][1]))
    if end > max_seq_len:
        raise ValueError(f"first_eight_bars_exceed_{max_seq_len}_tokens")
    selected_events = events[:end]
    tokens = tuple(event_token(event) for event in selected_events)
    try:
        input_ids = tuple(event2idx[token] for token in tokens)
    except KeyError as error:
        raise ValueError(f"event outside the frozen V1 vocabulary: {error}") from error

    chord_ranges = _track_ranges(tokens, "Chord")
    melody_ranges = _track_ranges(tokens, "Melody")
    if len(chord_ranges) != 8 or len(melody_ranges) != 8:
        raise ValueError(
            f"expected_eight_chord_and_melody_segments:{len(chord_ranges)}/{len(melody_ranges)}"
        )
    targets = [pad_token] * len(input_ids)
    segments = [0] * len(input_ids)
    next_source_token = event_token(events[end]) if end < len(events) else "EOS_None"
    if next_source_token == "Track_Chord":
        continuation_token = event2idx["Track_Chord"]
    elif next_source_token == "EOS_None":
        continuation_token = eos_token
    else:
        raise ValueError(
            f"unexpected token after eighth Melody segment:{next_source_token}"
        )
    for bar, (start, stop) in enumerate(melody_ranges):
        segments[start:stop] = [1] * (stop - start)
        if bar < 7:
            if stop >= len(input_ids):
                raise ValueError("nonfinal Melody bar has no following Chord token")
            targets[start:stop] = input_ids[start + 1 : stop + 1]
        else:
            targets[start : stop - 1] = input_ids[start + 1 : stop]
            # Match the original V1 next-token task.  A cropped eight-bar
            # window from a longer piece transitions back to Track_Chord;
            # only a source that truly ends here transitions to EOS.
            targets[stop - 1] = continuation_token

    alignment = align_dense_tokens_to_anchor_slots(tokens)
    example = AnchorDenseProductionExample(
        dataset=dataset,
        split=split,
        source_id=source_id,
        source_sha256=source_sha256,
        pad_token=pad_token,
        tokens=tokens,
        input_ids=input_ids,
        target_ids=tuple(targets),
        segment_ids=tuple(segments),
        alignment=alignment,
        planner_example=compile_anchor_planner_example(tokens),
    )
    example.validate(pad_token=pad_token)
    return example


def load_v1_vocabulary(dictionary_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = pickle.loads(dictionary_path.read_bytes())
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("unexpected V1 dictionary payload")
    event2idx, idx2event = dict(payload[0]), dict(payload[1])
    if len(event2idx) != len(idx2event):
        raise ValueError("V1 vocabulary directions differ in size")
    return event2idx, idx2event


def load_production_split(
    *,
    root: Path,
    dataset: str,
    split: str,
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
    max_seq_len: int = 768,
) -> tuple[list[AnchorDenseProductionExample], list[dict[str, str]]]:
    root = root.resolve()
    split_path = root / "data_splits" / f"{split}.pkl"
    names = list(pickle.loads(split_path.read_bytes()))
    examples: list[AnchorDenseProductionExample] = []
    exclusions: list[dict[str, str]] = []
    for name in names:
        path = root / "events" / str(name)
        digest = sha256_file(path)
        try:
            example = build_production_example_from_payload(
                dataset=dataset,
                split=split,
                source_id=str(name),
                source_sha256=digest,
                payload=pickle.loads(path.read_bytes()),
                event2idx=event2idx,
                eos_token=eos_token,
                pad_token=pad_token,
                max_seq_len=max_seq_len,
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            exclusions.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_id": str(name),
                    "source_sha256": digest,
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
            continue
        examples.append(example)
    return examples, exclusions


def tensorize_predicted_anchor_arrays(
    arrays: Sequence[PredictedAnchorArrays], *, device: Any
) -> dict[str, Any]:
    """Tensorize validated predictions without importing Torch at module load."""

    import torch

    if not arrays:
        raise ValueError("at least one predicted anchor array is required")
    for item in arrays:
        item.validate()
    return {
        "active": torch.tensor(
            [item.active_ids for item in arrays], dtype=torch.long, device=device
        ),
        "degree": torch.tensor(
            [item.degree_ids for item in arrays], dtype=torch.long, device=device
        ),
        "octave": torch.tensor(
            [item.octave_ids for item in arrays], dtype=torch.long, device=device
        ),
    }


def collate_dense_production_examples(
    examples: Sequence[AnchorDenseProductionExample],
    *,
    predictions: Mapping[tuple[str, str], PredictedAnchorArrays],
    pad_token: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("at least one dense production example is required")
    for example in examples:
        example.validate(pad_token=pad_token)
        if example.key not in predictions:
            raise ValueError(f"missing predicted anchors for {example.key}")
    maximum = max(example.sequence_length for example in examples)
    shape = (len(examples), maximum)
    inputs = torch.full(shape, pad_token, dtype=torch.long, device=device)
    targets = torch.full(shape, pad_token, dtype=torch.long, device=device)
    segments = torch.zeros(shape, dtype=torch.long, device=device)
    slots = torch.zeros(shape, dtype=torch.long, device=device)
    melody_mask = torch.zeros(shape, dtype=torch.long, device=device)
    attention_mask = torch.zeros(shape, dtype=torch.long, device=device)
    arrays = []
    for row, example in enumerate(examples):
        length = example.sequence_length
        inputs[row, :length] = torch.tensor(
            example.input_ids, dtype=torch.long, device=device
        )
        targets[row, :length] = torch.tensor(
            example.target_ids, dtype=torch.long, device=device
        )
        segments[row, :length] = torch.tensor(
            example.segment_ids, dtype=torch.long, device=device
        )
        slots[row, :length] = torch.tensor(
            example.alignment.anchor_slot_ids, dtype=torch.long, device=device
        )
        melody_mask[row, :length] = torch.tensor(
            example.alignment.melody_condition_mask,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1
        arrays.append(predictions[example.key])
    return {
        "inputs": inputs,
        "targets": targets,
        "segment_ids": segments,
        "token_anchor_slot_ids": slots,
        "melody_condition_mask": melody_mask,
        "attention_mask": attention_mask,
        "anchor_condition": tensorize_predicted_anchor_arrays(arrays, device=device),
    }
