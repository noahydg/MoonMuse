"""Real V90 Stage 2 token adapter for the V101 neural integration smoke."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SELECTION_NAMESPACE = "v101-neural-smoke"
FROZEN_SELECTION = {
    "hooktheory": "dPoDJQaBgnM.pkl",
    "emopia": "Q2_2uLw9Zs60A4_0.pkl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def event_key(event: Mapping[str, Any]) -> str:
    return f"{event['name']}_{event['value']}"


@dataclass(frozen=True)
class Stage2NeuralExample:
    dataset: str
    source_id: str
    source_sha256: str
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    token_slot_ids: tuple[int, ...]
    condition: Mapping[str, Any]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def target_count(self) -> int:
        pad_token = max(self.input_ids + self.targets)
        return sum(target != pad_token for target in self.targets)


def load_v98_rows(sidecars_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in sidecars_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("V98 sidecar file is empty")
    return rows


def select_frozen_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for dataset, expected_source in FROZEN_SELECTION.items():
        candidates = [row for row in rows if row["dataset"] == dataset]
        if not candidates:
            raise ValueError(f"V98 has no {dataset} candidates")
        chosen = min(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{SELECTION_NAMESPACE}|{dataset}|{row['source_id']}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        if chosen["source_id"] != expected_source:
            raise ValueError(
                f"frozen V101 selection drift for {dataset}: {chosen['source_id']}"
            )
        selected[dataset] = chosen
    return selected


def load_vocabulary(dictionary_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = load_pickle(dictionary_path)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("unexpected V90 dictionary payload")
    event2idx, idx2event = payload
    if len(event2idx) != len(idx2event):
        raise ValueError("V90 dictionary directions differ in size")
    return dict(event2idx), dict(idx2event)


def build_example_from_payload(
    *,
    dataset: str,
    source_id: str,
    source_sha256: str,
    payload: Sequence[Any],
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
    condition: Mapping[str, Any],
    bars: int = 8,
    max_seq_len: int = 512,
) -> Stage2NeuralExample:
    if len(payload) != 3:
        raise ValueError("unexpected V90 event payload")
    chord_ranges, melody_ranges, events = payload
    if len(chord_ranges) < bars or len(melody_ranges) < bars:
        raise ValueError("selected source has fewer than eight paired bars")
    end = max(chord_ranges[bars - 1][1], melody_ranges[bars - 1][1])
    if end > max_seq_len:
        raise ValueError("selected first-eight-bar sequence exceeds V101 maximum")
    selected_events = events[:end]
    try:
        input_ids = [event2idx[event_key(event)] for event in selected_events]
    except KeyError as exc:
        raise ValueError(f"event is outside the frozen V90 dictionary: {exc}") from exc

    targets = [pad_token] * len(input_ids)
    token_slots = [0] * len(input_ids)
    for bar in range(bars):
        slot = bar // 2
        for start, stop in (chord_ranges[bar], melody_ranges[bar]):
            if not 0 <= start < stop <= len(input_ids):
                raise ValueError("bar range is outside the selected token sequence")
            token_slots[start:stop] = [slot] * (stop - start)

        start, stop = melody_ranges[bar]
        if bar < bars - 1:
            targets[start:stop] = input_ids[start + 1 : stop + 1]
        else:
            targets[start : stop - 1] = input_ids[start + 1 : stop]
            targets[stop - 1] = eos_token

    if not any(target != pad_token for target in targets):
        raise ValueError("selected example has no Stage 2 melody targets")
    if not (len(input_ids) == len(targets) == len(token_slots)):
        raise AssertionError("V101 token fields differ in length")
    return Stage2NeuralExample(
        dataset=dataset,
        source_id=source_id,
        source_sha256=source_sha256,
        input_ids=tuple(input_ids),
        targets=tuple(targets),
        token_slot_ids=tuple(token_slots),
        condition=condition,
    )


def load_frozen_examples(
    *,
    sidecars_path: Path,
    events_root: Path,
    max_seq_len: int = 512,
) -> tuple[list[Stage2NeuralExample], dict[str, Any]]:
    rows = select_frozen_rows(load_v98_rows(sidecars_path))
    dictionary_paths = {
        dataset: events_root
        / f"{dataset}_true_melody_v90_functional"
        / "dictionary.pkl"
        for dataset in rows
    }
    dictionary_hashes = {
        dataset: sha256_file(path) for dataset, path in dictionary_paths.items()
    }
    if len(set(dictionary_hashes.values())) != 1:
        raise ValueError("HookTheory and EMOPIA V90 dictionaries are not byte-identical")
    event2idx, idx2event = load_vocabulary(dictionary_paths["hooktheory"])
    pad_token = len(event2idx)
    eos_token = event2idx["EOS_None"]
    examples = []
    source_paths = {}
    for dataset in ("hooktheory", "emopia"):
        row = rows[dataset]
        source_path = (
            events_root
            / f"{dataset}_true_melody_v90_functional"
            / "events"
            / row["source_id"]
        )
        actual_hash = sha256_file(source_path)
        if actual_hash != row["source_sha256"]:
            raise ValueError(f"V98 source hash mismatch: {source_path}")
        examples.append(
            build_example_from_payload(
                dataset=dataset,
                source_id=row["source_id"],
                source_sha256=actual_hash,
                payload=load_pickle(source_path),
                event2idx=event2idx,
                eos_token=eos_token,
                pad_token=pad_token,
                condition=row["realizer_condition"],
                max_seq_len=max_seq_len,
            )
        )
        source_paths[dataset] = str(source_path.resolve())
    metadata = {
        "dictionary_sha256": next(iter(dictionary_hashes.values())),
        "dictionary_paths": {
            key: str(value.resolve()) for key, value in dictionary_paths.items()
        },
        "vocabulary_without_pad": len(event2idx),
        "vocabulary_with_pad": len(event2idx) + 1,
        "pad_token": pad_token,
        "eos_token": eos_token,
        "source_paths": source_paths,
        "idx2event_size": len(idx2event),
    }
    return examples, metadata


def collate_examples(
    examples: Sequence[Stage2NeuralExample], *, pad_token: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("at least one neural example is required")
    max_length = max(example.sequence_length for example in examples)
    inputs = torch.full(
        (len(examples), max_length), pad_token, dtype=torch.long, device=device
    )
    targets = torch.full_like(inputs, pad_token)
    token_slots = torch.zeros_like(inputs)
    for index, example in enumerate(examples):
        length = example.sequence_length
        inputs[index, :length] = torch.tensor(
            example.input_ids, dtype=torch.long, device=device
        )
        targets[index, :length] = torch.tensor(
            example.targets, dtype=torch.long, device=device
        )
        token_slots[index, :length] = torch.tensor(
            example.token_slot_ids, dtype=torch.long, device=device
        )
    return inputs, targets, token_slots
