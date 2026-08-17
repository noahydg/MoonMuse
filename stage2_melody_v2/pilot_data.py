"""Full-split, label-independent data contract for the V103 motif pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .conditioning import encode_realizer_condition
from .motif_memory import extract_cell_motifs, parse_melody_onsets
from .online_memory import PUBLIC_FIELDS
from .planner import build_eight_bar_plan


MANIFEST_SCHEMA = "moonmuse-stage2-v2-v103-data-manifest-v1"
CONDITION_SCHEMA = "moonmuse-stage2-v2-realizer-condition-v1"
MIN_FIRST8_ONSETS = 16
MIN_CELL_ONSETS = 3
MAX_SEQUENCE_LENGTH = 768
FORBIDDEN_CONDITION_KEYS = frozenset(("relation_ids", "cadence_ids"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_vocabulary(dictionary_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = load_pickle(dictionary_path)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("unexpected V90 dictionary payload")
    event2idx, idx2event = payload
    if len(event2idx) != len(idx2event):
        raise ValueError("V90 dictionary directions differ in size")
    return dict(event2idx), dict(idx2event)


def event_key(event: Mapping[str, Any]) -> str:
    return "{}_{}".format(event["name"], event["value"])


@dataclass(frozen=True)
class V103TokenizedExample:
    dataset: str
    source_id: str
    source_sha256: str
    pad_token: int
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    token_slot_ids: tuple[int, ...]
    condition: Mapping[str, Any]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def target_count(self) -> int:
        return sum(target != self.pad_token for target in self.targets)


def build_tokenized_example(
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
    max_seq_len: int = MAX_SEQUENCE_LENGTH,
) -> V103TokenizedExample:
    if len(payload) != 3:
        raise ValueError("unexpected V90 event payload")
    chord_ranges, melody_ranges, events = payload
    if len(chord_ranges) < bars or len(melody_ranges) < bars:
        raise ValueError("selected source has fewer than eight paired bars")
    end = max(chord_ranges[bars - 1][1], melody_ranges[bars - 1][1])
    if end > max_seq_len:
        raise ValueError("first-eight-bar sequence exceeds V103 maximum")
    try:
        input_ids = [event2idx[event_key(event)] for event in events[:end]]
    except KeyError as exc:
        raise ValueError("event is outside the frozen V90 dictionary: {}".format(exc))
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
    return V103TokenizedExample(
        dataset=dataset,
        source_id=source_id,
        source_sha256=source_sha256,
        pad_token=pad_token,
        input_ids=tuple(input_ids),
        targets=tuple(targets),
        token_slot_ids=tuple(token_slots),
        condition=condition,
    )


def zero_motif_memory(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the same structural condition with all motif content removed."""

    output = copy.deepcopy(dict(payload))
    output["motif_interval_ids"] = [[], [], [], []]
    output["motif_rhythm_ids"] = [[], [], [], []]
    output["motif_lengths"] = [0, 0, 0, 0]
    return output


def shuffled_source_memory(
    payload: Mapping[str, Any], donor: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace the only causally visible motif with another piece's first motif."""

    output = copy.deepcopy(dict(payload))
    donor_condition = donor["condition"] if "condition" in donor else donor
    for field in ("motif_interval_ids", "motif_rhythm_ids"):
        output[field][0] = list(donor_condition[field][0])
    output["motif_lengths"][0] = int(donor_condition["motif_lengths"][0])
    return output


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def canonical_content_fingerprint(events: Iterable[Mapping[str, Any]]) -> str:
    """Hash the musical event content while ignoring split/container metadata."""

    retained = {
        "Key",
        "Bar",
        "Beat",
        "Chord",
        "Chord_Root",
        "Chord_Quality",
        "Note_Octave",
        "Note_Degree",
        "Note_Duration",
        "EOS",
    }
    canonical = [
        [str(event.get("name")), scalar(event.get("value"))]
        for event in events
        if event.get("name") in retained
    ]
    return canonical_sha256(canonical)


def load_split(path: Path) -> list[str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise ValueError("unexpected V90 split payload: {}".format(path))
    names = [str(value) for value in payload]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source ID in split: {}".format(path))
    return names


def label_independent_condition(
    events: Sequence[Mapping[str, Any]], *, emotion_quadrant: str = None
) -> dict[str, Any]:
    """Build the V103 condition while deleting unvalidated semantic fields."""

    _, onsets, bar_count = parse_melody_onsets(events)
    first8 = [onset for onset in onsets if onset.bar < 8]
    motifs = extract_cell_motifs(first8)
    counts = [motif.onset_count for motif in motifs]
    if bar_count < 8:
        raise ValueError("fewer_than_8_melody_bars")
    if len(first8) < MIN_FIRST8_ONSETS:
        raise ValueError("fewer_than_16_first8_onsets")
    if min(counts) < MIN_CELL_ONSETS:
        raise ValueError("cell_with_fewer_than_3_onsets")
    encoded = encode_realizer_condition(
        build_eight_bar_plan(emotion_quadrant), motifs
    ).to_dict()
    condition = {field: encoded[field] for field in PUBLIC_FIELDS}
    condition = json.loads(canonical_json(condition))
    if condition.get("schema") != CONDITION_SCHEMA:
        raise ValueError("unexpected V103 condition schema")
    if FORBIDDEN_CONDITION_KEYS.intersection(condition):
        raise AssertionError("semantic condition field entered V103")
    for target_slot, source_slot in enumerate(condition["source_slot_ids"]):
        if int(source_slot) < -1 or int(source_slot) >= target_slot:
            raise ValueError("motif source must be absent or strictly earlier")
    return condition


def inspect_source(
    *,
    dataset: str,
    split: str,
    source_id: str,
    source_path: Path,
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset": dataset,
        "split": split,
        "source_id": source_id,
        "source_sha256": "",
        "content_sha256": "",
        "eligible": False,
        "exclusion_reason": "",
    }
    if not source_path.is_file():
        row["exclusion_reason"] = "missing_source"
        return row
    row["source_sha256"] = sha256_file(source_path)
    try:
        payload = load_pickle(source_path)
        if not isinstance(payload, (list, tuple)) or len(payload) != 3:
            raise ValueError("unexpected_v90_event_payload")
        events = payload[2]
        if not isinstance(events, list):
            raise ValueError("unexpected_v90_event_list")
        row["content_sha256"] = canonical_content_fingerprint(events)
        condition = label_independent_condition(events)
        example = build_tokenized_example(
            dataset=dataset,
            source_id=source_id,
            source_sha256=row["source_sha256"],
            payload=payload,
            event2idx=event2idx,
            eos_token=eos_token,
            pad_token=pad_token,
            condition=condition,
            max_seq_len=max_sequence_length,
        )
        row.update(
            {
                "eligible": True,
                "sequence_length": example.sequence_length,
                "target_count": example.target_count,
                "condition": condition,
                "condition_sha256": canonical_sha256(condition),
            }
        )
    except Exception as exc:
        row["exclusion_reason"] = "{}: {}".format(type(exc).__name__, exc)
    return row


def build_split_manifest(
    *,
    dataset: str,
    corpus_root: Path,
    split: str,
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> list[dict[str, Any]]:
    if split not in {"train", "valid"}:
        raise ValueError("split must be train or valid")
    names = load_split(corpus_root / "data_splits" / "{}.pkl".format(split))
    return [
        inspect_source(
            dataset=dataset,
            split=split,
            source_id=source_id,
            source_path=corpus_root / "events" / source_id,
            event2idx=event2idx,
            eos_token=eos_token,
            pad_token=pad_token,
            max_sequence_length=max_sequence_length,
        )
        for source_id in sorted(names)
    ]


def eligible_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if bool(row.get("eligible"))]


def manifest_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(canonical_json(row) for row in rows) + "\n"


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    retained = eligible_rows(rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if row.get("eligible"):
            continue
        reason = str(row.get("exclusion_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "listed_count": len(rows),
        "eligible_count": len(retained),
        "eligible_rate": len(retained) / len(rows) if rows else 0.0,
        "minimum_sequence_length": min(
            (int(row["sequence_length"]) for row in retained), default=None
        ),
        "maximum_sequence_length": max(
            (int(row["sequence_length"]) for row in retained), default=None
        ),
        "exclusion_counts": dict(sorted(reasons.items())),
    }


def overlap_report(
    train_rows: Sequence[Mapping[str, Any]],
    valid_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_ids = {str(row["source_id"]) for row in train_rows}
    valid_ids = {str(row["source_id"]) for row in valid_rows}
    train_content = {
        str(row["content_sha256"])
        for row in train_rows
        if row.get("content_sha256")
    }
    valid_content = {
        str(row["content_sha256"])
        for row in valid_rows
        if row.get("content_sha256")
    }
    return {
        "source_id_overlap_count": len(train_ids & valid_ids),
        "source_id_overlap_examples": sorted(train_ids & valid_ids)[:20],
        "content_overlap_count": len(train_content & valid_content),
        "content_overlap_examples": sorted(train_content & valid_content)[:20],
    }


def select_validation_rows(
    rows: Sequence[Mapping[str, Any]], *, dataset: str, maximum: int = 128
) -> list[dict[str, Any]]:
    retained = eligible_rows(rows)
    return sorted(
        retained,
        key=lambda row: hashlib.sha256(
            "v103-validation|{}|{}".format(dataset, row["source_id"]).encode(
                "utf-8"
            )
        ).hexdigest(),
    )[:maximum]


def example_from_manifest_row(
    row: Mapping[str, Any],
    *,
    events_root: Path,
    event2idx: Mapping[str, int],
    eos_token: int,
    pad_token: int,
) -> V103TokenizedExample:
    dataset = str(row["dataset"])
    source_path = (
        events_root
        / "{}_true_melody_v90_functional".format(dataset)
        / "events"
        / str(row["source_id"])
    )
    actual_hash = sha256_file(source_path)
    if actual_hash != row["source_sha256"]:
        raise ValueError("V103 source hash drift: {}".format(source_path))
    condition = dict(row["condition"])
    if FORBIDDEN_CONDITION_KEYS.intersection(condition):
        raise ValueError("semantic field entered V103 manifest")
    return build_tokenized_example(
        dataset=dataset,
        source_id=str(row["source_id"]),
        source_sha256=actual_hash,
        payload=load_pickle(source_path),
        event2idx=event2idx,
        eos_token=eos_token,
        pad_token=pad_token,
        condition=condition,
        max_seq_len=MAX_SEQUENCE_LENGTH,
    )
