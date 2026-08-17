"""Model-facing categorical tensors for the V2 planner/realizer interface.

The module intentionally returns Python tuples rather than Torch tensors so the data
contract can be validated on CPU before any model or vocabulary is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .motif_memory import CellMotif
from .schema import CadenceGoal, MotifRelation, PhrasePlan, PhraseRole


ROLE_TO_ID = {value: index for index, value in enumerate(PhraseRole)}
RELATION_TO_ID = {value: index for index, value in enumerate(MotifRelation)}
CADENCE_TO_ID = {value: index for index, value in enumerate(CadenceGoal)}
INTERVAL_TOKENS = (
    "PAD",
    "S0",
    "U1",
    "U2",
    "U3",
    "U4",
    "D1",
    "D2",
    "D3",
    "D4",
)
INTERVAL_TO_ID = {value: index for index, value in enumerate(INTERVAL_TOKENS)}
RHYTHM_PAD = 0
MAX_RHYTHM_GAP = 32


@dataclass(frozen=True)
class EncodedRealizerCondition:
    role_ids: tuple[int, ...]
    relation_ids: tuple[int, ...]
    cadence_ids: tuple[int, ...]
    source_slot_ids: tuple[int, ...]
    bar_to_slot_ids: tuple[int, ...]
    motif_interval_ids: tuple[tuple[int, ...], ...]
    motif_rhythm_ids: tuple[tuple[int, ...], ...]
    motif_lengths: tuple[int, ...]

    def validate(self) -> None:
        if not all(len(values) == 4 for values in (
            self.role_ids,
            self.relation_ids,
            self.cadence_ids,
            self.source_slot_ids,
            self.motif_interval_ids,
            self.motif_rhythm_ids,
            self.motif_lengths,
        )):
            raise ValueError("V97 realizer conditioning requires four phrase cells")
        if len(self.bar_to_slot_ids) != 8:
            raise ValueError("bar-to-slot mapping must cover eight bars")
        if tuple(self.bar_to_slot_ids) != (0, 0, 1, 1, 2, 2, 3, 3):
            raise ValueError("unexpected V97 bar-to-slot mapping")
        for index, (intervals, rhythms, length) in enumerate(
            zip(self.motif_interval_ids, self.motif_rhythm_ids, self.motif_lengths)
        ):
            if len(intervals) != len(rhythms):
                raise ValueError(f"motif feature length mismatch in cell {index}")
            if length != len(intervals):
                raise ValueError(f"motif length mismatch in cell {index}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "moonmuse-stage2-v2-realizer-condition-v1",
            "role_ids": self.role_ids,
            "relation_ids": self.relation_ids,
            "cadence_ids": self.cadence_ids,
            "source_slot_ids": self.source_slot_ids,
            "bar_to_slot_ids": self.bar_to_slot_ids,
            "motif_interval_ids": self.motif_interval_ids,
            "motif_rhythm_ids": self.motif_rhythm_ids,
            "motif_lengths": self.motif_lengths,
            "id_maps": {
                "roles": {key.value: value for key, value in ROLE_TO_ID.items()},
                "relations": {
                    key.value: value for key, value in RELATION_TO_ID.items()
                },
                "cadences": {
                    key.value: value for key, value in CADENCE_TO_ID.items()
                },
                "intervals": INTERVAL_TO_ID,
            },
        }


def encode_realizer_condition(
    plan: PhrasePlan, motifs: Sequence[CellMotif]
) -> EncodedRealizerCondition:
    plan.validate()
    if len(motifs) != len(plan.slots):
        raise ValueError("one observed motif cell is required per phrase slot")
    if [motif.slot_index for motif in motifs] != list(range(4)):
        raise ValueError("motif cells must use contiguous slot indices")
    interval_ids = []
    rhythm_ids = []
    lengths = []
    for motif in motifs:
        if len(motif.interval_tokens) != len(motif.rhythm_gaps):
            raise ValueError("interval and rhythm transition counts differ")
        interval_ids.append(
            tuple(INTERVAL_TO_ID[token] for token in motif.interval_tokens)
        )
        rhythm_ids.append(
            tuple(min(MAX_RHYTHM_GAP, max(1, gap)) for gap in motif.rhythm_gaps)
        )
        lengths.append(len(motif.interval_tokens))
    condition = EncodedRealizerCondition(
        role_ids=tuple(ROLE_TO_ID[slot.role] for slot in plan.slots),
        relation_ids=tuple(RELATION_TO_ID[slot.relation] for slot in plan.slots),
        cadence_ids=tuple(CADENCE_TO_ID[slot.cadence_goal] for slot in plan.slots),
        source_slot_ids=tuple(
            -1 if slot.source_slot is None else slot.source_slot for slot in plan.slots
        ),
        bar_to_slot_ids=tuple(bar // 2 for bar in range(plan.total_bars)),
        motif_interval_ids=tuple(interval_ids),
        motif_rhythm_ids=tuple(rhythm_ids),
        motif_lengths=tuple(lengths),
    )
    condition.validate()
    return condition
