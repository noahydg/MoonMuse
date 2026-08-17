"""Model-facing contract for the hierarchical Stage 2 anchor planner.

Stage 1 finishes before Stage 2, so the planner may read the complete eight-bar
chord sequence and requested emotion.  It predicts 32 ordered anchor slots.
Only explicit planner predictions may be serialized for the later dense-note
realizer; the training targets are never exposed through that inference API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .anchor_skeleton import ANCHOR_SLOT_COUNT, AnchorSkeleton, extract_anchor_skeleton
from .hard_return import CHORD_INTERVALS, parse_chord_grids, split_tokens
from .motif_memory import MAJOR_DEGREES


PLANNER_SCHEMA = "moonmuse-stage2-v2-anchor-planner-example-v127-v1"
PLANNER_CONTEXT_SCHEMA = "moonmuse-stage2-v2-anchor-planner-context-v128-v1"
REALIZER_SCHEMA = "moonmuse-stage2-v2-anchor-realizer-condition-v127-v1"

DEGREE_LABELS = tuple(MAJOR_DEGREES)
DEGREE_TO_ID = {label: index for index, label in enumerate(DEGREE_LABELS)}
QUALITY_LABELS = tuple(CHORD_INTERVALS)
QUALITY_TO_ID = {label: index for index, label in enumerate(QUALITY_LABELS)}
EMOTION_LABELS = ("Q1", "Q2", "Q3", "Q4")
EMOTION_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}

MISSING_CLASS_ID = -1
UNKNOWN_CHORD_ID = len(DEGREE_LABELS)
UNKNOWN_QUALITY_ID = len(QUALITY_LABELS)
UNKNOWN_EMOTION_ID = len(EMOTION_LABELS)

PREVIOUS_REST = 0
PREVIOUS_ACTIVE = 1
PREVIOUS_BOS = 2


@dataclass(frozen=True)
class AnchorPlannerContext:
    """Inference-visible inputs; contains no current or future melody labels."""

    schema: str
    key_mode_id: int
    emotion_id: int
    slot_ids: tuple[int, ...]
    bar_ids: tuple[int, ...]
    beat_ids: tuple[int, ...]
    chord_root_ids: tuple[int, ...]
    chord_quality_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.schema != PLANNER_CONTEXT_SCHEMA:
            raise ValueError("unexpected anchor planner context schema")
        fields = (
            self.slot_ids,
            self.bar_ids,
            self.beat_ids,
            self.chord_root_ids,
            self.chord_quality_ids,
        )
        if not all(len(field) == ANCHOR_SLOT_COUNT for field in fields):
            raise ValueError("anchor planner context must contain exactly 32 slots")
        if self.slot_ids != tuple(range(ANCHOR_SLOT_COUNT)):
            raise ValueError("anchor planner context slots must be ordered 0..31")
        if self.bar_ids != tuple(index // 4 for index in range(ANCHOR_SLOT_COUNT)):
            raise ValueError("anchor planner context bar IDs are inconsistent")
        if self.beat_ids != tuple((index % 4) * 4 for index in range(ANCHOR_SLOT_COUNT)):
            raise ValueError("anchor planner context beat IDs are inconsistent")
        if not 0 <= self.key_mode_id < 2:
            raise ValueError("anchor planner context has an invalid key mode")
        if not 0 <= self.emotion_id <= UNKNOWN_EMOTION_ID:
            raise ValueError("anchor planner context has an invalid emotion")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class AnchorPlannerExample:
    schema: str
    key_mode_id: int
    emotion_id: int
    slot_ids: tuple[int, ...]
    bar_ids: tuple[int, ...]
    beat_ids: tuple[int, ...]
    chord_root_ids: tuple[int, ...]
    chord_quality_ids: tuple[int, ...]
    previous_active_ids: tuple[int, ...]
    previous_degree_ids: tuple[int, ...]
    previous_octave_ids: tuple[int, ...]
    target_active_ids: tuple[int, ...]
    target_degree_ids: tuple[int, ...]
    target_octave_ids: tuple[int, ...]
    source_anchor_fingerprint: str

    def validate(self) -> None:
        fields = (
            self.slot_ids,
            self.bar_ids,
            self.beat_ids,
            self.chord_root_ids,
            self.chord_quality_ids,
            self.previous_active_ids,
            self.previous_degree_ids,
            self.previous_octave_ids,
            self.target_active_ids,
            self.target_degree_ids,
            self.target_octave_ids,
        )
        if not all(len(field) == ANCHOR_SLOT_COUNT for field in fields):
            raise ValueError("anchor planner example must contain exactly 32 slots")
        if self.slot_ids != tuple(range(ANCHOR_SLOT_COUNT)):
            raise ValueError("anchor slot IDs must be ordered 0..31")
        if self.previous_active_ids[0] != PREVIOUS_BOS:
            raise ValueError("first autoregressive input must be BOS")
        for index in range(1, ANCHOR_SLOT_COUNT):
            expected = (
                PREVIOUS_ACTIVE
                if self.target_active_ids[index - 1]
                else PREVIOUS_REST
            )
            if self.previous_active_ids[index] != expected:
                raise ValueError("previous-active stream is not causally shifted")
            if self.previous_degree_ids[index] != self.target_degree_ids[index - 1]:
                raise ValueError("previous-degree stream is not causally shifted")
            if self.previous_octave_ids[index] != self.target_octave_ids[index - 1]:
                raise ValueError("previous-octave stream is not causally shifted")
        for active, degree, octave in zip(
            self.target_active_ids,
            self.target_degree_ids,
            self.target_octave_ids,
        ):
            if active not in (0, 1):
                raise ValueError("anchor activity target must be binary")
            if not active and (degree != MISSING_CLASS_ID or octave != MISSING_CLASS_ID):
                raise ValueError("rest targets must not carry pitch labels")
            if active and not (
                0 <= degree < len(DEGREE_LABELS) and octave >= 0
            ):
                raise ValueError("active targets require degree and octave labels")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def planner_context_from_example(example: AnchorPlannerExample) -> AnchorPlannerContext:
    example.validate()
    context = AnchorPlannerContext(
        schema=PLANNER_CONTEXT_SCHEMA,
        key_mode_id=example.key_mode_id,
        emotion_id=example.emotion_id,
        slot_ids=example.slot_ids,
        bar_ids=example.bar_ids,
        beat_ids=example.beat_ids,
        chord_root_ids=example.chord_root_ids,
        chord_quality_ids=example.chord_quality_ids,
    )
    context.validate()
    return context


def compile_anchor_planner_context(tokens: Iterable[str]) -> AnchorPlannerContext:
    """Compile inference inputs from a chord plan without requiring Melody bars."""

    preamble, segments = split_tokens(tokens)
    key_values = [
        token.removeprefix("Key_")
        for token in preamble
        if token.startswith("Key_")
    ]
    if len(key_values) != 1:
        raise ValueError(f"expected one Key token, found {len(key_values)}")
    key = key_values[0]
    emotion = next(
        (
            token.removeprefix("Emotion_")
            for token in preamble
            if token.startswith("Emotion_")
        ),
        None,
    )
    chord_bars = [events for track, events in segments if track == "Chord"]
    if len(chord_bars) < 8:
        raise ValueError("anchor planner context requires eight Stage 1 chord bars")
    chord_grids = parse_chord_grids(chord_bars[:8])
    roots: list[int] = []
    qualities: list[int] = []
    for bar in range(8):
        for beat in (0, 4, 8, 12):
            chord = chord_grids[bar].state_at(beat)
            roots.append(
                DEGREE_TO_ID.get(chord.root, UNKNOWN_CHORD_ID)
                if chord is not None
                else UNKNOWN_CHORD_ID
            )
            qualities.append(
                QUALITY_TO_ID.get(chord.quality, UNKNOWN_QUALITY_ID)
                if chord is not None
                else UNKNOWN_QUALITY_ID
            )
    context = AnchorPlannerContext(
        schema=PLANNER_CONTEXT_SCHEMA,
        key_mode_id=int(key != key.upper()),
        emotion_id=EMOTION_TO_ID.get(emotion, UNKNOWN_EMOTION_ID),
        slot_ids=tuple(range(ANCHOR_SLOT_COUNT)),
        bar_ids=tuple(index // 4 for index in range(ANCHOR_SLOT_COUNT)),
        beat_ids=tuple((index % 4) * 4 for index in range(ANCHOR_SLOT_COUNT)),
        chord_root_ids=tuple(roots),
        chord_quality_ids=tuple(qualities),
    )
    context.validate()
    return context


def planner_example_from_skeleton(skeleton: AnchorSkeleton) -> AnchorPlannerExample:
    if len(skeleton.slots) != ANCHOR_SLOT_COUNT:
        raise ValueError("anchor skeleton must contain exactly 32 slots")
    active = tuple(int(slot.active) for slot in skeleton.slots)
    degrees = tuple(
        DEGREE_TO_ID.get(slot.degree, MISSING_CLASS_ID)
        if slot.active
        else MISSING_CLASS_ID
        for slot in skeleton.slots
    )
    octaves = tuple(
        int(slot.octave) if slot.active and slot.octave is not None else MISSING_CLASS_ID
        for slot in skeleton.slots
    )
    if any(slot.active and degree == MISSING_CLASS_ID for slot, degree in zip(skeleton.slots, degrees)):
        raise ValueError("active anchor degree is outside the fixed functional vocabulary")
    example = AnchorPlannerExample(
        schema=PLANNER_SCHEMA,
        key_mode_id=int(skeleton.key != skeleton.key.upper()),
        emotion_id=EMOTION_TO_ID.get(skeleton.emotion, UNKNOWN_EMOTION_ID),
        slot_ids=tuple(range(ANCHOR_SLOT_COUNT)),
        bar_ids=tuple(slot.bar for slot in skeleton.slots),
        beat_ids=tuple(slot.beat for slot in skeleton.slots),
        chord_root_ids=tuple(
            DEGREE_TO_ID.get(slot.chord_root, UNKNOWN_CHORD_ID)
            for slot in skeleton.slots
        ),
        chord_quality_ids=tuple(
            QUALITY_TO_ID.get(slot.chord_quality, UNKNOWN_QUALITY_ID)
            for slot in skeleton.slots
        ),
        previous_active_ids=(PREVIOUS_BOS,) + tuple(
            PREVIOUS_ACTIVE if value else PREVIOUS_REST for value in active[:-1]
        ),
        previous_degree_ids=(MISSING_CLASS_ID,) + degrees[:-1],
        previous_octave_ids=(MISSING_CLASS_ID,) + octaves[:-1],
        target_active_ids=active,
        target_degree_ids=degrees,
        target_octave_ids=octaves,
        source_anchor_fingerprint=skeleton.fingerprint(),
    )
    example.validate()
    return example


def compile_anchor_planner_example(tokens: Iterable[str]) -> AnchorPlannerExample:
    return planner_example_from_skeleton(extract_anchor_skeleton(tokens))


def compile_realizer_anchor_condition(
    example: AnchorPlannerExample | AnchorPlannerContext,
    *,
    predicted_active_ids: Sequence[int],
    predicted_degree_ids: Sequence[int],
    predicted_octave_ids: Sequence[int],
) -> dict:
    """Serialize explicit planner predictions for the dense-note realizer.

    The function deliberately has no ``use_targets`` switch.  Callers must pass
    predictions, which prevents an inference path from silently substituting
    ground-truth future melody anchors.
    """

    example.validate()
    fields = (predicted_active_ids, predicted_degree_ids, predicted_octave_ids)
    if not all(len(field) == ANCHOR_SLOT_COUNT for field in fields):
        raise ValueError("realizer anchor predictions must contain 32 slots")
    anchors = []
    for index, (active, degree, octave) in enumerate(zip(*fields)):
        active = int(active)
        degree = int(degree)
        octave = int(octave)
        if active not in (0, 1):
            raise ValueError("predicted activity must be binary")
        if active:
            if not 0 <= degree < len(DEGREE_LABELS) or octave < 0:
                raise ValueError("active prediction has invalid pitch labels")
            degree_label = DEGREE_LABELS[degree]
            octave_label: int | None = octave
        else:
            if degree != MISSING_CLASS_ID or octave != MISSING_CLASS_ID:
                raise ValueError("rest prediction must not carry pitch labels")
            degree_label = None
            octave_label = None
        anchors.append(
            {
                "slot": index,
                "bar": example.bar_ids[index],
                "beat": example.beat_ids[index],
                "active": bool(active),
                "degree": degree_label,
                "octave": octave_label,
            }
        )
    return {
        "schema": REALIZER_SCHEMA,
        "inference_source": "planner_predictions",
        "ground_truth_anchor_fields_present": False,
        "relation_labels_present": False,
        "cadence_labels_present": False,
        "anchors": anchors,
    }
