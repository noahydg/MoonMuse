"""Pure planning helpers for condition-skeleton-locked Stage 3 inference."""

from __future__ import annotations


FORCED = "forced"
SAMPLED = "sampled"

ATTRIBUTE_NAME_BY_PREFIX = {
    "Tempo_": "Tempo",
    "Note_Duration_": "Note_Duration",
    "Note_Velocity_": "Note_Velocity",
}


def discard_incomplete_identity_prefixes(
    condition_events: list[str],
) -> tuple[list[str], list[tuple[int, str, str]]]:
    """Drop condition prefixes that cannot encode a complete musical identity.

    Stage 2 can end at its event budget immediately after ``Note_Octave_*`` or
    can emit a replacement chord root before completing the previous root.
    Those prefixes do not define a note or chord and are already ignored by
    the note-flow validator.  Removing only the incomplete prefix lets the
    strict planner keep validating every complete octave-degree and
    root-quality pair.
    """
    cleaned: list[str] = []
    discarded: list[tuple[int, str, str]] = []
    for index, event in enumerate(condition_events):
        next_event = (
            condition_events[index + 1]
            if index + 1 < len(condition_events)
            else "<EOF>"
        )
        if event.startswith("Note_Octave_") and not next_event.startswith(
            "Note_Degree_"
        ):
            discarded.append((index, event, next_event))
            continue
        if event.startswith("Chord_Root_") and not next_event.startswith(
            "Chord_Quality_"
        ):
            discarded.append((index, event, next_event))
            continue
        cleaned.append(event)
    return cleaned, discarded


def build_performance_bar_plan(condition_events: list[str]) -> list[tuple[str, str]]:
    """Expand a Stage 2 note skeleton into forced and sampled Stage 3 slots.

    Bar, beat, chord, octave, and degree events are copied from the condition.
    Tempo is sampled at every chord-bearing beat, while duration and velocity
    are sampled after each complete octave-degree note identity. This mirrors
    the m2p training representation without allowing pitch/onset drift.
    """
    if not condition_events or condition_events[0] != "Bar_None":
        raise ValueError("condition bar must start with Bar_None")
    plan: list[tuple[str, str]] = []
    index = 0
    previous_beat = -1
    while index < len(condition_events):
        event = condition_events[index]
        if event.startswith("Beat_"):
            beat = int(event.rsplit("_", 1)[1])
            if beat < previous_beat:
                raise ValueError("condition beats must be monotonic within a bar")
            previous_beat = beat
            plan.append((FORCED, event))
            next_event = condition_events[index + 1] if index + 1 < len(condition_events) else ""
            if next_event.startswith("Chord_Root_"):
                plan.append((SAMPLED, "Tempo_"))
            index += 1
            continue
        if event.startswith("Chord_Root_"):
            if index + 1 >= len(condition_events) or not condition_events[
                index + 1
            ].startswith("Chord_Quality_"):
                raise ValueError("Chord_Root must be followed by Chord_Quality")
            plan.append((FORCED, event))
            plan.append((FORCED, condition_events[index + 1]))
            index += 2
            continue
        if event.startswith("Note_Octave_"):
            if index + 1 >= len(condition_events) or not condition_events[
                index + 1
            ].startswith("Note_Degree_"):
                raise ValueError("Note_Octave must be followed by Note_Degree")
            plan.append((FORCED, event))
            plan.append((FORCED, condition_events[index + 1]))
            plan.append((SAMPLED, "Note_Duration_"))
            plan.append((SAMPLED, "Note_Velocity_"))
            index += 2
            continue
        if event.startswith(("Chord_Quality_", "Note_Degree_")):
            raise ValueError(f"orphaned condition token: {event}")
        plan.append((FORCED, event))
        index += 1
    return plan


def allowed_indices(event2idx: dict[str, int], prefix: str) -> list[int]:
    allowed = sorted(index for event, index in event2idx.items() if event.startswith(prefix))
    if not allowed:
        raise ValueError(f"vocabulary has no events with prefix {prefix}")
    return allowed


def bounded_allowed_indices(
    event2idx: dict[str, int], prefix: str, bounds: dict | None
) -> list[int]:
    """Return attribute tokens inside an inclusive numeric range.

    Non-numeric continuation tokens remain admissible because they carry the
    current attribute value rather than selecting a new out-of-range value.
    """
    allowed = allowed_indices(event2idx, prefix)
    if bounds is None:
        return allowed
    if not isinstance(bounds, dict) or set(bounds) < {"min", "max"}:
        raise ValueError(f"invalid bounds for {prefix}: {bounds}")
    lower = float(bounds["min"])
    upper = float(bounds["max"])
    if lower > upper:
        raise ValueError(f"inverted bounds for {prefix}: {lower} > {upper}")
    bounded = []
    numeric_count = 0
    for event, index in event2idx.items():
        if not event.startswith(prefix):
            continue
        suffix = event[len(prefix):]
        try:
            value = float(suffix)
        except ValueError:
            bounded.append(index)
            continue
        if lower <= value <= upper:
            bounded.append(index)
            numeric_count += 1
    if numeric_count == 0:
        raise ValueError(
            f"bounds for {prefix} admit no numeric vocabulary values: {bounds}"
        )
    return sorted(bounded)


def bounds_for_prefix(attribute_bounds: dict | None, prefix: str) -> dict | None:
    if attribute_bounds is None:
        return None
    attribute = ATTRIBUTE_NAME_BY_PREFIX.get(prefix)
    if attribute is None:
        raise ValueError(f"unsupported bounded attribute prefix: {prefix}")
    if attribute not in attribute_bounds:
        raise ValueError(f"missing bounds for {attribute}")
    return attribute_bounds[attribute]


def validate_plan_vocabulary(
    plan: list[tuple[str, str]], event2idx: dict[str, int]
) -> None:
    for kind, value in plan:
        if kind == FORCED:
            if value not in event2idx:
                raise KeyError(f"condition token missing from vocabulary: {value}")
        elif kind == SAMPLED:
            allowed_indices(event2idx, value)
        else:
            raise ValueError(f"unknown plan step kind: {kind}")
