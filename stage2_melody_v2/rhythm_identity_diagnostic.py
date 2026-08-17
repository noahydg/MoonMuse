"""Read-only rhythm-identity evidence beside, never inside, Stage 2 controls.

The V115 interface deliberately serializes several frozen observations without
choosing one as a controller signal.  It has no threshold, predicted relation,
loss, or write target, and it never changes the D-071 five-axis control payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .factorized_control import FactorizedControlPlan


DIAGNOSTIC_SCHEMA = (
    "moonmuse-stage2-v2-rhythm-identity-diagnostic-interface-v115-v1"
)
PAIR_SCHEMA = f"{DIAGNOSTIC_SCHEMA}-pair"
PLANNED_IDENTITY_RELATIONS = frozenset(
    {"SAME_IDENTITY", "DIFFERENT_IDENTITY"}
)
MODEL_NAMES = ("length", "unordered", "full")
CONTROL_AXES = ("z_id", "o_t", "d_pitch", "d_rhythm", "e_t")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def normalize_rhythm_signature(gaps: Sequence[int]) -> tuple[int, ...]:
    """Return a positive GCD-normalized gap signature."""

    if len(gaps) < 2:
        raise ValueError("rhythm diagnostic requires at least two gaps")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in gaps):
        raise ValueError("rhythm gaps must be positive integers")
    divisor = math.gcd(*gaps)
    signature = tuple(value // divisor for value in gaps)
    if math.gcd(*signature) != 1:
        raise AssertionError("rhythm signature did not reach canonical GCD form")
    return signature


def _validate_probability(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be finite and inside [0, 1]")


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class RhythmIdentityScores:
    exact_signature_equal: bool
    length_probability: float
    unordered_probability: float
    full_probability: float

    def validate(self) -> None:
        if not isinstance(self.exact_signature_equal, bool):
            raise ValueError("exact_signature_equal must be Boolean")
        for label, value in (
            ("length_probability", self.length_probability),
            ("unordered_probability", self.unordered_probability),
            ("full_probability", self.full_probability),
        ):
            _validate_probability(value, label)

    @property
    def full_minus_unordered(self) -> float:
        return float(self.full_probability) - float(self.unordered_probability)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "exact_signature_equal": self.exact_signature_equal,
            "length_probability": float(self.length_probability),
            "unordered_probability": float(self.unordered_probability),
            "full_probability": float(self.full_probability),
            "full_minus_unordered": self.full_minus_unordered,
        }


@dataclass(frozen=True)
class RhythmIdentityPairDiagnostic:
    target_slot: int
    comparison_source_slot: int
    identity_source_slot: Optional[int]
    planned_identity_relation: str
    source_signature: tuple[int, ...]
    target_signature: tuple[int, ...]
    scores: RhythmIdentityScores
    schema: str = PAIR_SCHEMA

    def validate(self) -> None:
        if self.schema != PAIR_SCHEMA:
            raise ValueError("unexpected rhythm-identity pair schema")
        if self.target_slot not in (1, 2, 3):
            raise ValueError("diagnostic target slot must be 1, 2, or 3")
        if not 0 <= self.comparison_source_slot < self.target_slot:
            raise ValueError("comparison source must be an earlier slot")
        if self.identity_source_slot is not None and not (
            0 <= self.identity_source_slot < self.target_slot
        ):
            raise ValueError("identity source must be an earlier slot")
        if self.planned_identity_relation not in PLANNED_IDENTITY_RELATIONS:
            raise ValueError("unexpected planned identity relation")
        expected_relation = (
            "SAME_IDENTITY"
            if self.identity_source_slot == self.comparison_source_slot
            else "DIFFERENT_IDENTITY"
        )
        if self.planned_identity_relation != expected_relation:
            raise ValueError("planned identity relation differs from frozen pointers")
        if normalize_rhythm_signature(self.source_signature) != self.source_signature:
            raise ValueError("source signature is not in canonical GCD form")
        if normalize_rhythm_signature(self.target_signature) != self.target_signature:
            raise ValueError("target signature is not in canonical GCD form")
        if self.scores.exact_signature_equal != (
            self.source_signature == self.target_signature
        ):
            raise ValueError("exact-signature flag differs from serialized signatures")
        self.scores.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "target_slot": self.target_slot,
            "comparison_source_slot": self.comparison_source_slot,
            "identity_source_slot": self.identity_source_slot,
            "planned_identity_relation": self.planned_identity_relation,
            "planned_relation_is_not_ground_truth": True,
            "source_signature": self.source_signature,
            "target_signature": self.target_signature,
            "observations": self.scores.to_dict(),
        }


@dataclass(frozen=True)
class RhythmIdentityDiagnosticInterface:
    dataset: str
    source_id: str
    source_sha256: str
    factorized_control_sha256: str
    model_sha256: Mapping[str, str]
    comparisons: tuple[RhythmIdentityPairDiagnostic, ...]
    schema: str = DIAGNOSTIC_SCHEMA
    diagnostic_only: bool = True
    controller_consumable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema != DIAGNOSTIC_SCHEMA:
            raise ValueError("unexpected rhythm-identity diagnostic schema")
        if not self.dataset or not self.source_id:
            raise ValueError("dataset and source ID must be non-empty")
        _validate_sha256(self.source_sha256, "source_sha256")
        _validate_sha256(
            self.factorized_control_sha256, "factorized_control_sha256"
        )
        if tuple(self.model_sha256) != MODEL_NAMES:
            raise ValueError("diagnostic model lineage must retain length/unordered/full")
        for name in MODEL_NAMES:
            _validate_sha256(str(self.model_sha256[name]), f"{name} model hash")
        if self.diagnostic_only is not True or self.controller_consumable is not False:
            raise ValueError("rhythm identity interface must remain diagnostic-only")
        if tuple(item.target_slot for item in self.comparisons) != (1, 2, 3):
            raise ValueError("interface requires one ordered comparison for slots 1/2/3")
        for item in self.comparisons:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "dataset": self.dataset,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "factorized_control_sha256": self.factorized_control_sha256,
            "diagnostic_only": self.diagnostic_only,
            "controller_consumable": self.controller_consumable,
            "model_lineage": {
                f"{name}_model_sha256": self.model_sha256[name]
                for name in MODEL_NAMES
            },
            "comparisons": [item.to_dict() for item in self.comparisons],
            "axis_separation": {
                "control_axes": CONTROL_AXES,
                "read_only_observation": True,
                "excluded_from_encoded_control": True,
                "planner_action": "NONE",
            },
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_rhythm_identity_diagnostic_interface(
    *,
    dataset: str,
    source_id: str,
    source_sha256: str,
    factorized_control_sha256: str,
    control: FactorizedControlPlan,
    observed_rhythm_gaps: Sequence[Sequence[int]],
    scores_by_target: Mapping[int, RhythmIdentityScores],
    model_sha256: Mapping[str, str],
) -> RhythmIdentityDiagnosticInterface:
    """Build a separate read-only interface from frozen observations and scores."""

    control.validate()
    if len(observed_rhythm_gaps) != len(control.slots):
        raise ValueError("one observed rhythm sequence is required per control slot")
    signatures = tuple(
        normalize_rhythm_signature(tuple(gaps)) for gaps in observed_rhythm_gaps
    )
    comparisons: list[RhythmIdentityPairDiagnostic] = []
    for target_slot in (1, 2, 3):
        slot = control.slots[target_slot]
        source_slot = slot.comparison_source_slot
        if source_slot is None:
            raise ValueError("target diagnostic slot lacks a comparison source")
        if target_slot not in scores_by_target:
            raise ValueError(f"missing diagnostic scores for target slot {target_slot}")
        relation = (
            "SAME_IDENTITY"
            if slot.identity_source_slot == source_slot
            else "DIFFERENT_IDENTITY"
        )
        comparisons.append(
            RhythmIdentityPairDiagnostic(
                target_slot=target_slot,
                comparison_source_slot=source_slot,
                identity_source_slot=slot.identity_source_slot,
                planned_identity_relation=relation,
                source_signature=signatures[source_slot],
                target_signature=signatures[target_slot],
                scores=scores_by_target[target_slot],
            )
        )
    interface = RhythmIdentityDiagnosticInterface(
        dataset=dataset,
        source_id=source_id,
        source_sha256=source_sha256,
        factorized_control_sha256=factorized_control_sha256,
        model_sha256=dict(model_sha256),
        comparisons=tuple(comparisons),
        metadata={
            "formalStudyData": False,
            "source_kind": "frozen_v98_observed_sidecar",
            "learned_here": False,
            "calibrated_here": False,
            "source_decision": "D-077",
            "claim_boundary": (
                "Read-only representation evidence; not a control target, selected "
                "model, operation label, prediction, or production decision."
            ),
        },
    )
    interface.validate()
    return interface
