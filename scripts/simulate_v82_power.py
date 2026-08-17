#!/usr/bin/env python3
"""Prospective power simulation for the frozen v82 two-form listening design.

The simulation deliberately requires every co-primary endpoint to pass twice:
once across listeners and once across the 24 hash-selected stimulus slots.  It
therefore avoids treating repeated ratings from either facet as independent.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


SLOT_COUNT = 24
CELL_COUNT = 12
SLOTS_PER_CELL = 2
ALPHA_ONE_SIDED = 0.025
QUALITY_NI_MARGIN = -0.30


@dataclass(frozen=True)
class SimulationAssumptions:
    baseline_arousal_probability: float = 0.65
    candidate_arousal_probability: float = 0.80
    participant_logit_sd: float = 0.65
    slot_logit_sd: float = 0.45
    version_logit_sd: float = 0.25
    baseline_quality_mean: float = 3.50
    candidate_quality_difference: float = 0.00
    participant_quality_sd: float = 0.55
    slot_quality_sd: float = 0.35
    version_quality_sd: float = 0.25
    residual_quality_sd: float = 0.75


def logit(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one")
    return math.log(probability / (1 - probability))


def lower_t_bound(values: np.ndarray, alpha: float = ALPHA_ONE_SIDED) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two one-dimensional values are required")
    standard_error = values.std(ddof=1) / math.sqrt(values.size)
    return float(values.mean() - stats.t.ppf(1 - alpha, values.size - 1) * standard_error)


def form_system_matrix() -> np.ndarray:
    """Return [form, slot], with 1=candidate and 0=baseline.

    Each seed-by-quadrant cell owns two slots.  Their assignments are reversed
    between forms, so every form contains one candidate and one baseline in
    every cell and every slot is evaluated under both systems across forms.
    """

    matrix = np.zeros((2, SLOT_COUNT), dtype=np.int8)
    for cell in range(CELL_COUNT):
        first = cell * SLOTS_PER_CELL
        matrix[0, first] = 1
        matrix[1, first + 1] = 1
    if not np.all(matrix.sum(axis=1) == SLOT_COUNT // 2):
        raise AssertionError("each form must contain 12 candidate stimuli")
    if not np.all(matrix.sum(axis=0) == 1):
        raise AssertionError("each slot must reverse system across forms")
    return matrix


def simulate_once(
    participants: int,
    rng: np.random.Generator,
    assumptions: SimulationAssumptions,
) -> dict[str, bool]:
    if participants < 4 or participants % 2:
        raise ValueError("participants must be an even integer of at least four")
    per_form = participants // 2
    form = np.repeat(np.arange(2), per_form)
    systems = form_system_matrix()[form]

    participant_logit = rng.normal(0, assumptions.participant_logit_sd, participants)
    slot_logit = rng.normal(0, assumptions.slot_logit_sd, SLOT_COUNT)
    version_logit = rng.normal(0, assumptions.version_logit_sd, (2, SLOT_COUNT))
    baseline_logit = logit(assumptions.baseline_arousal_probability)
    candidate_effect = (
        logit(assumptions.candidate_arousal_probability) - baseline_logit
    )
    linear = (
        baseline_logit
        + candidate_effect * systems
        + participant_logit[:, None]
        + slot_logit[None, :]
        + np.where(systems == 1, version_logit[1], version_logit[0])
    )
    probabilities = 1 / (1 + np.exp(-linear))
    arousal = rng.binomial(1, probabilities).astype(float)

    participant_quality = rng.normal(0, assumptions.participant_quality_sd, participants)
    slot_quality = rng.normal(0, assumptions.slot_quality_sd, SLOT_COUNT)
    version_quality = rng.normal(0, assumptions.version_quality_sd, (2, SLOT_COUNT))
    quality = (
        assumptions.baseline_quality_mean
        + assumptions.candidate_quality_difference * systems
        + participant_quality[:, None]
        + slot_quality[None, :]
        + np.where(systems == 1, version_quality[1], version_quality[0])
        + rng.normal(0, assumptions.residual_quality_sd, (participants, SLOT_COUNT))
    )
    quality = np.clip(quality, 1.0, 5.0)

    participant_arousal_diff = np.array(
        [
            arousal[index, systems[index] == 1].mean()
            - arousal[index, systems[index] == 0].mean()
            for index in range(participants)
        ]
    )
    participant_quality_diff = np.array(
        [
            quality[index, systems[index] == 1].mean()
            - quality[index, systems[index] == 0].mean()
            for index in range(participants)
        ]
    )

    slot_arousal_diff = np.empty(SLOT_COUNT)
    slot_quality_diff = np.empty(SLOT_COUNT)
    for slot in range(SLOT_COUNT):
        candidate_rows = systems[:, slot] == 1
        baseline_rows = ~candidate_rows
        slot_arousal_diff[slot] = (
            arousal[candidate_rows, slot].mean()
            - arousal[baseline_rows, slot].mean()
        )
        slot_quality_diff[slot] = (
            quality[candidate_rows, slot].mean()
            - quality[baseline_rows, slot].mean()
        )

    arousal_listener_pass = lower_t_bound(participant_arousal_diff) > 0
    arousal_stimulus_pass = lower_t_bound(slot_arousal_diff) > 0
    quality_listener_pass = (
        lower_t_bound(participant_quality_diff) > QUALITY_NI_MARGIN
    )
    quality_stimulus_pass = lower_t_bound(slot_quality_diff) > QUALITY_NI_MARGIN
    arousal_pass = arousal_listener_pass and arousal_stimulus_pass
    quality_pass = quality_listener_pass and quality_stimulus_pass
    return {
        "arousal_listener_pass": arousal_listener_pass,
        "arousal_stimulus_pass": arousal_stimulus_pass,
        "quality_listener_pass": quality_listener_pass,
        "quality_stimulus_pass": quality_stimulus_pass,
        "arousal_coprimary_pass": arousal_pass,
        "quality_coprimary_pass": quality_pass,
        "overall_claim_pass": arousal_pass and quality_pass,
    }


def simulate(
    participant_counts: list[int],
    simulations: int,
    seed: int,
    assumptions: SimulationAssumptions,
) -> dict:
    if simulations < 100:
        raise ValueError("at least 100 simulations are required")
    rng = np.random.default_rng(seed)
    by_count = {}
    for participants in participant_counts:
        outcomes = [
            simulate_once(participants, rng, assumptions)
            for _ in range(simulations)
        ]
        keys = outcomes[0]
        by_count[str(participants)] = {
            key: sum(row[key] for row in outcomes) / simulations for key in keys
        }
    return {
        "schema": "moonmuse-v82-prospective-power-simulation-v1",
        "seed": seed,
        "simulations_per_participant_count": simulations,
        "participant_counts": participant_counts,
        "design": {
            "forms": 2,
            "stimulus_slots": SLOT_COUNT,
            "ratings_per_participant": SLOT_COUNT,
            "candidate_ratings_per_participant": SLOT_COUNT // 2,
            "baseline_ratings_per_participant": SLOT_COUNT // 2,
            "one_sided_alpha": ALPHA_ONE_SIDED,
            "quality_noninferiority_margin": QUALITY_NI_MARGIN,
            "claim_requires_listener_and_stimulus_analysis": True,
            "claim_requires_both_coprimary_endpoints": True,
        },
        "assumptions": asdict(assumptions),
        "estimated_power": by_count,
        "boundary": (
            "This is a design calculation under explicit assumptions, not an observed "
            "result. It must not be used to claim that the candidate will pass."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, nargs="+", default=[48, 60, 72])
    parser.add_argument("--simulations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=820260809)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = simulate(
        participant_counts=args.participants,
        simulations=args.simulations,
        seed=args.seed,
        assumptions=SimulationAssumptions(),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
