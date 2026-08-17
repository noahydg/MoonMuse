"""Utilities for auditing Stage 3 per-attribute gradient interactions.

The summary and routing helpers are Torch-free so audit reports remain
testable on the local CPU-only evidence workstation. Tensor gradient helpers
import Torch lazily and are used only by the AutoDL diagnostic.
"""

import math
import statistics


PRIMARY_SCOPE = "shared_backbone"
PRIMARY_PAIR = "total_tempo__total_velocity"


def _finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _distribution(values):
    values = sorted(_finite(values))
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
            "mean": None,
            "negative_fraction": None,
            "zero_fraction": None,
        }

    def quantile(fraction):
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "minimum": values[0],
        "q25": quantile(0.25),
        "median": statistics.median(values),
        "q75": quantile(0.75),
        "maximum": values[-1],
        "mean": statistics.fmean(values),
        "negative_fraction": sum(value < 0 for value in values) / len(values),
        "zero_fraction": sum(value == 0 for value in values) / len(values),
    }


def summarize_gradient_records(records):
    """Aggregate per-batch norms and cosines without changing their meaning."""
    if not records:
        raise ValueError("at least one gradient record is required")
    scopes = sorted(records[0]["scopes"])
    summary = {"batch_count": len(records), "scopes": {}}
    for scope in scopes:
        norm_names = sorted(records[0]["scopes"][scope]["norms"])
        pair_names = sorted(records[0]["scopes"][scope]["cosines"])
        summary["scopes"][scope] = {
            "norms": {
                name: _distribution([
                    record["scopes"][scope]["norms"].get(name) for record in records
                ])
                for name in norm_names
            },
            "cosines": {
                name: _distribution([
                    record["scopes"][scope]["cosines"].get(name) for record in records
                ])
                for name in pair_names
            },
        }
    return summary


def classify_tempo_velocity_interaction(summary):
    """Route the next experiment using prospectively declared diagnostics.

    This is an exploratory engineering label, not a statistical test or a
    scientific acceptance gate.
    """
    scope = summary["scopes"][PRIMARY_SCOPE]
    interaction = scope["cosines"][PRIMARY_PAIR]
    tempo_norm = scope["norms"]["total_tempo"]["median"]
    velocity_norm = scope["norms"]["total_velocity"]["median"]
    if interaction["count"] < 8:
        return {
            "label": "insufficient_batches",
            "reason": "fewer than eight finite shared-backbone cosine observations",
            "tempo_to_velocity_norm_ratio": None,
        }

    norm_ratio = None
    if tempo_norm is not None and velocity_norm not in (None, 0):
        norm_ratio = tempo_norm / velocity_norm
    if interaction["median"] <= -0.05 and interaction["negative_fraction"] >= 0.75:
        label = "shared_gradient_conflict_supported"
        reason = "tempo and velocity total-component gradients oppose on most batches"
    elif interaction["median"] < 0 and interaction["negative_fraction"] >= 0.60:
        label = "mixed_shared_gradient_conflict"
        reason = "tempo and velocity conflict is present but not dominant"
    elif norm_ratio is not None and norm_ratio < 0.50:
        label = "tempo_gradient_underpowered"
        reason = "tempo gradient is less than half the velocity gradient without dominant conflict"
    else:
        label = "conflict_and_underpower_not_supported"
        reason = "neither the declared conflict nor tempo-underpower pattern is supported"
    return {
        "label": label,
        "reason": reason,
        "tempo_to_velocity_norm_ratio": norm_ratio,
    }


def add_gradient_lists(left, right):
    """Add two autograd gradient lists while preserving unused parameters."""
    if len(left) != len(right):
        raise ValueError("gradient lists must have the same length")
    combined = []
    for left_grad, right_grad in zip(left, right):
        if left_grad is None:
            combined.append(right_grad)
        elif right_grad is None:
            combined.append(left_grad)
        else:
            combined.append(left_grad + right_grad)
    return combined


def gradient_norm(gradients, indices=None):
    torch = __import__("torch")
    if indices is None:
        indices = range(len(gradients))
    squared = None
    for index in indices:
        gradient = gradients[index]
        if gradient is None:
            continue
        value = torch.sum(gradient.detach().float() ** 2)
        squared = value if squared is None else squared + value
    if squared is None:
        return 0.0
    return float(torch.sqrt(squared).item())


def gradient_cosine(left, right, indices=None):
    torch = __import__("torch")
    if len(left) != len(right):
        raise ValueError("gradient lists must have the same length")
    if indices is None:
        indices = range(len(left))
    dot = None
    left_sq = None
    right_sq = None
    for index in indices:
        left_grad = left[index]
        right_grad = right[index]
        if left_grad is None or right_grad is None:
            continue
        left_value = left_grad.detach().float()
        right_value = right_grad.detach().float()
        pair_dot = torch.sum(left_value * right_value)
        pair_left_sq = torch.sum(left_value ** 2)
        pair_right_sq = torch.sum(right_value ** 2)
        dot = pair_dot if dot is None else dot + pair_dot
        left_sq = pair_left_sq if left_sq is None else left_sq + pair_left_sq
        right_sq = pair_right_sq if right_sq is None else right_sq + pair_right_sq
    if dot is None or left_sq.item() == 0.0 or right_sq.item() == 0.0:
        return None
    return float((dot / torch.sqrt(left_sq * right_sq)).item())
