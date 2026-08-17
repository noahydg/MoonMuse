"""Counterfactual arousal-alignment helpers for Stage 3 training.

The public specification builder has no Torch dependency so it can be audited
locally.  Tensor operations import Torch lazily and run only in the training
environment.
"""

import math


NUMERIC_ATTRIBUTES = {
    "velocity": ("Note_Velocity_", 1),
    "tempo": ("Tempo_", 1),
    "duration": ("Note_Duration_", -1),
}

MASS_ATTRIBUTES = {
    "note_event_mass": ("Note_Octave_", 1),
    "onset_event_mass": ("Beat_", 1),
}


def _numeric_suffix(token, prefix):
    if not token.startswith(prefix):
        return None
    try:
        value = float(token[len(prefix):])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _attribute_config(config, name, default_margin, default_weight):
    raw = config.get("attributes", {}).get(name, {})
    enabled = bool(raw.get("enabled", True))
    margin = float(raw.get("margin", default_margin))
    weight = float(raw.get("weight", default_weight))
    if margin < 0:
        raise ValueError("{} margin must be non-negative".format(name))
    if weight <= 0:
        raise ValueError("{} weight must be positive".format(name))
    return enabled, margin, weight


def build_alignment_spec(event2idx, config):
    """Build a validated, JSON-serializable alignment specification."""
    try:
        high_index = int(event2idx["Arousal_High"])
        low_index = int(event2idx["Arousal_Low"])
    except KeyError as exc:
        raise ValueError("factorized arousal tokens are required") from exc

    attributes = []
    for name, (prefix, direction) in NUMERIC_ATTRIBUTES.items():
        enabled, margin, weight = _attribute_config(config, name, 0.05, 1.0)
        if not enabled:
            continue
        tokens = []
        for token, index in event2idx.items():
            value = _numeric_suffix(token, prefix)
            if value is not None:
                tokens.append((int(index), value, token))
        tokens.sort()
        if len(tokens) < 2:
            raise ValueError("{} requires at least two numeric tokens".format(name))
        minimum = min(item[1] for item in tokens)
        maximum = max(item[1] for item in tokens)
        if maximum <= minimum:
            raise ValueError("{} token values have no range".format(name))
        attributes.append({
            "name": name,
            "kind": "numeric",
            "direction": direction,
            "margin": margin,
            "weight": weight,
            "token_indices": [item[0] for item in tokens],
            "token_names": [item[2] for item in tokens],
            "normalized_values": [
                (item[1] - minimum) / (maximum - minimum) for item in tokens
            ],
            "raw_min": minimum,
            "raw_max": maximum,
        })

    for name, (prefix, direction) in MASS_ATTRIBUTES.items():
        enabled, margin, weight = _attribute_config(config, name, 0.002, 0.25)
        if not enabled:
            continue
        tokens = sorted(
            (int(index), token) for token, index in event2idx.items()
            if token.startswith(prefix)
        )
        if not tokens:
            raise ValueError("{} has no matching tokens".format(name))
        attributes.append({
            "name": name,
            "kind": "probability_mass",
            "direction": direction,
            "margin": margin,
            "weight": weight,
            "token_indices": [item[0] for item in tokens],
            "token_names": [item[1] for item in tokens],
        })

    if not attributes:
        raise ValueError("at least one arousal-alignment attribute is required")
    return {
        "schema": "moonmuse-stage3-counterfactual-arousal-alignment-v1",
        "high_token_index": high_index,
        "low_token_index": low_index,
        "attributes": attributes,
    }


def _torch():
    import torch
    return torch


def flip_arousal_tokens(dec_input, spec):
    """Flip exactly one High/Low control token in every batch item."""
    torch = _torch()
    high = spec["high_token_index"]
    low = spec["low_token_index"]
    high_mask = dec_input.eq(high)
    low_mask = dec_input.eq(low)
    high_counts = high_mask.sum(dim=1)
    low_counts = low_mask.sum(dim=1)
    valid = high_counts.add(low_counts).eq(1)
    if not bool(torch.all(valid).item()):
        raise ValueError("every sequence must contain exactly one Arousal_High/Low token")
    flipped = dec_input.clone()
    flipped[high_mask] = low
    flipped[low_mask] = high
    return flipped, high_counts.eq(1)


def _token_mask(target, token_indices):
    torch = _torch()
    indices = torch.as_tensor(token_indices, device=target.device, dtype=target.dtype)
    return target.unsqueeze(-1).eq(indices).any(dim=-1)


def _masked_sample_mean(values, mask):
    counts = mask.sum(dim=1)
    active = counts.gt(0)
    means = (values * mask.to(values.dtype)).sum(dim=1) / counts.clamp_min(1).to(values.dtype)
    return means, active


def _numeric_expectation(logits, attribute):
    torch = _torch()
    indices = torch.as_tensor(
        attribute["token_indices"], device=logits.device, dtype=torch.long
    )
    values = torch.as_tensor(
        attribute["normalized_values"], device=logits.device, dtype=logits.dtype
    )
    conditional = torch.softmax(logits.index_select(-1, indices), dim=-1)
    return (conditional * values).sum(dim=-1)


def _probability_mass(logits, attribute):
    torch = _torch()
    indices = torch.as_tensor(
        attribute["token_indices"], device=logits.device, dtype=torch.long
    )
    group_log_mass = torch.logsumexp(logits.index_select(-1, indices), dim=-1)
    total_log_mass = torch.logsumexp(logits, dim=-1)
    return torch.exp(group_log_mass - total_log_mass)


def compute_counterfactual_alignment_loss(
    real_logits,
    counterfactual_logits,
    dec_input,
    dec_target,
    pad_token,
    spec,
):
    """Return directional hinge loss and auditable High-minus-Low deltas."""
    torch = _torch()
    _, original_is_high = flip_arousal_tokens(dec_input, spec)
    choose_real = original_is_high[:, None, None]
    high_logits = torch.where(choose_real, real_logits, counterfactual_logits)
    low_logits = torch.where(choose_real, counterfactual_logits, real_logits)
    valid_target = dec_target.ne(pad_token)

    weighted_losses = []
    weights = []
    diagnostics = {}
    for attribute in spec["attributes"]:
        if attribute["kind"] == "numeric":
            mask = _token_mask(dec_target, attribute["token_indices"])
            high_values = _numeric_expectation(high_logits, attribute)
            low_values = _numeric_expectation(low_logits, attribute)
        elif attribute["kind"] == "probability_mass":
            mask = valid_target
            high_values = _probability_mass(high_logits, attribute)
            low_values = _probability_mass(low_logits, attribute)
        else:
            raise ValueError("unsupported alignment kind: {}".format(attribute["kind"]))

        high_means, active = _masked_sample_mean(high_values, mask)
        low_means, _ = _masked_sample_mean(low_values, mask)
        if not bool(torch.any(active).item()):
            continue
        difference = high_means[active] - low_means[active]
        signed_difference = difference * float(attribute["direction"])
        attribute_loss = torch.relu(float(attribute["margin"]) - signed_difference).mean()
        weight = float(attribute["weight"])
        weighted_losses.append(attribute_loss * weight)
        weights.append(weight)
        diagnostics[attribute["name"]] = {
            "loss": attribute_loss,
            "high_minus_low": difference.mean(),
            "active_samples": int(active.sum().item()),
        }

    if not weighted_losses:
        raise ValueError("batch has no positions for enabled alignment attributes")
    alignment_loss = torch.stack(weighted_losses).sum() / sum(weights)
    return {
        "alignment_loss": alignment_loss,
        "attributes": diagnostics,
    }
