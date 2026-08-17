"""Helpers for a Stage 3 reconstruction loss over performance attributes only.

The specification builder is Torch-free so configuration and vocabulary
coverage can be audited on a CPU-only workstation. Tensor masking imports
Torch lazily in the training environment.
"""

ATTRIBUTE_PREFIXES = {
    "tempo": "Tempo_",
    "duration": "Note_Duration_",
    "velocity": "Note_Velocity_",
}


def build_attribute_objective_spec(event2idx, config):
    mode = config.get("mode", "all_tokens")
    if mode == "all_tokens":
        return {
            "schema": "moonmuse-stage3-reconstruction-objective-v1",
            "mode": mode,
            "attributes": [],
            "token_indices": [],
            "token_names": [],
        }
    if mode != "performance_attributes_only":
        raise ValueError("unsupported reconstruction objective mode: {}".format(mode))
    attributes = config.get("attributes", ["tempo", "duration", "velocity"])
    if not isinstance(attributes, list) or not attributes:
        raise ValueError("performance_attributes_only requires a non-empty attributes list")
    unknown = sorted(set(attributes) - set(ATTRIBUTE_PREFIXES))
    if unknown:
        raise ValueError("unsupported performance attributes: {}".format(unknown))
    if len(set(attributes)) != len(attributes):
        raise ValueError("performance attributes must be unique")

    tokens = []
    counts = {}
    for attribute in attributes:
        prefix = ATTRIBUTE_PREFIXES[attribute]
        matched = sorted(
            (int(index), token)
            for token, index in event2idx.items()
            if token.startswith(prefix)
        )
        if not matched:
            raise ValueError("{} has no matching vocabulary tokens".format(attribute))
        counts[attribute] = len(matched)
        tokens.extend(matched)
    tokens.sort()
    return {
        "schema": "moonmuse-stage3-reconstruction-objective-v1",
        "mode": mode,
        "attributes": list(attributes),
        "attribute_token_counts": counts,
        "token_indices": [item[0] for item in tokens],
        "token_names": [item[1] for item in tokens],
    }


def mask_reconstruction_targets(target, pad_token, spec):
    """Replace non-objective targets with PAD and report selected positions."""
    if spec["mode"] == "all_tokens":
        return target, int(target.ne(pad_token).sum().item())
    import torch

    indices = torch.as_tensor(
        spec["token_indices"], device=target.device, dtype=target.dtype
    )
    selected = target.unsqueeze(-1).eq(indices).any(dim=-1)
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        raise ValueError("batch contains no selected performance-attribute targets")
    masked = target.clone()
    masked[~selected] = pad_token
    return masked, selected_count
