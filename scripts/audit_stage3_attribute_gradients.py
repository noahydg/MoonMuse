#!/usr/bin/env python3
"""Read-only gradient interaction audit for a trained Stage 3 checkpoint."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "stage3_performance") not in sys.path:
    sys.path.insert(0, str(ROOT / "stage3_performance"))

from stage3_arousal_alignment import (  # noqa: E402
    build_alignment_spec,
    compute_counterfactual_alignment_loss,
    flip_arousal_tokens,
)
from stage3_gradient_conflict import (  # noqa: E402
    add_gradient_lists,
    classify_tempo_velocity_interaction,
    gradient_cosine,
    gradient_norm,
    summarize_gradient_records,
)
from stage3_performance.dataloader import REMISkylineToMidiTransformerDataset  # noqa: E402
from stage3_performance.model.music_gpt2 import MusicGPT2  # noqa: E402
from utils import pickle_load  # noqa: E402


ATTRIBUTES = ("tempo", "duration", "velocity")
PAIR_NAMES = (
    ("total_tempo", "total_velocity"),
    ("alignment_tempo", "alignment_velocity"),
    ("reconstruction_tempo", "reconstruction_velocity"),
    ("reconstruction_tempo", "alignment_tempo"),
    ("reconstruction_velocity", "alignment_velocity"),
    ("total_tempo", "total_duration"),
    ("total_velocity", "total_duration"),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def token_mask(target, token_indices):
    indices = torch.as_tensor(token_indices, device=target.device, dtype=target.dtype)
    return target.unsqueeze(-1).eq(indices).any(dim=-1)


def reconstruction_components(logits, target, pad_token, attributes):
    masks = {
        name: token_mask(target, attributes[name]["token_indices"])
        for name in ATTRIBUTES
    }
    counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    total_count = sum(counts.values())
    if total_count == 0 or any(count == 0 for count in counts.values()):
        raise ValueError("batch must contain tempo, duration, and velocity targets")
    flat_logits = logits.reshape(-1, logits.size(-1))
    components = {}
    raw = {}
    for name in ATTRIBUTES:
        masked_target = target.clone()
        masked_target[~masks[name]] = pad_token
        loss_sum = F.cross_entropy(
            flat_logits,
            masked_target.reshape(-1).long(),
            ignore_index=pad_token,
            reduction="sum",
        )
        raw[name] = loss_sum / counts[name]
        components[name] = loss_sum / total_count
    return components, raw, counts


def parameter_scopes(named_parameters):
    scopes = {
        "full_model": [],
        "shared_backbone": [],
        "input_conditioning": [],
        "output_head": [],
    }
    for index, (name, _) in enumerate(named_parameters):
        scopes["full_model"].append(index)
        if name.startswith("transformer_decoder."):
            scopes["shared_backbone"].append(index)
        if name.startswith("token_emb.") or name.startswith("segemb."):
            scopes["input_conditioning"].append(index)
        if name.startswith("dec_out_proj."):
            scopes["output_head"].append(index)
    if any(not indices for indices in scopes.values()):
        raise ValueError("one or more declared parameter scopes are empty")
    return scopes


def load_model(config, dataset, checkpoint, device):
    model_conf = config["model"]
    model = MusicGPT2(
        dataset.vocab_size,
        model_conf["n_layer"],
        model_conf["n_head"],
        model_conf["d_model"],
        model_conf["d_ff"],
        model_conf["d_embed"],
        use_segment_emb=model_conf["use_segemb"],
        n_segment_types=model_conf.get("n_segment_types", 2),
        use_chord_mhot_emb=False,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = [
        name for name in load_result.unexpected_keys
        if not (name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"))
    ]
    if missing or unexpected:
        raise ValueError("checkpoint mismatch missing={} unexpected={}".format(missing, unexpected))
    model.eval()
    return model


def write_markdown(report, path):
    summary = report["gradient_summary"]
    interaction = summary["scopes"]["shared_backbone"]["cosines"][
        "total_tempo__total_velocity"
    ]
    route = report["diagnostic_route"]
    lines = [
        "# Stage 3 tempo/velocity gradient audit (v68)",
        "",
        "- Status: read-only diagnostic complete",
        "- Training performed: no",
        "- Checkpoint modified: no",
        "- Split: `{}`".format(report["dataset"]["split"]),
        "- Batches/examples: `{}/{}`".format(
            summary["batch_count"], report["dataset"]["example_count"]
        ),
        "- Route: `{}`".format(route["label"]),
        "- Reason: {}".format(route["reason"]),
        "",
        "## Primary shared-backbone observation",
        "",
        "The prospectively primary quantity is the cosine between the actual tempo and velocity total-objective gradient contributions.",
        "",
        "- median cosine: `{:.6f}`".format(interaction["median"]),
        "- negative fraction: `{:.3f}`".format(interaction["negative_fraction"]),
        "- interquartile range: `[{:.6f}, {:.6f}]`".format(interaction["q25"], interaction["q75"]),
        "- median tempo/velocity gradient-norm ratio: `{}`".format(
            "{:.6f}".format(route["tempo_to_velocity_norm_ratio"])
            if route["tempo_to_velocity_norm_ratio"] is not None else "n/a"
        ),
        "",
        "This is an exploratory optimization diagnostic, not a statistical test, listening result, or paper acceptance gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="stage3_performance/config/emopia_finetune_attribute_only_alignment_gpt2.yaml")
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.max_batches < 8:
        parser.error("--max-batches must be at least 8")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the declared AutoDL protocol")
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    representation = "functional"
    data_conf = config["data_loader"]
    vocab_path = Path(data_conf["vocab_path"].format(representation))
    split_path = Path(data_conf["train_split"] if args.split == "train" else data_conf["val_split"])
    event2idx = pickle_load(str(vocab_path))[0]
    alignment_conf = config["arousal_alignment"]
    alignment_spec = build_alignment_spec(event2idx, alignment_conf)
    attributes = {item["name"]: item for item in alignment_spec["attributes"]}
    if set(attributes) != set(ATTRIBUTES):
        raise ValueError("v68 requires exactly tempo, duration, and velocity alignment")

    dataset = REMISkylineToMidiTransformerDataset(
        data_dir=data_conf["data_path"].format(representation),
        vocab_file=str(vocab_path),
        model_dec_seqlen=config["model"]["max_len"],
        pieces=pickle_load(str(split_path)),
        pad_to_same=True,
        predict_key=False,
    )
    batch_size = args.batch_size or data_conf["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = load_model(config, dataset, str(checkpoint_path), device)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    scopes = parameter_scopes(named_parameters)
    alignment_weight = float(alignment_conf["weight"])
    alignment_weight_sum = sum(float(attributes[name]["weight"]) for name in ATTRIBUTES)
    records = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        dec_input = batch["dec_input"].to(device)
        dec_target = batch["dec_target"].to(device)
        track_mask = batch["track_mask"].to(device)
        real_logits = model(dec_input, seg_inp=track_mask, chord_inp=None)
        counterfactual_input, _ = flip_arousal_tokens(dec_input, alignment_spec)
        counterfactual_logits = model(counterfactual_input, seg_inp=track_mask, chord_inp=None)

        recon_components, recon_raw, target_counts = reconstruction_components(
            real_logits, dec_target, dataset.pad_token, attributes
        )
        alignment_result = compute_counterfactual_alignment_loss(
            real_logits,
            counterfactual_logits,
            dec_input,
            dec_target,
            dataset.pad_token,
            alignment_spec,
        )
        objective_tensors = {}
        raw_values = {}
        for name in ATTRIBUTES:
            objective_tensors["reconstruction_{}".format(name)] = recon_components[name]
            raw_values["reconstruction_{}".format(name)] = float(recon_raw[name].item())
            alignment_raw = alignment_result["attributes"][name]["loss"]
            alignment_scale = (
                alignment_weight * float(attributes[name]["weight"]) / alignment_weight_sum
            )
            objective_tensors["alignment_{}".format(name)] = alignment_raw * alignment_scale
            raw_values["alignment_{}".format(name)] = float(alignment_raw.item())

        gradients = {}
        objective_names = list(objective_tensors)
        for objective_index, name in enumerate(objective_names):
            gradients[name] = list(torch.autograd.grad(
                objective_tensors[name],
                parameters,
                retain_graph=objective_index + 1 < len(objective_names),
                allow_unused=True,
            ))
        for name in ATTRIBUTES:
            gradients["total_{}".format(name)] = add_gradient_lists(
                gradients["reconstruction_{}".format(name)],
                gradients["alignment_{}".format(name)],
            )

        scope_records = {}
        for scope_name, indices in scopes.items():
            scope_records[scope_name] = {
                "norms": {
                    name: gradient_norm(gradient_list, indices)
                    for name, gradient_list in gradients.items()
                },
                "cosines": {
                    "{}__{}".format(left, right): gradient_cosine(
                        gradients[left], gradients[right], indices
                    )
                    for left, right in PAIR_NAMES
                },
            }
        records.append({
            "batch_index": batch_index,
            "piece_ids": list(batch["piece_id"]),
            "target_counts": target_counts,
            "raw_objective_values": raw_values,
            "scaled_objective_values": {
                name: float(value.detach().item()) for name, value in objective_tensors.items()
            },
            "scopes": scope_records,
        })
        print("[gradient-audit] batch {}/{} pieces={} tempo_velocity_cosine={}".format(
            batch_index + 1,
            args.max_batches,
            len(batch["piece_id"]),
            scope_records["shared_backbone"]["cosines"]["total_tempo__total_velocity"],
        ), flush=True)
        del real_logits, counterfactual_logits, gradients, objective_tensors

    if len(records) != args.max_batches:
        raise ValueError("requested {} batches but observed {}".format(args.max_batches, len(records)))
    summary = summarize_gradient_records(records)
    route = classify_tempo_velocity_interaction(summary)
    report = {
        "schema": "moonmuse-stage3-gradient-conflict-audit-v1",
        "status": "read_only_gradient_diagnostic_complete",
        "formal_study_data": False,
        "training_performed": False,
        "checkpoint_modified": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
        },
        "configuration": {
            "path": str(config_path),
            "sha256": sha256(config_path),
            "alignment_weight": alignment_weight,
        },
        "dataset": {
            "split": args.split,
            "split_path": str(split_path),
            "split_sha256": sha256(split_path),
            "vocab_path": str(vocab_path),
            "vocab_sha256": sha256(vocab_path),
            "batch_size": batch_size,
            "batch_count": len(records),
            "example_count": sum(len(record["piece_ids"]) for record in records),
            "seed": args.seed,
            "shuffle": False,
        },
        "parameter_scopes": {
            name: {
                "tensor_count": len(indices),
                "parameter_count": sum(parameters[index].numel() for index in indices),
            }
            for name, indices in scopes.items()
        },
        "prospective_routing": {
            "primary_scope": "shared_backbone",
            "primary_pair": "total_tempo__total_velocity",
            "strong_conflict": "median <= -0.05 and negative_fraction >= 0.75",
            "mixed_conflict": "median < 0 and negative_fraction >= 0.60",
            "tempo_underpowered": "no dominant conflict and median tempo/velocity norm ratio < 0.50",
        },
        "diagnostic_route": route,
        "gradient_summary": summary,
        "records": records,
    }
    (output_dir / "gradient_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "batch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "batch_index", "piece_ids", "tempo_targets", "duration_targets", "velocity_targets",
            "shared_total_tempo_norm", "shared_total_velocity_norm", "shared_tempo_velocity_cosine",
        ])
        writer.writeheader()
        for record in records:
            shared = record["scopes"]["shared_backbone"]
            writer.writerow({
                "batch_index": record["batch_index"],
                "piece_ids": "|".join(record["piece_ids"]),
                "tempo_targets": record["target_counts"]["tempo"],
                "duration_targets": record["target_counts"]["duration"],
                "velocity_targets": record["target_counts"]["velocity"],
                "shared_total_tempo_norm": shared["norms"]["total_tempo"],
                "shared_total_velocity_norm": shared["norms"]["total_velocity"],
                "shared_tempo_velocity_cosine": shared["cosines"]["total_tempo__total_velocity"],
            })
    write_markdown(report, output_dir / "README.md")
    print(json.dumps({
        "status": report["status"],
        "route": route,
        "output": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
