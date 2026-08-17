#!/usr/bin/env python3
"""Filter a frozen Stage 3 source manifest to one training seed."""

import argparse
import json
from pathlib import Path


def filter_manifest(value, seed):
    sources = [row for row in value["sources"] if int(row["seed"]) == seed]
    if len(sources) != 4:
        raise ValueError("expected four sources for seed {}, found {}".format(seed, len(sources)))
    result = dict(value)
    result["status"] = "prospective_single_seed_pilot_subset"
    result["source_count"] = 4
    result["expected_output_count"] = 8
    result["seeds"] = [seed]
    result["sources"] = sources
    result["parent_manifest"] = str(Path("source_manifest.json"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = filter_manifest(json.loads(args.input.read_text(encoding="utf-8")), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
