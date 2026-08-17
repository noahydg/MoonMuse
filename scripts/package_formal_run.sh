#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: package_formal_run.sh RUN_DIR [DEST_DIR]}"
RUN_ABS="$(cd "$RUN_DIR" && pwd -P)"
RUN_NAME="$(basename "$RUN_ABS")"
DEST_DIR="${2:-$ROOT/formal-artifacts}"
mkdir -p "$DEST_DIR"

INFERENCE_DIR="${INFERENCE_DIR:-$(python - "$RUN_ABS" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidates = sorted(
    (path for path in run_dir.glob("inference-seed*") if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for candidate in candidates:
    gate = candidate / "generation_gate.json"
    if not gate.is_file():
        continue
    try:
        report = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if report.get("passed"):
        print(os.fspath(candidate.resolve()))
        break
PY
)}"
test -n "$INFERENCE_DIR"
test -s "$RUN_ABS/training_validation.json"
test -s "$INFERENCE_DIR/generation_gate.json"

python - "$RUN_ABS/training_validation.json" "$INFERENCE_DIR/generation_gate.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("passed"):
        raise SystemExit(f"gate did not pass: {path}")
PY

EVIDENCE_LIST="$DEST_DIR/${RUN_NAME}-evidence-files.txt"
CHECKPOINT_LIST="$DEST_DIR/${RUN_NAME}-checkpoint-files.txt"
SAMPLE_LIST="$DEST_DIR/${RUN_NAME}-sample-files.txt"

find "$RUN_ABS" -maxdepth 1 -type f -print \
  | sed "s#^$RUN_ABS/##" > "$EVIDENCE_LIST"
find "$RUN_ABS/logs" -type f -print \
  | sed "s#^$RUN_ABS/##" >> "$EVIDENCE_LIST"
find "$INFERENCE_DIR" -maxdepth 1 -type f -print \
  | sed "s#^$RUN_ABS/##" >> "$EVIDENCE_LIST"
find "$INFERENCE_DIR" -type f -name '*.yaml' -print \
  | sed "s#^$RUN_ABS/##" >> "$EVIDENCE_LIST"
sort -u "$EVIDENCE_LIST" -o "$EVIDENCE_LIST"

find "$RUN_ABS/checkpoints" -type f -path '*/params/best_params.pt' -print \
  | sed "s#^$RUN_ABS/##" > "$CHECKPOINT_LIST"

find "$INFERENCE_DIR" -type f \
  \( -path '*/midi/*.mid' \
     -o -path '*/stage1_chord/*.txt' \
     -o -path '*/stage2_melody/*.txt' \
     -o -path '*/stage2_performance/*.txt' \
     -o -path '*/stage3_performance/*.txt' \) -print \
  | sed "s#^$RUN_ABS/##" > "$SAMPLE_LIST"

test -s "$EVIDENCE_LIST"
test -s "$CHECKPOINT_LIST"
test -s "$SAMPLE_LIST"

tar -czf "$DEST_DIR/${RUN_NAME}-evidence.tar.gz" -C "$RUN_ABS" -T "$EVIDENCE_LIST"
tar -czf "$DEST_DIR/${RUN_NAME}-best-params.tar.gz" -C "$RUN_ABS" -T "$CHECKPOINT_LIST"
tar -czf "$DEST_DIR/${RUN_NAME}-samples.tar.gz" -C "$RUN_ABS" -T "$SAMPLE_LIST"

sha256sum \
  "$DEST_DIR/${RUN_NAME}-evidence.tar.gz" \
  "$DEST_DIR/${RUN_NAME}-best-params.tar.gz" \
  "$DEST_DIR/${RUN_NAME}-samples.tar.gz" \
  > "$DEST_DIR/${RUN_NAME}-artifacts.sha256"

printf 'FORMAL_RUN_PACKAGED run_dir=%s destination=%s\n' "$RUN_DIR" "$DEST_DIR"
