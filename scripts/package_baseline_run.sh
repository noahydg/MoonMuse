#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: package_baseline_run.sh RUN_DIR [DEST_DIR]}"
RUN_ABS="$(cd "$RUN_DIR" && pwd)"
RUN_NAME="$(basename "$RUN_ABS")"
DEST_DIR="${2:-$RUN_ABS/../artifacts}"
mkdir -p "$DEST_DIR"

test -s "$RUN_ABS/generation_gate.json"
test -s "$RUN_ABS/midi_descriptors.json"

python - "$RUN_ABS/generation_gate.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if not report.get("passed"):
    raise SystemExit(f"gate did not pass: {sys.argv[1]}")
PY

EVIDENCE_LIST="$DEST_DIR/${RUN_NAME}-evidence-files.txt"
SAMPLE_LIST="$DEST_DIR/${RUN_NAME}-sample-files.txt"

find "$RUN_ABS" -maxdepth 1 -type f ! -name '*.mid' -print \
  | sed "s#^$RUN_ABS/##" > "$EVIDENCE_LIST"
find "$RUN_ABS/logs" -type f -print \
  | sed "s#^$RUN_ABS/##" >> "$EVIDENCE_LIST"
sort -u "$EVIDENCE_LIST" -o "$EVIDENCE_LIST"

find "$RUN_ABS" -maxdepth 1 -type f -name 'samp_*_Q[1-4]_full.mid' -print \
  | sed "s#^$RUN_ABS/##" > "$SAMPLE_LIST"

test -s "$EVIDENCE_LIST"
test -s "$SAMPLE_LIST"

tar -czf "$DEST_DIR/${RUN_NAME}-evidence.tar.gz" \
  -C "$RUN_ABS" -T "$EVIDENCE_LIST"
tar -czf "$DEST_DIR/${RUN_NAME}-samples.tar.gz" \
  -C "$RUN_ABS" -T "$SAMPLE_LIST"

sha256sum \
  "$DEST_DIR/${RUN_NAME}-evidence.tar.gz" \
  "$DEST_DIR/${RUN_NAME}-samples.tar.gz" \
  > "$DEST_DIR/${RUN_NAME}-artifacts.sha256"

printf 'BASELINE_RUN_PACKAGED run_dir=%s destination=%s\n' \
  "$RUN_DIR" "$DEST_DIR"
