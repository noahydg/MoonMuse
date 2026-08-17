#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_ROOT="${1:?usage: run_stage3_arousal_alignment_pipeline.sh FROZEN_SOURCE_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d-%H%M%S)}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-generation/stage3-arousal-alignment-campaign-${PIPELINE_ID}}"
OUT_ROOT="${OUT_ROOT:-generation/stage3-arousal-alignment-gate-c-${PIPELINE_ID}}"

PYTHON_BIN="$PYTHON_BIN" CAMPAIGN_ID="$PIPELINE_ID" CAMPAIGN_DIR="$CAMPAIGN_DIR" \
  bash scripts/run_stage3_arousal_alignment_campaign.sh
printf '0\n' > "$CAMPAIGN_DIR/campaign.rc"

set +e
PYTHON_BIN="$PYTHON_BIN" RUN_ID="$PIPELINE_ID" OUT_ROOT="$OUT_ROOT" \
  bash scripts/run_stage3_arousal_alignment_gate_c.sh "$SOURCE_ROOT" "$CAMPAIGN_DIR"
gate_rc=$?
set -e
printf '%s\n' "$gate_rc" > "$CAMPAIGN_DIR/gate_c_handoff.rc"
printf 'pipeline_id=%s\ncampaign_dir=%s\nout_root=%s\ngate_c_rc=%s\n' \
  "$PIPELINE_ID" "$CAMPAIGN_DIR" "$OUT_ROOT" "$gate_rc" \
  > "$CAMPAIGN_DIR/pipeline_status.txt"
exit "$gate_rc"
