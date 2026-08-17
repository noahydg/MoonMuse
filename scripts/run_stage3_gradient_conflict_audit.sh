#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CHECKPOINT="${1:?usage: run_stage3_gradient_conflict_audit.sh V67_BEST_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-generation/stage3-gradient-conflict-audit-${RUN_ID}}"
BATCHES="${BATCHES:-16}"
CONFIG="${CONFIG:-stage3_performance/config/emopia_finetune_attribute_only_alignment_gpt2.yaml}"

test -s "$CHECKPOINT"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
checkpoint_before="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
cat > "$RUN_ROOT/manifest.txt" <<EOF
protocol=stage3-tempo-velocity-gradient-conflict-audit-v1
formal_study_data=false
training_performed=false
checkpoint=$CHECKPOINT
checkpoint_sha256_before=$checkpoint_before
split=valid
shuffle=false
seed=42
batches=$BATCHES
EOF

"$PYTHON_BIN" scripts/audit_stage3_attribute_gradients.py "$CHECKPOINT" \
  --config "$CONFIG" --split valid --max-batches "$BATCHES" \
  --seed 42 --num-workers 0 --device cuda:0 --output-dir "$RUN_ROOT/audit" \
  2>&1 | tee "$RUN_ROOT/audit.log"
"$PYTHON_BIN" scripts/validate_stage3_gradient_conflict_audit.py \
  "$RUN_ROOT/audit/gradient_audit.json" --expected-batches "$BATCHES" \
  --output "$RUN_ROOT/validation.json" | tee "$RUN_ROOT/validation.log"
checkpoint_after="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
test "$checkpoint_before" = "$checkpoint_after"
printf 'checkpoint_sha256_after=%s\n' "$checkpoint_after" >> "$RUN_ROOT/manifest.txt"
sha256sum stage3_gradient_conflict.py scripts/audit_stage3_attribute_gradients.py \
  scripts/validate_stage3_gradient_conflict_audit.py \
  scripts/run_stage3_gradient_conflict_audit.sh "$CONFIG" "$CHECKPOINT" \
  > "$RUN_ROOT/input_and_code_sha256.txt"
printf 'STAGE3_GRADIENT_CONFLICT_AUDIT_COMPLETE run_root=%s batches=%s\n' \
  "$RUN_ROOT" "$BATCHES" | tee "$RUN_ROOT/status.txt"
tar -czf "$RUN_ROOT.tar.gz" "$RUN_ROOT"
sha256sum "$RUN_ROOT.tar.gz" > "$RUN_ROOT.tar.gz.sha256"
