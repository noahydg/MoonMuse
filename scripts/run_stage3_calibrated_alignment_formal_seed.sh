#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_CHECKPOINT="${1:?usage: run_stage3_calibrated_alignment_formal_seed.sh V63_SEED_CHECKPOINT}"
SEED="${SEED:?set SEED to 42, 43, or 44}"
case "$SEED" in 42|43|44) ;; *) printf 'unsupported seed: %s\n' "$SEED" >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-generation/stage3-calibrated-alignment-formal-seed${SEED}-${RUN_ID}}"
CHECKPOINT_ROOT="$RUN_DIR/checkpoints/stage3_calibrated_alignment_finetune"
LOG_ROOT="$RUN_DIR/logs"
CONFIG=stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml

test -s "$SOURCE_CHECKPOINT"
test ! -e "$RUN_DIR"
mkdir -p "$LOG_ROOT"
source_hash_before="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
cat > "$RUN_DIR/training_manifest.txt" <<EOF
protocol=stage3-performance-calibrated-alignment-formal-v1
seed=$SEED
max_epochs=$MAX_EPOCHS
early_stopping_patience=$PATIENCE
source_checkpoint=$SOURCE_CHECKPOINT
source_checkpoint_sha256_before=$source_hash_before
conditioning=factorized_va
optimizer_transfer=reset_from_v63_alignment_checkpoint
selection=lowest_total_validation_objective
reconstruction_objective=performance_attributes_only
alignment_weight=2.25
tempo_alignment_coefficient=0.500
duration_alignment_coefficient=0.875
velocity_alignment_coefficient=0.875
formal_study_data=false
EOF
sha256sum stage3_attribute_objective.py stage3_arousal_alignment.py \
  stage3_performance/train.py "$CONFIG" \
  scripts/run_stage3_calibrated_alignment_formal_seed.sh \
  scripts/validate_stage3_attribute_objective_formal_run.py \
  scripts/validate_stage3_calibrated_alignment_formal_run.py \
  "$SOURCE_CHECKPOINT" > "$RUN_DIR/training_code_and_source_sha256.txt"

"$PYTHON_BIN" scripts/validate_checkpoint.py "$SOURCE_CHECKPOINT" \
  --output "$RUN_DIR/source_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/source_checkpoint_validation.log"
printf 'seed %s：v69 校准对齐正式微调（最多 %s epochs）\n' "$SEED" "$MAX_EPOCHS" \
  > "$RUN_DIR/current_stage.txt"
{ time "$PYTHON_BIN" stage3_performance/train.py \
  -c "$CONFIG" -m gpt2 -r functional --seed "$SEED" \
  --num_workers "$WORKERS" --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
  --ckpt_dir "$CHECKPOINT_ROOT" --pretrained_param_path "$SOURCE_CHECKPOINT" \
  --reset_optimizer; } 2>&1 | tee "$LOG_ROOT/training.log"

test -s "$CHECKPOINT_ROOT/params/best_params.pt"
test -s "$CHECKPOINT_ROOT/optim/best_optim.pt"
"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$CHECKPOINT_ROOT/params/best_params.pt" \
  --optimizer "$CHECKPOINT_ROOT/optim/best_optim.pt" \
  --reference "$SOURCE_CHECKPOINT" \
  --output "$RUN_DIR/formal_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/formal_checkpoint_validation.log"
"$PYTHON_BIN" scripts/validate_stage3_calibrated_alignment_formal_run.py \
  "$RUN_DIR" --seed "$SEED" --allow-missing-status \
  --output "$RUN_DIR/formal_run_validation.json" \
  2>&1 | tee "$LOG_ROOT/formal_run_validation.log"
source_hash_after="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
test "$source_hash_before" = "$source_hash_after"
printf 'source_checkpoint_sha256_after=%s\n' "$source_hash_after" >> "$RUN_DIR/training_manifest.txt"
printf 'STAGE3_CALIBRATED_ALIGNMENT_FORMAL_PASS seed=%s run_dir=%s\n' \
  "$SEED" "$RUN_DIR" | tee "$RUN_DIR/status.txt"
printf 'seed %s：v69 正式微调与 checkpoint 门禁完成\n' "$SEED" \
  > "$RUN_DIR/current_stage.txt"
