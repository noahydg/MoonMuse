#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:?set SEED to 42, 43, or 44}"
case "$SEED" in 42|43|44) ;; *) printf 'unsupported seed: %s\n' "$SEED" >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-generation/stage3-arousal-alignment-formal-seed${SEED}-${RUN_ID}}"
LEGACY_RUN_DIR="${LEGACY_RUN_DIR:?set LEGACY_RUN_DIR to the matching formal Full run}"
LEGACY_CHECKPOINT="$LEGACY_RUN_DIR/checkpoints/stage3_finetune/params/best_params.pt"
LEGACY_DATA_ROOT="events/m2p/emopia_m2p_functional"
FACTORIZED_DATA_ROOT="events/m2p/emopia_m2p_factorized_va_functional"
MIGRATED_CHECKPOINT="$RUN_DIR/migrated_stage3_finetune_params.pt"
CHECKPOINT_ROOT="$RUN_DIR/checkpoints/stage3_arousal_alignment_finetune"
LOG_ROOT="$RUN_DIR/logs"

test -s "$LEGACY_CHECKPOINT"
mkdir -p "$LOG_ROOT"
cat > "$RUN_DIR/training_manifest.txt" <<EOF
protocol=stage3-arousal-alignment-formal-v1
run_id=$RUN_ID
seed=$SEED
workers=$WORKERS
max_epochs=$MAX_EPOCHS
early_stopping_patience=$PATIENCE
run_dir=$RUN_DIR
legacy_run_dir=$LEGACY_RUN_DIR
legacy_checkpoint=$LEGACY_CHECKPOINT
conditioning=factorized_va
selection=lowest_total_validation_objective
optimizer_transfer=reset_after_checkpoint_migration
objective=token_cross_entropy_plus_paired_counterfactual_directional_hinge_v1
alignment_weight=0.20
EOF

sha256sum stage3_arousal_alignment.py stage3_performance/train.py \
  stage3_performance/dataloader.py \
  stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml \
  scripts/audit_stage3_arousal_supervision.py \
  scripts/migrate_stage3_factorized_checkpoint.py \
  scripts/validate_stage3_arousal_alignment_formal_run.py \
  scripts/run_stage3_arousal_alignment_formal_seed.sh \
  "$LEGACY_CHECKPOINT" > "$RUN_DIR/training_code_and_source_sha256.txt"

printf 'seed %s：审计监督信号\n' "$SEED" > "$RUN_DIR/current_stage.txt"
"$PYTHON_BIN" scripts/audit_stage3_arousal_supervision.py \
  "$FACTORIZED_DATA_ROOT" --output "$RUN_DIR/arousal_supervision_audit.json" \
  2>&1 | tee "$LOG_ROOT/arousal_supervision_audit.log"

printf 'seed %s：迁移匹配的旧 Stage3 checkpoint\n' "$SEED" > "$RUN_DIR/current_stage.txt"
"$PYTHON_BIN" scripts/migrate_stage3_factorized_checkpoint.py \
  --checkpoint "$LEGACY_CHECKPOINT" \
  --legacy-dictionary "$LEGACY_DATA_ROOT/dictionary.pkl" \
  --factorized-dictionary "$FACTORIZED_DATA_ROOT/dictionary.pkl" \
  --output "$MIGRATED_CHECKPOINT" \
  2>&1 | tee "$LOG_ROOT/checkpoint_migration.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py "$MIGRATED_CHECKPOINT" \
  --output "$RUN_DIR/migrated_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/migrated_checkpoint_validation.log"

printf 'seed %s：唤醒度对齐正式微调（最多 %s epochs）\n' "$SEED" "$MAX_EPOCHS" \
  > "$RUN_DIR/current_stage.txt"
{ time "$PYTHON_BIN" stage3_performance/train.py \
  -c stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml \
  -m gpt2 -r functional --seed "$SEED" --num_workers "$WORKERS" \
  --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
  --ckpt_dir "$CHECKPOINT_ROOT" \
  --pretrained_param_path "$MIGRATED_CHECKPOINT" --reset_optimizer; } \
  2>&1 | tee "$LOG_ROOT/training.log"

test -s "$CHECKPOINT_ROOT/params/best_params.pt"
test -s "$CHECKPOINT_ROOT/optim/best_optim.pt"
"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$CHECKPOINT_ROOT/params/best_params.pt" \
  --optimizer "$CHECKPOINT_ROOT/optim/best_optim.pt" \
  --reference "$MIGRATED_CHECKPOINT" \
  --output "$RUN_DIR/formal_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/formal_checkpoint_validation.log"
"$PYTHON_BIN" scripts/validate_stage3_arousal_alignment_formal_run.py \
  "$RUN_DIR" --seed "$SEED" --allow-missing-status \
  --output "$RUN_DIR/formal_run_validation.json" \
  2>&1 | tee "$LOG_ROOT/formal_run_validation.log"
printf 'STAGE3_AROUSAL_ALIGNMENT_FORMAL_PASS seed=%s run_dir=%s\n' "$SEED" "$RUN_DIR" \
  | tee "$RUN_DIR/status.txt"
printf 'seed %s：正式微调与 checkpoint 门禁完成\n' "$SEED" \
  > "$RUN_DIR/current_stage.txt"
