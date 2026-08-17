#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEGACY_CHECKPOINT="${1:?usage: run_stage3_arousal_alignment_smoke.sh LEGACY_STAGE3_FINETUNE_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
STEPS="${STEPS:-100}"
WORKERS="${WORKERS:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-generation/stage3-arousal-alignment-smoke-${RUN_ID}}"
LEGACY_DATA_ROOT="events/m2p/emopia_m2p_functional"
FACTORIZED_DATA_ROOT="events/m2p/emopia_m2p_factorized_va_functional"
MIGRATED_CHECKPOINT="$RUN_ROOT/migrated_stage3_finetune_params.pt"
TRAIN_ROOT="$RUN_ROOT/checkpoints/stage3_arousal_alignment_smoke"
LOG_ROOT="$RUN_ROOT/logs"

test -s "$LEGACY_CHECKPOINT"
mkdir -p "$LOG_ROOT"
printf 'python scripts/show_stage3_arousal_alignment_progress.py %q --expected-batches %q --watch 5\n' \
  "$RUN_ROOT" "$STEPS" > "$RUN_ROOT/dashboard_command.txt"

printf '审计 Stage3 唤醒度标签和真实表现信号\n' > "$RUN_ROOT/current_stage.txt"
"$PYTHON_BIN" scripts/audit_stage3_arousal_supervision.py \
  "$FACTORIZED_DATA_ROOT" --output "$RUN_ROOT/arousal_supervision_audit.json" \
  2>&1 | tee "$LOG_ROOT/arousal_supervision_audit.log"

printf '迁移匹配的 Stage3 检查点（优化器重置）\n' > "$RUN_ROOT/current_stage.txt"
"$PYTHON_BIN" scripts/migrate_stage3_factorized_checkpoint.py \
  --checkpoint "$LEGACY_CHECKPOINT" \
  --legacy-dictionary "$LEGACY_DATA_ROOT/dictionary.pkl" \
  --factorized-dictionary "$FACTORIZED_DATA_ROOT/dictionary.pkl" \
  --output "$MIGRATED_CHECKPOINT" \
  2>&1 | tee "$LOG_ROOT/checkpoint_migration.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py "$MIGRATED_CHECKPOINT" \
  --output "$RUN_ROOT/migrated_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/migrated_checkpoint_validation.log"

printf 'Stage3 反事实唤醒度对齐 smoke 训练（%s batches）\n' "$STEPS" \
  > "$RUN_ROOT/current_stage.txt"
"$PYTHON_BIN" stage3_performance/train.py \
  -c stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml \
  -m gpt2 -r functional \
  --seed "$SEED" --max_steps "$STEPS" --num_workers "$WORKERS" \
  --ckpt_dir "$TRAIN_ROOT" \
  --pretrained_param_path "$MIGRATED_CHECKPOINT" \
  --reset_optimizer \
  2>&1 | tee "$LOG_ROOT/training.log"

"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$TRAIN_ROOT/params/smoke_params.pt" \
  --optimizer "$TRAIN_ROOT/optim/smoke_optim.pt" \
  --reference "$MIGRATED_CHECKPOINT" \
  --output "$RUN_ROOT/smoke_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/smoke_checkpoint_validation.log"

"$PYTHON_BIN" scripts/validate_stage3_arousal_alignment_smoke.py \
  "$RUN_ROOT" --expected-steps "$STEPS" \
  --output "$RUN_ROOT/smoke_validation.json" \
  2>&1 | tee "$LOG_ROOT/smoke_validation.log"

printf 'STAGE3_AROUSAL_ALIGNMENT_SMOKE_PASS run_root=%s seed=%s steps=%s\n' \
  "$RUN_ROOT" "$SEED" "$STEPS" | tee "$RUN_ROOT/status.txt"
printf '已完成：新目标可训练；等待正式训练决策，不代表音乐质量已提升\n' \
  > "$RUN_ROOT/current_stage.txt"
