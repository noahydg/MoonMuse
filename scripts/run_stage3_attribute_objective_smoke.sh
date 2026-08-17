#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_CHECKPOINT="${1:?usage: run_stage3_attribute_objective_smoke.sh FACTORIZED_ALIGNMENT_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
STEPS="${STEPS:-100}"
WORKERS="${WORKERS:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-generation/stage3-attribute-objective-smoke-${RUN_ID}}"
TRAIN_ROOT="$RUN_ROOT/checkpoints/stage3_attribute_objective_smoke"
LOG_ROOT="$RUN_ROOT/logs"
CONFIG=stage3_performance/config/emopia_finetune_attribute_only_alignment_gpt2.yaml

test -s "$SOURCE_CHECKPOINT"
test ! -e "$RUN_ROOT"
mkdir -p "$LOG_ROOT"
cat > "$RUN_ROOT/smoke_manifest.txt" <<EOF
protocol=stage3-performance-attribute-only-objective-smoke-v1
formal_study_data=false
training_performed=true
source_checkpoint=$SOURCE_CHECKPOINT
seed=$SEED
steps=$STEPS
configuration=$CONFIG
optimizer_reset=true
EOF

"$PYTHON_BIN" scripts/validate_checkpoint.py "$SOURCE_CHECKPOINT" \
  --output "$RUN_ROOT/source_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/source_checkpoint_validation.log"

printf 'Stage 3 属性专用重构 + 唤醒度对齐 smoke（%s batches）\n' "$STEPS" \
  > "$RUN_ROOT/current_stage.txt"
"$PYTHON_BIN" stage3_performance/train.py \
  -c "$CONFIG" -m gpt2 -r functional \
  --seed "$SEED" --max_steps "$STEPS" --num_workers "$WORKERS" \
  --ckpt_dir "$TRAIN_ROOT" \
  --pretrained_param_path "$SOURCE_CHECKPOINT" --reset_optimizer \
  2>&1 | tee "$LOG_ROOT/training.log"

"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$TRAIN_ROOT/params/smoke_params.pt" \
  --optimizer "$TRAIN_ROOT/optim/smoke_optim.pt" \
  --reference "$SOURCE_CHECKPOINT" \
  --output "$RUN_ROOT/smoke_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/smoke_checkpoint_validation.log"
"$PYTHON_BIN" scripts/validate_stage3_attribute_objective_smoke.py \
  "$RUN_ROOT" --expected-steps "$STEPS" \
  --output "$RUN_ROOT/smoke_validation.json" \
  2>&1 | tee "$LOG_ROOT/smoke_validation.log"

sha256sum stage3_attribute_objective.py stage3_performance/train.py \
  "$CONFIG" scripts/run_stage3_attribute_objective_smoke.sh \
  scripts/validate_stage3_attribute_objective_smoke.py "$SOURCE_CHECKPOINT" \
  > "$RUN_ROOT/input_and_code_sha256.txt"
printf 'STAGE3_ATTRIBUTE_OBJECTIVE_SMOKE_PASS run_root=%s seed=%s steps=%s\n' \
  "$RUN_ROOT" "$SEED" "$STEPS" | tee "$RUN_ROOT/status.txt"
printf '已完成：新训练目标可运行；尚未证明生成质量或情感控制提升\n' \
  > "$RUN_ROOT/current_stage.txt"
