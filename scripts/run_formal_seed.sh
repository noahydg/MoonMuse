#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:?set SEED to one of 42, 43, or 44}"
WORKERS="${WORKERS:-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-generation/formal-full-seed${SEED}-${RUN_ID}}"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

run_phase() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name" | tee "$LOG_DIR/$name.log"
  { time "$@"; } 2>&1 | tee -a "$LOG_DIR/$name.log"
}

require_checkpoint() {
  local phase="$1"
  local reference="${2:-}"
  test -s "$CKPT_DIR/$phase/params/best_params.pt"
  test -s "$CKPT_DIR/$phase/optim/best_optim.pt"
  local args=(
    "$CKPT_DIR/$phase/params/best_params.pt"
    --optimizer "$CKPT_DIR/$phase/optim/best_optim.pt"
    --output "$CKPT_DIR/$phase/checkpoint_integrity.json"
  )
  if [[ -n "$reference" ]]; then
    args+=(--reference "$reference")
  fi
  python scripts/validate_checkpoint.py "${args[@]}" \
    | tee "$LOG_DIR/${phase}_checkpoint_integrity.log"
}

cat > "$RUN_DIR/training_manifest.txt" <<EOF
protocol=formal-full-20260804
run_id=$RUN_ID
seed=$SEED
workers=$WORKERS
pretrain_max_epochs=$PRETRAIN_EPOCHS
finetune_max_epochs=$FINETUNE_EPOCHS
early_stopping_patience=$PATIENCE
run_dir=$RUN_DIR
selection=lowest_validation_loss
optimizer_transfer=reset_for_finetuning
objective=token_cross_entropy
EOF

sha256sum \
  scripts/run_formal_seed.sh scripts/preflight.py \
  scripts/validate_checkpoint.py scripts/validate_pilot.py \
  stage1_chord/train.py stage2_melody/train.py stage3_performance/train.py \
  stage1_chord/config/hooktheory_pretrain.yaml \
  stage1_chord/config/emopia_finetune.yaml \
  stage2_melody/config/pop1k7_pretrain_new_gpt2.yaml \
  stage2_melody/config/emopia_finetune_new_gpt2.yaml \
  stage3_performance/config/pop1k7_pretrain_new_gpt2.yaml \
  stage3_performance/config/emopia_finetune_new_gpt2.yaml \
  > "$RUN_DIR/training_code_sha256.txt"

python scripts/preflight.py | tee "$LOG_DIR/preflight.log"
python -V 2>&1 | tee "$LOG_DIR/python.log"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
  | tee "$LOG_DIR/torch.log"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee "$LOG_DIR/gpu.log"

run_phase stage1_pretrain python stage1_chord/train.py \
  -c stage1_chord/config/hooktheory_pretrain.yaml -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage1_pretrain"
require_checkpoint stage1_pretrain

run_phase stage1_finetune python stage1_chord/train.py \
  -c stage1_chord/config/emopia_finetune.yaml -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage1_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage1_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage1_finetune \
  "$CKPT_DIR/stage1_pretrain/params/best_params.pt"

run_phase stage2_pretrain python stage2_melody/train.py \
  -c stage2_melody/config/pop1k7_pretrain_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage2_pretrain"
require_checkpoint stage2_pretrain

run_phase stage2_finetune python stage2_melody/train.py \
  -c stage2_melody/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage2_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage2_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage2_finetune \
  "$CKPT_DIR/stage2_pretrain/params/best_params.pt"

run_phase stage3_pretrain python stage3_performance/train.py \
  -c stage3_performance/config/pop1k7_pretrain_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage3_pretrain"
require_checkpoint stage3_pretrain

run_phase stage3_finetune python stage3_performance/train.py \
  -c stage3_performance/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --patience "$PATIENCE" --ckpt_dir "$CKPT_DIR/stage3_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage3_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage3_finetune \
  "$CKPT_DIR/stage3_pretrain/params/best_params.pt"

python scripts/validate_pilot.py "$RUN_DIR" \
  --require-checkpoint-integrity \
  --output "$RUN_DIR/training_validation.json" \
  | tee "$LOG_DIR/training_validation.log"
printf 'FORMAL_FULL_TRAINING_COMPLETE seed=%s run_dir=%s\n' "$SEED" "$RUN_DIR" \
  | tee "$RUN_DIR/status.txt"
