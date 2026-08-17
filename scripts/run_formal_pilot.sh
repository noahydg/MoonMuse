#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
PILOT_DIR="${PILOT_DIR:-generation/pilot-${RUN_ID}}"
CKPT_DIR="$PILOT_DIR/checkpoints"
LOG_DIR="$PILOT_DIR/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

run_phase() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name" | tee "$LOG_DIR/$name.log"
  { time "$@"; } 2>&1 | tee -a "$LOG_DIR/$name.log"
}

require_checkpoint() {
  local phase="$1"
  test -s "$CKPT_DIR/$phase/params/best_params.pt"
  test -s "$CKPT_DIR/$phase/optim/best_optim.pt"
}

cat > "$PILOT_DIR/pilot_manifest.txt" <<EOF
run_id=$RUN_ID
seed=$SEED
workers=$WORKERS
pretrain_epochs=$PRETRAIN_EPOCHS
finetune_epochs=$FINETUNE_EPOCHS
pilot_dir=$PILOT_DIR
optimizer_transfer=reset_for_finetuning
EOF

python scripts/preflight.py | tee "$LOG_DIR/preflight.log"
python -V 2>&1 | tee "$LOG_DIR/python.log"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
  | tee "$LOG_DIR/torch.log"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  | tee "$LOG_DIR/gpu.log"

run_phase stage1_pretrain python stage1_chord/train.py \
  -c stage1_chord/config/hooktheory_pretrain.yaml -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage1_pretrain"
require_checkpoint stage1_pretrain

run_phase stage1_finetune python stage1_chord/train.py \
  -c stage1_chord/config/emopia_finetune.yaml -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage1_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage1_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage1_finetune

run_phase stage2_pretrain python stage2_melody/train.py \
  -c stage2_melody/config/pop1k7_pretrain_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage2_pretrain"
require_checkpoint stage2_pretrain

run_phase stage2_finetune python stage2_melody/train.py \
  -c stage2_melody/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage2_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage2_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage2_finetune

run_phase stage3_pretrain python stage3_performance/train.py \
  -c stage3_performance/config/pop1k7_pretrain_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$PRETRAIN_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage3_pretrain"
require_checkpoint stage3_pretrain

run_phase stage3_finetune python stage3_performance/train.py \
  -c stage3_performance/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  --seed "$SEED" --num_workers "$WORKERS" --max_epochs "$FINETUNE_EPOCHS" \
  --ckpt_dir "$CKPT_DIR/stage3_finetune" \
  --pretrained_param_path "$CKPT_DIR/stage3_pretrain/params/best_params.pt" \
  --reset_optimizer
require_checkpoint stage3_finetune

python scripts/validate_pilot.py "$PILOT_DIR" | tee "$LOG_DIR/pilot_validation.log"
printf 'PILOT_TRAINING_COMPLETE run_id=%s pilot_dir=%s\n' "$RUN_ID" "$PILOT_DIR" \
  | tee "$PILOT_DIR/status.txt"
