#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE2_SOURCE="${1:?usage: run_training_pad_loss_smoke.sh STAGE2_SOURCE_CHECKPOINT STAGE3_SOURCE_CHECKPOINT}"
STAGE3_SOURCE="${2:?usage: run_training_pad_loss_smoke.sh STAGE2_SOURCE_CHECKPOINT STAGE3_SOURCE_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
STEPS="${STEPS:-100}"
WORKERS="${WORKERS:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-generation/training-pad-loss-smoke-${RUN_ID}-v73}"
LOG_ROOT="$RUN_ROOT/logs"
STAGE2_ROOT="$RUN_ROOT/checkpoints/stage2"
STAGE3_ROOT="$RUN_ROOT/checkpoints/stage3"
STAGE2_CONFIG=stage2_melody/config/emopia_finetune_new_gpt2.yaml
STAGE3_CONFIG=stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml

test -s "$STAGE2_SOURCE"
test -s "$STAGE3_SOURCE"
test ! -e "$RUN_ROOT"
mkdir -p "$LOG_ROOT"
stage2_hash_before="$(sha256sum "$STAGE2_SOURCE" | awk '{print $1}')"
stage3_hash_before="$(sha256sum "$STAGE3_SOURCE" | awk '{print $1}')"
cat > "$RUN_ROOT/smoke_manifest.txt" <<EOF
protocol=moonmuse-training-pad-loss-correction-smoke-v1
formal_study_data=false
training_performed=true
seed=$SEED
steps_per_stage=$STEPS
stage2_source_checkpoint=$STAGE2_SOURCE
stage2_source_sha256_before=$stage2_hash_before
stage3_source_checkpoint=$STAGE3_SOURCE
stage3_source_sha256_before=$stage3_hash_before
stage2_config=$STAGE2_CONFIG
stage3_config=$STAGE3_CONFIG
optimizer_reset=true
EOF

"$PYTHON_BIN" scripts/validate_music_gpt2_pad_loss.py \
  --output "$RUN_ROOT/pad_loss_tensor_validation.json" \
  2>&1 | tee "$LOG_ROOT/pad_loss_tensor_validation.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py "$STAGE2_SOURCE" \
  --output "$RUN_ROOT/stage2_source_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/stage2_source_checkpoint_validation.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py "$STAGE3_SOURCE" \
  --output "$RUN_ROOT/stage3_source_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/stage3_source_checkpoint_validation.log"

"$PYTHON_BIN" stage2_melody/train.py \
  -c "$STAGE2_CONFIG" -m gpt2 -r functional --seed "$SEED" \
  --max_steps "$STEPS" --num_workers "$WORKERS" \
  --ckpt_dir "$STAGE2_ROOT" --pretrained_param_path "$STAGE2_SOURCE" \
  --reset_optimizer 2>&1 | tee "$LOG_ROOT/stage2_training.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$STAGE2_ROOT/params/smoke_params.pt" \
  --optimizer "$STAGE2_ROOT/optim/smoke_optim.pt" \
  --reference "$STAGE2_SOURCE" \
  --output "$RUN_ROOT/stage2_smoke_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/stage2_smoke_checkpoint_validation.log"

"$PYTHON_BIN" stage3_performance/train.py \
  -c "$STAGE3_CONFIG" -m gpt2 -r functional --seed "$SEED" \
  --max_steps "$STEPS" --num_workers "$WORKERS" \
  --ckpt_dir "$STAGE3_ROOT" --pretrained_param_path "$STAGE3_SOURCE" \
  --reset_optimizer 2>&1 | tee "$LOG_ROOT/stage3_training.log"
"$PYTHON_BIN" scripts/validate_checkpoint.py \
  "$STAGE3_ROOT/params/smoke_params.pt" \
  --optimizer "$STAGE3_ROOT/optim/smoke_optim.pt" \
  --reference "$STAGE3_SOURCE" \
  --output "$RUN_ROOT/stage3_smoke_checkpoint_integrity.json" \
  2>&1 | tee "$LOG_ROOT/stage3_smoke_checkpoint_validation.log"

stage2_hash_after="$(sha256sum "$STAGE2_SOURCE" | awk '{print $1}')"
stage3_hash_after="$(sha256sum "$STAGE3_SOURCE" | awk '{print $1}')"
test "$stage2_hash_before" = "$stage2_hash_after"
test "$stage3_hash_before" = "$stage3_hash_after"
cat >> "$RUN_ROOT/smoke_manifest.txt" <<EOF
stage2_source_sha256_after=$stage2_hash_after
stage3_source_sha256_after=$stage3_hash_after
EOF

"$PYTHON_BIN" scripts/validate_training_pad_loss_smoke.py "$RUN_ROOT" \
  --expected-steps "$STEPS" --output "$RUN_ROOT/execution_validation.json" \
  2>&1 | tee "$LOG_ROOT/execution_validation.log"
sha256sum \
  stage2_melody/model/music_gpt2.py stage3_performance/model/music_gpt2.py \
  stage2_melody/train.py stage3_performance/train.py \
  "$STAGE2_CONFIG" "$STAGE3_CONFIG" \
  scripts/validate_music_gpt2_pad_loss.py \
  scripts/validate_training_pad_loss_smoke.py \
  scripts/run_training_pad_loss_smoke.sh \
  "$STAGE2_SOURCE" "$STAGE3_SOURCE" \
  > "$RUN_ROOT/input_and_code_sha256.txt"
printf 'TRAINING_PAD_LOSS_SMOKE_PASS run_root=%s steps_per_stage=%s\n' \
  "$RUN_ROOT" "$STEPS" | tee "$RUN_ROOT/status.txt"
tar -czf "$RUN_ROOT.tar.gz" "$RUN_ROOT"
sha256sum "$RUN_ROOT.tar.gz" > "$RUN_ROOT.tar.gz.sha256"
