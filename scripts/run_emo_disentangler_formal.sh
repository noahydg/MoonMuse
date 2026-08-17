#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:?set BASELINE_ROOT to the EMO-Disentangler repository}"
MOONMUSE_ROOT="${MOONMUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASELINE_ROOT/generation/formal-author-checkpoint-${CAMPAIGN_ID}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$BASELINE_ROOT/formal-artifacts/formal-author-checkpoint-${CAMPAIGN_ID}}"
SEEDS="${SEEDS:-42 43 44}"
N_GROUPS="${N_GROUPS:-25}"
EXPECTED_CANDIDATES="$((N_GROUPS * 4))"

cd "$BASELINE_ROOT"
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$ARTIFACT_ROOT"

cat > "$OUTPUT_ROOT/campaign_manifest.txt" <<EOF
protocol=emo-disentangler-author-checkpoint-campaign-v2
campaign_id=$CAMPAIGN_ID
seeds=$SEEDS
n_groups=$N_GROUPS
expected_candidates_per_seed=$EXPECTED_CANDIDATES
python_bin=$PYTHON_BIN
output_root=$OUTPUT_ROOT
artifact_root=$ARTIFACT_ROOT
execution=sequential_stop_on_failure
EOF

sha256sum \
  "$MOONMUSE_ROOT/scripts/run_emo_disentangler_formal.sh" \
  "$MOONMUSE_ROOT/scripts/package_baseline_run.sh" \
  "$MOONMUSE_ROOT/scripts/aggregate_generation_runs.py" \
  "$MOONMUSE_ROOT/scripts/evaluate_control_proxies.py" \
  > "$OUTPUT_ROOT/campaign_code_sha256.txt"

run_dirs=()
for seed in $SEEDS; do
  out="$OUTPUT_ROOT/seed${seed}"
  mkdir -p "$out/logs"

  cat > "$out/inference_manifest.txt" <<EOF
protocol=emo-disentangler-author-checkpoint-20260804
seed=$seed
n_groups=$N_GROUPS
expected_candidates=$EXPECTED_CANDIDATES
stage1_temperature=1.2
stage1_top_p=0.97
stage2_backbone=gpt2
stage2_temperature=1.2
stage2_top_p=0.97
descriptor_population=validation_passed_midi_only
campaign_id=$CAMPAIGN_ID
python_bin=$PYTHON_BIN
EOF

  sha256sum \
    "$MOONMUSE_ROOT/scripts/run_emo_disentangler_formal.sh" \
    "$MOONMUSE_ROOT/scripts/validate_midi.py" \
    "$MOONMUSE_ROOT/scripts/evaluate_baseline_generation.py" \
    "$MOONMUSE_ROOT/scripts/evaluate_midi_descriptors.py" \
    stage1_compose/inference.py \
    stage1_compose/config/emopia_finetune.yaml \
    stage2_accompaniment/inference.py \
    stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
    best_weight/Functional-two/emopia_lead_sheet_finetune/ep016_loss0.685_params.pt \
    best_weight/Functional-two/emopia_acccompaniment_finetune_gpt2/ep300_loss0.120_params.pt \
    > "$out/code_and_model_sha256.txt"

  "$PYTHON_BIN" -V 2>&1 | tee "$out/logs/python.log"
  "$PYTHON_BIN" -c \
    'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
    | tee "$out/logs/torch.log"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | tee "$out/logs/gpu.log"

  "$PYTHON_BIN" stage1_compose/inference.py \
    --configuration stage1_compose/config/emopia_finetune.yaml \
    --representation functional \
    --mode lead_sheet \
    --inference_params best_weight/Functional-two/emopia_lead_sheet_finetune/ep016_loss0.685_params.pt \
    --output_dir "$out" \
    --n_groups "$N_GROUPS" \
    --seed "$seed" \
    | tee "$out/logs/stage1.log"

  "$PYTHON_BIN" stage2_accompaniment/inference.py \
    --model_type gpt2 \
    --configuration stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
    --representation functional \
    --inference_params best_weight/Functional-two/emopia_acccompaniment_finetune_gpt2/ep300_loss0.120_params.pt \
    --output_dir "$out" \
    --seed "$seed" \
    | tee "$out/logs/stage2.log"

  "$PYTHON_BIN" "$MOONMUSE_ROOT/scripts/validate_midi.py" "$out" \
    --output "$out/midi_validation.json" \
    | tee "$out/logs/midi_validation.log"
  "$PYTHON_BIN" "$MOONMUSE_ROOT/scripts/evaluate_baseline_generation.py" "$out" \
    --expected-candidates "$EXPECTED_CANDIDATES" \
    --min-valid-midi 90 \
    --min-valid-per-quadrant 20 \
    --min-median-notes 16 \
    --output "$out/generation_gate.json" \
    | tee "$out/logs/generation_gate.log"
  "$PYTHON_BIN" "$MOONMUSE_ROOT/scripts/evaluate_midi_descriptors.py" "$out" \
    --pattern 'samp_*_Q[1-4]_full.mid' \
    --output-csv "$out/midi_descriptors.csv" \
    --output-json "$out/midi_descriptors.json" \
    | tee "$out/logs/midi_descriptors.log"
  printf 'BASELINE_FORMAL_COMPLETE seed=%s output=%s\n' "$seed" "$out" \
    | tee "$out/status.txt"
  bash "$MOONMUSE_ROOT/scripts/package_baseline_run.sh" "$out" "$ARTIFACT_ROOT"
  run_dirs+=("$out")
done

"$PYTHON_BIN" "$MOONMUSE_ROOT/scripts/aggregate_generation_runs.py" \
  --model-name EMO-Disentangler-author-checkpoint \
  --output-dir "$OUTPUT_ROOT/summary" \
  "${run_dirs[@]}"
"$PYTHON_BIN" "$MOONMUSE_ROOT/scripts/evaluate_control_proxies.py" \
  "$OUTPUT_ROOT/summary/descriptors_by_seed.csv" \
  --output-dir "$OUTPUT_ROOT/summary/control-proxies"

printf 'BASELINE_FORMAL_CAMPAIGN_COMPLETE campaign_id=%s output=%s summary=%s artifacts=%s\n' \
  "$CAMPAIGN_ID" "$OUTPUT_ROOT" "$OUTPUT_ROOT/summary" "$ARTIFACT_ROOT"
