#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: run_formal_generation.sh RUN_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SEED="$(sed -n 's/^seed=//p' "$RUN_DIR/training_manifest.txt")"
SEED="${SEED:-$MODEL_SEED}"
N_GROUPS="${N_GROUPS:-25}"
EXPECTED_CANDIDATES="$((N_GROUPS * 4))"
MIN_VALID_MIDI="${MIN_VALID_MIDI:-90}"
MIN_VALID_PER_QUADRANT="${MIN_VALID_PER_QUADRANT:-20}"
MIN_MEDIAN_NOTES="${MIN_MEDIAN_NOTES:-16}"
CLIP_BARS="${CLIP_BARS:-8}"
STAGE2_MAX_EVENTS="${STAGE2_MAX_EVENTS:-768}"
STAGE3_MAX_EVENTS="${STAGE3_MAX_EVENTS:-3072}"
MAX_REJECTIONS="${MAX_REJECTIONS:-512}"
MAX_SEGMENT_ATTEMPTS="${MAX_SEGMENT_ATTEMPTS:-512}"
OUT="${OUT:-$RUN_DIR/inference-seed${SEED}}"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$OUT"

cat > "$OUT/inference_manifest.txt" <<EOF
protocol=formal-full-generation-20260804
model_run_dir=$RUN_DIR
model_seed=$MODEL_SEED
inference_seed=$SEED
python_bin=$PYTHON_BIN
groups=$N_GROUPS
expected_candidates=$EXPECTED_CANDIDATES
min_valid_midi=$MIN_VALID_MIDI
min_valid_per_quadrant=$MIN_VALID_PER_QUADRANT
min_median_notes=$MIN_MEDIAN_NOTES
clip_bars=$CLIP_BARS
stage2_max_events=$STAGE2_MAX_EVENTS
stage3_max_events=$STAGE3_MAX_EVENTS
max_rejections_per_candidate=$MAX_REJECTIONS
max_sampling_attempts_per_segment=$MAX_SEGMENT_ATTEMPTS
stage1_mode=chord
stage1_temperature=1.1
stage1_top_p=0.97
stage1_max_dec_len=128
stage2_backbone=gpt2
stage2_temperature=1.0
stage2_top_p=0.97
stage3_backbone=gpt2
stage3_temperature=1.0
stage3_top_p=0.97
descriptor_population=validation_passed_midi_only
EOF

sha256sum \
  scripts/run_formal_generation.sh scripts/validate_midi.py \
  scripts/evaluate_pilot_generation.py scripts/evaluate_midi_descriptors.py \
  stage1_chord/inference.py stage2_melody/inference.py \
  stage3_performance/inference.py representations/pkl_conn.py txt2midi.py \
  "$CKPT_DIR/stage1_finetune/params/best_params.pt" \
  "$CKPT_DIR/stage2_finetune/params/best_params.pt" \
  "$CKPT_DIR/stage3_finetune/params/best_params.pt" \
  > "$OUT/inference_code_and_model_sha256.txt"

"$PYTHON_BIN" stage1_chord/inference.py \
  -c stage1_chord/config/emopia_finetune.yaml -r functional -m chord \
  -i "$CKPT_DIR/stage1_finetune/params/best_params.pt" \
  -o "$OUT/stage1_chord" -n "$N_GROUPS" --seed "$SEED" \
  2>&1 | tee "$OUT/stage1_generation.log"

"$PYTHON_BIN" stage2_melody/inference.py \
  -c stage2_melody/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  -i "$CKPT_DIR/stage2_finetune/params/best_params.pt" \
  -id "$OUT/stage1_chord" -od "$OUT/stage2_melody" --seed "$SEED" \
  --max_bars "$CLIP_BARS" --max_events "$STAGE2_MAX_EVENTS" \
  --max_rejections "$MAX_REJECTIONS" \
  --max_segment_attempts "$MAX_SEGMENT_ATTEMPTS" \
  2>&1 | tee "$OUT/stage2_generation.log"

"$PYTHON_BIN" representations/pkl_conn.py \
  -i "$OUT/stage2_melody" -o "$OUT/stage2_melody_conn" -m melody \
  2>&1 | tee "$OUT/stage2_connection.log"

"$PYTHON_BIN" stage3_performance/inference.py \
  -c stage3_performance/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  -i "$CKPT_DIR/stage3_finetune/params/best_params.pt" \
  -id "$OUT/stage2_melody_conn" -od "$OUT/stage3_performance" --seed "$SEED" \
  --max_bars "$CLIP_BARS" --max_events "$STAGE3_MAX_EVENTS" \
  --max_rejections "$MAX_REJECTIONS" \
  --max_segment_attempts "$MAX_SEGMENT_ATTEMPTS" \
  2>&1 | tee "$OUT/stage3_generation.log"

set +e
"$PYTHON_BIN" txt2midi.py -i "$OUT/stage3_performance" -o "$OUT/midi" \
  --max_bars "$CLIP_BARS" 2>&1 | tee "$OUT/conversion.log"
CONVERSION_RC=$?
"$PYTHON_BIN" scripts/validate_midi.py "$OUT/midi" --output "$OUT/midi_validation.json" \
  2>&1 | tee "$OUT/midi_validation.log"
VALIDATION_RC=$?
"$PYTHON_BIN" scripts/evaluate_pilot_generation.py "$OUT" \
  --expected-candidates "$EXPECTED_CANDIDATES" \
  --min-valid-midi "$MIN_VALID_MIDI" \
  --min-valid-per-quadrant "$MIN_VALID_PER_QUADRANT" \
  --min-median-notes "$MIN_MEDIAN_NOTES" \
  --output "$OUT/generation_gate.json" \
  2>&1 | tee "$OUT/generation_gate.log"
GATE_RC=$?
"$PYTHON_BIN" scripts/evaluate_midi_descriptors.py "$OUT/midi" \
  --output-csv "$OUT/midi_descriptors.csv" \
  --output-json "$OUT/midi_descriptors.json" \
  2>&1 | tee "$OUT/midi_descriptors.log"
DESCRIPTOR_RC=$?
set -e

FINAL_RC="$GATE_RC"
if [[ "$DESCRIPTOR_RC" -ne 0 ]]; then
  FINAL_RC="$DESCRIPTOR_RC"
fi
printf 'FORMAL_GENERATION_COMPLETE run_dir=%s output=%s conversion_rc=%s validation_rc=%s gate_rc=%s descriptor_rc=%s\n' \
  "$RUN_DIR" "$OUT" "$CONVERSION_RC" "$VALIDATION_RC" "$GATE_RC" "$DESCRIPTOR_RC" \
  | tee "$OUT/status.txt"
exit "$FINAL_RC"
