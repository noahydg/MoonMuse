#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PILOT_DIR="${1:?usage: run_pilot_inference.sh PILOT_DIR}"
SEED="${SEED:-42}"
N_GROUPS="${N_GROUPS:-2}"
OUT="${OUT:-$PILOT_DIR/inference-seed${SEED}}"
CKPT_DIR="$PILOT_DIR/checkpoints"
mkdir -p "$OUT"

python stage1_chord/inference.py \
  -c stage1_chord/config/emopia_finetune.yaml -r functional -m chord \
  -i "$CKPT_DIR/stage1_finetune/params/best_params.pt" \
  -o "$OUT/stage1_chord" -n "$N_GROUPS" --seed "$SEED"

python stage2_melody/inference.py \
  -c stage2_melody/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  -i "$CKPT_DIR/stage2_finetune/params/best_params.pt" \
  -id "$OUT/stage1_chord" -od "$OUT/stage2_melody" --seed "$SEED"

python representations/pkl_conn.py \
  -i "$OUT/stage2_melody" -o "$OUT/stage2_melody_conn" -m melody

python stage3_performance/inference.py \
  -c stage3_performance/config/emopia_finetune_new_gpt2.yaml -m gpt2 -r functional \
  -i "$CKPT_DIR/stage3_finetune/params/best_params.pt" \
  -id "$OUT/stage2_melody_conn" -od "$OUT/stage3_performance" --seed "$SEED"

set +e
python txt2midi.py -i "$OUT/stage3_performance" -o "$OUT/midi"
CONVERSION_RC=$?
python scripts/validate_midi.py "$OUT/midi" --output "$OUT/midi_validation.json"
VALIDATION_RC=$?
python scripts/evaluate_pilot_generation.py "$OUT" --output "$OUT/generation_gate.json"
GATE_RC=$?
set -e

printf 'PILOT_INFERENCE_COMPLETE pilot_dir=%s output=%s conversion_rc=%s validation_rc=%s gate_rc=%s\n' \
  "$PILOT_DIR" "$OUT" "$CONVERSION_RC" "$VALIDATION_RC" "$GATE_RC" \
  | tee "$OUT/status.txt"
exit "$GATE_RC"
