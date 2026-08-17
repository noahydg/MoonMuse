#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_FILE="${1:?usage: run_stage3_skeleton_lock_smoke.sh SOURCE_FILE CHECKPOINT OUTPUT_DIR CONDITION_A CONDITION_B}"
CHECKPOINT="${2:?usage: run_stage3_skeleton_lock_smoke.sh SOURCE_FILE CHECKPOINT OUTPUT_DIR CONDITION_A CONDITION_B}"
OUTPUT_DIR="${3:?usage: run_stage3_skeleton_lock_smoke.sh SOURCE_FILE CHECKPOINT OUTPUT_DIR CONDITION_A CONDITION_B}"
CONDITION_A="${4:?usage: run_stage3_skeleton_lock_smoke.sh SOURCE_FILE CHECKPOINT OUTPUT_DIR CONDITION_A CONDITION_B}"
CONDITION_B="${5:?usage: run_stage3_skeleton_lock_smoke.sh SOURCE_FILE CHECKPOINT OUTPUT_DIR CONDITION_A CONDITION_B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE3_CONFIG="${STAGE3_CONFIG:-stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml}"
SEED="${SEED:-42}"

test -s "$SOURCE_FILE"
test -s "$CHECKPOINT"
test -s "$STAGE3_CONFIG"
case "$CONDITION_A,$CONDITION_B" in
  Q1,Q4|Q4,Q1|Q2,Q3|Q3,Q2) ;;
  *) printf 'conditions must be Q1/Q4 or Q2/Q3\n' >&2; exit 2 ;;
esac
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/source_input"
cp "$SOURCE_FILE" "$OUTPUT_DIR/source_input/"

cat > "$OUTPUT_DIR/smoke_manifest.txt" <<EOF
protocol=stage3-condition-skeleton-lock-smoke-v1
formal_study_data=false
training_performed=false
source_file=$SOURCE_FILE
checkpoint=$CHECKPOINT
configuration=$STAGE3_CONFIG
seed=$SEED
conditions=$CONDITION_A,$CONDITION_B
lock_performance_to_condition_skeleton=1
EOF

"$PYTHON_BIN" representations/pkl_conn.py \
  -i "$OUTPUT_DIR/source_input" \
  -o "$OUTPUT_DIR/stage2_melody_conn" -m melody \
  2>&1 | tee "$OUTPUT_DIR/stage2_connection.log"

"$PYTHON_BIN" stage3_performance/inference.py \
  -c "$STAGE3_CONFIG" -m gpt2 -r functional -i "$CHECKPOINT" \
  -id "$OUTPUT_DIR/stage2_melody_conn" \
  -od "$OUTPUT_DIR/stage3_performance" \
  --seed "$SEED" --emotion-override "$CONDITION_A" "$CONDITION_B" \
  --conditioning-mode factorized_va --paired-condition-seed \
  --condition-track Melody --max_bars 8 --max_events 3072 \
  --temperature 1.0 --top-p 0.97 \
  --lock-performance-to-condition-skeleton \
  2>&1 | tee "$OUTPUT_DIR/stage3_generation.log"

"$PYTHON_BIN" txt2midi.py \
  -i "$OUTPUT_DIR/stage3_performance" -o "$OUTPUT_DIR/midi" \
  --max_bars 8 --condition-track Melody \
  2>&1 | tee "$OUTPUT_DIR/conversion.log"

"$PYTHON_BIN" scripts/validate_midi.py "$OUTPUT_DIR/midi" \
  --output "$OUTPUT_DIR/midi_validation.json" \
  2>&1 | tee "$OUTPUT_DIR/midi_validation.log"

"$PYTHON_BIN" scripts/validate_stage3_skeleton_lock.py "$OUTPUT_DIR" \
  --expected-files 2 --output "$OUTPUT_DIR/skeleton_validation.json" \
  2>&1 | tee "$OUTPUT_DIR/skeleton_validation.log"

sha256sum \
  "$SOURCE_FILE" "$CHECKPOINT" "$STAGE3_CONFIG" \
  stage3_performance/inference.py performance_skeleton.py \
  scripts/run_stage3_skeleton_lock_smoke.sh \
  > "$OUTPUT_DIR/input_and_code_sha256.txt"
printf 'STAGE3_SKELETON_LOCK_SMOKE_PASS output=%s\n' "$OUTPUT_DIR" \
  | tee "$OUTPUT_DIR/status.txt"
