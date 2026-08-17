#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_ROOT="${1:?usage: run_stage3_attribute_objective_seed_pilot.sh FROZEN_SOURCE_ROOT FORMAL_RUN_DIR}"
FORMAL_RUN_DIR="${2:?usage: run_stage3_attribute_objective_seed_pilot.sh FROZEN_SOURCE_ROOT FORMAL_RUN_DIR}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-generation/stage3-attribute-objective-seed${SEED}-pilot-${RUN_ID}}"
CONFIG=stage3_performance/config/emopia_finetune_attribute_only_alignment_gpt2.yaml
CHECKPOINT="$FORMAL_RUN_DIR/checkpoints/stage3_attribute_objective_finetune/params/best_params.pt"

test "$SEED" -eq 42
test -s "$SOURCE_ROOT/source_manifest.json"
test -s "$CHECKPOINT"
test ! -e "$OUT_ROOT"
mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/filter_stage3_source_manifest.py \
  "$SOURCE_ROOT/source_manifest.json" --seed "$SEED" \
  --output "$OUT_ROOT/source_manifest.json"
printf 'seed\tvalence\tstage3_rc\tconversion_rc\tvalidation_rc\tquality_rc\tdescriptor_rc\n' > "$OUT_ROOT/status.tsv"
execution_failure=0
for valence in positive negative; do
  case "$valence" in positive) conditions=(Q1 Q4) ;; negative) conditions=(Q2 Q3) ;; esac
  source_dir="$SOURCE_ROOT/sources/seed${SEED}/${valence}"
  pair_dir="$OUT_ROOT/seed${SEED}/${valence}"
  mkdir -p "$pair_dir"
  "$PYTHON_BIN" representations/pkl_conn.py -i "$source_dir" \
    -o "$pair_dir/stage2_melody_conn" -m melody 2>&1 | tee "$pair_dir/stage2_connection.log"
  set +e
  "$PYTHON_BIN" stage3_performance/inference.py -c "$CONFIG" -m gpt2 -r functional \
    -i "$CHECKPOINT" -id "$pair_dir/stage2_melody_conn" \
    -od "$pair_dir/stage3_performance" --seed "$SEED" \
    --emotion-override "${conditions[@]}" --conditioning-mode factorized_va \
    --paired-condition-seed --condition-track Melody --max_bars 8 --max_events 3072 \
    --temperature 1.0 --top-p 0.97 --lock-performance-to-condition-skeleton \
    2>&1 | tee "$pair_dir/stage3_generation.log"
  stage3_rc=${PIPESTATUS[0]}
  "$PYTHON_BIN" txt2midi.py -i "$pair_dir/stage3_performance" -o "$pair_dir/midi" \
    --max_bars 8 --condition-track Melody 2>&1 | tee "$pair_dir/conversion.log"
  conversion_rc=${PIPESTATUS[0]}
  "$PYTHON_BIN" scripts/validate_midi.py "$pair_dir/midi" \
    --output "$pair_dir/midi_validation.json" 2>&1 | tee "$pair_dir/midi_validation.log"
  validation_rc=${PIPESTATUS[0]}
  "$PYTHON_BIN" scripts/evaluate_quality_pilot.py "$pair_dir" \
    --event-dir-name stage3_performance --condition-track Melody \
    --expected-candidates 4 --expected-bars 8 --output "$pair_dir/quality_gate.json" \
    --output-csv "$pair_dir/quality_metrics.csv" 2>&1 | tee "$pair_dir/quality_gate.log"
  quality_rc=${PIPESTATUS[0]}
  "$PYTHON_BIN" scripts/evaluate_midi_descriptors.py "$pair_dir/midi" \
    --output-csv "$pair_dir/midi_descriptors.csv" --output-json "$pair_dir/midi_descriptors.json" \
    2>&1 | tee "$pair_dir/midi_descriptors.log"
  descriptor_rc=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SEED" "$valence" "$stage3_rc" \
    "$conversion_rc" "$validation_rc" "$quality_rc" "$descriptor_rc" >> "$OUT_ROOT/status.tsv"
  if [[ "$stage3_rc" -ne 0 || "$conversion_rc" -ne 0 || "$validation_rc" -ne 0 || "$descriptor_rc" -ne 0 ]]; then execution_failure=1; fi
done
printf '%s\n' "$execution_failure" > "$OUT_ROOT/execution.rc"
test "$execution_failure" -eq 0
"$PYTHON_BIN" scripts/validate_stage3_skeleton_lock.py "$OUT_ROOT" --expected-files 8 \
  --output "$OUT_ROOT/skeleton_validation.json" > "$OUT_ROOT/skeleton_validation.log" 2>&1
"$PYTHON_BIN" scripts/analyze_stage3_arousal_counterfactual.py "$OUT_ROOT" \
  --source-manifest "$OUT_ROOT/source_manifest.json" --output-dir "$OUT_ROOT/objective_analysis" \
  > "$OUT_ROOT/objective_analysis.log" 2>&1
set +e
"$PYTHON_BIN" scripts/evaluate_stage3_attribute_objective_seed_pilot.py \
  "$OUT_ROOT/skeleton_validation.json" "$OUT_ROOT/objective_analysis/objective_summary.json" \
  --output "$OUT_ROOT/gate_evaluation.json" > "$OUT_ROOT/gate_evaluation.log" 2>&1
gate_rc=$?
set -e
printf '%s\n' "$gate_rc" > "$OUT_ROOT/gate.rc"
if [[ "$gate_rc" -eq 0 ]]; then marker=STAGE3_ATTRIBUTE_OBJECTIVE_SEED_PILOT_PASS; else marker=STAGE3_ATTRIBUTE_OBJECTIVE_SEED_PILOT_FAIL_PRESERVED; fi
printf '%s output=%s gate_rc=%s\n' "$marker" "$OUT_ROOT" "$gate_rc" | tee "$OUT_ROOT/status.txt"
tar -czf "$OUT_ROOT.tar.gz" "$OUT_ROOT"
sha256sum "$OUT_ROOT.tar.gz" > "$OUT_ROOT.tar.gz.sha256"
exit "$gate_rc"
