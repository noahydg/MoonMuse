#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_ROOT="${1:?usage: run_stage3_arousal_counterfactual.sh FROZEN_SOURCE_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-generation/stage3-arousal-counterfactual-${RUN_ID}}"
CLIP_BARS="${CLIP_BARS:-8}"
STAGE3_MAX_EVENTS="${STAGE3_MAX_EVENTS:-3072}"
MAX_REJECTIONS="${MAX_REJECTIONS:-512}"
MAX_SEGMENT_ATTEMPTS="${MAX_SEGMENT_ATTEMPTS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.97}"
STAGE3_CONFIG="${STAGE3_CONFIG:-stage3_performance/config/emopia_finetune_new_gpt2.yaml}"
STAGE3_CHECKPOINT_REL="${STAGE3_CHECKPOINT_REL:-checkpoints/stage3_finetune/params/best_params.pt}"
CONDITIONING_MODE="${CONDITIONING_MODE:-legacy_quadrant}"
PROTOCOL="${PROTOCOL:-stage3-fixed-source-arousal-counterfactual-v1}"
TRAINING_PERFORMED="${TRAINING_PERFORMED:-false}"
PREVENT_DUPLICATE_ONSET_PITCHES="${PREVENT_DUPLICATE_ONSET_PITCHES:-0}"
LOCK_PERFORMANCE_TO_CONDITION_SKELETON="${LOCK_PERFORMANCE_TO_CONDITION_SKELETON:-0}"

case "$PREVENT_DUPLICATE_ONSET_PITCHES" in
  0) onset_constraint_args=() ;;
  1) onset_constraint_args=(--prevent-duplicate-onset-pitches) ;;
  *) printf 'PREVENT_DUPLICATE_ONSET_PITCHES must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$LOCK_PERFORMANCE_TO_CONDITION_SKELETON" in
  0) skeleton_lock_args=() ;;
  1) skeleton_lock_args=(--lock-performance-to-condition-skeleton) ;;
  *) printf 'LOCK_PERFORMANCE_TO_CONDITION_SKELETON must be 0 or 1\n' >&2; exit 2 ;;
esac
if [[ "$PREVENT_DUPLICATE_ONSET_PITCHES" -eq 1 \
   && "$LOCK_PERFORMANCE_TO_CONDITION_SKELETON" -eq 1 ]]; then
  printf 'onset masking and skeleton lock must not run together\n' >&2
  exit 2
fi

RUN_DIR_42="${RUN_DIR_42:-generation/formal-full-seed42-20260804-162930}"
RUN_DIR_43="${RUN_DIR_43:-generation/formal-full-seed43-20260804-204320}"
RUN_DIR_44="${RUN_DIR_44:-generation/formal-full-seed44-20260804-214837}"

test -s "$SOURCE_ROOT/source_manifest.json"
mkdir -p "$OUT_ROOT"
cp "$SOURCE_ROOT/source_manifest.json" "$OUT_ROOT/source_manifest.json"

cat > "$OUT_ROOT/campaign_manifest.txt" <<EOF
protocol=$PROTOCOL
formal_study_data=false
training_performed=$TRAINING_PERFORMED
source_root=$SOURCE_ROOT
stage3_config=$STAGE3_CONFIG
stage3_checkpoint_rel=$STAGE3_CHECKPOINT_REL
conditioning_mode=$CONDITIONING_MODE
clip_bars=$CLIP_BARS
stage3_max_events=$STAGE3_MAX_EVENTS
max_rejections=$MAX_REJECTIONS
max_segment_attempts=$MAX_SEGMENT_ATTEMPTS
temperature=$TEMPERATURE
top_p=$TOP_P
conditions_positive=Q1,Q4
conditions_negative=Q2,Q3
paired_condition_seed=true
prevent_duplicate_onset_pitches=$PREVENT_DUPLICATE_ONSET_PITCHES
lock_performance_to_condition_skeleton=$LOCK_PERFORMANCE_TO_CONDITION_SKELETON
selection_rule=lowest_sha256_within_model_seed_and_source_quadrant
EOF

run_dir_for_seed() {
  case "$1" in
    42) printf '%s\n' "$RUN_DIR_42" ;;
    43) printf '%s\n' "$RUN_DIR_43" ;;
    44) printf '%s\n' "$RUN_DIR_44" ;;
    *) printf 'unsupported seed: %s\n' "$1" >&2; return 2 ;;
  esac
}

for seed in 42 43 44; do
  run_dir="$(run_dir_for_seed "$seed")"
  test -s "$run_dir/$STAGE3_CHECKPOINT_REL"
  for valence in positive negative; do
    source_dir="$SOURCE_ROOT/sources/seed${seed}/${valence}"
    source_count="$(find "$source_dir" -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')"
    test "$source_count" -eq 2
  done
done

sha256sum \
  scripts/run_stage3_arousal_counterfactual.sh \
  scripts/validate_midi.py scripts/evaluate_quality_pilot.py \
  scripts/evaluate_midi_descriptors.py \
  stage3_performance/inference.py performance_skeleton.py emotion_conditioning.py \
  factorized_conditioning.py generation_budget.py "$STAGE3_CONFIG" \
  representations/pkl_conn.py txt2midi.py \
  "$SOURCE_ROOT/source_manifest.json" \
  "$RUN_DIR_42/$STAGE3_CHECKPOINT_REL" \
  "$RUN_DIR_43/$STAGE3_CHECKPOINT_REL" \
  "$RUN_DIR_44/$STAGE3_CHECKPOINT_REL" \
  > "$OUT_ROOT/code_source_and_model_sha256.txt"

printf 'seed\tvalence\tstage3_rc\tconversion_rc\tvalidation_rc\tquality_rc\tdescriptor_rc\n' \
  > "$OUT_ROOT/status.tsv"
execution_failure=0

for seed in 42 43 44; do
  run_dir="$(run_dir_for_seed "$seed")"
  checkpoint="$run_dir/$STAGE3_CHECKPOINT_REL"
  for valence in positive negative; do
    case "$valence" in
      positive) conditions=(Q1 Q4) ;;
      negative) conditions=(Q2 Q3) ;;
    esac
    source_dir="$SOURCE_ROOT/sources/seed${seed}/${valence}"
    pair_dir="$OUT_ROOT/seed${seed}/${valence}"
    mkdir -p "$pair_dir"

    printf 'seed %s %s：连接固定旋律源\n' "$seed" "$valence" \
      > "$OUT_ROOT/current_stage.txt"
    "$PYTHON_BIN" representations/pkl_conn.py \
      -i "$source_dir" -o "$pair_dir/stage2_melody_conn" -m melody \
      2>&1 | tee "$pair_dir/stage2_connection.log"

    printf 'seed %s %s：Stage 3 成对条件推理\n' "$seed" "$valence" \
      > "$OUT_ROOT/current_stage.txt"
    set +e
    "$PYTHON_BIN" stage3_performance/inference.py \
      -c "$STAGE3_CONFIG" \
      -m gpt2 -r functional -i "$checkpoint" \
      -id "$pair_dir/stage2_melody_conn" \
      -od "$pair_dir/stage3_performance" \
      --seed "$seed" --emotion-override "${conditions[@]}" \
      --conditioning-mode "$CONDITIONING_MODE" \
      --paired-condition-seed --condition-track Melody \
      --max_bars "$CLIP_BARS" --max_events "$STAGE3_MAX_EVENTS" \
      --max_rejections "$MAX_REJECTIONS" \
      --max_segment_attempts "$MAX_SEGMENT_ATTEMPTS" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" \
      "${onset_constraint_args[@]}" \
      "${skeleton_lock_args[@]}" \
      2>&1 | tee "$pair_dir/stage3_generation.log"
    stage3_rc=${PIPESTATUS[0]}

    "$PYTHON_BIN" txt2midi.py \
      -i "$pair_dir/stage3_performance" -o "$pair_dir/midi" \
      --max_bars "$CLIP_BARS" --condition-track Melody \
      2>&1 | tee "$pair_dir/conversion.log"
    conversion_rc=${PIPESTATUS[0]}

    "$PYTHON_BIN" scripts/validate_midi.py "$pair_dir/midi" \
      --output "$pair_dir/midi_validation.json" \
      2>&1 | tee "$pair_dir/midi_validation.log"
    validation_rc=${PIPESTATUS[0]}

    "$PYTHON_BIN" scripts/evaluate_quality_pilot.py "$pair_dir" \
      --event-dir-name stage3_performance --condition-track Melody \
      --expected-candidates 4 --expected-bars "$CLIP_BARS" \
      --output "$pair_dir/quality_gate.json" \
      --output-csv "$pair_dir/quality_metrics.csv" \
      2>&1 | tee "$pair_dir/quality_gate.log"
    quality_rc=${PIPESTATUS[0]}

    "$PYTHON_BIN" scripts/evaluate_midi_descriptors.py "$pair_dir/midi" \
      --output-csv "$pair_dir/midi_descriptors.csv" \
      --output-json "$pair_dir/midi_descriptors.json" \
      2>&1 | tee "$pair_dir/midi_descriptors.log"
    descriptor_rc=${PIPESTATUS[0]}
    set -e

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$valence" "$stage3_rc" "$conversion_rc" \
      "$validation_rc" "$quality_rc" "$descriptor_rc" \
      >> "$OUT_ROOT/status.tsv"
    if [[ "$stage3_rc" -ne 0 || "$conversion_rc" -ne 0 \
       || "$validation_rc" -ne 0 || "$descriptor_rc" -ne 0 ]]; then
      execution_failure=1
    fi
  done
done

printf '%s\n' "$execution_failure" > "$OUT_ROOT/execution.rc"
printf '完成：Stage 3 同源反事实已生成并等待冻结分析\n' \
  > "$OUT_ROOT/current_stage.txt"
tar -czf "$OUT_ROOT.tar.gz" "$OUT_ROOT"
sha256sum "$OUT_ROOT.tar.gz" > "$OUT_ROOT.tar.gz.sha256"
printf 'STAGE3_COUNTERFACTUAL_COMPLETE output=%s archive=%s execution_failure=%s\n' \
  "$OUT_ROOT" "$OUT_ROOT.tar.gz" "$execution_failure" \
  | tee "$OUT_ROOT/status.txt"
exit "$execution_failure"
