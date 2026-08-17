#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_ROOT="${1:?usage: run_v81_v70_population.sh SOURCE_ROOT V69_GENERATION_ROOT BOUNDS_JSON PROTOCOL}"
V69_GENERATION_ROOT="${2:?usage: run_v81_v70_population.sh SOURCE_ROOT V69_GENERATION_ROOT BOUNDS_JSON PROTOCOL}"
BOUNDS_JSON="${3:?usage: run_v81_v70_population.sh SOURCE_ROOT V69_GENERATION_ROOT BOUNDS_JSON PROTOCOL}"
PROTOCOL="${4:?usage: run_v81_v70_population.sh SOURCE_ROOT V69_GENERATION_ROOT BOUNDS_JSON PROTOCOL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-generation/v81-v70-baseline-superiority-${RUN_ID}}"
V81R1_ROOT="${V81R1_ROOT:-generation/v81-v70-baseline-superiority-20260809-v81r1}"
CONFIG="stage3_performance/config/emopia_finetune_calibrated_alignment_gpt2.yaml"
BASE_PROTOCOL="results/v81-v70-baseline-superiority-screen-protocol-20260809.md"

declare -A CHECKPOINT_SHA
CHECKPOINT_SHA[42]="f1836dd716c1d24e2691961981e464ab261309656f0a0238b0b50b0807d428fc"
CHECKPOINT_SHA[43]="326255d4cccf20560442284e4f32cc35f4f3a1e81d2b88092abf87b1eb2d83b2"
CHECKPOINT_SHA[44]="b309a528ba8bb2c58011d8386362bfc5664ac9c544775b8a705bffd9c5158de1"

test ! -e "$OUT_ROOT"
test -s "$SOURCE_ROOT/source_manifest.csv"
test -s "$SOURCE_ROOT/source_freeze.json"
test -s "$BOUNDS_JSON"
test -s "$PROTOCOL"

verify_sha() {
  local expected="$1"
  local file="$2"
  test -s "$file"
  test "$(sha256sum "$file" | awk '{print $1}')" = "$expected"
}

verify_sha "76ddaf78950098ec430ba9103249f9c68d1ed544ffe9f806ad296cfcfe2b0cd9" "$BOUNDS_JSON"
verify_sha "f8c2746f3dfbd873cdc72b446d27f60af26cdd23896e11bd420ec3935feb64ee" stage3_performance/inference.py
verify_sha "bfe7faf5395dd16231459f22ea9c15dc61d19f217040dfa8e887a4eaa4bc16f3" performance_skeleton.py
verify_sha "991f56782e5eb0273eac7bef2a559d24e632f21d4fd9b9fe3421b2d4b5de2c43" scripts/audit_v81_condition_grammar.py
verify_sha "fb9d038d19b4d9b7f80ca9283deb7b7a3c78b2cc3ac61f0f9cbf3220912c1b11" scripts/validate_v81r1_v81r2_overlap.py
verify_sha "f3c64522648ce96778881dc164357ab3f7092ac5603b91ff7887dc619dcd5af9" "$BASE_PROTOCOL"
verify_sha "bf5601964c80d0e5ddb3d8e0925b149e7086bf5cfa1c4aa444485b444d405a58" "$CONFIG"
verify_sha "864dbf32e556bcabe718794d1ac8bc62bdc64e6cd62c2ec8bcba27b01dead1fd" stage3_performance/model/music_gpt2.py
verify_sha "cd2f08ecadffd3ad6b4842356f304612ef35dd5478343f4836fb9357bca23d8e" stage3_performance/dataloader.py
verify_sha "225e7cbad922745e6cad74bd8fdeb1313684f278a3579c0d1ff91ae94df79fbd" stage3_performance/model/transformer_helpers.py
verify_sha "6a7c7e8882b185b617f6403bb8d93a11090f6118d44d6fb58d663a9a4b761fb3" stage3_performance/convert2midi.py
verify_sha "c8bb546824ef0632bffc8d556baed811379d622eefb262f27a19a5fed935e9ed" stage3_performance/convert_key.py
verify_sha "bbc15b6fb8218cf0eca5b02aad44c212a62ef8a34b36f582b267cfd52d566ca5" generation_budget.py
verify_sha "563018c968f0fbdb5933e8c8880feacedf9308306ded3970248b50d41b63711e" emotion_conditioning.py
verify_sha "62125ac302ec4de48d762ca344f75f12fd3b065a10ca5d8a671a1a2e52a60f24" factorized_conditioning.py
verify_sha "b94c359b31f02a5c9be3ce46ab2d49a24b1e87d2cea160c36bd9928b5acae5cb" txt2midi.py
verify_sha "382f2282fdc2ef1cf917e9963e290946750f4d1fe1ec9181a20f8050454bbf0c" representations/pkl_conn.py
"$PYTHON_BIN" -c 'import mido, torch, yaml; assert torch.cuda.is_available(), "CUDA is required"'

for seed in 42 43 44; do
  checkpoint="$V69_GENERATION_ROOT/stage3-calibrated-alignment-formal-seed${seed}-20260807-v69/checkpoints/stage3_calibrated_alignment_finetune/params/best_params.pt"
  test -s "$checkpoint"
  test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "${CHECKPOINT_SHA[$seed]}"
  for quadrant in Q1 Q2 Q3 Q4; do
    test "$(find "$SOURCE_ROOT/sources/seed${seed}/${quadrant}" -maxdepth 1 -type f -name '*.txt' | wc -l)" -eq 25
  done
done

grammar_audit_tmp="$(mktemp /tmp/v81r2-condition-grammar.XXXXXX)"
"$PYTHON_BIN" scripts/audit_v81_condition_grammar.py \
  --source-root "$SOURCE_ROOT/sources" --output "$grammar_audit_tmp" \
  --expected-files 300 --expected-issue-files 8 \
  --expected-discarded-prefixes 9

mkdir -p "$OUT_ROOT"
mv "$grammar_audit_tmp" "$OUT_ROOT/condition_grammar_audit.json"
cat > "$OUT_ROOT/run_manifest.txt" <<EOF
protocol=v81r2-condition-grammar-technical-amendment-v1
role=prospective_three_seed_automatic_screen
formal_study_data=false
source_root=$SOURCE_ROOT
source_manifest=$SOURCE_ROOT/source_manifest.csv
v69_generation_root=$V69_GENERATION_ROOT
v81r1_root=$V81R1_ROOT
configuration=$CONFIG
attribute_bounds=$BOUNDS_JSON
seeds=42,43,44
quadrants=Q1,Q2,Q3,Q4
expected_outputs=300
temperature=1.0
top_p=0.97
paired_condition_seed=true
condition_skeleton_locked=true
no_result_dependent_retry=true
EOF

{
  sha256sum "$SOURCE_ROOT/source_manifest.csv" "$SOURCE_ROOT/source_freeze.json" \
    "$BASE_PROTOCOL" "$PROTOCOL" "$BOUNDS_JSON" "$CONFIG" \
    stage3_performance/inference.py performance_skeleton.py \
    stage3_performance/model/music_gpt2.py stage3_performance/model/transformer_helpers.py \
    stage3_performance/dataloader.py stage3_performance/convert2midi.py \
    stage3_performance/convert_key.py performance_skeleton.py generation_budget.py \
    emotion_conditioning.py factorized_conditioning.py representations/pkl_conn.py txt2midi.py \
    scripts/run_v81_v70_population.sh scripts/validate_v81_v70_population.py \
    scripts/evaluate_quality_pilot_beat_normalized.py \
    scripts/audit_v81_condition_grammar.py scripts/validate_v81r1_v81r2_overlap.py \
    "$OUT_ROOT/condition_grammar_audit.json"
  for seed in 42 43 44; do
    sha256sum "$V69_GENERATION_ROOT/stage3-calibrated-alignment-formal-seed${seed}-20260807-v69/checkpoints/stage3_calibrated_alignment_finetune/params/best_params.pt"
  done
} > "$OUT_ROOT/input_and_code_sha256.txt"

printf 'seed\tquadrant\tconnection_rc\tstage3_rc\tconversion_rc\tvalidation_rc\tv74_rc\n' > "$OUT_ROOT/status.tsv"
execution_failure=0
for seed in 42 43 44; do
  checkpoint="$V69_GENERATION_ROOT/stage3-calibrated-alignment-formal-seed${seed}-20260807-v69/checkpoints/stage3_calibrated_alignment_finetune/params/best_params.pt"
  for quadrant in Q1 Q2 Q3 Q4; do
    source_dir="$SOURCE_ROOT/sources/seed${seed}/${quadrant}"
    group_dir="$OUT_ROOT/seed${seed}/${quadrant}"
    mkdir -p "$group_dir"
    set +e
    "$PYTHON_BIN" representations/pkl_conn.py -i "$source_dir" \
      -o "$group_dir/stage2_melody_conn" -m melody \
      > "$group_dir/stage2_connection.log" 2>&1
    connection_rc=$?
    "$PYTHON_BIN" stage3_performance/inference.py -c "$CONFIG" -m gpt2 -r functional \
      -i "$checkpoint" -id "$group_dir/stage2_melody_conn" \
      -od "$group_dir/stage3_performance" --seed "$seed" \
      --emotion-override "$quadrant" --conditioning-mode factorized_va \
      --paired-condition-seed --condition-track Melody --max_bars 8 --max_events 3072 \
      --temperature 1.0 --top-p 0.97 --lock-performance-to-condition-skeleton \
      --attribute-bounds-json "$BOUNDS_JSON" \
      > "$group_dir/stage3_generation.log" 2>&1
    stage3_rc=$?
    "$PYTHON_BIN" txt2midi.py -i "$group_dir/stage3_performance" -o "$group_dir/midi" \
      --max_bars 8 --condition-track Melody \
      > "$group_dir/conversion.log" 2>&1
    conversion_rc=$?
    "$PYTHON_BIN" scripts/validate_midi.py "$group_dir/midi" \
      --output "$group_dir/midi_validation.json" \
      > "$group_dir/midi_validation.log" 2>&1
    validation_rc=$?
    "$PYTHON_BIN" scripts/evaluate_quality_pilot_beat_normalized.py "$group_dir" \
      --event-dir-name stage3_performance --condition-track Melody \
      --expected-candidates 25 --expected-bars 8 \
      --output "$group_dir/quality_gate_v74.json" \
      --output-csv "$group_dir/quality_metrics_v74.csv" \
      > "$group_dir/quality_gate_v74.log" 2>&1
    v74_rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$quadrant" \
      "$connection_rc" "$stage3_rc" "$conversion_rc" "$validation_rc" "$v74_rc" \
      >> "$OUT_ROOT/status.tsv"
    if [[ "$connection_rc" -ne 0 || "$stage3_rc" -ne 0 || "$conversion_rc" -ne 0 || "$validation_rc" -ne 0 || "$v74_rc" -ne 0 ]]; then
      execution_failure=1
    fi
  done
done

set +e
"$PYTHON_BIN" scripts/validate_v81_v70_population.py \
  --run-root "$OUT_ROOT" --source-manifest "$SOURCE_ROOT/source_manifest.csv" \
  --protocol "$PROTOCOL" --output "$OUT_ROOT/population_validation.json" \
  > "$OUT_ROOT/population_validation.log" 2>&1
population_rc=$?
"$PYTHON_BIN" scripts/validate_v81r1_v81r2_overlap.py \
  --v81r1-root "$V81R1_ROOT" --v81r2-root "$OUT_ROOT" \
  --output "$OUT_ROOT/v81r1_v81r2_overlap_validation.json" \
  --expected-r1 204 --expected-r2 300 \
  > "$OUT_ROOT/v81r1_v81r2_overlap_validation.log" 2>&1
overlap_rc=$?
set -e
printf '%s\n' "$execution_failure" > "$OUT_ROOT/execution.rc"
printf '%s\n' "$population_rc" > "$OUT_ROOT/population_validation.rc"
printf '%s\n' "$overlap_rc" > "$OUT_ROOT/overlap_validation.rc"
if [[ "$execution_failure" -eq 0 && "$population_rc" -eq 0 && "$overlap_rc" -eq 0 ]]; then
  marker=V81_V70_POPULATION_EXECUTION_COMPLETE
  final_rc=0
else
  marker=V81_V70_POPULATION_EXECUTION_FAIL_PRESERVED
  final_rc=3
fi
printf '%s output=%s execution_failure=%s population_rc=%s overlap_rc=%s\n' \
  "$marker" "$OUT_ROOT" "$execution_failure" "$population_rc" "$overlap_rc" \
  | tee "$OUT_ROOT/status.txt"
tar -czf "$OUT_ROOT.tar.gz" "$OUT_ROOT"
sha256sum "$OUT_ROOT.tar.gz" > "$OUT_ROOT.tar.gz.sha256"
exit "$final_rc"
