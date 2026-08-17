#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_ROOT="${1:?usage: run_stage3_arousal_alignment_gate_c.sh FROZEN_SOURCE_ROOT FORMAL_CAMPAIGN_DIR}"
FORMAL_CAMPAIGN_DIR="${2:?usage: run_stage3_arousal_alignment_gate_c.sh FROZEN_SOURCE_ROOT FORMAL_CAMPAIGN_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-generation/stage3-arousal-alignment-gate-c-${RUN_ID}}"
ANALYSIS_DIR="$OUT_ROOT/gate_c_analysis"
GATE_JSON="$OUT_ROOT/gate_c_evaluation.json"

test -s "$SOURCE_ROOT/source_manifest.json"
grep -q '^STAGE3_AROUSAL_ALIGNMENT_CAMPAIGN_PASS ' "$FORMAL_CAMPAIGN_DIR/status.txt"
mkdir -p "$OUT_ROOT"
formal_run_for_seed() {
  awk -F '\t' -v seed="$1" \
    '$1 == seed && $4 == "PASS" {run_dir=$2} END {if (run_dir) print run_dir}' \
    "$FORMAL_CAMPAIGN_DIR/run_dirs.tsv"
}
RUN_DIR_42="$(formal_run_for_seed 42)"
RUN_DIR_43="$(formal_run_for_seed 43)"
RUN_DIR_44="$(formal_run_for_seed 44)"
test -n "$RUN_DIR_42" && test -n "$RUN_DIR_43" && test -n "$RUN_DIR_44"

for seed in 42 43 44; do
  run_var="RUN_DIR_${seed}"
  run_dir="${!run_var}"
  "$PYTHON_BIN" scripts/validate_stage3_arousal_alignment_formal_run.py \
    "$run_dir" --seed "$seed" --output "$OUT_ROOT/formal_seed${seed}_validation.json"
done

set +e
PYTHON_BIN="$PYTHON_BIN" RUN_ID="$RUN_ID" OUT_ROOT="$OUT_ROOT" \
  RUN_DIR_42="$RUN_DIR_42" RUN_DIR_43="$RUN_DIR_43" RUN_DIR_44="$RUN_DIR_44" \
  STAGE3_CONFIG=stage3_performance/config/emopia_finetune_factorized_va_alignment_gpt2.yaml \
  STAGE3_CHECKPOINT_REL=checkpoints/stage3_arousal_alignment_finetune/params/best_params.pt \
  CONDITIONING_MODE=factorized_va \
  PROTOCOL=stage3-arousal-alignment-fixed-source-counterfactual-v1 \
  TRAINING_PERFORMED=matched_arousal_alignment_stage3_finetune_seeds_42_43_44 \
  bash scripts/run_stage3_arousal_counterfactual.sh "$SOURCE_ROOT"
generation_rc=$?
set -e
printf '%s\n' "$generation_rc" > "$OUT_ROOT/generation.rc"
if [[ "$generation_rc" -ne 0 ]]; then
  printf 'STAGE3_AROUSAL_ALIGNMENT_GATE_C_EXECUTION_FAIL output=%s generation_rc=%s\n' \
    "$OUT_ROOT" "$generation_rc" | tee "$OUT_ROOT/alignment_gate_c_status.txt"
  tar -czf "$OUT_ROOT.tar.gz" "$OUT_ROOT"
  sha256sum "$OUT_ROOT.tar.gz" > "$OUT_ROOT.tar.gz.sha256"
  exit "$generation_rc"
fi

"$PYTHON_BIN" scripts/analyze_stage3_arousal_counterfactual.py "$OUT_ROOT" \
  --source-manifest "$SOURCE_ROOT/source_manifest.json" \
  --output-dir "$ANALYSIS_DIR" 2>&1 | tee "$OUT_ROOT/gate_c_analysis.log"
"$PYTHON_BIN" scripts/evaluate_stage3_factorized_gate_c.py \
  "$ANALYSIS_DIR/objective_summary.json" --output "$GATE_JSON" \
  2>&1 | tee "$OUT_ROOT/gate_c_evaluation.log" || gate_rc=$?
gate_rc="${gate_rc:-0}"
printf '%s\n' "$gate_rc" > "$OUT_ROOT/gate_c.rc"
if [[ "$gate_rc" -eq 0 ]]; then
  marker=STAGE3_AROUSAL_ALIGNMENT_GATE_C_PASS
elif [[ "$gate_rc" -eq 3 ]]; then
  marker=STAGE3_AROUSAL_ALIGNMENT_GATE_C_FAIL_PRESERVED
else
  marker=STAGE3_AROUSAL_ALIGNMENT_GATE_C_EVALUATION_ERROR
fi
printf '%s output=%s generation_rc=%s gate_rc=%s\n' \
  "$marker" "$OUT_ROOT" "$generation_rc" "$gate_rc" \
  | tee "$OUT_ROOT/alignment_gate_c_status.txt"
tar -czf "$OUT_ROOT.tar.gz" "$OUT_ROOT"
sha256sum "$OUT_ROOT.tar.gz" > "$OUT_ROOT.tar.gz.sha256"
exit "$gate_rc"
