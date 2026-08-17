#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44}"
WORKERS="${WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date +%Y%m%d-%H%M%S)}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-generation/stage3-arousal-alignment-campaign-${CAMPAIGN_ID}}"
RUN_DIR_42="${RUN_DIR_42:-generation/formal-full-seed42-20260804-162930}"
RUN_DIR_43="${RUN_DIR_43:-generation/formal-full-seed43-20260804-204320}"
RUN_DIR_44="${RUN_DIR_44:-generation/formal-full-seed44-20260804-214837}"

mkdir -p "$CAMPAIGN_DIR"
cat > "$CAMPAIGN_DIR/campaign_manifest.txt" <<EOF
protocol=stage3-arousal-alignment-campaign-v1
campaign_id=$CAMPAIGN_ID
seeds=$SEEDS
workers=$WORKERS
max_epochs=$MAX_EPOCHS
early_stopping_patience=$PATIENCE
execution=sequential_stop_on_failure
objective=token_cross_entropy_plus_paired_counterfactual_directional_hinge_v1
alignment_weight=0.20
EOF
printf 'seed\trun_dir\tsource\tstate\n' > "$CAMPAIGN_DIR/run_dirs.tsv"

legacy_run_for_seed() {
  case "$1" in
    42) printf '%s\n' "$RUN_DIR_42" ;;
    43) printf '%s\n' "$RUN_DIR_43" ;;
    44) printf '%s\n' "$RUN_DIR_44" ;;
    *) return 2 ;;
  esac
}

for seed in $SEEDS; do
  case "$seed" in 42|43|44) ;; *) printf 'unsupported seed: %s\n' "$seed" >&2; exit 2 ;; esac
  run_id="${CAMPAIGN_ID}-seed${seed}"
  run_dir="generation/stage3-arousal-alignment-formal-seed${seed}-${CAMPAIGN_ID}"
  legacy_run_dir="$(legacy_run_for_seed "$seed")"
  printf '%s\t%s\ttrain\tRUNNING\n' "$seed" "$run_dir" >> "$CAMPAIGN_DIR/run_dirs.tsv"
  printf 'seed %s：启动唤醒度对齐正式微调\n' "$seed" > "$CAMPAIGN_DIR/current_stage.txt"
  SEED="$seed" RUN_ID="$run_id" RUN_DIR="$run_dir" \
    LEGACY_RUN_DIR="$legacy_run_dir" PYTHON_BIN="$PYTHON_BIN" \
    WORKERS="$WORKERS" MAX_EPOCHS="$MAX_EPOCHS" PATIENCE="$PATIENCE" \
    bash scripts/run_stage3_arousal_alignment_formal_seed.sh
  "$PYTHON_BIN" scripts/validate_stage3_arousal_alignment_formal_run.py \
    "$run_dir" --seed "$seed" --output "$CAMPAIGN_DIR/seed${seed}_validation.json"
  printf '%s\t%s\ttrain\tPASS\n' "$seed" "$run_dir" >> "$CAMPAIGN_DIR/run_dirs.tsv"
done

printf 'STAGE3_AROUSAL_ALIGNMENT_CAMPAIGN_PASS campaign_dir=%s seeds=%s\n' \
  "$CAMPAIGN_DIR" "$SEEDS" | tee "$CAMPAIGN_DIR/status.txt"
printf '三种子正式微调完成：启动相同12源 Gate C\n' > "$CAMPAIGN_DIR/current_stage.txt"
