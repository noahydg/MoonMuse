#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-42 43 44}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date +%Y%m%d-%H%M%S)}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-generation/formal-campaign-${CAMPAIGN_ID}}"
mkdir -p "$CAMPAIGN_DIR"

cat > "$CAMPAIGN_DIR/campaign_manifest.txt" <<EOF
protocol=formal-full-campaign-20260804
campaign_id=$CAMPAIGN_ID
seeds=$SEEDS
execution=sequential_stop_on_failure
EOF

for seed in $SEEDS; do
  run_id="$(date +%Y%m%d-%H%M%S)"
  run_dir="generation/formal-full-seed${seed}-${run_id}"
  printf 'START seed=%s run_dir=%s\n' "$seed" "$run_dir" \
    | tee -a "$CAMPAIGN_DIR/status.txt"

  SEED="$seed" RUN_ID="$run_id" RUN_DIR="$run_dir" \
    bash scripts/run_formal_seed.sh
  SEED="$seed" bash scripts/run_formal_generation.sh "$run_dir"
  bash scripts/package_formal_run.sh "$run_dir" "$ROOT/formal-artifacts"

  printf 'PASS seed=%s run_dir=%s\n' "$seed" "$run_dir" \
    | tee -a "$CAMPAIGN_DIR/status.txt"
done

python scripts/aggregate_generation_runs.py \
  --model-name MoonMuse-full \
  --output-dir "$CAMPAIGN_DIR/three-seed-summary" \
  --discover-root generation \
  --seeds 42 43 44
python scripts/evaluate_control_proxies.py \
  "$CAMPAIGN_DIR/three-seed-summary/descriptors_by_seed.csv" \
  --output-dir "$CAMPAIGN_DIR/three-seed-summary/control-proxies"

printf 'FORMAL_CAMPAIGN_COMPLETE campaign_dir=%s\n' "$CAMPAIGN_DIR" \
  | tee -a "$CAMPAIGN_DIR/status.txt"
