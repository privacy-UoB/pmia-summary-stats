#!/usr/bin/env bash
set -euo pipefail

# 3 D17 Ordered_Noise experiments with random_sample_size=20
# Usage:
#   ./run_ordered_experiments.sh          # run all 3
#   ./run_ordered_experiments.sh a b      # run only experiments a, b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Define experiments: disease metric pool_idx random_sample_size output_file
declare -A EXP_DISEASE EXP_METRIC EXP_POOL EXP_SAMPLE EXP_OUTPUT
ALL_KEYS="a b"

EXP_DISEASE=([a]=D17  [b]=D17)
EXP_METRIC=( [a]=LLR  [b]=L1)
EXP_POOL=(   [a]=0    [b]=0)
EXP_SAMPLE=( [a]=20   [b]=20)
EXP_OUTPUT=( [a]=D17_LLR_random.pdf  [b]=D17_L1_random.pdf)

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    KEYS="$*"
else
    KEYS=$ALL_KEYS
fi

PIDS=()
for key in $KEYS; do
    disease=${EXP_DISEASE[$key]}
    metric=${EXP_METRIC[$key]}
    pool=${EXP_POOL[$key]}
    sample=${EXP_SAMPLE[$key]}
    output=${EXP_OUTPUT[$key]}
    log="runs/log_ordered_${key}_${disease}_${metric}_${pool}.txt"

    echo "[$key] Starting: $disease $metric pool=$pool sample=$sample -> $output (log: $log)"
    uv run python Ordered_Noise.py "$disease" "$metric" "$pool" "$sample" "$output" > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/log_ordered_*.txt"
echo ""

# Wait for all and report
FAILED=0
for i in "${!PIDS[@]}"; do
    key=$(echo $KEYS | cut -d' ' -f$((i+1)))
    if wait "${PIDS[$i]}"; then
        echo "[$key] Done (PID ${PIDS[$i]})"
    else
        echo "[$key] FAILED (PID ${PIDS[$i]})"
        FAILED=$((FAILED+1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "All experiments completed successfully."
    ls -la *.pdf 2>/dev/null || echo "No PDF files found."
else
    echo ""
    echo "$FAILED experiment(s) failed. Check runs/log_ordered_*.txt for details."
    exit 1
fi
