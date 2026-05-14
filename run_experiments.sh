#!/usr/bin/env bash
set -euo pipefail

# 7 Ordered_Noise experiments for zeroth-gpu
# Usage:
#   ./run_experiments.sh          # run all 7
#   ./run_experiments.sh a c f    # run only experiments a, c, f

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Define experiments: label disease metric pool_idx output_file
declare -A EXP_DISEASE EXP_METRIC EXP_POOL EXP_OUTPUT
ALL_KEYS="a b c d e f g"

EXP_DISEASE=([a]=D3  [b]=D3 [c]=D3  [d]=D3 [e]=D17  [f]=D17  [g]=D17)
EXP_METRIC=( [a]=LLR [b]=L1 [c]=LLR [d]=L1 [e]=LLR  [f]=LLR  [g]=L1)
EXP_POOL=(   [a]=1   [b]=1  [c]=0   [d]=0  [e]=1    [f]=0    [g]=0)
EXP_OUTPUT=( [a]=D3_LLR_case.pdf   [b]=D3_L1_case.pdf   [c]=D3_LLR_random.pdf \
             [d]=D3_L1_random.pdf   [e]=D17_LLR_case.pdf [f]=D17_LLR_random.pdf \
             [g]=D17_L1_random.pdf)

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
    output=${EXP_OUTPUT[$key]}
    log="runs/log_${key}_${disease}_${metric}_${pool}.txt"

    echo "[$key] Starting: $disease $metric pool=$pool -> $output (log: $log)"
    uv run python Ordered_Noise.py "$disease" "$metric" "$pool" "$output" > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/log_*.txt"
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
    echo "$FAILED experiment(s) failed. Check runs/log_*.txt for details."
    exit 1
fi
