#!/usr/bin/env bash
set -euo pipefail

# 4 D3 Ordered_Noise experiments: Fig 10 (case pool, n=20-25) and Fig 11 (random pool, n=65)
# Usage:
#   ./run_ordered_experiments.sh             # run all 4
#   ./run_ordered_experiments.sh 10a 11b     # run only experiments 10a, 11b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Define experiments: disease metric pool_idx random_sample_size output_file
declare -A EXP_DISEASE EXP_METRIC EXP_POOL EXP_SAMPLE EXP_OUTPUT
# 10a/10b: D3 case pool L1+LLR; 11a/11b: D3 random pool L1+LLR (sample=_ -> n=65 default)
ALL_KEYS="10a 10b 11a 11b"

EXP_DISEASE=([10a]=D3   [10b]=D3   [11a]=D3   [11b]=D3)
EXP_METRIC=( [10a]=LLR  [10b]=L1   [11a]=LLR  [11b]=L1)
EXP_POOL=(   [10a]=1    [10b]=1    [11a]=0    [11b]=0)
EXP_SAMPLE=( [10a]=_    [10b]=_    [11a]=_    [11b]=_)
EXP_OUTPUT=( [10a]=fig10a_D3_LLR_case.pdf   [10b]=fig10b_D3_L1_case.pdf \
             [11a]=fig11a_D3_LLR_random.pdf [11b]=fig11b_D3_L1_random.pdf)

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
