#!/usr/bin/env bash
set -euo pipefail

# Ordered experiments on D3 (prostate cancer):
#   Fig 3  (Ordered.py, no synthetic noise): 3a case pool, 3b random pool (n=65)
#   Fig 10 (Ordered_Noise.py, case pool, n=65)
#   Fig 11 (Ordered_Noise.py, random pool, n=65)
# Usage:
#   ./run_ordered_experiments.sh                  # run all
#   ./run_ordered_experiments.sh 3a 10a 11b       # run a subset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Optional seed override (defaults to 42 in each Python script).
SEED_ARG=${SEED:+--seed $SEED}

# Define experiments. Two flavours:
#   Fig 3 (Ordered.py):       disease pool_idx sample_size  -> one panel per pool, both metrics
#   Fig 10/11 (Ordered_Noise): disease metric pool_idx sample_size  -> noise sweep per metric
declare -A EXP_SCRIPT EXP_DISEASE EXP_METRIC EXP_POOL EXP_SAMPLE EXP_OUTPUT
ALL_KEYS="3a 3b 10a 10b 11a 11b"

EXP_SCRIPT=( [3a]=Ordered.py        [3b]=Ordered.py \
             [10a]=Ordered_Noise.py [10b]=Ordered_Noise.py \
             [11a]=Ordered_Noise.py [11b]=Ordered_Noise.py)
EXP_DISEASE=([3a]=D3    [3b]=D3    [10a]=D3   [10b]=D3   [11a]=D3   [11b]=D3)
EXP_METRIC=( [3a]=_     [3b]=_     [10a]=LLR  [10b]=L1   [11a]=LLR  [11b]=L1)
EXP_POOL=(   [3a]=1     [3b]=0     [10a]=1    [10b]=1    [11a]=0    [11b]=0)
EXP_SAMPLE=( [3a]=_     [3b]=_     [10a]=_    [10b]=_    [11a]=_    [11b]=_)
EXP_OUTPUT=( [3a]=fig3a_D3_case.pdf          [3b]=fig3b_D3_random.pdf \
             [10a]=fig10a_D3_LLR_case.pdf    [10b]=fig10b_D3_L1_case.pdf \
             [11a]=fig11a_D3_LLR_random.pdf  [11b]=fig11b_D3_L1_random.pdf)

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    KEYS="$*"
else
    KEYS=$ALL_KEYS
fi

PIDS=()
for key in $KEYS; do
    script=${EXP_SCRIPT[$key]}
    disease=${EXP_DISEASE[$key]}
    metric=${EXP_METRIC[$key]}
    pool=${EXP_POOL[$key]}
    sample=${EXP_SAMPLE[$key]}
    output=${EXP_OUTPUT[$key]}
    log="runs/log_ordered_${key}_${disease}_${metric}_${pool}.txt"

    if [ "$script" = "Ordered.py" ]; then
        echo "[$key] Starting: $script $disease pool=$pool sample=$sample -> $output (log: $log)"
        uv run python "$script" "$disease" "$pool" "$sample" "$output" $SEED_ARG > "$log" 2>&1 &
    else
        echo "[$key] Starting: $script $disease $metric pool=$pool sample=$sample -> $output (log: $log)"
        uv run python "$script" "$disease" "$metric" "$pool" "$sample" "$output" $SEED_ARG > "$log" 2>&1 &
    fi
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
