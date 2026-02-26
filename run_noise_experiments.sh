#!/usr/bin/env bash
set -euo pipefail

# 8 Noise.py experiments (Figures 2a-2d with variants)
# Usage:
#   ./run_noise_experiments.sh          # run all 8
#   ./run_noise_experiments.sh a c g    # run only experiments a, c, g

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Define experiments: dataset include_deviations disease pop_idx pool_idx random_sample_size output_file
declare -A EXP_DATASET EXP_DEVS EXP_DISEASE EXP_POP EXP_POOL EXP_SAMPLE EXP_OUTPUT
ALL_KEYS="a b c d e f g h"

#              dataset   devs   disease pop pool sample  output
EXP_DATASET=([a]=miRNA      [b]=miRNA       [c]=miRNA        [d]=miRNA       [e]=miRNA        [f]=miRNA         [g]=Timestamp  [h]=FitBit)
EXP_DEVS=(   [a]=true       [b]=true        [c]=false        [d]=true        [e]=true         [f]=false         [g]=true       [h]=true)
EXP_DISEASE=([a]=D3         [b]=D17         [c]=D17          [d]=D3          [e]=D17          [f]=D17           [g]=_          [h]=_)
EXP_POP=(    [a]=1          [b]=1           [c]=1            [d]=0           [e]=0            [f]=0             [g]=0          [h]=0)
EXP_POOL=(   [a]=1          [b]=1           [c]=1            [d]=0           [e]=0            [f]=0             [g]=0          [h]=0)
EXP_SAMPLE=( [a]=_          [b]=20          [c]=20           [d]=_           [e]=20           [f]=20            [g]=_          [h]=_)
EXP_OUTPUT=( [a]=fig2a_D3_case.pdf          [b]=fig2ai_D17_case_dev.pdf     [c]=fig2aii_D17_case_nodev.pdf \
             [d]=fig2b_D3_random.pdf        [e]=fig2bi_D17_random_dev.pdf   [f]=fig2bii_D17_random_nodev.pdf \
             [g]=fig2c_Timestamp.pdf        [h]=fig2d_FitBit.pdf)

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    KEYS="$*"
else
    KEYS=$ALL_KEYS
fi

PIDS=()
for key in $KEYS; do
    dataset=${EXP_DATASET[$key]}
    devs=${EXP_DEVS[$key]}
    disease=${EXP_DISEASE[$key]}
    pop=${EXP_POP[$key]}
    pool=${EXP_POOL[$key]}
    sample=${EXP_SAMPLE[$key]}
    output=${EXP_OUTPUT[$key]}
    log="runs/log_noise_${key}.txt"

    echo "[$key] Starting: $dataset devs=$devs disease=$disease pop=$pop pool=$pool sample=$sample -> $output (log: $log)"
    uv run python Noise.py "$dataset" "$devs" "$disease" "$pop" "$pool" "$sample" "$output" > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/log_noise_*.txt"
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
    echo "$FAILED experiment(s) failed. Check runs/log_noise_*.txt for details."
    exit 1
fi
