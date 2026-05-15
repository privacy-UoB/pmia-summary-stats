#!/usr/bin/env bash
set -euo pipefail

# Stratified miRNA experiments:
#   Fig 13 (Ordered_Timestamp.py, timestamp / D2 lung cancer)
#     13a: only disease-related miRNAs
#     13b: same-size random sample of non-disease-related miRNAs
#   Fig 7  (Ordered.py, case pool, stratified -- both subsets on the same panel)
#     7a: D1  (Wilms tumor)
#     7b: D17 (Renal cancer)
#     7c: D14 (Ovarian cancer)
# Usage:
#   ./run_fig7_fig13_experiments.sh                 # run all 5
#   ./run_fig7_fig13_experiments.sh 7a 13b          # run a subset
#   SEED=123 ./run_fig7_fig13_experiments.sh        # override seed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

uv sync

mkdir -p runs

# Optional seed override (defaults to 42 in each Python script).
SEED_ARG=${SEED:+--seed $SEED}

# Two flavours of invocation:
#   Ordered_Timestamp.py:  [selected_distribution] [output.pdf] [stratify_mode]
#   Ordered.py:            <disease> <pool_idx> [sample] [output.pdf] [stratify]
declare -A EXP_SCRIPT EXP_ARGS EXP_OUTPUT
ALL_KEYS="13a 13b 7a 7b 7c"

EXP_SCRIPT=( [13a]=Ordered_Timestamp.py [13b]=Ordered_Timestamp.py \
             [7a]=Ordered.py  [7b]=Ordered.py  [7c]=Ordered.py)
# Positional args excluding the output filename (which is appended below).
EXP_ARGS=(   [13a]="0"             [13b]="0" \
             [7a]="D1 1 _"   [7b]="D17 1 _"   [7c]="D14 1 _")
# Stratify positional arg, supplied after the output path.
declare -A EXP_STRATIFY
EXP_STRATIFY=([13a]=diseased [13b]=non_diseased \
              [7a]=stratified [7b]=stratified [7c]=stratified)
EXP_OUTPUT=( [13a]=fig13a_D2_diseased.pdf \
             [13b]=fig13b_D2_non_diseased.pdf \
             [7a]=fig7a_D1_case_stratified.pdf \
             [7b]=fig7b_D17_case_stratified.pdf \
             [7c]=fig7c_D14_case_stratified.pdf)

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    KEYS="$*"
else
    KEYS=$ALL_KEYS
fi

# Validate any provided subset against the known keys.
for key in $KEYS; do
    if [ -z "${EXP_SCRIPT[$key]:-}" ]; then
        echo "Unknown key: $key" >&2
        echo "Valid keys: $ALL_KEYS" >&2
        exit 2
    fi
done

PIDS=()
for key in $KEYS; do
    script=${EXP_SCRIPT[$key]}
    args=${EXP_ARGS[$key]}
    stratify=${EXP_STRATIFY[$key]}
    output=${EXP_OUTPUT[$key]}
    log="runs/log_${key}_${output%.pdf}.txt"

    echo "[$key] Starting: $script $args $output $stratify (log: $log)"
    # shellcheck disable=SC2086 # intentional word-splitting of $args / $SEED_ARG
    uv run python "$script" $args "$output" "$stratify" $SEED_ARG > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/log_*.txt"
echo ""

# Wait for all and report
FAILED=0
i=0
for key in $KEYS; do
    if wait "${PIDS[$i]}"; then
        echo "[$key] Done (PID ${PIDS[$i]})"
    else
        echo "[$key] FAILED (PID ${PIDS[$i]})"
        FAILED=$((FAILED+1))
    fi
    i=$((i+1))
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "All experiments completed successfully."
    ls -la results/fig{7,13}*.pdf 2>/dev/null || echo "No PDF files found."
else
    echo ""
    echo "$FAILED experiment(s) failed. Check runs/log_*.txt for details."
    exit 1
fi
