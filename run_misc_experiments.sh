#!/usr/bin/env bash
set -euo pipefail

# Figure 4 (Ordered_FitBit) and Figure 12b (Ordered_Timestamp, selected_distribution=0)
# Usage:
#   ./run_misc_experiments.sh             # run both
#   ./run_misc_experiments.sh 4           # run only Fig 4
#   ./run_misc_experiments.sh 12b         # run only Fig 12b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

declare -A EXP_CMD EXP_OUTPUT
ALL_KEYS="4 12b"

EXP_CMD=(   [4]="Ordered_FitBit.py"           [12b]="Ordered_Timestamp.py 0")
EXP_OUTPUT=([4]=fig4_FitBit.pdf               [12b]=fig12b_Timestamp.pdf)

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    KEYS="$*"
else
    KEYS=$ALL_KEYS
fi

PIDS=()
for key in $KEYS; do
    cmd=${EXP_CMD[$key]}
    output=${EXP_OUTPUT[$key]}
    log="runs/log_misc_${key}.txt"

    echo "[$key] Starting: $cmd -> $output (log: $log)"
    uv run python $cmd "$output" > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/log_misc_*.txt"
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
    echo "$FAILED experiment(s) failed. Check runs/log_misc_*.txt for details."
    exit 1
fi
