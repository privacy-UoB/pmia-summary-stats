#!/bin/sh
set -eu

# 4 D17 Ordered_Noise experiments
# Usage:
#   ./run_ordered_experiments.sh          # run all 4
#   ./run_ordered_experiments.sh a b      # run only experiments a, b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

mkdir -p runs

# Experiment definitions: key|disease|metric|pool_idx|sample|output
EXPERIMENTS="
a|D17|LLR|0|20|D17_LLR_random.pdf
b|D17|L1|0|20|D17_L1_random.pdf
c|D17|LLR|1|_|D17_LLR_case.pdf
d|D17|L1|1|_|D17_L1_case.pdf
"

# Use CLI args as subset, or run all
if [ $# -gt 0 ]; then
    SELECTED="$*"
else
    SELECTED="a b c d"
fi

PIDS=""
LAUNCHED=0
for exp in $EXPERIMENTS; do
    key=$(echo "$exp" | cut -d'|' -f1)
    # Skip if not selected
    case " $SELECTED " in
        *" $key "*) ;;
        *) continue ;;
    esac
    disease=$(echo "$exp" | cut -d'|' -f2)
    metric=$(echo "$exp" | cut -d'|' -f3)
    pool=$(echo "$exp" | cut -d'|' -f4)
    sample=$(echo "$exp" | cut -d'|' -f5)
    output=$(echo "$exp" | cut -d'|' -f6)
    log="runs/log_ordered_${key}_${disease}_${metric}_${pool}.txt"

    echo "[$key] Starting: $disease $metric pool=$pool sample=$sample -> $output (log: $log)"
    uv run python Ordered_Noise.py "$disease" "$metric" "$pool" "$sample" "$output" > "$log" 2>&1 &
    PIDS="$PIDS $!"
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "All $LAUNCHED experiments launched. PIDs:$PIDS"
echo "Monitor with: tail -f runs/log_ordered_*.txt"
echo ""

# Wait for all and report
FAILED=0
I=0
for pid in $PIDS; do
    I=$((I + 1))
    key=$(echo $SELECTED | cut -d' ' -f$I)
    if wait "$pid"; then
        echo "[$key] Done (PID $pid)"
    else
        echo "[$key] FAILED (PID $pid)"
        FAILED=$((FAILED + 1))
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
