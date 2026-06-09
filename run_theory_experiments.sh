#!/usr/bin/env bash
set -euo pipefail

# Figure 1 — theoretical minimum-error bound (Theorem 1).
# Deterministic, no datasets, runs in seconds.
#
# Usage:
#   ./run_theory_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install deps (uv sync is fast and cached)
uv sync

# Produces results/MinimumError_evolution_delta-n.pdf (Fig 1a)
#      and results/MinimumError_evolution_A-n.pdf     (Fig 1b)
uv run python fig_min_error.py

echo ""
echo "Figure 1 written to results/MinimumError_evolution_*.pdf"
