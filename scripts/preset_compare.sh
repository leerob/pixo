#!/usr/bin/env bash
set -euo pipefail

# Run the preset_compare bench and capture output.
# Default: sample-size 10, warm-up 0.5s, measurement 2s, no plots.
# Usage:
#   ./scripts/preset_compare.sh
#   ./scripts/preset_compare.sh -- --sample-size 20    # pass extra args after --

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT_DIR/target/preset_compare.log"

DEFAULT_ARGS=(--sample-size 10 --warm-up-time 0.5 --measurement-time 2 --noplot)

echo "==> Running preset_compare bench"
cargo bench --bench preset_compare -- "${DEFAULT_ARGS[@]}" "$@" | tee "$LOG"

echo "==> Output saved to $LOG"
