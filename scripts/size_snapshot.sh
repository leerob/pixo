#!/usr/bin/env bash
set -euo pipefail

# Run the size_snapshot bench (harnessless) and capture output.
# Usage:
#   ./scripts/size_snapshot.sh          # run default
#   ./scripts/size_snapshot.sh -- --nocapture  # pass extra args after --

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT_DIR/target/size_snapshot.log"

echo "==> Running size_snapshot bench"
cargo bench --bench size_snapshot "$@" | tee "$LOG"

echo "==> Output saved to $LOG"
