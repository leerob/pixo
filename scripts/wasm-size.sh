#!/usr/bin/env bash
set -euo pipefail

# Build the WASM artifact and report its size (raw and gzipped).
# If wasm-opt is available, also report an -Oz optimized artifact.
#
# Usage:
#   ./scripts/wasm-size.sh            # plain build and size report
#   ./scripts/wasm-size.sh --locked   # pass extra args to cargo build

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="$ROOT_DIR/target/wasm32-unknown-unknown/release/comprs.wasm"
OPT_ARTIFACT="$ROOT_DIR/target/wasm32-unknown-unknown/release/comprs.opt.wasm"

echo "==> Building WASM (release, feature=wasm)"
cargo build --release --target wasm32-unknown-unknown --features wasm "$@"

if [[ ! -f "$ARTIFACT" ]]; then
  echo "Artifact not found: $ARTIFACT" >&2
  exit 1
fi

report() {
  local file="$1"
  local label="$2"
  local size gz
  size=$(stat -c%s "$file")
  gz=$(gzip -c "$file" | wc -c | tr -d '[:space:]')
  printf "%-12s: %s bytes (gzipped: %s bytes)\n" "$label" "$size" "$gz"
}

echo "==> Size report"
report "$ARTIFACT" "raw"

if command -v wasm-opt >/dev/null 2>&1; then
  echo "==> Running wasm-opt -Oz"
  wasm-opt -Oz -o "$OPT_ARTIFACT" "$ARTIFACT"
  report "$OPT_ARTIFACT" "wasm-opt"
else
  echo "wasm-opt not found; skipping optimized artifact" >&2
fi
