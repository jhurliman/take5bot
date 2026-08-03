#!/usr/bin/env bash
# Rebuild the WASM engine package consumed by the web UI.
# simd128 is required: the transformer forward is 4x faster with the
# manual SIMD dot kernel (all evergreen browsers support WASM SIMD).
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build engine/take5-wasm --release --target web --out-dir ../../web/src/engine/pkg
rm -f web/src/engine/pkg/.gitignore
echo "Built web/src/engine/pkg"
