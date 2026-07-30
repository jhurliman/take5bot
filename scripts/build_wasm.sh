#!/usr/bin/env bash
# Rebuild the WASM engine package consumed by the web UI.
set -euo pipefail
cd "$(dirname "$0")/.."
wasm-pack build engine/take5-wasm --release --target web --out-dir ../../web/src/engine/pkg
rm -f web/src/engine/pkg/.gitignore
echo "Built web/src/engine/pkg"
