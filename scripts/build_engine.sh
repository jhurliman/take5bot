#!/usr/bin/env bash
# Build the Rust engine's Python extension and place it at py/take5_engine.so
# so anything with `py/` on its PYTHONPATH can `import take5_engine`.
# Uses abi3 (py>=3.10), so one build works across Python versions.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --manifest-path engine/Cargo.toml -p take5-py

mkdir -p py
case "$(uname -s)" in
Darwin) src=engine/target/release/libtake5_engine.dylib ;;
*) src=engine/target/release/libtake5_engine.so ;;
esac
cp "$src" py/take5_engine.so
echo "Built py/take5_engine.so"
