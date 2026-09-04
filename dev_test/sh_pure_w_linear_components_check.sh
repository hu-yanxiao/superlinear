#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 dev_test/sh_pure_w_linear_components_check.py "${PATCHED_BIN:-./bin/mlp-sus2}"
