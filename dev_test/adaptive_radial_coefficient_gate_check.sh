#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bin=${1:-./bin/mlp-sus2}

python3 dev_test/adaptive_radial_coefficient_gate_check.py "$bin"
