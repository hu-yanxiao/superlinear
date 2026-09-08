#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}

if [[ -z "${SUS2SHPT_ROOT:-}" ]]; then
  if [[ -d "../superlinear-adaptive-pytorch-wt/sus2shpt" ]]; then
    export SUS2SHPT_ROOT="$(cd ../superlinear-adaptive-pytorch-wt && pwd)"
  else
    echo "SUS2SHPT_ROOT must point at the adaptive PyTorch source tree" >&2
    exit 2
  fi
fi

python_bin=${PYTHON:-python3}
exec "$python_bin" dev_test/adaptive_pytorch_profile_check.py "$bin" "${@:2}"
