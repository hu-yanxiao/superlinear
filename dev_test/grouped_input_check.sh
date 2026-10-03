#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_dir=$(mktemp -d "${TMPDIR:-/tmp}/superlinear-grouped-unit.XXXXXX")
"${CXX:-c++}" -std=c++11 -Wall -Wextra -Werror \
  src/mlp/grouped_input.cpp dev_test/grouped_input_check.cpp -o "$task_dir/check"
"$task_dir/check"
