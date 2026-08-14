#!/usr/bin/env bash
# Run the shared exact compact-map LAMMPS parity harness for scaling_map=L.
set -Eeuo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
export SH_SCALING_MAP=L
exec "$script_dir/lammps_sh_scaling_map_k_parity.sh" "$@"
