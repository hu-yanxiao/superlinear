#!/usr/bin/env bash
# Launch the native superlinear GPU backend with the accepted Table policy.
set -euo pipefail
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$task_script_dir/env-kokkos.sh" ]]; then
  source "$task_script_dir/env-kokkos.sh"
fi
task_binary=${SUS2_LAMMPS_KOKKOS_BINARY:-"$task_script_dir/lmp-kokkos.bin"}
if [[ ! -x "$task_binary" ]]; then
  printf 'GPU executable not found: %s\nSet SUS2_LAMMPS_KOKKOS_BINARY to the built LAMMPS executable.\n' "$task_binary" >&2
  exit 2
fi
export SUS2_SL_KOKKOS_TABLE_CONTEXT=${SUS2_SL_KOKKOS_TABLE_CONTEXT:-1}
export SUS2_SL_KOKKOS_TABLE_PACKED=${SUS2_SL_KOKKOS_TABLE_PACKED:-1}
export SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE=${SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE:-1}
export SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE=${SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE:-0}
export SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE=${SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE:-0}
export SUS2_SL_KOKKOS_GENERIC=${SUS2_SL_KOKKOS_GENERIC:-0}
exec "$task_binary" "$@"
