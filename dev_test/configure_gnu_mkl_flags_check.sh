#!/usr/bin/env bash
# Configuration generation only: no compiler, MPI, or MKL installation needed.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/sus2-configure-flags.XXXXXX")
trap 'rm -rf "$task_tmp"' EXIT
mkdir -p "$task_tmp/bin"
# Autodetection may discover these names, but configure must never invoke them.
for name in mpicc mpicxx mpifc icc; do
    printf '#!/bin/sh\necho "unexpected compiler execution" >&2\nexit 99\n' > "$task_tmp/bin/$name"
    chmod +x "$task_tmp/bin/$name"
done
export PATH="$task_tmp/bin:$PATH"
export MKLROOT="$task_tmp/mock-mkl"
for compiler in gnu intel; do
    for mode in mpi serial; do
        work="$task_tmp/$compiler-$mode"
        mkdir -p "$work/make"
        args=("--compiler=$compiler" --blas=mkl "--prefix_fortran=$task_tmp/mock-fortran")
        if [[ "$mode" == serial ]]; then args+=(--no-mpi); fi
        (cd "$work" && bash "$repo/configure" "${args[@]}" > configure.log)
        config="$work/make/config.mk"
        grep -Fq -- "-I$MKLROOT/include" "$config"
        grep -Fq -- "-L$MKLROOT/lib/" "$config"
        grep -Fq -- '-lmkl_rt' "$config"
        grep -Fq -- '-DMLIP_INTEL_MKL' "$config"
        if [[ "$mode" == mpi ]]; then
            grep -Fq -- '-DMLIP_MPI' "$config"
        elif grep -Fq -- '-DMLIP_MPI' "$config"; then
            echo "MPI leaked into serial configuration" >&2; exit 1
        fi
        if [[ "$compiler" == gnu ]]; then
            grep -Fxq 'CXXFLAGS += -O3 -march=native -ffast-math -std=c++11' "$config"
            grep -Fxq "FORTRAN_DIR = $task_tmp/mock-fortran" "$config"
            grep -Fq -- '-lgfortran' "$config"
            if grep -Eq -- '-xHost|-ipo|-fp-model|-qopt|-qoverride|-fimf|-no-prec-div|-lifcore|^\$FORTRAN_DIR' "$config"; then
                echo "Intel flags or malformed Fortran assignment leaked into GNU MKL" >&2; exit 1
            fi
            if [[ "$mode" == mpi ]]; then grep -Fxq 'CXX_EXE = mpicxx' "$config";
            else grep -Fxq 'CXX_EXE = g++' "$config"; fi
        else
            grep -Fxq 'CXXFLAGS += -O3 -xHost -ipo -no-prec-div -fp-model fast=2 -qoverride-limits -qopt-zmm-usage=high  -qopt-mem-layout-trans=3  -qopt-streaming-stores=auto -fimf-use-svml=true -std=c++11 -qopt-report=5' "$config"
            grep -Fq -- '-lifcore' "$config"
            if grep -Eq -- '-march=native|-ffast-math|-lgfortran' "$config"; then
                echo "GNU flags leaked into Intel MKL" >&2; exit 1
            fi
        fi
        # Parse the generated make fragment too; do not run any build recipe.
        make -s -f "$config" -f - config-probe <<'MAKE'
.PHONY: config-probe
config-probe:
	@:
MAKE
        printf 'PASS %s %s MKL configuration\n' "$compiler" "$mode"
    done
done
