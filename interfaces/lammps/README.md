# ML-SUS2 LAMMPS Interface

## Current superlinear models: self-contained CPU interface

Use [`sus2/embedded`](README.superlinear-embedded-cpu.md). Its parser, model
evaluation, radial tables, MPI support, installer and Intel make targets are
all provided here in the **superlinear model-code repository**:

```text
ML-SUS2/pair_sus2_embedded.{cpp,h}
ML-SUS2/sus2_sl_*.{cpp,h}
MAKE/OPTIONS/Makefile.ml_sus2_embedded*
scripts/install_superlinear_embedded_cpu.sh
examples/static-carbon/in.diamond
```

Install with the dedicated script, not the historical installer below:

```bash
# From the superlinear repository root; use a separate LAMMPS source tree.
bash interfaces/lammps/scripts/install_superlinear_embedded_cpu.sh /path/to/lammps
make -C /path/to/lammps/src ml_sus2_embedded_avx2 -j4
```

See the [Static C input and element mapping](README.superlinear-embedded-cpu.md#static-carbon-example).
The numbered form is `pair_style sus2/embedded model.sus2 radial_mode direct`
followed by `pair_coeff * *`: type i maps to native species i-1, not by mass or
comments. [Direct](examples/static-carbon/in.direct) and [Table](examples/static-carbon/in.table)
templates read user-supplied `data.in` and `model.sus2`. Reinstall and rebuild
for this syntax addition; kernel mathematics are unchanged. Explicit symbol
mappings remain available.
No external SUS2 inference archive or training/Python runtime is required.
The sections below describe the **historical SUS2-SH interfaces**, not the
installation or model support of `sus2/embedded`.

## Historical SUS2-SH interfaces

This directory is the maintained LAMMPS interface snapshot for SUS2-SH models.
It was refreshed from the server runtime tree:

```text
/work/phy-weigw/apps/lammps-10Dec2025
```

The runtime pair styles remain:

```text
sus2mtp
sus2mtp/kk
sus2mtp/kk/device
sus2mtp/kk/host
```

Use `ML-SUS2` as the package name. Do not use the old `ML-SUS2-MTP`
name for new builds.

This is the spherical-harmonic line. It reads SUS2-SH model files with the SH
metadata and evaluates real spherical-harmonic angular channels. The original
moment-tensor LAMMPS interface remains in the `SUS2-MLIP` repository. Both
interfaces intentionally keep the same user-facing `sus2mtp` pair-style names
so existing LAMMPS input files need only point to the appropriate model file and
installed binary.

## Layout

```text
ML-SUS2/                 CPU base interface sources
ML-SUS2-KK/              Kokkos/CUDA override sources
MAKE/OPTIONS/            Intel CPU make targets used on the server
scripts/install_into_lammps.sh
```

The top-level `sus2-interface-20260410.tar.gz` archive contains the same
files for users who prefer the historical single-package distribution.

## Install Into A LAMMPS Tree

Start from a clean LAMMPS source tree, then run:

```bash
cd /path/to/SUS2-SH/interfaces/lammps
scripts/install_into_lammps.sh /path/to/lammps
```

The script copies:

- `ML-SUS2/*.cpp` and `ML-SUS2/*.h` into both `LAMMPS_ROOT/src/ML-SUS2/`
  and `LAMMPS_ROOT/src/`.
- `ML-SUS2-KK/pair_sus2_mtp_kokkos.{cpp,h}`,
  `ML-SUS2-KK/sus2_mtp_gate_cache_layout.h`, and
  `ML-SUS2-KK/sus2_mtp_rank_phase_layout.h` into
  `LAMMPS_ROOT/src/KOKKOS/`.
- `MAKE/OPTIONS/Makefile.ml_sus2_avx2` and
  `MAKE/OPTIONS/Makefile.ml_sus2_avx512_skx` into
  `LAMMPS_ROOT/src/MAKE/OPTIONS/`.
- `ML-SUS2` into the standard package list in
  `LAMMPS_ROOT/cmake/CMakeLists.txt`, if it is not already present.

The direct copy into `LAMMPS_ROOT/src/` is intentional: it makes the old
LAMMPS make workflow reliable even if the downstream tree does not have custom
package install glue for `ML-SUS2`.

## CPU Build

On the maintained server, the CPU binary is selected through this wrapper:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

The wrapper dispatches to AVX2 or AVX512 builds. For a fresh tree after
running the install script:

```bash
cd "$LAMMPS_ROOT/src"
make ml_sus2_avx2 -j 16
make ml_sus2_avx512_skx -j 16
```

The Intel make targets expect Intel MPI and oneAPI MKL variables such as
`MKLROOT` to be available.
For branch benchmarking, `make ml_sus2_avx2_noipo -j 16` is also provided as a
faster exact CPU build target.

## GPU / Kokkos Build

The maintained server GPU binary is:

```text
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

Preferred A100 build modules on the server:

```bash
module load cuda/12.4 nvhpc/22.11 gcc/11.2.0 cmake/3.25.2
export OMPI_CC=gcc
export OMPI_CXX=g++
```

Recommended CMake shape:

```bash
cd "$LAMMPS_ROOT"
mkdir -p build-sus2kk
cd build-sus2kk
cmake \
  -C ../cmake/presets/basic.cmake \
  -D BUILD_MPI=on \
  -D BUILD_OMP=off \
  -D PKG_ML-SUS2=on \
  -D PKG_KOKKOS=on \
  -D Kokkos_ENABLE_SERIAL=on \
  -D Kokkos_ENABLE_CUDA=on \
  -D Kokkos_ENABLE_CUDA_LAMBDA=on \
  -D Kokkos_ARCH_AMPERE80=on \
  -D KOKKOS_PREC=double \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D WITH_JPEG=OFF \
  -D WITH_PNG=OFF \
  ../cmake
cmake --build . -j 16
```

Before reusing a LAMMPS source tree for GPU CMake after an old make build,
clean stale make artifacts first:

```bash
make -C "$LAMMPS_ROOT/src" no-all purge
```

## Runtime Syntax

CPU:

```lammps
pair_style sus2mtp p.sus2 radial_mode table tabstep 1.0e-4
pair_coeff * *
```

GPU:

```lammps
pair_style sus2mtp/kk p.sus2 chunksize 129600 radial_mode table tabstep 1.0e-4
pair_coeff * *
```

`radial_mode` is optional and accepts `auto`, `table`, or `direct`. The default
`auto` mode preserves the historical behavior: `_lmp` radial-basis names use
the table and the corresponding names without `_lmp` use direct evaluation.
For a table-capable `_lmp` model, an explicit override lets that same model file
be benchmarked or run through direct evaluation without editing its metadata,
for example:

```lammps
pair_style sus2mtp/kk p.sus2 chunksize 129600 radial_mode direct
```

`tabstep` is optional and defaults to `1.0e-4 A`; it is used only in table
mode. Explicit table mode requires an `_lmp` radial basis. The table is built
only for species pairs that appear in the active simulation atom types.

Release3 SUS2-SH models may opt into the smooth polynomial cutoff by storing

```text
radial_cutoff_type = polynomial
radial_cutoff_p = p
```

immediately after `radial_basis_size`.  `p` is optional and defaults to `5`;
valid values are `1..64`.  If `radial_cutoff_type` is absent, both CPU and
Kokkos pair styles retain the historical `(r-r_cut)^2` cutoff exactly, so old
models remain binary-compatible.  The polynomial is zero at and above
`r_cut`, and its value and first derivative use the same stable factored
formula as the standalone release3 executable.  Direct and table radial modes
both honor the model metadata.

The current table path supports:

```text
RBChebyshev_sss_lmp
RBChebyshev_sss_rational_lmp
RBLaguerre_log1p_lmp
RBLaguerre_log1p_pos_lmp
RBJacobi_sss_lmp
```

Other radial basis names are parsed by the CPU side when implemented in the
SUS2 interface sources. Kokkos direct evaluation currently supports
`RBChebyshev_sss[_lmp]`, `RBLaguerre_log1p[_lmp]`, and
`RBLaguerre_log1p_pos[_lmp]`; unsupported explicit direct requests fail with a
diagnostic instead of silently selecting a different formula. The same check
applies when `auto` resolves a non-`_lmp` model to direct evaluation.

CPU `pair_style sus2mtp` supports release additive-node two-layer-gate models,
including `two_layer_gate_edge_l1` / full-L1 metadata. Kokkos pair styles
support no-gate SUS2-SH models, plain release additive-node two-layer-gate
models, and `two_layer_gate_edge_l1` / full-L1 models. Center two-layer-gate
models are still CPU-only.

Release3 model-topology support is:

| Model topology | CPU `sus2mtp` | Kokkos `sus2mtp/kk` |
| --- | --- | --- |
| historical full-`K` | yes | yes |
| outer `K` rank | yes | yes |
| gate + outer `K` rank | yes | yes |
| compatible single-layer gate readout | yes | yes |
| multi-layer gate | no | no |

SUS2-SH models may use `scaling_map=K`, `L`, or `LK` in both CPU and
Kokkos pair styles, for direct and table radial evaluation.  With the
serialized `mu=k*(lmax+1)+l` layout, `K` selects block `k`, `L` selects block
`l`, and `LK` selects the full `mu` block.  `L` therefore shares only the
pairwise `scal/s` transform across radial `k` at fixed angular `l`; it does
not collapse the distinct radial-basis channels.

The supported readout form is the single-layer neighbor-site gate with the
first-layer scalar energy head. Readout is incompatible with outer/gate rank,
residual, edge-L1, and multi-layer-gate modes. Multi-layer gate models must use
the standalone runtime. The standalone
`two_layer_gate_scaling_mode=decoupled` form is also unsupported by the CPU and
Kokkos pair styles.

For large Kokkos runs, the final stride-guard overlay reuses storage only when
the device views are layout- and stride-compatible; every other layout takes
the separate-workspace fallback. This is an exact storage optimization, not a
change to the model mathematics. With the rank-two `_lmp` table model and
`chunksize=1054560`, the measured one-A100 peak fell from 51,147 MiB to
30,831 MiB. The four-GPU million-atom run reduced the largest per-rank peak
from 13,236 MiB to 8,158 MiB. Its maximum force difference was
`2.515e-14 eV/angstrom`, with energy-per-atom and stress differences of
`8.831e-16 eV/atom` and `2.116e-10 bar`, respectively. A 1,000-step ABBA
comparison measured 1,588.3750 s for the final overlay and 1,589.8275 s for
its immediate predecessor.
