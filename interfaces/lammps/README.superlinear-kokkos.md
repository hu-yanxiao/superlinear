# superlinear Kokkos interface

`pair_style sus2/embedded/kk` is the new device backend for the current native
superlinear model format. It shares the standalone model reader and input
mapping with [sus2/embedded](README.superlinear-embedded-cpu.md). The historical
`sus2mtp/kk` implementation targets a different SUS2-SH format and is not this
backend. No production executable is replaced by these sources or installer.

The implementation covers Static/plain, Effect/legacy neighbor gate, Pure-W,
and Adaptive coefficient families, with their serialized product graphs,
shared/species readouts, independent gate radial coefficients/warps, optional
joint readout and ZBL. Adaptive includes full/shared-n projections,
none/tanh activation, shared/species amplitudes and the supported tensor
signal channels. It rejects unsupported historical multi-layer/residual/
edge-L1 gate extensions at setup. Global virial and per-atom energy are
implemented; per-atom/centroid virial is unavailable.

## Build and run target

Use an **isolated, clean LAMMPS 10 Dec 2025 source tree**, with its bundled
Kokkos **4.7.1**. Do not reuse the maintained CPU source tree, old-make object
directories, a historical `sus2mtp` install, or another Kokkos build cache.
The [dedicated installer](scripts/install_superlinear_embedded_kokkos.sh)
checks this boundary and installs only the standalone CPU units plus the new
Kokkos pair/device headers:

```bash
bash interfaces/lammps/scripts/install_superlinear_embedded_kokkos.sh /path/to/isolated-lammps
```

The tested NVIDIA route uses CUDA 12.4, GCC 11.2, CUDA-aware MPI and A100
(`AMPERE80`). Select that toolchain before configuring a **new** build directory:

```bash
export OMPI_CC=gcc OMPI_CXX=g++ NVCC_WRAPPER_DEFAULT_COMPILER=g++
cmake -S /path/to/isolated-lammps/cmake -B /path/to/new-cuda-build \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_C_COMPILER=gcc \
  -D CMAKE_CXX_COMPILER=/path/to/isolated-lammps/lib/kokkos/bin/nvcc_wrapper \
  -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
  -D MPIEXEC_EXECUTABLE="$(command -v mpirun)" \
  -D BUILD_MPI=ON -D BUILD_OMP=OFF -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=OFF \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D Kokkos_ARCH_AMPERE80=ON -D KOKKOS_PREC=double
cmake --build /path/to/new-cuda-build --target lmp --parallel 8
```

On the maintained cluster, compilation and tests belong in a GPU allocation.
The installer only copies sources; installation alone is not a CUDA build or
a qualification result. Other GPU architectures need their matching Kokkos
architecture setting and separate validation.

A one-GPU invocation is:

```bash
mpirun -n 1 /path/to/isolated-build/lmp -k on g 1 -sf kk \
  -pk kokkos neigh half newton on comm device -in in.superlinear
```

Use `newton on` in the input, and explicitly select the new pair:

```lammps
pair_style sus2/embedded/kk radial_mode table tabstep 0.001 table_max_mb auto
pair_coeff * * model.sus2
```

The numbered and explicit `model_elements` mappings are the same as the CPU
interface. Model metadata selects the family. CUDA-aware MPI may use device
communication; the framework's staged communication path is also implemented
for feature/adjoint payloads. Executed multi-rank coverage is listed below;
it is not a claim for arbitrary MPI configurations.

## Optional execution variants

These environment switches accept only `0` or `1`, default to `0`, and must be
set before launching MPI. They change execution layout, not model parameters.

| Switch | Effect when `1` |
| --- | --- |
| `SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE` | Reuse a bounded neighbor geometry/harmonic tile across channels. |
| `SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE` | Separate Gate, Message and OuterRaw forward kernels. |
| `SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE` | Reduce reverse edge contributions within an owned center. |
| `SUS2_SL_KOKKOS_GENERIC` | Use the generic device reference path; override the other three switches. |
| `SUS2_SL_KOKKOS_TABLE_CONTEXT` | Reuse per-neighbor frozen-table routing and Hermite weights in the optimized raw forward/reverse paths. |
| `SUS2_SL_KOKKOS_TABLE_PACKED` | Pack immutable frozen knots in unpadded tiles of up to four channels. |

Startup prints the four effective `SUS2_KOKKOS_*` values. Role specialization
coexists with atom-by-product-row graph scheduling and does not whitelist
historical channel dimensions. Choose variants from matched measurements:
a faster Carbon setting is not automatically a faster FeCN6 setting. No
universal fastest combination is selected automatically.

The two Table switches also default to `0`; neither changes Direct evaluation.
Startup prints their requested values; `SUS2_KOKKOS_TABLE_ACTIVE` reports actual
activation after table setup. Context reuse selects separate Gate/OuterRaw
forward kernels even if the role switch is `0`. Generic reference scheduling
does not use contexts, but can read packed tables. Compare context variants
with role specialization enabled in both arms to isolate lookup reuse.
Packing preserves table admission, knot precision/count, deduplication and the
configured byte cap. Contexts are bounded per-neighbor tiles or thread-local
state, never an all-edges radial cache. Dynamic source coefficients are not
frozen, and generic basis-vector lookup is unchanged.
Forward contexts use typed structure-of-arrays shared storage with neighbors
contiguous. Packed frozen radial producers assign adjacent lanes to adjacent
channels; message/basis producers retain their original scheduling. Neighbor
accumulation and the force chain retain their original order.

## Radial paths and memory

`radial_mode direct` evaluates native formulas on the selected device.
`table` and `auto` build and qualify cubic Hermite value/derivative knots on
the host at setup or cache rebuild, then upload immutable device handles and
knots. Both use the derivative of the same interpolant. The 36 native basis
families retain their current warp, cutoff and historical derivative semantics;
`min_dist` defines the table domain, not a universal hard lower cutoff.

Fixed Static/Effect and inner-gate contractions receive small scalar tables
first. Pure-W distance mixing and Adaptive outer coefficients use basis tables;
geometry-dependent Adaptive coefficients are never frozen. Legacy models with
a full distance mixer also request its basis tables. Rejected/unavailable
tables, nonfinite samples, capacity limits, and radii outside the table domain
fall back to **device Direct**. Device kernels do not call the CPU evaluator.
No fixed local B-array limits general basis widths; Shapeev retains its native
12-function coefficient storage limit.

`table_max_mb auto` bounds resident knot storage per MPI rank by 512 MiB;
on CUDA it uses `min(512 MiB, free_device_bytes/8)` measured during model setup.
An explicit nonnegative cap is used as supplied and is never enlarged. Tables
are pruned to the global union of species actually present in owned atoms and
ordered pairs between those species, rather than all serialized model types.
A changed species set rebuilds the cache. The knot cap does not bound graph,
feature, adjacency, parameter or communication storage.

`SUS2_KOKKOS_RADIAL` reports resident tables/bytes, hits, domain/capacity/invalid
Direct fallbacks and engine memory. Generic counters count basis-vector
requests at n=0; frozen counters count scalar contraction requests. By default
only the first force call after cache setup is counted, so contended global
counter updates do not distort ordinary timed steps. `profile_every N` enables
additional sampling. `counted_force_calls` and `counting=sampled` state this
scope; the counts are not totals over the whole trajectory or GPU qualification.

## Ownership and communication

The pair copies the full owned-center neighbor list into a device CSR while
retaining skin neighbors. Every force evaluation checks the current distance
against the physical cutoff; prefiltering the CSR to a previous timestep's
force cutoff would miss entering pairs. Neighbor rebuilds, sorting/migration,
layout changes and model reloads invalidate the cached adjacency.

The staged chain is inner graph → forward signal exchange → owned outer
energies and reverse derivatives → reverse **signal-adjoint sum** → inner
reverse derivatives. Feature/adjoint buffers are distinct from atom forces.
Coordinate derivatives atomically accumulate both endpoints, including ghost
force slots; LAMMPS' standard atom-force reverse communication subsequently
returns those contributions to their owners. Global energy/virial reductions
count owned centers. `SUS2_KOKKOS_COMM` force exports are ghost slots handed to
LAMMPS, not a measured count of completed MPI force messages.

Product graphs run by dependency level with atom-by-product-row parallelism;
reverse rows group contributions by destination. Graph-node arrays use an
atom-contiguous layout for those graph launches. The optional forward-tile
variant shares a bounded 16-neighbor geometry/harmonic tile across channels.
Reverse kernels combine channel derivatives before endpoint updates; the
optional center-reduction variant further groups edge contributions. These
variants are opt-in, not the default layout or evidence that every model shape
has already been performance tuned.

LAMMPS uses integer communication counts and grows pair buffers by 1.5. The
adapter checks its callback payloads and its force-call layout against half
the signed-integer range. LAMMPS can allocate initial border buffers before
these checks; extreme halo capacities remain an upstream integer-storage
limit, not a qualified configuration or a whitelist of model channel sizes.

## Executed validation and scope

The standalone [radial test](../../tests/radial_test_device.cpp) passed with
the exact bundled Kokkos 4.7.1 **Serial** backend: all 36 basis families,
legacy/polynomial cutoff settings, B=12/65 (Shapeev B<=12), modern B=1,
k=0..5, fixed/dynamic contractions, independent gate coefficients/warps,
pair/K/L/LK maps, endpoints, domain/nonfinite/capacity fallbacks and cache
reconfiguration. Direct/CPU and Table/CPU-Table comparisons use
`3e-12*(1+abs(reference))`; Table/Direct errors are recorded separately.
Table admission preserves the CPU midpoint checks, not a global error bound.

The [kernel comparison](tests/kokkos_kernel_parity.cpp),
[native-oracle MPI qualification](tests/kokkos_qualification.py) and
[CPU/GPU benchmark harness](tests/kokkos_benchmark.py) are separate controls,
not the evidence of a completed campaign by themselves.

The current isolated A100 binary (`993ce7ff…`) passed a 368-run role-focused
CUDA/MPI raw replay, including MPI 1/2/4 and all eight role/tile/reverse
combinations. Its 100-case continuous Direct check also passed the declared
finite acceptance policy. That check uses plain L1/K6/body-3 fixtures, not
every model shape. The original CPU comparison remains 73 PASS, 7 FAIL and
20 unsupported; the separate fixed high-precision budgets accept the 40
Shapeev / `RBChebyshev_s_lmp` cases without rewriting those verdicts.

The same binary subsequently passed the complete recorded model/option matrix:
1,963 core plus 486 extension positive launches and six expected collective-safe
rejections, using CUDA and MPI 1/2/4. Another 163 smoke positives and six smoke
rejections are counted separately. A postrun read-only replay checked the raw
E/F/global-virial outputs, native binding, effective flags, GPU identity and
source receipts without rewriting the original reports.

The [evidence index](tests/kokkos_evidence_20261025.json) binds those results,
the separate finite Direct100 acceptance and the final performance measurements
to their sources and hashes. The full-model report retains its conservative
`MODEL_SCOPE_PASS_RADIAL_UNRESOLVED` label; the independently accepted finite
Direct100 contract is recorded separately, not used to relabel old strict
failures. These finite checks do not prove arbitrary inputs, high-order
Table100, long-time NVE conservation or untested memory capacities. The native
legal limits include L=0..4, tensor order <=2 and <=L, and Shapeev B<=12;
channel dimensions are runtime values, not a 2/4/8 whitelist. No production
executable has been replaced.

Campaign-specific role/AB scripts retain their original binary/hash guards;
they are not interchangeable reproduction commands for this candidate. Use
the separately identified sealed campaign runners and archives, not edited
guards in an older script. The index does not reclassify other historical
strict-radial failures.

## Measured throughput

One A100-SXM4-80GB versus 120 physical CPU cores on one `1t81c` node. Both
backends use the same frozen models, geometry, NVE input and radial mode;
warmup is excluded and each value uses three matched Loop-time repeats.
Speedup is `median CPU120 seconds / median GPU1 seconds`: below 1 means the
CPU is faster. These results describe the three frozen trained models, not
every model with the same family name.

| Model | Atoms | Direct speedup | Table speedup |
| --- | ---: | ---: | ---: |
| Carbon Static | 1,000 | 1.267 | 0.677 |
| Carbon Static | 9,600 | 1.068 | 0.590 |
| Carbon Static | 101,568 | 1.002 | 0.551 |
| Carbon Static | 1,000,000 | 0.922 | 0.483 |
| Carbon Effect | 1,000 | 1.267 | 0.665 |
| Carbon Effect | 9,600 | 1.060 | 0.556 |
| Carbon Effect | 101,568 | 0.984 | 0.526 |
| Carbon Effect | 1,000,000 | 0.923 | 0.466 |
| FeCN6 | 1,736 | 1.242 | 1.325 |
| FeCN6 | 13,888 | 1.173 | 1.029 |
| FeCN6 | 111,104 | 1.002 | 0.736 |

All 66 GPU windows completed and their original logs were replayed. The
independent cross-check reproduced all 22 groups' medians and ratios. Direct
is near CPU120 throughput at the larger sizes; Carbon's CPU Table path remains
faster. Table is not an automatic GPU speedup, and FeCN6's GPU Direct path is
faster than GPU Table at the larger tested sizes.

The measured Carbon settings are `GENERIC=0`, `FORWARD_RADIAL_TILE=1`,
`REVERSE_CENTER_REDUCE=0`, `FORWARD_ROLE_SPECIALIZE=1`; FeCN6 uses the same
settings except `FORWARD_RADIAL_TILE=0`. Apply the full `SUS2_SL_KOKKOS_`
prefix to each switch above. They were selected from the matched 100k-scale
experiment, not asserted to be universal optima; the compiled defaults remain
all zero.

The largest sampled whole-GPU peaks were 6,809 MiB (Static), 9,475 MiB
(Effect), and 4,977 MiB (FeCN6), including each card's baseline allocation.
These are sampled framebuffer values, not exact process allocation. CPU
total-rank peak RSS remains unknown. Secondary Pair timings are LAMMPS
rank-mean wall times, not CUDA kernel timings.

### Numerical acceptance

Keep the original CPU comparisons and their strict verdicts. For ill-conditioned
high-order radial functions, also compare energy, site energies, forces and
global virial to an independent high-precision mathematical reference. Declare
and propagate the rounding budget before examining backend errors; do not fit
a tolerance to an observed maximum or globally relax the original tests.
Coefficient bits, geometry, harmonic normalization, FMA evaluation and library
function errors must be accounted for. A finite tested library-function bound
is not a guarantee for all future inputs. A Serial witness or an independently
compiled CUDA probe does not establish the arithmetic path of another binary.

Report continuous Direct, Hermite Table and historical native linear-table
contracts separately. In particular, a high-precision continuous reference for
`RBChebyshev_s_lmp` does not demonstrate equivalence to its historical native
linear-table runtime. A numerical budget check and a conventional absolute/
relative check against a rounded reference are distinct verdicts. Qualification
must name the tested binary, model, inputs, radial contract and scope; an
unexecuted case remains unexecuted.

The preimplementation review used the official
[NequIP/Allegro interface](https://github.com/mir-group/pair_nequip_allegro),
[SevenNet distributed implementation](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp),
and native [SNAP Kokkos kernels](https://github.com/lammps/lammps/blob/develop/src/KOKKOS/pair_snap_kokkos_impl.h).
Allegro supplies local GPU graph/ownership lessons; its strictly local
algorithm cannot establish correctness for superlinear neighbor-message
paths. The official `pair_nequip` is single-rank. SevenNet supplies the relevant
forward-feature/reverse-adjoint pattern, and SNAP illustrates device layouts,
dependency staging, fused derivatives and bounded workspaces. These design
references do not transfer another implementation's validation to this one.
