# Native superlinear CPU interface (ML-SUS2)

`sus2/superlinear` loads the current native SUS2-SH model with `MLMTPR` and
uses that same core for inference. It does not translate models, append `_lmp`
to radial names, or reuse the historical LAMMPS model reader. The old
`sus2mtp` and `sus2mtp/kk` styles remain available for their existing models.

## Model combinations

Architecture is read from the model, not selected or approximated by LAMMPS
options. In particular, changing `radial_mode` does not convert Legacy into
Adaptive or change elemental scaling.

| Native model setting | CPU implementation |
| --- | --- |
| Static / no environment correction | Native static radial and angular graph |
| Legacy / Effect | Native source-environment channel modulation and its full chain rule |
| Pure-W channelwise | Native Pure-W graph and derivative kernels |
| Adaptive | Full or shared-n projection; shared-n rank retained; none/tanh activation; shared or source-element amplitude |
| Inner and outer products | Independent cross / same-channel choices, including mixed choices |
| Readout and radial coordinate scaling | Shared or species readout; L / K / LK / pair scaling; independent gate scaling when enabled |
| Additional Adaptive branches | Tensor messages, joint readout, and their combination |
| Angular/body limits | Native L and per-body-order truncations; no historical benchmark-shape admission rule |
| ZBL | Once-only pair contribution, including an outer cutoff larger than the ML cutoff |

The native parser still determines which combinations are legal. Historical
residual and multilayer gate architectures are explicitly rejected by the staged
API; they are not silently approximated. Supporting a parameter dimension is
not the same as having benchmarked every value: the regression matrix covers
nonzero representative combinations at L=1..4, and the frozen FeCN6 model tests
K8/R3/O8 separately. Add new production models to the real-MPI qualification
even when their architecture is already covered.

## Input

```lammps
units metal
atom_style atomic
newton on
read_data system.data
pair_style sus2/superlinear model_elements H,C,N,O,S,Cl,K,Fe radial_mode direct
pair_coeff * * frozen.sus2 H C N O S Cl K Fe
```

`model_elements` lists chemical symbols in **native model species order**
(native types 0, 1, ...). The `pair_coeff` list maps each LAMMPS type (1, 2,
...) onto one of those symbols; it may reorder or repeat symbols. No implicit
mapping based on first occurrence is performed. The list is mandatory when
the model has no atomic-number metadata. For ZBL models, native atomic numbers
supply the list; an explicit list must agree with it. `NULL` and hybrid subset
mappings are rejected.

The options are key/value pairs on `pair_style`:

| Option | Values | Default |
| --- | --- | --- |
| `model_elements` | Comma-separated model-order symbols | Required unless ZBL metadata supplies them |
| `radial_mode` | `direct`, `table`, `auto` | `direct` |
| `tabstep` | Positive finite grid spacing, in Angstrom | `0.001` |
| `table_max_mb` | Nonnegative per-rank knot-storage budget, in MiB | `64` |

`direct` is the default native analytic reference. `table` explicitly requests the
native table implementation. `table` and `auto` both use direct evaluation for
distances outside the supported domain, invalid/unqualified basis tables, or
tables exceeding the native memory budget. Runtime table statistics report
hits and these fallbacks. The table approximates
the radial basis only; native angular products, messages, readouts, derivatives,
and physical model coefficients remain unchanged. Table accuracy must be measured
against `direct` for the target model and distance range. No file rewrite is
necessary when switching modes. The native default table memory budget is 64 MiB.
The first force evaluation logs rank-zero table/fallback counters and the exact
forward/reverse communication widths; counters are cumulative for that model.
Table modes are manual options. They are not automatically faster than analytic
evaluation; qualify both accuracy and throughput for the actual model before
selecting one for a campaign.

## Parallel computation and supported observables

Every rank constructs neighborhoods for owned atoms from a **full** LAMMPS
neighbor list. Ghost neighborhoods remain empty. Native first-stage features
are forwarded to ghosts using the model's exact forward width; native outer
evaluation then builds energy, direct forces and ghost adjoints. An adjoint
reverse uses the independently sized native reverse width. The native final
stage applies owner-side adjoints. LAMMPS subsequently performs its normal
force reverse, including final forces accumulated on ghosts. No atom gathering
or native periodic-image construction is performed.

Total energy, native site energy (`compute pe/atom`), forces, and global virial
are supported. Global pressure and NPT use the native configurational virial
`W_ab = -sum_edges (dE/dr_a) r_b`, copied with the **same sign**, no volume
division, and LAMMPS ordering `xx yy zz xy xz yz`. LAMMPS adds kinetic pressure
and unit conversion. The native CFG `PlusStress` order is
`xx yy zz yz xz xy`; compare named components rather than raw six-column order.
The adapter does not invoke a second force-dot-position virial calculation.

Per-atom virial and centroid stress are rejected until their decomposition is
implemented and verified. Consequently `compute stress/atom`,
`compute centroid/stress/atom`, and heat-flux workflows requiring those tensors
are unavailable. `newton pair on`, `units metal`, and neighborhoods for every
owned atom are required. Molecular atom styles must use
`special_bonds lj 1 1 1` so bonded exclusions do not remove model neighbors.
Hybrid pair styles and `neigh_modify include/exclude` are rejected: omitting
even one type pair can change a many-body environment. Rank-local evaluation
or model-loading errors abort the communicator rather than entering a collective
error barrier while other ranks are exchanging features.
Restart files do not embed the potential: reissue `pair_style`/`pair_coeff`.

## Build and installation

The native core lives in a separate static library and private header tree.
Only the new pair source is installed in LAMMPS `src/`; native files such as
`configuration.cpp`, `radial_basis.cpp`, and `utils.h` never overwrite LAMMPS
files. The native library is deliberately built without `MLIP_MPI`; all
distributed communication belongs to the LAMMPS communicator, including in
partition runs. It links the same native reader and inference objects used by
the current model implementation and a BLAS implementation.

On phy-weigw, retain the documented Intel MPI/compiler setup and load
`gcc/11.2.0` for the GNU C++ runtime. Use the maintained CPU old-make AVX2 route:

```bash
cd /path/to/SUS2-SH
module load gcc/11.2.0
CXX=mpiicpc CXXFLAGS='-O3 -xCORE-AVX2 -fp-model precise -fPIC -std=c++17' JOBS=16 \
  interfaces/lammps/scripts/build_superlinear_native.sh /absolute/native-build
interfaces/lammps/scripts/install_superlinear_cpu.sh \
  /work/phy-weigw/apps/lammps-10Dec2025 /absolute/native-build
make -C /work/phy-weigw/apps/lammps-10Dec2025/src ml_sus2_superlinear_avx2 -j 16
```

The native build detects `MKLROOT` and uses MKL ILP64, matching the maintained
Intel LAMMPS target. Without MKL it uses CBLAS with `-lblas` (macOS: Accelerate);
override `SUS2_NATIVE_BLAS_LIBS` when necessary. Use the same compiler family,
C++ ABI and BLAS integer ABI for the library and adapter. The build performs a
link/run smoke check before producing installable configuration. The output
also records compiler/BLAS preprocessor settings for the LAMMPS target.

The new installer adds the pair style and isolated library and preserves
historical CPU/Kokkos source files. The historical `install_into_lammps.sh`
continues installing only `sus2mtp`; it does not install this new adapter.
This CPU installation does not enable a GPU or Kokkos implementation of
`sus2/superlinear`. Follow the main interface README's CPU-artifact cleanup
instructions before a later GPU CMake build. Native source licensing remains
governed by the repository's `LICENSE`.

Qualification requires native `calc-efs` versus LAMMPS comparisons on identical
geometries for energy, all force components and six global virial components,
with direct/table compared separately; finite differences under strain establish
the pressure sign. MPI checks must include atoms crossing domain boundaries,
multiple decompositions, triclinic cells, and dynamics/NPT. A successful build
alone is not evidence that those numerical checks passed.
