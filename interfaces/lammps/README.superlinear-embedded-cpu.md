# Self-contained superlinear LAMMPS CPU interface

`PairSUS2Embedded` / `pair_style sus2/embedded` is compiled directly into
LAMMPS, like the historical `PairSUS2MTP`. The model reader, radial functions,
spherical harmonics, product graphs, energy/force/global-virial evaluation and
MPI feature exchanges all live in this repository's `interfaces/lammps/ML-SUS2/`.
It does not call or link the training runtime, an external native inference
archive, Python, or an ASE calculator. No separate interface checkout is needed.

## Install and build

Use a separate LAMMPS source tree. The validated route is LAMMPS **10 Dec 2025**,
the old make workflow, Intel classic `mpiicpc`, Intel MPI, MKL and TBB, targeting
AVX2. This is not a claim of qualification with arbitrary LAMMPS versions or
compiler toolchains. For example, on the maintained cluster:

```bash
module load gcc/11.2.0 compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0
```

The GCC module supplies the compatible C++ runtime; `mpiicpc` remains the
compiler. On another installation, initialize the corresponding Intel
environment so that `mpiicpc` and `MKLROOT` are available.

From the **superlinear repository root**:

```bash
LAMMPS_ROOT=/absolute/path/to/lammps
bash interfaces/lammps/scripts/install_superlinear_embedded_cpu.sh "$LAMMPS_ROOT"
make -C "$LAMMPS_ROOT/src" ml_sus2_embedded_avx2 -j4
"$LAMMPS_ROOT/src/lmp_ml_sus2_embedded_avx2" -h
```

Check that `sus2/embedded` appears in the pair-style list. The installer copies
the package-local sources to both `src/ML-SUS2/` and `src/`, and its two make
targets to `src/MAKE/OPTIONS/`; no `make yes-ML-SUS2` is required. Source files
remain maintained under **this model-code repository**, not in a temporary
benchmark directory or only in the downstream LAMMPS installation.

The installer preserves historical `sus2mtp` source/build files. It refuses a
tree containing the older `pair_sus2_superlinear.cpp` native wrapper; use a
separate tree rather than mixing the two implementations. Reinstallation
updates the installed embedded files, so keep local edits in the source
repository. No canonical server binary is automatically replaced.

Although the embedded inference engine has no BLAS dependency, the preserved
Intel LAMMPS make target links MKL/TBB for its general LAMMPS configuration.
The make target therefore still needs those libraries.

### Maintained cluster installation

The stable superlinear CPU LAMMPS executable is:

```text
/work/phy-weigw/app/superlinear/lmp
```

Load the Intel/GCC environment above before launching it with `mpirun`.
Use this executable for both `radial_mode direct` and `radial_mode table`.
The corresponding interface sources and installation scripts remain under
`interfaces/lammps/` in the superlinear repository. The cluster source mirror is
`/work/phy-weigw/20260321_Test/superlinear-radial-coefficient-gate-work-codex`;
the installed LAMMPS sources are under `/work/phy-weigw/apps/lammps-10Dec2025`.

For an update, synchronize the committed interface files, rerun the dedicated
installer, and rebuild `ml_sus2_embedded_avx2`. Reinstallation refreshes the
embedded sources; normal `make` recompiles them while reusing unchanged
LAMMPS core objects. Retain the target's `-fp-model precise` override for
`pair_sus2_embedded.o` and `sus2_sl_*.o`, and the existing Intel link settings.
Do not switch precision or remove optimization flags simply to shorten a build.

Before replacing the stable entry, run the native-reference E/F/virial
qualification, Direct/Table and numbered/explicit input checks, MPI and model
reload checks, and the provided carbon examples. Install only the passing
binary, back up an existing entry, and replace it atomically. Release provenance
is recorded beside it in `LAMMPS_PRODUCTION.json` (source commit, file hashes,
binary hash and qualification reports). The installer itself never publishes
the binary. A new installation does not restart existing jobs.

This is a separate superlinear entry: the historical
`/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi`, native training and
GPUMD executables are not replaced by this workflow. Interface/docs synchronization
does not imply updating unrelated training sources in an older server checkout.

## Static carbon example

For a native-exported one-species carbon Static model named `model.sus2`
and a LAMMPS data file with atom type 1 = C, the key lines are:

```lammps
units metal
atom_style atomic
boundary p p p
newton on
read_data data.in
mass 1 12.0107
pair_style sus2/embedded model.sus2 radial_mode direct
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
run 0
```

These model/data filenames are user-supplied, not bundled trained weights.
Paths are resolved from the LAMMPS working directory. The model file selects
Static versus other supported architectures; there is no extra `static` flag.
Numbered mapping sends **LAMMPS type i to native model species i-1**. Neither
`Masses` nor data-file comments infer elements. The number of declared LAMMPS
types must not exceed the model species count: this selects only its prefix,
not an arbitrary subset. The user must match the model's native species order.

Alternatively put the filename in `pair_coeff`, without symbols:

```lammps
pair_style sus2/embedded radial_mode direct
pair_coeff * * model.sus2
```

Specify the filename in only one place, never both commands. For arbitrary
subsets/reordering, the existing explicit syntax is unchanged:

```lammps
pair_style sus2/embedded model_elements C radial_mode direct
pair_coeff * * model.sus2 C
```

Here `model_elements` gives the **full model species order**; the symbols after
the filename map each **LAMMPS atom type**. This C example is equivalent to the
numbered form. For multielement models give the full model list and one symbol
per LAMMPS type. NULL mappings remain unsupported.

Reinstall and rebuild to obtain the numbered shorthand; old binaries do not
support it. This syntax change does not change kernel mathematics.
[Direct](examples/static-carbon/in.direct) and [Table](examples/static-carbon/in.table)
templates use your `data.in` and `model.sus2`; no checkpoint is bundled. From
the repository root, with those files in the working directory:

```bash
mpirun -n 4 "$LAMMPS_ROOT/src/lmp_ml_sus2_embedded_avx2" \
  -in interfaces/lammps/examples/static-carbon/in.direct
# Or use in.table with the same files.
```

The repository also provides a [self-generating 2016-atom diamond input](examples/static-carbon/in.diamond)
(7 x 6 x 6 conventional cells, lattice parameter 3.56054 angstrom). It runs a
single-point initialization by default, without requiring a geometry file:

```bash
mpirun -n 4 "$LAMMPS_ROOT/src/lmp_ml_sus2_embedded_avx2" \
  -var model "/absolute/path/to/static-C.sus2" -var radial table \
  -in interfaces/lammps/examples/static-carbon/in.diamond
```

For this variable-based input, use a model path without whitespace and leave
`${model}` unquoted inside the LAMMPS input so that LAMMPS expands it.
Use `-var radial direct` to evaluate the same model without interpolation. The
lattice parameter is an example starting geometry, not a universal equilibrium
value. After a successful single-point check, an illustrative 300 K, 1 bar NPT
section is:

```lammps
velocity all create 300.0 172900 mom yes rot yes dist gaussian
timestep 0.0005                    # metal units: ps; 0.5 fs
fix integrate all npt temp 300.0 300.0 0.05 iso 1.0 1.0 0.5
thermo 50
thermo_style custom step temp pe ke etotal press vol
run 1000                          # 0.5 ps; a short check, not equilibration
```

## Direct and table evaluation

`radial_mode` accepts `direct`, `table`, or `auto`; the default is `direct`.
For an explicit analytical reference use:

```lammps
pair_style sus2/embedded model.sus2 radial_mode direct
pair_coeff * *
```

Tables use cubic Hermite values and the derivative of the **same interpolant**.
Fixed inner/outer radial coefficient contractions are tabulated together with
their derivatives; identical complete requests share storage. Dynamic Adaptive
coefficients are not frozen into distance-only functions. `tabstep 0.001` is in
angstrom. The default `table_max_mb auto` allocates knot storage lazily for
actual requests, without a fixed 64 MiB cap. It does not reserve all tables in
advance. To impose an optional knot-storage limit, use a numeric value such as
`table_max_mb 64` (64 MiB **per MPI process**); `table_max_mb 0` disables knot
storage and evaluates directly. The limit does not cover total process memory
or request metadata. Out-of-domain, unsafe and over-budget requests, and failed
allocations, fall back to direct evaluation.

Frozen request metadata includes only ordered pairs of species actually present
among owned atoms across all MPI ranks. Declaring eight LAMMPS types does not
retain absent species: six present species produce 36 ordered pairs instead of
64, while preserving the original model species indices and input mapping.
The global presence set is refreshed at initialization and neighbor rebuilding,
covering type changes, atom creation/deletion, migration and repeated `run 0`.
Ranks with no owned atoms still participate. A new model clears the previous
selection. Knots remain lazy; pruning metadata alone does not claim savings for
unused tables that were never allocated.

Read-table mode is numerical interpolation and must be checked against direct
or native E/F/virial results for the chosen model and spacing. It is not always
faster. The initial `rank0_*` diagnostic is one rank's first evaluation, not a
whole-trajectory hit rate. Optional `profile_every 50` reports interval timing
and all-rank table counters; leave it at its default zero for fair speed tests.

## Supported scope and limits

- Native-exported Static, Legacy/Effect, Pure-W and Adaptive models, including
  supported inner/outer cross or diagonal graphs and species energy readout.
  Persisted native legality rules still apply, especially for Pure-W.
- `L`, `K`, `LK`, `pair` radial-coordinate sharing; supported separate inner
  coefficients/warps, Adaptive projections, joint/tensor paths and ZBL.
- `L=0..4`; valid K/radial/gate/outer dimensions follow native graph rules,
  not a benchmark-shape whitelist.
- Global virial and per-atom energy are supported. Per-atom/centroid virial,
  hybrid/NULL mappings, excluded or scaled neighborhoods, and old residual,
  multilayer and old edge-L1 models are not supported.
- Use `newton on` and complete full neighborhoods. Molecular atom styles need
  unscaled `special_bonds lj 1 1 1` environments. In particular, do not assume
  `compute stress/atom` is available just because global pressure works.
- Restart files do not serialize this pair's model; after `read_restart`,
  reissue `pair_style` and `pair_coeff` with the original model file.

The reader checks layout, finite coefficients, enums and graph bounds. It does
not independently reconstruct every canonical joint Clebsch-Gordan topology;
use native-exported models rather than hand-editing model graphs.

## Verification and source provenance

The 2026-10-19 package includes the qualified inference source and Intel build
files. The numbered shorthand changes pair-input handling, not kernel
mathematics. The rebuilt numbered-input binary passed:

- 92 existing native-oracle/MPI checks and 3 expected error-input rejections;
- 56 numbered/explicit input checks and 12 expected error-input rejections,
  including Direct/Table, multiple model families, prefixes, reordered
  explicit symbols, atom migration and short NVE checks;
- 2 Direct/Table checks reloading H/O, C/N and H/O models into the same pair
  instance without retaining inferred element metadata;
- 4 four-rank runs of the Direct/Table templates and self-generated diamond
  example, each with 2016 atoms.

The original tolerances were retained (direct 1e-9, table 2e-6 in the native
oracle comparisons). Separately, the earlier Static/Effect carbon package
passed eight model/mode/rank E/F/virial checks and 24 short NPT runs on
2016-atom diamond. These are bounded qualifications, not proof of long-time
stability for arbitrary models.

For a local installer-only regression (not compilation or numerical proof):

```bash
python3 interfaces/lammps/tests/embedded_install_check.py
```

`tests/embedded_efs.cpp` is a portable evaluator for comparison against an
independent native oracle. `tests/cpu_qualification.py` accepts a real LAMMPS
binary and an external native fixture tool supporting `eval-efs MODEL CFG`;
the oracle and frozen test datasets are test inputs, not runtime dependencies.
`tests/embedded_input_check.py` and `tests/embedded_reload_check.py` add the
numbered-input and same-instance reload regressions using those external
fixtures. Run these MPI tools inside a four-core CPU allocation; they do not
submit scheduler jobs themselves.

For focused storage/selection checks, compile and run the standalone regression
with an external eight-species model (the fixture is never modified):

```bash
c++ -O2 -std=c++17 -Iinterfaces/lammps/ML-SUS2 \
  interfaces/lammps/tests/embedded_radial_selection_check.cpp \
  interfaces/lammps/ML-SUS2/sus2_sl_model.cpp \
  interfaces/lammps/ML-SUS2/sus2_sl_radial.cpp \
  interfaces/lammps/ML-SUS2/sus2_sl_radial_math.cpp \
  -o /tmp/embedded-radial-selection-check
/tmp/embedded-radial-selection-check /absolute/path/to/eight-species-model.mtp
```

This checks noncontiguous original species indices, lazy construction,
selection reset, explicit zero/finite caps and default storage above 64 MiB.
It does not replace actual MPI lifecycle or native-reference E/F/virial checks.
