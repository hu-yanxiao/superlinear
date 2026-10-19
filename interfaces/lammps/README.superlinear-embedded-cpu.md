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

## Static carbon example

For a native-exported one-species carbon Static model named `static-C.sus2`
and a LAMMPS data file with atom type 1 = C, the key lines are:

```lammps
units metal
atom_style atomic
boundary p p p
newton on
read_data diamond.data
mass 1 12.0107
pair_style sus2/embedded model_elements C radial_mode table tabstep 0.001 table_max_mb 64
pair_coeff * * static-C.sus2 C
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
run 0
```

These model/data filenames are user-supplied, not bundled trained weights.
Paths are resolved from the LAMMPS working directory. The model file selects
Static versus other supported architectures; there is no extra `static` flag.
`model_elements` gives the **model species order**; the symbols after the model
filename in `pair_coeff` give the **LAMMPS atom-type order**. For this model both
are simply `C`. Do not substitute the historical `sus2mtp` syntax.

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
pair_style sus2/embedded model_elements C radial_mode direct
```

Tables use cubic Hermite values and the derivative of the **same interpolant**.
Fixed inner/outer radial coefficient contractions are tabulated together with
their derivatives; identical complete requests share storage. Dynamic Adaptive
coefficients are not frozen into distance-only functions. `tabstep 0.001` is in
angstrom. `table_max_mb 64` allows 64 MiB of knot storage **per MPI process**,
not 64 MiB for the whole job or a limit on total process memory. Metadata has a
separate bounded allocation. Out-of-domain, unsafe and over-budget requests
fall back to direct evaluation.

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

The 2026-10-19 package publishes the previously qualified inference source and
Intel build files byte for byte. Publication adds documentation and
installation/example entry points, not new model math.
The qualified candidate passed native-oracle comparisons and real MPI checks;
the Static/Effect carbon pair also passed eight model/mode/rank E/F/virial checks
and 24 short NPT runs on 2016-atom diamond. Those are bounded qualifications,
not proof of long-time stability for arbitrary models.

For a local installer-only regression (not compilation or numerical proof):

```bash
python3 interfaces/lammps/tests/embedded_install_check.py
```

`tests/embedded_efs.cpp` is a portable evaluator for comparison against an
independent native oracle. `tests/cpu_qualification.py` accepts a real LAMMPS
binary and an external native fixture tool supporting `eval-efs MODEL CFG`;
the oracle and frozen test datasets are test inputs, not runtime dependencies.
