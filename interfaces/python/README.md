# PySuperlinear: persistent C++ calculator for ASE

This package compiles the current superlinear inference engine directly.
The model is loaded once per calculator. Subsequent ASE calls pass NumPy arrays
directly to `MLMTPR::CalcEFS`: no subprocess launch or temporary CFG files.
No model mathematics, trained coefficients or native installed binary changes.

## Build from this source checkout

```sh
python -m pip install ./interfaces/python
```

Building from a separate interface copy requires
`-Ccmake.define.SUPERLINEAR_SOURCE_DIR=/path/to/superlinear/source`.
Record the audited revision with
`-Ccmake.define.SUPERLINEAR_SOURCE_REVISION=<git-commit>`.
A C++17 compiler, BLAS and Python development headers are required.
The package is separate from `pysus2sh`; neither installation overwrites the other.
Source-distribution builds need the matching source checkout; only the interface
is packaged here. Respect the source engine's existing license.

## ASE usage

```python
from ase.io import read
from ase.optimize import BFGSLineSearch
from pysuperlinear import SuperlinearCalculator

atoms = read("structure.extxyz")
# Mandatory: the FULL ordered species list used by model type indices,
# not only the species appearing in this particular structure.
model_species = ["C"]  # Example only: replace with your model's full type order.
atoms.calc = SuperlinearCalculator("model.sus2", species=model_species)
energy = atoms.get_potential_energy()  # eV, total
forces = atoms.get_forces()           # eV/Angstrom
stress = atoms.get_stress()           # eV/Angstrom^3, tensile-positive
BFGSLineSearch(atoms, maxstep=0.1).run(fmax=0.01, steps=1000)
```

Directly passing Atoms to BFGSLineSearch optimizes atoms only, with no cell filter.
Stress is `-PlusStress / abs(det(cell))`, ASE Voigt order xx yy zz yz xz xy.
Fully periodic triclinic cells, partial PBC and isolated molecules are supported;
only active lattice rows are passed to the native neighbor routine. For a zero
volume molecule, stress is unavailable, not silently reported as zero.
Empty structures, nonfinite values, unsupported elements and degenerate periodic
cells are rejected. E/F/stress outputs own their memory and survive subsequent calls.

The GIL is retained for native cache/diagnostic safety. Use independent processes,
one calculator per worker, and set OMP_NUM_THREADS/MKL_NUM_THREADS/
OPENBLAS_NUM_THREADS=1 before importing numerical libraries when running many
workers. Do not share one ASE calculator across concurrent threads or change a
model file behind a loaded calculator. ASE caching avoids duplicate evaluations
for unchanged geometry. No wall-clock timeout is imposed inside native calls;
use an isolated worker with a scheduler time limit for untrusted geometries.

## Validation

Run the current-mode tests against a matching native CLI:

```sh
PYSUPERLINEAR_TEST_BINARY="$PWD/bin/mlp-sus2" \
PYSUPERLINEAR_SOURCE_DIR="$PWD" \
python -m pytest interfaces/python/tests/test_current_modes.py interfaces/python/tests/test_phonons.py
```

These cover static, coordination-effect and Adaptive models, shared and
per-element readout, full/shared-n Adaptive projection, activation and joint
readout, with native E/F/virial parity and finite differences. The separate
`test_calculator.py` fixture uses an explicitly supplied 87-element model via
`PYSUPERLINEAR_TEST_MODEL` and `PYSUPERLINEAR_TEST_SPECIES` (full type order).
External CLI parity and timing are separately measured on a CPU allocation;
comparison is against the frozen binary, with finite differences for all three
force and six stress components. Strict parity uses `calc-efs --no-loss` at
1e-8 absolute/relative tolerance, not default rounded CFG values. Default CFG
rounding differences are recorded separately; timings retain the original CLI
adapter so they measure the actual former process/file overhead.
Do not replace an ongoing benchmark backend before these checks pass or mix
results with different fingerprints without explicit recorded qualification.

## Phonopy bands and phono3py lattice thermal conductivity

Install the optional workflow dependencies from this directory:

```sh
cd interfaces/python
python -m pip install '.[phonon]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The API targets phonopy 2.45.x / phono3py 3.21.x (optional dependencies are
restricted to those major versions). Those installed APIs and the official
[phonopy](https://phonopy.github.io/phonopy/phonopy-module.html) and
[phono3py](https://phonopy.github.io/phono3py/phono3py-api.html) references were
checked; newer major-version compatibility is not claimed. The force backend
is always this package's native `SuperlinearCalculator`, for static,
coordination-effect and Adaptive models accepted by that engine. Each run
records the actual model SHA256, full species order, engine revision/metadata,
package versions, structure (including masses), protocol and workflow hash.
Model identity and chemistry must be checked first; these examples do not infer
an element from a filename or silently substitute an older PySUS2 engine.

Supply one prepared, relaxed, fully periodic ASE-readable structure. The
**full model species order is mandatory**, even if this structure contains only
one of its elements. No relaxation or change of model coefficients occurs.
For a separately verified carbon diamond model, a two-atom primitive example is:

```sh
python examples/make_diamond.py --element C --lattice-constant 3.56054 --output diamond-C.extxyz
python examples/phonon_band.py --model model.sus2 --species C --structure diamond-C.extxyz \
  --supercell 3 3 3 --displacement 0.01 --band-path GXWKGLUWLK,UX --output band-C
python examples/lattice_kappa.py --model model.sus2 --species C --structure diamond-C.extxyz \
  --supercell 3 3 3 --fc2-supercell 4 4 4 --pair-cutoff 4.0 \
  --mesh 11 11 11 --temperatures 300 --displacement 0.01 --output kappa-C --dry-run
```

The lattice constant above is an explicit example, not a model-independent
equilibrium value. In a two-atom primitive cell, `3 3 3` is 54 atoms and
`4 4 4` is 128 atoms; replicating an eight-atom conventional cell instead would
give 216/512 atoms. Replication always acts on the **input** cell, before any
`--primitive-matrix` reduction. The default `P` treats it as already primitive.
Band special-point coordinates are in the resulting phonopy primitive basis.
Omit `--band-path` to let ASE identify a path for that cell, or use `--path-json`
with `{"paths":[[[0,0,0],[0.5,0,0]]],"labels":[["G","Q"]]}` for explicit
reduced reciprocal vertices. This is needed for cells without a suitable
recognized ASE lattice path.

Inspect the dry-run `metadata.json` for displacement counts and compact/full
FC3 memory estimates. To perform the actual calculation, repeat the same
command with `--resume` instead of `--dry-run`; alternatively choose a new
output directory. `--workers N` uses independent spawned native calculators
for the force stage, never threads sharing one model. RTA is default;
`--lbte` requests direct LBTE and may require large dense collision matrices.
The scripts do not impose a timeout inside native calls: use an appropriate
CPU scheduler allocation and wall time. Outputs remain under `--output`.

Results include raw/corrected displacement-force HDF5 and individual atomic
force checkpoints, displacement/parameter YAML, compact `fc2.hdf5` (and
`fc3.hdf5` for transport), band YAML/CSV/PNG/PDF or computed kappa
HDF5/CSV/PNG/PDF, stability HDF5 and `metadata.json`. Energies/forces use ASE
eV and eV/Angstrom; phonopy's VASP-unit convention converts these with masses
in atomic mass units to THz; phono3py returns kappa in W/(m K), with tensor
order xx yy zz yz xz xy. ASE tensile-positive stress is not used to construct
FCs; all FC2/FC3 entries come from displaced native forces.

By default undisplaced residual forces are subtracted, then the mean force
drift is removed per displacement; both raw arrays and corrections are saved.
Large undisplaced forces warn that the intended equilibrium needs checking.
`--keep-reference-forces` / `--keep-force-drift` disable those corrections.
Traditional FC2/FC3 translational/permutation symmetrization applies three
iterations (ASR); `--symmetrization symfc` requests the supported symmetry
projector, while `none` explicitly disables it. Traditional successive ASR
and permutation enforcement can slightly break space-group symmetry; it does
not impose rotational invariance. Optional pair exclusion uses phono3py's
included-pair mask: zero placeholders are marked as excluded, not reported as
calculated zero forces.

Frequency values below `-0.1 THz` by default are classified as substantial
imaginary modes. Band output preserves them; transport saves stability/FCs and
stops **without a kappa result**. Smaller negative values are reported
separately, not silently changed to positive frequencies. This sampled test
does not prove stability everywhere. The RTA/LBTE protocol uses tetrahedron
integration and intrinsic three-phonon scattering only; no NAC, isotope,
boundary or four-phonon contribution is implied.

Every single run is explicitly marked **unconverged**. The 11³ mesh, .01 A
displacement and 4 A pair cutoff above are a first computational test, not a
validated prediction. Check equilibrium/stress, displacement size, symmetry,
FC2/FC3 supercells, pair cutoff versus `--full-pairs`, mesh density and frequency
cutoff before quoting kappa. A displacement-pair cutoff is not the potential
cutoff: environment-dependent chains may extend FC interaction ranges.

Resume with the same command and `--resume`. Model, structure, ordered species,
physical protocol, engine/workflow and package-version fingerprints must match;
otherwise use a new directory. Checkpoints are atomic and each verifies the
displaced geometry. Cached FC files are **never** loaded: force sets are reused
and FCs rebuilt. Worker count and dry-run mode can change without changing
physics. Concurrent writers are rejected by `run.lock`; after an unclean kill,
confirm no process still owns that exact run before manually removing its lock.

The same functions are available without the scripts:

```python
from pysuperlinear.phonons import run_band_structure, run_thermal_conductivity

band = run_band_structure(atoms, "model.sus2", species=["C"], output="band-api")
kappa = run_thermal_conductivity(atoms, "model.sus2", species=["C"], output="kappa-api",
    supercell=(3, 3, 3), mesh=(11, 11, 11), pair_cutoff=4.0)
```

`tests/test_phonons.py` exercises checkpoint rejection, corrections, planning,
transport/stability and API plumbing using lightweight mocked calculators.
Those tests are not engine qualification. An opt-in real phonopy/native band
smoke requires explicit model/species/structure environment variables; actual
trained-model band/kappa qualification belongs on the intended CPU allocation.

Submit force evaluation, FC3 construction and the transport solver to a compute
allocation on a cluster. Prepare a dedicated Python environment; do not install
these dependencies into an active training environment.
