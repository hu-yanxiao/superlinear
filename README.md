# SUS2.superlinear

**Machine-learning interatomic potentials based on MPF (Material Property Field).**

## Installation

```bash
# Load an Intel classic + MKL environment first.
git clone https://github.com/hu-yanxiao/superlinear.git
cd superlinear
./configure --compiler=intel --blas=mkl
make -j4 mlp
./bin/mlp-sus2 help
```

### LAMMPS CPU interface

The matching interface sources, installer and make targets are shipped **in
this repository**, under [`interfaces/lammps/`](interfaces/lammps/README.superlinear-embedded-cpu.md).
Use the self-contained `sus2/embedded` pair style for current superlinear
models; it does not require the training executable, a native inference
library or Python at runtime. The historical `sus2mtp` interface remains
separate.

From the superlinear repository root, with Intel classic C++, Intel MPI, MKL
and TBB available, install into a separate LAMMPS source tree (validated with
LAMMPS 10 Dec 2025):

```bash
LAMMPS_ROOT=/absolute/path/to/lammps
bash interfaces/lammps/scripts/install_superlinear_embedded_cpu.sh "$LAMMPS_ROOT"
make -C "$LAMMPS_ROOT/src" ml_sus2_embedded_avx2 -j4
"$LAMMPS_ROOT/src/lmp_ml_sus2_embedded_avx2" -h
```

The help output must list `sus2/embedded`. No `make yes-ML-SUS2`, separate
interface repository, or native-library build is needed. Keep a tree with the
older native-wrapper interface separate; the installer will reject it.

On the maintained phy-weigw cluster, the stable **superlinear CPU LAMMPS**
entry is `/work/phy-weigw/app/superlinear/lmp`. Load the Intel environment
listed in the [build notes](interfaces/lammps/README.superlinear-embedded-cpu.md#install-and-build), then use:

```bash
mpirun -n 4 /work/phy-weigw/app/superlinear/lmp -in in.direct
# Or: -in in.table (with the same data.in and model.sus2)
```

This entry is separate from the historical SUS2 LAMMPS executable, the
training executable and GPUMD. Updating the source requires reinstalling the
interface and rebuilding; production installation follows numerical checks
and retains a backup of any previous binary.

#### Static C: key LAMMPS input lines

Assume `model.sus2` is your native-exported, **one-species carbon Static**
model, and `data.in` has one atom type (C). Paths are relative to the
directory where LAMMPS is launched; these are user-supplied files.

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

Static mode is read from the model file; there is no extra `static` pair-style
switch. Numbered mapping sends LAMMPS type i to native model species i-1.
`Masses` and data-file comments do not infer elements. The number of LAMMPS
types must not exceed the model species count; this form only selects a prefix.
Arbitrary subsets/reordering require the existing explicit `model_elements`
plus `pair_coeff` symbols syntax. For interpolation, replace the pair-style line with:

```lammps
pair_style sus2/embedded model.sus2 radial_mode table
# Optional defaults: tabstep 0.001 table_max_mb auto
```

Table mode uses conservative Hermite interpolation; `tabstep` is in angstrom.
The default `table_max_mb auto` grows knot storage lazily to meet actual
requests, without a fixed 64 MiB cap. Frozen request metadata includes only
species present anywhere in the system, retaining their original model indices.
An optional numeric `table_max_mb` limits knot storage in MiB **per MPI process**;
zero disables knot storage. This is not a limit on total process memory.
It retains direct fallback and is not always faster.
Direct is the default. Check E/F/virial for each new model and table spacing.
Ready-to-run inputs using your `data.in` and `model.sus2` in the working directory:
[Direct](interfaces/lammps/examples/static-carbon/in.direct) and
[Table](interfaces/lammps/examples/static-carbon/in.table).
Reinstall and rebuild LAMMPS to enable this numbered shorthand; old binaries
do not acquire it by editing an input file. Kernel mathematics are unchanged.
See the [2016-atom diamond example](interfaces/lammps/examples/static-carbon/in.diamond)
and [complete installation, MD and capability notes](interfaces/lammps/README.superlinear-embedded-cpu.md).

## Training

Choose one template, copy it to the repository root, and edit its data paths
and number of atomic types.

```bash
cp examples/train-static.yaml train.yaml
# Or: cp examples/train-effect.yaml train.yaml

./bin/mlp-sus2 train train.yaml --check-config
./bin/mlp-sus2 train train.yaml > train.log 2>&1

# With an MPI build, use this instead of the serial training command:
# mpirun -n 40 ./bin/mlp-sus2 train train.yaml > train.log 2>&1
```

### Training acceleration

Native `coordination-effect` parameter gradients use exact channelwise
contractions automatically on supported single-layer, shared-radial paths.
No enable flag is needed. Set `SUS2_SH_EFFECT_PARAM_FAST=0` to compare with
the reference implementation. The model and loss are unchanged.

These optional execution switches are **off by default**. Add them before the
training command, or before `mpirun` in an LSF script:

```bash
export SUS2_SH_SITE_DER_CACHE=1        # Reuse site derivatives where applicable
export SUS2_SH_ACCUM_SKIP_SITE_DERS=1  # Skip unused site-derivative output
export SUS2_SH_PRODUCT_HVT_REVERSE=1   # Alternative product-graph derivative schedule
```

They do not change the model or loss. Applicability and speedup depend on the
model's execution path. In `coordination-effect`, the default two-layer edge
cache takes precedence over `SITE_DER_CACHE`; it does not add another cache.
Leave `MLIP_CACHE_NEIGHBORHOODS` unset for automatic memory-aware neighbor
caching. Two-layer edge/tanh caches and MKL linear-solver fast paths are already
enabled when applicable; they need no extra enable flag.

### Static radial — L3/K4/O2

No coordination-effect layer: **K and O are used; R is absent**.

```yaml
schema_version: 1
model:
  kind: static-radial
  species_count: 2                 # Number of CFG types (0, 1, ...)
  l_max: 3                        # L
  radial_channels: 4              # K
  neighbor_cutoff: 6.5             # Angstrom
  radial:
    basis: chebyshev-sss
    basis_size: 10
    warp_sharing: K               # L / K / LK / pair
  interaction:
    channels: 2                   # O; 1 <= O <= K
    body_order: 4
    channel_coupling: cross        # Static mode uses cross
  readout:
    element_mode: per-element     # shared / per-element
training:
  profile: native-static-v1
  optimizer: bfgs
  iterations: 100                  # Maximum BFGS steps; may converge earlier
loss:
  energy_weight: 1
  force_weight: 0.01
  virial_weight: 0.001
data:
  train: data/train.cfg
  valid: data/val.cfg              # Remove if no validation set exists
```

### Coordination effect — L3/K4/R2/O2

Adds coordination-environment modulation. Inner and outer channel coupling
are independently selectable. `effect` is also accepted as a short kind name.

```yaml
schema_version: 1
model:
  kind: coordination-effect
  species_count: 2                 # Number of CFG types (0, 1, ...)
  l_max: 3                        # L
  radial_channels: 4              # K
  neighbor_cutoff: 6.5             # Angstrom
  radial:
    basis: chebyshev-sss
    basis_size: 10
    warp_sharing: K               # L / K / LK / pair
  coordination:
    channels: 2                   # R; 1 <= R <= K
    body_order: 4
    channel_coupling: cross        # cross / diagonal; inner graph
  interaction:
    channels: 2                   # O; 1 <= O <= K
    body_order: 4
    channel_coupling: diagonal     # cross / diagonal; outer graph
  readout:
    element_mode: per-element     # shared / per-element
training:
  profile: native-effect-v1
  optimizer: bfgs
  iterations: 100                  # Maximum BFGS steps; may converge earlier
loss:
  energy_weight: 1
  force_weight: 0.01
  virial_weight: 0.001
data:
  train: data/train.cfg
  valid: data/val.cfg              # Remove if no validation set exists
```

## Outputs and evaluation

No `directory` setting is needed. Each run automatically gets a new directory
under `runs/`, named from the model type and L/K/R/O with a unique run ID.
Static models omit R. The resolved directory is printed at startup.

Models and checkpoints are saved with the **`.sus2`** extension:

```text
runs/<your-run>/
  initial.sus2
  current.sus2
  best.sus2
  final.sus2
  optimizer.csv
```

```bash
./bin/mlp-sus2 calc-errors runs/<your-run>/final.sus2 data/val.cfg
./bin/mlp-sus2 calc-efs runs/<your-run>/final.sus2 data/val.cfg val.pred.cfg
```

Replace `<your-run>` with the generated directory printed in the log. Keep
prediction output separate from the original labels.

### Atomic descriptors

```bash
./bin/mlp-sus2 calc-descriptors runs/<your-run>/final.sus2 data/val.cfg descriptors.txt
```

Supports shared and `per-element` readout. Each structure starts with `#start N`,
followed by one row per atom: CFG type, then raw outer scalar descriptors.
Optional gate/joint readout scalars are appended. Final readout weights are not
applied, and features are not expanded into per-element blocks; the type column
identifies the element head. The output is descriptor text, not a CFG file.

## Continue training

Remove the entire `model` group, add
`data.model: runs/<your-run>/current.sus2`. A new continuation directory is
generated automatically. Keep the matching training profile: `native-static-v1` or `native-effect-v1`.
Then run the same training command.
This restores model parameters; BFGS optimizer history starts fresh.

## Python: relaxation, phonons and lattice thermal conductivity

```bash
python -m pip install './interfaces/python[phonon]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The in-process ASE calculator supports static, coordination-effect and Adaptive
models, including element-specific readout. It evaluates energy, forces and
stress with the native C++ engine; it does not launch the training executable.
Build requirements are a C++17 compiler, BLAS and Python development headers.
Use a dedicated Python environment and run the following scripts in a **new
working directory** with your `model.sus2`. On a cluster, run them on a CPU
compute allocation, not the login node.

This carbon-diamond example assumes a separately verified carbon model with
native type order `["C"]`. For another model, replace **every** `species=["C"]`
with its **full, ordered model species list**, even when the structure contains
only a subset of those elements. Unlike the LAMMPS numbered-type input, Python
requires this explicit mapping. The examples do not change model coefficients.

### 1. Relax atomic positions and the cell

Save as `01_relax.py`. `FrechetCellFilter` relaxes positions and the full cell at
zero external pressure. To hold the cell fixed, use `BFGS(atoms, ...)` instead.
Here `3.56` Angstrom is only a starting lattice constant, not a fitted result.

```python
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pysuperlinear import SuperlinearCalculator

atoms = bulk("C", "diamond", a=3.56)  # Two-atom primitive cell; fully periodic.
# Or: from ase.io import read; atoms = read("structure.extxyz")
atoms.calc = SuperlinearCalculator("model.sus2", species=["C"])
with Trajectory("relax.traj", "w", atoms) as trajectory:
    opt = BFGS(FrechetCellFilter(atoms), logfile="relax.log", maxstep=0.05)
    opt.attach(trajectory.write, interval=1)
    converged = opt.run(fmax=1e-4, steps=1000)

# Check real forces and stress separately from the cell filter's generalized fmax.
force_max = np.linalg.norm(atoms.get_forces(), axis=1).max()
stress_max = np.abs(atoms.get_stress()).max()
print("max force [eV/A]:", force_max, "max stress [eV/A^3]:", stress_max)
if not converged or force_max > 1e-3 or stress_max > 1e-4:
    raise RuntimeError("Relaxation not accepted; inspect relax.log before phonons.")
write("relaxed.json", atoms)  # Keep this exact geometry for bands and transport.
```

```bash
python 01_relax.py
```

The stress acceptance check above assumes the zero-pressure, variable-cell
example. For a deliberately fixed cell or nonzero target pressure, define the
corresponding stress criterion rather than applying this one unchanged.

### 2. Calculate the phonon band structure

Save as `02_band.py`. The default band path is chosen by ASE for the relaxed
primitive cell. A custom path can be passed with `band_path`; see the
[Python interface guide](interfaces/python/README.md#phonopy-bands-and-phono3py-lattice-thermal-conductivity).

```python
from ase.io import read
from pysuperlinear.phonons import run_band_structure

def main():
    band = run_band_structure(
        read("relaxed.json"), "model.sus2", species=["C"], output="band-C",
        supercell=(3, 3, 3), displacement=0.01, primitive_matrix="P",
        stability_mesh=(5, 5, 5), workers=1,
    )
    print(band["stability"])
    # Bands preserve imaginary frequencies; a completed band run is not a stability pass.
    if band["stability"]["substantial_imaginary_count"]:
        raise RuntimeError("Imaginary modes found; investigate before transport.")

if __name__ == "__main__":
    main()
```

```bash
python 02_band.py
```

Results include `band-C/band.png`, `band.pdf`, `band.yaml`, `band.csv`,
`fc2.hdf5` and `metadata.json`. The sampled stability check does not establish
stability at every wave vector. Do not discard imaginary modes to obtain kappa.

### 3. Plan, then calculate lattice thermal conductivity

After checking the bands, save as `03_kappa.py`. The first command below only
estimates displacement counts, FC3 memory and (if selected) LBTE matrix memory;
it evaluates **no forces or
thermal conductivity**. The second explicitly starts the calculation using the
same saved structure, model and protocol.

```python
import argparse
from ase.io import read
from pysuperlinear.phonons import run_thermal_conductivity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute the existing dry-run plan")
    args = parser.parse_args()
    result = run_thermal_conductivity(
        read("relaxed.json"), "model.sus2", species=["C"], output="kappa-C",
        supercell=(3, 3, 3), fc2_supercell=(4, 4, 4), displacement=0.01,
        pair_cutoff=None,  # All displacement pairs within this finite supercell.
        mesh=(11, 11, 11), temperatures=(300.0,), lbte=False,  # RTA; True selects LBTE.
        primitive_matrix="P", workers=1, dry_run=not args.run, resume=args.run,
    )
    if args.run:
        print("T [K]:", result["temperature_K"])
        print("kappa [W/(m K)], xx yy zz yz xz xy:", result["kappa_W_mK"])
    else:
        print(result["plan"])

if __name__ == "__main__":
    main()
```

```bash
python 03_kappa.py           # Inspect kappa-C/metadata.json and the printed plan first.
python 03_kappa.py --run     # Real force constants and transport; can be expensive.
```

Transport builds its own FC2/FC3 from forces and rechecks stability; it does not
reuse `band-C/fc2.hdf5`. Substantial imaginary modes stop transport without a
kappa result. Outputs include `kappa-C/kappa.csv`, HDF5/PNG/PDF results, FC2/FC3,
force checkpoints and provenance in `metadata.json`.

- **Convergence:** these are starting settings, not converged predictions.
  Check equilibrium, displacement amplitude, FC2/FC3 supercell sizes, q-mesh and
  frequency cutoff. A two-atom input gives 54/128 atoms for the example FC3/FC2
  supercells; an eight-atom conventional input gives 216/512 instead.
- **RTA or LBTE:** `lbte=False` is RTA; `True` requests direct LBTE, which can
  require much more memory. Changing this or another physical setting requires
  a new output directory and a new dry-run plan. The current wrapper includes
  intrinsic three-phonon scattering, **not** isotope, boundary, four-phonon or
  electronic contributions, and supplies no non-analytical correction (NAC).
- **Parallelism and restart:** increase `workers` only within the CPU allocation;
  these are independent force-evaluation processes, not threads sharing a model.
  Keep the `__main__` guard and the thread environment variables above. Resume a
  stopped run with identical inputs and `resume=True`; do not rerelax the cell or
  overwrite the model in place. The workflow verifies fingerprints and reuses
  force checkpoints, not arbitrary cached force-constant files.

See the [full Python interface guide](interfaces/python/README.md) for CLI
equivalents, displacement-pair cutoffs, output conventions and validation.
