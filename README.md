# SUS2.superlinear

**Machine-learning interatomic potentials based on MPF (Material Property Field).**

## Installation

Load GCC (C++ and Fortran), MPI and MKL (`MKLROOT` must be set), then:

```bash
git clone https://github.com/hu-yanxiao/superlinear.git
cd superlinear
./configure --compiler=gnu --blas=mkl
make -j4 mlp
./bin/mlp-sus2 help
```

Build on the target compute node; use a fresh build when changing compilers.
Intel classic is also supported with `--compiler=intel`. With Intel MPI and GCC,
set `I_MPI_CC=gcc I_MPI_CXX=g++ I_MPI_F90=gfortran` before building.

## Training with YAML

The [carbon-diamond example](example/Readme.md) is ready to run from the repository root:

```bash
./bin/mlp-sus2 train example/train-static.yaml --check-config
./bin/mlp-sus2 train example/train-static.yaml > train.log 2>&1
# Or: ./bin/mlp-sus2 train example/train-effect.yaml > train.log 2>&1
# MPI: mpirun -n 40 ./bin/mlp-sus2 train example/train-effect.yaml > train.log 2>&1
```

Both examples use 100 maximum BFGS steps and the supplied C-diamond train/validation data.

- [Static radial YAML](example/train-static.yaml): **L3/K4/O2**, without coordination effects; R is unused.
- [Coordination effect YAML](example/train-effect.yaml): **L3/K4/R2/O2**, inner `cross`, outer `diagonal`.
- For your own data, copy [train-static.yaml](examples/train-static.yaml) or
  [train-effect.yaml](examples/train-effect.yaml) and edit the data paths and species count.

Paths are relative to the YAML file. `elements: [C]` maps CFG type 0 to C;
alternatively, `species_count` specifies only the number of types.
Do not specify both with conflicting counts. `--check-config` validates settings
without training or creating output files.

### Training acceleration

Exact parameter-gradient contractions and per-atom reuse during linear
initialization are automatic on supported `coordination-effect` paths.
These additional switches are optional and off by default; export them before
training or `mpirun`:

```bash
export SUS2_SH_SITE_DER_CACHE=1
export SUS2_SH_ACCUM_SKIP_SITE_DERS=1
export SUS2_SH_PRODUCT_HVT_REVERSE=1
```

They change execution, not the model or loss; speedup depends on the model path.
The two-layer edge cache takes precedence over `SITE_DER_CACHE`.
Outer rank contractions also reuse validated weight rows and coefficient offsets
within each call, including formal loss/gradient evaluation. Arithmetic and
parameter updates are unchanged; direct-radial and pure-matrix branches retain
their existing paths.
Leave `MLIP_CACHE_NEIGHBORHOODS` unset for memory-aware automatic caching.
For reference comparisons, set `SUS2_SH_EFFECT_PARAM_FAST=0`,
`SUS2_SH_LINEAR_GATE_ADJOINT_FAST=0` and `SUS2_SH_LINEAR_FLAT_DERS=0`.

## Outputs and evaluation

Each run creates a unique model-named directory under `runs/`, relative to the
YAML file. No `directory` setting is needed. The path is printed at startup.

```text
runs/<run>/
  initial.sus2
  current.sus2
  best.sus2
  final.sus2
  optimizer.csv
```

Replace `MODEL` with the printed path to `final.sus2`:

```bash
./bin/mlp-sus2 calc-errors MODEL example/valid.cfg
./bin/mlp-sus2 calc-efs MODEL example/valid.cfg valid.pred.cfg
./bin/mlp-sus2 calc-descriptors MODEL example/valid.cfg descriptors.txt
```

Keep predictions separate from the original labels. Descriptors are raw outer
scalars, one row per atom with its CFG type; shared and per-element readout are
supported. Readout weights are not applied.

To continue training, remove the YAML `model` group and set `data.model` to the
checkpoint. Keep the matching `training.profile`. Model parameters are restored
in a new run directory; BFGS history starts fresh.

## LAMMPS CPU

The self-contained `sus2/embedded` interface runs without Python or the training
executable. With Intel classic C++, Intel MPI, MKL and TBB available:

```bash
LAMMPS_ROOT=/absolute/path/to/lammps
bash interfaces/lammps/scripts/install_superlinear_embedded_cpu.sh "$LAMMPS_ROOT"
make -C "$LAMMPS_ROOT/src" ml_sus2_embedded_avx2 -j4
```

After `read_data data.in`:

```lammps
pair_style sus2/embedded model.sus2 radial_mode direct
pair_coeff * *
```

LAMMPS type i maps to model species i−1; masses do not infer elements.
Use `radial_mode table` for optional interpolation; check E/F/virial against
Direct for each model. Arbitrary type reordering uses explicit element mapping.

[Direct input](interfaces/lammps/examples/static-carbon/in.direct) ·
[Table input](interfaces/lammps/examples/static-carbon/in.table) ·
[2016-atom diamond](interfaces/lammps/examples/static-carbon/in.diamond) ·
[Build and usage guide](interfaces/lammps/README.superlinear-embedded-cpu.md)

Maintained phy-weigw CPU entry: `/work/phy-weigw/app/superlinear/lmp`.
Source updates require reinstalling and rebuilding the interface.

## LAMMPS Kokkos/GPU

`sus2/embedded/kk` supports native Static, Effect, Pure-W and Adaptive models,
with Direct and Hermite Table evaluation. Use its dedicated installer in a
clean, isolated LAMMPS tree; the historical `sus2mtp/kk` binary is a different
interface. See the [build, supported options and measured CPU120/GPU results](interfaces/lammps/README.superlinear-kokkos.md).

Maintained phy-weigw GPU entry: `/work/phy-weigw/app/superlinear/lmp-kokkos`.
Its launcher enables the accepted frozen-table lookup and packing optimizations.
Use `SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE=1` for the measured Carbon configuration;
the measured FeCN6 configuration uses `0`.

## Python: ASE, phonons and thermal conductivity

**Python ≥3.9; Python 3.12 recommended.** Requires a C++17 compiler, BLAS and
Python development headers. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install './interfaces/python[phonon]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

For ASE only, install `./interfaces/python` without `[phonon]`.
Static, coordination-effect and Adaptive models support energy, forces and
stress, including per-element readout.

This example uses a **carbon-diamond** model and a previously relaxed,
fully periodic primitive cell saved as `relaxed.json`. Supply the model's
**full ordered species list**, not just the elements present in the structure.
Run force/phonon/transport calculations on a CPU compute allocation, not a cluster login node.

```python
from ase.io import read
from pysuperlinear import SuperlinearCalculator
from pysuperlinear.phonons import run_band_structure, run_thermal_conductivity

def main():
    atoms = read("relaxed.json")
    atoms.calc = SuperlinearCalculator("model.sus2", species=["C"])
    print(atoms.get_potential_energy())  # eV
    print(atoms.get_forces())            # eV/Angstrom

    band = run_band_structure(
        atoms, "model.sus2", species=["C"], output="band-C",
        supercell=(4, 4, 4), displacement=0.03,
    )
    if band["stability"]["substantial_imaginary_count"]:
        raise RuntimeError("Investigate imaginary phonons before transport.")

    result = run_thermal_conductivity(
        atoms, "model.sus2", species=["C"], output="kappa-C",
        supercell=(4, 4, 4), fc2_supercell=(4, 4, 4),
        displacement=0.03, pair_cutoff=None, mesh=(21, 21, 21),
        temperatures=(300.0,), lbte=True, isotope=True,
        dry_run=True,  # Inspect the plan before calculating FCs and conductivity.
    )
    print(result["plan"])

if __name__ == "__main__":
    main()
```

`isotope=True` adds **natural-abundance isotope scattering** without changing
input atomic masses. Set it to `False` for a comparison without isotope scattering.
After inspecting the plan, call `run_thermal_conductivity` with the same arguments,
`dry_run=False, resume=True` to compute kappa in W/(m·K). Resume only the thermal
call, not the completed band calculation.
`lbte=False` selects RTA; `True` selects direct LBTE, which can need substantial memory.

These are starting settings, **not converged predictions**. No boundary,
four-phonon or electronic scattering, or NAC, is included. Reuse an output
directory only with identical physical settings and `resume=True`.
See the [Python guide](interfaces/python/README.md) for relaxation, CLI examples,
outputs and tests.
