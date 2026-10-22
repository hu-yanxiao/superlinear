# PySuperlinear

Native C++ energy, forces and stress for ASE; phonopy/phono3py workflows use
the same calculator. Static, coordination-effect and Adaptive models support
shared and per-element readout. No training process or temporary CFG files are
needed for ASE evaluations.

## Install

**Python ≥3.9; Python 3.12 recommended.** Requires a C++17 compiler, BLAS and
Python development headers. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install './interfaces/python[phonon]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use `./interfaces/python` without `[phonon]` for ASE only.
The phonon extra requires phonopy 2.x (≥2.45) and phono3py 3.x (≥3.21);
the workflow was checked with phonopy 2.45.1 and phono3py 3.21.0.

Build from this source checkout. A separate interface copy needs
`-Ccmake.define.SUPERLINEAR_SOURCE_DIR=/absolute/path/to/superlinear`.
The package is independent of `pysus2sh`.

## ASE and relaxation

The example below uses **carbon diamond**, not silicon. Replace `["C"]` with
your model's **full ordered species list**, even if the structure contains only
a subset. Use a chemically suitable model; element names are not inferred
from its filename.

```python
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS
from pysuperlinear import SuperlinearCalculator

atoms = bulk("C", "diamond", a=3.56)  # Two-atom primitive cell; starting geometry.
atoms.calc = SuperlinearCalculator("model.sus2", species=["C"])
print(atoms.get_potential_energy())  # eV, total
print(atoms.get_forces())            # eV/Angstrom
print(atoms.get_stress())            # eV/Angstrom^3, tensile-positive

# Relax positions and the full cell at zero external pressure.
converged = BFGS(FrechetCellFilter(atoms), logfile="relax.log", maxstep=0.05).run(
    fmax=1e-4, steps=1000,
)
force_max = np.linalg.norm(atoms.get_forces(), axis=1).max()
stress_max = np.abs(atoms.get_stress()).max()
if not converged or force_max > 1e-3 or stress_max > 1e-4:
    raise RuntimeError("Relaxation not accepted; inspect relax.log.")
write("relaxed.json", atoms)  # Preserve the exact geometry and masses.
```

Use `BFGS(atoms)` to relax positions at fixed cell instead; choose a suitable
stress criterion for fixed-cell or nonzero-pressure work.
ASE stress is `-PlusStress / abs(det(cell))`, in xx yy zz yz xz xy order.
Partial-PBC systems and molecules are supported by the calculator; stress is
unavailable for zero-volume structures.

## Phonons and thermal conductivity

Start from an accepted, fully periodic, unconstrained relaxed unit cell.
Phonon workflows do not relax it automatically. From the repository root:

```bash
python interfaces/python/examples/phonon_band.py \
  --model model.sus2 --species C --structure relaxed.json \
  --supercell 4 4 4 --displacement 0.03 --output band-C

python interfaces/python/examples/lattice_kappa.py \
  --model model.sus2 --species C --structure relaxed.json \
  --supercell 4 4 4 --fc2-supercell 4 4 4 --full-pairs \
  --mesh 21 21 21 --temperatures 300 --displacement 0.03 \
  --lbte --isotope --output kappa-C --dry-run
```

Inspect `kappa-C/metadata.json` for displacement counts and memory estimates.
Then repeat the thermal command with **`--resume` instead of `--dry-run`**.
The plan evaluates no forces or conductivity; the band command is a real
calculation. On a cluster, run relaxation, bands and transport on a CPU compute
allocation, not the login node.

Equivalent thermal API:

```python
from ase.io import read
from pysuperlinear.phonons import run_thermal_conductivity

if __name__ == "__main__":
    result = run_thermal_conductivity(
        read("relaxed.json"), "model.sus2", species=["C"], output="kappa-api",
        supercell=(4, 4, 4), fc2_supercell=(4, 4, 4),
        displacement=0.03, pair_cutoff=None, mesh=(21, 21, 21),
        temperatures=(300.0,), lbte=True, isotope=True, dry_run=True,
    )
    print(result["plan"])
```

- **Isotopes:** examples enable natural-abundance scattering with
  `isotope=True` / `--isotope`. Input masses are preserved. Omit the option
  (API default `False`) for a no-isotope comparison in a new output directory.
- **Solver:** `lbte=True` / `--lbte` selects direct LBTE; the API/CLI default
  is RTA. Direct LBTE can require large dense matrices.
- **Cell and range:** replication acts on the input cell; 4³ gives 128 atoms
  for a two-atom diamond primitive cell, or 512 for an eight-atom conventional cell.
  `pair_cutoff` / `--pair-cutoff` truncates displacement pairs, not the potential;
  `None` / `--full-pairs` includes all pairs in the finite supercell.
- **Bands:** ASE chooses the default path; use `--band-path` or `--path-json`
  for another path. Frequencies are THz. Modes below −0.1 THz stop transport;
  smaller negative modes remain reported. A sampled stability pass is not a
  proof of stability everywhere.
- **Physics:** three-phonon plus optional isotope scattering; no boundary,
  four-phonon, electronic contribution or NAC. All single-protocol results
  remain **unconverged** until cell size, displacement, FC range and q-mesh are checked.

## Outputs and restart

Outputs include force checkpoints, compact `fc2.hdf5`/`fc3.hdf5`,
band or kappa CSV/HDF5/PNG/PDF, stability data and `metadata.json`.
Kappa is in W/(m·K), tensor order xx yy zz yz xz xy.
Reference-force subtraction and per-displacement drift removal are enabled;
`--keep-reference-forces` / `--keep-force-drift` disable them.
Traditional FC symmetrization is default; `--symmetrization symfc` or `none`
selects another policy.

Use `resume=True` / `--resume` only with identical model, geometry, masses,
species order, protocol (including isotopes), engine and package versions.
The workflow verifies fingerprints, reuses force checkpoints and rebuilds
force constants; it does not load arbitrary cached FC files. Concurrent writers
are rejected. After a crash, confirm no process owns the run before removing
a stale `run.lock`.

Increase `workers` / `--workers` within the CPU allocation for independent
force-evaluation processes. Keep the Python `__main__` guard and numerical
thread limits. Do not share a calculator across concurrent threads or modify
its loaded model file; use scheduler wall-time limits for untrusted inputs.

## Tests

```bash
python -m pip install './interfaces/python[test,phonon]'
python -m pytest interfaces/python/tests/test_phonons.py
```

These are lightweight workflow tests with mocked forces/transport, not model
qualification. [Current-mode tests](tests/test_current_modes.py) compare native
E/F/virial and finite differences against an explicitly selected matching CLI.
Real model/phonon qualification belongs on the intended CPU allocation.
