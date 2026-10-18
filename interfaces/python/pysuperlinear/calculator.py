from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.data import atomic_numbers

from ._core import SuperlinearModel


class SuperlinearCalculator(Calculator):
    """Persistent float64 C++ model with explicit type-index species order.

    Independent CPU processes should each own their calculator. No external
    process, temporary CFG, model reload, or gradient approximation is used.
    A calculator is not safe to share between concurrent ASE calculations.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, species, **kwargs):
        super().__init__(**kwargs)
        self.model_path = str(Path(model).expanduser().resolve(strict=True))
        if isinstance(species, str):
            raise ValueError("species must be a sequence of chemical symbols in full model type order")
        self.species = tuple(species)
        if not self.species or not all(isinstance(s, str) for s in self.species):
            raise ValueError("species must contain chemical symbols in full model type order")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be nonempty and unique, in exact model type order")
        try:
            numbers = [atomic_numbers[s] for s in self.species]
        except KeyError as exc:
            raise ValueError(f"unknown chemical symbol: {exc.args[0]}") from exc
        self.model = SuperlinearModel(self.model_path, numbers)
        self.calls = 0

    @property
    def metadata(self):
        return {**dict(self.model.metadata), "species": self.species}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("an ASE Atoms object is required")
        atoms = self.atoms
        # Guard periodic cells before native neighbor image generation.
        active = atoms.cell.array[atoms.pbc]
        if len(active):
            if not np.isfinite(active).all():
                raise ValueError("periodic lattice must be finite")
            if np.linalg.svd(active, compute_uv=False).min() < 0.5:
                raise ValueError("periodic lattice is degenerate or thinner than 0.5 A")
        result = self.model.calculate(
            np.asarray(atoms.numbers, dtype=np.int32),
            np.asarray(atoms.positions, dtype=np.float64),
            np.asarray(atoms.cell.array, dtype=np.float64),
            np.asarray(atoms.pbc, dtype=np.bool_),
        )
        self.calls += 1
        self.results = {
            "energy": float(result["energy"]),
            "free_energy": float(result["energy"]),
            "forces": np.asarray(result["forces"]),
            "superlinear_virial": np.asarray(result["virial"]),
        }
        if result["stress"] is not None:
            self.results["stress"] = np.asarray(result["stress"])
        elif "stress" in properties:
            raise PropertyNotImplementedError("stress requires a finite nonzero 3D cell volume")
