"""Current native model families: CLI parity, geometry FD and cache ownership.

PYSUPERLINEAR_TEST_BINARY selects the matching lossless-capable native CLI.
Fixture generation uses repository QA helpers; inference itself is in-process.
Run once in a compute allocation, with numerical-library threads set to one.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from pysuperlinear import SuperlinearCalculator

ROOT = Path(os.environ.get("PYSUPERLINEAR_SOURCE_DIR", Path(__file__).resolve().parents[3]))
CLI = os.environ.get("PYSUPERLINEAR_TEST_BINARY")
CASES = [(kind, head, projection, activation, message) for kind, projection, activation, message in (
    ("static", "full", "none", False), ("effect", "full", "none", False),
    ("adaptive", "full", "none", False), ("adaptive", "full", "tanh", False),
    ("adaptive", "shared-n", "none", False), ("adaptive", "shared-n", "tanh", True),
) for head in ("shared", "species")]


def helper():
    # Share native topology/population contracts rather than invent a Python model.
    sys.path.insert(0, str(ROOT / "dev_test"))
    spec = importlib.util.spec_from_file_location("descriptor_qa", ROOT / "dev_test/descriptors_species_readout_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*args):
    subprocess.run([str(CLI), *map(str, args)], check=True, capture_output=True, text=True, timeout=180)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=CASES, ids=lambda p: "-".join(map(str, p)))
def current_model(request, tmp_path):
    if not CLI:
        pytest.skip("set PYSUPERLINEAR_TEST_BINARY to the matching native CLI")
    qa = helper()
    kind, head, projection, activation, tensor = request.param
    joint = kind == "adaptive"
    top, materialized, model = (tmp_path / name for name in ("topology.sus2", "materialized.sus2", "model.sus2"))
    options = qa.topology(kind, head, joint, tensor, projection=projection, activation=activation)
    # pair sharing catches the historical K_=warp-count metadata mistake.
    options = ["--scaling-map=pair" if x.startswith("--scaling-map=") else x for x in options]
    run("init-sh", top, *options)
    run("roundtrip-model-dev", top, materialized)
    model.write_text(qa.populated(materialized.read_text(), head, joint, tensor))
    calc = SuperlinearCalculator(model, species=["H", "O"])
    frame = qa.fixture()[0]
    atoms = Atoms(numbers=np.array([1, 8, 1]), positions=frame["positions"], cell=frame["cell"], pbc=True)
    atoms.calc = calc
    frozen = digest(model)
    yield qa, model, atoms, (kind, head, projection, activation, tensor), tmp_path
    assert digest(model) == frozen, "Python/CLI evaluation modified the model"


def read_prediction(path):
    text = path.read_text()
    energy = float(re.search(r"(?m)^\s*Energy\s*\n\s*(\S+)", text)[1])
    atom = re.search(r"(?m)^\s*AtomData:\s*([^\n]*)\n", text)
    header = atom[1].split()
    n = int(re.search(r"(?m)^\s*Size\s*\n\s*(\d+)", text)[1])
    rows = np.array([[float(v) for v in line.split()] for line in text[atom.end():].splitlines()[:n]])
    force = rows[:, [header.index(k) for k in ("fx", "fy", "fz")]]
    stress = np.array([float(v) for v in re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]+)", text)[1].split()])
    assert stress.shape == (6,)
    return energy, force, stress


def test_current_modes_cli_and_all_geometry_derivatives(current_model):
    qa, model, atoms, mode, folder = current_model
    kind, head, projection, activation, tensor = mode
    meta = atoms.calc.metadata
    assert meta["L"] == 1 and meta["K"] == 2 and meta["scaling_map"] == "pair"
    assert meta["readout"] == head and meta["atomic_numbers"] == [1, 8]
    assert meta["adaptive_radial"] == (kind == "adaptive")
    assert meta["tensor_message_lmax"] == int(tensor)
    if kind == "adaptive":
        assert meta["projection"] == projection and meta["activation"] == activation
        assert meta["shared_n_rank"] == (2 if projection == "shared-n" else 1)
        assert meta["joint"] and meta["amplitude_mode"] == "species"
    cfg, prediction = folder / "input.cfg", folder / "prediction.cfg"
    cfg.write_text(qa.cfg_text([dict(cell=atoms.cell.array, positions=atoms.positions, types=np.array([0, 1, 0]))]))
    before = digest(cfg)
    run("calc-efs", model, cfg, prediction, "--no-loss")
    assert digest(cfg) == before
    energy, force, virial = read_prediction(prediction)
    calc = atoms.calc
    result = calc.model.calculate(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)
    assert np.isfinite(result["energy"]) and np.max(np.abs(result["forces"])) > 1e-12
    np.testing.assert_allclose(result["energy"], energy, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(result["forces"], force, atol=2e-9, rtol=2e-9)
    expected_stress = -virial / atoms.get_volume()
    np.testing.assert_allclose(result["stress"], expected_stress, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(atoms.get_forces(), force, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(atoms.get_stress(), expected_stress, atol=2e-9, rtol=2e-9)

    def energy_at(positions, cell):
        return calc.model.calculate(atoms.numbers, positions, cell, atoms.pbc)["energy"]

    step = 2e-5
    # Every force component and all six symmetric strain components, not one direction.
    for i in range(len(atoms)):
        for a in range(3):
            plus, minus = atoms.positions.copy(), atoms.positions.copy()
            plus[i, a] += step
            minus[i, a] -= step
            fd = -(energy_at(plus, atoms.cell.array) - energy_at(minus, atoms.cell.array)) / (2 * step)
            np.testing.assert_allclose(fd, force[i, a], atol=2e-6, rtol=2e-4)
    for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
        strain = np.zeros((3, 3))
        strain[a, b] = 1. if a == b else .5
        if a != b:
            strain[b, a] = .5
        plus, minus = np.eye(3) + step * strain, np.eye(3) - step * strain
        fd = (energy_at(atoms.positions @ plus, atoms.cell.array @ plus)
              - energy_at(atoms.positions @ minus, atoms.cell.array @ minus)) / (2 * step * atoms.get_volume())
        np.testing.assert_allclose(fd, expected_stress[k], atol=2e-7, rtol=2e-4)
    # Distinct calls must not overwrite returned arrays or leave geometry caches stale.
    saved = result["forces"].copy()
    changed = atoms.positions.copy()
    changed[0, 0] += .08
    altered = calc.model.calculate(atoms.numbers, changed, atoms.cell.array, atoms.pbc)
    assert np.max(np.abs(altered["forces"] - saved)) > 1e-12
    restored = calc.model.calculate(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)
    np.testing.assert_array_equal(result["forces"], saved)
    np.testing.assert_allclose(restored["forces"], saved, atol=2e-12, rtol=2e-12)
    assert restored["energy"] == pytest.approx(result["energy"], abs=2e-12, rel=2e-12)


def test_current_mapping_and_nonperiodic_guards(current_model):
    _, model, atoms, _, _ = current_model
    calc = atoms.calc
    for bad in (["H"], ["H", "H"], "HO", [1, 8], ["H", "Xx"]):
        with pytest.raises((ValueError, TypeError)):
            SuperlinearCalculator(model, bad)
    # The full mapping includes absent species: a H-only geometry still needs H/O.
    for indices in ([0, 2], [1]):
        mono = atoms[indices]
        mono.calc = SuperlinearCalculator(model, ["H", "O"])
        assert np.isfinite(mono.get_potential_energy())
    molecule = atoms.copy()
    molecule.set_cell(np.zeros((3, 3)))
    molecule.pbc = False
    molecule.calc = calc
    assert np.isfinite(molecule.get_forces()).all()
    from ase.calculators.calculator import PropertyNotImplementedError
    with pytest.raises(PropertyNotImplementedError):
        molecule.get_stress()
    slab = atoms.copy()
    slab.pbc = [True, True, False]
    slab.calc = calc
    assert np.isfinite(slab.get_forces()).all()
    bad_cell = atoms.cell.array.copy()
    bad_cell[1] = bad_cell[0]
    with pytest.raises(ValueError):
        calc.model.calculate(atoms.numbers, atoms.positions, bad_cell, atoms.pbc)
