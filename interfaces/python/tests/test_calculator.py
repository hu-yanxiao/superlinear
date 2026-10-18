import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError
from ase.optimize import BFGSLineSearch

from pysuperlinear import SuperlinearCalculator, SuperlinearModel

MODEL = os.environ.get("PYSUPERLINEAR_TEST_MODEL")
SPECIES = os.environ.get("PYSUPERLINEAR_TEST_SPECIES", "").split()
pytestmark = pytest.mark.skipif(not MODEL or not SPECIES, reason="explicit test model/species required")


@pytest.fixture
def calc():
    return SuperlinearCalculator(MODEL, SPECIES)


def sample(calc):
    a = bulk("Si", "diamond", a=5.43)
    a.positions[0] += [0.025, -0.018, 0.013]
    a.calc = calc
    return a


def test_ase_cache_and_geometry_invalidation(calc):
    a = sample(calc)
    e = a.get_potential_energy()
    f = a.get_forces()
    s = a.get_stress()
    assert calc.calls == 1
    assert a.get_potential_energy(force_consistent=True) == e
    a.positions[0, 0] += 0.015
    assert a.get_potential_energy() != e
    assert calc.calls == 2
    assert np.isfinite(f).all() and np.isfinite(s).all()


def test_output_ownership_and_A_B_A(calc):
    a = sample(calc)
    m = calc.model
    x = m.calculate(a.numbers, a.positions, a.cell.array, a.pbc)
    saved = x["forces"].copy()
    b = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 1, 1))
    m.calculate(b.numbers, b.positions, b.cell.array, b.pbc)
    y = m.calculate(a.numbers, a.positions, a.cell.array, a.pbc)
    np.testing.assert_array_equal(saved, x["forces"])
    np.testing.assert_allclose(y["forces"], saved, atol=1e-12, rtol=1e-12)
    assert x["energy"] == pytest.approx(y["energy"], abs=1e-12)


def test_stress_sign_and_finite_differences(calc):
    a = sample(calc)
    a.set_cell(a.cell.array @ np.array([[1.01, .015, 0], [.015, .99, .008], [0, .008, 1.02]]), scale_atoms=True)
    a.get_potential_energy()
    vir = calc.results["superlinear_virial"]
    stress = a.get_stress()
    np.testing.assert_allclose(stress, -vir[(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)] / a.get_volume(), atol=1e-14)
    forces = a.get_forces()
    h = 1e-5
    for k in range(3):
        p, m = a.copy(), a.copy()
        p.positions[0, k] += h
        m.positions[0, k] -= h
        p.calc = m.calc = calc
        fd = -(p.get_potential_energy()-m.get_potential_energy()) / (2*h)
        assert fd == pytest.approx(forces[0, k], abs=2e-6, rel=1e-5)
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
        deform = np.zeros((3, 3))
        deform[i, j] = 1 if i == j else .5
        if i != j:
            deform[j, i] = .5
        p, m = a.copy(), a.copy()
        p.set_cell(a.cell.array @ (np.eye(3)+h*deform), scale_atoms=True)
        m.set_cell(a.cell.array @ (np.eye(3)-h*deform), scale_atoms=True)
        p.calc = m.calc = calc
        fd = (p.get_potential_energy()-m.get_potential_energy()) / (2*h*a.get_volume())
        assert fd == pytest.approx(stress[k], abs=2e-7, rel=2e-5)


def test_permutation_translation(calc):
    a = sample(calc)
    e, f = a.get_potential_energy(), a.get_forces()
    b = a[::-1]
    b.positions += [0.11, 0.23, -0.13]
    b.calc = calc
    assert b.get_potential_energy() == pytest.approx(e, abs=1e-10)
    np.testing.assert_allclose(b.get_forces(), f[::-1], atol=1e-10)


def test_atoms_only_relaxation(calc):
    a = sample(calc)
    initial = a.cell.array.copy()
    opt = BFGSLineSearch(a, logfile=None, maxstep=.1)
    assert opt.run(fmax=.01, steps=100)
    np.testing.assert_array_equal(initial, a.cell.array)


def test_molecule_and_partial_pbc(calc):
    a = Atoms("H2O", positions=[[.76, 0, .59], [-.76, 0, .59], [0, 0, 0]])
    a.calc = calc
    e, f = a.get_potential_energy(), a.get_forces()
    with pytest.raises(PropertyNotImplementedError):
        a.get_stress()
    a.set_cell([20, 20, 20])
    a.set_pbc([True, True, False])
    assert a.get_potential_energy() == pytest.approx(e, abs=1e-10)
    np.testing.assert_allclose(a.get_forces(), f, atol=1e-10)
    assert np.isfinite(a.get_stress()).all()


@pytest.mark.parametrize("bad", ["empty", "shape", "nan", "unsupported", "singular"])
def test_invalid_native_inputs(calc, bad):
    a = sample(calc)
    n, p, c, b = a.numbers, a.positions.copy(), a.cell.array.copy(), a.pbc
    if bad == "empty":
        n, p = np.array([], dtype=int), np.empty((0, 3))
    elif bad == "shape":
        p = p.ravel()
    elif bad == "nan":
        p[0, 0] = np.nan
    elif bad == "unsupported":
        n = np.full(len(a), 118)
    elif bad == "singular":
        c[1] = c[0]
    with pytest.raises(ValueError):
        calc.model.calculate(n, p, c, b)


def test_invalid_species_and_missing_model(tmp_path):
    with pytest.raises(ValueError):
        SuperlinearCalculator(MODEL, ["Si", "Si"])
    with pytest.raises(ValueError):
        SuperlinearCalculator(MODEL, ["Si"])
    with pytest.raises((RuntimeError, ValueError)):
        SuperlinearModel(str(tmp_path / "missing.sus2"), [14])
