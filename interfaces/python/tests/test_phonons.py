"""Light API/IO tests; artificial test forces/kappa are NOT model qualification.

The module is loaded by filename so these tests do not need a compiled native
extension. Numerical native qualification is opt-in and uses an explicit model,
full species order and structure on the intended compute allocation.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

pytest.importorskip("phonopy")
pytest.importorskip("phono3py")
h5py = pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

SPEC = importlib.util.spec_from_file_location("_phonons_test_module", Path(__file__).parents[1] / "pysuperlinear" / "phonons.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
PATH = {"paths": [[[0, 0, 0], [.5, 0, 0]]], "labels": [["G", "Q"]]}


class ToyCalculator(Calculator):
    """Small analytic test-only calculator; never used by shipped examples."""
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.metadata = {"backend": "TEST-ONLY", "source_revision": "test-revision", "precision": "float64"}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        positions = self.atoms.positions
        drift = np.asarray([.2, -.1, .3])
        self.results = {"energy": float(np.sum(positions ** 2) - np.sum(positions * drift)),
                        "forces": -2 * positions + drift}


class Cell:
    def __init__(self, atoms):
        self.numbers, self.positions = atoms.numbers.copy(), atoms.positions.copy()
        self.cell, self.masses = atoms.cell.array.copy(), atoms.get_masses().copy()
        self.symbols = atoms.get_chemical_symbols()
        self.p2s_map = np.arange(len(atoms))

    def __len__(self):
        return len(self.numbers)


def displaced(cell, dx):
    atoms = workflow._atoms(workflow._payload(cell))
    atoms.positions[0, 0] += dx
    return Cell(atoms)


class ToyPhonopy:
    fits = 0

    def __init__(self, unitcell, supercell_matrix, **kwargs):
        atoms = workflow._atoms(workflow._payload(unitcell))
        self.primitive = Cell(atoms)
        self.supercell = Cell(atoms.repeat(np.diag(supercell_matrix)))
        self.symmetrized = False

    def generate_displacements(self, distance, is_plusminus):
        assert is_plusminus is True
        self.supercells_with_displacements = [displaced(self.supercell, distance), displaced(self.supercell, -distance)]

    def produce_force_constants(self, **kwargs):
        assert kwargs == {"calculate_full_force_constants": False, "fc_calculator": "traditional"}
        type(self).fits += 1
        self.force_constants = np.zeros((len(self.primitive), len(self.supercell), 3, 3))

    def symmetrize_force_constants(self, **kwargs):
        assert kwargs == {"level": 3, "use_symfc_projector": False}
        self.symmetrized = True

    def save(self, filename):
        Path(filename).write_text("test_fixture: true\n")

    def run_band_structure(self, paths):
        self.band_structure = SimpleNamespace(
            frequencies=[np.tile(np.arange(6), (len(path), 1)) for path in paths],
            distances=[np.linalg.norm(path - path[0], axis=1) for path in paths])

    def write_yaml_band_structure(self, filename):
        Path(filename).write_text("test_fixture: true\n")

    def run_mesh(self, mesh, is_gamma_center):
        assert is_gamma_center
        self.mesh = SimpleNamespace(frequencies=np.asarray([[0, 0, 0, 1, 2, 3.]]), qpoints=np.zeros((1, 3)))


class ToyPhono3py:
    unstable = False
    transport_calls = 0
    transport_options = []
    symmetrizations = []

    def __init__(self, unitcell, supercell_matrix, phonon_supercell_matrix, **kwargs):
        atoms = workflow._atoms(workflow._payload(unitcell))
        self.primitive = self.phonon_primitive = Cell(atoms)
        self.supercell = Cell(atoms.repeat(np.diag(supercell_matrix)))
        self.phonon_supercell = Cell(atoms.repeat(np.diag(phonon_supercell_matrix))) if phonon_supercell_matrix is not None else self.supercell

    def generate_displacements(self, distance, cutoff_pair_distance, is_plusminus):
        assert is_plusminus is True
        self.supercells_with_displacements = [displaced(self.supercell, distance), None, displaced(self.supercell, -distance)]
        self.phonon_supercells_with_displacements = [displaced(self.phonon_supercell, distance), displaced(self.phonon_supercell, -distance)]

    def save(self, filename):
        Path(filename).write_text("test_fixture: true\n")

    def produce_fc3(self, **kwargs):
        assert kwargs == {"is_compact_fc": True, "fc_calculator": "traditional"}
        assert self.forces.shape == (3, len(self.supercell), 3)
        np.testing.assert_array_equal(self.forces[1], 0)
        self.fc3 = np.zeros((len(self.primitive), len(self.supercell), len(self.supercell), 3, 3, 3))
        self.fc2 = np.zeros((len(self.primitive), len(self.supercell), 3, 3))

    def produce_fc2(self, **kwargs):
        assert kwargs == {"is_compact_fc": True, "fc_calculator": "traditional"}
        assert self.phonon_forces.shape == (2, len(self.phonon_supercell), 3)
        self.fc2 = np.zeros((len(self.phonon_primitive), len(self.phonon_supercell), 3, 3))

    def symmetrize_fc3(self, **kwargs):
        self.symmetrizations.append((3, kwargs))

    def symmetrize_fc2(self, **kwargs):
        self.symmetrizations.append((2, kwargs))

    def init_phph_interaction(self):
        pass

    def run_phonon_solver(self):
        pass

    def get_phonon_data(self):
        values = np.asarray([[0., 0., 0., 1., 2., 3.], [1., 2., 3., 4., 5., 6.]])
        if self.unstable:
            values[1, 0] = -.8
        return values, None, np.asarray([[0, 0, 0], [1, 0, 0]])

    def run_thermal_conductivity(self, is_LBTE, temperatures, write_kappa, is_isotope=False, mass_variances=None):
        type(self).transport_calls += 1
        self.transport_options.append(dict(is_LBTE=is_LBTE, is_isotope=is_isotope,
                                          mass_variances=None if mass_variances is None else np.asarray(mass_variances).copy()))
        assert write_kappa is False
        gamma = None
        if is_isotope:
            assert np.asarray(mass_variances).shape == (len(self.primitive),)
            gamma = np.repeat(np.asarray(mass_variances), 3).reshape(1, 1, -1)
        self.thermal_conductivity = SimpleNamespace(kappa=np.ones((1, len(temperatures), 6)), gamma_isotope=gamma)


@pytest.fixture
def toy(monkeypatch, tmp_path):
    import phonopy
    import phonopy.file_IO
    import phono3py
    import phono3py.file_IO
    calculators = []

    def calculator(model, species):
        assert species == ("C",)
        obj = ToyCalculator()
        calculators.append(obj)
        return obj

    def fc2(values, filename, p2s_map, **kwargs):
        with h5py.File(filename, "w") as handle:
            handle["force_constants"] = values
            handle["p2s_map"] = p2s_map

    def fc3(values, filename, p2s_map):
        with h5py.File(filename, "w") as handle:
            handle["fc3"] = values
            handle["p2s_map"] = p2s_map

    ToyPhonopy.fits = 0
    ToyPhono3py.unstable = False
    ToyPhono3py.transport_calls = 0
    ToyPhono3py.transport_options = []
    ToyPhono3py.symmetrizations = []
    monkeypatch.setattr(workflow, "_calculator", calculator)
    monkeypatch.setattr(phonopy, "Phonopy", ToyPhonopy)
    monkeypatch.setattr(phono3py, "Phono3py", ToyPhono3py)
    monkeypatch.setattr(phonopy.file_IO, "write_force_constants_to_hdf5", fc2)
    monkeypatch.setattr(phono3py.file_IO, "write_fc2_to_hdf5", fc2)
    monkeypatch.setattr(phono3py.file_IO, "write_fc3_to_hdf5", fc3)
    model = tmp_path / "model.sus2"
    model.write_text("TEST-ONLY model identity\n")
    return SimpleNamespace(model=model, atoms=bulk("C", "diamond", a=3.56054), calculators=calculators, out=tmp_path / "run")


def band(toy, **kwargs):
    return workflow.run_band_structure(toy.atoms, toy.model, ["C"], toy.out,
                                      supercell=(1, 1, 1), band_path=PATH, npoints=3, **kwargs)


def kappa(toy, **kwargs):
    return workflow.run_thermal_conductivity(toy.atoms, toy.model, ["C"], toy.out,
        supercell=(1, 1, 1), fc2_supercell=(1, 1, 1), mesh=(3, 3, 3), pair_cutoff=2., **kwargs)


def test_verified_dependency_api_signatures():
    from phonopy import Phonopy
    from phono3py import Phono3py
    assert "calculate_full_force_constants" in inspect.signature(Phonopy.produce_force_constants).parameters
    assert "cutoff_pair_distance" in inspect.signature(Phono3py.generate_displacements).parameters
    assert "is_compact_fc" in inspect.signature(Phono3py.produce_fc3).parameters
    assert "use_symfc_projector" in inspect.signature(Phono3py.symmetrize_fc3).parameters
    assert "is_LBTE" in inspect.signature(Phono3py.run_thermal_conductivity).parameters
    assert "is_isotope" in inspect.signature(Phono3py.run_thermal_conductivity).parameters
    assert "mass_variances" in inspect.signature(Phono3py.run_thermal_conductivity).parameters


def test_conversion_masses_and_input_guards():
    atoms = bulk("C", "diamond", a=3.56054)
    atoms.set_masses([12., 13.])
    converted = workflow.ase_to_phonopy(atoms)
    np.testing.assert_array_equal(converted.masses, [12., 13.])
    np.testing.assert_allclose(converted.positions, atoms.positions)
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match="fully periodic"):
        workflow.ase_to_phonopy(atoms)
    for bad in ([1, 2], [1, .5, 2], [0, 1, 1], [np.nan, 1, 1]):
        with pytest.raises(ValueError):
            workflow._triple(bad, "test")


@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_band_forces_resume_and_rebuild(toy):
    report = band(toy)
    assert report["status"] == "complete" and report["convergence"]["converged"] is False
    assert report["units"]["frequency"] == "THz" and report["units"]["fc2"] == "eV/Angstrom^2"
    assert toy.calculators[0].calls == 3 and ToyPhonopy.fits == 1
    with h5py.File(toy.out / "fc2-forces.hdf5") as handle:
        raw, corrected = handle["raw_forces"][:], handle["forces"][:]
        delta = raw[1:] - raw[0]
        np.testing.assert_allclose(corrected, delta - delta.mean(axis=1)[:, None, :], atol=1e-15)
        np.testing.assert_allclose(corrected.sum(axis=1), 0, atol=1e-15)
    # A bogus existing FC is never used as a resume input.
    (toy.out / "fc2.hdf5").write_bytes(b"not a valid old force constant file")
    again = band(toy, resume=True)
    assert ToyPhonopy.fits == 2 and toy.calculators[1].calls == 0
    assert again["force_evaluations"]["fc2"]["reused"] == 3
    with h5py.File(toy.out / "fc2.hdf5") as handle:
        assert handle.attrs["fingerprint"] == report["fingerprint"]
    assert (toy.out / "band.csv").exists() and (toy.out / "band.png").exists()
    assert not (toy.out / "run.lock").exists()


def test_dry_run_can_resume_but_new_protocol_cannot(toy):
    result = band(toy, dry_run=True)
    assert result["plan"]["supercell_atoms"] == 2 and toy.calculators[0].calls == 0
    assert not (toy.out / "fc2.hdf5").exists()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        band(toy, resume=True, displacement=.02)
    toy.model.write_text("changed model\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        band(toy, resume=True)


@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_geometry_checkpoint_rejects_stale_cache(toy):
    band(toy)
    path = toy.out / "fc2-force-cache" / "0000001.npz"
    with np.load(path, allow_pickle=False) as data:
        saved = dict(data)
    saved["geometry_sha256"] = "not the same geometry"
    np.savez_compressed(path, **saved)
    with pytest.raises(ValueError, match="stale force checkpoint"):
        band(toy, resume=True)
    assert json.loads((toy.out / "metadata.json").read_text())["status"] == "failed"


def test_force_planning_limit_and_writer_lock(toy):
    with pytest.raises(ValueError, match="displacement count"):
        band(toy, max_displacements=1)
    assert toy.calculators[0].calls == 0
    (toy.out / "run.lock").write_text("another process owns this run")
    with pytest.raises(FileExistsError):
        band(toy, resume=True, dry_run=True)


@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_kappa_included_mask_symmetry_and_units(toy):
    result = kappa(toy, temperatures=(200., 300.), lbte=True)
    assert result["result"] == "transport_computed" and result["kappa_status"] == "computed_unconverged"
    assert result["identity"]["protocol"]["solver"] == "LBTE"
    assert result["identity"]["protocol"]["isotope"] is False
    assert ToyPhono3py.transport_options[0]["is_isotope"] is False
    assert ToyPhono3py.transport_options[0]["mass_variances"] is None
    assert result["force_evaluations"]["fc3"]["excluded_pair_placeholders"] == 1
    assert ToyPhono3py.transport_calls == 1
    assert ToyPhono3py.symmetrizations == [(3, {"use_symfc_projector": False, "options": "level=3"}),
                                          (2, {"use_symfc_projector": False, "options": "level=3"})]
    with h5py.File(toy.out / "fc3-forces.hdf5") as handle:
        np.testing.assert_array_equal(handle["included"][:], [True, True, False, True])
    with h5py.File(toy.out / "kappa.hdf5") as handle:
        assert handle["kappa"].shape == (2, 6) and handle.attrs["unit"] == "W/(m K)"
        assert not handle.attrs["converged"]
        assert not handle.attrs["isotope"] and "gamma_isotope" not in handle


@pytest.mark.parametrize("lbte", [False, True])
@pytest.mark.parametrize("isotope", [False, True])
@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_isotope_rta_lbte_and_mass_invariance(toy, lbte, isotope):
    from phono3py.other.isotope import get_mass_variances
    masses = toy.atoms.get_masses().copy()
    expected_g = get_mass_variances(symbols=["C", "C"])
    result = kappa(toy, isotope=isotope, lbte=lbte)
    options = ToyPhono3py.transport_options[0]
    assert options["is_LBTE"] is lbte and options["is_isotope"] is isotope
    np.testing.assert_array_equal(toy.atoms.get_masses(), masses)
    protocol = result["identity"]["protocol"]
    assert protocol["isotope"] is isotope
    assert protocol["isotope_parameters"]["primitive_symbols"] == ["C", "C"]
    with h5py.File(toy.out / "kappa.hdf5") as handle:
        assert bool(handle.attrs["isotope"]) is isotope
        if isotope:
            np.testing.assert_array_equal(options["mass_variances"], expected_g)
            np.testing.assert_array_equal(result["isotope_parameters"]["mass_variances"], expected_g)
            assert result["isotope_scattering"]["positive_count"] > 0
            np.testing.assert_array_equal(handle["mass_variances"][:], expected_g)
            assert handle["primitive_symbols"][:].tolist() == [b"C", b"C"]
            assert np.isfinite(handle["gamma_isotope"][:]).all()
        else:
            assert options["mass_variances"] is None
            assert result["isotope_parameters"]["mass_variances"] is None
            assert "gamma_isotope" not in handle and "isotope_scattering" not in result


@pytest.mark.parametrize("flags", [[], ["--isotope"], ["--isotope", "--lbte"]])
def test_cli_isotope_forwarding(toy, monkeypatch, flags):
    import ase.io
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return dict(status="complete", result="TEST-ONLY", plan={}, convergence={"converged": False})

    package = ModuleType("pysuperlinear")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "pysuperlinear", package)
    monkeypatch.setitem(sys.modules, "pysuperlinear.phonons", SimpleNamespace(run_thermal_conductivity=run))
    monkeypatch.setattr(ase.io, "read", lambda *args, **kwargs: [toy.atoms])
    script = Path(__file__).parents[1] / "examples" / "lattice_kappa.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--model", "model.sus2", "--species", "C",
        "--structure", "relaxed.json", "--output", "new-output", "--full-pairs", *flags])
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["isotope"] is ("--isotope" in flags)
    assert calls[0]["lbte"] is ("--lbte" in flags)


def test_isotope_dry_run_fingerprint_and_resume_rejection(toy):
    result = kappa(toy, isotope=True, dry_run=True)
    assert result["identity"]["protocol"]["isotope"] is True
    assert result["isotope_parameters"]["primitive_symbols"] == ["C", "C"]
    assert len(result["isotope_parameters"]["mass_variances"]) == 2
    assert toy.calculators[0].calls == 0 and ToyPhono3py.transport_calls == 0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        kappa(toy, isotope=False, dry_run=True, resume=True)
    again = kappa(toy, isotope=True, dry_run=True, resume=True)
    assert again["fingerprint"] == result["fingerprint"]


@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_zero_natural_mass_variance_is_allowed(toy, monkeypatch):
    # Emulate a monoisotopic element's valid zero g; no positive-g assumption.
    import phono3py.other.isotope
    monkeypatch.setattr(phono3py.other.isotope, "get_mass_variances", lambda primitive: np.zeros(len(primitive)))
    result = kappa(toy, isotope=True)
    assert result["isotope_scattering"]["positive_count"] == 0
    np.testing.assert_array_equal(result["isotope_parameters"]["mass_variances"], [0., 0.])
    with h5py.File(toy.out / "kappa.hdf5") as handle:
        np.testing.assert_array_equal(handle["gamma_isotope"][:], 0.)


@pytest.mark.filterwarnings("ignore:.*undisplaced")
def test_unstable_modes_do_not_generate_fake_kappa(toy):
    ToyPhono3py.unstable = True
    with pytest.raises(workflow.UnstablePhononsError):
        kappa(toy)
    result = json.loads((toy.out / "metadata.json").read_text())
    assert result["stability"]["classification"] == "unstable"
    assert result["kappa_status"] == "not_computed_unstable"
    assert ToyPhono3py.transport_calls == 0 and not (toy.out / "kappa.hdf5").exists()
    assert (toy.out / "fc3.hdf5").exists() and (toy.out / "stability.hdf5").exists()


def test_stability_and_invalid_band_input():
    result = workflow._stability(np.asarray([-.01, -.3, 0, 4.]), .1)
    assert result["substantial_imaginary_count"] == result["small_negative_count"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        workflow._stability([np.nan], .1)
    with pytest.raises(ValueError, match="matching labels"):
        workflow._band_paths(Atoms(), {"paths": [[[0, 0, 0]]], "labels": [["G"]]}, 3)


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ("PYSUPERLINEAR_PHONON_SMOKE_MODEL", "PYSUPERLINEAR_PHONON_SMOKE_SPECIES", "PYSUPERLINEAR_PHONON_SMOKE_STRUCTURE")),
    reason="opt-in native/real-phonopy smoke on a compute allocation; explicit model/species/structure required")
def test_real_native_band_smoke(tmp_path):
    from ase.io import read
    from pysuperlinear.phonons import run_band_structure
    report = run_band_structure(read(os.environ["PYSUPERLINEAR_PHONON_SMOKE_STRUCTURE"]),
        os.environ["PYSUPERLINEAR_PHONON_SMOKE_MODEL"], os.environ["PYSUPERLINEAR_PHONON_SMOKE_SPECIES"].split(),
        tmp_path / "native-smoke", supercell=(1, 1, 1), band_path=PATH, npoints=3, stability_mesh=(2, 2, 2))
    assert report["status"] == "complete" and report["result"] == "band_computed"
