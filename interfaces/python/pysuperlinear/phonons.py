"""Finite-displacement phonopy/phono3py workflows using native Superlinear forces.

Optional dependencies: ``pip install pysuperlinear[phonon]``. Public entry
points accept an explicit fully periodic ASE unit cell and the FULL model type
order. No relaxation, unit-cell conversion, model fitting or legacy engine is
performed implicitly. Frequencies are THz; thermal conductivity is W/(m K).

API checked against phonopy 2.45.1 and phono3py 3.21.0 (see README). Forces are
checkpointed before fitting. Resume NEVER loads force constants: it verifies
the full fingerprint and each displaced geometry, then rebuilds FCs from forces.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import warnings

import numpy as np
from ase import Atoms

WORKFLOW_VERSION = 1
_WORKER_CALCULATOR = None


class UnstablePhononsError(RuntimeError):
    """Substantial imaginary modes prevent a meaningful lattice-kappa run."""


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(_jsonable(data), stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _finite(array, shape=None):
    array = np.asarray(array, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"nonfinite array or wrong shape: {array.shape}, expected {shape}")
    return array


def _triple(value, name):
    array = np.asarray(value)
    if array.shape != (3,) or not np.isfinite(array).all() or not np.equal(array, array.astype(int)).all() or np.any(array < 1):
        raise ValueError(f"{name} must contain three positive integers")
    return array.astype(int).tolist()


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _structure(atoms):
    if not isinstance(atoms, Atoms) or not len(atoms) or not np.all(atoms.pbc):
        raise ValueError("supply a nonempty, fully periodic ASE unit cell")
    if atoms.constraints:
        raise ValueError("phonon input must not contain ASE constraints")
    _finite(atoms.positions, (len(atoms), 3))
    cell = _finite(atoms.cell.array, (3, 3))
    if abs(np.linalg.det(cell)) < 1e-8 or np.linalg.svd(cell, compute_uv=False).min() < .5:
        raise ValueError("degenerate or too thin periodic unit cell")
    masses = _finite(atoms.get_masses(), (len(atoms),))
    if np.any(masses <= 0):
        raise ValueError("atomic masses must be positive")
    return dict(numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
                cell=cell.tolist(), masses=masses.tolist(), pbc=[True] * 3)


def ase_to_phonopy(atoms):
    """Preserve ASE Angstrom geometry, order and explicit masses (atomic mass units)."""
    from phonopy.structure.atoms import PhonopyAtoms
    _structure(atoms)
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                        positions=atoms.positions, masses=atoms.get_masses())


def _payload(cell):
    return dict(numbers=np.asarray(cell.numbers).tolist(), positions=np.asarray(cell.positions).tolist(),
                cell=np.asarray(cell.cell).tolist(), masses=np.asarray(cell.masses).tolist(), pbc=[True] * 3)


def _atoms(payload):
    return Atoms(numbers=payload["numbers"], positions=payload["positions"],
                 cell=payload["cell"], masses=payload["masses"], pbc=True)


def _calculator(model, species):
    from .calculator import SuperlinearCalculator
    return SuperlinearCalculator(model, species=species)


def _init_worker(model, species):
    global _WORKER_CALCULATOR
    _WORKER_CALCULATOR = _calculator(model, species)


def _evaluate(payload, calculator):
    atoms = _atoms(payload)
    atoms.calc = calculator
    forces = _finite(atoms.get_forces(), (len(atoms), 3)).copy()
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError("nonfinite native energy")
    return energy, forces


def _worker(payload):
    return _evaluate(payload, _WORKER_CALCULATOR)


def _versions():
    result = {"python": platform.python_version()}
    for package in ("pysuperlinear", "ase", "numpy", "phonopy", "phono3py", "h5py", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


@contextmanager
def _run(atoms, model, species, output, protocol, resume, workers):
    model = Path(model).expanduser().resolve(strict=True)
    species = tuple(species)
    if not species or len(set(species)) != len(species) or not set(atoms.get_chemical_symbols()).issubset(species):
        raise ValueError("supply the FULL unique model species order containing every structure element")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    structure = _structure(atoms)
    calculator = _calculator(str(model), species)
    identity = dict(schema=WORKFLOW_VERSION, model_sha256=_sha(model), structure=structure,
                    species=list(species), protocol=protocol, versions=_versions(),
                    workflow_source_sha256=_sha(__file__), engine=_jsonable(calculator.metadata))
    # Absolute model path is provenance, not physics. Allow an identical model
    # to move, but still bind resume to engine revision and all numerical input.
    identity["engine"].pop("model_path", None)
    fingerprint = _digest(identity)
    out = Path(output).expanduser().resolve()
    if resume:
        manifest = json.loads((out / "metadata.json").read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint mismatch (model/structure/species/protocol/engine/versions)")
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(fingerprint=fingerprint, identity=identity)
    lock = out / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    os.close(descriptor)
    manifest.update(status="running", model_path=str(model), workers=workers,
                    convergence=dict(converged=False, reason="single protocol; mesh, size, displacement and FC3 range convergence not established"),
                    units=dict(length="Angstrom", energy="eV", force="eV/Angstrom", mass="atomic mass unit",
                               fc2="eV/Angstrom^2", fc3="eV/Angstrom^3", frequency="THz", kappa="W/(m K)"),
                    thread_environment={key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    _atomic_json(out / "metadata.json", manifest)
    state = dict(out=out, model=model, species=species, calculator=calculator,
                 fingerprint=fingerprint, metadata=manifest, workers=workers)
    try:
        if workers > 1 and any(os.environ.get(key) != "1" for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            warnings.warn("set OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1 before Python when using force workers")
        from ase.io import write
        write(out / "unitcell.extxyz", atoms)
        yield state
        if _sha(model) != identity["model_sha256"]:
            raise RuntimeError("model file changed during calculation")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _atomic_json(out / "metadata.json", manifest)
        lock.unlink()


def _raw_forces(cells, state, label):
    """One persistent native calculator per process; atomic per-geometry cache."""
    cells = list(cells)
    natom = len(next(cell for cell in cells if cell is not None))
    cache = state["out"] / (label + "-force-cache")
    cache.mkdir(exist_ok=True)
    raw = np.zeros((len(cells), natom, 3), dtype=np.float64)
    energies = np.zeros(len(cells), dtype=np.float64)
    included = np.asarray([cell is not None for cell in cells], dtype=bool)
    pending = []
    reused = 0

    def store(index, energy, forces):
        forces = _finite(forces, (natom, 3))
        if not np.isfinite(energy):
            raise ValueError("nonfinite energy in force checkpoint")
        path = cache / f"{index:07d}.npz"
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, fingerprint=state["fingerprint"], geometry_sha256=_digest(_payload(cells[index])),
                                forces=forces, energy=float(energy))
        os.replace(temporary, path)
        raw[index], energies[index] = forces, energy

    for index, cell in enumerate(cells):
        if cell is None:
            continue  # Explicit phono3py excluded-pair placeholder, NOT a computed zero-force structure.
        if len(cell) != natom:
            raise ValueError("inconsistent displacement supercell sizes")
        path = cache / f"{index:07d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != state["fingerprint"] or str(data["geometry_sha256"]) != _digest(_payload(cell)):
                    raise ValueError(f"stale force checkpoint: {path}")
                raw[index] = _finite(data["forces"], (natom, 3))
                energies[index] = float(data["energy"])
                if not np.isfinite(energies[index]):
                    raise ValueError(f"nonfinite cached energy: {path}")
            reused += 1
        else:
            pending.append(index)
    if _sha(state["model"]) != state["metadata"]["identity"]["model_sha256"]:
        raise RuntimeError("model file changed before force evaluation")
    if state["workers"] == 1:
        for index in pending:
            store(index, *_evaluate(_payload(cells[index]), state["calculator"]))
    elif pending:
        # spawn avoids inheriting initialized BLAS/native state; bounded queue
        # avoids retaining thousands of force structures in submitted futures.
        with ProcessPoolExecutor(max_workers=state["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(str(state["model"]), state["species"])) as pool:
            iterator = iter(pending)
            active = {}
            def submit():
                index = next(iterator, None)
                if index is not None:
                    active[pool.submit(_worker, _payload(cells[index]))] = index
            for _ in range(2 * state["workers"]):
                submit()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    index = active.pop(future)
                    store(index, *future.result())
                    submit()
    state["metadata"].setdefault("force_evaluations", {})[label] = dict(
        requested=len(cells), evaluated=len(pending), reused=reused, excluded_pair_placeholders=int((~included).sum()))
    _atomic_json(state["out"] / "metadata.json", state["metadata"])
    return raw, energies, included


def _force_sets(supercell, displaced, state, label, subtract_reference, remove_drift):
    import h5py
    raw, energies, included = _raw_forces([supercell, *displaced], state, label)
    forces = raw[1:].copy()
    if subtract_reference:
        forces[included[1:]] -= raw[0]
    drift = np.mean(forces, axis=1)
    if remove_drift:
        forces[included[1:]] -= drift[included[1:], None, :]
    with h5py.File(state["out"] / (label + "-forces.hdf5"), "w") as handle:
        handle.attrs.update(fingerprint=state["fingerprint"], force_unit="eV/Angstrom")
        handle.create_dataset("raw_forces", data=raw, compression="gzip")
        handle.create_dataset("forces", data=forces, compression="gzip")
        handle.create_dataset("energies_eV", data=energies)
        handle.create_dataset("included", data=included)
        handle.create_dataset("drift_before_correction", data=drift)
    residual = float(np.max(np.linalg.norm(raw[0], axis=1)))
    state["metadata"].setdefault("force_diagnostics", {})[label] = dict(
        undisplaced_max_force_eV_A=residual, max_drift_eV_A=float(np.max(np.abs(drift))),
        subtract_reference=subtract_reference, remove_drift=remove_drift)
    if residual > .01:
        warnings.warn(f"{label}: undisplaced max force {residual:.4g} eV/A; relax and verify the intended equilibrium first")
    return _finite(forces, (len(displaced), len(supercell), 3))


def _stability(frequencies, tolerance):
    frequencies = _finite(frequencies)
    minimum = float(frequencies.min())
    substantial = int(np.count_nonzero(frequencies < -tolerance))
    return dict(min_frequency_THz=minimum, imaginary_tolerance_THz=tolerance,
                substantial_imaginary_count=substantial,
                small_negative_count=int(np.count_nonzero((frequencies < 0) & (frequencies >= -tolerance))),
                classification="unstable" if substantial else "no_substantial_imaginary_modes_on_sampled_points",
                sampled_values=int(frequencies.size))


def _symmetrize(obj, third, method):
    if method not in ("traditional", "symfc", "none"):
        raise ValueError("symmetrization must be traditional, symfc or none")
    if method == "none":
        warnings.warn("force-constant ASR/permutation symmetrization is disabled")
        return
    if third:
        kwargs = dict(use_symfc_projector=method == "symfc")
        if method == "traditional":
            kwargs["options"] = "level=3"
        obj.symmetrize_fc3(**kwargs)
        obj.symmetrize_fc2(**kwargs)
    else:
        obj.symmetrize_force_constants(level=3, use_symfc_projector=method == "symfc")


def _band_paths(atoms, band_path, npoints):
    """ASE named paths, or JSON-like {paths: reduced vertices, labels: nested}."""
    if not isinstance(npoints, int) or npoints < 2:
        raise ValueError("npoints must be at least two, per segment")
    if band_path is None or isinstance(band_path, str):
        from ase.dft.kpoints import parse_path_string
        path = atoms.cell.bandpath(path=band_path)
        names = parse_path_string(path.path)
        vertices = [[path.special_points[name] for name in part] for part in names]
    else:
        vertices, names = band_path["paths"], band_path.get("labels")
        if names is None:
            names = [[f"q{i}" for i in range(len(part))] for part in vertices]
    if len(vertices) != len(names) or not vertices:
        raise ValueError("band paths/labels mismatch or empty path")
    paths, labels = [], []
    for part, part_names in zip(vertices, names):
        part = _finite(part)
        if part.ndim != 2 or part.shape[1] != 3 or len(part) < 2 or len(part) != len(part_names):
            raise ValueError("each band path needs >=2 reduced 3-vectors and matching labels")
        for index in range(len(part) - 1):
            paths.append(np.linspace(part[index], part[index + 1], npoints))
            labels.append([str(part_names[index]), str(part_names[index + 1])])
    return paths, labels


def _plot_band(out, paths, distances, frequencies, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    offset, ticks, names = 0., [], []
    with (out / "band.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segment", "point", "distance_A^-1", "qx", "qy", "qz", "branch", "frequency_THz"])
        for segment, (qpoints, distance, frequency, label) in enumerate(zip(paths, distances, frequencies, labels)):
            x = np.asarray(distance) - distance[0] + offset
            ax.plot(x, frequency, color="#c52b2f", linewidth=1)
            if not ticks or abs(ticks[-1] - x[0]) > 1e-12:
                ticks.append(float(x[0])); names.append(label[0])
            elif names[-1] != label[0]:
                names[-1] += "|" + label[0]
            ticks.append(float(x[-1])); names.append(label[1])
            for point, (qpoint, xx, freq) in enumerate(zip(qpoints, x, frequency)):
                for branch, value in enumerate(freq):
                    writer.writerow([segment, point, f"{xx:.17g}", *[f"{q:.17g}" for q in qpoint], branch, f"{value:.17g}"])
            offset = float(x[-1])
    ax.set_xticks(ticks, [name.replace("G", r"$\Gamma$") if name == "G" else name for name in names])
    ax.set_xlim(0, offset)
    ax.set_ylabel("Frequency (THz)", fontweight="bold", color="black")
    ax.axhline(0, color="black", linewidth=.6)
    ax.grid(axis="x", color="0.8", linewidth=.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold"); label.set_color("black")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / ("band." + extension), dpi=180)
    plt.close(fig)


def run_band_structure(atoms, model, species, output, *, supercell=(3, 3, 3), displacement=.01,
                       band_path=None, npoints=51, stability_mesh=(5, 5, 5), primitive_matrix="P",
                       symprec=1e-5, symmetrization="traditional", subtract_reference=True,
                       remove_drift=True, imaginary_tolerance=.1, workers=1, resume=False,
                       dry_run=False, max_displacements=50000):
    """Write FC2 HDF5, phonopy YAML, band YAML/CSV/PNG/PDF and metadata.

    Input is preferably an explicitly prepared primitive cell. Supercell sizes
    multiply the INPUT cell, not an automatically reduced conventional cell.
    Named band paths are ASE paths for the resulting phonopy primitive basis;
    custom vertices use fractional coordinates in that reciprocal basis.
    ``dry_run`` counts displacements without any force evaluation.
    """
    from phonopy import Phonopy
    from phonopy.file_IO import write_force_constants_to_hdf5
    import h5py
    supercell, stability_mesh = _triple(supercell, "supercell"), _triple(stability_mesh, "stability_mesh")
    protocol = dict(task="band", supercell=supercell, displacement=_positive(displacement, "displacement"),
                    band_path=band_path, npoints=npoints, stability_mesh=stability_mesh,
                    primitive_matrix=primitive_matrix, symprec=_positive(symprec, "symprec"),
                    symmetrization=symmetrization, subtract_reference=bool(subtract_reference), remove_drift=bool(remove_drift),
                    imaginary_tolerance=_positive(imaginary_tolerance, "imaginary_tolerance"))
    with _run(atoms, model, species, output, protocol, resume, workers) as state:
        ph = Phonopy(ase_to_phonopy(atoms), np.diag(supercell), primitive_matrix=primitive_matrix,
                     symprec=symprec, calculator="vasp")
        ph.generate_displacements(distance=displacement, is_plusminus=True)
        cells = ph.supercells_with_displacements
        state["metadata"]["plan"] = dict(input_atoms=len(atoms), primitive_atoms=len(ph.primitive),
                                          supercell_atoms=len(ph.supercell), displacement_count=len(cells))
        ph.save(filename=str(state["out"] / "phonopy_disp.yaml"))
        if len(cells) > max_displacements:
            raise ValueError("displacement count exceeds max_displacements; inspect plan and choose a new protocol/output")
        if dry_run:
            state["metadata"]["result"] = "dry_run_no_forces_or_frequencies"
            return state["metadata"]
        ph.forces = _force_sets(ph.supercell, cells, state, "fc2", subtract_reference, remove_drift)
        ph.produce_force_constants(calculate_full_force_constants=False, fc_calculator="traditional")
        _symmetrize(ph, False, symmetrization)
        _finite(ph.force_constants)
        fcfile = state["out"] / "fc2.hdf5"
        write_force_constants_to_hdf5(ph.force_constants, filename=str(fcfile), p2s_map=ph.primitive.p2s_map,
                                      physical_unit="eV/angstrom^2", compression="gzip")
        with h5py.File(fcfile, "a") as handle:
            handle.attrs["fingerprint"] = state["fingerprint"]
        ph.save(filename=str(state["out"] / "phonopy_params.yaml"))
        basis = _atoms(_payload(ph.primitive))
        paths, labels = _band_paths(basis, band_path, npoints)
        ph.run_band_structure(paths)
        band = ph.band_structure
        frequencies = [_finite(value) for value in band.frequencies]
        ph.write_yaml_band_structure(filename=str(state["out"] / "band.yaml"))
        _plot_band(state["out"], paths, band.distances, frequencies, labels)
        ph.run_mesh(stability_mesh, is_gamma_center=True)
        mesh_frequencies = _finite(ph.mesh.frequencies)
        with h5py.File(state["out"] / "stability.hdf5", "w") as handle:
            handle.create_dataset("frequency_THz", data=mesh_frequencies)
            handle.create_dataset("qpoints", data=ph.mesh.qpoints)
        state["metadata"].update(result="band_computed", band_labels=labels,
            stability=_stability(np.concatenate([mesh_frequencies.ravel(), *[f.ravel() for f in frequencies]]), imaginary_tolerance),
            fc2_row_sum_max_eV_A2=float(np.max(np.abs(np.sum(ph.force_constants, axis=1)))))
    return state["metadata"]


def run_thermal_conductivity(atoms, model, species, output, *, supercell=(3, 3, 3),
                             fc2_supercell=None, displacement=.01, pair_cutoff=None,
                             mesh=(11, 11, 11), temperatures=(300.,), lbte=False,
                             primitive_matrix="P", symprec=1e-5, symmetrization="traditional",
                             subtract_reference=True, remove_drift=True, imaginary_tolerance=.1,
                             cutoff_frequency=1e-4, workers=1, resume=False, dry_run=False,
                             max_displacements=50000):
    """Compute intrinsic 3-phonon RTA (or explicit LBTE), never a surrogate kappa.

    ``pair_cutoff`` is phono3py's displacement-pair distance in Angstrom, NOT
    the model cutoff nor a proof of FC3 range convergence. None includes all
    pairs in the chosen supercell. No isotope/boundary/four-phonon scattering,
    NAC or convergence extrapolation is supplied. Substantial imaginary modes
    save diagnostics and raise before transport; no fake conductivity is saved.
    """
    from phono3py import Phono3py
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    import h5py
    supercell, mesh = _triple(supercell, "supercell"), _triple(mesh, "mesh")
    fc2_supercell = _triple(fc2_supercell, "fc2_supercell") if fc2_supercell is not None else None
    temperatures = [_positive(value, "temperature") for value in temperatures]
    if not temperatures:
        raise ValueError("supply at least one positive temperature")
    if pair_cutoff is not None:
        pair_cutoff = _positive(pair_cutoff, "pair_cutoff")
    protocol = dict(task="kappa", supercell=supercell, fc2_supercell=fc2_supercell,
                    displacement=_positive(displacement, "displacement"), pair_cutoff_A=pair_cutoff,
                    mesh=mesh, temperatures_K=temperatures, solver="LBTE" if lbte else "RTA",
                    primitive_matrix=primitive_matrix, symprec=_positive(symprec, "symprec"),
                    symmetrization=symmetrization, subtract_reference=bool(subtract_reference),
                    remove_drift=bool(remove_drift), imaginary_tolerance=_positive(imaginary_tolerance, "imaginary_tolerance"),
                    cutoff_frequency_THz=_positive(cutoff_frequency, "cutoff_frequency"),
                    integration="tetrahedron", isotope=False, boundary=False, nac=False)
    with _run(atoms, model, species, output, protocol, resume, workers) as state:
        ph3 = Phono3py(ase_to_phonopy(atoms), np.diag(supercell), primitive_matrix=primitive_matrix,
                       phonon_supercell_matrix=np.diag(fc2_supercell) if fc2_supercell else None,
                       symprec=symprec, calculator="vasp", cutoff_frequency=cutoff_frequency)
        ph3.generate_displacements(distance=displacement, cutoff_pair_distance=pair_cutoff, is_plusminus=True)
        cells = ph3.supercells_with_displacements
        fc2_cells = ph3.phonon_supercells_with_displacements if fc2_supercell else []
        count = sum(cell is not None for cell in cells) + len(fc2_cells)
        n, p = len(ph3.supercell), len(ph3.primitive)
        state["metadata"]["plan"] = dict(input_atoms=len(atoms), primitive_atoms=p, fc3_supercell_atoms=n,
            fc2_supercell_atoms=len(ph3.phonon_supercell) if fc2_supercell else n,
            fc3_displacements_evaluated=sum(cell is not None for cell in cells),
            fc3_excluded_pair_placeholders=sum(cell is None for cell in cells), fc2_displacements=len(fc2_cells),
            compact_fc3_bytes=p * n * n * 27 * 8, full_fc3_work_array_bytes=n ** 3 * 27 * 8,
            lbte_one_dense_collision_matrix_bytes=(int(np.prod(mesh)) * p * 3) ** 2 * 8 if lbte else None)
        state["metadata"]["range_warning"] = "Finite supercell and optional displacement-pair cutoff are unconverged; environment chains can extend FC interaction range beyond a pair cutoff."
        ph3.save(filename=str(state["out"] / "phono3py_disp.yaml"))
        if count > max_displacements:
            raise ValueError("displacement count exceeds max_displacements; inspect plan before starting forces")
        if dry_run:
            state["metadata"]["result"] = "dry_run_no_forces_or_transport"
            return state["metadata"]
        ph3.forces = _force_sets(ph3.supercell, cells, state, "fc3", subtract_reference, remove_drift)
        if fc2_supercell:
            ph3.phonon_forces = _force_sets(ph3.phonon_supercell, fc2_cells, state, "fc2", subtract_reference, remove_drift)
        ph3.produce_fc3(is_compact_fc=True, fc_calculator="traditional")
        if fc2_supercell:
            ph3.produce_fc2(is_compact_fc=True, fc_calculator="traditional")
        _symmetrize(ph3, True, symmetrization)
        _finite(ph3.fc2); _finite(ph3.fc3)
        write_fc3_to_hdf5(ph3.fc3, filename=str(state["out"] / "fc3.hdf5"), p2s_map=ph3.primitive.p2s_map)
        fc2_primitive = ph3.phonon_primitive if fc2_supercell else ph3.primitive
        write_fc2_to_hdf5(ph3.fc2, filename=str(state["out"] / "fc2.hdf5"), p2s_map=fc2_primitive.p2s_map,
                          physical_unit="eV/angstrom^2")
        for filename, unit in (("fc2.hdf5", "eV/Angstrom^2"), ("fc3.hdf5", "eV/Angstrom^3")):
            with h5py.File(state["out"] / filename, "a") as handle:
                handle.attrs.update(fingerprint=state["fingerprint"], physical_unit=unit)
        ph3.save(filename=str(state["out"] / "phono3py_params.yaml"))
        state["metadata"]["fc2_row_sum_max_eV_A2"] = float(np.max(np.abs(np.sum(ph3.fc2, axis=1))))
        state["metadata"]["fc3_second_third_index_sum_max_eV_A3"] = max(
            float(np.max(np.abs(np.sum(ph3.fc3, axis=axis)))) for axis in (1, 2))
        ph3.mesh_numbers = mesh
        ph3.init_phph_interaction()
        ph3.run_phonon_solver()
        frequencies, _, addresses = ph3.get_phonon_data()
        stability = _stability(frequencies, imaginary_tolerance)
        state["metadata"]["stability"] = stability
        with h5py.File(state["out"] / "stability.hdf5", "w") as handle:
            handle.create_dataset("frequency_THz", data=frequencies)
            handle.create_dataset("grid_address", data=addresses)
        if stability["substantial_imaginary_count"]:
            state["metadata"]["kappa_status"] = "not_computed_unstable"
            raise UnstablePhononsError("substantial imaginary phonons: inspect stability.hdf5/metadata.json; no meaningful kappa computed")
        ph3.run_thermal_conductivity(is_LBTE=bool(lbte), temperatures=temperatures, write_kappa=False)
        transport = ph3.thermal_conductivity
        if transport is None:
            raise RuntimeError("phono3py did not return a transport result")
        kappa = _finite(transport.kappa)
        if kappa.ndim != 3 or kappa.shape != (1, len(temperatures), 6):
            raise RuntimeError(f"unexpected phono3py kappa shape {kappa.shape}; expected one tetrahedron integration")
        kappa = kappa[0]
        with h5py.File(state["out"] / "kappa.hdf5", "w") as handle:
            handle.attrs.update(fingerprint=state["fingerprint"], unit="W/(m K)", solver=protocol["solver"], converged=False)
            handle.create_dataset("kappa", data=kappa)
            handle.create_dataset("temperature", data=temperatures)
            handle.create_dataset("mesh", data=mesh)
        with (state["out"] / "kappa.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["temperature_K", *["kappa_" + key + "_W_mK" for key in ("xx", "yy", "zz", "yz", "xz", "xy")]])
            for temperature, row in zip(temperatures, kappa):
                writer.writerow([f"{temperature:.17g}", *[f"{value:.17g}" for value in row]])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        for index, name in enumerate(("xx", "yy", "zz")):
            ax.plot(temperatures, kappa[:, index], "o-", label=name)
        ax.set_xlabel("Temperature (K)", fontweight="bold")
        ax.set_ylabel("Lattice thermal conductivity (W m$^{-1}$ K$^{-1}$)", fontweight="bold")
        ax.set_title(protocol["solver"] + " — unconverged protocol", fontweight="bold")
        ax.legend()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold"); label.set_color("black")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(state["out"] / ("kappa." + extension), dpi=180)
        plt.close(fig)
        state["metadata"].update(result="transport_computed", kappa_status="computed_unconverged",
                                  temperature_K=temperatures, kappa_W_mK=kappa.tolist())
    return state["metadata"]
