#!/usr/bin/env python3
"""Qualify the real MPI LAMMPS CPU adapter against native E/F/virial/site E.

The candidate native fixture tool must support ``eval-efs MODEL CFG`` and emit
``NATIVE_EFS_JSON {...}``. All selected legal cases run at every requested rank
count in direct and table modes. A failure is recorded, never silently skipped.
At least one requested rank count must be >=2: those counts also test prompt
rejection of hybrid filtering, neighbor exclusions, and rank-local overlaps.
NumPy is the only nonstandard Python dependency. This script submits no jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np


# Verified against LAMMPS units metal: pressure[bar] = virial[eV]/V[A^3]
# multiplied by force->nktv2p. Deliberately exclude the kinetic pressure term.
METAL_NKTV2P = 1.6021765e6


@dataclass
class Geometry:
    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray


def require(value, message):
    if not value:
        raise RuntimeError(message)


def validate_ranks(ranks):
    require(ranks and all(v > 0 for v in ranks), "Rank counts must be positive")
    require(len(ranks) == len(set(ranks)), "Rank counts must be unique")
    allocation = os.environ.get("LSB_DJOB_NUMPROC")
    if allocation is not None:
        try:
            available = int(allocation)
        except ValueError as error:
            raise RuntimeError("LSB_DJOB_NUMPROC must be a positive integer") from error
        require(available > 0, "LSB_DJOB_NUMPROC must be positive")
        require(max(ranks) <= available,
                f"Requested {max(ranks)} MPI ranks exceeds LSF allocation {available}")


def read_cfg(path):
    text = Path(path).read_text()
    block = re.search(r"BEGIN_CFG(.*?)END_CFG", text, re.S)
    require(block is not None, f"No CFG record in {path}")
    text = block[1]
    count = int(re.search(r"\bSize\s+(\d+)", text)[1])
    lines = re.search(r"Supercell\s*\n((?:[^\n]*\n){3})", text)[1].splitlines()
    cell = np.asarray([[float(v) for v in line.split()] for line in lines])
    match = re.search(r"AtomData:\s*([^\n]*)\n(.*)", text, re.S)
    columns = match[1].split()
    rows = [line.split() for line in match[2].splitlines()[:count]]
    require(len(rows) == count, "Incomplete CFG atom data")
    indices = [columns.index(key) for key in ("cartes_x", "cartes_y", "cartes_z")]
    positions = np.asarray([[float(row[k]) for k in indices] for row in rows])
    types = np.asarray([int(row[columns.index("type")]) for row in rows], dtype=int)
    require(np.isfinite(cell).all() and np.isfinite(positions).all(), "Non-finite CFG geometry")
    return Geometry(cell, positions, types)


def restricted(geometry):
    # Rows are cell vectors. A.T=Q R gives A Q=R.T, a restricted triclinic
    # box. Apply the same proper rotation to positions, then evaluate the
    # native oracle on this rotated geometry to avoid tensor-frame ambiguity.
    require(np.linalg.det(geometry.cell) > 0, "CFG cell must have positive volume")
    q, r = np.linalg.qr(geometry.cell.T)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q @ np.diag(signs)
    cell = np.diag(signs) @ r
    cell = cell.T
    require(np.linalg.det(q) > 0, "Cell conversion unexpectedly produced a reflection")
    cell[np.abs(cell) < 1e-14] = 0.0
    positions = geometry.positions @ q
    fractional = positions @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return Geometry(cell, fractional @ cell, geometry.types.copy())


def write_cfg(path, geometry):
    lines = ["BEGIN_CFG", " Size", f" {len(geometry.types)}", " Supercell"]
    lines += [" " + " ".join(f"{v:.17g}" for v in row) for row in geometry.cell]
    lines += [" AtomData: id type cartes_x cartes_y cartes_z"]
    lines += [f" {i+1} {int(t)} " + " ".join(f"{v:.17g}" for v in p)
              for i, (t, p) in enumerate(zip(geometry.types, geometry.positions))]
    Path(path).write_text("\n".join(lines + ["END_CFG", ""]))


def write_data(path, geometry, species, type_order=None):
    cell = geometry.cell
    require(np.max(np.abs(np.triu(cell, 1))) < 1e-12, "Expected restricted triclinic cell")
    lines = ["Native SUS2 CPU qualification", "", f"{len(geometry.types)} atoms",
             f"{species} atom types", "", f"0 {cell[0,0]:.17g} xlo xhi",
             f"0 {cell[1,1]:.17g} ylo yhi", f"0 {cell[2,2]:.17g} zlo zhi",
             f"{cell[1,0]:.17g} {cell[2,0]:.17g} {cell[2,1]:.17g} xy xz yz",
             "", "Atoms # atomic", ""]
    type_order = list(range(species)) if type_order is None else type_order
    native_to_lammps = {native: lammps + 1 for lammps, native in enumerate(type_order)}
    lines += [f"{i+1} {native_to_lammps[int(t)]} " + " ".join(f"{v:.17g}" for v in p)
              for i, (t, p) in enumerate(zip(geometry.types, geometry.positions))]
    Path(path).write_text("\n".join(lines) + "\n")


def read_dump(path):
    lines = Path(path).read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line == "ITEM: TIMESTEP"]
    require(starts, f"Missing atom dump: {path}")
    start = starts[-1]
    count = int(lines[start + 3])
    atom_line = next(i for i in range(start + 4, len(lines)) if lines[i].startswith("ITEM: ATOMS "))
    columns = lines[atom_line].split()[2:]
    data = np.asarray([[float(v) for v in line.split()]
                       for line in lines[atom_line+1:atom_line+1+count]])
    require(len(data) == count, "Incomplete LAMMPS dump")
    data = data[np.argsort(data[:, columns.index("id")])]
    require(np.array_equal(data[:, columns.index("id")], np.arange(1, count + 1)),
            "LAMMPS atom IDs changed or were lost")
    return {name: data[:, i] for i, name in enumerate(columns)}


def close(actual, expected, tolerance, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{name}: non-finite value")
    error = np.abs(actual - expected)
    allowed = tolerance + tolerance * np.maximum(np.abs(actual), np.abs(expected))
    require(np.all(error <= allowed), f"{name}: max abs {float(error.max()):.8g}; "
            f"max normalized {float(np.max(error / allowed)):.8g}")
    return float(np.max(error, initial=0.0))


def tokens(case):
    spec = case["spec"]
    family = spec["family"]
    result = {f"family:{family}": 10}
    if family == "legacy":
        result[f"legacy-graphs:{spec.get('gate')}:{spec.get('outer')}"] = 9
    if family == "pure":
        result[f"pure-gate:{spec.get('gate')}"] = 8
    if family == "adaptive":
        result[f"adaptive-projection:{spec.get('projection')}"] = 8
    for key in ("scale", "head", "source", "activation", "independent", "basis", "rank"):
        if key in spec:
            result[f"{key}:{spec[key]}"] = 2
    if spec.get("joint") or spec.get("message", 0):
        result[f"tensor-joint-l:{spec.get('l', 2)}"] = 10
    if spec.get("zbl"):
        result["zbl"] = 10
    return result


def select_cases(cases, maximum):
    if maximum == 0 or len(cases) <= maximum:
        return sorted(cases, key=lambda c: c["index"])
    chosen, covered = [], set()
    while len(chosen) < maximum:
        remaining = [c for c in cases if c not in chosen]
        selected = max(remaining, key=lambda c: (sum(w for k, w in tokens(c).items() if k not in covered),
                                                -int(c["index"])))
        chosen.append(selected)
        covered.update(tokens(selected))
    return sorted(chosen, key=lambda c: c["index"])


class Qualification:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        require(not self.out.exists() or (self.out.is_dir() and not any(self.out.iterdir())),
                f"Qualification output must be new or empty; preserve existing receipt and choose a unique attempt: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        # An exclusive marker reserves even an initially empty output folder
        # against concurrent invocations. Keep it with the immutable attempt.
        descriptor = os.open(self.out / ".qualification-attempt", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.report = {"status": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "lammps": str(args.lammps), "native": str(args.native),
                       "binary_sha256": hashlib.sha256(args.lammps.read_bytes()).hexdigest(),
                       "lammps_sha256": hashlib.sha256(args.lammps.read_bytes()).hexdigest(),
                       "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
                       "tolerance": {"direct": args.direct_tolerance, "table": args.table_tolerance},
                       "tabstep": args.tabstep, "runs": [], "negative_checks": []}

    def save(self):
        (self.out / "qualification.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def process(self, argv, cwd, log, timeout):
        # Each launcher owns a fresh local process group. Kill only that group
        # on timeout, including MPI proxies/children that outlive the launcher.
        # Never use a broad process-name match or kill unrelated MPI jobs.
        started = time.monotonic()
        process = subprocess.Popen([str(v) for v in argv], cwd=cwd, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        timed_out = False
        killed = False
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            output = error.output or ""
            try:
                os.killpg(process.pid, signal.SIGTERM)
                killed = True
            except ProcessLookupError:
                pass
            try:
                output, _ = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as error:
                output = error.output or output
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    killed = True
                except ProcessLookupError:
                    pass
                try:
                    output, _ = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    # Do not wait indefinitely for inherited output pipes.
                    process.stdout.close()
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2)
            # Even if the launcher exited and closed its pipes after TERM,
            # an MPI proxy could still be alive with detached output. Ensure
            # no surviving member of this specifically created group remains.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
            except ProcessLookupError:
                pass
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        Path(log).write_text(output)
        return {"returncode": process.returncode, "stdout": output,
                "elapsed_seconds": time.monotonic() - started, "timed_out": timed_out,
                "process_group_killed_on_timeout": killed}

    def command(self, argv, cwd, log):
        result = self.process(argv, cwd, log, self.args.timeout)
        require(not result["timed_out"], f"Command timed out; child process group terminated; log={log}")
        require(result["returncode"] == 0,
                f"Command failed ({result['returncode']}): {shlex.join(map(str, argv))}; log={log}")
        return result["stdout"]

    def negative(self, case, model, ranks, scenario):
        validate_ranks([ranks])
        require(ranks >= 2, "Negative communication regression requires >=2 MPI ranks")
        name = f"negative-{scenario}-mpi{ranks}"
        directory = self.out / name
        directory.mkdir(exist_ok=False)
        expected = {
            "hybrid": "sus2/superlinear does not support hybrid pair styles",
            "exclude": "neigh_modify include/exclude is not supported",
            "overlap": "distributed EFS neighborhood contains an invalid source or distance",
        }[scenario]
        record = {"name": name, "scenario": scenario, "ranks": ranks, "case": case["index"],
                  "spec": case["spec"],
                  "model": str(model), "expected_diagnostic": expected, "status": "RUNNING"}
        self.report["negative_checks"].append(record)
        self.save()
        try:
            cutoff = float(re.search(r"(?m)^\s*max_dist\s*=\s*([^\s]+)", model.read_text())[1])
            # Force an x-only partition. The coincident pair lies deep inside
            # rank zero's subdomain, outside every other rank's one-cutoff halo.
            # Other ranks therefore enter source forward communication without
            # independently discovering that invalid local pair.
            boundary = 40*cutoff/ranks
            geometry = Geometry(np.diag([40*cutoff, 3*cutoff, 3*cutoff]),
                                np.asarray([[2*cutoff, 1.5*cutoff, 1.5*cutoff],
                                            [2*cutoff, 1.5*cutoff, 1.5*cutoff],
                                            [boundary-.25*cutoff, 1.5*cutoff, 1.5*cutoff],
                                            [boundary+.25*cutoff, 1.5*cutoff, 1.5*cutoff],
                                            [28*cutoff, 1.5*cutoff, 1.5*cutoff]]),
                                np.asarray([0, 1, 0, 1, 0]))
            # The separate healthy pair straddles the first x partition. Rank
            # one genuinely needs a forward payload from the failing rank,
            # so the old collective error path cannot pass vacuously.
            if scenario != "overlap":
                geometry.positions[1, 0] += 0.35*cutoff
            write_data(directory / "system.data", geometry, 2)
            styles = ["units metal", "atom_style atomic", "boundary p p p", "newton on",
                      f"processors {ranks} 1 1", 'read_data "system.data"', "mass * 1.0"]
            if scenario == "hybrid":
                styles += [f"pair_style hybrid sus2/superlinear model_elements H,O radial_mode direct lj/cut {cutoff:.17g}",
                           f'pair_coeff * * sus2/superlinear "{model}" H O',
                           "pair_coeff 1 2 lj/cut 0.05 1.0"]
            else:
                styles += ["pair_style sus2/superlinear model_elements H,O radial_mode direct",
                           f'pair_coeff * * "{model}" H O']
            styles += ["neighbor 0.1 bin", "neigh_modify every 1 delay 0 check no"]
            if scenario == "exclude":
                styles += ["neigh_modify exclude type 1 2"]
            styles += ["run 0"]
            script = directory / "in.negative"
            script.write_text("\n".join(styles) + "\n")
            command = shlex.split(self.args.mpiexec) + ["-n", str(ranks), self.args.lammps,
                       "-in", script, "-log", directory / "lammps.log", "-screen", directory / "screen.log"]
            result = self.process(command, directory, directory / "launcher.log", self.args.negative_timeout)
            record.update({key: result[key] for key in ("returncode", "elapsed_seconds", "timed_out",
                                                       "process_group_killed_on_timeout")})
            require(not result["timed_out"], "Expected prompt rejection, but MPI timed out; child process group was terminated")
            require(result["returncode"] != 0, "Unsupported/invalid input unexpectedly succeeded")
            diagnostic = result["stdout"]
            for filename in ("screen.log", "lammps.log"):
                if (directory / filename).exists():
                    diagnostic += "\n" + (directory / filename).read_text()
            require(expected in diagnostic, f"Nonzero exit did not contain required diagnostic: {expected}")
            if scenario == "overlap":
                failing = sorted(set(int(v) for v in re.findall(r"ERROR on proc (\d+):", diagnostic)))
                require(failing == [0], f"Overlap must be rejected on rank zero only, observed ranks={failing}")
                record["diagnostic_ranks"] = failing
            record["status"] = "PASS"
            print("PASS", name, "expected rejection", flush=True)
        except Exception as error:
            record.update(status="FAIL", error=str(error))
            print("FAIL", name, str(error), flush=True)
        self.save()

    def oracle(self, model, geometry, directory, label):
        cfg = directory / f"{label}.cfg"
        write_cfg(cfg, geometry)
        output = self.command([self.args.native, "eval-efs", model, cfg], directory,
                              directory / f"{label}.native.log")
        records = [line.split("NATIVE_EFS_JSON", 1)[1].strip()
                   for line in output.splitlines() if line.startswith("NATIVE_EFS_JSON")]
        require(len(records) == 1, "Expected exactly one NATIVE_EFS_JSON oracle record")
        result = json.loads(records[0])
        require(all(k in result for k in ("energy", "forces", "virial", "site_energies")),
                "Native EFS oracle JSON is missing required fields")
        return result

    def run(self, case, model, geometry, ranks, mode, operation="run0"):
        validate_ranks([ranks])
        name = f"{int(case['index']):04d}-{case['spec']['family']}-{operation}-{mode}-mpi{ranks}"
        directory = self.out / name
        directory.mkdir(exist_ok=False)
        record = {"name": name, "case": case["index"], "spec": case["spec"],
                  "model": str(model), "ranks": ranks, "mode": mode, "operation": operation,
                  "status": "RUNNING"}
        self.report["runs"].append(record)
        self.save()
        try:
            text = model.read_text()
            species = int(re.search(r"(?m)^\s*species_count\s*=\s*(\d+)", text)[1])
            elements = case.get("_elements", ["H", "O"])
            type_order = case.get("_type_order", list(range(species)))
            require(species == len(elements), "Element list must match native model species count")
            require(sorted(type_order) == list(range(species)), "LAMMPS type order must be a model-species permutation")
            require(np.all((geometry.types >= 0) & (geometry.types < species)), "Fixture type outside native species map")
            record["model_elements"] = elements
            record["lammps_type_native_indices"] = type_order
            write_data(directory / "system.data", geometry, species, type_order)
            styles = ["units metal", "atom_style atomic", "boundary p p p", "newton on",
                      "atom_modify map array sort 0 0.0", 'read_data "system.data"', "mass * 1.0",
                      f"pair_style sus2/superlinear model_elements {','.join(elements)} radial_mode {mode} tabstep {self.args.tabstep:.17g}",
                      f'pair_coeff * * "{model}" ' + " ".join(elements[i] for i in type_order), "neighbor 0.4 bin",
                      "neigh_modify every 1 delay 0 check no", "compute site all pe/atom pair",
                      "compute pairpressure all pressure NULL pair"]
            styles += [f"variable w{k} equal c_pairpressure[{k}]*vol/{METAL_NKTV2P:.17g}"
                       for k in range(1, 7)]
            styles += ["thermo 1", "thermo_style custom step pe etotal " + " ".join(f"v_w{k}" for k in range(1, 7)),
                       "thermo_modify format float %.17g"]
            if operation == "nve":
                styles += ["velocity all create 5.0 4928459 mom yes rot no dist gaussian", "timestep 0.00001"]

            def snapshot(label, steps):
                return [f"dump observation all custom 1 {label}.dump id type proc x y z fx fy fz c_site",
                        "dump_modify observation sort id format float %.17g first yes", f"run {steps}",
                        'print "QUAL_THERMO ' + label + ' $(pe:%.17g) ' +
                        " ".join(f"$(v_w{k}:%.17g)" for k in range(1, 7)) + ' $(etotal:%.17g)"',
                        "undump observation"]

            styles += snapshot("initial", 0)
            if operation == "migration":
                delta = np.asarray([0.61, 0.57, 0.53]) @ geometry.cell
                styles += ["displace_atoms all move " + " ".join(f"{v:.17g}" for v in delta) + " units box"]
                styles += snapshot("final", 0)
            elif operation == "nve":
                styles += ["fix dynamics all nve"] + snapshot("final", 10) + ["unfix dynamics"]
            script = directory / "in.qualification"
            script.write_text("\n".join(styles) + "\n")
            launcher = shlex.split(self.args.mpiexec)
            self.command(launcher + ["-n", str(ranks), self.args.lammps, "-in", script,
                                    "-log", directory / "lammps.log", "-screen", directory / "screen.log"],
                         directory, directory / "launcher.log")
            screen = (directory / "screen.log").read_text()
            thermo = {}
            for line in screen.splitlines():
                if line.startswith("QUAL_THERMO "):
                    fields = line.split()
                    thermo[fields[1]] = np.asarray([float(v) for v in fields[2:]])
            labels = ["initial"] + ([] if operation in ("run0", "self-image") else ["final"])
            metrics = {}
            observed = {}
            tolerance = self.args.direct_tolerance if mode == "direct" else self.args.table_tolerance
            for label in labels:
                require(label in thermo and len(thermo[label]) == 8, "Missing high-precision LAMMPS E/virial output")
                atoms = read_dump(directory / f"{label}.dump")
                positions = np.column_stack([atoms[key] for key in ("x", "y", "z")])
                fractional = positions @ np.linalg.inv(geometry.cell)
                positions = (fractional - np.floor(fractional)) @ geometry.cell
                current = Geometry(geometry.cell, positions,
                                   np.asarray([type_order[int(t)-1] for t in atoms["type"]], dtype=int))
                reference = self.oracle(model, current, directory, label)
                forces = np.column_stack([atoms[key] for key in ("fx", "fy", "fz")])
                w = thermo[label][1:7]
                virial = np.asarray([[w[0], w[3], w[4]], [w[3], w[1], w[5]], [w[4], w[5], w[2]]])
                metrics[label] = {
                    "energy_max_abs": close(thermo[label][0], reference["energy"], tolerance, "energy"),
                    "force_max_abs": close(forces, reference["forces"], tolerance, "forces"),
                    "virial_max_abs": close(virial, reference["virial"], tolerance, "global virial"),
                    "site_max_abs": close(atoms["c_site"], reference["site_energies"], tolerance, "site energies"),
                    "site_sum_max_abs": close(np.sum(atoms["c_site"]), thermo[label][0], tolerance, "site sum"),
                    "occupied_ranks": sorted(set(atoms["proc"].astype(int).tolist())),
                }
                observed[label] = atoms
            if len(geometry.types) < ranks:
                require(len(metrics["initial"]["occupied_ranks"]) < ranks, "Empty-rank fixture unexpectedly populated every rank")
                record["verified_empty_rank"] = True
            if operation == "migration" and ranks > 1:
                changed = int(np.count_nonzero(observed["initial"]["proc"] != observed["final"]["proc"]))
                require(changed > 0, "Migration fixture did not change any owning rank")
                record["migrated_atoms"] = changed
            if operation == "nve":
                drift = float(abs(thermo["final"][-1] - thermo["initial"][-1]))
                close(thermo["final"][-1], thermo["initial"][-1], 2e-6, "NVE total energy conservation")
                record["nve_total_energy_drift"] = drift
            record.update(status="PASS", metrics=metrics)
            print("PASS", name, flush=True)
        except Exception as error:
            record.update(status="FAIL", error=str(error))
            print("FAIL", name, str(error), flush=True)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--ranks", default="1,2,4", help="MPI rank counts; include >=2 for required negative communication checks")
    parser.add_argument("--max-cases", type=int, default=12, help="0 selects every legal matrix case")
    parser.add_argument("--tabstep", type=float, default=0.001)
    parser.add_argument("--direct-tolerance", type=float, default=1e-9)
    parser.add_argument("--table-tolerance", type=float, default=2e-6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--negative-timeout", type=float, default=20,
                        help="Maximum seconds for each required >=2-rank expected-error test; timeout always fails and kills its local child process group")
    parser.add_argument("--extra-model", type=Path, help="Additional frozen model for direct/table and reordered type-map checks")
    parser.add_argument("--extra-cfg", type=Path, help="Geometry CFG in the additional model's native type order")
    parser.add_argument("--extra-elements", help="Comma-separated element symbols in the additional model's native species order")
    args = parser.parse_args()
    args.lammps, args.native = args.lammps.resolve(), args.native.resolve()
    args.matrix_dir = args.matrix_dir.resolve()
    require(args.lammps.is_file() and args.native.is_file(), "Candidate LAMMPS/native binaries must exist")
    require(args.max_cases >= 0 and args.tabstep > 0, "Invalid selection count or table spacing")
    ranks = [int(v) for v in args.ranks.split(",")]
    validate_ranks(ranks)
    require(any(v >= 2 for v in ranks), "Qualification requires at least one --ranks value >=2 for negative MPI regressions")
    require(args.negative_timeout > 0, "Negative-test timeout must be positive")
    matrix = json.loads((args.matrix_dir / "matrix.json").read_text())
    selected = select_cases(matrix["cases"], args.max_cases)
    require(selected, "No legal matrix cases were selected")
    qualification = Qualification(args)
    qualification.report["selected"] = [{"index": c["index"], "spec": c["spec"]} for c in selected]
    qualification.report["required_negative_ranks"] = [v for v in ranks if v >= 2]
    qualification.report["negative_timeout"] = args.negative_timeout
    qualification.report["unselected_indices"] = [c["index"] for c in matrix["cases"] if c not in selected]
    geometry = restricted(read_cfg(args.matrix_dir / "asymmetric.cfg"))
    models = {}
    for case in selected:
        model = args.matrix_dir / Path(case["model"]).name
        require(model.is_file(), f"Missing selected matrix model: {model}")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        require(not case.get("sha256") or digest == case["sha256"], f"Matrix model hash changed: {model}")
        models[case["index"]] = model
        for mode in ("direct", "table"):
            for count in ranks:
                qualification.run(case, model, geometry, count, mode)
    # Additional communication/physics cases use representative static and
    # tensor/joint or shared-n Adaptive models. These are extra executions,
    # never replacements for a failed primary model/rank/mode combination.
    edges = [next((c for c in selected if c["spec"]["family"] == "static"), selected[0])]
    adaptive = [c for c in selected if c["spec"]["family"] == "adaptive"]
    if adaptive:
        edges.append(max(adaptive, key=lambda c: (bool(c["spec"].get("joint")), int(c["spec"].get("l", 2)))))
    for case in edges:
        model = models[case["index"]]
        cutoff = float(re.search(r"(?m)^\s*max_dist\s*=\s*([^\s]+)", model.read_text())[1])
        periodic = Geometry(np.diag(np.asarray([0.72, 0.83, 0.94]) * cutoff),
                            np.asarray([[0.2, 0.2, 0.2]]) * cutoff, np.asarray([0]))
        for mode in ("direct", "table"):
            for count in ranks:
                qualification.run(case, model, periodic, count, mode, "self-image")
        for count in [v for v in ranks if v > 1]:
            qualification.run(case, model, geometry, count, "table", "migration")
    # Use the smallest adaptive topology for a brief, reproducible table NVE
    # check. The final geometry is independently re-evaluated by native code.
    dynamic = min(adaptive, key=lambda c: c["index"]) if adaptive else selected[0]
    for count in ranks:
        qualification.run(dynamic, models[dynamic["index"]], geometry, count, "table", "nve")
    negative_case = next((c for c in matrix["cases"] if c["spec"]["family"] == "legacy"), dynamic)
    negative_model = args.matrix_dir / Path(negative_case["model"]).name
    require(negative_model.is_file(), "Missing gated model for rank-local negative MPI checks")
    if negative_case.get("sha256"):
        require(hashlib.sha256(negative_model.read_bytes()).hexdigest() == negative_case["sha256"],
                "Negative MPI regression model hash changed")
    for count in [v for v in ranks if v >= 2]:
        for scenario in ("hybrid", "exclude", "overlap"):
            qualification.negative(negative_case, negative_model, count, scenario)
    if args.extra_model is not None or args.extra_cfg is not None or args.extra_elements is not None:
        require(args.extra_model is not None and args.extra_cfg is not None and args.extra_elements,
                "--extra-model, --extra-cfg and --extra-elements must be supplied together")
        model = args.extra_model.resolve()
        require(model.is_file() and args.extra_cfg.is_file(), "Extra model and geometry must exist")
        elements = args.extra_elements.split(",")
        require(all(elements) and len(elements) == len(set(elements)), "Extra element symbols must be nonempty and unique")
        extra_geometry = restricted(read_cfg(args.extra_cfg))
        for mapping, order in (("native", list(range(len(elements)))),
                               ("reverse", list(reversed(range(len(elements)))))):
            extra_case = {"index": 1000000 if mapping == "native" else 1000001,
                          "spec": {"family": "frozen-extra", "mapping": mapping},
                          "_elements": elements, "_type_order": order}
            for mode in ("direct", "table"):
                for count in ranks:
                    qualification.run(extra_case, model, extra_geometry, count, mode)
    failures = [run for run in qualification.report["runs"] + qualification.report["negative_checks"]
                if run["status"] != "PASS"]
    qualification.report.update(status="FAIL" if failures else "PASS", failures=len(failures),
                                completed=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    qualification.save()
    print(json.dumps({"status": qualification.report["status"], "runs": len(qualification.report["runs"]),
                      "negative_checks": len(qualification.report["negative_checks"]),
                      "failures": len(failures), "report": str(qualification.out / "qualification.json")}))
    return 1 if failures else 0


def compare_single_case(*, lammps, native, model, cfg, out, mpiexec="mpirun",
                        ranks=1, mode="direct", elements=("H", "O"),
                        type_order=None, tabstep=0.001, timeout=180,
                        direct_tolerance=1e-9, table_tolerance=2e-6):
    """Importable frozen-model single point, including an optional type map.

    ``elements`` is native species order; ``type_order`` maps LAMMPS types
    1..N to native indices 0..N-1. Raises on any mismatch and returns metrics.
    This helper needs no matrix directory and submits no scheduler job.
    """
    args = argparse.Namespace(lammps=Path(lammps).resolve(), native=Path(native).resolve(),
                              out=Path(out), mpiexec=mpiexec, tabstep=tabstep,
                              timeout=timeout, direct_tolerance=direct_tolerance,
                              table_tolerance=table_tolerance)
    validate_ranks([int(ranks)])
    case = {"index": 0, "spec": {"family": "frozen-single"}, "_elements": list(elements)}
    if type_order is not None:
        case["_type_order"] = list(type_order)
    check = Qualification(args)
    check.run(case, Path(model).resolve(), restricted(read_cfg(cfg)), int(ranks), mode)
    result = check.report["runs"][-1]
    check.report["status"] = result["status"]
    check.save()
    require(result["status"] == "PASS", result.get("error", "Frozen model qualification failed"))
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"QUALIFICATION_FATAL: {error}", file=sys.stderr)
        sys.exit(2)
