#!/usr/bin/env python3
"""LSF compute-only raw calc-descriptors/readout regression (no training).

Run this harness once, not under mpiexec. Each fresh --output contains all
fixtures, models, predictions, command logs and report.json. The same strict
checks apply to old binaries: an Adaptive descriptor bug is a failure, never an
expected-success exception. The default 12 cases include full/linear and
shared-n rank-2 linear/tanh Adaptive projection/response combinations. Optional
tensor cases add 2 more cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from adaptive_species_readout_check import NUMBER, assignment, numbers, put, put_numbers
from sh_effect_parameter_fast_check import active_text

ATOL, RTOL = 2e-10, 2e-10
SIGNAL = 1e-10
PERMUTATION = np.asarray([1, 2, 0])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture() -> list[dict]:
    cell = np.asarray([[7.4, .21, .09], [.12, 7.8, .18], [.05, .16, 8.1]])
    pos = np.asarray([[1.17, 1.39, 1.52], [2.63, 2.11, 2.74], [4.22, 3.05, 1.88]])
    types = np.asarray([0, 1, 0])
    c, s = np.cos(.37), np.sin(.37)
    d, t = np.cos(.21), np.sin(.21)
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.asarray(
        [[d, 0, t], [0, 1, 0], [-t, 0, d]])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2e-15)
    return [dict(cell=cell, positions=pos, types=types),
            dict(cell=cell @ rotation.T, positions=(pos @ rotation.T)[PERMUTATION],
                 types=types[PERMUTATION])]


def cfg_text(frames: list[dict]) -> str:
    lines = []
    for frame in frames:
        lines += ["BEGIN_CFG", " Size", f" {len(frame['types'])}", " Supercell"]
        lines += [" " + " ".join(f"{x:.17e}" for x in row) for row in frame["cell"]]
        lines += [" AtomData: id type cartes_x cartes_y cartes_z"]
        lines += [f" {i + 1} {int(kind)} " + " ".join(f"{x:.17e}" for x in xyz)
                  for i, (kind, xyz) in enumerate(zip(frame["types"], frame["positions"]))]
        lines += ["END_CFG", ""]
    return "\n".join(lines)


def descriptors(path: Path, frames: list[dict], width: int) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    cursor, blocks = 0, []
    for index, frame in enumerate(frames):
        assert cursor < len(lines), f"{path}: missing block {index}"
        marker = re.fullmatch(r"#start\s+(\d+)", lines[cursor])
        assert marker and int(marker[1]) == len(frame["types"]), (path, index, "#start N")
        cursor += 1
        rows = []
        for kind in frame["types"]:
            assert cursor < len(lines), (path, index, "missing atom row")
            fields = lines[cursor].split()
            assert len(fields) == width + 1, (path, index, "dimension", len(fields), width + 1)
            assert re.fullmatch(r"\d+", fields[0]) and int(fields[0]) == kind, (
                path, index, "type/atom order", fields[0], int(kind))
            rows.append([float(value) for value in fields[1:]])
            cursor += 1
        block = np.asarray(rows, dtype=np.float64)
        assert block.shape == (len(frame["types"]), width) and np.isfinite(block).all(), path
        blocks.append(block)
    assert cursor == len(lines), (path, "extra descriptor blocks/rows")
    return blocks


def predictions(path: Path, frames: list[dict]) -> np.ndarray:
    text = path.read_text()
    blocks = re.findall(r"(?ms)^\s*BEGIN_CFG\s*\n(.*?)^\s*END_CFG\s*$", text)
    assert len(blocks) == len(frames), (path, "prediction block count", len(blocks))
    energies = []
    for index, (block, frame) in enumerate(zip(blocks, frames)):
        size = re.search(r"(?m)^\s*Size\s*\n\s*(\d+)\s*$", block)
        assert size and int(size[1]) == len(frame["types"]), (path, index, "size")
        atom = re.search(r"(?m)^\s*AtomData:\s*([^\n]*)\n", block)
        assert atom, (path, index, "AtomData")
        header = atom[1].split()
        rows = block[atom.end():].splitlines()[:len(frame["types"])]
        assert len(rows) == len(frame["types"]), (path, index, "rows")
        values = np.asarray([[float(value) for value in row.split()] for row in rows])
        assert values.shape[1] == len(header) and np.isfinite(values).all(), (path, index)
        np.testing.assert_array_equal(values[:, header.index("type")], frame["types"])
        np.testing.assert_array_equal(values[:, header.index("id")], np.arange(1, len(rows) + 1))
        xyz = values[:, [header.index("cartes_" + axis) for axis in "xyz"]]
        np.testing.assert_allclose(xyz, frame["positions"], atol=2e-12, rtol=2e-12)
        energy = re.search(r"(?m)^\s*Energy\s*\n\s*(" + NUMBER + r")\s*$", block)
        assert energy, (path, index, "missing/invalid energy")
        stress = re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", block)
        assert stress, (path, index, "missing stress")
        stress_values = np.asarray([float(value) for value in stress[1].split()])
        assert stress_values.shape == (6,) and np.isfinite(stress_values).all(), (path, index)
        assert all(name in header for name in ("fx", "fy", "fz")), (path, index, "missing forces")
        energies.append(float(energy[1]))
    assert np.isfinite(energies).all(), path
    return np.asarray(energies)


def reconstruct(text: str, phi: list[np.ndarray], frames: list[dict], q: int, j: int,
                readout: str) -> np.ndarray:
    shift, center = numbers(text, "shift_coeffs"), numbers(text, "species_coeffs")
    assert shift.shape == center.shape == (2,) and np.isfinite(shift).all() and np.isfinite(center).all()
    heads = numbers(text, "moment_coeffs").reshape(2 if readout == "species" else 1, q)
    joint = (numbers(text, "joint_readout_coeffs").reshape(2 if readout == "species" else 1, j)
             if j else np.empty((2 if readout == "species" else 1, 0)))
    assert np.isfinite(heads).all() and np.isfinite(joint).all()
    energies = []
    for block, frame in zip(phi, frames):
        types = frame["types"]
        factor = np.ones(len(types)) if readout == "species" else center[types]
        head_rows = types if readout == "species" else np.zeros(len(types), dtype=int)
        value = np.sum(block[:, :q] * heads[head_rows], axis=1)
        if j:
            value += np.sum(block[:, q:] * joint[head_rows], axis=1)
        energies.append(np.sum(shift[types] + center[types] + factor * value))
    return np.asarray(energies)


def topology(mode: str, readout: str, joint: bool, tensor: bool, *,
             projection: str = "full", activation: str = "none") -> list[str]:
    args = ["--species-count=2", "--l-max=1", "--k-max=2", "--body-order=3",
            "--body-l-max=1,1,1,1", "--radial-basis-size=3", "--cutoff=4.8",
            "--radial-basis-type=RBChebyshev_sss", "--scaling-map=K", "--outer-k-rank=2",
            "--sh-factor-pruning=q-total", "--write-sh-scalar-info", f"--readout-mode={readout}"]
    if mode != "static":
        args += ["--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
                 "--two-layer-gate-shared-radial", "--two-layer-gate-k-rank=2",
                 "--two-layer-gate-body-order=3", "--two-layer-gate-factor-pruning=q-total"]
    if mode == "adaptive":
        args += ["--superlinear-mode=radial-coefficient-gate", f"--adaptive-radial-projection={projection}",
                 f"--adaptive-radial-shared-n-rank={2 if projection == 'shared-n' else 1}",
                 f"--adaptive-radial-activation={activation}", "--adaptive-radial-amplitude-mode=species"]
    if joint:
        args += ["--joint-readout=true"]
    if tensor:
        args += ["--tensor-message-lmax=1"]
    return args


def populated(source: str, readout: str, joint: bool, tensor: bool) -> str:
    text = put(active_text(source), "scaling", "1.0")
    text = put_numbers(text, "shift_coeffs", [-.27, .43])
    text = put_numbers(text, "species_coeffs", [.8, 1.4])
    q = int(assignment(text, "alpha_scalar_moments"))
    heads = .12 + .05 * np.sin(.41 * np.arange(q) + .2)
    if readout == "species":
        heads = np.concatenate([heads, -.2 + .07 * np.cos(.37 * np.arange(q) + .4)])
    assert len(numbers(text, "moment_coeffs")) == len(heads)
    text = put_numbers(text, "moment_coeffs", heads)
    if re.search(r"(?m)^\s*adaptive_radial_weights\s*=", text):
        count = len(numbers(text, "adaptive_radial_weights"))
        assert count > 0
        text = put_numbers(text, "adaptive_radial_weights", .7 + .2 * np.sin(.43 * np.arange(count) + .3))
        count = len(numbers(text, "adaptive_radial_amplitudes"))
        text = put_numbers(text, "adaptive_radial_amplitudes", .8 + .1 * np.cos(.29 * np.arange(count)))
    if joint:
        j = int(assignment(text, "joint_readout_scalar_count"))
        assert j > 0
        values = .09 + .04 * np.sin(.31 * np.arange(j) + .2)
        if readout == "species":
            values = np.concatenate([values, -.11 + .05 * np.cos(.47 * np.arange(j) + .3)])
        assert len(numbers(text, "joint_readout_coeffs")) == len(values)
        text = put_numbers(text, "joint_readout_coeffs", values)
    if tensor:
        count = len(numbers(text, "tensor_message_coeffs"))
        assert count > 0
        text = put_numbers(text, "tensor_message_coeffs", .06 + .03 * np.sin(.39 * np.arange(count) + .27))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, non-existing artifact directory")
    parser.add_argument("--with-tensor", action="store_true", help="add Adaptive+joint+tensor shared/species")
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not os.environ.get("LSB_JOBID"):
        parser.error("numerical verification requires a Linux LSF compute allocation")
    if any(int(os.environ.get(key, "1")) != 1 for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")):
        parser.error("run the harness once, not under multi-rank mpiexec")
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frames = fixture()
    immutable = {str(binary): sha(binary)}
    report = dict(status="RUNNING", protocol="raw descriptor contract; no training; loaded linear_mults=1",
                  job=os.environ["LSB_JOBID"], binary=str(binary), atol=ATOL, rtol=RTOL,
                  signal_floor=SIGNAL, rotation_permutation=PERMUTATION.tolist(), cases=[], commands=[])
    env = {key: value for key, value in os.environ.items() if not key.startswith("SUS2_SH_")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_DYNAMIC="FALSE",
               SUS2_SH_ANGULAR_BACKEND="ry")

    def save():
        report["immutable_sha256"] = immutable
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def protect(path):
        assert str(path) not in immutable, (path, "immutable path reused")
        immutable[str(path)] = sha(path)

    def write(path, text):
        assert not path.exists(), (path, "refuse overwrite")
        path.write_text(text)
        protect(path)

    def unchanged():
        for path, digest in immutable.items():
            assert sha(Path(path)) == digest, (path, "input changed")

    def run(label, *command):
        record = dict(label=label, command=[str(binary), *map(str, command)])
        report["commands"].append(record)
        result = subprocess.run(record["command"], cwd=out, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (out / f"{label}.log").write_text(result.stdout)
        record["returncode"] = result.returncode
        unchanged()
        save()
        if result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}; see {out / (label + '.log')}")

    cfg = out / "asymmetric-two-frame.cfg"
    write(cfg, cfg_text(frames))

    def evaluate(label, text, q, j, readout):
        model = out / f"{label}.sus2"
        raw, prediction = out / f"{label}.descriptors.txt", out / f"{label}.prediction.cfg"
        write(model, text)
        run(label + "-descriptors", "calc-descriptors", model, cfg, raw)
        run(label + "-efs", "calc-efs", model, cfg, prediction, "--no-loss")
        phi, energy = descriptors(raw, frames, q + j), predictions(prediction, frames)
        restored = reconstruct(text, phi, frames, q, j, readout)
        np.testing.assert_allclose(restored, energy, atol=ATOL, rtol=RTOL,
                                   err_msg=label + ": raw phi must reconstruct calc-efs energy")
        np.testing.assert_allclose(phi[1], phi[0][PERMUTATION], atol=ATOL, rtol=RTOL,
                                   err_msg=label + ": rotation/permutation or atom order")
        np.testing.assert_allclose(energy[0], energy[1], atol=ATOL, rtol=RTOL)
        assert np.max(np.abs(phi[0][0] - phi[0][2])) > SIGNAL, (label, "same-type row-order check vacuous")
        return phi, energy, dict(label=label, energy=energy.tolist(), reconstructed_energy=restored.tolist(),
                                reconstruction_max_abs=float(np.max(np.abs(restored - energy))),
                                rotation_max_abs=float(np.max(np.abs(phi[1] - phi[0][PERMUTATION]))))

    try:
        # Preserve the original 8-case order: old binaries first fail on the
        # same fifth (Adaptive/full/shared) case, before added v1.7 coverage.
        specs = [(mode, joint, False, "full", "none") for mode, joint in
                 (("static", False), ("effect", False), ("adaptive", False), ("adaptive", True))]
        specs += [("adaptive", False, False, "shared-n", activation) for activation in ("none", "tanh")]
        if args.with_tensor:
            specs += [("adaptive", True, True, "full", "none")]
        shared_n_linear = {}
        for mode, joint, tensor, projection, activation in specs:
            for readout in ("shared", "species"):
                suffix = ("-shared-n-" + ("linear" if activation == "none" else activation)
                          if projection == "shared-n" else "")
                label = mode + ("-joint" if joint else "") + ("-tensor" if tensor else "") + suffix + "-" + readout
                top, full = out / f"{label}.topology.sus2", out / f"{label}.materialized.sus2"
                run(label + "-init", "init-sh", top,
                    *topology(mode, readout, joint, tensor, projection=projection, activation=activation))
                protect(top)
                run(label + "-materialize", "roundtrip-model-dev", top, full)
                protect(full)
                text = populated(full.read_text(), readout, joint, tensor)
                if projection == "shared-n":
                    assert assignment(text, "adaptive_radial_projection") == "shared-n"
                    assert int(assignment(text, "adaptive_radial_shared_n_rank")) == 2
                    assert assignment(text, "adaptive_radial_activation") == activation
                q = int(assignment(text, "alpha_scalar_moments"))
                j = int(assignment(text, "joint_readout_scalar_count")) if joint else 0
                assert q > 0
                base_phi, base_energy, base = evaluate(label + "-base", text, q, j, readout)
                case = dict(label=label, mode=mode, readout=readout, q=q, j=j, tensor=tensor,
                            projection=projection if mode == "adaptive" else None,
                            activation=activation if mode == "adaptive" else None,
                            shared_n_rank=2 if projection == "shared-n" else None,
                            atoms_per_block=3, blocks=2, checks=[base])
                report["cases"].append(case)
                if projection == "shared-n":
                    if activation == "none":
                        shared_n_linear[readout] = (base_phi, base_energy)
                    else:
                        linear_phi, linear_energy = shared_n_linear[readout]
                        response_phi = max(float(np.max(np.abs(a - b)))
                                           for a, b in zip(base_phi, linear_phi))
                        response_energy = float(np.max(np.abs(base_energy - linear_energy)))
                        assert response_phi > SIGNAL and response_energy > SIGNAL, (
                            label, "tanh-vs-linear response check vacuous", response_phi, response_energy)
                        case["tanh_vs_linear"] = dict(descriptor_change_max_abs=response_phi,
                                                      energy_change_max_abs=response_energy)
                changes = [("heads", put_numbers(text, "moment_coeffs", numbers(text, "moment_coeffs") + .37)),
                           ("bias", put_numbers(put_numbers(text, "shift_coeffs", [.34, -.18]),
                                                "species_coeffs", [1.15, .62]))]
                if joint:
                    changes += [("joint-heads", put_numbers(text, "joint_readout_coeffs",
                                                           numbers(text, "joint_readout_coeffs") + .29))]
                    assert max(float(np.max(np.abs(block[:, q:]))) for block in base_phi) > SIGNAL, (
                        label, "joint columns vacuous")
                for kind, changed in changes:
                    phi, energy, check = evaluate(label + "-" + kind, changed, q, j, readout)
                    for actual, reference in zip(phi, base_phi):
                        # No descriptor computation may depend on a final head or additive bias.
                        np.testing.assert_array_equal(actual, reference, err_msg=label + ": raw readout dependence")
                    delta = float(np.max(np.abs(energy - base_energy)))
                    assert delta > SIGNAL, (label, kind, "head/bias edit did not change energy", delta)
                    check["raw_unchanged_exactly"] = True
                    check["energy_change_max_abs"] = delta
                    case["checks"].append(check)
                if mode == "adaptive":
                    weights = numbers(text, "adaptive_radial_weights")
                    assert np.all(weights != 0), (label, "nonzero-U fixture")
                    zero = put_numbers(text, "adaptive_radial_weights", np.zeros_like(weights))
                    phi, energy, check = evaluate(label + "-zero-U", zero, q, j, readout)
                    delta_phi = max(float(np.max(np.abs(a - b))) for a, b in zip(phi, base_phi))
                    delta_energy = float(np.max(np.abs(energy - base_energy)))
                    assert delta_phi > SIGNAL and delta_energy > SIGNAL, (
                        label, "Adaptive U sensitivity missing", delta_phi, delta_energy)
                    check.update(descriptor_change_max_abs=delta_phi, energy_change_max_abs=delta_energy)
                    case["checks"].append(check)
                save()
                print(label, "PASS", f"Q={q} J={j}", flush=True)
        unchanged()
        report["status"] = "PASS"
        report["inputs_unchanged"] = True
        report["case_count"] = len(report["cases"])
        save()
        print("PASS", len(report["cases"]), "cases;", out / "report.json", flush=True)
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        try:
            unchanged()
            report["inputs_unchanged"] = True
        except AssertionError as changed:
            report["inputs_unchanged"] = False
            report["immutability_error"] = str(changed)
        save()
        raise


if __name__ == "__main__":
    main()
