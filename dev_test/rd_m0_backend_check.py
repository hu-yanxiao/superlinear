#!/usr/bin/env python3
"""Server-only RY / RD_m0 native CLI equivalence and derivative qualification.

Every backend receives the identical, hashed MTP bytes in a fresh process.
No model conversion or parameter rescaling is part of this comparison.  The
public runtime presently loads L<=4; L=6 is checked as an explicit rejected
boundary, NOT counted as a successful numerical test.  The independent angular
unit test must qualify EvalD through L=6 separately.

Usage (inside an LSF compute allocation):
  python dev_test/rd_m0_backend_check.py /path/to/mlp-sus2 --outdir NEW_DIRECTORY
Optional MODEL_PATHS is a JSON list of {"model": "/abs/model.mtp", "cfg":
"/abs/data.cfg"}; exactly the first two complete configurations are copied to
the new result directory.  Sources are never modified.  This is a correctness
suite, not a benchmark or a deployment certificate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

# These modules are import-safe: their large suites only run under __main__.
from adaptive_species_readout_check import (
    NUMBER, ZERO_REG, assignment, asymmetric_shared_text, dev_fields,
    numbers, put, put_numbers, require_server, write_fixture,
)
from adaptive_tensor_message_check import read_fixture, write_geometry

BACKEND_ENV = "SUS2_SH_ANGULAR_BACKEND"
BACKENDS = (None, "ry", "rd-m0")
BANNER = "[RDM0_ANGULAR] backend=rd-m0"
EFS_ATOL = EFS_RTOL = 2e-10
GRAD_ATOL, GRAD_RTOL = 5e-9, 2e-9
FD_ATOL, FD_RTOL = 5e-8, 5e-5
GEOM_ATOL, GEOM_RTOL = 2e-6, 2e-4
SEED = 20260913


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction(path: Path) -> list[np.ndarray]:
    blocks = re.findall(r"(?ms)^BEGIN_CFG\s*\n.*?^END_CFG\s*$", path.read_text())
    assert blocks, f"no complete predicted CFG: {path}"
    result = []
    for block in blocks:
        energy = float(re.search(r"(?m)^\s*Energy\s*\n\s*(" + NUMBER + ")", block)[1])
        match = re.search(r"(?ms)^\s*AtomData:([^\n]*)\n(.*?)^\s*Energy", block)
        columns = match[1].split()
        indices = [columns.index(axis) for axis in ("fx", "fy", "fz")]
        rows = [row.split() for row in match[2].strip().splitlines()]
        force = [float(row[index]) for row in rows for index in indices]
        stress = [float(x) for x in re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", block)[1].split()]
        assert len(stress) == 6
        values = np.asarray([energy, *force, *stress], dtype=float)
        assert np.isfinite(values).all(), f"nonfinite prediction: {path}"
        result.append(values)
    return result


def compare(a: list[np.ndarray], b: list[np.ndarray], *, exact=False) -> dict:
    assert len(a) == len(b)
    differences = []
    for left, right in zip(a, b):
        if exact:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=EFS_ATOL, rtol=EFS_RTOL)
        differences.append(float(np.max(np.abs(left - right))))
    return {"configurations": len(a), "max_abs": max(differences),
            "exact": exact, "atol": 0.0 if exact else EFS_ATOL,
            "rtol": 0.0 if exact else EFS_RTOL}


def topology_options(spec: dict) -> list[str]:
    lmax = spec["lmax"]
    body = spec.get("body", 4)
    limits = [lmax] * (5 if body == 6 else 4)
    if body == 6:
        limits[-1] = min(lmax, 1)
    opts = ["--species-count=2", f"--l-max={lmax}", "--k-max=3",
            f"--body-order={body}", "--body-l-max=" + ",".join(map(str, limits)),
            "--radial-basis-size=3", "--radial-basis-type=RBChebyshev_sss",
            "--min-dist=1.1", "--cutoff=4.8", f"--scaling-map={spec.get('scaling', 'K')}",
            "--superlinear-mode=radial-coefficient-gate",
            f"--sh-factor-pruning={spec.get('outer', 'same-k')}", "--outer-k-rank=3",
            "--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
            "--two-layer-gate-k-rank=2", f"--two-layer-gate-body-order={body}",
            f"--two-layer-gate-factor-pruning={spec.get('gate', 'q-total')}",
            "--two-layer-gate-shared-radial", "--write-sh-scalar-info",
            f"--readout-mode={spec.get('readout', 'shared')}",
            f"--adaptive-radial-projection={spec.get('projection', 'full')}",
            f"--adaptive-radial-amplitude-mode={spec.get('amplitude', 'shared')}",
            f"--tensor-message-lmax={spec.get('tensor', 0)}"]
    if spec.get("joint"):
        opts.append("--joint-readout")
    return opts


def patterned(source: str, spec: dict) -> str:
    # This reused initializer changes only numerical arrays. Restore the
    # native-generated readout metadata; array length already matches it.
    # Native full/shared v1.2 omits this optional field when saving.
    readout = spec.get("readout", "shared")
    text = put(asymmetric_shared_text(source), "readout_mode", readout)
    if readout == "species":
        count = int(assignment(text, "alpha_scalar_moments"))
        heads = numbers(text, "moment_coeffs").reshape(2, count)
        heads[1] += .037 * np.cos(np.arange(count) * .29 + .11)
        text = put_numbers(text, "moment_coeffs", heads.ravel())
    if spec.get("tensor"):
        count = len(numbers(text, "tensor_message_coeffs"))
        assert count == spec["tensor"] * 3 * 2
        values = .11 * np.sin(.37 * np.arange(count) + .21) + .043 * np.cos(.19 * np.arange(count))
        text = put_numbers(text, "tensor_message_coeffs", values)
    if spec.get("joint"):
        count = len(numbers(text, "joint_readout_coeffs"))
        assert count > 0
        values = .03 + .06 * np.sin(.37 * np.arange(count) + .23)
        text = put_numbers(text, "joint_readout_coeffs", values)
    return text


class Check:
    def __init__(self, binary: Path, out: Path, reference_binary: Path | None = None):
        require_server()
        self.binary = binary.resolve(strict=True)
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.binary_sha = sha(self.binary)
        self.reference = reference_binary.resolve(strict=True) if reference_binary else None
        self.reference_sha = sha(self.reference) if self.reference else None
        self.report = {"status": "RUNNING", "binary": str(self.binary),
                       "binary_sha256": self.binary_sha, "seed": SEED,
                       "reference_binary": str(self.reference) if self.reference else None,
                       "reference_binary_sha256": self.reference_sha,
                       "lsb_jobid": os.environ["LSB_JOBID"],
                       "host": os.uname().nodename,
                       "scope": "native CLI FP64 mathematics; no speed/deployment claim",
                       "supported_runtime_lmax": [0, 1, 2, 3, 4],
                       "angular_unit_lmax6_is_separate": True,
                       "tolerances": {"EFS": [EFS_ATOL, EFS_RTOL],
                                      "RY_RD_analytic_gradient": [GRAD_ATOL, GRAD_RTOL],
                                      "parameter_FD": [FD_ATOL, FD_RTOL],
                                      "geometry_FD": [GEOM_ATOL, GEOM_RTOL]},
                       "script_sha256": sha(Path(__file__)),
                       "helper_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                                         ("adaptive_species_readout_check.py", "adaptive_tensor_message_check.py")},
                       "processes": [], "cases": {}, "extra": {}, "failures": []}
        self.cfg = self.out / "asymmetric.input.cfg"
        write_fixture(self.cfg)
        self.flush()

    def flush(self):
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def run(self, directory: Path, name: str, *args, backend=None, extra_env=None,
            allow_failure=False, calculation=False, timeout=300, binary=None):
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / (name + ".log")
        assert not log.exists(), f"refusing log overwrite: {log}"
        environment = dict(os.environ)
        # Ambient experimental switches must not silently alter the comparison.
        for key in tuple(environment):
            if key.startswith("SUS2_SH_"):
                del environment[key]
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        if backend is not None:
            environment[BACKEND_ENV] = backend
        environment.update(extra_env or {})
        command = [str(binary or self.binary), *map(str, args)]
        start = time.monotonic()
        try:
            proc = subprocess.run(command, cwd=directory, env=environment, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            output, code = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            output += "\nTEST_PROCESS_TIMEOUT\n"
            code = -999
        log.write_text(output)
        self.report["processes"].append({"command": command, "cwd": str(directory),
            "backend": "unset/default" if backend is None else backend,
            "extra_env": extra_env or {}, "exit_code": code, "log": str(log),
            "seconds": time.monotonic() - start, "log_sha256": sha(log)})
        self.flush()
        if not allow_failure:
            assert code == 0, f"{name}: exit={code}; see {log}\n{output[-3000:]}"
        if calculation and code == 0:
            if backend == "rd-m0":
                assert BANNER in output, f"RD selector not evidenced in {log}"
                for token in ("gate_reduce=RD_then_Yl0", "outer_reduce=RD_then_Yl0", "physical_aux_cache=1"):
                    assert token in output, (name, token)
            else:
                assert BANNER not in output, f"unexpected RD backend: {log}"
        return output, code

    def calc(self, directory, name, model, cfg=None, backend=None, extra_env=None, binary=None):
        cfg = cfg or self.cfg
        model, cfg = Path(model).resolve(), Path(cfg).resolve()
        target = (directory / (name + ".prediction.cfg")).resolve()
        assert target not in (model, cfg) and not target.exists()
        identity = (sha(model), sha(cfg))
        self.run(directory, name, "calc-efs", model, cfg, target, backend=backend,
                 extra_env=extra_env, calculation=True, binary=binary)
        assert identity == (sha(model), sha(cfg)), "a prediction modified its immutable inputs"
        return prediction(target)

    def parity(self, directory, name, model, cfg=None, extra_env=None):
        values = [self.calc(directory, name + "-" + (backend or "default"), model, cfg,
                            backend=backend, extra_env=extra_env) for backend in BACKENDS]
        result = {"default_vs_explicit_ry": compare(values[0], values[1], exact=True),
                  "ry_vs_rd_m0": compare(values[1], values[2])}
        if self.reference and extra_env is None:
            reference = self.calc(directory, name + "-reference", model, cfg, binary=self.reference)
            result["new_ry_vs_reference_default"] = compare(values[1], reference)
        return result, values[1]

    def direction(self, directory, name, model, begin, end, weights, backend):
        assert end > begin
        attempts, streak = [], 0
        # Fixed refinement schedule and original strict CLI bounds. Failed
        # coarse differences are retained; no permissive diagnostic flags.
        for h in (2e-5, 5e-6, 1.25e-6, 3.125e-7, 7.8125e-8):
            output, code = self.run(directory, f"{name}-{backend}-{h:.8g}",
                "check-loss-gradient-direction-dev", model, self.cfg,
                "--max-configs=1", f"--energy-weight={weights[0]}",
                f"--force-weight={weights[1]}", f"--stress-weight={weights[2]}", *ZERO_REG,
                f"--coeff-start={begin}", f"--coeff-end={end}", f"--displacement={h:.17g}",
                f"--abs-tolerance={FD_ATOL}", f"--rel-tolerance={FD_RTOL}",
                backend=backend, allow_failure=True, calculation=True)
            fields = dev_fields(output)
            required = ("analytic_directional", "finite_difference", "base_loss", "checked_coeffs")
            assert all(key in fields for key in required), (name, code, output[-1000:])
            a, fd = float(fields["analytic_directional"]), float(fields["finite_difference"])
            assert all(math.isfinite(float(fields[key])) for key in required)
            assert int(fields["checked_coeffs"]) == end - begin
            passed = code == 0 and bool(np.isclose(a, fd, atol=FD_ATOL, rtol=FD_RTOL))
            attempts.append({"h": h, "analytic": a, "finite_difference": fd,
                             "base_loss": float(fields["base_loss"]), "exit_code": code, "pass": passed})
            streak = streak + 1 if passed else 0
            if streak == 2:
                break
        assert streak == 2, (name, "no two consecutive strict FD passes", attempts)
        return {"begin": begin, "end": end, "weights": weights,
                "analytic": a, "attempts": attempts}

    def paired_direction(self, directory, name, model, begin, end, weights):
        ry = self.direction(directory, name, model, begin, end, weights, "ry")
        rd = self.direction(directory, name, model, begin, end, weights, "rd-m0")
        np.testing.assert_allclose(ry["analytic"], rd["analytic"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        return {"ry": ry, "rd_m0": rd, "analytic_difference": abs(ry["analytic"] - rd["analytic"])}

    def case(self, name, spec):
        directory = self.out / name
        directory.mkdir()
        record = self.report["cases"][name] = {"status": "RUNNING", "spec": spec}
        top, materialized, model = [directory / x for x in ("topology.mtp", "materialized.mtp", "patterned.mtp")]
        self.run(directory, "init-sh", "init-sh", top, *topology_options(spec))
        self.run(directory, "materialize", "roundtrip-model-dev", top, materialized)
        model.write_text(patterned(materialized.read_text(), spec))
        record.update(model=str(model), model_sha256=sha(model), cfg_sha256=sha(self.cfg))
        record["EFS"], values = self.parity(directory, "efs", model)
        assert float(np.linalg.norm(values[0][1:-6])) > 1e-8, "vacuous force fixture"
        roundtrips = []
        for backend in ("ry", "rd-m0"):
            path = directory / f"roundtrip-{backend}.mtp"
            self.run(directory, "roundtrip-" + backend, "roundtrip-model-dev", model, path, backend=backend)
            roundtrips.append(path.read_bytes())
        assert roundtrips[0] == roundtrips[1], "backend leaked into serialized model format/parameters"
        record["roundtrip_bytes_identical"] = True
        output, _ = self.run(directory, "layout", "check-loss-gradient-dev", model, self.cfg,
            "--max-configs=1", "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
            *ZERO_REG, "--coeff-start=0", "--coeff-end=1", "--displacement=5e-6",
            f"--abs-tolerance={FD_ATOL}", f"--rel-tolerance={FD_RTOL}", backend="ry", calculation=True)
        layout = dev_fields(output)
        record["layout"] = layout
        count = int(layout["coeff_count"])
        record["mixed_global"] = self.paired_direction(directory, "mixed-all", model, 0, count, [1., .1, .01])
        for backend in ("ry", "rd-m0"):
            output, _ = self.run(directory, "geometry-FD-" + backend, "check-efs-fd-dev", model, self.cfg,
                "--max-configs=1", "--max-atoms=3", "--displacement=2e-5",
                f"--abs-tolerance={GEOM_ATOL}", f"--rel-tolerance={GEOM_RTOL}",
                backend=backend, calculation=True)
            fields = dev_fields(output)
            assert int(fields["force_components"]) == 9 and int(fields["stress_components"]) == 9
            record["geometry_FD_" + backend] = fields
        if spec.get("deep"):
            self.deep(directory, record, model, layout, values)
        assert record["model_sha256"] == sha(model)
        record["status"] = "PASS"
        self.flush()
        print(f"PASS {name}", flush=True)
        return top, model

    def deep(self, directory, record, model, layout, baseline):
        names = ["gate_k_rank", "adaptive_radial_amplitude", "adaptive_radial_weight",
                 "tensor_message", "joint_readout", "radial", "gate_radial", "gate_type"]
        ranges = {key: (int(layout[key + "_begin"]), int(layout[key + "_end"])) for key in names}
        ranges["scaling"] = (2, int(layout["radial_begin"]))
        directions = record["block_directions"] = {}
        for key, (begin, end) in ranges.items():
            assert end > begin, (key, "required nonempty parameter block")
            directions[key] = self.paired_direction(directory, "mixed-" + key, model, begin, end, [1., .1, .01])
            # Componentwise FD supplements the directional contraction; it is
            # explicitly a two-coordinate sample, not an all-parameter claim.
            self.run(directory, "component-FD-" + key, "check-loss-gradient-dev", model, self.cfg,
                "--max-configs=1", "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
                *ZERO_REG, f"--coeff-start={begin}", f"--coeff-end={min(end, begin + 2)}",
                "--displacement=5e-6", f"--abs-tolerance={FD_ATOL}", f"--rel-tolerance={FD_RTOL}",
                backend="rd-m0", calculation=True)
        # Isolate F and virial: mixed E/F/S can conceal a missing force chain.
        isolated = record["isolated_objectives"] = {}
        for observable, weights in (("force", [0., 1., 0.]), ("virial", [0., 0., 1.])):
            isolated[observable] = {}
            for key in ("gate_k_rank", "adaptive_radial_weight", "tensor_message", "joint_readout"):
                begin, end = ranges[key]
                result = self.paired_direction(directory, observable + "-" + key, model, begin, end, weights)
                assert abs(result["rd_m0"]["analytic"]) > 1e-10, (observable, key, "vacuous isolated derivative")
                isolated[observable][key] = result
        counterfactuals = record["nonzero_feature_sensitivity"] = {}
        for key in ("tensor_message_coeffs", "joint_readout_coeffs", "adaptive_radial_weights"):
            text = model.read_text()
            path = directory / ("zero-" + key + ".mtp")
            path.write_text(put_numbers(text, key, np.zeros_like(numbers(text, key))))
            alternate = self.calc(directory, "zero-" + key, path, backend="rd-m0")
            effect = float(np.max(np.abs(alternate[0] - baseline[0])))
            assert effect > 1e-8, (key, "vacuous nonzero fixture", effect)
            counterfactuals[key] = effect
        # Cover the existing reference EFS path and the optional full radial
        # derivative cache without disabling the mandatory adaptive edge cache.
        record["cache_variants"] = {}
        for name, environment in (("reference-efs", {"SUS2_SH_ADAPTIVE_EFS_REFERENCE": "1"}),
                                  ("full-radial-dr", {"SUS2_SH_ADAPTIVE_RADIAL_DR_FULL_CACHE": "1"})):
            comparison, variant = self.parity(directory, name, model, extra_env=environment)
            comparison["variant_vs_default_ry"] = compare(baseline, variant)
            record["cache_variants"][name] = comparison
        geometry = read_fixture(self.cfg)
        changed = {**geometry, "positions": geometry["positions"].copy()}
        changed["positions"][1] += np.array([.013, -.007, .011])
        perturbed = directory / "perturbed.input.cfg"
        write_geometry(perturbed, changed)
        sequence = directory / "ABA.input.cfg"
        sequence.write_text(self.cfg.read_text() + perturbed.read_text() + self.cfg.read_text())
        record["geometry_cache_ABA"], values = self.parity(directory, "ABA", model, sequence)
        compare([values[0]], [values[2]], exact=True)
        assert np.max(np.abs(values[0] - values[1])) > 1e-8

    def initialization_training(self, topology):
        directory = self.out / "initialization-training"
        directory.mkdir()
        results = {}
        for backend in ("ry", "rd-m0"):
            work = directory / backend
            work.mkdir()
            initialized = work / "initialized.mtp"
            trained = work / "trained.mtp"
            shared = ["--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
                      "--std-weight=0", "--stdd-weight=0", "--optimizer=bfgs",
                      "--bfgs-line-search=wolfe", "--optimizer-conv-tol=0"]
            output, _ = self.run(work, "initialize", "train", topology, self.cfg,
                "--max-iter=0", "--init-params=same", f"--init-random-seed={SEED}", *shared,
                f"--trained-pot-name={initialized}", f"--curr-pot-name={work/'init-current.mtp'}",
                f"--best-pot-name={work/'init-best.mtp'}", backend=backend, calculation=True)
            assert "[adaptive-pytorch-profile] fresh=1" in output
            trace = work / "trace.csv"
            output, _ = self.run(work, "train-one-point", "train", initialized, self.cfg,
                "--max-iter=1", *shared, f"--trained-pot-name={trained}",
                f"--curr-pot-name={work/'current.mtp'}", f"--best-pot-name={work/'best.mtp'}",
                f"--bfgs-trace-file={trace}", backend=backend, calculation=True)
            assert "[adaptive-pytorch-profile] fresh=0" in output
            with trace.open(newline="") as stream:
                rows = list(csv.reader(stream))
            assert rows[0] == ["step", "total_loss", "efs_loss", "energy_mae_mev_atom", "force_mae_mev_a", "stress_mae_ev"]
            assert len(rows) >= 2 and all(len(row) == 6 and all(math.isfinite(float(v)) for v in row) for row in rows[1:])
            results[backend] = {"initialized": str(initialized), "trained": str(trained),
                                "accepted_points": len(rows) - 1, "trace": rows,
                                "max_iter_1_is_smoke_not_learning_claim": True}
        for state in ("initialized", "trained"):
            models = [Path(results[backend][state]) for backend in ("ry", "rd-m0")]
            metadata = []
            for model in models:
                text = model.read_text()
                metadata.append({key: assignment(text, key) for key in (
                    "version", "readout_mode", "alpha_moments_count", "alpha_scalar_moments",
                    "adaptive_radial_amplitude_count", "adaptive_radial_weight_count")})
            assert metadata[0] == metadata[1]
            scale_keys = re.findall(r"(?m)^\s*([A-Za-z_0-9]*optimizer_scales)\s*=", models[0].read_text())
            assert scale_keys, "missing optimizer normalization metadata"
            scale_comparisons = {}
            for key in scale_keys:
                left, right = [numbers(path.read_text(), key) for path in models]
                np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-10, err_msg=key)
                scale_comparisons[key] = float(np.max(np.abs(left - right))) if left.size else 0.0
            a = self.calc(directory, state + "-ry", models[0], backend="ry")
            b = self.calc(directory, state + "-rd", models[1], backend="rd-m0")
            results[state + "_comparison"] = {"EFS": compare(a, b), "metadata": metadata[0],
                                                "optimizer_scale_max_abs": scale_comparisons}
        self.report["extra"]["initialization_training"] = {"status": "PASS", **results}

    def boundary(self, model):
        directory = self.out / "boundaries"
        directory.mkdir()
        output, code = self.run(directory, "invalid-selector", "calc-efs", model, self.cfg,
            directory / "invalid.prediction.cfg", backend="not-a-backend", allow_failure=True)
        assert code != 0 and "SUS2_SH_ANGULAR_BACKEND must be ry or rd-m0" in output
        spec = {"lmax": 6}
        top = directory / "L6.topology.mtp"
        self.run(directory, "L6-init", "init-sh", top, *topology_options(spec))
        for backend in ("ry", "rd-m0"):
            output, code = self.run(directory, "L6-load-" + backend, "roundtrip-model-dev", top,
                directory / ("L6-" + backend + ".mtp"), backend=backend, allow_failure=True)
            assert code != 0 and "supports sh_l_max in [0,4]" in output
        self.report["extra"]["runtime_L6"] = {"status": "EXPECTED_UNSUPPORTED",
            "numerical_pass": False, "reason": "native model loader retains L<=4; angular unit test is separate"}
        self.report["extra"]["invalid_selector"] = {"status": "PASS", "fail_closed": True}

    def actual_models(self):
        entries = json.loads(os.environ.get("MODEL_PATHS", "[]"))
        assert isinstance(entries, list), "MODEL_PATHS must be a JSON list"
        output = self.report["extra"]["actual_models"] = {"status": "NOT_REQUESTED" if not entries else "RUNNING", "cases": []}
        for index, entry in enumerate(entries):
            assert isinstance(entry, dict) and set(entry) == {"model", "cfg"}, entry
            source_model, source_cfg = Path(entry["model"]).resolve(strict=True), Path(entry["cfg"]).resolve(strict=True)
            directory = self.out / f"actual-{index:02d}"
            directory.mkdir()
            model, cfg = directory / "frozen.mtp", directory / "first-two.input.cfg"
            model.write_bytes(source_model.read_bytes())
            blocks, current = [], []
            with source_cfg.open() as stream:
                for line in stream:
                    if line.strip() == "BEGIN_CFG":
                        assert not current, "nested CFG"
                        current = [line]
                    elif current:
                        current.append(line)
                        if line.strip() == "END_CFG":
                            blocks.append("".join(current))
                            current = []
                            if len(blocks) == 2:
                                break
            assert len(blocks) == 2, "actual comparison requires two complete configurations"
            cfg.write_text("".join(blocks))
            identity = sha(source_model)
            assert sha(model) == identity
            comparison, _ = self.parity(directory, "actual", model, cfg)
            assert sha(source_model) == identity
            output["cases"].append({"source_model": str(source_model), "source_cfg": str(source_cfg),
                "model_sha256": identity, "selected_cfg_sha256": sha(cfg), "selected_indices": [0, 1],
                "status": "PASS", "EFS": comparison})
        if entries:
            output["status"] = "PASS"

    def checked(self, label, action):
        try:
            result = action()
            self.flush()
            return result
        except Exception as error:
            failure = {"label": label, "error": str(error), "traceback": traceback.format_exc()}
            self.report["failures"].append(failure)
            if label in self.report["cases"]:
                self.report["cases"][label]["status"] = "FAIL"
            else:
                self.report["extra"].setdefault(label, {})["status"] = "FAIL"
            self.flush()
            print(f"FAIL {label}: {error}", flush=True)
            return None

    def check(self):
        specifications = [(f"L{lmax}-full-shared", {"lmax": lmax}) for lmax in range(5)]
        specifications.append(("L1-tensor-joint", {"lmax": 1, "tensor": 1, "joint": True}))
        for index, (gate, outer) in enumerate((("same-k", "same-k"), ("same-k", "q-total"),
                                             ("q-total", "same-k"), ("q-total", "q-total"))):
            specifications.append((f"L2-extended-{index}", {"lmax": 2, "body": 6, "gate": gate,
                "outer": outer, "projection": "shared-n", "amplitude": "species", "readout": "species",
                "scaling": ("K", "L", "LK", "K")[index], "tensor": 2, "joint": True, "deep": index == 0}))
        built = {}
        for name, spec in specifications:
            result = self.checked(name, lambda name=name, spec=spec: self.case(name, spec))
            if result:
                built[name] = result
        # Independent stages are not skipped simply because another fixture
        # failed; however a missing prerequisite remains an explicit failure.
        if "L2-extended-0" in built:
            self.checked("initialization_training", lambda: self.initialization_training(built["L2-extended-0"][0]))
        else:
            self.report["failures"].append({"label": "initialization_training", "error": "prerequisite fixture failed"})
        if "L0-full-shared" in built:
            self.checked("boundaries", lambda: self.boundary(built["L0-full-shared"][1]))
        else:
            self.report["failures"].append({"label": "boundaries", "error": "prerequisite fixture failed"})
        self.checked("actual_models", self.actual_models)
        if sha(self.binary) != self.binary_sha:
            self.report["failures"].append({"label": "identity", "error": "binary changed during qualification"})
        if self.reference and sha(self.reference) != self.reference_sha:
            self.report["failures"].append({"label": "identity", "error": "reference binary changed during qualification"})
        self.report["positive_case_count"] = len(specifications)
        self.report["positive_cases_passed"] = sum(case["status"] == "PASS" for case in self.report["cases"].values())
        self.report["status"] = "FAIL" if self.report["failures"] else "PASS"
        self.flush()
        print(json.dumps({"status": self.report["status"], "report": str(self.out / 'report.json'),
                          "positive_cases_passed": self.report["positive_cases_passed"],
                          "positive_case_count": len(specifications), "L6": "unsupported, not numerical PASS"}), flush=True)
        return self.report["status"] == "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--reference-binary", type=Path, help="Optional frozen pre-RD native binary; same-model EFS comparison")
    parser.add_argument("--outdir", type=Path, help="Must not exist; default is a new directory in the current working directory")
    args = parser.parse_args()
    require_server()
    if args.outdir is None:
        # Reserve a unique name without subsequently deleting/reusing it.
        parent = Path(tempfile.mkdtemp(prefix="rd-m0-", dir=Path.cwd()))
        args.outdir = parent / "verification"
    check = Check(args.binary, args.outdir, args.reference_binary)
    raise SystemExit(0 if check.check() else 1)


if __name__ == "__main__":
    main()
