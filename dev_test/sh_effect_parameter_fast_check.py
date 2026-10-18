#!/usr/bin/env python3
"""LSF-only exact effect-parameter parity, not a training/performance protocol.

The existing driver uses vibrations E/F/virial weights 1/.1/.01, no priors,
one MPI rank and one CPU. Compare loss AND every physical gradient component.
Source model/data are immutable; every generated artifact stays under --output.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from adaptive_species_readout_check import LOSS_ARGS, NUMBER, numbers, put_numbers, write_fixture
from rd_m0_benchmark import read_dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_text(text):
    """Populate small models without altering topology/counts or scale signs."""
    for key, bias, amplitude in (
        ("scal_coeffs", 1.7, .13), ("two_layer_gate_scaling_coeffs", 2.1, .17),
        ("two_layer_gate_radial_coeffs", .17, .06),
        ("two_layer_gate_type_coeffs", 1.1, .13),
        ("two_layer_gate_additive_coeffs", .19, .04),
        ("two_layer_gate_weights", .07, .05),
        ("two_layer_gate_k_rank_weights", .31, .13),
        ("outer_k_rank1_weights", .29, .11),
        ("moment_coeffs", .09, .07), ("species_coeffs", -.11, .19),
        ("adaptive_radial_weights", .05, .07),
        ("adaptive_radial_amplitudes", .13, .05),
        ("second_layer_gate_weights", .06, .04),
        ("second_layer_gate_type_coeffs", 1.05, .09),
        ("second_layer_gate_additive_coeffs", .09, .04),
        ("second_layer_gate_body_mix_weights", .11, .04),
        ("two_layer_gate_edge_l1_weights", .02, .015),
        ("two_layer_gate_body_mix_weights", .35, .07),
    ):
        if re.search(rf"(?m)^\s*{key}\s*=", text):
            count = len(numbers(text, key))
            text = put_numbers(text, key, bias + amplitude * np.sin(.43 * np.arange(count) + .3))
    # Radial rows have N radial coefficients followed by S endpoint factors.
    block = re.search(r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)", text)
    if not block:
        raise AssertionError("missing materialized radial coefficients")
    row = 0

    def radial(match):
        nonlocal row
        row += 1
        size = len(re.findall(NUMBER, match[0]))
        vals = [.16 + .013 * row + .025 * np.sin(.61 * (row + j)) for j in range(size)]
        return "{" + ", ".join(f"{x:.17e}" for x in vals) + "}"

    return text[:block.start()] + block[1] + re.sub(r"\{[^}]*\}", radial, block[2]) + text[block.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "driver", "frozen-model", "frozen-cfg", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--check-default", action="store_true",
                        help="also compare the automatic path with the switch unset")
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not os.environ.get("LSB_JOBID"):
        parser.error("requires a Linux LSF compute allocation")
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))) != 1:
        parser.error("run the Python harness once, not under a multi-rank mpiexec")
    for key in ("binary", "driver", "frozen_model", "frozen_cfg"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cpu = min(os.sched_getaffinity(0))
    source_hashes = {str(p): sha(p) for p in (args.frozen_model, args.frozen_cfg)}
    report = dict(protocol="full loss+physical gradient parity; vibrations weights 1/.1/.01; no regularizers; not template training",
                  atol=2e-9, rtol=2e-9, cpu=cpu, job=os.environ["LSB_JOBID"],
                  binary_sha256=sha(args.binary), driver_sha256=sha(args.driver),
                  sources=source_hashes, cases=[], commands=[])
    report["unsupported"] = {"center_gate": "Current native interface accepts neighbor gate only; no successful center-path qualification is claimed."}
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUS2_SH_")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_DYNAMIC="FALSE", SUS2_SH_ANGULAR_BACKEND="ry")

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, updates=None, expected_error=None):
        process_env = {**env, **(updates or {})}
        record = dict(label=label, command=list(map(str, command)), flags={k: v for k, v in process_env.items()
                      if k.startswith("SUS2_SH_") or k.endswith("NUM_THREADS") or k == "MKL_DYNAMIC"})
        report["commands"].append(record)
        result = subprocess.run(list(map(str, command)), cwd=out, env=process_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / (label + ".log")).write_text(result.stdout)
        record["returncode"] = result.returncode
        save()
        if expected_error is not None:
            if result.returncode == 0 or expected_error not in result.stdout:
                raise AssertionError(f"{label}: expected rejection {expected_error!r}")
            return result.stdout
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")
        return result.stdout

    def compare(label, model, cfg, maximum, cache, expected=None):
        snapshots = []
        modes = (0, 1, None) if args.check_default else (0, 1)
        for fast in modes:
            dump = out / f"{label}-cache{cache}-fast{fast if fast is not None else 'auto'}.jsonl"
            updates = {"SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE": str(1 - cache)}
            if fast is not None:
                updates["SUS2_SH_EFFECT_PARAM_FAST"] = str(fast)
            run(dump.stem, ["taskset", "-c", cpu, args.driver, model, cfg, "gradient", maximum, 1, 1, dump],
                updates)
            metadata, values = read_dump(dump)
            if expected is not None and metadata["coeff_count"] != expected:
                raise AssertionError(f"{label}: expected {expected} coefficients, got {metadata}")
            if not np.isfinite(values).all() or len(values) != metadata["coeff_count"] + 1:
                raise AssertionError(f"{label}: incomplete/nonfinite physical gradient")
            if not np.any(values[1:] != 0):
                raise AssertionError(f"{label}: vacuous all-zero gradient")
            snapshots.append((metadata, values))
        left = snapshots[0][1]
        for metadata, right in snapshots[1:]:
            if snapshots[0][0] != metadata:
                raise AssertionError(f"{label}: dump metadata differ")
            np.testing.assert_allclose(right, left, atol=2e-9, rtol=2e-9, err_msg=label)
        delta = np.max(np.stack([np.abs(left - row[1]) for row in snapshots[1:]]), axis=0)
        magnitude = np.max(np.stack([np.abs(row[1]) for row in snapshots]), axis=0)
        record = dict(label=label, cache=cache, model_sha256=sha(model), cfg_sha256=sha(cfg),
                      switch_modes=["auto" if mode is None else mode for mode in modes],
                      metadata=snapshots[0][0], finite=True, loss_abs_error=float(delta[0]),
                      gradient_max_abs_error=float(delta[1:].max()),
                      gradient_max_relative_error=float(np.max(delta[1:] / np.maximum(1e-300, magnitude[1:]))),
                      max_scaled_error=float(np.max(delta / (2e-9 + 2e-9 * np.abs(left)))),
                      all_physical_components_checked=len(left) - 1)
        report["cases"].append(record)
        save()
        print(label, "cache", cache, "PASS", record["gradient_max_abs_error"], flush=True)

    try:
        frozen = out / "frozen.sus2"
        data = out / "frozen.cfg"
        shutil.copyfile(args.frozen_model, frozen)
        shutil.copyfile(args.frozen_cfg, data)
        # Qualification starts with the actual 1428-coefficient production seed.
        compare("frozen-first-two", frozen, data, 2, 1, expected=1428)
        compare("frozen-first-two", frozen, data, 2, 0, expected=1428)
        cfg = out / "asymmetric.cfg"
        write_fixture(cfg)
        base = ["--species-count=2", "--l-max=1", "--k-max=2", "--body-order=3",
                "--body-l-max=1,1,1,1", "--radial-basis-size=3", "--cutoff=4.8",
                "--write-sh-scalar-info"]
        cases = []
        # Four graph combinations; rotate sharing modes and element heads.
        for index, (inner, outer) in enumerate(itertools.product(("same-k", "q-total"), repeat=2)):
            cases.append((f"ranked-{inner}-{outer}", ["--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
                "--two-layer-gate-shared-radial", "--two-layer-gate-k-rank=1", "--outer-k-rank=2",
                f"--two-layer-gate-factor-pruning={inner}", f"--sh-factor-pruning={outer}",
                f"--scaling-map={('K', 'L', 'LK', 'pair')[index]}",
                f"--readout-mode={('shared', 'species')[index % 2]}"]))
        cases += [
            ("decoupled", ["--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full", "--two-layer-gate-shared-radial",
                "--two-layer-gate-decoupled-scaling", "--two-layer-gate-k-rank=1", "--outer-k-rank=2", "--scaling-map=LK"]),
            ("unranked", ["--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full", "--two-layer-gate-shared-radial"]),
            ("l2-body-combo", ["--l-max=2", "--body-l-max=2,2,2,2", "--two-layer-gate", "--two-layer-gate-mode=mu-body-linear-combo", "--two-layer-gate-shared-radial"]),
            ("l4-ranked", ["--l-max=4", "--body-l-max=4,4,4,4", "--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full", "--two-layer-gate-shared-radial", "--two-layer-gate-k-rank=1", "--outer-k-rank=2"]),
            ("edge-l1", ["--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full", "--two-layer-gate-shared-radial", "--two-layer-gate-edge-l1"]),
            ("static-fallback", ["--outer-k-rank=2"]),
            ("adaptive-fallback", ["--superlinear-mode=radial-coefficient-gate", "--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
                "--two-layer-gate-shared-radial", "--two-layer-gate-k-rank=1", "--outer-k-rank=2"]),
            ("multilayer-fallback", ["--two-layer-gate", "--multi-layer-gate", "--second-layer-gate-body-order=3"]),
        ]
        fd_model = None
        for label, options in cases:
            topology, model = out / (label + "-topology.sus2"), out / (label + ".sus2")
            overridden = {option.split("=", 1)[0] for option in options}
            case_base = [option for option in base if option.split("=", 1)[0] not in overridden]
            run(label + "-init", [args.binary, "init-sh", topology, *case_base, *options])
            run(label + "-materialize", [args.binary, "roundtrip-model-dev", topology, model])
            model.write_text(active_text(model.read_text()))
            for cache in (1, 0):
                if label == "adaptive-fallback" and cache == 0:
                    modes = (0, 1, None) if args.check_default else (0, 1)
                    for mode in modes:
                        updates = {"SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE": "1"}
                        if mode is not None:
                            updates["SUS2_SH_EFFECT_PARAM_FAST"] = str(mode)
                        run(f"adaptive-cache-disabled-rejected-{mode}",
                            ["taskset", "-c", cpu, args.driver, model, cfg, "gradient", 1, 1, 1, "-"],
                            updates, expected_error="adaptive radial coefficients require the exact edge primitive cache")
                    report["unsupported"]["adaptive_without_edge_cache"] = {
                        "reason": "Existing adaptive contract requires its edge primitive cache.",
                        "rejected_switch_modes": ["auto" if mode is None else mode for mode in modes]}
                    save()
                else:
                    compare(label, model, cfg, 1, cache)
            if fd_model is None:
                fd_model = model
        fd_log = run("full-parameter-finite-difference", [args.binary, "check-loss-gradient-dev", fd_model, cfg,
            *LOSS_ARGS, "--coeff-start=0", "--coeff-end=0"], {"SUS2_SH_EFFECT_PARAM_FAST": "1"})
        fields = dict(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", fd_log))
        if int(fields.get("checked_coeffs", "0")) != int(fields.get("coeff_count", "-1")):
            raise AssertionError("finite differences did not check all stored physical coefficients")
        report["finite_difference"] = fields
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        raise
    finally:
        report["sources_unchanged"] = all(sha(path) == digest for path, digest in source_hashes.items())
        save()
        if not report["sources_unchanged"]:
            raise AssertionError("frozen source model/data changed")


if __name__ == "__main__":
    main()
