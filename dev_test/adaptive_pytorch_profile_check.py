#!/usr/bin/env python3
"""Server-only regression for C++ adaptive-radial PyTorch-aligned defaults.

This test is intentionally a standalone dev_test harness.  It creates a tiny
adaptive-radial topology, exercises the C++ train/calc/dev-gradient commands,
and compares the physical complete checkpoint against the PyTorch loss and
regularizer path when SUS2SHPT_ROOT points at the matching PyTorch checkout.

No production modules are imported from this C++ repository.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import re
import argparse
import shlex
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(
    description="server-only adaptive radial PyTorch-profile regression"
)
PARSER.add_argument("binary", nargs="?", default=str(ROOT / "bin/mlp-sus2"))
PARSER.add_argument(
    "--mpi-command",
    default="",
    help="optional MPI command prefix, e.g. 'mpiexec.hydra -n 2'",
)
ARGS = PARSER.parse_args()
BIN = Path(ARGS.binary).resolve()
MPI_COMMAND = shlex.split(ARGS.mpi_command) if ARGS.mpi_command else []
TMP_PARENT = ROOT / ".local_tmp"
TMP_PARENT.mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="adaptive-pytorch-profile-", dir=TMP_PARENT))
CFG = TMP / "tiny_efs.cfg"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
ENV = dict(
    os.environ,
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_DYNAMIC="FALSE",
)
PROFILE_TRAIN_ARGS = (
    "--energy-weight=1",
    "--force-weight=0.1",
    "--stress-weight=0.01",
    "--std-weight=0",
    "--stdd-weight=0",
)
PROFILE_PHYSICAL_REG_ARGS = (
    "--radial-smooth=1e-6",
    "--radial-smooth-grid=128",
    "--gate-x-l2=0",
    "--scalar-head-l2=1e-5",
    "--gate-scalar-l2=0",
    "--gate-mix-l2=0",
    "--gate-full-l2=1e-4",
)


def run(
    label: str,
    *args: object,
    expect_success: bool = True,
    env: dict[str, str] | None = None,
    mpi: bool = False,
) -> str:
    command = ([*MPI_COMMAND, str(BIN)] if mpi else [str(BIN)]) + [*map(str, args)]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(ENV, **(env or {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (TMP / f"{label}.log").write_text(result.stdout, encoding="utf-8")
    if expect_success and result.returncode:
        raise AssertionError(f"{label}: exit {result.returncode}\n{result.stdout[-8000:]}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"{label}: expected failure\n{result.stdout[-4000:]}")
    return result.stdout


def write_cfg() -> None:
    CFG.write_text(
        """BEGIN_CFG
 Size
       3
 Supercell
   7.400000000000    0.210000000000    0.090000000000
   0.120000000000    7.800000000000    0.180000000000
   0.050000000000    0.160000000000    8.100000000000
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1.170000000000 1.390000000000 1.520000000000  0.031000000000 -0.023000000000  0.017000000000
 2 1 2.630000000000 2.110000000000 2.740000000000 -0.019000000000  0.041000000000 -0.029000000000
 3 0 4.220000000000 3.050000000000 1.880000000000  0.013000000000 -0.011000000000  0.037000000000
 Energy
   -4.812345678900
 PlusStress: xx yy zz yz xz xy
   0.041000000000 -0.027000000000 0.018000000000 0.006000000000 -0.004000000000 0.003000000000
END_CFG
""",
        encoding="utf-8",
    )


def numbers(text: str, field: str) -> list[float]:
    match = re.search(rf"(?ms)^\s*{re.escape(field)}\s*=\s*\{{(.*?)\}}", text)
    if match is None:
        raise AssertionError(f"missing field {field}")
    return [float(x) for x in re.findall(NUM, match.group(1))]


def optional_numbers(text: str, field: str) -> list[float] | None:
    if not re.search(rf"(?m)^\s*{re.escape(field)}\s*=", text):
        return None
    return numbers(text, field)


def replace_numbers(text: str, field: str, values: Iterable[float]) -> str:
    replacement = f"{field} = {{{', '.join(f'{float(v):.17e}' for v in values)}}}"
    out, count = re.subn(
        rf"(?ms)^\s*{re.escape(field)}\s*=\s*\{{.*?\}}",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise AssertionError((field, count))
    return out


def scalar_value(text: str, field: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(field)}\s*=\s*({NUM})\s*$", text)
    if match is None:
        raise AssertionError(f"missing scalar field {field}")
    return float(match.group(1))


def radial_rows(text: str) -> list[list[float]]:
    match = re.search(
        r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*(?:alpha_moments_count|alpha_index_basic_count)\s*=)",
        text,
    )
    if match is None:
        raise AssertionError("missing radial_coeffs")
    return [[float(x) for x in re.findall(NUM, row)] for row in re.findall(r"\{([^}]*)\}", match.group(2))]


def replace_radial_rows(text: str, rows: list[list[float]]) -> str:
    match = re.search(
        r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*(?:alpha_moments_count|alpha_index_basic_count)\s*=)",
        text,
    )
    if match is None:
        raise AssertionError("missing radial_coeffs")
    old_rows = re.findall(r"\{([^}]*)\}", match.group(2))
    if len(old_rows) != len(rows):
        raise AssertionError(("radial row count", len(old_rows), len(rows)))
    rendered = iter("{" + ", ".join(f"{x:.17e}" for x in row) + "}" for row in rows)
    block = re.sub(r"\{[^}]*\}", lambda _: next(rendered), match.group(2))
    return text[: match.start()] + match.group(1) + block + text[match.end() :]


def wave(count: int, amplitude: float, bias: float, phase: float) -> np.ndarray:
    return np.asarray([bias + amplitude * math.cos(phase * (i + 1)) for i in range(count)], dtype=float)


def field(text: str, key: str, default: str | None = None) -> str:
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", text)
    if matches:
        return matches[-1]
    if default is not None:
        return default
    raise AssertionError(f"missing {key} in {text[-1000:]}")


def last_dev_line(output: str, marker: str = "checked_coeffs=") -> str:
    for line in reversed(output.strip().splitlines()):
        if marker in line:
            return line
    raise AssertionError(f"missing dev output marker {marker!r} in {output[-2000:]}")


def dev_direction(coeff_count: int) -> np.ndarray:
    pattern = np.empty(coeff_count, dtype=float)
    for local_index in range(coeff_count):
        phase = local_index % 5
        value = float(phase - 2)
        pattern[local_index] = 1.0 if value == 0.0 else value
    return pattern


def parse_prediction(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    energy = float(re.search(r"(?m)^\s*Energy\s*\n\s*(" + NUM + ")", text).group(1))
    atom_block = re.search(r"(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy", text).group(1)
    forces = [float(x) for row in atom_block.strip().splitlines() for x in row.split()[-3:]]
    stress = [float(x) for x in re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", text).group(1).split()]
    values = np.asarray([energy, *forces, *stress], dtype=float)
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"nonfinite prediction in {path}")
    return values


def calc_efs(label: str, model: Path) -> np.ndarray:
    out = TMP / f"{label}.cfg"
    run(label, "calc-efs", model, CFG, out)
    return parse_prediction(out)


def calc_efs_mpi(label: str, model: Path) -> np.ndarray:
    if not MPI_COMMAND:
        raise AssertionError("calc_efs_mpi called without --mpi-command")
    out = TMP / f"{label}.cfg"
    run(label, "calc-efs", model, CFG, out, mpi=True)
    return parse_prediction(out)


def model_signature(path: Path) -> dict[str, np.ndarray | float]:
    text = path.read_text(encoding="utf-8")
    fields = (
        "adaptive_radial_weights",
        "adaptive_radial_amplitudes",
        "moment_coeffs",
        "species_coeffs",
        "scal_coeffs",
    )
    signature: dict[str, np.ndarray | float] = {
        "scaling": scalar_value(text, "scaling"),
        "radial_coeffs": np.asarray([x for row in radial_rows(text) for x in row], dtype=float),
    }
    for name in fields:
        signature[name] = np.asarray(numbers(text, name), dtype=float)
    return signature


def compare_signatures(label: str, left: Path, right: Path, tolerance: float = 2.0e-12) -> None:
    a = model_signature(left)
    b = model_signature(right)
    if set(a) != set(b):
        raise AssertionError(f"{label}: signature keys differ")
    worst = 0.0
    worst_key = ""
    for key in a:
        av = np.asarray(a[key], dtype=float)
        bv = np.asarray(b[key], dtype=float)
        if av.shape != bv.shape:
            raise AssertionError(f"{label}: shape differs for {key}: {av.shape} vs {bv.shape}")
        err = float(np.max(np.abs(av - bv))) if av.size else 0.0
        if err > worst:
            worst = err
            worst_key = key
    if worst > tolerance:
        raise AssertionError(f"{label}: model signature mismatch {worst_key} {worst:.3e}")


def make_topology() -> Path:
    topology = TMP / "fresh_l1k2_o2_topology.sus2"
    run(
        "init-topology",
        "init-sh",
        topology,
        "--species-count=2",
        "--l-max=1",
        "--k-max=2",
        "--body-order=3",
        "--body-l-max=1,1,1,1",
        "--radial-basis-size=3",
        "--cutoff=4.8",
        "--radial-basis-type=RBChebyshev_sss",
        "--scaling-map=K",
        "--superlinear-mode=radial-coefficient-gate",
        "--outer-k-rank=2",
        "--two-layer-gate",
        "--two-layer-gate-mode=mu-scalar-full",
        "--two-layer-gate-body-order=3",
        "--two-layer-gate-factor-pruning=q-total",
        "--two-layer-gate-k-rank=1",
        "--two-layer-gate-shared-radial",
        "--write-sh-scalar-info",
    )
    return topology


def materialized_asymmetric_model(topology: Path) -> Path:
    model = TMP / "asymmetric_complete_seed.sus2"
    run("roundtrip-topology", "roundtrip-model-dev", topology, model)
    text = model.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^scaling = .*$", "scaling = 1.0", text)
    rows = radial_rows(text)
    rows = [
        [0.071 * (r + 1) + 0.023 * (c + 1) + 0.037 * math.sin(0.43 * (r + c + 1)) for c, _ in enumerate(row)]
        for r, row in enumerate(rows)
    ]
    text = replace_radial_rows(text, rows)
    for name, amp, bias, phase in (
        ("adaptive_radial_weights", 0.19, -0.013, 0.61),
        ("adaptive_radial_amplitudes", 0.11, 0.23, 0.47),
        ("two_layer_gate_radial_coeffs", 0.09, 0.05, 0.67),
        ("two_layer_gate_type_coeffs", 0.08, 1.07, 0.53),
        ("two_layer_gate_k_rank_weights", 0.17, 0.04, 0.71),
        ("moment_coeffs", 0.013, 0.0, 0.37),
    ):
        text = replace_numbers(text, name, wave(len(numbers(text, name)), amp, bias, phase))
    text = replace_numbers(text, "species_coeffs", [0.21, -0.14])
    text = replace_numbers(text, "scal_coeffs", [1.8 + 0.03 * i for i in range(len(numbers(text, "scal_coeffs")))])
    model.write_text(text, encoding="utf-8")
    return model


def train_zero_iter(topology: Path) -> Path:
    complete = TMP / "fresh_auto_zeroiter_complete.sus2"
    log = run(
        "fresh-auto-zeroiter",
        "train",
        topology,
        CFG,
        f"--trained-pot-name={complete}",
        "--max-iter=0",
        "--init-params=same",
        *PROFILE_TRAIN_ARGS,
    )
    if not complete.exists():
        raise AssertionError("--max-iter=0 did not write the complete checkpoint")
    required_hints = ("adaptive", "radial")
    if not all(hint in log.lower() for hint in required_hints):
        raise AssertionError("fresh auto training log did not identify adaptive radial handling")
    before = calc_efs("zeroiter-prediction-before", complete)
    resumed = TMP / "fresh_auto_zeroiter_resumed.sus2"
    run(
        "complete-zeroiter-resume",
        "train",
        complete,
        CFG,
        f"--trained-pot-name={resumed}",
        "--max-iter=0",
        "--init-params=same",
        *PROFILE_TRAIN_ARGS,
    )
    after = calc_efs("zeroiter-prediction-after", resumed)
    err = float(np.max(np.abs(before - after)))
    if err > 2.0e-10:
        raise AssertionError(f"complete checkpoint zero-iteration resume changed EFS: {err}")
    print(f"PASS complete-zeroiter-roundtrip max_abs={err:.3e}", flush=True)
    return complete


def mpi_zero_iter_checks(topology: Path, serial_complete: Path) -> None:
    if not MPI_COMMAND:
        return
    mpi_fresh = TMP / "fresh_auto_zeroiter_mpi2.sus2"
    run(
        "mpi-fresh-auto-zeroiter",
        "train",
        topology,
        CFG,
        f"--trained-pot-name={mpi_fresh}",
        "--max-iter=0",
        "--init-params=same",
        *PROFILE_TRAIN_ARGS,
        mpi=True,
    )
    if not mpi_fresh.exists():
        raise AssertionError("MPI fresh zero-step did not write complete checkpoint")
    serial_efs = calc_efs("mpi-serial-complete-serial-efs", serial_complete)
    mpi_fresh_serial_efs = calc_efs("mpi-fresh-complete-serial-efs", mpi_fresh)
    mpi_fresh_mpi_efs = calc_efs_mpi("mpi-fresh-complete-mpi-efs", mpi_fresh)
    fresh_serial_err = float(np.max(np.abs(serial_efs - mpi_fresh_serial_efs)))
    fresh_mpi_err = float(np.max(np.abs(mpi_fresh_serial_efs - mpi_fresh_mpi_efs)))
    if fresh_serial_err > 2.0e-10 or fresh_mpi_err > 2.0e-10:
        raise AssertionError(
            f"MPI fresh zero-step EFS mismatch: serial={fresh_serial_err:.3e} mpi={fresh_mpi_err:.3e}"
        )
    compare_signatures("MPI fresh zero-step scales/physical params", serial_complete, mpi_fresh)

    mpi_resumed = TMP / "complete_zeroiter_resumed_mpi2.sus2"
    run(
        "mpi-complete-zeroiter-resume",
        "train",
        serial_complete,
        CFG,
        f"--trained-pot-name={mpi_resumed}",
        "--max-iter=0",
        "--init-params=same",
        *PROFILE_TRAIN_ARGS,
        mpi=True,
    )
    resumed_serial_efs = calc_efs("mpi-resumed-complete-serial-efs", mpi_resumed)
    resumed_mpi_efs = calc_efs_mpi("mpi-resumed-complete-mpi-efs", mpi_resumed)
    resume_serial_err = float(np.max(np.abs(serial_efs - resumed_serial_efs)))
    resume_mpi_err = float(np.max(np.abs(resumed_serial_efs - resumed_mpi_efs)))
    if resume_serial_err > 2.0e-10 or resume_mpi_err > 2.0e-10:
        raise AssertionError(
            f"MPI complete zero-step EFS mismatch: serial={resume_serial_err:.3e} mpi={resume_mpi_err:.3e}"
        )
    compare_signatures("MPI complete zero-step scales/physical params", serial_complete, mpi_resumed)
    # A second case has work on both ranks. Duplicating the configuration must
    # leave RMS calibration and the mean-loss + once-only regularizers unchanged.
    double_cfg = TMP / "two_identical_efs.cfg"
    double_cfg.write_text(CFG.read_text() * 2, encoding="utf-8")
    one_step_models = []
    for label, mpi in (("serial", False), ("mpi2", True)):
        trained = TMP / f"two_active_{label}_one_step.sus2"
        run(
            f"two-active-{label}-one-step", "train", serial_complete, double_cfg,
            f"--trained-pot-name={trained}", "--max-iter=1", "--init-params=same",
            *PROFILE_TRAIN_ARGS, mpi=mpi,
        )
        one_step_models.append(trained)
    one_step_error = float(np.max(np.abs(
        calc_efs("two-active-serial-efs", one_step_models[0])
        - calc_efs("two-active-mpi-efs", one_step_models[1])
    )))
    if one_step_error > 2.0e-8:
        raise AssertionError(f"MPI two-active-rank one-step mismatch: {one_step_error}")
    print(
        "PASS mpi-zeroiter-fixed-scales "
        + json.dumps(
            {
                "mpi_command": MPI_COMMAND,
                "fresh_serial_max_abs": fresh_serial_err,
                "fresh_mpi_max_abs": fresh_mpi_err,
                "resume_serial_max_abs": resume_serial_err,
                "resume_mpi_max_abs": resume_mpi_err,
                "two_active_one_step_max_abs": one_step_error,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def train_std_stdd_one_step(model: Path) -> None:
    for label, std_weight, stdd_weight in (
        ("std-only", "0.04", "0"),
        ("stdd-only", "0", "0.03"),
    ):
        out_model = TMP / f"{label}_one_step.sus2"
        log = run(
            f"train-{label}",
            "train",
            model,
            CFG,
            f"--trained-pot-name={out_model}",
            "--max-iter=1",
            "--init-params=same",
            "--energy-weight=1",
            "--force-weight=0.1",
            "--stress-weight=0.01",
            f"--std-weight={std_weight}",
            f"--stdd-weight={stdd_weight}",
        )
        if not out_model.exists():
            raise AssertionError(f"{label} one-step training did not write model")
        if not np.all(np.isfinite(calc_efs(f"{label}-one-step-efs", out_model))):
            raise AssertionError(f"{label} one-step model produced nonfinite EFS")
        lowered = log.lower()
        if "std" not in lowered:
            raise AssertionError(f"{label} training log did not report std/stdd terms")
    print("PASS std-stdd-one-step-training count=2", flush=True)


def gradient_checks(model: Path) -> dict[str, object]:
    run("weight-norm-roundtrip", "check-radial-weight-norm-roundtrip-dev", model)
    for block in ("outer", "gate1"):
        run(
            f"weight-norm-{block}-gradient", "check-radial-weight-norm-gradient-dev",
            model, CFG, f"--radial-block={block}", "--mu=0", "--max-configs=1",
            "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
            "--std-weight=0.03", "--stdd-weight=0.02", *PROFILE_PHYSICAL_REG_ARGS,
            "--displacement=2e-5", "--abs-tolerance=5e-6", "--rel-tolerance=5e-4",
        )
    print("PASS signed-weight-norm-roundtrip-and-pullback outer=1 gate1=1", flush=True)
    output = run(
        "loss-gradient-std-stdd",
        "check-loss-gradient-dev",
        model,
        CFG,
        "--max-configs=1",
        "--energy-weight=1",
        "--force-weight=0.1",
        "--stress-weight=0.01",
        "--std-weight=0.03",
        "--stdd-weight=0.02",
        *PROFILE_PHYSICAL_REG_ARGS,
        "--displacement=2e-5",
        "--abs-tolerance=5e-6",
        "--rel-tolerance=5e-4",
    )
    last = last_dev_line(output)
    if int(field(last, "checked_coeffs", "0")) <= 0:
        raise AssertionError(f"gradient check was vacuous: {last}")
    layout = output
    ranges = {}
    for prefix in ("adaptive_radial_amplitude", "adaptive_radial_weight"):
        begin = int(field(layout, prefix + "_begin", "-1"))
        end = int(field(layout, prefix + "_end", "-1"))
        if begin < 0 or end <= begin:
            raise AssertionError(f"missing active {prefix} range in dev output")
        ranges[prefix] = (begin, end)
        direction = run(
            f"{prefix}-direction",
            "check-loss-gradient-direction-dev",
            model,
            CFG,
            "--max-configs=1",
            "--energy-weight=1",
            "--force-weight=0.1",
            "--stress-weight=0.01",
            "--std-weight=0",
            "--stdd-weight=0",
            *PROFILE_PHYSICAL_REG_ARGS,
            "--displacement=2e-5",
            "--abs-tolerance=5e-6",
            "--rel-tolerance=5e-4",
            f"--coeff-start={begin}",
            f"--coeff-end={end}",
        )
        dlast = last_dev_line(direction)
        analytic = float(field(dlast, "analytic_directional", field(dlast, "analytic", "nan")))
        fd = float(field(dlast, "finite_difference"))
        if not (math.isfinite(analytic) and math.isfinite(fd)):
            raise AssertionError(f"nonfinite directional gradient: {dlast}")
        if max(abs(analytic), abs(fd)) <= 1e-10:
            raise AssertionError(f"vacuous directional gradient for {prefix}: {dlast}")
        ranges[prefix + "_directional"] = {"analytic": analytic, "finite_difference": fd}
    print(f"PASS cpp-gradient-std-stdd {last}", flush=True)
    return {"last": last, "ranges": ranges}


def std_value_gradient_checks(model: Path) -> None:
    """Check that std/stdd see the same loss on the gradient and value paths."""
    for label, std_weight, stdd_weight in (
        ("std", "0.05", "0"),
        ("stdd", "0", "0.05"),
    ):
        output = run(
            f"{label}-value-gradient",
            "check-loss-gradient-dev",
            model,
            CFG,
            "--max-configs=1",
            "--energy-weight=0",
            "--force-weight=0",
            "--stress-weight=0",
            f"--std-weight={std_weight}",
            f"--stdd-weight={stdd_weight}",
            "--radial-smooth=0",
            "--radial-smooth-grid=128",
            "--gate-x-l2=0",
            "--scalar-head-l2=0",
            "--gate-scalar-l2=0",
            "--gate-mix-l2=0",
            "--gate-full-l2=0",
            "--displacement=2e-5",
            "--abs-tolerance=1e-5",
            "--rel-tolerance=1e-4",
        )
        line = last_dev_line(output)
        value = float(field(line, "gradient_path_loss"))
        base = float(field(line, "base_loss"))
        if not (math.isfinite(value) and math.isfinite(base)) or abs(value - base) > 1e-10:
            raise AssertionError(f"{label} gradient/value loss mismatch: {line}")
    print("PASS std-stdd-gradient-value-loss count=2", flush=True)


def check_rejections(topology: Path) -> None:
    # These options are controlled by the adaptive PyTorch-alignment profile.
    # std/stdd remain deliberately accepted and are covered by gradient_checks.
    rejected = {
        "--adaptive-radial-initialization=legacy": "adaptive",
        "--adaptive-radial-init-relative-rms=0.2": "adaptive",
        "--bfgs-coordinate-normalization=none": "normalization",
        "--scalar-weight-l2-preset=none": "l2",
        "--scalar-head-l2=0": "l2",
        "--gate-full-l2=0": "l2",
        "--radial-smooth=0": "radial",
        "--radial-smooth-grid=64": "radial",
        "--gate-x-l2=1e-4": "gate",
        "--weighting=structures": "weighting",
        "--scale-by-force=1": "scale",
        "--force-loss=log-cosh": "force",
        "--force-log-cosh-scale=2": "force",
    }
    for option, expected in rejected.items():
        out = run(
            "reject-" + re.sub(r"[^A-Za-z0-9]+", "_", option).strip("_"),
            "train",
            topology,
            CFG,
            f"--trained-pot-name={TMP / ('reject_' + str(abs(hash(option))) + '.sus2')}",
            "--max-iter=0",
            "--init-params=same",
            option,
            expect_success=False,
        )
        lowered = out.lower()
        if expected not in lowered and "deprecated" not in lowered and "fixed" not in lowered and "adaptive" not in lowered:
            raise AssertionError(f"unexpected rejection diagnostic for {option}: {out[-2000:]}")
    print(f"PASS deprecated-profile-options-rejected count={len(rejected)}", flush=True)


def pytorch_profile_parity(model_path: Path, cpp_grad: dict[str, object]) -> None:
    torch_root = os.environ.get("SUS2SHPT_ROOT")
    if not torch_root:
        raise AssertionError("SUS2SHPT_ROOT is required for PyTorch profile parity")
    sys.path.insert(0, torch_root)
    torch = importlib.import_module("torch")
    load_cfg = importlib.import_module("sus2shpt.io.cfg").load_cfg
    mtp_io = importlib.import_module("sus2shpt.io.sus2")
    SUS2SHPyTorch = importlib.import_module("sus2shpt.model.sus2_sh").SUS2SHPyTorch
    training = importlib.import_module("sus2shpt.training")
    required_private_api = (
        "_prepare_training_graphs",
        "_configuration_batch_loss",
        "_global_regularization_loss",
    )
    missing_private_api = [name for name in required_private_api if not hasattr(training, name)]
    if missing_private_api:
        raise AssertionError(
            "PyTorch training private API changed; update this alignment test: "
            + ", ".join(missing_private_api)
        )

    def make_model(path: Path):
        loaded = mtp_io.load_mtp_model(path)
        model = SUS2SHPyTorch(loaded, device="cpu", dtype=torch.float64)
        if not getattr(model, "adaptive_radial", False):
            raise AssertionError("PyTorch did not load adaptive radial model")
        return model

    cfgs = load_cfg(CFG)
    options = training.TrainingOptions(
        energy_weight=1.0,
        force_weight=0.1,
        stress_weight=0.01,
        weighting="vibrations",
        loss_style="legacy",
        radial_smooth=1.0e-6,
        radial_smooth_grid=128,
        scalar_head_l2=1.0e-5,
        gate_scalar_l2=0.0,
        gate_mix_l2=0.0,
        gate_full_l2=1.0e-4,
        gate_x_l2=0.0,
        radial_weight_norm=False,
        adaptive_amplitude_conditioning=False,
        device="cpu",
        dtype=torch.float64,
    )

    def loss_value(path: Path) -> tuple[float, float, float]:
        model = make_model(path)
        graphs = training._prepare_training_graphs(model, cfgs, workers=0)
        pred_loss = training._configuration_batch_loss(model, graphs, options, create_graph=True)
        reg_loss = training._global_regularization_loss(model, options)
        total = pred_loss + reg_loss
        if not torch.isfinite(total):
            raise AssertionError("PyTorch profile loss is nonfinite")
        return float(pred_loss.detach()), float(reg_loss.detach()), float(total.detach())

    pt_pred, pt_reg, pt_total = loss_value(model_path)
    cpp_full = last_dev_line(run(
        "cpp-loss-profile-for-pytorch",
        "check-loss-gradient-dev",
        model_path,
        CFG,
        "--max-configs=1",
        "--energy-weight=1",
        "--force-weight=0.1",
        "--stress-weight=0.01",
        "--std-weight=0",
        "--stdd-weight=0",
        *PROFILE_PHYSICAL_REG_ARGS,
        "--coeff-start=0",
        "--coeff-end=1",
    ))
    cpp_total = float(field(cpp_full, "base_loss"))
    cpp_reg = float(field(last_dev_line(run(
        "cpp-regularizer-profile-for-pytorch",
        "check-loss-gradient-dev",
        model_path,
        CFG,
        "--max-configs=1",
        "--energy-weight=0",
        "--force-weight=0",
        "--stress-weight=0",
        "--std-weight=0",
        "--stdd-weight=0",
        *PROFILE_PHYSICAL_REG_ARGS,
        "--coeff-start=0",
        "--coeff-end=1",
    )), "base_loss"))

    if abs(pt_total - cpp_total) > 2.0e-8:
        raise AssertionError(f"PyTorch/C++ total loss mismatch: pt={pt_total} cpp={cpp_total}")
    if abs(pt_reg - cpp_reg) > 2.0e-8:
        raise AssertionError(f"PyTorch/C++ regularizer mismatch: pt={pt_reg} cpp={cpp_reg}")

    def perturb_file(src: Path, dst: Path, name: str, delta: float, direction: np.ndarray) -> None:
        text = src.read_text(encoding="utf-8")
        values = np.asarray(numbers(text, name), dtype=float)
        if len(values) != len(direction):
            raise AssertionError((name, len(values), len(direction)))
        text = replace_numbers(text, name, values + delta * direction)
        dst.write_text(text, encoding="utf-8")

    h = 2.0e-5
    for tensor_name, range_name in (
        ("adaptive_radial_amplitudes", "adaptive_radial_amplitude_directional"),
        ("adaptive_radial_weights", "adaptive_radial_weight_directional"),
    ):
        plus = TMP / f"pt_fd_{tensor_name}_plus.sus2"
        minus = TMP / f"pt_fd_{tensor_name}_minus.sus2"
        begin, end = cpp_grad["ranges"][range_name.replace("_directional", "")]
        direction = dev_direction(end - begin)
        perturb_file(model_path, plus, tensor_name, h, direction)
        perturb_file(model_path, minus, tensor_name, -h, direction)
        pt_fd = (loss_value(plus)[2] - loss_value(minus)[2]) / (2.0 * h)
        cpp_fd = float(cpp_grad["ranges"][range_name]["finite_difference"])
        if abs(pt_fd - cpp_fd) > 2.0e-6:
            raise AssertionError(f"PyTorch/C++ {tensor_name} directional FD mismatch: pt={pt_fd} cpp={cpp_fd}")
    print(
        "PASS pytorch-profile-loss-regularizer-directional "
        + json.dumps({"pt_total": pt_total, "cpp_total": cpp_total, "pt_reg": pt_reg, "cpp_reg": cpp_reg}, sort_keys=True),
        flush=True,
    )


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"missing mlp-sus2 binary: {BIN}")
    write_cfg()
    topology = make_topology()
    auto_complete = train_zero_iter(topology)
    mpi_zero_iter_checks(topology, auto_complete)
    check_rejections(topology)
    asymmetric = materialized_asymmetric_model(topology)
    train_std_stdd_one_step(asymmetric)
    gradient_info = gradient_checks(asymmetric)
    std_value_gradient_checks(asymmetric)
    # Also check that the auto-complete model is accepted by the physical
    # PyTorch profile; the asymmetric model stresses nonzero a/U directions.
    pytorch_profile_parity(asymmetric, gradient_info)
    calc_efs("auto-complete-final-smoke", auto_complete)
    print(f"ADAPTIVE_PYTORCH_PROFILE_CHECK_PASS evidence={TMP}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
