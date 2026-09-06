#!/usr/bin/env python3
"""Regression checks for --superlinear-mode=radial-coefficient-gate.

The mode adapts the per-(l,o,n) radial coefficient as

    d_jlon = c_lon + a_lon * sum_q U_lonq S_jq

where O is --outer-k-rank, K is --k-max, O <= K, and the model stores the
new tensors as adaptive_radial_weights (U) and adaptive_radial_amplitudes (a).
This test intentionally avoids production-code imports: it exercises only the
developer CLI and generated fixtures.
"""
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "bin/mlp-sus2"
DEFAULT_OLD_BIN = ROOT.parent / "superlinear-pure-w-channelwise-wt/bin/mlp-sus2"
OLD_BIN = Path(os.environ.get("OLD_BIN", DEFAULT_OLD_BIN)).resolve()
(ROOT / ".codex_tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="adaptive-radial-coeff-gate-", dir=ROOT / ".codex_tmp"))
CFG = TMP / "asym_two_species.cfg"
ENV = dict(
    os.environ,
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


CFG.write_text(
    """BEGIN_CFG
 Size
       5
 Supercell
   8.0000000000    0.1700000000    0.0600000000
   0.0300000000    7.7000000000    0.2100000000
   0.0800000000    0.0400000000    8.3000000000
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1.130000 1.720000 1.410000  0.030000 -0.020000  0.010000
 2 1 2.470000 1.190000 3.060000 -0.010000  0.040000 -0.030000
 3 0 3.810000 2.680000 1.970000  0.020000  0.010000 -0.040000
 4 1 5.080000 3.330000 4.440000 -0.030000 -0.010000  0.020000
 5 0 2.260000 4.510000 3.720000  0.015000 -0.025000  0.035000
 Energy
   -7.456789
 PlusStress: xx yy zz yz xz xy
   0.030 -0.020 0.010 0.004 -0.003 0.002
END_CFG
""",
    encoding="utf-8",
)


def run_bin(bin_path, label, *args, env=None, expect_success=True):
    result = subprocess.run(
        [str(bin_path), *map(str, args)],
        cwd=ROOT,
        env=dict(ENV, **(env or {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (TMP / f"{label}.log").write_text(result.stdout, encoding="utf-8")
    if expect_success and result.returncode:
        raise AssertionError(f"{label}: exit {result.returncode}\n{result.stdout[-7000:]}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"{label}: expected failure\n{result.stdout[-3000:]}")
    return result.stdout


def run(label, *args, env=None, expect_success=True):
    return run_bin(BIN, label, *args, env=env, expect_success=expect_success)


def search(src, pattern, name):
    match = re.search(pattern, src)
    if not match:
        raise AssertionError(f"missing {name}")
    return match


def integer(src, key, default=None):
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*(\d+)\s*$", src)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise AssertionError(f"missing integer {key}")


def vector(src, key):
    match = search(
        src,
        rf"(?ms)^\s*{re.escape(key)}\s*=\s*\{{([^}}]*)\}}",
        key,
    )
    body = match.group(1).strip()
    return [float(x) for x in re.findall(NUM, body)] if body else []


def optional_vector(src, key):
    match = re.search(rf"(?ms)^\s*{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", src)
    if not match:
        return None
    body = match.group(1).strip()
    return [float(x) for x in re.findall(NUM, body)] if body else []


def replace_vector(src, key, values):
    replacement = key + " = {" + ", ".join(f"{x:.17e}" for x in values) + "}"
    out, count = re.subn(
        rf"(?ms)^\s*{re.escape(key)}\s*=\s*\{{[^}}]*\}}",
        replacement,
        src,
    )
    if count != 1:
        raise AssertionError((key, count))
    return out


def radial_rows(src):
    match = search(
        src,
        r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)",
        "radial_coeffs",
    )
    rows = [
        [float(x) for x in re.findall(NUM, row)]
        for row in re.findall(r"\{([^}]*)\}", match.group(2))
    ]
    if not rows:
        raise AssertionError("empty radial_coeffs block")
    return rows


def replace_radial_rows(src, rows):
    match = search(
        src,
        r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)",
        "radial_coeffs",
    )
    old_rows = re.findall(r"\{([^}]*)\}", match.group(2))
    if len(old_rows) != len(rows):
        raise AssertionError(("radial row count", len(old_rows), len(rows)))
    new_rows = iter(
        "{" + ", ".join(f"{x:.17e}" for x in row) + "}" for row in rows
    )
    block = re.sub(r"\{[^}]*\}", lambda _: next(new_rows), match.group(2))
    return src[: match.start()] + match.group(1) + block + src[match.end() :]


def field(text, key, default=None):
    matches = re.findall(rf"\b{re.escape(key)}=([^\s]+)", text)
    if matches:
        return matches[-1]
    if default is not None:
        return default
    raise AssertionError(f"missing layout field {key} in: {text[-1000:]}")


def predict(label, model, env=None):
    out = TMP / f"{label}.cfg"
    run(label, "calc-efs", model, CFG, out, env=env)
    text = out.read_text(encoding="utf-8")
    energy = float(search(text, r"(?m)^\s*Energy\s*\n\s*(" + NUM + ")", "Energy")[1])
    atoms = search(text, r"(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy", "AtomData")[1]
    forces = [float(x) for row in atoms.strip().splitlines() for x in row.split()[-3:]]
    stress = [float(x) for x in search(text, r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", "PlusStress")[1].split()]
    if len(forces) != 15 or len(stress) != 6:
        raise AssertionError((len(forces), len(stress)))
    return [energy, *forces, *stress]


def close(a, b, label, atol=2.0e-10):
    error = max(abs(x - y) for x, y in zip(a, b))
    if error > atol:
        raise AssertionError((label, error, a, b))
    print(f"PASS {label}: max_abs={error:.3e}", flush=True)


def patterned(count, amplitude, bias=0.0, phase=0.73):
    return [bias + amplitude * math.cos(phase * (i + 1)) for i in range(count)]


def make_model(mapping, kmax, outer, gate_rank=None, decoupled=False):
    if gate_rank is None:
        gate_rank = kmax
    tag = f"{mapping}-k{kmax}-o{outer}-g{gate_rank}-dec{int(decoupled)}"
    initial = TMP / f"{tag}.initial.mtp"
    model = TMP / f"{tag}.mtp"
    args = [
        "init-sh",
        initial,
        "--species-count=2",
        "--l-max=2",
        f"--k-max={kmax}",
        "--body-order=4",
        "--body-l-max=2,2,2,2",
        "--radial-basis-size=4",
        "--cutoff=5.2",
        "--radial-basis-type=RBChebyshev_sss",
        f"--scaling-map={mapping}",
        "--superlinear-mode=radial-coefficient-gate",
        f"--outer-k-rank={outer}",
        "--two-layer-gate",
        "--two-layer-gate-mode=mu-scalar-full",
        "--two-layer-gate-body-order=4",
        f"--two-layer-gate-k-rank={gate_rank}",
        "--write-sh-scalar-info",
    ]
    if decoupled:
        args.append("--two-layer-gate-decoupled-scaling")
    run(tag + "-init", *args)
    run(tag + "-roundtrip-seed", "roundtrip-model-dev", initial, model)
    src = model.read_text(encoding="utf-8")

    for obsolete in (
        "two_layer_gate_radial_mixing",
        "two_layer_gate_radial_mixing_coeffs",
        "two_layer_gate_additive_coeffs",
        "two_layer_gate_readout",
    ):
        if obsolete in src:
            raise AssertionError(f"obsolete block present in adaptive-radial mode: {obsolete}")

    u_count = len(vector(src, "adaptive_radial_weights"))
    a_count = len(vector(src, "adaptive_radial_amplitudes"))
    recorded_u_count = integer(
        src, "adaptive_radial_weight_count",
        integer(src, "adaptive_radial_weights_count", u_count),
    )
    recorded_a_count = integer(
        src, "adaptive_radial_amplitude_count",
        integer(src, "adaptive_radial_amplitudes_count", a_count),
    )
    if recorded_u_count != u_count:
        raise AssertionError("adaptive_radial_weight_count mismatch")
    if recorded_a_count != a_count:
        raise AssertionError("adaptive_radial_amplitude_count mismatch")
    if u_count == 0 or a_count == 0:
        raise AssertionError("adaptive-radial tensors must be nonempty")

    src = re.sub(r"(?m)^scaling = .*$", "scaling = 1.0", src)
    rows = radial_rows(src)
    rows = [
        [0.08 * (row_index + 1) + 0.019 * (col + 1) + 0.11 * math.sin(0.41 * (row_index + col + 1))
         for col, _ in enumerate(row)]
        for row_index, row in enumerate(rows)
    ]
    src = replace_radial_rows(src, rows)
    src = replace_vector(src, "adaptive_radial_weights", patterned(u_count, 0.37, -0.02, 0.61))
    src = replace_vector(src, "adaptive_radial_amplitudes", patterned(a_count, 0.24, 0.31, 0.47))
    src = replace_vector(src, "two_layer_gate_radial_coeffs", patterned(len(vector(src, "two_layer_gate_radial_coeffs")), 0.18, 0.06, 0.67))
    src = replace_vector(src, "two_layer_gate_type_coeffs", patterned(len(vector(src, "two_layer_gate_type_coeffs")), 0.14, 1.08, 0.53))
    gate_weights = optional_vector(src, "two_layer_gate_weights")
    if gate_weights is not None:
        if gate_weights:
            raise AssertionError("adaptive-radial mode should not keep old additive/readout gate weights")
        src = replace_vector(src, "two_layer_gate_weights", [])
    src = replace_vector(src, "two_layer_gate_k_rank_weights", patterned(len(vector(src, "two_layer_gate_k_rank_weights")), 0.29, 0.12, 0.71))
    src = replace_vector(src, "species_coeffs", [0.17, -0.09])
    src = replace_vector(src, "moment_coeffs", patterned(len(vector(src, "moment_coeffs")), 0.011, 0.0, 0.37))
    src = replace_vector(src, "scal_coeffs", [
        (2.45 if i % 8 < 4 else 1.72) + 0.021 * i
        for i in range(len(vector(src, "scal_coeffs")))
    ])
    if decoupled:
        src = replace_vector(src, "two_layer_gate_scaling_coeffs", [
            (2.20 if i % 8 < 4 else 1.51) + 0.017 * i
            for i in range(len(vector(src, "two_layer_gate_scaling_coeffs")))
        ])
    model.write_text(src, encoding="utf-8")
    return model


def run_gradient(label, model, env=None):
    output = run(
        label,
        "check-loss-gradient-dev",
        model,
        CFG,
        "--max-configs=1",
        "--energy-weight=1",
        "--force-weight=0.35",
        "--stress-weight=0.17",
        "--radial-smooth=0",
        "--gate-x-l2=0",
        "--gate-scalar-l2=0",
        "--gate-mix-l2=0",
        "--gate-full-l2=0",
        "--scalar-head-l2=0",
        "--displacement=1e-5",
        "--abs-tolerance=5e-8",
        "--rel-tolerance=5e-5",
        env=env,
    )
    print(f"PASS {label}: {output.strip().splitlines()[-1]}", flush=True)
    return output


def run_regularized_gradient(label, model, env=None):
    output = run(
        label,
        "check-loss-gradient-dev",
        model,
        CFG,
        "--max-configs=1",
        "--energy-weight=1",
        "--force-weight=0.35",
        "--stress-weight=0.17",
        "--std-weight=0.1",
        "--radial-smooth=1e-6",
        "--radial-smooth-grid=128",
        "--gate-x-l2=0",
        "--gate-scalar-l2=0",
        "--gate-mix-l2=0",
        "--gate-full-l2=1e-4",
        "--scalar-head-l2=0",
        "--displacement=1e-5",
        "--abs-tolerance=2e-6",
        "--rel-tolerance=2e-4",
        env=env,
    )
    print(f"PASS {label}: {output.strip().splitlines()[-1]}", flush=True)
    return output


def regularizer_loss(label, model, env=None):
    output = run(
        label,
        "check-loss-gradient-dev",
        model,
        CFG,
        "--max-configs=1",
        "--energy-weight=0",
        "--force-weight=0",
        "--stress-weight=0",
        "--std-weight=0",
        "--stdd-weight=0",
        "--radial-smooth=1e-6",
        "--radial-smooth-grid=128",
        "--gate-x-l2=0",
        "--gate-scalar-l2=0",
        "--gate-mix-l2=0",
        "--gate-full-l2=1e-4",
        "--scalar-head-l2=0",
        "--displacement=1e-5",
        "--abs-tolerance=2e-6",
        "--rel-tolerance=2e-4",
        "--coeff-start=0",
        "--coeff-end=1",
        env=env,
    )
    last = output.strip().splitlines()[-1]
    return float(field(last, "base_loss"))


def run_efs_fd(label, model, env=None):
    output = run(
        label,
        "check-efs-fd-dev",
        model,
        CFG,
        "--max-configs=1",
        "--displacement=2e-5",
        "--abs-tolerance=4e-7",
        "--rel-tolerance=4e-4",
        env=env,
    )
    last = output.strip().splitlines()[-1]
    if int(field(last, "force_components", "0")) <= 0:
        raise AssertionError((label, "force finite differences were not executed", last))
    if int(field(last, "stress_components", "0")) != 9:
        raise AssertionError((label, "stress finite differences did not cover all tensor components", last))
    force_signal = max(
        abs(float(field(last, "force_analytic", "0"))),
        abs(float(field(last, "force_fd", "0"))),
    )
    stress_signal = max(
        abs(float(field(last, "stress_dedl_analytic", "0"))),
        abs(float(field(last, "stress_dedl_fd", "0"))),
    )
    if force_signal <= 1e-10 or stress_signal <= 1e-10:
        raise AssertionError((label, "force/stress finite-difference branch was vacuous", last))
    print(f"PASS {label}: {last}", flush=True)
    return output


def gauge_fold_model(tag, src):
    amplitudes = vector(src, "adaptive_radial_amplitudes")
    weights = vector(src, "adaptive_radial_weights")
    if not amplitudes or not weights or len(weights) % len(amplitudes) != 0:
        raise AssertionError((tag, "invalid adaptive a/U sizes for gauge folding"))
    q_count = len(weights) // len(amplitudes)
    scales = [0.73 + 0.09 * ((row % 5) + 1) for row in range(len(amplitudes))]
    folded_a = [a * scales[row] for row, a in enumerate(amplitudes)]
    folded_u = [
        weights[row * q_count + q] / scales[row]
        for row in range(len(amplitudes))
        for q in range(q_count)
    ]
    out = replace_vector(src, "adaptive_radial_amplitudes", folded_a)
    out = replace_vector(out, "adaptive_radial_weights", folded_u)
    model = TMP / f"{tag}.gauge-folded.mtp"
    model.write_text(out, encoding="utf-8")
    return model


def directional(model, begin, end, label, env):
    output = run(
        label,
        "check-loss-gradient-direction-dev",
        model,
        CFG,
        "--max-configs=1",
        "--energy-weight=1",
        "--force-weight=0.35",
        "--stress-weight=0.17",
        "--radial-smooth=0",
        "--gate-x-l2=0",
        "--gate-scalar-l2=0",
        "--gate-mix-l2=0",
        "--gate-full-l2=0",
        "--scalar-head-l2=0",
        "--displacement=1e-5",
        "--abs-tolerance=5e-8",
        "--rel-tolerance=5e-5",
        f"--coeff-start={begin}",
        f"--coeff-end={end}",
        env=env,
    )
    last = output.strip().splitlines()[-1]
    analytic = float(search(last, r"\banalytic(?:_directional)?=([^\s]+)", "analytic")[1])
    fd = float(search(last, r"\bfinite_difference=([^\s]+)", "finite_difference")[1])
    return analytic, fd, last


def check_case(mapping, kmax, outer, gate_rank=None, decoupled=False, regularized=False):
    if gate_rank is None:
        gate_rank = kmax
    tag = f"{mapping}-k{kmax}-o{outer}-g{gate_rank}-dec{int(decoupled)}"
    model = make_model(mapping, kmax, outer, gate_rank, decoupled)
    src = model.read_text(encoding="utf-8")
    cache_on = {
        "SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE": "0",
        "SUS2_SH_SITE_DER_CACHE": "1",
        "SUS2_SH_PRODUCT_ROWS": "1",
        "SUS2_SH_PRODUCT_HVT_REVERSE": "1",
    }
    full_efs = dict(cache_on, SUS2_SH_EFS_RADIAL_FULL="1")

    reference = predict(tag + "-cache", model, cache_on)
    close(reference, predict(tag + "-full-efs", model, full_efs), tag + " cache/full-efs")
    if max(map(abs, reference[1:16])) <= 1e-6 or max(map(abs, reference[16:])) <= 1e-7:
        raise AssertionError(f"{tag}: fixture did not activate force and stress branches")

    zero_u = TMP / f"{tag}.zero-u.mtp"
    zero_u.write_text(
        replace_vector(src, "adaptive_radial_weights", [0.0] * len(vector(src, "adaptive_radial_weights"))),
        encoding="utf-8",
    )
    zero_a = TMP / f"{tag}.zero-a.mtp"
    zero_a.write_text(
        replace_vector(src, "adaptive_radial_amplitudes", [0.0] * len(vector(src, "adaptive_radial_amplitudes"))),
        encoding="utf-8",
    )
    for ablated, name in ((zero_u, "zero-U"), (zero_a, "zero-a")):
        ablated_result = predict(tag + "-" + name, ablated, cache_on)
        signal = max(abs(a - b) for a, b in zip(reference[1:], ablated_result[1:]))
        if signal <= 1e-7:
            raise AssertionError((tag, name, "adaptive branch is inactive", signal))
        run_gradient(tag + "-" + name + "-gradient", ablated, cache_on)

    roundtrip = TMP / f"{tag}.roundtrip.mtp"
    run(tag + "-roundtrip", "roundtrip-model-dev", model, roundtrip)
    close(reference, predict(tag + "-roundtrip-predict", roundtrip, cache_on), tag + " model IO")

    run_efs_fd(tag + "-efs-fd", model, cache_on)
    layout = run_gradient(tag + "-mixed-gradient-cache", model, cache_on)
    if int(field(layout, "checked_coeffs")) != int(field(layout, "coeff_count")):
        raise AssertionError((tag, "main mixed gradient did not finite-difference all coefficients", layout))
    if regularized:
        regularized_layout = run_regularized_gradient(tag + "-regularized-gradient-cache", model, cache_on)
        if int(field(regularized_layout, "checked_coeffs")) != int(field(regularized_layout, "coeff_count")):
            raise AssertionError((tag, "regularized gradient did not finite-difference all coefficients", regularized_layout))
        folded = gauge_fold_model(tag, src)
        close(reference, predict(tag + "-gauge-folded", folded, cache_on), tag + " a/U gauge-fold EFS")
        base_regularizer = regularizer_loss(tag + "-regularizer-loss", model, cache_on)
        folded_regularizer = regularizer_loss(tag + "-gauge-folded-regularizer-loss", folded, cache_on)
        if abs(base_regularizer - folded_regularizer) > 1.0e-11:
            raise AssertionError((tag, "a/U gauge-fold regularizer mismatch", base_regularizer, folded_regularizer))

    # If the implementation exposes named coefficient ranges, make sure both
    # new adaptive tensors have active analytic directional gradients.
    for prefix in ("adaptive_radial_weight", "adaptive_radial_amplitude"):
        begin = int(field(layout, prefix + "_begin", "-1"))
        end = int(field(layout, prefix + "_end", "-1"))
        if begin < 0 or end <= begin:
            raise AssertionError((tag, "missing required adaptive layout range", prefix, layout))
        analytic, fd, last = directional(
            model, begin, end, tag + "-" + prefix + "-direction", cache_on
        )
        if max(abs(analytic), abs(fd)) <= 1e-10:
            raise AssertionError((tag, prefix, "inactive directional probe", last))
    wb = int(field(layout, "adaptive_radial_weight_begin"))
    we = int(field(layout, "adaptive_radial_weight_end"))
    ab = int(field(layout, "adaptive_radial_amplitude_begin"))
    ae = int(field(layout, "adaptive_radial_amplitude_end"))
    za_u, za_fd, za_last = directional(
        zero_a, wb, we, tag + "-zero-a-weight-direction", cache_on
    )
    if max(abs(za_u), abs(za_fd)) > 1e-10:
        raise AssertionError((tag, "zero-a should zero U gradient", za_last))
    za_a, za_a_fd, za_a_last = directional(
        zero_a, ab, ae, tag + "-zero-a-amplitude-direction", cache_on
    )
    if max(abs(za_a), abs(za_a_fd)) <= 1e-10:
        raise AssertionError((tag, "zero-a should keep amplitude gradient active", za_a_last))
    zu_u, zu_fd, zu_last = directional(
        zero_u, wb, we, tag + "-zero-u-weight-direction", cache_on
    )
    if max(abs(zu_u), abs(zu_fd)) <= 1e-10:
        raise AssertionError((tag, "zero-U with nonzero a should keep U gradient active", zu_last))

    unit_center = TMP / f"{tag}.unit-center.mtp"
    unit_center.write_text(
        replace_vector(src, "species_coeffs", [1.0] * len(vector(src, "species_coeffs"))),
        encoding="utf-8",
    )
    readout = run(
        tag + "-linear-readout",
        "check-linear-readout-dev",
        unit_center,
        CFG,
        "--max-configs=1",
        "--max-atoms=0",
        "--check-stress",
        "--abs-tolerance=3e-9",
        "--rel-tolerance=3e-9",
        env=cache_on,
    )
    if int(field(readout, "stress_components")) != 9:
        raise AssertionError("linear readout did not cover full stress components")
    reference_components = dict(cache_on, SUS2_SH_ADAPTIVE_COMPONENTS="reference")
    run(
        tag + "-linear-readout-reference-components",
        "check-linear-readout-dev",
        unit_center,
        CFG,
        "--max-configs=1",
        "--max-atoms=0",
        "--check-stress",
        "--abs-tolerance=3e-9",
        "--rel-tolerance=3e-9",
        env=reference_components,
    )
    run(
        tag + "-linear-components-fd",
        "check-linear-components-fd-dev",
        unit_center,
        CFG,
        "--max-configs=1",
        "--max-atoms=0",
        "--displacement=2e-5",
        "--abs-tolerance=4e-7",
        "--rel-tolerance=4e-5",
        env=cache_on,
    )
    run(
        tag + "-linear-components-fd-reference-components",
        "check-linear-components-fd-dev",
        unit_center,
        CFG,
        "--max-configs=1",
        "--max-atoms=0",
        "--displacement=2e-5",
        "--abs-tolerance=4e-7",
        "--rel-tolerance=4e-5",
        env=reference_components,
    )
    print(f"PASS {tag}", flush=True)


def legacy_unmodified_parity():
    if not OLD_BIN.exists() or not BIN.exists():
        print(f"SKIP legacy parity: old={OLD_BIN.exists()} new={BIN.exists()}", flush=True)
        return
    old_model = TMP / "legacy-pure-w-old.mtp"
    old_roundtrip = TMP / "legacy-pure-w-old.roundtrip.mtp"
    run_bin(
        OLD_BIN,
        "legacy-old-init",
        "init-sh",
        old_model,
        "--species-count=2",
        "--l-max=2",
        "--k-max=2",
        "--body-order=4",
        "--body-l-max=2,2,2,2",
        "--radial-basis-size=3",
        "--cutoff=5.2",
        "--scaling-map=K",
        "--superlinear-mode=pure-w-channelwise",
        "--outer-k-rank=2",
        "--two-layer-gate",
        "--two-layer-gate-mode=mu-scalar-full",
        "--two-layer-gate-body-order=4",
        "--two-layer-gate-k-rank=2",
    )
    run_bin(OLD_BIN, "legacy-old-roundtrip", "roundtrip-model-dev", old_model, old_roundtrip)
    old_out = TMP / "legacy.old.cfg"
    new_out = TMP / "legacy.new.cfg"
    run_bin(OLD_BIN, "legacy-old-predict", "calc-efs", old_roundtrip, CFG, old_out)
    run_bin(BIN, "legacy-new-predict", "calc-efs", old_roundtrip, CFG, new_out)
    close(predict_values(old_out), predict_values(new_out), "legacy pure-W old/new parity")


def predict_values(path):
    text = Path(path).read_text(encoding="utf-8")
    energy = float(search(text, r"(?m)^\s*Energy\s*\n\s*(" + NUM + ")", "Energy")[1])
    atoms = search(text, r"(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy", "AtomData")[1]
    forces = [float(x) for row in atoms.strip().splitlines() for x in row.split()[-3:]]
    stress = [float(x) for x in search(text, r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", "PlusStress")[1].split()]
    return [energy, *forces, *stress]


print(f"Evidence: {TMP}", flush=True)
if not BIN.exists():
    raise SystemExit(f"missing mlp-sus2 binary: {BIN}")
legacy_unmodified_parity()

for invalid_outer in (3,):
    run(
        f"invalid-o{invalid_outer}-gt-k",
        "init-sh",
        TMP / "invalid.mtp",
        "--species-count=2",
        "--l-max=2",
        "--k-max=2",
        "--body-order=4",
        "--body-l-max=2,2,2,2",
        "--radial-basis-size=4",
        "--cutoff=5.2",
        "--superlinear-mode=radial-coefficient-gate",
        f"--outer-k-rank={invalid_outer}",
        "--two-layer-gate",
        "--two-layer-gate-mode=mu-scalar-full",
        "--two-layer-gate-k-rank=2",
        expect_success=False,
    )

run(
    "invalid-o4-gt-rawk-with-gate-rank1",
    "init-sh",
    TMP / "invalid-o4-k3-g1.mtp",
    "--species-count=2",
    "--l-max=2",
    "--k-max=3",
    "--body-order=4",
    "--body-l-max=2,2,2,2",
    "--radial-basis-size=4",
    "--cutoff=5.2",
    "--superlinear-mode=radial-coefficient-gate",
    "--outer-k-rank=4",
    "--two-layer-gate",
    "--two-layer-gate-mode=mu-scalar-full",
    "--two-layer-gate-k-rank=1",
    expect_success=False,
)

for scaling_map in ("K", "L", "LK"):
    check_case(scaling_map, kmax=2, outer=1, regularized=(scaling_map == "K"))
    check_case(scaling_map, kmax=2, outer=2)

check_case("LK", kmax=3, outer=3, gate_rank=1)
check_case("K", kmax=3, outer=2, gate_rank=1, decoupled=True)

print("PASS adaptive radial coefficient gate regression", flush=True)
