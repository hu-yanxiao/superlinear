#!/usr/bin/env python3
"""Contract checks for adaptive radial gate same-k topology.

This script is intended for server-side execution after building mlp-sus2.  It
does not depend on test data: it initializes small adaptive models, checks the
serialized topology/counts, verifies model roundtrip preserves explicit
outer/gate pruning metadata, and confirms resume rejects mismatched explicit
graph modes.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def run(cmd: list[str], *, expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_ok and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    if not expect_ok and proc.returncode == 0:
        raise SystemExit(f"command unexpectedly succeeded: {' '.join(cmd)}\n{proc.stdout}")
    return proc


def scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([^\n]+?)\s*$", text)
    if match is None:
        raise AssertionError(f"missing scalar key {key}")
    return match.group(1).strip()


def integer(text: str, key: str) -> int:
    return int(scalar(text, key))


def gate_basic_index(
    channel: int,
    l: int,
    m: int,
    *,
    lmax: int,
    gate_rank: int,
    basic_begin: int,
    basic_count: int,
) -> int:
    local = (
        gate_rank * ((lmax + 1) * (lmax + 1) - (l + 1) * (l + 1))
        + (gate_rank - 1 - channel) * (2 * l + 1)
        + (m + l)
    )
    if local < 0 or local >= basic_count:
        raise AssertionError("test reconstructed invalid gate basic index")
    return basic_begin + local


def corrupt_same_k_product_lane(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    lmax = integer(text, "sh_l_max")
    gate_rank = integer(text, "two_layer_gate_k_rank")
    basic_begin = integer(text, "two_layer_gate_k_rank_basic_begin")
    basic_count = integer(text, "two_layer_gate_k_rank_basic_count")
    product_begin = integer(text, "two_layer_gate_k_rank_product_begin")
    product_end = integer(text, "two_layer_gate_k_rank_product_end")
    products_match = re.search(
        r"(?ms)^sh_products\s*=\s*\{(.*?)\}\s*\nalpha_scalar_moments",
        text,
    )
    if products_match is None:
        raise AssertionError("missing sh_products block")
    body = products_match.group(1)
    product_matches = list(
        re.finditer(r"\{(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*([^{}]+?)\}", body)
    )
    if product_end > len(product_matches):
        raise AssertionError("product range exceeds parsed sh_products")

    channel = {}
    for ch in range(gate_rank):
        for l in range(lmax, -1, -1):
            for m in range(-l, l + 1):
                channel[
                    gate_basic_index(
                        ch,
                        l,
                        m,
                        lmax=lmax,
                        gate_rank=gate_rank,
                        basic_begin=basic_begin,
                        basic_count=basic_count,
                    )
                ] = ch

    corrupt_index = None
    corrupt_left = None
    for idx in range(product_begin, product_end):
        match = product_matches[idx]
        left = int(match.group(1))
        right = int(match.group(2))
        target_moment = int(match.group(3))
        left_ch = channel.get(left)
        right_ch = channel.get(right)
        if left_ch is None or right_ch is None:
            continue
        if left_ch != right_ch:
            raise AssertionError("original same-k model is already cross-channel")
        other_ch = (right_ch + 1) % gate_rank
        corrupt_left = gate_basic_index(
            other_ch,
            0,
            0,
            lmax=lmax,
            gate_rank=gate_rank,
            basic_begin=basic_begin,
            basic_count=basic_count,
        )
        if corrupt_left == left:
            corrupt_left = gate_basic_index(
                (right_ch + 2) % gate_rank,
                0,
                0,
                lmax=lmax,
                gate_rank=gate_rank,
                basic_begin=basic_begin,
                basic_count=basic_count,
            )
        corrupt_index = idx
        channel[target_moment] = left_ch
        break

    if corrupt_index is None or corrupt_left is None:
        raise AssertionError("could not find gate product to corrupt")
    match = product_matches[corrupt_index]
    replacement = "{" + ", ".join(
        [
            str(corrupt_left),
            match.group(2),
            match.group(3),
            match.group(4).strip(),
        ]
    ) + "}"
    new_body = body[: match.start()] + replacement + body[match.end() :]
    target.write_text(
        text[: products_match.start(1)]
        + new_body
        + text[products_match.end(1) :],
        encoding="utf-8",
    )


def init_model(
    binary: Path,
    out: Path,
    gate_factor: str,
    *,
    outer_factor: Optional[str] = None,
    outer_rank: int,
    gate_rank: int,
    body_order: int,
    gate_body_order: int,
    body_l_max: str,
) -> None:
    cmd = [
        str(binary),
        "init-sh",
        str(out),
        "--species-count=2",
        "--l-max=3",
        "--k-max=8",
        f"--body-order={body_order}",
        f"--body-l-max={body_l_max}",
        "--cutoff=6.5",
        "--radial-basis-size=10",
        "--radial-basis-type=RBChebyshev_sss",
        "--scaling-map=K",
        "--superlinear-mode=radial-coefficient-gate",
        f"--outer-k-rank={outer_rank}",
        f"--two-layer-gate-k-rank={gate_rank}",
        f"--two-layer-gate-body-order={gate_body_order}",
        f"--two-layer-gate-factor-pruning={gate_factor}",
    ]
    if outer_factor is not None:
        cmd.append(f"--sh-factor-pruning={outer_factor}")
    run(cmd)


def tiny_cfg(path: Path) -> None:
    path.write_text(
        """BEGIN_CFG
 Size
    2
 Supercell
    8.0 0.0 0.0
    0.0 8.0 0.0
    0.0 0.0 8.0
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
    1    0       0.0           0.0           0.0          0.0         0.0         0.0
    2    1       1.9           0.2           0.1          0.0         0.0         0.0
 Energy
    0.0
 PlusStress:  xx          yy          zz          yz          xz          xy
    0.0         0.0         0.0         0.0         0.0         0.0
END_CFG
""",
        encoding="utf-8",
    )


def check_model(
    path: Path,
    expected_factor: str,
    *,
    expected_q: int,
    expected_outer_rank: int,
    expected_gate_rank: int,
    expected_gate_body_order: int,
    expected_outer_channel_mode: str = "same-channel",
) -> None:
    text = path.read_text(encoding="utf-8")
    if scalar(text, "outer_radial_mode") != "adaptive-coefficients":
        raise AssertionError("model is not adaptive-coefficients")
    if scalar(text, "outer_channel_mode") != expected_outer_channel_mode:
        raise AssertionError("model has wrong outer channel mode")
    if scalar(text, "two_layer_gate_factor_pruning") != expected_factor:
        raise AssertionError("wrong serialized gate factor-pruning mode")
    if integer(text, "outer_k_rank") != expected_outer_rank:
        raise AssertionError("wrong outer rank")
    if integer(text, "two_layer_gate_k_rank") != expected_gate_rank:
        raise AssertionError("wrong gate rank")
    if integer(text, "two_layer_gate_body_order_max") != expected_gate_body_order:
        raise AssertionError("wrong gate body-order")
    q = integer(text, "two_layer_gate_scalar_count")
    if q != expected_q:
        raise AssertionError(f"wrong gate scalar count: got {q}, expected {expected_q}")
    rows = 4 * expected_outer_rank * 10
    if integer(text, "adaptive_radial_amplitude_count") != rows:
        raise AssertionError("wrong adaptive row count")
    if integer(text, "adaptive_radial_weight_count") != rows * q:
        raise AssertionError("adaptive weight count does not match rows*Q")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        default=os.environ.get("MLP_BIN", "bin/mlp-sus2"),
        help="Path to built mlp-sus2 binary",
    )
    args = parser.parse_args(argv)
    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise SystemExit(f"missing binary: {binary}")

    with tempfile.TemporaryDirectory(prefix="adaptive_gate_same_k_") as tmp:
        root = Path(tmp)
        same = root / "same_k.mtp"
        qtotal = root / "q_total.mtp"
        cross_outer_same_gate = root / "cross_outer_same_gate.mtp"
        init_model(
            binary,
            same,
            "same-k",
            outer_rank=8,
            gate_rank=8,
            body_order=6,
            gate_body_order=6,
            body_l_max="3,3,3,2,1",
        )
        init_model(
            binary,
            qtotal,
            "q-total",
            outer_rank=2,
            gate_rank=4,
            body_order=6,
            gate_body_order=4,
            body_l_max="3,3,3,2,1",
        )
        init_model(
            binary,
            cross_outer_same_gate,
            "same-k",
            outer_factor="q-total",
            outer_rank=2,
            gate_rank=8,
            body_order=6,
            gate_body_order=6,
            body_l_max="3,3,3,2,1",
        )

        check_model(
            same,
            "same-k",
            expected_q=232,
            expected_outer_rank=8,
            expected_gate_rank=8,
            expected_gate_body_order=6,
        )
        check_model(
            qtotal,
            "q-total",
            expected_q=348,
            expected_outer_rank=2,
            expected_gate_rank=4,
            expected_gate_body_order=4,
        )
        check_model(
            cross_outer_same_gate,
            "same-k",
            expected_q=232,
            expected_outer_rank=2,
            expected_gate_rank=8,
            expected_gate_body_order=6,
            expected_outer_channel_mode="cross-channel",
        )

        same_rt = root / "same_k_roundtrip.mtp"
        qtotal_rt = root / "q_total_roundtrip.mtp"
        cross_outer_same_gate_rt = root / "cross_outer_same_gate_roundtrip.mtp"
        run([str(binary), "roundtrip-model-dev", str(same), str(same_rt)])
        run([str(binary), "roundtrip-model-dev", str(qtotal), str(qtotal_rt)])
        run(
            [
                str(binary),
                "roundtrip-model-dev",
                str(cross_outer_same_gate),
                str(cross_outer_same_gate_rt),
            ]
        )
        check_model(
            same_rt,
            "same-k",
            expected_q=232,
            expected_outer_rank=8,
            expected_gate_rank=8,
            expected_gate_body_order=6,
        )
        check_model(
            qtotal_rt,
            "q-total",
            expected_q=348,
            expected_outer_rank=2,
            expected_gate_rank=4,
            expected_gate_body_order=4,
        )
        check_model(
            cross_outer_same_gate_rt,
            "same-k",
            expected_q=232,
            expected_outer_rank=2,
            expected_gate_rank=8,
            expected_gate_body_order=6,
            expected_outer_channel_mode="cross-channel",
        )

        corrupt = root / "same_k_cross_lane_corrupt.mtp"
        corrupt_same_k_product_lane(same, corrupt)
        corrupt_result = run(
            [str(binary), "roundtrip-model-dev", str(corrupt), str(root / "corrupt_out.mtp")],
            expect_ok=False,
        )
        if "same-k gate product crosses gate channels" not in corrupt_result.stdout:
            raise AssertionError("corrupt same-k gate product was not rejected by provenance validator")

        cfg = root / "tiny.cfg"
        tiny_cfg(cfg)
        wrong_same = run(
            [
                str(binary),
                "train",
                str(same),
                str(cfg),
                "--two-layer-gate-factor-pruning=q-total",
                "--max-iter=0",
            ],
            expect_ok=False,
        )
        if "two-layer-gate-factor-pruning does not match" not in wrong_same.stdout:
            raise AssertionError("same-k resume mismatch did not fail at gate-pruning check")
        wrong_qtotal = run(
            [
                str(binary),
                "train",
                str(qtotal),
                str(cfg),
                "--two-layer-gate-factor-pruning=same-k",
                "--max-iter=0",
            ],
            expect_ok=False,
        )
        if "two-layer-gate-factor-pruning does not match" not in wrong_qtotal.stdout:
            raise AssertionError("q-total resume mismatch did not fail at gate-pruning check")
        wrong_outer = run(
            [
                str(binary),
                "train",
                str(cross_outer_same_gate),
                str(cfg),
                "--sh-factor-pruning=same-k",
                "--max-iter=0",
            ],
            expect_ok=False,
        )
        if "sh-factor-pruning must match" not in wrong_outer.stdout:
            raise AssertionError("cross-channel outer resume mismatch did not fail at outer-pruning check")

    print("adaptive_gate_same_k_contract_check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
