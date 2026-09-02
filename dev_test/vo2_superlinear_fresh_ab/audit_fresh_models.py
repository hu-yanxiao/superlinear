#!/usr/bin/env python3
"""Audit deterministic fresh A/B0/B SUS2-SH model artifacts."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

COMMON_BLOCKS = (
    "scal_coeffs",
    "radial_coeffs",
    "outer_k_rank1_weights",
    "two_layer_gate_k_rank_weights",
    "two_layer_gate_radial_coeffs",
    "two_layer_gate_additive_coeffs",
    "two_layer_gate_type_coeffs",
    "two_layer_gate_weights",
    "species_coeffs",
    "moment_coeffs",
)

TOPOLOGY_FIELDS = (
    "scaling_map",
    "species_count",
    "sh_l_max",
    "sh_k_max",
    "sh_body_order",
    "sh_body_l_max",
    "radial_basis_type",
    "min_dist",
    "max_dist",
    "radial_basis_size",
    "outer_k_rank",
    "two_layer_gate_k_rank",
    "two_layer_gate_mode",
    "two_layer_gate_body_order_max",
)


def read_model(path: Path) -> tuple[str, dict[str, str], dict[str, tuple[float, ...]]]:
    text = path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if match:
            fields[match.group(1)] = match.group(2)

    vectors: dict[str, tuple[float, ...]] = {}
    for match in re.finditer(r"(?ms)^\s*([A-Za-z0-9_]+)\s*=\s*\{([^}]*)\}", text):
        vectors[match.group(1)] = tuple(float(x) for x in NUMBER.findall(match.group(2)))

    radial = re.search(
        r"(?ms)^\s*radial_coeffs\s*$\s*(.*?)^\s*alpha_moments_count\s*=",
        text,
    )
    if not radial:
        raise SystemExit(f"missing multiline radial_coeffs block: {path}")
    radial_rows = re.findall(r"\{([^}]*)\}", radial.group(1))
    vectors["radial_coeffs"] = tuple(
        float(x) for row in radial_rows for x in NUMBER.findall(row)
    )
    if not vectors["radial_coeffs"]:
        raise SystemExit(f"empty multiline radial_coeffs block: {path}")
    return text, fields, vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("bzero", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--mix-scale", type=float, default=1.0e-4)
    args = parser.parse_args()

    models = {
        "A": read_model(args.a),
        "B0": read_model(args.bzero),
        "B": read_model(args.b),
    }
    a_text, a_fields, a_vectors = models["A"]

    if "two_layer_gate_radial_mixing = full" in a_text:
        raise SystemExit("A unexpectedly enables radial mixing")
    for name in ("B0", "B"):
        if "two_layer_gate_radial_mixing = full" not in models[name][0]:
            raise SystemExit(f"{name} does not enable full radial mixing")

    for field in TOPOLOGY_FIELDS:
        if field not in a_fields:
            raise SystemExit(f"A is missing topology field {field}")
        for name in ("B0", "B"):
            candidate = models[name][1]
            if candidate.get(field) != a_fields[field]:
                raise SystemExit(
                    f"topology mismatch {field}: A={a_fields[field]!r} "
                    f"{name}={candidate.get(field)!r}"
                )

    common_values = 0
    for block in COMMON_BLOCKS:
        expected = a_vectors.get(block)
        if not expected:
            raise SystemExit(f"A is missing nonempty common block {block}")
        common_values += len(expected)
        for name in ("B0", "B"):
            candidate = models[name][2].get(block)
            if not candidate:
                raise SystemExit(f"{name} is missing nonempty common block {block}")
            if candidate != expected:
                raise SystemExit(f"common initialization mismatch in {block}: A versus {name}")

    zero_mix = models["B0"][2].get("two_layer_gate_radial_mixing_coeffs")
    active_mix = models["B"][2].get("two_layer_gate_radial_mixing_coeffs")
    if not zero_mix or any(value != 0.0 for value in zero_mix):
        raise SystemExit("B0 radial mixing coefficients are not exactly zero")
    if not active_mix or len(active_mix) != len(zero_mix):
        raise SystemExit("B radial mixing coefficient block is missing or has the wrong size")
    if not all(math.isfinite(value) and abs(value) <= args.mix_scale for value in active_mix):
        raise SystemExit("B radial mixing coefficient is nonfinite or outside the requested scale")
    if not any(value != 0.0 for value in active_mix):
        raise SystemExit("B radial mixing coefficients are all zero")

    print(
        "fresh model audit passed: "
        f"common_blocks={len(COMMON_BLOCKS)} common_values={common_values} "
        f"mixing_coeffs={len(active_mix)} max_abs_mix={max(map(abs, active_mix)):.17e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
