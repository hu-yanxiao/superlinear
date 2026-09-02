#!/usr/bin/env python3
"""Measure or enforce numerical parity between two calc-efs CFG files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def numbers(path: Path) -> list[float]:
    return [float(value) for value in NUMBER.findall(path.read_text(encoding="utf-8"))]


def efs_values(path: Path) -> dict[str, list[float]]:
    text = path.read_text(encoding="utf-8")
    size_match = re.search(r"(?m)^\s*Size\s*\n\s*(\d+)\s*$", text)
    atom_match = re.search(r"(?m)^\s*AtomData:\s*(.*?)$", text)
    energy_match = re.search(r"(?m)^\s*Energy\s*\n\s*(" + NUMBER.pattern + r")", text)
    stress_match = re.search(r"(?m)^\s*PlusStress:.*\n\s*(.*?)$", text)
    if not (size_match and atom_match and energy_match and stress_match):
        raise SystemExit(f"unable to parse Energy/Force/Stress fields: {path}")

    size = int(size_match.group(1))
    columns = atom_match.group(1).split()
    try:
        force_indices = [columns.index(name) for name in ("fx", "fy", "fz")]
    except ValueError as exc:
        raise SystemExit(f"AtomData force columns are missing: {path}") from exc
    atom_lines = text[atom_match.end() :].splitlines()[1 : size + 1]
    if len(atom_lines) != size:
        raise SystemExit(f"AtomData row count mismatch: {path}")
    forces: list[float] = []
    for line in atom_lines:
        values = [float(value) for value in NUMBER.findall(line)]
        if len(values) != len(columns):
            raise SystemExit(f"AtomData column mismatch: {path}: {line}")
        forces.extend(values[index] for index in force_indices)
    stress = [float(value) for value in NUMBER.findall(stress_match.group(1))]
    if len(stress) != 6:
        raise SystemExit(f"PlusStress component count mismatch: {path}")
    return {
        "energy": [float(energy_match.group(1))],
        "force": forces,
        "stress": stress,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--require-nonzero", action="store_true")
    parser.add_argument("--min-efs-effect", type=float, default=0.0)
    args = parser.parse_args()

    reference = numbers(args.reference)
    candidate = numbers(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit(f"numeric length mismatch: {len(reference)} != {len(candidate)}")
    max_abs = 0.0
    max_rel = 0.0
    worst = -1
    for index, (lhs, rhs) in enumerate(zip(reference, candidate)):
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            raise SystemExit(f"non-finite value at token {index}: {lhs}, {rhs}")
        absolute = abs(lhs - rhs)
        relative = absolute / max(1.0, abs(lhs), abs(rhs))
        if absolute > max_abs:
            max_abs = absolute
            worst = index
        max_rel = max(max_rel, relative)
    print(
        f"numeric_tokens={len(reference)} max_abs={max_abs:.17e} "
        f"max_rel={max_rel:.17e} worst_token={worst}"
    )
    reference_efs = efs_values(args.reference)
    candidate_efs = efs_values(args.candidate)
    efs_max_abs: dict[str, float] = {}
    for field in ("energy", "force", "stress"):
        lhs = reference_efs[field]
        rhs = candidate_efs[field]
        if len(lhs) != len(rhs):
            raise SystemExit(f"{field} component count mismatch")
        efs_max_abs[field] = max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)
    print(
        "efs_max_abs "
        f"energy={efs_max_abs['energy']:.17e} "
        f"force={efs_max_abs['force']:.17e} "
        f"stress={efs_max_abs['stress']:.17e}"
    )
    if args.tolerance is not None and max(max_abs, max_rel) > args.tolerance:
        raise SystemExit("numeric parity tolerance exceeded")
    if args.require_nonzero and max(efs_max_abs.values()) <= args.min_efs_effect:
        raise SystemExit(
            "expected an Energy/Force/Stress effect greater than "
            f"{args.min_efs_effect:.17e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
