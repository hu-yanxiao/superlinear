#!/usr/bin/env python3
"""Compare two calc-efs CFG files by their complete numeric payload."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path


NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def numbers(path: str) -> list[float]:
    return [float(value) for value in NUMBER.findall(Path(path).read_text())]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_cfg_numeric.py reference.cfg candidate.cfg")
    reference = numbers(sys.argv[1])
    candidate = numbers(sys.argv[2])
    if len(reference) != len(candidate):
        raise SystemExit(f"numeric length mismatch: {len(reference)} != {len(candidate)}")
    max_abs = 0.0
    max_rel = 0.0
    worst = -1
    for index, (lhs, rhs) in enumerate(zip(reference, candidate)):
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            raise SystemExit(f"non-finite value at numeric token {index}: {lhs}, {rhs}")
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
    if max_abs > 1.0e-12 and max_rel > 1.0e-12:
        raise SystemExit("numeric parity tolerance exceeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
