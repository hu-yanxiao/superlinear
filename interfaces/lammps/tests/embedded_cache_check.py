#!/usr/bin/env python3
"""Compare exact CPU cache routes with a separately built reference engine.

Uses the bundled two-cutoff native/PT fixtures and the embedded_efs harness.
The simulated domains include empty owners. Real MPI must be qualified separately.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

FLAGS = ("SUS2_SL_STEP_RADIAL_CACHE", "SUS2_SL_ADAPTIVE_PROJECTED_COMM",
         "SUS2_SL_ADAPTIVE_COEFFICIENT_CACHE", "SUS2_SL_STEP_RADIAL_CACHE_MB",
         "SUS2_SL_EFFECT_SOURCE_CACHE", "SUS2_SL_EFFECT_FORWARD_STREAM",
         "SUS2_SL_STEP_ANGULAR_CACHE", "SUS2_SL_STEP_ANGULAR_CACHE_MB")


def flatten(value):
    if isinstance(value, list):
        return [item for row in value for item in flatten(row)]
    return [float(value)]


def evaluate(binary, model, cfg, domains, mode, route, log):
    env = os.environ.copy()
    for key in FLAGS:
        env.pop(key, None)
    if route == "off":
        env.update({key: "0" for key in FLAGS if not key.endswith("_MB")})
    if route == "zero-budget":
        env[FLAGS[3]] = "0"
        env["SUS2_SL_STEP_ANGULAR_CACHE_MB"] = "0"
    if route == "angular-off":
        env["SUS2_SL_STEP_ANGULAR_CACHE"] = "0"
    if route == "angular-only":
        env["SUS2_SL_STEP_RADIAL_CACHE"] = "0"
    result = subprocess.run([str(binary), str(model), str(cfg), str(domains), mode],
                            env=env, text=True, capture_output=True, timeout=60)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Engine failed: {log}")
    records = [line.split(" ", 1)[1] for line in result.stdout.splitlines()
               if line.startswith("NATIVE_EFS_JSON ")]
    if len(records) != 1:
        raise RuntimeError(f"Missing or repeated EFS record: {log}")
    return json.loads(records[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.candidate, args.reference = args.candidate.resolve(), args.reference.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = Path(__file__).resolve().parent / "fixtures/two_cutoff/fixtures_v2"
    report = {"status": "RUNNING", "absolute_tolerance": 1e-9, "runs": []}
    try:
        for family in ("static", "effect", "adaptive_full", "adaptive_shared-n"):
            for mode in ("direct", "table"):
                for domains in (1, 2, 40):
                    label = f"{family}-{mode}-{domains}"
                    model, cfg = fixtures / (family + ".mtp"), fixtures / "geometries.cfg"
                    ref = evaluate(args.reference, model, cfg, domains, mode, "off",
                                   args.out / (label + "-reference.log"))
                    for route in ("default", "off", "zero-budget", "angular-off", "angular-only"):
                        actual = evaluate(args.candidate, model, cfg, domains, mode, route,
                                          args.out / (label + "-" + route + ".log"))
                        errors = {}
                        for key in ("energy", "forces", "virial", "site_energies"):
                            a, b = flatten(actual[key]), flatten(ref[key])
                            if len(a) != len(b) or not all(math.isfinite(x) for x in a+b):
                                raise RuntimeError(f"Invalid {key}: {label}-{route}")
                            errors[key] = max((abs(x-y) for x, y in zip(a, b)), default=0.)
                        passed = all(value <= 1e-9 for value in errors.values())
                        report["runs"].append(dict(case=label, route=route,
                                                   errors=errors, passed=passed))
                        if not passed:
                            raise RuntimeError(f"Exact cache mismatch: {label}-{route}: {errors}")
        report["status"] = "PASS"
    finally:
        (args.out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(report['runs'])} exact cache comparisons")


if __name__ == "__main__":
    main()
