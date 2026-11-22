#!/usr/bin/env python3
"""Compare embedded direct and contracted tables to the native full-precision E/F/virial."""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


def native_efs(path):
    lines = [line.strip() for line in path.read_text().splitlines()]
    energy = float(lines[lines.index("Energy") + 1])
    count = int(lines[lines.index("Size") + 1])
    header = next(i for i, line in enumerate(lines) if line.startswith("AtomData:"))
    fields = lines[header].split()[1:]
    forces = [[float(line.split()[fields.index(key)]) for key in ("fx", "fy", "fz")]
              for line in lines[header+1:header+1+count]]
    i = next(i for i, line in enumerate(lines) if line.startswith("PlusStress:"))
    xx, yy, zz, yz, xz, xy = map(float, lines[i+1].split())
    return dict(energy=energy, forces=forces, virial=[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def errors(got, ref):
    result = {}
    for key, reference in ref.items():
        observed = got[key]
        if key == "energy":
            delta = [abs(observed-reference)]
        else:
            if len(observed) != len(reference) or any(
                len(a) != len(b) for a, b in zip(observed, reference)
            ):
                raise AssertionError(f"{key} shape mismatch")
            delta = [abs(x-y) for a, b in zip(observed, reference) for x, y in zip(a, b)]
        if not all(math.isfinite(value) for value in delta):
            raise AssertionError(f"nonfinite {key}")
        result[key] = max(delta, default=0.0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    fixtures = Path(__file__).resolve().parent/"fixtures"/"effect-mix"
    for arg in ("native", "embedded"):
        p.add_argument("--"+arg, required=True, type=Path)
    p.add_argument("--cfg", type=Path, default=fixtures/"geometry.cfg")
    p.add_argument("models", nargs="*", type=Path, default=sorted(fixtures.glob("*.sus2")))
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        for model in args.models:
            cfg = Path(temp)/"oracle.cfg"
            subprocess.run([args.native.resolve(), "calc-efs", model.resolve(), args.cfg.resolve(),
                            cfg, "--no-loss"], check=True, stdout=subprocess.DEVNULL)
            ref = native_efs(cfg)
            for mode, budget in (("direct", "auto"), ("table", "auto"), ("table", "0")):
                for domains in (1,2):
                    text = subprocess.check_output([args.embedded, model, args.cfg, str(domains), mode,
                                                    "0.001", budget, "present"], text=True)
                    records = dict((line.split(" ",1)[0],json.loads(line.split(" ",1)[1]))
                                   for line in text.splitlines() if "_JSON " in line)
                    result = errors(records["NATIVE_EFS_JSON"], ref)
                    stats = records["RADIAL_STATS_JSON"]
                    table = mode == "table" and budget != "0"
                    if not all(value <= (2e-6 if table else 2e-9) for value in result.values()):
                        raise AssertionError((model, mode, budget, domains, result))
                    if (table and not stats["hits"]) or (budget=="0" and not stats["memory_fallbacks"]):
                        raise AssertionError((model, mode, budget, stats))
                    print(model.name, mode, budget, domains, json.dumps(result))
if __name__ == "__main__":
    main()
