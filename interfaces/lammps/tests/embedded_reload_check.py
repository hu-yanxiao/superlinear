#!/usr/bin/env python3
"""Check same-pair reload does not retain inferred element metadata.

Uses external native oracle fixtures. Only copies inside the new test output
are changed; model checkpoints and source fixtures are never modified.
"""
import argparse
import json
from pathlib import Path
import re

from cpu_qualification import Qualification, read_cfg, restricted, require, validate_ranks


class ReloadCheck(Qualification):
    def command(self, argv, cwd, log):
        if "-in" in argv:
            path = Path(argv[argv.index("-in")+1])
            text = path.read_text()
            # Both models carry their own atomic-number metadata. This tests
            # auto-symbol mapping, not model_elements explicitly fixed by input.
            text = text.replace(" model_elements H,O", "")
            coeff = next(line for line in text.splitlines() if line.startswith("pair_coeff "))
            text = text.replace(coeff, coeff+'\nrun 0\n'+
                f'pair_coeff * * "{self.args.reload_model}" C N\nrun 0\n'+coeff, 1)
            path.write_text(text)
        return super().command(argv, cwd, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lammps", "native", "matrix-dir", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("lammps", "native", "matrix_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.mpiexec="mpirun"
    args.pair_style="sus2/embedded"
    args.tabstep=0.001
    args.direct_tolerance=1e-9
    args.table_tolerance=2e-6
    args.timeout=120
    validate_ranks([4])
    check=ReloadCheck(args)
    cases=json.loads((args.matrix_dir/"matrix.json").read_text())["cases"]
    primary=next(c for c in cases if c["spec"].get("zbl"))
    adaptive=next(c for c in cases if c["spec"]["family"]=="adaptive")
    model=args.matrix_dir/Path(primary["model"]).name
    secondary=args.matrix_dir/Path(adaptive["model"]).name
    # The disabled-ZBL Adaptive fixture gains only mapping metadata. No
    # coefficients, graph, precision, cutoff or atom geometry are changed.
    text=secondary.read_text()
    require(not re.search(r'^\s*zbl_enabled\s*=\s*true',text,re.M), "Expected disabled ZBL")
    text=re.sub(r'^\s*zbl_atomic_numbers\s*=.*$', '', text, flags=re.M)
    args.reload_model=check.out/"adaptive-map-C-N.sus2"
    args.reload_model.write_text(text+'\nzbl_atomic_numbers = {6, 7}\n')
    geometry=restricted(read_cfg(args.matrix_dir/"asymmetric.cfg"))
    for mode in ("direct", "table"):
        check.run(primary, model, geometry, 4, mode)
    failures=[r for r in check.report["runs"] if r["status"]!="PASS"]
    check.report.update(status="FAIL" if failures else "PASS", failures=len(failures))
    check.save()
    print(json.dumps({"status":check.report["status"],"failures":failures}))
    return bool(failures)


if __name__=="__main__":
    raise SystemExit(main())
