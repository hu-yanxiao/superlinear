#!/usr/bin/env python3
"""Real-MPI numbered-input regression; needs the external native oracle fixtures.

No jobs are submitted here. Run inside a four-core CPU allocation. E/F/global
virial/site energies use the same strict tolerances as cpu_qualification.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

import numpy as np
from cpu_qualification import (Geometry, Qualification, read_cfg, restricted,
                               require, validate_ranks, write_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lammps", "native", "matrix-dir", "carbon-model", "carbon-cfg", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("lammps", "native", "matrix_dir", "carbon_model", "carbon_cfg", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.mpiexec = "mpirun"
    args.pair_style = "sus2/embedded"
    args.tabstep = 0.001
    args.direct_tolerance = 1e-9
    args.table_tolerance = 2e-6
    args.timeout = 120
    validate_ranks([1, 4])
    check = Qualification(args)
    matrix = json.loads((args.matrix_dir/"matrix.json").read_text())
    geometry = restricted(read_cfg(args.matrix_dir/"asymmetric.cfg"))
    selected = [next(c for c in matrix["cases"] if c["spec"]["family"] == family)
                for family in ("static", "legacy", "pure", "adaptive")]
    zbl_case = next(c for c in matrix["cases"] if c["spec"].get("zbl"))
    selected.append(zbl_case)
    serial = 0
    for original in selected:
        model = args.matrix_dir/Path(original["model"]).name
        require(hashlib.sha256(model.read_bytes()).hexdigest() == original["sha256"], "Fixture hash changed")
        for form in ("numbered-style", "numbered-coeff", "symbols"):
            serial += 1
            case = dict(original, index=serial, _input_form=form)
            for mode in ("direct", "table"):
                check.run(case, model, geometry, 4, mode)
        # A prefix-only system must keep type 1 -> native species 0, regardless
        # of mass (the qualification deliberately assigns mass 1 to all types).
        serial += 1
        prefix = Geometry(geometry.cell.copy(), geometry.positions.copy(), np.zeros_like(geometry.types))
        case = dict(original, index=serial, _input_form="numbered-style", _declared_types=1)
        check.run(case, model, prefix, 4, "direct")
        serial += 1
        case = dict(original, index=serial, _type_order=[1, 0])
        check.run(case, model, geometry, 4, "direct")
    carbon = restricted(read_cfg(args.carbon_cfg))
    for form in ("numbered-style", "numbered-coeff", "symbols"):
        serial += 1
        case = {"index":serial, "spec":{"family":"carbon"}, "_elements":["C"], "_input_form":form}
        for mode in ("direct", "table"):
            for ranks in (1, 4):
                check.run(case, args.carbon_model, carbon, ranks, mode)
    adaptive = selected[3]
    model = args.matrix_dir/Path(adaptive["model"]).name
    for form in ("numbered-style", "numbered-coeff"):
        serial += 1
        case = dict(adaptive, index=serial, _input_form=form)
        check.run(case, model, geometry, 4, "table", "migration")
        check.run(case, model, geometry, 4, "table", "nve")

    model = args.matrix_dir/Path(selected[0]["model"]).name
    zbl_model = args.matrix_dir/Path(zbl_case["model"]).name
    negative = [
        ("missing-model", "pair_style sus2/embedded", "pair_coeff * *", 2, "Specify the model filename in pair_style or pair_coeff"),
        ("double-model", f'pair_style sus2/embedded "{model}"', f'pair_coeff * * "{model}"', 2, "Specify the model filename only once"),
        ("too-many-types", f'pair_style sus2/embedded "{model}"', "pair_coeff * *", 3, "atom type count exceeds model species count"),
        ("missing-value", "pair_style sus2/embedded radial_mode", "pair_coeff * *", 2, "options require key/value pairs"),
        ("unknown-option", f'pair_style sus2/embedded "{model}" bad_option 2', "pair_coeff * *", 2, "Unknown sus2/embedded option"),
        ("bad-mode", f'pair_style sus2/embedded "{model}" radial_mode bogus', "pair_coeff * *", 2, "radial_mode must be"),
        ("missing-symbol", "pair_style sus2/embedded model_elements H,O", f'pair_coeff * * "{model}" H', 2, "Expected pair_coeff"),
        # LAMMPS rejects non-wildcard ranges before calling a many-body pair's
        # coeff method; assert its diagnostic rather than the adapter's one.
        ("bad-range", f'pair_style sus2/embedded "{model}"', "pair_coeff 1 1", 2, "Pair_coeff must start with * *"),
        ("missing-file", "pair_style sus2/embedded absent.sus2", "pair_coeff * *", 2, "cannot open"),
        ("element-count", f'pair_style sus2/embedded "{model}" model_elements H', "pair_coeff * *", 2, "model_elements length"),
        ("metadata-conflict", f'pair_style sus2/embedded "{zbl_model}" model_elements O,H', "pair_coeff * *", 2, "disagrees with ZBL"),
        ("null-symbol", "pair_style sus2/embedded model_elements H,O", f'pair_coeff * * "{model}" H NULL', 2, "Unmapped pair_coeff symbol"),
    ]
    for name, style, coeff, count, expected in negative:
        directory = check.out/("input-negative-"+name)
        directory.mkdir()
        write_data(directory/"data.in", geometry, count)
        script = directory/"in.test"
        script.write_text('\n'.join(["units metal", "atom_style atomic", "boundary p p p", "newton on",
                                    "read_data data.in", "mass * 1.0", style, coeff, "run 0", ""]))
        result = check.process(shlex.split(args.mpiexec)+["-n","4",args.lammps,"-in",script],
                               directory, directory/"stdout.log", 20)
        passed = not result["timed_out"] and result["returncode"] != 0 and expected in result["stdout"]
        check.report["negative_checks"].append(dict(name=name,status="PASS" if passed else "FAIL",
              expected=expected,returncode=result["returncode"],timed_out=result["timed_out"]))
        check.save()
    failures = [r for r in check.report["runs"]+check.report["negative_checks"] if r["status"] != "PASS"]
    check.report.update(status="FAIL" if failures else "PASS", failures=len(failures))
    check.save()
    print(json.dumps({"status":check.report["status"],"runs":len(check.report["runs"]),
                      "negative_checks":len(check.report["negative_checks"]),"failures":failures}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
