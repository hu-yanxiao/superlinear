#!/usr/bin/env python3
"""Focused real-MPI species/storage regression against a frozen CPU binary.

Run inside a four-core allocation. No scheduler jobs are submitted. Model and
CFG fixtures are read-only; tolerances match the existing qualification suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

import numpy as np
from cpu_qualification import (Geometry, METAL_NKTV2P, Qualification, close,
                               read_cfg, read_dump, require, restricted,
                               validate_ranks, write_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "model", "cfg", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    for name in ("candidate", "reference", "model", "cfg", "out"):
        setattr(args, name, getattr(args, name).resolve())
    validate_ranks([4])
    params = SimpleNamespace(out=args.out, lammps=args.candidate, native=args.reference,
                             pair_style="sus2/embedded", direct_tolerance=1e-9,
                             table_tolerance=2e-6, tabstep=.001, timeout=240)
    runner = Qualification(params)
    report = dict(status="RUNNING", binary_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  cfg_sha256=hashlib.sha256(args.cfg.read_bytes()).hexdigest(),
                  tolerance=dict(direct=1e-9, table=2e-6), runs=[], comparisons=[])
    def save():
        (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    text = args.model.read_text()
    require(int(re.search(r"(?m)^\s*species_count\s*=\s*(\d+)", text)[1]) == 8,
            "This regression requires an eight-species model")
    cutoff = float(re.search(r"(?m)^\s*max_dist\s*=\s*(\S+)", text)[1])
    water = restricted(read_cfg(args.cfg))
    require(set(water.types.tolist()) == {0, 1, 2, 3, 6, 7}, "Expected FeCN6 six-species fixture")
    # All six species begin on rank zero; three ranks have no owned atoms.
    # A regular hexagon keeps every distinct-atom distance inside the radial
    # domain, while a large box removes periodic self images and other halos.
    length = 40*cutoff
    angles = np.arange(6)*np.pi/3
    xyz = np.column_stack([4*cutoff+.35*cutoff*np.cos(angles),
                           4*cutoff+.35*cutoff*np.sin(angles), np.full(6,4*cutoff)])
    sparse = Geometry(np.eye(3)*length, xyz, np.asarray([0,1,2,3,6,7]))

    def run(name, binary, geometry, mode, cap=None, lifecycle=False):
        directory = args.out/name
        directory.mkdir()
        write_data(directory/"system.data", geometry, 8)
        style = f'pair_style sus2/embedded "{args.model}" radial_mode {mode} tabstep 0.001 profile_every 1'
        if cap is not None:
            style += f" table_max_mb {cap}"
        commands = ["units metal", "atom_style atomic", "boundary p p p", "newton on",
                    "processors 4 1 1", "atom_modify map array sort 0 0.0", "read_data system.data",
                    "mass * 1.0", style, "pair_coeff * *", "neighbor 0.4 bin",
                    "neigh_modify every 1 delay 0 check no", "compute site all pe/atom pair",
                    "compute pairpressure all pressure NULL pair"]
        commands += [f"variable w{k} equal c_pairpressure[{k}]*vol/{METAL_NKTV2P:.17g}" for k in range(1,7)]
        commands += ["thermo 1", "thermo_style custom step pe "+" ".join(f"v_w{k}" for k in range(1,7)),
                     "thermo_modify format float %.17g"]
        labels = []
        def snapshot(label):
            labels.append(label)
            commands.extend([f"dump observation all custom 1 {label}.dump id type proc x y z fx fy fz c_site",
                             "dump_modify observation sort id format float %.17g first yes", "run 0",
                             'print "SPECIES_THERMO '+label+' $(pe:%.17g) '+
                             " ".join(f"$(v_w{k}:%.17g)" for k in range(1,7))+'"', "undump observation"])
        snapshot("initial")
        if lifecycle:
            snapshot("repeat")
            commands.append("set atom 1 type 5")
            snapshot("type_changed")
            commands.append(f"create_atoms 6 single {4*cutoff:.17g} {4*cutoff:.17g} {4*cutoff:.17g} units box")
            snapshot("created")
            commands.extend(["group inserted type 6", "delete_atoms group inserted compress no", "group inserted delete"])
            snapshot("deleted")
            commands.append("set atom 1 type 1")
            snapshot("restored")
            commands.append(f"displace_atoms all move {0.51*length:.17g} 0 0 units box")
            snapshot("migrated")
            # Reissue coeff into the same pair object; the previous presence
            # selection and knot state must be discarded before initialization.
            commands.append(f'pair_coeff * *')
            snapshot("reloaded")
        script = directory/"in.check"
        script.write_text("\n".join(commands)+"\n")
        result = runner.process(shlex.split(args.mpiexec)+["-n","4",binary,"-in",script,
                                "-log",directory/"lammps.log","-screen",directory/"screen.log"],
                                directory,directory/"launcher.log",240)
        require(not result["timed_out"] and result["returncode"] == 0, f"{name} failed; see {directory}")
        screen = (directory/"screen.log").read_text()
        thermos = {parts[1]: np.asarray([float(v) for v in parts[2:]])
                   for line in screen.splitlines() if line.startswith("SPECIES_THERMO ")
                   for parts in [line.split()]}
        observed = {}
        for label in labels:
            require(label in thermos and len(thermos[label]) == 7,"Missing E/virial output")
            atoms = read_dump(directory/(label+".dump"))
            observed[label] = dict(thermo=thermos[label], forces=np.column_stack([atoms[k] for k in ("fx","fy","fz")]),
                                   sites=atoms["c_site"], types=atoms["type"], owners=atoms["proc"])
            close(sum(atoms["c_site"]),thermos[label][0],2e-6,"site sum")
        record = dict(name=name, status="PASS", ranks=4, mode=mode, cap=cap, snapshots=labels)
        if lifecycle:
            require(len(set(observed["initial"]["owners"])) < 4,"Sparse fixture has no empty rank")
            require(np.any(observed["initial"]["owners"] != observed["migrated"]["owners"]),"No atoms migrated")
            require(set(observed["created"]["types"]) == {2,3,4,5,6,7,8},"Creation/type-change fixture failed")
            record.update(verified_empty_ranks=True,verified_migration=True)
        if binary == args.candidate and mode == "table":
            selections = [(int(a),int(b),int(c),int(d),alloc) for a,b,c,d,alloc in
                          re.findall(r"SUS2 radial table species: active=(\d+)/(\d+) pairs=(\d+)/(\d+) allocation=(\S+)",screen)]
            require(selections and selections[0][:4] == (6,8,36,64),"Missing six-of-eight global selection")
            if lifecycle:
                require(any(s[:4] == (7,8,49,64) for s in selections),"Created species absent from global selection")
                require(selections[-1][:4] == (6,8,36,64),"Delete/reload retained stale selection")
            counters = [tuple(map(int,values)) for values in re.findall(
                r"allrank_table_hits=(\d+) allrank_direct_fallbacks=(\d+) domain=(\d+) memory=(\d+) invalid=(\d+)",screen)]
            require(counters,"Missing all-rank radial counters")
            if cap in (None,"auto"):
                require(all(v[3] == 0 for v in counters),"Default/auto table memory fallback")
                require(all(v[4] == 0 and v[1] == v[2] for v in counters),
                        "Safe in-domain table request unexpectedly fell back")
                require(any(v[0]>0 for v in counters),"Table mode produced no table hits")
            if cap in ("0","0.001"):
                require(any(v[3]>0 for v in counters),"Explicit zero/small cap produced no memory fallback")
            record.update(selections=selections,allrank_counters=counters)
        report["runs"].append(record)
        save()
        return observed

    def compare(name, actual, expected, tolerance):
        require(actual.keys() == expected.keys(),"Snapshot labels differ")
        errors = {}
        for label in actual:
            errors[label] = {key:close(actual[label][key],expected[label][key],tolerance,name+" "+label+" "+key)
                             for key in ("thermo","forces","sites")}
            require(np.array_equal(actual[label]["types"],expected[label]["types"]),"Native model type indices changed")
        report["comparisons"].append(dict(name=name,status="PASS",tolerance=tolerance,errors=errors))
        save()

    try:
        old = run("fecn6-reference-direct",args.reference,water,"direct")
        direct = run("fecn6-candidate-direct",args.candidate,water,"direct")
        compare("fecn6 direct/frozen",direct,old,1e-9)
        table = run("fecn6-candidate-table",args.candidate,water,"table")
        compare("fecn6 table/direct",table,direct,2e-6)
        old = run("lifecycle-reference-direct",args.reference,sparse,"direct",lifecycle=True)
        direct = run("lifecycle-candidate-direct",args.candidate,sparse,"direct",lifecycle=True)
        compare("lifecycle direct/frozen",direct,old,1e-9)
        for label,cap in (("auto","auto"),("zero","0"),("small","0.001")):
            table = run("lifecycle-candidate-"+label,args.candidate,sparse,"table",cap,lifecycle=True)
            compare("lifecycle "+label+"/direct",table,direct,2e-6 if label=="auto" else 1e-9)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL",error=str(error))
    save()
    print(json.dumps(dict(status=report["status"],runs=len(report["runs"]),report=str(args.out/"report.json"))))
    return report["status"] != "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
