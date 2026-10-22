#!/usr/bin/env python3
"""Native Superlinear forces -> phono3py FC2/FC3 and intrinsic lattice kappa."""
import argparse
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--species", nargs="+", required=True, help="FULL model type-index species order")
    parser.add_argument("--structure", type=Path, required=True, help="one explicit periodic unit cell, preferably primitive")
    parser.add_argument("--output", type=Path, required=True, help="new directory, or matching directory with --resume")
    parser.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 3], help="FC3 replication of INPUT cell")
    parser.add_argument("--fc2-supercell", nargs=3, type=int, help="independent harmonic replication; default same as FC3")
    pairs = parser.add_mutually_exclusive_group(required=True)
    pairs.add_argument("--pair-cutoff", type=float, help="Angstrom displacement-pair cutoff; explicitly unconverged truncation")
    pairs.add_argument("--full-pairs", action="store_true", help="all displacement pairs in the finite supercell")
    parser.add_argument("--displacement", type=float, default=.01, help="Angstrom")
    parser.add_argument("--mesh", nargs=3, type=int, default=[11, 11, 11])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[300.])
    parser.add_argument("--lbte", action="store_true", help="direct LBTE instead of default RTA; can require large dense matrices")
    parser.add_argument("--isotope", action="store_true", help="natural-isotope scattering; preserves input atomic masses")
    parser.add_argument("--primitive-matrix", choices=["P", "auto", "F", "I", "A", "C", "R"], default="P")
    parser.add_argument("--symmetrization", choices=["traditional", "symfc", "none"], default="traditional")
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--imaginary-tolerance", type=float, default=.1, help="THz; substantial instability stops before kappa")
    parser.add_argument("--cutoff-frequency", type=float, default=1e-4, help="phono3py transport frequency cutoff, THz")
    parser.add_argument("--keep-force-drift", action="store_true")
    parser.add_argument("--keep-reference-forces", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="count/plan only, including FC memory estimates")
    parser.add_argument("--max-displacements", type=int, default=50000)
    args = parser.parse_args()
    from ase.io import read
    from pysuperlinear.phonons import run_thermal_conductivity
    frames = read(args.structure, index=":")
    if len(frames) != 1:
        parser.error("structure must contain exactly one unit cell")
    result = run_thermal_conductivity(frames[0], args.model, args.species, args.output,
        supercell=args.supercell, fc2_supercell=args.fc2_supercell, pair_cutoff=args.pair_cutoff,
        displacement=args.displacement, mesh=args.mesh, temperatures=args.temperatures, lbte=args.lbte, isotope=args.isotope,
        primitive_matrix=args.primitive_matrix, symmetrization=args.symmetrization, symprec=args.symprec,
        imaginary_tolerance=args.imaginary_tolerance, cutoff_frequency=args.cutoff_frequency,
        subtract_reference=not args.keep_reference_forces, remove_drift=not args.keep_force_drift,
        workers=args.workers, resume=args.resume, dry_run=args.dry_run, max_displacements=args.max_displacements)
    print(json.dumps({key: result[key] for key in ("status", "result", "plan", "convergence")}, indent=2))


if __name__ == "__main__":
    main()
