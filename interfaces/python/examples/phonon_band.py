#!/usr/bin/env python3
"""Native Superlinear forces -> phonopy FC2 and phonon band, with force resume."""
import argparse
import json
import os
from pathlib import Path

# Set these BEFORE importing NumPy/ASE/native code. Override explicitly in the
# scheduler environment if the force stage and phonopy stage need other limits.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--species", nargs="+", required=True, help="FULL species list in model type-index order")
    parser.add_argument("--structure", type=Path, required=True, help="one fully periodic ASE-readable unit cell, preferably primitive")
    parser.add_argument("--output", type=Path, required=True, help="new directory, or matching directory with --resume")
    parser.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 3], metavar=("N1", "N2", "N3"))
    parser.add_argument("--displacement", type=float, default=.01, help="Angstrom")
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--band-path", help="ASE special-point path, e.g. GXWKGLUWLK,UX for diamond/fcc")
    paths.add_argument("--path-json", type=Path, help='{"paths":[[[qx,qy,qz],...],...],"labels":[["G","X",...],...]}')
    parser.add_argument("--npoints", type=int, default=51, help="points per line segment")
    parser.add_argument("--stability-mesh", nargs=3, type=int, default=[5, 5, 5])
    parser.add_argument("--primitive-matrix", choices=["P", "auto", "F", "I", "A", "C", "R"], default="P")
    parser.add_argument("--symmetrization", choices=["traditional", "symfc", "none"], default="traditional")
    parser.add_argument("--symprec", type=float, default=1e-5)
    parser.add_argument("--imaginary-tolerance", type=float, default=.1, help="THz; below its negative is classified unstable")
    parser.add_argument("--keep-force-drift", action="store_true")
    parser.add_argument("--keep-reference-forces", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="independent spawned native force processes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="generate/count displacements; no forces")
    parser.add_argument("--max-displacements", type=int, default=50000)
    args = parser.parse_args()
    from ase.io import read
    from pysuperlinear.phonons import run_band_structure
    frames = read(args.structure, index=":")
    if len(frames) != 1:
        parser.error("structure must contain exactly one unit cell")
    band_path = json.loads(args.path_json.read_text()) if args.path_json else args.band_path
    result = run_band_structure(frames[0], args.model, args.species, args.output,
        supercell=args.supercell, displacement=args.displacement, band_path=band_path,
        npoints=args.npoints, stability_mesh=args.stability_mesh, primitive_matrix=args.primitive_matrix,
        symmetrization=args.symmetrization, symprec=args.symprec, imaginary_tolerance=args.imaginary_tolerance,
        subtract_reference=not args.keep_reference_forces, remove_drift=not args.keep_force_drift,
        workers=args.workers, resume=args.resume, dry_run=args.dry_run, max_displacements=args.max_displacements)
    print(json.dumps({key: result[key] for key in ("status", "result", "plan", "convergence")}, indent=2))


if __name__ == "__main__":
    main()
