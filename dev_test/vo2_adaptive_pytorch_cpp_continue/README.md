# Final PyTorch weights to C++ continuation: VO2 9.1

Server-only continuation of the completed epoch-500 adaptive-radial O2/O4/O8
`current.best.mtp` files, not the older Sep 6 inference snapshots.
Each independent LSF job requests queue 33, 200 MPI ranks (5 hosts x 40), and
5000 accepted optimizer steps maximum. No GPU or model initialization is used.

Source models: `/work/phy-weigw/hyx/vo2/9.1-adaptive-radial-pytorch-20260906-be41881/O{2,4,8}`.
Fresh destination: `/work/phy-weigw/hyx/vo2/9.1-adaptive-pytorch-to-cpp-5000-33-200c-20260907/O{2,4,8}`.
The preparation script pins the final model, full 9.1 data, and binary hashes;
it refuses to overwrite the destination and never changes the PyTorch runs.

The optimized CPU executable is the deployed `cb685aa` build, SHA256
`37657b894abb7321926cf5ef37f66f19761ac6a55bdd0cd39814981d9a65d888`.
This campaign changes test scripts only, not compiled model code. Its measured
2.2-2.5x EFS speedup must not be represented as a measured training-loop speedup.
Full parameter gradients continue to use their required full derivative cache.

Continuation uses BFGS/Wolfe, exact signed radial weight-norm coordinates, and
EFS-GN coordinate normalization (4 probes, floor 1e-6). These charts preserve
the loaded physical coefficients and fold exports into the existing MTP format.
Adam/EMA optimizer history does not transfer to BFGS. No balanced/random init,
physical sign repair, Rescale, head-only prefit, periodic linear solve, or
cutoff/topology change is requested. The existing optimizer logs its initial
physical-coordinate roundtrip and pre-objective parameter delta.

Keep PyTorch's E/F/S weights 1/0.1/0.01, vibrations weighting, radial smoothing
1e-6 (128 points), recommended scalar/gate L2, and gate-x L2=0. Legacy CPU
STD/STDD penalties are explicitly zero because they were absent from this
PyTorch training objective. `--optimizer-conv-tol=0` prevents a positive-small-
improvement plateau from stopping before the requested budget; numerical
failure or other existing terminal conditions still terminate safely.

Each allocated job first verifies old-oracle versus optimized E/F/stress on
three frozen 9.1 configurations and a zero-step model roundtrip, then starts
the full-data 5000-step run from the untouched input. `current.mtp`, `best.mtp`,
and `trained.mtp` are separate native-model outputs; accepted steps are recorded
in `trace.csv` and `training_result.txt`. An early successful return is not
equivalent to completing all 5000 steps: check the actual accepted count.

After syncing the committed scripts, run `prepare.sh FULL_SCRIPT_COMMIT` on
the server, then submit `bsub -J arad_ptcpp_O2 < train.lsf` from O2 and similarly
for O4/O8. Never resubmit into a run that has already started. Retain submission
receipts and resolve scheduler status before retrying an ambiguous submission.
