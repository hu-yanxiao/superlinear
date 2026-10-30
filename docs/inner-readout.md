# Optional inner energy readout

Set `model.readout.inner: true` or pass `--inner-readout` to add a zero-initialized constant and inner scalar energy head. The default is off. Coefficients are saved as `[b0, bq...]`; older `--two-layer-gate-readout` models retain their Q-only layout.

For the new head, linear initialization jointly solves the outer and inner energy coefficients, including energy, force and virial rows, while keeping center species amplitudes and all other parameters fixed. `--inner-readout-linear-init` requests this solve on a trained checkpoint. The old Q-only option retains its readout-only behavior. The constant contributes no force or virial; dependent constant columns need not have unique fitted coefficients.

Supported models are shared Scalar Legacy/Effect models (including ranked outer models) and Adaptive radial coefficient models, with full or shared-n projections and linear or tanh activation. Pure radial matrices and species readouts are rejected when enabling the head. Enabling the head does not change the existing center species scale.

Adaptive keeps its fixed PyTorch training profile: default training does not perform a legacy linear warmup or periodic refit. To explicitly fit the outer and inner heads of a complete checkpoint, use `--inner-readout-linear-init --max-iter=0 --std-weight=0 --stdd-weight=0` (or `initialization.inner_readout_linear: true` in YAML). All nonlinear coefficients, including elemental `t`, remain fixed; this is not a bilinear solve. The profile retains its prescribed head ridge regularization, so a synthetic recovery check should account for its small regularization bias. Do not pass legacy profile overrides such as `--radial-smooth`, `--force-loss`, or `--scale-by-force` for Adaptive. Enabling a zero head and resuming without this option preserves the normal training flow.

When the joint invariant head is enabled, the frozen solve includes its linear coefficients in the same equation system.

Regression checks (same tiny two-element dataset for serial or MPI):

```sh
python3 dev_test/inner_readout_solver_check.py
python3 dev_test/inner_readout_solver_check.py --adaptive
python3 dev_test/inner_readout_solver_check.py --adaptive --joint --projection shared-n --activation tanh --mpi-command 'mpirun -n 2'
```

The checks cover same-data E/F/virial recovery, elemental scale preservation, constant zero force/virial, default Adaptive resume without refit, and force plus lattice finite differences for every linear design column. CFG output precision and the prescribed ridge limit recovery precision.
