# Adaptive-radial PyTorch training profile in native C++

This note documents the native C++ training profile used for adaptive-radial
superlinear models exported from the PyTorch implementation.  The purpose is to
make the C++ optimizer use the same fixed normalization and regularization
math as the current PyTorch training code while keeping the saved `.sus2` model
in the ordinary physical coefficient format.

The profile applies only when the loaded model uses adaptive radial
coefficients:

```text
w_jlon = sum_q U_lonq S_jq
b_jlon = c_lon + a_lon w_jlon
```

There is no activation in this adaptive readout.  Energy, force, and stress
loss weights remain user training-loss hyperparameters.  Regularization and
normalization controls are fixed by this profile, except for the legacy
`--std-weight` and `--stdd-weight` penalties.

## Fixed normalization contract

The C++ evaluator always stores and saves physical coefficients.  The PyTorch
profile builds an optimizer coordinate chart around those physical
coefficients; it does not change inference or add runtime metadata needed for
prediction.

For every training shard, C++ accumulates:

```text
gate_q_sq  = sum_atoms S_jq^2
outer_p_sq = sum_atoms raw_outer_jp^2
force_sq   = sum_force_components F_ref^2
```

The global RMS scales are reduced across MPI ranks:

```text
gate_scale_q  = clamp(1 / rms(S_q), 1e-12, 1e12)
outer_scale_p = clamp(1 / rms(raw_outer_p), 1e-12, 1e12)
output_scale  = clamp(rms(F_ref_component), 1e-12, 1e6)
```

If an RMS is effectively zero, the corresponding scale is one.  Force targets
are required because `output_scale` is defined from the force-component RMS.

Calibration-sample boundary: native C++ uses all training configurations for
these statistics. PyTorch uses the same feature-RMS formulas but may cap the
gate/adaptive/outer calibration subset with `--normalization-max-configs`
(the formal launch templates use 2048; the CLI default is 256). PyTorch's
force-RMS output scale still uses all training configurations. Thus formula
alignment does not imply identical feature scales on a dataset larger than
the PyTorch cap. Fresh initial coefficients can differ statistically; for a
complete checkpoint this only changes the optimizer chart, not the loaded
physical energy/force function or saved regularization priors.

The scalar readout coordinate is scaled so that the C++ physical contribution
matches PyTorch:

```text
E_i scalar term = type_factor_i * sum_p coeff_p * linear_mult_p * raw_outer_ip

coord_scale_p = |linear_mult_p| / (output_scale * outer_scale_p)
```

Equivalently, a unit optimizer-coordinate perturbation corresponds to a
physical stored coefficient change of
`output_scale * outer_scale_p / |linear_mult_p|`.  Multiplication by
`linear_mult_p` in the evaluator then gives the same effective coefficient as
PyTorch export:

```text
physical_head_p = readout_output_scale * outer_scalar_scale_p * train_head_p
```

Adaptive amplitudes use the saved amplitude optimizer scale:

```text
a_physical_lon = scale_a_lon * a_optimizer_lon
```

Adaptive weights use the fixed gate feature scale:

```text
U_physical_lonq = scale_U_q * U_optimizer_lonq
```

For a fresh C++ initialization, `scale_U_q` is the raw gate RMS inverse used to
initialize symmetry-breaking `U`.  For a complete checkpoint, the physical
model is preserved and the saved adaptive optimizer-scale metadata remains the
regularization prior.

## Regularization contract

The adaptive profile fixes these native settings:

```text
radial_smooth        = 1e-6
radial_smooth_grid   = 128
scalar_head_l2       = 1e-5
adaptive_response_l2 = gate_full_l2 = 1e-4
gate_scalar_l2       = 0
gate_mix_l2          = 0
gate_x_l2            = 0
```

For `readout_mode = species`, native full nonlinear training also fixes the
additive `species_coeffs`/`h[type]` entries as a gauge anchor and leaves
`shift_coeffs` trainable.  This removes the otherwise exact additive redundancy
in `shift[type] + h[type]` while preserving the physical model and the complete
saved coefficient layout.

The scalar-head L2 penalty is applied to the physical exported readout:

```text
mean_p (coeff_p * linear_mult_p)^2
```

The adaptive folded-response L2 penalty is applied to the fixed normalized-gate
response, not to raw `U` alone:

```text
mean_lonq (a_lon * U_physical_lonq / prior_U_q)^2
```

This matches PyTorch's
`a * U_working * gate_scalar_scale / adaptive_radial_weight_optimizer_scales`
because PyTorch exports `U_physical = U_working * gate_scalar_scale`.
Here `prior_U_q` is the saved `adaptive_radial_weight_optimizer_scales`,
not the newly measured `scale_U_q` used for optimization coordinates on resume.
The prior must never be replaced by a new calibration when continuing a model.

The radial smoothness penalty uses the physical baseline radial rows and the
folded adaptive response rows:

```text
lambda * cutoff / ((L+1) O) *
  [ sum_lo c_lo^T G_lo c_lo
    + sum_loq response_loq^T G_lo response_loq ]
```

`G_lo` averages the midpoint-grid Gram matrix over all ordered element pairs
and grid points and includes the model's global radial scaling. No
species prefactors and no final scalar readout coefficients enter this
regularizer.

## Loss averaging and configurable loss weights

The profile keeps PyTorch's legacy loss averaging convention:

```text
energy:  mean_cfg [ energy_weight * (Delta E_cfg)^2 / N_cfg ]
force:   mean_cfg [ force_weight  * sum_atom,xyz (Delta F)^2 ]
stress:  mean_cfg [ stress_weight * ||Delta stress||_F^2 / N_cfg ]
```

Forces use component-wise squared error with `weighting=vibrations`.  The
profile rejects overrides of the normalization/loss-shape controls
`--weighting`, `--scale-by-force`, `--force-loss`, and
`--force-log-cosh-scale`; their adaptive values are fixed to PyTorch's
`vibrations`, zero force-normalization, and L2 force loss.

The E/F/stress weights are deliberately still configurable, for example:

```text
--energy-weight=1 --force-weight=0.01 --stress-weight=0.001
```

or another explicitly chosen training-loss weighting.  These weights are not
part of the frozen normalization/regularization profile.

## Fresh initialization

Fresh adaptive models use the balanced adaptive-radial initializer followed by
the PyTorch fixed-RMS profile.  The initializer:

1. computes raw gate scalar RMS values;
2. initializes distinct deterministic symmetry-breaking `U_lonq` values with
   row scale `1 / sqrt(Q)` and the raw-gate RMS inverse;
3. sets `a_lon` from the fixed adaptive relative RMS `0.03`;
4. rescales baseline radial rows and adaptive amplitudes by a common outer
   moment RMS factor;
5. fits composition-only atomic shifts with minimum-norm least squares:

```text
min_e0 || counts_cfg dot e0 - E_cfg ||_2
```

and stores:

```text
shift_coeff[type]   = e0[type] - 1
species_coeff[type] = 1
```

The fresh scalar head is Gaussian in normalized optimization coordinates,
with standard deviation `0.1 / sqrt(number_of_outer_scalars)`. C++ and PyTorch
do not share an identical random-number stream; this is not a bitwise replay
of PyTorch initialization.

Fresh C++ adaptive profile training is currently BFGS-only.  It skips legacy
Rescale, 75-step pretraining, head-only warmup/refits, random sign repair,
fixed atomic-energy penalties, and guard-compression penalties.

## Complete-checkpoint behavior

A complete adaptive `.sus2` checkpoint is not reinitialized.  The native trainer:

- preserves physical coefficients exactly;
- recomputes fixed-RMS optimizer coordinate scales from the supplied training
  data;
- keeps saved adaptive optimizer-scale metadata as the regularization prior;
- verifies the optimizer-coordinate round trip before the first proposal.

The recomputed optimizer chart changes only optimization coordinates and
gradient pullback.  It does not alter the saved model function unless the
optimizer later accepts a step.

## STD and STDD penalties

`--std-weight` and `--stdd-weight` remain selectable because they are native
legacy penalties not present in the PyTorch optimizer.  For adaptive PyTorch
profile runs, the STD value path and gradient path both use the force scale
constant `200`:

```text
std_atom_weight = 200 / (200 + |F_ref_i|^2)
```

This avoids the old non-adaptive inconsistency where the loss value used `20`
while the gradient used `200`.  The compatibility fix is scoped to adaptive
profile training.

## Unsupported or intentionally not claimed

ZBL is not supported by the current native adaptive PyTorch profile.  PyTorch
can compute force-RMS output scaling on residual targets after subtracting ZBL;
the current C++ profile rejects adaptive profile calibration when ZBL is
enabled.

The native profile is not an Adam, EMA, or PyTorch trajectory replay.  It
matches the fixed normalization, foldable coordinate maps, physical
regularization definitions, and mean-loss convention.  Optimizer dynamics,
checkpoint selection, and accepted-step history are native C++ BFGS behavior.

## Minimal launch templates

PyTorch entry point on the server:

```bash
/work/phy-weigw/app/superlinear/superlinear-pytorch ...
```

Native C++ adaptive profile continuation from a complete checkpoint:

```bash
mpiexec.hydra -np 200 /work/phy-weigw/app/superlinear/mlp-sus2 train input.sus2 train.cfg \
  --init-params=same \
  --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 \
  --std-weight=0 --stdd-weight=0 \
  --max-iter=5000 \
  --curr-pot-name=current.sus2 \
  --best-pot-name=best.sus2 \
  --trained-pot-name=trained.sus2
```

Native C++ fresh adaptive training from an incomplete adaptive model:

```bash
mpiexec.hydra -np 200 /work/phy-weigw/app/superlinear/mlp-sus2 train fresh_adaptive.sus2 train.cfg \
  --init-params=same \
  --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 \
  --std-weight=0 --stdd-weight=0 \
  --max-iter=5000 \
  --curr-pot-name=current.sus2 \
  --best-pot-name=best.sus2 \
  --trained-pot-name=trained.sus2
```

For adaptive models, the profile supplies the fixed radial smoothness, scalar
L2, folded adaptive response L2, signed radial weight norm, and
`pytorch-fixed-rms` coordinate chart automatically.  Do not add legacy
normalization or regularization overrides unless the adaptive profile code
explicitly exposes them.

## Server validation, 2026-09-08

Campaign: `/work/phy-weigw/20260321_Test/superlinear-alignment-20260908`.
The CPU program was built on phy-weigw, not on the developer laptop.
Profile validation job `4258472` passed:

- serial/MPI fresh initialization and complete zero-step continuation, including
  an empty MPI shard: maximum E/F/virial difference `0`;
- one accepted training step with two active MPI ranks versus serial: maximum
  E/F/virial difference `0`;
- 118 physical-coefficient finite differences, nonzero adaptive a/U directional
  derivatives, and signed radial weight-norm roundtrip/pullbacks;
- separate STD-only and STDD-only value/gradient and training-step checks;
- 13 retired profile-override rejection cases;
- C++ / PyTorch total loss `1.46024147252` / `1.4602414725237303`,
  regularization `3.09699303358e-5` / `3.0969930335812736e-5`.

The independent composition QR/SVD check passed rank-deficient VO2 and
full-rank examples. The existing adaptive radial regression also passed in
job `4258469`, covering K/L/LK, independent gate scaling, cached/recomputed
paths, model I/O, zero-a/zero-U, force/virial differences and mixed gradients.

## Selective native fine-tuning

For a complete adaptive checkpoint, native BFGS/L-BFGS can restrict the
optimizer coordinates with, for example:

```sh
mlp-sus2 train initial.sus2 train.cfg --init-params=same --optimizer=bfgs \
  --max-iter=10000 --energy-weight=10 --force-weight=1 --stress-weight=0.1 \
  --std-weight=0 --stdd-weight=0 \
  --trainable-groups=shift,readout,radial,adaptive_weight,adaptive_amplitude \
  --curr-pot-name=current.sus2 --trained-pot-name=trained.sus2
```

The option is an allowlist: `shift` selects element energy offsets; `readout`
selects the outer scalar head; `radial` selects active outer radial rows and
outer type factors; `adaptive_weight` selects U; `adaptive_amplitude` selects a.
Unlisted coefficients remain fixed. In the example this freezes all coordinate
scaling, inner gate radial/type/rank parameters, and the redundant species
readout bias. Consequently the pre-U gate scalar on a fixed geometry is fixed.
Unused outer rows and redundant type coefficients remain excluded even when
`radial` is requested; stored coefficient count is not active parameter count.
Joint heads and tensor-message coefficients, if present, are not selected by
these five names.

Omitting the option preserves the original all-active behavior. Empty entries,
unknown/duplicate groups, incomplete models, random initialization, legacy
`--fine-tune`, linear refits and residual stages are rejected. Native training
does not import PyTorch LR, clip, batch or optimizer state. Its full-batch
squared E/F/stress normalization and fixed physical regularizers are those
documented above, not a promise of PyTorch minibatch-MSE trajectory parity.
Complete-model calibration changes optimizer coordinates only, not the
starting physical coefficients. Every objective checks frozen physical
coefficients exactly against the optimizer-entry reference.

Regression: `dev_test/adaptive_selective_train_check.py BIN` (optionally
`--mpi-command 'mpiexec.hydra -n 2'`) exercises each group, the five-group
combination, MPI/serial EFS, idle ranks, invalid options and zero-step resume.
