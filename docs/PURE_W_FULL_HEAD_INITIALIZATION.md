# Full-objective pure-W head initialization and fixed dense coordinates

These are two separate opt-in controls on the balanced pure-W protocol. Existing
defaults, the saved physical model format, and inference kernels are unchanged.
The purpose is to test whether matching the initial head fit to the actual
training objective and accounting for correlated head features improves
optimization. Neither intervention guarantees better final accuracy.

## Conditional physical objective

Let `h` denote the stored scalar-head coefficients. Hold W, the gate, radial
scaling, species-center prefactors and all other physical coefficients fixed.
Each energy, force, stress and site energy is affine in h. The physical head
column for site i is

    B[i,q] = center_linear[type_i] * linear_mults[q] * scalar_basis[i,q].

Both prefactors are necessary: the derivative is with respect to the stored h,
not an auxiliary linear-fit coefficient. A weighted L2 row is
`w*(b^T h + c - y)^2`. Accumulating all such rows gives

    J(h) = h^T A h - 2*r^T h + c0 + J_fixed,
    grad_h J = 2*(A h-r),
    A = sum_rows w*b*b^T + physical_head_ridge.

For std, use rows `B[i,:] - mean_type(B)` and weights
`std_weight*200/[N*(200+|F_ref[i]|^2)]`. For stdd, use the per-species mean
columns, reference total energy per atom and weight `stdd_weight`. The means
and their derivatives are included; no stop-gradient is applied to them.
The physical head ridge includes `linear_mults[q]^2` and exactly the same
configuration and head-count normalization as the nonlinear objective.

`--pure-w-head-solve=full-quadratic` first performs the same existing balanced
EFS-only fit, establishing the common species-center prefactors and the common
control model, then refines only the scalar head with a conditional solve
`A h=r`. The physical objective is not damped or
truncated. A non-positive-definite system is rejected; any physical ridge must
be requested explicitly through the existing head-L2 option. This changes the
initial physical head, while leaving its calibrated W and gate fixed.

The reference is the actual objective-gradient path used by Wolfe BFGS. The
historical value-only path uses 20 instead of 200 in std. This feature does not
change that historical behavior and must not be evaluated with Armijo-value.

## Independent optimizer-coordinate control

`--bfgs-coordinate-normalization=efs-head-full` builds a fixed full head metric
`H=2*A/N_configs`, including off-diagonal correlations, std/stdd and physical
head L2. With chart-only stabilization `delta`, factor

    H + delta*I = L L^T,
    z = L^T*(h-h_ref),
    h = h_ref + L^(-T)*z,
    grad_z = L^(-1)*grad_h.

This is an invertible coordinate change, not a change to model features or
physical regularization. The complete physical gradient is pulled back after
all penalties are included. No eigendirections or model channels are removed.
The positive diagonal shift is only a numerical floor for this chart and is
reported. The map stays fixed throughout the optimizer run; it is not
differentiated with respect to positions or W. Saved models remain physical h.

An identity inverse Hessian in z is an intentional head preconditioner. BFGS
trajectories may differ even though the initial physical coefficients, E/F/S,
objective and penalties are identical in a coordinate-only comparison.

## Controlled experiments

For each O in **1, 2, 4, 8**, use the same database, seed and other parameters:

| Control | `pure-w-head-solve` | `bfgs-coordinate-normalization` |
|---|---|---|
| Existing balanced reference | `efs` | `efs-head` |
| Initial head solve only | `full-quadratic` | `efs-head` |
| Fixed dense coordinates only | `efs` | `efs-head-full` |
| Both | `full-quadratic` | `efs-head-full` |

For example, after creating an incomplete pure-W seed:

```sh
bin/mlp-sus2 train seed.sus2 train.cfg \
  --pure-w-initialization=balanced --init-random-seed=1 \
  --pure-w-head-solve=full-quadratic \
  --bfgs-coordinate-normalization=efs-head-full \
  --radial-parameterization=weight-norm --bfgs-line-search=wolfe \
  --scalar-head-l2=1e-5 --std-weight=0.1 --max-iter=100
```

Report startup objective, accepted steps, objective-gradient evaluations,
line-search trials and time to matched error separately. `max-iter=0` compares
initialized physical models without running BFGS. Use held-out data before
claiming improved generalization; the VO2 training-only comparison is not that
test.

## Efficiency and boundaries

After the common EFS-only initialization, the full matrix is assembled once with symmetric BLAS updates and packed MPI
reduction. When both controls are active, its head metric is reused at startup
instead of repeating that full-objective data scan. The initial EFS-only scan
is still separate in this implementation. Initialization matrix work is quadratic in
the head count and its factorization is cubic; applying the triangular chart
is quadratic per optimizer call. There is no added MD/inference-time work.

This is an initialization/optimizer change, not the proposed basis-first
reordering of the runtime distance mixer. Existing runtime kernels stay as the
reference while initialization controls are tested independently.

Both new CLI controls require balanced fresh initialization, L2 force loss
and Wolfe BFGS. Full-quadratic refinement additionally rejects periodic
EFS-only `do-lin` replacement. The first dense
implementation requires equal nonzero species-center prefactors because the
existing E/F/S component representation is not species-resolved. Unequal
centers, unsupported residual heads or nonquadratic losses are rejected rather
than approximated. Complete checkpoints are not silently reinitialized.

## Regression checks

```sh
bash dev_test/full_head_coordinate_scaling_check.sh
python3 dev_test/sh_pure_w_full_head_init_check.py bin/mlp-sus2
bash dev_test/sh_scalar_head_quadratic_objective_check.sh MODEL.sus2 TWO_CONFIGS.cfg
```

The mathematical regression compares the assembled quadratic against the
independent actual `CalcObjectiveFunctionGrad` path at several physical heads,
then finite-differences that actual complete objective in dense coordinates.
It includes std, stdd, nonunit signed `linear_mults`, common nonunit center
prefactors, relative-force weights, force normalization and structure weighting.
The site-component test separately covers unequal species-center prefactors.
The tiny CLI regression covers all four O values and all four controls, exact
initialized-model parity across charts, W semantic prefixes, cache built/hit
boundaries, model roundtrip and the first accepted step.

For MPI, the CLI test accepts `SUS2_TEST_LAUNCHER='mpiexec -np 2'`. The C++
test driver accepts an object directory as its third argument and environment
variables for the matching compiler, compile/link flags and MPI runner. A clean
full rebuild is required after the trainer class layout changes; do not link
old objects that instantiate its inline constructors against the new header.
