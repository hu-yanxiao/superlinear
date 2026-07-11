# Release3 independent L-BFGS optimizer

Date: 2026-07-11

## Scope

Release3 keeps dense BFGS with Wolfe line search as the default training path.
The new L-BFGS implementation is selected explicitly with
`--optimizer=lbfgs`. It is an independent optimizer, not a mode inside the
dense BFGS implementation. Gauss-Newton and Levenberg-Marquardt are reserved
for a later change.

This is a training-only feature. `lbfgs.cpp` is linked into the `mlp` target,
but not into `libinterface`, so the LAMMPS/MD interface is unchanged.

## Mathematical update

For two consecutive accepted points,

```text
s_k = x_(k+1) - x_k
y_k = grad f(x_(k+1)) - grad f(x_k)
```

Rejected line-search trials never create a curvature pair. With the positive
diagonal preconditioner `D`, the initial inverse-Hessian operator used by the
two-loop recursion is

```text
H_k^0 = gamma_k D.
```

After damping and accepting a pair, the scale for the next recursion is

```text
gamma_(k+1) = (s_k^T y_tilde_k) / (y_tilde_k^T D y_tilde_k).
```

The pair is checked with a scale-invariant relative curvature measure. If its
raw curvature is below the diagonal Powell threshold, it is replaced by

```text
y_tilde = theta y + (1 - theta) B_k^0 s,
B_k^0 = (H_k^0)^(-1),
theta = (1 - delta) (s^T B_k^0 s) / (s^T B_k^0 s - s^T y).
```

This enforces `s^T y_tilde >= delta s^T B_k^0 s`. A pair is stored only when
the damped curvature remains finite, positive, and above the configured
relative threshold.

## Line search and state rules

- L-BFGS supports total-loss value-only Armijo line search.
- The expensive gradient is evaluated only after a value trial passes Armijo.
- The gradient evaluation's total loss is authoritative and is checked again
  before the gradient is accepted. A cache/value mismatch is backtracked.
- A newly accepted curvature pair changes the search direction and resets its
  Armijo trial step to the configured initial value (default `1`). A reduced
  step from the previous direction is not reused for the new L-BFGS direction.
- External coefficient changes, TrainLinear refreshes, coordinate-mask
  changes, diagonal-preconditioner changes, and recovery restarts invalidate
  all curvature history.
- Per-run diagnostics are reset at each `MTPR_trainer::Train()` call.
- The L-BFGS history is replicated on active MPI ranks. Objective and gradient
  reductions are broadcast before every optimizer transition.
- Training-loop collectives use the active training communicator, including
  runs with more MPI ranks than local configurations.

## Options

```text
--optimizer=lbfgs
--optimizer-line-search=armijo-value
--lbfgs-history=20
--lbfgs-curvature-tol=1e-10
--lbfgs-damping=0.2
--lbfgs-armijo-c1=1e-4
--lbfgs-initial-step=1
--lbfgs-step-growth=1.5
```

The generic `--optimizer-conv-tol` and `--optimizer-trace-file` options have
legacy `--bfgs-*` aliases. Existing commands without `--optimizer` retain the
historical BFGS/Wolfe defaults.

## Verification entry points

- Core tests are registered in `dev_src/mlp/dev_self_test.cpp`.
- End-to-end serial and MPI smoke coverage is in
  `dev_test/sh_release3_lbfgs_optimizer_smoke.sh`.
- The smoke test checks accepted-loss monotonicity, finite diagnostics,
  history limits, serial/two-rank agreement, and the inactive-rank path.

## 2026-07-11 server verification

The validated MPI binary was built in
`/work/phy-weigw/20260321_Test/SUS2-SH-release3-lbfgs-work-codex`.
Its SHA-256 is
`7a4a91dc4480431c47c90d25036bd137d4da98853ea3a2e727c51cc55e60bf31`.

- The core scaled-quadratic probe converged from `320.68` to
  `4.8328182851781155e-28` with four accepted curvature pairs.
- The release3 smoke passed in serial, two-active-rank, and
  three-rank/two-configuration modes. Accepted total losses were finite and
  monotone, and the active MPI runs agreed on accepted losses.
- A fixed-start default-BFGS regression against the pre-change release3 binary
  had the same 18 trace fields and 1223 model numeric fields. The maximum
  coefficient difference was `9.03e-11`; this is the last-bit change from
  relinking with Intel IPO, not an optimizer-path change.
- The monolithic `self-test` command could not run to the new tests in this
  server checkout because its existing first test fixture
  `configurations/reading_test.cfg` is absent. The focused core and end-to-end
  tests above were run instead.

The fixed-start optimizer benchmark used 40 MPI ranks, 128 configurations,
the same trained coupled one-gate model (SHA-256
`81db4f1a8e2eca60d914f1d4b3bac89ed00f1163df3145f4c27e4c02ec173d19`),
the same total-loss terms, and value-only Armijo for both methods. Results are
stored under
`/work/phy-weigw/hyx/xxx-b/6.17/6.24/update1/update2/update3/update4/update5/release3/optimizer-ab-20260711`.

| Optimizer | Accepted steps | Runtime | Final total loss |
| --- | ---: | ---: | ---: |
| Dense BFGS | 8 | 151 s | 0.7640437187083877 |
| L-BFGS, final step-reset logic | 8 | 72 s | 0.7854378435683030 |
| L-BFGS, approximately equal wall time | 24 | 162 s | 0.7854291136742261 |

L-BFGS therefore reduced accepted-step cost on this case, but did not match
dense BFGS convergence per step or per wall time. It remains an opt-in
memory-scalable alternative; dense BFGS remains the release3 default. The
first L-BFGS benchmark also exposed that reusing a heavily backtracked step for
a newly formed L-BFGS direction froze progress. The final implementation
resets the trial step after every accepted curvature pair, and the focused
benchmark above uses that corrected behavior.
