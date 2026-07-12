# Gate linear-component mu-chain correction (2026-07-12)

## Scope

This correction is carried by both the `release2` and `release3` branches.
It affects the energy/force/stress design matrix used by `TrainLinear` for
gated SUS2-SH models. Direct prediction and the LAPACK linear solver are not
changed.

`release2` uses the corrected one-gate chain. `release3` uses the same
one-gate chain and also preserves each linear column while propagating through
the second gate.

## Correct contraction

For atom `i`, outer linear column `a`, radial channel `mu`, gate scalar `q`,
and coordinate `x`, define

\[
A_{ia\mu}=\frac{\partial B_{ia}^{\mathrm{outer}}}{\partial h_{i\mu}},
\qquad
W_{\mu q}=\frac{\partial h_{i\mu}}{\partial s_{iq}},
\qquad
D_{iqx}=\frac{\partial s_{iq}}{\partial x}.
\]

The gate contribution to a linear-component derivative is

\[
\frac{\partial B_{ia}^{\mathrm{outer}}}{\partial x}\bigg|_{\mathrm{gate}}
=
\sum_{\mu,q} A_{ia\mu} W_{\mu q} D_{iqx}.
\]

The old path summed `A` over `mu` before applying the gate derivative. That is
only valid for a degenerate gate projection and gives an incorrect force and
stress design matrix for general `mu`-dependent gate weights. The corrected
path retains the full `A[i,a,mu]` tensor. For one gate, the exact contraction
is reassociated as `A * (W * D)` to avoid materializing the larger
`a x q` intermediate.

For the second gate in `release3`, the column-specific adjoint field is kept
through both gate reverse paths before coordinate derivatives are accumulated.

## Regression evidence

The deterministic regression is:

```bash
bash dev_test/sh_gate_linear_mu_components_check.sh
```

It uses a distorted configuration and non-degenerate `mu`-dependent gate,
additive, and type coefficients. The old and corrected maximum errors were:

| Branch | Old absolute error | Corrected absolute error |
| --- | ---: | ---: |
| `release2` | `6.42787530939e-05` | `7.54184761517e-11` |
| `release3` | `6.41298384134e-05` | `1.12326140143e-10` |

Additional server checks passed:

- `release2`: full loss-gradient matrix, one-gate force finite difference,
  and shared-radial compatibility.
- `release3`: multi-layer gate smoke including linear-component finite
  difference and loss directional derivative, one-gate force finite
  difference, and shared-radial compatibility.

## Server artifacts

- `release2` training binary SHA-256:
  `a4c41dac98657820d6f34b78cacfbb2b5701dd2927de766db3faca7437e3eab2`
- `release3` training binary SHA-256:
  `bea8ca1af259db2a49bb4b42a30aa59c22231d8f404e21b13d7c23ba709fe67a`
- Pre-fix binaries and changed source files are retained under
  `.codex_tmp/backups/gate-linear-mu-chain-20260712-pre-fix` in each server
  source tree.

This is a training-code correction. The existing release2 LAMMPS CPU and
Kokkos binaries were not rebuilt or replaced.
