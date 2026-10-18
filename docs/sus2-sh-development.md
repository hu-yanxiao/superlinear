# SUS2-SH Development Notes

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

## Current implementation target

SUS2-SH is developed by copying the SUS2-MLIP developer version and replacing the angular expression with real spherical harmonics. The original training flow remains the driver: `mlp-sus2 train`, E/F/S equations, nonlinear BFGS, linear sub-solve, model save/load, and MPI execution stay in the SUS2 code path.

## Exact-body mu two-layer gate

The `<historical-branch>` branch has an experimental exact-body gate for
SUS2-SH. The gate signal is per atom and per mu channel:

\[
h_{j,\mu}
=
\sum_{r:\operatorname{body\_order}(s_{q_r})=k_\mu+1}
w_r s_{q_r}(j)
\]

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
\left[1+A\tanh(a_{z_j,\mu}h_{j,\mu})\right]
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

This is `two_layer_gate_site_mode = neighbor`. The optional
`two_layer_gate_site_mode = double` also applies the center factor
\(g_{i,\mu}=1+A\tanh(a_{z_i,\mu}h_{i,\mu})\):

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
g_{i,\mu}g_{j,\mu}R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

In code, `k` is zero-based, so the exact scalar body order is `k_internal + 2`.
The full branch specification, server path, binary path, and verification list
are recorded in `docs/sus2-sh-mu-body-order-gate.md`.

The model initializer accepts:

```bash
--two-layer-gate
```

When enabled, `init-sh` writes `sh_scalar_info` plus explicit gate metadata:

```text
two_layer_gate_enabled = true
two_layer_gate_mode = mu-body-order
two_layer_gate_body_order_max = <int>
two_layer_gate_include_one_body = false
two_layer_gate_site_mode = neighbor|double
two_layer_gate_weight_count = <count>
two_layer_gate_scalar_indices = {...}
two_layer_gate_weights = {...}
```

The gate scalar set is selected automatically from exact body orders
`2..k_max+1`. `--two-layer-gate-body-order` is rejected in this branch, and
`--body-order` must be at least `--k-max + 1`. The only new trainable
coefficients are the shared `w_q` gate weights. They are stored after the
radial/scaling nonlinear block and before the mirrored linear coefficients in
`regression_coeffs`. They initialize to zero, so \(1+A\tanh(0)=1\) and
`calc-efs` reduces to ordinary SH until a gate weight becomes nonzero.

Implementation notes:

- The forward path first computes unmodulated first-layer gate scalars for all
  atoms and all mu channels with the ordinary `r_c` neighbor list, then
  evaluates the final SH moments using the neighbor scale
  `1 + A*tanh(a*h_{j,mu})`.
- Double-site mode reuses the same per-atom per-mu gate signals and multiplies
  the center and neighbor factors in the final SH edge pass.
- The force and training-gradient paths are configuration-level because a site
  energy centered at atom `a` depends on atom `c` when `c` is in the first-layer
  shell of neighbor `b`. This gives an effective mathematical radius `2*r_c`
  without building a physical `2*r_c` neighbor list.
- Reverse mode accumulates `dL/dh_{j,mu}` from final edges, groups those
  adjoints by the exact body order of each mu channel, then backpropagates only
  through the matching first-layer scalar bucket. The `w_q` gradients are
  accumulated even when all `w_q` start at zero, which is required for the gate
  to leave the zero state during training.
- Force-loss gradients include both gate-chain terms: the direct
  `d Phi_q^(0) / dx` contribution to `w_q`, and the mixed final-layer tangent
  contribution `dA_b / dw_q`. The direct gate scalar directional derivative is
  evaluated as a basic-moment tangent followed by SH product-graph tangent
  propagation, avoiding a full per-neighbor/per-scalar derivative matrix.
- LAMMPS and GPUMD should not use a real `2*r_c` halo for this feature. The
  intended interface design is to compute `h_{atom,mu}` on owned atoms,
  communicate the per-mu vector to ghosts, use it in the main pass,
  reverse-communicate `dE/dh_{atom,mu}`, then run the first-layer gate reverse
  pass on owners. The required interface buffers are two-layer-only:
  `gate_signal[nall][mu]`, `gate_adjoint[nall][mu]`, and a compact owned-atom
  first-layer gate-moment cache. The gate is neighbor centered, not pair
  symmetric; final edge `a-b` uses the mu-dependent signal of atom `b`, while
  edge `b-a` uses the mu-dependent signal of atom `a`.

Validation added:

```bash
bash dev_test/sh_two_layer_gate_mu_body_order_check.sh
bash dev_test/sh_two_layer_gate_double_mode_check.sh
bash dev_test/sh_two_layer_gate_init_check.sh ./bin/mlp-sus2
bash dev_test/sh_two_layer_gate_zero_compat_check.sh ./bin/mlp-sus2
bash dev_test/sh_two_layer_gate_forward_check.sh ./bin/mlp-sus2
bash dev_test/sh_two_layer_gate_force_fd_check.sh ./bin/mlp-sus2
bash dev_test/sh_two_layer_gate_train_weight_check.sh ./bin/mlp-sus2
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh ./bin/mlp-sus2
LAMMPS_BIN=<historical-path> \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
GATE_SITE_MODE=double \
LAMMPS_BIN=<historical-path> \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
```

Local serial build and the focused gate tests pass with the developer build
command:

```bash
make mlp -j2 USE_MPI=0 CXX_EXE=g++ CC_EXE=gcc FC_EXE=true \
  CXXFLAGS='-std=c++11 -O0 -DMLIP_DEV' CPPFLAGS='-O0 -I./cblas' \
  LDFLAGS='-framework Accelerate' TARGET_PRERQ=
```

Remote server validation was run in:

```text
<historical-path>
```

The MPI build must include `-DMLIP_MPI`; `USE_MPI=1` selects the MPI object
directory and compiler wrapper but does not add the macro by itself. The server
build command used for validation is:

```bash
make mlp -j8 USE_MPI=1 CXX_EXE=mpicxx CC_EXE=mpicc FC_EXE=true \
  CXXFLAGS='-std=c++11 -O2 -DMLIP_DEV -DMLIP_MPI' \
  CPPFLAGS='-O2 -I./cblas' TARGET_PRERQ=
```

The MPI build completed and the same six gate scripts passed against:

```text
<historical-path>
```

Additional derivative checks on the previous nonzero-gate failure model
`archived_two_layer_gate_500x100/gate_100_p.sus2`:

```text
gate force-loss gradient block: checked_coeffs=54, worst_index=-1
EFS finite difference: force_abs_err=7.68e-7, stress_abs_err=3.85e-7
```

Short 500-configuration, 100-step benchmark:

```text
source data: /work/phy-weigw/hyx/200w/14w.cfg
MPI ranks  : 120
subset     : first 500 configurations, 83000 atoms
```

Both runs used the same final SH topology:

```text
species-count=4, l-max=4, k-max=4, body-order=6.5,
body-l-max=4,4,4,2,1, cutoff=6.0,
radial-basis-size=10, radial-basis-type=RBLaguerre_log1p,
energy-weight=1, force-weight=0.01, stress-weight=0.001,
do-samp, init-params=same, do-lin, do-lin-rescale,
do-lin-steps=2000, do-lin-freq=50, max-iter=100,
scal-range=3.5,1.5
```

The two-layer run additionally used:

```text
--two-layer-gate --two-layer-gate-body-order=3
```

Result:

| model | runtime | Energy MAE | Force MAE | status |
|---|---:|---:|---:|---|
| plain SH | 334 s | 4.488 meV/atom | 205.006 meV/A | completed `max-iter=100` |
| two-layer gate, pre-fix | 720 s | 7.659 meV/atom | 351.241 meV/A | line search stopped at formal step 17 |
| two-layer gate, fixed | 1737 s | 5.501 meV/atom | 208.771 meV/A | completed `max-iter=100` |
| two-layer gate, stable chainfix | 1386 s | 5.243 meV/atom | 241.660 meV/A | completed `max-iter=100` |
| two-layer gate, directed-edge cache | 613 s | 5.160 meV/atom | 227.230 meV/A | completed `max-iter=100` |

Benchmark paths:

```text
plain/pre-fix: <historical-path>
fixed run    : <historical-path>
fixed job id : 3740179
stable chainfix: <historical-path>
directed-edge cache: <historical-path>
directed-edge job id: 3740785
```

The pre-fix gate weights did move from zero, reaching roughly `1.8e-1` max
absolute magnitude before the run stopped. The failure was traced to the
training gradient, not to `CalcEFS`: finite differences showed force and stress
predictions were consistent, while the force-loss gradient for gate weights was
not. After adding the missing gate-chain terms, the fixed two-layer run
completed 100 formal steps. It is now close to the plain SH force MAE on this
small subset but remains about 5.2x slower than the plain run, so the next
optimization target is reducing repeated first-layer gate reverse work and
adding gate-specific scale control or staged training if larger runs show
instability.

2026-06-01 follow-up notes:

- The step-17 line-search failure was traced to incomplete chain terms in the
  training gradient path, not to `CalcEFS` prediction. In particular, the
  linear-component force/stress path must include the two-layer gate chain
  correction so `TrainLinear` solves the same quadratic that the nonlinear
  objective differentiates.
- After `do-lin`, the BFGS state should sync the current coefficient vector but
  should not reset the accumulated Hessian approximation. Resetting the Hessian
  discards useful curvature memory and was not needed after the gradient fix.
- Acceleration work is constrained to paths introduced by the two-layer gate.
  Do not claim speedups from generic single-layer SH work such as common
  `TrainLinear`, common BFGS, or generic SH accumulation unless the code is
  explicitly gated to two-layer-only execution.
- Accepted two-layer-only cache: `PrepareTwoLayerGateValues` now keeps selected
  gate scalar values and the corresponding full first-layer moment vector per
  atom. The cache is reused by the gate-weight gradient and by two-layer
  weighted scalar derivative paths. A trial that also forced reuse in the gate
  directional-derivative path was measured and rejected because it was slower.

Server verification after the accepted cache used:

```text
<historical-path>
<historical-path>
```

Key checker results:

```text
check-two-layer-gate-fastpath-dev: checked_values=648, checked_derivatives=90234,
  value_abs_err=0, der_abs_err=0, weighted_der_abs_err=8.88e-16
check-linear-components-fd-dev: checked_components=3024, abs_err=1.60e-08
check-efs-fd-dev: force_abs_err=1.01e-06, stress_abs_err=3.64e-07
check-loss-gradient-dev gate-weight block: checked_coeffs=54, abs_err=0
```

One-step 120-rank profile comparison on the same 500-configuration subset:

| profile | path suffix | avg_us_total | avg_us_moment | avg_us_hvt_reverse | runtime_sec | status |
|---|---|---:|---:|---:|---:|---|
| original two-layer profile | `archived_two_layer_gate_profile_1step` | 258.569 | 169.974 | 16.3229 | 27 | reference |
| scalar cache only | `archived_two_layer_gate_profile_1step_gatecache` | 258.053 | 170.196 | 16.0840 | 24 | measured |
| accepted scalar+moment cache | `archived_two_layer_gate_profile_1step_momentcache` | 255.885 | 167.429 | 15.9410 | 24 | accepted |
| rejected directional-cache trial | `archived_two_layer_gate_profile_1step_directioncache` | 257.064 | 168.175 | 16.0600 | 26 | reverted |

Two-layer-specific profile timers were added with
`SUS2_SH_TWO_LAYER_PROFILE=1` and `SUS2_SH_TWO_LAYER_PROFILE_INTERVAL=<n>`.
They report only the new gate paths: gate preparation, gated final SH
accumulation, final site-energy adjoints, gate scalar directional derivatives,
mixed final-layer tangent gradients, gate-weight gradients, scalar-parameter
gate-chain gradients, and the two-layer linear-component correction. On
`archived_two_layer_gate_profile_1step_finalcache`, the gradient-side profile was:

```text
avg_us_total=242634
avg_us_prepare_gate=19178.3
avg_us_main_gated_grad=42874.8
avg_us_energy_adjoint=28670.7
avg_us_directional=40977.6
avg_us_tangent_grad=57159.1
avg_us_gate_weight=12.6762
avg_us_scalar_param=53759.1
```

An edge-primitive cache inside the two-layer-only final tangent gradient reuses
per-neighbor SH values, radial-basis values, and contracted radial
value/sigma/shift derivatives between the tangent seed pass and the coefficient
gradient pass. It does not change the generic single-layer SH accumulator. On
`archived_two_layer_gate_profile_1step_edgecache`, the same 1-step profile became:

```text
avg_us_total=225239
avg_us_prepare_gate=19196.7
avg_us_main_gated_grad=42970.1
avg_us_energy_adjoint=28633.7
avg_us_directional=40824.8
avg_us_tangent_grad=38785.8
avg_us_gate_weight=12.9078
avg_us_scalar_param=54814.2
runtime_sec=21
```

The edge cache reduced the final tangent-gradient stage by about 32% versus the
previous final-cache profile (`57159.1 -> 38785.8 us`) and reduced the
two-layer gradient-side total by about 7% (`242634 -> 225239 us`). The remaining
largest two-layer-only hotspots are scalar-parameter gate-chain gradients and
gate scalar directional derivatives.

The gate scalar directional derivative path was then fused so it computes
first-layer gate moment values and their directional tangent in one required
neighbor loop, followed by a single required-product pass that propagates both
the value and tangent. This removes the separate value-only gate pass from the
directional stage. On `archived_two_layer_gate_profile_1step_dirfuse`, the same
profile became:

```text
avg_us_total=204135
avg_us_prepare_gate=19173.1
avg_us_main_gated_grad=42933.7
avg_us_energy_adjoint=28749.6
avg_us_directional=22480.5
avg_us_tangent_grad=37146.7
avg_us_gate_weight=25.0266
avg_us_scalar_param=53625.4
runtime_sec=21
```

Compared with the edge-cache profile, this reduces `avg_us_directional` by about
45% (`40824.8 -> 22480.5 us`) and the two-layer gradient-side total by about 9%
(`225239 -> 204135 us`). Relative to the pre-edge-cache final-cache profile,
the combined two-layer-only optimizations reduce the measured gradient-side
total by about 16% (`242634 -> 204135 us`). The next high-value target is the
scalar-parameter stage: reusing only the final scalar directional outputs is not
mathematically sufficient; a correct reuse cache must include the full gate
moment tangent over the selected product-graph closure.

The scalar-parameter stage now reuses that full gate moment tangent. During the
directional pass, atoms with nonzero `energy_gate_adjoint` store the complete
`D_x m(v)` vector after required-product tangent propagation. The later
scalar-parameter pass scales this vector by `energy_gate_adjoint` and skips the
duplicated neighbor derivative pass plus product tangent propagation; the final
edge-level coefficient-gradient pass is still evaluated directly. This keeps the
chain rule exact while avoiding reuse of an insufficient scalar-only tangent. On
`archived_two_layer_gate_profile_1step_tangentcache`, the same profile became:

```text
avg_us_total=185140
avg_us_prepare_gate=19233.9
avg_us_main_gated_grad=43195.5
avg_us_energy_adjoint=28759.8
avg_us_directional=22904.8
avg_us_tangent_grad=37566.3
avg_us_gate_weight=13.8214
avg_us_scalar_param=33087.6
runtime_sec=20
```

Compared with the directional-fuse profile, this reduces `avg_us_scalar_param`
by about 38% (`53625.4 -> 33087.6 us`) and the two-layer gradient-side total by
about 9% (`204135 -> 185140 us`). Relative to the pre-edge-cache final-cache
profile, the accepted two-layer-only optimizations now reduce the measured
gradient-side total by about 24% (`242634 -> 185140 us`). A direct HVT-reverse
fusion trial inside scalar-param was also checked and profiled but rejected
because it was slightly slower (`204135 -> 205153 us` total).

A configuration-level directed-edge primitive cache was then added for
two-layer paths only. It is built once per `Configuration`/`Neighborhoods`
inside the two-layer evaluation/training entry points and is indexed by the
directed center-neighbor slot `i -> j`; the reverse edge is never reused because
the real SH parity and ordered center/outer element radial parameters differ.
The cache stores only shared edge primitives and radial contractions:
`Y_lm`, `dY_lm/dx`, radial basis values, radial basis derivatives with respect
to coordinate/sigma/shift channels, and contracted radial values/derivatives.
It does not reuse first-layer gate scalar products as final-layer scalar
products. The final gated SH moments are still rebuilt as
`B_i^(1)(a)=sum_j [1+f_j] edge_i(a,j)` and the outer product graph is still
propagated from those gated moments.

The cache is consumed only when a two-layer caller has provided an atom index
or set `active_two_layer_edge_cache_atom_index_`; otherwise the old SH code path
runs unchanged. It now feeds gate preparation, final gated SH accumulation,
final energy adjoints, gate directional derivatives, final-layer tangent
gradients, scalar-parameter gate-chain gradients, and the two-layer
force-chain correction. On
`archived_two_layer_gate_profile_1step_edgeprimitive`, the same 1-step profile
became:

```text
avg_us_total=108769
avg_us_prepare_gate=34997.7
avg_us_main_gated_grad=26715.8
avg_us_energy_adjoint=10295.6
avg_us_directional=3172.88
avg_us_tangent_grad=16030.8
avg_us_gate_weight=15.4981
avg_us_scalar_param=17059.2
runtime_sec=17
```

Compared with the previous tangent-cache profile, this reduces the measured
two-layer gradient-side total by about 41% (`185140 -> 108769 us`). Relative to
the pre-directed-cache final-cache profile, the cumulative accepted
two-layer-only optimizations reduce the gradient-side total by about 55%
(`242634 -> 108769 us`).

## Residual staged two-layer mode

This is a historical note for the previous direct-scale gate experiment. The
`<historical-branch>` branch rejects `--two-layer-residual` together with
`--two-layer-gate`, because that mode depended on legacy `two_layer_gate_bias`
and direct-scale metadata that are not part of the exact-body mu gate.

The older developer branch had an opt-in residual two-layer variant for staged
training:

```text
E = E0 + E1
E0_i = sum_s c_s^0 Phi_s^(0)(i) + one-body_i
g_j = b + sum_q w_q Phi_q^(0)(j)
B_i^(1)(k,l,m) = sum_j g_j R_{k,l}(r_ij) Y_lm(rhat_ij)
E1_i = sum_s c_s^1 Phi_s^(1)(i)
```

This mode is initialized with `--two-layer-residual` together with
`--two-layer-gate`. It uses direct neighbor scaling metadata:

```text
two_layer_residual_enabled = true
two_layer_gate_scale_mode = direct
two_layer_gate_bias = 1
```

So the final layer does not use the old literal `1 + f(n_j)` expression.
Instead, the scale is the direct scalar `g_j`; the fixed bias starts at 1 so
the B-stage residual linear solve has nonzero E1 columns while the trainable
inner weights `w_q` still initialize to zero. Old `--two-layer-gate` models
without residual metadata keep the legacy `1+f` scale and old coefficient
layout.

Perception radius:

- `E0` is an ordinary one-cutoff SH readout from unmodulated moments.
- `E1` is effectively two-cutoff because the outer site energy around center
  `i` uses `g_j`, and `g_j` depends on the one-cutoff environment around
  neighbor `j`.
- One-body terms are included only in `E0`.

The staged training CLI is:

```bash
--two-layer-residual-staged \
--stage-a-steps=<int> \
--stage-b-steps=<int> \
--stage-c-steps=<int>
```

Stage A optimizes the E0 nonlinear block, E0 scalar coefficients, and species
one-body coefficients. Stage B freezes E0 and optimizes the E1 residual block:
gate shared-radial coefficients, gate weights, and final scalar coefficients,
excluding species one-body columns. Stage C releases all active residual
parameters. A and B each begin with one linear solve projected onto that
stage's active block; inactive columns are subtracted into the right-hand side
using the current frozen coefficients, so the B-stage linear solve is a true
residual solve rather than an independent refit. During the subsequent BFGS
part of A/B/C, staged training disables the ordinary internal `do-lin` and
`do-lin-rescale` refreshes. This avoids repeatedly re-solving or re-scaling
the same block inside the stage and ensures C does not perform a simultaneous
full linear solve.

Implementation notes:

- The residual coefficient layout is
  `[base nonlinear][gate radial][gate weights][E0 scalar coeffs][E1 linear tail]`.
- `LinearEquationCount()` is larger than the saved E1 linear tail because the
  staged normal equations include extra E0 scalar columns.
- `CalcSHResidualSiteEnergyDers()` evaluates E0 by temporarily using the
  ordinary SH path with cached residual E0 coefficients, then evaluates E1 with
  the direct gated final moments. The temporary non-residual recursion remaps
  component-gradient offsets back to the residual layout; this is required
  because disabling residual mode changes `LinearCoeffOffset()`.
- In residual mode, scalar readouts are not multiplied by the center species
  linear coefficient. The temporary ordinary SH readout therefore sets the
  center-linear gauge to 1 for both E0 and E1 scalar paths, then adds the E0
  one-body term explicitly. This keeps direct E/F prediction, coefficient
  gradients, and staged linear columns on the same model even if a test model
  has non-unit `species_coeffs`.
- The gate-chain force/stress terms are still attached only to E1, since E0
  has no second-layer dependency.
- In E1-only stage B, the residual gradient wrapper skips outer base
  radial/type/scaling gradient accumulation because those coefficients are not
  active in that stage. The gate adjoint, gate radial, gate weights, and E1
  scalar-linear gradients are still accumulated and are covered by a staged
  active-only loss-gradient finite-difference check.
- Legacy two-layer tests are intentionally left unchanged and still exercise
  the old `1+f` mode.

Server validation used:

```text
<historical-path>
```

Build command:

```bash
make mlp -j8 USE_MPI=1 CXX_EXE=mpicxx CC_EXE=mpicc FC_EXE=true \
  CXXFLAGS='-std=c++11 -O2 -DMLIP_DEV -DMLIP_MPI' \
  CPPFLAGS='-O2 -I./cblas' TARGET_PRERQ=
```

Focused residual and legacy checks run on the server:

```bash
bash dev_test/sh_two_layer_residual_init_check.sh
bash dev_test/sh_two_layer_residual_forward_check.sh
bash dev_test/sh_two_layer_residual_staged_train_check.sh
bash dev_test/sh_two_layer_residual_force_fd_check.sh
bash dev_test/sh_two_layer_residual_loss_gradient_check.sh
bash dev_test/sh_two_layer_residual_readout_check.sh
bash dev_test/sh_two_layer_gate_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_init_check.sh
```

Additional legacy gate checks were also run:

```bash
bash dev_test/sh_two_layer_gate_forward_check.sh
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_gate_train_weight_check.sh
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
```

The residual direct-readout check with non-unit species coefficients reported
`energy_abs_err=1.42e-14` and force absolute error below `1e-18` between
direct E/F prediction and the current linear component readout. The residual
finite-difference force checker passed with max absolute error below
`1e-6 eV/A`. The residual loss-gradient checker passed in both full residual
mode and E1-stage active-only mode; the E1-stage check covered 66 active
coefficients and reported no failing coefficient block.

The 500-configuration, 100-step comparison used the first 500 structures from
`/work/phy-weigw/hyx/200w/5.31-new-44421/` and ran with 120 MPI ranks. The
plain reference run is:

```text
<historical-path>
plain_100.log
```

Plain run result:

```text
runtime_sec=316
Energy MAE = 4.746 meV/atom
Energy RMSE = 5.797 meV/atom
Force MAE  = 179.118 meV/A
Force RMSE = 251.379 meV/A
```

Important timing note: this plain run did not use `--skip-preinit`. Its log has
a pre-training block with 75 BFGS steps (`step=0..74`) ending at
`Pre-training ended`, then a separate formal block with 100 BFGS steps
(`step=0..99`). Therefore `runtime_sec=316` is the total wall time for
pre-training plus formal 100 steps, while the reported final MAE/RMSE values
are from the formal block's final step.

The residual staged run is:

```text
<historical-path>
job id: 3741968
residual_100.log
```

It uses A/B/C = `30/30/40` steps and the same final SH topology plus:

```text
--two-layer-gate --two-layer-gate-body-order=3 \
--two-layer-gate-shared-radial --two-layer-residual \
--two-layer-residual-staged --stage-a-steps=30 \
--stage-b-steps=30 --stage-c-steps=40
```

Residual 100-step result after staged-speed fixes:

```text
runtime_sec=313
Energy MAE = 5.869 meV/atom
Energy RMSE = 7.623 meV/atom
Force MAE  = 251.120 meV/A
Force RMSE = 348.672 meV/A
```

The log shows exactly two linear solves in the staged residual run:

```text
TrainLinear[residual stage A E0 solve] n=1008 full_n=2012
TrainLinear[residual stage B E1 residual solve] n=1004 full_n=2012
```

There are no `rescale bracket`, `rescale fine trial`, or `bfgs refresh step 0`
linear solves inside A/B, and no `TrainLinear` after stage C starts. The older
correctness-first residual run under
`archived_residual_staged_500x100_readoutfix` took `runtime_sec=995` with force
MAE `281.395 meV/A`; the current staged-speed run is therefore about `3.18x`
faster and has lower force error for this short 100-step comparison. The plain
SH reference remains more accurate at 100 formal steps. Do not use the `316s`
plain runtime as a pure formal-100-step speed baseline, because it includes the
75-step pre-training block; a fair pure-speed comparison should rerun the plain
case with `--skip-preinit` or report the two plain blocks separately.

Two-layer-only optimization notes from the read-only performance audit:

- The next highest-value C-stage path is reusing the two-layer edge/gate cache
  from forward loss evaluation when immediately accumulating gradients for the
  same configuration, neighborhoods, and coefficients.
- Residual full C still allocates and clears separate full-size E0 and E1
  temporary gradient vectors per site. A lower-risk improvement is to reuse
  member scratch buffers; a more invasive improvement is direct residual
  scatter into the final coefficient layout.
- The staged active-index normal equation can still be improved by generating
  only active residual rows while preserving the inactive frozen contribution
  as an aggregate RHS subtraction.
- Any future component-generation optimization must still subtract frozen
  inactive contributions into the RHS; otherwise B-stage residual solves would
  drop the fixed E0 model.

## Training model

The training `.sus2` keeps the ordinary SUS2 radial/scaling/linear parameters and adds `potential_tag = SUS2-SH`. For SH models, `alpha_index_basic` stores `{k, l, m}`. Internally this is converted to `mu = k * (l_max + 1) + l`, so the radial channel is still compatible with the SUS2 coefficient layout.

```text
B[k,l,m] = sum_j R_{k,l}(r_ij) Y_lm_real(rhat_ij)
```

The product graph stores real-basis Clebsch-Gordan contractions as `sh_products = {left, right, target, coeff}`. GPUMD must read these stored coefficients directly; they include the real tesseral basis phase convention used by training.

## l/k/body enumeration

Use `--body-l-max=body2,body3,body4,body5` for body-specific angular truncation through 5body, and `--body-l-max=body2,body3,body4,body5,body6` when `--body-order=6`. For example, `--l-max=3 --body-l-max=3,3,2,2,2` keeps 2-body and 3-body terms through `l=3`, while 4-body, 5body, and 6body terms use `l<=2`.

Factors are enumerated in canonical order with both `l1 >= l2 >= ...` and `k1 >= k2 >= ...`. Only even parity scalar paths are kept. The graph is built as the minimal shared DAG closure of the selected scalar targets; unused `(k,l,m)` channels and unused tensor-product intermediates are not generated.

The 6body path uses five SH factors and the rank-diagonal standard tree:

```text
A = (q0 x q1) -> L
B = (q2 x q3) -> L
C = (B  x q4) -> L
S = (A  x C ) -> 0
```

This is an intentional truncation of the full 2+3 tree. It preserves the scalar construction in a standard CG tree while reducing the number of 6body scalar paths by enforcing the same intermediate rank `L` on the pair and triple branches.

## l3k3 scalar counts

With `k=3`, `l=3`, even parity, and all body orders through 5:

- `--body-l-max=3,3,3,3`: 2-body 3, 3-body 24, 4-body 80, 5-body 615, total 722 scalars.
- `--body-l-max=3,3,2,2`: the raw enumeration has 317 candidate scalars. After removing CG paths with zero intermediate tensors, the active model has 261 scalar basis functions.
- `--body-order=6 --body-l-max=3,3,2,2,2`: the rank-diagonal 6body extension generates 621 active scalar basis functions in total. This keeps the 261 active 2body-through-5body scalars and adds the rank-diagonal 6body scalars after the same zero-path pruning.

Zero intermediate tensors can arise when identical tensor factors are coupled into an exchange-antisymmetric rank. For example, `(q10_l2_k2 x q10_l2_k2) -> L=1` is zero because swapping the two identical `l=2` factors gives the CG phase `(-1)^(2+2-1) = -1`. The generator must combine CG records before allocating nodes, and must not keep scalars depending on these zero tensors.

## GPUMD export plan

The export should include the header, channel table, coupling table, scalar table, and trained parameters. The coupling table should be generated from the same stored `sh_products` graph so GPUMD does not reconstruct a different CG convention.

Current SH support is implemented for `l <= 4`. The l=4 implementation uses the same real tesseral convention as the existing l<=3 path:

- `m = 0`: real `Y_l0`.
- `m > 0`: `sqrt(2) * (-1)^m * Re(Y_lm)`.
- `m < 0`: `sqrt(2) * (-1)^|m| * Im(Y_l|m|)`.

## Current Cu-Zr test run

Server project:

```text
<historical-path>
```

Training path:

```text
<historical-path>
```

Model initialization:

```bash
../bin/mlp-sus2 init-sh untrained_sus2sh_l3k3_b5_l3322.sus2 \
  --species-count=2 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss
```

Release3 accepts `--scaling-map=K|L|LK` in both `init-sh` and inline
`train ... --init-sh` construction. The default remains `LK`. With internal
\(\mu=k(l_{\max}+1)+l\), `K` maps only the pairwise `scal` and `s` parameters
to \(k\); it does not merge the full-\(\mu\) radial coefficient rows,
spherical-harmonic channels, product graph, gate features, or K-rank graph.
Consequently an already trained `LK` model cannot be silently converted to
`K`: such a conversion is exact only when all `l` blocks at each fixed `k`
already have identical `scal/s` values.

The release3 LAMMPS interfaces preserve this sharing without changing the
full-\(\mu\) descriptor.  In the CPU direct-radial path, all angular channels
at fixed \(k\) reuse one radial-coordinate recurrence whenever the basis does
not require a separate \(\mu\)-local sigma.  The Kokkos interface deliberately
uses the original release3 generic CUDA kernels for both compact `K` models
and their exactly expanded `LK` counterparts; only the host-side mapping of
`scal/s` differs.  This keeps the established `LK` machine path unchanged and
prevents a K-only specialization from slowing existing models.  The LAMMPS
table path is likewise shared by `K` and `LK` after table construction.

As of the inline-init update, `init-sh` writes only the SH topology and radial
metadata. It intentionally omits `species_coeffs` and `moment_coeffs`, so
`MLMTPR::Load()` treats the model as uninitialized and the ordinary training
path initializes/fits parameters. Training can also build the same topology
directly. With `--inline-sh-model=<path>`, an existing file is used as the
starting model for continued training; a missing file is created from the
`init-sh` options:

```bash
../bin/mlp-sus2 train train.cfg --init-sh \
  --inline-sh-model=untrained_sus2sh_l3k3_b5_l3322.sus2 \
  --species-count=2 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0 \
  --do-samp \
  --do-lin \
  --do-lin-rescale \
  --curr-pot-name=current.sus2 \
  --trained-pot-name=p.sus2
```

The corrected generated model has 48 basic SH channels, 1416 product graph records, 525 total moment nodes, and 261 scalar basis functions. The previous bad graph with 317 scalar basis functions was archived as `untrained_sus2sh_l3k3_b5_l3322.bad-shgraph.sus2`; it kept zero intermediate tensors such as nodes 171/172/173 and produced structural zero descriptor columns.

The l=4 probe model was generated with:

```bash
../bin/mlp-sus2 init-sh untrained_sus2sh_l4k3_b5_l4422.sus2 \
  --species-count=2 \
  --l-max=4 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=4,4,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss
```

The generated l=4/body-l-max `4,4,2,2` model has 75 basic SH channels, 1470 product graph records, 558 total moment nodes, and 267 scalar basis functions.

Single-configuration `calc-efs`, descriptor generation on `smoke5.cfg`, and a 5-configuration one-step BFGS smoke train complete successfully. The fixed descriptor check found zero structural all-zero columns on `smoke5.cfg`.

The Cu-Zr training file used here has energy and force labels but no stress/virial labels, so the submitted queue-33 job explicitly uses `--stress-weight=0`.

Queue-33 full training script:

```text
bsub_sus2sh_l3k3_b5_l3322_33.lsf
```

The active run uses 160 MPI ranks across four 40-core nodes, uses Hydra with the LSF hostfile, includes `--do-samp`, and keeps the SUS2 `do-lin` flow:

```bash
mpiexec.hydra -machinefile $LSB_DJOB_HOSTFILE -np $NP \
  <historical-path> train \
  untrained_sus2sh_l3k3_b5_l3322.sus2 \
  /work/phy-weigw/hyx/cu-zr/sus2-sh/train.cfg \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0 \
  --stdd-weight=0.0 \
  --std-weight=0.0 \
  --do-samp \
  --init-params=same \
  --do-lin \
  --do-lin-rescale \
  --do-lin-steps=5000 \
  --do-lin-freq=50 \
  --max-iter=10000 \
  --curr-pot-name=current.sus2 \
  --trained-pot-name=p.sus2
```

The first 40-rank test job was stopped and its logs were preserved under `killed_3657712/`. The bad-graph 160-rank job `3657721` was stopped and archived under `killed_3657721_bad_sh_graph/`. The corrected 261-scalar job is `3657797`.

## SH runtime optimization notes

The current safe optimizations keep the SH mathematical basis, CG coupling graph, scalar list, and parameter layout unchanged:

- The value-only forward path now evaluates only real SH values and skips SH coordinate derivatives.
- The evaluator avoids `std::pow` for `l <= 4` and uses explicit inverse-radius products.
- Small SH value/derivative work arrays are stack buffers (`lmax <= 4`) instead of per-call heap vectors.
- Per-neighbor radial contractions are cached by `mu=(k,l)` and reused across all `m` channels. This removes repeated `R`-dimension dot products in forward, force derivative, site-energy derivative, and direct BFGS coefficient-gradient paths.
- The linear-equation assembly skips force or stress component construction when the corresponding training weight is zero. This is important for the current Cu-Zr dataset because it has no stress labels and is trained with `--stress-weight=0`.
- Runtime SH evaluation now uses direct real spherical harmonic polynomials for values and coordinate derivatives. The older complex evaluator remains only as an internal reference path; ordinary forward, force, and BFGS gradient paths use the real `double` evaluator.
- Direct real SH normalization constants are initialized once and reused, instead of recomputing `sqrt(...)` factors for every neighbor.
- Force-weighted BFGS coefficient-gradient accumulation now caches each neighbor's real SH values/derivatives, radial basis values/derivatives, and `mu=(k,l)` radial contractions. This removes the previous duplicate `EvalRealSH`/`RB_Calc` pass inside `AccumulateSHCombinationGrad` when force residuals are active.
- The force/stress linear block buffers use `resize` instead of zero-filling `assign`, because the block rows are fully overwritten before BLAS assembly.

Validation against the pre-optimization universal binary:

- `smoke5.cfg` `calc-efs`: exact file match.
- `smoke5.cfg` one-step training: identical `current.sus2`.
- First 1000 Cu-Zr training structures `calc-efs`: numeric token max absolute difference `0.0`; text diff only showed `+0.00000/-0.00000` stress formatting.
- First 1000 Cu-Zr training structures one-step training: same metrics; `current.sus2` max numeric token difference `1.735e-18`.
- First 1000 Cu-Zr training structures timing on one rank: `calc-efs` improved from about `9.05 s` to `7.01 s`; one-step training improved from about `67.07 s` to `57.38 s`.
- Stress-gating validation with `--stress-weight=0`: `smoke5.cfg` one-step linear path produced identical `current.sus2`. A 300-structure l=4 linear solve reduced elapsed time from about `136.3 s` to `126.3 s`.
- Stress-active validation with synthetic `PlusStress` and `--stress-weight=0.01`: `current.sus2` matched the pre-gating binary exactly.
- l=4 regression: `calc-efs` and one-step training completed on `smoke5.cfg`; finite-difference force validation on six atom/direction components gave max absolute error about `5.0e-7 eV/A`.
- Direct-real SH formula check: independent pointwise comparison against the previous complex evaluator found l=4 value differences around `4.4e-16` and derivative differences around `5.3e-15`.
- Direct-real runtime regression against the complex-evaluator backup: l3/l4 `calc-efs` matched exactly; deterministic one-step BFGS from complete models matched exactly for l3 and to about `1.7e-11` max numeric token difference for l4.
- Direct-real final finite-difference l=4 check: six force components retained max absolute error about `5.0e-7 eV/A`.
- Direct-real `calc-efs` timing on the first 1000 Cu-Zr structures, one rank, four alternating old/new samples: complex-evaluator mean `6.82 s`, direct-real mean `6.37 s` (about `1.07x`). The login-node timing is noisy, so this should be treated as indicative rather than a hard benchmark.
- Direct-real one-step BFGS timing on 300 structures from a complete model did not improve in the short run (`26.0 s` old vs `26.8 s` new), while the trained parameter files matched to `2.6e-18`. This suggests the current training bottleneck is dominated by coefficient-gradient assembly and linear algebra around the SH values, not by complex-vs-real SH evaluation alone.
- SH neighbor-cache validation against `bin/mlp-sus2.before-sh-neighbor-cache-20260513`: l3 `smoke5.cfg` with fixed `ini.sus2 --skip-preinit` matched exactly for `calc-efs`, `current.sus2`, and `p.sus2`.
- SH neighbor-cache l4 validation with the l4/body-l-max `4,4,2,2` model matched exactly for `calc-efs`, `current.sus2`, and `p.sus2`.
- SH neighbor-cache 300-structure BFGS timing from fixed `ini.sus2 --skip-preinit`: old runs `26.23 s`, `24.90 s`; new runs `21.19 s`, `20.70 s`; mean speedup about `1.22x`. The resulting `current.sus2` numeric max difference was `1.735e-18`.

Rejected or not-yet-retained optimization attempts:

- Lowering the force block BLAS threshold from `n >= 1000` to `n >= 128` was not retained. On a 300-structure `--do-lin` test it changed the linear solution and training metrics, likely because the normal equation is ill-conditioned enough that different BLAS summation order changes the solve. Keep this as an opt-in experiment only after a dedicated normal-matrix/RHS audit.
- Replacing selected SH gradient inner-loop `mu_to_*` lookups with `basic_*_cache_` was not retained as a standalone change. It is mathematically expected to be equivalent, but it did not provide a useful validated speed result and must be reintroduced only with fixed-start BFGS regression tests.

Larger but more invasive optimization candidates are intentionally deferred:

- Stream force/stress rows directly into the linear normal equations instead of materializing full `basis_ders`, `forces_cmpnts`, and `stress_cmpnts`.
- Fuse prediction and BFGS residual-gradient accumulation so a neighborhood's SH/radial state is not rebuilt once for E/F prediction and again for coefficient gradients.
- Add a final scalar-rooted graph closure/reindexing pass if future scalar generation leaves unreachable product nodes.

## SUS2-SH-TT shared improvements synced back to SUS2-SH

The TT-specific factorized model path is intentionally not part of this tree, but the plain SH runtime can reuse the training-side graph optimizations proven in the TT branch.

Synced generic changes:

- Product graph row program: groups all products with the same target moment and enables `SUS2_SH_PRODUCT_ROWS=1` for forward and derivative propagation.
- Site derivative cache: enables `SUS2_SH_SITE_DER_CACHE=1` to compute SH site derivatives once per neighborhood before force/stress projection.
- Coefficient-gradient fast paths: adds value-only radial basis evaluation, cached basic SH/radial indices, radial coefficient accumulation buffers, `SUS2_SH_ACCUM_SKIP_SITE_DERS=1`, and `SUS2_SH_PRODUCT_HVT_REVERSE=1`.
- SH init basis controls: default `init-sh` uses the full q-index total-order graph (`--sh-factor-pruning=q-total`). `legacy` adds separate monotonic `l`/`k` pruning, while `same-k` retains complete angular coupling but only within one radial `k` channel, giving the exact hierarchy `same-k` subset `legacy` subset `q-total`. Gate1 and gate2 can independently select these outer-graph subgraphs with `--two-layer-gate-factor-pruning` and `--second-layer-gate-factor-pruning`; their defaults inherit outer then gate1. `--write-sh-scalar-info` writes optional scalar metadata and is automatic for gate models.
- SH harmonic support now covers `l <= 6`; existing l<=4 models remain compatible.
- `init-sh` can write `RBLaguerre_log1p` and `RBJacobi_sss` SH models in addition to the Chebyshev SSS variants. Lowercase aliases `laguerre_log1p` and `jacobi_sss` are canonicalized to the saved model names. `RBJacobi_sss` supports `--k-max <= 6` because Jacobi blocks are indexed as `k=0..5`.

Regression scripts added under `dev_test/`:

- `sh_product_row_program_equivalence_check.sh`
- `sh_site_der_cache_equivalence_check.sh`
- `sh_accum_skip_site_ders_train_check.sh`
- `sh_product_hvt_reverse_train_check.sh`
- `sh_factor_tuple_total_order_check.sh`
- `sh_init_format_compat_check.sh`
- `sh_lmax_5_6_init_smoke.sh`
- `sh_radial_basis_init_smoke.sh`
