# Adaptive radial coefficient gate

This is a separate `radial-coefficient-gate` model mode. It does not reinterpret
existing release3, cross-channel superlinear, or pure-W same-channel files.

## Mathematical contract

Write `L = l_max + 1`, `K = k_max`, `O = outer_k_rank`, `N = rb_size`, and
`Q = gate_scalar_count`. The supported width is **1 <= O <= K**. The existing
gate equivariant moments, upstream gate-rank projection, and invariant product
graph produce raw scalar features `S[j,q]`. There is no gate activation.

\[
w_{jlon}=\sum_{q=0}^{Q-1}U_{lonq}S_{jq},\qquad
b_{jlon}=c_{lon}+a_{lon}w_{jlon},
\]
\[
R_{ij,lo}=\sum_{n=0}^{N-1}b_{jlon}\,
T_n(r_{ij};\theta^{Z_iZ_j}_{\sigma(l,o)}),\qquad
A_{i,olm}=\sum_j v_{Z_i,lo}v_{Z_j,lo}R_{ij,lo}Y_{lm}(\widehat r_{ij}).
\]

`T` includes the existing model's global radial `scaling` and cutoff convention.
Element coefficients `v` outside `R` are trainable constants, independent of the
environment. Gate species coefficients remain a separate parameter block.

| scaling map | sigma(l,o) | existing mu slot |
| --- | --- | --- |
| LK | o*L+l | o*L+l |
| K | o | o*L+l |
| L | l | o*L+l |

Each scaling slot retains the full ordered element-pair slope/shift parameters.
The gate still uses all K channels and its original `sigma(l,k)` mapping. When
gate scaling is shared, gradients from both layers **add to the same slots**.
Decoupled gate scaling retains its own slots. Shared scaling never implies
sharing the trainable `c`, `a`, or `U` across output channels.

By default, the outer invariant graph uses one k=1 angular topology,
independently replicated O times. Products never combine distinct output lanes,
and if the k=1 topology has P invariants, the outer head has O*P invariants.
When initialized with `--sh-factor-pruning=q-total`, the same adaptive radial
coefficient primitives feed a cross-channel outer invariant graph instead, so
outer products may combine different output lanes. The native energy convention
is unchanged: `E_i = shift[type_i] + h[type_i] * (1 + sum_p d_p * Phi_ip)`,
including the saved linear multipliers in `d`.

The storage order for adaptive rows is `(l*O+o)*N+n`; `q` is contiguous in `U`.
The base `c` and species constants occupy the existing `mu=o*L+l` radial rows;
base rows with k>=O are inactive, while their shared scaling slots can still be
used by the gate. `a` and `U` are explicit trainable blocks. The product `a*U`
can be folded exactly, but is not detached or treated as constant during training.

## Chain rule

For an arbitrary coordinate direction, use dots for directional derivatives:

\[
\dot w=U\dot S,\quad \dot b=a\dot w,\quad
\dot R=\sum_n(\dot b_nT_n+b_nT'_n\dot r),
\]
\[
\dot A=\sum_j v_iv_j(\dot RY+R\dot Y).
\]

The force/stress loss is evaluated by propagating `(A, dot A)` through the outer
product graph and reversing this **dual graph**, not by finite differences. For
`z += g*x*y`, with value/tangent adjoints `(bar z, bar dot z)`, the pullback is

\[
\bar x\mathrel{+}=g(\bar z\,y+\overline{\dot z}\,\dot y),\quad
\overline{\dot x}\mathrel{+}=g\overline{\dot z}\,y
\]

and the symmetric update to y. Both updates are necessary when x=y. Radial
scaling gradients include the mixed `d/dr d/slope` and `d/dr d/shift` terms and
the `dot b` contribution. After summing **all edges receiving atom j**, the
readout pullback is

\[
\nabla_c L=\sum_j\bar b_j,\quad
\nabla_a L=\sum_j(\bar b_jw_j+\overline{\dot b_j}\dot w_j),
\]
\[
\nabla_U L=\sum_j a(\bar b_jS_j^T+\overline{\dot b_j}\dot S_j^T),
\quad \bar S=U^T(a\bar b),\quad
\overline{\dot S}=U^T(a\overline{\dot b}).
\]

The existing analytic gate mixed-derivative engine receives these two independent
scalar seed arrays. There is no division by `a` or `w`; zero-value cases are
well-defined. Forces from the gate propagate to every neighbor of j, including
periodic repeated images and atoms outside i's immediate neighborhood.

Native stress-gradient callers already negate the stored stress residual. Thus
their edge direction is `f_weight[i]-f_weight[j]+stress_weight*r_ij`; adding a
second stress minus sign here would be incorrect. Stored EFS virial remains
`-sum_edges dE/dr_vec outer r_vec`.

## Execution and cache contract

- Raw gate scalar values are calculated once per atom and read out with GEMM.
- Adaptive coefficient values are cached per atom, not expanded per edge and
  scalar-feature index. Each edge contracts only N terms per (l,o).
- Angular adjoints contract over m before radial/readout pullback.
- All incoming-edge coefficient adjoints are summed before the gate reverse.
- Force/stress parameter gradients use one coordinate direction, not a full
  parameter-by-coordinate Jacobian or per-parameter reevaluation.
- EFS uses radial values and r derivatives; mixed gradients require all five
  derivative blocks. A requested full-cache EFS can share primitives with the
  next parameter gradient once, only after exact equality checks of every
  coefficient, global scaling, edge vector/distance, index and species. Other
  evaluations rebuild; energy-only calls and mutations invalidate this reuse.
- The exact primitive cache is currently mandatory for this mode. Disabling it
  is explicitly rejected; it must not silently select the old model's math.

### Exact EFS cache and gate access optimization

Gate type coefficients are resolved once at the cache/force routine boundary,
not through the mode-dependent offset helpers for every edge/basic node. The
borrowed matrix pointer is local to that evaluation; parameter vector updates
or resizes cannot leave a persistent stale pointer. Incomplete parameter blocks
retain the existing per-entry getter fallback and species bounds checks.

Ordinary adaptive EFS (shared-radial gate, no radial mixing) stores only the
coordinate derivative in an `edges * eval_blocks * N` array. The adaptive
consumer identifies compact versus full layout by the actual array size, not
the current environment setting. Shared-scaling gate contraction still receives
its established full-stride scratch, with the coordinate entries populated.
No gate basis channels are removed: for shared scaling all K basis groups can
still be needed even when O is smaller than K.

The unused environment-independent outer-mu value/derivative arrays are omitted
only on this EFS-only path. Requested full parameter/mixed derivative caches
retain the original arrays and five derivative blocks. Cache validation rejects
a compact buffer when parameter derivatives are required. Switching between
energy, EFS, linear components and full gradients preserves this distinction.

Diagnostic switches (not model-file fields):

- `SUS2_SH_GATE_TYPE_REFERENCE=1`: use the original per-entry gate type getter.
- `SUS2_SH_ADAPTIVE_EFS_REFERENCE=1`: retain full-stride radial coordinate
  storage and the old outer-mu arrays; mathematical outputs are unchanged.
- `SUS2_SH_ADAPTIVE_RADIAL_DR_FULL_CACHE=1`: disable just radial derivative
  compaction, for separate memory/performance diagnosis.

`codex_tests/adaptive_radial_benchmark.cpp` includes same-instance reference
switching and energy/EFS/components/full-gradient interleaving, plus mutation
checks. These switches are distinct from `SUS2_SH_ADAPTIVE_COMPONENTS`, which
selects only the linear-component evaluation algorithm, not ordinary EFS.

Initialization, regression evidence and measured performance are recorded after
their corresponding checks complete; the existence of this contract is not a
claim that a training run or benchmark has passed.

## Balanced fresh initialization

`--adaptive-radial-initialization=balanced` is a fresh-seed path only. It is not
applied when resuming a complete checkpoint. Resume should keep the saved
physical `c`, `a`, `U`, linear head, and optimizer-scale metadata unchanged.

The initializer uses the raw gate scalars `S[j,q]` produced by the upstream gate
graph. There is no activation, clipping, or extra nonlinear transform in the
adaptive radial readout:

\[
w_{jlon}=\sum_q U_{lonq}S_{jq}.
\]

The default target correction strength is
`--adaptive-radial-init-relative-rms=0.03`. The initialization sequence is:

1. Measure the training-set RMS of every raw scalar feature `S_q`.
2. Initialize each `U_lonq` with deterministic symmetry-breaking directions,
   scaled as `noise / (sqrt(Q) * rms(S_q))`.
3. Temporarily use `a=1`, measure the edge RMS of the baseline radial response
   from `c` and the adaptive correction response from `U*S`, then set each
   `(l,o)` amplitude block so the adaptive correction RMS is the requested
   relative fraction of the baseline RMS.
4. Measure the resulting outer primitive moment RMS and apply one common
   per-`(l,o)` factor to both `c` and `a`, preserving the calibrated correction
   ratio while keeping the outer moment scale near one.
5. Fit the linear head once, then skip the legacy `Rescale` and 75-step
   pretraining path.

Degenerate statistics are guarded. Non-finite reduced moments are fatal.
Near-zero raw scalar RMS falls back to unit RMS for that scalar. Near-zero
baseline RMS uses unit baseline scale so `a` remains nonzero. Near-zero adaptive
response RMS leaves the divisor at one and clamps the resulting amplitude to a
finite range. These guards avoid NaN/Inf coordinates while still making the
failure visible in the rank-0 initialization log.

Optimizer-coordinate metadata is non-predictive. The model file remains in
physical coordinates, and prediction must be identical if the metadata defaults
to one. The current names are:

- `adaptive_radial_weight_optimizer_scales[q] = 1 / rms(S_q)`, the physical
  natural scale for `U`. The initializer's extra `1/sqrt(Q)` is not folded into
  this metadata.
- `adaptive_radial_amplitude_optimizer_scales[row] = abs(initial a[row])`, after
  calibration and moment normalization.

The BFGS centered diagonal chart uses the inverse of these physical scales. In
other words, if the physical scale is `s`, the optimizer coordinate scale is
`d=1/s`; gradients are pulled back through the same existing diagonal-coordinate
machinery used by the scalar-head charts. Regularization and folding should
therefore use the physical `a` and `U` values, not optimizer coordinates.

The legacy `--gate-x-l2` penalty is not defined for this folded adaptive
response. Adaptive radial coefficient models default it to zero and reject
explicit nonzero values. Use the folded adaptive gate regularization terms for
this mode instead.

## Validation recorded on 2026-09-06

The C++ CLI regression suite passes with Clang/Accelerate, including K/L/LK,
shared and decoupled scaling, O=1, O=K, O greater than gate rank, model roundtrip,
Cartesian force and all nine virial components, full mixed E/F/stress parameter
gradients, nonzero STD and folded regularizers, zero-a and zero-U derivatives,
and rowwise a/U refactorization invariance. Batched linear components agree with
the reference reverse pass. The old pure-W model's predictions remain unchanged.
Run `dev_test/adaptive_radial_coefficient_gate_check.sh /absolute/path/mlp-sus2`.

`codex_tests/adaptive_radial_benchmark.cpp` separately checks cache reuse against
fresh gradients after coordinate, species, global scaling and coefficient
mutations; every tested component agreed exactly in the local build.

A fresh balanced small-fixture BFGS run completed three updates: the physical
startup loss was 0.00196948034310 and the last accepted loss 0.00193391678528.
The optimizer-to-physical roundtrip error was zero. A subsequent two-update
checkpoint resume using `efs-gn` also passed, without recalibration. These are
implementation smoke tests, not VO2 accuracy results.

The separate macOS ASAN/UBSAN build compiles but its ASAN runtime deadlocks in
shadow-memory initialization before `main`; it is not a passing sanitizer test.
Server compiler/MPI and VO2 testing must be verified separately.
