# Release3 BFGS coordinate normalization

Release3 provides an opt-in fixed diagonal coordinate system for dense BFGS:

```text
--bfgs-coordinate-normalization=efs-gn
```

The default is `none`, which preserves the previous release3 trajectory.

## Exact coordinate transformation

Let \(x_j\) be a physical model coefficient, \(x_j^{\mathrm{ref}}\) its fixed
value at the beginning of the BFGS stage, and \(d_j>0\) a fixed scale. BFGS
stores

\[
z_j=d_j\left(x_j-x_j^{\mathrm{ref}}\right).
\]

Before every model evaluation,

\[
x_j=x_j^{\mathrm{ref}}+\frac{z_j}{d_j},
\]

and the physical gradient is pulled back as

\[
\frac{\partial L}{\partial z_j}
=
\frac{1}{d_j}\frac{\partial L}{\partial x_j}.
\]

Therefore

\[
\delta L
=
\sum_j\frac{\partial L}{\partial x_j}\delta x_j
=
\sum_j\frac{\partial L}{\partial z_j}\delta z_j.
\]

The forward model, objective, physical gradient, and represented potential are
unchanged. Only the coordinates seen by BFGS differ. Centering makes the
initial optimizer coordinate exactly zero, so the first pack/unpack operation
recovers the initial physical coefficient without a floating-point
multiply/divide round trip.

The scaling is applied to:

- element-dependent `species_coeffs` prefactors;
- the outer scalar head and, when present, the residual E0 scalar head;
- gate1 and gate2 additive coefficients;
- gate scalar/full mixing weights and body-order mixing weights;
- gate1 edge-L1 weights when that mode is present.

Radial rows are not included. They retain either the physical
parameterization or the independent norm/direction chart selected by
`--radial-parameterization=weight-norm`.

## Scale construction

For the energy, force, and stress residual outputs, define the generalized
Gauss-Newton matrix

\[
H_{\mathrm{GN}}
=
J^{\mathsf T}WJ,
\]

where \(J\) is the Jacobian of the predicted E/F/S outputs with respect to the
physical coefficients and \(W\) contains the exact local loss curvatures and
training weights.

The desired diagonal is

\[
h_j=(H_{\mathrm{GN}})_{jj}.
\]

Materializing \(J\) or \(H_{\mathrm{GN}}\) would be expensive. Split the
objective into configuration contributions \(J_s^{\mathsf T}W_sJ_s\).
For each configuration and deterministic Rademacher probe
\(\xi_s^{(p)}\),

\[
v_s^{(p)}
=
J_s^{\mathsf T}W_s^{1/2}\xi_s^{(p)}.
\]

Then

\[
\mathbb E\left[\left(v_{s,j}^{(p)}\right)^2\right]
=
(J_s^{\mathsf T}W_sJ_s)_{jj},
\]

so the implementation estimates

\[
\widehat h_j
=
\sum_s\frac{1}{P}\sum_{p=1}^{P}
\left(v_{s,j}^{(p)}\right)^2.
\]

Each \(v_s^{(p)}\) uses the existing complete E/F/S reverse chain. It therefore
includes the element prefactor, scalar head, gate, radial, force, and stress
dependencies without introducing a separate derivative implementation.
Squaring before the MPI reduction removes zero-mean cross terms between
configurations and ranks. This lowers estimator variance and requires one
diagonal `MPI_Allreduce` after all probes rather than one gradient reduction
per probe.

For componentwise L2 forces,

\[
\frac{\partial^2 \ell_F}{\partial r_F^2}=2.
\]

For the configured log-cosh force loss with scale \(s\),

\[
\frac{\partial^2 \ell_F}{\partial r_F^2}
=
1-\tanh^2\left(\frac{r_F}{s}\right).
\]

The exact diagonal curvature of coefficient-only regularizers already present
in the objective is added for the scalar head, gate1 mixing weights, and fixed
atomic-energy constraint. No new regularizer is introduced.

Within each parameter block, let \(\widetilde h\) be the median positive
diagonal. The stabilized value is

\[
h_j^{\mathrm{stab}}
=
\max\left(\widehat h_j,\tau\widetilde h\right),
\]

where \(\tau\) is `--bfgs-coordinate-floor`. Define the block reference
curvature

\[
h_B^{\mathrm{ref}}=\min(\widetilde h,1).
\]

The value \(1\) is the curvature scale implied by the previous identity
inverse-Hessian initialization in physical coordinates. The coordinate scale
is

\[
d_j=
\max\left(
1,
\sqrt{\frac{h_j^{\mathrm{stab}}}{h_B^{\mathrm{ref}}}}
\right).
\]

Thus the transformed Hessian

\[
H_z=D^{-1}H_xD^{-1}
\]

has a reduced diagonal spread in stiff directions. The lower bound \(d_j\ge1\)
is a stability policy: with an identity inverse Hessian in optimizer
coordinates, the initial physical direction is

\[
p_{x,j}=-\frac{1}{d_j^2}\frac{\partial L}{\partial x_j}.
\]

Consequently EFS-GN may damp a high-curvature direction, but it never takes a
larger initial physical step than legacy BFGS in a low-curvature or noisy probe
direction. Low-curvature coordinates that would require \(d_j<1\) retain the
identity scale. If a block has no positive estimated curvature, the whole
block retains identity coordinates.

## Stage and compatibility semantics

- If a step-zero `TrainLinear` refresh is enabled, it runs first. The scale is
  then built once from that physical model before BFGS receives its first
  optimizer-coordinate vector.
- It remains fixed for the rest of the `Train()` call. Later linear refreshes
  do not silently change the BFGS coordinate system.
- BFGS is initialized only after the fixed coordinate map exists. There is no
  physical-coordinate BFGS state to restart or reinterpret.
- Scale construction snapshots every physical coefficient and requires exact
  equality after all forward and reverse probes. Any probe path that modifies
  a model coefficient aborts training before BFGS starts.
- Probe-only `linear_coeffs` and `max_linear` statistics are restored before
  the first optimizer objective. Normal E/F/S evaluation owns and rebuilds its
  transient gate and edge caches; the coordinate initializer does not clear
  those buffers out of band.
- After all radial and EFS-GN transforms are installed, the trainer performs a
  complete physical-to-optimizer-to-physical round trip and aborts if any
  coefficient changes beyond floating-point reconstruction tolerance.
- The exact total objective at the loaded physical point is logged before the
  first BFGS proposal. This is distinct from `step=0`, which is the first
  accepted post-proposal point.
- The estimator uses `--bfgs-coordinate-probes=4` by default.
- The current scale estimator includes E/F/S and coefficient-only L2
  curvature. `std`, `stdd`, radial smoothness, gate-X, compression guards, and
  generic constraint curvature are not included in the scale estimate; they
  remain fully present in the exact objective and exact gradient used by
  BFGS.
- Model files store only physical coefficients. No coordinate scales, probe
  state, or optimizer metadata are serialized.
- Previously trained release3 models remain readable. Models trained with
  this option remain readable by release3 inference binaries that do not know
  about the option.
- Inference has no additional normalization work.

## Validation

Run:

```bash
dev_test/sh_release3_bfgs_coordinate_normalization_check.sh
```

The regression executes a standalone finite-difference check of the affine
coordinate pullback, verifies exact preservation of the physical coefficient
vector during scale construction, checks deterministic scales and
accepted-step traces, and verifies coexistence with radial weight
normalization and the unchanged model format. The same coordinate check is
also compiled into `self-test-dev`; the standalone check does not depend on
the historical public-test fixtures.

### Physical-model resume validation (2026-07-17)

An earlier implementation installed the EFS-GN coordinate map after dense
BFGS had already received the physical-coordinate vector and initialized its
state. Restarting BFGS inside the first training-loop iteration then mixed the
old state sequence with the new coordinate chart. A trained physical model
therefore started at a different objective even though an isolated coefficient
round trip was accurate.

The fixed sequence is:

1. load the physical model;
2. run the optional step-zero `TrainLinear` refresh, if explicitly enabled;
3. construct and freeze the EFS-GN coordinate map;
4. verify the complete physical-to-optimizer-to-physical round trip;
5. initialize dense BFGS in optimizer coordinates;
6. evaluate and log the unchanged physical startup objective.

The final Intel/MPI binary had SHA256
`be0f694b1f99cf1c3b1e08fb6bbe555370f45f2801fb3224daada086943cae0a`.
Using the same trained physical model in both modes:

- the one-configuration startup objective was exactly
  `1.32790627180701410e-02` for both `none` and `efs-gn`;
- the full 2307-configuration, 153483-atom, 288-rank startup objective was
  exactly `2.56662650749638566e-01` for both modes;
- the combined radial and EFS-GN physical round-trip maximum absolute error
  was `6.66133814775093924e-16`;
- both full-data jobs completed with exit code zero, and the EFS-GN scale
  construction took `6.72 s` in that run.

The full-data records are under
`release3/2/1-layer/5bd/2/weight-norm/`
`bfgs-coordinate-normalization-physical-resume-fix/clean-final-v2-full-ab`.
