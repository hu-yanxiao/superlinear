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
\sqrt{\frac{h_j^{\mathrm{stab}}}{h_B^{\mathrm{ref}}}}.
\]

Thus the transformed Hessian

\[
H_z=D^{-1}H_xD^{-1}
\]

has a nearly uniform diagonal within each selected block. When
\(\widetilde h\ge 1\), the block is also damped by its absolute GN curvature.
When \(\widetilde h<1\), the block median has \(d=1\): relative anisotropy is
removed without making the entire low-curvature block take a larger first
physical step than legacy BFGS. If a block has no positive estimated
curvature, it retains identity coordinates.

## Stage and compatibility semantics

- The scale is built once after the step-zero `TrainLinear` refresh, if that
  refresh is enabled.
- It remains fixed for the rest of the `Train()` call. Later linear refreshes
  do not silently change the BFGS coordinate system.
- Building the scale resets the BFGS state and starts the stage from the same
  physical coefficients.
- Scale construction snapshots every physical coefficient and requires exact
  equality after all forward and reverse probes. Any probe path that modifies
  a model coefficient aborts training before BFGS starts.
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
