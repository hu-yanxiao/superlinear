# Release3 radial weight normalization

Release3 provides an opt-in optimizer parameterization for the outer radial
coefficient rows and, when present, the independent gate1 radial contraction
rows:

```text
--radial-parameterization=weight-norm
```

The default remains `physical`, so existing commands retain their previous
behavior.

For one physical outer radial row

\[
\mathbf c=(c_0,c_1,\ldots,c_{R-1}),\qquad c_0>0,
\]

the optimizer stores one log-scale coordinate and `R-1` direction coordinates.
Let

\[
\eta=\log\lVert\mathbf c\rVert_2,
\qquad \mathbf y\in\mathbb R^{R-1},
\qquad r=\lVert\mathbf y\rVert_2,
\]

and define

\[
\theta=\frac{\pi}{2}\tanh r.
\]

The forward map is

\[
c_0=e^\eta\cos\theta,
\qquad
\mathbf c_\perp=e^\eta\sin\theta\frac{\mathbf y}{r},
\]

with the continuous limit \(\mathbf c_\perp=0\) at \(r=0\). The inverse map
uses

\[
\theta=\operatorname{atan2}(\lVert\mathbf c_\perp\rVert_2,c_0),
\qquad
r=\operatorname{atanh}\left(\frac{2\theta}{\pi}\right),
\qquad
\mathbf y=r\frac{\mathbf c_\perp}{\lVert\mathbf c_\perp\rVert_2}.
\]

This is an exact `R`-to-`R` coordinate transformation. It adds no optimizer
coordinate and has no internal `v -> a v` redundancy in the chart. It does not
change the represented radial function or reduce the model parameter count.
Any physical scale gauge between a radial row and downstream coefficients
remains part of the model. The map is one-to-one between finite optimizer
coordinates and physical rows with \(c_0>0\).

Gate1 rows do not have the outer model's positive-first-coefficient
constraint. For each gate1 row, the trainer therefore selects once

\[
p=\operatorname*{arg\,max}_{a}|c_a|,
\qquad
\sigma=\operatorname{sign}(c_p),
\]

and uses a signed full-sphere pivot chart. With

\[
\theta_g=\pi\tanh\left(\frac{r}{2}\right),
\]

the map is

\[
c_p=\sigma e^\eta\cos\theta_g,
\qquad
\mathbf c_{\setminus p}
=e^\eta\sin\theta_g\frac{\mathbf y}{r}.
\]

Its inverse direction radius is

\[
r=2\operatorname{atanh}\left(\frac{\theta_g}{\pi}\right).
\]

Thus the selected physical coefficient may cross \(c_p=0\); the chart does
not constrain gate1 to its initial pivot hemisphere. It covers every nonzero
direction except the single antipodal chart pole, which is approached only as
\(r\to\infty\). The factor of two preserves the outer chart's local angular
slope \(\pi/2\) at \(r=0\).

The pivot \(p\) and sign \(\sigma\) are optimizer-only state reconstructed
from the physical row at the beginning of training. They are not written to
the model. Choosing the largest-magnitude component keeps the initial point
away from the chart boundary without changing the row or refitting downstream
coefficients.

The hyperbolic tangent avoids the poor conditioning of the direct ratio chart
\(c_a/c_0\). In the trained `release3/2/1-layer/5bd` model,

\[
\min_\mu\frac{c_{\mu0}}{\lVert\mathbf c_\mu\rVert_2}
=4.16\times10^{-3}.
\]

The largest direct-ratio coordinate would be `160.65`, while the largest
direction radius in this chart is `3.313`.

If \(q_a=\partial L/\partial c_a\), then

\[
\frac{\partial L}{\partial\eta}
=\sum_{a=0}^{R-1}q_ac_a.
\]

For \(r>0\), let \(\mathbf p=\mathbf y/r\). The direction pullback is

\[
\frac{\partial L}{\partial\mathbf y}
=e^\eta\left[
\theta'(r)
\left(-q_0\sin\theta
+(\mathbf q_\perp\!\cdot\!\mathbf p)\cos\theta\right)\mathbf p
+\frac{\sin\theta}{r}
\left(\mathbf q_\perp
-(\mathbf q_\perp\!\cdot\!\mathbf p)\mathbf p\right)
\right],
\]

where

\[
\theta'(r)=\frac{\pi}{2}\operatorname{sech}^2r.
\]

At \(r=0\), the implementation uses the analytic limit

\[
\frac{\partial L}{\partial\mathbf y}
=e^\eta\frac{\pi}{2}\mathbf q_\perp.
\]

## Compatibility

- Model files continue to store only the physical outer and gate1 radial
  coefficient values.
- Existing release3 models remain readable in either parameterization. When an
  old model has a non-positive first outer radial coefficient, training uses
  the existing release3 sign-repair and linear-refit path before creating
  optimizer coordinates. Gate1 rows use the signed pivot chart without any
  coefficient modification.
- Models trained with weight normalization remain readable by older release3
  inference binaries because no new model field is written.
- Inference uses the saved physical coefficients and has no normalization cost.
- Independent gate1 radial contraction rows use the same norm/direction
  parameterization. A gate1 mode that reuses the outer radial rows needs no
  additional coordinates. Gate2 radial rows retain their existing physical
  parameterization.
- Pack, unpack, and gradient pullback cost
  \(O((K_{\rm outer}+K_{\rm gate1})R)\) once per objective evaluation, outside
  the atom and edge loops. The inference path has no added normalization work.
- `TrainLinear` skips its legacy in-place `Orthogonalize()` call in
  weight-normalized mode because that projection would modify the physical
  feature map outside the optimizer coordinates. The row norm remains a
  trainable coordinate; this is an opt-in optimization path, while the default
  physical mode is unchanged.
- Trial changes of each row's log norm are step-limited before evaluation, so
  an extreme BFGS/L-BFGS proposal is reduced by the active line search instead
  of overflowing during reconstruction.

## Validation

Run:

```bash
dev_test/sh_release3_radial_weight_norm_check.sh
```

The check covers outer and gate1 all-row coordinate round trips, including a
negative gate1 pivot; finite-difference derivatives of the complete training
objective; equality of the initial objective; model-format compatibility; and
a `TrainLinear` refresh without radial-row projection. Models after an
optimizer step are not required to match because the two coordinate systems
intentionally produce different BFGS trajectories.

### Recorded server validation (2026-07-16)

The Intel/MPI release3 build at
`/work/phy-weigw/20260321_Test/SUS2-SH-release3-radial-weight-norm-work-codex`
passed the regression above. Additional checks used the trained model in
`release3/2/1-layer/5bd/2`:

- All 20 outer and 20 gate1 rows round-tripped with maximum relative error
  `5.38e-13`.
- A complete energy/force/stress objective finite difference for gate1 row
  `mu=5` had maximum relative error `2.43e-5` and maximum absolute error
  `2.28e-4`.
- The pre-weight-normalization release3 inference binary read a newly saved
  model and completed `calc-efs` after loading its required GCC runtime.
- On the same trained model and one 48-atom structure, a stable paired one-step
  timing was `18.50 s` for weight normalization and `18.45 s` for physical
  coordinates, a `0.27%` difference. Other repetitions showed host I/O
  outliers, so this is an overhead smoke test rather than a throughput claim.
