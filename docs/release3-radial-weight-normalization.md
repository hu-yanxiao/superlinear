# Release3 radial weight normalization

Release3 provides an opt-in optimizer parameterization for the outer radial
coefficient rows:

```text
--radial-parameterization=weight-norm
```

The default remains `physical`, so existing commands retain their previous
behavior.

For one physical radial row

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

- Model files continue to store only the physical `radial_coeffs` values.
- Existing release3 models remain readable in either parameterization. When an
  old model has a non-positive first radial coefficient, training uses the
  existing release3 sign-repair and linear-refit path before creating optimizer
  coordinates.
- Models trained with weight normalization remain readable by older release3
  inference binaries because no new model field is written.
- Inference uses the saved physical coefficients and has no normalization cost.
- The first implementation covers the outer radial rows only. Gate radial
  tables keep their existing physical parameterization.
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

The check covers all-row coordinate round trips, finite-difference derivatives
of the complete training objective, equality of the initial objective,
model-format compatibility, and a `TrainLinear` refresh without radial-row
projection. Models after an optimizer step are not required to match because
the two coordinate systems intentionally produce different BFGS trajectories.
