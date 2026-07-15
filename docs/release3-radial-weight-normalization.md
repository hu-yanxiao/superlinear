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

the optimizer stores one log-scale coordinate and `R-1` direction coordinates:

\[
\eta=\log\lVert\mathbf c\rVert_2,
\qquad
z_a=\frac{c_a}{c_0},\quad a=1,\ldots,R-1.
\]

The forward map is

\[
D=1+\sum_{a=1}^{R-1}z_a^2,
\qquad
c_0=\frac{e^\eta}{\sqrt D},
\qquad
c_a=c_0z_a.
\]

This is an exact `R`-to-`R` coordinate transformation. It adds no optimizer
coordinate and has no internal `v -> a v` redundancy in the chart. It does not
change the represented radial function or reduce the model parameter count.
Any physical scale gauge between a radial row and downstream coefficients
remains part of the model.
If `q_a` is the objective derivative with respect to the physical coefficient
`c_a`, the optimizer gradient is

\[
\frac{\partial L}{\partial\eta}
=\sum_{a=0}^{R-1}q_ac_a,
\]

\[
\frac{\partial L}{\partial z_a}
=c_0q_a-\frac{z_a}{D}\sum_{b=0}^{R-1}q_bc_b,
\qquad a=1,\ldots,R-1.
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

The check covers the coordinate round trip, finite-difference derivatives of
the complete training objective, equality of initial predictions and loss,
model-format compatibility, and a `TrainLinear` refresh without radial-row
projection.
