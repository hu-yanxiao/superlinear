# Release3 radial weight normalization

Release3 provides an opt-in optimizer parameterization for the outer radial
coefficient rows and, when present, the independent gate1 radial contraction
rows:

```text
--radial-parameterization=weight-norm
```

The default remains `physical`, so existing commands retain their previous
behavior.

For every nonzero physical outer or gate1 radial row

\[
\mathbf c=(c_0,c_1,\ldots,c_{R-1})\ne\mathbf 0,
\]

the trainer first selects

\[
p=\operatorname*{arg\,max}_{a}|c_a|,
\qquad
\sigma=\operatorname{sign}(c_p),
\]

directly from the loaded physical row. The optimizer then stores one log-scale
coordinate and `R-1` direction coordinates. Let

\[
\eta=\log\lVert\mathbf c\rVert_2,
\qquad \mathbf y\in\mathbb R^{R-1},
\qquad r=\lVert\mathbf y\rVert_2,
\]

and define

\[
\theta=\pi\tanh\left(\frac{r}{2}\right).
\]

The forward map is

\[
c_p=\sigma e^\eta\cos\theta,
\qquad
\mathbf c_{\setminus p}=e^\eta\sin\theta\frac{\mathbf y}{r},
\]

with the continuous limit \(\mathbf c_{\setminus p}=0\) at \(r=0\). The
inverse map uses

\[
\theta=\operatorname{atan2}
\left(\lVert\mathbf c_{\setminus p}\rVert_2,\sigma c_p\right),
\qquad
r=2\operatorname{atanh}\left(\frac{\theta}{\pi}\right),
\qquad
\mathbf y=r\frac{\mathbf c_{\setminus p}}
{\lVert\mathbf c_{\setminus p}\rVert_2}.
\]

This is an exact `R`-to-`R` coordinate transformation. It adds no optimizer
coordinate and has no internal `v -> a v` redundancy in the chart. It does not
change the represented radial function or reduce the model parameter count.
Any physical scale gauge between a radial row and downstream coefficients
remains part of the model. The signed full-sphere chart covers every nonzero
direction except the single antipodal chart pole, which is approached only as
\(r\to\infty\). The selected coefficient may cross \(c_p=0\), so neither outer
nor gate1 rows are constrained to the initial pivot hemisphere.

The pivot \(p\) and sign \(\sigma\) are optimizer-only state reconstructed
from the physical row at the beginning of training. They are not written to
the model. Choosing the largest-magnitude component keeps the initial point
away from the chart boundary without changing the row or refitting downstream
coefficients.

The hyperbolic tangent avoids the poor conditioning of a direct ratio chart.
Selecting the largest-magnitude component also guarantees that the initial
physical row is well away from the chart's antipodal boundary, including when
its first coefficient is negative.

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
\left(-\sigma q_p\sin\theta
+(\mathbf q_{\setminus p}\!\cdot\!\mathbf p)\cos\theta\right)\mathbf p
+\frac{\sin\theta}{r}
\left(\mathbf q_{\setminus p}
-(\mathbf q_{\setminus p}\!\cdot\!\mathbf p)\mathbf p\right)
\right],
\]

where

\[
\theta'(r)=\frac{\pi}{2}\operatorname{sech}^2\left(\frac r2\right).
\]

At \(r=0\), the implementation uses the analytic limit

\[
\frac{\partial L}{\partial\mathbf y}
=e^\eta\frac{\pi}{2}\mathbf q_{\setminus p}.
\]

## Compatibility

- Model files continue to store only the physical outer and gate1 radial
  coefficient values.
- Existing release3 models remain readable in either parameterization. In
  `weight-norm` mode, both outer and gate1 rows use signed pivots and are
  round-trip verified without coefficient repair or a compensating linear
  solve. The legacy positive-first-coefficient repair remains confined to the
  default `physical` optimizer parameterization.
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

The check covers outer and gate1 all-row coordinate round trips, including
negative outer and gate1 pivots; finite-difference derivatives of the complete
training objective; equality of the initial objective; combined radial and
EFS-GN coordinate normalization on a continued physical model; model-format
compatibility; and a `TrainLinear` refresh without radial-row projection.
Models after an optimizer step are not required to match because the two
coordinate systems intentionally produce different BFGS trajectories.

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
