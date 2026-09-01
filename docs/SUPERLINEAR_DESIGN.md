# Release3 superlinear per-l radial mixer design

This note defines and records the implemented standalone release3 superlinear
extension.  The implementation is optional and old release3 model files remain
an exact fallback path.

## Scope

The extension adds an optional edge-distance-dependent matrix

\[
M_l(r_{ij}) \in \mathbb{R}^{R_\mathrm{outer}\times R_\mathrm{gate}}
\]

for each angular order `l`. The gate rank is the input channel space and the
outer rank is the output channel space. The matrix is shared over all `m`
components of the same `l`; it does not mix different `l` values and does not
mix different `m` values. Therefore it is a scalar radial operation on the
multiplicity space of a fixed irreducible representation, and it preserves
rotational equivariance.

The default implementation should be a residual term on top of the existing
release3 outer-rank path:

\[
O_{ij,a l m}
= O^{(0)}_{ij,a l m}
+ \Delta O_{ij,a l m},
\]

where \(O^{(0)}\) is the current `outer_k_rank` contribution. All new
superlinear coefficients initialize to zero, so an existing model or a newly
initialized model with the feature disabled has identical numerical behavior to
current release3.

## Existing rank semantics

For an edge \(e=(i,j)\), define the current gate-scaled outer edge primitive as

\[
X^p_{e,k l m}
= s^p_{e,k l}\, R_{k l}(r_e)\, Y_{l m}(\hat r_e),
\]

where \(p\) is the active pass. In the gate pass, \(s^g\) is the first-layer
type/radial scale. In the outer pass, \(s^o\) includes the release3 two-layer
gate pair scale, including the edge-L1 contribution when enabled.

Current release3 first sums raw primitives over neighbors and then applies a
constant channel contraction:

\[
B^p_{i,k l m}=\sum_{e:i\leftarrow j} X^p_{e,k l m},
\]

\[
G_{i,c l m}=\sum_k W^g_{l,c,k}B^g_{i,k l m},
\qquad
O^{(0)}_{i,a l m}=\sum_k W^o_{l,a,k}B^o_{i,k l m}.
\]

The superlinear term cannot be inserted after this neighbor sum, because
distance information has already been collapsed. The implementation point must
be the edge-primitive accumulation loop.

## Superlinear forward definition

In the outer pass, first compress each outer-pass edge primitive into the gate
rank space:

\[
Z_{e,c l m} = \sum_{k=0}^{K-1} W^g_{l,c,k} X^o_{e,k l m}.
\]

Then apply the per-`l` radial mixer:

\[
\Delta O_{e,a l m}
= \sum_{c=0}^{R_\mathrm{gate}-1} M_{l,a,c}(r_e) Z_{e,c l m}.
\]

Finally accumulate into the outer-rank primitive range:

\[
O_{i,a l m}
= O^{(0)}_{i,a l m}
+ \sum_{e:i\leftarrow j}\Delta O_{e,a l m}.
\]

The outer Clebsch-Gordan product graph and scalar readout consume
\(O_{i,a l m}\) exactly where the current outer-rank primitive values are
consumed. No downstream invariant graph changes are required.

The gate pass itself remains unchanged: it still constructs first-layer gate
scalars from the gate-rank product graph. The same \(W^g\) matrix is reused by
the superlinear outer-pass edge compressor, so the learned gate-rank channel
space becomes the input basis for the radial mixer.

## Radial parameterization

Use a separate learned coefficient block over the model's existing raw radial
basis:

\[
M_{l,a,c}(r)
= \sum_{b=0}^{B_s-1} C_{l,a,c,b}\,\phi_b(r).
\]

Here \(B_s\) equals the model radial basis size.  For each `l`, the implemented
\(\phi_{l,b}^{t_i t_j}(r)\) is the already available raw-basis evaluation for
`mu=(k=0,l)`, including the model cutoff and the selected type-pair radial
scaling map.  It is a scalar with respect to rotations and is independent of
`m`, but it can depend on the ordered type pair through that existing scaling.
With `scaling_map=L`, each `l` uses its own scaling block; with
`scaling_map=K`, all `l` values reuse the `k=0` scaling block.  Reverse-mode
and force/stress tangent paths include derivatives with respect to both scaling
parameters, so this reuse is part of the fitted model rather than a detached
feature.

Saved parameter layout:

\[
\operatorname{index}(l,a,c,b)
= (((l R_\mathrm{outer}+a)R_\mathrm{gate}+c)B_s+b).
\]

Metadata fields are explicit and omitted when disabled:

- `two_layer_gate_radial_mixing = full`
- `two_layer_gate_radial_mixing_coeff_count`
- `two_layer_gate_radial_mixing_coeffs`

For controlled continuation from a trained release3 checkpoint, the developer
binary provides:

```bash
mlp-sus2 enable-two-layer-gate-radial-mixing-dev input.mtp output.mtp
```

It preserves every existing coefficient, inserts the new block at its formal
layout offset, and initializes that block to exactly zero.  The upgraded model
therefore has identical energy, force, and stress before optimization.

The feature is valid only when both `two_layer_gate_k_rank > 0` and
`outer_k_rank > 0`. The expected weight count is

\[
(L_\mathrm{max}+1)R_\mathrm{outer}R_\mathrm{gate}B_s.
\]

All weights default to zero. This makes the feature an exact no-op at
initialization and preserves old release3 behavior.

## Reverse pass

Let \(\bar O_{i,a l m}\) be the adjoint arriving from the outer product graph.
For each edge \(e:i\leftarrow j\), the superlinear contribution gives

\[
\bar M_{l,a,c}(r_e) \mathrel{+}= \bar O_{i,a l m} Z_{e,c l m},
\]

\[
\bar Z_{e,c l m} \mathrel{+}=
\sum_a M_{l,a,c}(r_e)\bar O_{i,a l m}.
\]

For the matrix coefficients:

\[
\frac{\partial E}{\partial C_{l,a,c,b}}
\mathrel{+}=
\sum_{e,m}\bar O_{i,a l m}\,
Z_{e,c l m}\,\phi_b(r_e).
\]

The edge-compressor transpose contributes to the existing gate-rank weights and
raw edge primitive adjoints:

\[
\bar W^g_{l,c,k} \mathrel{+}= \bar Z_{e,c l m} X^o_{e,k l m},
\]

\[
\bar X^o_{e,k l m} \mathrel{+}=
\sum_c W^g_{l,c,k}\bar Z_{e,c l m}.
\]

These adjoints add to, not replace, the current gate-rank and outer-rank
adjoints. If gate-rank weights are shared between the first-layer gate graph and
the superlinear outer compressor, their gradient is the sum of both paths.

## Force and stress derivatives

For a coordinate or lattice degree of freedom \(q\),

\[
\frac{\partial \Delta O_{e,a l m}}{\partial q}
=
\sum_c
\left[
\frac{\partial M_{l,a,c}(r_e)}{\partial q} Z_{e,c l m}
+ M_{l,a,c}(r_e)
  \frac{\partial Z_{e,c l m}}{\partial q}
\right],
\]

with

\[
\frac{\partial M_{l,a,c}(r_e)}{\partial q}
=
\sum_b C_{l,a,c,b}\,\phi'_b(r_e)
\frac{\partial r_e}{\partial q}.
\]

The \(Z\) derivative expands through the current outer-pass edge primitive:

\[
\frac{\partial Z_{e,c l m}}{\partial q}
=
\sum_k W^g_{l,c,k}
\frac{\partial X^o_{e,k l m}}{\partial q}.
\]

The existing derivative of \(X^o\) already contains radial, spherical harmonic,
type/gate scale, and edge-L1 terms. The superlinear implementation must only
add the new \(M\) and \(\phi'_b\) chain terms and must preserve the existing
virial convention:

\[
\sigma_{\alpha\beta}
\mathrel{-}= \frac{\partial E}{\partial r_{e,\alpha}} r_{e,\beta}.
\]

Stress parity is therefore not a separate formula; it is the same edge
coordinate derivative accumulated into the stress tensor.

## Implementation insertion points

### Standalone C++ release3

Model metadata and parameter ownership:

- `dev_src/mtpr.h`: add flags, basis size, weight storage, offset/count/index
  accessors, and cache buffers if needed.
- `dev_src/mtpr.cpp`: read/write the new fields, validate counts and rank
  preconditions, include the new block in regression coefficient offsets, and
  keep disabled models byte-compatible in behavior.
- `dev_src/sh_model_init.cpp`: add init options, zero-initialize the new weight
  block, and write metadata only when enabled.
- `dev_src/mlp/mtpr_train.cpp`: add train CLI option validation so loaded-model
  topology cannot be silently changed by command-line flags.

Forward and derivatives:

- `dev_src/mtpr_sh.cpp::CalcSHMomentValuesOnly`: compute \(\phi_b(r)\), edge
  gate-rank \(Z\), and residual \(\Delta O\) inside the neighbor loop before
  outer products are applied.
- The edge-primitive cache remains available.  When it is active, the
  superlinear path reuses cached SH and contracted radial primitives and
  recomputes only the small set of raw radial-basis blocks needed by
  \(\phi_{l,b}\).  The older site-derivative-cache route stays disabled for a
  superlinear model unless the edge cache supplies the complete primitive set.
- `dev_src/mtpr_sh.cpp::CalcSHMomentValuesWithGradientCache` and the mixed-loss
  tangent path propagate the coefficient, radial-scaling, gate-rank and
  upstream outer-radial derivatives needed by BFGS fitting.
- `dev_src/mtpr_sh.cpp::CalcSHSiteEnergyDers`: after product-graph backprop,
  propagate adjoints through \(M_l(r)\), \(Z\), \(W^g\), and \(X^o\).
- Existing constant contraction helpers should remain as the fallback path for
  models without superlinear metadata.

`outer_k_rank == 1` needs special care. Current release3 can absorb constant
rank-one outer weights into radial coefficients. That transformation is valid
only for constant \(W^o\). It must be disabled when the superlinear radial mixer
is enabled.

### PyTorch trainer

The PyTorch trainer is the fastest place to establish the math:

- `sus2shpt/io/mtp.py`: add fields, parser, validator, and serializer.
- `sus2shpt/topology_init.py`: initialize the zero residual block and preserve
  field ordering.
- `sus2shpt/model/sus2_sh.py`: add a parameter tensor with shape
  `(lmax + 1, outer_k_rank, gate_k_rank, superlinear_basis_size)`.
- `sus2shpt/model/sus2_sh.py::_basic_moments_from_geometry`: add the
  edge-local path. Do not implement this by changing only
  `_contract_rank_moments`, because that function sees only atom-summed
  moments.
- `sus2shpt/training.py`: put the new tensor in a parameter group. Force and
  stress gradients are then handled by autograd, provided the edge formula uses
  differentiable `pair_r`, positions, and lattice tensors.

### LAMMPS / Kokkos

LAMMPS and Kokkos must either implement the same metadata and edge derivative
path or reject superlinear models at load time. Silent fallback to the old rank
path would be incorrect because the new model changes the energy expression.

For Kokkos, keep the parameter layout `l -> outer -> gate -> radial_basis` and
mirror it as a device view. In the force kernels, compute \(Z\) and \(MZ\) per
edge and per `l`, using small local arrays over `R_gate` and `R_outer`.

## Performance layout

The hot loop should be grouped by `l`:

1. evaluate or load \(X^o_{e,k l m}\);
2. compute all \(Z_{e,c l m}\) for that `l`;
3. evaluate \(\phi_b(r_e)\) once per edge;
4. form \(M_l(r_e)\);
5. accumulate all \(a,m\) outputs into the outer-rank primitive targets.

This avoids a full dense edge tensor of shape
`edge_count x l_count x R_outer x R_gate x m_count`. The temporary storage per
edge is only \(R_\mathrm{gate}+R_\mathrm{outer}R_\mathrm{gate}\), plus the
radial basis vector. The current hot path first computes the `K -> R_gate`
compression once per edge and `l`, builds all `R_outer x R_gate` kernels once,
and then applies them to the `2l+1` SH components.  Coefficients \(C\) remain
live contiguous parameters and are never baked into the cache.

After hoisting the `m`-independent work, the extra forward arithmetic cost is
approximately

\[
\sum_l \left[
K R_\mathrm{gate}
+ R_\mathrm{gate}R_\mathrm{outer}B_s
+ (2l+1)R_\mathrm{gate}R_\mathrm{outer}
\right]
\]

multiply-adds per edge. With small ranks this is controlled, but it is an
edge-stage cost, not a product-graph-stage cost.

## Acceptance criteria

1. With the feature disabled, all existing release3 rank/gate tests pass with
   unchanged outputs.
2. With the feature enabled and all `superlinear_rank_radial_weights = 0`,
   energy, force, and stress match the disabled model to roundoff.
3. Standalone C++ finite differences validate gradients with respect to
   positions, lattice, gate-rank weights, superlinear weights, base radial
   weights, gate weights, type/additive weights, and radial scaling.
4. C++ finite differences validate force and stress for at least one model with
   `two_layer_gate_k_rank > 1`, `outer_k_rank > 1`, and nonzero superlinear
   weights.
5. Energy-only, force-only, stress-only and combined EFS directional
   derivatives agree with central finite differences on an active nonzero
   model.
6. Serialization round trip preserves all new fields and weights exactly.
7. Old release3 models load in the new binary without adding implicit weights.
8. Old binaries either reject new superlinear models clearly or are never given
   such models.
9. LAMMPS/Kokkos parity is required before using superlinear models outside the
   standalone executable; this repository does not silently claim that parity.
10. Cache-on/cache-off parity must cover neighborhood cache, edge primitive
    cache, force derivative cache, and stress.
