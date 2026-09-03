# Release3 direct K-to-outer distance mixer

This note defines the implemented standalone release3 superlinear extension.
The mixer is optional and residual: disabling it, or setting every mixer
coefficient to zero, recovers the release3 outer path exactly.

## Scope

The extension adds an edge-distance-dependent matrix

\[
M_l(r_{ij}) \in \mathbb{R}^{R_\mathrm{outer}\times K}
\]

for each angular order `l`. Here `K = sh_k_max`; the gate rank is not a mixer
dimension. The matrix is shared over all `m` components of a fixed `l`, and it
does not mix different `l` or `m` values. It therefore acts only on the
multiplicity space of one irreducible representation and preserves rotational
equivariance.

The implementation is a residual term on top of the existing
release3 outer-rank path:

\[
O_{ij,a l m}
= O^{(0)}_{ij,a l m}
+ \Delta O_{ij,a l m},
\]

where \(O^{(0)}\) is the current `outer_k_rank` contribution. New mixer
coefficients initialize to zero.

## Rank semantics and gate scalar

The gate rank \(R_g\) belongs only to the upstream gate-scalar generator. The
first gate pass contracts its equivariant `K` channels to `R_g`, builds its
Clebsch-Gordan graph, and produces scalar invariants \(S^g_{j,q}\). A scalar
readout then returns `K` gate channels for every `l`:

\[
h_{j,k l}=\sum_q A_{l k q}S^g_{j,q},
\qquad k=0,\ldots,K-1.
\]

Thus `two_layer_gate_k_rank` controls the cost of constructing the gate
scalars, while `sh_k_max` controls the width presented to the distance mixer.
Changing \(R_g\) must not change the mixer coefficient shape.

The release affine species gate forms the actual direct mixer input

\[
u_{ij,k l}=\tau_{z_i,k l}
\left(\tau_{z_j,k l}+\alpha_{z_j,k l}h_{j,k l}\right).
\]

The tanh gate variant uses the corresponding release multiplier, but the same
rule applies: the pair buffer passed to the mixer is the complete scalar
\(u_{ij,kl}\).

The baseline outer primitive remains

\[
X^{(0)}_{ij,k l m}
=u_{ij,k l}R_{k l}(r_{ij})Y_{lm}(\hat r_{ij}),
\qquad
O^{(0)}_{i,a l m}=\sum_{j,k}W^o_{l,a,k}X^{(0)}_{ij,k l m}.
\]

This old radial factor is used only by the baseline path. It is deliberately
not multiplied into the direct mixer input.

## Direct mixer forward definition

The residual mixes the `K` scalar channels directly to the outer rank:

\[
\Delta O_{i,a l m}
=\sum_j\sum_{k=0}^{K-1}
M^{z_i z_j}_{l,a,k}(r_{ij})
u_{ij,k l}Y_{lm}(\hat r_{ij}),
\]

\[
O_{i,a l m}=O^{(0)}_{i,a l m}+\Delta O_{i,a l m}.
\]

There is no intermediate
\(x=R(r)u\), no \(K\to R_g\) compression \(z=W^g x\), and no second radial
factor after scalar compression. The distance matrix \(M(r)\) is the only
radial function in the residual branch.

The outer Clebsch-Gordan product graph and scalar readout consume
\(O_{i,a l m}\) exactly where the current outer-rank primitive values are
consumed. No downstream invariant graph changes are required.

The gate pass itself remains unchanged. The gate-rank matrix \(W^g\) affects the
residual only upstream, through the generated scalar \(h\); it is not reused as
an edge compressor.

## Radial parameterization

Use a separate learned coefficient block over the model's raw radial basis:

\[
M^{pq}_{l,a,k}(r)
= \sum_{b=0}^{B_s-1} C_{l,a,k,b}\,
\phi_b\!\left(r;\lambda^{pq}_{\sigma(k,l)},s^{pq}_{\sigma(k,l)}\right).
\]

Here \(B_s\) is the radial basis size. The coefficient tensor \(C\) is shared
over species; ordered-pair dependence enters through the existing slope/shift
parameters of \(\phi\). The scaling-map lookup is

\[
\sigma(k,l)=
\begin{cases}
k, & \texttt{scaling\_map=K},\\
l, & \texttt{scaling\_map=L},\\
k(L_{\max}+1)+l, & \texttt{scaling\_map=LK}.
\end{cases}
\]

Consequently `K` gives each `k` one scaling channel shared across `l`, `L`
shares one `l` channel across all `k`, and `LK` gives every `(k,l)` its own
channel. This sharing applies only to the radial coordinate warp; the mixer
coefficients remain distinct for every `(l,a,k)`.

Saved parameter layout:

\[
\operatorname{index}(l,a,k,b)
= (((l R_\mathrm{outer}+a)K+k)B_s+b).
\]

Metadata fields are explicit and omitted when disabled:

- `two_layer_gate_radial_mixing = full`
- `two_layer_gate_radial_mixing_layout = direct-k`
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

The feature requires a two-layer gate, `two_layer_gate_k_rank > 0`, and
`outer_k_rank > 0`. The expected coefficient count is

\[
(L_\mathrm{max}+1)R_\mathrm{outer}K B_s.
\]

All coefficients default to zero. New direct-mixer models carry the explicit
`direct-k` marker. A model containing the historical gate-rank mixer metadata
but no layout marker is rejected rather than silently interpreting its
`R_g`-shaped coefficient vector as `K`-shaped data.

## Reverse pass

Let \(\bar O_{i,a l m}\) be the adjoint arriving from the outer product graph.
For each edge, the direct residual gives

\[
\bar M_{l,a,k}(r_{ij}) \mathrel{+}=
\sum_m\bar O_{i,a l m}u_{ij,k l}Y_{lm},
\]

\[
\bar u_{ij,k l}\mathrel{+}=
\sum_{a,m}\bar O_{i,a l m}M_{l,a,k}(r_{ij})Y_{lm}.
\]

For the matrix coefficients:

\[
\frac{\partial E}{\partial C_{l,a,k,b}}
\mathrel{+}=
\sum_{i,j,m}\bar O_{i,a l m}\,
u_{ij,k l}Y_{lm}\,\phi_b(r_{ij}).
\]

The \(\bar u\) adjoint is propagated through the species-affine gate into
\(h\), then through the scalar readout, gate Clebsch-Gordan graph, gate-rank
contraction, and gate radial primitives. This is the only residual path to the
gate-rank weights. The residual creates no direct adjoint for the baseline
outer radial coefficients.

## Force and stress derivatives

For a coordinate or lattice degree of freedom \(q\), the complete chain rule is

\[
\frac{\partial \Delta O_{e,a l m}}{\partial q}
=
\sum_k
\left[
\frac{\partial M_{l,a,k}}{\partial q}u_{e,k l}Y_{lm}
+M_{l,a,k}\frac{\partial u_{e,k l}}{\partial q}Y_{lm}
+M_{l,a,k}u_{e,k l}\frac{\partial Y_{lm}}{\partial q}
\right],
\]

with

\[
\frac{\partial M_{l,a,k}(r_e)}{\partial q}
=
\sum_b C_{l,a,k,b}\,\phi'_b(r_e)
\frac{\partial r_e}{\partial q}.
\]

The edge-local force loop treats the already computed gate scalar as fixed and
adds the first and third terms, \(M'uY+Mu\,dY\). It accumulates \(\bar u\) for
the second term; the gate-tangent/backprop machinery then differentiates the
neighbor's scalar environment. This separation prevents double counting. It
also makes explicit why no derivative of the baseline \(R_{kl}\) belongs to
the residual. The implementation preserves the existing virial convention:

\[
\sigma_{\alpha\beta}
\mathrel{-}= \frac{\partial E}{\partial r_{e,\alpha}} r_{e,\beta}.
\]

Stress parity is therefore not a separate formula; it is the same edge
coordinate derivative accumulated into the stress tensor.

### Forward-state requirement for force and EFS gradients

The reverse and mixed tangent/reverse passes must consume exactly the same
forward moment state as the value-only evaluator.  In particular, define

\[
O_i = O_i^{(0)} + \Delta O_i(C).
\]

The product graph is evaluated at this complete (O_i), so its Jacobian and
Hessian-vector products must also be evaluated at the complete (O_i).  It is
not sufficient to add the direct derivatives of \(\Delta O_i\) later while the
product graph still sees only \(O_i^{(0)}\).  That omission drops terms such as

\[
\frac{\partial^2 P}{\partial O^2}
\left[\Delta O,\frac{\partial O}{\partial q}\right]
\]

from force, stress, and combined EFS parameter gradients whenever the outer
graph (P(O)) is nonlinear.

For this reason `CalcSHMomentValuesOnly` and
`CalcSHMomentValuesWithGradientCache` call the same exact edge-forward helper,
and both add \(\Delta O\) before the Clebsch-Gordan product graph.  The active
coefficient state is refreshed at every internally callable moment/site entry,
so a coefficient update cannot leave the gradient path on a stale zero-mixer
branch.

The regression in
`dev_test/sh_superlinear_radial_mixing_product_graph_fd_check.sh` deliberately
uses a strong nonzero mixer and a nonlinear product graph.  The pre-fix binary
fails central finite differences for force, stress, and combined EFS with
order-one relative errors, while the corrected binary passes energy, force,
stress, and EFS for both base-radial and mixer-coefficient blocks with product
HVT reverse enabled and disabled.  A separate zero-(C) case verifies that the
direct \(\partial L/\partial C\) remains nonzero and correct even though the
forward residual is exactly zero.

## Implementation insertion points

### Standalone C++ release3

Model metadata and parameter ownership:

- `dev_src/mtpr.h`: owns the mode, direct-`K` coefficient layout, offsets, and
  reusable scratch/cache buffers.
- `dev_src/mtpr.cpp`: read/write the new fields, validate counts and rank
  preconditions, include the block in regression coefficient offsets, and
  reject the obsolete unmarked gate-rank layout.
- `dev_src/sh_model_init.cpp`: add init options, zero-initialize the new weight
  block, and write metadata only when enabled.
- `dev_src/mlp/mtpr_train.cpp`: add train CLI option validation so loaded-model
  topology cannot be silently changed by command-line flags.

Forward and derivatives:

- `dev_src/mtpr_sh.cpp::CalcSHMomentValuesOnly`: compute \(\phi_b(r)\), apply
  `M(l,a,k)` directly to the pair scalar `u(mu)`, and accumulate
  \(\Delta O\) inside the neighbor loop.
- The edge-primitive cache remains available.  When it is active, the
  superlinear path reuses cached SH and raw radial-basis values. It optionally
  caches the small kernel tensor `(edge,l,outer,k)` and its radial derivative;
  it never stores an `m`-expanded mixer tensor.
- `dev_src/mtpr_sh.cpp::CalcSHMomentValuesWithGradientCache` and the mixed-loss
  tangent path propagate mixer coefficients, mixer scaling, and gate-scalar
  adjoints needed by BFGS fitting.
- `dev_src/mtpr_sh.cpp::CalcSHSiteEnergyDers`: after product-graph backprop,
  propagates adjoints through \(M_l(r)uY\), then sends \(\bar u\) through the
  existing two-layer gate graph.
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
  `(lmax + 1, outer_k_rank, kmax, superlinear_basis_size)`.
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

For Kokkos, keep the parameter layout `l -> outer -> k -> radial_basis` and
mirror it as a device view. In the force kernels, compute `M*u` per edge and
per `l`, using a small local array over `R_outer`.

## Performance layout

The hot loop is grouped by `l`:

1. load the `K` pair scalars \(u_{e,kl}\);
2. evaluate or load the required raw basis block for each `(k,l)`;
3. form and contract \(\sum_k M_{l,a,k}(r_e)u_{e,kl}\) for every `a`;
4. multiply the contracted `a` values by all `2l+1` SH components.

This avoids a full dense edge tensor of shape
`edge_count x l_count x R_outer x K x m_count`. Temporary forward storage is
only `R_outer`, plus the already available radial basis vectors. When the
edge cache is active, the cached tensor is `(edge,l,R_outer,K)`; coefficients
remain live contiguous parameters and are not baked into saved model state.

After hoisting the `m`-independent work, the extra forward arithmetic cost is
approximately

\[
\sum_l \left[
R_\mathrm{outer}K B_s
+ R_\mathrm{outer}K
+ (2l+1)R_\mathrm{outer}
\right]
\]

multiply-adds per edge, excluding raw-basis evaluation already shared through
the scaling-map cache. Removing the old `K -> R_gate` edge compression both
matches the intended mathematics and eliminates one dense contraction from
the hot path.

## Reproducible fresh initialization

`train ... --init-sh` accepts `--init-random-seed=<uint64>`.  The common
nonlinear coefficient stream is generated first.  The optional mixer stream is
drawn only afterward with
`--two-layer-gate-radial-mixing-init-scale=<nonnegative double>`.  Thus A
(mixer off), B0 (mixer enabled at zero), and B (active mixer) have bitwise
identical common initial coefficients for the same seed.

With an explicit seed, `--init-params=random`, and `--max-iter=0`, the trained
output is saved immediately after randomization and MPI broadcast, before
shift correction, rescaling, any linear solve, readout initialization, or
BFGS.  This is the required source for a genuine from-scratch A/B experiment;
an upgraded trained checkpoint is not a fresh comparison.

`dev_test/sh_init_random_seed_radial_mixing_check.sh` verifies repeatability,
different-seed separation, common-block identity including the multiline base
radial block, exact zero B0 coefficients, bounded active B coefficients, and
strict CLI validation.  The VO2 server protocol is recorded under
`dev_test/vo2_superlinear_fresh_ab/`.

## Acceptance criteria

1. With the feature disabled, all existing release3 rank/gate tests pass with
   unchanged outputs.
2. With the feature enabled and all `two_layer_gate_radial_mixing_coeffs = 0`,
   energy, force, and stress match the disabled model to roundoff.
3. Standalone C++ finite differences validate gradients with respect to
   positions, lattice, gate-rank weights through the upstream scalar graph,
   mixer weights, base/gate radial weights, type/additive weights, and all
   mixer scaling-map layouts.
4. C++ finite differences validate force and stress for at least one model with
   `two_layer_gate_k_rank > 1`, `outer_k_rank > 1`, and nonzero superlinear
   weights.
5. Energy-only, force-only, stress-only and combined EFS directional
   derivatives agree with central finite differences on an active nonzero
   model.
6. Serialization round trip preserves the `direct-k` marker and all weights
   exactly; the coefficient count is `(lmax+1)*outer_k_rank*kmax*rb_size`.
7. Old release3 models load in the new binary without adding implicit weights.
8. Historical unmarked gate-rank mixer models and old binaries reject the new
   layout clearly; neither may reinterpret it silently.
9. LAMMPS/Kokkos parity is required before using superlinear models outside the
   standalone executable; this repository does not silently claim that parity.
10. Cache-on/cache-off parity must cover neighborhood cache, edge primitive
    cache, force derivative cache, and stress.
