# SUS2-SH SO2 Edge L1 Gate Design

## Goal

Add a first-version edge-projected `L=1` gate to the existing SUS2-SH
two-layer gate. The new term must be exact, rotationally invariant after edge
projection, differentiable through coordinates and nonlinear parameters, and
disabled by default so old two-layer-gate model files remain readable.

## Mathematical Definition

The current gate builds an atom-local scalar signal for every target
`mu = (k, l)`:

```text
s0_mu(j) = existing two-layer-gate scalar/body signal for atom j and target mu
```

The edge-projected first-version term uses raw first-layer `l=1` moment
channels. For each source channel `nu = (k_source, 1)`:

```text
H_{nu,m}(j) = M^{(0)}_{nu,1m}(j),  m = -1, 0, 1
chi_nu(i,j) = sum_{m=-1}^{1} H_{nu,m}(j) Y_{1m}(rhat_ij)
```

For every target `mu`, the effective edge signal is:

```text
s_mu(i,j) = s0_mu(j) + sum_nu W_edge_l1[mu, nu] chi_nu(i,j)
```

The existing bounded gate activation remains unchanged:

```text
g_mu(i,j) = 1 + A tanh(a_{type(j)} s_mu(i,j))
```

The source set is deterministic and derived from the existing `mu` layout:

```text
nu(k) = k * (sh_l_max + 1) + 1,  k = 0, ..., sh_k_max - 1
```

This requires `sh_l_max >= 1`. In the first version, edge L1 is only valid for
`two_layer_gate_site_mode = neighbor`. `double` site mode is rejected because the
center projection convention must be designed separately.

## Exactness And Invariance

The scalar `chi_nu(i,j)` is the contraction of two real `l=1` objects in the
same real spherical-harmonic basis. Under a global rotation, both
`H_{nu,m}(j)` and `Y_{1m}(rhat_ij)` rotate by the same irreducible
representation, so their scalar contraction is invariant. The final energy
surface remains rotationally invariant because the final SH readout sees only
the scalar gate multiplier `g_mu(i,j)`.

The implementation must use the same real SH normalization for `H` and
`Y_1(rhat_ij)`. It must not replace the projection with Cartesian dot products
unless a source-level normalization proof is added.

## Reverse-Mode Contract

For an edge adjoint `bar_s_mu(i,j)` from the existing tanh gate derivative:

```text
bar_W_edge_l1[mu,nu] += bar_s_mu(i,j) chi_nu(i,j)
bar_chi_nu(i,j) += sum_mu bar_s_mu(i,j) W_edge_l1[mu,nu]
bar_H_{nu,m}(j) += bar_chi_nu(i,j) Y_{1m}(rhat_ij)
bar_Y_{1m}(i,j) += bar_chi_nu(i,j) H_{nu,m}(j)
```

The direct edge-coordinate contribution is:

```text
d chi_nu / d x = sum_m H_{nu,m}(j) d Y_{1m}(rhat_ij) / d x
```

The first-layer force-chain contribution is the reverse propagation of
`bar_H_{nu,m}(j)` through the same raw first-layer moment construction that
produced `H_{nu,m}(j)`.

For nonlinear parameter gradients, the directional tangent must include both
terms:

```text
delta chi_nu = sum_m delta H_{nu,m}(j) Y_{1m}(rhat_ij)
             + sum_m H_{nu,m}(j) delta Y_{1m}(rhat_ij)
```

The existing atom-major tangent buffer is not sufficient for the edge term. The
edge L1 tangent is edge-major or recomputed per edge because `chi_nu(i,j)`
depends on `rhat_ij`.

## Metadata And Compatibility

Old models do not contain edge L1 metadata. Loading them must set:

```text
two_layer_gate_edge_l1_enabled = false
two_layer_gate_edge_l1_source_mu_indices = {}
two_layer_gate_edge_l1_weights = {}
```

New models with the option enabled save:

```text
two_layer_gate_edge_l1_enabled = true
two_layer_gate_edge_l1_source_mu_count = sh_k_max
two_layer_gate_edge_l1_source_mu_indices = {1, 1 + (sh_l_max + 1), ...}
two_layer_gate_edge_l1_weight_count = radial_func_count * sh_k_max
two_layer_gate_edge_l1_weights = {0, ...}
```

The new weights are nonlinear coefficients placed after existing two-layer-gate
body-mix weights and before residual `E0` and linear coefficients:

```text
[base nonlinear][gate radial][gate additive][gate scalar weights][gate body mix]
[edge L1 weights][residual E0][linear mirror]
```

Zero `two_layer_gate_edge_l1_weights` must reproduce the current two-layer-gate
energy, forces, stresses, and coefficient gradients to numerical precision.

## Efficient Implementation Shape

The forward pass remains two-layer:

1. Build the existing scalar gate signal cache `s0_mu(atom)`.
2. If edge L1 is enabled, build `H(atom, source, m)` once per configuration.
3. For each final edge `(i,j)`, compute `chi(source)` from `H(j, source, m)`
   and cached or freshly evaluated `Y_1(rhat_ij)`.
4. Compute the dense micro-GEMV `edge_l1_signal_by_mu = W_edge_l1 * chi`.
5. Pass `s0_mu(j) + edge_l1_signal_by_mu` to the existing pair gate buffer.

The atom-keyed tanh cache is disabled for edge L1 signals because
`s_mu(i,j)` is edge-dependent. This preserves correctness first; a later exact
optimization can add an edge-keyed cache if profiles show the tanh path is hot.

The reverse pass stores two additional atom-major arrays:

```text
two_layer_gate_edge_l1_values_: atom * source_count * 3
two_layer_gate_edge_l1_adjoints_: atom * source_count * 3
```

Per-edge scratch arrays are reused inside the hot loops:

```text
edge_l1_chi[source_count]
edge_l1_signal_by_mu[radial_func_count]
edge_l1_chi_adj[source_count]
edge_l1_weight_grad[radial_func_count * source_count] accumulated into coeff grads
```

No approximation, pruning, low-rank factorization, quantization, or approximate
communication is part of this first version.

## Validation Contract

The first version is complete only when all of these checks pass on the server:

1. `init-sh --two-layer-gate --two-layer-gate-edge-l1` writes the new metadata
   with zero weights and correct source `mu` indices.
2. Loading an old model without edge L1 metadata succeeds and preserves the old
   metadata on save.
3. A model with edge L1 enabled and all edge L1 weights zero matches the same
   model without edge L1 for energy, forces, stresses, and loss gradients.
4. Nonzero edge L1 weights change the prediction on an asymmetric three-atom
   or larger configuration.
5. `check-efs-fd-dev` passes for nonzero edge L1 weights.
6. `check-loss-gradient-dev` passes for a coefficient range covering edge L1
   weights, gate additive coefficients, and gate radial coefficients.
7. A rotated CFG gives equal energies and rotated forces for nonzero edge L1.
8. Server profile in `/work/phy-weigw/hyx/xxx-b/so2-gate/` compares the same
   CLI/data with edge L1 off and on.
