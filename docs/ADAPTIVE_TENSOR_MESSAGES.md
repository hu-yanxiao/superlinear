# Optional tensor environment messages

This option augments the adaptive radial-coefficient-gate model. It is not a
replacement for either gate or outer scalar graph.

## Mathematical definition

Let `z[j,g,L,M]` be the existing gate basic moments after the K-to-G projection,
before the invariant product graph. For selected complete irreps `L=1..Lmsg`,

```
H[j,o,L,M] = sum_g B[L,o,g] z[j,g,L,M]
A[i,o,L,M] = sum_j v[i,o,L] v[j,o,L] R[i,j,o,L]
                   (Y[L,M](u_ij) + H[j,o,L,M])
```

`R` is the existing environment-dependent radial function, including its
unchanged scalar-gate dependence. Other angular orders use the old expression.
There is no tensor bias, component activation, extra amplitude, or additional
radial evaluation. `B` mixes multiplicity channels only and is independent of
`M`; therefore `H` transforms in the same real spherical-harmonic irrep as
`Y`. All `2L+1` magnetic components and natural parity are retained. An invariant
outer contraction followed by either shared or species readout consequently
preserves rotations, reflections, translations, and atom permutations.

Both gate graph choices and both outer graph choices remain independent:
`same-k` or `q-total` can be selected in any combination. Existing restrictions,
including `O <= K`, still apply; there is no new restriction `G=O` or `O<=G`.

## Option, initialization, and optimizer coordinates

Use `--tensor-message-lmax=0|1|2` with
`--superlinear-mode=radial-coefficient-gate`. The default is `0`, disabling this
feature. A positive value must not exceed `--l-max`. A loaded model inherits its
saved value when the option is omitted; explicitly mismatched values are errors.

Fresh physical `B` coefficients are zero, without consuming random numbers.
Thus enabling the option initially retains the previous function; derivatives
with respect to `B` can nevertheless be nonzero. There is no extra `B` penalty
in this version. All previous loss and regularization terms are unchanged.

Training can use one fixed RMS for every complete `(L,g)` irrep:

```
rho[L,g]^2 = sum_(j,M) z[j,g,L,M]^2 / (N_atoms * (2L+1))
B[L,o,g] = Theta[L,o,g] / rho[L,g]
dLoss/dTheta = (dLoss/dB) / rho[L,g]
```

The positive finite guarded scale `1/rho` is optimizer metadata, not a
prediction-time operation. Native files always contain physical `B`; changing
optimizer coordinates cannot change energy, force, or stress. The same scale is
used for all `M` and all receiving `o` of a source `(L,g)`. This is essential for
equivariance. A per-magnetic-component normalization is not allowed.

## Layout and compatibility

Enabled models use `version = 1.4.0`. Required fields are explicit `readout_mode`,
`tensor_message_lmax`, `tensor_message_coeff_count`, and `tensor_message_coeffs`.
Physical coefficient order is `((L-1)*O+o)*G+g`, with `Lmsg*O*G` parameters.
`tensor_message_optimizer_scales` is optional, uses the same count/order, and
defaults to one; there is no separate scale-count field.

The header places `tensor_message_lmax` immediately after `readout_mode`. The
arrays follow the adaptive radial arrays and their optional optimizer scales.
Missing, duplicate, invalid, truncated, excess, or nonfinite tensor metadata is
rejected, as are incompatible model modes and active fields in older versions.
Disabling the feature retains previous 1.1/1.2/1.3 serialization semantics.

Current GPUMD binaries do **not** support version 1.4 tensor messages. They must
reject such models; renaming the version does not produce a compatible model.
This native/PyTorch feature does not implement a GPUMD interface.

## Execution and chain rule

The prepared gate cache retains selected `z` primitives. `H` is formed once per
atom, then `Y+H` is fused into the existing edge/radial loop. There is no
edge-expanded `H` cache or nested traversal of each neighbor's neighborhood.

For a force/virial loss directional derivative,

```
dotH = B dotz
dotA = sum_j vv [dotR (Y+H) + R (dotY+dotH)]
dLoss/dB = sum_(j,M) [barH z + bardotH dotz]
```

All receiving-edge adjoints are accumulated before the node pullback. Scalar
seeds through `R` and direct equivariant `z` value/tangent seeds traverse the
gate reverse/HVP together. This includes derivatives through gate projection,
radial coefficients, element coefficients, scaling, and atomic geometry.
Direct `z` adjoint scratch is one atom's moment buffer, not an atom-by-full-graph
tensor. Linear-component calculations reset the new adjoints for every readout
head and retain both batched and reference paths.

The default linear-solver component chart remains the unweighted `Phi` chart
used by `TrainLinear`. For diagnostic reconstruction of an existing shared
readout with unequal center factors `h_Z`, the optional fixed-center chart uses
`sum_i h_Zi Phi_iq`; the center factors are held fixed, not differentiated in
these columns. `check-linear-readout-dev` selects this chart automatically for
adaptive shared models. `check-linear-components-fd-dev
--fixed-center-readout-columns` checks its geometry derivatives independently.
Species readout has no multiplicative `h_Z`, so its columns are unchanged.
The element-constant columns continue to be atom counts with coefficient
`shift_Z+h_Z`. Neither chart changes the physical model or training defaults.

For `G=O=8`, enabling `L=1,2` adds 128 parameters and 64 message components per
atom. Leading new node work is 512 multiply-adds per atom, plus 64 additions per
edge in the fused angular expression. These operation counts are not measured
end-to-end speed claims; gradient work, cache traffic, and graph costs also
matter.

Server validation is provided by `dev_test/adaptive_tensor_message_check.py`
and `dev_test/adaptive_tensor_message_integration.py`. It includes nonzero
coefficients, finite differences of energy/force/virial losses, symmetry,
zero-message parity, graph/readout combinations, serialization, native/PyTorch
export parity, and real optimizer updates. The historical FP32 tiny-step
finite-difference limitation is separate and is not resolved by this feature.
