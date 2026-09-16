# Adaptive radial projection and source-species amplitudes

For generalized retained rank and independent `tanh`, see
[ADAPTIVE_RADIAL_RANK_ACTIVATION.md](ADAPTIVE_RADIAL_RANK_ACTIVATION.md).
The formulas below remain the backward-compatible `d=1`, activation `none`
special case.

Two independent options extend the adaptive-coefficients model. Existing
models and commands retain `full` projection and `shared` amplitudes.

```
--adaptive-radial-projection=full|shared-n
--adaptive-radial-amplitude-mode=shared|species
```

They require `--superlinear-mode=radial-coefficient-gate`. They do not change
gate/outer graph pruning, gate rank, outer rank, element-pair radial scaling,
shared/species energy readout, or optional tensor messages. A resume command
must match the stored options; changing them is not an implicit conversion.

## Mathematics and layout

Let `L = l_max + 1`, `O = outer rank`, `N = radial basis size`, `Q = gate
scalar count`, and `S = species count`. The gate scalars `s[j,q]` belong to
the environment centered on source atom `j`. The radial coefficient on an
edge received by atom `i` is

```
d[j,l,o,n] = c[l,o,n] + a[type(j),l,o,n] * h[j,l,o,n]
R[i,j,l,o] = sum_n d[j,l,o,n] * T[n](r[i,j]; scaling[type(i),type(j)])
```

Here `T` denotes the existing evaluated radial basis including its fixed
model/basis scale factors. This is the same physical convention as extracting
the model-wide `scaling` outside the sum in the GPUMD implementation; it is
not an extra trainable multiplier or a second application of that scale.

Shared amplitudes omit the first index. Full projection is

```
h[j,l,o,n] = sum_q U[l,o,n,q] * s[j,q].
```

Shared-n projection replaces it with

```
h[j,l,o] = sum_q U[l,o,q] * s[j,q]
d[j,l,o,n] = c[l,o,n] + a[type(j),l,o,n] * h[j,l,o].
```

The source type is `type(j)`, not the receiving atom's type. The existing
outer element prefactors still multiply the complete radial contribution.
No feature, coefficient, or force is threshold-pruned.

| Option | Physical stored layout | Count |
| --- | --- | --- |
| full projection | U[L,O,N,Q] | L O N Q |
| shared-n projection | U[L,O,Q] | L O Q |
| shared amplitude | a[L,O,N] | L O N |
| species amplitude | a[S,L,O,N] | S L O N |

The last index is contiguous. Base radial coefficients retain their existing
native `mu = o*L+l` serialization; the new arrays use the layouts above.
For the VO2 L4/O8/N10/Q348 case, projection parameters fall from 111,360
to 11,136 and species amplitudes grow from 320 to 640.

Shared-n is an intentional structural restriction, not an exact general
compression of a trained full projection: at fixed `(l,o,type)`, the
effective response matrix over `(n,q)` has rank at most one. Different
source species can have different radial shapes through `a`, while sharing
the gate-feature direction `U[l,o,:]`.

## Exact execution and derivatives

The forward matrix product has only `P=L*O` output columns for shared-n,
versus `P=L*O*N` for full. The small result is then expanded into the cached
physical radial coefficients `d`. U is never expanded to a full matrix.

For coefficient adjoints `bar_d`, the shared-n reverse is

```
bar_a[type(j),l,o,n] += bar_d[j,l,o,n] * h[j,l,o]
bar_h[j,l,o] = sum_n a[type(j),l,o,n] * bar_d[j,l,o,n]
bar_U[l,o,q] += sum_j bar_h[j,l,o] * s[j,q]
bar_s[j,q] += sum_l,o U[l,o,q] * bar_h[j,l,o].
```

The n reduction occurs before both reverse matrix products. Force/virial
training additionally propagates coordinate tangents through the same
projection and reduction. Amplitude gradients include both `bar_d*h` and
`bar_d_tangent*h_tangent`; U gradients include the corresponding scalar
tangent product. No derivative divides by an amplitude or signal, so zero
values retain their correct gradients. All source-environment contributions
continue through the existing gate geometric reverse to all affected atoms
and the virial. Linear-component evaluation uses the same compact reduction
independently for every readout head.

## Initialization and training coordinates

The existing data-calibrated initialization remains in use. Gate scalar RMS
is shared over all source species. Shared-n initializes one U vector per
`(l,o)` using the old full layout's n=0 deterministic seed index. The response
RMS is calibrated after contraction with the actual radial basis. One
amplitude per `(l,o)` is then copied across all n and all species, without
additional random draws. The outer-moment normalization multiplies c and
all species amplitudes by the same per-`(l,o)` factor.

Amplitude optimizer-scale metadata follows the physical amplitude layout.
Projection optimizer-scale metadata remains one positive finite Q-vector,
shared across n, `(l,o)`, and species. It represents a fixed optimization
coordinate chart and does not alter the exported physical coefficients.

The response L2 prior uses `a*U / q_scale`, averaged over the physical
`(l,o,n,q)` response entries and, when enabled, over species. The radial
smoothness prior uses the same effective response coefficients, averaging
over element pairs with the source species selecting a. Thus copying shared
amplitudes to all species preserves both prior values and the corresponding
sum of species gradients. Compact U gradients sum all reused n/species
contributions; penalties are not norms of the redundant factors separately.

## Model format

Any nondefault option is written as MTP version `1.5.0`, with these mandatory
headers in order after `outer_radial_mode`:

```
readout_mode = shared|species
adaptive_radial_projection = full|shared-n
adaptive_radial_amplitude_mode = shared|species
tensor_message_lmax = 0|1|2
```

The existing `adaptive_radial_amplitude_count`, `adaptive_radial_amplitudes`,
`adaptive_radial_weight_count`, and `adaptive_radial_weights` fields use the
physical counts above. Optional amplitude scales have amplitude-count
entries; optional weight scales have Q entries. Tensor B/count/scales retain
their v1.4 position after adaptive optimizer metadata, and are required for
positive tensor lmax (scales optional) and forbidden for lmax zero.

The reader accepts explicit v1.5 full/shared models. The native writer
canonicalizes the version from active features, so a double-default model
can be saved with its older compatible version. Versions 1.1--1.4 retain
the previous meanings and reject the new mode headers. Version 1.5 rejects
missing, duplicate, misplaced, unknown, nonfinite, incorrectly sized, or
malformed new metadata rather than silently ignoring it. Actual supported
runtime modes must be verified in each consumer before installing an export.

## Verification boundary

This document describes implementation semantics, not a performance result.
Server compilation, forward/EFS parity, full mixed parameter derivatives,
regularization finite differences, format rejection, initialization, and
legacy regression tests are required before deployment. Scientific execution
and compilation are performed on server compute jobs, not on the workstation.
