# Separate radial support for l=0 and l>0

SH models support exactly two physical radial cutoffs, shared by their inner and
outer layers. The existing `max_dist` is the l=0 cutoff and the neighbor-search
radius. Optional `angular_max_dist` is the cutoff for every l>0 channel:

```text
max_dist = 6.5
angular_max_dist = 5.0
radial_basis_size = 8
```

The new value must be finite and satisfy `0 < angular_max_dist <= max_dist`.
Omission means the original single-cutoff model. Equal values keep the original
path. This does not change K/R/O, the product graph, mixed-l couplings, or the
parameter count. It does change spatial support: train or retrain with this
setting rather than relabeling an existing model as an equivalent one.

## Configuration

Add this to a native training YAML model group:

```yaml
model:
  # Keep the other settings from train-effect.yaml / train-static.yaml.
  neighbor_cutoff: 6.5
  angular_cutoff: 5.0
```

For `init-sh-model`, add `--cutoff=6.5 --angular-cutoff=5.0` to the existing
initialization command. The PT initializer accepts the same CLI options and
`SHInitOptions(cutoff=6.5, angular_cutoff=5.0, ...)`.

Use the exported model with the updated C++, LAMMPS, Python or GPUMD reader.
Older readers are not qualified for this metadata and may reject or silently
ignore it. All simulation endpoints must be upgraded together.

```lammps
pair_style sus2/embedded model.sus2 radial_mode direct
pair_coeff * *
# Alternative, after checking interpolation accuracy for this model:
# pair_style sus2/embedded model.sus2 radial_mode table tabstep 0.001
```

The cutoffs come from the model, not extra LAMMPS pair-style arguments.
The native Python calculator uses the same C++ implementation; metadata reports
`radial_cutoff` and `angular_cutoff` in angstroms. `cutoff` remains the full
neighbor radius, including any longer optional ZBL term.

## Mathematics and execution

Each radial primitive is `f(r; rc_l) P_n(r, sigma, shift)`, with `rc_l=max_dist`
for l=0 and `rc_l=angular_max_dist` otherwise. Values and coordinate derivatives
are exactly zero outside their own support. Warp coordinates and original
radial basis normalization are unchanged. The legacy quadratic envelope has
a continuous first derivative; the optional native polynomial envelope retains
its existing higher-order smoothness.

The new split supports `RBChebyshev_sss` and `RBLaguerre_log1p`. Other historical
native basis types keep single-cutoff support and reject unequal cutoffs.
Non-SH native models also reject unequal cutoffs.

- CPU kernels select the active shell range once per edge. Far edges evaluate
  only l=0 harmonics, radial contractions and coordinate derivatives. Equal
  numerical warps reuse the raw radial recurrence before applying envelopes.
- Far l=0 edges retain gate/Adaptive gradients through the full inner
  environment. The product graph is not pruned or restricted to radial scalars.
- Embedded interpolation uses separate keys, endpoints and contexts for the
  two grids. Packing combines only equal-grid lanes in constant admission time.
  Both grids share the existing memory budget; insufficient capacity or
  sub-minimum-distance evaluation falls back to exact direct evaluation.
- PT training caches short-edge indices with immutable training graphs and
  gathers compact short-edge tensors before angular computation. General
  prediction rebuilds indices for current coordinates. Mixed derivatives,
  normalization and radial regularization use the matching support. The new
  split path explicitly falls back from compiled-edge execution to native
  eager autograd; unified-cutoff dispatch is unchanged.
- GPUMD retains shell-aware Static/Effect cooperative kernels and Adaptive
  basis reuse. Single-envelope Adaptive experimental candidates fall back to
  shell-aware kernels. There is no separate GPU neighbor partition yet: far
  lanes skip high-l work but may leave idle lanes in a cooperative block.
  Runtime acceleration must be measured, not inferred from the cutoff ratio.

## Verification

`native_tests/two_cutoff_radial_check.cpp` checks radial values, coordinate and
mixed parameter derivatives. `native_tests/two_cutoff_native_check.py` checks
E/F/virial, linear solve columns and regularization derivatives.
`interfaces/lammps/tests/two_cutoff_check.py` compares direct, interpolated and
zero-budget fallback EFS against independent PT fixtures in one/two halo
domains. `interfaces/python/tests/test_two_cutoff.py` checks the native Python
API. Cross-framework fixtures include nonzero environment effects, near/far
edges, both physical endpoints, and a separate periodic-image case.
