# Superlinear gate development checks

This file records the local, standalone checks for the release3 superlinear
gate paths.  In this branch the checked implementations include the per-`l`
distance-dependent gate-to-outer radial mixer
(`--two-layer-gate-radial-mixing=full`), the edge-conditioned `L=1` gate
(`--two-layer-gate-edge-l1`), and the first-gate scalar energy readout
(`--two-layer-gate-readout`).

## One-command local suite

Build the developer binary, then run:

```bash
bash dev_test/sh_superlinear_gate_regression_suite.sh
```

The suite assumes the binary is available at `./bin/mlp-sus2`, matching the
existing `dev_test` convention.  Set
`SUS2_SH_SUPERLINEAR_INCLUDE_PROFILE=0` to skip profile-log smoke checks, or
`SUS2_SH_SUPERLINEAR_INCLUDE_READOUT=0` to skip the independent gate-readout
regressions.

## Coverage matrix

| Risk | Check |
| --- | --- |
| Old two-layer gate model still loads and predicts | `dev_test/sh_two_layer_gate_edge_l1_init_check.sh` |
| Zero new weights preserve old predictions | `dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh` |
| New term is numerically active on an asymmetric structure | `dev_test/sh_two_layer_gate_edge_l1_forward_check.sh` |
| Parameter gradients match finite differences | `dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh` |
| Energy/force/stress coordinate derivatives match finite differences | `dev_test/sh_two_layer_gate_edge_l1_efs_fd_check.sh` |
| Per-`l` radial mixer metadata and zero-weight backward compatibility | `dev_test/sh_two_layer_gate_radial_mixing_check.sh` and the `two_layer_gate_radial_mixing` suite block |
| Active mixer changes E/F/stress and survives save/load exactly | `two_layer_gate_radial_mixing` suite block |
| E-, F-, S-, and EFS-loss gradients cover base species/scaling/radial, gate radial/additive/type/weights, gate rank, outer rank, and mixer coefficients | `two_layer_gate_radial_mixing` suite block |
| Active mixer coordinate finite differences and rotation invariance/equivariance hold for E/F/stress | `two_layer_gate_radial_mixing` suite block |
| Linear component readout remains consistent with direct E/F | `dev_test/sh_two_layer_gate_edge_l1_linear_components_check.sh` |
| Rotation invariance/equivariance holds for energy, force, and stress | `dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh` |
| Pair/weighted cache profile instrumentation is present and disable switches work | `dev_test/sh_two_layer_gate_edge_l1_pair_cache_profile_check.sh`, `dev_test/sh_two_layer_gate_edge_l1_weighted_cache_profile_check.sh` |
| Gate-readout descriptors, pair energy, LMP-table route, and zero-gate coordinate FD are covered | `dev_test/sh_two_layer_gate_readout_diagnostics_check.sh`, `dev_test/sh_two_layer_gate_readout_zero_gate_efs_fd_check.sh` |

For performance claims beyond smoke-level instrumentation, use the established
server-side matched profile workflow rather than local laptop timing.

The active mixer fixture deliberately makes every checked parameter block
nonzero. This prevents a zero-initialized residual from producing a vacuous
finite-difference pass. Its rotated CFG also rotates positions, forces and the
full symmetric stress tensor.
