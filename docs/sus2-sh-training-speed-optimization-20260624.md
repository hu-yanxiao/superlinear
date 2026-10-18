# SUS2-SH Training Speed Optimization 2026-06-24

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

## Scope

This note records an exact `TrainLinear` optimization for SUS2-SH gate-mode
training. The change does not alter the loss, optimizer state, model format, or
training commands.

## Implemented Change

`MTPR_trainer::AddToSLAE` now accumulates the linear normal matrix as a
symmetric matrix:

- rank-1 rows use `cblas_dsyr` instead of full `cblas_dger`;
- force and stress row blocks use `cblas_dsyrk` instead of full `cblas_dgemm`;
- MPI `TrainLinear` reduction packs and reduces only the upper triangle, then
  unpacks it before the existing `SolveSLAE()` path.

This is mathematically the same normal equation matrix,

```text
H = sum_r alpha_r a_r a_r^T
```

but only the upper triangle is materialized and communicated. `SolveSLAE()`
still calls `SymmetrizeSLAE()` before solving.

## Validation

Server build tree:

```text
<historical-path>
```

Correctness command:

```bash
bash /work/phy-weigw/20260321_Test/run_train_speed_correctness.sh
```

Passed checks:

- `dev_test/sh_two_layer_gate_loss_gradient_check.sh`
- `dev_test/sh_two_layer_gate_mu_scalar_full_loss_gradient_check.sh`
- `dev_test/sh_two_layer_gate_double_mode_check.sh`
- `dev_test/sh_two_layer_gate_force_fd_check.sh`
- `dev_test/sh_two_layer_residual_loss_gradient_check.sh`
- `dev_test/sh_two_layer_residual_force_fd_check.sh`
- `dev_test/sh_two_layer_gate_shared_radial_check.sh`
- `dev_test/sh_two_layer_gate_mu_body_order_check.sh`

## Performance Evidence

Benchmark script:

```text
/work/phy-weigw/hyx/xxx-b/test/train_speed_tangent_force_ab.lsf
```

Queue/layout:

```text
1t88c, 96 MPI ranks, span[ptile=96]
```

The benchmark used `force-weight=0.1`, `stress-weight=0`, and the existing
`l4k4` gate training command from `/work/phy-weigw/hyx/xxx-b/test`.

Results:

```text
job 3834844, dsyr/dsyrk only:
  main runtime: 129 s
  opt  runtime: 127 s
  TrainLinear build_sum: 78.49 s -> 76.99 s

job 3834854, dsyr/dsyrk + packed upper reduction:
  main runtime: 130 s
  opt  runtime: 127 s
  TrainLinear build_sum: 77.53 s -> 76.42 s
  TrainLinear reduce_sum: 16.81 s -> 16.27 s
```

For job `3834854`, accepted-step loss and MAE lines were identical between the
main and optimized binaries. Final `main_force_p.sus2` and `opt_force_p.sus2`
were byte-identical by SHA256.

## Notes

The measured speedup is small but repeatable on this benchmark. The largest
remaining general opportunities are outside this patch:

- BFGS line-search strategy and cache reuse for rejected/accepted trials.
- A compact batched SH/gate product-graph evaluator for repeated gate
  forward/reverse/JVP walks.
- More targeted profiling of `TrainLinear` packing versus MPI transfer on
  larger linear systems.
