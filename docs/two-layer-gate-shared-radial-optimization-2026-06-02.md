# Two-layer gate shared-radial optimization notes (2026-06-02)

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

Repository branch: `developer`

Server tree:

```text
<historical-path>
```

Server binary after the kept changes:

```text
<historical-path>
sha256: 82048c3493ff72b9e83455f74175b72ded487ad3ce14e0e502ce146adb9e19dd
```

## Kept changes

The kept changes only affect the two-layer gate evaluation path.

- Added two-level directed-edge primitive cache capability:
  - full cache for parameter-gradient paths,
  - coordinate-only cache for forward/linear component paths.
- Reused the directed-edge primitive cache in active two-layer `CalcSHBasisFuncsDers`.
- Replaced the temporary `gate_basis_ders` matrix in linear component building with direct accumulation into per-atom gate linear adjoints.
- Skipped full `basis_ders` zeroing only when evaluating an active two-layer gated basis; all scalar derivative rows are overwritten in that path.
- In `AccumulateSHBasisGateDers`, reused the row-product program locally for
  the per-edge gate adjoint propagation. This avoids clearing the full
  moment-derivative buffer for every directed edge while keeping the original
  fallback path for builds without product rows.

## Rejected attempts

These were compiled/tested but not kept.

- Fusing gate JVP into `CalcSHBasisFuncsDers` increased `main_components` and total linear build time.
- Forcing product-row evaluation in active gated basis was mathematically equivalent but slower on the 500-sample benchmark.
- Directly accumulating terminal scalar gate adjoints avoided one intermediate store but increased `gate_adjoints` time from about 90 ms to about 130 ms.
- Inlining the energy-only gate adjoint `dE_i/dg_j` into the main nonlinear
  gradient traversal removed the `avg_us_energy_adjoint` replay bucket, but
  increased the following gate-chain work enough to make total gradient time
  slower. On the fair 500x55 profile, the reference
  `skip_basis_zero` run had `avg_us_total ~= 141.7 ms`; the inline-adjoint
  run reached `avg_us_total ~= 154.9 ms`.
- Caching the first-layer gate reverse adjoint `df/dM0` in
  `PrepareTwoLayerGateValues` was FD-correct but not profitable. It moved
  work into `prepare_gate` and did not materially reduce `tangent_grad` or
  `scalar_param`, so the combined profile stayed slower than the kept path.

## Verification

Server smoke/FD checks passed:

```text
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_residual_force_fd_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_residual_loss_gradient_check.sh
bash .local_tmp/shared_radial_loss_gradient_check.sh
./bin/mlp-sus2 check-linear-components-fd-dev \
  .local_tmp/sh_two_layer_gate_shared_radial_loss_gradient/shared_nonzero.sus2 \
  .local_tmp/sh_two_layer_gate_shared_radial_loss_gradient/one.cfg \
  --max-configs=1 --max-atoms=1 --displacement=1.0e-5 \
  --abs-tolerance=5.0e-5 --rel-tolerance=5.0e-4
```

The shared-radial non-residual loss-gradient check covered the gate-radial
coefficient segment:

```text
checked_coeffs=154
gate_radial_begin=86
gate_radial_end=110
abs_err=0
rel_err=0
```

The shared-radial linear-component FD check produced:

```text
checked_components=99
coeff_count=33
abs_err=1.11022310324e-11
```

Additional checks for the rejected energy-adjoint / `df/dM0` cache candidates:

```text
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
bash .local_tmp/shared_radial_loss_gradient_check.sh
./bin/mlp-sus2 check-linear-components-fd-dev \
  .local_tmp/sh_two_layer_gate_shared_radial_loss_gradient/shared_nonzero.sus2 \
  .local_tmp/sh_two_layer_gate_shared_radial_loss_gradient/one.cfg \
  --max-configs=1 --max-atoms=1 --displacement=1.0e-5 \
  --abs-tolerance=5.0e-5 --rel-tolerance=5.0e-4
```

Server binaries for these rejected candidates:

```text
inline energy adjoint: 6f6f1a272984ea8a249d68e5c29a8384b70d7b630cd138c85f106d74a902fef6
inline energy adjoint + df/dM0 cache: 16b2dd2e939c25a143cdb3a695799b5e4685c0d6697b48bc53a0e6492a8b2e92
```

## Performance evidence

500-sample active two-layer shared-radial profile command was based on:

```text
<historical-path>
```

Clean reference:

```text
<historical-path>
avg_us_total ~= 412635
avg_us_main_components ~= 229895
avg_us_gate_adjoints ~= 90148
TrainLinear build_s at step 50 ~= 2.148 s
```

Kept direct/cache/basis-reuse profile:

```text
<historical-path>
avg_us_total ~= 375481
avg_us_main_components ~= 201274
avg_us_gate_adjoints ~= 90128
TrainLinear build_s at step 50 ~= 1.976 s
```

Kept skip-basis-zero profile:

```text
<historical-path>
avg_us_total ~= 369464
avg_us_main_components ~= 195490
avg_us_gate_adjoints ~= 90190
TrainLinear build_s at step 50 ~= 1.952 s
```

Kept row-local gate-adjoint profile:

```text
<historical-path>
exit_code=0
runtime_sec=534
avg_us_total ~= 366544
avg_us_main_components ~= 198300
avg_us_gate_adjoints ~= 82953
```

Net observed linear component build improvement versus the clean reference is
about 11% on this 500-sample, 120-rank profile. The final row-local change is
a small additional gain over the previous kept path: total linear component
build time improves by about 0.8%, while the two-layer gate-adjoint subpath
improves by about 8%.
