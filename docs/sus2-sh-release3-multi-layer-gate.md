# SUS2-SH Release3 Multi-Layer Gate

## Source Lineage

Release3 is branched from the corrected release2 server source tree:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-release2-loss-guard-work-codex
```

The release3 server working tree is:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-release3-work-codex
```

The release2 LAMMPS interface binaries are intentionally unchanged:

```text
/work/phy-weigw/app/release2/lmp.ml-sus2_release2_cpu
/work/phy-weigw/app/release2/lmp.ml-sus2_release2_kokkos_sm80
```

Loss-guard is a training-code feature and is not part of the MD interface
contract.

## Mathematical Structure

Release2 uses an environment-sensitive outer endpoint coefficient

```text
c^{out,eff}_{j,\mu} = c^{out}_{Z_j,\mu} + a^{out}_{Z_j,\mu} s_{j,\mu}.
```

Release3 keeps that form and makes the inner pair endpoint coefficient used by
the first gate environment-sensitive as well:

```text
c^{g1,eff}_{j,\mu} = c^{g1}_{Z_j,\mu} + a^{g1}_{Z_j,\mu} t_{j,\mu}.
```

Here `s_{j,mu}` is the first-gate scalar signal and `t_{j,mu}` is the
second-gate scalar signal. Both signals are built in the same style as the
outer PES scalar expansion: radial functions and spherical harmonics form edge
features, scalar products are selected by body order, and the selected scalar
terms are contracted per `mu`.

The three radial contractions share the same global radial nonlinear
parameters `s` and `scal`:

```text
outer PES radial layer
first gate scalar layer
second gate scalar layer
```

Other layer-specific parameters are independent: type coefficients, additive
coefficients, scalar weights, and body-order mixing weights.

## CLI

The new release3 flags are:

```text
--multi-layer-gate
--second-layer-gate-body-order=<int>
--second-layer-gate-mode=mu-body-linear-combo|mu-scalar-full
```

`--multi-layer-gate` implies `--two-layer-gate` for `init-sh`.
`--second-layer-gate-body-order` defaults to the outer `--body-order`, and must
be in `[2, --body-order]`.

The first-gate body-order default is also aligned with the outer body-order in
release3. Explicit older `--two-layer-gate-body-order` settings are still
accepted.

## Serialization

Release2 model files remain compatible when `multi_layer_gate_enabled` is not
present. Release3 multi-layer models add the second-gate metadata and parameter
blocks after the release2 first-gate block and before residual/species/linear
coefficient data.

The release3 coefficient layout is:

```text
shared radial nonlinear coefficients
first-gate radial/type/additive/scalar/body-mix coefficients
first-gate edge-L1 coefficients
second-gate radial/type/additive/scalar/body-mix coefficients
residual E0 coefficients
linear readout coefficients
```

The first-gate edge-L1 path is not enabled together with multi-layer gate in
this implementation.

## Development Checks

The release3 smoke workflow is:

```text
dev_test/sh_release3_multi_layer_gate_smoke.sh
```

The server test directory used during implementation is:

```text
/work/phy-weigw/hyx/xxx-b/6.17/6.24/update1/update2/update3/update4/update5/release3
```

Implementation smoke results on the release3 server mirror:

```text
binary: /work/phy-weigw/20260321_Test/SUS2-SH-release3-work-codex/bin/mlp-sus2
binary SHA-256: 4b30677d3831cb2a9069bc46be166ebaa5f3c710b21d5a573ea40b2e72b4c8ce
build mode: non-IPO MPI build with -DMLIP_MPI -DMLIP_INTEL_MKL -O3 -std=c++11
```

Checked:

```text
dev_test/sh_release3_multi_layer_gate_smoke.sh
checked_coeffs=287 coeff_count=287 abs_err=0.00183662811573 rel_err=0.000311410923406
```

Additional coordinate finite-difference smoke:

```text
check-efs-fd-dev release3_multi_gate_probe.mtp one.cfg --max-configs=1 --max-atoms=1
force_abs_err=7.1054273576e-11
stress_abs_err=4.71228099703e-10
stress_rel_err=0.000237772333266
```

The Intel MPI wrapper relink of the final source snapshot timed out in the
final `mpiicpc` link step during this development pass. The tested binary above
contains the release3 multi-layer gate implementation used by the smoke checks;
the later source-only edits are diagnostic output and formatting.

## Exact Optimization Notes

First exact release3 multi-gate optimization pass:

```text
scope: gate value and gate-seed memory flow
math: unchanged
```

Changes:

```text
reuse object-level scratch buffers for full/body-combo gate seed projections
avoid zero-filling gate value buffers immediately before full overwrite
avoid zero-filling GEMM output buffers immediately before beta=0 GEMM
cache second-gate neighbor signal pointers in first-gate hot loops
```

Server build after this pass:

```text
binary: /work/phy-weigw/20260321_Test/SUS2-SH-release3-work-codex/bin/mlp-sus2
binary SHA-256: 9dc53e10ee9260e9eb5140e55a144e3a213cea80d00a1435b633ab3cc345ba97
```

Checked:

```text
dev_test/sh_release3_multi_layer_gate_smoke.sh
checked_coeffs=287 coeff_count=287 abs_err=0.00183662811573 rel_err=0.000311410923406

check-efs-fd-dev release3_multi_gate_probe.mtp one.cfg --max-configs=1 --max-atoms=1
force_abs_err=7.1054273576e-11
stress_abs_err=4.71228099703e-10
stress_rel_err=0.000237772333266
```
