# Linear initialization checks

`native_tests/linear_components_benchmark.cpp` links against the native MPI
objects, excluding the CLI main. It evaluates a frozen model without an optimizer:

```text
driver MODEL CFG OUTPUT.dump [repeats=1] [max_configs=1] [warmup=0]
```

Only `CalcEFSComponents` is timed. The dump contains every energy, force and
virial feature column, plus labels. Inputs and physical model coefficients must
remain unchanged. The Python checker also compares the complete weighted normal
matrix and right-hand side for E/F/V weights 1/0.1/0.01.

```bash
python dev_test/linear_components_make_cases.py --binary bin/mlp-sus2 --out CASES
python dev_test/linear_components_benchmark_check.py run \
  --driver DRIVER --cases CASES/cases.json --out RESULTS \
  --reference-env SUS2_SH_LINEAR_GATE_ADJOINT_FAST=0 \
  --reference-env SUS2_SH_LINEAR_FLAT_DERS=0 \
  --candidate-env SUS2_SH_LINEAR_GATE_ADJOINT_FAST=1 \
  --candidate-env SUS2_SH_LINEAR_FLAT_DERS=1 --require-coverage
```

The gate fast path stores the scalar-to-basic Jacobian once per center and
contracts each neighbor/channel tangent with it. This preserves all scalars and
channels. Pure-matrix, edge-L1, multilayer and product-row modes, or scratch
requirements above 64 MiB, use the reference path. The flat derivative path
only changes traversal of contiguous neighbor/Cartesian coordinates.

Qualification on 2026-10-26 covered 16 frozen L1–L4 static/effect cases with
shared/species readout, finite differences, and a 120-atom production frame.
All feature columns and normal equations passed `atol=rtol=1e-10` against the
reference paths and across Intel classic/GCC builds. A nonzero shared static
model also passed against SUS2-SH release3; its old reader cannot load the
species-readout format. The shared readout reconstruction check uses a separate
unit-center-prefactor model because the legacy linear chart differs from the
physical nonunit-prefactor chart. Original nonunit models are retained for the
feature and derivative checks.

The qualified deployment uses GCC 11.2.0, Intel MPI/MKL and
`-O3 -march=skylake-avx512 -ffast-math`; GNU configure defaults to `-march=native`.
On the same CPU, gate reuse reduced the effect production-frame feature build
from 24.9 s to 6.2 s. This is an initialization-kernel measurement, not a full
BFGS iteration speedup. Switching the compiler also affects performance;
neither AVX-512 nor a particular compiler is universally faster.
