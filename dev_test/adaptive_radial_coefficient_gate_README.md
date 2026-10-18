# Adaptive Radial Coefficient Gate Regression Notes

These tests target the new C++ `--superlinear-mode=radial-coefficient-gate`
path. They are intentionally standalone `dev_test` artifacts and do not import
production code.

Run from the repository root:

```bash
bash dev_test/adaptive_radial_coefficient_gate_check.sh ./bin/mlp-sus2
```

The Python body creates its own asymmetric two-species fixture under
`.local_tmp/`, initializes models with `K`, `L`, and `LK` scaling maps, covers
`O=1`, `O=2`, `O=K`, and `O=K` with gate-rank less than `K`, checks `O>K`
rejection against raw `K`, and exercises decoupled gate scaling. Every accepted
model is patched so base radial
coefficients `c`, adaptive weights `U`, adaptive amplitudes `a`, gate radial
coefficients, gate type coefficients, gate-rank weights, species coefficients,
scaling coefficients, and scalar head coefficients are all asymmetric and
nonzero.

Coverage in one invocation:

- model file format and roundtrip for `adaptive_radial_weights` and
  `adaptive_radial_amplitudes`
- rejection of the obsolete radial-mixing/additive/readout blocks in this mode
- cached versus recomputed E/F/stress parity
- coordinate finite differences for forces and stress
- mixed native-training loss gradients with energy, force, and stress weights
- one all-parameter finite-difference gradient with nonzero STD, adaptive
  radial smoothing, and gate-full L2 regularization
- rowwise a/U gauge-folding parity: `a[row] -> a[row] * s[row]` and
  `U[row,q] -> U[row,q] / s[row]` preserve E/F/stress and the CLI
  regularizer-only loss
- zero-`U` and zero-`a` branches, while retaining nonzero base `c`
- required named directional-gradient probes for adaptive tensor ranges via
  `adaptive_radial_weight_begin/end` and `adaptive_radial_amplitude_begin/end`
- full linear readout and linear component finite-difference checks, for both
  the default batched adaptive components and the reference component path;
  these use a generated unit-center copy because the dev helper intentionally
  compares the raw linear-solver basis convention

The fixture prints its evidence directory first. The evidence directory is created under `.local_tmp` and is intentionally left in place for diagnosis.

The companion standalone source
`native_tests/adaptive_radial_benchmark.cpp` measures `CalcEFS`,
full-cache `AccumulateEFSCombinationGrad`, and `CalcEFSComponents` on identical
input configurations per process/rank. It also runs an in-process stale-state
guard sequence across EFS, full-gradient, energy-only, coefficient/scaling/
geometry/species mutation, and EFS. A second in-process guard compares
full-cache gradient reuse against fresh-model gradients after no mutation and
after direct geometry, species, global-scaling, and parameter mutations.
