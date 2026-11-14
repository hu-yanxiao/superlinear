Effect adjoint seed rows

`SUS2_SL_KOKKOS_EFFECT_SEED_ROWS=1` enables destination-row scheduling only for non-generic Effect with gate readout disabled. The raw executable defaults to 0 and keeps the old atom loop; the accepted 2026-11-15 launcher defaults to 1. Constructor groups compressed gate nodes, preserving ascending q within each target. Each thread starts with the existing ga value and applies each q's increasing-mu sum in original order, preserving duplicate destinations and joint seeds without atomics. New metadata lives outside State so existing raw-kernel captures do not grow. Installer includes the helper header automatically.

Portable Serial test (from repository root):

```sh
c++ -O2 -std=c++17 -I$KOKKOS_PREFIX/include -Iinterfaces/lammps/ML-SUS2-KK interfaces/lammps/tests/kokkos_seed_test.cpp -L$KOKKOS_PREFIX/lib -lkokkoscore -ldl -pthread -o /tmp/kokkos-seed-test
/tmp/kokkos-seed-test
```

Tests duplicate destinations, existing nonzero seeds, F1/7/40, owned0/1/17 and Q0/5, plus negative/out-of-range owned types against the original q/mu loop with zero numerical difference. CUDA compilation, real EFS parity and performance are separate requirements. The helper repeats the original owned type guard using lightweight types/error captures; invalid species set error3 and preserve that atom adjoints. Test the Generic, gate-readout, Static, Pure and Adaptive fallbacks on GPU before qualification. Existing nonzero adjoints are checked by the helper test; the native format permits joint readout only for Adaptive, which retains the old path.

The [2026-11-15 evidence index](kokkos_effect_seed_evidence_20261115.json) records executed CUDA helper/engine and single-rank LAMMPS validation. Older multi-rank qualification is not inherited by this binary.
