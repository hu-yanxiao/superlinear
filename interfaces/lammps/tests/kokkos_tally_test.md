Optional global tally prototype

`SUS2_SL_KOKKOS_FUSED_TALLY=1` combines global energy and six virial reductions into one custom reduction. The flag is read once per model setup (`coeff`); reload rereads it. Energy-only keeps the scalar reduction. Default/unset and `0` preserve the existing scalar reductions. Force accumulation, model kernel, error checks and per-atom energy are unchanged. An empty owner or both global flags disabled returns zero without launching a reduction. The installer automatically includes `sus2_sl_kokkos_*.h`.

From repository root, with a Serial Kokkos prefix:

```sh
c++ -O2 -std=c++17 -I$KOKKOS_PREFIX/include -Iinterfaces/lammps/ML-SUS2-KK interfaces/lammps/tests/kokkos_tally_test.cpp -L$KOKKOS_PREFIX/lib -lkokkoscore -ldl -pthread -o /tmp/kokkos-tally-test
/tmp/kokkos-tally-test
```

This verifies 36 Serial cases: all four energy/virial flag combinations, empty/singleton/multiple owners, signed data and cancellation, against the old scalar reductions with finite results and absolute tolerance1e-9. Unrequested input arrays are poisoned with NaN to check they do not propagate. It is not CUDA or full LAMMPS qualification. GPU validation must compile with production NVCC/Kokkos4.7.1, test all flag combinations and empty MPI owners against flag0, and compare global E/virial, force and per-atom energy on real frozen models plus trajectory parity. A custom reduction may use a different GPU summation tree; mathematical equality does not guarantee bit identity. Benchmark only after parity passes, separating thermo/tally steps from steps without requested global values.
