Portable stateful angular-cache regression (run from repository root):

```sh
c++ -O2 -std=c++17 -Iinterfaces/lammps/ML-SUS2 interfaces/lammps/tests/embedded_angular_cache_check.cpp interfaces/lammps/ML-SUS2/sus2_sl_*.cpp -o /tmp/embedded-angular-cache-check
/tmp/embedded-angular-cache-check interfaces/lammps/tests/fixtures/two_cutoff/fixtures_v2/effect.mtp
```

Pass other supported Effect models to qualify L0 through L4 and single/dual cutoffs. This test creates synthetic directed neighborhoods, not physical MD structures or benchmark data. One owned center and distinct ghost sources exercise a 1MiB admission prefix plus uncached suffix; deterministic ghost signals are explicitly injected. Successive calls reuse one Kernel across large, small, empty, and changed/reversed large neighborhoods. Zero-budget and 1MiB routes compare E/F/virial/site against a fresh angular-cache-disabled Kernel at 1e-9, across direct/table, radial-cache, source-cache and forward-stream combinations. All values must remain finite.

This covers geometric/type/layout invalidation and pointer consumers, including angular/outer shell probes. Vector normalization can move nominal equal-cutoff probes by rounding; exact boundary qualification remains in the separate bundled two-cutoff framework. This does not replace MPI communication, physical trajectory or allocation-failure tests. No model files are written. Cache flags are process-local.
