# Two-cutoff regression fixtures

These are synthetic workspace test inputs, frozen from the project's
`two_cutoff_20261107` cross-framework qualification. They contain initialized
two-species model coefficients and small artificial cells, rather than a
material training dataset or a third-party empirical potential. They are
distributed as test assets under the repository's [BSD license](../../../../../LICENSE).

The `.mtp` files use the native SUS2-SH serialization format. This format does
not establish which evaluator generated a reference result. The JSON files
contain independent FP64 E/F/virial reference values: the 29 retained records
were re-evaluated with the project's PyTorch implementation on CPU using
`torch.float64` (PyTorch 2.8.0), agreeing within `1.4e-15` absolute error. The
local qualification also checks Cartesian and cell derivatives independently
with finite differences. These small fixtures qualify arithmetic and cutoff
semantics, rather than model accuracy on physical systems.

`fixtures_v2` covers Static, Effect, Adaptive full, and Adaptive shared-n.
Its seven geometries include a mixed near/far environment, the angular cutoff
and both sides, both sides of the radial cutoff, and periodic images.
`fixtures_hi_l` adds Effect with L=3, K=4, gate rank 3, and outer rank 3.
Shared-o and grouped projections are outside the formal CPU main support
contract and are deliberately excluded.

Each manifest contains only the files used by the regression. The retained
file bytes and their original SHA-256 hashes are unchanged. Unused validation
summaries, shared-o models, and generated execution artifacts are not bundled.

From the repository root, build and run the standalone test:

```sh
c++ -O2 -std=c++17 -Iinterfaces/lammps/ML-SUS2 \
  interfaces/lammps/tests/embedded_efs.cpp \
  interfaces/lammps/ML-SUS2/sus2_sl_*.cpp -o /tmp/sus2-embedded-efs
python3 interfaces/lammps/tests/embedded_two_cutoff_check.py \
  --embedded /tmp/sus2-embedded-efs --out /tmp/sus2-two-cutoff-check
```

The test defaults to these bundled fixtures; `--fixtures` can select an
external directory with the same layout. `--reference-embedded` adds comparison
against a separately compiled main binary. `--sn-dir` adds optional external
SnS/SnSe regression data and requires that reference binary. The standalone
harness explicitly simulates ghost forward/reverse exchange across two
domains; real LAMMPS MPI qualification is separate. Python uses only its
standard library, and the test submits no jobs.
