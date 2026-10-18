# SUS2-SH Shared-Radial Gate Implementation Notes

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

**Goal:** Add an optional two-layer gate radial mode where the inner gate reuses
the same bottom radial basis, scaling, shift, and SH primitive cache as the
outer SUS2-SH path, but owns independent radial contraction coefficients.

**Model form:** In legacy mode the gate scalar uses the outer radial
contraction. With `--two-layer-gate-shared-radial`, the gate uses

```text
R^gate_mu(r) = sum_xi c^gate_{mu,xi} b_xi(r; sigma, shift)
```

while the final outer SH moments still use the ordinary outer `R_mu(r)`. The
shared quantities are only the directed-edge primitives: distance, SH values
and derivatives, bottom radial basis values and derivatives, scaling, shift,
and type coefficients. The gate scalar product graph is not reused as an outer
product graph.

**Parameter layout:** For shared-radial models, the trainable gate radial block
is inserted after the base nonlinear block and before the two-layer gate
weights:

```text
[base nonlinear][gate radial coeffs][gate scalar weights][linear coeffs]
```

The gate radial count is `radial_func_count * radial_basis_size`.

**Implementation points:**

- `dev_src/sh_model_init.cpp` writes `two_layer_gate_radial_mode =
  shared-radial`, count, and initial coefficients.
- `dev_src/mtpr.cpp` loads/saves the optional metadata and accounts for the new
  offset in `DistributeCoeffs`, pruning, and random radial initialization.
- `dev_src/mtpr_sh.cpp` keeps outer SH directed-edge contractions in
  `two_layer_edge_mu_*` and adds gate-only contractions in
  `two_layer_edge_gate_mu_*`. The gate arrays are filled only for the gate's
  required `mu` indices.
- Legacy two-layer models keep the old coefficient layout because the gate
  radial count is zero unless the new metadata/flag is enabled.

**Server validation:** The server tree was synced and built at
`<historical-path>`.

```bash
rm -rf obj/mpi
make mlp -j8 USE_MPI=1 CXX_EXE=mpicxx CC_EXE=mpicc FC_EXE=true \
  CXXFLAGS='-std=c++11 -O2 -DMLIP_DEV -DMLIP_MPI' \
  CPPFLAGS='-O2 -I./cblas' TARGET_PRERQ=
```

The initial link attempt failed because stale `obj/mpi` objects had been built
with an Intel toolchain; removing `obj/mpi` fixed the toolchain mix.

Passed checks:

```bash
dev_test/sh_two_layer_gate_shared_radial_check.sh
./bin/mlp-sus2 check-loss-gradient-dev ... --coeff-start=86 --coeff-end=110 \
  --displacement=1e-4 --radial-smooth=0
./bin/mlp-sus2 check-efs-fd-dev ... --max-atoms=0 --displacement=1e-4
dev_test/sh_two_layer_gate_init_check.sh
dev_test/sh_two_layer_gate_zero_compat_check.sh
dev_test/sh_two_layer_gate_forward_check.sh
dev_test/sh_two_layer_gate_loss_gradient_check.sh
dev_test/sh_two_layer_gate_force_fd_check.sh
dev_test/sh_two_layer_gate_train_weight_check.sh
```
