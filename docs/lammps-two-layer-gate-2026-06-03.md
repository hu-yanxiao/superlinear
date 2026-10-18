# LAMMPS two-layer gate support notes (2026-06-03)

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

Repository branch: `developer`

Server tree:

```text
<historical-path>
```

LAMMPS source tree:

```text
/work/phy-weigw/apps/lammps-10Dec2025
```

## CPU implementation

The CPU `pair_style sus2mtp` path now parses non-residual SUS2-SH two-layer
gate models and evaluates active gate models through an explicit two-pass
normal-cutoff algorithm.

The physical receptive field of a center atom is two SH cutoffs, because the
outer edge `i -> j` depends on the gate value of `j`, and that gate value
depends on neighbors `k` within the normal cutoff of `j`.  The LAMMPS
implementation does not request a physical `2R` neighbor list.  It evaluates
the model as two message-passing steps over the existing full neighbor list:

1. For owned atoms, compute first-layer gate values `g_i` from unmodulated SH
   scalar products using the normal cutoff.
2. Forward-communicate `g_i` to ghost atoms.
3. Evaluate the outer gated SH energy and direct force terms with
   `B_i = sum_j g_j edge(i,j)`, while accumulating the adjoint
   `dE_i / dg_j` on neighbor atoms.
4. Reverse-communicate gate adjoints from ghosts back to owned atoms.
5. Revisit the first-layer gate SH graph to propagate `dE / dg_i` to force
   contributions on the gate-neighbor edges.

This preserves the mathematical two-cutoff receptive field without increasing
the neighbor-list radius.

## LAMMPS model support

Supported in CPU LAMMPS:

- Plain SUS2-SH models.
- Gate-zero SUS2-SH models. These keep the old single-layer compute path.
- Active non-residual two-layer gate models.
- `--two-layer-gate-shared-radial` models with independent gate radial
  coefficients.
- Gate models without `--two-layer-gate-shared-radial`, where the gate layer
  uses the same radial functions as the outer layer.

Rejected in CPU LAMMPS:

- Residual two-layer readout models.
- Environment-gate plus two-layer gate combinations.

## `_lmp` table semantics

Training/dev `_lmp` tables store the radial contraction without type
coefficients, because the training evaluator multiplies type coefficients in
the SH edge path.

LAMMPS `_lmp` tables keep the existing LAMMPS convention: the table stores the
precontracted radial value including the center/out species factor.  For active
two-layer shared-radial gate models, LAMMPS builds a second table:

```text
two_layer_gate_radial_list
two_layer_gate_radial_der_list
```

using the gate radial coefficients and the same species-factor convention as
the outer table.  The runtime edge evaluator chooses the outer or gate table
based on the current pass.

## CPU verification

The server development binary passed:

```bash
bash dev_test/sh_two_layer_gate_lmp_table_check.sh
```

The LAMMPS CPU smoke test is:

```bash
LAMMPS_BIN=/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2 \
  bash dev_test/lammps_two_layer_gate_lmp_smoke.sh
```

It builds a small nonzero two-layer gate model, creates a matching `_lmp`
model, and checks:

- direct radial vs `_lmp` radial in LAMMPS,
- one MPI rank vs two MPI ranks for direct radial,
- one MPI rank vs two MPI ranks for `_lmp` radial,
- gate-zero parsing through the old compute path.

## Kokkos/GPU plan

The Kokkos device path currently has an explicit guard for active two-layer
gate models.  This prevents silent use of the old single-layer Kokkos
algorithm on a gate model.

The planned Kokkos implementation should mirror the CPU phases, but keep all
large per-atom/per-edge arrays in Kokkos views:

1. Add device views for gate values, gate adjoints, first-layer gate moments,
   first-layer gate reverse adjoints, and optional shared-radial gate `_lmp`
   tables.
2. Add a gate-value kernel over owned atoms.  It should reuse the existing SH
   product graph and normal neighbor list.  Gate-zero should still bypass this
   path.
3. Add Kokkos forward communication for the one-scalar `g_i` field.  The base
   Pair communication layout already uses one forward scalar in the CPU path.
4. Add the outer gated SH kernel.  It computes `B_i = sum_j g_j edge(i,j)`,
   energy, direct edge forces, and the neighbor gate adjoint
   `dE_i / dg_j`.
5. Add reverse communication for the one-scalar gate adjoint.
6. Add a gate-chain force kernel over owned atoms to propagate `dE/dg_i`
   through the first-layer gate SH graph.
7. Copy the CPU `_lmp` gate table build into Kokkos host setup, then mirror the
   outer and gate tables to device views.

Important constraints:

- Do not request a `2R` Kokkos neighbor list for this model.  The two-layer
  receptive field comes from two normal-cutoff passes plus communication.
- Keep active gate and non-gate Kokkos paths separate so the existing SH Kokkos
  hot path is unchanged.
- Device memory scales with per-atom gate scalars and per-team SH scratch, not
  with a `2R` neighbor list.

Suggested Kokkos tests:

- CPU direct vs Kokkos direct for an active nonzero gate model.
- CPU `_lmp` vs Kokkos `_lmp` for an active shared-radial gate model.
- One GPU rank vs two GPU ranks with atoms straddling a domain boundary.
- Gate-zero model equivalence with the existing single-layer Kokkos result.
