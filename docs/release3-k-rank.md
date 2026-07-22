# Release3 K-channel contraction

## Mathematical definition

For every angular shell `(l,m)`, the ordinary SUS2-SH edge accumulation first
produces `K` radial channels

\[
M_{i,k l m}=\sum_j A_{ij,k l m}.
\]

With outer rank `R`, release3 inserts the exact linear contraction

\[
\widetilde M_{i,c l m}
=\sum_{k=0}^{K-1}W^{(l)}_{c k}M_{i,k l m},
\qquad c=0,\ldots,R-1,
\]

and constructs the outer Clebsch-Gordan product graph from the contracted
channels `c`, not from the original channels `k`. The saved weight layout is

\[
\operatorname{index}(l,c,k)=(lR+c)K+k.
\]

The force/stress reverse pass is the transpose of the same map,

\[
\frac{\partial E}{\partial M_{i,k l m}}
=\sum_c W^{(l)}_{c k}
\frac{\partial E}{\partial\widetilde M_{i,c l m}}.
\]

The first-layer gate can use an independent rank `R_g` and matrix `W_g` before
its own scalar product graph. This does not alter the release additive-node
gate definition; it only changes which channel space is used to construct the
first-layer invariant scalars.

`R=K` with identity matrices is exactly the full-`K` model written through an
extra change of representation. `R<K` is a smaller hypothesis class whose
radial functions, contraction matrices, and scalar heads are optimized
together; it is not a post-training approximation of an arbitrary full model.

For outer `R=1`, a separate matrix is redundant with the freely trainable
radial rows. Release3 therefore absorbs the rank-one weights into the radial
coefficients and evaluates a direct-radial rank-one channel. There is no outer
`W` parameter block in that canonical form.

Rank reduces the Clebsch-Gordan product graph, not the initial edge basis. The
runtime still evaluates the original `K(l_max+1)^2` raw edge moments before
the exact contraction, so total inference speedup is smaller than the scalar
count reduction and is workload dependent. For `l_max=4`, `K=4`, the raw edge
stage still has 100 `(k,l,m)` moments; a trainable rank-`R` contraction adds
`25 R K` multiply-adds per forward or reverse stage. CPU rank models currently
use the generic exact gate path rather than the specialized direct-main/static
fixed gate shortcuts, so a smaller graph is not by itself a guarantee of a
faster full LAMMPS step. Use a matched single-node benchmark before making a
backend-specific speed claim.

### Reference CPU inference benchmark

The final pre-release CPU candidate was measured on 2026-07-22 on one
`1t88c` node with 96 MPI ranks, one thread per rank, and a fixed 10,368-atom
system. Each fresh process ran 100 warm-up steps followed by 800 timed steps;
the order of full, rank-one, and rank-two models was rotated over three
repeats. All three models used `l_max=4`, `K=4`, body order 6, and the same
release gate family.

| Outer graph | Timed loop repeats (s) | Median (s) | MAD (s) | Full / case |
| --- | --- | ---: | ---: | ---: |
| full `K=4` | 13.4812, 14.1710, 14.2946 | 14.1710 | 0.1236 | 1.0000 |
| rank 1, direct radial | 13.1178, 13.1932, 13.0919 | 13.1178 | 0.0259 | 1.0803 |
| rank 2 | 13.5346, 14.3975, 14.0463 | 14.0463 | 0.3512 | 1.0089 |

Rank one reduced loop time by about 7.43% in this workload. The rank-two
difference was only about 0.88%, smaller than its run-to-run variation, so it
should be treated as a tie rather than a demonstrated speedup. These numbers
compare models through the same final candidate binary; they are not an
old-versus-new LAMMPS implementation comparison.

## CLI

```bash
bin/mlp-sus2 init-sh rank2.mtp \
  --species-count=2 \
  --l-max=4 \
  --k-max=4 \
  --body-order=6 \
  --body-l-max=4,4,4,2,1 \
  --radial-basis-size=5 \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-factor-pruning=q-total \
  --outer-k-rank=2
```

- `--outer-k-rank=0` or omission keeps the historical full-`K` path.
- `--outer-k-rank1` is a compatibility alias for `--outer-k-rank=1`.
- `--two-layer-gate-k-rank=<R_g>` contracts the first-layer gate graph too.
- Both ranks must be in `[1,K]` when enabled.
- Gate rank currently requires outer rank.
- Rank models require an explicit one-layer `--two-layer-gate` and
  `q-total` pruning for each contracted graph.
- Rank is a model topology. A loaded model cannot change rank during `train`;
  generate a new model instead.
- Rank is currently incompatible with readout, residual, edge-L1, and
  multi-layer-gate modes.

## Model format and runtime compatibility

Feature-free models retain `version = 1.1.0` and the historical rank-off
serialization. Models containing outer/gate rank or gate readout use
`version = 1.2.0`. The release3 standalone reader accepts both versions and
canonicalizes the version from the actual feature set when saving.

| Runtime | rank=0 | outer rank | gate + outer rank | gate readout |
| --- | --- | --- | --- | --- |
| release3 standalone | yes | yes | yes | yes |
| release3 CPU LAMMPS `sus2mtp` | yes | yes | yes | no, explicit error |
| release3 Kokkos LAMMPS `sus2mtp/kk` | yes | yes | yes | no, explicit error |
| older LAMMPS/other backends | 1.1 models only unless separately updated | not declared compatible | not declared compatible | not declared compatible |

The CPU and Kokkos interfaces share the same host-side model reader, including
the `1.2.0` and unsupported-readout checks. Their forward contraction and
transpose force chain are separate CPU/device implementations and must both
pass the standalone/LAMMPS energy, force, and virial parity test before a
binary is published.

Additional current limits:

- the LAMMPS SH reader supports `0 <= l_max <= 4`;
- `PruneSpecies` is not implemented for rank or readout models;
- disabling rank performs no contraction and writes no rank metadata;
- readout remains a standalone-only experimental feature in this release;
- standalone `two_layer_gate_scaling_mode=decoupled` models are not supported
  by the CPU or Kokkos LAMMPS pair styles and are rejected explicitly.

## Required checks

```bash
dev_test/sh_k_rank_cli_smoke.sh ./bin/mlp-sus2
dev_test/sh_k_rank_active_math_check.sh ./bin/mlp-sus2
dev_test/sh_k_rank_mpi_smoke.sh ./bin/mlp-sus2
dev_test/sh_release3_model_format_check.sh ./bin/mlp-sus2

dev_test/lammps_sh_k_rank_parity.sh \
  --mlp ./bin/mlp-sus2 \
  --cpu /path/to/cpu/lmp \
  --kokkos /path/to/kokkos/lmp
```

For the device Kokkos runtime, use a half neighbor-list package configuration
with Newton pair forces enabled, for example
`-k on g 1 -sf kk -pk kokkos neigh half newton on`. The pair style rejects the
default full-neighbor/Newton-off combination before model evaluation.

The LAMMPS parity matrix covers rank-off, outer rank-one direct radial, outer
rank two with asymmetric weights, `R=K` identity, gate/outer ranks `1/1` and
`2/2`, both independent mixed-rank layouts `R_g/R=1/2` and `2/1`, and a
default `mu-body-linear-combo` gate-rank case. It also checks that CPU and
Kokkos reject dynamic readout and decoupled gate-scaling models with explicit
diagnostics. Active
parameter-gradient, energy/force/stress directional-derivative, coordinate and
cell finite-difference, roundtrip, and serial/MPI checks are kept in the
standalone test suite.
