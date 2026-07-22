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
`25 R K` multiply-adds per forward or reverse stage.

The CPU outer-rank path uses the same exact direct main-layer edge dot as the
rank-off path. Product reverse first applies `W^T` to recover the raw
`(k,l,m)` adjoints, after which force and gate-adjoint evaluation can recompute
the edge radial/SH terms without storing a dense edge Jacobian. The main
workspace also clears only the active raw prefix and contracted outer suffix;
the skipped full-`K` graph is validated as an inactive closed gap. Gate-rank
models retain the conservative generic main path until their independent gate
fast paths are validated. Use a matched single-node benchmark before making a
backend-specific speed claim.

The Kokkos rank path already recomputes edge derivatives directly on the
device. Its rank-specific optimization is instead stage-aware storage: the
gate pass clears and caches only the validated gate-graph prefix, while the
outer pass clears only the raw SH basics and the closed contracted suffix.
Rank-off models retain the historical full-view `deep_copy`. Their historical
flattened cache kernel is also retained while `chunk_size * width` fits in a
signed `int`; only an overflow-sized chunk switches to the same exact
two-dimensional cache mapping. The prefix/suffix closure checks are part of
model loading, so skipped buffer regions cannot be referenced silently by a
malformed graph.

The active-range kernels use a two-dimensional hierarchical launch rather
than flattening `(chunk_size, width)` into one signed `int` range.  In
particular, `TeamPolicy` assigns one league rank to each active moment index
and `TeamThreadRange` covers the atoms in the current chunk.  The previous
flattened expression would exceed `INT_MAX` for, for example,
`600000 * 4000 = 2400000000`; the hierarchical form never represents that
Cartesian-product cardinality in a signed integer.  It also avoids integer
division and remainder in the device kernel.  The lightweight
`lammps_kk_rank_clear_overflow_check.sh` regression proves the integer bound,
exhaustively checks a smaller Cartesian product and the raw/suffix mapping,
and guards all five source kernels against reintroducing a flattened range.
For the conservative maximum supported moment width of 4000, the last safe
rank-off flattened chunk is 536,870 atoms; 536,871 atoms selects the wide
fallback before the product is formed.

### Reference CPU inference benchmark

The optimized pre-release CPU candidate was measured on 2026-07-22 on one
`1t88c` node with 96 MPI ranks, one thread per rank, and a fixed 10,368-atom
system. Each fresh process ran 100 warm-up steps followed by 800 timed steps;
the case order was rotated over three repeats. All models used `l_max=4`,
`K=4`, body order 6, and the same release gate family.

| Outer graph | Timed loop repeats (s) | Median (s) | MAD (s) | Full / case |
| --- | --- | ---: | ---: | ---: |
| full `K=4` | 13.7027, 13.8520, 13.6882 | 13.7027 | 0.0145 | 1.0000 |
| rank 1, direct radial | 11.0659, 11.7143, 11.4748 | 11.4748 | 0.2395 | 1.1942 |
| rank 2 | 11.8644, 11.5563, 12.0640 | 11.8644 | 0.1996 | 1.1549 |

Rank one reduced loop time by 16.26% and rank two by 13.42% relative to the
same-binary full model. A same-binary switch-off control isolates the direct
main-layer change:

| Outer graph | Direct off median (s) | Direct on median (s) | Off / on |
| --- | ---: | ---: | ---: |
| rank 1 | 12.9558 | 11.4748 | 1.1291 |
| rank 2 | 13.6233 | 11.8644 | 1.1483 |

The previous generic-rank candidate medians were 13.1178 s for rank one and
14.0463 s for rank two. The exact direct-main and active-range-clear changes
therefore reduced their median loop times by 12.53% and 15.53%, respectively.
The benchmark artifacts are stored under
`codex_release3_rank_cpu_direct_sparse_20260722_3966177` on the reference
server.

For context, an initial six-repeat comparison found the fixed release2 CPU
binary 2.62% faster than the pre-optimization release3 rank-capable candidate
for the same full-`K=4` model. Rank is inactive in that model, so this was not
extra rank mathematics. Inspection found that rank-off still made an empty
per-atom contraction call whose compiled function created a nontrivial stack
frame. The final candidate moves that guard to the caller and avoids the call
entirely. A fresh six-repeat, single-`1t88c`-node comparison then measured
14.4589 s for release2 and 14.29025 s for release3; the paired-ratio median
favored release3 by 1.43%. Because the release2 repeats in that run had a
0.2638 s MAD, the conservative conclusion is that the earlier rank-off
regression is gone, not that release3 has a reproducible one-percent speedup.
All preflight energy, pressure, and force values agreed, and all formal-run
stderr files were empty. The final artifacts are under
`codex_release2_vs_release3_guard_20260722_3966235` on the reference server.

The compiled release2 Kokkos binary is also compatible with the same full-`K4`
model. On one A100 with 105,456 atoms, six balanced old/new repeats measured
21.4910 s for release2 and 21.80935 s for the pre-active-clear release3 GPU
candidate, so release2 used 1.46% less time. The compatibility preflight gave
zero energy difference, a maximum force difference of
`7.237e-15 eV/angstrom`, and a maximum pressure difference of
`8.731e-11 bar`. Release2 and release3 use different CUDA runtime and MPI
stacks, so this is a whole-binary comparison and cannot be attributed solely
to rank dispatch.

### Reference GPU rank optimization benchmark

The rank-specific active-clear change was compared with its immediately
preceding Kokkos candidate on one exclusive A100-80GB. The fixed 105,456-atom
system used 40 warm-up and 200 timed steps in six balanced old/new repeats.

| Model | Old median (s) | Active-clear median (s) | Old / new paired median | New faster |
| --- | ---: | ---: | ---: | ---: |
| full `K=4` control | 43.61910 | 43.61610 | 1.00019 | 4 / 6 |
| outer rank 1 | 38.09165 | 31.86425 | 1.19567 | 5 / 6 |
| outer rank 2 | 39.83915 | 32.75515 | 1.21587 | 6 / 6 |

The full-model control is unchanged within 0.02%. Rank-one and rank-two loop
times decreased by 16.35% and 17.78% from their respective medians. All 18
old/new energy controls passed, with maximum absolute difference
`1.164e-10 eV`, and all 48 timing/profile stderr files were empty. Fenced
profiles localized the gain to first/main clear and cache work; the main
physical compute stages remained approximately one-to-one. The benchmark
artifacts are under
`codex_release3_rank_gpu_active_clear_ab_20260722/job_3966685`.

The final overflow-safe binary adds only a rank-off wide-cache fallback to
that active-clear candidate. It independently passed the complete CPU/Kokkos
matrix on both one and two GPUs with `KOKKOS_CHUNKSIZE=3` and device
communication (jobs 3966777 and 3966778).

### Million-atom GPU capacity check

The final overflow-safe candidate was also exercised on one A100-80GB with
1,054,560 atoms (`48 * 130 * 13 * 13`), a full neighbor list containing
138,147,360 entries, and both outer-rank models. The chunk-dependent pair
workspace is exactly

```text
M_chunk = 2 * chunk_size * alpha_moments_count * sizeof(double).
```

This is 55,120 bytes per chunk atom for rank one
(`alpha_moments_count=3445`) and 61,728 bytes for rank two
(`alpha_moments_count=3858`). The per-atom gate cache and LAMMPS/Kokkos atom,
ghost, and neighbor storage are separate from this workspace.

Both tested model files explicitly use
`radial_basis_type = RBChebyshev_sss_lmp`. The pair input supplies
`tabstep 0.0001`; the runtime log confirms construction of the preinterpolation
table with 55,001 grid points through the 5.5-A cutoff. Therefore the capacity
and timing numbers below are for the `_lmp` table-lookup path, not analytic
radial evaluation.

| Model | Largest routine sweep chunk | Peak GPU memory (MiB) | Two-step loop (s) | Result |
| --- | ---: | ---: | ---: | --- |
| rank 1 | 655,360 | 71,749 | 3.39233 | pass |
| rank 2 | 655,360 | 75,881 | 3.42359 | pass |

At `chunksize=65536`, the rank-one step-zero energy was
`-6811553.6046968848 eV` and pressure was `-152632.04171433244 bar`, agreeing
with ten replicas of the 105,456-atom reference. Forces were finite, the atom
count remained 1,054,560 through two NVE steps, and stderr was empty for every
successful chunk/model case.

A focused rank-two upper-bound test succeeded at chunk sizes 688,128, 720,896,
and 737,280, with respective peaks 77,809, 79,737, and 80,701 MiB. The next
case, 753,664, failed cleanly during allocation of a 34.05-GiB Kokkos view.
Thus the measured hard limit lies in `[737280, 753664)`. Because 737,280 leaves
only about 1.2 GiB free, 655,360 is the largest tested setting with roughly
6 GiB margin, while 524,288 is the more conservative default recommendation
for this model/system (68,165 MiB peak). Larger chunks did not monotonically
reduce the short loop time after launch overhead was amortized. Artifacts are
under `SUS2-SH-release3-rank-million-smoke-20260722/job_3966782` and
`upper_job_3966785` on the reference server.

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
dev_test/lammps_kk_rank_clear_overflow_check.sh

dev_test/lammps_sh_k_rank_parity.sh \
  --mlp ./bin/mlp-sus2 \
  --cpu /path/to/cpu/lmp \
  --kokkos /path/to/kokkos/lmp
```

For the device Kokkos runtime, use a half neighbor-list package configuration
with Newton pair forces enabled, for example
`-k on g 1 -sf kk -pk kokkos newton on neigh half comm device`. The pair style
rejects the default full-neighbor/Newton-off combination before model
evaluation.  Set `KOKKOS_CHUNKSIZE=3` in the parity script to force its
eight-atom fixture through multiple chunks and exercise chunk-local clear and
cache boundaries.

The LAMMPS parity matrix covers rank-off, outer rank-one direct radial, outer
rank two with asymmetric weights, `R=K` identity, gate/outer ranks `1/1` and
`2/2`, both independent mixed-rank layouts `R_g/R=1/2` and `2/1`, and a
default `mu-body-linear-combo` gate-rank case. It also checks that CPU and
Kokkos reject dynamic readout and decoupled gate-scaling models with explicit
diagnostics. Active
parameter-gradient, energy/force/stress directional-derivative, coordinate and
cell finite-difference, roundtrip, and serial/MPI checks are kept in the
standalone test suite.
