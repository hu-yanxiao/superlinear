# Exact pure-W inference execution

This changes execution only. The pure-W channelwise model, serialized physical
coefficients, scalar head, optimizer coordinates, initialization, regularization,
and single-channel angular product topology are unchanged. Outputs do not acquire
cross-O products. Floating-point summation order may change; bitwise equality is
not promised for the new paths.

## Execution backends

`SUS2_SH_PURE_W_KERNEL` selects `auto`, `legacy`, `packed`, or `basis`.
The setting is read once per process. `legacy` retains the edge-first reference.
The default `auto` conservatively selects basis-first only for eligible EFS
calls with O>=8, R<=10 and at least 16 neighbors/site on average. Small-O,
sparse-neighbor and large-R cases retain legacy execution. Full mixed-gradient
calls retain legacy execution under `auto`, regardless of O. This is a measured
CPU-workload crossover guard, not a universal claim of optimal dispatch.

`packed` mirrors physical `C[l,o,k,b]` into contiguous output lanes
`Cpack[l,k,b,o]`. O=1/2/4/8/16 use compile-time-sized accumulators; other widths
use an exact generic loop. Each physical W and W-prime retains the original
radial-index accumulation order. The existing edge-cache layout is unchanged.
No coefficient normalization or approximation occurs during packing.

`basis` additionally enables the basis-first EFS path when eligible. Full
parameter-derivative cache requests still use edge-first execution (with packed
W construction). Energy-only calls, linear-component assembly and mixed
parameter derivatives also retain their existing paths. Thus this is **not** a
new mixed-Hessian backend or a new energy-only backend.

`SUS2_SH_PURE_W_GRAPH=batched` optionally validates and executes one canonical
angular product instruction stream across O contiguous, independent lanes.
The default is `legacy`. It verifies operand/target/coefficient correspondence
for every lane, including physical lane ownership and instruction order; an incompatible equivalent graph
falls back. `SUS2_SH_PRODUCT_ROWS=1` takes precedence. Serialization still keeps
the original graph, so old model readers and saved models are unaffected.

## Exact basis-first algebra and force chain

Write the full existing edge gate/type factor as x, including its species
factors. For radial basis b and sharing map sigma(k,l),

\[
 W_{lok}(r_{ij})=\sum_b C_{lokb}
 \phi_b(r_{ij};s_{\sigma(k,l)}^{z_i z_j}),\qquad
 A_{iolm}=\sum_{j,k}W_{lok}(r_{ij})x_{ij,kl}Y_{lm}(\widehat r_{ij}).
\]

The alternate contraction is

\[
 Q_{ilkbm}=\sum_j x_{ij,kl}
 \phi_b(r_{ij};s_{\sigma(k,l)}^{z_i z_j})Y_{lm}(\widehat r_{ij}),
 \qquad A_{iolm}=\sum_{k,b}C_{lokb}Q_{ilkbm}.
\]

Q retains **all k, b and m indices**. It is not gate-rank compression, and does
not insert a separate single-radial-function multiplier before W. Species and
K/L/LK-dependent radial evaluation remains inside the neighbor sum.

After differentiating the unchanged scalar graph and physical head, the
adjoint (including the species-center head multiplier) gives

\[
 \bar Q_{ilkbm}=\sum_o C_{lokb}\bar A_{iolm}.
\]

For one edge define

\[
 B_{jklm}=\sum_b\phi_b(r_{ij})\bar Q_{ilkbm},\qquad
 D_{jklm}=\sum_b\partial_r\phi_b(r_{ij})\bar Q_{ilkbm}.
\]

Then the direct-edge and input derivatives are

\[
 \bar x_{ij,kl}=\sum_m Y_{lm}B_{jklm},
\]
\[
 \partial_{\mathbf r_{ij}}E\big|_{x\ \mathrm{fixed}}
 =\sum_{k,l,m}x_{ij,kl}
 \left[Y_{lm}D_{jklm}\widehat{\mathbf r}_{ij}
       +B_{jklm}\nabla_{\mathbf r_{ij}}Y_{lm}\right].
\]

The existing gate Jacobian multiplies bar-x, and the existing global gate-force
pass propagates it through neighboring environments. It must not be omitted.
The caller retains its original force signs, stress assembly and energy offsets.
Contracting b into B/D before angular derivatives avoids recomputing four
angular dot products for every b; this is algebraic reassociation, not pruning.

For clarity, the corresponding parameter identities would be
`bar-C = bar-A Q^T` and `bar-s = sum(bar-Q x Y dphi/ds)`, plus all gate/scaling
chains. **The implementation deliberately keeps the already-validated full
mixed-derivative path for those consumers.** Existing finite-difference tests
verify both paths against the same physical objective.

## Cache and fallback contract

The caller must explicitly mark an EFS-only cache build. Basis-first requires
a derivative cache without parameter derivatives, pure-W mode, and no multilayer
gate, center gate, residual, active edge-L1 gate weights or LAMMPS radial table.
Unsupported combinations fall back to the reference path. Disabling edge
caches also falls back. Ordinary three-argument cache callers cannot silently
receive an inference-only cache.

Packing rereads the actual coefficients on **every cache build**. Optimizers
may edit coefficients in place, so pointer equality is not a validity test.
Cache clearing invalidates execution flags; a subsequent full derivative cache
cannot be satisfied by an inference-only cache. The regression test alternates
EFS, full-cache EFS, energy-only, and EFS on the same object while changing both
geometry and parameters, including strongly dilated/denser neighborhoods.

An eligible fresh inference cache allocates no edge W or W-prime arrays. It
stores compact radial derivatives `[edge,radial-evaluation-block,b]` instead,
plus bounded per-center Q/adjoint scratch. Shared raw radial values already
exist in both paths. Inactive scratch retains its size/capacity after a previous
full-cache call, avoiding redundant zero-fills on mixed inference/gradient
workloads. Validity flags prevent those stale values from being read. This is
an allocation/layout statement, **not** a guarantee that an existing training
process releases that much RSS. The profile distinguishes active `W_entries`
from retained `W_storage_entries`.

With E edges, N centers, H=(l_max+1)^2, L=l_max+1, radial size R, gate width K,
and P single-channel product instructions, dominant work is approximately

- Edge-first: `E*K*O*(L*R + H) + N*O*P`.
- Basis-first: `E*K*R*H + N*O*K*R*H + N*O*P`.

Constants, cache traffic and BLAS call sizes matter: smaller O can favor the
edge-first path. Linear O scaling of the independent graph is preserved; an
O-independent edge contraction does not make total inference independent of O.

## Diagnostics and regression commands

`SUS2_SH_PURE_W_PROFILE=1` emits rank-zero `[PURE_W_EXECUTION]` markers and
packing, W-build, Q aggregation, mixing, graph and basis-force times. Turn this
off for performance comparisons. These markers distinguish actual basis-first
execution from a mathematically valid fallback.

```bash
python3 dev_test/sh_pure_w_execution_check.py /absolute/path/to/mlp-sus2
bash dev_test/sh_pure_w_execution_state_check.sh MODEL CFG OBJECT_DIR
c++ -std=c++11 -O2 -I. codex_tests/pure_w_packed_kernel_check.cpp \
  dev_src/pure_w_execution.cpp -o /tmp/packed-check
/tmp/packed-check
```

The execution suite covers O=1/2/4/8, K/L/LK sharing, decoupled scaling, zero W,
nonparallel nonzero gate/W/head parameters, radial size larger than K*O, save/load,
cache disabled/enabled, E/F/stress finite differences and mixed gradients.
The packed unit also covers O=3/16, value-only kernels and in-place repacking.
The execution suite also legally reverses independent output-channel blocks in
saved models. It verifies that original and reordered models give the same EFS
and that the batched graph falls back instead of reassigning physical lanes.
The state-test driver accepts `CXX`, `SUS2_TEST_CPPFLAGS`,
`SUS2_TEST_LDFLAGS`, `SUS2_TEST_RUNNER` and `SUS2_TEST_BUILD_DIR` for MPI/MKL
builds. Link against a clean object set rebuilt after the model header changes.

Formal model training directories are not modified by these tests. Benchmark
inputs and pre-change binaries are frozen separately; fixed-call same-host EFS
and gradient timings are not interchangeable with full training wall time.

## Measured CPU result (2026-09-05)

Final candidate, queue 33, one host `b07u17a`, 40 MPI ranks, four interleaved
repeats, identical frozen O=1/2/4/8 models and three 96-atom VO2 configurations.
Each rank repeats the fixed-call diagnostic; the table is the rank-zero warm
internal kernel time per configuration, **not** distributed throughput or an
end-to-end MD measurement. Gradient calls 2-3 and EFS calls 4-12 are retained.
STD/STDD and parameter regularization are disabled in this timing probe.
The control is the pre-change `9d72433` executable with the **same O/model**,
not the separate historical 9.1 architecture. Positive changes mean slower.

| O | Old EFS ms | Auto EFS ms | Change | Old gradient ms | Auto gradient ms | Change |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 124.248 | 126.430 | +1.76% | 371.066 | 375.161 | +1.10% |
| 2 | 137.056 | 139.139 | +1.52% | 399.565 | 400.967 | +0.35% |
| 4 | 160.474 | 162.031 | +0.97% | 461.957 | 465.538 | +0.78% |
| 8 | 206.655 | 181.144 | -12.34% | 576.338 | 586.534 | +1.77% |

O8 EFS median absolute deviations were 0.307 ms (old) and 0.372 ms (auto).
The gain is an inference-kernel result: this change does **not** establish a
training-backward speedup. Small-O and mixed-call overheads remain visible;
they are not rounded away as zero. The graph and packed-only alternatives stay
opt-in because their preliminary gains were not sufficiently general.

Final validation passed the execution/FD matrix including legally reordered
graphs; all four real VO2 EFS comparisons (maximum scaled difference
`9.9998e-13`); and two-rank changing-geometry/coefficient/cache-state checks
(maximum scaled E/F/stress difference `1.01086e-12`). Packed W/W-prime unit
error was `1.77636e-15`. Local UBSan and its state-transition checks also passed.
The full 12-step objective trajectories for O1/2/4/8 were checked before the
last cache/graph guards; the cache refinement additionally rechecked O8 training.
The final guards do not change the default training arithmetic. The final
source was rerun through all-width FD, EFS, reordered-graph and state tests.

Evidence is under the server checkout's
`.codex_tmp/pure_w_inference_20260905/r6/`, with `pilot_4246415/summary.json`,
`checks/`, `execution-check.log` and `candidate.sha256`. The corresponding
local archive is under the workspace-level `.codex_tmp/pure_w_inference_20260905/`.
Linux binary SHA256:
`5dfa614ab7dbba17bda6f09c73dcae911f7136717851b0bd6b6de8679fc07aca`.

The model-adapted historical gate 500x55 regression remains a **separate queued
test**, job `4246418`, queue `1t81c`, one host/120 ranks, at this report's
checkpoint. It retains the historical 75-step pretraining and 55 formal steps.
It is not counted among the completed tests above. Prior candidate-only queued
jobs were cancelled before execution; active formal training was not changed.
