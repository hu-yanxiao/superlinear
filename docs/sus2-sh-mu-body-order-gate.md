# SUS2-SH Mu Body Linear-Combo Gate

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

## Branch And Scope

This experiment lives on:

```text
<historical-branch>
```

It is branched directly from SUS2-SH `origin/main`. The server work directory is:

```text
<historical-path>
```

The server binary is:

```text
<historical-path>
```

The target smoke/test case directory is:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/
```

## Server Binary Source Map

Current SUS2-SH main/developer reference binary for branch speed comparisons:

```text
<historical-path>
```

Source relationship:

```text
server source tree: <historical-path>
server checkout branch name: developer
GitHub repository: git@github.com:hu-yanxiao/SUS2-SH.git
GitHub branches at this commit: main, developer
commit: dd3010572b8d34fc53e7c371ea93ecc2104f74c1
binary SHA-256: aca08ae4d763c155af18dfa2c6a51b4c39379eef7712c29179d787ac4f499659
```

Current mu-body-order gate test binary:

```text
<historical-path>
```

Source relationship:

```text
server source mirror: <historical-path>
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.local_tmp/sus2-sh-developer-tanh
GitHub branch: <historical-branch>
binary code commit: a827a02c13978fd7f06be757497d6b1cdb5bff9c
binary SHA-256: 0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
```

The server mu-body gate source mirror is not a git checkout. Treat the local
worktree and GitHub branch as the source authority, then sync confirmed files to
the server mirror before compiling. Documentation-only commits made after the
binary code commit do not imply that the binary must be rebuilt.

Current optimized LAMMPS mu-body gate binary:

```text
<historical-path>
```

Source relationship as of 2026-06-16 03:00 CST:

```text
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.local_tmp/sus2-sh-developer-tanh
local branch: <historical-branch>
GitHub/code commit: a8ec8a98b41e99f12821567d61f5cd8b602c4221
server SUS2-SH source mirror: <historical-path>
server LAMMPS build src: <historical-path>
LAMMPS binary SHA-256: e77ee6eaa74ef6eb9ad8b4723dc649be907856deb881ef734981cd77cd864054
pair_sus2_mtp.cpp SHA-256: 51a09e0d403af744b2e05770a45021b4adc9cbc508ddb245c9ff3041b67fea80
pair_sus2_mtp.h SHA-256: 6bc7a1624aae0b0f162f02bb4fcc77f983a8700a4960f81f78ad1f040918bd55
```

The stable binary above is the same file content as:

```text
<historical-path>
```

Final exact LAMMPS optimization status:

```text
test directory: <historical-path>
main LAMMPS baseline binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
40-rank parity job for final SHA-256 e77...4054: 3801127, DONE
40-rank direct parity job final vs group_contract reference: 3801130, DONE
40-rank performance job for final SHA-256 e77...4054: 3801128, DONE
```

Verified 40-rank parity for the final binary against the immediately previous
branchless exact candidate:

```text
combo abs_dE=0, max_force_diff=0
full abs_dE=0, max_force_diff=0
```

Verified 40-rank direct parity for the final stable binary against the accepted
grouped-contraction reference:

```text
reference binary: <historical-path>
final binary: <historical-path>
combo abs_dE=0, max_force_diff=0
full abs_dE=0, max_force_diff=0
```

Final 40-rank LAMMPS speed comparison:

```text
summary: <historical-path>
host: b07u32a
replicate: 2 2 2
run steps: 5000
atoms after replicate: 384
```

| Case | Mean Loop Time | Relative To Main |
| --- | ---: | ---: |
| main old gate | `14.760067 s` | `1.0000x` |
| mu-body linear-combo | `16.754067 s` | `1.1351x` |
| mu-scalar full | `17.457800 s` | `1.1828x` |

Both active mu gate modes are within the target `1.2x` main-branch LAMMPS
speed envelope on this benchmark. The exact optimizations promoted in the
final binary are:

- mode-specific gate-force cache strategy: body-linear-combo uses cached basic
  Jacobians; mu-scalar-full caches radial values/derivatives and recomputes the
  compact SH Jacobian on demand;
- grouped \(\mu\)-major SH Jacobian contraction for the on-demand full mode;
- direct stride-4 body-linear-combo gate-signal and adjoint contractions;
- branchless full-mode on-demand contraction over grouped SH basic terms;
- single-gate no-center main-layer force/adjoint path, leaving the double-gate
  center path separate.

Current LAMMPS mu-body gate trial binary for exact gate performance work:

```text
<historical-path>
```

Source relationship as of 2026-06-15 21:16 CST:

```text
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.local_tmp/sus2-sh-developer-tanh
local branch: <historical-branch>
GitHub commit: c28fe23
server SUS2-SH source mirror: <historical-path>
server LAMMPS build src: <historical-path>
LAMMPS trial binary SHA-256: e2f8f2d6e23eed2196f28c207fd2b2da9ef96022943853dcdcc7555dae42c270
pair_sus2_mtp.cpp SHA-256: b5ed9f68d797655d581d1e969390e29ce1f925bef0f2bcdafac1f8a32db06872
pair_sus2_mtp.h SHA-256: 8d8e1c3e68b545f3e211e6accdbc2aaf29356cbf66333afc3467675680809ebd
```

The interface source hashes above were checked to match between the local
worktree, the server SUS2-SH source mirror, and the LAMMPS build source tree.

Current exact LAMMPS optimization status for this trial:

```text
test directory: <historical-path>
main LAMMPS baseline binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
40-rank parity job for trial SHA-256 e2f8...c270: 3800824, DONE
40-rank speed job for trial SHA-256 e2f8...c270: 3800825, DONE
96-rank parity job for trial SHA-256 e2f8...c270: 3800826, submitted
96-rank speed job for trial SHA-256 e2f8...c270: 3800827, submitted
```

Verified login-node and 40-rank parity for the current trial:

```text
combo abs_dE=0, max_force_diff=0
full login max_force_diff=6.634440e-15
full 40-rank max_force_diff=6.689640e-15
```

Clean 40-rank LAMMPS speed comparison for trial SHA-256 `e2f8...c270`:

```text
summary: <historical-path>
host: b08u22a
replicate: 2 2 2
run steps: 5000
atoms after replicate: 384
```

| Case | Mean Loop Time | Relative To Main | Relative To Accepted |
| --- | ---: | ---: | ---: |
| main old gate | `11.872133 s` | `1.0000x` | - |
| accepted mu-body linear-combo | `17.081967 s` | `1.4388x` | `1.0000x` |
| e2f8 mu-body linear-combo | `16.299800 s` | `1.3729x` | `0.9542x` |
| accepted mu-scalar full | `18.118667 s` | `1.5262x` | `1.0000x` |
| e2f8 mu-scalar full | `17.290133 s` | `1.4564x` | `0.9543x` |

Short 40-rank profile for trial SHA-256 `e2f8...c270`:

```text
job id: 3800839
summary: <historical-path>
host: b08u03a
profile steps: 20
```

| Mode | Mean Total | First Layer | Forward Comm | Main Layer | Reverse Comm | Gate Force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mu-body linear-combo | `0.003324401 s` | `0.001049122 s` | `0.000482758 s` | `0.001607793 s` | `0.000611042 s` | `0.000525844 s` |
| mu-scalar full | `0.003486166 s` | `0.001122569 s` | `0.000550462 s` | `0.001393776 s` | `0.000564622 s` | `0.000735240 s` |

The stage timings are reduced by MPI max per stage, so the component means do
not need to sum exactly to the total. The actionable signal is that full mode
still spends more in gate-force adjoint work, while communication is already a
large fraction of the 40-rank short-run profile.

Rejected exact LAMMPS signal-derivative-cache candidate:

```text
candidate binary: <historical-path>
candidate SHA-256 after alias fix: ba05caca66e9690b15b189acc6c63449aa1db4df8e5e114526e10cc0cd4c3c34
login parity: combo max_force_diff=1.0e-20, full max_force_diff=1.3878e-16
login profile combo mean_total=0.087373956 s
login profile full mean_total=0.219957343 s
decision: rejected; exact but much slower than e2f8 due 4/20-RHS product-graph reverse work moved into the first layer
```

This candidate precomputed \(\partial h_{i,s}/\partial r_{ij}\) for every gate
signal \(s\) and edge, reducing the late gate-force cache from
`alpha_index_basic_count` entries per edge to `gate_signal_stride` entries per
edge. The math is exact after explicitly handling product-graph square terms
where `a0 == a1`, but the extra multi-RHS scalar-product reverse pass dominates
the saved memory traffic on the tested model. Do not promote this candidate.

Do not use this historical directory as the current main reference:

```text
<historical-path>
```

It was observed at detached commit `6a5ca72dd5a31766c0e0da0711a44682d8fa461c`,
older than the current main/developer reference, and produced an invalid speed
comparison because it rejected `RBLaguerre_log1p` during `init-sh`.

Fresh 40-rank CPU speed comparison on 2026-06-15:

```text
job id: 3797952
summary: /work/phy-weigw/hyx/xxx-b/test/cmp_20260615_093351_main_combo_full_summary.txt
train cfg: /work/phy-weigw/hyx/xxx-b/test/train.cfg
max-iter: 30
radial basis: RBLaguerre_log1p
```

| Case | Binary | Gate option | Runtime | Relative to main |
| --- | --- | --- | ---: | ---: |
| main/developer old gate | `SUS2-SH-work-<historical-branch>` | `--two-layer-gate-body-order=5` | `462 s` | `1.00x` |
| mu-body linear-combo | `SUS2-SH-mu-body-gate-work-<historical-branch>` | `--two-layer-gate-mode=mu-body-linear-combo` | `654 s` | `1.42x` |
| mu-scalar full | `SUS2-SH-mu-body-gate-work-<historical-branch>` | `--two-layer-gate-mode=mu-scalar-full` | `603 s` | `1.31x` |

All three runs exited with code 0.

Submitted neighbor-site full test job in that directory:

```text
job id: 3791918
script: <historical-path>
log: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/mu_body_gate_3000_additive.log
```

Submitted double-site full test job in that directory:

```text
job id: 3792530
script: <historical-path>
log: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/mu_body_double_gate_3000_additive.log
status at submission: PEND
```

Double-site implementation commit hash:

```text
e9de9dd
```

## Formula

The gate first forms one shared scalar per body order:

\[
H_{a,B}
=
\sum_{q:\operatorname{body\_order}(s_q)=B}
c_q s_q(a),
\qquad
B=2,\ldots,k_{\max}+1 .
\]

Each \(\mu\) channel then mixes those body-order scalars with its own
coefficients:

\[
h_{a,\mu}
=
\sum_{B=2}^{k_{\max}+1}
W_{\mu B}H_{a,B}.
\]

Equivalently,

\[
h_{a,\mu}
=
\sum_{B=2}^{k_{\max}+1}
W_{\mu B}
\sum_{q:\operatorname{body\_order}(s_q)=B}
c_q s_q(a).
\]

The additive tanh coefficient is species-only:

\[
a_{z,\mu}=a_z .
\]

The default neighbor-site SH moment is:

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
\left[1+A\tanh(a_{z_j}h_{j,\mu})\right]
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

The optional double-site gate first defines

\[
g_{a,\mu}=1+A\tanh(a_{z_a}h_{a,\mu})
\]

and then evaluates

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
g_{i,\mu}g_{j,\mu}
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

Default initialization is:

\[
c_q^{(0)}=0,\qquad W_{\mu B}^{(0)}=1,\qquad a_z^{(0)}=1.
\]

Thus \(h_{a,\mu}^{(0)}=0\), the initial gate factor is exactly 1, and the
model starts from ordinary SUS2-SH while \(c_q\) still has a nonzero first-step
gradient.

## CLI And Model Rules

- `--two-layer-gate-body-order` is not a normal option for this branch. Passing
  it explicitly is an error.
- `--two-layer-gate` requires:

  \[
  \texttt{body-order} \ge \texttt{k-max}+1
  \]

- `two_layer_gate_scalar_indices` automatically contains scalar basis functions
  with body orders `2..k_max+1`.
- `two_layer_gate_weights` is the shared \(c_q\) vector and has length equal to
  `two_layer_gate_scalar_indices`.
- `two_layer_gate_body_mix_weights` is the \(W_{\mu B}\) matrix flattened as
  `radial_func_count * k_max`.
- The branch does not save or use `two_layer_gate_bias` / \(G_0\). Zero gate
  weights give \(h_{j,\mu}=0\), hence \(1+A\tanh(0)=1\), so the model reduces to
  ordinary SUS2-SH.
- New gate models write:

  ```text
  two_layer_gate_mode = mu-body-linear-combo|mu-scalar-full
  two_layer_gate_site_mode = neighbor|double
  ```

  Legacy two-layer gate metadata is rejected instead of being silently read as
  the new mode.
- `--two-layer-residual` is not supported by this mu-body-order gate branch.

## Implementation Notes

- Gate caches are per atom times mu: \(h_{j,\mu}\), not one scalar per atom.
- The implementation caches \(s_q\), compresses them to the small \(H_B\) vector,
  and then applies the \(W_{\mu B}\) matrix. This keeps the core signal build at
  \(O(N_q+N_\mu k_{\max})\) per atom instead of \(O(N_\mu N_q)\).
- Gate adjoints are also per atom times mu. During reverse mode, adjoints first
  accumulate into body-order buckets and then into the shared \(c_q\) scalar
  weights and scalar-value chain.
- The optional `mu-scalar-full` mode keeps one independent gate coefficient per
  \((\mu,q)\) pair while reusing the same scalar basis values \(s_q\). Its
  forward signal, force-chain scalar seeds, directional derivative, and gate
  weight gradient are batched through BLAS matrix/vector products instead of
  nested \(\mu,q\) loops.
- The existing tanh cache, edge cache, site derivative cache, and shared-radial
  gate paths are preserved.
- CPU LAMMPS reads `two_layer_gate_mode` and `two_layer_gate_site_mode`,
  supports both `mu-body-linear-combo` and `mu-scalar-full`, communicates
  compact per-body-order gate signals/adjoints for `mu-body-linear-combo`,
  communicates per-mu signals/adjoints for exact `mu-scalar-full`, applies the
  same neighbor or double-site gate factor, and rejects legacy gate fields.
- CPU LAMMPS now propagates gate force-chain derivatives by caching first-layer
  basic moment Jacobians per active edge and doing one combined scalar-product
  reverse pass per center atom, avoiding the old edge-by-scalar derivative
  materialization.
- Kokkos device gate evaluation is explicitly rejected for this branch rather
  than silently using the old scalar-gate kernels.

Interface verification:

```text
CPU LAMMPS pair source built into the independent branch LAMMPS work tree:
<historical-path>
Kokkos device pair style rejects mu-body-order gate models at settings time.
```

## Final Server Sync And Speed Decision

Final synchronized server binary:

```text
<historical-path>
```

Final binary SHA-256:

```text
0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
```

Independent CPU LAMMPS binary for this branch:

```text
<historical-path>
<historical-path>
<historical-path>
```

LAMMPS binary SHA-256:

```text
7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
```

Final build and smoke evidence on the server:

```text
BUILD_RC=0
SMOKE_RC=0
```

The accepted optimization keeps the exact-body mu math but merges per-body-order
gate adjoints once per atom instead of rerunning the scalar reverse path once
for every body-order bucket. This preserves the existing tanh cache, edge
primitive cache, site-derivative cache, and shared-radial paths.

500x55 profile directory:

```text
<historical-path>
```

Training-speed decision at matched profile settings:

| Version | Gradient total | Forward total | Decision |
| --- | ---: | ---: | --- |
| legacy scalar gate | `71.738 ms` | `61.323 ms` | old reference |
| accepted exact-body opt1 | `76.593 ms` | `62.535 ms` | synchronized |
| rejected final clear/cache candidate | `79.046 ms` | `62.561 ms` | rejected |

The final clear/cache candidate was stopped after collecting profile data
because it remained slower at `calls=570`. Its LSF job `3792234` therefore
ended with MPI interrupt text from the intentional stop, not from a correctness
failure.

## Double-Site Optimization Update 2026-06-12

The final double-site optimization keeps the existing exact-body gate math and
the existing neighbor-site fast paths. It adds a small center-gate scratch cache
for repeated `PrepareTwoLayerGatePairMuBuffers()` calls with the same center
atom and signal pointer, avoids computing `sech^2` when no derivative buffer is
requested, and avoids materializing the neighbor multiplier buffer unless the
caller asks for it. The LAMMPS CPU pair style also reuses per-pair mu adjoint
scratch arrays instead of allocating `std::vector<double>` inside the edge
adjoint loops.

Matched 500x55 profile directory:

```text
<historical-path>
```

Final matched profile:

| Mode | Runtime | Linear total | Gradient total | Forward total |
| --- | ---: | ---: | ---: | ---: |
| neighbor-site | `478 s` | `428.333 ms` | `90.000 ms` | `68.500 ms` |
| double-site | `485 s` | `423.258 ms` | `96.217 ms` | `69.497 ms` |

The double-site runtime overhead at the matched profile settings is therefore
`485 / 478 = 1.015x`. Before this update the same comparison was `501 / 478 =
1.048x`, so the measured double-site overhead dropped from about `4.8%` to
about `1.5%`.

Fresh server verification after this optimization:

```text
mlp-sus2 build: PASS, SHA-256 86eb63f7d3a8087e335ebe0d78e7681c1363649107ba63cd67f32b568e87ec6c
sh_two_layer_gate_mu_body_order_check.sh: PASS
sh_two_layer_gate_double_mode_check.sh: PASS
sh_two_layer_gate_loss_gradient_check.sh: PASS
sh_two_layer_gate_force_fd_check.sh: PASS
LAMMPS neighbor additive smoke: PASS, max_abs=8.694700e-05
LAMMPS double additive smoke: PASS, max_abs=8.695400e-05
LAMMPS 1-rank vs 2-rank: PASS, max_abs=0
```

## Full-Mode And LAMMPS Efficiency Update

The retained `mu-scalar-full` mode is no longer a legacy fallback. It uses the
same scalar basis cache as `mu-body-linear-combo`, but stores an independent
flattened weight matrix over \((\mu,q)\). The hot training paths use BLAS for
the full weight projection, scalar-seed projection, directional derivative,
outer-product gate-weight gradient, and scalar-parameter seed accumulation.

Matched smoke-profile logs in `/work/phy-weigw/hyx/xxx-b/test`:

| Mode | Runtime | Forward total | Prepare gate | Main forward | Force chain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `262 s` | `165.620 ms` | `83.691 ms` | `58.805 ms` | `23.124 ms` |
| `mu-scalar-full` | `283 s` | `148.455 ms` | `78.817 ms` | `58.670 ms` | `10.968 ms` |

Fresh server verification after rebuilding current source:

```text
mlp-sus2 build: PASS, SHA-256 0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
CPU LAMMPS no-IPO build: PASS, SHA-256 7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
sh_two_layer_gate_mu_scalar_full_init_check.sh: PASS
sh_two_layer_gate_mu_scalar_full_loss_gradient_check.sh: PASS
sh_two_layer_gate_loss_gradient_check.sh: PASS
sh_two_layer_gate_force_fd_check.sh: PASS
LAMMPS mu-body-linear-combo smoke: PASS, max_abs=0
LAMMPS mu-body-linear-combo neighbor additive: PASS, max_abs=8.790000e-05
LAMMPS mu-body-linear-combo double additive: PASS, max_abs=8.788400e-05
LAMMPS mu-scalar-full smoke: PASS, max_abs=0
LAMMPS mu-scalar-full neighbor additive: PASS, max_abs=8.694700e-05
LAMMPS mu-scalar-full double additive: PASS, max_abs=8.695400e-05
LAMMPS 1-rank vs 2-rank: PASS, max_abs=0
```

## LAMMPS Gate-Mode Efficiency Update 2026-06-15

The independent CPU LAMMPS branch binary was rebuilt from the current
`<historical-branch>` source after optimizing the exact gate dataflow:

```text
source mirror: <historical-path>
binary: <historical-path>
binary SHA-256: 7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
pre-optimization backup: <historical-path>
```

Implementation notes:

- `mu-body-linear-combo` stores and communicates only the body-channel signals
  \(H_b\) and body-channel adjoints. Each receiving rank reconstructs
  \(h_\mu=\sum_b c_{\mu b}H_b\) locally before the tanh gate. During reverse
  communication, each per-mu gate adjoint is contracted back to
  \(dE/dH_b=\sum_\mu c_{\mu b}\,dE/dh_\mu\). This is algebraically exact and
  reduces the l4k4 communication width from 20 doubles to 4 doubles per atom.
- `mu-scalar-full` remains exact and still communicates the 20 per-mu
  \(h_\mu\) values/adjoints. The scalar basis is shared, and a scalar-major
  transposed weight view is used for the full-mode gate-force reverse
  contraction.
- The first-layer edge derivative cache now reserves once and writes by offset
  instead of appending large derivative slices with repeated vector `insert()`.
- LAMMPS gate communication pack/unpack uses exact block copies for contiguous
  gate-signal buffers; reverse unpack still accumulates per entry because LAMMPS
  reverse communication returns contributions to owned atoms.

Real B-system LAMMPS benchmark:

```text
directory: <historical-path>
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
queue/cores: 1t88c, 96 MPI ranks
run: 5000 steps, tabstep 0.0005, trained models converted to *_lmp radial basis
main binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
new-branch binary: <historical-path>
```

| Model/mode | Mean loop time | Relative to main |
| --- | ---: | ---: |
| main old gate | `4.339053 s` | `1.0000x` |
| `mu-body-linear-combo` | `5.337373 s` | `1.2301x` |
| `mu-scalar-full` | `6.796363 s` | `1.5663x` |

The first body-channel communication build on the same benchmark gave
`mu-body-linear-combo = 5.396523 s` (`1.2254x` against that run's main
baseline) and `mu-scalar-full = 6.999947 s` (`1.5894x`). The final build keeps
the full-mode scalar-major reverse optimization because it improves the full
absolute loop time, while the combo absolute time remains in the same range.

Numerical parity against the pre-body-channel-communication binary:

```text
directory: <historical-path>
job: 3798420
combo abs_dE=0.000000e+00, max_force_diff=8.600000e-19, rms_force_diff=2.533799e-20
full  abs_dE=0.000000e+00, max_force_diff=3.903132e-15, rms_force_diff=6.774959e-16
```

Short 200-step profile on the same B 2x2x2 setup:

```text
directory: <historical-path>
job: 3798421
```

| Mode | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `0.00116393` | `0.00028527` | `0.00034180` | `0.00043722` | `0.00046945` | `0.00016602` |
| `mu-scalar-full` | `0.00269330` | `0.00032071` | `0.00110001` | `0.00042119` | `0.00117981` | `0.00021778` |

The remaining `mu-scalar-full` gap is dominated by exact per-mu forward and
reverse communication. With arbitrary independent \(w_{\mu q}\), communicating
20 per-mu \(h_\mu\) values/adjoints is the compact exact representation; the
alternative of communicating all scalar basis values would be much larger.
Approximate low-rank compression is intentionally not used. The tested B-model
full weight matrix is numerically close to low-rank, but not exactly low-rank:
rank-1 SVD reconstruction leaves about `1e-4` max absolute residual and rank-7
still leaves about `3e-10`. The body-mix matrix is also only numerically close
to rank-1, with about `1.7e-10` rank-1 residual. These are approximation
opportunities, not exact algorithmic reductions, and are therefore out of scope.

### No-Zero Gate-Value LAMMPS Update 2026-06-15

The current accepted CPU LAMMPS branch binary adds two exact dataflow
optimizations on top of the body-channel communication build:

- The first-layer gate moment cache is reused as a member buffer instead of
  reallocated every step, and the scalar scratch buffer is resized once.
- `mu-scalar-full` accumulates the full gate projection through a scalar-major
  transposed weight view so each scalar value is loaded once and the 20
  \(\mu\)-weights are contiguous.
- `two_layer_gate_values` is no longer pre-zeroed over all local and ghost atoms
  before the gate forward pass. Each local atom's gate signal is fully written by
  the first-layer pass, static fixed atoms are copied from their cache, and ghost
  signals are fully written by LAMMPS forward communication. The adjoint buffer
  is still zeroed because reverse communication accumulates into it.

Accepted independent CPU LAMMPS binary:

```text
binary: <historical-path>
binary SHA-256: 9e89565433abd450bf49ecb85e702b84f6bf290614c6b499d3ec353e9cae83d9
source code commit: f8a098f643eecfdfe9b9b2e0fc144d8046518083
previous backup: <historical-path>
trial binary: <historical-path>
source mirror: <historical-path>
SUS2-SH server mirror: <historical-path>
```

Numerical parity against the previous canonical branch binary:

```text
directory: <historical-path>
job: 3798701
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=5.922340e-15, rms_force_diff=1.163142e-15
```

Real B-system LAMMPS speed comparison in one matched 96-rank job:

```text
directory: <historical-path>
job: 3798702
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
run: 5000 steps, tabstep 0.0005, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main |
| --- | ---: | ---: |
| main old gate | `4.514740 s` | `1.0000x` |
| canonical `mu-body-linear-combo` before this update | `5.518320 s` | `1.2223x` |
| cache/scalar-major `mu-body-linear-combo` trial | `5.479610 s` | `1.2137x` |
| accepted no-zero `mu-body-linear-combo` | `5.434567 s` | `1.2037x` |
| canonical `mu-scalar-full` before this update | `6.908270 s` | `1.5302x` |
| cache/scalar-major `mu-scalar-full` trial | `6.894727 s` | `1.5272x` |
| accepted no-zero `mu-scalar-full` | `6.755803 s` | `1.4964x` |

Short 200-step profile with the accepted binary:

```text
directory: <historical-path>
job: 3798812
```

| Mode | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `0.00122208` | `0.00027351` | `0.00034509` | `0.00042426` | `0.00056258` | `0.00015259` |
| `mu-scalar-full` | `0.00310239` | `0.00037310` | `0.00140737` | `0.00045936` | `0.00150011` | `0.00023858` |

The no-zero update gives a small exact improvement, but the same conclusion
holds: on this very small 384-atom/96-rank benchmark, the remaining full-mode
gap is communication dominated. `mu-body-linear-combo` must communicate 4
body-channel signals/adjoints for the l4k4 model; `mu-scalar-full` must
communicate 20 independent per-\(\mu\) signals/adjoints. The LAMMPS pack/unpack
path already sends contiguous signal blocks. Reducing the full-mode width below
20 would require an exact low-rank/sparse identity in the trained weight matrix
or a change in the mathematical model; no approximate compression is used.

### Raw Edge/Comm Direct LAMMPS Update 2026-06-15

The current accepted CPU LAMMPS branch binary adds exact data-layout and hot-loop
optimizations on top of the no-zero build:

- Gate edge staging uses LAMMPS-managed raw contiguous buffers instead of
  per-step `std::vector` clear/reserve/push/resize traffic.
- The common l4k4 `mu-body-linear-combo` path specializes the four body-channel
  signal and adjoint maps while keeping the same per-\(\mu\) accumulation order.
- Forward and reverse gate communication add fixed-width stride 4 and stride 20
  pack/unpack paths. The communicated values and adjoint sums are unchanged.

Accepted independent CPU LAMMPS binary:

```text
binary: <historical-path>
binary SHA-256: 89e29919ddf899990f5f8e4e1e0a55be5ce0a31bc6bc04c0acb038bc795714ad
source code commit: 560558dee1f1f2bb38b7b815039256d669d1984e
previous backup: <historical-path>
trial binary: <historical-path>
source mirror: <historical-path>
SUS2-SH server mirror: <historical-path>
```

Numerical parity against the previous canonical branch binary:

```text
directory: <historical-path>
job: 3800591
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
```

Real B-system LAMMPS speed comparison in one matched 96-rank job:

```text
directory: <historical-path>
job: 3800594
node: be1u39a
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
run: 5000 steps, tabstep 0.0005, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main | Relative to previous branch binary |
| --- | ---: | ---: | ---: |
| main old gate | `4.336693 s` | `1.0000x` | n/a |
| previous `mu-body-linear-combo` | `5.574307 s` | `1.2854x` | `1.0000x` |
| accepted raw-edge/comm-direct `mu-body-linear-combo` | `5.321253 s` | `1.2270x` | `0.9546x` |
| previous `mu-scalar-full` | `6.871523 s` | `1.5845x` | `1.0000x` |
| accepted raw-edge/comm-direct `mu-scalar-full` | `6.809597 s` | `1.5702x` | `0.9910x` |

An earlier same-node full-speed run for the direct version without the
forward-pack specialization gave `mu-body-linear-combo = 5.313430 s`
(`1.1989x` relative to that job's main run) and `mu-scalar-full = 6.914083 s`.
Across the matched jobs, the accepted code gives a repeatable improvement over
the previous branch binary. The remaining `mu-scalar-full` gap is still
communication dominated because this mode exactly carries 20 independent
per-\(\mu\) gate signals and adjoints for the l4k4 model.

### Direct Derivative-Flow LAMMPS Update 2026-06-15

The current retained CPU LAMMPS branch source adds another exact memory-flow
cleanup on top of the raw-edge/comm-direct build:

- The first-layer gate SH pass writes the raw edge derivative planes directly
  into the gate derivative cache, instead of first writing the ordinary
  `moment_jacobian_{x,y,z}` slabs and then copying them.
- The first-layer gate moment buffer is cleared only through the gate scalar
  dependency width, not the full model moment width.
- The late gate-force scalar seed is accumulated directly into
  `nbh_energy_ders_wrt_moments`; the temporary scalar scratch vector is removed.

Retained independent CPU LAMMPS binary:

```text
binary: <historical-path>
binary SHA-256: 08b53882086b4e99eabfbbed60a84f5b00932d0ac33a9a878d1d3a754c3ac0f8
source code commit: 826415d4dcc64ee0a38ba445c2302d6f7aa0d5ec
source mirror: <historical-path>
SUS2-SH server mirror: <historical-path>
pair_sus2_mtp.cpp SHA-256: 6f80c0d1a1df873806050e02bf7f2c9c9d0e16b73d19abf288d76b4b5a2ae375
pair_sus2_mtp.h SHA-256: db75fb2da94ef56eed5d19e9260fddd589f39d1e8402fae74c878ddc582ea810
```

Current login-node parity against the previous `e2f8` canonical branch binary:

```text
script: /work/phy-weigw/20260321_Test/run_directfill_seed_login_parity.sh
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
```

Current login-node short profile against the previous `e2f8` canonical branch
binary:

```text
script: /work/phy-weigw/20260321_Test/run_directfill_seed_login_profile.sh
directory: <historical-path>
```

| Mode/binary | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2f8` `mu-body-linear-combo` | `0.083261775` | `0.029778039` | `0.000018864` | `0.038592064` | `0.000013142` | `0.014833429` |
| retained `mu-body-linear-combo` | `0.079464204` | `0.025972551` | `0.000019252` | `0.038679035` | `0.000014232` | `0.014754995` |
| `e2f8` `mu-scalar-full` | `0.088467262` | `0.031465805` | `0.000089547` | `0.037235977` | `0.000054461` | `0.019555161` |
| retained `mu-scalar-full` | `0.082831310` | `0.026925359` | `0.000088189` | `0.036981662` | `0.000055227` | `0.018714115` |

The login-node profile is only a hot-path direction check; the real LAMMPS
speed comparison remains the submitted 40-rank/96-rank B-system jobs in the same
test directory.

Completed 40-rank short profile on the B-system 2x2x2 LAMMPS case:

```text
job: 3800991
node: b07u38a
summary: <historical-path>
```

| Mode/binary | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2f8` `mu-body-linear-combo` | `0.003463722` | `0.001082097` | `0.000418600` | `0.001682915` | `0.000592926` | `0.000633548` |
| retained `mu-body-linear-combo` | `0.003338736` | `0.000965856` | `0.000370418` | `0.001693265` | `0.000599926` | `0.000610887` |
| `e2f8` `mu-scalar-full` | `0.003881440` | `0.001175615` | `0.000508676` | `0.001652033` | `0.000620843` | `0.000888017` |
| retained `mu-scalar-full` | `0.003810493` | `0.001045825` | `0.000467634` | `0.001706453` | `0.000669589` | `0.000885084` |

Completed 40-rank real LAMMPS speed comparison on the B-system 2x2x2 case:

```text
job: 3801017
node: b07u38a
summary: <historical-path>
run: 5000 steps, 384 B atoms, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main | Relative to e2f8 |
| --- | ---: | ---: | ---: |
| main old gate | `14.571533 s` | `1.0000x` | n/a |
| `e2f8` `mu-body-linear-combo` | `17.898400 s` | `1.2283x` | `1.0000x` |
| retained no-IPO `mu-body-linear-combo` | `17.440233 s` | `1.1969x` | `0.9744x` |
| retained IPO `mu-body-linear-combo` | `17.425167 s` | `1.1958x` | `0.9736x` |
| `e2f8` `mu-scalar-full` | `19.782833 s` | `1.3576x` | `1.0000x` |
| retained no-IPO `mu-scalar-full` | `19.227633 s` | `1.3195x` | `0.9719x` |
| retained IPO `mu-scalar-full` | `19.432767 s` | `1.3336x` | `0.9823x` |

The retained no-IPO build is the best full-mode result in this matched job, and
IPO is neutral/slightly better only for combo. The direct derivative-flow
cleanup therefore remains useful but not sufficient to make either new gate
mode reach the old main gate speed on this small 40-rank B benchmark.

At the time this note was updated, the remaining pending job was:

```text
96-rank speed: 3801039
```

Rejected exact fast-dispatch candidate:

```text
trial binary: <historical-path>
trial SHA-256: 67b4c698d963eec3f9e1f48542bd067eb5c82530615d8d0a9a1dc2de204d6e34
login parity: combo/full zero energy and force differences against the retained binary
login profile: combo total 0.079477085 -> 0.079802535, full total 0.083516155 -> 0.083533995
decision: rejected; exact but no measured speed gain.
```

The benchmark models were also checked for exact structural compression
opportunities. The trained `mu-body-linear-combo` body-mix rows and
`mu-scalar-full` rows are numerically close to each other, but they are not
exactly duplicate rows. Therefore the communication width cannot be reduced
without changing the mathematical model or using approximation, which is out of
scope for this branch.

### Full-Matrix Optimization Policy 2026-06-16

LAMMPS optimization candidates are now evaluated as model-family backends, not
as single-case fixes. Each exact candidate must be benchmarked over the full
representative matrix before it is promoted:

```text
l/k: l2k2, l2k3, l3k3, l4k4
factor pruning: legacy, q_total
gate weight mode: mu-body-linear-combo, mu-scalar-full
LAMMPS model form: trained model converted to *_lmp read-table radial basis
LAMMPS cell: B-system first training structure, replicate 6 6 6
baseline: current SUS2-SH main/developer LAMMPS binary
branch binary: independent mu-body-order-gate LAMMPS binary
```

The goal remains a mathematically exact implementation. If one backend is
consistently best across the matrix, it becomes the default path. If different
model sizes or pruning modes favor different exact kernels, model loading should
dispatch by stable model metadata such as \(l_{\max}\), \(k_{\max}\),
`alpha_index_basic` size, gate mode, signal stride, and scalar count. A
candidate that only improves one corner case is retained only as a specialized
dispatch path when the full matrix shows that it does not regress the other
cases.

The current candidate-matrix harness records both raw loop times and
main-relative ratios for all cases, so backend selection can be based on the
observed matrix rather than on one l4k4 benchmark.

Each new LAMMPS optimization version must be treated as a candidate backend and
run over the whole matrix above before promotion. The decision record must keep:

- source commit or source file hashes for the candidate;
- binary path and SHA-256;
- full matrix output directory;
- median and mean ratio against the same-case main binary;
- worst-case ratio, number of cases above `1.20x`, and the best backend per
  case.

This avoids overfitting the implementation to one model size. If a candidate
improves only a subset such as large `l4k4` models or only
`mu-body-linear-combo`, it can be retained only behind a model-load dispatch
path whose predicate is based on stable model metadata. If no single exact
backend covers all representative cases, the dispatch table should be derived
from the full matrix and revalidated whenever the backend code changes.

### Promoted LAMMPS Product-Array Backend 2026-06-17

The retained single-gate LAMMPS backend is the product-array implementation in
code commit:

```text
047490880ddcc2f5eb90c48007edbc28dceeb97d
```

Server/source relationship:

```text
branch: <historical-branch>
LAMMPS source tree: /work/phy-weigw/apps/lammps-10Dec2025/src
candidate binary: <historical-path>
binary SHA-256: 3804c2ff4dc2f8d2dea9dc98ef518f9f62cd4d65a4d02895880926f013a94340
pair_sus2_mtp.cpp SHA-256: 99607417815d5ff7581a94b00a5fd4d0ea22a3ba8f9543724e8ab49f89a7b796
pair_sus2_mtp.h SHA-256: 7bcc504efe60e2eb96c4392b96bca1efd4853b47cd7af02bdd60f8d2084dea0f
```

Parity verification:

```text
job: 3802892
directory: <historical-path>
result: 12 cases PASS, max_abs_energy=0, max_abs_force=0, rms_force=0
```

Full 40-rank large-cell speed verification:

```text
job: 3802925
directory: <historical-path>
main binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
main SHA-256: 46a2096aebdcb8d05030c3f7164227c402986334f942c558042d569888e9344a
run_steps: 800
replicate: 6 6 6
pruning: q_total
gate site mode: single gate
```

Median LAMMPS loop-time ratios against the current main/developer LAMMPS
binary:

| l/k | Gate weight mode | Main median | Product-array median | Ratio |
| --- | --- | ---: | ---: | ---: |
| l2k2 | `mu-body-linear-combo` | `6.643130 s` | `7.890470 s` | `1.1878x` |
| l2k2 | `mu-scalar-full` | `6.527200 s` | `7.654530 s` | `1.1727x` |
| l2k3 | `mu-body-linear-combo` | `11.007000 s` | `12.487000 s` | `1.1345x` |
| l2k3 | `mu-scalar-full` | `11.146700 s` | `12.763800 s` | `1.1451x` |
| l3k3 | `mu-body-linear-combo` | `16.323700 s` | `19.458500 s` | `1.1920x` |
| l3k3 | `mu-scalar-full` | `16.319300 s` | `19.572900 s` | `1.1994x` |
| l4k4 | `mu-body-linear-combo` | `50.756900 s` | `53.271400 s` | `1.0495x` |
| l4k4 | `mu-scalar-full` | `50.667400 s` | `60.127400 s` | `1.1867x` |

Decision: promote this product-array backend as the current exact LAMMPS
implementation for single-gate `q_total` models with \(l \le 4\), \(k \le 4\).
All measured representative cases are at or below the `1.20x` target. The
tightest case is `l3k3` `mu-scalar-full` at `1.1994x`, so future changes should
rerun this matrix before promotion.

Rejected exact candidates during this optimization round:

- `stride2` direct body-adjoint specialization preserved parity but slowed the
  l2k2 hot case (`1.2823x` vs main in
  `<historical-path>`);
  it was reverted and is not present in the promoted source.
- `SUS2_LAMMPS_GATE_MU_INDEXED_BASIC=1` preserved the mathematical path but
  increased l2k2 `mu-body-linear-combo` median to `11.258000 s`
  (`1.6477x` vs main) in
  `<historical-path>`;
  it remains off and is not promoted.

## Verification Checklist

Local serial checks:

```bash
bash dev_test/sh_two_layer_gate_mu_body_order_check.sh
bash dev_test/sh_two_layer_gate_double_mode_check.sh
bash dev_test/sh_two_layer_gate_init_check.sh
bash dev_test/sh_two_layer_gate_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_forward_check.sh
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
bash dev_test/sh_two_layer_gate_train_weight_check.sh
bash dev_test/sh_two_layer_gate_lmp_table_check.sh
bash dev_test/sh_plain_to_gate_upgrade_check.sh
LAMMPS_BIN=<historical-path> \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
GATE_SITE_MODE=double \
LAMMPS_BIN=<historical-path> \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
```

Server checks:

- MPI `mlp-sus2` build in the independent work directory.
- `body-order < k-max+1` fails.
- Explicit `--two-layer-gate-body-order` fails.
- `k-max=4, body-order=6` gate scalars contain only body orders `2,3,4,5`.
- Single nonzero gate weights in body-order buckets `2,3,4,5` affect only their
  matching k channels.
- Zero gate weights match ordinary SH `calc-efs` energy and force.
- Double-site gate zero weights also match ordinary SH `calc-efs`, and nonzero
  double-site gate weights pass the loss-gradient check.
- CPU LAMMPS neighbor-site and double-site gate models match `mlp-sus2
  calc-efs` on the additive shared-radial smoke case. The 1-rank and 2-rank
  LAMMPS runs are bitwise identical in the checked energy/force values.
- Smoke run in `/work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/` with the new
  independent binary.

## 1t88c 96-Rank LAMMPS Status - 2026-06-18

The earlier 40-rank product-array result above is historical. Current LAMMPS
speed conclusions for this branch should use a single `1t88c` allocation with
96 ranks and `span[ptile=96]`.

Current accepted branch LAMMPS binaries:

```text
AVX2 stable trial:
<historical-path>

1t88c/Sapphire-Rapids trial:
<historical-path>
```

Source and build record:

```text
GitHub branch: <historical-branch>
LAMMPS code commit: 337b3a6a1100b7f549398d8265c8d8767b83b8ae
build job: 3806942
build directory: <historical-path>
LAMMPS source tree: /work/phy-weigw/apps/lammps-10Dec2025/src
server source mirror: <historical-path>
pair_sus2_mtp.cpp SHA-256: d51d12b20dd160472d8b7a3097c98241362527614c81c3466139c427c025bf0c
pair_sus2_mtp.h SHA-256: 43493ad85dc9e4cccadf327294eb3c15766305cf24ba4bd1510c2566015264df
AVX2 binary SHA-256: 2fbde6c67935795157bf8566fa21dd2f159a5c06944ed088fdd5825b76c65703
SPR-low binary SHA-256: 608884071b9053e2739ea5f174c742dffc0d3980a67fa56b5cf504b9054fce88
SPR-low flags: -xSAPPHIRERAPIDS -qopt-zmm-usage=low -O3 -fp-model fast=2 -no-prec-div -qoverride-limits -fimf-use-svml=true -qopt-streaming-stores=auto
```

Current full l/k matrix:

```text
job: 3806946
summary: <historical-path>
main binary: <historical-path>
main SHA-256: 8719bec551be72abcf9c8c6e93bad55ab5b78327f505052cbf9683c1f22deb61
run_steps: 800
replicate: 6 6 6
pruning: q_total
gate site mode: single gate
```

Median LAMMPS loop-time ratios against the current main/developer LAMMPS
reference:

| l/k | Gate weight mode | Main median | Guarded SPR median | Ratio |
| --- | --- | ---: | ---: | ---: |
| l2k2 | `mu-body-linear-combo` | `2.382050 s` | `2.412690 s` | `1.0129x` |
| l2k2 | `mu-scalar-full` | `2.544760 s` | `2.733580 s` | `1.0742x` |
| l2k3 | `mu-body-linear-combo` | `3.483920 s` | `3.566550 s` | `1.0237x` |
| l2k3 | `mu-scalar-full` | `3.613250 s` | `3.990340 s` | `1.1044x` |
| l2k4 | `mu-body-linear-combo` | `9.736700 s` | `10.426700 s` | `1.0709x` |
| l2k4 | `mu-scalar-full` | `9.339690 s` | `11.098900 s` | `1.1884x` |
| l3k3 | `mu-body-linear-combo` | `4.264790 s` | `4.758500 s` | `1.1158x` |
| l3k3 | `mu-scalar-full` | `4.221570 s` | `4.939370 s` | `1.1700x` |
| l3k4 | `mu-body-linear-combo` | `7.235610 s` | `8.324090 s` | `1.1504x` |
| l3k4 | `mu-scalar-full` | `7.218510 s` | `9.008680 s` | `1.2480x` |
| l4k3 | `mu-body-linear-combo` | `6.473510 s` | `7.700900 s` | `1.1896x` |
| l4k3 | `mu-scalar-full` | `6.659330 s` | `7.604340 s` | `1.1419x` |
| l4k4 | `mu-body-linear-combo` | `10.547300 s` | `12.268100 s` | `1.1632x` |
| l4k4 | `mu-scalar-full` | `10.618700 s` | `13.354700 s` | `1.2577x` |

Decision:

- `mu-body-linear-combo` is within the `1.20x` target for all tested q-total
  models with `l<=4, k<=4`.
- `mu-scalar-full` is within target for l2k2, l2k3, l2k4, l3k3, and l4k3, but
  remains above target for l3k4 and l4k4. The current worst valid 1t88c result
  is l4k4 `mu-scalar-full` at `1.2577x`.
- Therefore the branch has not globally reached the `1.20x` target for both
  gate weight modes. Earlier "l4k4 within 1.2x" notes should be treated as
  combo-mode, 40-rank, candidate-specific, or invalid-baseline evidence unless
  the exact binary family, ISA, queue, rank layout, model, and gate mode match.

Rejected exact candidates from the 2026-06-18 1t88c round:

- `lmp.ml-sus2_mu_body_gate_spr_low.stride20_groupadj_noipo_trial`
  (build job 3806948, binary SHA-256
  `0f2cfeb611ebd360687d334c87e3b78e2c81798306ab9a655a91301ac51a934b`) was
  slower than guarded_envbatch in direct A/B testing:
  `<historical-path>`.
  It also introduced a max force difference of about `6.9e-15` in AVX2 run-0
  parity, so it is not promoted.
- Runtime `SUS2_LAMMPS_GATE_COMPACT_MOMENT_CACHE=1` slowed both l3k4 and l4k4
  `mu-scalar-full`.
- Runtime `SUS2_LAMMPS_GATE_REQUIRED_BASIC=1` plus
  `SUS2_LAMMPS_GATE_REQUIRED_DOT=1` was mixed: l3k4 `mu-scalar-full` improved
  slightly, but l4k4 slowed. Keep the default heuristics.

## Formal App Installs - 2026-06-18

The last exact optimization attempts after the accepted guarded-envbatch source
were rejected. The branch was restored to the accepted source hashes below, and
the formal CPU LAMMPS binaries are installed under `/work/phy-weigw/app/`.

```text
GitHub branch: <historical-branch>
branch HEAD at build time: 74b8e61aba78e24198172f0ed5872eb52940182f
LAMMPS source tree used for app builds: /work/phy-weigw/apps/lammps-10Dec2025/src
server LAMMPS mirror: <historical-path>
pair_sus2_mtp.cpp SHA-256: d51d12b20dd160472d8b7a3097c98241362527614c81c3466139c427c025bf0c
pair_sus2_mtp.h SHA-256: 43493ad85dc9e4cccadf327294eb3c15766305cf24ba4bd1510c2566015264df
```

Installed binaries:

| Target | Queue | Binary | SHA-256 | Build job | Status |
| --- | --- | --- | --- | ---: | --- |
| Generic AVX2 | `33` | `/work/phy-weigw/app/lmp.sus2-sh-mu-body-order-gate_33_avx2` | `77272a531821825d84453d20a1c228116ac191ba2d72ad7a85b8e06bad2b4691` | 3807120 | DONE |
| High performance xHost | `1t88c` | `/work/phy-weigw/app/lmp.sus2-sh-mu-body-order-gate_1t88c_xhost_high` | `ecdd45153e958624080bde652233934c6aef98f0433df9649a7b9100411fb4e7` | 3807121 | DONE after linked binary copy; final `make` exited only on duplicate `liblammps.a` symlink |
| High performance xHost | `1t75c` | `/work/phy-weigw/app/lmp.sus2-sh-mu-body-order-gate_1t75c_xhost_high` | pending | 3807125 | queued on `1t75c` host slot limit |
| High performance xHost | `1t81c` | `/work/phy-weigw/app/lmp.sus2-sh-mu-body-order-gate_1t81c_xhost_high` | pending | 3807127 | queued with dependency `done(3807125)` |

Build records are stored in:

```text
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/
```

The `33` build uses:

```text
-xCORE-AVX2 -O3 -fp-model fast=2 -no-prec-div -qoverride-limits -fimf-use-svml=true -qopt-streaming-stores=never
```

The target-CPU builds use:

```text
-xHost -qopt-zmm-usage=high -O3 -fp-model fast=2 -no-prec-div -qoverride-limits -fimf-use-svml=true -qopt-streaming-stores=auto
```

Rejected final attempts are archived under:

```text
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/rejected-test-summaries/
```

The rejected final attempts were:

- `directfulladj`: parity-correct but slower than the accepted source for the
  tested full-mode cases.
- `profile_split`: diagnostic only; it showed the full scalar contraction is not
  the main bottleneck for l4k4 `mu-scalar-full`.
- `skipzero_backprop`: parity-correct but slower than the accepted source for
  l3k4 and l4k4 `mu-scalar-full`.
- earlier `unroll16_20` and `full_envbatch` candidates were also archived as
  rejected summaries.

## Kokkos GPU LAMMPS v21 - 2026-06-18

This round optimizes the Kokkos GPU LAMMPS interface for the experimental
`<historical-branch>` branch. The accepted Kokkos binary is installed at:

```text
/work/phy-weigw/app/lmp.sus2-sh-mu-body-order-gate_kokkos_sm80_v21_split_gate_force_kernel
```

Source and binary record:

```text
branch: <historical-branch>
source commit: this documentation commit; final pushed hash is recorded in the server source map
server source mirror: <historical-path>
server build tree: <historical-path>
build job: 3812933
Kokkos binary SHA-256: 9ac60065bb2590e18647633684634d2d1c2751eae1afbd709a0b8bf1f2720dc3
pair_sus2_mtp_kokkos.cpp SHA-256: 72a31557e1903e61170ce29ee3b4e3eb1dcb1d5a1d7d5e5d53abfc1ff12cbeb0
pair_sus2_mtp_kokkos.h SHA-256: c06e42d35a53afc1247fa8ce01917b18735f53a1fd8414829cdb6ce415af3a1e
```

The v21 optimization keeps the exact math and does not change atomic or
centroid virial formulas. The direct gate chain fast path remains disabled when
`eflag_atom`, `vflag_atom`, or `cvflag_atom` is set, so
`compute centroid/stress/atom` continues to use the `v_tally_xyz` pair tally
path. The performance change is limited to splitting the direct-chain team force
kernel into:

- a combo-mode local-adjoint kernel that keeps the small signal adjoint
  accumulation array and reduces atomics;
- a normal full-mode kernel that does not instantiate the combo-only local
  adjoint array, reducing register/local-memory pressure.

Centroid stress oracle:

```text
job: 3812936
directory: <historical-path>
input check: compute ccent all centroid/stress/atom NULL pair
combo: dE=0, dP=0, force_max=9.1593399531575415e-15, centroid_rel=8.2133422715458233e-15 PASS
full:  dE=0, dP=0, force_max=2.1316282072803006e-14, centroid_rel=1.3829233550611508e-14 PASS
```

Kokkos GPU speed matrix:

```text
job: 3812939
node: c04u01g
main reference binary: /work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
mu binary: <historical-path>
cell: B-system first training structure, replicate 6 6 6
run steps: 800
reps: 3
models: q-total l2k2, l2k3, l2k4, l3k3, l3k4, l4k3, l4k4
```

Median loop-time ratios against the current main Kokkos gate reference:

| l/k | Gate weight mode | Main median | v21 median | Ratio | Status |
| --- | --- | ---: | ---: | ---: | --- |
| l2k2 | `mu-body-linear-combo` | `2.49414 s` | `2.63148 s` | `1.0551x` | PASS |
| l2k2 | `mu-scalar-full` | `2.49414 s` | `2.50149 s` | `1.0029x` | PASS |
| l2k3 | `mu-body-linear-combo` | `4.64907 s` | `5.22997 s` | `1.1249x` | PASS |
| l2k3 | `mu-scalar-full` | `4.64907 s` | `5.07160 s` | `1.0909x` | PASS |
| l2k4 | `mu-body-linear-combo` | `30.2593 s` | `36.2827 s` | `1.1991x` | PASS |
| l2k4 | `mu-scalar-full` | `30.2593 s` | `36.2037 s` | `1.1964x` | PASS |
| l3k3 | `mu-body-linear-combo` | `7.62669 s` | `8.67004 s` | `1.1368x` | PASS |
| l3k3 | `mu-scalar-full` | `7.62669 s` | `8.31637 s` | `1.0904x` | PASS |
| l3k4 | `mu-body-linear-combo` | `17.7583 s` | `20.9923 s` | `1.1821x` | PASS |
| l3k4 | `mu-scalar-full` | `17.7583 s` | `20.8855 s` | `1.1761x` | PASS |
| l4k3 | `mu-body-linear-combo` | `13.8074 s` | `16.2601 s` | `1.1776x` | PASS |
| l4k3 | `mu-scalar-full` | `13.8074 s` | `15.7815 s` | `1.1430x` | PASS |
| l4k4 | `mu-body-linear-combo` | `27.3667 s` | `31.7116 s` | `1.1588x` | PASS |
| l4k4 | `mu-scalar-full` | `27.3667 s` | `32.0424 s` | `1.1709x` | PASS |

Archived reports:

```text
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/kokkos-v21-20260618/kokkos_matrix_report.txt
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/kokkos-v21-20260618/kokkos_matrix_summary.tsv
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/kokkos-v21-20260618/centroid_oracle_report.txt
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/kokkos-v21-20260618/source.sha256
/work/phy-weigw/app/sus2-sh-mu-body-order-gate-build-info/kokkos-v21-20260618/binary.sha256
```

Decision: v21 is the accepted Kokkos GPU LAMMPS implementation for this branch.
For the tested q-total models with `l<=4, k<=4`, both `mu-body-linear-combo`
and `mu-scalar-full` meet the `main x1.20` speed target on the fixed GPU
benchmark while preserving centroid stress correctness.
