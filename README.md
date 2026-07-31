# SUS2-SH

SUS2-SH is the spherical-harmonic line of SUS2. It keeps the original C++
SUS2 training flow, BFGS optimizer, CFG/MTP file conventions, and model I/O,
while replacing moment-tensor angular channels with real spherical harmonics.

The current formal line is **SUS2-SH Release 3**, branch `release3`, released
on 2026-07-23.

The basic SH channel is

```text
M_i,mu,m = sum_j R_ZiZj,mu(r_ij) Y_lm(rhat_ij)
```

where `mu = (l,k)`. The saved `.mtp` file stores the SH basic moments, the
real Clebsch-Gordan scalar product graph, and the radial/model parameters used
by the CPU trainer, LAMMPS-SH, Kokkos-SH, and GPUMD-SUS2-SH backends.

## Repository Role

| Repository | Role |
| --- | --- |
| `SUS2-SH` | CPU/reference SUS2-SH model generation, training, model I/O, and SH LAMMPS source. |
| `SUS2-SH-GPU` | CUDA training/evaluation line for the same SUS2-SH model format. |
| `GPUMD-SUS2-SH` | GPUMD runtime backend for trained SUS2-SH models. |
| `PySUS2SH` | Python/ASE/phonon workflow package for SUS2-SH models. |

Implementation notes are in `docs/sus2-sh-development.md`. The inherited
SUS2-MLIP/MLIP documentation remains useful for CFG and MTP file formats.

## Build

```bash
./configure
make mlp
make libinterface
```

The training executable is `bin/mlp-sus2`.

## Basic SH Training CLI

There are two equivalent ways to start a normal single-layer SH training run.

Create an untrained model first:

```bash
bin/mlp-sus2 init-sh untrained.mtp \
  --species-count=4 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss

bin/mlp-sus2 train untrained.mtp train.cfg \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0.001 \
  --do-samp \
  --do-lin \
  --do-lin-rescale \
  --curr-pot-name=current.mtp \
  --trained-pot-name=p.mtp
```

Or create/continue the model directly from `train`:

```bash
bin/mlp-sus2 train train.cfg --init-sh \
  --inline-sh-model=untrained.mtp \
  --species-count=4 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0.001 \
  --do-samp \
  --do-lin \
  --do-lin-rescale \
  --curr-pot-name=current.mtp \
  --trained-pot-name=p.mtp
```

Main SH topology options:

| Option | Meaning |
| --- | --- |
| `--species-count=<int>` | Number of atomic species in CFG type order. |
| `--l-max=<int>` | Maximum SH angular index `l`; generator supports `0..6`, optimized runtimes are mainly validated for `l <= 4`. |
| `--k-max=<int>` | Number of radial channels per `l`; `k` is indexed from `0` to `k_max-1`. |
| `--scaling-map=K\|LK` | Select the `scal/s` map. `K` shares one pairwise block across all `l` at fixed `k`; `LK` keeps one block per `(k,l)`. Default=`LK`. |
| `--body-order=<int>` | Maximum scalar body order, supported range `2..6`. |
| `--body-l-max=<list>` | Body-specific angular cutoffs. Use four values for body orders `2..5`, or five values for `2..6`; example `3,3,2,2`. |
| `--cutoff=<double>` / `--max-dist=<double>` | Main SH radial cutoff. `--cutoff` is the preferred spelling. |
| `--min-dist=<double>` | Initial radial scaling reference. It does not define a hard lower interaction cutoff. |
| `--radial-basis-size=<int>` | Number of primitive radial basis functions used to form each `R_mu`. |
| `--radial-basis-type=<name>` | Supported model-generation types: `RBChebyshev_sss`, `RBChebyshev_sss_rational`, `RBLaguerre_log1p`, `RBJacobi_sss`; LAMMPS table variants use the same model fields with `_lmp` names. |
| `--sh-factor-pruning=same-k|legacy|q-total` | Outer scalar graph pruning rule, default `q-total`. `q-total` keeps the complete total-ordered `q=(l,k)` tuples; `legacy` additionally requires separate monotonic `l` and `k`; `same-k` keeps complete angular coupling but requires every factor in a scalar to use one common radial channel `k`. Thus `same-k` is a subgraph of `legacy`, which is a subgraph of `q-total`. |
| `--write-sh-scalar-info` | Explicitly save scalar metadata; this is written automatically for gate models. |
| `--potential-name=<string>` | Name written into the `.mtp` file. |
| `--scaling=<double>` | Initial global scaling written to the model, default `0.01`. |

`scaling_map=K` changes only the pair-dependent radial-coordinate parameters:

\[
\mu=k(l_{\max}+1)+l,\qquad
b_K(\mu)=k,\qquad b_{LK}(\mu)=\mu.
\]

Thus `scal[pair,b(mu)]` and `s[pair,b(mu)]` are shared across `l` only in
`K` mode. The radial contraction coefficients \(c_{\mu,\xi}\), SH moments,
product graph, gate channels, and optional outer/gate K-rank contractions all
retain the complete \(\mu=(k,l)\) layout. Omitting `--scaling-map` preserves
the release3 `LK` model and runtime path.

Common training options:

| Option | Meaning |
| --- | --- |
| `--energy-weight=<double>` | Energy contribution weight, default `1`. |
| `--force-weight=<double>` | Force contribution weight, default `0.01`. |
| `--stress-weight=<double>` | Stress contribution weight, default `0.001`. |
| `--force-loss=l2|log-cosh` | Nonlinear force loss. Default is componentwise L2/MSE; `log-cosh` is more robust to large force outliers. |
| `--force-log-cosh-scale=<double>` | Scale for `log-cosh` force loss in eV/A, default `2`. |
| `--valid-cfgs=<path>` | Optional validation CFG file. |
| `--optimizer=bfgs|lbfgs` | Nonlinear optimizer. The default remains dense BFGS; L-BFGS is an independent opt-in path. |
| `--max-iter=<int>` | Maximum accepted optimizer iterations, default `1000`. |
| `--curr-pot-name=<path>` | Atomically save the latest evaluated and accepted model during training. |
| `--best-pot-name=<path>` | Atomically save the best fully evaluated boundary/accepted model. By default a `.best.mtp` path is derived from `--curr-pot-name`; an existing best file is copied to a timestamped archive before a new run writes it. It must not alias the current or final-model path. |
| `--trained-pot-name=<path>` | Atomically save the final trained model, default `Trained.mtp_`. |
| `--optimizer-conv-tol=<double>` | Generic alias for `--bfgs-conv-tol`. |
| `--optimizer-trace-file=<path>` | Generic alias for `--bfgs-trace-file`. L-BFGS traces include curvature/history diagnostics. |
| `--optimizer-line-search=wolfe|armijo-value` | Generic line-search selector. L-BFGS supports value-only Armijo; BFGS retains its historical Wolfe default. |
| `--radial-parameterization=physical|weight-norm` | Optimizer coordinates for outer and independent gate1 radial coefficient rows. Default `physical` preserves the release3 path; `weight-norm` separates each row's log norm from its direction without changing the saved model format or inference path. |
| `--bfgs-coordinate-normalization=none|efs-gn` | Optional fixed diagonal BFGS coordinates for `species_coeffs`, scalar heads, and gate additive/mixing weights. `efs-gn` estimates the E/F/S generalized Gauss-Newton diagonal once per training stage; the default `none` preserves the previous path. |
| `--bfgs-coordinate-probes=<int>` | Deterministic Hutchinson probe count used by `efs-gn`, default `4`. |
| `--bfgs-coordinate-floor=<double>` | Per-block relative curvature floor for `efs-gn`, default `1e-6`. |
| `--bfgs-conv-tol=<double>` | Stop when improvement over 50 BFGS iterations is below this factor, default `1e-3`. |
| `--bfgs-trace-file=<path>` | Optional CSV trace of accepted BFGS steps. |
| `--lbfgs-history=<int>` | Accepted curvature-pair history size, default `20`. |
| `--lbfgs-curvature-tol=<double>` | Relative damped-curvature acceptance threshold, default `1e-10`. |
| `--lbfgs-damping=<double>` | Diagonal Powell damping threshold in `[0,1)`, default `0.2`. |
| `--lbfgs-armijo-c1=<double>` | Total-loss Armijo sufficient-decrease coefficient, default `1e-4`. |
| `--lbfgs-initial-step=<double>` | Initial trial step after a reset, default `1`. |
| `--lbfgs-step-growth=<double>` | Growth factor for the next trial after an unbacktracked accepted step, default `1.5`. |
| `--radial-smooth=<double>` | H1 smoothness penalty on `dR/dr`, default `1e-6`; set `0` to disable. |
| `--radial-smooth-grid=<int>` | Midpoint grid for `--radial-smooth`, default `128`. |
| `--atomic-energies=<e0,e1,...>` | Enforce isolated element energy constraints `shift_t + species_t = e_t`. |
| `--atomic-energy-weight=<double>` | Penalty weight for `--atomic-energies`, default `1e8`. |
| `--weighting=<vibrations|molecules|structures>` | Configuration weighting mode, default `vibrations`. |
| `--init-params=random|same` | Parameter initialization when a model is not already fitted. Default `random`. |
| `--do-lin` | Enable linear pre-fitting/linear solves inside nonlinear training. |
| `--do-lin-rescale` | Run the subset rescale pass before the full-data `TrainLinear` solve used by `--do-lin`. |
| `--do-lin-steps=<int>` | Number of accepted BFGS steps for which `--do-lin` stays active, default `1000`. |
| `--do-lin-freq=<int>` | Run `TrainLinear` every N accepted BFGS steps while `--do-lin` is active, default `50`. |
| implicit initial linear warmup | Untrained or incomplete initial models run one rescale plus full-data `TrainLinear` warmup before BFGS even without `--do-lin`; complete trained models do not unless requested by explicit options. |
| `--fine-tune` | Continue from a complete trained model, freeze scaling coefficients, and run one initial rescale plus linear solve before BFGS. |
| `--do-samp` | Disable random sampling used in pre-training. |
| `--skip-preinit` | Skip the 75 preliminary iterations used when parameters are not given. |
| `--shift` | Disable the trainer's internal shift correction. |
| `--update-mindist` | Update model `min_dist` from the training-set minimum distance. |

Best-checkpoint writes remain atomic and accepted-point-only, but their
per-improvement messages are quiet by default. Set
`SUS2_SH_CHECKPOINT_UPDATE_TRACE=1` only when those diagnostic messages are
needed.

## Gate Model Training CLI

The release3 production gate is a linear additive-node update. It does not use
`tanh` or any other activation. Let `B_i,q^gate` be the first-layer invariant
scalars. The `mu-scalar-full` mode forms

```text
h_i,mu = sum_q W_mu,q B_i,q^gate
```

while `mu-body-linear-combo` first pools the scalars by body order and then
mixes those body-order channels for each `mu`. The neighbor node coefficient in
the outer edge is updated linearly,

```text
u_i,j,mu = s_center[Z_i,mu]
             (s_outer[Z_j,mu] + a[Z_j,mu] h_j,mu),

M_i,mu,m^main = sum_j R_i,j,mu(r_ij) Y_lm(rhat_ij) u_i,j,mu.
```

Thus zero gate weights recover the ungated node coefficient exactly. The
release path is neighbor-site only; the older center/double-site and
`--two-layer-gate-tanh-amplitude` controls are obsolete. Gate models use their
own trained radial contraction coefficients for `B_i,q^gate` by default.

Initialize and train a gate model directly:

```bash
bin/mlp-sus2 train train.cfg --init-sh \
  --inline-sh-model=gate_untrained.mtp \
  --species-count=4 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBLaguerre_log1p \
  --two-layer-gate \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0.001 \
  --do-samp \
  --do-lin \
  --do-lin-rescale \
  --curr-pot-name=current.mtp \
  --trained-pot-name=p.mtp
```

Continue a trained or partially trained plain SH model as a gate model:

```bash
bin/mlp-sus2 train plain_sh.mtp train.cfg \
  --two-layer-gate \
  --curr-pot-name=current_gate.mtp \
  --trained-pot-name=gate.mtp
```

Gate-specific options:

| Option | Meaning |
| --- | --- |
| `--two-layer-gate` | Add two-layer gate metadata to a new SH model, or upgrade a plain SH model before continuing training. |
| `--two-layer-gate-mode=mu-body-linear-combo|mu-scalar-full` | Gate projection from first-layer scalars to the `mu` node channels. `mu-scalar-full` keeps an independent coefficient for every `(mu,q)` pair. |
| `--two-layer-gate-body-order=<int>` | Gate1 scalar body-order cutoff in `[2, --body-order]`; default is the outer body order. |
| `--two-layer-gate-factor-pruning=same-k|legacy|q-total` | Gate1 scalar subgraph. It defaults to the outer mode and cannot be richer than `--sh-factor-pruning`. |
| `--two-layer-gate-site-mode=neighbor` | Gate acts on neighbor endpoint `j`; `neighbor` is the only release mode. |
| `--two-layer-gate-shared-radial` | Compatibility switch for the default behavior: gate models use an independent first-layer radial contraction table. |
| `--two-layer-gate-readout` | Add a linear energy head on the first-layer gate scalars. Release3 standalone, CPU LAMMPS, and Kokkos LAMMPS support the compatible single-layer neighbor-site form. |
| `--outer-k-rank=<R>` | Contract the outer `k` channels to rank `R` before building the outer scalar graph. `0`/omitted keeps the full graph; `R=1` uses the direct-radial form. |
| `--two-layer-gate-k-rank=<R_g>` | Independently contract the first-layer gate `k` channels. It currently requires an outer-rank model. |
| `--second-layer-gate-factor-pruning=same-k|legacy|q-total` | Gate2 scalar subgraph for `--multi-layer-gate`. It defaults to gate1 and cannot be richer than the outer mode. |
| `--multi-layer-gate` | Add release3 gate2 on the gate1 endpoint coefficients. This topology is supported by the standalone runtime, but not by the release3 LAMMPS pair styles. |
| `--second-layer-gate-body-order=<int>` | Gate2 scalar body-order cutoff in `[2, --body-order]`; default is the outer body order. |
| `--inline-sh-model=<path>` | In `train train.cfg --init-sh`, create this file if missing or continue from it if it already exists. |
| `--two-layer-residual` | Experimental residual two-layer mode `E=E0+E1`; not the current production gate path. |
| `--two-layer-residual-staged` | Stage residual training into A/B/C phases; requires a residual model and positive stage step counts. |
| `--stage-a-steps=<int>` | BFGS steps for residual stage A. |
| `--stage-b-steps=<int>` | BFGS steps for residual stage B. |
| `--stage-c-steps=<int>` | BFGS steps for residual stage C. |

For current production work, use `--two-layer-gate`; add
`--multi-layer-gate` only when gate2 is required. The saved scalar index lists
fully define each gate subgraph, so inference and LAMMPS readers do not need a
runtime pruning-mode branch. Rank-model mathematics, constraints, and runtime
compatibility are documented in
[`docs/release3-k-rank.md`](docs/release3-k-rank.md).

Gate readout is defined only for the single-layer neighbor-site gate topology.
It is mutually exclusive with outer/gate `K` rank, two-layer residual,
edge-L1, and multi-layer-gate modes.

## Effective Pair Energy CLI

`calc-eij` writes the effective-pair (EP) projection of the SUS2-SH scalar
energy for each unordered neighbor image in a CFG file:

```bash
bin/mlp-sus2 calc-eij pot.mtp in.cfg out.eij
```

Each output row has four whitespace-separated columns:

```text
type_i type_j distance EP_energy
```

The two type columns are sorted so that `type_i <= type_j`. The distance is the
length of that specific neighbor image. If a small periodic cell contains more
than one image of the same atom pair within the cutoff, those images are written
as separate rows rather than merged by atom index.

The EP energy omits one-body element terms such as `shift_coeffs` and
`species_coeffs`. It is the pair projection of the SH scalar linear head:

```text
E_EP(I,j) =
  u_ZI sum_q v_q lambda_q
    sum_m C_q(m) A_{I<-j,mu_1,m_1}
      prod_{nu=2..n_q} A_{I,mu_nu,m_nu}
```

The unordered row for an `I,J` image sums the two directed projections
`E_EP(I,J) + E_EP(J,I)` when both directions are present. The command supports
plain SH and the release neighbor-site additive gate, and uses the same node
update as ordinary `calc-efs`.

`calc-eij` requires `sh_scalar_info` metadata in the `.mtp` file. New gate
models write this metadata automatically; for plain SH models, initialize or
rewrite the model with `--write-sh-scalar-info` before using this command. This
diagnostic reports the SUS2-SH scalar projection and does not add ZBL pair
terms.

## ZBL CLI

ZBL is a fixed short-range repulsive term stored inside the `.mtp` file. Once a
model contains `zbl_enabled = true`, SUS2-SH prediction/training, LAMMPS-SH,
Kokkos-SH, and GPUMD-SUS2-SH read the model metadata and add the ZBL term
automatically; no extra runtime flag is required in LAMMPS or GPUMD.

Add ZBL when creating a model:

```bash
bin/mlp-sus2 init-sh untrained_zbl.mtp \
  --species-count=4 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --zbl-elements=H,O,Cl,K \
  --zbl-inner=0.7 \
  --zbl-outer=1.4
```

Add ZBL to an existing trained SH or gate model and continue training:

```bash
bin/mlp-sus2 train trained_without_zbl.mtp train.cfg \
  --zbl-elements=H,O,Cl,K \
  --zbl-inner=0.7 \
  --zbl-outer=1.4 \
  --curr-pot-name=current_zbl.mtp \
  --trained-pot-name=p_zbl.mtp
```

Use NEP-style typewise pair cutoffs:

```bash
bin/mlp-sus2 train trained_without_zbl.mtp train.cfg \
  --zbl-elements=H,O,Cl,K \
  --zbl-outer=1.4 \
  --zbl-typewise-cutoff-factor=0.7 \
  --curr-pot-name=current_zbl.mtp \
  --trained-pot-name=p_zbl.mtp
```

ZBL options:

| Option | Meaning |
| --- | --- |
| `--zbl-elements=<e0,e1,...>` | Enables ZBL and gives one element symbol or atomic number per model species, in CFG type order. Required for any ZBL use. |
| `--zbl-inner=<double>` | Inner radius for fixed-cutoff ZBL. Default `0.7`. Inside this radius the unswitched ZBL is used. |
| `--zbl-outer=<double>` | Global outer cutoff. Default `1.4`. Fixed-cutoff mode uses this value for every pair. |
| `--zbl-typewise-cutoff-factor=<double>` | Enable typewise cutoffs. The inner cutoff becomes `0`; each pair uses `outer_ij = min(zbl_outer, factor*(Rcov_i + Rcov_j))`. The code uses the maximum pair `outer_ij` as the ZBL neighbor cutoff, so no pair term is missed. Default factor is `0.7` when this option is used. |

During training, when ZBL is enabled, the trainer precomputes the ZBL residual
on the training set, subtracts it from the target E/F/S during optimization,
and restores the full reference for final reporting. This keeps the fitted SH
part focused on the non-ZBL residual while the saved model still carries the
complete SH+ZBL potential.

## Dataset Format

SUS2-SH uses the same CFG format as SUS2-MLIP/MLIP:

```text
BEGIN_CFG
 Size
     192
 Supercell
  16.7849720000    0.0000000000    0.0000040000
   0.0000000000   17.3600060000   -0.0000010000
   0.0000030000   -0.0000000000   12.4404560000
AtomData:  id type cartes_x cartes_y cartes_z fx fy fz
      1      0     2.498644     3.809912     3.110113    -0.009924    -0.001443     0.000004
Energy
        -165049.45992
PlusStress:  xx yy zz yz xz xy
        1.1012082011734257 -0.7401052299730059 0.2790165512621857 0.0012779654132595323 -0.0002538647260112984 -2.4887947553777763e-10
END_CFG
```

Conversion scripts for CFG and ASE-readable formats are in `python_tool/`.

## External Runtime Interfaces

### LAMMPS-SH

The maintained SH LAMMPS interface is in `interfaces/lammps/`. It uses the
same package and pair-style names as the SUS2-MLIP interface:

```text
ML-SUS2
pair_style sus2mtp
pair_style sus2mtp/kk
```

Install into a clean LAMMPS tree:

```bash
cd interfaces/lammps
scripts/install_into_lammps.sh /path/to/lammps
```

Table radial types such as `RBChebyshev_sss_lmp`,
`RBChebyshev_sss_rational_lmp`, `RBLaguerre_log1p_lmp`,
`RBLaguerre_log1p_pos_lmp`, and `RBJacobi_sss_lmp` enable list-based radial
evaluation in LAMMPS. The `tabstep` keyword controls table spacing and defaults
to `1.0e-4 A`. For radial bases supported by the selected CPU or Kokkos
backend, the optional `radial_mode auto|table|direct` keyword can run a
table-capable `_lmp` model through direct evaluation without editing the model
file; `auto` remains the backward-compatible default.

The release3 CPU and Kokkos pair styles accept the historical full-`K` models,
optional outer/gate `K`-rank models, and compatible single-layer gate-readout
models documented in
[`docs/release3-k-rank.md`](docs/release3-k-rank.md). Readout remains
incompatible with rank, residual, edge-L1, and multi-layer-gate topologies.
Both pair styles accept `scaling_map=K` as well as the default `LK` map.  The
compact `K` map shares only `scal/s` across angular channels; it does not
collapse the full `mu=(k,l)` radial, SH, gate, or rank channels.
Multi-layer gate models are not supported by either LAMMPS pair style. The
standalone decoupled gate-scaling mode is likewise not implemented by the
release3 LAMMPS pair styles and is rejected with an explicit diagnostic.

The final Kokkos stride-guard overlay keeps the million-atom ranked-gate
workspace exact while reusing mutually exclusive device phases. On one
A100-80GB at `chunksize=1054560`, the measured peak fell from 51,147 MiB to
30,831 MiB. A four-GPU run reduced the largest per-rank peak from 13,236 MiB
to 8,158 MiB and retained energy, force, and stress parity. The final
1,000-step ABBA comparison measured 1,588.3750 s versus 1,589.8275 s for its
immediate predecessor, so the memory reduction did not introduce a measured
speed penalty.
The Kokkos pair style requires a half neighbor-list package configuration with
Newton pair forces enabled, for example
`-k on g 1 -sf kk -pk kokkos neigh half newton on`; the input should also keep
`newton on`.

### GPUMD-SUS2-SH

The GPUMD backend is hosted separately at
<https://github.com/hu-yanxiao/GPUMD-SUS2-SH>. It loads trained SUS2-SH `.mtp`
files, supports single-layer SH, ZBL, and the current additive-node gate path,
and exposes runtime options through the GPUMD `potential` line. The current
GPUMD reader is declared compatible with the historical `1.1.0` feature set;
release3 `1.2.0` rank and gate-readout models require a separately updated
backend and must not be assumed compatible.
