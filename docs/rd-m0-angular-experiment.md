# RD m0 angular backend

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

## Maintained-branch integration, 2026-09-14

The RD implementation and its qualification tools are merged into the maintained
`main` branch. `ry` remains the default; this promotion
does not change model mathematics, initialization, optimizer settings or MTP
format. Use the regular installed executable with an explicit selection:

```sh
SUS2_SH_ANGULAR_BACKEND=rd-m0 /work/phy-weigw/app/superlinear/mlp-sus2 calc-efs model.sus2 input.cfg rd.cfg --no-loss
```

PyTorch carries the equivalent opt-in feature on `<historical-branch>`.
Its accepted values are only `ry` and `rd-m0`. Native `hybrid`/`ry-new` are
retained for diagnostics and must not be exported into a PyTorch invocation.
GPUMD source integration and binary acceptance are separate: its unresolved
FP32 finite-difference checks are not certified by this native promotion.
The experiment paths and measurements below are retained historical evidence.

## Original experiment contract

This branch adds optional C++ angular execution backends for the current
`radial-coefficient-gate` superlinear model. It does not add parameters or
change the saved model format, CG graphs, model physics or training objective.
The production/default backend remains `ry`.

## Definition

Let Q_ij map the fixed unit z axis onto the unit edge direction rhat_ij.
In the existing real spherical-harmonic convention,

    Y_lm(rhat_ij) = D^l_m0(Q_ij) c_l,
    c_l = Y_l0(zhat) = sqrt((2l+1)/(4*pi)).

Only the full `rd-m0` backend changes the gate and outer primitive reductions to

    h_i,klm = c_l sum_j R_gate,ij,kl D^l_m0(Q_ij),
    A_i,olm = c_l sum_j R_outer,ij,ol D^l_m0(Q_ij).

Here each R includes its original scalar species/environment factors. The
factor c_l is applied after the complete neighbor reduction and before channel
rank contraction and the existing CG product/scalar graphs. Thus all downstream
tensors, graph inputs and readout parameters retain their original physical
normalization. Gate/outer cross or same topology and shared/species readout
are unchanged.

This is the first, restricted column-rotation experiment: it is not a general
Wigner matrix, a learned rotation, or a new environment-transport model. The
rotation Q is not unique, but D_m0 is independent of its right-hand z-axis roll.
The weighted sum of these columns is not itself a rotation matrix. Rotating the
fixed axial tensor onto each edge precedes summation; there is no single
rotation applied to an already summed neighborhood.

## Backends, implementation and derivatives

All four selections use the same physical model and coefficients:

| Selection | Values-only angular kernel | Angular kernel with derivatives | Neighbor reduction / raw angular cache |
| --- | --- | --- | --- |
| `ry` (default) | Existing real SH | Existing real SH | Physical Y; no raw D cache |
| `ry-new` | New direction-cosine recurrence, immediately normalized to Y | New recurrence, immediately normalized to Y/dY | Physical Y; no raw D/dD cache |
| `hybrid` | Existing real SH | Same new physical Y/dY kernel as `ry-new` | Physical Y; no raw D/dD cache |
| `rd-m0` | New D recurrence | New D/dD recurrence | Raw D reduction, then c_l restoration; retains raw and physical cache views |

`ry-new` and `hybrid` are physical-only ablations, not deferred-normalization
backends. Their recurrence workspace is the final Y/dY buffer and they do not
allocate the extra raw angular caches used by `rd-m0`. `hybrid` preserves the
old values-only route while using the new derivative route; it does not have
a separate faster EFS kernel than `ry-new`.

`dev_src/rd_m0_angular.h` independently evaluates the normalized real rotation
column using direction-cosine recurrences. It does not call the old spherical
harmonic evaluator or divide an already evaluated Y by c_l. There are no
`acos`, `atan2` or inverse transverse-radius factors at the poles.

For n = r/|r|, derivatives include the complete normalization chain rule:

    dD/dr_a = (dD/dn_a - n_a sum_b n_b dD/dn_b) / |r|.
    d(c_l sum_j R_j D_j)/dr = c_l sum_j (dR_j D_j + R_j dD_j).

In `rd-m0`, raw D/dD caches serve the experimental reductions. Physical Y/dY cache views
are produced directly as c_l D/c_l dD for existing reverse, mixed-loss-gradient
and auxiliary code. Physical adjoints must not be multiplied by c_l a second
time. Cached gate moments are always physical and are not rescaled on reuse.
For an existing tensor-message angular addition Y+H, the experimental raw
reduction uses D+H/c_l, preserving exactly the original tensor-message model.

The shared angular helper supports l=0..6; the existing native model runtime
limit remains l<=4. This experiment does not change that limit. Historical
non-adaptive model paths may use the independent physical-Y angular adapter,
but are not claimed to use deferred normalization in their outer reduction.

## Usage

Set the backend before starting the process (selection is cached per process):

```sh
SUS2_SH_ANGULAR_BACKEND=ry /path/to/experimental/mlp-sus2 calc-efs model.sus2 input.cfg ry.cfg --no-loss
SUS2_SH_ANGULAR_BACKEND=ry-new /path/to/experimental/mlp-sus2 calc-efs model.sus2 input.cfg ry-new.cfg --no-loss
SUS2_SH_ANGULAR_BACKEND=hybrid /path/to/experimental/mlp-sus2 calc-efs model.sus2 input.cfg hybrid.cfg --no-loss
SUS2_SH_ANGULAR_BACKEND=rd-m0 /path/to/experimental/mlp-sus2 calc-efs model.sus2 input.cfg rd.cfg --no-loss
```

An unset variable selects `ry`; values outside these four are rejected. RD adaptive setup
prints a `[RDM0_ANGULAR] backend=rd-m0` diagnostic. The same selection works for
`init-sh`, training and evaluation. No model conversion or reinitialization is
needed to switch an existing model between these mathematically equivalent
backends. Floating-point summation order differs, so bitwise equality or
identical long optimization trajectories is not promised.

`calc-efs --no-loss` is an additive diagnostic option on this branch, using
the existing lossless CFG serializer (17 significant digits). Default CFG
formatting is unchanged. Do not infer double-precision parity from ordinary
CFG files: those round forces to six and stresses to five decimal places.

The full `rd-m0` experiment intentionally keeps physical angular cache views as
well as raw D/dD. Its extra angular storage is H doubles per edge in values-only
mode or 4H doubles per edge with derivatives, H=(lmax+1)^2. Mathematical
equivalence alone does not imply a speedup; performance needs measurement.
The original experiment did not change GPUMD or production installed binaries. The
[separate PyTorch implementation branch](https://github.com/hu-yanxiao/SUS2-SH-PyTorch)
is not part of this native implementation or its validation claims. Its own
qualification is independent. Its snapshot `0de63a3` passed 45 CPU/CUDA
float64 checks in server job `4272217`; see that repository's RD document
for the separate autograd, native-export and batch-dependent timings.

## Validation

Compile and run on the server, not the developer laptop:

```sh
g++ -std=c++11 -O3 -Idev_src native_tests/rd_m0_angular_check.cpp -o rd_m0_angular_check
./rd_m0_angular_check
python3 dev_test/rd_m0_backend_check.py /path/to/experimental/mlp-sus2
```

The standalone test covers independent analytic references, Cartesian
derivatives, exact/near poles, parity and rotations. The CLI regression checks
native model/backend parity, coordinate and mixed EFS parameter derivatives,
and model generation/training. See its output for the actual tested cases;
unsupported model angular orders are not counted as numerical passes.

### Original two-backend qualification, 2026-09-13

Original build job: `4270352` (queue 33). Original lossless qualification job:
`4270355` (queue 33, host b07u36a). Both completed successfully.
Binary SHA256:
`5da3a8e54f75f986f97ffb6623775264c4571be54acfe609fe2cd55a68f497e8`.

- Standalone angular test: 135851 assertions / 102 direction-scale cases;
  maximum independent SH value difference 2.665e-15, derivative difference
  multiplied by radius 6.303e-15, five-point FD difference multiplied by radius
  8.434e-11. Includes l=0..6 and exact/near poles.
- Native runtime: all 10 fixtures passed (l=0..4, all four gate/outer
  cross/same combinations, nonzero tensor and joint branches, full/shared-n
  projection, shared/species amplitude/readout, K/L/LK scaling).
- Candidate RY/RD predictions were written with verified 17-significant-digit
  output. Largest absolute component difference across each fixture's native
  E/F/virial vector was 2.132e-14. Coordinate/strain FD and mixed parameter FD
  passed, including separately isolated energy, force and virial objectives.
- Both backends completed initialization and a real BFGS update (trace steps
  0 and 1; physical parameters changed). After that one-update smoke, the
  largest E/F/virial component difference was 4.858e-17. This is not a claim
  about long training trajectories or fitted accuracy.
- Two frozen, trained VO2 models (shared-n/species-amplitude/shared readout;
  full-U/species readout), each evaluated on the first two updated training
  configurations, passed lossless RY/RD parity. Maximum absolute component
  differences were respectively 3.658e-13 and 6.382e-13, in each component's
  native E/F/virial units. This was not a full-dataset accuracy benchmark.
- The production reference binary agreed with candidate RY at the legacy
  CFG output precision. That separate comparison is not FP64 parity evidence.
- Native model l=6 rejection was checked explicitly; it is unsupported, not
  a passing native numerical case. Runtime l<=4 remains unchanged.

Full evidence is retained on the server under
`/work/phy-weigw/20260321_Test/superlinear-rd-m0-experiment-20260913/qualification-4270355/`.
The original two-backend experimental installation was
`/work/phy-weigw/app/superlinear/experimental/rd-m0/mlp-sus2`.
Select `SUS2_SH_ANGULAR_BACKEND=rd-m0` explicitly. No speed improvement or MD
stability result is claimed by this mathematical qualification.

### Four-backend qualification, 2026-09-13

Build job `4272207` and qualification job `4272208` use the optimized
four-backend implementation. Qualification completed with **PASS, 10/10
positive fixtures, zero failures** on `b07u36a`. The qualified executable is
`/work/phy-weigw/20260321_Test/superlinear-rd-m0-optimized-20260913/runtime/mlp-sus2`,
SHA256 `0c31698276079574a618d36f41fe698171cab392d8e6f806ed94e09e6d2cd062`.

- All four backends passed lossless E/F/virial parity on all ten fixtures,
  covering native l=0..4 and all gate/outer cross/same combinations. Default
  and explicitly selected `ry` remained identical in the recorded checks.
- RY/RD coordinate and mixed-gradient checks cover all fixtures. The deep
  `L2-extended-0` fixture additionally checks all four backends for mixed
  gradients, isolated energy/force/virial objectives, parameter-block
  directions, values-only energy-to-force component finite differences,
  geometry A/B/A cache reuse, and full-radial-derivative/reference-EFS caches.
  This is targeted coverage, not every parameter direction in every fixture.
- All four backends passed model initialization, normalization comparison and
  a real one-update BFGS smoke (accepted trace steps 0 and 1, changed physical
  parameters). These short checks do not establish long-training equivalence.
- Both frozen trained VO2 models passed lossless all-backend checks on the
  first two updated-data configurations. The largest absolute E/F/virial
  component differences versus RY were 3.658e-13 for shared-n/species-amplitude
  shared readout and 6.382e-13 for full-U/species readout. Components retain
  their respective native units; this is not a full-dataset accuracy test.
- Invalid selectors fail closed. Native l=6 remains explicitly unsupported
  and is not counted as a numerical pass. The standalone angular l<=6 test
  is separate from native-model support. The legacy-formatted production
  comparison remains only a legacy-precision check, not FP64 evidence.

Evidence: `qualification-4272208/report.json` under the optimized server
directory above. The four-backend driver parity tolerances and individual FD
attempts are retained there; passing is not a claim of bitwise equality.

## Paired CPU performance test

`native_tests/rd_m0_benchmark.cpp` is a standalone driver linked against the
qualified native objects. `dev_test/rd_m0_benchmark.py` runs it only inside a
Linux LSF compute allocation, freezes both models and the selected CFG data,
and writes a live `summary.json`. This benchmark does not update parameters.

Job `4271804` was submitted to queue 33 on 2026-09-13 and started on
`b07u36a`. The server working directory is
`/work/phy-weigw/20260321_Test/superlinear-rd-m0-performance-20260913/`;
its report is `result-4271804/summary.json` and its runner log is
`logs/benchmark-4271804.log`. Submission is not a completed performance result;
check the report status and complete paired aggregates before drawing a
conclusion.

The test uses the two frozen VO2 models from qualification. The models have
different readout layouts as well as different U layouts, so only compare
RY and RD within the same model. It uniformly selects 500 configurations
(47112 atoms) from the updated 2162-configuration VO2 training set, including
indices in the added-data interval. Each fresh child process uses the same
single allowed CPU, one MPI rank, and one BLAS/OpenMP thread.

Four timing modes are reported separately:

- `energy`: energy and site energies, with prebuilt neighborhoods.
- `efs`: energy, forces and virial, with prebuilt neighborhoods.
- `gradient`: actual E/F/S residual loss and its full physical coefficient
  gradient, with weights 1/0.1/0.01 and vibrations weighting. Regularization,
  STD/STDD scaling, optimizer updates, and line searches are excluded. Every
  configuration gets a fresh forward pass; only its forward-to-backward
  cache is reused.
- `efs-total`: complete CalcEFS, including fresh neighborhood construction.

Loading, one full warmup pass, checksums, and output are excluded from all
timers. Pilot runs compare full-precision energy/site/force/virial outputs or
every physical gradient component, then choose equal measured pass counts
for both backends. Four pairs alternate RY/RD order. The reported paired
ratio is **RD time / RY time**; a value above one means RD is slower. Median,
range and median absolute deviation are retained, along with peak RSS and
checks that neither model coefficients nor reference labels changed.

This measures single-core CPU latency, not MPI scaling, optimizer convergence
or GPUMD throughput. Peak RSS includes loaded configurations and resident
prebuilt neighborhoods (also in `efs-total`), and therefore is not a minimal
streaming-inference memory estimate. CPU binding reduces scheduling noise but
does not eliminate shared-node load effects.

### Resumable four-backend ablation

`dev_test/rd_m0_ablation_benchmark.py` is a separate runner; it does not replace
the original runner or overwrite the original 500-CFG campaign. It accepts
`--models`, `--modes`, `--backends`, and `--count` filters, retaining `ry` as the
reference and at least four pairs. Four-backend order is balanced across pairs:
RY/new/RD/hybrid, new/hybrid/RY/RD, hybrid/RD/new/RY, RD/RY/hybrid/new.
The same fixed-objective driver, immutable reference labels, one full warmup
per fresh process, and full component/physical-gradient dump checks apply.

Add `--resume` to the identical invocation to continue a partial pair. Resume
requires the same source/model/data/binary and runner/dependency hashes,
parameters, relevant runtime environment, host and original CPU identity.
The selected CFG bytes are frozen. Only cleanly completed processes with
matching timing/log/dump evidence are reused; incomplete work is rerun, not
counted. The campaign lock prevents concurrent writers. A completed resume
does not rerun successful measurements. Changing inputs or parameters requires
a new output directory, not silently mixing results into an old campaign.

Every accepted timed process must last at least three seconds. If any backend
is too short, the entire section moves to a larger common repeat count;
earlier attempts remain recorded but are excluded from the active aggregate.
Statistics include only matching pair indices and repeat-count generations.
`COMPLETE` requires all requested pairs, independently for each section and
backend. A partially running campaign can therefore contain completed sections
without being a completed campaign. The independent interruption/resume smoke
is `dev_test/rd_m0_ablation_resume_check.py`. Server job `4272218` passed the
partial-pair interruption, resume, complete-resume no-rerun and changed-count
rejection checks. Evidence is `resume-4272218/report.json` under the optimized
server campaign root. This is a two-CFG runner smoke, not a speed result.

### Completed 100-CFG energy/EFS sections, 2026-09-13 16:28 +08:00

Job `4272209`, queue 33, is the four-backend diagnostic campaign in
`/work/phy-weigw/20260321_Test/superlinear-rd-m0-optimized-20260913/ablation100-4272209/`.
The snapshot `summary.json` updated at `2026-09-13T16:28:45.969868+08:00`
was still `RUNNING`. Only the following **completed 4/4-pair sections** are
reported here: energy and EFS for the shared-n/species-amplitude/shared-readout
model. EFS-total and gradient are not completed results in this snapshot.

This diagnostic uses **100 CFG / 9,408 atoms**, uniformly selected from the
same updated 2,162-CFG VO2 source, on one pinned CPU of `b07u36a` (Xeon Gold
6133, 2.50 GHz). It is not the earlier **500 CFG / 47,112 atoms** campaign,
nor a completed new 500-CFG verification. The new benchmark driver SHA256 is
`c5d8cfaa8df90ec352c8e4e2d8446b054967dc7504264ac08220a02fe570a07a`;
the shared-n model SHA256 remains
`6e5de8b306061cadbdefdba812e30567166d388f49b899f9b2ab61eeaf2ca449`.

Ratios are per-pair **backend time / RY time**, then median across four pairs.
Ranges describe observed paired scatter, not a confidence interval. Throughput
change is `100*(1/median_ratio-1)`, not the negative of the time change.

| Mode | Backend | Paired time ratio, median [min, max] | Time change | Throughput change | Pairs |
| --- | --- | --- | --- | --- | --- |
| Energy | `ry-new` | 1.01230 [1.00773, 1.02342] | +1.23% | -1.21% | 4/4 |
| Energy | `hybrid` | 1.00123 [0.99555, 1.00519] | +0.12% | -0.12% | 4/4 |
| Energy | `rd-m0` | 1.01524 [1.01268, 1.01763] | +1.52% | -1.50% | 4/4 |
| EFS | `ry-new` | 0.84471 [0.84177, 0.85454] | -15.53% | +18.38% | 4/4 |
| EFS | `hybrid` | 0.84034 [0.83814, 0.84846] | -15.97% | +19.00% | 4/4 |
| EFS | `rd-m0` | 0.84362 [0.84221, 0.85013] | -15.64% | +18.54% | 4/4 |

Energy uses two measured passes per process (200 configuration calls); EFS
uses one (100 calls). Median RY costs are 408.734 us/atom for energy and
971.184 us/atom for EFS. Loading and the full warmup are outside these timers.
All 100-CFG energy/EFS full-precision output checks passed; the largest
absolute E/F/virial component difference was 5.756e-13 in native units.

On this model, the new physical derivative kernel accounts for essentially
the measured EFS gain: deferred normalization is not required to obtain it.
`hybrid` avoids the small energy-only regression of `ry-new` (direct paired
hybrid/new energy ratio 0.98878) and is effectively tied with original RY for
energy. Hybrid and new use the same derivative path; their roughly 0.58%
paired EFS difference is not evidence of an additional algorithmic speedup.
Four shared-node pairs do not establish cross-CPU scaling, optimizer speed,
MD throughput, another model's performance, or the final 500-CFG result.
