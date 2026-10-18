# Optional joint gate/outer readout

This option adds an independent invariant head to adaptive-coefficient
superlinear models. The existing gate graph, outer graph and main readout are
unchanged. It works with either graph's same-k or crossed mode, either adaptive
projection/amplitude layout, shared/species readout and tensor messages.

For each center atom i and l >= 1, the two inputs are

```
Amean[i,l,m] = sum_o A[i,o,l,m] / O
hmean[i,l,m] = sum_g z[i,g,l,m] / G
```

Here A is the actual outer primitive before scalarization, including tensor
edge corrections if enabled. z is the actual post-rank gate primitive before
scalarization, not the gate scalar vector or the raw K-channel primitive.
The joint inputs themselves introduce no tanh or additional nonlinear
normalization. The independent adaptive radial projection may optionally use
tanh; that does not change the definition of this joint head.

The independent K=2 q-total graph has input 0=Amean and input 1=hmean. It uses
the outer body's order and angular limits. l=0 is excluded during leaf
enumeration. All pure-A, pure-h and mixed scalar invariants are retained in
the original CG construction order. Its indices are local and do not change
the main graph's indices.

The extra site energy is `h_Z * sum_p d[p]*Psi[p]` for shared readout and
`sum_p d[Z,p]*Psi[p]` for species readout. h_Z is the existing shared-readout
center coefficient. The added physical coefficients are a separate tail after
the original complete parameter vector. The original `LinearCoeffCount()` is
unchanged; the linear-equation API has additional joint columns.

## Creation and upgrade

Joint readout requires `--superlinear-mode=radial-coefficient-gate` and
`--l-max` at least 1. The following two-element example uses K8/G4/O8,
crossed gate, same-k outer, shared-n projection and source-species amplitudes.
Choose the species count and model settings to match your dataset. Run fresh
examples in a new working directory so existing training outputs are retained.

Generate the topology independently, then perform data-dependent
initialization without any optimizer updates:

```bash
mlp-sus2 init-sh joint.topology.sus2 \
  --superlinear-mode=radial-coefficient-gate --joint-readout \
  --species-count=2 --l-max=3 --k-max=8 \
  --body-order=6 --body-l-max=3,3,3,2,1 \
  --two-layer-gate-k-rank=4 --two-layer-gate-body-order=4 \
  --two-layer-gate-factor-pruning=q-total --outer-k-rank=8 \
  --sh-factor-pruning=same-k --scaling-map=K \
  --cutoff=6.5 --radial-basis-size=10 \
  --adaptive-radial-projection=shared-n \
  --adaptive-radial-amplitude-mode=species --readout-mode=shared
mlp-sus2 train joint.topology.sus2 train.cfg \
  --init-params=same --init-random-seed=20260912 \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --max-iter=0 --trained-pot-name=joint.initialized.sus2
```

The equivalent inline topology creation and initialization is:

```bash
mlp-sus2 train train.cfg --init-sh \
  --inline-sh-model=joint-inline.topology.sus2 \
  --superlinear-mode=radial-coefficient-gate --joint-readout \
  --species-count=2 --l-max=3 --k-max=8 \
  --body-order=6 --body-l-max=3,3,3,2,1 \
  --two-layer-gate-k-rank=4 --two-layer-gate-body-order=4 \
  --two-layer-gate-factor-pruning=q-total --outer-k-rank=8 \
  --sh-factor-pruning=same-k --scaling-map=K \
  --cutoff=6.5 --radial-basis-size=10 \
  --adaptive-radial-projection=shared-n \
  --adaptive-radial-amplitude-mode=species --readout-mode=shared \
  --init-params=same --init-random-seed=20260912 \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --max-iter=0 --trained-pot-name=joint-inline.initialized.sus2
```

Inline mode accepts exactly one positional argument, `train.cfg`; do not add
a positional model filename. If `--inline-sh-model` already exists, it is
reused instead of creating a fresh topology. Use a new path for a fresh run.
`init-sh` alone does not perform data-dependent RMS calibration. The
`train ... --max-iter=0` calls above do that initialization; choose a positive
step limit to continue directly into training. Fresh initialization preserves
the original main initialization and gives the joint head zero coefficients
without consuming additional random draws.

For an existing adaptive model:

```
mlp-sus2 enable-joint-readout-dev old.sus2 enabled.sus2
mlp-sus2 train old.sus2 train.cfg --joint-readout=true [training options]
```

Enabling preserves every original physical coefficient and initializes only
the additional head to zero. It therefore preserves energy, forces and virial
before training, while the new head has trainable feature gradients. Removing
an existing head implicitly during resume is rejected.

The fixed-RMS training chart adds independent positive joint factors
`S[p] = output_scale * D_joint[p]`. Stored physical coefficients obey
`C_joint[p] = S[p] * q_joint[p]`. Only the new joint head is regularized in
this fixed optimizer chart:

```
R_joint = lambda / N_joint * sum (C_joint / S)^2
dR_joint/dC_joint = 2 * lambda / N_joint * C_joint / S^2
```

For a species readout, each species uses the same per-scalar S, and N_joint
counts all species/scalar coefficients. The joint term is averaged separately
and added to the unchanged original physical-head penalty, so adding features
does not dilute the original penalty. Linear-solve ridge and physical-coordinate
Hessian terms use the same `1/S^2` factors.

S is fixed during training and saved in `joint_readout_optimizer_scales`.
Fresh initialization, or enabling joint readout directly within the current
`train old.sus2 ... --joint-readout` invocation, calibrates it from the data.
Loading a joint-enabled checkpoint preserves its saved S rather than silently
recalibrating it from the new dataset or current features. Changing S while
keeping physical C fixed preserves energy, forces and virial but generally
changes R_joint; it is therefore a change to the optimization prior, not merely
a harmless reparameterization of the full training objective.

The standalone `enable-joint-readout-dev` command has no dataset and writes an
uncalibrated unit chart, S=1, with C=0. Loading that output retains its explicit
unit chart; the loader does not guess calibration state from whether all
scales equal 1. For data-calibrated initialization, prefer direct
`train old.sus2 train.cfg --joint-readout` (optionally with `--max-iter=0`) or
the fresh topology/initialization examples above.

## File and execution contract

Enabled models use version 1.6.0 (1.7.0 when combined with retained rank >1 or
adaptive tanh) with explicit readout, adaptive-layout and
tensor-lmax metadata (including tensor lmax=0). A `joint_readout_*` block
follows those fields and precedes `outer_k_rank1_basic_begin`. The block stores
the complete independent graph, shared/species physical head coefficients and
one positive optimizer-scale factor per joint scalar. The loader rejects
missing, duplicate, malformed and noncanonical graph metadata. Disabled
models retain the previous file versions and evaluation path.

Forward execution reuses the existing gate and outer primitive caches.
Reverse execution sends 1/O and 1/G mean adjoints to both branches. Mixed
force/virial parameter derivatives carry both value and directional adjoints;
the direct gate-primitive path is not detached. Linear component columns reuse
the cached graph values and physical radial contractions.

Server numerical qualification is required before release; this document does
not assert unrun test or performance results.

The updated GPUMD-superlinear source implements the independent joint head
for v1.6 and combined rank/tanh/joint v1.7 models. It checks the complete
canonical graph, supports RY/RD and shared/species readout, and propagates
both outer and gate force adjoints. An all-zero head uses the old fast path
without joint buffers or kernel launches. Install only after server
CPU/GPU qualification; older GPUMD binaries still reject these models.
Never downgrade a version header or remove joint metadata to bypass that
check: doing so would discard part of the trained energy model. See the
GPUMD repository's `docs/adaptive-latent-joint-v17.md` for its precision and
qualification contract.
