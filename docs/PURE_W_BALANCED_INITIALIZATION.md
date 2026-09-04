# Balanced pure-W initialization and fixed head coordinates

`--pure-w-initialization=balanced` is an explicit new protocol for an incomplete,
one-gate pure-W seed, not a conversion of a trained model. Defaults and serialized
physical models remain compatible. Ordinary `--init-params=same` still retains
supplied parameters; complete checkpoints must not enter the new initializer.

## Architecture and sequence

1. Keep the gate, K/L/LK scaling, type factors and global radial scale. Replace only
   physical W coefficients with indexed random directions. The random stream is
   keyed by seed and `(l,o,k,b)`, independently of O, and has sampling variance
   `1/(K*B)`. Neither the fan-in nor the common random prefix changes with O.
2. Evaluate the real gate and outer basic moments on all training atoms and MPI
   reduce `q[l,o]^2 = mean_i sum_m A[i,o,l,m]^2/(2*l+1)`. Multiply the entire K-by-B
   W output block by `target_rms/q[l,o]`. All m share one factor, preserving rotation
   equivariance. The default target is 1; this is not an energy/force output scale.
3. Fit the existing linear head once against weighted E/F/stress. Bypass the
   fitted-head global Rescale search and inherited 75-step pretraining phase.
4. Pack W using its existing signed-pivot weight-norm chart. Establish fixed head
   optimizer coordinates, then start BFGS.

The random realization is not guaranteed to have exactly zero empirical mean,
orthogonal outputs or unit feature variance; the data calibration controls its
feature scale. Across O, shared W and gate prefixes match but refitted heads need
not match. Calibration is folded into physical W, adds no runtime normalization,
and has no position-dependent divisor. RMS <= 1e-12 blocks remain unchanged and
are reported, not deleted or infinitely amplified. Factors are clipped to
`[1e-6,1e6]`; clipped blocks are not claimed to attain the target. Non-finite
statistics fail explicitly. Logs give per-block `rms_before`, `factor`,
`rms_after`, and `status`; `rms_after` is algebraic, with independent reevaluation
covered by tests.

## Physical coefficients versus optimizer coordinates

For physical scalar-head h, a fixed reference and positive scales d define

    z_a = d_a*(h_a-h_ref_a),
    h_a = h_ref_a+z_a/d_a,
    grad_z[a] = grad_h[a]/d_a.

The new `efs-head` mode calculates weighted E/F/stress Gauss-Newton diagonal
entries H_aa directly from the linear component columns, including the actual
force-loss curvature and physical head L2 curvature. It needs no random probes
or full Gram matrix. With positive block median m and floor f,
`d_a=sqrt(max(H_aa,f*m))`, clipped to `[1e-8,1e8]`; uninformative zero columns use
identity if no positive floor exists. Unlike `efs-gn`, it can enlarge weak
directions as well as shrink stiff ones. This is diagonal preconditioning, not
decorrelation or full whitening.

The physical objective is unchanged: evaluate R at `h_ref+z/d`, never replace
physical L2 by isotropic L2 on z. Data and regularization gradients are combined
before pullback. The E/F/stress parameter/position mixed derivatives remain those
of the physical model because d is fixed and independent of atomic positions.
An identity BFGS inverse-H in z is an intentional preconditioner, not preservation
of the previous trajectory. Saved models contain physical coefficients; optimizer
history is not reused across different coordinate maps.

`efs-head` currently requires a common species center prefactor, as after
TrainLinear; it rejects unequal prefactors rather than computing wrong columns.
Residual two-layer heads are also rejected. Use existing `efs-gn` for general
trained checkpoints. W weight-norm rows are not
scaled twice: their chart is `C=exp(a)*u(v)`, and a compensated rowwise constant
scale would merely translate a, not improve that chart's local conditioning.

## Usage and boundaries

After ordinary `init-sh` generates an incomplete pure-W seed:

```sh
bin/mlp-sus2 train seed.mtp train.cfg \
  --pure-w-initialization=balanced --init-random-seed=1 \
  --pure-w-init-target-rms=1 --max-iter=100 \
  --curr-pot-name=current.mtp --trained-pot-name=trained.mtp
```

Balanced mode defaults to weight-norm and `efs-head` with BFGS. Explicit coordinate
mode `none` or `efs-gn` is allowed for ablations. `--max-iter=0` still calibrates W,
fits the head and exports the initialized physical model, without BFGS. Without
balanced mode, the old zero-step `same` snapshot behavior is unchanged.

This initial implementation requires weight-norm and L2 force loss for its initial
linear solve, and excludes complete models,
`init-params=random`, `do-lin-rescale`, `fine-tune`, `update-mindist`, multi-layer
or residual stages, ZBL and compression guards. Periodic `do-lin` without Rescale
is optional. These boundaries prevent a second initialization or inconsistent
linear target. Standard resume does not repeat calibration, for example:

```sh
bin/mlp-sus2 train current.mtp train.cfg --init-params=same \
  --radial-parameterization=weight-norm \
  --bfgs-coordinate-normalization=efs-gn --max-iter=100
```

No initializer guarantees a global optimum or eliminates feature correlations.
Validation must cover common prefixes, rotations, physical and coordinate
derivatives, zero blocks, MPI, reload, unchanged penalties, line-search counts
and held-out errors before adopting this protocol for formal training.
