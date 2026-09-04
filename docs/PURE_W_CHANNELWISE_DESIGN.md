# Pure distance matrix with channelwise outer products

Development branch: `codex/pure-w-channelwise`, forked from `0002956`.
This is a new model variant, not an in-place change to existing checkpoints or running jobs.

## Agreed mathematical contract

The gate retains cross-channel invariant construction. Its scalar readout produces K channels for each angular order. The complete pair scalar x includes the existing species/type and gate-affine factors.

For each l, the physical distance matrix W_l(r) has shape K x O. Its output channel is o; its input channel is k:

    A[i,o,l,m] = sum(j,k) x[i,j,k,l] W[l,k,o](r[i,j]) Y[l,m](rhat[i,j]).

    W[l,k,o](r; p,q) = sum(b) C[l,k,o,b] phi[b](r; scaling[p,q,sigma(k,l)]).

There is no separate outer P diag(R) contribution. The gate's own radial generator is retained. K, O and gate rank are distinct quantities; O may exceed K.

Scaling is indexed by input channels, never output channels:

- K map: sigma(k,l) = k.
- LK map: sigma(k,l) = k*(lmax+1)+l.
- Existing L map, when selected, keeps sigma(k,l) = l.

All O matrix entries with the same input (k,l) share that pair's scaling parameters. Scaling derivatives sum contributions over o and all relevant angular channels. The matrix has (lmax+1)*K*O*radial_basis_size physical coefficients. Internal coefficient layout may remain output-major for compatibility with the established mixer kernels; this does not change the mathematical K x O convention and must be documented explicitly.

The outer graph is the K=1 angular graph applied to O channels independently. Each product's factors have the same o, while all allowed angular couplings are retained. If a one-channel graph has S angular scalar features, the outer scalar feature count is O*S (excluding an independent species constant). Gate and outer plans have independent scalar selections; neither must be a subgraph of the other. Their generators and evaluator primitives remain shared.

## Model and command interface

New explicit model metadata distinguishes `outer_channel_mode = same-channel` and `outer_radial_mode = pure-matrix`. Missing metadata preserves the old interpretation. New init mode is selected with `--superlinear-mode=pure-w-channelwise`; existing model/CLI paths remain supported. Pure mode requires the existing two-layer gate and full distance-matrix coefficient block; legacy P has zero trainable coefficients. It must not fall back to a legacy radial path when every W coefficient is zero.

The branch's train default is `--init-params=same`: retain the physical parameters supplied in the model file, without random reinitialization. This does NOT mean making all W entries identical. New model W initial values are deterministic and nonzero, with distinct output directions to avoid permutation-symmetry locking:

    C[l,k,o,b] = 0.01 / sqrt(K) * (1 + 0.25*cos((o+1)*(k*B+b+1))).

This deterministic seed is not an identity/residual initialization. Saved pure-mode checkpoints are resumed without resetting W or scaling. A pure model with `--max-iter=0 --init-params=same` saves the initial physical snapshot before linear warmup/optimization. The legacy max-iter=0 behavior is unchanged.

## Optimization and regularization

Apply the established signed-pivot radial weight-normalization chart to each W[l,k,o,:] physical coefficient vector. Retain its established zero-row handling and exact physical-value roundtrip. The gate's own independent radial rows still use their existing chart. In pure mode, obsolete outer radial rows are not normalized or trained merely as unused compatibility storage.

Move the outer radial smoothness penalty to the actual W(r) entries, including all first/second chain-rule terms through shared scaling parameters. Regularization is defined in physical coefficient/function space, not on arbitrary weight-normalization coordinates. Do not silently assume the old factorized parameter penalty or optimization trajectory is unchanged. Retain gate-specific regularization. Document any reduction/normalization convention over matrix entries.

With N element types and Q radial quadrature points, the implemented objective is

    R_W = lambda * r_cut / (Q*(lmax+1)*K*O*N*N)
          * sum(t,l,k,o,p,q) [dW[l,k,o](r_t;p,q)/dr]^2,
    r_t = (t+0.5)*r_cut/Q.

It includes the model's fixed global radial scaling, not the type/gate factors already in x. All O terms contribute to the same input-channel slope/shift gradient. Gate scalar/full L2 and scalar-head L2 retain their existing meanings; no new gate-radial smoothness is introduced. The historical hardcoded 1e-9 unit-norm/inter-row orthogonality penalty on factorized R is explicitly disabled in pure mode, not imposed on W output channels. Direct nonzero calls to that obsolete penalty are rejected. Weight-norm itself is an exact coordinate chart, not a unit-norm constraint or an additional regularizer.

## Execution and chain rule

The CG angular graph is constructed for K=1 once, then its indices are bound to each output channel. Gate and outer reuse the angular CG cache during construction. The current model file/evaluator contains the compiled O channel instances of that template; it does not yet use a compressed runtime DAG or a dedicated SIMD O-axis kernel.

For any coordinate or parameter theta,

    dA = sum(j,k) [(dx)*W*Y + x*(dW)*Y + x*W*(dY)],
    dW = sum(b) [(dC_b)*phi_b + C_b*(dphi_b)].

For a shared scaling parameter s, dphi_b/ds is accumulated for every associated k,l,o. Force-loss parameter derivatives also require d²W/(dr ds) and the derivatives of the gate signal x; they are not obtained by differentiating W alone. The linear E/F/stress design matrix must use the same complete W primitives and gate chain as the final predictor.

## Usage and validation

Example of a new model (O can exceed K):

```bash
bin/mlp-sus2 init-sh pure.mtp --superlinear-mode=pure-w-channelwise \
  --species-count=2 --l-max=2 --k-max=2 --outer-k-rank=4 \
  --two-layer-gate-mode=mu-scalar-full --two-layer-gate-k-rank=2 \
  --body-order=4 --body-l-max=2,2,2,2 --scaling-map=LK \
  --radial-basis-size=4 --cutoff=5
bin/mlp-sus2 train pure.mtp train.cfg --init-params=same \
  --radial-parameterization=weight-norm --radial-smooth=1e-6
```

Focused checks, using a freshly built `bin/mlp-sus2`:

```bash
bash dev_test/sh_pure_w_channelwise_init_check.sh
bash dev_test/sh_pure_w_model_contract_check.sh
python3 dev_test/sh_pure_w_physics_check.py
bash dev_test/sh_pure_w_regularization_check.sh
bash dev_test/sh_pure_w_linear_components_check.sh
```

The model-contract C++ harness links existing serial objects. Other checks use the CLI only. Test artifacts are generated outside tracked source paths or under `.codex_tmp/`.

`calc-eij` is explicitly rejected for pure models until a compatible edge-energy decomposition is implemented. `calc-efs`, training, linear E/F/stress components and their gradients are the supported paths; separate GPU/GPUMD/LAMMPS ports are not included in this change.

## Migration and compatibility boundaries

For unchanged gate/outer graph and identical radial basis, cutoff and scaling, old effective kernels can be folded exactly: C_new[l,k,o,b] = C_old[l,k,o,b] + P[l,o,k]*a[l,k,b]. The entire W must not be initialized to zero to emulate a zero residual. Changing a cross-channel outer graph into a channelwise graph is a separate hypothesis-class change and is not an exact general checkpoint conversion.

Existing running FeCN6 and VO2 jobs and the original branch are out of scope for mutations. Legacy model loading must retain its old physical function. New format validation must enforce stage-local references, scalar mappings and parameter offsets. Runtime value, force, stress, mixed force-loss gradients and weight-normalized optimizer coordinates require numerical validation, including O>K, K/L/LK scaling, zero/nonzero W, cache/recompute and serialization roundtrip cases.
