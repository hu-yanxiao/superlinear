# Adaptive radial retained rank and activation

These options extend the existing adaptive-coefficients model without changing
the gate or outer graphs:

```
--adaptive-radial-projection=full|shared-n  # default full
--adaptive-radial-shared-n-rank=d          # default 1, 1 <= d <= N
--adaptive-radial-activation=none|tanh      # default none
--adaptive-radial-amplitude-mode=shared|species
```

Full projection requires `d=1` (the rank option is otherwise irrelevant).
Activation is independent: full projection also supports tanh. The default
keeps old model semantics, coefficient counts and model version. Shared-n with
d=1 and no activation is exactly the existing shared-n model. Source species
means Z_j, not receiving species Z_i. All gate/outer cross/same combinations,
readout modes, tensor messages, joint readout and RY/RD angular execution remain
independent of these options.

## Physical formulas and storage

Let s_jq be invariant gate scalars before the adaptive projection. For shared-n:

```
h[j,l,o,t] = sum_q U[l,o,t,q] * s[j,q]
x[j,l,o,t] = phi(h[j,l,o,t])
delta_c[j,l,o,n] = sum_t B[Z_j?,l,o,n,t] * x[j,l,o,t]
R[i,j,l,o] = sum_n (c[l,o,n] + delta_c[j,l,o,n]) * T[n](r_ij; scaling[Z_i,Z_j])
```

Here phi is identity or tanh; **B is the persisted adaptive amplitude array**,
not an additional factor multiplying a second amplitude. U is `[L,O,d,Q]`,
B is `[L,O,N,d]` or `[S,L,O,N,d]`, with `t` innermost in B. For d=1 the flat
arrays are unchanged from the old format. Counts are L O d Q and S? L O N d.
O and N do not change when d changes. Full projection instead uses
`delta_c[j,l,o,n] = a[Z_j?,l,o,n] * phi(sum_q U[l,o,n,q]*s[j,q])`.

All inputs to phi are scalars, hence rotational/reflection invariance is
preserved. When phi is identity the effective linear response is
`C[Z_j?,l,o,n,q] = sum_t B[Z_j?,l,o,n,t]*U[l,o,t,q]`.
Its radial rank is at most d. With tanh this is only the Jacobian at s=0;
there is no exact global folding into the previous linear model. Enabling
tanh on an existing nonzero model changes E/F/S; resume rejects silent changes
to rank, activation or layout. No arbitrary full-to-low-rank truncation occurs.

## Force and mixed force/parameter derivatives

For a coordinate/strain direction dot(s), compute raw dot(h)=U dot(s), then
dot(x)=phi'(h) dot(h). After reducing receiving-edge adjoints through B:

```
bar_h     = phi'(h)*bar_x + phi''(h)*dot(h)*bar_dot_x
bar_dot_h = phi'(h)*bar_dot_x
bar_B     += bar_c*x + bar_dot_c*dot(x)
bar_U     += bar_h*s + bar_dot_h*dot(s)
bar_s     += U^T*bar_h
bar_dot_s += U^T*bar_dot_h
```

For tanh, `phi'=1-x*x`, `phi''=-2*x*(1-x*x)`. The second derivative term is
essential for force/stress training. No division by B, U, or x is used, so
zero-amplitude and zero-signal cases retain the correct gradients. Tensor and
joint branches add their adjoints separately to the original gate/outer graph.

## Initialization and normalization

Fresh `init-sh` writes deterministic distinct latent directions; initial B is
0.01/sqrt(d). Data-balanced fresh initialization preserves the prior d=1/none
scheme. It normalizes projected input features and uses distinct deterministic
U rows. For tanh it additionally measures each raw preactivation row RMS and
scales nondegenerate rows to 0.5, avoiding feature-correlation-induced immediate
saturation. This is a fresh-model calibration only, not a resume transform.

Next the actual activated radial response with common B/sqrt(d) is measured;
its strength is calibrated to the configured relative radial RMS (existing
default 0.03). Final outer-moment RMS normalization scales c and B, never U,
so it does not change the nonlinear operating point. Resumes retain the exact
physical U and activation; optimizer-coordinate scales do not alter E/F/S.

## Regularization

Response L2 and radial smoothness use C=sum_t B_t U_t, including all cross-t
terms. They are not separate factor norms and cannot be replaced with a sum
of individual latent-channel penalties. With identity this is the true folded
linear response; with tanh it is explicitly an **origin-Jacobian prior**, not
the global nonlinear response norm. It does not bound learned B or guarantee
bounded force/extrapolation. Existing baseline-radial and scalar readout
regularization remain unchanged.

For d>1 shared-n, native priors contract the latent Gram matrix
`G[t,u]=sum_q U[t,q]*U[u,q]/scale[q]^2` exactly. The pair/radial-grid loop
therefore uses d-dimensional vectors, not Q-dimensional folded responses.
U gradients accumulate a d-by-d metric and contract with U once per (l,o).
This retains cross-t terms while removing Q from the expensive S-squared
radial-grid work. The d=1 legacy prior route is retained unchanged.

## Execution and serialization

Projection uses one compact per-atom GEMM with L O d outputs. Activation is
cached once per atom; U is never expanded to `[L,O,N,Q]` in forward/backward.
Radial expansion costs O(atoms L O N d), and receiving-edge coefficient
adjoints are reduced before compact projection backpropagation. Training
caches raw dot(h) for the phi'' term; ordinary EFS does not allocate it.

Models using d>1 or tanh use version 1.7.0. Ordered metadata after readout:

```
adaptive_radial_projection = shared-n
adaptive_radial_amplitude_mode = shared
adaptive_radial_shared_n_rank = 2
adaptive_radial_activation = tanh
tensor_message_lmax = 0
joint_readout_enabled = false
```

An enabled joint head uses the complete existing ordered joint block instead
of the single false marker. Missing, duplicated, invalid or misplaced fields
are rejected; old schemas cannot silently carry new semantics. Old v1.2-v1.6
models default to d=1 and no activation.

## Qualification

Run `dev_test/adaptive_rank_activation_check.py` in a server LSF allocation,
not on the local workstation. Minimal checks include native E/F/S coordinate
FD, mixed parameter directions (including force-only and stress-only), exact
identity folding, response-prior gradients, linear readout components, fresh
and preserve-function resume, malformed schema, and combined nonzero tensor
plus joint readout. `--stage full` adds every gate/outer/readout combination.
The harness writes a report and does not claim PASS until all checks finish.
Separate PT and GPUMD parity/qualification are required before deployment.
