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
No tanh or additional nonlinear normalization is introduced.

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

Add `--joint-readout` to `init-sh` or inline `train train.cfg --init-sh`.
Fresh initialization preserves the original main initialization and gives the
joint head zero coefficients without consuming additional random draws.

For an existing adaptive model:

```
mlp-sus2 enable-joint-readout-dev old.mtp enabled.mtp
mlp-sus2 train old.mtp train.cfg --joint-readout=true [training options]
```

Enabling preserves every original physical coefficient and initializes only
the additional head to zero. It therefore preserves energy, forces and virial
before training, while the new head has trainable feature gradients. Removing
an existing head implicitly during resume is rejected.

The fixed-RMS training chart adds independent positive joint factors. Stored
physical coefficients obey `d_joint = output_scale * D_joint * theta_joint`.
Changing this optimizer chart does not change the physical head. The joint
head's physical-coefficient L2 is averaged separately and added to the old
head penalty, so adding features does not dilute the old penalty.

## File and execution contract

Enabled models use version 1.6.0 with explicit readout, adaptive-layout and
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
not assert unrun test or performance results. The separate GPUMD interface is
not automatically compatible with version 1.6.0.
