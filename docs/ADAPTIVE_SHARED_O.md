# Adaptive shared-o projection

`shared-o` shares only the environment projection across the outer channel O.
Let Q be the number of inner scalar features and N the radial basis size:

```
h[j,l,n] = sum_q U[l,n,q] S[j,q]
c[j,l,o,n] = c0[l,o,n] + a[Z_j,l,o,n] phi(h[j,l,n])
```

The optional source-species index in `a` is omitted with shared amplitudes.
`phi` is `none` (identity) or `tanh`. Each O channel retains its own base radial
coefficients and amplitudes; the channels do not become identical.

## Select at initialization

In the existing native training YAML, use:

```yaml
model:
  kind: coordination-adaptive
  response:
    layout: shared-o
    latent_channels: 1
    activation: linear       # linear or tanh
    element_mode: shared     # shared or source-element
```

This is a fragment to merge into an existing model/data/training configuration.
The CLI equivalent is `--adaptive-radial-projection=shared-o`.
`share-o` is accepted as an input alias and saved as `shared-o`.
The `shared-n` latent-rank option must remain 1 for this mode.
Full and shared-n defaults and stored layouts are unchanged.

The new model format is **1.10.0**. It supports the existing same-channel,
cross-channel and grouped-cross outer graphs, with the usual graph bounds,
amplitude/readout options, inner readout, joint readout and tensor modes.
A trained arbitrary full model cannot be converted losslessly by averaging U;
choose shared-o at initialization or resume a checkpoint with the same layout.

## Compact execution and derivatives

U contains `(L+1)*N*Q` parameters instead of `(L+1)*O*N*Q`.
Forward evaluation projects once per source atom. Reverse evaluation first
sums coefficient adjoints over O and then applies the compact U transpose:

```
bar_h[j,l,n] = phi'(h[j,l,n]) * sum_o a[Z_j,l,o,n] bar_c[j,l,o,n]
```

Native E/F/virial parameter derivatives retain the mixed derivative terms,
including tanh's second derivative. Linear energy heads retain the complete
joint normal equations, including off-diagonal Gram blocks. Physical radial
regularization includes the coefficient multiplicities and cross terms.

For the OFF23 L4/K6/R4/O4 configuration with N=8 and Q=1503, full Adaptive has
245,119 parameters and shared-o has **64,759**. This is a model-specific count,
not a universal count for all L4/K6/R4/O4 graphs.

## Inference interfaces

LAMMPS, GPUMD and PySuperlinear read the projection from the model; do not add
an inference input flag. LAMMPS supports both `radial_mode direct` and
`radial_mode table`. Its shared-o CPU path caches compact source projections
once per call and sums compact edge adjoints before the U transpose; Kokkos
uses the same compact layout. Existing GPUMD Adaptive radial paths consume
the resulting coefficients without expanding U across O. GPUMD retains its
existing default `sus2_float` precision and unsupported-path guards.

```python
from pysuperlinear import SuperlinearCalculator
atoms.calc = SuperlinearCalculator("shared_o.sus2", species=["H", "O"])
assert atoms.calc.metadata["projection"] == "shared-o"
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

Rebuild the native Python extension after updating source. Element order must
match the model's type order. PT initialization/export accepts the same
`adaptive_radial_projection="shared-o"` option; its CLI uses
`--adaptive-radial-projection shared-o`.
