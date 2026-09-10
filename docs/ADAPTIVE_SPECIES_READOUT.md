# Adaptive species readout

This document records the native C++ model-file contract for the optional
species-resolved final readout of adaptive radial coefficient gate models.

## Scope

`readout_mode = species` is supported only for
`outer_radial_mode = adaptive-coefficients`.  The default remains
`readout_mode = shared`, including old files that do not contain the field.
The readout choice is independent of both the gate scalar pruning mode and the
outer channel mode: gate same-k/cross and outer same-channel/cross-channel are
all valid when the underlying adaptive radial model is otherwise supported.

Species-readout model files are saved as `version = 1.3.0` and must contain
`readout_mode = species`.  Shared-readout files continue to use the existing
1.1/1.2 formats.  Older native or GPUMD readers should reject 1.3.0 instead of
silently evaluating a species readout as a shared readout.  GPUMD support for
species readout is not included in this change.

## Energy convention

The scalar graph and raw scalar order do not change.  If `Q` is the number of
active outer scalars and `S` is the number of species, the linear block remains:

- first `S` entries: additive `species_coeffs`;
- remaining entries: `moment_coeffs`.

For shared readout, `moment_coeffs` has `Q` entries and the historical formula
is unchanged:

```text
E_i = shift[type_i] + h[type_i]
    + h[type_i] * sum_q d[q] * linear_mults[q] * Phi_i[q]
```

For species readout, `moment_coeffs` has `S*Q` entries in row-major
`[type, q]` order:

```text
E_i = shift[type_i] + h[type_i]
    + sum_q d[type_i, q] * linear_mults[q] * Phi_i[q]
```

The additive `h[type]` term is retained deliberately.  It is no longer also
used as an environment-dependent prefactor in species mode.

During native full nonlinear training, the additive `h[type]` entries are kept
fixed as the species-readout gauge anchor.  `shift[type]` remains trainable.
This removes the exact redundancy between the two additive terms without
changing EFS evaluation, gradients, linear components, or the saved model
layout.

## Exact shared-to-species conversion

The development command

```bash
mlp-sus2 convert-adaptive-readout-dev shared.mtp species.mtp --readout-mode=species
```

converts a trained adaptive shared-readout model without changing its EFS
predictions by setting:

```text
d_species[type, q] = h[type] * d_shared[q]
```

and preserving `h[type]` as the additive species bias.  This is a physical EFS
conversion only.  It is not a statement that regularization terms or optimizer
coordinate penalties have identical numerical values after conversion, because
the trained head parameterization is different.

## Training and initialization notes

Fresh adaptive species-readout topology files keep the same non-readout
initialization sequence as shared readout.  Missing complete linear coefficients
load to the same small physical default response as shared mode: `h = 1e-3` and
`d[type,q] = 1e-6`.  Complete saved models write the full species readout block.
The saved `h[type]` values are preserved when loading or converting models, but
native nonlinear training does not optimize them in species mode.

Resume must not change readout mode implicitly.  If `--readout-mode` is passed
while training from a complete checkpoint, it must match the loaded model; use
the explicit conversion command for a structural change.

Scalar-head L2, linear components, exact head coordinates and adaptive mixed
gradients treat the species head width as `S*Q` and map each row back to
`q = local % Q` for `linear_mults`.
