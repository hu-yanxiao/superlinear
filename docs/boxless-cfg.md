# Molecular CFG configurations

A CFG frame with Cartesian coordinates and no `Supercell` section represents a
fully nonperiodic molecule. The internal cell is zero; no artificial vacuum box
is introduced. Each frame resets the cell, so periodic and molecular frames may
be mixed in one file. Saving a zero-cell configuration omits `Supercell` again.
Existing nonzero cell vectors retain their periodic meaning (including the
one- and two-vector format emitted by the CFG writer).

For example, with `elements: [H, O]`, an isolated water geometry is:

```text
BEGIN_CFG
 Size
 3
 AtomData: id type cartes_x cartes_y cartes_z
 1 1 0.000 0.000 0.000
 2 0 0.957 0.000 0.000
 3 0 -0.240 0.927 0.000
END_CFG
```

This geometry-only example is suitable for prediction. For training, add the
quantum calculation's `Energy` label and/or `fx fy fz` columns. The same input
works for Static and Effect models; no new training option is required.

Energy and forces need no volume and can be evaluated and fitted for molecules.
`PlusStress:` is the existing internal energy-valued tensor convention; it can
be retained for a molecule. It is not a Cauchy stress in pressure units. The
legacy `Virial:` input conversion divides by cell volume, and therefore requires
a nonzero-volume cell. Likewise `Configuration::virial()` requires volume and
rejects a molecule instead of producing infinities. Fractional coordinates
still require a nonsingular cell. A malformed explicit `Supercell` is an error;
it is never silently replaced by a molecular configuration.

The small parser regression can be run independently of a model build:

```sh
mkdir -p .local_tmp/boxless-check
c++ -std=c++11 -I. native_tests/boxless_cfg_check.cpp src/configuration.cpp src/neighborhoods.cpp src/common/utils.cpp -o .local_tmp/boxless-check/check
.local_tmp/boxless-check/check .local_tmp/boxless-check
```
