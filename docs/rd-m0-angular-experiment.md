# RD m0 angular backend experiment

This branch adds an optional C++ execution backend for the current
`radial-coefficient-gate` superlinear model. It does not add parameters or
change the saved model format. The production/default backend remains `ry`.

## Definition

Let Q_ij map the fixed unit z axis onto the unit edge direction rhat_ij.
In the existing real spherical-harmonic convention,

    Y_lm(rhat_ij) = D^l_m0(Q_ij) c_l,
    c_l = Y_l0(zhat) = sqrt((2l+1)/(4*pi)).

The experimental gate and outer primitive reductions are

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

## Implementation and derivatives

`dev_src/rd_m0_angular.h` independently evaluates the normalized real rotation
column using direction-cosine recurrences. It does not call the old spherical
harmonic evaluator or divide an already evaluated Y by c_l. There are no
`acos`, `atan2` or inverse transverse-radius factors at the poles.

For n = r/|r|, derivatives include the complete normalization chain rule:

    dD/dr_a = (dD/dn_a - n_a sum_b n_b dD/dn_b) / |r|.
    d(c_l sum_j R_j D_j)/dr = c_l sum_j (dR_j D_j + R_j dD_j).

Raw D/dD caches serve the experimental reductions. Physical Y/dY cache views
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
SUS2_SH_ANGULAR_BACKEND=ry /path/to/experimental/mlp-sus2 calc-efs model.mtp input.cfg ry.cfg --no-loss
SUS2_SH_ANGULAR_BACKEND=rd-m0 /path/to/experimental/mlp-sus2 calc-efs model.mtp input.cfg rd.cfg --no-loss
```

An unset variable selects `ry`; other values are rejected. RD adaptive setup
prints a `[RDM0_ANGULAR] backend=rd-m0` diagnostic. The same selection works for
`init-sh`, training and evaluation. No model conversion or reinitialization is
needed to switch an existing model between these mathematically equivalent
backends. Floating-point summation order differs, so bitwise equality or
identical long optimization trajectories is not promised.

`calc-efs --no-loss` is an additive diagnostic option on this branch, using
the existing lossless CFG serializer (17 significant digits). Default CFG
formatting is unchanged. Do not infer double-precision parity from ordinary
CFG files: those round forces to six and stresses to five decimal places.

This first experiment intentionally keeps physical angular cache views as well
as raw D/dD. The extra angular storage is H doubles per edge in values-only
mode or 4H doubles per edge with derivatives, H=(lmax+1)^2. Mathematical
equivalence alone does not imply a speedup; performance needs measurement.
PyTorch, GPUMD and production installed binaries are not changed by this branch.

## Validation

Compile and run on the server, not the developer laptop:

```sh
g++ -std=c++11 -O3 -Idev_src codex_tests/rd_m0_angular_check.cpp -o rd_m0_angular_check
./rd_m0_angular_check
python3 dev_test/rd_m0_backend_check.py /path/to/experimental/mlp-sus2
```

The standalone test covers independent analytic references, Cartesian
derivatives, exact/near poles, parity and rotations. The CLI regression checks
native model/backend parity, coordinate and mixed EFS parameter derivatives,
and model generation/training. See its output for the actual tested cases;
unsupported model angular orders are not counted as numerical passes.
