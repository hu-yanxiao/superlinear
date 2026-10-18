# SUS2.superlinear

**Machine-learning interatomic potentials based on MPF (Material Property Field).**

## Installation

```bash
# Load an Intel classic + MKL environment first.
git clone https://github.com/hu-yanxiao/superlinear.git
cd superlinear
./configure --compiler=intel --blas=mkl
make -j4 mlp
./bin/mlp-sus2 help
```

## Training

Choose one template, copy it to the repository root, and edit its data paths
and number of atomic types.

```bash
cp examples/train-static.yaml train.yaml
# Or: cp examples/train-effect.yaml train.yaml

./bin/mlp-sus2 train train.yaml --check-config
./bin/mlp-sus2 train train.yaml > train.log 2>&1

# With an MPI build, use this instead of the serial training command:
# mpirun -n 40 ./bin/mlp-sus2 train train.yaml > train.log 2>&1
```

### Static radial — L3/K4/O2

No coordination-effect layer: **K and O are used; R is absent**.

```yaml
schema_version: 1
model:
  kind: static-radial
  species_count: 2                 # Number of CFG types (0, 1, ...)
  l_max: 3                        # L
  radial_channels: 4              # K
  neighbor_cutoff: 6.5             # Angstrom
  radial:
    basis: chebyshev-sss
    basis_size: 10
    warp_sharing: K               # L / K / LK / pair
  interaction:
    channels: 2                   # O; 1 <= O <= K
    body_order: 4
    channel_coupling: cross        # Static mode uses cross
  readout:
    element_mode: per-element     # shared / per-element
training:
  profile: native-static-v1
  optimizer: bfgs
  iterations: 100
loss:
  energy_weight: 1
  force_weight: 0.01
  virial_weight: 0.001
data:
  train: data/train.cfg
  valid: data/val.cfg              # Remove if no validation set exists
```

### Coordination effect — L3/K4/R2/O2

Adds coordination-environment modulation. Inner and outer channel coupling
are independently selectable. `effect` is also accepted as a short kind name.

```yaml
schema_version: 1
model:
  kind: coordination-effect
  species_count: 2                 # Number of CFG types (0, 1, ...)
  l_max: 3                        # L
  radial_channels: 4              # K
  neighbor_cutoff: 6.5             # Angstrom
  radial:
    basis: chebyshev-sss
    basis_size: 10
    warp_sharing: K               # L / K / LK / pair
  coordination:
    channels: 2                   # R; 1 <= R <= K
    body_order: 4
    channel_coupling: cross        # cross / diagonal; inner graph
  interaction:
    channels: 2                   # O; 1 <= O <= K
    body_order: 4
    channel_coupling: diagonal     # cross / diagonal; outer graph
  readout:
    element_mode: per-element     # shared / per-element
training:
  profile: native-effect-v1
  optimizer: bfgs
  iterations: 100
loss:
  energy_weight: 1
  force_weight: 0.01
  virial_weight: 0.001
data:
  train: data/train.cfg
  valid: data/val.cfg              # Remove if no validation set exists
```

## Outputs and evaluation

No `directory` setting is needed. Each run automatically gets a new directory
under `runs/`, named from the model type and L/K/R/O with a unique run ID.
Static models omit R. The resolved directory is printed at startup.

Models and checkpoints are saved with the **`.sus2`** extension:

```text
runs/<your-run>/
  initial.sus2
  current.sus2
  best.sus2
  final.sus2
  optimizer.csv
```

```bash
./bin/mlp-sus2 calc-errors runs/<your-run>/final.sus2 data/val.cfg
./bin/mlp-sus2 calc-efs runs/<your-run>/final.sus2 data/val.cfg val.pred.cfg
```

Replace `<your-run>` with the generated directory printed in the log. Keep
prediction output separate from the original labels.

## Continue training

Remove the entire `model` group, add
`data.model: runs/<your-run>/current.sus2`. A new continuation directory is
generated automatically. Keep the matching training profile: `native-static-v1` or `native-effect-v1`.
Then run the same training command.

## Notes

- Paths are relative to the YAML file. Existing output files are not overwritten.
- Optional `output.directory` overrides the automatically generated directory.
- `species_count` is the number of CFG types. Alternatively, use
  `elements: [H, O]` in CFG type order.
- `iterations` is the BFGS iteration limit, **not epochs**.
- `--check-config` checks options only; it does not read data or checkpoints.
- The native YAML trainer runs on CPU in FP64. No Adam/GPU settings are implied.
