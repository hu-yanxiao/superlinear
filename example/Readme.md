# Carbon diamond: YAML training example

The supplied data are **carbon diamond (C), not silicon**. CFG type **0 = C**.

| File | Structures | Atoms per structure |
|---|---:|---:|
| `train.cfg` | 200 | 64 |
| `valid.cfg` | 114 | 64 |

The reference diamond lattice constant is 3.56054 Å.

## Train

From the repository root, choose one YAML:

```bash
# Static radial: L3/K4/O2
./bin/mlp-sus2 train example/train-static.yaml --check-config
./bin/mlp-sus2 train example/train-static.yaml > static.log 2>&1

# Coordination effect: L3/K4/R2/O2
./bin/mlp-sus2 train example/train-effect.yaml --check-config
./bin/mlp-sus2 train example/train-effect.yaml > effect.log 2>&1

# MPI alternative:
# mpirun -n 24 ./bin/mlp-sus2 train example/train-effect.yaml > effect.log 2>&1
```

[Static YAML](train-static.yaml) · [Effect YAML](train-effect.yaml)

Both train from scratch for at most **100 BFGS steps** and use E/F/virial
weights **1 / 0.01 / 0.001**. Paths resolve relative to the YAML file.
The saved `untrained.sus2` is not used by these fresh-training templates.

## Evaluate

Each run prints its unique output directory under `example/runs/`.
Replace `MODEL` with that run's `final.sus2`:

```bash
./bin/mlp-sus2 calc-errors MODEL example/train.cfg
./bin/mlp-sus2 calc-errors MODEL example/valid.cfg
./bin/mlp-sus2 calc-efs MODEL example/valid.cfg valid.pred.cfg
```

[Installation and usage](../README.md)
