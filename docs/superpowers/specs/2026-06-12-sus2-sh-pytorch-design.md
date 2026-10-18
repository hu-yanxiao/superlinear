# SUS2-SH-PyTorch Design

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

Date: 2026-06-12

## Purpose

`SUS2-SH-PyTorch` is a new private training and research repository for the SUS2-SH model family. It is separate from `PySUS2SH`, which remains the Python application interface. The first implementation phase is C++ parity, not training.

The phase-one goal is:

1. Load an existing C++ SUS2-SH `.sus2` model.
2. Load a CFG structure set through ASE-compatible parsing.
3. Reproduce C++ `calc-efs` energy, force, and stress for the same model and structure.
4. Use PyTorch autograd for force and stress derivatives where practical.

Training with Adam, LBFGS, batching, and `.sus2` export are later phases. They should be built only after parity is stable.

## C++ Reference Branch

The phase-one C++ reference is the SUS2-SH `<historical-branch>` branch, not `main` and not the older `developer` branch.

The current reference commit is:

```text
103cd94c01542ac19ad645422eb46dc1917e9bd8
```

The local reference worktree is:

```text
/Users/hu-yanxiao/Projects/SUS2MLIP/.local_tmp/sus2-sh-developer-tanh
```

That path is a SUS2-SH worktree despite being located under the local `SUS2MLIP/.local_tmp` directory. The repository remote is:

```text
git@github.com:hu-yanxiao/SUS2-SH.git
```

The server reference tree is:

```text
<historical-path>
```

Any parity assumptions for `mu-body-order` or double gate must be taken from this branch. The `main` branch must not be used as the mathematical source for the new gate mode.

## Golden Case

The primary parity case is the latest double-gate mu-body-order model:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/double/ini.sus2
<historical-path>
```

This case covers:

- SUS2-SH spherical-harmonic basis.
- `mu-body-order` gate mode.
- `two_layer_gate_site_mode = double`.
- Shared gate weights grouped by exact body order.
- Energy, force, and stress targets.
- The current stress sign convention used by the C++ code.

Regression cases should include plain SH and single-gate mu-body-order models from the same `<historical-branch>` line after the golden case is working.

## Repository Boundary

The new repository name is:

```text
SUS2-SH-PyTorch
```

Initial visibility is private.

The repository should not reuse or modify the local `PySUS2SH` project. Code can inspect the SUS2-SH `<historical-branch>` branch for formulas and parameter layout, but it should not depend on building the C++ source as a Python extension in phase one.

## Phase-One Scope

Phase one implements a parity runner:

```text
sus2shpt parity --model ini.sus2 --cfg train_one.cfg --cpp-mlp /path/to/mlp-sus2
```

The command should:

1. Read model metadata and parameters from `.sus2`.
2. Read one or more structures from CFG.
3. Evaluate PyTorch energy, forces, and stress.
4. Call C++ `mlp-sus2 calc-efs` as the oracle.
5. Write a parity report with max abs error, RMSE, and relative error for energy, force components, and stress components.

Phase one does not implement:

- Dataset-scale training.
- Adam/LBFGS optimizer loops.
- `.sus2` export of updated parameters.
- Python `init-sh`.
- GPU performance optimization beyond keeping tensor shapes compatible with later batching.

## Model Coverage

The PyTorch model must support the latest C++ math used by the `<historical-branch>` golden case.

For each atom pair edge `i <- j`, the SH moment channel uses radial basis values and real spherical harmonics:

```latex
B_{i,\mu m} = \sum_{j \in N(i)} T_{ij,\mu} R_{\mu}(r_{ij}) Y_{lm}(\hat r_{ij})
```

where `mu` is the internal `(l, k)` channel. Without a gate, the edge type factor is:

```latex
T_{ij,\mu}=t_{z_i}t_{z_j}
```

For the current mu-body-order gate:

```latex
h_{a,\mu} =
\sum_{q: body\_order(s_q)=K_\mu} w_q s_q(a),
\qquad K_\mu = k_{\mu,\mathrm{internal}} + 2
```

The C++ internal `k` is zero-based, so the selected scalar body order is `k_internal + 2`.

Single neighbor gate:

```latex
g_{j,\mu} = 1 + A \tanh(a_{z_j,\mu} h_{j,\mu})
```

Double gate:

```latex
g_{ij,\mu} =
\left[1 + A \tanh(a_{z_j,\mu} h_{j,\mu})\right]
\left[1 + A \tanh(a_{z_i,\mu} h_{i,\mu})\right]
```

The effective edge factor in double mode is:

```latex
T_{ij,\mu}=t_{z_i}t_{z_j}g_{ij,\mu}
```

The implementation must preserve exact body-order selection. It must not use `<= body_order` for the mu gate buckets.

## Stress Definition

Stress parity follows the C++ convention, not a new PyTorch convention.

The C++ finite-difference checker compares lattice derivatives as:

```latex
\frac{\partial E}{\partial L} = -L^{-T} S
```

where `S` is the stored C++ stress tensor. Phase one should implement the same sign and tensor convention, then verify it against C++ `calc-efs` and finite differences.

If PyTorch computes virial-like quantities internally, the final reported tensor must be converted to the C++ stored stress convention before comparison.

## Architecture

The package layout should keep parsing, math, and parity reporting separate:

```text
sus2shpt/
  cli.py
  io/
    cfg.py
    mtp.py
  model/
    radial.py
    spherical_harmonics.py
    sh_products.py
    gate.py
    sus2_sh.py
  parity/
    cpp_oracle.py
    metrics.py
    report.py
  tests/
```

### IO Layer

`io.sus2` parses enough of the `.sus2` file to reconstruct the exact C++ model:

- Species count and element mapping.
- `l_max`, `k_max`, `body_order`, body-l limits.
- Radial basis type and coefficients.
- SH product program or scalar basis mapping.
- Linear coefficients and nonlinear coefficients.
- Gate metadata:
  - `two_layer_gate_enabled`.
  - `two_layer_gate_mode = mu-body-order`.
  - `two_layer_gate_site_mode = neighbor | double`.
  - `two_layer_gate_tanh_amplitude`.
  - shared-radial gate coefficients.
  - additive coefficients.
  - scalar indices and body-order bucket metadata.
  - shared gate weights.

`io.cfg` should read MLIP CFG robustly and expose an ASE-compatible representation. ASE can be used for file handling, but the project should keep a small CFG adapter so the exact C++ stress, force, and species fields remain accessible.

### Model Layer

The model layer should be pure PyTorch and deterministic for parity:

- Use `torch.float64` by default.
- Avoid stochastic operations.
- Keep tensors on CPU first.
- Add CUDA only after CPU parity passes.
- Keep parameter tensors in one-to-one correspondence with `.sus2` parameter blocks.

The SH product contraction should be represented as an explicit computation graph that matches the C++ scalar basis order. This is more important than speed in phase one.

### Autograd Derivatives

Forces:

```latex
F_{ia} = -\frac{\partial E}{\partial R_{ia}}
```

Stress should be computed through a differentiable lattice deformation path. The parity runner should use the same deformation convention as the C++ finite-difference checker:

```text
plus.Deform(plus.lattice.inverse() * deformed_lattice)
```

The PyTorch stress path should be validated by both:

1. Direct comparison to C++ stored stress.
2. Energy finite differences with lattice perturbations.

## Data Flow

Phase-one parity data flow:

```text
.sus2 -> MTPModelSpec -> SUS2SHPyTorch
.cfg -> StructureBatch -> positions/lattice/species tensors
SUS2SHPyTorch -> E
autograd(E, positions) -> F
autograd(E, lattice_deformation) -> stress
C++ calc-efs -> oracle E/F/stress
metrics -> parity report
```

Batching should be represented at the API boundary but not optimized initially. A simple list of structures is acceptable for phase one.

## Tests

Required phase-one tests:

1. `.sus2` parser smoke test on the golden `ini.sus2`.
2. CFG parser smoke test on `train_one.cfg`.
3. Plain SH energy parity on a small known model.
4. Single-gate energy/force/stress parity from the same `<historical-branch>` branch.
5. Double-gate energy/force/stress parity on the golden case.
6. Stress finite-difference sign check.
7. Gate exact body-order bucket check:
   - `k_internal = 0` uses body order 2 scalars.
   - `k_internal = 1` uses body order 3 scalars.
   - `k_internal = 2` uses body order 4 scalars.
   - `k_internal = 3` uses body order 5 scalars.

Initial tolerances should be conservative:

```text
energy abs error: <= 1e-8 eV for one structure if all operations match
force abs error: <= 1e-6 eV/A initially
stress abs error: <= 1e-6 in C++ stored stress units initially
```

If exact parity is blocked by known radial or spherical-harmonic implementation differences, the report must identify the first mismatching layer before widening tolerances.

## Server Environment

Development and verification should run on the phy-weigw server for parity with the C++ binary.

The C++ oracle binary for the current gate branch is:

```text
<historical-path>
```

The PyTorch environment can be a dedicated conda environment, for example:

```text
sus2-sh-pytorch
```

Expected packages:

- Python 3.11 or 3.12.
- PyTorch with CUDA support if available.
- ASE.
- NumPy.
- PyYAML or TOML support for reports/configs.
- pytest.

Phase one should work on CPU before relying on CUDA.

## Later Phases

After phase-one parity:

1. Add Python `init-sh` reconstruction.
2. Add mini-batch training with Adam.
3. Add optional LBFGS or hybrid Adam-to-LBFGS training.
4. Add exact `.sus2` export of the original parameterization.
5. Add GPU batching and memory-aware neighbor handling.
6. Add parity tests for exported `.sus2` files by loading them back into C++.

## Risks

Main risks:

- C++ scalar basis ordering is subtle; the parser must preserve it exactly.
- Stress sign mistakes are easy; the C++ finite-difference convention is the source of truth.
- Gate body-order mapping must be exact, not cumulative.
- Double-gate derivatives are more complex, so parity should compare forward energy first, then force, then stress.
- Radial basis variants such as `RBLaguerre_log1p` need exact C++ formula parity.

Risk mitigation:

- Build layer-by-layer parity tests.
- Keep C++ oracle calls available in the parity runner.
- Use `float64` throughout phase one.
- Avoid optimizing tensor layout until numerical parity is established.

## Acceptance Criteria

Phase one is complete when:

1. `SUS2-SH-PyTorch` exists as a private repository.
2. The golden double-gate case loads without manual model edits.
3. PyTorch energy, force, and stress match C++ `calc-efs` within the agreed tolerances.
4. The parity report records command lines, model path, cfg path, C++ binary path, C++ reference commit, and error metrics.
5. Plain SH and single-gate regression cases from the same C++ reference branch also pass.

