# SUS2-SH Residual Staged Two-Layer Implementation Plan

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in SUS2-SH residual two-layer mode with \(E=E_0+E_1\), direct neighbor gate scale, and staged A/B/C training controls.

**Architecture:** Keep the legacy `--two-layer-gate` path unchanged. The residual mode adds a first-layer E0 scalar readout over unmodulated SH scalars and keeps the existing final-layer scalar readout as E1. The direct gate scale is explicit model metadata and defaults to legacy `1+f` for old models; residual models use `f=b+\sum_q w_q\Phi_q^{(0)}` with fixed bias initialized to 1 so the E1 linear residual solve has nonzero columns at B-stage start.

**Tech Stack:** C++ SUS2-SH developer code, existing MLIP trainer, BLAS/MPI linear solve, shell dev tests.

---

### Task 1: Model Metadata And Layout

**Files:**
- Modify: `dev_src/mtpr.h`
- Modify: `dev_src/mtpr.cpp`
- Modify: `dev_src/sh_model_init.cpp`
- Modify: `dev_src/mlp/mtpr_train.cpp`
- Modify: `src/mlp/mlp_commands.cpp`

- [x] Add `two_layer_residual_enabled_`, `two_layer_gate_scale_mode_`, and `two_layer_residual_e0_coeffs_` to `MLMTPR`.
- [x] Add offset helpers: residual E0 coeff count/offset, linear equation count, and linear storage mapping.
- [x] Extend load/save/init-sh metadata:
  - `two_layer_residual_enabled = true`
  - `two_layer_gate_scale_mode = direct`
  - `two_layer_gate_bias = 1`
  - `two_layer_residual_e0_coeff_count = alpha_scalar_moments`
  - `two_layer_residual_e0_coeffs = {...}`
- [x] Keep old files without mode as legacy `1+f`.

### Task 2: Forward E0 + E1

**Files:**
- Modify: `dev_src/mtpr_sh.cpp`
- Modify: `dev_src/mtpr.cpp`

- [x] Make `PrepareTwoLayerGateValues()` support `legacy` and `direct` modes.
- [x] Add residual site energy path:
  - compute E0 from unmodulated one-cutoff scalars and `two_layer_residual_e0_coeffs_`;
  - compute E1 from gated final scalars and existing `linear_coeffs`;
  - preserve gate adjoint only for E1.
- [x] Extend `CalcEFSComponents()` / `CalcEComponents()` to emit linear columns for species, E1 scalars, and E0 scalars.

### Task 3: Linear Solves And Staged Training

**Files:**
- Modify: `dev_src/mtpr_trainer.h`
- Modify: `dev_src/mtpr_trainer.cpp`
- Modify: `dev_src/mlp/mtpr_train.cpp`

- [x] Add trainer stage enum: none/full, E0, E1, all.
- [x] Let `TrainLinear()` solve only the active linear block and preserve frozen blocks.
- [x] Add CLI controls:
  - `--two-layer-residual-staged`
  - `--stage-a-steps=<int>`
  - `--stage-b-steps=<int>`
  - `--stage-c-steps=<int>`
- [x] Run phases A, B, C sequentially; each phase may exit early by existing convergence criteria.

### Task 4: Tests

**Files:**
- Create: `dev_test/sh_two_layer_residual_init_check.sh`
- Create: `dev_test/sh_two_layer_residual_forward_check.sh`
- Create: `dev_test/sh_two_layer_residual_staged_train_check.sh`
- Create: `dev_test/sh_two_layer_residual_force_fd_check.sh`
- Create: `dev_test/sh_two_layer_residual_loss_gradient_check.sh`
- Create: `dev_test/sh_two_layer_residual_readout_check.sh`

- [x] Test residual init writes metadata and round-trips through save/load.
- [x] Test direct gate mode does not change legacy zero-gate behavior unless residual mode is explicitly enabled.
- [x] Test a tiny staged train updates E0 in A and E1 in B without clobbering frozen blocks.
- [x] Test residual EFS force finite differences.
- [x] Test residual loss-gradient finite differences, including staged-offset linear coefficient mapping.
- [x] Test residual direct E/F readout against the current linear component columns with non-unit species coefficients.
- [x] Run existing two-layer gate tests to protect legacy mode.

### Task 5: Server Verification

**Files:**
- Modify: `docs/sus2-sh-development.md`

- [x] Sync to `<historical-path>`.
- [x] Build on server.
- [x] Run focused dev tests on server.
- [x] Run 500-sample 100-step comparison under `/work/phy-weigw/hyx/200w/5.31-new-44421/`.
- [x] Record commands, paths, and metrics.
