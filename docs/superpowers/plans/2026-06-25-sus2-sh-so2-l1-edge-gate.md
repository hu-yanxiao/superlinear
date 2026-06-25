# SUS2-SH SO2 L1 Edge Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first exact edge-projected `L=1` SUS2-SH gate with zero-weight compatibility, analytic coordinate/parameter derivatives, rotational-invariance checks, and server-side profile comparison.

**Architecture:** Reuse the existing two-layer scalar gate and final SH evaluator. Add a separate edge L1 metadata/weight block, build raw first-layer `l=1` source moments per atom, combine them with edge `Y_1(rhat_ij)` inside the existing pair gate buffer path, and add reverse-mode adjoints for both direct edge projection and first-layer `H` construction.

**Tech Stack:** C++11, existing SUS2-SH `MLMTPR`, real spherical harmonics from `dev_src/mtpr_sh.cpp`, existing shell regression tests in `dev_test`, server build/test path `/work/phy-weigw/20260321_Test/SUS2-SH-so2-l1-gate-work-codex`, benchmark path `/work/phy-weigw/hyx/xxx-b/so2-gate/`.

---

## File Structure

- Modify `dev_src/sh_model_init.cpp`: parse `--two-layer-gate-edge-l1`, validate `l-max >= 1` and neighbor site mode, and write zero-initialized metadata.
- Modify `src/mlp/mlp_commands.cpp`: include the new init option in SH option detection and help text.
- Modify `dev_src/mlp/dev_mlp_commands.cpp`: include the new init option in dev help text and add coefficient-name reporting for edge L1 weights.
- Modify `dev_src/mtpr.h`: add metadata fields, value/adjoint caches, scratch buffers, offsets, accessors, and declarations.
- Modify `dev_src/mtpr.cpp`: load/save metadata, coefficient layout, compatibility defaults, source `mu` construction, value-cache orchestration, force-chain orchestration, and coefficient gradient accumulation.
- Modify `dev_src/mtpr_sh.cpp`: compute raw `l=1` gate vectors, compute edge signals in forward paths, disable atom-keyed tanh cache for edge signals, accumulate force/reverse/tangent terms, and support edge-cache/site-cache paths.
- Add `dev_test/sh_two_layer_gate_edge_l1_init_check.sh`: metadata, rejection, and old-model load compatibility.
- Add `dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh`: edge-L1-zero output/gradient compatibility.
- Add `dev_test/sh_two_layer_gate_edge_l1_forward_check.sh`: nonzero forward sensitivity on asymmetric geometry.
- Add `dev_test/sh_two_layer_gate_edge_l1_fd_check.sh`: nonzero force/stress finite differences.
- Add `dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh`: coefficient finite differences covering edge L1 weights and existing gate parameters.
- Add `dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh`: nonzero rotated-energy/force invariance.
- Add `dev_test/sh_two_layer_gate_edge_l1_server_profile_notes.sh`: records the exact remote commands used for off/on profile runs.

## Task 1: Failing Metadata And Compatibility Tests

**Files:**
- Create: `dev_test/sh_two_layer_gate_edge_l1_init_check.sh`
- Create: `dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh`

- [ ] **Step 1: Create the failing init/metadata test**

Add executable `dev_test/sh_two_layer_gate_edge_l1_init_check.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_init"
mkdir -p "$tmp_dir"
edge_model="$tmp_dir/edge_l1.mtp"
legacy_model="$tmp_dir/legacy_gate.mtp"
legacy_saved="$tmp_dir/legacy_gate.saved.mtp"
reject_log="$tmp_dir/reject_double.log"

./bin/mlp-sus2 init-sh "$edge_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-edge-l1 >/dev/null

grep -q "two_layer_gate_enabled = true" "$edge_model"
grep -q "two_layer_gate_edge_l1_enabled = true" "$edge_model"
grep -q "two_layer_gate_edge_l1_source_mu_count = 3" "$edge_model"
grep -q "two_layer_gate_edge_l1_source_mu_indices = {1, 4, 7}" "$edge_model"
grep -q "two_layer_gate_edge_l1_weight_count = 27" "$edge_model"
grep -q "two_layer_gate_edge_l1_weights" "$edge_model"

if ./bin/mlp-sus2 init-sh "$tmp_dir/reject_double.mtp" \
  --species-count=1 \
  --l-max=1 \
  --k-max=2 \
  --body-order=3 \
  --body-l-max=1,1,1,1 \
  --radial-basis-size=3 \
  --cutoff=5.0 \
  --two-layer-gate \
  --two-layer-gate-site-mode=double \
  --two-layer-gate-edge-l1 >"$reject_log" 2>&1; then
  echo "edge L1 gate should reject double site mode in the first version" >&2
  exit 1
fi
grep -q "edge L1" "$reject_log"

./bin/mlp-sus2 init-sh "$legacy_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate >/dev/null

./bin/mlp-sus2 train "$legacy_model" example/train.cfg \
  --max-iter=0 \
  --trained-pot-name="$legacy_saved" \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >/dev/null

grep -q "two_layer_gate_enabled = true" "$legacy_saved"
if grep -q "two_layer_gate_edge_l1_enabled = true" "$legacy_saved"; then
  echo "legacy gate save unexpectedly enabled edge L1 metadata" >&2
  exit 1
fi
```

- [ ] **Step 2: Create the failing zero-compatibility test**

Add executable `dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_zero"
mkdir -p "$tmp_dir"
base="$tmp_dir/base_gate.mtp"
edge="$tmp_dir/edge_gate_zero.mtp"
base_pred="$tmp_dir/base_pred.cfg"
edge_pred="$tmp_dir/edge_pred.cfg"

common_opts=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --write-sh-scalar-info
  --two-layer-gate
)

./bin/mlp-sus2 init-sh "$base" "${common_opts[@]}" >/dev/null
./bin/mlp-sus2 init-sh "$edge" "${common_opts[@]}" --two-layer-gate-edge-l1 >/dev/null
./bin/mlp-sus2 calc-efs "$base" example/train.cfg "$base_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$edge" example/train.cfg "$edge_pred" >/dev/null

python3 - "$base_pred" "$edge_pred" <<'PY'
import math
import sys

def values(path):
    out = []
    fcols = None
    scols = None
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("Energy"):
                out.append(float(line.split()[1]))
            elif line.startswith("PlusStress:"):
                out.extend(float(x) for x in line.split()[1:])
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                fcols = [header.index(name) for name in ("fx", "fy", "fz")]
            elif fcols is not None and line[:1].isdigit():
                parts = line.split()
                out.extend(float(parts[idx]) for idx in fcols)
    return out

a = values(sys.argv[1])
b = values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"value count mismatch: {len(a)} vs {len(b)}")
err = max(abs(x - y) for x, y in zip(a, b))
if not math.isfinite(err) or err > 1.0e-10:
    raise SystemExit(f"zero edge-L1 gate predictions differ: {err}")
print(f"zero edge-L1 compatibility OK: max_err={err:.3e}")
PY
```

- [ ] **Step 3: Verify both tests fail before production code**

Run:

```bash
chmod +x dev_test/sh_two_layer_gate_edge_l1_init_check.sh \
  dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_init_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh
```

Expected: both fail because `--two-layer-gate-edge-l1` is not recognized and no
edge L1 metadata exists.

## Task 2: Metadata, Coefficient Layout, And Save/Load

**Files:**
- Modify: `dev_src/sh_model_init.cpp`
- Modify: `src/mlp/mlp_commands.cpp`
- Modify: `dev_src/mlp/dev_mlp_commands.cpp`
- Modify: `dev_src/mtpr.h`
- Modify: `dev_src/mtpr.cpp`
- Test: `dev_test/sh_two_layer_gate_edge_l1_init_check.sh`

- [ ] **Step 1: Add init option parsing and validation**

Use this option name:

```text
--two-layer-gate-edge-l1
```

In `dev_src/sh_model_init.cpp`, set:

```cpp
const bool two_layer_gate_edge_l1 = HasOpt(opts, "two-layer-gate-edge-l1");
```

Validation:

```cpp
if (two_layer_gate_edge_l1 && !two_layer_gate)
    ERROR("--two-layer-gate-edge-l1 requires --two-layer-gate.");
if (two_layer_gate_edge_l1 && lmax < 1)
    ERROR("--two-layer-gate-edge-l1 requires --l-max >= 1.");
if (two_layer_gate_edge_l1 && two_layer_gate_site_mode != "neighbor")
    ERROR("--two-layer-gate-edge-l1 first version supports neighbor site mode only.");
```

Add the option to `HasSphericalHarmonicInitOptions()` and both public/dev help
texts.

- [ ] **Step 2: Write metadata from `init-sh`**

After existing body-mix weights in `dev_src/sh_model_init.cpp`, write:

```cpp
if (two_layer_gate_edge_l1) {
    const int edge_l1_source_count = kmax;
    const int edge_l1_weight_count = radial_func_count * edge_l1_source_count;
    ofs << "two_layer_gate_edge_l1_enabled = true\n";
    ofs << "two_layer_gate_edge_l1_source_mu_count = "
        << edge_l1_source_count << "\n";
    ofs << "two_layer_gate_edge_l1_source_mu_indices = {";
    for (int k = 0; k < edge_l1_source_count; ++k) {
        if (k != 0)
            ofs << ", ";
        ofs << k * (lmax + 1) + 1;
    }
    ofs << "}\n";
    ofs << "two_layer_gate_edge_l1_weight_count = "
        << edge_l1_weight_count << "\n";
    ofs << "two_layer_gate_edge_l1_weights = {";
    for (int i = 0; i < edge_l1_weight_count; ++i) {
        if (i != 0)
            ofs << ", ";
        ofs << 0.0;
    }
    ofs << "}\n";
}
```

- [ ] **Step 3: Add `MLMTPR` fields and offsets**

Add fields:

```cpp
bool two_layer_gate_edge_l1_enabled_ = false;
std::vector<int> two_layer_gate_edge_l1_source_mu_indices_;
std::vector<double> two_layer_gate_edge_l1_weights_;
std::vector<double> two_layer_gate_edge_l1_values_;
std::vector<double> two_layer_gate_edge_l1_adjoints_;
std::vector<double> two_layer_gate_edge_l1_signal_buffer_;
std::vector<double> two_layer_gate_edge_l1_chi_buffer_;
std::vector<double> two_layer_gate_edge_l1_chi_adj_buffer_;
std::vector<double>* active_two_layer_gate_edge_l1_adjoints_ = nullptr;
const std::vector<double>* active_two_layer_gate_edge_l1_values_ = nullptr;
```

Add helpers:

```cpp
bool TwoLayerGateEdgeL1Enabled() const;
bool HasNonzeroTwoLayerGateEdgeL1Weights() const;
int TwoLayerGateEdgeL1SourceCount() const;
int TwoLayerGateEdgeL1WeightCount() const;
int TwoLayerGateEdgeL1WeightOffset() const;
int TwoLayerGateEdgeL1WeightIndex(int mu, int source) const;
void BuildTwoLayerGateEdgeL1Sources();
```

Set `TwoLayerResidualE0CoeffOffset()` to use:

```cpp
return TwoLayerGateEdgeL1WeightOffset() + TwoLayerGateEdgeL1WeightCount();
```

- [ ] **Step 4: Add load/save with old-model defaults**

In `Load()`, clear all edge L1 fields before parsing. After existing
`two_layer_gate_body_mix_weights` parsing, accept optional
`two_layer_gate_edge_l1_enabled`. If the next token is not the edge L1 token,
leave edge L1 disabled and continue parsing the old-format model.

For enabled models, validate source count, source indices, and weight count:

```cpp
if (two_layer_gate_edge_l1_enabled_) {
    if (sh_l_max_ < 1)
        ERROR("SUS2-SH edge L1 gate requires sh_l_max >= 1");
    if (two_layer_gate_site_mode_ != "neighbor")
        ERROR("SUS2-SH edge L1 gate first version supports neighbor site mode only");
    BuildTwoLayerGateEdgeL1Sources();
    if (static_cast<int>(two_layer_gate_edge_l1_weights_.size())
        != TwoLayerGateEdgeL1WeightCount())
        ERROR("SUS2-SH edge L1 gate weight count is inconsistent");
}
```

In `Save()`, write the edge L1 block only when enabled.

- [ ] **Step 5: Run metadata test**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_init_check.sh
```

Expected: PASS.

## Task 3: Edge L1 Values And Zero-Compatible Forward Path

**Files:**
- Modify: `dev_src/mtpr.h`
- Modify: `dev_src/mtpr.cpp`
- Modify: `dev_src/mtpr_sh.cpp`
- Test: `dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh`
- Create: `dev_test/sh_two_layer_gate_edge_l1_forward_check.sh`

- [ ] **Step 1: Add a failing nonzero forward-sensitivity test**

Create `dev_test/sh_two_layer_gate_edge_l1_forward_check.sh` that initializes an
edge L1 model, edits `two_layer_gate_edge_l1_weights` to deterministic nonzero
values, and verifies `calc-efs` changes relative to the zero-weight model.

The Python edit block must replace only edge L1 weights:

```python
match = re.search(r"two_layer_gate_edge_l1_weights = \{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("missing two_layer_gate_edge_l1_weights")
weights = [float(x) for x in re.findall(r"[-+0-9.eE]+", match.group(1))]
for i in range(len(weights)):
    weights[i] = 0.02 * ((i % 5) - 2)
replacement = "two_layer_gate_edge_l1_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
text = text[:match.start()] + replacement + text[match.end():]
```

Run it before implementation. Expected: FAIL because nonzero edge L1 weights do
not affect the gate yet.

- [ ] **Step 2: Compute raw `l=1` first-layer values**

Add `CalcTwoLayerGateEdgeL1ValuesOnly(const Neighborhood& nbh, std::vector<double>& out, int cache_atom_index)` in `dev_src/mtpr_sh.cpp`.

Implementation rules:

```text
out[source * 3 + (m + 1)] accumulates M^{(0)}_{nu,1m}
source mu is two_layer_gate_edge_l1_source_mu_indices_[source]
radial values use the same gate radial path as scalar gate values
the real SH component index is 1 * 1 + m + 1
```

Call it from `PrepareTwoLayerGateValues()` after scalar gate values are ready:

```cpp
two_layer_gate_edge_l1_values_.assign(
    static_cast<size_t>(cfg.size()) * TwoLayerGateEdgeL1SourceCount() * 3, 0.0);
for (int ind = 0; ind < cfg.size(); ++ind) {
    double* dst = two_layer_gate_edge_l1_values_.data()
        + static_cast<size_t>(ind) * TwoLayerGateEdgeL1SourceCount() * 3;
    CalcTwoLayerGateEdgeL1ValuesOnly(neighborhoods[ind],
                                     two_layer_gate_edge_l1_value_buffer_,
                                     ind);
    std::copy(two_layer_gate_edge_l1_value_buffer_.begin(),
              two_layer_gate_edge_l1_value_buffer_.end(), dst);
}
```

- [ ] **Step 3: Add edge-specific signal formation**

Add helper:

```cpp
const double* PrepareTwoLayerGateEdgeL1NeighborSignals(
    const Neighborhood& nbh,
    int neighbor_index,
    const double* sh_values_use,
    const double* base_signal_by_mu);
```

The helper returns `base_signal_by_mu` when edge L1 is disabled or all edge L1
weights are zero. Otherwise it fills `two_layer_gate_edge_l1_signal_buffer_`:

```cpp
signal[mu] = base_signal_by_mu == nullptr ? 0.0 : base_signal_by_mu[mu];
chi[source] = sum_m H[atom_j, source, m] * Y_1m(edge i->j);
signal[mu] += sum_source W_edge_l1[mu, source] * chi[source];
```

When this helper returns the scratch signal, pass `neighbor_atom_index = -1` to
`PrepareTwoLayerGatePairMuBuffers()` so the atom-keyed tanh cache is bypassed.

- [ ] **Step 4: Run zero and forward tests**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_forward_check.sh
```

Expected: both PASS.

## Task 4: Coordinate Reverse And Force Finite Differences

**Files:**
- Modify: `dev_src/mtpr.h`
- Modify: `dev_src/mtpr.cpp`
- Modify: `dev_src/mtpr_sh.cpp`
- Create: `dev_test/sh_two_layer_gate_edge_l1_fd_check.sh`

- [ ] **Step 1: Add failing force/stress finite-difference test**

Create a shell script that builds a nonzero edge L1 model and runs:

```bash
./bin/mlp-sus2 check-efs-fd-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --displacement=1e-6 \
  --abs-tolerance=2e-5 \
  --rel-tolerance=2e-4
```

Run it before reverse implementation. Expected: FAIL because the direct
`dY_1(rhat_ij)` and first-layer `H` force-chain terms are missing.

- [ ] **Step 2: Accumulate direct edge-projection adjoints**

Inside `CalcSHSiteEnergyDers()`, after computing `gate_adjoint_by_mu` for an
edge, compute:

```cpp
bar_chi[source] = sum_mu gate_adjoint_by_mu[mu] * W_edge_l1[mu, source];
bar_H[source,m] += bar_chi[source] * Y_1m(edge);
buff_site_energy_ders_[j][a] += bar_chi[source] * H[source,m] * dY_1m(edge)/dx_a;
```

Use the already evaluated `sh_values_use` and `sh_ders_use` with `l=1` indices.
Accumulate `bar_H` into `active_two_layer_gate_edge_l1_adjoints_` for atom `j`.

- [ ] **Step 3: Propagate `H` adjoints through the first layer**

Add `AccumulateTwoLayerGateEdgeL1ForceChain(Configuration&, const Neighborhoods&)`.
For each atom `j`, seed the raw `l=1` source moments from
`two_layer_gate_edge_l1_adjoints_`, then reuse the same edge radial/SH
derivative convention as `CalcTwoLayerGateWeightedScalarDersForScalarSeeds()`.

Call it after `AccumulateTwoLayerGateForceChain()` in `CalcEFS()` and in any
configuration-level path that populates forces/stresses.

- [ ] **Step 4: Run finite-difference test**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_fd_check.sh
```

Expected: PASS.

## Task 5: Parameter Gradients

**Files:**
- Modify: `dev_src/mtpr.cpp`
- Modify: `dev_src/mtpr_sh.cpp`
- Create: `dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh`

- [ ] **Step 1: Add failing loss-gradient test**

Create a script that builds a nonzero edge L1 model and runs
`check-loss-gradient-dev` over the edge L1 weight range:

```bash
edge_offset=$(python3 - "$nonzero_model" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
radial = int(re.search(r"radial_funcs_count = (\d+)", text).group(1))
source = int(re.search(r"two_layer_gate_edge_l1_source_mu_count = (\d+)", text).group(1))
print(radial * source)
PY
)
```

Use the dev command's coefficient-name output after implementation to locate
the exact coefficient range. The initial failing run can use a conservative
range that includes the new block once offsets are exposed.

- [ ] **Step 2: Accumulate direct edge L1 weight gradients**

After final edge reverse computes `gate_adjoint_by_mu` and `chi[source]`, add:

```cpp
out_grads_accumulator[TwoLayerGateEdgeL1WeightOffset()
    + TwoLayerGateEdgeL1WeightIndex(mu, source)] += gate_adjoint_by_mu[mu] * chi[source];
```

Use BLAS only if an edge-major `gate_adjoint_by_mu` cache is introduced. The
first version can accumulate in the hot edge loop because `source_count = kmax`
is small.

- [ ] **Step 3: Add tangent support**

Extend the gate-chain tangent path so the final-layer gate tangent on an edge is:

```text
delta s_mu(i,j) =
delta s0_mu(j)
+ sum_source delta W[mu,source] chi_source(i,j)
+ sum_source W[mu,source] delta chi_source(i,j)
```

For base radial/gate radial parameter checks:

```text
delta chi_source =
sum_m delta H[source,m](j) Y_1m(edge)
+ sum_m H[source,m](j) delta Y_1m(edge)
```

The `delta Y_1m` part is nonzero for final-layer radial-scaling parameters only
through coordinate tangents; coefficient-only checks for edge L1 weights use the
direct `delta W * chi` term.

- [ ] **Step 4: Run loss-gradient tests**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_mu_scalar_full_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_double_mode_check.sh
```

Expected: PASS for edge L1 and unchanged existing gate gradient checks.

## Task 6: Rotation Invariance

**Files:**
- Create: `dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh`

- [ ] **Step 1: Add failing rotation check**

Create a script that:

1. Builds a nonzero edge L1 model.
2. Copies one small CFG from `example/train.cfg`.
3. Rotates positions, lattice, forces, and stresses by a fixed orthogonal
   matrix.
4. Runs `calc-efs` on original and rotated CFG.
5. Verifies equal energies and rotated forces within `2e-8`.

Use this fixed rotation matrix:

```python
R = [
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
]
```

- [ ] **Step 2: Run rotation check**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh
```

Expected: PASS. A sign or real-SH normalization error usually appears here even
when zero compatibility passes.

## Task 7: Server Build, Math Tests, And Profile

**Files:**
- Create: `dev_test/sh_two_layer_gate_edge_l1_server_profile_notes.sh`

- [ ] **Step 1: Sync isolated server path**

Use the documented SSH route and create/update:

```bash
/work/phy-weigw/20260321_Test/SUS2-SH-so2-l1-gate-work-codex
```

The binary under test is:

```bash
/work/phy-weigw/20260321_Test/SUS2-SH-so2-l1-gate-work-codex/bin/mlp-sus2
```

- [ ] **Step 2: Build on the server**

Run from the isolated server source path:

```bash
make -j
```

Expected: `bin/mlp-sus2` exists and reports the new CLI option through help or
successful `init-sh`.

- [ ] **Step 3: Run server correctness checks**

Run:

```bash
bash dev_test/sh_two_layer_gate_edge_l1_init_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_forward_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_fd_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh
```

Expected: all PASS on the server binary.

- [ ] **Step 4: Profile in `/work/phy-weigw/hyx/xxx-b/so2-gate/`**

Use the existing scripts and CLI settings in:

```bash
/work/phy-weigw/hyx/xxx-b/so2-gate/
```

Run two variants that differ only by `--two-layer-gate-edge-l1` and the
corresponding nonzero edge L1 metadata. Set the existing profile environment:

```bash
SUS2_SH_TWO_LAYER_FORWARD_PROFILE=1
SUS2_SH_TWO_LAYER_GRAD_PROFILE=1
```

Record:

```text
command line
model metadata diff
total wall time
forward profile sections
gradient profile sections
call counts
speed ratio on/off
```

Store the exact commands in
`dev_test/sh_two_layer_gate_edge_l1_server_profile_notes.sh` and the raw server
logs under the benchmark directory.

## Self-Review Checklist

- Spec coverage: metadata, old-model compatibility, zero compatibility,
  forward sensitivity, force FD, parameter FD, rotation invariance, and profile
  comparison each map to a task.
- Exactness: the plan uses no pruning, low-rank factorization, quantization, or
  approximate communication.
- First-version boundary: `double` site mode is rejected for edge L1.
- Cache correctness: atom-keyed tanh cache is bypassed for edge-dependent gate
  signals.
- Gradient risk: parameter tangents are treated as edge-dependent, not copied
  from atom-major buffers.
