#!/usr/bin/env python3
"""Native executable regression; fixtures only, no production inputs or jobs.

Run in an initialized scientific Python environment on allocated CPU resources.
No third-party packages are required. Keep fixtures/logs for qualification.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "bin/mlp-sus2").resolve()
ROOT = Path(tempfile.mkdtemp(prefix="grouped-native-cli-"))
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_DYNAMIC="FALSE")
MPI = shlex.split(os.environ.get("GROUPED_TEST_MPI_COMMAND", ""))
CHECKS = 0


def run(label, args, failure=None, mpi=False):
    global CHECKS
    result = subprocess.run((MPI if mpi else []) + [str(BIN)] + list(map(str, args)),
                            cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=180)
    output = result.stdout + result.stderr
    (ROOT / (label + ".log")).write_text(output)
    if failure:
        assert result.returncode != 0 and failure in output, (label, result.returncode, output[-5000:])
    else:
        assert result.returncode == 0, (label, result.returncode, output[-5000:])
    CHECKS += 1
    return output


def yaml(name, text):
    path = ROOT / name
    path.write_text(text)
    return path


MODEL = """model:
  kind: coordination-adaptive
  elements: [H, O]
  l_max: 1
  radial_channels: 2
  neighbor_cutoff: 3.5
  radial:
    basis_size: 3
    warp_sharing: LK
  coordination:
    channels: 1
    body_order: 3
  interaction:
    channels: 1
    body_order: 3
"""
LEGACY = ["--superlinear-mode=radial-coefficient-gate", "--species-count=2", "--l-max=1", "--k-max=2",
          "--cutoff=3.5", "--radial-basis-size=3", "--two-layer-gate-k-rank=1",
          "--two-layer-gate-body-order=3", "--outer-k-rank=1", "--body-order=3"]
CFG = ROOT / "tiny.cfg"
CFG.write_text("""BEGIN_CFG
 Size
   3
 Supercell
   12 0 0
   0 12 0
   0 0 12
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1.17 1.39 1.52 0.031 -0.023 0.017
 2 1 2.63 2.11 2.74 -0.019 0.041 -0.029
 3 0 4.22 3.05 1.88 0.013 -0.011 0.037
 Energy
   -4.8123456789
 PlusStress: xx yy zz yz xz xy
   0.041 -0.027 0.018 0.006 -0.004 0.003
END_CFG
""")
init_config = yaml("model.yaml", MODEL + "output:\n  model: yaml-seed.mtp\n")
run("init-check", ["init-sh", init_config, "--check-config"])
assert not (ROOT / "yaml-seed.mtp").exists()
run("init-yaml", ["init-sh", init_config])
run("init-legacy", ["init-sh", ROOT / "legacy-seed.mtp"] + LEGACY)
assert (ROOT / "yaml-seed.mtp").read_bytes() == (ROOT / "legacy-seed.mtp").read_bytes(), "topology differs"
CHECKS += 1

for sharing, semantic_alias in [("L", "angular"), ("K", "channel"), ("LK", "channel-angular")]:
    canonical_config = yaml("warp-" + sharing + ".yaml", MODEL.replace("warp_sharing: LK", "warp_sharing: " + sharing)
                            + "output:\n  model: warp-" + sharing + ".mtp\n")
    alias_config = yaml("warp-alias-" + sharing + ".yaml", MODEL.replace("warp_sharing: LK", "warp_sharing: " + semantic_alias)
                        + "output:\n  model: warp-alias-" + sharing + ".mtp\n")
    run("warp-canonical-" + sharing, ["init-sh", canonical_config])
    run("warp-alias-" + sharing, ["init-sh", alias_config])
    run("warp-legacy-" + sharing, ["init-sh", ROOT / ("warp-legacy-" + sharing + ".mtp")]
        + LEGACY + ["--scaling-map=" + sharing])
    assert (ROOT / ("warp-" + sharing + ".mtp")).read_bytes() == (ROOT / ("warp-alias-" + sharing + ".mtp")).read_bytes() \
        == (ROOT / ("warp-legacy-" + sharing + ".mtp")).read_bytes(), "warp sharing topology differs: " + sharing
    CHECKS += 1

for label, layout, semantic_alias, rank in [("full", "full", "per-basis", 1),
                                           ("shared-n", "shared-n", "shared-latent", 1),
                                           ("shared-n-2", "shared-n", "shared-latent", 2)]:
    response = "  response:\n    layout: " + layout + "\n    latent_channels: " + str(rank) + "\n"
    canonical_config = yaml("response-" + label + ".yaml", MODEL + response
                            + "output:\n  model: response-" + label + ".mtp\n")
    alias_config = yaml("response-alias-" + label + ".yaml", MODEL + response.replace("layout: " + layout, "layout: " + semantic_alias)
                        + "output:\n  model: response-alias-" + label + ".mtp\n")
    run("response-canonical-" + label, ["init-sh", canonical_config])
    run("response-alias-" + label, ["init-sh", alias_config])
    run("response-legacy-" + label, ["init-sh", ROOT / ("response-legacy-" + label + ".mtp")]
        + LEGACY + ["--adaptive-radial-projection=" + layout, "--adaptive-radial-shared-n-rank=" + str(rank)])
    assert (ROOT / ("response-" + label + ".mtp")).read_bytes() == (ROOT / ("response-alias-" + label + ".mtp")).read_bytes() \
        == (ROOT / ("response-legacy-" + label + ".mtp")).read_bytes(), "response topology differs: " + label
    CHECKS += 1

train_config = yaml("train.yaml", MODEL + """initialization:
  seed: 41
training:
  optimizer: bfgs
  iterations: 1
loss:
  energy_weight: 1
  force_weight: 0.1
  virial_weight: 0.01
data:
  train: tiny.cfg
  valid: tiny.cfg
output:
  directory: yaml-run
""")
output = run("train-check", ["train", "--config", train_config, "--check-config"])
assert "CONFIG_CHECK_PASS" in output and not (ROOT / "yaml-run").exists()
run("train-yaml", ["train", train_config])
(ROOT / "legacy-run").mkdir()
run("train-legacy", ["train", CFG] + LEGACY + ["--init-sh", "--init-params=same", "--init-random-seed=41",
    "--max-iter=1", "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01", "--valid-cfgs=" + str(CFG),
    "--inline-sh-model=" + str(ROOT / "legacy-run/initial.mtp"), "--curr-pot-name=" + str(ROOT / "legacy-run/current.mtp"),
    "--best-pot-name=" + str(ROOT / "legacy-run/best.mtp"), "--trained-pot-name=" + str(ROOT / "legacy-run/final.mtp"),
    "--optimizer-trace-file=" + str(ROOT / "legacy-run/optimizer.csv")])
for field in ["initial.mtp", "final.mtp", "optimizer.csv"]:
    assert (ROOT / "yaml-run" / field).read_bytes() == (ROOT / "legacy-run" / field).read_bytes(), field + " differs"
    CHECKS += 1

run("predict-yaml", ["calc-efs", ROOT / "yaml-run/final.mtp", CFG, ROOT / "yaml-pred.cfg"])
run("predict-legacy", ["calc-efs", ROOT / "legacy-run/final.mtp", CFG, ROOT / "legacy-pred.cfg"])
assert (ROOT / "yaml-pred.cfg").read_bytes() == (ROOT / "legacy-pred.cfg").read_bytes(), "E/F/stress differs"
CHECKS += 1

run("reject-overwrite", ["train", train_config], failure="refusing to reuse/overwrite")
run("reject-epoch", ["train", train_config, "--epochs=2000", "--check-config"], failure="not supported")
run("reject-conflict", ["train", train_config, "--max-iter=2", "--check-config"], failure="conflicting")
run("reject-unknown", ["train", train_config, "--model.unknown=1", "--check-config"], failure="unknown setting")

continue_config = yaml("continue.yaml", """data:
  model: yaml-run/final.mtp
  train: tiny.cfg
training:
  iterations: 0
output:
  directory: resumed
""")
run("continue", ["train", continue_config])
assert (ROOT / "resumed/final.mtp").exists() and not (ROOT / "resumed/initial.mtp").exists()
bad_continue = yaml("incomplete.yaml", """data:
  model: yaml-seed.mtp
  train: tiny.cfg
training:
  iterations: 0
output:
  directory: invalid-resume
""")
run("reject-incomplete", ["train", bad_continue], failure="requires a complete checkpoint")

# Disable coordination through the existing gate-free ranked SH implementation.
# Compare to its legacy CLI, not to an adaptive model with a zeroed U matrix.
STATIC_MODEL = """model:
  kind: coordination-adaptive
  elements: [H, O]
  l_max: 1
  radial_channels: 2
  neighbor_cutoff: 3.5
  radial:
    basis_size: 3
    warp_sharing: LK
  coordination:
    enabled: false
  interaction:
    channels: 2
    body_order: 3
    channel_coupling: cross
"""
STATIC_LEGACY = ["--superlinear-mode=legacy", "--species-count=2", "--l-max=1", "--k-max=2",
                 "--cutoff=3.5", "--radial-basis-size=3", "--outer-k-rank=2", "--body-order=3",
                 "--sh-factor-pruning=q-total"]
static_kind_config = yaml("static-kind.yaml", STATIC_MODEL.replace("kind: coordination-adaptive", "kind: 'static-radial'")
                          .replace("  coordination:\n    enabled: false\n", "") + "output:\n  model: static-kind.mtp\n")
run("static-quoted-kind", ["init-sh", static_kind_config])
for sharing in ["L", "K", "LK"]:
    config = yaml("static-" + sharing + ".yaml", STATIC_MODEL.replace("warp_sharing: LK", "warp_sharing: " + sharing)
                  + "output:\n  model: static-" + sharing + ".mtp\n")
    run("static-init-" + sharing, ["init-sh", config])
    run("static-legacy-init-" + sharing, ["init-sh", ROOT / ("static-legacy-" + sharing + ".mtp")]
        + STATIC_LEGACY + ["--scaling-map=" + sharing])
    text = (ROOT / ("static-" + sharing + ".mtp")).read_text()
    assert "two_layer_gate" not in text and "adaptive_radial" not in text, "static topology contains coordination/response"
    assert (ROOT / ("static-" + sharing + ".mtp")).read_bytes() == (ROOT / ("static-legacy-" + sharing + ".mtp")).read_bytes()
    CHECKS += 1

STATIC_TRAINING = """initialization:
  parameters: random
  seed: 41
training:
  optimizer: bfgs
  iterations: 1
regularization:
  radial_smoothness: 0
  radial_smoothness_grid: 32
  energy_readout_l2: 1e-5
loss:
  energy_weight: 1
  force_weight: 0.1
  virial_weight: 0.01
data:
  train: tiny.cfg
  valid: tiny.cfg
output:
  directory: static-run
"""
static_train_config = yaml("static-train.yaml", STATIC_MODEL + STATIC_TRAINING)
# Adaptive initialization earlier in this same fixture root legitimately uses
# ini.mtp. Static runs must neither create nor modify that inherited artifact.
def undeclared_initial_snapshot():
    path = ROOT / "ini.mtp"
    return (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None


static_undeclared_before = undeclared_initial_snapshot()
run("static-train-yaml", ["train", static_train_config])
assert undeclared_initial_snapshot() == static_undeclared_before, "static grouped run changed an undeclared pretraining output"
(ROOT / "static-legacy-run").mkdir()
run("static-train-legacy", ["train", CFG] + STATIC_LEGACY + ["--init-sh", "--init-params=random", "--init-random-seed=41",
    "--skip-preinit", "--max-iter=1", "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
    "--radial-smooth=0", "--radial-smooth-grid=32", "--scalar-head-l2=1e-5", "--valid-cfgs=" + str(CFG),
    "--inline-sh-model=" + str(ROOT / "static-legacy-run/initial.mtp"), "--curr-pot-name=" + str(ROOT / "static-legacy-run/current.mtp"),
    "--best-pot-name=" + str(ROOT / "static-legacy-run/best.mtp"), "--trained-pot-name=" + str(ROOT / "static-legacy-run/final.mtp"),
    "--optimizer-trace-file=" + str(ROOT / "static-legacy-run/optimizer.csv")])
for field in ["initial.mtp", "final.mtp", "optimizer.csv"]:
    assert (ROOT / "static-run" / field).read_bytes() == (ROOT / "static-legacy-run" / field).read_bytes(), "static " + field + " differs"
    CHECKS += 1
run("static-predict-yaml", ["calc-efs", ROOT / "static-run/final.mtp", CFG, ROOT / "static-yaml-pred.cfg"])
run("static-predict-legacy", ["calc-efs", ROOT / "static-legacy-run/final.mtp", CFG, ROOT / "static-legacy-pred.cfg"])
assert (ROOT / "static-yaml-pred.cfg").read_bytes() == (ROOT / "static-legacy-pred.cfg").read_bytes()
CHECKS += 1
static_continue_config = yaml("static-continue.yaml", """data:
  model: static-run/final.mtp
  train: tiny.cfg
training:
  profile: 'native-static-v1'
  iterations: 1
output:
  directory: static-resumed
""")
run("static-continue", ["train", static_continue_config])
assert (ROOT / "static-resumed/final.mtp").exists() and not (ROOT / "static-resumed/initial.mtp").exists()
default_static_config = yaml("static-defaults.yaml", STATIC_MODEL + """training:
  iterations: 1
data:
  train: tiny.cfg
output:
  directory: static-defaults
""")
default_output = run("static-default-training", ["train", default_static_config])
assert "Initial untrained model warmup rescale+linear solve done" in default_output and "Pre-training started" not in default_output
assert "coordination_response_l2" not in default_output and (ROOT / "static-defaults/final.mtp").exists()
assert undeclared_initial_snapshot() == static_undeclared_before
wrong_profile = yaml("static-wrong-profile.yaml", static_continue_config.read_text().replace("native-static-v1", "native-adaptive-v1")
                     .replace("static-resumed", "static-wrong-profile"))
run("static-reject-wrong-profile", ["train", wrong_profile], failure="does not match the checkpoint architecture")
wrong_adaptive_profile = yaml("adaptive-wrong-profile.yaml", continue_config.read_text()
                              .replace("  iterations: 0", "  profile: native-static-v1\n  iterations: 0")
                              .replace("directory: resumed", "directory: adaptive-wrong-profile"))
run("adaptive-reject-wrong-profile", ["train", wrong_adaptive_profile], failure="does not match the checkpoint architecture")
run("static-reject-response", ["train", static_train_config, "--response-layout=full", "--check-config"], failure="unavailable")
run("static-reject-coordination", ["train", static_train_config, "--coord-channels=1", "--check-config"], failure="unavailable")
run("static-reject-diagonal", ["train", static_train_config, "--interaction-coupling=diagonal", "--check-config"], failure="conflict")

# New Legacy grouped entry: true independent ranked gate/outer graphs, species
# head allocation and native warmup/training, not just option translation.
for inner in ("cross", "diagonal"):
    for outer in ("cross", "diagonal"):
        label = "legacy-" + inner + "-" + outer
        legacy_model = """model:
  kind: coordination-legacy
  elements: [H, O]
  l_max: 1
  radial_channels: 4
  neighbor_cutoff: 3.5
  radial:
    basis_size: 3
    warp_sharing: LK
  coordination:
    channels: 2
    body_order: 3
    channel_coupling: INNER
  interaction:
    channels: 3
    body_order: 3
    channel_coupling: OUTER
  readout:
    element_mode: per-element
""".replace("INNER", inner).replace("OUTER", outer)
        config = yaml(label + ".yaml", legacy_model + STATIC_TRAINING.replace("directory: static-run", "directory: " + label))
        run(label + "-check", ["train", config, "--check-config"])
        assert not (ROOT / label).exists()
        native_output = run(label + "-train", ["train", config])
        assert "Pre-training started" not in native_output
        final = ROOT / label / "final.mtp"
        text = final.read_text()
        assert "version = 1.8.0" in text and "readout_mode = species" in text
        assert undeclared_initial_snapshot() == static_undeclared_before
        run(label + "-predict", ["calc-efs", final, CFG, ROOT / (label + "-pred.cfg")])
        resumed = yaml(label + "-continue.yaml", """data:
  model: FINAL
  train: tiny.cfg
training:
  profile: native-legacy-v1
  iterations: 1
output:
  directory: RESUMED
""".replace("FINAL", str(final)).replace("RESUMED", label + "-resumed"))
        run(label + "-continue", ["train", resumed])
        assert (ROOT / (label + "-resumed") / "final.mtp").exists()
        assert not (ROOT / (label + "-resumed") / "initial.mtp").exists()

# Gate-free species head uses the same Legacy readout math and profile guard.
static_species_config = yaml("static-species.yaml", STATIC_MODEL +
    "  readout:\n    element_mode: per-element\n" +
    STATIC_TRAINING.replace("directory: static-run", "directory: static-species"))
run("static-species-train", ["train", static_species_config])
run("static-species-predict", ["calc-efs", ROOT / "static-species/final.mtp", CFG, ROOT / "static-species-pred.cfg"])

if MPI:
    output = run("mpi-check", ["train", train_config, "--check-config"], mpi=True)
    assert output.count("CONFIG_CHECK_PASS") == 1
    CHECKS += 1
    static_mpi_config = yaml("static-mpi.yaml", STATIC_MODEL + STATIC_TRAINING.replace("directory: static-run", "directory: static-mpi"))
    run("static-mpi-train", ["train", static_mpi_config], mpi=True)
    assert (ROOT / "static-mpi/final.mtp").exists() and undeclared_initial_snapshot() == static_undeclared_before

print("GROUPED_NATIVE_CLI_PASS checks=" + str(CHECKS) + " fixtures=" + str(ROOT))
