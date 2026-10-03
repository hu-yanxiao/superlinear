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

if MPI:
    output = run("mpi-check", ["train", train_config, "--check-config"], mpi=True)
    assert output.count("CONFIG_CHECK_PASS") == 1
    CHECKS += 1

print("GROUPED_NATIVE_CLI_PASS checks=" + str(CHECKS) + " fixtures=" + str(ROOT))
