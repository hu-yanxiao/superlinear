#!/usr/bin/env python3
"""Deterministic MPI startup-race regression on allocated CPU resources.

The MPI_Init interposer delays one rank *after* collective initialization, so
the old entrypoint writes initial.sus2 before that rank checks output reuse.
The interposer is a test fixture only; it is never installed with the trainer.
Fixtures and logs are retained for inspection. No production data is consumed.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
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
initialization:
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
output:
  directory: run
"""
FRAME = """BEGIN_CFG
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
"""
LEGACY = ["--superlinear-mode=radial-coefficient-gate", "--species-count=2",
          "--l-max=1", "--k-max=2", "--cutoff=3.5", "--radial-basis-size=3",
          "--two-layer-gate-k-rank=1", "--two-layer-gate-body-order=3",
          "--outer-k-rank=1", "--body-order=3", "--init-sh", "--init-params=same",
          "--init-random-seed=41", "--max-iter=1", "--energy-weight=1",
          "--force-weight=0.1", "--stress-weight=0.01"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--old-binary", type=Path)
    parser.add_argument("--mpi-command", default="mpiexec.hydra -n 4")
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    assert args.ranks >= 2
    binary = args.binary.resolve()
    root = Path(tempfile.mkdtemp(prefix="grouped-mpi-startup-"))
    print("fixtures=" + str(root), flush=True)
    shim = root / "mpi_startup_delay.so"
    compile_result = subprocess.run(shlex.split(os.environ.get("CC_MPI", "mpicc")) +
        ["-std=c99", "-D_DEFAULT_SOURCE", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
         str(REPO / "dev_test/grouped_mpi_startup_delay.c"), "-o", str(shim)],
        capture_output=True, text=True, timeout=120)
    (root / "shim-compile.log").write_text(compile_result.stdout + compile_result.stderr)
    assert compile_result.returncode == 0, compile_result.stderr
    mpi = shlex.split(args.mpi_command)
    base_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", MKL_DYNAMIC="FALSE")
    delayed_env = dict(base_env, LD_PRELOAD=str(shim),
                       GROUPED_TEST_DELAY_RANK=str(args.ranks - 1), GROUPED_TEST_DELAY_MS="2000")

    def fixture(label):
        path = root / label
        path.mkdir()
        (path / "tiny.cfg").write_text(FRAME * args.ranks)
        (path / "train.yaml").write_text(MODEL)
        return path

    def run(label, exe, cwd, command, delay=True):
        result = subprocess.run(mpi + [str(exe)] + command, cwd=cwd,
                                env=delayed_env if delay else base_env,
                                capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        (root / (label + ".log")).write_text(output)
        if delay:
            assert "test-only MPI startup delay:" in output, "MPI interposer did not execute"
        return result, output

    if args.old_binary:
        old_dir = fixture("old-race")
        result, output = run("old-race", args.old_binary.resolve(), old_dir, ["train", "train.yaml"])
        assert result.returncode != 0 and (old_dir / "run/initial.sus2").exists(), output[-5000:]
        assert "inline initial model written" in output and ("Abort(2)" in output or "MPI_Abort" in output), output
        assert not (old_dir / "run/final.sus2").exists(), "old binary unexpectedly completed"
        print("PASS old binary reproduces startup race", flush=True)

    delayed_dir = fixture("fixed-delayed")
    result, output = run("fixed-delayed", binary, delayed_dir, ["train", "train.yaml"])
    assert result.returncode == 0 and (delayed_dir / "run/final.sus2").exists(), output[-5000:]
    print("PASS delayed grouped fresh MPI training", flush=True)
    normal_dir = fixture("fixed-normal")
    result, output = run("fixed-normal", binary, normal_dir, ["train", "train.yaml"], delay=False)
    assert result.returncode == 0, output[-5000:]
    legacy_dir = fixture("fixed-legacy")
    (legacy_dir / "run").mkdir()
    command = ["train", "tiny.cfg"] + LEGACY + ["--inline-sh-model=run/initial.sus2",
        "--curr-pot-name=run/current.sus2", "--best-pot-name=run/best.sus2",
        "--trained-pot-name=run/final.sus2", "--optimizer-trace-file=run/optimizer.csv"]
    result, output = run("fixed-legacy", binary, legacy_dir, command)
    assert result.returncode == 0, output[-5000:]
    fields = ["initial.sus2", "final.sus2", "optimizer.csv"]
    before = {field: (delayed_dir / "run" / field).read_bytes() for field in fields}
    for field in fields:
        assert before[field] == (normal_dir / "run" / field).read_bytes(), "delay changed " + field
        assert before[field] == (legacy_dir / "run" / field).read_bytes(), "legacy parity: " + field
    print("PASS exact model/optimizer parity: delayed, normal and legacy", flush=True)
    # Early rejection may abort before the delayed rank reaches its interposer
    # marker. The delay is qualified above; these cases test output protection.
    result, output = run("reject-reuse", binary, delayed_dir, ["train", "train.yaml"], delay=False)
    assert result.returncode != 0 and "refusing to reuse/overwrite" in output, output[-5000:]
    assert before == {field: (delayed_dir / "run" / field).read_bytes() for field in fields}
    print("PASS existing output still rejected and unmodified", flush=True)
    check_dir = fixture("check-only")
    result, output = run("check-only", binary, check_dir, ["train", "train.yaml", "--check-config"], delay=False)
    assert result.returncode == 0 and "CONFIG_CHECK_PASS" in output and not (check_dir / "run").exists(), output[-5000:]
    print("PASS MPI check-config remains read-only", flush=True)
    print("GROUPED_MPI_STARTUP_PASS", flush=True)


if __name__ == "__main__":
    main()
