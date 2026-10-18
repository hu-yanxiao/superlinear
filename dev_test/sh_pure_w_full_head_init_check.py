#!/usr/bin/env python3
"""Tiny CLI 2x2 regression for pure-W full head solve and full head chart.

For O=1,2,4,8, every control starts from the same raw incomplete seed:

  default EFS solve / full-quadratic solve
        x
  diagonal efs-head chart / full-Gram efs-head-full chart

This checks causal boundaries and CLI wiring, not convergence quality. All
evidence is retained below the worktree's ignored .local_tmp directory. Set
SUS2_TEST_LAUNCHER='mpirun -np 2' to repeat the whole test with two MPI ranks.
"""

import csv
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "bin/mlp-sus2"
CFG = ROOT / "native_tests/pure_w_regularization.cfg"
(ROOT / ".local_tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="pure-w-full-head-controls-", dir=ROOT / ".local_tmp"))
LAUNCHER = shlex.split(os.environ.get("SUS2_TEST_LAUNCHER", ""))
RANKS = (1, 2, 4, 8)
SOLVES = ("efs", "full-quadratic")
COORDINATES = ("efs-head", "efs-head-full")
NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
ENV = dict(
    os.environ,
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MLIP_CACHE_NEIGHBORHOODS="1",
    SUS2_SH_PRODUCT_ROWS="0",
    SUS2_SH_SITE_DER_CACHE="1",
    SUS2_SH_ACCUM_SKIP_SITE_DERS="1",
    SUS2_SH_PRODUCT_HVT_REVERSE="1",
    SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE="0",
    SUS2_MTPR_OPTIMIZER_PERF_COUNTERS="1",
)


def run(label, *arguments):
    directory = TMP / label
    directory.mkdir(parents=True)
    command = [*LAUNCHER, str(BIN), *map(str, arguments)]
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    result = subprocess.run(
        command,
        cwd=directory,
        env=ENV,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    (directory / "output.log").write_text(result.stdout)
    assert result.returncode == 0, (label, result.returncode, result.stdout[-5000:])
    return directory, result.stdout


def values(path, key):
    source = path.read_text()
    match = re.search(r"(?ms)^\s*" + re.escape(key) + r"\s*=\s*\{([^}]*)\}", source)
    assert match, (path, key)
    result = [float(value) for value in re.findall(NUMBER, match[1])]
    assert result and all(math.isfinite(value) for value in result), (path, key)
    return result


def startup(log):
    match = re.search(
        r"optimizer physical startup objective total_loss=(" + NUMBER + r")", log
    )
    assert match, log[-3000:]
    result = float(match[1])
    assert math.isfinite(result)
    return result


def semantic_w_output(path, rank, output):
    data = values(path, "two_layer_gate_radial_mixing_coeffs")
    # Tiny topology is L=0,1; K=2; radial basis size=3.
    assert len(data) == 2 * rank * 2 * 3
    result = []
    stride = 2 * 3
    for ell in range(2):
        begin = (ell * rank + output) * stride
        result.extend(data[begin : begin + stride])
    return result


def make_seed(rank):
    directory, _ = run(
        f"O{rank}/seed",
        "init-sh",
        "seed.sus2",
        "--species-count=2",
        "--l-max=1",
        "--k-max=2",
        "--body-order=3",
        "--body-l-max=1,1,1,1",
        "--radial-basis-size=3",
        "--cutoff=5",
        "--scaling=1",
        "--scaling-map=LK",
        "--two-layer-gate",
        "--two-layer-gate-mode=mu-scalar-full",
        "--two-layer-gate-body-order=3",
        "--two-layer-gate-k-rank=2",
        f"--outer-k-rank={rank}",
        "--write-sh-scalar-info",
        "--superlinear-mode=pure-w-channelwise",
    )
    seed = directory / "seed.sus2"
    assert seed.is_file() and "moment_coeffs" not in seed.read_text()
    return seed


def train(rank, seed, solve, coordinate):
    label = f"O{rank}/{solve}__{coordinate}"
    solve_option = () if solve == "efs" else ("--pure-w-head-solve=full-quadratic",)
    directory, log = run(
        label,
        "train",
        seed,
        CFG,
        "--pure-w-initialization=balanced",
        "--pure-w-init-target-rms=1",
        "--init-random-seed=123",
        "--init-params=same",
        "--energy-weight=1",
        "--force-weight=0.3",
        "--stress-weight=0.1",
        "--force-loss=l2",
        "--std-weight=0.1",
        "--stdd-weight=0",
        "--scalar-head-l2=1e-5",
        "--radial-smooth=1e-6",
        "--gate-x-l2=0",
        "--optimizer=bfgs",
        "--bfgs-line-search=wolfe",
        "--radial-parameterization=weight-norm",
        f"--bfgs-coordinate-normalization={coordinate}",
        *solve_option,
        "--max-iter=1",
        "--trained-pot-name=trained.sus2",
        "--curr-pot-name=current.sus2",
        "--optimizer-trace-file=trace.csv",
    )
    for forbidden in (
        "Initial untrained model warmup",
        "Entering Rescale",
        "Pre-training started",
        "Rescaling selected scaling=",
    ):
        assert forbidden not in log, (label, forbidden)
    assert f"head_solve={solve}" in log
    assert f"BFGS coordinate normalization initialized mode={coordinate}" in log
    assert "optimizer coordinate physical roundtrip verified" in log
    blocks = re.findall(
        r"Pure-W init block l=(\d+) o=(\d+) rms_before=(\S+) "
        r"factor=(\S+) rms_after=(\S+) status=(\S+)",
        log,
    )
    assert len(blocks) == 2 * rank, (label, len(blocks), 2 * rank)
    assert all(
        status == "scaled"
        and math.isfinite(float(after))
        and abs(float(after) - 1.0) <= 2e-12
        for _, _, _, _, after, status in blocks
    ), blocks

    quadratic_builds = log.count("scalar-head physical quadratic built")
    quadratic_solves = log.count("scalar-head full quadratic solve done")
    if solve == "full-quadratic":
        assert quadratic_solves == 1 and "physical_objective=true" in log
        assert log.count(
            "scalar-head physical quadratic built "
            "ctx=balanced pure-W full conditional head solve"
        ) == 1
        match = re.search(
            r"scalar-head full quadratic solve done[^\n]*before=(" + NUMBER
            + r")[^\n]*after=(" + NUMBER + r")",
            log,
        )
        assert match and float(match[2]) <= float(match[1]) + 1e-10 * max(1.0, abs(float(match[1])))
    else:
        assert quadratic_solves == 0
    if coordinate == "efs-head-full":
        assert "coordinate_chart=full-cholesky-centered" in log
        expected_cache = "hit" if solve == "full-quadratic" else "built"
        assert f"metric_cache={expected_cache}" in log
        assert quadratic_builds == 1
    elif solve == "full-quadratic":
        assert quadratic_builds == 1
    else:
        assert quadratic_builds == 0

    ini = directory / "ini.sus2"
    trained = directory / "trained.sus2"
    assert ini.is_file() and trained.is_file()
    with (directory / "trace.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == 1 and all(
        math.isfinite(float(value)) for row in trace for value in row.values()
    )
    counter = [
        line for line in log.splitlines() if "SUS2_MTPR_OPTIMIZER_PERF_COUNTERS " in line
    ]
    assert len(counter) == 1 and "accepted_steps=1" in counter[0]
    return {
        "directory": directory,
        "log": log,
        "ini": ini,
        "startup": startup(log),
        "accepted_loss": float(trace[0]["total_loss"]),
    }


print(f"Evidence: {TMP}", flush=True)
controls = {}
for outer_rank in RANKS:
    raw_seed = make_seed(outer_rank)
    controls[outer_rank] = {}
    for head_solve in SOLVES:
        controls[outer_rank][head_solve] = {}
        for head_coordinates in COORDINATES:
            controls[outer_rank][head_solve][head_coordinates] = train(
                outer_rank, raw_seed, head_solve, head_coordinates
            )

    # Coordinate choice is downstream of initialization. Both fresh processes
    # must save the same physical initialized model for a fixed solve choice.
    for head_solve in SOLVES:
        diagonal = controls[outer_rank][head_solve]["efs-head"]
        full = controls[outer_rank][head_solve]["efs-head-full"]
        assert diagonal["ini"].read_bytes() == full["ini"].read_bytes()
        error = abs(diagonal["startup"] - full["startup"]) / max(
            1.0, abs(diagonal["startup"]), abs(full["startup"])
        )
        assert error <= 2e-12, (outer_rank, head_solve, error)

    # The full solve follows the legacy EFS solve on identical calibrated W, so
    # its exact conditional physical objective cannot be worse before BFGS.
    for head_coordinates in COORDINATES:
        efs = controls[outer_rank]["efs"][head_coordinates]
        full = controls[outer_rank]["full-quadratic"][head_coordinates]
        assert values(efs["ini"], "two_layer_gate_radial_mixing_coeffs") == values(
            full["ini"], "two_layer_gate_radial_mixing_coeffs"
        )
        tolerance = 2e-10 * max(1.0, abs(efs["startup"]))
        assert full["startup"] <= efs["startup"] + tolerance, (
            outer_rank,
            head_coordinates,
            efs["startup"],
            full["startup"],
        )

# Same random seed and calibration retain semantic W output prefixes as O grows.
for small_rank, large_rank in zip(RANKS, RANKS[1:]):
    small = controls[small_rank]["full-quadratic"]["efs-head"]["ini"]
    large = controls[large_rank]["full-quadratic"]["efs-head"]["ini"]
    for output in range(small_rank):
        small_values = semantic_w_output(small, small_rank, output)
        large_values = semantic_w_output(large, large_rank, output)
        error = max(
            abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
            for lhs, rhs in zip(small_values, large_values)
        )
        assert error <= 2e-11, (small_rank, large_rank, output, error)

print(
    "PASS O1/O2/O4/O8 2x2: same initialized models/startups across charts, "
    "full solve nonincrease, full-Gram built/hit cache boundaries and W prefixes",
    flush=True,
)
print("pure_w_full_head_init_check_ok", flush=True)
