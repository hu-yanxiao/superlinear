#!/usr/bin/env python3
"""CLI regression for opt-in balanced pure-W initialization (public tiny fixture).

Run with Python 3 and an optional mlp-sus2 binary path. For the same checks with
empty MPI ranks, set SUS2_TEST_LAUNCHER='mpirun -n 2' using an MPI binary. Every
command works in an isolated evidence directory, never in the source root.
This is a protocol/one-step smoke check, not a convergence-quality benchmark.
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
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'bin/mlp-sus2'
CFG = ROOT / 'native_tests/pure_w_regularization.cfg'
(ROOT / '.local_tmp').mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix='pure-w-balanced-cli-', dir=ROOT / '.local_tmp'))
LAUNCHER = shlex.split(os.environ.get('SUS2_TEST_LAUNCHER', ''))
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MLIP_CACHE_NEIGHBORHOODS='1', SUS2_SH_PRODUCT_ROWS='0',
           SUS2_SH_SITE_DER_CACHE='1', SUS2_SH_ACCUM_SKIP_SITE_DERS='1',
           SUS2_SH_PRODUCT_HVT_REVERSE='1', SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE='0')
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
COMMON = ['--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1',
          '--scalar-head-l2=1e-5', '--radial-smooth=1e-6', '--gate-x-l2=0']


def run(label, *args, failure=None, env=None):
    directory = TMP / label
    directory.mkdir()
    command = [*LAUNCHER, str(BIN), *map(str, args)]
    (directory / 'command.txt').write_text(shlex.join(command) + '\n')
    result = subprocess.run(command, cwd=directory, env=dict(ENV, **(env or {})),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180)
    (directory / 'output.log').write_text(result.stdout)
    if failure is None:
        assert result.returncode == 0, (label, result.returncode, result.stdout[-5000:])
    else:
        assert result.returncode != 0 and failure in result.stdout, (
            label, result.returncode, failure, result.stdout[-5000:])
    return result.stdout


def values(source, key):
    match = re.search(rf'(?m)^\s*{key}\s*=\s*\{{([^}}]*)\}}', source)
    assert match, key
    result = [float(x) for x in re.findall(NUM, match[1])]
    assert all(math.isfinite(x) for x in result), key
    return result


def global_scaling(source):
    # Save omits the multiplicative identity; Load defaults missing scaling to 1.
    match = re.search(r'(?m)^scaling\s*=\s*(' + NUM + ')', source)
    return float(match[1]) if match else 1.0


def predict(label, model):
    destination = TMP / (label + '.cfg')
    run(label, 'calc-efs', model, CFG, destination)
    source = destination.read_text()
    energy = float(re.search(r'(?m)^\s*Energy\s*\n\s*(' + NUM + ')', source)[1])
    stress = list(map(float, re.search(r'(?m)^\s*PlusStress:[^\n]*\n([^\n]*)', source)[1].split()))
    atoms = re.search(r'(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy', source)[1].strip().splitlines()
    forces = [float(x) for line in atoms for x in line.split()[-3:]]
    assert len(forces) == 12 and len(stress) == 6
    return [energy, *forces, *stress]


def near(a, b, label, tolerance=2e-11):
    assert len(a) == len(b), label
    error = max((abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b)), default=0.0)
    assert error <= tolerance, (label, error)


def seed(label, width=1, pure=True):
    model = TMP / (label + '.sus2')
    run(label, 'init-sh', model, '--species-count=2', '--l-max=1', '--k-max=2',
        '--body-order=3', '--body-l-max=1,1,1,1', '--radial-basis-size=3',
        '--cutoff=5', '--scaling=1', '--scaling-map=LK', '--two-layer-gate',
        '--two-layer-gate-mode=mu-scalar-full', '--two-layer-gate-body-order=3',
        '--two-layer-gate-k-rank=2', f'--outer-k-rank={width}', '--write-sh-scalar-info',
        *(['--superlinear-mode=pure-w-channelwise'] if pure else []))
    return model


def train(label, model, balanced=True, steps=0, extra=(), failure=None, env=None):
    output = TMP / (label + '.sus2')
    log = run(label, 'train', model, CFG, *COMMON, '--init-params=same',
              f'--max-iter={steps}', f'--trained-pot-name={output}',
              f'--curr-pot-name={TMP / (label + ".current.sus2")}',
              f'--optimizer-trace-file={TMP / (label + ".csv")}',
              *(['--pure-w-initialization=balanced', '--init-random-seed=123'] if balanced else []),
              *extra, failure=failure, env=env)
    if failure is not None:
        assert not output.exists(), label + ' wrote a rejected model'
    return output, log


def no_legacy_initialization(log):
    for forbidden in ['Rescale scaling=', 'Rescaling selected scaling=',
                      'Initial untrained model warmup', 'Entering Rescale',
                      'Pre-training started', 'Pre-training ended']:
        assert forbidden not in log, forbidden


print(f'Evidence: {TMP}', flush=True)
one = seed('seed-o1')
three = seed('seed-o3', width=3)
base, base_log = train('balanced-o1', one)
repeat, repeat_log = train('balanced-o1-repeat', one)
wide, wide_log = train('balanced-o3', three)
other, _ = train('balanced-other-seed', one, extra=('--init-random-seed=124',))
uncached, _ = train('balanced-uncached', one, env={
    'MLIP_CACHE_NEIGHBORHOODS': '0', 'SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE': '1',
    'SUS2_SH_SITE_DER_CACHE': '0', 'SUS2_SH_PRODUCT_HVT_REVERSE': '0'})
for log in (base_log, repeat_log, wide_log):
    assert 'Pure-W balanced initialization seed=123 target_rms=1 sites=4' in log
    assert 'Pure-W balanced initialization complete' in log
    assert 'optimizer physical startup objective' not in log
    no_legacy_initialization(log)
    rows = re.findall(r'rms_after=(' + NUM + r') status=(\S+)', log)
    assert rows and all(status == 'scaled' and abs(float(rms) - 1.0) < 2e-12
                        for rms, status in rows), rows
assert base.read_text() == repeat.read_text(), 'same seed is not deterministic'
base_src, wide_src = base.read_text(), wide.read_text()
w1 = values(base_src, 'two_layer_gate_radial_mixing_coeffs')
w3 = values(wide_src, 'two_layer_gate_radial_mixing_coeffs')
assert w1 != values(other.read_text(), 'two_layer_gate_radial_mixing_coeffs')
# Physical layout L,O,K,B: one shared output from each of the two l blocks.
assert len(w1) == 2*1*2*3 and len(w3) == 2*3*2*3
near(w1, w3[:6] + w3[18:24], 'cross-O calibrated W prefix')
near(w1, values(uncached.read_text(), 'two_layer_gate_radial_mixing_coeffs'), 'cache/recompute W')
for key in ['two_layer_gate_radial_coeffs', 'two_layer_gate_weights',
            'two_layer_gate_k_rank_weights', 'two_layer_gate_type_coeffs',
            'two_layer_gate_additive_coeffs', 'scal_coeffs']:
    near(values(base_src, key), values(wide_src, key), 'cross-O unchanged ' + key, 0.0)
assert global_scaling(base_src) == global_scaling(one.read_text()) == 1.0
assert max(map(abs, values(base_src, 'moment_coeffs'))) > 1e-5, 'vacuous head fit'
reference = predict('balanced-prediction', base)
assert max(map(abs, reference[1:13])) > 1e-5, 'vacuous force signal'
near(reference, predict('uncached-prediction', uncached), 'cache/recompute fitted EFS')
train('complete-rejected', base, failure='only accepts an incomplete fresh seed')
same, same_log = train('same-reload', base, balanced=False)
assert 'Pure-W same initial snapshot saved' in same_log
assert 'Pure-W balanced initialization' not in same_log
near(reference, predict('same-prediction', same), 'same resume EFS', 0.0)
assert values(base_src, 'two_layer_gate_radial_mixing_coeffs') == values(
    same.read_text(), 'two_layer_gate_radial_mixing_coeffs')
print('PASS max0 head fit, seed/O prefixes, fixed global scale, cache parity and same reload', flush=True)

legacy = seed('seed-legacy', pure=False)
train('legacy-rejected', legacy, failure='requires a pure-W channelwise model')
for label, option, error in [
    ('target-zero', '--pure-w-init-target-rms=0', 'should be positive'),
    ('target-negative', '--pure-w-init-target-rms=-1', 'should be positive'),
    ('target-nan', '--pure-w-init-target-rms=nan', 'should be a finite double'),
    ('physical', '--radial-parameterization=physical', 'requires BFGS and weight-norm'),
    ('log-cosh', '--force-loss=log-cosh', 'requires --force-loss=l2 for its initial linear head solve'),
    ('random', '--init-params=random', 'requires --init-params=same'),
    ('rescale', '--do-lin-rescale', 'incompatible with do-lin-rescale'),
    ('fine-tune', '--fine-tune', 'incompatible with do-lin-rescale'),
    ('mindist', '--update-mindist', 'incompatible with do-lin-rescale'),
    ('mix-scale', '--two-layer-gate-radial-mixing-init-scale=1', 'uses --pure-w-init-target-rms'),
    ('negative-steps', '--max-iter=-1', 'requires --max-iter >= 0'),
]:
    train(label + '-rejected', one, extra=(option,), failure=error)
train('target-without-mode-rejected', one, balanced=False,
      extra=('--pure-w-init-target-rms=1',), failure='requires --pure-w-initialization=balanced')
train('unknown-mode-rejected', one, balanced=False,
      extra=('--pure-w-initialization=unknown',), failure='should be legacy or balanced')
print('PASS incompatible options rejected without output checkpoints', flush=True)

trained, log = train('balanced-one-step', one, steps=1)
no_legacy_initialization(log)
assert 'initialized mode=efs-head probes=0' in log
assert 'outer_target=pure-W' in log and 'physical_load=roundtrip-verified' in log
assert log.index('Pure-W balanced initialization complete') < log.index('initialized mode=efs-head')
startup = re.search(r'optimizer physical startup objective total_loss=(' + NUM + ')', log)
assert startup, log[-3000:]
start_loss = float(startup[1])
with (TMP / 'balanced-one-step.csv').open() as stream:
    losses = [float(row['total_loss']) for row in csv.DictReader(stream)]
assert losses and all(math.isfinite(x) for x in losses), losses
assert losses[-1] < start_loss - 1e-10 * max(1.0, abs(start_loss)), (start_loss, losses)
initial = TMP / 'balanced-one-step' / 'ini.sus2'
near(values(base_src, 'two_layer_gate_radial_mixing_coeffs'),
     values(initial.read_text(), 'two_layer_gate_radial_mixing_coeffs'), 'max0/max1 initialized W', 0.0)
assert global_scaling(trained.read_text()) == 1.0
print(f'PASS one-step balanced BFGS: objective {start_loss:.12g} -> {losses[-1]:.12g}', flush=True)
print('pure_w_balanced_init_check_ok', flush=True)
