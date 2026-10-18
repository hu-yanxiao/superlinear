#!/usr/bin/env python3
"""Exact head-coordinate regression; keeps all generated evidence in .local_tmp."""
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'bin/mlp-sus2'
(ROOT / '.local_tmp').mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix='pure-w-efs-head-', dir=ROOT / '.local_tmp'))
CFG = ROOT / 'native_tests/pure_w_regularization.cfg'
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(label, *args, failure=None):
    result = subprocess.run([str(BIN), *map(str, args)], cwd=ROOT, env=ENV,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (TMP / (label + '.log')).write_text(result.stdout)
    if failure is not None:
        assert result.returncode and failure in result.stdout, (label, result.stdout[-3000:])
    else:
        assert result.returncode == 0, (label, result.stdout[-5000:])
    return result.stdout


def vector(src, key, values):
    output, count = re.subn(rf'(?m)^\s*{key}\s*=\s*\{{[^}}]*\}}',
                           key + ' = {' + ', '.join(f'{x:.17e}' for x in values) + '}', src)
    assert count == 1, key
    return output


def values(src, key):
    return list(map(float, re.search(rf'(?m)^\s*{key}\s*=\s*\{{([^}}]*)\}}', src)[1].split(',')))


def startup(log):
    match = re.search(r'optimizer physical startup objective total_loss=([0-9.eE+-]+)', log)
    assert match, log[-3000:]
    return float(match[1])


print(f'Evidence: {TMP}', flush=True)
initial, seed = TMP / 'initial.sus2', TMP / 'seed.sus2'
run('init', 'init-sh', initial, '--species-count=2', '--l-max=1', '--k-max=2',
    '--body-order=4', '--body-l-max=1,1,1,1', '--radial-basis-size=3', '--cutoff=5',
    '--scaling=0.3', '--superlinear-mode=pure-w-channelwise', '--outer-k-rank=2',
    '--two-layer-gate-k-rank=2', '--two-layer-gate-mode=mu-scalar-full',
    '--two-layer-gate-body-order=4', '--write-sh-scalar-info')
run('complete', 'roundtrip-model-dev', initial, seed)
src = seed.read_text()
src = vector(src, 'species_coeffs', [1.7, 1.7])
for key, amplitude, bias in [('two_layer_gate_radial_mixing_coeffs', 0.3, 0.05),
                             ('two_layer_gate_radial_coeffs', 0.25, 0.03),
                             ('two_layer_gate_weights', 0.12, 0.0),
                             ('moment_coeffs', 0.12, 0.0)]:
    src = vector(src, key, [bias + amplitude * math.cos(.71 * (i + 1))
                            for i in range(len(values(src, key)))])
seed.write_text(src)
common = ['--max-iter=1', '--skip-preinit', '--init-params=same', '--no-do-lin',
          '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1',
          '--scalar-head-l2=1e-5', '--radial-smooth=1e-6', '--gate-x-l2=0']


def train(label, mode='efs-head', extra=()):
    return run(label, 'train', seed, CFG, *common,
               '--radial-parameterization=weight-norm',
               f'--bfgs-coordinate-normalization={mode}',
               f'--optimizer-trace-file={TMP / (label + ".csv")}',
               f'--trained-pot-name={TMP / (label + ".sus2")}',
               f'--curr-pot-name={TMP / (label + ".current.sus2")}', *extra)


physical = train('physical', 'none')
head = train('head', extra=('--bfgs-coordinate-probes=1',))
repeat = train('head-repeat', extra=('--bfgs-coordinate-probes=17',))
for log in (head, repeat):
    assert 'initialized mode=efs-head probes=0' in log
    assert 'physical_reference=exact' in log
    assert 'optimizer coordinate physical roundtrip verified' in log
    blocks = re.findall(r'BFGS coordinate normalization block name=([^ ]+)', log)
    assert blocks == ['scalar-head'], blocks
    assert abs(startup(log) - startup(physical)) <= 2e-12 * max(1.0, abs(startup(physical)))
    assert 'Rescale scaling=' not in log
blocks = lambda log: re.findall(r'BFGS coordinate normalization block[^\n]+', log)
assert blocks(head) == blocks(repeat), (blocks(head), blocks(repeat))
assert (TMP / 'head.csv').read_text() == (TMP / 'head-repeat.csv').read_text()
with (TMP / 'head.csv').open() as stream:
    losses = [float(row['total_loss']) for row in csv.DictReader(stream)]
assert losses and all(math.isfinite(x) for x in losses)
assert losses[0] <= startup(head) + 2e-12 * max(1.0, abs(startup(head)))

# Step-zero TrainLinear must precede fixed-coordinate construction and leave
# the same physical post-fit objective regardless of the optimization chart.
linear_extra = ('--do-lin', '--do-lin-freq=1', '--do-lin-steps=1')
linear_physical = train('linear-physical', 'none', linear_extra)
linear_head = train('linear-head', 'efs-head', linear_extra)
if 'TrainLinear solve done' in linear_head:  # This progress line is MPI-only.
    assert linear_head.index('TrainLinear solve done') < linear_head.index('initialized mode=efs-head')
assert abs(startup(linear_head) - startup(head)) > 1e-8, 'step-zero linear fit did not change this fixture'
assert abs(startup(linear_head) - startup(linear_physical)) <= 2e-12 * max(1.0, abs(startup(linear_physical)))

# Nonquadratic force loss is supported: exact columns use output curvature,
# not the squared residual-gradient estimator.
robust = train('robust', extra=('--force-loss=log-cosh', '--force-log-cosh-scale=0.1'))
assert 'initialized mode=efs-head' in robust

# Standard-deviation penalties are not approximated/replaced by the EFS metric;
# they remain in the physical objective and receive the same exact pullback.
weighted_extra = ('--std-weight=2', '--stdd-weight=0.05',
                  '--scale-by-force=0.2', '--weighting=structures')
weighted_physical = train('weighted-physical', 'none', weighted_extra)
weighted_head = train('weighted-head', 'efs-head', weighted_extra)
assert abs(startup(weighted_head) - startup(weighted_physical)) <= 2e-12 * max(1.0, abs(startup(weighted_physical)))

# Do not approximate a general checkpoint with unequal center prefactors.
unequal = TMP / 'unequal.sus2'
unequal.write_text(vector(src, 'species_coeffs', [1.7, 1.8]))
run('unequal-rejected', 'train', unequal, CFG, *common,
    '--radial-parameterization=weight-norm', '--bfgs-coordinate-normalization=efs-head',
    f'--trained-pot-name={TMP / "rejected.sus2"}',
    failure='efs-head requires equal center linear prefactors')
print('PASS efs-head exact deterministic columns, physical objective, WN and TrainLinear compatibility', flush=True)
