#!/usr/bin/env python3
"""Compare balanced initialization/BFGS with one and two active MPI ranks.

Uses only supplied files, writes all generated artifacts below --output-dir,
and does not modify the binary, seed, fixture or any source. Example:
  python3 sh_pure_w_mpi_initialization_check.py /path/mlp-sus2 seed.mtp \
      pure_w_regularization.cfg --output-dir evidence
Run this Python process once (not under mpirun); it launches both MPI commands.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def perturb_fixture(source, variant):
    blocks = re.findall(r'(?ms)^\s*BEGIN_CFG\b.*?^\s*END_CFG\b', source)
    if not blocks:
        raise ValueError('fixture needs a text BEGIN_CFG/END_CFG block')
    lines = blocks[0].strip().splitlines()
    size = int(re.search(r'(?m)^\s*Size\s*\n\s*(\d+)', blocks[0])[1])
    header = next(i for i, line in enumerate(lines) if line.strip().startswith('AtomData:'))
    columns = lines[header].split(':', 1)[1].split()
    indices = [columns.index(name) for name in ('cartes_x', 'cartes_y', 'cartes_z')]
    if size < 2:
        raise ValueError('fixture needs at least two atoms for non-rigid perturbations')
    for atom in range(size):
        fields = lines[header + 1 + atom].split()
        if len(fields) != len(columns):
            raise ValueError('fixture AtomData rows do not match the header')
        for axis, column in enumerate(indices):
            phase = (atom + 1) * (axis + 1) * 0.73
            offset = 0.017 * math.sin(phase) if variant == 0 else -0.023 * math.cos(phase + 0.4)
            fields[column] = format(float(fields[column]) + offset, '.17g')
        lines[header + 1 + atom] = ' ' + ' '.join(fields)
    return '\n'.join(lines) + '\n', size


def array(source, key):
    match = re.search(r'(?m)^\s*' + re.escape(key) + r'\s*=\s*\{([^}]*)\}', source)
    if not match:
        raise ValueError('model missing ' + key)
    values = [float(x) for x in re.findall(NUM, match[1])]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid coefficients for ' + key)
    return values


def compare(label, lhs, rhs, atol, rtol):
    if len(lhs) != len(rhs):
        raise AssertionError((label, 'size', len(lhs), len(rhs)))
    max_abs = 0.0
    max_ratio = 0.0
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError((label, index, a, b, 'non-finite'))
        error = abs(a - b)
        tolerance = atol + rtol * max(abs(a), abs(b))
        max_abs = max(max_abs, error)
        max_ratio = max(max_ratio, error / tolerance)
        if error > tolerance:
            raise AssertionError((label, index, a, b, error, tolerance))
    return {'max_abs': max_abs, 'max_tolerance_ratio': max_ratio,
            'atol': atol, 'rtol': rtol, 'count': len(lhs)}


def parse_result(directory, ranks, atom_count):
    log = (directory / 'output.log').read_text()
    execution = re.search(r'optimizer execution mode=([^\n]+)', log)
    if not execution or not re.search(r'\btrain_ranks=' + str(ranks) + r'\b', execution[1]):
        raise AssertionError(('wrong active rank count', ranks, execution[1] if execution else None))
    if 'train_comm_is_world=true' not in execution[1]:
        raise AssertionError(('not all MPI ranks active', ranks, execution[1]))
    if not re.search(r'Pure-W balanced initialization seed=123 target_rms=1 sites='
                     + str(atom_count) + r'\b', log):
        raise AssertionError(('wrong global calibration count', ranks))
    if 'initialized mode=efs-head probes=0' not in log:
        raise AssertionError(('missing exact head coordinates', ranks))
    for bad in ['Initial untrained model warmup', 'Entering Rescale',
                'Pre-training started', 'Rescaling selected scaling=']:
        if bad in log:
            raise AssertionError(('unexpected legacy initialization', bad))
    match = re.search(r'optimizer physical startup objective total_loss=(' + NUM + ')', log)
    if not match:
        raise AssertionError('missing startup physical loss')
    startup = float(match[1])
    with (directory / 'trace.csv').open() as stream:
        trace = list(csv.DictReader(stream))
    if not trace:
        raise AssertionError(('no accepted BFGS step', ranks))
    accepted = float(trace[0]['total_loss'])
    if not math.isfinite(startup) or not math.isfinite(accepted) or accepted > startup + 2e-12:
        raise AssertionError(('invalid/increasing first step', ranks, startup, accepted))
    diagnostics = {}
    for match in re.finditer(r'BFGS coordinate normalization block name=(\S+)([^\n]+)', log):
        fields = dict(re.findall(r'\b(\w+)=(' + NUM + r')', match[2]))
        diagnostics[match[1]] = {key: float(value) for key, value in fields.items()}
    if set(diagnostics) != {'scalar-head'}:
        raise AssertionError(('unexpected coordinate blocks', diagnostics))
    return {'ranks': ranks, 'execution': execution[1], 'startup_loss': startup,
            'first_step_loss': accepted, 'diagnostics': diagnostics,
            'calibrated_w': array((directory / 'ini.mtp').read_text(),
                                  'two_layer_gate_radial_mixing_coeffs')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('seed', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mpi-launcher', default='mpiexec.hydra',
                        help='launcher and fixed options, shell-split; script appends -np 2; one rank runs directly')
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    inputs = {key: getattr(args, key).resolve() for key in ('binary', 'seed', 'fixture')}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError('missing input: ' + str(path))
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix='pure-w-mpi-active-')).resolve()
    print('Evidence: ' + str(output), flush=True)
    hashes_before = {key: digest(path) for key, path in inputs.items()}
    source = inputs['fixture'].read_text()
    cfg0, size0 = perturb_fixture(source, 0)
    cfg1, size1 = perturb_fixture(source, 1)
    if cfg0 == cfg1:
        raise AssertionError('fixtures must have different coordinates')
    dataset = output / 'two-distinct-configs.cfg'
    dataset.write_text(cfg0 + '\n' + cfg1)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MLIP_CACHE_NEIGHBORHOODS='1',
               SUS2_SH_PRODUCT_ROWS='0', SUS2_SH_SITE_DER_CACHE='1',
               SUS2_SH_ACCUM_SKIP_SITE_DERS='1', SUS2_SH_PRODUCT_HVT_REVERSE='1',
               SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE='0')
    results = []
    for ranks in (1, 2):
        directory = output / ('ranks-' + str(ranks))
        directory.mkdir()
        launch = [] if ranks == 1 else [*shlex.split(args.mpi_launcher), '-np', '2']
        command = [*launch, str(inputs['binary']),
                   'train', str(inputs['seed']), str(dataset), '--pure-w-initialization=balanced',
                   '--init-params=same', '--init-random-seed=123', '--max-iter=1',
                   '--optimizer=bfgs', '--radial-parameterization=weight-norm',
                   '--bfgs-coordinate-normalization=efs-head', '--energy-weight=1',
                   '--force-weight=0.3', '--stress-weight=0.1', '--force-loss=l2',
                   '--std-weight=0.1', '--scalar-head-l2=1e-5', '--radial-smooth=1e-6',
                   '--gate-x-l2=0', '--trained-pot-name=trained.mtp',
                   '--curr-pot-name=current.mtp', '--optimizer-trace-file=trace.csv']
        (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (directory / 'output.log').open('w') as stream:
            result = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode:
            raise AssertionError(('training failed', ranks, result.returncode, str(directory / 'output.log')))
        results.append(parse_result(directory, ranks, size0 + size1))
    one, two = results
    checks = {'calibrated_w': compare('calibrated W', one['calibrated_w'], two['calibrated_w'], 2e-13, 2e-12),
              'startup_loss': compare('startup loss', [one['startup_loss']], [two['startup_loss']], 2e-10, 2e-9),
              'first_step_loss': compare('first-step loss', [one['first_step_loss']], [two['first_step_loss']], 2e-10, 2e-8)}
    first, second = one['diagnostics']['scalar-head'], two['diagnostics']['scalar-head']
    if set(first) != set(second):
        raise AssertionError(('different diagnostic fields', first, second))
    for key in ('count', 'floored', 'identity'):
        if first[key] != second[key]:
            raise AssertionError(('different discrete head diagnostics', key, first[key], second[key]))
    keys = sorted(first)
    checks['head_curvature_diagnostics'] = compare(
        'head curvature/scale diagnostics', [first[key] for key in keys],
        [second[key] for key in keys], 2e-9, 2e-9)
    checks['head_curvature_diagnostics']['fields'] = keys
    hashes_after = {key: digest(path) for key, path in inputs.items()}
    if hashes_before != hashes_after:
        raise AssertionError('input file modified during test')
    report = {'status': 'PASS', 'input_paths': {key: str(path) for key, path in inputs.items()},
              'input_sha256': hashes_before, 'dataset_sha256': digest(dataset),
              'configurations': 2, 'atoms': size0 + size1, 'results': results, 'checks': checks,
              'coverage': 'Two distinct small configurations, 1/2 active ranks; fixed-setup numerical regression, not all MPI/data inputs.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for key, details in checks.items():
        print('PASS {} max_abs={:.4g} tolerance_ratio={:.4g}'.format(
            key, details['max_abs'], details['max_tolerance_ratio']), flush=True)
    print('pure_w_balanced_mpi_active_check_ok', flush=True)


if __name__ == '__main__':
    main()

