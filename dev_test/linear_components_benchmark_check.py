#!/usr/bin/env python3
"""Frozen-feature parity and native vibrations normal-equation checks.

run --driver DRIVER --cases cases.json --out NEW_DIR
    [--reference-env FLAG=0] [--candidate-env FLAG=1] [--require-coverage]
cases.json is a list of {name, model, cfg}; all input paths must be absolute.
Coverage optionally requires static/effect x shared/species, a small L/K model,
and the current L4/K4/R3/O4 effect. No training or model mutation is performed.
compare REF.dump CAND.dump also works without a native runtime.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np


def read_dump(path):
    frames = []
    with Path(path).open('rb') as handle:
        if handle.read(8) != b'LCOMP001':
            raise ValueError('unknown component dump')
        def integer():
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError('truncated component dump')
            return struct.unpack('=Q', raw)[0]
        def vector(expected):
            count = integer()
            if count != expected:
                raise ValueError('component vector shape mismatch')
            raw = handle.read(8 * count)
            if len(raw) != 8 * count:
                raise ValueError('truncated component array')
            values = np.frombuffer(raw, dtype='=f8').copy()
            if not np.isfinite(values).all():
                raise ValueError('nonfinite component/target')
            return values
        count = integer()
        if not 0 < count <= 10000:
            raise ValueError('invalid bounded frame count')
        for _ in range(count):
            atoms, columns, energy, force, virial = [integer() for _ in range(5)]
            if not atoms or not columns or any(x not in (0, 1) for x in (energy, force, virial)):
                raise ValueError('invalid frame metadata')
            frames.append(dict(atoms=atoms, columns=columns, labels=(energy, force, virial),
                               energy=vector(columns),
                               force=vector(atoms * columns * 3).reshape(atoms, columns, 3),
                               virial=vector(columns * 9).reshape(columns, 9),
                               targets=vector(1 + atoms * 3 + 9)))
        if handle.read(1):
            raise ValueError('unexpected trailing dump bytes')
    return frames


def normal_equations(frames):
    """Native full-column vibrations objective: E²/N + .1 F² + .01 V9²/N.

    No fixed-atom gauge, residual-only stage, or ridge prior. These tests cover
    the complete shared/species linear feature system; missing labels skip rows.
    """
    columns = frames[0]['columns']
    matrix = np.zeros((columns, columns))
    rhs = np.zeros(columns)
    for frame in frames:
        if frame['columns'] != columns:
            raise ValueError('column count changed between frames')
        atoms, labels, target = frame['atoms'], frame['labels'], frame['targets']
        groups = ((labels[0], frame['energy'][None, :], target[:1], 1. / atoms),
                  (labels[1], frame['force'].transpose(0, 2, 1).reshape(-1, columns),
                   target[1:1 + atoms * 3], .1),
                  (labels[2], frame['virial'].T, target[1 + atoms * 3:], .01 / atoms))
        for present, features, values, weight in groups:
            if present:
                matrix += weight * (features.T @ features)
                rhs += weight * (features.T @ values)
    return matrix, rhs


def difference(reference, candidate, atol, rtol):
    if reference.shape != candidate.shape:
        raise ValueError('reference/candidate array shapes differ')
    error = np.abs(reference - candidate)
    bound = atol + rtol * np.maximum(np.abs(reference), np.abs(candidate))
    worst = int(np.argmax(error / bound))
    return dict(max_abs=float(error.max()), max_scaled=float((error / bound).max()),
                worst_index=list(np.unravel_index(worst, error.shape)),
                passed=bool(np.all(error <= bound)), elements=int(error.size))


def compare(reference, candidate, atol=1e-10, rtol=1e-10):
    left, right = read_dump(reference), read_dump(candidate)
    if len(left) != len(right):
        raise ValueError('frame counts differ')
    result = {'frames': [], 'atol': atol, 'rtol': rtol}
    passed = True
    for a, b in zip(left, right):
        if any(a[key] != b[key] for key in ('atoms', 'columns', 'labels')):
            raise ValueError('frame metadata differ')
        if not np.array_equal(a['targets'], b['targets']):
            raise ValueError('frozen labels changed')
        row = {key: difference(a[key], b[key], atol, rtol) for key in ('energy', 'force', 'virial')}
        passed &= all(value['passed'] for value in row.values())
        result['frames'].append(row)
    a_matrix, a_rhs = normal_equations(left)
    b_matrix, b_rhs = normal_equations(right)
    result['normal_matrix'] = difference(a_matrix, b_matrix, atol, rtol)
    result['normal_rhs'] = difference(a_rhs, b_rhs, atol, rtol)
    result['passed'] = bool(passed and result['normal_matrix']['passed'] and result['normal_rhs']['passed'])
    return result


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def assignments(values):
    result = {}
    for value in values:
        key, separator, setting = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('environment overrides require NAME=VALUE')
        result[key] = setting
    return result


def run(args):
    cases = json.loads(Path(args.cases).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty list')
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', driver=str(Path(args.driver).resolve()),
                  driver_sha256=digest(args.driver), cases=[],
                  normal_equation_scope='full native vibrations feature columns; no ridge/gauge')
    seen, coverage = set(), set()
    for case in cases:
        name = case['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
            raise ValueError('case names must be unique filename-safe strings')
        seen.add(name)
        model, cfg = Path(case['model']), Path(case['cfg'])
        if not model.is_absolute() or not cfg.is_absolute():
            raise ValueError('frozen model/CFG paths must be absolute')
        text = model.read_text()
        metadata = dict(re.findall(r'^\s*(\w+)\s*=\s*([^\n]+)', text, re.M))
        readout = metadata.get('readout_mode', 'shared').strip()
        effect = metadata.get('two_layer_gate_enabled', 'false').strip() == 'true'
        coverage.add(('effect' if effect else 'static') + '_' + readout)
        lmax, kmax = int(metadata['sh_l_max']), int(metadata['sh_k_max'])
        if lmax <= 2 and kmax <= 2:
            coverage.add('small')
        if effect and (lmax, kmax, int(metadata.get('two_layer_gate_k_rank', 0)),
                       int(metadata.get('outer_k_rank', 0))) == (4, 4, 3, 4):
            coverage.add('production')
        hashes = {'model': digest(model), 'cfg': digest(cfg)}
        trials, reference_dump = [], None
        for label in ('reference', 'candidate', 'candidate', 'reference'):
            index = len(trials)
            dump = output / (name + '-' + str(index) + '-' + label + '.dump')
            env = dict(os.environ)
            ref, cand = assignments(args.reference_env), assignments(args.candidate_env)
            for key in ref.keys() | cand.keys():
                env.pop(key, None)
            env.update(ref if label == 'reference' else cand)
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_DYNAMIC='FALSE')
            command = [str(Path(args.driver).resolve()), str(model), str(cfg), str(dump),
                       str(args.repeats), str(args.max_configs), str(args.warmup)]
            proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout)
            (output / (dump.stem + '.log')).write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError('driver failed: ' + str(proc.returncode) + ' ' + dump.stem)
            summary = json.loads(next(line.partition('LINEAR_COMPONENTS_JSON ')[2] for line in proc.stdout.splitlines()
                                      if line.startswith('LINEAR_COMPONENTS_JSON ')))
            if not summary['finite'] or not summary['model_unchanged'] or not math.isfinite(summary['seconds']):
                raise ValueError('driver invariant failed')
            if reference_dump is None:
                reference_dump = dump
            parity = compare(reference_dump, dump, args.atol, args.rtol)
            trials.append(dict(mode=label, summary=summary, parity=parity))
            if not parity['passed']:
                (output/'failed-parity.json').write_text(json.dumps(parity, indent=2, default=int))
                raise AssertionError('component/normal-equation parity failed: ' + dump.stem)
        if hashes != {'model': digest(model), 'cfg': digest(cfg)}:
            raise AssertionError('frozen input file changed')
        report['cases'].append(dict(name=name, model=str(model), cfg=str(cfg), input_sha256=hashes, trials=trials))
        (output/'report.json').write_text(json.dumps(report, indent=2, default=int) + '\n')
    required = {'static_shared', 'static_species', 'effect_shared', 'effect_species', 'small', 'production'}
    if args.require_coverage and not required <= coverage:
        raise ValueError('missing qualification coverage: ' + ','.join(sorted(required - coverage)))
    report.update(status='PASS', coverage=sorted(coverage))
    (output/'report.json').write_text(json.dumps(report, indent=2, default=int) + '\n')
    print(json.dumps({'status': 'PASS', 'cases': len(cases), 'coverage': sorted(coverage)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('compare')
    check.add_argument('reference'); check.add_argument('candidate')
    native = sub.add_parser('run')
    for key in ('driver', 'cases', 'out'):
        native.add_argument('--' + key, required=True)
    native.add_argument('--reference-env', action='append', default=[])
    native.add_argument('--candidate-env', action='append', default=[])
    native.add_argument('--require-coverage', action='store_true')
    native.add_argument('--repeats', type=int, default=1)
    native.add_argument('--max-configs', type=int, default=1)
    native.add_argument('--warmup', type=int, default=0)
    native.add_argument('--timeout', type=float, default=900)
    for target in (check, native):
        target.add_argument('--atol', type=float, default=1e-10)
        target.add_argument('--rtol', type=float, default=1e-10)
    args = parser.parse_args()
    if args.atol <= 0 or args.rtol < 0:
        parser.error('require atol>0 and rtol>=0')
    if args.command == 'compare':
        result = compare(args.reference, args.candidate, args.atol, args.rtol)
        print(json.dumps(result, indent=2, default=int))
        return 0 if result['passed'] else 1
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
