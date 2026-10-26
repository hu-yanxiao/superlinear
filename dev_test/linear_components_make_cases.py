#!/usr/bin/env python3
"""Generate isolated, nonzero static/effect feature fixtures on a compute node.

python dev_test/linear_components_make_cases.py --binary /abs/mlp-sus2 --out NEW_DIR
Creates 16 complete frozen models: L1..4 x static/effect x shared/species.
Only init-sh, roundtrip-model-dev and finite-difference/readout checks are run;
there is no train invocation. All outputs stay in the new fixture directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from adaptive_species_readout_check import numbers, put_numbers, write_fixture
from sh_effect_parameter_fast_check import active_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topology_options(lmax, kind, readout):
    k, r, o, basis = (4, 3, 4, 8) if lmax == 4 else (2, 1, 2, 3)
    limits = (lmax, lmax, min(lmax, 3), min(lmax, 2))
    options = [
        '--superlinear-mode=legacy', '--species-count=2', f'--l-max={lmax}',
        f'--k-max={k}', '--body-order=5',
        '--body-l-max=' + ','.join(map(str, limits)),
        '--radial-basis-type=RBChebyshev_sss', f'--radial-basis-size={basis}',
        '--min-dist=1.1', '--cutoff=6.0', '--scaling=0.1', '--scaling-map=K',
        f'--outer-k-rank={o}', f'--readout-mode={readout}',
        '--sh-factor-pruning=' + ('q-total' if kind == 'static' else 'same-k'),
        '--write-sh-scalar-info',
    ]
    if kind == 'effect':
        options += ['--two-layer-gate', '--two-layer-gate-mode=mu-scalar-full',
                    '--two-layer-gate-shared-radial', '--two-layer-gate-body-order=4',
                    f'--two-layer-gate-k-rank={r}', '--two-layer-gate-factor-pruning=q-total']
    return options


def env_options(values):
    result = {}
    for value in values:
        key, separator, setting = value.partition('=')
        if not separator or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('expected NAME=VALUE for --fd-env')
        result[key] = setting
    return result


def reconstruction_chart(text, readout):
    """Legacy shared raw TrainLinear columns omit center h_Z.

    Preserve the nonunit-h frozen model for FD and full feature/normal-equation
    parity. Only direct readout reconstruction uses a separately named h=1 copy;
    the historical native checker has no Legacy fixed-center chart option.
    Species heads are already additive and need no such copy.
    """
    if readout == 'shared':
        return put_numbers(text, 'species_coeffs', np.ones(len(numbers(text, 'species_coeffs'))))
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fd-env', action='append', default=[], help='candidate switch for native FD checks')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if not sys.platform.startswith('linux') or not os.environ.get('LSB_JOBID'):
        parser.error('requires an existing Linux LSF compute allocation; never submits one')
    if int(os.environ.get('PMI_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))) != 1:
        parser.error('invoke once, not under multi-rank mpiexec')
    binary = args.binary.resolve(strict=True)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    env = {k: v for k, v in os.environ.items() if not k.startswith('SUS2_SH_')}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_DYNAMIC='FALSE')
    cfg = root/'asymmetric-two-species.cfg'
    write_fixture(cfg)
    report = dict(status='RUNNING', binary=str(binary), binary_sha256=sha(binary),
                  cfg_sha256=sha(cfg), job=os.environ['LSB_JOBID'],
                  fixture='three asymmetric atoms, both species, nonzero E/F/virial labels',
                  commands=[], finite_differences=[], cases=[])

    def save():
        (root/'generation-report.json').write_text(json.dumps(report, indent=2) + '\n')
        (root/'cases.json').write_text(json.dumps(report['cases'], indent=2) + '\n')

    def run(label, argv, overrides=None):
        command = [str(binary), *map(str, argv)]
        proc = subprocess.run(command, cwd=root, env={**env, **(overrides or {})},
                              capture_output=True, text=True, timeout=args.timeout)
        (root/(label + '.log')).write_text(proc.stdout + proc.stderr)
        report['commands'].append(dict(label=label, argv=command, returncode=proc.returncode))
        save()
        if proc.returncode:
            raise RuntimeError('native check failed: ' + label)
        return proc.stdout

    try:
        for lmax in range(1, 5):
            for kind in ('static', 'effect'):
                for readout in ('shared', 'species'):
                    label = f'{kind}-L{lmax}-{readout}'
                    topology, full = root/(label + '-topology.sus2'), root/(label + '-full.sus2')
                    activated, frozen = root/(label + '-activated.sus2'), root/(label + '-frozen.sus2')
                    options = topology_options(lmax, kind, readout)
                    run(label+'-init', ['init-sh', topology, *options])
                    run(label+'-materialize', ['roundtrip-model-dev', topology, full])
                    text = active_text(full.read_text())
                    activated.write_text(text)
                    # Native re-import validates every populated array and count.
                    run(label+'-freeze', ['roundtrip-model-dev', activated, frozen])
                    for key in ('outer_k_rank1_weights', 'moment_coeffs', 'species_coeffs'):
                        values = numbers(frozen.read_text(), key)
                        if not len(values) or not np.isfinite(values).all() or not np.all(values != 0):
                            raise AssertionError('inactive/nonfinite fixture block: ' + key)
                    if kind == 'effect':
                        for key in ('two_layer_gate_weights', 'two_layer_gate_radial_coeffs',
                                    'two_layer_gate_k_rank_weights', 'two_layer_gate_type_coeffs',
                                    'two_layer_gate_additive_coeffs'):
                            values = numbers(frozen.read_text(), key)
                            if not len(values) or not np.isfinite(values).all() or not np.all(values != 0):
                                raise AssertionError('zero effect fixture block: ' + key)
                    model_sha = sha(frozen)
                    # Small L2 representatives check every linear column for
                    # shared and species heads, not a checksum or one column.
                    if kind == 'effect' and lmax == 2:
                        checked = run(label+'-components-fd', [
                            'check-linear-components-fd-dev', frozen, cfg,
                            '--max-configs=1', '--max-atoms=3', '--coeff-start=0', '--coeff-end=0',
                            '--displacement=2e-5', '--abs-tolerance=5e-8', '--rel-tolerance=5e-5'],
                            env_options(args.fd_env))
                        fields = dict(re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)', checked))
                        if int(fields.get('checked_components', '0')) <= 0:
                            raise AssertionError('finite differences checked no components')
                        readout_model = frozen
                        chart = 'original species additive readout'
                        if readout == 'shared':
                            readout_model = root/(label + '-unit-center-readout-check.sus2')
                            readout_model.write_text(reconstruction_chart(frozen.read_text(), readout))
                            chart = ('separate unit-center Legacy reconstruction check; original nonunit-center '
                                     'frozen model retained for FD/features/normal equations')
                        readout_hash = sha(readout_model)
                        run(label+'-readout', ['check-linear-readout-dev', readout_model, cfg,
                            '--max-configs=1', '--max-atoms=3', '--check-stress',
                            '--abs-tolerance=2e-9', '--rel-tolerance=2e-9'], env_options(args.fd_env))
                        if sha(readout_model) != readout_hash:
                            raise AssertionError('readout check changed its frozen chart')
                        report['finite_differences'].append(dict(case=label, fields=fields,
                            environment=env_options(args.fd_env), reconstruction_chart=chart,
                            reconstruction_model=str(readout_model), reconstruction_sha256=readout_hash))
                    if sha(frozen) != model_sha or sha(cfg) != report['cfg_sha256']:
                        raise AssertionError('native checks changed frozen fixture')
                    report['cases'].append(dict(name=label, model=str(frozen), cfg=str(cfg),
                        lmax=lmax, kind=kind, readout=readout, model_sha256=model_sha,
                        topology_options=options))
                    save()
        if len(report['cases']) != 16 or len(report['finite_differences']) != 2:
            raise AssertionError('incomplete fixture qualification matrix')
        if sha(binary) != report['binary_sha256']:
            raise AssertionError('native binary changed during fixture generation')
        report['status'] = 'PASS'
        print(json.dumps({'status': 'PASS', 'cases': 16, 'fd_effect_readouts': ['shared', 'species'],
                          'cases_json': str(root/'cases.json')}))
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
