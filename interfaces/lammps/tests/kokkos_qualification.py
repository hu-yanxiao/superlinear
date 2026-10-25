#!/usr/bin/env python3
"""Run the established native-oracle MPI matrix through the Kokkos pair.

The original CPU qualification equations, geometries and tolerances are reused.
CUDA must prove its actual backend on every positive LAMMPS launch. Serial is
reported HOST_ONLY and cannot qualify CUDA. This script never submits a job.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

import cpu_qualification as cpu
from kokkos_benchmark import BASE, CARBON, ELEMENTS, MODELS, allocation, probe, require, save, sha
from kokkos_oracle_binding import validate_reference, MATRIX320
from kokkos_gpu_identity import assigned_gpu, runtime_device


def coverage(matrix, matrix_dir):
    cases = matrix['cases']
    specs = [c['spec'] for c in cases]
    def values(key):
        return sorted({s[key] for s in specs if key in s}, key=str)
    model_shapes = []
    for case in cases:
        path = matrix_dir / Path(case['model']).name
        require(path.is_file(), f'Missing matrix model: {path}')
        require(not case.get('sha256') or sha(path) == case['sha256'], f'Matrix model changed: {path}')
        text = path.read_text()
        shape = {}
        for field in ('sh_l_max', 'sh_k_max', 'outer_k_rank', 'two_layer_gate_k_rank'):
            match = re.search(r'^\s*' + field + r'\s*=\s*(\d+)', text, re.M)
            if match:
                shape[field] = int(match[1])
        model_shapes.append(dict(index=case['index'], shape=shape))
    checks = dict(families=set(values('family')) >= {'static', 'legacy', 'pure', 'adaptive', 'plain'},
                  scaling=set(values('scale')) >= {'L', 'K', 'LK', 'pair'},
                  readout=set(values('head')) >= {'shared', 'species'},
                  projection=set(values('projection')) >= {'full', 'shared-n'},
                  activation=set(values('activation')) >= {'none', 'tanh'},
                  source=set(values('source')) >= {'shared', 'species'},
                  l_1_through_4=set(values('l')) >= {1, 2, 3, 4},
                  independent_gate_warp=True in values('independent'),
                  joint=True in values('joint'), tensor=any(s.get('message', 0) >= 1 for s in specs),
                  zbl=True in values('zbl'),
                  odd_dimensions=any(v > 1 and v % 2 for item in model_shapes for v in item['shape'].values()))
    return dict(status='COMPLETE_INPUT_COVERAGE' if all(checks.values()) else 'INCOMPLETE_INPUT_COVERAGE',
                required_checks=checks, spec_axes={k: values(k) for k in ('family', 'scale', 'head', 'projection',
                'rank', 'activation', 'source', 'l', 'message', 'joint', 'independent', 'basis', 'zbl')},
                model_shapes=model_shapes, cases=len(cases),
                executions=['all matrix models x MPI1/2/4 x Direct/Table', 'periodic self images',
                            'MPI ownership migration', 'native final geometry E/F/global virial',
                            'short NVE conservation', 'rank-local rejection without deadlock'],
                communication=['forward feature payload', 'reverse signal adjoints', 'reverse atom forces'],
                gate='Input coverage is not an executed PASS; every requested positive/negative check must finish')


def backend_evidence(text, backend, needs_reverse=False, expected_device=None):
    marker = re.findall(r'SUS2_KOKKOS_BACKEND\s+(\w+)', text)
    require(marker, 'Missing SUS2_KOKKOS_BACKEND marker')
    if backend == 'cuda':
        require(set(marker) == {'CUDA'}, f'Expected actual CUDA backend, observed {marker}')
    else:
        require(set(marker) <= {'Serial', 'OpenMP'}, f'Host control unexpectedly used {marker}')
    counters = [dict(signal_reverse=int(a), force_exports=int(b)) for a, b in
                re.findall(r'SUS2_KOKKOS_COMM\s+signal_reverse=(\d+)\s+force_exports=(\d+)', text)]
    if needs_reverse:
        require(any(c['signal_reverse'] > 0 for c in counters), 'No observed reverse signal communication')
        require(any(c['force_exports'] > 0 for c in counters), 'No observed ghost force exports to LAMMPS')
    device = runtime_device(text, expected_device) if backend == 'cuda' and expected_device is not None else []
    return dict(backend=marker, communication_counters=counters, runtime_device=device,
                force_exports_are_not_mpi_counts=True,
                force_reverse_validation='Native E/F/global-virial agreement across MPI1/2/4 and ownership migration; exports marker alone is not proof',
                gpu_execution_observed=backend == 'cuda', host_control=backend != 'cuda')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lammps', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--oracle-binding-audit', type=Path)
    parser.add_argument('--extension-matrix', type=Path, help='Original-oracle frozen57 matrix; never regenerated on GPU')
    parser.add_argument('--backend', choices=('cuda', 'serial'), default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-cases', type=int, default=0, help='0=all; a subset never earns full qualification')
    parser.add_argument('--fail-fast', action='store_true', help='Stop after the first failed launch/check; preserve unexecuted slots as NOT_RUN')
    parser.add_argument('--ranks', default='1,2,4')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--extra-model', type=Path, default=Path(MODELS['fecn6'][0]))
    parser.add_argument('--extra-cfg', type=Path, default=BASE / 'superlinear-lammps-cpu-20261018/validation-inputs/water.cfg')
    parser.add_argument('--extra-elements', default=','.join(ELEMENTS))
    parser.add_argument('--frozen-carbon-dir', type=Path, default=CARBON)
    args = parser.parse_args()
    args.matrix_dir = args.matrix_dir.resolve()
    matrix = json.loads((args.matrix_dir / 'matrix.json').read_text())
    audit = coverage(matrix, args.matrix_dir)
    if args.prepare_only:
        require(not args.out.exists(), 'Preserve saved qualification plan')
        args.out.mkdir(parents=True)
        save(args.out / 'matrix-plan.json', dict(status='PREPARED_NOT_RUN', coverage=audit,
                                                matrix_sha256=sha(args.matrix_dir / 'matrix.json')))
        print(json.dumps(dict(status='PREPARED_NOT_RUN', coverage=audit['status'])))
        return 0
    require(args.lammps and args.native, '--lammps and --native are required to execute')
    reference = validate_reference(args.native, args.oracle_binding_audit, args.extension_matrix)
    if args.backend == 'cuda':
        require(reference['mode'] == 'VALIDATED_PORTABLE', 'New CUDA attempts require the independently audited portable native reference')
        require(sha(args.matrix_dir / 'matrix.json') == MATRIX320, 'Frozen core320 matrix changed')
    require(sha(args.extra_model) == MODELS['fecn6'][1], 'Frozen FeCN6 model changed')
    require(sha(args.extra_cfg) == 'c63ffa1dfeb47841604fe118b7d50af3d8e9e55fe8df7abd882078ccb790852e',
            'Frozen FeCN6 CFG changed')
    require(os.environ.get('LSB_JOBID'), 'Qualification requires a compute allocation, never a login node')
    hardware = dict(job_id=os.environ['LSB_JOBID'], queue=os.environ.get('LSB_QUEUE'),
                    hosts=os.environ.get('LSB_MCPU_HOSTS'), cpu=probe(['lscpu']))
    if args.backend == 'cuda':
        hardware = allocation('cuda1')
    gpu = assigned_gpu(hardware) if args.backend == 'cuda' else None
    ranks = [int(v) for v in args.ranks.split(',')]
    require(set(ranks) >= {1, 2, 4}, 'Full qualification requires MPI1/2/4')
    cpu.validate_ranks(ranks)
    real_class = cpu.Qualification
    instances = []

    class KokkosQualification(real_class):
        def __init__(self, inherited_args):
            inherited_args.pair_style = 'sus2/embedded/kk'
            super().__init__(inherited_args)
            instances.append(self)
            self.report.update(backend=args.backend, hardware=hardware, coverage=audit,
                               gpu_qualified=False, fail_fast=args.fail_fast,
                               native_reference_binding=reference,
                               allocated_gpu=gpu,
                               signal_reverse_distinct_from_force_reverse=True)
            self.save()

        def run(self, *values, **options):
            super().run(*values, **options)
            if args.fail_fast and self.report['runs'][-1]['status'] != 'PASS':
                raise RuntimeError('Fail-fast: ' + self.report['runs'][-1]['name'] + ': ' +
                                   self.report['runs'][-1].get('error', 'failed check'))

        def process(self, argv, cwd, log, timeout):
            command = [str(v) for v in argv]
            candidate = str(self.args.lammps)
            if candidate not in command:
                return super().process(command, cwd, log, timeout)
            index = command.index(candidate) + 1
            launch = ['-k', 'on', 'g', '1'] if args.backend == 'cuda' else ['-k', 'on', 't', '1']
            # LAMMPS forbids the global full/newton-on combination. The pair
            # independently requests REQ_FULL; default half here preserves
            # standard newton-on atom-force reverse communication.
            launch += ['-sf', 'kk', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on', 'comm', 'device']
            command[index:index] = launch
            result = super().process(command, cwd, log, timeout)
            if result['returncode'] == 0 and 'in.qualification' in ' '.join(command):
                screen = Path(cwd) / 'screen.log'
                text = result['stdout'] + (screen.read_text() if screen.exists() else '')
                record = self.report['runs'][-1]
                gated = record['spec']['family'] in ('legacy', 'pure', 'adaptive', 'frozen-extra', 'frozen-carbon-effect')
                record['device_evidence'] = backend_evidence(text, args.backend, gated and record['ranks'] > 1, expected_device=gpu)
                self.save()
            return result

        def negative(self, case, model, ranks, scenario):
            """Same rank-local regression geometry; accept actual KK diagnostics.

            Do not require the previous CPU kernel's error wording. Original
            screen/launcher logs and the exact matched diagnostic are retained.
            """
            name = f'negative-{scenario}-mpi{ranks}'
            folder = self.out / name
            folder.mkdir()
            expected = {
                'hybrid': ('sus2/embedded does not support hybrid pair styles',
                           'sus2/embedded/kk does not support hybrid pair styles'),
                'exclude': ('neigh_modify include/exclude is not supported',),
                'overlap': ('SUS2 Kokkos encountered invalid neighbor geometry or model indices',),
            }[scenario]
            record = dict(name=name, scenario=scenario, ranks=ranks, case=case['index'], status='RUNNING')
            self.report['negative_checks'].append(record)
            self.save()
            try:
                cutoff = float(re.search(r'^\s*max_dist\s*=\s*(\S+)', model.read_text(), re.M)[1])
                boundary = 40*cutoff/ranks
                geometry = cpu.Geometry(cpu.np.diag([40*cutoff, 3*cutoff, 3*cutoff]),
                    cpu.np.asarray([[2*cutoff, 1.5*cutoff, 1.5*cutoff],
                                    [2*cutoff, 1.5*cutoff, 1.5*cutoff],
                                    [boundary-.25*cutoff, 1.5*cutoff, 1.5*cutoff],
                                    [boundary+.25*cutoff, 1.5*cutoff, 1.5*cutoff],
                                    [28*cutoff, 1.5*cutoff, 1.5*cutoff]]), cpu.np.asarray([0, 1, 0, 1, 0]))
                if scenario != 'overlap':
                    geometry.positions[1, 0] += .35*cutoff
                cpu.write_data(folder / 'system.data', geometry, 2)
                style = self.args.pair_style
                lines = ['units metal', 'atom_style atomic', 'boundary p p p', 'newton on',
                         f'processors {ranks} 1 1', 'read_data system.data', 'mass * 1.0']
                if scenario == 'hybrid':
                    lines += [f'pair_style hybrid {style} model_elements H,O radial_mode direct lj/cut {cutoff}',
                              f'pair_coeff * * {style} "{model}" H O', 'pair_coeff 1 2 lj/cut 0.05 1.0']
                else:
                    lines += [f'pair_style {style} model_elements H,O radial_mode direct', f'pair_coeff * * "{model}" H O']
                lines += ['neighbor 0.1 bin', 'neigh_modify every 1 delay 0 check no']
                if scenario == 'exclude':
                    lines += ['neigh_modify exclude type 1 2']
                script = folder / 'in.negative'
                script.write_text('\n'.join(lines + ['run 0']) + '\n')
                command = shlex.split(self.args.mpiexec) + ['-n', str(ranks), self.args.lammps,
                    '-in', script, '-log', folder / 'lammps.log', '-screen', folder / 'screen.log']
                result = self.process(command, folder, folder / 'launcher.log', self.args.negative_timeout)
                record.update({key: result[key] for key in ('returncode', 'elapsed_seconds', 'timed_out',
                                                          'process_group_killed_on_timeout')})
                require(not result['timed_out'] and result['returncode'] != 0, 'Expected prompt collective-safe rejection')
                diagnostic = result['stdout']
                for filename in ('screen.log', 'lammps.log'):
                    if (folder / filename).exists():
                        diagnostic += '\n' + (folder / filename).read_text()
                matched = next((s for s in expected if s in diagnostic), None)
                require(matched, 'Nonzero exit has no expected rejection diagnostic')
                record['matched_actual_diagnostic'] = matched
                if scenario == 'overlap':
                    failed = sorted(set(map(int, re.findall(r'ERROR on proc (\d+):', diagnostic))))
                    require(failed == [0], f'Overlap must be rejected only by owning rank0; observed {failed}')
                    record['diagnostic_ranks'] = failed
                record['status'] = 'PASS'
            except Exception as error:
                record.update(status='FAIL', error=str(error))
            self.save()
            if args.fail_fast and record['status'] != 'PASS':
                raise RuntimeError('Fail-fast: ' + name + ': ' + record.get('error', 'failed negative check'))

    cpu.Qualification = KokkosQualification
    old_argv = sys.argv
    sys.argv = [__file__, '--lammps', str(args.lammps.resolve()), '--native', str(args.native.resolve()),
                '--matrix-dir', str(args.matrix_dir), '--out', str(args.out.resolve()),
                '--pair-style', 'sus2/embedded', '--max-cases', str(args.max_cases),
                '--ranks', args.ranks, '--timeout', str(args.timeout), '--mpiexec', args.mpiexec]
    for key in ('extra_model', 'extra_cfg', 'extra_elements'):
        if getattr(args, key) is not None:
            sys.argv += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    failure = None
    try:
        code = cpu.main()
        check = instances[-1]
        geometry = cpu.restricted(cpu.read_cfg(args.frozen_carbon_dir / 'qualification64.cfg'))
        for i, family in enumerate(('static', 'effect')):
            model = args.frozen_carbon_dir / f'{family}.sus2'
            require(sha(model) == MODELS[family][1], f'Frozen carbon {family} changed')
            case = dict(index=1000010+i, spec=dict(family='frozen-carbon-'+family), _elements=['C'])
            for mode in ('direct', 'table'):
                for count in ranks:
                    check.run(case, model, geometry, count, mode)
        if any(r['status'] != 'PASS' for r in check.report['runs']):
            code = 1
        check.save()
    except Exception as error:
        code, failure = 2, str(error)
    finally:
        sys.argv, cpu.Qualification = old_argv, real_class
    path = args.out / 'qualification.json'
    if path.exists():
        report = json.loads(path.read_text())
        report['failures'] = sum(r['status'] != 'PASS' for r in report['runs'] + report['negative_checks'])
        # Enumerate the entire primary matrix, even for a smoke subset or an
        # interrupted run. No absent case is silently counted as zero or PASS.
        primary = {(r['case'], r['mode'], r['ranks']): r for r in report['runs'] if r['operation'] == 'run0'}
        planned = [c['index'] for c in matrix['cases']] + [1000000, 1000001, 1000010, 1000011]
        slots = [dict(case=c, mode=m, ranks=n, status=primary.get((c, m, n), {}).get('status', 'NOT_RUN'))
                 for c in planned for m in ('direct', 'table') for n in ranks]
        report['primary_slots'] = slots
        report['primary_counts'] = {s: sum(v['status'] == s for v in slots)
                                    for s in ('PASS', 'FAIL', 'RUNNING', 'NOT_RUN')}
        report['extension_gate'] = dict(status='NOT_RUN', required_categories=[
            'body_combinations', 'static_all_L', 'odd_R_K_ne_O', 'skin_md_sort', 'zbl_nonzero'],
            note='Core input coverage never qualifies extended GPU scope; use kokkos_complete_qualification.py')
        full = not report.get('unselected_indices') and audit['status'] == 'COMPLETE_INPUT_COVERAGE'
        if code or failure:
            report.update(status='FAIL', gpu_qualified=False, fatal_error=failure)
        elif args.backend == 'serial':
            report.update(status='HOST_ONLY', gpu_qualified=False, full_input_coverage=full)
        elif not full:
            report.update(status='PARTIAL_GPU_PASS', gpu_qualified=False, full_input_coverage=False)
        else:
            report.update(status='CORE_GPU_PASS', gpu_qualified=False, full_input_coverage=True)
        save(path, report)
        print(json.dumps(dict(status=report['status'], gpu_qualified=report['gpu_qualified'], report=str(path))))
    if failure:
        print(f'KOKKOS_QUALIFICATION_FATAL: {failure}', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
