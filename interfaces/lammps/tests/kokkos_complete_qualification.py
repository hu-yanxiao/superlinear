#!/usr/bin/env python3
"""Read-only evidence gate joining the complete core and supplemental CUDA runs.

No labels alone grant a PASS: model metadata, exact run slots, actual device
logs, native E/F/virial/site records, rebuilds, sorting and ZBL forces are read
again. This performs no simulations and never submits jobs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
import cpu_qualification as cpu
from kokkos_benchmark import save, sha
from kokkos_qualification import backend_evidence
from kokkos_extended_qualification import CATEGORIES, validate_inputs
from kokkos_extended_fixtures import field
from kokkos_oracle_binding import validate_reference, PORTABLE


def verify_run(record, directory, tolerance, expected_device=None):
    cpu.require(record['status'] == 'PASS', 'Unfinished/failed run: ' + record['name'])
    folder = directory / record['name']
    text = (folder / 'screen.log').read_text()
    backend_evidence(text, 'cuda', expected_device=expected_device)
    cpu.require('Loop time of ' in text, 'No completed LAMMPS run')
    thermo = {}
    for line in text.splitlines():
        if line.startswith('QUAL_THERMO '):
            fields = line.split()
            thermo[fields[1]] = np.asarray(list(map(float, fields[2:])))
    labels = ['initial'] + ([] if record['operation'] in ('run0', 'self-image') else ['final'])
    for label in labels:
        cpu.require(label in thermo and len(thermo[label]) == 8, 'Missing thermo ' + label)
        atoms = cpu.read_dump(folder / (label + '.dump'))
        native_lines = [line.split('NATIVE_EFS_JSON', 1)[1].strip()
                        for line in (folder / (label + '.native.log')).read_text().splitlines()
                        if line.startswith('NATIVE_EFS_JSON')]
        cpu.require(len(native_lines) == 1, 'Native record missing/ambiguous')
        native = json.loads(native_lines[0])
        w = thermo[label][1:7]
        virial = [[w[0], w[3], w[4]], [w[3], w[1], w[5]], [w[4], w[5], w[2]]]
        cpu.close(thermo[label][0], native['energy'], tolerance, 'energy')
        cpu.close(np.column_stack([atoms[k] for k in ('fx', 'fy', 'fz')]), native['forces'], tolerance, 'force')
        cpu.close(virial, native['virial'], tolerance, 'global virial')
        cpu.close(atoms['c_site'], native['site_energies'], tolerance, 'site energy')
        cpu.close(np.sum(atoms['c_site']), thermo[label][0], tolerance, 'site sum')
    if record['operation'] == 'nve':
        cpu.close(thermo['final'][-1], thermo['initial'][-1], 2e-6, 'NVE energy conservation')
    return folder, text


def exact_slots(runs, expected):
    keys = [(r['case'], r['mode'], r['ranks'], r['operation']) for r in runs]
    cpu.require(len(keys) == len(set(keys)), 'Duplicate qualification run slot')
    cpu.require(set(keys) == set(expected), 'Missing or extra qualification run slots')


def verify_core(path):
    report = json.loads(path.read_text())
    cpu.require(report['status'] == 'CORE_GPU_PASS' and not report['gpu_qualified'], 'Core matrix is not complete')
    cpu.require(report['backend'] == 'cuda' and report['failures'] == 0, 'Core is not native CUDA PASS')
    binding = report.get('native_reference_binding', {})
    cpu.require(report['native_sha256'] == PORTABLE and binding.get('mode') == 'VALIDATED_PORTABLE', 'Missing audited portable reference binding')
    validate_reference(Path(report['native']), Path(binding['audit_path']), Path(binding['extension_matrix']))
    cpu.require(sha(Path(report['lammps'])) == report['binary_sha256'] and
                sha(Path(report['native'])) == report['native_sha256'], 'Core executable/oracle changed')
    cpu.require(not report['unselected_indices'] and len(report['selected']) == 320, 'Core320 incomplete')
    cpu.require(report['primary_counts'] == dict(PASS=1944, FAIL=0, RUNNING=0, NOT_RUN=0), 'Primary slots incomplete')
    cases = [c['index'] for c in report['selected']] + [1000000, 1000001, 1000010, 1000011]
    expected = [(c, m, n, 'run0') for c in cases for m in ('direct', 'table') for n in (1, 2, 4)]
    # The established CPU driver adds two representative periodic/migration
    # models and one NVE model; these must remain executed, not merely planned.
    runs = report['runs']
    edges = {r['case'] for r in runs if r['operation'] == 'self-image'}
    dynamics = {r['case'] for r in runs if r['operation'] == 'nve'}
    cpu.require(len(edges) == 2 and len(dynamics) == 1, 'Missing periodic/migration/NVE representatives')
    expected += [(c, m, n, 'self-image') for c in edges for m in ('direct', 'table') for n in (1, 2, 4)]
    expected += [(c, 'table', n, 'migration') for c in edges for n in (2, 4)]
    expected += [(c, 'table', n, 'nve') for c in dynamics for n in (1, 2, 4)]
    exact_slots(runs, expected)
    for run in runs:
        verify_run(run, path.parent, report['tolerance'][run['mode']], report['allocated_gpu'])
        if run['operation'] == 'migration':
            folder = path.parent / run['name']
            initial, final = (cpu.read_dump(folder / (s + '.dump')) for s in ('initial', 'final'))
            cpu.require(np.count_nonzero(initial['proc'] != final['proc']) > 0, 'No observed ownership migration')
    negatives = report['negative_checks']
    cpu.require({(r['scenario'], r['ranks']) for r in negatives} ==
                {(s, n) for s in ('hybrid', 'exclude', 'overlap') for n in (2, 4)} and len(negatives) == 6,
                'Missing/duplicate negative regressions')
    cpu.require(all(r['status'] == 'PASS' and r['returncode'] != 0 and not r['timed_out'] and
                    r.get('matched_actual_diagnostic') for r in negatives), 'Negative rejection not proved')
    for record in negatives:
        folder = path.parent / record['name']
        diagnostic = '\n'.join((folder / f).read_text() for f in ('launcher.log', 'screen.log', 'lammps.log') if (folder / f).exists())
        cpu.require(record['matched_actual_diagnostic'] in diagnostic, 'Negative diagnostic missing in logs')
        if record['scenario'] == 'overlap':
            cpu.require(set(map(int, re.findall(r'ERROR on proc (\d+):', diagnostic))) == {0}, 'Overlap was not rank-local')
    return report


def verify_extension(path, binary_sha, native_sha):
    report = json.loads(path.read_text())
    cpu.require(report.get('schema') == 'sus2-kokkos-extension-v1', 'Wrong extension schema')
    cpu.require(report['binary_sha256'] == binary_sha and report['native_sha256'] == native_sha, 'Extension binary/oracle identity mismatch')
    cpu.require(report['status'] == 'PASS' and report['backend'] == 'CUDA' and
                report['oracle_kind'] == 'native' and report['full_extension'], 'Extension not complete native CUDA')
    binding = report.get('native_reference_binding', {})
    cpu.require(binding.get('mode') == 'VALIDATED_PORTABLE', 'Extension lacks independently audited portable reference')
    validate_reference(Path(report['native']), Path(binding['audit_path']), Path(binding['extension_matrix']))
    # Re-read actual model files and compare the complete current generator
    # plan, rather than trusting coverage_categories or self-reported shapes.
    models = {r['case']: Path(r['model']) for r in report['runs']}
    cpu.require(len({p.parent for p in models.values()}) == 1, 'Ambiguous fixture directory')
    directory = next(iter(models.values())).parent
    matrix = json.loads((directory / 'matrix.json').read_text())
    cpu.require(sha(directory / 'matrix.json') == report['fixture_matrix_sha256'], 'Fixture matrix changed')
    metadata = validate_inputs(matrix, directory)
    cpu.require(len(metadata) == 57, 'Current57 supplemental model scope changed')
    cases = matrix['cases']
    representatives = [next(c for c in cases if c['spec']['family'] == f)['index'] for f in ('static', 'legacy', 'pure', 'adaptive')]
    expected = [(c['index'], m, n, 'run0') for c in cases for m in ('direct', 'table') for n in (1, 2, 4)]
    expected += [(c, m, n, op) for c in representatives for m in ('direct', 'table') for n in (1, 2, 4)
                 for op in ('skin-cache', 'skin-rebuild', 'sorting')]
    expected += [(c['index'], m, n, 'zbl-distance-' + str(d)) for c in cases if c['spec'].get('zbl_probe')
                 for d in ((.25, .5, 1.4) if c['spec']['typewise'] else (.65, 1.4, 5.4))
                 for m in ('direct', 'table') for n in (1, 2, 4)]
    exact_slots(report['runs'], expected)
    observed = dict(rebuilds=0, mpi1_sort=False, nonzero_zbl=False, zero_zbl=False)
    by_index = {c['index']: c for c in cases}
    for run in report['runs']:
        case = by_index[run['case']]
        model = models[run['case']]
        cpu.require(sha(model) == case['sha256'] == run['model_sha256'], 'Run model changed')
        cpu.require(run['spec'] == case['spec'] and run['device_backend'] == 'CUDA' and
                    run['native_comparison'] == 'PASS', 'Run case/device/native mismatch')
        folder, text = verify_run(run, path.parent, report['tolerance'][run['mode']], report['allocated_gpu'])
        op = run['operation']
        if op in ('skin-cache', 'skin-rebuild'):
            builds = [int(v) for v in re.findall(r'Neighbor list builds =\s*(\d+)', text)]
            cpu.require(builds and (builds[-1] == 0 if op == 'skin-cache' else builds[-1] > 0), 'Wrong rebuild observation')
            initial, final = (cpu.read_dump(folder / (s + '.dump')) for s in ('initial', 'final'))
            distance = lambda d: np.linalg.norm([d[k][1] - d[k][0] for k in ('x', 'y', 'z')])
            cutoff = float(field(model.read_text(), 'max_dist'))
            cpu.require(distance(initial) > cutoff > distance(final), 'No actual cutoff crossing')
            cpu.require(max(np.max(np.abs(final[k])) for k in ('fx', 'fy', 'fz')) > 1e-10, 'Vacuous skin force')
            script = (folder / 'in.qualification').read_text()
            cpu.require('run 30' in script and 'neigh_modify every 1 delay 0 check yes' in script and
                        ('neighbor 2.0 bin' if op == 'skin-cache' else 'neighbor 0.1 bin') in script, 'Wrong skin protocol')
            observed['rebuilds'] += builds[-1]
        if op == 'sorting':
            raw = (folder / 'initial.dump').read_text().splitlines()
            start = next(i for i, v in enumerate(raw) if v.startswith('ITEM: ATOMS '))
            cols = raw[start].split()[2:]
            ids = [int(v.split()[cols.index('id')]) for v in raw[start+1:start+9]]
            cpu.require(len(ids) == 8 and ids != sorted(ids), 'Sorting did not produce nonidentity storage')
            cpu.require('atom_modify map array sort 1 0.2' in (folder / 'in.qualification').read_text(), 'Sorting not enabled')
            observed['mpi1_sort'] |= run['ranks'] == 1
        if op.startswith('zbl-distance'):
            spec = case['spec']
            cpu.require(spec.get('zero_ml') and spec.get('zbl_probe'), 'ZBL not an isolated zero-ML control')
            text_model = model.read_text()
            for key in ('moment_coeffs', 'species_coeffs', 'joint_readout_coeffs'):
                for values in re.findall(r'^\s*' + key + r'\s*=\s*\{([^}]+)\}', text_model, re.M):
                    cpu.require(all(float(v) == 0 for v in values.split(',') if v.strip()), 'ZBL ML control not zero')
            atoms = cpu.read_dump(folder / 'initial.dump')
            force = max(float(np.max(np.abs(atoms[k]))) for k in ('fx', 'fy', 'fz'))
            distance = float(op.split('-')[-1])
            active = spec['zbl'] and (not spec['typewise'] or distance == .25)
            if active:
                cpu.require(force > 1e-4, 'ZBL never actually triggered')
                observed['nonzero_zbl'] = True
            if not spec['zbl']:
                cpu.require(force < 1e-12, 'Disabled ZBL zero-ML control has force')
                observed['zero_zbl'] = True
    cpu.require(observed['rebuilds'] > 0 and all(observed[k] for k in ('mpi1_sort', 'nonzero_zbl', 'zero_zbl')), 'Missing live extension evidence')
    return dict(status='PASS', categories=CATEGORIES, verified_runs=len(expected), observations=observed)


def read_sha_receipt(path):
    records = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
        cpu.require(match and match[2] not in records, 'Malformed/duplicate SHA receipt: ' + str(path))
        records[match[2]] = match[1]
    cpu.require(records, 'Empty SHA receipt: ' + str(path))
    return records


def verify_radial(path, source, installed_receipt, candidate_sha):
    report = json.loads(path.read_text())
    cpu.require(report.get('schema') == 'sus2-radial-device-qualification-v1' and report.get('status') == 'PASS',
                'Radial qualification is not PASS')
    backend = report['backend']
    cpu.require(backend['kokkos_version'] == 40701 and backend['cuda'] and
                backend['execution_space'] == 'Cuda' and backend['kernel_cuda_arch'] > 0,
                'Radial qualification did not actually execute CUDA')
    # The build's own successful terminal and binary receipts bind this source
    # manifest to the candidate; no user-entered source SHA is accepted/needed.
    suffix = '.installed.sha256'
    cpu.require(installed_receipt.name.endswith(suffix), 'Expected actual build installed-source receipt')
    prefix = installed_receipt.parent / installed_receipt.name[:-len(suffix)]
    cpu.require(Path(str(prefix) + '.exit').read_text().strip() == '0', 'Candidate build did not finish successfully')
    binaries = read_sha_receipt(Path(str(prefix) + '.binary.sha256'))
    cpu.require(len(binaries) == 1 and next(iter(binaries.values())) == candidate_sha, 'Installed receipt belongs to another binary')
    binary_name = next(iter(binaries))
    binary_path = Path(binary_name) if Path(binary_name).is_absolute() else prefix.parent / binary_name
    cpu.require(sha(binary_path) == candidate_sha, 'Build-receipted candidate binary changed')
    installed = read_sha_receipt(installed_receipt)
    cpu.require(source is not None, '--radial-source is required to re-read the standalone source snapshot')
    source = source.resolve()
    for relative, digest in report['sources_sha256'].items():
        cpu.require(sha(source / relative) == digest, 'Standalone radial source changed: ' + relative)
        if relative.startswith('interfaces/lammps/ML-SUS2-KK/'):
            target = 'src/KOKKOS/' + Path(relative).name
        elif relative.startswith('interfaces/lammps/ML-SUS2/'):
            target = 'src/' + Path(relative).name
        else:
            continue
        cpu.require(installed.get(target) == digest, 'Standalone radial source differs from the successful candidate install: ' + target)
    required = {'interfaces/lammps/ML-SUS2-KK/sus2_sl_kokkos_radial.h'} | {
        'interfaces/lammps/ML-SUS2/' + name for name in ('sus2_sl_model.cpp', 'sus2_sl_model.h',
        'sus2_sl_radial.cpp', 'sus2_sl_radial.h', 'sus2_sl_radial_math.cpp', 'sus2_sl_radial_math.h', 'sus2_sl_zbl.h')}
    cpu.require(required <= set(report['sources_sha256']), 'Missing mathematical source provenance')
    evidence = {Path(name): digest for name, digest in report['evidence_sha256'].items()}
    cpu.require({'test.log', 'environment.txt', 'binary.sha256', 'compile_commands.json'} <= {p.name for p in evidence},
                'Missing radial log/compiler/environment/binary evidence')
    for evidence_path, digest in evidence.items():
        cpu.require(sha(evidence_path) == digest, 'Radial evidence changed: ' + str(evidence_path))
        if evidence_path.name == 'binary.sha256':
            standalone_binaries = read_sha_receipt(evidence_path)
            cpu.require(len(standalone_binaries) == 1, 'Ambiguous standalone binary receipt')
            for name, binary_digest in standalone_binaries.items():
                cpu.require(sha(Path(name)) == binary_digest, 'Standalone radial binary changed')
    logs = [v for v in evidence if v.name == 'test.log']
    cpu.require(len(logs) == 1, 'Ambiguous radial raw log')
    log = logs[0]
    # Run the frozen snapshot's complete234-slot parser again. The raw log,
    # source files and all derived case/backend/summary fields must agree.
    parser = source / 'tests/radial_test_cuda/report.py'
    with tempfile.TemporaryDirectory(prefix='sus2-radial-finalize-') as temporary:
        output = Path(temporary) / 'reparsed.json'
        result = subprocess.run([sys.executable, str(parser), '--source', str(source), '--log', str(log),
            '--evidence', str(log.parent), '--output', str(output), '--require-cuda'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        cpu.require(result.returncode == 0, 'Radial raw-log reparse failed: ' + result.stdout[-2000:])
        parsed = json.loads(output.read_text())
    for key in ('schema', 'status', 'backend', 'counts', 'summary', 'cases', 'sources_sha256'):
        cpu.require(parsed[key] == report[key], 'Radial derived evidence mismatch: ' + key)
    return dict(status='PASS', report=str(path), report_sha256=sha(path), backend=backend,
                counts=parsed['counts'], parser_sha256=sha(parser), installed_receipt=str(installed_receipt),
                installed_receipt_sha256=sha(installed_receipt), binary_receipt_sha256=sha(Path(str(prefix) + '.binary.sha256')),
                candidate_sha256=candidate_sha, sources_bound_to_candidate=True,
                oracle=report['oracle'], table_direct_error_is_separate_not_pass_tolerance=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-report', type=Path, required=True)
    p.add_argument('--extension-report', type=Path, action='append', default=[])
    p.add_argument('--radial-report', type=Path)
    p.add_argument('--radial-source', type=Path, help='Actual standalone source snapshot; hashes are read, never hand-entered')
    p.add_argument('--installed-source-receipt', type=Path, help='Actual build-cuda-rN.installed.sha256; sibling binary/exit receipts are read automatically')
    p.add_argument('--require-complete', action='store_true', help='Return nonzero for WAITING_RADIAL (otherwise preserve a successful core+extension job)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cpu.require(not a.out.exists(), 'Preserve previous qualification receipt')
    result = dict(status='INCOMPLETE', gpu_qualified=False, extension_gate={k: 'NOT_RUN' for k in CATEGORIES},
                  radial_gate=dict(status='NOT_RUN'),
                  core_report=str(a.core_report), extension_reports=list(map(str, a.extension_report)))
    try:
        core = verify_core(a.core_report)
        result.update(binary_sha256=core['binary_sha256'], native_sha256=core['native_sha256'], core_status='CORE_GPU_PASS')
        evidence = [verify_extension(v, core['binary_sha256'], core['native_sha256']) for v in a.extension_report]
        if evidence:
            result.update(status='WAITING_RADIAL', extension_gate={k: 'PASS' for k in CATEGORIES}, extension_evidence=evidence)
            missing = [name for name in ('radial_report', 'radial_source', 'installed_source_receipt') if getattr(a, name) is None]
            if missing:
                result['radial_gate']['missing'] = missing
            else:
                radial = verify_radial(a.radial_report, a.radial_source, a.installed_source_receipt, core['binary_sha256'])
                result.update(status='GPU_PASS', gpu_qualified=True, radial_gate=radial)
    except Exception as error:
        result.update(status='FAIL', error=str(error))
    save(a.out, result)
    print(json.dumps(result))
    return 0 if result['gpu_qualified'] or result['status'] == 'WAITING_RADIAL' and not a.require_complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
