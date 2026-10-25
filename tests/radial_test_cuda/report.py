"""Re-read completed native-oracle cases and retain raw-log/source provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_success(cases, counts, summary, failures):
    if counts != {'Direct': 146, 'Table': 87, 'Fallback': 1} or not all(c['pass'] for c in cases):
        raise ValueError(f'Incomplete or failed case set: {counts}')
    if failures or summary.get('numerical_failures', 0) or summary.get('fatal', False):
        raise ValueError('Numerical failures or fatal error cannot qualify')
    direct = [c for c in cases if c['mode'] == 'Direct']
    table = [c for c in cases if c['mode'] == 'Table']
    families = {c['basis'] for c in direct}
    direct_keys = {(c['basis'], c['B'], c['cutoff']) for c in direct}
    table_keys = {(c['basis'], c['cutoff'], c['adaptive'], c['gate'], c['map']) for c in table}
    if len(families) != 36 or len(direct_keys) != 146 or len(table_keys) != 87:
        raise ValueError('Duplicate or missing family/case slot')
    for basis in families:
        for cutoff in ('legacy', 'polynomial'):
            if (basis, 12, cutoff) not in direct_keys or (basis, cutoff, False, False, 'LK') not in table_keys:
                raise ValueError('Missing native family/cutoff slot')
            if basis != 'RBShapeev' and (basis, 65, cutoff) not in direct_keys:
                raise ValueError('Missing arbitrary-B slot')
    for basis in ('RBChebyshev_sss', 'RBChebyshev_sss_rational', 'RBLaguerre_log1p', 'RBJacobi_sss'):
        if (basis, 1, 'polynomial') not in direct_keys:
            raise ValueError('Missing B=1 slot')
    for basis in ('RBChebyshev_sss', 'RBLaguerre_log1p', 'RBJacobi_sss'):
        if (basis, 'polynomial', True, False, 'LK') not in table_keys:
            raise ValueError('Missing dynamic basis-table slot')
        for mapping in ('pair', 'K', 'L', 'LK'):
            if (basis, 'polynomial', True, True, mapping) not in table_keys:
                raise ValueError('Missing independently warped gate slot')
    expected = {'direct_cases': 146, 'table_cases': 87, 'fallback_cases': 1, 'families': 36,
                'relative_scaled_tolerance': 3e-12, 'pass': True}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected summary or tolerance')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--write-failed-report', action='store_true', help='Save diagnostic FAIL report; still exit 1')
    a = p.parse_args()
    records = {}
    for line in a.log.read_text().splitlines():
        for marker in ('RADIAL_BACKEND', 'RADIAL_CASE', 'RADIAL_SUMMARY', 'RADIAL_FAILURE'):
            if line.startswith(marker + ' '):
                records.setdefault(marker, []).append(json.loads(line[len(marker) + 1:]))
    if len(records.get('RADIAL_BACKEND', [])) != 1 or len(records.get('RADIAL_SUMMARY', [])) != 1:
        raise SystemExit('Missing or duplicate backend/summary evidence')
    backend = records['RADIAL_BACKEND'][0]
    if backend['kokkos_version'] != 40701:
        raise SystemExit('Not the canonical 4.7.1 Kokkos build')
    if a.require_cuda and not (backend['cuda'] and backend['execution_space'] == 'Cuda' and backend['kernel_cuda_arch'] > 0):
        raise SystemExit('Missing actual CUDA kernel execution evidence')
    cases = records.get('RADIAL_CASE', [])
    counts = {m: sum(c['mode'] == m for c in cases) for m in ('Direct', 'Table', 'Fallback')}
    summary = records['RADIAL_SUMMARY'][0]
    failures = records.get('RADIAL_FAILURE', [])
    status, validation_error = 'PASS', None
    try:
        validate_success(cases, counts, summary, failures)
    except ValueError as e:
        if not a.write_failed_report:
            raise SystemExit(str(e))
        status, validation_error = 'FAIL', str(e)
    sources = {}
    for relative in ('tests/radial_test_device.cpp', 'interfaces/lammps/ML-SUS2-KK/sus2_sl_kokkos_radial.h',
                     'interfaces/lammps/ML-SUS2/sus2_sl_model.cpp', 'interfaces/lammps/ML-SUS2/sus2_sl_model.h',
                     'interfaces/lammps/ML-SUS2/sus2_sl_radial.cpp', 'interfaces/lammps/ML-SUS2/sus2_sl_radial.h',
                     'interfaces/lammps/ML-SUS2/sus2_sl_radial_math.cpp', 'interfaces/lammps/ML-SUS2/sus2_sl_radial_math.h',
                     'interfaces/lammps/ML-SUS2/sus2_sl_zbl.h'):
        sources[relative] = sha(a.source / relative)
    evidence = {str(f): sha(f) for f in (a.log, a.evidence / 'environment.txt', a.evidence / 'binary.sha256',
                                       a.evidence / 'build/compile_commands.json') if f.is_file()}
    result = {'schema': 'sus2-radial-device-qualification-v1', 'status': status, 'backend': backend,
              'scope': 'standalone radial only; not Pair/MPI/performance qualification',
              'oracle': 'separate native CPU Radial implementation; CPU Table shares Hermite algorithm; not third-party physics oracle',
              'comparison': 'abs(actual-reference)<=3e-12*(1+abs(reference)); explicitly matching NaN/Inf',
              'table_direct_error_is_separate_not_pass_tolerance': True,
              'sources_sha256': sources, 'evidence_sha256': evidence, 'counts': counts,
              'summary': summary, 'cases': cases, 'numerical_failures': failures, 'validation_error': validation_error}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'{status} {a.output}: {counts}; backend={backend["execution_space"]}')
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
