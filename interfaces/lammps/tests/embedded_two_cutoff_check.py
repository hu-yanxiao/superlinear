#!/usr/bin/env python3
"""Host embedded-engine E/F/virial regression against external FP64 fixtures.

Uses embedded_efs.cpp's explicit forward/reverse ghost exchange for domains=2;
this does not qualify real LAMMPS MPI. No jobs are submitted. shared-o and
grouped projections are deliberately outside the formal main support contract.
Requires only the Python standard library. Output includes every subprocess log.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def flatten(value):
    if isinstance(value, list):
        return [v for row in value for v in flatten(row)]
    return [float(value)]


def close(actual, expected, tolerance, label):
    a, b = flatten(actual), flatten(expected)
    require(len(a) == len(b), f'{label}: size mismatch')
    require(all(math.isfinite(v) for v in a+b), f'{label}: nonfinite value')
    delta = [abs(x-y) for x, y in zip(a, b)]
    require(all(d <= tolerance*(1+max(abs(x), abs(y)))
                for x, y, d in zip(a, b, delta)),
            f'{label}: max absolute error {max(delta, default=0):.12g}')
    return max(delta, default=0)


def compare(actual, expected, tolerance):
    return {key: close(actual[key], expected[key], tolerance, key)
            for key in ('energy', 'forces', 'virial', 'site_energies') if key in expected}


def read_geometry(block):
    lines = block.splitlines()
    n = int(lines[next(i for i, v in enumerate(lines) if v.strip() == 'Size')+1])
    start = next(i for i, v in enumerate(lines) if v.strip() == 'Supercell')+1
    cell = [[float(v) for v in row.split()] for row in lines[start:start+3]]
    start = next(i for i, v in enumerate(lines) if 'AtomData:' in v)+1
    keys = lines[start-1].split('AtomData:', 1)[1].split()
    types, positions = [], []
    for row in lines[start:start+n]:
        row = row.split()
        types.append(int(row[keys.index('type')]))
        positions.append([float(row[keys.index('cartes_'+a)]) for a in 'xyz'])
    return dict(cell=cell, types=types, positions=positions)


def write_geometry(path, g):
    lines = ['BEGIN_CFG', ' Size', f' {len(g["types"])}', ' Supercell']
    lines += [' '+' '.join(f'{v:.17g}' for v in row) for row in g['cell']]
    lines += [' AtomData: id type cartes_x cartes_y cartes_z']
    lines += [f' {i+1} {t} '+' '.join(f'{v:.17g}' for v in row)
              for i, (t, row) in enumerate(zip(g['types'], g['positions']))]
    path.write_text('\n'.join(lines+['END_CFG', '']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embedded', type=Path, required=True)
    p.add_argument('--reference-embedded', type=Path)
    p.add_argument('--fixtures', type=Path,
                   default=Path(__file__).resolve().parent/'fixtures'/'two_cutoff',
                   help='Fixture root containing fixtures_v2 and fixtures_hi_l (default: bundled tests)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sn-dir', type=Path, help='SnS directory with model.sus2 and two CFGs')
    args = p.parse_args()
    args.embedded = args.embedded.resolve()
    if args.reference_embedded:
        args.reference_embedded = args.reference_embedded.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', runs=[], binary=str(args.embedded),
                  binary_sha256=hashlib.sha256(args.embedded.read_bytes()).hexdigest(),
                  fixtures=str(args.fixtures.resolve()),
                  reference_binary=str(args.reference_embedded) if args.reference_embedded else None,
                  sn87_enabled=bool(args.sn_dir),
                  scope='standalone CPU; simulated ghost communication; no real MPI',
                  excluded=['adaptive shared-o', 'adaptive grouped'])
    report_path = out/'qualification.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    def check(name, body):
        record = dict(name=name)
        report['runs'].append(record)
        try:
            record.update(status='PASS', **body())
        except Exception as error:
            record.update(status='FAIL', error=str(error))
        save()
    serial = 0
    def evaluate(model, geometry, domains=1, mode='direct', budget='auto', binary=None,
                 reverse_stream='1'):
        nonlocal serial
        serial += 1
        cfg = out/f'geometry-{serial:05d}.cfg'
        write_geometry(cfg, geometry)
        command = [str(binary or args.embedded), str(model), str(cfg), str(domains),
                   mode, '.001', budget, 'present']
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=90,
                                env=dict(os.environ, SUS2_SL_REVERSE_STREAM=reverse_stream))
        (out/f'eval-{serial:05d}.log').write_text(result.stdout)
        require(result.returncode == 0, f'{command}: {result.stdout[-1200:]}')
        def parse(marker):
            return json.loads(next(v.split(marker, 1)[1] for v in result.stdout.splitlines()
                                   if v.startswith(marker)))
        return parse('NATIVE_EFS_JSON '), parse('RADIAL_STATS_JSON ')
    models = []
    for directory, names in [('fixtures_v2', ['static', 'effect', 'adaptive_full', 'adaptive_shared-n']),
                             ('fixtures_hi_l', ['effect_l3k4r3o3'])]:
        root = args.fixtures.resolve()/directory
        manifest = json.loads((root/'manifest.json').read_text())
        for filename, digest in manifest.items():
            require(hashlib.sha256((root/filename).read_bytes()).hexdigest() == digest,
                    f'External FP64 fixture changed: {root/filename}')
        blocks = re.findall(r'BEGIN_CFG.*?END_CFG', (root/'geometries.cfg').read_text(), re.S)
        geometries = [read_geometry(b) for b in blocks]
        for name in names:
            model = root/(name+'.mtp')
            oracle = json.loads((root/(name+'.efs.json')).read_text())
            require(oracle['dtype'] == 'float64', 'Oracle is not FP64')
            models.append((name, model, geometries))
            for entry in oracle['records']:
                index = entry['config']
                expected = dict(energy=entry['energy'], forces=entry['forces'],
                                virial=entry['plus_stress'])
                for domains in (1, 2):
                    for mode, budget in [('direct', 'auto'), ('table', 'auto'), ('table', '0')]:
                        def body(model=model, g=geometries[index], expected=expected,
                                 mode=mode, budget=budget, domains=domains):
                            actual, stats = evaluate(model, g, domains, mode, budget)
                            error = compare(actual, expected, 2e-6 if mode == 'table' else 1e-9)
                            close(sum(actual['site_energies']), actual['energy'], 1e-12, 'site sum')
                            close([sum(row[a] for row in actual['forces']) for a in range(3)],
                                  [0, 0, 0], 1e-10, 'force conservation')
                            # Mixed frame contains near and angular-far edges. Ensure table
                            # coverage/fallback tests cannot pass through unused options.
                            if index == 0 and mode == 'table':
                                if budget == 'auto':
                                    require(stats['hits'] > 0 and stats['tables'] > 0, 'No table hits')
                                else:
                                    require(stats['bytes'] == 0 and stats['memory_fallbacks'] > 0,
                                            'Zero-budget fallback not exercised')
                            return dict(max_abs=error, radial_stats=stats)
                        check(f'oracle/{name}/frame{index}/domains{domains}/{mode}/{budget}', body)
    # Interior finite differences independently check coordinate and cell reverse.
    # Use a triaxial displacement plus off-diagonal deformation, including gate
    # environments and periodic images. Coordinates and cell transform together.
    for name, model, geometries in models:
        g = geometries[0]
        for mode in ('direct', 'table'):
            def body(model=model, mode=mode):
                text = model.read_text()
                radial = float(re.search(r'^\s*max_dist\s*=\s*(\S+)', text, re.M)[1])
                angular = float(re.search(r'^\s*angular_max_dist\s*=\s*(\S+)', text, re.M)[1])
                def pair(distance):
                    return dict(cell=[[4*radial, 0, 0], [0, 4*radial, 0], [0, 0, 4*radial]],
                                types=[0, 1], positions=[[0, 0, 0], [distance, 0, 0]])
                endpoint, _ = evaluate(model, pair(radial), 2, mode)
                outside, _ = evaluate(model, pair(radial+1e-5), 2, mode)
                far, _ = evaluate(model, pair((radial+angular)/2), 2, mode)
                error = compare(endpoint, outside, 1e-12)
                close(endpoint['forces'], [[0, 0, 0], [0, 0, 0]], 1e-12, 'radial endpoint force')
                close(endpoint['virial'], [[0]*3 for _ in range(3)], 1e-12, 'radial endpoint virial')
                far_force = max(map(abs, flatten(far['forces'])))
                require(far_force > 1e-8, 'Far-shell l=0 interaction was lost or test was vacuous')
                return dict(endpoint_max_abs=error, far_shell_max_force=far_force)
            check(f'radial-endpoint-and-far-shell/{name}/{mode}', body)
        for mode in ('direct', 'table'):
            def body(g=g, model=model, mode=mode):
                base, _ = evaluate(model, g, 2, mode)
                step = 2e-6
                f_actual, f_fd = [], []
                for atom in range(len(g['types'])):
                    for axis in range(3):
                        plus, minus = copy.deepcopy(g), copy.deepcopy(g)
                        plus['positions'][atom][axis] += step
                        minus['positions'][atom][axis] -= step
                        ep = evaluate(model, plus, 2, mode)[0]['energy']
                        em = evaluate(model, minus, 2, mode)[0]['energy']
                        f_actual.append(base['forces'][atom][axis])
                        f_fd.append(-(ep-em)/(2*step))
                v_actual, v_fd = [], []
                for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
                    def strain(sign):
                        moved = copy.deepcopy(g)
                        for rows in (moved['positions'], moved['cell']):
                            for row in rows:
                                row[a] += sign*step*row[b]
                        return evaluate(model, moved, 2, mode)[0]['energy']
                    v_fd.append(-(strain(1)-strain(-1))/(2*step))
                    v_actual.append(base['virial'][a][b])
                return dict(force_fd_error=close(f_actual, f_fd, 2e-6, 'force finite difference'),
                            virial_fd_error=close(v_actual, v_fd, 2e-6, 'virial finite difference'))
            check(f'finite-difference/{name}/{mode}', body)
        for mode in ('direct', 'table'):
            def body(g=g, model=model, mode=mode):
                optimized, _ = evaluate(model, g, 2, mode)
                reference, _ = evaluate(model, g, 2, mode, reverse_stream='0')
                return dict(max_abs=compare(optimized, reference, 1e-10))
            check(f'reference-reverse/{name}/{mode}', body)
    # Formal-main single-cutoff compatibility: remove the new field, then compare
    # against a separately compiled unchanged main binary, including site energy.
    for name, original, geometries in models:
        old = out/(name+'-single-cutoff.mtp')
        old.write_text(re.sub(r'^\s*angular_max_dist\s*=.*\n', '', original.read_text(), flags=re.M))
        equal = out/(name+'-equal-cutoffs.mtp')
        radial = re.search(r'^\s*max_dist\s*=\s*(\S+)', old.read_text(), re.M)[1]
        equal.write_text(old.read_text()+'\nangular_max_dist = '+radial+'\n')
        for domains in (1, 2):
            for mode in ('direct', 'table'):
                def body(old=old, equal=equal, g=geometries[0], domains=domains, mode=mode):
                    candidate, _ = evaluate(old, g, domains, mode)
                    explicit, _ = evaluate(equal, g, domains, mode)
                    errors = dict(equal_cutoffs=compare(explicit, candidate, 1e-12))
                    if args.reference_embedded:
                        reference, _ = evaluate(old, g, domains, mode, binary=args.reference_embedded)
                        errors['formal_main'] = compare(candidate, reference, 1e-9)
                    return dict(max_abs=errors)
                check(f'single-cutoff/{name}/domains{domains}/{mode}', body)
    if args.sn_dir:
        root = args.sn_dir.resolve()
        for cfg in ('SnS_Pnma.cfg', 'SnSe_Rocksalt.cfg'):
            g = read_geometry((root/cfg).read_text())
            for domains in (1, 2):
                for mode in ('direct', 'table'):
                    def body(g=g, domains=domains, mode=mode):
                        actual, stats = evaluate(root/'model.sus2', g, domains, mode)
                        require(args.reference_embedded, 'SnS check requires --reference-embedded')
                        reference, _ = evaluate(root/'model.sus2', g, domains, mode,
                                                binary=args.reference_embedded)
                        old_reverse, _ = evaluate(root/'model.sus2', g, domains, mode,
                                                  reverse_stream='0')
                        return dict(max_abs=compare(actual, reference, 1e-9), radial_stats=stats,
                                    reference_reverse=compare(actual, old_reverse, 1e-9))
                    check(f'sn87/{cfg}/domains{domains}/{mode}', body)
    report.update(status='FAIL' if any(r['status'] != 'PASS' for r in report['runs']) else 'PASS',
                  subprocesses=serial, failures=sum(r['status'] != 'PASS' for r in report['runs']))
    save()
    print(json.dumps(dict(status=report['status'], checks=len(report['runs']),
                          failures=report['failures'], report=str(report_path))))
    return report['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
