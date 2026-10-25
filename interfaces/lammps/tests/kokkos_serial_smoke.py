#!/usr/bin/env python3
"""Tiny real-LAMMPS Serial controls versus the separate embedded CPU engine.

This is HOST_ONLY, not training-oracle qualification or a throughput benchmark.
No MPI library/allocation is needed: a one-rank-only launcher rejects >1 ranks.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
import numpy as np
import cpu_qualification as cpu
from kokkos_qualification import backend_evidence


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--launch':
        cpu.require(sys.argv[2:4] == ['-n', '1'], 'Serial launcher only supports exactly one rank')
        binary = sys.argv[4]
        command = [binary, '-k', 'on', 't', '1', '-sf', 'kk', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on',
                   'comm', 'device'] + sys.argv[5:]
        os.execv(binary, command)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lammps', type=Path, required=True)
    p.add_argument('--cpu-oracle', type=Path, required=True)
    p.add_argument('--matrix-dir', type=Path, required=True)
    p.add_argument('--carbon-dir', type=Path, required=True, help='Directory with static.sus2/effect.sus2')
    p.add_argument('--carbon-cfg', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    inherited = argparse.Namespace(lammps=args.lammps.resolve(), native=args.cpu_oracle.resolve(), out=args.out,
        mpiexec=shlex.join([sys.executable, str(Path(__file__).resolve()), '--launch']),
        pair_style='sus2/embedded/kk', timeout=120, tabstep=.001, direct_tolerance=1e-9, table_tolerance=2e-6)

    class Check(cpu.Qualification):
        def oracle(self, model, geometry, directory, label):
            cfg = directory / f'{label}.cfg'
            cpu.write_cfg(cfg, geometry)
            output = self.command([self.args.native, model, cfg, '1', 'direct'], directory,
                                  directory / f'{label}.cpu-embedded.log')
            return json.loads(next(line.split('NATIVE_EFS_JSON ', 1)[1] for line in output.splitlines()
                                   if line.startswith('NATIVE_EFS_JSON ')))

        def process(self, argv, cwd, log, timeout):
            candidate = str(self.args.lammps) in [str(v) for v in argv]
            if candidate:
                script = Path(cwd) / 'in.qualification'
                record = self.report['runs'][-1]
                if record['spec'].get('continuous30'):
                    text = script.read_text()
                    text = re.sub(r'^displace_atoms all move .*$',
                        'group moving id 2\ntimestep 0.0005\nfix prescribed moving move linear -40 0 0 units box', text, flags=re.M)
                    text = text.replace('neighbor 0.4 bin', 'neighbor 0.1 bin')
                    text = text.replace('neigh_modify every 1 delay 0 check no', 'neigh_modify every 1 delay 0 check yes')
                    text = text.replace('run 0\nprint "QUAL_THERMO final ', 'run 30\nprint "QUAL_THERMO final ')
                    script.write_text(text)
                # Exercise force clearing and repeated initialization in the
                # same live Pair object, retaining a second per-atom snapshot.
                with script.open('a') as stream:
                    stream.write('dump repeated all custom 1 repeated.dump id type proc x y z fx fy fz c_site\n'
                                 'dump_modify repeated sort id format float %.17g first yes\nrun 0\nundump repeated\n')
            result = super().process(argv, cwd, log, timeout)
            if candidate and result['returncode'] == 0:
                text = (Path(cwd) / 'screen.log').read_text()
                self.report['runs'][-1]['backend_evidence'] = backend_evidence(text, 'serial')
            return result

    check = Check(inherited)
    check.report.update(reference_kind='separate standalone embedded CPU engine; not native training oracle',
                        backend='Serial', gpu_qualified=False)
    carbon = cpu.restricted(cpu.read_cfg(args.carbon_cfg))
    matrix = json.loads((args.matrix_dir / 'matrix.json').read_text())
    asymmetric = cpu.restricted(cpu.read_cfg(args.matrix_dir / 'asymmetric.cfg'))
    cases = []
    for index, family in enumerate(('static', 'effect')):
        cases.append((dict(index=100000+index, spec=dict(family=family), _elements=['C']),
                      args.carbon_dir / f'{family}.sus2', carbon))
    selectors = [lambda s: s['family'] == 'adaptive' and not s.get('joint'),
                 lambda s: s['family'] == 'adaptive' and s.get('joint'),
                 lambda s: s['family'] == 'adaptive' and s.get('independent'),
                 lambda s: s['family'] == 'legacy' and s.get('gate') != s.get('outer'),
                 lambda s: s['family'] == 'adaptive' and s.get('projection') == 'shared-n' and s.get('rank') == 3]
    for select in selectors:
        case = next(c for c in matrix['cases'] if select(c['spec']))
        if any(c[0]['index'] == case['index'] for c in cases):
            continue
        cases.append((case, args.matrix_dir / Path(case['model']).name, asymmetric))
    for case, model, geometry in cases:
        for mode in ('direct', 'table'):
            check.run(case, model.resolve(), geometry, 1, mode)
            record = check.report['runs'][-1]
            if record['status'] == 'PASS':
                folder = check.out / record['name']
                original = cpu.read_dump(folder / 'initial.dump')
                repeated = cpu.read_dump(folder / 'repeated.dump')
                try:
                    record['repeated_run'] = {key: cpu.close(repeated[key], original[key], 1e-9, key)
                                              for key in ('fx', 'fy', 'fz', 'c_site')}
                except Exception as error:
                    record.update(status='FAIL', error='repeated run: ' + str(error))
                check.save()
    case, model, _ = cases[1]
    cutoff = float(re.search(r'^\s*max_dist\s*=\s*(\S+)', model.read_text(), re.M)[1])
    periodic = cpu.Geometry(np.diag(np.array([.72, .83, .94])*cutoff),
                            np.array([[.2, .2, .2]])*cutoff, np.array([0]))
    for mode in ('direct', 'table'):
        check.run(case, model.resolve(), periodic, 1, mode, 'self-image')
    # Initially there is no pair even in the 0.1A skin list. Prescribed motion
    # brings it inside the force cutoff in one uninterrupted 30-step run.
    # This is an integration/cache control, not a physical NVE trajectory.
    original = next(c for c in matrix['cases'] if c['spec']['family'] == 'adaptive' and c['spec'].get('independent'))
    model = args.matrix_dir / Path(original['model']).name
    cutoff = float(re.search(r'^\s*max_dist\s*=\s*(\S+)', model.read_text(), re.M)[1])
    moving_case = dict(original, index=110000, spec=dict(original['spec'], continuous30=True))
    moving = cpu.Geometry(np.diag([3*cutoff]*3),
                          np.array([[cutoff, cutoff, cutoff], [2*cutoff+.15, cutoff, cutoff]]), np.array([0, 1]))
    for mode in ('direct', 'table'):
        check.run(moving_case, model.resolve(), moving, 1, mode, 'migration')
        record = check.report['runs'][-1]
        if record['status'] == 'PASS':
            folder = check.out / record['name']
            try:
                initial, final = (cpu.read_dump(folder / f'{label}.dump') for label in ('initial', 'final'))
                distance = lambda atoms: float(np.linalg.norm(np.array([atoms[k][1]-atoms[k][0] for k in ('x', 'y', 'z')])))
                text = (folder / 'screen.log').read_text()
                builds = list(map(int, re.findall(r'Neighbor list builds = (\d+)', text)))
                cpu.require(distance(initial) > cutoff+.1 and distance(final) < cutoff, 'No new neighbor entered cutoff')
                cpu.require(builds and max(builds) > 0, 'No neighbor rebuild observed during 30 steps')
                cpu.require(np.linalg.norm(np.array([final[k] for k in ('fx', 'fy', 'fz')])) > 1e-12,
                            'Final new-neighbor force comparison was vacuous')
                record['continuous30'] = dict(steps=30, timestep_ps=.0005, skin_A=.1,
                    initial_distance_A=distance(initial), final_distance_A=distance(final), cutoff_A=cutoff,
                    observed_neighbor_builds=max(builds), final_cpu_reference=True,
                    note='Prescribed relative motion tests CSR rebuild/new neighbors; not an NVE stability claim')
            except Exception as error:
                record.update(status='FAIL', error='continuous30: ' + str(error))
            check.save()
    failures = [r for r in check.report['runs'] if r['status'] != 'PASS']
    check.report.update(status='FAIL' if failures else 'HOST_ONLY', failures=len(failures), gpu_qualified=False)
    check.save()
    print(json.dumps(dict(status=check.report['status'], checks=len(check.report['runs']),
                          failures=len(failures), report=str(check.out / 'qualification.json'))))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
