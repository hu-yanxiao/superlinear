#!/usr/bin/env python3
"""Prepare/run/summarize frozen CPU120 versus CUDA1 NVE throughput; never submit jobs.

prepare and summarize are safe on a login node. run requires a scheduler
allocation. Each immutable attempt contains three independent same-seed runs.
Missing, pending and failed results never become zero throughput or PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import threading
import time

BASE = Path('/work/phy-weigw/20260321_Test')
CPU = '/work/phy-weigw/app/superlinear/lmp'
CPU_SHA = '4155408b7c98563ae1059ee618025b47b38eafe284468b26e2da3e04ac45bc71'
ELEMENTS = ['H', 'C', 'N', 'O', 'S', 'Cl', 'K', 'Fe']
MASSES = [1.00794, 12.0107, 14.0067, 15.9994, 32.065, 35.453, 39.0983, 55.845]
CARBON = BASE / 'superlinear-carbon-diamond-speed-20261019'
MODELS = {
    'static': (str(CARBON / 'static.sus2'), '08087d1a7ba1cb70c9d7a4bf139bbe716156807ce8880edd9537544a1bcf7097'),
    'effect': (str(CARBON / 'effect.sus2'), '7f9a6495a0bc7229b4e1c45d3c29eea3c1feb880376e21218e9747ef2d13743f'),
    'fecn6': (str(BASE / 'fecn6-k4-water-scaling-10ps-20261016/frozen/model.mtp'), 'b7c488b4e1899054234b063749aded0274da049c0ec7e35072101b4dd538a6f4'),
}
WATER = str(BASE / 'superlinear-lammps-cpu-20261018/validation-inputs/water.data')
WATER_SHA = '4622d00893a7ada2478f12d06f593201f6e0ebff08e68b629af64526009779fc'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def manifest():
    cases = []
    for family in ('static', 'effect'):
        for replicate, warm, steps in (((5, 5, 5), 200, 2000), ((10, 10, 12), 50, 500),
                                       ((23, 23, 24), 20, 100), ((50, 50, 50), 10, 30)):
            atoms = 8 * math.prod(replicate)
            cases.append(dict(id=f'carbon-{family}-{atoms}', family=family, atoms=atoms,
                              replicate=list(replicate), warmup=warm, steps=steps))
    for replicate, warm, steps in (((1, 1, 1), 200, 2000), ((2, 2, 2), 50, 500), ((4, 4, 4), 20, 100)):
        atoms = 1736 * math.prod(replicate)
        cases.append(dict(id=f'fecn6-{atoms}', family='fecn6', atoms=atoms,
                          replicate=list(replicate), warmup=warm, steps=steps))
    return dict(schema=1, status='PREPARED_NOT_RUN', models=MODELS, cpu_binary=CPU,
                cpu_sha256=CPU_SHA, water_data=WATER, water_sha256=WATER_SHA,
                protocol=dict(ensemble='NVE', temperature_K=330, timestep_ps=0.0005,
                              velocity_seed=172900, neighbor_skin_A=1.0,
                              neighbor='every 1 delay 0 check yes', thermo_every=10,
                              tabstep_A=0.001, repeats=3, warmup_excluded=True,
                              independent_identical_seed=True),
                backends=dict(cpu120=dict(queue='1t81c', ranks=120, omp=1, single_node=True),
                              cuda1=dict(queue='gpu-phy-zhangwq', ranks=1, gpus=1, exclusive=True)),
                cases=cases, radial_modes=['direct', 'table'],
                comparison='speedup = median_CPU120_loop_seconds / median_CUDA1_loop_seconds',
                npt_validation='Separate correctness/stability protocol; never combined with NVE throughput')


def input_text(plan, case, mode, backend, npt=False):
    model = plan['models'][case['family']][0]
    style = 'sus2/embedded' if backend == 'cpu120' else 'sus2/embedded/kk'
    p = plan['protocol']
    lines = ['units metal', 'atom_style atomic', 'boundary p p p', 'newton on', 'atom_modify map array']
    if case['family'] == 'fecn6':
        lines += [f'read_data "{plan["water_data"]}"', 'replicate ' + ' '.join(map(str, case['replicate']))]
        elements, masses = ELEMENTS, MASSES
    else:
        nx, ny, nz = case['replicate']
        lines += ['lattice diamond 3.56054', f'region system block 0 {nx} 0 {ny} 0 {nz}',
                  'create_box 1 system', 'create_atoms 1 box']
        elements, masses = ['C'], [12.0107]
    lines += [f'mass {i+1} {v:.17g}' for i, v in enumerate(masses)]
    lines += [f'pair_style {style} model_elements {",".join(elements)} radial_mode {mode} tabstep {p["tabstep_A"]}',
              f'pair_coeff * * "{model}" ' + ' '.join(elements),
              f'neighbor {p["neighbor_skin_A"]} bin', 'neigh_modify ' + p['neighbor'],
              f'velocity all create {p["temperature_K"]} {p["velocity_seed"]} mom yes rot yes dist gaussian',
              f'timestep {p["timestep_ps"]}',
              ('fix integrate all npt temp 330 330 0.05 iso 1 1 0.5' if npt else 'fix integrate all nve'),
              f'thermo {p["thermo_every"]}', 'thermo_style custom step atoms temp pe ke etotal press vol',
              'thermo_modify format float %.17g flush yes', f'run {case["warmup"]}',
              'reset_timestep 0', f'run {case["steps"]}', 'print "BENCHMARK_COMPLETE"']
    return '\n'.join(lines) + '\n'


def prepare(directory):
    require(not directory.exists(), f'Preserve existing plan: {directory}')
    directory.mkdir(parents=True)
    plan = manifest()
    for case in plan['cases']:
        for mode in plan['radial_modes']:
            for backend in plan['backends']:
                target = directory / case['id'] / mode / backend
                target.mkdir(parents=True)
                (target / 'in.benchmark').write_text(input_text(plan, case, mode, backend))
        if case['atoms'] in (1000, 1736):
            check = dict(case, warmup=20, steps=100)
            for mode in plan['radial_modes']:
                for backend in plan['backends']:
                    target = directory / 'npt-validation' / case['id'] / mode / backend
                    target.mkdir(parents=True)
                    (target / 'in.validation').write_text(input_text(plan, check, mode, backend, True))
    save(directory / 'manifest.json', plan)
    return plan


def probe(command):
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8)
        return dict(returncode=result.returncode, stdout=result.stdout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=None, error=str(error))


def allocation(backend):
    require(os.environ.get('LSB_JOBID'), 'run requires an LSF compute allocation; never run on login node')
    count = int(os.environ.get('LSB_DJOB_NUMPROC', '0'))
    queue = os.environ.get('LSB_QUEUE')
    hosts = os.environ.get('LSB_MCPU_HOSTS', '').split()
    require(len(hosts) == 2 and int(hosts[1]) == count, 'Require exactly one allocated node')
    require(count >= (120 if backend == 'cpu120' else 1), 'Insufficient CPU allocation')
    require(queue == ('1t81c' if backend == 'cpu120' else 'gpu-phy-zhangwq'), 'Wrong benchmark queue')
    result = dict(job_id=os.environ['LSB_JOBID'], queue=queue, node=hosts[0], slots=count,
                  cpu=probe(['lscpu']), gpu=None,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_allocation=os.environ.get('LSB_GPU_REQ'))
    if backend == 'cuda1':
        result['gpu'] = probe(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'])
        require(result['gpu']['returncode'] == 0, 'CUDA benchmark requires actual nvidia-smi device evidence')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible:
            require(len(visible.split(',')) == 1 and visible not in ('-1', 'NoDevFiles'), 'Require one visible allocated GPU')
    return result


class MemorySampler:
    """Legacy process-group RSS sampling, not all-MPI-process memory.

    GPU framebuffer observations belong to the separate verified UUID sampler;
    MPI children can escape the launcher's PGID, so PID-filtered GPU sums are
    not a reliable substitute. Keep the old GPU field explicitly unknown.
    """
    def __init__(self, pgid, cuda):
        self.pgid, self.cuda = pgid, cuda
        self.stop_event = threading.Event()
        self.samples, self.rss_kib, self.gpu_mib = 0, None, None
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            try:
                rows = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,rss='], text=True).splitlines()
                owned = [(int(p), int(r)) for p, g, r in map(str.split, rows) if int(g) == self.pgid]
                if owned:
                    self.rss_kib = max(self.rss_kib or 0, sum(r for _, r in owned))
                    self.samples += 1
            except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(0.25)

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        return dict(sample_interval_s=0.25, samples=self.samples, sampled_group_peak_rss_KiB=self.rss_kib,
                    sampled_group_peak_gpu_MiB=self.gpu_mib, exact_peak=False,
                    attribution='RSS sum of same-process-group PIDs; MPI daemons/children escaping the group are not counted',
                    gpu_group_observation='UNKNOWN; PGID-filtered GPU sums are not collected')


def parse_log(text, case, ranks, cuda=False):
    require('BENCHMARK_COMPLETE' in text, 'Missing terminal completion marker')
    require(not re.search(r'(?im)^.*(?:\bERROR\b|\bnan\b|\b[+-]?inf\b|Lost atoms).*$', text),
            'Non-finite thermodynamics or LAMMPS error')
    loops = re.findall(r'Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms', text)
    require(len(loops) == 2, f'Expected warmup and timed windows, observed {len(loops)}')
    seconds, procs, steps, atoms = loops[-1]
    require(tuple(map(int, loops[0][1:])) == (ranks, case['warmup'], case['atoms']),
            'Wrong warmup run dimensions')
    seconds = float(seconds)
    require(math.isfinite(seconds) and seconds > 0, 'Invalid measured loop time')
    require((int(procs), int(steps), int(atoms)) == (ranks, case['steps'], case['atoms']), 'Wrong actual run dimensions')
    if cuda:
        require(set(re.findall(r'SUS2_KOKKOS_BACKEND\s+(\w+)', text)) == {'CUDA'},
                'Missing actual CUDA backend proof; Serial is not GPU qualification')
    tables = [dict(kind=k, bytes=int(v)) for k, v in re.findall(r'(?<![A-Za-z0-9_])(max_rank_table_bytes|bytes|SUS2_KOKKOS_TABLE_BYTES)[=\s]+(\d+)', text)]
    return dict(loop_seconds=seconds, actual_atoms=int(atoms), iterations=int(steps), ranks=int(procs),
                device_backend='CUDA' if cuda else None,
                atom_steps_per_s=int(atoms)*int(steps)/seconds, table_memory_observations=tables,
                table_memory_scope='CPU bytes is rank0 at initialization unless explicitly max_rank; no inferred all-rank sum')


def run(args):
    plan = json.loads(args.manifest.read_text())
    case = next(c for c in plan['cases'] if c['id'] == args.case)
    npt = args.validation_npt
    require(not npt or case['atoms'] in (1000, 1736), 'Short NPT validation uses the smallest fixed structures')
    if npt:
        case = dict(case, warmup=20, steps=100)
    hardware = allocation(args.backend)
    require(args.backend == 'cpu120' or args.candidate, 'CUDA execution requires --candidate')
    binary = Path(plan['cpu_binary'] if args.backend == 'cpu120' else args.candidate).resolve()
    expected = plan['cpu_sha256'] if args.backend == 'cpu120' else args.candidate_sha256
    require(expected and sha(binary) == expected, 'Binary hash mismatch or missing candidate SHA')
    require(args.backend == 'cpu120' or str(binary) != CPU, 'GPU candidate must not replace/use canonical CPU executable')
    model, digest = plan['models'][case['family']]
    require(sha(model) == digest, 'Frozen model changed')
    if case['family'] == 'fecn6':
        require(sha(plan['water_data']) == plan['water_sha256'], 'Frozen water geometry changed')
    require(not args.out.exists(), 'Preserve existing attempt; choose a new --out')
    args.out.mkdir(parents=True)
    source_binary = binary
    binary = (args.out / 'frozen-lmp').resolve()
    shutil.copy2(source_binary, binary)
    require(sha(binary) == expected, 'Executable changed during snapshot')
    report = dict(status='RUNNING', case=case, radial_mode=args.radial, backend=args.backend,
                  kind='npt-validation' if npt else 'throughput',
                  manifest_sha256=sha(args.manifest), protocol=plan['protocol'], hardware=hardware,
                  binary=str(binary), source_binary=str(source_binary), binary_sha256=expected,
                  model_sha256=digest, runs=[])
    save(args.out / 'report.json', report)
    ranks = 120 if args.backend == 'cpu120' else 1
    report['protocol'] = dict(plan['protocol'], ensemble='NPT' if npt else 'NVE')
    repeats = 1 if npt else plan['protocol']['repeats']
    for repeat in range(repeats):
        folder = args.out / f'repeat-{repeat+1}'
        folder.mkdir()
        script = folder / 'in.benchmark'
        script.write_text(input_text(plan, case, args.radial, args.backend, npt))
        command = shlex.split(args.mpiexec) + ['-n', str(ranks), str(binary)]
        if args.backend == 'cuda1':
            command += ['-k', 'on', 'g', '1', '-sf', 'kk', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on', 'comm', 'device']
        command += ['-in', str(script.resolve()), '-log', 'none']
        record = dict(repeat=repeat+1, status='RUNNING', command=command,
                      returncode=None, timed_out=False)
        report['runs'].append(record)
        save(args.out / 'report.json', report)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        sampler = None
        gpu_sampler = None
        process = None
        try:
            require(sha(model) == digest and sha(binary) == expected, 'Frozen model/binary changed between repetitions')
            if (args.backend == 'cuda1' and getattr(args, 'expected_gpu', None) is not None
                    and getattr(args, 'gpu_runtime_evidence', None) is not None):
                from kokkos_gpu_memory import GpuMemorySampler
                gpu_sampler = GpuMemorySampler(args.expected_gpu, args.gpu_runtime_evidence)
                gpu_sampler.start()  # Verified same-allocation preflight; before this Popen.
            with (folder / 'stdout.log').open('w') as log:
                process = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                sampler = MemorySampler(process.pid, False)
                sampler.thread.start()
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    record['timed_out'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
                    raise RuntimeError('MPI timeout; only owned process group terminated')
            record['returncode'] = code
            require(code == 0, f'LAMMPS exit {code}')
            record.update(parse_log((folder / 'stdout.log').read_text(), case, ranks, args.backend == 'cuda1'), status='PASS')
            if args.backend == 'cuda1' and getattr(args, 'expected_gpu', None) is not None:
                from kokkos_gpu_identity import runtime_device
                record['runtime_device'] = runtime_device((folder / 'stdout.log').read_text(), args.expected_gpu)
        except Exception as error:
            record.update(status='FAIL', error=str(error))
        finally:
            if process is not None:
                record['returncode'] = process.returncode
            record['memory'] = sampler.finish() if sampler else None
            if record['memory'] is None:
                record['memory'] = dict(sampled_group_peak_rss_KiB=None,
                    sampled_group_peak_gpu_MiB=None, exact_peak=False,
                    attribution='RSS same-process-group only; no RSS sample available')
            gpu_memory = gpu_sampler.finish() if gpu_sampler is not None else None
            record['memory'].update(gpu_device_memory=gpu_memory,
                sampled_device_peak_gpu_MiB=(gpu_memory['sampled_device_peak_used_MiB']
                    if gpu_memory and gpu_memory['attribution_verified'] else None),
                gpu_observation_status=(gpu_memory['status'] if gpu_memory else
                    'UNKNOWN_NO_VERIFIED_PREFLIGHT' if args.backend == 'cuda1' else 'NOT_REQUESTED'),
                gpu_scope='Whole verified exclusive UUID framebuffer including driver/context baseline; sampled, not exact or per-process. Unknown stays null; no PGID filtering.')
            if gpu_memory and gpu_memory['identity_errors']:
                record.update(status='FAIL',error='Assigned GPU identity changed in UUID memory samples')
            save(args.out / 'report.json', report)
        if record['status'] != 'PASS':
            break
    report['status'] = 'PASS' if len(report['runs']) == repeats and all(r['status'] == 'PASS' for r in report['runs']) else 'FAIL'
    if report['status'] == 'PASS':
        report['median_loop_seconds'] = statistics.median(r['loop_seconds'] for r in report['runs'])
        report['median_atom_steps_per_s'] = case['atoms'] * case['steps'] / report['median_loop_seconds']
    report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(args.out / 'report.json', report)
    return report


def summarize(plan, paths):
    reports = [json.loads(p.read_text()) for p in paths]
    rows = []
    for case in plan['cases']:
        for mode in plan['radial_modes']:
            row = dict(case=case['id'], atoms=case['atoms'], radial_mode=mode, cpu120=None, cuda1=None, speedup=None)
            selected = []
            for backend in plan['backends']:
                matching = [r for r in reports if r.get('kind') == 'throughput' and
                            r.get('case', {}).get('id') == case['id'] and r.get('radial_mode') == mode and r.get('backend') == backend]
                require(len(matching) <= 1, f'Duplicate attempts for {case["id"]}/{mode}/{backend}; select explicit reports')
                if matching:
                    r = matching[0]
                    if r['status'] == 'PASS':
                        require(len(r.get('runs', [])) == plan['protocol']['repeats'] and
                                all(x.get('status') == 'PASS' and x.get('returncode') == 0 and
                                    x.get('actual_atoms') == case['atoms'] and
                                    x.get('ranks') == (120 if backend == 'cpu120' else 1) and
                                    x.get('iterations') == case['steps'] for x in r['runs']),
                                'PASS report lacks three completed runs with the requested actual dimensions')
                        require(r.get('case') == case and r.get('protocol') == plan['protocol'] and
                                r.get('model_sha256') == plan['models'][case['family']][1],
                                'Result differs from frozen case/protocol/model')
                        if backend == 'cpu120':
                            require(r.get('binary_sha256') == plan['cpu_sha256'], 'Wrong production CPU baseline')
                        else:
                            require(all(x.get('device_backend') == 'CUDA' for x in r['runs']) and
                                    r.get('hardware', {}).get('gpu', {}).get('returncode') == 0,
                                    'GPU PASS report lacks executed CUDA/device proof')
                        require(math.isfinite(r['median_loop_seconds']) and r['median_loop_seconds'] > 0,
                                'Invalid PASS loop seconds')
                    row[backend] = dict(status=r['status'], hardware=r.get('hardware'), runs=r.get('runs'),
                                        median_loop_seconds=r.get('median_loop_seconds'),
                                        median_atom_steps_per_s=r.get('median_atom_steps_per_s'))
                    selected.append(r)
            row['status'] = 'NOT_RUN' if not selected else 'INCOMPLETE'
            if len(selected) == 2 and all(r['status'] == 'PASS' for r in selected):
                cpu = next(r for r in selected if r['backend'] == 'cpu120')
                gpu = next(r for r in selected if r['backend'] == 'cuda1')
                require(cpu['protocol'] == gpu['protocol'] and cpu['case'] == gpu['case'] and
                        cpu['model_sha256'] == gpu['model_sha256'] and
                        cpu['manifest_sha256'] == gpu['manifest_sha256'], 'Unmatched CPU/GPU scientific protocol')
                row.update(status='PASS', speedup=cpu['median_loop_seconds']/gpu['median_loop_seconds'])
            elif any(r['status'] == 'FAIL' for r in selected):
                row['status'] = 'FAIL'
            rows.append(row)
    return dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'INCOMPLETE', rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('run')
    for key in ('manifest', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--case', required=True); p.add_argument('--radial', choices=('direct', 'table'), required=True)
    p.add_argument('--backend', choices=('cpu120', 'cuda1'), required=True)
    p.add_argument('--candidate'); p.add_argument('--candidate-sha256'); p.add_argument('--mpiexec', default='mpirun')
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--validation-npt', action='store_true', help='Separate 20+100-step NPT validation; excluded from throughput summaries')
    p = commands.add_parser('summarize'); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reports', type=Path, nargs='*', default=[]); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.out)
    elif args.command == 'run':
        result = run(args)
    else:
        require(not args.out.exists(), 'Refusing to overwrite a saved summary')
        result = summarize(json.loads(args.manifest.read_text()), args.reports)
        save(args.out, result)
    print(json.dumps(dict(status=result['status'], output=str(args.out))))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
