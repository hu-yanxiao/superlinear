#!/usr/bin/env python3
"""Sequential exclusive-CUDA1 throughput matrix, gated by real CUDA smoke.

Runs all11 cases x Direct/Table x3 repetitions without changing the frozen
scientific protocol. Does not submit jobs, rebuild code or claim full numerical
qualification. A fatal case stops the matrix; missing windows remain NOT_RUN.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import time
from types import SimpleNamespace
import cpu_qualification as cpu
import kokkos_benchmark as bench
from kokkos_qualification import backend_evidence
from kokkos_gpu_identity import assigned_gpu as bound_gpu, runtime_device, lsf_gpu_id
from kokkos_oracle_binding import validate_reference, PORTABLE
from kokkos_scheduler_readiness import scheduler_readiness

MANIFEST_SHA = '7cc245a643fb8f9cb920cb0d86dc37e7850ff9810436cf260dd3bb2c013a6492'


def validate_smoke(path, candidate_sha):
    report = json.loads(path.read_text())
    bench.require(report.get('status') in ('PARTIAL_GPU_PASS', 'CORE_GPU_PASS', 'GPU_PASS') and
                  report.get('backend') == 'cuda' and report.get('failures') == 0,
                  'Timing requires a completed actual CUDA smoke PASS')
    bench.require(report.get('binary_sha256') == candidate_sha, 'Smoke used a different candidate')
    bench.require(len(report.get('selected', [])) >= 20, 'Smoke must include at least20 matrix models')
    binding = report.get('native_reference_binding', {})
    bench.require(report.get('native_sha256') == PORTABLE and binding.get('mode') == 'VALIDATED_PORTABLE',
                  'Smoke did not use the independently audited portable native reference')
    validate_reference(Path(report['native']), Path(binding['audit_path']), Path(binding['extension_matrix']))
    runs = report.get('runs', [])
    expected = {(c, mode, ranks) for c in (1000000, 1000001, 1000010, 1000011)
                for mode in ('direct', 'table') for ranks in (1, 2, 4)}
    observed = {(r['case'], r['mode'], r['ranks']) for r in runs if r.get('operation') == 'run0'}
    expected |= {(c['index'], mode, ranks) for c in report['selected']
                 for mode in ('direct', 'table') for ranks in (1, 2, 4)}
    bench.require(expected == observed and len(observed) == sum(r.get('operation') == 'run0' for r in runs),
                  'Smoke lacks selected primary slots/frozen Carbon2/FeCN or has duplicates')
    bench.require(any(r.get('operation') == 'self-image' for r in runs) and
                  any(r.get('operation') == 'nve' for r in runs), 'Smoke lacks periodic-image or NVE controls')
    bench.require(any(r.get('operation') == 'migration' and r.get('migrated_atoms', 0) > 0 for r in runs),
                  'Smoke lacks observed ownership migration')
    negatives = report.get('negative_checks', [])
    bench.require({(r['scenario'], r['ranks']) for r in negatives} ==
                  {(s, n) for s in ('hybrid', 'exclude', 'overlap') for n in (2, 4)} and
                  all(r['status'] == 'PASS' and r['returncode'] != 0 and not r['timed_out'] for r in negatives),
                  'Smoke lacks collective-safe negative checks')
    for run in runs:
        bench.require(run['status'] == 'PASS' and run.get('metrics') and
                      run.get('device_evidence', {}).get('gpu_execution_observed'), 'Incomplete smoke run')
        backend_evidence((path.parent / run['name'] / 'screen.log').read_text(), 'cuda', expected_device=report['allocated_gpu'])
    return dict(path=str(path.resolve()), sha256=bench.sha(path), status=report['status'],
                binary_sha256=candidate_sha, positive_runs=len(runs),
                scope='CUDA smoke gate only; full core/extension/radial qualification tracked separately')


def assigned_gpu(hardware=None):
    return bound_gpu(hardware)


def ready_gpu(hardware,out):
    """Retain each current-job publication attempt before strict GPU binding."""
    path=out/'scheduler-readiness.json'
    scheduler_readiness(hardware,lsf_gpu_id,lambda value:bench.save(path,value))
    # Keep the exact successful raw query, never replace it with a fresh read
    # which could again be incomplete. assigned_gpu consumes this cached raw.
    hardware['scheduler_job']=hardware['gpu_scheduler_evidence']
    return assigned_gpu(hardware)


def require_idle_gpu(gpu):
    # A completed MPI process can linger very briefly in NVML. Bounded polling
    # permits teardown, never terminates an existing process or waits for hours.
    for attempt in range(5):
        result = bench.probe(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                              '--format=csv,noheader,nounits'])
        bench.require(result['returncode'] == 0, 'Cannot verify absence of competing GPU work')
        competing = [r for r in csv.reader(io.StringIO(result['stdout']), skipinitialspace=True)
                     if r and r[0].strip() == gpu['uuid']]
        if not competing:
            break
        if attempt < 4:
            time.sleep(.25)
    bench.require(not competing, 'Allocated timing GPU already has compute processes; no process was stopped')
    return dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                gpu_uuid=gpu['uuid'], competing_processes=competing)


def planned_windows(plan):
    return [dict(case=c['id'], radial_mode=m, repeat=i, status='NOT_RUN')
            for c in plan['cases'] for m in plan['radial_modes'] for i in (1, 2, 3)]


def runtime_preflight(plan,binary,expected_sha,gpu,out,mpiexec,timeout):
    """One untimed eight-atom setup/UUID check, never a formal timing window."""
    out.mkdir(exist_ok=False)
    case=dict(id='runtime-preflight',family='static',atoms=8,replicate=[1,1,1],warmup=0,steps=1)
    script=out/'in.preflight';script.write_text(bench.input_text(plan,case,'direct','cuda1'))
    command=shlex.split(mpiexec)+['-n','1',str(binary),'-k','on','g','1','-sf','kk',
        '-pk','kokkos','neigh','half','newton','on','comm','device','-in',str(script.resolve()),'-log','none']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    receipt=dict(status='STARTING',formal_timing_window=False,command=command,
        job_id=os.environ.get('LSB_JOBID'),expected_gpu_uuid=gpu['uuid'],cwd=str(out),
        input_sha256=bench.sha(script),binary_sha256_before=bench.sha(binary),
        model_sha256=bench.sha(plan['models']['static'][0]),
        returncode=None,timed_out=None,log_sha256=None,
        environment_subset={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY',
            'SUS2_SL_KOKKOS_GENERIC','SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE',
            'SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE','SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    target=out/'process.json';bench.save(target,receipt)
    try:
        bench.require(receipt['binary_sha256_before']==expected_sha,'Preflight candidate changed')
        bench.require(receipt['model_sha256']==plan['models']['static'][1],'Preflight frozen model changed')
        result=cpu.Qualification.process(SimpleNamespace(env=env),command,out,out/'stdout.log',min(timeout,180.))
        receipt.update({k:result[k] for k in ('returncode','timed_out','elapsed_seconds','process_group_killed_on_timeout')})
        bench.require(result['returncode']==0 and not result['timed_out'],'Untimed CUDA runtime preflight failed')
        text=(out/'stdout.log').read_text()
        bench.parse_log(text,case,1,True)  # Dimensions/backend only; no throughput is retained.
        receipt['runtime_device']=runtime_device(text,gpu)
        from kokkos_gpu_memory import validate_identity
        receipt['gpu_identity']=validate_identity(gpu,receipt['runtime_device'])
        receipt.update(status='VERIFIED_RUNTIME_PREFLIGHT',scope='Untimed same-binary/switches/allocation setup and UUID confirmation only')
    except Exception as error:
        receipt.update(status='FAIL',error=str(error));raise
    finally:
        receipt['binary_sha256_after']=bench.sha(binary)
        receipt['log_sha256']=bench.sha(out/'stdout.log') if (out/'stdout.log').exists() else None
        if receipt['binary_sha256_after']!=expected_sha:receipt.update(status='FAIL',error='Preflight binary changed')
        bench.save(target,receipt)
    bench.require(receipt['status']=='VERIFIED_RUNTIME_PREFLIGHT','Preflight identity was not verified')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--smoke-report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mpiexec', default='mpirun')
    p.add_argument('--timeout', type=float, default=14400)
    a = p.parse_args()
    bench.require(bench.sha(a.manifest) == MANIFEST_SHA, 'Use the exact CPU baseline manifest, not a regenerated variant')
    plan = json.loads(a.manifest.read_text())
    bench.require(plan == json.loads(json.dumps(bench.manifest())), 'Manifest scientific protocol changed')
    bench.require(bench.sha(a.candidate) == a.candidate_sha256 and str(a.candidate.resolve()) != bench.CPU,
                  'Candidate identity mismatch or production executable selected')
    smoke = validate_smoke(a.smoke_report, a.candidate_sha256)
    bench.require(not a.out.exists(), 'Preserve earlier timing attempts; choose a new output')
    a.out.mkdir(parents=True)
    hardware = bench.allocation('cuda1')
    gpu = ready_gpu(hardware,a.out)
    first_idle = require_idle_gpu(gpu)
    inputs = a.out / 'runinputs'
    inputs.mkdir()
    binary = (inputs / 'lmp').resolve()
    shutil.copy2(a.candidate, binary)
    shutil.copy2(a.manifest, inputs / 'manifest.json')
    bench.require(bench.sha(binary) == a.candidate_sha256, 'Candidate changed during freeze')
    windows = planned_windows(plan)
    result = dict(schema='sus2-kokkos-timing-matrix-v1', status='RUNNING', gpu_qualified=False,
                  qualification_scope='Throughput only; smoke gate is not full numerical qualification',
                  binary_sha256=a.candidate_sha256, binary=str(binary), source_binary=str(a.candidate.resolve()),
                  manifest_sha256=bench.sha(a.manifest), protocol=plan['protocol'],
                  hardware=hardware, allocated_gpu=gpu, smoke=smoke,
                  scheduler_readiness=dict(path=str(a.out/'scheduler-readiness.json'),
                      sha256=bench.sha(a.out/'scheduler-readiness.json')),
                  driver_sha256=bench.sha(__file__),
                  window_status_scope='Completed-group snapshots; active_group/report.json is authoritative for in-flight repetitions',
                  active_group=None, windows=windows, idle_observations=[first_idle], groups=[])
    path = a.out / 'matrix-report.json'
    bench.save(path, result)
    fatal = None
    try:
        preflight=runtime_preflight(plan,binary,a.candidate_sha256,gpu,a.out/'runtime-preflight',a.mpiexec,a.timeout)
        result['runtime_preflight']=dict(path=str(a.out/'runtime-preflight/process.json'),
            sha256=bench.sha(a.out/'runtime-preflight/process.json'),evidence=preflight)
        bench.save(path,result)
        for case in plan['cases']:
            for mode in plan['radial_modes']:
                result['idle_observations'].append(require_idle_gpu(gpu))
                current_gpu = assigned_gpu(hardware)
                bench.require(all(current_gpu[k] == gpu[k] for k in ('uuid', 'pci_bus_id', 'model', 'memory_total_MiB')), 'Assigned GPU identity changed within timing allocation')
                folder = a.out / 'results' / (case['id'] + '__' + mode)
                result['active_group'] = dict(case=case['id'], radial_mode=mode, report=str(folder / 'report.json'))
                for w in windows:
                    if w['case'] == case['id'] and w['radial_mode'] == mode:
                        w['status'] = 'NOT_RUN'
                bench.save(path, result)
                args = argparse.Namespace(manifest=inputs / 'manifest.json', case=case['id'], radial=mode,
                    backend='cuda1', candidate=str(binary), candidate_sha256=a.candidate_sha256,
                    validation_npt=False, out=folder, mpiexec=a.mpiexec, timeout=a.timeout,
                    expected_gpu=gpu,gpu_runtime_evidence=preflight['runtime_device'])
                group = bench.run(args)
                # Each timed positive launch must prove actual runtime UUID,
                # local device count/name/memory, not just request a CUDA pair.
                for run in group['runs']:
                    if run['status'] == 'PASS':
                        run['runtime_device'] = runtime_device((folder / ('repeat-' + str(run['repeat'])) / 'stdout.log').read_text(), gpu)
                bench.save(folder / 'report.json', group)
                result['groups'].append(dict(case=case['id'], radial_mode=mode, status=group['status'],
                                             report=str(folder / 'report.json')))
                result['active_group'] = None
                for w in windows:
                    if w['case'] == case['id'] and w['radial_mode'] == mode:
                        run = next((r for r in group['runs'] if r['repeat'] == w['repeat']), None)
                        if run:
                            w.update(status=run['status'], report=str(folder / 'report.json'))
                bench.save(path, result)
                bench.require(group['status'] == 'PASS', 'Fail-fast: throughput case failed; later windows remain NOT_RUN')
        bench.require(bench.sha(binary) == a.candidate_sha256, 'Frozen candidate changed')
    except Exception as error:
        fatal = str(error)
    result.update(status='FAIL' if fatal else 'PASS', fatal_error=fatal,
                  counts={s: sum(w['status'] == s for w in windows) for s in ('PASS', 'FAIL', 'RUNNING', 'NOT_RUN')},
                  completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    bench.save(path, result)
    print(json.dumps(dict(status=result['status'], counts=result['counts'], report=str(path))))
    return 1 if fatal else 0


if __name__ == '__main__':
    raise SystemExit(main())
