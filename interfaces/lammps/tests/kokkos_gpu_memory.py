"""UUID-bound exclusive-GPU memory sampling, independent of MPI process groups.

This module never submits jobs, launches a candidate, changes device settings,
or signals candidate/user processes. Only owned nvidia-smi query subprocesses
have bounded timeouts. Use a new profiling attempt, never retrofit old peaks.

GpuMemorySampler(gpu, runtime_evidence).start() must precede the profiled launch;
runtime_evidence is the actual same-allocation CUDA preflight result returned by
kokkos_gpu_identity.runtime_device(). finish() returns JSON-serializable raw
query events and sampled device/process peaks. A sampled whole-device peak is
not an exact process allocation peak and includes the idle driver baseline.
"""
import copy
import csv
import datetime
import io
import math
import re
import subprocess
import threading
import time


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _pci(value):
    m = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    _require(m is not None, 'Missing/malformed verified PCI bus identity')
    return tuple(int(x, 16) for x in m.groups())


def validate_identity(gpu, runtime_evidence):
    """Require supplied LSF/sole-NVML identity and actual CUDA confirmation."""
    _require(isinstance(gpu, dict), 'Require an already verified assigned-GPU dictionary, not ordinal zero')
    _require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu.get('uuid', '')),
             'A complete physical GPU UUID is mandatory; no device-index fallback')
    _pci(gpu.get('pci_bus_id', ''))
    _require(gpu.get('model') and isinstance(gpu.get('memory_total_MiB'), int) and gpu['memory_total_MiB'] > 0,
             'Missing verified GPU name/total memory')
    _require(gpu.get('binding_method') in ('EXPLICIT_CVD_UUID', 'EXPLICIT_CVD_ORDINAL_NAMESPACE',
                                          'LSF_SINGLE_NAMESPACE_UUID'), 'Unverified allocation binding method')
    _require(gpu.get('cuda_mps_pipe_directory') in (None, ''), 'This exclusive-card sampler requires no MPS')
    scheduler = gpu.get('scheduler_evidence', {})
    _require(scheduler.get('returncode') == 0, 'Missing successful scheduler identity query')
    text = scheduler.get('stdout', '')
    _require('GPU REQUIREMENT DETAILS:' in text and 'GPU_ALLOCATION:' in text,
             'Missing authoritative scheduler GPU sections')
    detail = text.split('GPU REQUIREMENT DETAILS:', 1)[1].split('GPU_ALLOCATION:', 1)[0]
    _require('Effective:' in detail, 'Missing effective allocation, not echoed submission text')
    effective = re.sub(r'\s+', '', detail.split('Effective:', 1)[1])
    _require(re.search(r'(?:^|:)num=1(?:/host)?(?:$|:)', effective) and
             'mode=exclusive_process' in effective and 'mps=no' in effective and 'j_exclusive=yes' in effective,
             'Require effective exclusive single GPU, no MPS, job-exclusive allocation')
    host_id = gpu.get('lsf_host_gpu_id')
    _require(isinstance(host_id, int) and host_id >= 0, 'Missing scheduler host GPU ID')
    table = text.split('GPU_ALLOCATION:', 1)[1]
    ids = []
    hosts = set()
    for line in table.splitlines():
        parts = line.split()
        if len(parts) >= 3 and not parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            hosts.add(parts[0])
            ids.append(int(parts[2]))
        elif len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            ids.append(int(parts[1]))
    _require(len(hosts) == 1 and ids and set(ids) == {host_id}, 'Allocation is not one host and one assigned card')
    external = text.split('EXTERNAL MESSAGES:')[-1].split('RESOURCE REQUIREMENT DETAILS:')[0]
    host = next(iter(hosts)).split('.')[0]
    allocated = re.findall(re.escape(host) + r':gpus=([^;\s]+)', external)
    _require(allocated and set(allocated) == {str(host_id)}, 'External assignment and GPU table differ')
    _require(isinstance(runtime_evidence, list) and runtime_evidence, 'Actual CUDA runtime evidence is mandatory')
    for r in runtime_evidence:
        _require(r.get('identity_verified') is True and r.get('device') == 0 and r.get('visible_devices') == 1,
                 'Runtime identity must already verify one selected local CUDA device')
        _require(r.get('uuid') == gpu['uuid'] and r.get('name') == gpu['model'], 'Runtime/NVML UUID or name mismatch')
        _require(isinstance(r.get('memory_bytes'), int) and 0 < r['memory_bytes'] <= gpu['memory_total_MiB'] * 1048576,
                 'Runtime memory does not fit verified assigned GPU')
    return dict(uuid=gpu['uuid'], pci_bus_id=gpu['pci_bus_id'], effective_request=effective,
                lsf_host_gpu_id=host_id, nvml_visible_index=gpu.get('nvml_visible_index'),
                cuda_runtime_ordinal=0, runtime_count=1,
                attribution='Whole assigned exclusive physical GPU, not a process-group PID sum')


def default_runner(argv, timeout):
    """Run only the monitoring query; never execute scheduler/control commands."""
    try:
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False)
    except subprocess.TimeoutExpired as error:
        decode = lambda x: x.decode(errors='replace') if isinstance(x, bytes) else (x or '')
        return dict(returncode=None, stdout=decode(error.stdout), stderr=decode(error.stderr), timed_out=True)
    except OSError as error:
        return dict(returncode=None, stdout='', stderr=str(error), timed_out=False)


def _unknown(reason, **kwargs):
    return dict(status='UNKNOWN', reason=reason, **kwargs)


def _rows(query):
    if query.get('timed_out') or query.get('returncode') != 0:
        return None
    try:
        return [r for r in csv.reader(io.StringIO(query.get('stdout', '')), skipinitialspace=True, strict=True)
                if r and any(v.strip() for v in r)]
    except (csv.Error, TypeError):
        return None


def _mib(value):
    return int(value.strip()) if re.fullmatch(r'\d+', value.strip()) else None


def parse_device_query(query, gpu):
    rows = _rows(query)
    if rows is None:
        return _unknown('query_error_or_timeout', used_MiB=None, total_MiB=None)
    if len(rows) != 1 or len(rows[0]) != 4:
        return _unknown('empty_or_malformed_device_inventory', used_MiB=None, total_MiB=None)
    uuid, pci, used, total = [x.strip() for x in rows[0]]
    try:
        matches = uuid == gpu['uuid'] and _pci(pci) == _pci(gpu['pci_bus_id'])
    except ValueError:
        matches = False
    if not matches:
        return dict(status='IDENTITY_ERROR', reason='selected_UUID_or_PCI_changed', used_MiB=None, total_MiB=None)
    used, total = _mib(used), _mib(total)
    if used is None or total is None or total != gpu['memory_total_MiB'] or used > total:
        return _unknown('unsupported_or_invalid_memory', used_MiB=None, total_MiB=total)
    return dict(status='OK', reason=None, used_MiB=used, total_MiB=total)


def parse_apps_query(query, gpu):
    rows = _rows(query)
    if rows is None:
        return _unknown('query_error_or_timeout', processes=[], sum_used_MiB=None)
    if not rows:
        return dict(status='EMPTY', reason='no_visible_compute_processes', processes=[], sum_used_MiB=None)
    processes, seen = [], set()
    unknown = False
    for row in rows:
        if len(row) != 4:
            return _unknown('malformed_compute_process_row', processes=processes, sum_used_MiB=None)
        uuid, pid, name, used = [v.strip() for v in row]
        if uuid != gpu['uuid']:
            return dict(status='IDENTITY_ERROR', reason='compute_process_UUID_changed',
                        processes=processes, sum_used_MiB=None)
        if not pid.isdigit() or int(pid) <= 0 or (uuid, int(pid)) in seen:
            return _unknown('invalid_or_duplicate_process_PID', processes=processes, sum_used_MiB=None)
        seen.add((uuid, int(pid)))
        memory = _mib(used)
        if memory is not None and memory > gpu['memory_total_MiB']:
            memory = None
        unknown |= memory is None
        processes.append(dict(gpu_uuid=uuid, pid=int(pid), process_name=name, used_gpu_memory_MiB=memory))
    return dict(status='PARTIAL_UNKNOWN' if unknown else 'OK', reason='unsupported_process_memory' if unknown else None,
                processes=processes, sum_used_MiB=None if unknown else sum(p['used_gpu_memory_MiB'] for p in processes))


class GpuMemorySampler:
    def __init__(self, assigned_gpu, runtime_evidence, *, interval_s=0.25, query_timeout_s=3.0, runner=None):
        self.identity = validate_identity(assigned_gpu, runtime_evidence)
        _require(math.isfinite(interval_s) and interval_s > 0, 'Positive finite sampling interval required')
        _require(math.isfinite(query_timeout_s) and query_timeout_s > 0, 'Positive finite query timeout required')
        self.gpu = copy.deepcopy(assigned_gpu)
        self.runtime_evidence = copy.deepcopy(runtime_evidence)
        self.interval_s, self.query_timeout_s = interval_s, query_timeout_s
        self.runner = runner or default_runner
        self.events = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._thread = None
        self._finished = False
        self._finish_result = None

    def _query(self, fields, kind):
        command = ['nvidia-smi', '-i', self.gpu['uuid'], '--query-' + kind + '=' + fields,
                   '--format=csv,noheader,nounits']
        began = time.monotonic_ns()
        try:
            raw = dict(self.runner(command, self.query_timeout_s))
        except Exception as error:
            raw = dict(returncode=None, stdout='', stderr=repr(error), timed_out=False)
        for key in ('stdout', 'stderr'):
            value = raw.get(key) or ''
            raw[key] = value.decode(errors='replace') if isinstance(value, bytes) else str(value)
        raw.setdefault('timed_out', False)
        raw.update(command=command, elapsed_s=(time.monotonic_ns()-began)/1e9)
        return raw

    def sample_once(self, phase='running'):
        _require(phase in ('baseline', 'running', 'final'), 'Invalid sampling phase')
        _require(not self._finished, 'Sampler already finished')
        with self._query_lock:
            event = dict(phase=phase, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         monotonic_ns=time.monotonic_ns())
            device = self._query('uuid,pci.bus_id,memory.used,memory.total', 'gpu')
            apps = self._query('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
            event.update(device_query=device, process_query=apps,
                         device=parse_device_query(device, self.gpu), processes=parse_apps_query(apps, self.gpu))
            with self._lock:
                self.events.append(event)
            return copy.deepcopy(event)

    def _loop(self):
        while not self._stop.is_set():
            self.sample_once('running')
            self._stop.wait(self.interval_s)

    def start(self):
        _require(self._thread is None and not self._finished, 'Sampler cannot be started twice')
        self.sample_once('baseline')
        self._thread = threading.Thread(target=self._loop, name='exclusive-GPU-memory-query', daemon=True)
        self._thread.start()
        return self

    def finish(self):
        if self._finish_result is not None:
            return copy.deepcopy(self._finish_result)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2*self.query_timeout_s+1)
        alive = self._thread is not None and self._thread.is_alive()
        if not alive:
            self.sample_once('final')
        self._finished = True
        with self._lock:
            events = copy.deepcopy(self.events)
        valid = [e['device']['used_MiB'] for e in events if e['device']['status'] == 'OK']
        run = [e['device']['used_MiB'] for e in events if e['phase'] != 'baseline' and e['device']['status'] == 'OK']
        baseline = [e['device']['used_MiB'] for e in events if e['phase'] == 'baseline' and e['device']['status'] == 'OK']
        processes = [e['processes']['sum_used_MiB'] for e in events
                     if e['phase'] != 'baseline' and e['processes']['status'] == 'OK']
        identity_errors = sum(e[k]['status'] == 'IDENTITY_ERROR' for e in events for k in ('device', 'processes'))
        unknown_device = sum(e['device']['status'] != 'OK' for e in events)
        process_valid = sum(e['processes']['status'] == 'OK' for e in events)
        status = ('INCOMPLETE' if alive else 'IDENTITY_ERROR' if identity_errors else 'UNKNOWN' if not run
                  else 'PARTIAL' if unknown_device else 'OBSERVED')
        self._finish_result = dict(schema='sus2-exclusive-gpu-memory-v1', status=status,
                                  identity=self.identity, runtime_evidence=self.runtime_evidence,
                                  sample_interval_s=self.interval_s, exact_peak=False,
                                  baseline_device_used_MiB=baseline[0] if baseline else None,
                                  sampled_device_peak_used_MiB=max(run) if run else None,
                                  sampled_device_peak_including_baseline_MiB=max(valid) if valid else None,
                                  sampled_process_peak_sum_MiB=max(processes) if processes else None,
                                  device_valid_samples=len(valid), unknown_device_samples=unknown_device,
                                  process_valid_samples=process_valid,
                                  process_unknown_or_empty_samples=len(events)-process_valid,
                                  process_observation_status=('UNKNOWN' if not process_valid else
                                                              'PARTIAL' if process_valid<len(events) else 'OBSERVED'),
                                  identity_errors=identity_errors, sampler_thread_still_running=alive,
                                  attribution_verified=bool(run) and not identity_errors and not alive,
                                  pgid_filter_applied=False, events=events,
                                  scope='Whole assigned exclusive GPU framebuffer usage, including driver/context baseline; sampled, not exact or incremental process allocation. Empty/error samples are unknown, never zero.')
        return copy.deepcopy(self._finish_result)
