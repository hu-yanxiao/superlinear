#!/usr/bin/env python3
"""Resumable, same-core native angular ablation benchmark (Linux LSF only).

Example, a shorter diagnostic before a separate 500-CFG campaign:
  python dev_test/rd_m0_ablation_benchmark.py --binary /abs/rd_m0_benchmark \
    --training-cfg /abs/train.cfg --models-dir /abs/frozen-models \
    --out /abs/new-results --count 100 --models sharedn_sharedreadout \
    --modes energy efs gradient --backends ry ry-new hybrid rd-m0
Resume with the identical command plus --resume. All selection parameters,
input hashes, binary, runner/dependency code, and the original host/core must
match. The original CPU must remain inside the new allocation's allowed mask.
Changing count from 100 to 500 requires a new output directory.

Successful process records are committed atomically after a clean exit and
validation of timing/log/dump. Partial pairs resume at the missing backend;
failed/interrupted attempts are retained, never treated as measurements.
All gradients in pilot dumps are checked, not only checksum summaries. Every
fresh process warms up with one complete dataset pass. Pilot timings calibrate
repeats; reported paired samples must contain at least 3 measured seconds.
One too-short trial promotes the entire section to a longer common repeat
count, preserving old attempts but excluding their generation from statistics.

The old rd_m0_benchmark.py is imported only for read-only parsing/statistics;
its executable runner and existing campaign evidence are not modified.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

import numpy as np

from rd_m0_benchmark import distribution, read_dump, sha


MODEL_FILES = {
    'sharedn_sharedreadout': 'vo2-sharedn-speciesamp.sus2',
    'fullu_speciesreadout': 'vo2-fullu-speciesreadout.sus2',
}
MODES = ('energy', 'efs', 'gradient', 'efs-total')
BACKENDS = ('ry', 'ry-new', 'hybrid', 'rd-m0')
MIN_SECONDS = 3.0
ATOL = RTOL = 2e-9
SCHEMA = 'rd_m0_ablation_campaign_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now().astimezone().isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_dataset(path, count):
    # Byte-preserving selection: never parse/rewrite coordinates or E/F/S labels.
    blocks, current = [], []
    with path.open('rb') as stream:
        for line in stream:
            if line.strip() == b'BEGIN_CFG':
                require(not current, 'nested CFG in source dataset')
                current = [line]
            elif current:
                current.append(line)
                if line.strip() == b'END_CFG':
                    blocks.append(b''.join(current))
                    current = []
    require(not current, 'truncated CFG in source dataset')
    require(len(blocks) >= count, 'not enough complete source configurations')
    indices = [i*(len(blocks)-1)//(count-1) for i in range(count)]
    require(len(set(indices)) == count, 'nonunique selected indices')
    return b''.join(blocks[i] for i in indices), indices, len(blocks)


def cpu_signature(cpu):
    source = Path('/proc/cpuinfo').read_text()
    for block in source.strip().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if fields.get('processor') == str(cpu):
            keys = ('vendor_id', 'cpu family', 'model', 'model name', 'stepping',
                    'physical id', 'core id', 'microcode', 'flags')
            return {key: fields.get(key) for key in keys}
    raise RuntimeError('selected processor is absent from /proc/cpuinfo')


def process_start_ticks(pid):
    try:
        # The comm field can itself contain spaces and parentheses.
        fields = Path('/proc').joinpath(str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19]  # Field 22: starttime; fields[0] is field 3/state.
    except (OSError, IndexError):
        return None


def backend_orders(backends, pairs):
    # Williams orders balance positions and carryover; for an even number of
    # treatments each reverse order is already in the rotated set. For odd
    # counts append the explicit reverses. Persist the exact actual schedule.
    n = len(backends)
    base = [0] + [((i+1)//2 if i % 2 else n-i//2) for i in range(1, n)]
    orders = [[backends[(index+shift) % n] for index in base] for shift in range(n)]
    if n % 2 and n > 1:
        orders += [list(reversed(order)) for order in orders]
    return [orders[pair % len(orders)] for pair in range(pairs)]


class TooShort(RuntimeError):
    def __init__(self, seconds):
        self.seconds = seconds
        super().__init__('measured process duration below 3 seconds')


class Campaign:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        if args.resume:
            require(self.out.is_dir(), '--resume requires an existing campaign directory')
        else:
            self.out.mkdir(parents=True, exist_ok=False)
        self.lock = (self.out/'.runner.lock').open('a+')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another runner holds this campaign lock')
        self.binary = args.binary.resolve(strict=True)
        source_cfg = args.training_cfg.resolve(strict=True)
        source_sha = sha(source_cfg)
        payload, indices, total = select_dataset(source_cfg, args.count)
        require(sha(source_cfg) == source_sha, 'source dataset changed during selection')
        model_payloads, source_models = {}, {}
        for name in args.models:
            source = (args.models_dir/MODEL_FILES[name]).resolve(strict=True)
            model_payloads[name] = source.read_bytes()
            source_models[name] = {'path': str(source), 'sha256': digest_bytes(model_payloads[name])}
        self.data = self.out/'inputs'/('vo2-uniform-' + str(args.count) + '.cfg')
        self.models = {name: self.out/'inputs'/MODEL_FILES[name] for name in args.models}
        helper = Path(__file__).with_name('rd_m0_benchmark.py').resolve(strict=True)
        runtime_environment = {key: os.environ.get(key, '') for key in (
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'MKL_INTERFACE_LAYER', 'MKL_THREADING_LAYER',
            'OMP_PROC_BIND', 'OMP_PLACES', 'I_MPI_PIN_DOMAIN')}
        self.contract = dict(schema=SCHEMA, binary={'path': str(self.binary), 'sha256': sha(self.binary)},
            source_dataset={'path': str(source_cfg), 'sha256': source_sha, 'configs': total},
            source_models=source_models, selected_sha256=digest_bytes(payload), selected_indices=indices,
            runner_sha256=sha(Path(__file__)), helper_sha256=sha(helper),
            count=args.count, pairs=args.pairs, target_seconds=args.target_seconds,
            process_timeout=args.process_timeout, models=args.models, modes=args.modes, backends=args.backends,
            warmup_passes=1, minimum_measured_seconds=MIN_SECONDS, atol=ATOL, rtol=RTOL,
            runtime_environment=runtime_environment, orders=backend_orders(args.backends, args.pairs),
            objective={'weighting': 'vibrations', 'efs_weights': [1, .1, .01],
                       'regularization': False, 'gradient': 'full-physical', 'optimizer_steps': 0})
        self.contract_id = digest_bytes(canonical(self.contract))
        affinity = sorted(os.sched_getaffinity(0))
        require(affinity, 'no allocated processor affinity')
        manifest_path = self.out/'manifest.json'
        if args.resume:
            manifest = json.loads(manifest_path.read_text())
            require(manifest['contract'] == self.contract and manifest['contract_id'] == self.contract_id,
                    'resume contract mismatch: parameters, source/model/data/binary or runner hash changed')
            self.placement = manifest['placement']
            require(self.placement['host'] == socket.gethostname(), 'resume requires the original host')
            require(self.placement['cpu'] in affinity, 'original CPU is not in the current allocation mask')
            require(cpu_signature(self.placement['cpu']) == self.placement['cpu_signature'],
                    'resume CPU identity changed')
        else:
            for folder in ('inputs', 'logs', 'checks', 'processes'):
                (self.out/folder).mkdir()
            self.data.write_bytes(payload)
            for name, value in model_payloads.items():
                self.models[name].write_bytes(value)
            self.placement = {'host': socket.gethostname(), 'cpu': affinity[0],
                              'cpu_signature': cpu_signature(affinity[0])}
            atomic_json(manifest_path, {'contract': self.contract, 'contract_id': self.contract_id,
                                       'placement': self.placement, 'created': now()})
        os.sched_setaffinity(0, {self.placement['cpu']})
        self.identities = {str(self.binary): self.contract['binary']['sha256'],
                           str(self.data): self.contract['selected_sha256']}
        self.identities.update({str(self.models[name]): source_models[name]['sha256'] for name in args.models})
        self.check_inputs()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith('SUS2_SH_')}
        self.env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        BLIS_NUM_THREADS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
                        I_MPI_PIN='0', I_MPI_FABRICS='shm')
        summary = self.out/'summary.json'
        if args.resume and summary.exists():
            self.report = json.loads(summary.read_text())
            require(self.report['contract_id'] == self.contract_id, 'summary contract mismatch')
        else:
            self.report = dict(schema=SCHEMA, contract_id=self.contract_id, started=now(),
                manifest=str(manifest_path), placement=self.placement, configs=args.count,
                source_config_count=total, selected_indices=indices, input_sha256=self.identities,
                sections={}, process_records=[], checks=[], trials=[], aggregates={}, sessions=[],
                scope={
                    'energy': 'Energy/site energies with prebuilt neighborhoods',
                    'efs': 'Full E/F/virial with prebuilt neighborhoods',
                    'gradient': 'Actual EFS residual loss and full physical gradient; no regularizer/optimizer',
                    'efs-total': 'CalcEFS including fresh neighborhood construction'},
                limitations=[
                    'Single-core FP64 CPU latency; no MPI/GPU/optimizer convergence claim.',
                    'Every process has a full warmup; pilot calibration is excluded from paired statistics.',
                    'RSS includes input configurations and resident neighborhoods.',
                    'Resume requires the same host/core and unchanged inputs, binary, selections and scripts.',
                    'Same-core results can still be affected by external load; ratios use complete matched pairs only.'])
        self.report.update(status='RUNNING', error=None, finished=None)
        self.report['sessions'].append({'started': now(), 'jobid': os.environ['LSB_JOBID'],
                                       'command': sys.argv, 'affinity_before': affinity, 'resume': args.resume})
        self.records = []
        for path in sorted((self.out/'processes').glob('*.json')):
            record = json.loads(path.read_text())
            require(record['contract_id'] == self.contract_id, 'foreign process record: ' + str(path))
            if record['status'] == 'RUNNING':
                if record.get('pid'):
                    live_ticks = process_start_ticks(record['pid'])
                    require(live_ticks is None or live_ticks != record.get('start_ticks'),
                            'an interrupted child is still running; wait/stop it before resume: ' + str(record['pid']))
                record.update(status='INTERRUPTED', finished=now(),
                              error='Runner stopped before committing a verified clean process exit; not reused')
                atomic_json(path, record)
            if record['status'] == 'PASS':
                self.verify_record(record)
            self.records.append(record)
        self.save()

    def check_inputs(self):
        for path, expected in self.identities.items():
            require(sha(path) == expected, 'immutable benchmark input changed: ' + path)

    def verify_record(self, record):
        require(record.get('exit_code') == 0 and 'timing' in record,
                'incomplete successful process record')
        require(record['host'] == self.placement['host'] and record['cpu'] == self.placement['cpu'],
                'completed-process placement differs from the campaign core')
        for path, expected in record['artifact_sha256'].items():
            require(sha(path) == expected, 'completed-process artifact changed: ' + path)
        self.validate_timing(record['timing'], record['spec'])

    def validate_timing(self, timing, spec):
        require(timing['backend'] == spec['backend'] and timing['mode'] == spec['mode'], 'backend/mode mismatch')
        require(timing['configs'] == self.args.count and timing['max_configs'] == self.args.count,
                'configuration count mismatch')
        require(timing['repeats'] == spec['repeats'] and timing['warmup_passes'] == 1,
                'repeat/warmup count mismatch')
        require(timing['calls'] == self.args.count*spec['repeats'], 'configuration-call count mismatch')
        require(timing['atoms'] > 0 and timing['coeff_count'] > 0, 'empty workload')
        require(timing['dtype'] == 'float64' and timing['ranks'] == 1 and timing['cpu_affinity_count'] == 1,
                'not a single-core FP64 benchmark')
        for flag in ('finite', 'reference_unchanged', 'model_unchanged'):
            require(timing[flag] is True, 'failed driver guard: ' + flag)
        require(timing['regularization'] is False and timing['optimizer_steps'] == 0,
                'unexpected regularizer or optimizer')
        gradient = spec['mode'] == 'gradient'
        require(timing['physical_gradient'] == gradient
                and timing['objective_dataset_calls'] == (spec['repeats'] if gradient else 0),
                'incorrect full-gradient objective scope')
        require(timing['weighting'] == 'vibrations' and timing['efs_weights'] == [1, .1, .01],
                'wrong physical loss normalization')
        require(timing['neighborhoods_prebuilt'] == (spec['mode'] != 'efs-total'), 'wrong neighborhood scope')
        require(timing['timing_scope'] == ('efs_with_neighborhood_build' if spec['mode'] == 'efs-total' else 'kernel_only'),
                'wrong measured scope')
        require(timing['checksums_timed'] is False, 'checksums included in kernel timer')
        require(all(math.isfinite(value) for value in timing.values() if isinstance(value, (float, int))),
                'nonfinite timing/checksum')
        require(timing['seconds'] > 0 and len(timing['pass_seconds']) == spec['repeats']
                and all(math.isfinite(value) and value > 0 for value in timing['pass_seconds']),
                'invalid per-pass timings')
        require(math.isclose(sum(timing['pass_seconds']), timing['seconds'], rel_tol=1e-12, abs_tol=1e-12),
                'timer sum mismatch')

    def completed(self, logical):
        candidates = [record for record in self.records if record['logical_id'] == logical and record['status'] == 'PASS']
        require(len(candidates) <= 1, 'duplicate successful logical process: ' + logical)
        return candidates[0] if candidates else None

    def logical(self, model, mode, backend, pair=None, generation=0):
        stage = 'pilot' if pair is None else 'g' + str(generation) + '-pair' + str(pair)
        return model + '--' + mode + '--' + stage + '--' + backend

    def refresh(self):
        trials, aggregates = [], {}
        for key, section in self.report['sections'].items():
            if 'repeats' not in section:
                continue
            model, mode = key.split('/')
            rows = []
            for pair, order in enumerate(self.contract['orders']):
                for position, backend in enumerate(order):
                    record = self.completed(self.logical(model, mode, backend, pair, section['generation']))
                    if record is not None and record['timing']['seconds'] >= MIN_SECONDS:
                        rows.append({**record['timing'], 'model': model, 'pair': pair, 'position': position,
                                     'generation': section['generation'], 'logical_id': record['logical_id'],
                                     'process_record': record['record_path']})
            trials.extend(rows)
            aggregate = {'repeats': section['repeats'], 'generation': section['generation'],
                         'atoms': section['atoms'],
                         'expected_pairs': self.args.pairs, 'relative_to_ry': {}, 'by_backend': {}}
            for backend in self.args.backends:
                present = [row for row in rows if row['backend'] == backend]
                if present:
                    aggregate['by_backend'][backend] = {field: distribution([row[field] for row in present])
                        for field in ('seconds', 'us_per_atom', 'atoms_per_second', 'peak_rss_kib')}
                    # Retain the original runner's direct backend metric keys.
                    aggregate[backend] = aggregate['by_backend'][backend]
            ry = {row['pair']: row for row in rows if row['backend'] == 'ry'}
            for backend in self.args.backends:
                if backend == 'ry':
                    continue
                matched = [row for row in rows if row['backend'] == backend and row['pair'] in ry]
                ratios = [row['seconds']/ry[row['pair']]['seconds'] for row in matched]
                if ratios:
                    median = statistics.median(ratios)
                    comparison = dict(status='COMPLETE' if len(ratios) == self.args.pairs else 'PARTIAL',
                        pair_indices=[row['pair'] for row in matched], paired_time_ratio_over_ry=distribution(ratios),
                        paired_ratios=ratios, median_ratio_minus_one_percent=100*(median-1),
                        throughput_change_from_median_ratio_percent=100*(1/median-1),
                        all_pairs_same_direction=all(value > 1 for value in ratios) or all(value < 1 for value in ratios))
                    aggregate['relative_to_ry'][backend] = comparison
                    if backend == 'rd-m0':
                        aggregate['paired_time_ratio_rd_over_ry'] = comparison['paired_time_ratio_over_ry']
                        aggregate['median_ratio_minus_one_percent'] = comparison['median_ratio_minus_one_percent']
                        aggregate['all_pairs_same_direction'] = comparison['all_pairs_same_direction']
            aggregate['status'] = 'COMPLETE' if len(rows) == self.args.pairs*len(self.args.backends) else 'PARTIAL'
            aggregates[key] = aggregate
        self.report['trials'], self.report['aggregates'] = trials, aggregates
        self.report['process_records'] = self.records

    def save(self):
        self.refresh()
        self.report['updated'] = now()
        atomic_json(self.out/'summary.json', self.report)

    def execute(self, model, mode, backend, repeats=1, pair=None, generation=0):
        logical = self.logical(model, mode, backend, pair, generation)
        spec = {'model': model, 'mode': mode, 'backend': backend, 'repeats': repeats,
                'pair': pair, 'generation': generation, 'dump': pair is None}
        old = self.completed(logical)
        if old is not None:
            require(old['spec'] == spec, 'successful-process parameters differ on resume')
            self.verify_record(old)
            return old
        self.check_inputs()
        attempts = [record['attempt'] for record in self.records if record['logical_id'] == logical]
        attempt = max(attempts, default=-1) + 1
        stem = logical + '.attempt' + str(attempt)
        # Exclusive logs mean a pre-metadata interrupted attempt is retained too.
        while (self.out/'logs'/(stem + '.log')).exists() or (self.out/'processes'/(stem + '.json')).exists():
            attempt += 1
            stem = logical + '.attempt' + str(attempt)
        log = self.out/'logs'/(stem + '.log')
        dump = self.out/'checks'/(stem + '.jsonl') if spec['dump'] else None
        path = self.out/'processes'/(stem + '.json')
        command = [str(self.binary), str(self.models[model]), str(self.data), mode,
                   str(self.args.count), str(repeats), '1', str(dump) if dump else '-']
        record = {'contract_id': self.contract_id, 'logical_id': logical, 'spec': spec,
                  'status': 'RUNNING', 'attempt': attempt, 'started': now(), 'command': command,
                  'record_path': str(path), 'log': str(log), 'dump': str(dump) if dump else None,
                  'host': self.placement['host'], 'cpu': self.placement['cpu'], 'jobid': os.environ['LSB_JOBID']}
        self.records.append(record)
        atomic_json(path, record)
        self.save()
        print('START', logical, 'repeats', repeats, 'log', log, flush=True)
        start, process = time.perf_counter(), None
        try:
            with log.open('x') as stream:
                process = subprocess.Popen(command, env=dict(self.env, SUS2_SH_ANGULAR_BACKEND=backend),
                    stdout=stream, stderr=subprocess.STDOUT, cwd=self.out, start_new_session=True)
                record['pid'] = process.pid
                record['start_ticks'] = process_start_ticks(process.pid)
                atomic_json(path, record)
                record['exit_code'] = process.wait(timeout=self.args.process_timeout)
            record['process_seconds'] = time.perf_counter()-start
            require(record['exit_code'] == 0, 'driver exit ' + str(record['exit_code']) + ': ' + str(log))
            rows = [json.loads(line[len('BENCH_JSON '):]) for line in log.read_text().splitlines()
                    if line.startswith('BENCH_JSON ')]
            require(len(rows) == 1, 'missing/duplicate timing JSON: ' + str(log))
            record['timing'] = rows[0]
            self.validate_timing(record['timing'], spec)
            artifacts = [log]
            if dump:
                metadata, values = read_dump(dump)
                require(metadata['mode'] == mode and metadata['configs'] == self.args.count
                        and metadata['atoms'] == record['timing']['atoms']
                        and metadata['coeff_count'] == record['timing']['coeff_count'], 'dump metadata mismatch')
                require(values.size > 0 and np.isfinite(values).all(), 'empty/nonfinite full dump')
                artifacts.append(dump)
            self.check_inputs()
            record['artifact_sha256'] = {str(artifact): sha(artifact) for artifact in artifacts}
            record['status'] = 'PASS'
        except BaseException as error:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            record['status'] = 'INTERRUPTED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAIL'
            record['error'] = repr(error)
            if process is not None:
                record['exit_code'] = process.returncode
            raise
        finally:
            record['finished'] = now()
            record['process_seconds'] = time.perf_counter()-start
            atomic_json(path, record)
            self.save()
        return record

    def qualify(self, model, mode, pilots):
        reference_meta, reference = read_dump(pilots['ry']['dump'])
        for backend, record in pilots.items():
            metadata, values = read_dump(record['dump'])
            require(metadata == reference_meta and values.shape == reference.shape, 'full-dump shape mismatch')
            np.testing.assert_allclose(values, reference, atol=ATOL, rtol=RTOL)
            require(record['timing']['reference_fingerprint_fnv1a64']
                    == pilots['ry']['timing']['reference_fingerprint_fnv1a64'], 'backend reference labels differ')
            if backend == 'ry':
                continue
            check = dict(model=model, mode=mode, backend=backend, reference_backend='ry',
                status='PASS', values=int(values.size), full_gradient_checked=mode == 'gradient', atol=ATOL, rtol=RTOL,
                max_abs=float(np.max(np.abs(values-reference))),
                max_scaled=float(np.max(np.abs(values-reference)/np.maximum(1., np.maximum(abs(values), abs(reference))))),
                dumps={name: pilots[name]['dump'] for name in ('ry', backend)})
            self.report['checks'] = [old for old in self.report['checks'] if
                (old['model'], old['mode'], old['backend']) != (model, mode, backend)] + [check]
        self.save()

    def section(self, model, mode):
        key = model + '/' + mode
        section = self.report['sections'].setdefault(key, {'status': 'CALIBRATING'})
        pilots = {backend: self.execute(model, mode, backend) for backend in self.args.backends}
        self.qualify(model, mode, pilots)
        section['atoms'] = pilots['ry']['timing']['atoms']
        if 'repeats' not in section:
            fastest = min(record['timing']['seconds'] for record in pilots.values())
            # Extra headroom avoids repeatedly paying a full warmup for tiny
            # datasets; original large-VO2 cases naturally retain repeats=1.
            section.update(repeats=max(1, math.ceil(1.15*self.args.target_seconds/fastest)), generation=0)
        while True:
            repeats = section['repeats']
            section['estimated_wall_seconds'] = sum(record['process_seconds'] + self.args.pairs*(
                record['process_seconds'] + (repeats-1)*record['timing']['seconds']) for record in pilots.values())
            section['estimate_scope'] = 'pilot plus all pairs, including each fresh process warmup; excludes earlier short generations'
            section['status'] = 'RUNNING'
            self.save()
            try:
                for pair, order in enumerate(self.contract['orders']):
                    for backend in order:
                        record = self.execute(model, mode, backend, repeats, pair, section['generation'])
                        require(record['timing']['atoms'] == pilots['ry']['timing']['atoms'], 'pair atom count mismatch')
                        require(record['timing']['reference_fingerprint_fnv1a64']
                                == pilots['ry']['timing']['reference_fingerprint_fnv1a64'], 'pair reference labels differ')
                        if record['timing']['seconds'] < MIN_SECONDS:
                            raise TooShort(record['timing']['seconds'])
                    print('PAIR_COMPLETE', key, pair, flush=True)
            except TooShort as error:
                require(section['generation'] < 4, 'five timing generations remained too short')
                section.setdefault('superseded_generations', []).append({
                    'generation': section['generation'], 'repeats': repeats, 'reason': str(error),
                    'short_seconds': error.seconds})
                section['generation'] += 1
                section['repeats'] = max(repeats+1, math.ceil(repeats*1.25*self.args.target_seconds/error.seconds))
                self.save()
                continue
            section['status'] = 'PASS'
            self.save()
            return

    def run(self):
        try:
            for model in self.args.models:
                for mode in self.args.modes:
                    self.section(model, mode)
            self.check_inputs()
            self.report['status'] = 'PASS'
        except BaseException as error:
            self.report['status'] = 'INTERRUPTED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAIL'
            self.report['error'] = repr(error)
            raise
        finally:
            self.report['finished'] = now()
            self.save()
        print(json.dumps({'status': self.report['status'], 'summary': str(self.out/'summary.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--training-cfg', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--target-seconds', type=float, default=5.)
    parser.add_argument('--process-timeout', type=float, default=7200.)
    parser.add_argument('--models', nargs='+', choices=tuple(MODEL_FILES), default=list(MODEL_FILES))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(sys.platform == 'linux' and os.environ.get('LSB_JOBID'), 'requires a Linux LSF compute allocation')
    require(args.count >= 2 and args.pairs >= 4, 'count must be >=2 and pairs >=4')
    require(math.isfinite(args.target_seconds) and args.target_seconds >= MIN_SECONDS,
            'target seconds must be finite and >=3')
    require(math.isfinite(args.process_timeout) and args.process_timeout > args.target_seconds,
            'process timeout must exceed target seconds')
    for values in (args.models, args.modes, args.backends):
        require(len(values) == len(set(values)), 'duplicate filter selection')
    require('ry' in args.backends, '--backends must include ry for the paired reference')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Campaign(args).run()


if __name__ == '__main__':
    main()
