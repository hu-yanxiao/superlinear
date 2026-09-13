#!/usr/bin/env python3
"""Server/LSF-only interruption, partial-pair resume and immutable-contract smoke.

Runs the real benchmark with 2 CFGs, energy, ry/hybrid, 4 pairs and a 3s target.
All files/processes belong to the newly created --outdir; no global job control.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(campaign):
    return {path: json.loads(path.read_text()) for path in (campaign/'processes').glob('*.json')}


def passed(campaign):
    return {str(path): sha(path) for path, item in records(campaign).items() if item['status'] == 'PASS'}


def start_ticks(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def stop_owned(process, campaign):
    if process.poll() is not None:
        return
    process.terminate()  # The runner normally terminates/waits on its own child.
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    # Last-resort cleanup is restricted to this fresh campaign's recorded,
    # still-identical child process groups; PID reuse never grants ownership.
    for item in records(campaign).values():
        pid, ticks = item.get('pid'), item.get('start_ticks')
        if item['status'] == 'RUNNING' and pid and ticks and start_ticks(pid) == ticks:
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'training-cfg', 'models-dir', 'outdir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stage-timeout', type=float, default=600.)
    args = parser.parse_args()
    require(sys.platform == 'linux' and os.environ.get('LSB_JOBID'), 'requires a Linux LSF allocation')
    require(0 < args.stage_timeout <= 3600, 'stage timeout must be in (0,3600] seconds')
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    campaign = out/'campaign'
    runner = Path(__file__).with_name('rd_m0_ablation_benchmark.py').resolve(strict=True)
    command = [sys.executable, '-u', str(runner), '--binary', str(args.binary.resolve(strict=True)),
        '--training-cfg', str(args.training_cfg.resolve(strict=True)),
        '--models-dir', str(args.models_dir.resolve(strict=True)), '--out', str(campaign),
        '--count', '2', '--pairs', '4', '--models', 'sharedn_sharedreadout',
        '--modes', 'energy', '--backends', 'ry', 'hybrid', '--target-seconds', '3']
    report = {'status': 'RUNNING', 'command': command, 'stages': []}
    active = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)

    def launch(label, extra=()):
        with (out/(label+'.log')).open('x') as stream:
            return subprocess.Popen(command+list(extra), stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)

    def finish(label, extra=(), success=True):
        nonlocal active
        active = launch(label, extra)
        code = active.wait(timeout=args.stage_timeout)
        require((code == 0) == success, label+' unexpected exit '+str(code))
        report['stages'].append({'name': label, 'exit_code': code, 'log': str(out/(label+'.log'))})
        active = None

    try:
        active = launch('interrupt-first-pair')
        deadline = time.monotonic()+args.stage_timeout
        while True:
            qualified = [item for item in records(campaign).values() if item['status'] == 'PASS'
                and item['spec']['pair'] is not None and item['timing']['seconds'] >= 3.]
            if qualified:
                first = min(qualified, key=lambda item: item['started'])
                siblings = [item for item in qualified if item['spec']['pair'] == first['spec']['pair']
                            and item['spec']['generation'] == first['spec']['generation']]
                require(len(siblings) == 1, 'missed the partial-pair interruption window')
                break
            require(active.poll() is None, 'runner ended before the first qualified pair record; inspect log')
            require(time.monotonic() < deadline, 'timeout awaiting first qualified pair record')
            time.sleep(.05)
        original = passed(campaign)
        stop_owned(active, campaign)
        require(active.returncode != 0, 'interrupted runner unexpectedly completed')
        report['stages'].append({'name': 'interrupt-first-pair', 'exit_code': active.returncode,
                                'first_logical_id': first['logical_id'], 'preserved_pass': original})
        active = None
        finish('resume-partial', ['--resume'])
        final_pass = passed(campaign)
        require(all(final_pass.get(path) == digest for path, digest in original.items()),
                'completed process records were rerun, removed or modified during resume')
        summary = json.loads((campaign/'summary.json').read_text())
        require(summary['status'] == 'PASS', 'resumed campaign did not pass')
        aggregate = summary['aggregates']['sharedn_sharedreadout/energy']
        require(aggregate['status'] == 'COMPLETE', 'remaining partial pairs were not completed')
        comparison = aggregate['relative_to_ry']['hybrid']
        require(comparison['pair_indices'] == [0, 1, 2, 3]
                and comparison['paired_time_ratio_over_ry']['n'] == 4, 'incorrect matched-pair count')
        logical = [item['logical_id'] for item in records(campaign).values() if item['status'] == 'PASS']
        require(len(logical) == len(set(logical)), 'duplicate successful logical process')
        finish('resume-complete', ['--resume'])
        require(passed(campaign) == final_pass, 'complete resume created/changed successful records')
        finished_hash = sha(campaign/'summary.json')
        finish('reject-count3', ['--resume', '--count', '3'], success=False)
        require('resume contract mismatch' in (out/'reject-count3.log').read_text(), 'wrong rejection reason')
        require(sha(campaign/'summary.json') == finished_hash and passed(campaign) == final_pass,
                'rejected parameters polluted the completed summary or measurements')
        report.update(status='PASS', completed_pairs=4, pass_records=len(final_pass))
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        try:
            if active is not None:
                stop_owned(active, campaign)
        finally:
            (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'PASS', 'report': str(out/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
