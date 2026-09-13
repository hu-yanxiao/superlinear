#!/usr/bin/env python3
"""Server-only, same-core, paired RY/RD native performance comparison.

Uses an independently linked driver built from the already-qualified native
objects. Loading, warmup and output are outside its timers. No model update.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    median = statistics.median(values)
    return dict(n=len(values), median=median, min=min(values), max=max(values),
                mad=statistics.median(abs(x-median) for x in values))


def read_dump(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    metadata = rows[0]
    assert metadata['schema'] == 'rd_m0_benchmark_dump_v1'
    values = []
    if metadata['mode'] == 'gradient':
        assert len(rows) == 2 and len(rows[1]['gradient']) == metadata['coeff_count']
        values = [rows[1]['loss'], *rows[1]['gradient']]
    else:
        assert len(rows)-1 == metadata['configs']
        for index, row in enumerate(rows[1:]):
            assert row['index'] == index and len(row['site_energies']) == row['atoms']
            values.extend([row['energy'], *row['site_energies']])
            if metadata['mode'] != 'energy':
                assert len(row['forces']) == row['atoms']*3 and len(row['virial']) == 9
                values.extend([*row['forces'], *row['virial']])
    return metadata, np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--training-cfg', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--target-seconds', type=float, default=5.)
    args = parser.parse_args()
    if sys.platform != 'linux' or not os.environ.get('LSB_JOBID'):
        raise SystemExit('Numerical benchmarks require a Linux LSF compute allocation')
    assert args.count >= 2 and args.pairs >= 4 and args.target_seconds >= 3.
    binary = args.binary.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    (out/'checks').mkdir()
    (out/'inputs').mkdir()
    affinity = sorted(os.sched_getaffinity(0))
    assert affinity, 'no available CPU affinity'
    os.sched_setaffinity(0, {affinity[0]})
    models = {}
    for label, filename in [('sharedn_sharedreadout', 'vo2-sharedn-speciesamp.mtp'),
                            ('fullu_speciesreadout', 'vo2-fullu-speciesreadout.mtp')]:
        source = (args.models_dir/filename).resolve(strict=True)
        target = out/'inputs'/filename
        target.write_bytes(source.read_bytes())
        models[label] = target
    blocks, current = [], []
    with args.training_cfg.open() as stream:
        for line in stream:
            if line.strip() == 'BEGIN_CFG':
                assert not current
                current = [line]
            elif current:
                current.append(line)
                if line.strip() == 'END_CFG':
                    blocks.append(''.join(current))
                    current = []
    assert not current and len(blocks) >= args.count
    indices = [i*(len(blocks)-1)//(args.count-1) for i in range(args.count)]
    assert len(set(indices)) == args.count
    data = out/'inputs'/f'vo2-uniform-{args.count}.cfg'
    data.write_text(''.join(blocks[i] for i in indices))
    env = {k:v for k,v in os.environ.items() if not k.startswith('SUS2_SH_')}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_DYNAMIC='FALSE', I_MPI_PIN='0', I_MPI_FABRICS='shm')
    identities = {str(p): sha(p) for p in [binary, data, *models.values()]}
    report = dict(status='RUNNING', started=datetime.datetime.now().astimezone().isoformat(),
        host=os.uname().nodename, jobid=os.environ['LSB_JOBID'], affinity_before=affinity,
        pinned_cpu=affinity[0], selected_indices=indices, source_config_count=len(blocks),
        configs=args.count, input_sha256=identities, command=sys.argv,
        process_records=[], checks=[], trials=[], aggregates={},
        scope={'energy': 'Energy/site-energy kernel; prebuilt neighborhoods',
               'efs': 'Full energy/forces/virial kernel; prebuilt neighborhoods',
               'gradient': 'Actual EFS residual loss + full physical coefficient gradient; prebuilt neighborhoods; weights 1/.1/.01, vibrations, no regularization or optimizer',
               'efs-total': 'CalcEFS including fresh neighborhood construction'},
        limitations=['Single-core latency, not MPI scaling or GPUMD throughput.',
                     'No optimizer update, line search, or model-quality comparison.',
                     'Peak RSS includes loaded configurations and resident prebuilt neighborhoods.',
                     'Allocated CPU affinity is respected; external node load can still introduce noise.'])

    def save():
        (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def run(label, backend, model, mode, repeats=1, dump=False):
        path = out/'checks'/(label+'.txt') if dump else '-'
        command = [str(binary), str(model), str(data), mode, str(args.count),
                   str(repeats), '1', str(path)]
        start = time.perf_counter()
        result = subprocess.run(command, env=dict(env, SUS2_SH_ANGULAR_BACKEND=backend),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=3600, cwd=out)
        log = out/'logs'/(label+'.log')
        log.write_text(result.stdout)
        rec = dict(label=label, backend=backend, mode=mode, command=command,
                   log=str(log), exit_code=result.returncode,
                   process_seconds=time.perf_counter()-start)
        report['process_records'].append(rec)
        save()
        assert result.returncode == 0, f'{label}: exit {result.returncode}; {result.stdout[-1600:]}'
        rows = [json.loads(line[len('BENCH_JSON '):]) for line in result.stdout.splitlines()
                if line.startswith('BENCH_JSON ')]
        assert len(rows) == 1, (label, rows)
        timing = rows[0]
        assert timing['cpu_affinity_count'] == 1 and timing['ranks'] == 1
        assert timing['backend'] == backend and timing['mode'] == mode
        assert timing['configs'] == args.count and timing['repeats'] == repeats
        assert math.isfinite(timing['seconds']) and timing['seconds'] > 0
        assert all(math.isfinite(x) for x in timing.values() if isinstance(x, (float,int)))
        rec['timing'] = timing
        save()
        return timing, path

    try:
        for model_label, model in models.items():
            for mode in ('energy', 'efs', 'gradient', 'efs-total'):
                key = model_label+'/'+mode
                pilots, dumps = {}, {}
                for backend in ('ry', 'rd-m0'):
                    pilots[backend], dumps[backend] = run(
                        model_label+'-'+mode+'-pilot-'+backend, backend, model, mode, dump=True)
                (meta_left,left), (meta_right,right) = [read_dump(dumps[backend]) for backend in ('ry','rd-m0')]
                assert meta_left == meta_right
                assert left.shape == right.shape and left.size and np.isfinite(left).all() and np.isfinite(right).all()
                np.testing.assert_allclose(left, right, atol=2e-9, rtol=2e-9)
                report['checks'].append(dict(model=model_label, mode=mode, values=int(left.size),
                    max_abs=float(np.max(np.abs(left-right))),
                    max_scaled=float(np.max(np.abs(left-right)/np.maximum(1.,np.maximum(abs(left),abs(right)))))))
                repeats = min(1000, max(1, math.ceil(args.target_seconds/min(p['seconds'] for p in pilots.values()))))
                assert pilots['ry']['atoms'] == pilots['rd-m0']['atoms']
                paired_ratios, per_backend = [], {'ry':[], 'rd-m0':[]}
                for pair in range(args.pairs):
                    order = ('ry','rd-m0') if pair % 2 == 0 else ('rd-m0','ry')
                    matched = {}
                    for position, backend in enumerate(order):
                        timing, _ = run(f'{model_label}-{mode}-pair{pair}-pos{position}-{backend}',
                                        backend, model, mode, repeats)
                        assert timing['atoms'] == pilots['ry']['atoms']
                        matched[backend] = timing
                        per_backend[backend].append(timing)
                        report['trials'].append({**timing, 'model':model_label, 'mode':mode,
                                                'pair':pair, 'position':position, 'backend':backend})
                        save()
                    paired_ratios.append(matched['rd-m0']['seconds']/matched['ry']['seconds'])
                    print(key, 'pair', pair, 'RD/RY=', paired_ratios[-1], flush=True)
                aggregate = dict(repeats=repeats, atoms=pilots['ry']['atoms'],
                                 paired_time_ratio_rd_over_ry=distribution(paired_ratios))
                for backend, trials in per_backend.items():
                    aggregate[backend] = {field:distribution([t[field] for t in trials])
                                          for field in ('seconds','us_per_atom','peak_rss_kib')}
                aggregate['median_ratio_minus_one_percent'] = 100*(statistics.median(paired_ratios)-1)
                aggregate['all_pairs_same_direction'] = all(x>1 for x in paired_ratios) or all(x<1 for x in paired_ratios)
                report['aggregates'][key] = aggregate
                save()
        assert identities == {path:sha(path) for path in identities}
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = repr(error)
        raise
    finally:
        report['finished'] = datetime.datetime.now().astimezone().isoformat()
        save()
    print(json.dumps({'status':report['status'], 'summary':str(out/'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
