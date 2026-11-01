#!/usr/bin/env python3
"""Cross-interface grouped E/F/virial check using independently exported PT fixtures.

The embedded executable is built from embedded_efs.cpp; --native is mlp-sus2.
No jobs are submitted. Tables retain the existing interpolation tolerances.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures', type=Path, required=True)
p.add_argument('--embedded', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
p.add_argument('--pt-source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.pt_source.resolve()))
from sus2shpt.io.cfg import load_cfg

a.output.mkdir(parents=True, exist_ok=False)
expected = json.loads((a.fixtures / 'expected-efs.json').read_text())
report = []
def flat(e, f, v):
    return np.concatenate(([e], np.asarray(f).ravel(), np.asarray(v).ravel()))
def run(tag, argv, setting='1'):
    q = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                       timeout=120, env={**os.environ, 'SUS2_SL_GROUPED_GRAPH':setting})
    (a.output / (tag + '.log')).write_text(q.stdout + q.stderr)
    assert q.returncode == 0, (tag, q.stdout[-1500:], q.stderr[-1500:])
    return q.stdout

for name, ref in sorted(expected.items()):
    model = a.fixtures / (name + '.mtp')
    cfg = a.fixtures / 'tiny.cfg'
    wanted = flat(ref['E'], ref['F'], ref['V'])
    prediction = a.output / (name + '.cfg')
    run(name + '-native', [a.native, 'calc-efs', model, cfg, prediction, '--no-loss'])
    c = load_cfg(prediction)[0]
    native = flat(c.energy, [x.force for x in c.atoms], c.stress)
    np.testing.assert_allclose(native, wanted, atol=2e-10, rtol=2e-10)
    checks = {'native_pt_max_abs': float(np.max(np.abs(native-wanted)))}
    reference = None
    for mode in ('direct', 'table'):
        for domains in (1, 2):
            for setting in ('0', '1'):
                tag = f'{name}-{mode}-{domains}-grouped{setting}'
                text = run(tag, [a.embedded, model, cfg, domains, mode, '.001'], setting)
                out = json.loads(next(x for x in text.splitlines() if x.startswith('NATIVE_EFS_JSON ')).split(' ', 1)[1])
                path = json.loads(next(x for x in text.splitlines() if x.startswith('GROUPED_GRAPH_JSON ')).split(' ', 1)[1])
                assert path['groups'] == (2 if setting == '1' else 0), (tag, path)
                result = flat(out['energy'], out['forces'], out['virial'])
                tolerance = 2e-10 if mode == 'direct' else 2e-7
                np.testing.assert_allclose(result, wanted, atol=tolerance, rtol=tolerance)
                if setting == '0': reference = result
                else: np.testing.assert_array_equal(result, reference)
                checks[f'{mode}_domains{domains}_grouped{setting}_max_abs'] = float(np.max(np.abs(result-wanted)))
    report.append({'case': name, 'checks': checks})
    print(name, 'PASS', flush=True)
(a.output / 'report.json').write_text(json.dumps({'status':'PASS','cases':report}, indent=2)+'\n')
