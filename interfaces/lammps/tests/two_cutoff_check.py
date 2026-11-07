#!/usr/bin/env python3
"""Compare embedded two-cutoff EFS to independent, nontrivial PT fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures', type=Path, required=True)
p.add_argument('--embedded', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
frames = [s + 'END_CFG\n' for s in (a.fixtures/'geometries.cfg').read_text().split('END_CFG') if s.strip()]
runs = []
with tempfile.TemporaryDirectory() as tmp:
    for model in sorted(a.fixtures.glob('*.mtp')):
        expected = json.loads(model.with_suffix('.efs.json').read_text())['records']
        assert len(expected) == len(frames)
        assert np.max(np.abs(expected[0]['forces'])) > .01, 'Degenerate force fixture'
        for frame, (cfg_text, ref) in enumerate(zip(frames, expected)):
            cfg = Path(tmp)/'frame.cfg'
            cfg.write_text(cfg_text)
            target = np.r_[ref['energy'], np.array(ref['forces']).ravel(), np.array(ref['plus_stress']).ravel()]
            for domains in (1, 2):
                # Include exact direct fallback when the table budget is zero,
                # cached two-grid interpolation and both reverse implementations.
                for mode, budget, stream in [('direct','auto','1'), ('table','auto','1'), ('table','0','0')]:
                    env = dict(os.environ, SUS2_SL_EDGE_METADATA='1', SUS2_SL_REVERSE_STREAM=stream)
                    q = subprocess.run([str(a.embedded.resolve()), str(model.resolve()), str(cfg), str(domains),
                                        mode, '.001', budget, 'present'], env=env, text=True, capture_output=True, timeout=30)
                    assert q.returncode == 0, (model, frame, mode, q.stderr)
                    out = json.loads(next(l.split(' ',1)[1] for l in q.stdout.splitlines() if l.startswith('NATIVE_EFS_JSON ')))
                    value = np.r_[out['energy'], np.array(out['forces']).ravel(), np.array(out['virial']).ravel()]
                    exact = mode == 'direct' or budget == '0'
                    np.testing.assert_allclose(value, target, atol=2e-10 if exact else 2e-6,
                                               rtol=2e-9 if exact else 2e-6,
                                               err_msg=f'{model.stem} frame={frame} {mode}/{budget} domains={domains}')
                    stats = json.loads(next(l.split(' ',1)[1] for l in q.stdout.splitlines() if l.startswith('RADIAL_STATS_JSON ')))
                    if mode == 'table' and budget == 'auto' and frame == 0:
                        assert stats['hits'] > 0, stats
                    if budget == '0':
                        assert stats['bytes'] == 0, stats
                    runs.append(dict(model=model.stem, frame=frame, domains=domains, mode=mode, budget=budget,
                                     max_error=float(np.max(np.abs(value-target))), radial=stats))
result = dict(status='PASS', runs=runs)
a.report.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(status='PASS', runs=len(runs), max_error=max(x['max_error'] for x in runs))))
