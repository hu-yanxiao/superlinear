#!/usr/bin/env python3
"""Embedded direct/table and halo-domain checks against independent PT EFS fixtures."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--embedded',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
runs=[]
for model in sorted(a.fixtures.glob('*.mtp')):
 if model.stem.endswith('_full'):continue
 expected=np.loadtxt(model.with_name(model.stem+'_efs.txt'))
 for domains in (1,2):
  for mode in ('direct','table'):
   q=subprocess.run([str(a.embedded.resolve()),str(model),str(a.fixtures/'tiny.cfg'),str(domains),mode],text=True,capture_output=True,timeout=30)
   assert q.returncode==0,(model,domains,mode,q.stderr)
   out=json.loads(next(s.split('NATIVE_EFS_JSON ',1)[1] for s in q.stdout.splitlines() if s.startswith('NATIVE_EFS_JSON ')))
   value=np.r_[out['energy'],np.asarray(out['forces']).ravel(),np.asarray(out['virial']).ravel()]
   np.testing.assert_allclose(value,expected,atol=2e-10 if mode=='direct' else 2e-6,rtol=2e-9 if mode=='direct' else 2e-6)
   stats=json.loads(next(s.split('RADIAL_STATS_JSON ',1)[1] for s in q.stdout.splitlines() if s.startswith('RADIAL_STATS_JSON ')))
   if mode=='table':assert stats['hits']>0,stats
   runs.append(dict(case=model.stem,domains=domains,mode=mode,max_error=float(np.max(np.abs(value-expected))),radial=stats))
assert len(runs)>=56
result=dict(status='PASS',runs=runs,reference='independent PT FP64 with tied-full parity; 1/2 halo domains; direct/table')
a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',runs=len(runs),max_error=max(x['max_error'] for x in runs))))
