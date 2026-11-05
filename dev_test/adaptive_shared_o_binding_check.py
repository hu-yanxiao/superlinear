#!/usr/bin/env python3
"""Check native inference against PT fixtures and independently tied full U."""
import argparse
import json
from pathlib import Path
import numpy as np
from pysuperlinear._core import SuperlinearModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    lines=(args.fixtures/'tiny.cfg').read_text().splitlines()
    at=next(i for i,s in enumerate(lines) if s.strip()=='Supercell')
    cell=np.array([[float(x) for x in s.split()] for s in lines[at+1:at+4]])
    at=next(i for i,s in enumerate(lines) if 'AtomData:' in s)
    rows=[]
    for line in lines[at+1:]:
        fields=line.split()
        if len(fields)<5: break
        rows.append([float(x) for x in fields])
    rows=np.array(rows);numbers=np.array([1 if t==0 else 8 for t in rows[:,1]],dtype=np.int32)
    positions=rows[:,2:5].copy()
    report={}
    for path in sorted(args.fixtures.glob('*.mtp')):
        if path.stem.endswith('_full'): continue
        model=SuperlinearModel(str(path),[1,8])
        full_path=path.with_name(path.stem+'_full.mtp')
        full=SuperlinearModel(str(full_path),[1,8]) if full_path.exists() else model
        def calc(m,x=positions):return m.calculate(numbers,x,cell,[True]*3)
        r=calc(model);f=calc(full)
        diffs={k:float(np.max(np.abs(np.asarray(r[k])-np.asarray(f[k])))) for k in ('energy','forces','virial')}
        for k in diffs:np.testing.assert_allclose(r[k],f[k],atol=2e-11,rtol=2e-10)
        prediction=np.r_[r['energy'],np.asarray(r['forces']).ravel(),np.asarray(r['virial']).ravel()]
        ref=np.loadtxt(path.with_name(path.stem+'_efs.txt'))
        np.testing.assert_allclose(prediction,ref,atol=2e-10,rtol=2e-9)
        eps=1e-5;fd=np.empty_like(positions)
        for i in range(len(numbers)):
            for d in range(3):
                plus=positions.copy();minus=positions.copy();plus[i,d]+=eps;minus[i,d]-=eps
                fd[i,d]=-(calc(model,plus)['energy']-calc(model,minus)['energy'])/(2*eps)
        np.testing.assert_allclose(fd,r['forces'],atol=2e-9,rtol=2e-6)
        report[path.stem]={'tied_full_max_error':diffs if full_path.exists() else None,'PT_max_error':float(np.max(np.abs(prediction-ref))),
                           'force_fd_max_error':float(np.max(np.abs(fd-r['forces'])))}
    assert len(report)>=12, 'expected all same/cross/grouped, shared/species, none/tanh fixtures'
    args.report.write_text(json.dumps({'status':'PASS','cases':report},indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':len(report),'report':str(args.report)}))


if __name__=='__main__':main()
