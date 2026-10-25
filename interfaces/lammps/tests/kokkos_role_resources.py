#!/usr/bin/env python3
"""Optional allocated-job binary resource evidence; never a throughput claim."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import cpu_qualification as cpu
from kokkos_benchmark import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tool',default='/share/apps/cuda/12.4/bin/cuobjdump')
    a=p.parse_args();report=dict(schema='sus2-role-optional-cuda-resource-v1',status='UNAVAILABLE',
        optional=True,throughput_evidence=False,job_id=os.environ.get('LSB_JOBID'),commands=[])
    try:
        if not os.environ.get('LSB_JOBID'):raise RuntimeError('Resource probe requires existing compute allocation')
        a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
        report['binary_sha256_before']=sha(a.candidate)
        owner=SimpleNamespace(env=dict(os.environ))
        for label,command in [('version',[a.tool,'--version']),('resource',[a.tool,'--dump-resource-usage',str(a.candidate)])]:
            log=a.out/(label+'.log')
            rec=dict(label=label,command=command,log=str(log),returncode=None,timed_out=None)
            report['commands'].append(rec)
            result=cpu.Qualification.process(owner,command,a.out,log,120)
            rec.update({k:result[k] for k in ('returncode','timed_out','elapsed_seconds','process_group_killed_on_timeout')})
            rec['log_sha256']=sha(log)
            if result['returncode']!=0 or result['timed_out']:raise RuntimeError('Optional cuobjdump probe unavailable/failed')
        report['binary_sha256_after']=sha(a.candidate)
        if report['binary_sha256_after']!=report['binary_sha256_before']:raise RuntimeError('Resource binary identity changed')
        report['status']='OBSERVED'
    except Exception as error:report['error']=str(error)
    finally:
        if a.out.is_dir() and not (a.out/'resource.json').exists():save(a.out/'resource.json',report)
    print(json.dumps(report))
    return 0  # Optional tool availability must never turn mathematical PASS into EXIT.


if __name__=='__main__':raise SystemExit(main())
