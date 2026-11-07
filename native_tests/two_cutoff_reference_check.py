#!/usr/bin/env python3
"""Compare full-precision native E/F/virial to independently generated JSON fixtures."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixtures',type=Path)
p.add_argument('--helper',type=Path,default=Path('.local_tmp/two_cutoff_predict'));p.add_argument('--out',type=Path,default=Path('.local_tmp/two_cutoff_oracle'))
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def numbers(tree):
 if isinstance(tree,list):return [v for t in tree for v in numbers(t)]
 return [float(tree)]
worst={}
for fixture in sorted(a.fixtures.glob('*.efs.json')):
 name=fixture.name[:-9];model=a.fixtures/(name+'.mtp');out=a.out/(name+'.json')
 result=subprocess.run([str(a.helper.resolve()),str(model),str(a.fixtures/'geometries.cfg'),str(out)],capture_output=True,text=True,timeout=120)
 (a.out/(name+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,(name,result.stderr)
 expected=json.loads(fixture.read_text())['records'];actual=json.loads(out.read_text())['records'];assert len(actual)==len(expected)
 errors={key:0. for key in ('energy','forces','plus_stress')}
 for truth,got in zip(expected,actual):
  assert truth['config']==got['config']
  for key in errors:
   x,y=numbers(truth[key]),numbers(got[key]);assert len(x)==len(y)
   errors[key]=max(errors[key],max(abs(u-v) for u,v in zip(x,y)))
 assert max(errors.values())<1e-9,(name,errors)
 worst[name]=errors
(a.out/'summary.json').write_text(json.dumps(worst,indent=2)+'\n');print(json.dumps(worst))
