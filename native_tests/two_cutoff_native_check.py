#!/usr/bin/env python3
"""Bounded native two-cutoff E/F/virial, mixed-gradient and column FD matrix.

Run after the serial/MPI executable and two_cutoff_seed_model helper are built.
All generated fixtures/logs live under --out; no reference/production files change.
"""
import argparse,json,subprocess,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,default=Path('bin/mlp-sus2'))
p.add_argument('--seed-helper',type=Path,default=Path('.local_tmp/two_cutoff_seed_model'))
p.add_argument('--columns-helper',type=Path,default=Path('.local_tmp/fixed_center_columns_check'))
p.add_argument('--out',type=Path,default=Path('.local_tmp/two_cutoff_matrix'))
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
exe=str(a.binary.resolve());seed=str(a.seed_helper.resolve());columns=str(a.columns_helper.resolve())
def call(argv,tag,success=True):
 r=subprocess.run(list(map(str,argv)),capture_output=True,text=True,timeout=120)
 (a.out/(tag+'.log')).write_text(r.stdout+r.stderr)
 assert (r.returncode==0)==success,(tag,r.returncode,(r.stdout+r.stderr)[-1500:])
 return r
cfg=a.out/'mixed.cfg';cfg.write_text('''BEGIN_CFG
 Size
    4
 Supercell
    16 0 0
    0 16 0
    0 0 16
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
    1 0 0 0 0 0 0 0
    2 1 4.8 .3 .1 0 0 0
    3 0 5.3 -.2 .4 0 0 0
    4 1 1.5 1.8 -.7 0 0 0
 Energy
    0
 PlusStress: xx yy zz yz xz xy
    0 0 0 0 0 0
END_CFG
''')
base=['--species-count=2','--l-max=2','--k-max=2','--body-order=3',
 '--cutoff=6.5','--angular-cutoff=5','--radial-basis-size=4']
modes={
 'static':[],
 'effect':['--two-layer-gate','--two-layer-gate-mode=mu-scalar-full','--two-layer-gate-k-rank=2','--outer-k-rank=2','--inner-readout'],
 'adaptive':['--superlinear-mode=radial-coefficient-gate','--two-layer-gate-k-rank=2','--outer-k-rank=2','--inner-readout'],
 'shared-o':['--superlinear-mode=radial-coefficient-gate','--two-layer-gate-k-rank=2','--outer-k-rank=2','--inner-readout',
             '--adaptive-radial-projection=shared-o','--adaptive-radial-activation=tanh','--tensor-message-lmax=1','--joint-readout']}
results=[]
cases=[(basis,sm,mode,2,2,4,2,[]) for basis in ('RBChebyshev_sss','RBLaguerre_log1p')
       for sm in ('K','L','LK','pair') for mode in modes]
cases += [
 ('RBChebyshev_sss','pair','static',1,3,3,3,[]),
 ('RBLaguerre_log1p','L','effect',3,3,5,3,[]),
 ('RBChebyshev_sss','K','shared-o',4,3,5,3,[]),
 ('RBLaguerre_log1p','pair','shared-o',1,3,3,3,[]),
 ('RBChebyshev_sss','LK','adaptive',3,3,5,3,['--adaptive-radial-projection=shared-n','--adaptive-radial-shared-n-rank=2','--adaptive-radial-amplitude-mode=species']),
 ('RBLaguerre_log1p','L','effect',2,3,5,3,['--two-layer-gate-decoupled-scaling']),
 ('RBChebyshev_sss','K','effect',3,3,3,3,['--radial-cutoff-type=polynomial']),
]
for index,(basis,sm,mode,lmax,kmax,rbsize,outer,extra) in enumerate(cases):
   opts=modes[mode] + (['--two-layer-gate','--two-layer-gate-body-order=3'] if mode!='static' else [])
   opts=[x if not x.startswith('--outer-k-rank=') else '--outer-k-rank='+str(outer) for x in opts]
   settings=[x for x in base if not x.startswith(('--l-max=','--k-max=','--radial-basis-size='))]
   settings += ['--l-max='+str(lmax),'--k-max='+str(kmax),'--radial-basis-size='+str(rbsize)]
   tag=basis+'_'+sm+'_'+mode + ('_extra'+str(index) if index>=32 else '')
   initial=a.out/(tag+'.sus2');model=a.out/(tag+'_seeded.sus2')
   call([exe,'init-sh',initial,*settings,'--radial-basis-type='+basis,'--scaling-map='+sm,*opts,*extra],tag+'_init')
   call([seed,initial,model],tag+'_seed')
   assert 'angular_max_dist = 5' in model.read_text(),tag
   call([exe,'check-efs-fd-dev',model,cfg,'--displacement=1e-6','--abs-tolerance=3e-8','--rel-tolerance=1e-5'],tag+'_efs')
   # Full parameter mixed F/stress plus physical radial-prior directional check.
   call([exe,'check-loss-gradient-direction-dev',model,cfg,'--energy-weight=1','--force-weight=1','--stress-weight=.01',
     '--gate-x-l2=0','--radial-smooth=1e-4','--radial-smooth-grid=8','--displacement=1e-5',
     '--abs-tolerance=3e-7','--rel-tolerance=1e-4'],tag+'_mixed')
   call([exe,'check-linear-components-fd-dev',model,cfg,'--max-atoms=1','--check-stress','--fixed-center-readout-columns',
     '--displacement=1e-6','--abs-tolerance=3e-8','--rel-tolerance=1e-5'],tag+'_components')
   call([columns,model,cfg],tag+'_reconstruct')
   results.append(tag)
# Equal explicit cutoff and absent field must preserve the same E/F/W bytes.
model=a.out/'RBChebyshev_sss_K_effect_seeded.sus2';text=model.read_text()
for tag,field in [('equal','angular_max_dist = 6.5'),('absent','')]:
 m=a.out/(tag+'.sus2');m.write_text(re.sub(r'angular_max_dist\s*=\s*[^\s]+',field,text))
 call([exe,'calc-efs',m,cfg,a.out/(tag+'.cfg')],tag+'_prediction')
assert (a.out/'equal.cfg').read_bytes()==(a.out/'absent.cfg').read_bytes()
for n,value in enumerate(('0','-1','6.6','nan','inf')):
 bad=a.out/('bad'+str(n)+'.sus2');bad.write_text(re.sub(r'angular_max_dist\s*=\s*[^\s]+','angular_max_dist = '+value,text))
 call([seed,bad,a.out/'bad-output.sus2'],'reject_'+str(n),False)
bad=a.out/'unsupported.sus2';bad.write_text(text.replace('RBChebyshev_sss','RBChebyshev_sss_rational'))
call([seed,bad,a.out/'bad-output.sus2'],'reject_unsupported',False)
summary=dict(status='PASS',models=len(results),cases=results,equal_vs_absent='byte-identical EFS',invalid_and_unsupported='rejected')
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
