from pathlib import Path
import subprocess,re,json,argparse,math,shlex
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Nontrivial two-element frozen inner/outer linear-solve regression')
parser.add_argument('--binary',type=Path,default=R/'bin/mlp-sus2')
parser.add_argument('--work-dir',type=Path,default=R/'.local_tmp/inner_readout_solver')
parser.add_argument('--adaptive',action='store_true')
parser.add_argument('--joint',action='store_true')
parser.add_argument('--projection',choices=['full','shared-n'],default='full')
parser.add_argument('--activation',choices=['none','tanh'],default='none')
parser.add_argument('--mpi-command',default='',help='Optional prefix for the fitted solve, e.g. mpirun -n 2')
args=parser.parse_args(); B=args.binary.resolve(); D=args.work_dir.resolve(); D.mkdir(parents=True,exist_ok=True)
def run(label,*args):
 prefix=shlex.split(globals()['args'].mpi_command) if label=='synthetic-fit' else []
 p=subprocess.run([*prefix,str(B),*map(str,args)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=D);(D/(label+'.log')).write_text(p.stdout);assert p.returncode==0,p.stdout[-2000:];return p.stdout
def replace(s,k,v):return re.sub(r'(\b'+k+r'\s*=\s*\{)[^}]*(\})',lambda m:m[1]+', '.join(map(str,v))+m[2],s)
def numbers(p):
 s=p.read_text();m=re.search(r'Energy\s*\n\s*(\S+)',s); f=[];w=[]
 for line in s.splitlines():
  a=line.split()
  if len(a)==9 and a[0].isdigit():f.extend(map(float,a[-3:]))
  if len(a)==6:
   try:w=list(map(float,a))
   except:pass
 return [float(m[1])],f,w
adaptive_options=['--superlinear-mode=radial-coefficient-gate','--outer-k-rank=2','--two-layer-gate-k-rank=2','--adaptive-radial-projection='+args.projection,'--adaptive-radial-activation='+args.activation]+(['--joint-readout'] if args.joint else []) if args.adaptive else []
run('init','init-sh',D/'init.sus2','--species-count=2','--l-max=1','--k-max=2','--body-order=3','--body-l-max=1,1,1,1','--radial-basis-size=3','--cutoff=5','--two-layer-gate','--two-layer-gate-mode=mu-scalar-full','--two-layer-gate-body-order=2','--inner-readout',*adaptive_options)
run('complete','roundtrip-model-dev',D/'init.sus2',D/'model.sus2')
s=(D/'model.sus2').read_text()
def vector_size(s,k):
 m=re.search(r'\b'+k+r'\s*=\s*\{([^}]*)\}',s);return len(m[1].split(',')) if m else 0
outer_count=vector_size(s,'moment_coeffs'); inner_count=vector_size(s,'two_layer_gate_readout_coeffs')
s=replace(s,'species_coeffs',[.7,1.3]);s=replace(s,'moment_coeffs',[.02*math.cos(i) for i in range(outer_count)]);s=replace(s,'two_layer_gate_readout_coeffs',[.03*math.sin(i+1) for i in range(inner_count)])
if args.joint:
 s=replace(s,'joint_readout_coeffs',[.025*math.sin(i+.2) for i in range(vector_size(s,'joint_readout_coeffs'))])
(D/'model.sus2').write_text(s)
(D/'one.cfg').write_text('''BEGIN_CFG
 Size
 4
 Supercell
 12 0 0
 0 12 0
 0 0 12
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1 1 1 0 0 0
 2 1 2.4 1.3 1.1 0 0 0
 3 0 1.3 2.8 1.4 0 0 0
 4 1 2 2 3 0 0 0
 Energy
 0
 PlusStress: xx yy zz yz xz xy
 0 0 0 0 0 0
END_CFG
''')
run('target','calc-efs',D/'model.sus2',D/'one.cfg',D/'target.cfg')
zero=replace(replace(s,'moment_coeffs',[0]*outer_count),'two_layer_gate_readout_coeffs',[0]*inner_count);zero=replace(zero,'joint_readout_coeffs',[0]*vector_size(zero,'joint_readout_coeffs'));(D/'zero.sus2').write_text(zero)
constant=replace(zero,'two_layer_gate_readout_coeffs',[.25]+[0]*(inner_count-1));(D/'constant.sus2').write_text(constant)
for tag in ['zero','constant']:run(tag,'calc-efs',D/(tag+'.sus2'),D/'one.cfg',D/(tag+'.cfg'))
a=numbers(D/'zero.cfg');b=numbers(D/'constant.cfg');assert a[1:]==b[1:],'constant changed forces/virial'
assert abs(b[0][0]-a[0][0]-1.0)<1e-10,'constant must use the existing elemental t'
if args.adaptive:
 log=run('default-resume','train',D/'model.sus2',D/'target.cfg','--max-iter=0','--init-params=same','--trained-pot-name='+str(D/'default-resume.sus2'))
 assert 'linear initialization start' not in log and 'TrainLinear' not in log,'default Adaptive unexpectedly refit heads'
 run('default-predict','calc-efs',D/'default-resume.sus2',D/'one.cfg',D/'default.cfg')
 assert numbers(D/'default.cfg')==numbers(D/'target.cfg'),'default Adaptive resume changed predictions'
run('synthetic-fit','train',D/'zero.sus2',D/'target.cfg','--inner-readout-linear-init','--max-iter=0','--skip-preinit','--init-params=same','--std-weight=0','--stdd-weight=0',*(['--energy-weight=10000','--force-weight=10000','--stress-weight=10000'] if args.adaptive else ['--force-loss=l2','--scale-by-force=0','--radial-smooth=0']),'--trained-pot-name='+str(D/'synthetic-fit.sus2'))
run('fit-predict','calc-efs',D/'synthetic-fit.sus2',D/'one.cfg',D/'fit.cfg')
c=numbers(D/'target.cfg');assert len(c[1])==12 and len(c[2])==6 and max(map(abs,c[1]))>1e-5 and max(map(abs,c[2]))>1e-5
d=numbers(D/'fit.cfg');errs=[max([abs(x-y) for x,y in zip(v,w)] or [0]) for v,w in zip(c,d)];assert max(errs)<2.1e-6,errs # CFG text rounds forces to 1e-6 and virials to 1e-5
fit=(D/'synthetic-fit.sus2').read_text()
for key in ['moment_coeffs','two_layer_gate_readout_coeffs']+(['joint_readout_coeffs'] if args.joint else []):
 m=re.search(r'\b'+key+r'\s*=\s*\{([^}]*)\}',fit)
 assert m and any(abs(float(x))>1e-12 for x in m[1].split(',') if x.strip()),'fitted head was silently omitted: '+key
for key in ['species_coeffs','scal_coeffs','radial_coeffs','two_layer_gate_weights','adaptive_radial_amplitudes','adaptive_radial_weights','two_layer_gate_k_rank_weights','two_layer_gate_radial_coeffs','two_layer_gate_type_coeffs']:
 def vec(t):
  m=re.search(r'\b'+key+r'\s*=\s*\{([^}]*)\}',t);return [float(x) for x in m[1].split(',') if x.strip()] if m else None
 assert vec(fit)==vec(zero),key
run('design-columns-fd','check-linear-components-fd-dev',D/'model.sus2',D/'one.cfg','--max-atoms=4','--fixed-center-readout-columns','--check-stress','--abs-tolerance=3e-6','--rel-tolerance=3e-5')
print(json.dumps({'constant_force_virial_unchanged':True,'synthetic_E_F_W_max_errors':errs,'frozen_parameters':True}))
