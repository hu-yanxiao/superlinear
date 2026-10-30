from pathlib import Path
import subprocess,re,json,argparse,math
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Nontrivial two-element frozen inner/outer linear-solve regression')
parser.add_argument('--binary',type=Path,default=R/'bin/mlp-sus2')
parser.add_argument('--work-dir',type=Path,default=R/'.local_tmp/inner_readout_solver')
args=parser.parse_args(); B=args.binary.resolve(); D=args.work_dir.resolve(); D.mkdir(parents=True,exist_ok=True)
def run(label,*args):
 p=subprocess.run([str(B),*map(str,args)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=D);(D/(label+'.log')).write_text(p.stdout);assert p.returncode==0,p.stdout[-2000:];return p.stdout
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
run('init','init-sh',D/'init.sus2','--species-count=2','--l-max=1','--k-max=2','--body-order=3','--body-l-max=1,1,1,1','--radial-basis-size=3','--cutoff=5','--two-layer-gate','--two-layer-gate-mode=mu-scalar-full','--two-layer-gate-body-order=2','--inner-readout')
run('complete','roundtrip-model-dev',D/'init.sus2',D/'model.sus2')
s=(D/'model.sus2').read_text()
s=replace(s,'species_coeffs',[.7,1.3]);s=replace(s,'moment_coeffs',[.02*math.cos(i) for i in range(8)]);s=replace(s,'two_layer_gate_readout_coeffs',[.03*math.sin(i+1) for i in range(3)])
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
zero=replace(replace(s,'moment_coeffs',[0]*8),'two_layer_gate_readout_coeffs',[0]*3);(D/'zero.sus2').write_text(zero)
constant=replace(zero,'two_layer_gate_readout_coeffs',[.25,0,0]);(D/'constant.sus2').write_text(constant)
for tag in ['zero','constant']:run(tag,'calc-efs',D/(tag+'.sus2'),D/'one.cfg',D/(tag+'.cfg'))
a=numbers(D/'zero.cfg');b=numbers(D/'constant.cfg');assert a[1:]==b[1:],'constant changed forces/virial'
run('synthetic-fit','train',D/'zero.sus2',D/'target.cfg','--inner-readout-linear-init','--max-iter=0','--skip-preinit','--init-params=same','--force-loss=l2','--scale-by-force=0','--std-weight=0','--stdd-weight=0','--radial-smooth=0','--trained-pot-name='+str(D/'synthetic-fit.sus2'))
run('fit-predict','calc-efs',D/'synthetic-fit.sus2',D/'one.cfg',D/'fit.cfg')
c=numbers(D/'target.cfg');assert len(c[1])==12 and len(c[2])==6 and max(map(abs,c[1]))>1e-5 and max(map(abs,c[2]))>1e-5
d=numbers(D/'fit.cfg');errs=[max([abs(x-y) for x,y in zip(v,w)] or [0]) for v,w in zip(c,d)];assert max(errs)<1e-6,errs
fit=(D/'synthetic-fit.sus2').read_text()
for key in ['species_coeffs','scal_coeffs','radial_coeffs','two_layer_gate_weights']:
 def vec(t):
  m=re.search(r'\b'+key+r'\s*=\s*\{([^}]*)\}',t);return [float(x) for x in m[1].split(',')] if m else None
 assert vec(fit)==vec(zero),key
print(json.dumps({'constant_force_virial_unchanged':True,'synthetic_E_F_W_max_errors':errs,'frozen_parameters':True}))
