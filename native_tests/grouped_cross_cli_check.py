"""Run native topology endpoints, non-power block, malformed metadata and EFS checks.
Usage: python3 native_tests/grouped_cross_cli_check.py /path/mlp-sus2 /path/group-check
The checker is built from grouped_cross_check.cpp and ordinary serial objects.
"""
import pathlib,subprocess,sys,tempfile
binary=pathlib.Path(sys.argv[1]).resolve();check=pathlib.Path(sys.argv[2]).resolve()
fixed=pathlib.Path(sys.argv[3]).resolve() if len(sys.argv)>3 else None
cfg=pathlib.Path(__file__).resolve().with_name('pure_w_regularization.cfg')
base=['--species-count=2','--l-max=1','--k-max=4','--two-layer-gate','--two-layer-gate-mode=mu-scalar-full','--two-layer-gate-body-order=3','--two-layer-gate-k-rank=2','--two-layer-gate-shared-radial','--body-order=3','--body-l-max=1,1,1,1','--sh-factor-pruning=q-total','--two-layer-gate-factor-pruning=q-total','--radial-basis-size=4','--cutoff=4','--min-dist=1','--init-params=random','--init-random-seed=1']
def run(args,ok=True):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True)
 assert (p.returncode==0)==ok,(args,p.stdout,p.stderr)
 return p.stdout
with tempfile.TemporaryDirectory(prefix='grouped-cross-') as tmp:
 root=pathlib.Path(tmp)
 for o,b in [(4,1),(4,2),(4,4),(6,3)]:
  model=root/f'o{o}b{b}.sus2'
  run([binary,'init-sh',model,*base,f'--outer-k-rank={o}',f'--outer-channel-block-size={b}'])
  print(run([check,model,cfg,root/'roundtrip.sus2']).strip())
  if fixed:print(run([fixed,model,cfg]).strip())
  text=model.read_text();assert 'version = 1.9.0' in text
  for replacement in ['outer_channel_block_size = 0',f'outer_channel_block_size = {o+1}']:
   invalid=root/'invalid.sus2';invalid.write_text(text.replace(f'outer_channel_block_size = {b}',replacement))
   run([binary,'calc-efs',invalid,cfg,root/'bad.cfg'],False)
 # One group must preserve the native cross graph and model numerics exactly.
 plain=root/'plain.sus2';run([binary,'init-sh',plain,*base,'--outer-k-rank=4'])
 a=root/'plain.cfg';c=root/'group.cfg'
 run([binary,'calc-efs',plain,cfg,a]);run([binary,'calc-efs',root/'o4b4.sus2',cfg,c])
 assert a.read_text()==c.read_text(),'G=1 changed native cross-channel EFS'
 if fixed:
  print(run([fixed,plain,cfg]).strip())
  for name,extra in [('inner',['--inner-readout']),('species',['--readout-mode=species'])]:
   model=root/(name+'.sus2');run([binary,'init-sh',model,*base,'--outer-k-rank=4','--outer-channel-block-size=2',*extra])
   print(run([fixed,model,cfg]).strip())
print('grouped_cross_cli_check PASS')
