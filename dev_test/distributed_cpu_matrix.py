#!/usr/bin/env python3
"""Nonzero modern-model fixtures for the staged CPU interface.

Every generated legal combination is recorded; no failed combination is skipped.
The optional checker compares native/staged direct/table E/F/virial and FD.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess

NUMBER=r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

def array_match(text,key):
    matches=list(re.finditer(rf'(?m)^([ \t]*{re.escape(key)}[ \t]*=[ \t]*)\{{([^}}]*)\}}',text))
    if len(matches)>1:raise AssertionError('duplicate array '+key)
    return matches[0] if matches else None

def array(text,key,required=True):
    match=array_match(text,key)
    if not match:
        if required:raise AssertionError('missing array '+key)
        return []
    return [float(x) for x in re.findall(NUMBER,match[2])]

def scalar(text,key,default=None):
    matches=re.findall(rf'(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^\n\r{{}}]+)',text)
    if len(matches)>1:raise AssertionError('duplicate scalar '+key)
    if not matches:
        if default is None:raise AssertionError('missing scalar '+key)
        return default
    return matches[0].strip()

def put_array(text,key,fn):
    match=array_match(text,key)
    if not match: return text
    values=re.findall(NUMBER,match[2])
    return text[:match.start()]+match[1]+'{'+', '.join(f'{fn(i):.17e}' for i in range(len(values)))+'}'+text[match.end():]

def nonzero(text):
    text,count=re.subn(r'(?m)^[ \t]*scaling[ \t]*=.*$', 'scaling = 1.0',text)
    if count!=1:raise AssertionError('expected one model scaling field')
    radial=re.search(r'(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)',text)
    if not radial: raise AssertionError('missing radial coefficients')
    counter=[0]
    def row(m):
        counter[0]+=1
        n=len(re.findall(NUMBER,m[0])); assert n==5
        return '{'+', '.join(f'{(.08+.013*counter[0]+.017*math.sin(.4*(i+counter[0]))) if i<3 else (1.05+.07*counter[0]+.11*math.sin(.67*(i+counter[0]))):.17e}' for i in range(n))+'}'
    text=text[:radial.start()]+radial[1]+re.sub(r'\{[^}]*\}',row,radial[2])+text[radial.end():]
    for key,bias,amp in [('moment_coeffs',.04,.12),('outer_k_rank1_weights',.13,.07),
                         ('two_layer_gate_weights',.08,.05),('two_layer_gate_k_rank_weights',.23,.11),
                         ('two_layer_gate_radial_coeffs',.13,.04),('two_layer_gate_type_coeffs',1.07,.08),
                         ('two_layer_gate_radial_mixing_coeffs',.14,.03),
                         ('adaptive_radial_weights',.3,.8),('adaptive_radial_amplitudes',.23,.07),
                         ('tensor_message_coeffs',.02,.06),('joint_readout_coeffs',.01,.04)]:
        text=put_array(text,key,lambda i,b=bias,a=amp:b+a*math.sin(i*.61+.3))
    text=put_array(text,'species_coeffs',lambda i:[-.7,1.3][i])
    text=put_array(text,'scal_coeffs',lambda i:1.8+.03*i)
    # Shared and independent gate warps deliberately differ.
    text=put_array(text,'two_layer_gate_scaling_coeffs',lambda i:1.5+.017*i)
    return text

def assert_fixture(text,spec):
    """Validate the actual serialized fixture, rather than trusting CLI options."""
    l=spec.get('l',1);body=spec.get('body',3);family=spec['family']
    expected={'scaling':'1.0','scaling_map':spec['scale'],'species_count':'2',
              'sh_l_max':str(l),'sh_k_max':'3','sh_body_order':str(body),
              'radial_basis_size':'3','radial_funcs_count':str(3*(l+1)),
              'radial_basis_type':spec.get('basis','RBChebyshev_sss')}
    if spec.get('outer_rank',3):expected['outer_k_rank']=str(spec.get('outer_rank',3))
    elif scalar(text,'outer_k_rank1_enabled','false')!='false':raise AssertionError('plain fixture has an outer rank')
    for key,value in expected.items():
        observed=scalar(text,key,'1.0' if key=='scaling' else None)
        if key=='scaling':
            if float(observed)!=1.0:raise AssertionError('model scaling was not changed')
        elif observed!=value:raise AssertionError(f'{key}: expected {value}, observed {observed}')
    head=scalar(text,'readout_mode','shared')
    if head!=spec['head']:raise AssertionError('readout_mode mismatch')
    mode=scalar(text,'outer_radial_mode','legacy-plus-mixer')
    required=['species_coeffs','moment_coeffs','scal_coeffs']
    if family in ('plain','static','legacy'):
        if mode!='legacy-plus-mixer':raise AssertionError('Legacy radial mode mismatch')
        if spec.get('outer_rank',3):required+=['outer_k_rank1_weights']
    else:
        if mode!=('pure-matrix' if family=='pure' else 'adaptive-coefficients'):
            raise AssertionError('modern radial mode mismatch')
        if array(text,'outer_k_rank1_weights'):raise AssertionError('modern factor shell is not empty')
    outer=scalar(text,'outer_channel_mode','cross-channel')
    if outer!=('same-channel' if spec['outer']=='same-k' else 'cross-channel'):
        raise AssertionError('outer channel mode mismatch')
    if family not in ('plain','static'):
        for key,value in {'two_layer_gate_enabled':'true','two_layer_gate_mode':'mu-scalar-full',
                          'two_layer_gate_radial_mode':'shared-radial',
                          'two_layer_gate_factor_pruning':spec['gate'],'two_layer_gate_k_rank':'2'}.items():
            if scalar(text,key)!=value:raise AssertionError('gate metadata mismatch: '+key)
        required+=['two_layer_gate_radial_coeffs','two_layer_gate_type_coeffs','two_layer_gate_k_rank_weights']
    elif scalar(text,'two_layer_gate_enabled','false')!='false':
        raise AssertionError('static fixture unexpectedly has a gate')
    if spec.get('independent'):
        if scalar(text,'two_layer_gate_scaling_mode')!='decoupled':
            raise AssertionError('independent gate warp not enabled')
        required+=['two_layer_gate_scaling_coeffs']
    if family=='pure':required+=['two_layer_gate_radial_mixing_coeffs']
    if family=='adaptive':
        defaults={'adaptive_radial_projection':'full','adaptive_radial_amplitude_mode':'shared',
                  'adaptive_radial_shared_n_rank':'1','adaptive_radial_activation':'none',
                  'tensor_message_lmax':'0','joint_readout_enabled':'false'}
        selected={'adaptive_radial_projection':spec['projection'],'adaptive_radial_amplitude_mode':spec['source'],
                  'adaptive_radial_shared_n_rank':str(spec['rank']),'adaptive_radial_activation':spec['activation'],
                  'tensor_message_lmax':str(spec.get('message',0)),
                  'joint_readout_enabled':str(bool(spec.get('joint'))).lower()}
        for key,value in selected.items():
            if scalar(text,key,defaults[key])!=value:raise AssertionError('Adaptive metadata mismatch: '+key)
        required+=['adaptive_radial_weights','adaptive_radial_amplitudes']
        if spec.get('message'):required+=['tensor_message_coeffs']
        if spec.get('joint'):required+=['joint_readout_coeffs']
        latent=spec['rank'] if spec['projection']=='shared-n' else 1
        amplitude_count=(l+1)*3*3*latent*(2 if spec['source']=='species' else 1)
        if len(array(text,'adaptive_radial_amplitudes'))!=amplitude_count:
            raise AssertionError('Adaptive species/latent amplitude layout mismatch')
        projection_rows=(l+1)*3*(spec['rank'] if spec['projection']=='shared-n' else 3)
        if len(array(text,'adaptive_radial_weights'))!=projection_rows*int(scalar(text,'two_layer_gate_scalar_count')):
            raise AssertionError('Adaptive U projection layout mismatch')
        if spec.get('message') and len(array(text,'tensor_message_coeffs'))!=6*spec['message']:
            raise AssertionError('tensor message coefficient layout mismatch')
    summary={}
    for key in required:
        values=array(text,key)
        if not values or not all(math.isfinite(x) for x in values) or not any(abs(x)>1e-12 for x in values):
            raise AssertionError('missing, empty, zero, or non-finite required array '+key)
        summary[key]={'count':len(values),'max_abs':max(map(abs,values))}
    blocks={'pair':1,'K':3,'L':l+1,'LK':3*(l+1)}[spec['scale']]
    if len(array(text,'scal_coeffs'))!=8*blocks:raise AssertionError('scaling-map layout mismatch')
    expected_limits=[l,l,l,l] if body!=6 else [l,l,l,min(l,2),1]
    if array(text,'sh_body_l_max')!=expected_limits:raise AssertionError('body angular limits mismatch')
    if len(array(text,'species_coeffs'))!=2:raise AssertionError('species coefficient count mismatch')
    if family not in ('plain','static') and len(array(text,'two_layer_gate_radial_coeffs'))!=9*(l+1):
        raise AssertionError('gate radial coefficient layout mismatch')
    if spec.get('zbl') and (scalar(text,'zbl_enabled')!='true' or array(text,'zbl_atomic_numbers')!=[1,8]):
        raise AssertionError('site-energy regression ZBL metadata mismatch')
    return summary

def ablation_candidates(spec):
    family=spec['family']
    if family=='pure':
        yield f'pure-matrix/l{spec.get("l",1)}','two_layer_gate_radial_mixing_coeffs'
    if family=='adaptive':
        projection=spec['projection']+('-r'+str(spec['rank']) if spec['projection']=='shared-n' else '')
        yield 'U/'+projection,'adaptive_radial_weights'
        yield 'amplitude/'+spec['source'],'adaptive_radial_amplitudes'
        if spec['activation']=='tanh':yield 'tanh/'+projection,'activation:none'
        if spec.get('message'):yield 'tensor/'+spec['projection']+'/l'+str(spec['message']),'tensor_message_coeffs'
        if spec.get('joint'):yield 'joint/'+spec['head'],'joint_readout_coeffs'

def cases(full):
    scales=('L','K','LK','pair') if full else ('pair',)
    graphs=('q-total','same-k')
    heads=('shared','species') if full else ('species',)
    for scale,head in itertools.product(scales,heads):
        yield dict(family='static',scale=scale,head=head,gate='q-total',outer='q-total')
    for scale,gate,outer,head in itertools.product(scales,graphs,graphs,heads):
        yield dict(family='legacy',scale=scale,head=head,gate=gate,outer=outer)
    for scale,gate in itertools.product(scales,graphs):
        yield dict(family='pure',scale=scale,head='shared',gate=gate,outer='same-k')
    projections=('full','shared-n')
    activations=('none','tanh') if full else ('tanh',)
    sources=('shared','species') if full else ('species',)
    for scale,gate,outer,head,proj,activation,source in itertools.product(scales,graphs,graphs,heads,projections,activations,sources):
        yield dict(family='adaptive',scale=scale,head=head,gate=gate,outer=outer,
                   projection=proj,rank=2 if proj=='shared-n' else 1,activation=activation,source=source)
    # Important intersections: tensor + joint + rank/tanh + independent warp,
    # all angular levels, body-specific truncation, and Laguerre basis.
    for l in (1,2,3,4):
        yield dict(family='adaptive',scale='K',head='species',gate='q-total',outer='same-k',
                   projection='shared-n',rank=3,activation='tanh',source='species',l=l,
                   message=min(2,l),joint=True,independent=True,body=6,
                   basis='RBLaguerre_log1p' if l==3 else 'RBChebyshev_sss')
        yield dict(family='legacy',scale='LK',head='species',gate='same-k',outer='q-total',
                   l=l,independent=True,body=6)
    for l,scale in ((2,'L'),(3,'K'),(4,'LK')):
        yield dict(family='pure',scale=scale,head='shared',gate='same-k',outer='same-k',
                   l=l,body=6,independent=True)
    yield dict(family='adaptive',scale='L',head='shared',gate='same-k',outer='q-total',
               projection='shared-n',rank=1,activation='none',source='shared',l=2)
    yield dict(family='adaptive',scale='LK',head='species',gate='same-k',outer='q-total',
               projection='full',rank=1,activation='none',source='shared',l=2,message=2,independent=True)
    yield dict(family='adaptive',scale='pair',head='shared',gate='q-total',outer='q-total',
               projection='full',rank=1,activation='none',source='shared',l=2,joint=True)
    for zbl in (False,True):
        yield dict(family='plain',scale='pair',head='shared',gate='q-total',outer='q-total',
                   outer_rank=0,site_regression=True,zbl=zbl)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--checker',type=Path)
    p.add_argument('--ablation-oracle',type=Path,
                   help='Fixture tool supporting eval-efs; enables measured feature-effect assertions')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--full',action='store_true')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    cfg=args.out/'asymmetric.cfg'
    cfg.write_text('''BEGIN_CFG
 Size
 3
 Supercell
 7.4 .21 .09
 .12 7.8 .18
 .05 .16 8.1
 AtomData: id type cartes_x cartes_y cartes_z
 1 0 1.17 1.39 1.52
 2 1 2.63 2.11 2.74
 3 0 4.22 3.05 1.88
END_CFG
''')
    rows=[];qualified_ablations={}
    def run(name,cmd):
        cp=subprocess.run(list(map(str,cmd)),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},timeout=600)
        (args.out/(name+'.log')).write_text(cp.stdout)
        if cp.returncode: raise RuntimeError(name+'\n'+cp.stdout[-5000:])
        return cp.stdout
    def efs(name,path):
        output=run(name,[args.ablation_oracle,'eval-efs',path,cfg])
        lines=[line for line in output.splitlines() if line.startswith('NATIVE_EFS_JSON ')]
        if len(lines)!=1:raise AssertionError('missing native ablation result')
        result=json.loads(lines[0].split(' ',1)[1])
        if len(result['forces'])!=3 or any(len(row)!=3 for row in result['forces']) or \
           len(result['site_energies'])!=3 or len(result['virial'])!=3 or any(len(row)!=3 for row in result['virial']):
            raise AssertionError('wrong native ablation result shape')
        values=[result['energy']]+list(itertools.chain.from_iterable(result['forces']))+list(itertools.chain.from_iterable(result['virial']))
        if not all(math.isfinite(x) for x in values+result['site_energies']):
            raise AssertionError('non-finite native ablation observable')
        if abs(sum(result['site_energies'])-result['energy'])>1e-10*(1+abs(result['energy'])):
            raise AssertionError('native ablation site-energy sum mismatch')
        return values
    report={'status':'RUNNING','full':args.full,'cases':rows,'feature_ablations':qualified_ablations}
    try:
        for index,spec in enumerate(cases(args.full)):
            label=f'{index:04d}-{spec["family"]}'; top=args.out/(label+'.topology.sus2'); material=args.out/(label+'.material.sus2')
            model=args.out/(label+'.sus2'); l=spec.get('l',1); body=spec.get('body',3)
            limits=[l,l,l,l] if body!=6 else [l,l,l,min(l,2),1]
            opts=['--species-count=2',f'--l-max={l}','--k-max=3',f'--body-order={body}',
                  '--body-l-max='+','.join(map(str,limits)),'--radial-basis-size=3','--cutoff=4.8',
                  '--radial-basis-type='+spec.get('basis','RBChebyshev_sss'),'--scaling-map='+spec['scale'],
                  '--outer-k-rank='+str(spec.get('outer_rank',3)),
                  '--sh-factor-pruning='+spec['outer'],'--readout-mode='+spec['head']]
            if spec.get('zbl'):opts+=['--zbl-elements=H,O','--zbl-inner=1.0','--zbl-outer=4.0']
            if spec['family'] not in ('plain','static'):
                opts+=['--two-layer-gate','--two-layer-gate-mode=mu-scalar-full','--two-layer-gate-radial-mode=shared-radial',
                       '--two-layer-gate-k-rank=2','--two-layer-gate-factor-pruning='+spec['gate'],f'--two-layer-gate-body-order={body}']
            if spec.get('independent'):opts+=['--two-layer-gate-decoupled-scaling']
            if spec['family']=='pure':opts+=['--superlinear-mode=pure-w-channelwise','--two-layer-gate-radial-mixing=full']
            if spec['family']=='adaptive':
                opts+=['--superlinear-mode=radial-coefficient-gate','--adaptive-radial-projection='+spec['projection'],
                       '--adaptive-radial-shared-n-rank='+str(spec['rank']),'--adaptive-radial-activation='+spec['activation'],
                       '--adaptive-radial-amplitude-mode='+spec['source'],'--tensor-message-lmax='+str(spec.get('message',0))]
                if spec.get('joint'):opts+=['--joint-readout=true']
            run(label+'-init',[args.native,'init-sh',top,*opts])
            run(label+'-materialize',[args.native,'roundtrip-model-dev',top,material])
            material_text=material.read_text()
            text=nonzero(material_text); array_summary=assert_fixture(text,spec)
            for key,summary in array_summary.items():
                previous=array(material_text,key);current=array(text,key)
                if len(previous)!=len(current) or previous==current:
                    raise AssertionError('required array was not materialized and mutated: '+key)
                summary['mutation_l2_norm']=math.sqrt(math.fsum((a-b)**2 for a,b in zip(previous,current)))
            model.write_text(text)
            roundtrip=args.out/(label+'.roundtrip.sus2')
            run(label+'-schema',[args.native,'roundtrip-model-dev',model,roundtrip])
            assert_fixture(roundtrip.read_text(),spec)
            row=dict(index=index,spec=spec,model=str(model.resolve()),sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                     required_arrays=array_summary)
            if args.checker:
                output=run(label+'-distributed',[args.checker,model,cfg])
                lines=[x for x in output.splitlines() if x.startswith('DISTRIBUTED_CPU_PASS ')]
                if len(lines)!=1:raise AssertionError('missing numerical result')
                row['result']=json.loads(lines[0].split(' ',1)[1])
            if args.ablation_oracle:
                if spec.get('site_regression'):
                    efs(label+'-site-energy-sum',model)
                    row['site_energy_sum_regression']='PASS'
                pending=[(feature,key) for feature,key in ablation_candidates(spec) if feature not in qualified_ablations]
                if pending:
                    baseline=efs(label+'-effect-baseline',model)
                    row['feature_ablations']={}
                    for feature,key in pending:
                        name=label+'-effect-'+re.sub('[^a-zA-Z0-9-]','-',feature)
                        altered=args.out/(name+'.sus2')
                        if key=='activation:none':
                            changed,count=re.subn(r'(?m)^([ \t]*adaptive_radial_activation[ \t]*=)[ \t]*tanh[ \t]*$',
                                                  r'\1 none',text)
                            if count!=1:raise AssertionError('missing active tanh field')
                        else:
                            if not array(text,key):raise AssertionError('vacuous ablation '+key)
                            changed=put_array(text,key,lambda i:0.0)
                        altered.write_text(changed)
                        observed=efs(name,altered)
                        deltas=[abs(a-b) for a,b in zip(baseline,observed)]
                        # Exceed the unchanged table-vs-direct check tolerance
                        # by at least tenfold in an observable component.
                        ratio=max(abs(a-b)/(2e-7+2e-7*max(abs(a),abs(b))) for a,b in zip(baseline,observed))
                        if max(deltas)<=1e-6 or ratio<=10:
                            raise AssertionError(f'vacuous {feature}: max_abs={max(deltas):.4g}, tolerance_ratio={ratio:.4g}')
                        effect={'case':index,'max_abs_effect':max(deltas),'table_tolerance_ratio':ratio}
                        qualified_ablations[feature]=effect;row['feature_ablations'][feature]=effect
            rows.append(row)
            (args.out/'matrix.json').write_text(json.dumps(report,indent=2)+'\n')
            print(label,'PASS' if args.checker else 'FIXTURE_READY',flush=True)
        if args.ablation_oracle:
            expected={feature for spec in cases(args.full) for feature,key in ablation_candidates(spec)}
            if expected!=set(qualified_ablations):raise AssertionError('feature ablation coverage is incomplete')
        report['status']='PASS' if args.checker else 'FIXTURES_ONLY'
    except Exception as error:
        report['status']='FAILED';report['error']=str(error);raise
    finally:(args.out/'matrix.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
