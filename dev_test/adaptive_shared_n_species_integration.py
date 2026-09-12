#!/usr/bin/env python3
"""Server-allocation-only compact adaptive projection/source-species audit.

No tolerances are relaxed to turn an unsupported precision or vacuous zero
derivative into a pass. Native CFG formatting has its own output tolerance.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cpp', 'old-cpp', 'native-root', 'pytorch-root', 'outdir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--stage', choices=('minimal', 'full', 'tensor-extended'), default='minimal')
    p.add_argument('--native-only', action='store_true')
    args = p.parse_args()
    if sys.platform == 'darwin' or not os.environ.get('LSB_JOBID'):
        raise SystemExit('Run numerical checks only in a server LSF allocation')
    sys.path.insert(0, str(args.native_root.resolve()/'dev_test'))
    sys.path.insert(0, str(args.pytorch_root.resolve()))
    import numpy as np
    from adaptive_species_readout_check import (assignment, numbers, put, put_numbers,
        asymmetric_shared_text, write_fixture, parse_prediction, dev_fields, ZERO_REG)
    from adaptive_tensor_message_check import TensorCheck, topology_args
    out = args.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    native = TensorCheck(args.cpp, out/'native', args.old_cpp)
    native.cfg = native.out/'asymmetric.cfg'; write_fixture(native.cfg)
    report = {'status':'RUNNING', 'stage':args.stage, 'device':args.device,
              'native_only':args.native_only, 'job_id':os.environ['LSB_JOBID'],
              'checks':{}, 'cases':{}}
    def record(name, value):
        report['checks'][name] = value
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(value), flush=True)
    def close(name, x, y, atol=2e-9, rtol=2e-8):
        def arr(a):
            return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
        x, y = arr(x), arr(y)
        assert x.shape == y.shape, (name, x.shape, y.shape)
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=name)
        record(name, {'max_abs':float(abs(x-y).max()), 'atol':atol, 'rtol':rtol})
    def cfg_prediction(label, model, *, reference=False, binary=None):
        target = native.out/(label+'.prediction.cfg')
        assert target.resolve() not in (Path(model).resolve(), native.cfg.resolve())
        native.run(label, 'calc-efs', model, native.cfg, target, binary=binary,
                   env={'SUS2_SH_ADAPTIVE_EFS_REFERENCE':str(int(reference))})
        return parse_prediction(target)
    def make_case(label, projection, amplitude_mode, gate='q-total', outer='same-k',
                  readout='shared', message=0, scaling='K', deep_checks=False):
        top = native.out/(label+'.topology.mtp')
        full = native.out/(label+'.materialized.mtp')
        native.run(label+'-init', 'init-sh', top,
            *topology_args(gate=gate, outer=outer, readout=readout, message=message, scaling=scaling),
            '--adaptive-radial-projection='+projection,
            '--adaptive-radial-amplitude-mode='+amplitude_mode)
        native.run(label+'-materialize', 'roundtrip-model-dev', top, full)
        text = put(asymmetric_shared_text(full.read_text()), 'readout_mode', readout)
        if message:
            size = len(numbers(text, 'tensor_message_coeffs'))
            text = put_numbers(text, 'tensor_message_coeffs', .07*np.cos(np.arange(size)*.41+.27))
        D, Q = 3*3*3, int(assignment(text, 'two_layer_gate_scalar_count'))
        acount = D*(2 if amplitude_mode == 'species' else 1)
        wcount = Q*(D if projection == 'full' else D//3)
        assert len(numbers(text, 'adaptive_radial_amplitudes')) == acount
        assert len(numbers(text, 'adaptive_radial_weights')) == wcount
        assert int(assignment(text,'adaptive_radial_amplitude_count')) == acount
        assert int(assignment(text,'adaptive_radial_weight_count')) == wcount
        if projection != 'full' or amplitude_mode != 'shared':
            assert assignment(text,'version') == '1.5.0'
            assert assignment(text,'adaptive_radial_projection') == projection
            assert assignment(text,'adaptive_radial_amplitude_mode') == amplitude_mode
        model = native.out/(label+'.mtp'); model.write_text(text)
        roundtrip = native.out/(label+'.roundtrip.mtp')
        native.run(label+'-roundtrip', 'roundtrip-model-dev', model, roundtrip)
        value = cfg_prediction(label+'-fast', model)
        close(label+'-native-roundtrip', value, cfg_prediction(label+'-roundtrip-efs',roundtrip), 2e-10,2e-10)
        close(label+'-native-reference',value,cfg_prediction(label+'-reference',model,reference=True),2e-10,2e-10)
        if projection == 'full' and amplitude_mode == 'shared':
            close(label+'-old-binary',value,cfg_prediction(label+'-old',model,binary=args.old_cpp),2e-10,2e-10)
        else:
            native.run(label+'-old-reject', 'roundtrip-model-dev', model,
                native.out/(label+'.old-should-not-exist.mtp'), binary=args.old_cpp, fail=True)
        layout = native.layout(label+'-layout',model)
        ranges = {key:(int(layout[key+'_begin']),int(layout[key+'_end']))
                  for key in ('adaptive_radial_amplitude','adaptive_radial_weight','gate_k_rank',
                              'gate_radial','gate_type','radial')}
        ranges['scaling'] = (2,int(layout['radial_begin']))
        if message:
            ranges['tensor_message'] = (int(layout['tensor_message_begin']),int(layout['tensor_message_end']))
            assert ranges['tensor_message'][1]-ranges['tensor_message'][0] == message*3*2
        assert ranges['adaptive_radial_amplitude'][1]-ranges['adaptive_radial_amplitude'][0] == acount
        assert ranges['adaptive_radial_weight'][1]-ranges['adaptive_radial_weight'][0] == wcount
        directions = {}
        keys = tuple(ranges) if label == 'target' or deep_checks else ('adaptive_radial_amplitude','adaptive_radial_weight')
        if message and 'tensor_message' not in keys:
            keys += ('tensor_message',)
        for key in keys:
            directions[key] = native.direction(label+'-mixed-'+key,model,*ranges[key])
        isolated_directions = {}
        if amplitude_mode == 'species' and (label == 'target' or deep_checks):
            begin,_ = ranges['adaptive_radial_amplitude']
            for z in range(2):
                directions['amplitude_source_'+str(z)] = native.direction(
                    label+'-amplitude-source-'+str(z),model,begin+z*D,begin+(z+1)*D)
            for observable,weights in (('force',(0.,1.,0.)),('virial',(0.,0.,1.))):
                isolated_directions[observable] = {}
                isolated_keys = tuple(ranges) if deep_checks else ('adaptive_radial_amplitude','adaptive_radial_weight','gate_k_rank')
                for key in isolated_keys:
                    isolated_directions[observable][key] = native.direction(
                        label+'-'+observable+'-'+key,model,*ranges[key],weights=weights)
        native.run(label+'-geometry-fd','check-efs-fd-dev',model,native.cfg,
            '--max-configs=1','--max-atoms=3','--displacement=2e-5',
            '--abs-tolerance=2e-6','--rel-tolerance=2e-4')
        if label == 'target' or deep_checks:
            native.run(label+'-linear-EFS','check-linear-readout-dev',model,native.cfg,
                '--max-configs=1','--max-atoms=3','--check-stress',
                '--abs-tolerance=2e-9','--rel-tolerance=2e-9')
            # Shared readout has two intentional component charts: the raw
            # Phi columns used by TrainLinear and fixed-center h[type]*Phi.
            # Exercise the compact n reduction independently for both.
            charts=(('raw',()),('fixed-center',('--fixed-center-readout-columns',))) if readout == 'shared' else (('species',()),)
            implementations=(('batched',{'SUS2_SH_ADAPTIVE_COMPONENTS':'batched'}),
                             ('reference',{'SUS2_SH_ADAPTIVE_COMPONENTS':'reference'})) if deep_checks else (('',{}),)
            for implementation,env in implementations:
                for chart,flags in charts:
                    suffix=chart+('-'+implementation if implementation else '')
                    fields=dev_fields(native.run(label+'-components-FD-'+suffix,
                        'check-linear-components-fd-dev',model,native.cfg,
                        '--max-configs=1','--max-atoms=3',*flags,'--displacement=2e-5',
                        '--abs-tolerance=5e-8','--rel-tolerance=5e-5',env=env))
                    assert int(fields['checked_components'])>0,(label,suffix,'vacuous component check')
                    record(label+'-components-FD-'+suffix,fields)
        case = dict(model=str(model),topology=str(top),projection=projection,amplitude=amplitude_mode,
                    gate=gate,outer=outer,readout=readout,message=message,scaling=scaling,
                    amplitude_count=acount,weight_count=wcount,layout=layout,directions=directions,
                    deep_checks=deep_checks,isolated_directions=isolated_directions)
        report['cases'][label] = case
        record(label+'-native', {k:v for k,v in case.items() if k not in ('directions','layout')})
        return case

    make_case('target','shared-n','species')
    if args.stage == 'full':
        make_case('legacy','full','shared')
        for projection,amp in (('shared-n','shared'),('full','species'),('shared-n','species')):
            for gate in ('same-k','q-total'):
                for outer in ('same-k','q-total'):
                    for readout in ('shared','species'):
                        label=f'{projection}_{amp}_{gate}_{outer}_{readout}'
                        make_case(label,projection,amp,gate,outer,readout)
        for projection,amp in (('shared-n','shared'),('full','species'),('shared-n','species')):
            for msg,scaling in ((1,'L'),(2,'LK')):
                label=f'{projection}_{amp}_tensor{msg}_{scaling}'
                make_case(label,projection,amp,'same-k','q-total','species',msg,scaling)
    elif args.stage == 'tensor-extended':
        make_case('tensor_L1_cross_same_shared','shared-n','species',
                  'q-total','same-k','shared',1,'L',deep_checks=True)
        make_case('tensor_LK2_same_cross_shared','shared-n','species',
                  'same-k','q-total','shared',2,'LK',deep_checks=True)
    if args.native_only:
        report['status']='PASS'; record('native-complete',{'cases':len(report['cases'])})
        return

    import torch
    from sus2shpt.io.cfg import load_cfg
    from sus2shpt.io.mtp import load_mtp_model,save_mtp_model
    from sus2shpt.model.sus2_sh import SUS2SHPyTorch,_apply_product_levels
    from sus2shpt.model.neighbor import build_pair_list
    from sus2shpt.training import (TrainingOptions,rescale_initial_parameters,
        _configuration_batch_forward,_prepare_training_graph,_maybe_parametrize_radial_weight_norm)
    from sus2shpt.adaptive_training import parametrize_adaptive_amplitudes
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise AssertionError('CUDA unavailable: this is not a passing skip')
    cfg=load_cfg(native.cfg)[0]
    def create(path,kernel='vectorized'):
        obj=SUS2SHPyTorch(load_mtp_model(path),device=args.device,dtype=torch.float64)
        obj.adaptive_kernel=kernel
        return obj
    def efs(obj,configuration=cfg):
        r=obj.forward_configuration(configuration,compute_forces=True,compute_stress=True,create_graph=True)
        return torch.cat((r.energy.reshape(-1),r.forces.reshape(-1),r.stress.reshape(-1)))
    def native_efs9(label,path):
        dest=native.out/(label+'.efs9.cfg')
        native.run(label,'calc-efs',path,native.cfg,dest)
        c=load_cfg(dest)[0]
        return np.r_[c.energy,np.asarray([a.force for a in c.atoms]).ravel(),np.asarray(c.stress).ravel()]
    def loss(obj,weights=(1.,.1,.01)):
        r=obj.forward_configuration(cfg,create_graph=True)
        f=r.forces.new_tensor(np.asarray([a.force for a in cfg.atoms]))
        s=r.stress.new_tensor(np.asarray(cfg.stress))
        return weights[0]*(r.energy-float(cfg.energy)).square()/cfg.size+weights[1]*(r.forces-f).square().sum()+weights[2]*(r.stress-s).square().sum()/cfg.size
    def packed_index(begin,rank,lmax,l,lane,m):
        return begin+rank*((lmax+1)**2-(l+1)**2)+(rank-1-lane)*(2*l+1)+(m+l)
    def oracle_energy(obj,positions,cell,types,source=True):
        """Literal contractions, independent of compact adaptive kernels."""
        pair=build_pair_list(positions,cell,obj.max_dist)
        geo=obj._edge_geometry(positions,types,pair.centers,pair.neighbors,pair.shifts@cell)
        gate=obj._moment_values_from_geometry(atom_count=len(types),centers=pair.centers,
            neighbors=pair.neighbors,geometry=geo,gate_pass=True,gate_signal=None,edge_l1_values=None)
        S=gate.index_select(1,obj.gate_scalar_moment_indices)*obj.gate_scalar_scale
        radial=obj._raw_radial_values(geo.pair_r,geo.center_types,geo.neighbor_types)
        G,O=obj.gate_k_rank,obj.outer_k_rank
        gb=int(obj.mtp.spec.two_layer_gate_k_rank_basic_begin)
        ob=int(obj.mtp.spec.outer_k_rank1_basic_begin)
        values=[];targets=[]
        compact=getattr(obj.mtp.spec,'adaptive_radial_projection','full')=='shared-n'
        species=getattr(obj.mtp.spec,'adaptive_radial_amplitude_mode','shared')=='species'
        for l in range(obj.lmax+1):
            for o in range(O):
                mu=o*(obj.lmax+1)+l
                if compact:
                    signal=sum(S[:,q]*obj.adaptive_radial_weights[l,o,q] for q in range(S.shape[1]))
                    signal=signal.index_select(0,pair.neighbors)[:,None]
                else:
                    signal=sum(S[:,q,None]*obj.adaptive_radial_weights[l,o,:,q] for q in range(S.shape[1]))
                    signal=signal.index_select(0,pair.neighbors)
                a=(obj.adaptive_radial_amplitudes[types[pair.neighbors if source else pair.centers],l,o,:]
                   if species else obj.adaptive_radial_amplitudes[l,o,:])
                coeff=obj.radial_coeffs[mu,:obj.rb_size]+a*signal
                R=(coeff*radial[:,mu,:]).sum(1)
                vv=obj.radial_coeffs[mu,obj.rb_size+geo.center_types]*obj.radial_coeffs[mu,obj.rb_size+geo.neighbor_types]
                for m in range(-l,l+1):
                    angular=geo.sh_values[:,l*l+l+m]
                    if 1<=l<=obj.tensor_message_lmax:
                        H=sum(obj.tensor_message_coeffs[l-1,o,g]*obj.tensor_message_irrep_scale[l-1,g]
                              *gate[:,packed_index(gb,G,obj.lmax,l,g,m)] for g in range(G))
                        angular=angular+H.index_select(0,pair.neighbors)
                    values.append(positions.new_zeros(len(types)).index_add(0,pair.centers,vv*R*angular))
                    targets.append(packed_index(ob,O,obj.lmax,l,o,m))
        moments=positions.new_zeros((len(types),obj.alpha_moments_count)).index_add(
            1,torch.tensor(targets,device=obj.device_arg),torch.stack(values,1))
        moments=_apply_product_levels(moments,obj.alpha_moments_count,obj.outer_product_levels)
        scalars=moments.index_select(1,obj.alpha_moment_mapping)*obj.outer_scalar_scale
        head=obj.readout_output_coeffs()
        r=(scalars*head[types]).sum(1) if obj.readout_mode=='species' else scalars@head
        h=obj.species_coeffs[types]
        return (obj.shift_coeffs[types]+h+(r if obj.readout_mode=='species' else h*r)).sum()
    def oracle_efs(obj,source=True):
        pos=torch.tensor(np.asarray([a.position for a in cfg.atoms]),device=obj.device_arg,dtype=obj.dtype,requires_grad=True)
        cell=torch.tensor(np.asarray(cfg.supercell),device=obj.device_arg,dtype=obj.dtype)
        types=torch.tensor([a.type for a in cfg.atoms],device=obj.device_arg,dtype=torch.long)
        E=oracle_energy(obj,pos,cell,types,source)
        F=-torch.autograd.grad(E,pos,create_graph=True)[0]
        strained=cell.detach().clone().requires_grad_(True)
        fraction=pos.detach()@torch.linalg.inv(cell)
        Es=oracle_energy(obj,fraction@strained,strained,types,source)
        V=-torch.autograd.grad(Es,strained,create_graph=True)[0].T@strained
        return torch.cat((E.reshape(-1),F.reshape(-1),V.reshape(-1)))
    names=('adaptive_radial_amplitudes','adaptive_radial_weights','two_layer_gate_k_rank_weights',
           'two_layer_gate_radial_coeffs','two_layer_gate_type_coeffs','radial_coeffs','scal_coeffs','moment_coeffs')
    mapping=dict(adaptive_radial_amplitudes='adaptive_radial_amplitude',adaptive_radial_weights='adaptive_radial_weight',
                 two_layer_gate_k_rank_weights='gate_k_rank',two_layer_gate_radial_coeffs='gate_radial',
                 two_layer_gate_type_coeffs='gate_type',radial_coeffs='radial',scal_coeffs='scaling',
                 tensor_message_coeffs='tensor_message')
    for label,case in report['cases'].items():
        path=Path(case['model']);obj=create(path);ref=create(path,'reference')
        if case['message']: local_names=names+('tensor_message_coeffs',)
        else: local_names=names
        current=efs(obj)
        close(label+'-PT-native',current,native_efs9(label+'-PT-native',path),8e-6,5e-7)
        close(label+'-PT-reference',current,efs(ref))
        close(label+'-literal-source-oracle',current,oracle_efs(obj))
        if case['amplitude']=='species':
            wrong=oracle_efs(obj,source=False)
            difference=float((current.detach()-wrong.detach()).abs().max())
            assert difference>1e-10,(label,'vacuous source-vs-receiver test',difference)
            record(label+'-source-not-receiver',{'wrong_receiver_max_difference':difference})
        gradients=torch.autograd.grad(loss(obj),[getattr(obj,n) for n in local_names])
        reference_gradients=torch.autograd.grad(loss(ref),[getattr(ref,n) for n in local_names])
        for name,g,rg in zip(local_names,gradients,reference_gradients):
            assert torch.isfinite(g).all() and float(g.abs().max())>1e-12,(label,name,'vacuous')
            close(label+'-mixed-'+name,g,rg,5e-8,2e-7)
            key=mapping.get(name)
            if key in case['directions']:
                flat=g.reshape(-1);direction=torch.arange(flat.numel(),device=obj.device_arg)%5-2
                direction=torch.where(direction==0,1,direction).to(flat)
                close(label+'-native-direction-'+name,(flat*direction).sum(),
                      case['directions'][key]['analytic_directional'],5e-8,5e-6)
            if case['deep_checks'] and name=='adaptive_radial_amplitudes':
                for z in range(2):
                    flat=g[z].reshape(-1);direction=torch.arange(flat.numel(),device=obj.device_arg)%5-2
                    direction=torch.where(direction==0,1,direction).to(flat)
                    close(label+'-native-direction-amplitude-source-'+str(z),(flat*direction).sum(),
                          case['directions']['amplitude_source_'+str(z)]['analytic_directional'],5e-8,5e-6)
        if case['deep_checks']:
            for observable,weights in (('force',(0.,1.,0.)),('virial',(0.,0.,1.))):
                isolated=torch.autograd.grad(loss(obj,weights),[getattr(obj,n) for n in local_names])
                for name,g in zip(local_names,isolated):
                    key=mapping.get(name)
                    if key not in case['isolated_directions'][observable]: continue
                    assert torch.isfinite(g).all(),(label,observable,name,'nonfinite')
                    flat=g.reshape(-1);direction=torch.arange(flat.numel(),device=obj.device_arg)%5-2
                    direction=torch.where(direction==0,1,direction).to(flat)
                    close(label+'-native-'+observable+'-direction-'+name,(flat*direction).sum(),
                          case['isolated_directions'][observable][key]['analytic_directional'],5e-8,5e-6)
        if case['message'] and case['projection']=='shared-n' and case['amplitude']=='species':
            # Make the literal irrep-scale factor nonvacuous without changing
            # the physical model or the parameter chart used above for native
            # directional comparisons.
            scaled=create(path)
            scale=1.17+.09*torch.arange(scaled.tensor_message_irrep_scale.numel(),
                device=scaled.device_arg,dtype=scaled.dtype).reshape_as(scaled.tensor_message_irrep_scale)
            scaled.set_tensor_message_irrep_scale(scale,preserve_function=True)
            close(label+'-tensor-scale-preserve',efs(scaled),current)
            close(label+'-tensor-scale-literal-oracle',efs(scaled),oracle_efs(scaled))
        if label=='target':
            cfg2=replace(cfg,atoms=[replace(a,position=np.asarray(a.position)+[.031*(i+1),-.017*i,.009]) for i,a in enumerate(cfg.atoms)])
            packed=_configuration_batch_forward(obj,[_prepare_training_graph(obj,c) for c in (cfg,cfg2)],create_graph=True)
            singles=[efs(obj,c) for c in (cfg,cfg2)]
            pv=torch.cat((packed.graph_energies.reshape(-1),packed.forces.reshape(-1),packed.stress.reshape(-1)))
            sv=torch.cat((torch.stack([x[0] for x in singles]),torch.cat([x[1:-9] for x in singles]),torch.cat([x[-9:] for x in singles])))
            close('target-packed-EFS',pv,sv)
            pg=torch.autograd.grad(pv.square().sum(),[getattr(obj,n) for n in names])
            sg=torch.autograd.grad(sv.square().sum(),[getattr(obj,n) for n in names])
            for name,g,rg in zip(names,pg,sg): close('target-packed-mixed-'+name,g,rg,5e-8,2e-7)
            before=efs(obj).detach()
            rescale_initial_parameters(obj,[cfg,cfg2],preserve_function=True,min_scale=1e-12,max_scale=1e12)
            obj.set_readout_output_scale(2.37,preserve_function=True)
            parametrize_adaptive_amplitudes(obj)
            _maybe_parametrize_radial_weight_norm(obj,TrainingOptions(radial_weight_norm=True))
            close('target-normalization-weightnorm-preserve',efs(obj),before)
            exported=out/'target-normalized.mtp';save_mtp_model(obj.to_mtp_model(),exported)
            close('target-normalized-export-PT',efs(create(exported)),before)
            close('target-normalized-export-native',native_efs9('target-normalized-native',exported),before,8e-6,5e-7)
    report['status']='PASS';record('complete',{'cases':len(report['cases'])})

if __name__=='__main__':
    main()
