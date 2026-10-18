#!/usr/bin/env python3
"""Tensor-message native/PT audit: minimal first, full before deployment.

Numerics are prohibited outside a server LSF allocation. In-memory double
checks keep strict tolerances; native CFG text output has a separate existing
serialization tolerance. No zero-B-only result is accepted as feature proof.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cpp","old-cpp","native-root","pytorch-root","cfg","outdir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--stage",choices=("minimal","full"),default="minimal")
    parser.add_argument("--native-report",type=Path,help="Reuse a completed same-stage native report in this allocation/filesystem")
    args=parser.parse_args()
    if sys.platform=="darwin" or not os.environ.get("LSB_JOBID"):
        raise SystemExit("Run numerics only within a server LSF compute allocation")
    sys.path.insert(0,str(args.pytorch_root.resolve()))
    sys.path.insert(0,str(args.native_root.resolve()/"dev_test"))
    import numpy as np
    import torch
    import adaptive_tensor_message_check as helper
    from sus2shpt.io.cfg import load_cfg
    from sus2shpt.io.sus2 import MTPParseError, load_mtp_model, save_mtp_model
    from sus2shpt.model.sus2_sh import SUS2SHPyTorch, _apply_product_levels
    from sus2shpt.model.neighbor import build_pair_list
    from sus2shpt.topology_init import TopologyInitializationOptions, materialize_topology_parameters
    from sus2shpt.training import (TrainingOptions, rescale_initial_parameters,
        _configuration_batch_forward, _prepare_training_graph, _maybe_parametrize_radial_weight_norm,
        _apply_species_readout_gauge_fix)
    from sus2shpt.adaptive_training import parametrize_adaptive_amplitudes
    torch.set_num_threads(1)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise AssertionError("CUDA unavailable, not a passing skip")
    out=args.outdir.resolve(); out.mkdir(parents=True,exist_ok=True)
    native=helper.TensorCheck(args.cpp,out/"native",args.old_cpp)
    if args.native_report:
        native_report_path=args.native_report.resolve()
        native.report=json.loads(native_report_path.read_text())
        assert native.report["status"]=="PASS" and native.report["stage"]==args.stage
        # Reuse immutable models/report, not their output destination. All
        # subsequent calc-efs/init logs belong to THIS integration attempt.
        native_artifacts=native_report_path.parent
        native.cfg=native_artifacts/"asymmetric.cfg"
    else:
        native.check(args.stage)
        native_artifacts=native.out
        native_report_path=(native_artifacts/"report.json").resolve()
    assert native_report_path.is_file()
    (out/"native_report_path.txt").write_text(str(native_report_path)+"\n")
    report={"status":"RUNNING","stage":args.stage,"device":args.device,
            "native_report":str(native_report_path),"native_report_reused":bool(args.native_report),"checks":{}}
    def record(name,value):
        report["checks"][name]=value
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,json.dumps(value),flush=True)
    def array(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    def close(name,a,b,*,atol=2e-9,rtol=2e-8):
        a,b=array(a),array(b)
        assert a.shape==b.shape,(name,a.shape,b.shape)
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)
        record(name,{"max_abs":float(abs(a-b).max()),"atol":atol,"rtol":rtol})
    def model(path,kernel="vectorized",device=None):
        obj=SUS2SHPyTorch(load_mtp_model(path),device=device or args.device,dtype=torch.float64)
        obj.adaptive_kernel=kernel
        return obj
    def efs(obj,cfg,graph=True):
        value=obj.forward_configuration(cfg,compute_forces=True,compute_stress=True,create_graph=graph)
        return torch.cat((value.energy.reshape(-1),value.forces.reshape(-1),value.stress.reshape(-1)))
    def objective(obj,cfg,weights=(1.,.1,.01)):
        value=obj.forward_configuration(cfg,create_graph=True)
        target_f=value.forces.new_tensor(np.asarray([a.force for a in cfg.atoms]))
        target_s=value.stress.new_tensor(np.asarray(cfg.stress))
        return (weights[0]*(value.energy-float(cfg.energy)).square()/cfg.size
                +weights[1]*(value.forces-target_f).square().sum()
                +weights[2]*(value.stress-target_s).square().sum()/cfg.size)
    def native_efs(label,path,cfg_path):
        target=out/(label+".cfg")
        native.run(label,"calc-efs",path,cfg_path,target)
        return np.concatenate([np.r_[cfg.energy,np.asarray([a.force for a in cfg.atoms]).ravel(),np.asarray(cfg.stress).ravel()] for cfg in load_cfg(target)])
    def finite_difference(label,obj,cfg,name,gradient=None,weights=(1.,.1,.01)):
        parameter=getattr(obj,name); initial=parameter.detach().clone()
        direction=torch.cos(torch.arange(parameter.numel(),device=obj.device_arg,dtype=obj.dtype)*.37+.21).reshape_as(parameter)
        if gradient is None: gradient=torch.autograd.grad(objective(obj,cfg,weights),parameter)[0]
        analytic=float((gradient*direction).sum().detach()); streak=0; records=[]
        assert abs(analytic)>1e-10,(label,"vacuous",analytic)
        try:
            for h in (2e-5,5e-6,1.25e-6,3.125e-7,7.8125e-8):
                with torch.no_grad(): parameter.copy_(initial+h*direction)
                plus=float(objective(obj,cfg,weights).detach())
                with torch.no_grad(): parameter.copy_(initial-h*direction)
                minus=float(objective(obj,cfg,weights).detach())
                fd=(plus-minus)/(2*h)
                passed=bool(np.isclose(fd,analytic,atol=5e-7,rtol=5e-5))
                records.append({"h":h,"analytic":analytic,"fd":fd,"pass":passed})
                record(label,records)
                streak=streak+1 if passed else 0
                if streak>=2: break
            assert streak>=2,(label,records)
        finally:
            with torch.no_grad(): parameter.copy_(initial)

    def packed_index(begin,rank,lmax,l,lane,m):
        # Independent contract indexing, not model.gate_rank_target_indices.
        return begin+rank*((lmax+1)**2-(l+1)**2)+(rank-1-lane)*(2*l+1)+(m+l)

    def oracle_energy(obj,positions,cell,types):
        """Literal H=sum_g B*z and edge vv*R*(Y+H) in scalar loops.

        Reuses established radial/SH/gate-graph primitives, never the new H
        builder or adaptive outer kernel. Both receives are summed separately
        and tensor z is selected from the post-P/pre-products primitive slots.
        This oracle is used before any normalization/parameter re-expression.
        """
        pair=build_pair_list(positions,cell,obj.max_dist)
        geometry=obj._edge_geometry(positions,types,pair.centers,pair.neighbors,pair.shifts@cell)
        gate=obj._moment_values_from_geometry(atom_count=len(types),centers=pair.centers,
            neighbors=pair.neighbors,geometry=geometry,gate_pass=True,gate_signal=None,edge_l1_values=None)
        selected=gate.index_select(1,obj.gate_scalar_moment_indices)*obj.gate_scalar_scale
        radial=obj._raw_radial_values(geometry.pair_r,geometry.center_types,geometry.neighbor_types)
        B=obj.tensor_message_coeffs
        assert torch.equal(obj.tensor_message_irrep_scale,torch.ones_like(obj.tensor_message_irrep_scale))
        gb=int(obj.sus2.spec.two_layer_gate_k_rank_basic_begin); ob=int(obj.sus2.spec.outer_k_rank1_basic_begin)
        G,O=obj.gate_k_rank,obj.outer_k_rank; contributions=[]; targets=[]
        for l in range(obj.lmax+1):
            for o in range(O):
                mu=o*(obj.lmax+1)+l
                w=sum(selected[:,q,None]*obj.adaptive_radial_weights[l,o,:,q] for q in range(selected.shape[1]))
                coeff=obj.radial_coeffs[mu,:obj.rb_size]+obj.adaptive_radial_amplitudes[l,o]*w.index_select(0,pair.neighbors)
                R=(radial[:,mu,:]*coeff).sum(1)
                vv=obj.radial_coeffs[mu,obj.rb_size+geometry.center_types]*obj.radial_coeffs[mu,obj.rb_size+geometry.neighbor_types]
                for m in range(-l,l+1):
                    angular=geometry.sh_values[:,l*l+m+l]
                    if 1<=l<=obj.tensor_message_lmax:
                        H=sum(B[l-1,o,g]*gate[:,packed_index(gb,G,obj.lmax,l,g,m)] for g in range(G))
                        angular=angular+H.index_select(0,pair.neighbors)
                    value=positions.new_zeros(len(types)).index_add(0,pair.centers,vv*R*angular)
                    contributions.append(value); targets.append(packed_index(ob,O,obj.lmax,l,o,m))
        moments=positions.new_zeros((len(types),obj.alpha_moments_count)).index_add(
            1,torch.tensor(targets,device=obj.device_arg),torch.stack(contributions,1))
        moments=_apply_product_levels(moments,obj.alpha_moments_count,obj.outer_product_levels)
        scalars=moments.index_select(1,obj.alpha_moment_mapping)*obj.outer_scalar_scale
        head=obj.readout_output_coeffs()
        scalar_sum=(scalars*head[types]).sum(1) if obj.readout_mode=="species" else scalars@head
        h=obj.species_coeffs[types]
        sites=obj.shift_coeffs[types]+h+(scalar_sum if obj.readout_mode=="species" else h*scalar_sum)
        return sites.sum()

    def oracle_efs(obj,cfg):
        positions=torch.tensor(np.asarray([a.position for a in cfg.atoms]),device=obj.device_arg,dtype=obj.dtype,requires_grad=True)
        cell=torch.tensor(np.asarray(cfg.supercell),device=obj.device_arg,dtype=obj.dtype)
        types=torch.tensor([a.type for a in cfg.atoms],device=obj.device_arg,dtype=torch.long)
        E=oracle_energy(obj,positions,cell,types)
        F=-torch.autograd.grad(E,positions,create_graph=True)[0]
        strained=cell.detach().clone().requires_grad_(True)
        fraction=positions.detach()@torch.linalg.inv(cell)
        Es=oracle_energy(obj,fraction@strained,strained,types)
        S=-torch.autograd.grad(Es,strained,create_graph=True)[0].T@strained
        return torch.cat((E.reshape(-1),F.reshape(-1),S.reshape(-1)))

    cfg=load_cfg(native.cfg)[0]
    cfg2=replace(cfg,atoms=[replace(a,position=np.asarray(a.position)+[.037*(i+1),-.019*i,.011]) for i,a in enumerate(cfg.atoms)])
    names=("tensor_message_coeffs","moment_coeffs","species_coeffs","shift_coeffs",
           "adaptive_radial_amplitudes","adaptive_radial_weights","two_layer_gate_k_rank_weights",
           "two_layer_gate_radial_coeffs","two_layer_gate_type_coeffs","radial_coeffs","scal_coeffs")
    mapping={"tensor_message_coeffs":"tensor_message","adaptive_radial_amplitudes":"adaptive_radial_amplitude",
             "adaptive_radial_weights":"adaptive_radial_weight","two_layer_gate_k_rank_weights":"gate_k_rank",
             "two_layer_gate_radial_coeffs":"gate_radial","two_layer_gate_type_coeffs":"gate_type",
             "radial_coeffs":"radial","scal_coeffs":"scaling"}
    for label,case in native.report["cases"].items():
        path=Path(case["nonzero"]); obj=model(path); reference=model(path,"reference")
        assert obj.tensor_message_lmax==case["message"] and obj.tensor_message_coeffs.shape==(case["message"],3,2)
        assert obj.readout_mode==case["readout"]
        current=efs(obj,cfg)
        close(label+"-native-EFS",current,native_efs(label+"-native",path,native.cfg),atol=8e-6,rtol=5e-7)
        close(label+"-reference-EFS",current,efs(reference,cfg))
        close(label+"-literal-contract-EFS",current,oracle_efs(obj,cfg))
        if label=="same_same_species_K":
            columns=[i for i,info in enumerate(obj.sus2.spec.sh_scalar_info) if info.body_order==6]
            assert columns,"small fixture must retain sixth-order outer invariants"
            saved=obj.moment_coeffs.detach().clone()
            with torch.no_grad(): obj.moment_coeffs[...,columns]=0.
            ablated=efs(obj,cfg).detach()
            with torch.no_grad(): obj.moment_coeffs.copy_(saved)
            effect=float((current.detach()-ablated).abs().max())
            assert effect>1e-12,("body6 test is vacuous",effect)
            record("body6-active",{"columns":columns,"EFS_effect":effect})
        grads=torch.autograd.grad(objective(obj,cfg),[getattr(obj,n) for n in names])
        rg=torch.autograd.grad(objective(reference,cfg),[getattr(reference,n) for n in names])
        for name,a,b in zip(names,grads,rg):
            assert torch.isfinite(a).all() and float(a.abs().max())>1e-12,(label,name,"vacuous")
            close(label+"-reference-mixed-"+name,a,b,atol=5e-8,rtol=2e-7)
        # Compare gradients of a fully mixed E/F/virial objective with the
        # independently assembled literal message oracle, not just its values.
        og=torch.autograd.grad(oracle_efs(obj,cfg).square().sum(),[getattr(obj,n) for n in names])
        cg=torch.autograd.grad(efs(obj,cfg).square().sum(),[getattr(obj,n) for n in names])
        for name,a,b in zip(names,cg,og): close(label+"-literal-mixed-"+name,a,b,atol=5e-8,rtol=2e-7)
        for name,native_name in mapping.items():
            g=grads[names.index(name)].reshape(-1)
            d=torch.arange(g.numel(),device=obj.device_arg)%5-2
            d=torch.where(d==0,1,d).to(g)
            close(label+"-native-direction-"+name,(g*d).sum(),case["directions"][native_name]["analytic_directional"],atol=5e-8,rtol=5e-6)
        finite_difference(label+"-B-mixed-FD",obj,cfg,"tensor_message_coeffs",grads[0])
        if label=="same_same_species_K":
            for name in ("two_layer_gate_k_rank_weights","adaptive_radial_weights","adaptive_radial_amplitudes","radial_coeffs","scal_coeffs"):
                finite_difference(label+"-mixed-FD-"+name,obj,cfg,name,grads[names.index(name)])
            finite_difference(label+"-B-force-only-FD",obj,cfg,"tensor_message_coeffs",weights=(0.,1.,0.))
            finite_difference(label+"-B-virial-only-FD",obj,cfg,"tensor_message_coeffs",weights=(0.,0.,1.))
            for observable,weights in (("force",(0.,1.,0.)),("virial",(0.,0.,1.))):
                for name,native_name in (("tensor_message_coeffs","tensor_message"),
                                          ("two_layer_gate_k_rank_weights","gate_k_rank")):
                    g=torch.autograd.grad(objective(obj,cfg,weights),getattr(obj,name))[0].reshape(-1)
                    d=torch.arange(g.numel(),device=obj.device_arg)%5-2
                    d=torch.where(d==0,1,d).to(g)
                    target=case["isolated_directions"][observable][native_name]["analytic_directional"]
                    close(label+"-native-"+observable+"-direction-"+name,(g*d).sum(),target,
                          atol=5e-8,rtol=5e-6)
        old=model(case["legacy"]); zero=model(case["zero"])
        close(label+"-zero-B-EFS",efs(old,cfg),efs(zero,cfg),atol=2e-10,rtol=2e-10)
        old_grads=torch.autograd.grad(objective(old,cfg),[getattr(old,n) for n in names[1:]])
        zero_grads=torch.autograd.grad(objective(zero,cfg),[getattr(zero,n) for n in names[1:]])
        for name,a,b in zip(names[1:],old_grads,zero_grads): close(label+"-zero-B-old-gradient-"+name,a,b,atol=5e-9,rtol=2e-9)
        zero_B_gradient=torch.autograd.grad(objective(zero,cfg),zero.tensor_message_coeffs)[0]
        assert torch.isfinite(zero_B_gradient).all() and float(zero_B_gradient.abs().max())>1e-10
        packed=_configuration_batch_forward(obj,[_prepare_training_graph(obj,c) for c in (cfg,cfg2)],create_graph=True)
        pflat=torch.cat((packed.graph_energies.reshape(-1),packed.forces.reshape(-1),packed.stress.reshape(-1)))
        singles=[efs(obj,c) for c in (cfg,cfg2)]
        sflat=torch.cat((torch.stack([s[0] for s in singles]),torch.cat([s[1:-9] for s in singles]),torch.cat([s[-9:] for s in singles])))
        close(label+"-packed-EFS",pflat,sflat)
        pg=torch.autograd.grad(pflat.square().sum(),[getattr(obj,n) for n in names])
        sg=torch.autograd.grad(sflat.square().sum(),[getattr(obj,n) for n in names])
        for name,a,b in zip(names,pg,sg): close(label+"-packed-mixed-"+name,a,b,atol=5e-8,rtol=2e-7)

    first=native.report["cases"]["same_same_species_K"]
    obj=model(first["nonzero"])
    # Double-precision covariance, excluding rounded native output.
    data=helper.read_fixture(native.cfg)
    axis=np.asarray([.31,-.47,.73]); axis/=np.linalg.norm(axis); angle=.713
    K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    Q=np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*K
    before=efs(obj,cfg).detach(); F=before[1:10].reshape(3,3); S=before[-9:].reshape(3,3)
    symmetry_cfgs=[]
    for label,q in (("rotation",Q),("reflection",Q@np.diag([-1.,1.,1.])),("inversion",-np.eye(3))):
        transformed=replace(cfg,supercell=data["cell"]@q.T,atoms=[replace(a,position=np.asarray(a.position)@q.T) for a in cfg.atoms])
        symmetry_cfgs.append((label,q,transformed))
        value=efs(obj,transformed); qt=F.new_tensor(q)
        close(label+"-energy",value[:1],before[:1]); close(label+"-forces",value[1:10].reshape(3,3),F@qt.T)
        close(label+"-virial",value[-9:].reshape(3,3),qt@S@qt.T)
    translated=replace(cfg,atoms=[replace(a,position=np.asarray(a.position)+[1.137,-.219,2.071]) for a in cfg.atoms])
    close("translation-EFS",efs(obj,translated),before)
    order=[2,0,1]; permuted=replace(cfg,atoms=[replace(cfg.atoms[i],id=j+1) for j,i in enumerate(order)])
    value=efs(obj,permuted)
    close("permutation-energy",value[:1],before[:1]); close("permutation-forces",value[1:10].reshape(3,3),F[order]); close("permutation-virial",value[-9:],before[-9:])

    # Preserve nonzero physical B through all existing training re-expressions.
    stats=rescale_initial_parameters(obj,[cfg,cfg2],preserve_function=True,min_scale=1e-12,max_scale=1e12)
    record("fixed-RMS",stats)
    obj.set_readout_output_scale(2.37,preserve_function=True)
    parametrize_adaptive_amplitudes(obj)
    _maybe_parametrize_radial_weight_norm(obj,TrainingOptions(radial_weight_norm=True))
    close("normalization-output-weightnorm-preserve",efs(obj,cfg),before)
    assert obj.tensor_message_irrep_scale.shape==(2,2),"one scale per (L,g), no M-specific scaling"
    assert torch.isfinite(obj.tensor_message_irrep_scale).all() and (obj.tensor_message_irrep_scale>0).all()
    assert not torch.equal(obj.tensor_message_irrep_scale,torch.ones_like(obj.tensor_message_irrep_scale)),"normalization test must use nontrivial irrep scales"
    for label,q,transformed in symmetry_cfgs:
        value=efs(obj,transformed); qt=F.new_tensor(q)
        close("normalized-"+label+"-energy",value[:1],before[:1])
        close("normalized-"+label+"-forces",value[1:10].reshape(3,3),F@qt.T)
        close("normalized-"+label+"-virial",value[-9:].reshape(3,3),qt@S@qt.T)
    folded=out/"folded.sus2"; save_mtp_model(obj.to_mtp_model(),folded)
    close("folded-PT",efs(model(folded),cfg),before)
    close("folded-native",native_efs("folded-native",folded,native.cfg),before,atol=8e-6,rtol=5e-7)
    # Disposable zero-B optimizer proves learnability from the prescribed init.
    train=model(first["zero"]); opts=TrainingOptions()
    _apply_species_readout_gauge_fix(train,opts)
    h=train.species_coeffs.detach().clone(); start_B=train.tensor_message_coeffs.detach().clone()
    optimizer=torch.optim.Adam([p for p in train.parameters() if p.requires_grad],lr=1e-4); losses=[]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); loss=objective(train,cfg)
        assert torch.isfinite(loss); loss.backward()
        assert train.tensor_message_coeffs.grad is not None and torch.isfinite(train.tensor_message_coeffs.grad).all()
        assert float(train.tensor_message_coeffs.grad.abs().max())>1e-12
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in train.parameters())
        torch.nn.utils.clip_grad_norm_(train.parameters(),10.); optimizer.step(); losses.append(float(loss.detach()))
    assert not torch.equal(start_B,train.tensor_message_coeffs)
    close("Adam-frozen-h",train.species_coeffs,h,atol=0,rtol=0)
    trained=out/"adam3.sus2"; save_mtp_model(train.to_mtp_model(),trained)
    close("Adam-export",native_efs("Adam-native",trained,native.cfg),efs(train,cfg),atol=8e-6,rtol=5e-7)
    record("Adam3",{"losses":losses,"B_updated":True,"B_max_abs":float(train.tensor_message_coeffs.abs().max().detach())})

    if args.stage=="full":
        bad_paths=sorted(native_artifacts.glob("bad-*.sus2"))
        assert bad_paths
        for path in bad_paths:
            try: load_mtp_model(path)
            except (MTPParseError,ValueError): pass
            else: raise AssertionError("PT accepted invalid model: "+str(path))
        record("PT-schema-rejection",[p.name for p in bad_paths])
        source=load_cfg(args.cfg.resolve()); assert len(source)>=2
        selected=[source[0],source[-1]]; selected_path=out/"selected.cfg"
        # Lossless writer preserves the real inputs; do not compare rounded
        # native positions against unrounded PT coordinates.
        chunks=[]
        for i,c in enumerate(selected):
            path=out/f"selected{i}.cfg"
            helper.write_geometry(path,dict(cell=np.asarray(c.supercell),positions=np.asarray([a.position for a in c.atoms]),
                types=[a.type for a in c.atoms],forces=np.asarray([a.force for a in c.atoms]),energy=c.energy,stress=np.asarray(c.stress)))
            chunks.append(path.read_text())
        selected_path.write_text("\n".join(chunks)); selected=load_cfg(selected_path)
        tops={}
        for msg in (0,2):
            top=out/f"production-m{msg}.topology.sus2"
            native.run(f"production-init{msg}","init-sh",top,*helper.topology_args(production=True,message=msg)); tops[msg]=top
        options=TopologyInitializationOptions(init_params="same",seed=20260910,head_scale=.1,scaling_slope_start=3.5,scaling_slope_stop=1.5)
        init={msg:materialize_topology_parameters(load_mtp_model(top),options) for msg,top in tops.items()}
        production={msg:SUS2SHPyTorch(mtp,device=args.device,dtype=torch.float64) for msg,mtp in init.items()}
        off,on=production[0],production[2]
        assert sum(p.numel() for p in off.parameters())==76116
        assert sum(p.numel() for p in on.parameters())==76244
        assert on.tensor_message_coeffs.shape==(2,8,8) and on.scalar_count==232
        assert on.moment_coeffs.shape==(2,232)
        close("production-fresh-B",on.tensor_message_coeffs,torch.zeros_like(on.tensor_message_coeffs),atol=0,rtol=0)
        for name in names[1:]: close("production-no-RNG-change-"+name,getattr(off,name),getattr(on,name),atol=0,rtol=0)
        for msg,objp in production.items():
            save_mtp_model(objp.to_mtp_model(),out/("production-disabled.sus2" if msg==0 else "production-zero.sus2"))
        with torch.no_grad(): on.tensor_message_coeffs.copy_(.017*torch.sin(torch.arange(128,device=on.device_arg,dtype=on.dtype).reshape(2,8,8)*.31+.2))
        path=out/"production-nonzero.sus2"; save_mtp_model(on.to_mtp_model(),path)
        all_efs=torch.cat([efs(on,c) for c in selected]); zero_efs=torch.cat([efs(off,c) for c in selected])
        assert float((all_efs-zero_efs).abs().max())>1e-10
        close("production-native-EFS",all_efs,native_efs("production-native",path,selected_path),atol=8e-6,rtol=5e-7)
        close("production-roundtrip",all_efs,torch.cat([efs(model(path),c) for c in selected]))
        if args.device=="cuda": close("production-CPU-CUDA",all_efs,torch.cat([efs(model(path,device="cpu"),c) for c in selected]),atol=5e-8,rtol=2e-7)
        record("production",{"parameters":76244,"B_count":128,"gate_Q":232,"outer_Q":232,"head":464,
            "cfg_indices":[0,len(source)-1],"atoms":[c.size for c in selected],"timing_models_same_nonB":True})
    report["status"]="PASS"; record("complete",{"stage":args.stage,"cases":len(native.report["cases"]),"full":args.stage=="full"})
    (out/("FULL_PASS" if args.stage=="full" else "MINIMAL_PASS")).write_text("PASS\n")


if __name__=="__main__": main()
