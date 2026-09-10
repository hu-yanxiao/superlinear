#!/usr/bin/env python3
"""Species-readout native/PT integration; only run in an LSF allocation."""
from __future__ import annotations
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cpp", "old-cpp", "native-root", "pytorch-root", "cfg", "outdir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if sys.platform == "darwin" or not os.environ.get("LSB_JOBID"):
        raise SystemExit("Only run inside a server LSF compute allocation")
    sys.path.insert(0, str(args.pytorch_root.resolve()))
    import numpy as np
    import torch
    from sus2shpt.io.cfg import load_cfg
    from sus2shpt.io.mtp import MTPParseError, load_mtp_model, save_mtp_model, convert_shared_readout_to_species
    from sus2shpt.model.sus2_sh import SUS2SHPyTorch
    from sus2shpt.topology_init import TopologyInitializationOptions, materialize_topology_parameters
    from sus2shpt.training import (TrainingOptions, rescale_initial_parameters,
                                   _maybe_parametrize_radial_weight_norm,
                                   _configuration_batch_forward, _prepare_training_graph,
                                   _scalar_weight_l2_penalty, _apply_species_readout_gauge_fix)
    from sus2shpt.adaptive_training import parametrize_adaptive_amplitudes
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise AssertionError("CUDA unavailable: refusing to report skipped checks as PASS")
    out = args.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    helper_path = args.native_root.resolve()/"dev_test/adaptive_species_readout_check.py"
    loader = importlib.util.spec_from_file_location("species_native_checks", helper_path)
    helper = importlib.util.module_from_spec(loader); loader.loader.exec_module(helper)
    native = helper.NativeCheck(args.cpp, out/"native", args.old_cpp)
    native.check()
    report = {"status": "RUNNING", "device": args.device, "job_id": os.environ["LSB_JOBID"], "checks": {}}
    def record(name, result):
        report["checks"][name] = result
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(name, json.dumps(result), flush=True)
    def close(name, left, right, serialized=False, atol=None, rtol=None):
        def array(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        left, right = array(left), array(right)
        atol = (8e-6 if serialized else 2e-9) if atol is None else atol
        rtol = (5e-7 if serialized else 2e-8) if rtol is None else rtol
        np.testing.assert_allclose(left, right, atol=atol, rtol=rtol, err_msg=name)
        record(name, {"max_abs": float(abs(left-right).max()), "atol": atol, "rtol": rtol,
                      "rounded_native_cfg": serialized})
    def model(path, kernel="vectorized", device=None):
        obj = SUS2SHPyTorch(load_mtp_model(path), device=device or args.device, dtype=torch.float64)
        obj.adaptive_kernel = kernel
        return obj
    def efs(obj, cfg, graph=False):
        result = obj.forward_configuration(cfg, compute_forces=True, compute_stress=True, create_graph=graph)
        return torch.cat((result.energy.reshape(-1), result.forces.reshape(-1), result.stress.reshape(-1)))
    def loss(obj, cfg):
        value = obj.forward_configuration(cfg, create_graph=True)
        force = value.forces.new_tensor(np.asarray([a.force for a in cfg.atoms]))
        virial = value.stress.new_tensor(cfg.stress)
        return (value.energy-float(cfg.energy)).square()/cfg.size + .1*(value.forces-force).square().sum() + .01*(value.stress-virial).square().sum()/cfg.size
    def native_values(label, path, cfg_path):
        target = out/(label+".cfg")
        native.run(label, "calc-efs", path, cfg_path, target)
        return np.concatenate([np.r_[c.energy, np.asarray([a.force for a in c.atoms]).ravel(), np.asarray(c.stress).ravel()] for c in load_cfg(target)])
    def lossless(path, cfgs):
        rows=[]
        for cfg in cfgs:
            rows.extend(["BEGIN_CFG", " Size", f" {cfg.size}", " Supercell"])
            rows.extend(" " + " ".join(f"{v:.17e}" for v in row) for row in cfg.supercell)
            rows.append(" AtomData: id type cartes_x cartes_y cartes_z fx fy fz")
            for atom in cfg.atoms:
                rows.append(f" {atom.id} {atom.type} " + " ".join(f"{v:.17e}" for v in (*atom.position, *atom.force)))
            rows.extend([" Energy", f" {cfg.energy:.17e}", " PlusStress: xx yy zz yz xz xy"])
            stress=np.asarray(cfg.stress)
            rows.append(" " + " ".join(f"{stress[i,j]:.17e}" for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))))
            rows.extend(["END_CFG", ""])
        path.write_text("\n".join(rows))
        reloaded=load_cfg(path)
        for a,b in zip(cfgs,reloaded):
            np.testing.assert_array_equal(a.supercell,b.supercell)
            np.testing.assert_array_equal([x.position for x in a.atoms],[x.position for x in b.atoms])
            np.testing.assert_array_equal([x.force for x in a.atoms],[x.force for x in b.atoms])
            np.testing.assert_array_equal(a.stress,b.stress)
            assert a.energy==b.energy
        return reloaded

    cfg = load_cfg(native.cfg)[0]
    fresh_native=native.out/"fresh-zeroiter.mtp"
    close("native-fresh-fixed-profile-PT",efs(model(fresh_native),cfg),native_values("fresh-fixed-profile",fresh_native,native.cfg),serialized=True)
    for path in sorted(native.out.glob("bad-*.mtp")):
        if path.name.endswith(".out.mtp"): continue
        try:
            load_mtp_model(path)
        except MTPParseError:
            pass
        else:
            raise AssertionError(f"PyTorch accepted invalid schema: {path.name}")
    record("PT-invalid-schema",{"rejected": [p.name for p in sorted(native.out.glob("bad-*.mtp")) if not p.name.endswith(".out.mtp")]})
    shared_path, free_path = native.out/"shared.mtp", native.out/"species.free.mtp"
    shared = load_mtp_model(shared_path)
    converted = convert_shared_readout_to_species(shared)
    assert converted.spec.readout_mode == "species"
    conversion_path = out/"pt-converted.mtp"; save_mtp_model(converted, conversion_path)
    close("PT-shared-conversion", efs(model(shared_path), cfg), efs(model(conversion_path), cfg))
    # The exact structural invariant belongs to the in-memory conversion.
    # Comparing against a saved/reparsed model would also test text rounding
    # of unchanged nonlinear parameters, which is a separate EFS check above.
    memory_shared=SUS2SHPyTorch(shared,device=args.device,dtype=torch.float64)
    memory_converted=SUS2SHPyTorch(converted,device=args.device,dtype=torch.float64)
    close("PT-raw-scalars-unchanged",memory_shared.forward_configuration(cfg,compute_forces=False,compute_stress=False).scalar_values,
          memory_converted.forward_configuration(cfg,compute_forces=False,compute_stress=False).scalar_values,atol=0.,rtol=0.)
    close("PT-native-converted", efs(model(conversion_path), cfg), native_values("pt-converted-native", conversion_path, native.cfg), serialized=True)
    obj, ref = model(free_path), model(free_path, "reference")
    close("PT-native-free", efs(obj,cfg), native_values("pt-free-native",free_path,native.cfg), serialized=True)
    close("reference-vectorized-EFS", efs(obj,cfg), efs(ref,cfg))
    # The returned scalar matrix is the actual readout input. No helper that
    # selects head rows participates in this independent center-type oracle.
    value=obj.forward_configuration(cfg,create_graph=True)
    z=torch.tensor([a.type for a in cfg.atoms],device=args.device)
    expected=torch.stack([value.scalar_values[z==s].sum(0) for s in range(2)])
    actual=torch.autograd.grad(value.energy,obj.moment_coeffs,retain_graph=True)[0]
    close("center-species-head-energy-gradient",actual,expected)
    expected_sites=obj.shift_coeffs[z]+obj.species_coeffs[z]+(value.scalar_values*obj.moment_coeffs[z]).sum(1)
    close("manual-site-readout",value.site_energies,expected_sites)
    hb=torch.autograd.grad(value.energy,obj.species_coeffs,retain_graph=True)[0]
    close("h-energy-gradient-counts",hb,[2.,1.],atol=0.,rtol=0.)
    for field in ("forces","stress"):
        g=torch.autograd.grad(getattr(value,field).square().sum(),obj.species_coeffs,retain_graph=True,allow_unused=True)[0]
        if g is not None: np.testing.assert_array_equal(g.detach().cpu().numpy(),np.zeros(2))
    record("h-force-virial-gradient",{"exact_zero":True})
    body6=[i for i,info in enumerate(obj.mtp.spec.sh_scalar_info) if info.body_order==6]
    assert body6,"fixture lost sixth-order readout paths"
    saved_head=obj.moment_coeffs.detach().clone()
    with torch.no_grad(): obj.moment_coeffs[:,body6]=0.
    ablated=efs(obj,cfg).detach()
    with torch.no_grad(): obj.moment_coeffs.copy_(saved_head)
    body6_effect=float((ablated-efs(obj,cfg).detach()).abs().max())
    assert body6_effect>1e-12,"body-six test is vacuous"
    record("body6-active",{"columns":body6,"max_EFS_effect":body6_effect})
    names=("moment_coeffs","species_coeffs","shift_coeffs","adaptive_radial_amplitudes",
           "adaptive_radial_weights","two_layer_gate_k_rank_weights","two_layer_gate_radial_coeffs",
           "two_layer_gate_type_coeffs","radial_coeffs","scal_coeffs")
    grads=torch.autograd.grad(loss(obj,cfg),[getattr(obj,n) for n in names])
    refgrads=torch.autograd.grad(loss(ref,cfg),[getattr(ref,n) for n in names])
    for name,a,b in zip(names,grads,refgrads):
        assert torch.isfinite(a).all() and float(a.abs().max())>1e-12,name
        close("mixed-gradient-"+name,a,b,atol=5e-8,rtol=2e-7)
    # Native-direction patterns are defined by local coefficient index, not
    # by physical values. Use exact same pattern for a/U/P and both head rows.
    layout=helper.dev_fields((native.out/"layout.log").read_text())
    mapping={"adaptive_radial_amplitudes":"adaptive_radial_amplitude",
             "adaptive_radial_weights":"adaptive_radial_weight","two_layer_gate_k_rank_weights":"gate_k_rank"}
    for name,native_name in mapping.items():
        g=grads[names.index(name)].reshape(-1)
        direction=torch.arange(g.numel(),device=args.device)%5-2
        direction=torch.where(direction==0,1,direction).to(g)
        expected_dir=float((g*direction).sum().detach())
        measured=native.report["checks"][native_name+"_mixed_fd"]
        close("native-PT-direction-"+name,float(measured["analytic_directional"]),expected_dir,atol=5e-8,rtol=5e-6)
    for s in range(2):
        g=grads[0][s]; direction=torch.arange(g.numel(),device=args.device)%5-2
        direction=torch.where(direction==0,1,direction).to(g)
        close(f"native-PT-head-row{s}",float(native.report["checks"][f"head_species_{s}_mixed_fd"]["analytic_directional"]),
              (g*direction).sum(),atol=5e-8,rtol=5e-6)
    # Independent head-coordinate FD of mixed EFS loss. Two successful steps
    # rule out a fortunate cancellation; never widen a tolerance to force PASS.
    for s in range(2):
        parameter=obj.moment_coeffs; original=parameter.detach().clone()
        direction=torch.zeros_like(parameter)
        direction[s]=torch.cos(torch.arange(parameter.shape[1],device=args.device,dtype=obj.dtype)*.37+.21)
        analytic=float((grads[0]*direction).sum().detach()); records=[]; streak=0
        try:
            for h in (2e-5,5e-6,1.25e-6,3.125e-7,7.8125e-8):
                with torch.no_grad(): parameter.copy_(original+h*direction)
                plus=float(loss(obj,cfg).detach())
                with torch.no_grad(): parameter.copy_(original-h*direction)
                minus=float(loss(obj,cfg).detach())
                fd=(plus-minus)/(2*h)
                passed=bool(np.isclose(fd,analytic,atol=5e-7,rtol=5e-5))
                records.append({"h":h,"analytic":analytic,"fd":fd,"pass":passed})
                record(f"PT-head-row{s}-FD",records)
                streak=streak+1 if passed else 0
                if streak>=2: break
            assert streak>=2 and abs(analytic)>1e-10
        finally:
            with torch.no_grad(): parameter.copy_(original)
    # Two geometrically unequal fixtures exercise real packed training graph
    # indexing and mixed gradients, including all body-six paths.
    cfg2=replace(cfg,atoms=[replace(a,position=np.asarray(a.position)+[.037*(i+1),-.019*i,.011]) for i,a in enumerate(cfg.atoms)])
    graphs=[_prepare_training_graph(obj,c) for c in (cfg,cfg2)]
    packed=_configuration_batch_forward(obj,graphs,create_graph=True)
    singles=[efs(obj,c,True) for c in (cfg,cfg2)]
    pack_flat=torch.cat((packed.graph_energies.reshape(-1),packed.forces.reshape(-1),packed.stress.reshape(-1)))
    single_flat=torch.cat((torch.stack([x[0] for x in singles]),torch.cat([x[1:-9] for x in singles]),torch.cat([x[-9:] for x in singles])))
    close("packed-EFS",pack_flat,single_flat)
    pg=torch.autograd.grad(pack_flat.square().sum(),[getattr(obj,n) for n in names])
    sg=torch.autograd.grad(single_flat.square().sum(),[getattr(obj,n) for n in names])
    for name,a,b in zip(names,pg,sg): close("packed-mixed-"+name,a,b,atol=5e-8,rtol=2e-7)
    # Head L2 mean is over S*Q, not just Q. Conversion preserves EFS, not L2.
    opts=TrainingOptions(scalar_head_l2=.013)
    penalty=_scalar_weight_l2_penalty(obj,opts)
    close("species-head-L2-denominator",penalty,.013*obj.moment_coeffs.square().mean(),atol=1e-13,rtol=1e-12)
    native_reg=helper.dev_fields(native.run("head-L2", "check-loss-gradient-dev", free_path,native.cfg,
                             *[a for a in helper.LOSS_ARGS if not a.startswith("--scalar-head-l2=")],
                             "--scalar-head-l2=0.013","--coeff-start=0","--coeff-end=1"))
    close("native-PT-head-L2-loss",float(native_reg["base_loss"])-float(layout["base_loss"]),penalty,atol=5e-10,rtol=5e-6)
    before=efs(obj,cfg).detach()
    stats=rescale_initial_parameters(obj,[cfg,cfg2],preserve_function=True,min_scale=1e-12,max_scale=1e12)
    record("fixed-RMS",{k:float(v) for k,v in stats.items()})
    obj.set_readout_output_scale(2.37,preserve_function=True)
    parametrize_adaptive_amplitudes(obj)
    _maybe_parametrize_radial_weight_norm(obj,TrainingOptions(radial_weight_norm=True))
    close("fixed-RMS-output-weightnorm-preserve",efs(obj,cfg),before)
    folded=out/"folded.mtp"; save_mtp_model(obj.to_mtp_model(),folded)
    close("folded-PT",efs(model(folded),cfg),before)
    close("folded-native",native_values("folded-native",folded,native.cfg),before,serialized=True)
    native.run("folded-linear-components","check-linear-readout-dev",folded,native.cfg,
               "--max-configs=1","--max-atoms=3","--check-stress","--abs-tolerance=2e-9","--rel-tolerance=2e-9")
    # Physical dh checks above use an unconstrained model. Apply the same
    # training-only gauge helper as train_model only now, before optimization.
    initial_h=obj.species_coeffs.detach().clone()
    gauge_stats=_apply_species_readout_gauge_fix(obj,opts)
    assert gauge_stats is not None and not obj.species_coeffs.requires_grad
    assert obj.shift_coeffs.requires_grad and obj.moment_coeffs.requires_grad
    np.testing.assert_array_equal(initial_h.cpu().numpy(),obj.species_coeffs.detach().cpu().numpy())
    record("PT-training-gauge",gauge_stats)
    optimizer=torch.optim.Adam([p for p in obj.parameters() if p.requires_grad],lr=1e-4)
    initial_head=obj.moment_coeffs.detach().clone(); losses=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True); objective=loss(obj,cfg)+_scalar_weight_l2_penalty(obj,opts)
        assert torch.isfinite(objective)
        objective.backward()
        assert obj.species_coeffs.grad is None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters())
        torch.nn.utils.clip_grad_norm_(obj.parameters(),10.); optimizer.step()
        losses.append(float(objective.detach()))
    assert not torch.equal(initial_head,obj.moment_coeffs)
    np.testing.assert_array_equal(initial_h.cpu().numpy(),obj.species_coeffs.detach().cpu().numpy())
    adam=out/"adam3.mtp"; save_mtp_model(obj.to_mtp_model(),adam)
    close("Adam-export-native",native_values("adam-native",adam,native.cfg),efs(obj,cfg),serialized=True)
    record("Adam3",{"steps":3,"losses":losses,"h_exactly_unchanged":True,"head_updated":True,
                    "formal_initialization_reused":False})

    # Species heads must be orthogonal to the choice of gate/outer pruning.
    # Run the remaining contracts on real native-generated, body-six DAGs;
    # no synthetic metadata switch is used to impersonate another topology.
    combinations=native.report["checks"]["pruning-combinations"]
    assert set(combinations)=={"gate_same_outer_same","gate_same_outer_cross",
                               "gate_cross_outer_same","gate_cross_outer_cross"}
    def lane_provenance(mtp,stage):
        spec=mtp.spec
        rank=int(spec.two_layer_gate_k_rank if stage=="gate" else spec.outer_k_rank)
        begin=int(spec.two_layer_gate_k_rank_basic_begin if stage=="gate" else spec.outer_k_rank1_basic_begin)
        pbegin=int(spec.two_layer_gate_k_rank_product_begin if stage=="gate" else spec.outer_k_rank1_product_begin)
        pend=int(spec.two_layer_gate_k_rank_product_end) if stage=="gate" else len(mtp.sh_products)
        scalar_indices=spec.two_layer_gate_scalar_moment_indices if stage=="gate" else mtp.alpha_moment_mapping
        lmax=int(spec.sh_l_max)
        deps={}
        for lane in range(rank):
            for l in range(lmax,-1,-1):
                start=begin+rank*((lmax+1)**2-(l+1)**2)+(rank-1-lane)*(2*l+1)
                for m in range(2*l+1): deps[start+m]={lane}
        for product in mtp.sh_products[pbegin:pend]:
            assert product.left in deps and product.right in deps,(stage,product)
            deps.setdefault(product.target,set()).update(deps[product.left]|deps[product.right])
        return [deps[int(q)] for q in scalar_indices]
    for label,case in combinations.items():
        free_case=Path(case["species"]); shared_case=Path(case["shared"])
        opt=model(free_case); reference=model(free_case,"reference")
        mtp=opt.mtp
        assert mtp.spec.readout_mode=="species"
        assert mtp.spec.two_layer_gate_factor_pruning==case["gate_pruning"]
        assert mtp.spec.outer_channel_mode==("same-channel" if case["outer_pruning"]=="same-k" else "cross-channel")
        provenance={}
        for stage,pruning in (("gate",case["gate_pruning"]),("outer",case["outer_pruning"])):
            lanes=lane_provenance(mtp,stage)
            cross_count=sum(len(x)>1 for x in lanes)
            assert (cross_count>0)==(pruning=="q-total"),(label,stage,cross_count)
            provenance[stage]={"Q":len(lanes),"cross_lane_scalars":cross_count}
        record(label+"-actual-graph",provenance)
        current_efs=efs(opt,cfg)
        close(label+"-native-EFS",current_efs,native_values(label+"-native",free_case,native.cfg),serialized=True)
        close(label+"-reference-EFS",current_efs,efs(reference,cfg))
        pt_converted=out/(label+".PT-converted.mtp")
        save_mtp_model(convert_shared_readout_to_species(load_mtp_model(shared_case)),pt_converted)
        close(label+"-PT-conversion",efs(model(shared_case),cfg),efs(model(pt_converted),cfg))
        close(label+"-conversion-native-parity",efs(model(pt_converted),cfg),
              native_values(label+"-converted-native",pt_converted,native.cfg),serialized=True)
        species_result=opt.forward_configuration(cfg,create_graph=True)
        rows=torch.stack([species_result.scalar_values[z==s].sum(0) for s in range(2)])
        row_grad=torch.autograd.grad(species_result.energy,opt.moment_coeffs)[0]
        close(label+"-center-row-gradient",row_grad,rows)
        mixed=torch.autograd.grad(loss(opt,cfg),[getattr(opt,n) for n in names])
        mixed_ref=torch.autograd.grad(loss(reference,cfg),[getattr(reference,n) for n in names])
        for name,actual,expected in zip(names,mixed,mixed_ref):
            assert torch.isfinite(actual).all() and float(actual.abs().max())>1e-12,(label,name)
            close(label+"-mixed-"+name,actual,expected,atol=5e-8,rtol=2e-7)
        for name,native_name in mapping.items():
            gradient=mixed[names.index(name)].reshape(-1)
            direction=torch.arange(gradient.numel(),device=args.device)%5-2
            direction=torch.where(direction==0,1,direction).to(gradient)
            measured=case["directions"][native_name]
            close(label+"-native-mixed-"+name,float(measured["analytic_directional"]),
                  (gradient*direction).sum(),atol=5e-8,rtol=5e-6)
        for s in range(2):
            gradient=mixed[0][s]
            direction=torch.arange(gradient.numel(),device=args.device)%5-2
            direction=torch.where(direction==0,1,direction).to(gradient)
            close(label+f"-native-head{s}",float(case["directions"][f"head_species_{s}"]["analytic_directional"]),
                  (gradient*direction).sum(),atol=5e-8,rtol=5e-6)
        packed_case=_configuration_batch_forward(opt,[_prepare_training_graph(opt,c) for c in (cfg,cfg2)],create_graph=True)
        singles_case=[efs(opt,c,True) for c in (cfg,cfg2)]
        packed_values=torch.cat((packed_case.graph_energies.reshape(-1),packed_case.forces.reshape(-1),packed_case.stress.reshape(-1)))
        single_values=torch.cat((torch.stack([v[0] for v in singles_case]),torch.cat([v[1:-9] for v in singles_case]),torch.cat([v[-9:] for v in singles_case])))
        close(label+"-packed-EFS",packed_values,single_values)
        packed_grads=torch.autograd.grad(packed_values.square().sum(),[getattr(opt,n) for n in names])
        single_grads=torch.autograd.grad(single_values.square().sum(),[getattr(opt,n) for n in names])
        for name,actual,expected in zip(names,packed_grads,single_grads):
            close(label+"-packed-mixed-"+name,actual,expected,atol=5e-8,rtol=2e-7)
    record("all-four-pruning-combinations",{"count":4,"gate_modes":["same-k","q-total"],
                                             "outer_modes":["same-k","q-total"],"body_order":6,"gate_body_order":6})

    # Production geometry and parameter volume; avoid full production finite
    # differences (the small fixture above retains all six body orders).
    source_cfg=load_cfg(args.cfg.resolve()); assert len(source_cfg)>=2
    selected_path=out/"selected.cfg"
    selected=lossless(selected_path,[source_cfg[0],source_cfg[-1]])
    top={}
    for mode in ("shared","species"):
        top[mode]=out/f"production-{mode}.topology.mtp"
        native.run("production-init-"+mode,"init-sh",top[mode],*helper.topology_args(True,readout=mode))
    for key in helper.GRAPH_FIELDS:
        assert helper.assignment(top["shared"].read_text(),key)==helper.assignment(top["species"].read_text(),key)
    mtp=materialize_topology_parameters(load_mtp_model(top["species"]),TopologyInitializationOptions(init_params="same",seed=20260910,head_scale=.1,scaling_slope_start=3.5,scaling_slope_stop=1.5))
    production=SUS2SHPyTorch(mtp,device=args.device,dtype=torch.float64)
    assert production.moment_coeffs.shape==(2,232)
    assert production.scalar_count==232 and len(mtp.spec.two_layer_gate_scalar_indices)==232
    assert sum(p.numel() for p in production.parameters())==76116
    stats=rescale_initial_parameters(production,selected,max_configs=2,min_scale=1e-12,max_scale=1e12,preserve_function=False)
    production.set_readout_output_scale(1.71,preserve_function=True)
    prod_path=out/"production-initial.mtp"; save_mtp_model(production.to_mtp_model(),prod_path)
    all_efs=torch.cat([efs(production,c) for c in selected])
    close("production-native-PT-EFS",native_values("production-native",prod_path,selected_path),all_efs,serialized=True)
    close("production-fold",torch.cat([efs(model(prod_path),c) for c in selected]),all_efs)
    if args.device=="cuda":
        cpu=model(prod_path,device="cpu")
        close("production-CUDA-CPU",torch.cat([efs(cpu,c) for c in selected]),all_efs,atol=5e-8,rtol=2e-7)
    record("production",{"gate_Q":232,"outer_Q":232,"head":464,"parameters":76116,
                         "real_cfg_indices":[0,len(source_cfg)-1],"atom_counts":[c.size for c in selected]})
    report["status"]="PASS"; record("complete",True); (out/"PASS").write_text("PASS\n")


if __name__=="__main__": main()
