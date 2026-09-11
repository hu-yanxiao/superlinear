#!/usr/bin/env python3
"""Server-only nonzero tensor-message native mathematical regression.

Run --stage minimal first. Full adds the eight graph/readout combinations,
rotations/reflections, legacy-zero parity, schema rejection, and real BFGS.
The sibling species checker supplies process/CFG parsing helpers only; its
scientific suite is not called. All reports distinguish partial from full PASS.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

from adaptive_species_readout_check import (
    NativeCheck, NUMBER, GRAPH_FIELDS, ZERO_REG, assignment, numbers, put,
    put_numbers, dev_fields, asymmetric_shared_text, write_fixture, parse_prediction,
)

PARAMETER_FIELDS = ("shift_coeffs", "scal_coeffs", "species_coeffs", "moment_coeffs",
                    "adaptive_radial_amplitudes", "adaptive_radial_weights",
                    "two_layer_gate_k_rank_weights", "two_layer_gate_radial_coeffs",
                    "two_layer_gate_type_coeffs")
MIXED_NAMES = ("tensor_message", "gate_k_rank", "adaptive_radial_amplitude",
               "adaptive_radial_weight", "scaling", "radial", "gate_radial", "gate_type")


def topology_args(*, production=False, readout="species", gate="same-k", outer="same-k",
                  scaling="K", message=2, species=2):
    l, k, g, o, n = (3, 8, 8, 8, 10) if production else (2, 3, 2, 3, 3)
    return [f"--species-count={species}", f"--l-max={l}", f"--k-max={k}",
            "--body-order=6", "--body-l-max="+("3,3,3,2,1" if production else "2,2,2,2,1"),
            f"--radial-basis-size={n}", f"--cutoff={6.5 if production else 4.8}",
            "--radial-basis-type=RBChebyshev_sss", f"--scaling-map={scaling}",
            "--superlinear-mode=radial-coefficient-gate", f"--sh-factor-pruning={outer}",
            f"--outer-k-rank={o}", "--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
            f"--two-layer-gate-k-rank={g}", "--two-layer-gate-body-order=6",
            f"--two-layer-gate-factor-pruning={gate}", "--two-layer-gate-shared-radial",
            "--write-sh-scalar-info", f"--readout-mode={readout}", f"--tensor-message-lmax={message}"]


def tensor_text(source, enabled_template, values):
    """Enable via genuine init-sh metadata order, preserving every old value."""
    source = re.sub(r"(?m)^\s*tensor_message_(?:lmax|coeff_count|coeffs|optimizer_scales)\s*=.*\n?", "", source)
    source = put(source, "version", "1.4.0")
    # Copy the new field ordering from the native-generated complete model.
    lines = enabled_template.splitlines()
    fields=["tensor_message_lmax", "tensor_message_coeff_count", "tensor_message_coeffs"]
    if re.search(r"(?m)^\s*tensor_message_optimizer_scales\s*=",enabled_template):
        fields.append("tensor_message_optimizer_scales")
    for key in fields:
        index = next(i for i, line in enumerate(lines) if re.match(rf"^\s*{key}\s*=", line))
        anchor = next(re.match(r"^\s*([A-Za-z_0-9]+)\s*=", lines[i])[1]
                      for i in range(index-1, -1, -1)
                      if re.match(r"^\s*([A-Za-z_0-9]+)\s*=", lines[i]))
        pattern = rf"(?m)^\s*{anchor}\s*=.*\n"
        match = re.search(pattern, source)
        assert match, (key, anchor)
        source = source[:match.end()] + lines[index].strip() + "\n" + source[match.end():]
    return put_numbers(source, "tensor_message_coeffs", values)


def read_fixture(path):
    text = path.read_text()
    cell = np.asarray([[float(x) for x in line.split()] for line in
        re.search(r"(?ms)^\s*Supercell\s*\n(.*?)^\s*AtomData:", text)[1].strip().splitlines()])
    atom_rows = [line.split() for line in
        re.search(r"(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy", text)[1].strip().splitlines()]
    positions = np.asarray([[float(v) for v in row[2:5]] for row in atom_rows])
    forces = np.asarray([[float(v) for v in row[5:8]] for row in atom_rows])
    stress6 = np.asarray([float(v) for v in re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)",text)[1].split()])
    return dict(cell=cell, positions=positions, types=[int(row[1]) for row in atom_rows], forces=forces,
                energy=float(re.search(r"(?m)^\s*Energy\s*\n\s*("+NUMBER+")",text)[1]), stress=stress_matrix(stress6))


def stress_matrix(s):
    return np.array([[s[0],s[5],s[4]],[s[5],s[1],s[3]],[s[4],s[3],s[2]]])


def write_geometry(path, data):
    # CFG has only six symmetric stress entries. A theoretically symmetric
    # Q S Q^T can acquire a tiny antisymmetric part from matrix-product
    # roundoff; reject anything larger instead of silently dropping it.
    stress=np.asarray(data["stress"],dtype=float)
    assert stress.shape==(3,3) and np.isfinite(stress).all(),"CFG stress must be finite 3x3"
    bound=32*np.finfo(float).eps*float(abs(stress).max())
    antisymmetric=float(abs(stress-stress.T).max())
    assert antisymmetric<=bound,("CFG stress is not symmetric within roundoff",antisymmetric,bound)
    canonical_stress=stress.copy()
    for i in range(3):
        for j in range(i+1,3):
            canonical_stress[i,j]=canonical_stress[j,i]=.5*stress[i,j]+.5*stress[j,i]
    canonical={**data,"stress":canonical_stress}
    rows=["BEGIN_CFG", " Size", f" {len(data['types'])}", " Supercell"]
    rows += [" "+" ".join(f"{x:.17e}" for x in row) for row in data["cell"]]
    rows += [" AtomData: id type cartes_x cartes_y cartes_z fx fy fz"]
    rows += [f" {i+1} {z} "+" ".join(f"{x:.17e}" for x in (*p,*f))
             for i,(z,p,f) in enumerate(zip(data["types"],data["positions"],data["forces"]))]
    rows += [" Energy", f" {data['energy']:.17e}", " PlusStress: xx yy zz yz xz xy",
             " "+" ".join(f"{canonical_stress[i,j]:.17e}" for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))), "END_CFG", ""]
    path.write_text("\n".join(rows))
    reread=read_fixture(path)
    # Cell, positions, forces, energy and types remain bit-exact. Stress is
    # also bit-exact against the explicitly canonicalized six-entry tensor.
    for key in canonical: np.testing.assert_array_equal(canonical[key],reread[key],err_msg=key)


class TensorCheck(NativeCheck):
    def calc(self,label,model,cfg=None,binary=None):
        # Several symmetry inputs intentionally use <label>.cfg. Native
        # calc-efs opens its output before consuming its input, so the inherited
        # naming convention would truncate precisely those test geometries.
        source=Path(cfg if cfg is not None else self.cfg).resolve()
        potential=Path(model).resolve()
        target=(self.out/f"{label}.prediction.cfg").resolve()
        assert target not in (source,potential),("prediction output aliases an input",target)
        assert source.is_file() and potential.is_file(),(source,potential)
        self.run(label,"calc-efs",potential,source,target,binary=binary)
        assert target.is_file() and target.stat().st_size>0,("empty prediction",target)
        return parse_prediction(target)

    def layout(self, label, model, binary=None):
        return dev_fields(self.run(label,"check-loss-gradient-dev",model,self.cfg,
            "--max-configs=1","--energy-weight=1","--force-weight=0.1","--stress-weight=0.01",
            *ZERO_REG,"--coeff-start=0","--coeff-end=1","--displacement=5e-6",
            "--abs-tolerance=5e-8","--rel-tolerance=5e-5",binary=binary))

    @staticmethod
    def ranges(layout):
        result={n:(int(layout[n+"_begin"]),int(layout[n+"_end"]))
                for n in MIXED_NAMES if n!="scaling"}
        # shift precedes scal; all base scaling coefficients precede radial.
        result["scaling"]=(2,int(layout["radial_begin"]))
        return result

    def direction(self,label,model,begin,end,*,binary=None,nonzero=True,weights=(1.,.1,.01)):
        assert end>begin,(label,begin,end)
        records=[]; streak=0
        for h in (2e-5,5e-6,1.25e-6,3.125e-7,7.8125e-8):
            # Run with non-terminating diagnostic tolerances, then enforce the
            # original strict bound ourselves over two refined steps. A coarse
            # truncation error is not an excuse to widen the acceptance bound.
            raw=self.run(f"{label}-{h:.2g}","check-loss-gradient-direction-dev",model,self.cfg,
                "--max-configs=1",f"--energy-weight={weights[0]}",
                f"--force-weight={weights[1]}",f"--stress-weight={weights[2]}",
                *ZERO_REG,f"--coeff-start={begin}",f"--coeff-end={end}",f"--displacement={h}",
                "--abs-tolerance=1e300","--rel-tolerance=1e300",binary=binary)
            values=dev_fields(raw)
            a,fd=float(values["analytic_directional"]),float(values["finite_difference"])
            assert math.isfinite(a) and math.isfinite(fd),(label,values)
            passed=bool(np.isclose(a,fd,atol=5e-8,rtol=5e-5))
            records.append({"h":h,"analytic":a,"fd":fd,"pass":passed})
            self.record(label+"-refinement",records)
            streak=streak+1 if passed else 0
            if streak>=2: break
        assert streak>=2,(label,records)
        if nonzero: assert abs(a)>1e-10,(label,"vacuous derivative",a)
        return {"analytic_directional":a,"records":records,"begin":begin,"end":end,"weights":list(weights)}

    def case(self,label,*,gate="same-k",outer="same-k",readout="species",scaling="K",message=2):
        paths={}
        for msg in (0,message):
            top=self.out/f"{label}.m{msg}.topology.mtp"
            full=self.out/f"{label}.m{msg}.materialized.mtp"
            self.run(label+f"-init{msg}","init-sh",top,
                *topology_args(gate=gate,outer=outer,readout=readout,scaling=scaling,message=msg))
            self.run(label+f"-materialize{msg}","roundtrip-model-dev",top,full)
            paths[msg]=full
        old,template=paths[0].read_text(),paths[message].read_text()
        assert assignment(template,"version")=="1.4.0"
        assert int(assignment(template,"tensor_message_lmax"))==message
        assert len(numbers(template,"tensor_message_coeffs"))==message*3*2
        np.testing.assert_array_equal(numbers(template,"tensor_message_coeffs"),np.zeros(message*3*2))
        for key in GRAPH_FIELDS: assert assignment(old,key)==assignment(template,key),(label,key)
        # Start both files from exactly the same pre-existing coefficients.
        # asymmetric_shared_text mutates scalar arrays, not graph metadata.
        base=asymmetric_shared_text(old)
        base=put(base,"readout_mode",readout)
        if readout=="species":
            q=int(assignment(base,"alpha_scalar_moments"))
            h=numbers(base,"moment_coeffs").reshape(2,q)
            h[1]+=.037*np.cos(np.arange(q)*.29+.11)
            base=put_numbers(base,"moment_coeffs",h.ravel())
        legacy=self.out/f"{label}.legacy.mtp"; legacy.write_text(base)
        zero=self.out/f"{label}.zero.mtp"
        zero.write_text(tensor_text(base,template,np.zeros(message*3*2)))
        nonzero=self.out/f"{label}.nonzero.mtp"
        B=.11*np.sin(.37*np.arange(message*3*2)+.21)+.043*np.cos(.19*np.arange(message*3*2))
        nonzero.write_text(tensor_text(base,template,B))
        before=self.calc(label+"-legacy",legacy)
        zero_efs=self.calc(label+"-zero",zero)
        np.testing.assert_allclose(before,zero_efs,atol=2e-10,rtol=2e-10)
        values=self.calc(label+"-nonzero",nonzero)
        effect=float(abs(values-before).max())
        assert effect>1e-8,(label,"nonzero B made no measurable change",effect)
        layout=self.layout(label+"-layout",nonzero)
        ranges=self.ranges(layout)
        assert ranges["tensor_message"][1]-ranges["tensor_message"][0]==message*3*2
        directions={name:self.direction(label+"-mixed-"+name,nonzero,*ranges[name]) for name in MIXED_NAMES}
        isolated={}
        if label=="same_same_species_K":
            begin,end=ranges["tensor_message"]
            self.run(label+"-B-every-component-FD","check-loss-gradient-dev",nonzero,self.cfg,
                "--max-configs=1","--energy-weight=1","--force-weight=0.1","--stress-weight=0.01",
                *ZERO_REG,f"--coeff-start={begin}",f"--coeff-end={end}","--displacement=5e-6",
                "--abs-tolerance=5e-8","--rel-tolerance=5e-5")
            # The mixed objective alone can mask a force/virial-chain omission
            # behind its energy term. Isolate both physical observables and
            # test B plus the upstream projection P separately.
            for observable,weights in (("force",(0.,1.,0.)),("virial",(0.,0.,1.))):
                isolated[observable]={name:self.direction(label+"-"+observable+"-"+name,
                    nonzero,*ranges[name],weights=weights)
                    for name in ("tensor_message","gate_k_rank")}
        self.run(label+"-geometry-FD","check-efs-fd-dev",nonzero,self.cfg,"--max-configs=1","--max-atoms=3",
            "--displacement=2e-5","--abs-tolerance=2e-6","--rel-tolerance=2e-4")
        self.run(label+"-linear-EFS","check-linear-readout-dev",nonzero,self.cfg,"--max-configs=1","--max-atoms=3",
            "--check-stress","--abs-tolerance=2e-9","--rel-tolerance=2e-9")
        if readout=="shared":
            # TrainLinear retains its raw-Phi chart. The physical fixed-center
            # head chart is h[type] * Phi; check BOTH coordinate derivative
            # definitions on the same asymmetric h and nonzero-B model.
            for chart,flags in (("raw",()),("fixed-center",("--fixed-center-readout-columns",))):
                values=dev_fields(self.run(label+"-components-FD-"+chart,
                    "check-linear-components-fd-dev",nonzero,self.cfg,
                    "--max-configs=1","--max-atoms=3",*flags,"--displacement=2e-5",
                    "--abs-tolerance=5e-8","--rel-tolerance=5e-5"))
                assert int(values["checked_components"])>0
                self.record(label+"-components-FD-"+chart,values)
        result={"gate":gate,"outer":outer,"readout":readout,"scaling":scaling,"message":message,
                "legacy":str(legacy),"zero":str(zero),"nonzero":str(nonzero),"layout":layout,
                "directions":directions,"isolated_directions":isolated,
                "B_effect":effect,"B_count":len(B),"topology":str(paths[message])}
        self.report.setdefault("cases",{})[label]=result
        self.record(label+"-summary",{k:result[k] for k in ("gate","outer","readout","scaling","message","B_effect","B_count")})
        return result

    def zero_legacy_gradients(self,label,case):
        assert self.old_cpp,"--old-cpp required for full old-format regression"
        legacy,zero=Path(case["legacy"]),Path(case["zero"])
        np.testing.assert_allclose(self.calc(label+"-old-binary",legacy,binary=self.old_cpp),
                                   self.calc(label+"-zero-new",zero),atol=2e-10,rtol=2e-10)
        oldlayout=self.layout(label+"-old-layout",legacy,binary=self.old_cpp)
        newlayout=self.layout(label+"-zero-layout",zero)
        for name in MIXED_NAMES[1:]:
            if name=="scaling": ro,rn=(2,int(oldlayout["radial_begin"])),(2,int(newlayout["radial_begin"]))
            else: ro=tuple(int(oldlayout[name+x]) for x in ("_begin","_end")); rn=tuple(int(newlayout[name+x]) for x in ("_begin","_end"))
            a=self.direction(label+"-old-"+name,legacy,*ro,binary=self.old_cpp,nonzero=False)
            b=self.direction(label+"-zero-"+name,zero,*rn,nonzero=False)
            np.testing.assert_allclose(a["analytic_directional"],b["analytic_directional"],atol=5e-9,rtol=2e-9,err_msg=name)
        for name,ro,rn in (("shift",(0,2),(0,2)),
                ("linear",(int(oldlayout["linear_begin"]),int(oldlayout["coeff_count"])),
                          (int(newlayout["linear_begin"]),int(newlayout["coeff_count"])))):
            a=self.direction(label+"-old-"+name,legacy,*ro,binary=self.old_cpp,nonzero=False)
            b=self.direction(label+"-zero-"+name,zero,*rn,nonzero=False)
            np.testing.assert_allclose(a["analytic_directional"],b["analytic_directional"],atol=5e-9,rtol=2e-9,err_msg=name)
        self.record(label+"-old-gradient-parity",{"old_cpp":str(self.old_cpp),"directions":[*MIXED_NAMES[1:],"shift","linear"]})
        if case["readout"]=="shared":
            # The historical diagnostic contracts raw Phi with d and thus
            # assumes unit species h. Document this pre-existing chart limit
            # using the old binary, rather than weakening the tensor fixture.
            args=("check-linear-readout-dev",self.cfg,"--max-configs=1","--max-atoms=3",
                  "--check-stress","--abs-tolerance=2e-9","--rel-tolerance=2e-9")
            old=dev_fields(self.run(label+"-old-asymmetric-chart-limit",args[0],legacy,*args[1:],
                                    binary=self.old_cpp,fail=True))
            errors=[float(old[key]) for key in ("energy_abs_err","force_abs_err","stress_abs_err")]
            assert all(math.isfinite(x) for x in errors) and max(errors)>1e-8
            unit=self.out/(label+"-unit-h-chart.mtp")
            unit.write_text(put_numbers(legacy.read_text(),"species_coeffs",np.ones(2)))
            self.run(label+"-old-unit-chart",args[0],unit,*args[1:],binary=self.old_cpp)
            self.run(label+"-new-asymmetric-fixed-chart",args[0],legacy,*args[1:])
            self.record(label+"-legacy-linear-chart",{"old_asymmetric_h_expected_error":old,
                "old_unit_h_pass":True,"new_asymmetric_fixed_center_pass":True,
                "asymmetric_feature_fixture_unchanged":True})

    def transformations(self,case):
        data=read_fixture(self.cfg); model=Path(case["nonzero"])
        before=self.calc("symmetry-base",model)
        force=before[1:10].reshape(3,3); stress=stress_matrix(before[10:])
        axis=np.array([.31,-.47,.73]); axis/=np.linalg.norm(axis); angle=.713
        cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        Q=np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*cross
        transforms={"rotation":Q,"reflection":Q@np.diag([-1.,1.,1.]),"inversion":-np.eye(3)}
        for name,matrix in transforms.items():
            current={**data,"positions":data["positions"]@matrix.T,"cell":data["cell"]@matrix.T,
                     "forces":data["forces"]@matrix.T,"stress":matrix@data["stress"]@matrix.T}
            path=self.out/f"{name}.cfg"; write_geometry(path,current)
            prediction=self.calc(name,model,path)
            np.testing.assert_allclose(prediction[0],before[0],atol=8e-6,rtol=5e-7)
            np.testing.assert_allclose(prediction[1:10].reshape(3,3),force@matrix.T,atol=8e-6,rtol=5e-7)
            np.testing.assert_allclose(stress_matrix(prediction[10:]),matrix@stress@matrix.T,atol=8e-6,rtol=5e-7)
        translated={**data,"positions":data["positions"]+np.array([1.137,-.219,2.071])}
        path=self.out/"translation.cfg"; write_geometry(path,translated)
        np.testing.assert_allclose(self.calc("translation",model,path),before,atol=8e-6,rtol=5e-7)
        order=np.array([2,0,1]); permuted={**data,"positions":data["positions"][order],"forces":data["forces"][order],"types":np.asarray(data["types"])[order].tolist()}
        path=self.out/"permutation.cfg"; write_geometry(path,permuted)
        prediction=self.calc("permutation",model,path)
        np.testing.assert_allclose(prediction[[0,*range(10,16)]],before[[0,*range(10,16)]],atol=8e-6,rtol=5e-7)
        np.testing.assert_allclose(prediction[1:10].reshape(3,3),force[order],atol=8e-6,rtol=5e-7)
        self.record("native-symmetries",{"cases":[*transforms,"translation","permutation"],"cell_transformed":True,"CFG_output_tolerance":[8e-6,5e-7]})

    def schema(self,case):
        text=Path(case["nonzero"]).read_text(); B=numbers(text,"tensor_message_coeffs")
        bad={"short-B":put_numbers(text,"tensor_message_coeffs",B[:-1]),
             "extra-B":put_numbers(text,"tensor_message_coeffs",[*B,.23]),
             "nan-B":put(text,"tensor_message_coeffs","{nan, "+", ".join(map(str,B[1:]))+"}"),
             "inf-B":put(text,"tensor_message_coeffs","{inf, "+", ".join(map(str,B[1:]))+"}"),
             "no-B":re.sub(r"(?m)^tensor_message_coeffs.*\n?","",text),
             "no-count":re.sub(r"(?m)^tensor_message_coeff_count.*\n?","",text),
             "wrong-count":put(text,"tensor_message_coeff_count",str(len(B)+1)),
             "no-lmax":re.sub(r"(?m)^tensor_message_lmax.*\n?","",text),
             "zero-v14":put(text,"tensor_message_lmax","0"),
             "large-lmax":put(text,"tensor_message_lmax","3"),
             "negative-lmax":put(text,"tensor_message_lmax","-1"),
             "old-version-active":put(text,"version","1.3.0"),
             "no-readout":re.sub(r"(?m)^readout_mode.*\n?","",text),
             "invalid-readout":put(text,"readout_mode","invalid")}
        assert re.search(r"(?m)^tensor_message_optimizer_scales\s*=",text),"template should include default optimizer metadata"
        bad.update({"short-optimizer-scales":put_numbers(text,"tensor_message_optimizer_scales",np.ones(len(B)-1)),
            "extra-optimizer-scales":put_numbers(text,"tensor_message_optimizer_scales",np.ones(len(B)+1)),
            "zero-optimizer-scale":put_numbers(text,"tensor_message_optimizer_scales",[0.,*np.ones(len(B)-1)]),
            "negative-optimizer-scale":put_numbers(text,"tensor_message_optimizer_scales",[-1.,*np.ones(len(B)-1)]),
            "nan-optimizer-scale":put(text,"tensor_message_optimizer_scales","{nan, "+", ".join(["1."]*(len(B)-1))+"}")})
        for name,content in bad.items():
            path=self.out/f"bad-{name}.mtp"; path.write_text(content)
            self.run("reject-"+name,"roundtrip-model-dev",path,self.out/f"rejected-{name}.mtp",fail=True)
        self.run("reject-nonadaptive","init-sh",self.out/"nonadaptive.mtp","--species-count=2","--l-max=2","--k-max=3","--body-order=3","--tensor-message-lmax=2",fail=True)
        self.run("reject-loaded-mismatch","train",case["nonzero"],self.cfg,"--max-iter=0","--tensor-message-lmax=1",fail=True)
        self.run("reject-loaded-disable","train",case["nonzero"],self.cfg,"--max-iter=0","--tensor-message-lmax=0",fail=True)
        self.run("reject-loaded-enable","train",case["legacy"],self.cfg,"--max-iter=0","--tensor-message-lmax=2",fail=True)
        if self.old_cpp:
            self.run("old-reject-v14","roundtrip-model-dev",case["nonzero"],self.out/"old-rejected.mtp",binary=self.old_cpp,fail=True)
            top=self.out/"mono.topology.mtp"; full=self.out/"mono.mtp"
            self.run("mono-init","init-sh",top,*topology_args(species=1))
            self.run("mono-materialize","roundtrip-model-dev",top,full)
            self.run("old-reject-mono-v14","roundtrip-model-dev",full,self.out/"old-rejected-mono.mtp",binary=self.old_cpp,fail=True)
        roundtrip=self.out/"roundtrip.mtp"
        self.run("tensor-roundtrip","roundtrip-model-dev",case["nonzero"],roundtrip)
        np.testing.assert_allclose(numbers(roundtrip.read_text(),"tensor_message_coeffs"),B,atol=0,rtol=2e-15)
        np.testing.assert_allclose(self.calc("roundtrip-efs",roundtrip),self.calc("before-roundtrip",case["nonzero"]),atol=2e-10,rtol=2e-10)
        # Optimizer coordinate metadata must not alter the stored physical B
        # or the E/F/virial function. Its omission means identity coordinates.
        baseline=self.calc("optimizer-scale-baseline",case["nonzero"])
        for label,scaled_text in (("absent",re.sub(r"(?m)^tensor_message_optimizer_scales.*\n?","",text)),
                ("nontrivial",put_numbers(text,"tensor_message_optimizer_scales",.7+.09*np.arange(len(B))))):
            source=self.out/f"optimizer-scales-{label}.mtp"; source.write_text(scaled_text)
            saved=self.out/f"optimizer-scales-{label}-roundtrip.mtp"
            self.run("optimizer-scales-"+label,"roundtrip-model-dev",source,saved)
            np.testing.assert_allclose(numbers(saved.read_text(),"tensor_message_coeffs"),B,atol=0,rtol=2e-15)
            np.testing.assert_allclose(self.calc("optimizer-scales-"+label+"-efs",saved),baseline,atol=2e-10,rtol=2e-10)
        self.record("schema",{"invalid_cases":list(bad),"old_binary_checked":bool(self.old_cpp)})

    def optimizer(self):
        fresh=[]
        for msg in (0,2):
            top=self.out/f"fresh.m{msg}.topology.mtp"; model=self.out/f"fresh.m{msg}.mtp"
            self.run(f"fresh-init{msg}","init-sh",top,*topology_args(message=msg))
            output=self.run(f"fresh-profile{msg}","train",top,self.cfg,"--max-iter=0","--init-params=same",
                "--init-random-seed=20260910","--energy-weight=1","--force-weight=0.1","--stress-weight=0.01",
                "--std-weight=0","--stdd-weight=0",f"--trained-pot-name={model}")
            assert "[adaptive-pytorch-profile] fresh=1" in output
            fresh.append(model)
        for key in PARAMETER_FIELDS:
            np.testing.assert_allclose(numbers(fresh[0].read_text(),key),numbers(fresh[1].read_text(),key),atol=0,rtol=2e-15,err_msg=key)
        def radial_block(path):
            raw=re.search(r"(?ms)^\s*radial_coeffs\s*\n(.*?)(?=^\s*alpha_moments_count\s*=)",path.read_text())
            assert raw,"missing native radial/type coefficient block"
            return np.asarray([float(v) for v in re.findall(NUMBER,raw[1])])
        np.testing.assert_allclose(radial_block(fresh[0]),radial_block(fresh[1]),atol=0,rtol=2e-15,
                                   err_msg="fresh base radial/type coefficients and RNG sequence")
        np.testing.assert_array_equal(numbers(fresh[1].read_text(),"tensor_message_coeffs"),np.zeros(12))
        np.testing.assert_allclose(self.calc("fresh-disabled",fresh[0]),self.calc("fresh-enabled",fresh[1]),atol=2e-10,rtol=2e-10)
        trained=self.out/"bfgs2.mtp"; trace=self.out/"bfgs2.csv"
        output=self.run("bfgs2","train",fresh[1],self.cfg,"--max-iter=2","--optimizer=bfgs","--bfgs-line-search=wolfe",
            "--optimizer-conv-tol=0","--energy-weight=1","--force-weight=0.1","--stress-weight=0.01",
            "--std-weight=0","--stdd-weight=0",f"--trained-pot-name={trained}",f"--bfgs-trace-file={trace}",
            f"--curr-pot-name={self.out/'bfgs-current.mtp'}",f"--best-pot-name={self.out/'bfgs-best.mtp'}",timeout=300)
        assert "[adaptive-pytorch-profile] fresh=0" in output
        with trace.open() as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==2 and [int(row["step"]) for row in rows]==[0,1]
        assert all(math.isfinite(float(v)) for row in rows for v in row.values())
        B=numbers(trained.read_text(),"tensor_message_coeffs")
        assert np.isfinite(B).all() and float(abs(B).max())>1e-12,"actual optimizer failed to update zero B"
        np.testing.assert_array_equal(numbers(fresh[1].read_text(),"species_coeffs"),numbers(trained.read_text(),"species_coeffs"))
        self.calc("bfgs-trained",trained)
        self.record("native-optimizer",{"accepted_points":2,"B_max_abs":float(abs(B).max()),"h_frozen":True,"fresh_B_zero_no_rng_change":True})

    def check(self,stage="minimal"):
        self.cfg=self.out/"asymmetric.cfg"; write_fixture(self.cfg)
        first=self.case("same_same_species_K")
        if stage=="full":
            for i,(gate,outer) in enumerate((("same-k","same-k"),("same-k","q-total"),("q-total","same-k"),("q-total","q-total"))):
                for readout in ("shared","species"):
                    if i==0 and readout=="species": continue
                    label=f"{'same' if gate=='same-k' else 'cross'}_{'same' if outer=='same-k' else 'cross'}_{readout}"
                    self.case(label,gate=gate,outer=outer,readout=readout,scaling=("K","L","LK")[i%3],message=1 if readout=="shared" else 2)
            self.zero_legacy_gradients("legacy-species",first)
            shared=next(c for c in self.report["cases"].values() if c["readout"]=="shared")
            self.zero_legacy_gradients("legacy-shared",shared)
            self.transformations(first); self.schema(first); self.optimizer()
            assert len(self.report["cases"])==8
        self.report.update(status="PASS",stage=stage)
        self.record("coverage",{"stage":stage,"cases":len(self.report["cases"]),"all_requested_coverage":stage=="full"})
        (self.out/("FULL_PASS" if stage=="full" else "MINIMAL_PASS")).write_text("PASS\n")
        return self.report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp",type=Path,required=True)
    parser.add_argument("--old-cpp",type=Path)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--stage",choices=("minimal","full"),default="minimal")
    args=parser.parse_args()
    if args.stage=="full" and not args.old_cpp: parser.error("full requires --old-cpp")
    TensorCheck(args.cpp,args.outdir,args.old_cpp).check(args.stage)
