#!/usr/bin/env python3
"""Server-only native species-readout contract and mixed-derivative regression.

No PyTorch dependency. Also importable by the cross-engine campaign verifier.
The fixture is intentionally asymmetric, two-species, and retains body-six paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
ZERO_REG = ["--std-weight=0", "--stdd-weight=0", "--radial-smooth=0",
            "--scalar-head-l2=0", "--gate-scalar-l2=0", "--gate-mix-l2=0",
            "--gate-full-l2=0", "--gate-x-l2=0"]
LOSS_ARGS = ["--max-configs=1", "--energy-weight=1", "--force-weight=0.1",
             "--stress-weight=0.01", *ZERO_REG,
             "--displacement=2e-5", "--abs-tolerance=5e-8", "--rel-tolerance=5e-5"]
GRAPH_FIELDS = ("alpha_moments_count", "alpha_index_basic", "alpha_index_times",
                "sh_products", "alpha_moment_mapping", "alpha_scalar_moments",
                "two_layer_gate_k_rank_basic_begin", "two_layer_gate_k_rank_basic_count",
                "two_layer_gate_k_rank_product_begin", "two_layer_gate_k_rank_product_end",
                "two_layer_gate_scalar_moment_indices", "two_layer_gate_scalar_count")


def require_server() -> None:
    if sys.platform == "darwin" or not os.environ.get("LSB_JOBID"):
        raise SystemExit("Numerical verification requires a server LSF compute allocation")


def assignment(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([^\n]*)", text)
    if not match:
        raise AssertionError(f"missing {key}")
    return match.group(1).strip()


def numbers(text: str, key: str) -> np.ndarray:
    return np.asarray([float(x) for x in re.findall(NUMBER, assignment(text, key))])


def put(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[^\n]*"
    if re.search(pattern, text):
        return re.sub(pattern, f"{key} = {value}", text, count=1)
    if key == "readout_mode":
        # Native metadata parsing is ordered, unlike the PT field dictionary.
        # Place this optional field at its schema anchor, never at EOF.
        anchor = re.search(r"(?m)^outer_radial_mode\s*=\s*[^\n]*\n", text)
        assert anchor, "readout_mode requires the adaptive outer_radial_mode anchor"
        return text[:anchor.end()] + f"{key} = {value}\n" + text[anchor.end():]
    return text.rstrip() + f"\n{key} = {value}\n"


def put_numbers(text: str, key: str, values) -> str:
    return put(text, key, "{" + ", ".join(f"{float(x):.17e}" for x in values) + "}")


def dev_fields(output: str) -> dict[str, str]:
    return dict(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", output))


def topology_args(production=False, species_count=2, readout="shared", *,
                  gate_pruning="same-k", outer_pruning="same-k") -> list[str]:
    l, k, g, o, n = (3, 8, 8, 8, 10) if production else (1, 2, 2, 2, 3)
    return [f"--species-count={species_count}", f"--l-max={l}", f"--k-max={k}",
            "--body-order=6", f"--body-l-max={'3,3,3,2,1' if production else '1,1,1,1,1'}",
            f"--radial-basis-size={n}", f"--cutoff={6.5 if production else 4.8}",
            "--radial-basis-type=RBChebyshev_sss", "--scaling-map=K",
            "--superlinear-mode=radial-coefficient-gate", f"--sh-factor-pruning={outer_pruning}",
            f"--outer-k-rank={o}", "--two-layer-gate", "--two-layer-gate-mode=mu-scalar-full",
            f"--two-layer-gate-k-rank={g}", "--two-layer-gate-body-order=6",
            f"--two-layer-gate-factor-pruning={gate_pruning}", "--two-layer-gate-shared-radial",
            "--write-sh-scalar-info", f"--readout-mode={readout}"]


def asymmetric_shared_text(source: str) -> str:
    """Populate the same small physical fixture for every gate/outer graph."""
    text = put(put(source, "scaling", "1.0"), "readout_mode", "shared")
    radial = re.search(r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)", text)
    assert radial, "missing radial block"
    row_index = 0
    def radial_row(match):
        nonlocal row_index
        row_index += 1
        row_length = len(re.findall(NUMBER, match[0]))
        assert row_length == 5, "tiny radial rows require N=3 radial and S=2 type coefficients"
        values = [(.09 + .013 * row_index + .017 * math.sin(.43 * (j + row_index)))
                  if j < 3 else (1.07 + .08 * row_index + .11 * math.sin(.67 * (j + row_index)))
                  for j in range(row_length)]
        return "{" + ", ".join(f"{v:.17e}" for v in values) + "}"
    block = re.sub(r"\{[^}]*\}", radial_row, radial[2])
    text = text[:radial.start()] + radial[1] + block + text[radial.end():]
    for key, bias, amp in (("adaptive_radial_weights", .03, .08),
                           ("adaptive_radial_amplitudes", .23, .07),
                           ("two_layer_gate_radial_coeffs", .13, .04),
                           ("two_layer_gate_type_coeffs", 1.07, .08),
                           ("two_layer_gate_k_rank_weights", .23, .11),
                           ("moment_coeffs", .04, .12)):
        text = put_numbers(text, key, bias + amp * np.sin(np.arange(len(numbers(text, key))) * .61 + .3))
    text = put_numbers(text, "species_coeffs", [-.7, 1.3])
    return put_numbers(text, "scal_coeffs", 1.8 + .03 * np.arange(len(numbers(text, "scal_coeffs"))))


def write_fixture(path: Path) -> None:
    path.write_text("""BEGIN_CFG
 Size
 3
 Supercell
 7.4 .21 .09
 .12 7.8 .18
 .05 .16 8.1
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1.17 1.39 1.52 .031 -.023 .017
 2 1 2.63 2.11 2.74 -.019 .041 -.029
 3 0 4.22 3.05 1.88 .013 -.011 .037
 Energy
 -4.8123456789
 PlusStress: xx yy zz yz xz xy
 .041 -.027 .018 .006 -.004 .003
END_CFG
""")


def parse_prediction(path: Path) -> np.ndarray:
    text = path.read_text()
    energy = float(re.search(r"(?m)^\s*Energy\s*\n\s*(" + NUMBER + ")", text)[1])
    atoms = re.search(r"(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy", text)[1]
    forces = [float(v) for line in atoms.strip().splitlines() for v in line.split()[-3:]]
    stress = [float(v) for v in re.search(r"(?m)^\s*PlusStress:[^\n]*\n([^\n]*)", text)[1].split()]
    result = np.asarray([energy, *forces, *stress])
    assert np.isfinite(result).all(), path
    return result


class NativeCheck:
    def __init__(self, cpp: Path, out: Path, old_cpp: Path | None = None):
        require_server()
        self.cpp, self.out = cpp.resolve(), out.resolve()
        self.old_cpp = old_cpp.resolve() if old_cpp else None
        self.out.mkdir(parents=True, exist_ok=True)
        self.report = {"status": "RUNNING", "checks": {}}

    def run(self, label, *args, binary=None, fail=False, env=None, timeout=1800):
        result = subprocess.run([str(binary or self.cpp), *map(str, args)], cwd=self.out,
                                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                     "OPENBLAS_NUM_THREADS": "1", **(env or {})},
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout)
        (self.out / f"{label}.log").write_text(result.stdout)
        if (result.returncode == 0) == fail:
            raise AssertionError(f"{label}: unexpected exit {result.returncode}\n{result.stdout[-6000:]}")
        return result.stdout

    def record(self, label, value):
        self.report["checks"][label] = value
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")
        if label == "pruning-combinations":
            summary = {name: {"gate": case["gate_pruning"], "outer": case["outer_pruning"],
                              "coeff_count": int(case["layout"]["coeff_count"]),
                              "mixed_directions": len(case["directions"])}
                       for name, case in value.items()}
            print(label, json.dumps(summary), flush=True)
        else:
            print(label, json.dumps(value), flush=True)

    def calc(self, label, model, cfg=None, binary=None):
        target = self.out / f"{label}.cfg"
        self.run(label, "calc-efs", model, cfg or self.cfg, target, binary=binary)
        return parse_prediction(target)

    def check(self):
        self.cfg = self.out / "asymmetric.cfg"
        write_fixture(self.cfg)
        shared_top = self.out / "shared.topology.sus2"
        species_top = self.out / "species.topology.sus2"
        for mode, path in (("shared", shared_top), ("species", species_top)):
            self.run(f"init-{mode}", "init-sh", path, *topology_args(readout=mode))
        a, b = shared_top.read_text(), species_top.read_text()
        assert assignment(b, "version") == "1.3.0"
        assert assignment(b, "readout_mode") == "species"
        for key in GRAPH_FIELDS:
            assert assignment(a, key) == assignment(b, key), key
        self.record("raw_graph_unchanged", {"fields": list(GRAPH_FIELDS)})
        fresh = self.out/"fresh-zeroiter.sus2"
        profile_args = ["--max-iter=0", "--init-params=same", "--init-random-seed=20260910",
                        "--energy-weight=1", "--force-weight=0.1", "--stress-weight=0.01",
                        "--std-weight=0", "--stdd-weight=0", "--readout-mode=species"]
        fresh_log = self.run("fresh-profile", "train", species_top, self.cfg,
                             f"--trained-pot-name={fresh}", *profile_args)
        assert "[adaptive-pytorch-profile] fresh=1" in fresh_log
        assert assignment(fresh.read_text(),"readout_mode")=="species"
        resumed = self.out/"resumed-zeroiter.sus2"
        resume_log = self.run("resume-profile", "train", fresh, self.cfg,
                              f"--trained-pot-name={resumed}", *profile_args)
        assert "[adaptive-pytorch-profile] fresh=0" in resume_log
        np.testing.assert_allclose(self.calc("fresh-profile-efs",fresh),self.calc("resumed-profile-efs",resumed),atol=2e-10,rtol=2e-10)
        self.record("native-fixed-profile",{"fresh_zeroiter":str(fresh),"resume_preserves_EFS":True})
        # Resume the unmodified fresh checkpoint. Two accepted points (steps
        # 0 and 1) include an actual update, rather than merely the initial
        # evaluated point. Keep the normal production normalization/profile.
        gauge_trained=self.out/"gauge-two-step.sus2"
        gauge_trace=self.out/"gauge-two-step.csv"
        gauge_log=self.run("gauge-two-step","train",fresh,self.cfg,
                          *[a for a in profile_args if not a.startswith("--max-iter=")],
                          "--max-iter=2","--optimizer=bfgs","--bfgs-line-search=wolfe",
                          "--optimizer-conv-tol=0",f"--trained-pot-name={gauge_trained}",
                          f"--curr-pot-name={self.out/'gauge-current.sus2'}",
                          f"--best-pot-name={self.out/'gauge-best.sus2'}",
                          f"--bfgs-trace-file={gauge_trace}",timeout=300)
        assert "[adaptive-pytorch-profile] fresh=0" in gauge_log
        with gauge_trace.open(newline="") as stream:
            rows=list(csv.reader(stream))
        assert rows[0]==["step","total_loss","efs_loss","energy_mae_mev_atom","force_mae_mev_a","stress_mae_ev"]
        assert len(rows)==3,"gauge regression requires exactly two accepted optimizer points"
        for i,row in enumerate(rows[1:]):
            assert len(row)==6 and int(row[0])==i and all(math.isfinite(float(v)) for v in row[1:]),row
        fresh_text,trained_text=fresh.read_text(),gauge_trained.read_text()
        np.testing.assert_array_equal(numbers(fresh_text,"species_coeffs"),numbers(trained_text,"species_coeffs"))
        changed={}
        for key in ("shift_coeffs","moment_coeffs","scal_coeffs","adaptive_radial_amplitudes",
                    "adaptive_radial_weights","two_layer_gate_radial_coeffs",
                    "two_layer_gate_type_coeffs","two_layer_gate_k_rank_weights"):
            old,new=numbers(fresh_text,key),numbers(trained_text,key)
            assert old.shape==new.shape and np.isfinite(new).all(),key
            changed[key]=float(abs(new-old).max()) if old.size else 0.
        assert max(changed.values())>1e-12,"accepted optimizer points did not update any active parameter"
        self.calc("gauge-trained-efs",gauge_trained)
        self.record("native-trained-gauge",{"accepted_points":2,"steps":[0,1],"h_exactly_unchanged":True,
                                            "active_max_abs_changes":changed,"timeout_seconds":300})
        shared = self.out / "shared.sus2"
        self.run("materialize", "roundtrip-model-dev", shared_top, shared)
        text = asymmetric_shared_text(shared.read_text())
        shared.write_text(text)
        q = int(assignment(text, "alpha_scalar_moments"))
        converted = put(put(text, "version", "1.3.0"), "readout_mode", "species")
        converted = put_numbers(converted, "moment_coeffs", (np.array([-.7, 1.3])[:, None] * numbers(text, "moment_coeffs")[None, :]).ravel())
        species = self.out / "species.converted.sus2"
        species.write_text(converted)
        command_converted = self.out / "species.native-converted.sus2"
        self.run("native-conversion", "convert-adaptive-readout-dev", shared, command_converted, "--readout-mode=species")
        conversion_text = command_converted.read_text()
        for key in ("shift_coeffs", "scal_coeffs", "species_coeffs", "adaptive_radial_amplitudes",
                    "adaptive_radial_weights", "two_layer_gate_k_rank_weights",
                    "two_layer_gate_radial_coeffs", "two_layer_gate_type_coeffs", "moment_coeffs"):
            # Native Save emits scientific precision(15), i.e. 16 significant
            # digits; comparing physical arrays is not a bitwise text test.
            np.testing.assert_allclose(numbers(conversion_text,key),numbers(converted,key),atol=0.,rtol=2e-15,err_msg=key)
        def radial_values(source):
            block=re.search(r"(?ms)^\s*radial_coeffs\s*\n(.*?)(?=^\s*alpha_moments_count\s*=)",source)
            assert block
            return np.asarray([float(v) for row in re.findall(r"\{([^}]*)\}",block[1]) for v in re.findall(NUMBER,row)])
        np.testing.assert_allclose(radial_values(conversion_text),radial_values(converted),atol=0.,rtol=2e-15)
        for key in GRAPH_FIELDS:
            assert assignment(text,key)==assignment(conversion_text,key),key
        shared_efs, species_efs = self.calc("shared-efs", shared), self.calc("converted-efs", species)
        np.testing.assert_allclose(shared_efs, species_efs, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(shared_efs,self.calc("native-converted-efs",command_converted),atol=2e-10,rtol=2e-10)
        assert np.linalg.norm(species_efs[1:10]) > 1e-8, "vacuous force fixture"
        self.record("shared_to_species", {"Q": q, "max_abs": float(abs(shared_efs-species_efs).max())})
        zero_shared, zero_species = self.out/"zero-h.shared.sus2", self.out/"zero-h.species.sus2"
        zero_shared.write_text(put_numbers(text, "species_coeffs", [0., 1.3]))
        zero_head = numbers(converted, "moment_coeffs").reshape(2, q).copy(); zero_head[0] = 0.
        zero_species.write_text(put_numbers(put_numbers(converted, "species_coeffs", [0., 1.3]), "moment_coeffs", zero_head.ravel()))
        np.testing.assert_allclose(self.calc("zero-shared", zero_shared), self.calc("zero-species", zero_species), atol=2e-10, rtol=2e-10)
        zero_native = self.out/"zero-h.native-converted.sus2"
        self.run("zero-h-native-conversion","convert-adaptive-readout-dev",zero_shared,zero_native,"--readout-mode=species")
        np.testing.assert_allclose(self.calc("zero-native",zero_native),self.calc("zero-species-check",zero_species),atol=2e-10,rtol=2e-10)
        self.record("zero_h_conversion", {"division_by_h": False})
        # Free rows not restricted to the shared rank-one manifold.
        free = self.out / "species.free.sus2"
        head = numbers(converted, "moment_coeffs").reshape(2, q)
        head[1] += .06 * np.cos(np.arange(q) * .37 + .1)
        free.write_text(put_numbers(converted, "moment_coeffs", head.ravel()))
        base = self.calc("free-efs", free)
        for z, count in enumerate((2, 1)):
            bias_model = self.out / f"bias{z}.sus2"
            biases = np.array([-.7, 1.3]); biases[z] += .17
            bias_model.write_text(put_numbers(free.read_text(), "species_coeffs", biases))
            changed = self.calc(f"bias{z}-efs", bias_model)
            expected = np.zeros_like(base); expected[0] = count * .17
            np.testing.assert_allclose(changed-base, expected, atol=2e-10, rtol=2e-10)
        self.record("species_bias_is_additive_only", {"counts": [2, 1]})
        layout = dev_fields(self.run("layout", "check-loss-gradient-dev", free, self.cfg,
                                    *LOSS_ARGS, "--coeff-start=0", "--coeff-end=1"))
        linear = int(layout["linear_begin"])
        assert int(layout["coeff_count"]) - linear == 2 + 2*q
        ranges = {f"head_species_{z}": (linear+2+z*q, linear+2+(z+1)*q) for z in range(2)}
        ranges.update({p: (int(layout[p+"_begin"]), int(layout[p+"_end"])) for p in
                       ("adaptive_radial_amplitude", "adaptive_radial_weight", "gate_k_rank")})
        for name, (begin, end) in ranges.items():
            result = dev_fields(self.run(name+"-mixed-fd", "check-loss-gradient-direction-dev",
                                        free, self.cfg, *LOSS_ARGS,
                                        f"--coeff-start={begin}", f"--coeff-end={end}"))
            assert end > begin
            assert abs(float(result["analytic_directional"])) > 1e-10, (name, result)
            self.record(name+"_mixed_fd", result)
        for cmd, extra in (("check-linear-readout-dev", ["--check-stress", "--abs-tolerance=2e-9", "--rel-tolerance=2e-9"]),
                           ("check-linear-components-fd-dev", ["--displacement=2e-5", "--abs-tolerance=5e-8", "--rel-tolerance=5e-5"]),
                           ("check-efs-fd-dev", ["--displacement=2e-5", "--abs-tolerance=2e-6", "--rel-tolerance=2e-4"])):
            result = dev_fields(self.run(cmd, cmd, free, self.cfg, "--max-configs=1", "--max-atoms=3", *extra))
            self.record(cmd, result)
        self.run("weightnorm-fold", "check-radial-weight-norm-roundtrip-dev", free)
        bad = {
            "short-head": put_numbers(converted, "moment_coeffs", head.ravel()[:-1]),
            "extra-head": put_numbers(converted, "moment_coeffs", [*head.ravel(), .123]),
            "old-version-species": put(converted, "version", "1.2.0"),
            "v13-shared": put(converted, "readout_mode", "shared"),
            "v13-no-mode": re.sub(r"(?m)^readout_mode[^\n]*\n?", "", converted),
            "invalid-mode": put(converted, "readout_mode", "banana"),
        }
        for name, content in bad.items():
            path = self.out / f"bad-{name}.sus2"; path.write_text(content)
            self.run(name, "roundtrip-model-dev", path, self.out/f"bad-{name}.out.sus2", fail=True)
        self.run("nonadaptive-species", "init-sh", self.out/"nonadaptive.sus2",
                 "--species-count=2", "--l-max=1", "--k-max=2", "--body-order=3",
                 "--readout-mode=species", fail=True)
        default = self.out / "default-shared.sus2"
        default.write_text(re.sub(r"(?m)^readout_mode[^\n]*\n?", "", text))
        np.testing.assert_allclose(self.calc("default-shared", default), shared_efs, atol=2e-10, rtol=2e-10)
        if self.old_cpp:
            np.testing.assert_allclose(self.calc("old-default-shared", default, binary=self.old_cpp), shared_efs,
                                       atol=2e-10, rtol=2e-10)
            self.run("old-reject-species", "roundtrip-model-dev", species, self.out/"old-species.out.sus2",
                     binary=self.old_cpp, fail=True)
            mono_top = self.out/"mono-species.topology.sus2"
            self.run("mono-init", "init-sh", mono_top, *topology_args(species_count=1, readout="species"))
            mono = self.out/"mono-species.sus2"
            self.run("mono-materialize", "roundtrip-model-dev", mono_top, mono)
            self.run("old-reject-mono-species", "roundtrip-model-dev", mono, self.out/"old-mono.out.sus2",
                     binary=self.old_cpp, fail=True)
        self.record("schema_rejection", {"cases": list(bad), "old_binary_checked": bool(self.old_cpp)})
        self.check_pruning_combinations(shared, free, layout)
        self.report["status"] = "PASS"
        self.record("artifacts", {"shared": str(shared), "species": str(free), "cfg": str(self.cfg)})
        (self.out/"PASS").write_text("PASS\n")
        return self.report

    def check_pruning_combinations(self, same_shared, same_free, same_layout):
        """All four pruning combinations, without repeating the heavy suite."""
        direction_names = ("head_species_0", "head_species_1", "adaptive_radial_amplitude",
                           "adaptive_radial_weight", "gate_k_rank")
        combinations = {
            "gate_same_outer_same": {
                "gate_pruning": "same-k", "outer_pruning": "same-k",
                "shared": str(same_shared), "species": str(same_free),
                "converted": str(self.out/"species.native-converted.sus2"),
                "layout": same_layout,
                "directions": {name:self.report["checks"][name+"_mixed_fd"] for name in direction_names},
            }
        }
        for gate, outer in (("same-k", "q-total"), ("q-total", "same-k"), ("q-total", "q-total")):
            label=f"gate_{'same' if gate=='same-k' else 'cross'}_outer_{'same' if outer=='same-k' else 'cross'}"
            paths={mode:self.out/f"{label}.{mode}.topology.sus2" for mode in ("shared", "species")}
            for mode,path in paths.items():
                self.run(label+"-init-"+mode,"init-sh",path,
                         *topology_args(readout=mode,gate_pruning=gate,outer_pruning=outer))
            for key in GRAPH_FIELDS:
                assert assignment(paths["shared"].read_text(),key)==assignment(paths["species"].read_text(),key),(label,key)
            shared=self.out/f"{label}.shared.sus2"
            self.run(label+"-materialize","roundtrip-model-dev",paths["shared"],shared)
            shared.write_text(asymmetric_shared_text(shared.read_text()))
            converted=self.out/f"{label}.converted.sus2"
            self.run(label+"-convert","convert-adaptive-readout-dev",shared,converted,"--readout-mode=species")
            shared_values=self.calc(label+"-shared",shared)
            converted_values=self.calc(label+"-converted",converted)
            np.testing.assert_allclose(shared_values,converted_values,atol=2e-10,rtol=2e-10,err_msg=label)
            text=converted.read_text(); q=int(assignment(text,"alpha_scalar_moments"))
            expected_head=np.array([-.7,1.3])[:,None]*numbers(shared.read_text(),"moment_coeffs")[None,:]
            np.testing.assert_allclose(numbers(text,"moment_coeffs").reshape(2,q),expected_head,atol=0.,rtol=2e-15)
            head=expected_head.copy(); head[1]+=.06*np.cos(np.arange(q)*.37+.1)
            free=self.out/f"{label}.species-free.sus2"
            free.write_text(put_numbers(text,"moment_coeffs",head.ravel()))
            values=self.calc(label+"-free",free)
            assert np.linalg.norm(values[1:10])>1e-8,(label,"vacuous force fixture")
            layout=dev_fields(self.run(label+"-layout","check-loss-gradient-dev",free,self.cfg,
                                      *LOSS_ARGS,"--coeff-start=0","--coeff-end=1"))
            linear=int(layout["linear_begin"])
            assert int(layout["coeff_count"])-linear==2+2*q
            ranges={f"head_species_{z}":(linear+2+z*q,linear+2+(z+1)*q) for z in range(2)}
            ranges.update({n:(int(layout[n+"_begin"]),int(layout[n+"_end"])) for n in direction_names[2:]})
            directions={}
            for name,(begin,end) in ranges.items():
                result=dev_fields(self.run(label+"-"+name,"check-loss-gradient-direction-dev",free,self.cfg,
                                          *LOSS_ARGS,f"--coeff-start={begin}",f"--coeff-end={end}"))
                assert end>begin and abs(float(result["analytic_directional"]))>1e-10,(label,name,result)
                directions[name]=result
            self.run(label+"-linear","check-linear-readout-dev",free,self.cfg,"--max-configs=1",
                     "--max-atoms=3","--check-stress","--abs-tolerance=2e-9","--rel-tolerance=2e-9")
            self.run(label+"-geometry-FD","check-efs-fd-dev",free,self.cfg,"--max-configs=1",
                     "--max-atoms=2","--displacement=2e-5","--abs-tolerance=2e-6","--rel-tolerance=2e-4")
            combinations[label]={"gate_pruning":gate,"outer_pruning":outer,"shared":str(shared),
                                 "species":str(free),"converted":str(converted),"layout":layout,
                                 "gate_Q":int(assignment(text,"two_layer_gate_scalar_count")),"outer_Q":q,
                                 "directions":directions,"conversion_max_abs":float(abs(shared_values-converted_values).max())}
            self.record("pruning-combinations",combinations)
        assert len(combinations)==4
        self.record("pruning-combinations",combinations)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpp", type=Path, required=True)
    p.add_argument("--old-cpp", type=Path)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    NativeCheck(a.cpp, a.outdir, a.old_cpp).check()
