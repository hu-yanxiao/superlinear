#!/usr/bin/env python3
"""Small native CLI regression for public templates and model-output contracts.

Requires a freshly built serial executable, not MKL, production data, or jobs.
Fixtures and logs are retained in an isolated temporary directory.
"""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve()
ROOT = Path(tempfile.mkdtemp(prefix="superlinear-release-"))
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
CHECKS = 0


def run(label, args, failure=None):
    global CHECKS
    result = subprocess.run([str(BIN)] + list(map(str, args)), cwd=ROOT, env=ENV,
                            capture_output=True, text=True, timeout=180)
    output = result.stdout + result.stderr
    (ROOT / (label + ".log")).write_text(output)
    if failure:
        assert result.returncode != 0 and failure in output, (label, output[-4000:])
    else:
        assert result.returncode == 0, (label, result.returncode, output[-4000:])
    CHECKS += 1
    return output


def final_path(output):
    value = re.search(r'^resolved_option --trained-pot-name=(.+)$', output, re.M)
    assert value, output[-4000:]
    return Path(shlex.split(value.group(1))[0])


def yaml(name, text):
    path = ROOT / name
    path.write_text(text)
    return path


data = ROOT / "data"
data.mkdir()
fixture = """BEGIN_CFG
 Size
   3
 Supercell
   15 0 0
   0 15 0
   0 0 15
 AtomData: id type cartes_x cartes_y cartes_z fx fy fz
 1 0 1.17 1.39 1.52 0.031 -0.023 0.017
 2 1 2.63 2.11 2.74 -0.019 0.041 -0.029
 3 0 4.22 3.05 1.88 0.013 -0.011 0.037
 Energy
   -4.8123456789
 PlusStress: xx yy zz yz xz xy
   0.041 -0.027 0.018 0.006 -0.004 0.003
END_CFG
"""
for file in ("train.cfg", "val.cfg"):
    (data / file).write_text(fixture)

readme = (REPO / "README.md").read_text()
blocks = re.findall(r"```yaml\n(.*?)```", readme, re.S)
assert len(blocks) == 2
finals = {}
configs = {}
for index, family in enumerate(("static", "effect")):
    template = (REPO / "examples" / ("train-" + family + ".yaml")).read_text()
    assert template[template.index("schema_version:"):].rstrip() == blocks[index].rstrip()
    assert "radial_channels: 4" in template and "channels: 2" in template
    assert "output:" not in template and "directory:" not in template
    if family == "static":
        assert "coordination:" not in template
    else:
        assert "kind: coordination-effect" in template and "native-effect-v1" in template
    config = yaml(family + ".yaml", re.sub(r"iterations:\s*\d+", "iterations: 1", template, count=1))
    configs[family] = config
    first = final_path(run(family + "-check", ["train", config, "--check-config"]))
    second = final_path(run(family + "-check-repeat", ["train", config, "--check-config"]))
    assert first != second and not first.parent.exists() and not second.parent.exists(), "check-config created output"
    output = run(family + "-train", ["train", config])
    final = final_path(output)
    assert "-l3-k4-" in str(final.parent)
    assert ("-r2-o2-" in str(final.parent)) == (family == "effect")
    assert final.name == "final.sus2"
    for name in ("initial.sus2", "current.sus2", "best.sus2", "final.sus2", "optimizer.csv"):
        assert (final.parent / name).is_file(), (family, name)
    finals[family] = final
    run(family + "-errors", ["calc-errors", final, data / "val.cfg"])
    run(family + "-predict", ["calc-efs", final, data / "val.cfg", ROOT / (family + ".pred.cfg")])
    run(family + "-roundtrip", ["roundtrip-model-dev", final, ROOT / (family + ".roundtrip.sus2")])
    assert final.read_bytes() == (ROOT / (family + ".roundtrip.sus2")).read_bytes()
    run(family + "-reject-save", ["roundtrip-model-dev", final, ROOT / (family + ".bad.mtp")], "must end in .sus2")
    assert not (ROOT / (family + ".bad.mtp")).exists()

assert (data / "val.cfg").read_text() == fixture, "prediction overwrote reference labels"
# Repeated real executions of the identical YAML get fresh output directories.
repeat = final_path(run("static-train-repeat", ["train", configs["static"]]))
assert repeat != finals["static"] and repeat.is_file()
assert not list(ROOT.rglob("*.tmp*")), "atomic temporary models were left behind"

# Old model filenames remain read-compatible, but writing those suffixes fails.
old_input = ROOT / "historical.mtp"
old_input.write_bytes(finals["effect"].read_bytes())
run("old-input-predict", ["calc-efs", old_input, data / "val.cfg", ROOT / "old-input.pred.cfg"])
assert (ROOT / "old-input.pred.cfg").read_bytes() == (ROOT / "effect.pred.cfg").read_bytes()

resume = yaml("resume.yaml", """schema_version: 1
data:
  model: """ + str(old_input) + """
  train: data/train.cfg
  valid: data/val.cfg
training:
  profile: native-effect-v1
  iterations: 1
loss:
  energy_weight: 1
  force_weight: 0.01
  virial_weight: 0.001
""")
resumed = final_path(run("resume", ["train", resume]))
assert resumed.is_file() and "resume-historical-" in str(resumed.parent)
assert not (resumed.parent / "initial.sus2").exists()

for kind, profile in (("effect", "native-effect-v1"), ("coordination-legacy", "native-legacy-v1")):
    alias = configs["effect"].read_text().replace("coordination-effect", kind)
    alias = alias.replace("native-effect-v1", profile)
    config = yaml("alias-" + kind + ".yaml", alias)
    aliased = final_path(run("alias-" + kind, ["train", config]))
    assert aliased.read_bytes() == finals["effect"].read_bytes(), kind

bad = yaml("bad-output.yaml", configs["effect"].read_text() + "output:\n  final: bad.mtp\n")
run("reject-explicit-bad-output", ["train", bad, "--check-config"], "partial explicit model output")
bad = yaml("bad-output-with-dir.yaml", configs["effect"].read_text()
           + "output:\n  directory: bad-output\n  final: bad.mtp\n")
run("reject-explicit-bad-suffix", ["train", bad, "--check-config"], "must end in .sus2")
assert not (ROOT / "bad-output").exists()

# Historical positional init-sh is inactive in the YAML adapter, so its direct
# topology writer must enforce the output suffix independently.
init_opts = ["--species-count=2", "--l-max=1", "--k-max=2", "--body-order=3",
             "--cutoff=3.5", "--radial-basis-size=3"]
run("native-init-bad-output", ["init-sh", ROOT / "native.bad.mtp"] + init_opts,
    "must end in .sus2")
assert not (ROOT / "native.bad.mtp").exists()
run("native-init-good-output", ["init-sh", ROOT / "native.good.sus2"] + init_opts)
assert (ROOT / "native.good.sus2").is_file()

print(json.dumps({"result": "PASS", "checks": CHECKS, "fixtures": str(ROOT),
                  "templates": "L3/K4/R2/O2; static has no R",
                  "scope": "serial static/effect, automatic paths, save/read/continue; no MKL Adaptive or real MPI qualification"}, indent=2))
