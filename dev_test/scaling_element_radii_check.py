#!/usr/bin/env python3
"""Native element-radii initialization, map semantics, random/resume and old-reader checks."""
import argparse, json, math, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="bin/mlp-sus2")
    p.add_argument("--old-binary")
    p.add_argument("--output", default=".local_tmp/scaling-element-radii")
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(a.binary).resolve()
    checks = []
    base = (ROOT / "native_tests/pure_w_regularization.cfg").read_text()
    cfg = out / "two-compositions.cfg"
    cfg.write_text(base + re.sub(r"(?m)^(\s*\d+\s+)1(\s+)", r"\g<1>0\2", base))

    def run(label, *args, b=None, fail=False):
        q = subprocess.run(
            [str(b or binary), *map(str, args)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        (out / (label + ".log")).write_text(q.stdout + q.stderr)
        assert (q.returncode != 0) == fail, (label, (q.stdout + q.stderr)[-2500:])
        checks.append(label)

    def vals(s, key):
        return [
            float(x)
            for x in re.search(r"(?m)^\s*" + key + r"\s*=\s*\{([^}]*)\}", s)[1].split(
                ","
            )
        ]

    def expected(sharing, K):
        blocks = {"K": K, "LK": K * 2, "L": 2, "pair": 1}[sharing]
        result = []
        cov = [0.32, 0.63]
        vdw = [1.674686, 1.714534]
        for block in range(blocks):
            t = (
                (block if sharing == "K" else block // 2) / (K - 1)
                if K > 1 and sharing in ("K", "LK")
                else 0
            )
            result += [3 - 2 * t] * 4
            result += [
                cov[i] + cov[j] + t * ((vdw[i] + vdw[j]) - (cov[i] + cov[j]))
                for i in range(2)
                for j in range(2)
            ]
        return result

    def check(s, sharing, K):
        want = expected(sharing, K)
        got = vals(s, "scal_coeffs")
        assert len(want) == len(got)
        assert max(abs(x - y) for x, y in zip(got, want)) < 2e-14, (
            sharing,
            K,
            got,
            want,
        )
        if re.search(r"^two_layer_gate_scaling_coeffs\s*=", s, re.M):
            assert (
                max(
                    abs(x - y)
                    for x, y in zip(vals(s, "two_layer_gate_scaling_coeffs"), want)
                )
                < 2e-14
            )

    common = [
        "--l-max=1",
        "--body-order=3",
        "--radial-basis-size=3",
        "--cutoff=4",
        "--outer-k-rank=2",
    ]
    for mode in ("static", "effect", "tanh", "adaptive"):
        extra = (
            []
            if mode == "static"
            else [
                "--two-layer-gate",
                "--two-layer-gate-mode=mu-scalar-full",
                "--two-layer-gate-k-rank=2",
                "--two-layer-gate-radial-mode=shared-radial",
                "--two-layer-gate-decoupled-scaling",
            ]
        )
        if mode == "tanh":
            extra += ["--effect-gate-projection=tanh-gate"]
        if mode == "adaptive":
            extra += ["--superlinear-mode=radial-coefficient-gate"]
        for sharing in ("K", "LK", "L", "pair"):
            label = mode + "-" + sharing
            initial = out / (label + ".sus2")
            complete = out / (label + "-complete.sus2")
            run(
                label + "-init",
                "init-sh",
                initial,
                *common,
                *extra,
                "--k-max=3",
                "--scaling-map=" + sharing,
                "--elements=H,O"
            )
            check(initial.read_text(), sharing, 3)
            run(label + "-roundtrip", "roundtrip-model-dev", initial, complete)
            check(complete.read_text(), sharing, 3)
            if a.old_binary:
                run(
                    label + "-old-template",
                    "roundtrip-model-dev",
                    initial,
                    out / (label + "-old.sus2"),
                    b=Path(a.old_binary).resolve(),
                )
                for name, b in [("new", binary), ("old", Path(a.old_binary).resolve())]:
                    run(
                        label + "-" + name + "-efs",
                        "calc-efs",
                        complete,
                        ROOT / "native_tests/pure_w_regularization.cfg",
                        out / (label + "-" + name + ".cfg"),
                        "--no-loss",
                        b=b,
                    )
                assert (out / (label + "-new.cfg")).read_bytes() == (
                    out / (label + "-old.cfg")
                ).read_bytes()
    # K=1 uses the first endpoint; permutation must follow actual symbols.
    for elements in ("H,O", "O,H", "Og,H"):
        path = out / ("elements-" + elements.replace(",", "-") + ".sus2")
        run(
            path.stem,
            "init-sh",
            path,
            "--l-max=1",
            "--k-max=1",
            "--body-order=2",
            "--elements=" + elements,
        )
        if elements == "H,O":
            check(path.read_text(), "LK", 1)
        if elements == "O,H":
            assert (
                max(
                    abs(x - y)
                    for x, y in zip(
                        vals(path.read_text(), "scal_coeffs")[4:8],
                        [1.26, 0.95, 0.95, 0.64],
                    )
                )
                < 1e-14
            )
    # No element mapping preserves the original default file exactly.
    for label, options in [("implicit", []), ("legacy", ["--scaling-init=legacy"])]:
        run(
            "legacy-" + label,
            "init-sh",
            out / ("legacy-" + label + ".sus2"),
            *common,
            "--k-max=3",
            *options
        )
    assert (out / "legacy-implicit.sus2").read_bytes() == (
        out / "legacy-legacy.sus2"
    ).read_bytes()
    run(
        "manual-auto",
        "init-sh",
        out / "manual.sus2",
        *common,
        "--elements=H,O",
        "--scal-range=8,6",
        "--s-range=2,4"
    )
    assert "scaling_init = legacy" in (out / "manual.sus2").read_text()
    run(
        "manual-conflict",
        "init-sh",
        out / "bad.sus2",
        *common,
        "--elements=H,O",
        "--scaling-init=element-radii",
        "--scal-range=8,6",
        fail=True
    )
    run(
        "missing-elements",
        "init-sh",
        out / "bad.sus2",
        *common,
        "--scaling-init=element-radii",
        fail=True
    )
    run("duplicates", "init-sh", out / "bad.sus2", *common, "--elements=H,H", fail=True)
    # Random fresh initialization must leave both outer and independent inner warp fixed.
    for mode in ("static", "effect", "tanh"):
        src = out / (mode + "-K.sus2")
        target = out / (mode + "-random.sus2")
        run(
            mode + "-random",
            "train",
            src,
            cfg,
            "--max-iter=0",
            "--init-params=random",
            "--init-random-seed=71",
            "--skip-preinit",
            "--no-do-lin",
            "--trained-pot-name=" + str(target),
            "--curr-pot-name=" + str(out / (mode + "-current.sus2")),
        )
        check(target.read_text(), "K", 3)
        resumed = out / (mode + "-resumed.sus2")
        run(
            mode + "-resume",
            "train",
            target,
            cfg,
            "--max-iter=1",
            "--fine-tune",
            "--bfgs-coordinate-normalization=none",
            "--elements=H,O",
            "--scaling-init=element-radii",
            "--radial-parameterization=weight-norm",
            "--skip-preinit",
            "--no-do-lin",
            "--trained-pot-name=" + str(resumed),
            "--curr-pot-name=" + str(out / (mode + "-resume-current.sus2")),
        )
        run(
            mode + "-resume-roundtrip",
            "roundtrip-model-dev",
            target,
            out / (mode + "-roundtrip.sus2"),
        )
        assert vals(target.read_text(), "scal_coeffs") == vals(
            resumed.read_text(), "scal_coeffs"
        )
    for index, malformed in enumerate(("", "{}", "H,,O", "{H,O]", "H,O,")):
        run(
            "bad-elements-" + str(index),
            "init-sh",
            out / "bad.sus2",
            *common,
            "--elements=" + malformed,
            fail=True
        )
    run(
        "zbl-conflict",
        "init-sh",
        out / "bad.sus2",
        *common,
        "--elements=H,O",
        "--zbl-elements=O,H",
        fail=True
    )
    run(
        "resume-mapping-conflict",
        "train",
        out / "tanh-random.sus2",
        ROOT / "native_tests/pure_w_regularization.cfg",
        "--max-iter=0",
        "--elements=O,H",
        fail=True,
    )
    config = out / "grouped.yaml"
    config.write_text(
        "schema_version: 1\nmodel:\n  kind: static-radial\n  elements: [H, O]\n  l_max: 1\n  radial_channels: 3\n  radial:\n    warp_sharing: LK\n  coordination:\n    enabled: false\n  interaction:\n    channels: 2\n    body_order: 3\noutput:\n  model: "
        + str(out / "yaml.sus2")
        + "\n"
    )
    (out / "yaml.sus2").unlink(missing_ok=True)
    run("yaml-mapping", "init-sh", config)
    check((out / "yaml.sus2").read_text(), "LK", 3)
    prune_seed = out / "prune-in.sus2"
    prune_full = out / "prune-full.sus2"
    pruned = out / "pruned.sus2"
    run(
        "prune-init",
        "init-sh",
        prune_seed,
        "--elements=H,O",
        "--zbl-elements=H,O",
        "--l-max=1",
        "--k-max=2",
        "--body-order=3",
    )
    run("prune-complete", "roundtrip-model-dev", prune_seed, prune_full)
    run("prune", "prune-model", prune_full, pruned, "--species=1")
    run("prune-reload", "roundtrip-model-dev", pruned, out / "prune-reloaded.sus2")
    assert (
        "scaling_elements = {O}" in pruned.read_text()
        and "zbl_atomic_numbers = {8}" in pruned.read_text()
    )
    if a.old_binary:
        run(
            "old-default-init",
            "init-sh",
            out / "old-default.sus2",
            *common,
            "--k-max=3",
            b=Path(a.old_binary).resolve()
        )
        assert (out / "old-default.sus2").read_bytes() == (
            out / "legacy-implicit.sus2"
        ).read_bytes()
    (out / "report.json").write_text(
        json.dumps(dict(status="PASS", checks=checks), indent=2) + "\n"
    )
    print("PASS", len(checks), "native radii checks;", out)


if __name__ == "__main__":
    main()
