#!/usr/bin/env python3
"""Exercise independent Legacy gate/outer topology with distinct K/R/O."""
import pathlib
import re
import subprocess
import sys
import tempfile


def field(text, key):
    matches = re.findall(r"^\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*$", text, re.M)
    assert len(matches) == 1, (key, matches)
    return matches[0]


binary = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="legacy-channel-init-") as folder:
    records = {}
    for gate in ("q-total", "same-k"):
        for outer in ("q-total", "same-k"):
            for readout in ("shared", "species"):
                path = pathlib.Path(folder) / f"{gate}-{outer}-{readout}.mtp"
                subprocess.run([binary, "init-sh", str(path), "--species-count=2",
                                "--l-max=2", "--k-max=4", "--body-order=3",
                                "--radial-basis-size=4", "--superlinear-mode=legacy",
                                "--outer-k-rank=3", "--two-layer-gate",
                                "--two-layer-gate-k-rank=2",
                                "--two-layer-gate-mode=mu-scalar-full",
                                f"--sh-factor-pruning={outer}",
                                f"--two-layer-gate-factor-pruning={gate}",
                                f"--readout-mode={readout}"], check=True,
                               stdout=subprocess.DEVNULL)
                text = path.read_text()
                expected = "1.2.0" if (gate, outer, readout) == ("q-total", "q-total", "shared") else "1.8.0"
                assert field(text, "version") == expected
                saved = path.with_suffix(".saved.mtp")
                reloaded = path.with_suffix(".reloaded.mtp")
                for source, target in ((path, saved), (saved, reloaded)):
                    subprocess.run([binary, "roundtrip-model-dev", str(source), str(target)],
                                   check=True, stdout=subprocess.DEVNULL)
                    assert field(target.read_text(), "version") == expected
                if expected == "1.8.0":
                    canonical = saved.read_text()
                    invalid_texts = [canonical + "\nreadout_mode = " + readout + "\n",
                                     re.sub(r"^\s*readout_mode\s*=.*\n", "", canonical, flags=re.M),
                                     canonical + "\nadaptive_radial_weights = {}\n"]
                    for index, invalid_text in enumerate(invalid_texts):
                        invalid = path.with_suffix(f".invalid{index}.mtp")
                        invalid.write_text(invalid_text)
                        result = subprocess.run([binary, "roundtrip-model-dev", str(invalid), str(reloaded)],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        assert result.returncode != 0, (gate, outer, readout, index)
                assert int(field(text, "two_layer_gate_k_rank_basic_count")) == 2 * 9
                assert int(field(text, "outer_k_rank1_basic_count")) == 3 * 9
                gate_begin = int(field(text, "two_layer_gate_k_rank_product_begin"))
                gate_end = int(field(text, "two_layer_gate_k_rank_product_end"))
                assert gate_begin <= gate_end == int(field(text, "outer_k_rank1_product_begin"))
                records[gate, outer, readout] = (len(re.findall(r"-?\d+", field(text, "two_layer_gate_scalar_indices"))),
                                                int(field(text, "alpha_scalar_moments")))
    for readout in ("shared", "species"):
        for gate in ("q-total", "same-k"):
            assert records[gate, "q-total", readout][0] == records[gate, "same-k", readout][0]
        for outer in ("q-total", "same-k"):
            assert records["q-total", outer, readout][1] == records["same-k", outer, readout][1]
    plain = pathlib.Path(folder) / "plain-species.mtp"
    subprocess.run([binary, "init-sh", str(plain), "--species-count=2", "--l-max=2",
                    "--k-max=4", "--body-order=3", "--readout-mode=species"],
                   check=True, stdout=subprocess.DEVNULL)
    saved_plain = plain.with_suffix(".saved.mtp")
    subprocess.run([binary, "roundtrip-model-dev", str(plain), str(saved_plain)],
                   check=True, stdout=subprocess.DEVNULL)
    assert field(saved_plain.read_text(), "version") == "1.8.0"
    assert field(saved_plain.read_text(), "readout_mode") == "species"
    print("PASS: 8 Legacy K4/R2/O3 independent-channel initializer cases + unranked species")
