#!/usr/bin/env python3
"""Verify the standalone installer leaves historical source/config intact."""
from pathlib import Path
import subprocess
import tempfile

installer = Path(__file__).resolve().parents[1]/"scripts/install_superlinear_embedded_cpu.sh"
with tempfile.TemporaryDirectory(prefix="sus2-embedded-install-") as temporary:
    root=Path(temporary)
    (root/"src/MAKE/OPTIONS").mkdir(parents=True)
    (root/"src/ML-SUS2").mkdir()
    (root/"src/pair.h").write_text("// LAMMPS test sentinel\n")
    protected={"src/sus2_mtp_zbl.h":"historical zbl\n",
               "src/ML-SUS2/sus2_mtp_zbl.h":"historical package zbl\n",
               "src/pair_sus2_mtp.cpp":"historical pair\n",
               "src/MAKE/OPTIONS/Makefile.ml_sus2_avx2":"historical make\n"}
    for path,text in protected.items(): (root/path).write_text(text)
    for repeat in range(2):
        subprocess.run(["bash",str(installer),str(root)],check=True,capture_output=True,text=True)
        for path,text in protected.items():
            assert (root/path).read_text()==text,path
    assert (root/"src/pair_sus2_embedded.cpp").exists()
    assert (root/"src/sus2_sl_zbl.h").exists()
    assert not (root/"lib").exists(),"Unexpected external library installation"
    (root/"src/pair_sus2_superlinear.cpp").write_text("native wrapper sentinel\n")
    rejected=subprocess.run(["bash",str(installer),str(root)],capture_output=True,text=True)
    assert rejected.returncode!=0 and "separate source tree" in rejected.stderr
print("PASS: repeatable standalone install, historical files untouched, wrapper conflict rejected")
