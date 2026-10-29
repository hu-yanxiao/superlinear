#include "src/configuration.h"
#include "src/neighborhoods.h"
#include <fstream>
#include <iostream>
// Supply a disposable directory as argv[1]. Exercises the same reused object
// used by LoadCfgs, including periodic -> molecule -> periodic transitions.
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string path = std::string(argv[1]) + "/mixed.cfg";
    const std::string atom = " AtomData: id type cartes_x cartes_y cartes_z\n 1 0 0 0 0\n";
    { std::ofstream out(path);
      out << "BEGIN_CFG\n Size\n 1\n Supercell\n 5 0 0\n 0 5 0\n 0 0 5\n" << atom << "END_CFG\n";
      out << "BEGIN_CFG\n Size\n 1\n" << atom << " PlusStress: xx yy zz yz xz xy\n 1 2 3 0 0 0\nEND_CFG\n";
      out << "BEGIN_CFG\n Size\n 1\n Supercell\n 6 0 0\n 0 6 0\n 0 0 6\n" << atom << "END_CFG\n"; }
    Configuration cfg;
    std::ifstream in(path);
    if (!cfg.Load(in) || cfg.lattice.det() != 125) return 3;
    if (!cfg.Load(in) || cfg.lattice.det() != 0 || cfg.stresses[0][0] != 1) return 4;
    try { cfg.virial(); return 5; } catch (const MlipException&) {}
    const std::string saved = std::string(argv[1]) + "/saved.cfg";
    { std::ofstream out(saved); cfg.Save(out); }
    { Configuration roundtrip; std::ifstream input(saved);
      if (!roundtrip.Load(input) || roundtrip.lattice.det() != 0 || roundtrip.stresses[0][0] != 1) return 6; }
    if (!cfg.Load(in) || cfg.lattice.det() != 216) return 7;
    const std::string probe = std::string(argv[1]) + "/probe.cfg";
    const auto load_probe = [&](const std::string& cell, const std::string& stress) {
        { std::ofstream out(probe); out << "BEGIN_CFG\n Size\n 1\n" << cell << atom << stress << "END_CFG\n"; }
        std::ifstream input(probe); Configuration result; result.Load(input); return result;
    };
    const std::string invalid[] = {
        " Supercell\n", " Supercell\n nan 0 0\n",
        " Supercell\n infinity 0 0\n", " Supercell\n x 0 0\n",
        " Supercell\n 5 0\n 0 5 0\n 0 0 5\n",
        " Supercell\n 5 0 0 9\n"};
    for (const auto& cell : invalid) {
        try { load_probe(cell, ""); return 8; } catch (const MlipException&) {}
    }
    try { load_probe("", " Virial: xx yy zz yz xz xy\n 1 2 3 0 0 0\n"); return 9; }
    catch (const MlipException&) {}
    const std::string valid[] = {" Supercell\n 5 0 0\n",
        " Supercell\n 5 0 0\n 0 5 0\n", " Supercell\n 0 0 0\n 0 0 0\n 0 0 0\n"};
    for (const auto& cell : valid) {
        Configuration original = load_probe(cell, "");
        { std::ofstream out(saved); original.Save(out); }
        Configuration copy; std::ifstream input(saved);
        if (!copy.Load(input) || copy.lattice != original.lattice) return 10;
    }
    Configuration molecule;
    molecule.lattice = Matrix3(0,0,0,0,0,0,0,0,0);
    molecule.resize(3);
    molecule.pos(0) = Vector3(0, 0, 0);
    molecule.pos(1) = Vector3(1, 0, 0);
    molecule.pos(2) = Vector3(0, 2, 0);
    Configuration boxed = molecule;
    boxed.lattice = Matrix3(100,0,0,0,100,0,0,0,100);
    Neighborhoods molecular_nbhs(molecule, 2.5), boxed_nbhs(boxed, 2.5);
    for (int i = 0; i < 3; ++i) {
        if (molecular_nbhs[i].count != 2 || boxed_nbhs[i].count != 2) return 11;
        for (int j = 0; j < 2; ++j) {
            const int neighbor = molecular_nbhs[i].inds[j];
            bool found = false;
            for (int k = 0; k < 2; ++k)
                if (boxed_nbhs[i].inds[k] == neighbor &&
                    fabs(boxed_nbhs[i].dists[k] - molecular_nbhs[i].dists[j]) < 1e-12)
                    found = true;
            if (!found) return 12;
        }
    }
    std::cout << "boxless CFG reuse and roundtrip passed\n";
}
