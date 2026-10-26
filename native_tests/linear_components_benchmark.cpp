// Frozen-model, one-rank/one-core linear feature benchmark. No optimizer.
// Usage: driver MODEL CFG DUMP [repeats=1] [max_configs=1] [warmup=0]
// Only CalcEFSComponents is timed; neighborhoods, copies and I/O are excluded.
// DUMP: native-endian LCOMP001, uint64 frame count, then per frame five uint64
// (atoms, columns, has_E, has_F, has_virial), followed by length-prefixed f64
// vectors: E[col], F[atom,col,xyz], V[col,row,col], targets[E,F,V].
#include "dev_src/mtpr.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/stat.h>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef MLIP_MPI
#include <mpi.h>
#endif

namespace {
double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void Finite(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite model/feature/target");
}
void U64(std::ostream& out, std::uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
void Vector(std::ostream& out, const std::vector<double>& values) {
    U64(out, values.size());
    for (double x : values) Finite(x);
    if (!values.empty()) out.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(double));
}
struct Frame {
    std::uint64_t atoms, columns, has_e, has_f, has_v;
    std::vector<double> e, f, v, targets;
};
Frame Capture(MLMTPR& model, const Configuration& source) {
    Frame frame;
    frame.atoms=source.size(); frame.columns=model.LinearEquationCount();
    frame.has_e=source.has_energy(); frame.has_f=source.has_forces(); frame.has_v=source.has_stresses();
    frame.e.assign(model.energy_cmpnts, model.energy_cmpnts+frame.columns);
    for (int atom=0; atom<source.size(); ++atom)
        for (int col=0; col<static_cast<int>(frame.columns); ++col)
            for (int axis=0; axis<3; ++axis)
                frame.f.push_back(model.forces_cmpnts(atom,col,axis));
    for (int col=0; col<static_cast<int>(frame.columns); ++col)
        for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
            frame.v.push_back(model.stress_cmpnts[col][a][b]);
    frame.targets.push_back(frame.has_e ? source.energy : 0.0);
    for (int atom=0; atom<source.size(); ++atom) for (int a=0; a<3; ++a)
        frame.targets.push_back(frame.has_f ? source.force(atom,a) : 0.0);
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
        frame.targets.push_back(frame.has_v ? source.stresses[a][b] : 0.0);
    return frame;
}
int Run(int argc, char** argv) {
    if (argc<4 || argc>7) throw std::runtime_error("usage: driver MODEL CFG DUMP [repeats=1] [max_configs=1] [warmup=0]");
    const int repeats=argc>4 ? std::stoi(argv[4]) : 1;
    const int limit=argc>5 ? std::stoi(argv[5]) : 1;
    const int warmup=argc>6 ? std::stoi(argv[6]) : 0;
    if (repeats<1 || limit<1 || warmup<0) throw std::runtime_error("invalid bounded repeats/configs/warmup");
    struct stat existing;
    if (stat(argv[3],&existing)==0) throw std::runtime_error("refusing to overwrite an existing dump/input");
    MLMTPR model(argv[1]);
    if (!model.HasCompleteParameters()) throw std::runtime_error("requires a complete frozen model");
    const std::vector<double> before(model.Coeff(),model.Coeff()+model.CoeffCount());
    for (double x:before) Finite(x);
    const double scaling=model.scaling; Finite(scaling);
    const std::vector<double> multipliers=model.linear_mults;
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("cannot open CFG");
    std::vector<Configuration> configs;
    std::vector<Neighborhoods> neighborhoods;
    Configuration cfg;
    while (static_cast<int>(configs.size())<limit && cfg.Load(input)) {
        if (!cfg.size()) throw std::runtime_error("empty configuration");
        configs.push_back(cfg);
        neighborhoods.emplace_back(configs.back(),model.CutOff());
    }
    if (configs.empty()) throw std::runtime_error("no CFG frames");
    double seconds=0.0;
    std::vector<Frame> frames;
    for (int pass=-warmup; pass<repeats; ++pass)
        for (size_t i=0;i<configs.size();++i) {
            Configuration work=configs[i];
            const double start=Now();
            model.CalcEFSComponents(work,neighborhoods[i],true,true);
            const double elapsed=Now()-start;
            if (pass>=0) seconds+=elapsed;
            if (pass==repeats-1) frames.push_back(Capture(model,configs[i]));
            if (std::memcmp(before.data(),model.Coeff(),before.size()*sizeof(double)) ||
                scaling!=model.scaling || multipliers!=model.linear_mults)
                throw std::runtime_error("model physical coefficients/scaling changed");
        }
    std::ofstream output(argv[3],std::ios::binary);
    if (!output) throw std::runtime_error("cannot open dump");
    output.write("LCOMP001",8); U64(output,frames.size());
    for (const Frame& frame:frames) {
        U64(output,frame.atoms); U64(output,frame.columns);
        U64(output,frame.has_e); U64(output,frame.has_f); U64(output,frame.has_v);
        Vector(output,frame.e); Vector(output,frame.f); Vector(output,frame.v); Vector(output,frame.targets);
    }
    output.close();
    if (!output) throw std::runtime_error("dump write failed");
    std::cout<<std::setprecision(17)<<"LINEAR_COMPONENTS_JSON {\"frames\":"<<frames.size()
             <<",\"columns\":"<<model.LinearEquationCount()<<",\"seconds\":"<<seconds
             <<",\"repeats\":"<<repeats<<",\"finite\":true,\"model_unchanged\":true"
             <<",\"timing_scope\":\"CalcEFSComponents_only_prebuilt_neighborhoods\"}\n";
    return 0;
}
}
int main(int argc,char** argv) {
    for (const char* key:{"OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"}) setenv(key,"1",1);
    setenv("MKL_DYNAMIC","FALSE",1);
#ifdef __linux__
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0,sizeof(allowed),&allowed)!=0) return 3;
    int cpu=0; while (cpu<CPU_SETSIZE && !CPU_ISSET(cpu,&allowed)) ++cpu;
    if (cpu==CPU_SETSIZE) return 3;
    cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(cpu,&selected);
    if (sched_setaffinity(0,sizeof(selected),&selected)!=0) return 3;
#endif
#ifdef MLIP_MPI
    MPI_Init(&argc,&argv);
    int ranks=0; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if (ranks!=1) { MPI_Finalize(); return 3; }
#endif
    int code=0;
    try { code=Run(argc,argv); }
    catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; code=1; }
#ifdef MLIP_MPI
    MPI_Finalize();
#endif
    return code;
}
