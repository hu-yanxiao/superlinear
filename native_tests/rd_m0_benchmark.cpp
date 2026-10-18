// One-process, one-CPU, fixed-model angular-backend benchmark.
// Link against the same production objects as mlp-sus2, excluding main.o.
// Usage: model cfg energy|efs|gradient|efs-total max_configs repeats warmup_passes dump|-
// max_configs=0 means all configurations; warmup_passes must be >=1.
// Main modes use identical prebuilt neighborhoods. Timings exclude loading,
// neighborhood construction, complete-dataset warmup, checksums and dump I/O.
// The extra efs-total control includes neighborhood construction via CalcEFS(cfg).
// Gradient mode calls the real unregularized nonlinear-regression objective:
//   sum_cfg [dE^2/N + 0.1 sum_ia dF_ia^2 + 0.01 ||dVirial||_F^2/N].
// It performs fresh EFS prediction and the full physical-coefficient VJP on
// every pass, including force/stress mixed derivatives, without optimization.
// The production full-edge cache may be reused only between the forward and
// gradient of the same configuration. No cached-prediction objective is used.
// Optional dump is JSONL at 17-digit precision: one metadata record followed
// by per-configuration predictions, or one loss/full-gradient record.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "dev_src/mtpr.h"
#include "dev_src/non_linear_regression.h"
#include "dev_src/rd_m0_angular.h"
#include <mpi.h>
#include <sys/resource.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double Now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int Integer(const char* text, const char* name, int minimum)
{
    size_t consumed=0;
    const std::string input(text);
    long long value=std::stoll(input,&consumed);
    if (consumed!=input.size() || value<minimum
        || value>std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ")+name);
    return static_cast<int>(value);
}

void Finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw std::runtime_error(std::string("nonfinite ")+name);
}

double AsDouble(long double value, const char* name)
{
    const double result=static_cast<double>(value);
    Finite(result,name);
    return result;
}

// This is a bitwise mutation guard, not a cryptographic provenance hash.
// The runner should separately hash the immutable input files and binary.
class Fingerprint {
    uint64_t hash_=UINT64_C(14695981039346656037);
public:
    void Word(uint64_t word) {
        for (int b=0;b<8;++b) {
            hash_^=static_cast<unsigned char>(word & 255);
            hash_*=UINT64_C(1099511628211);
            word>>=8;
        }
    }
    void Real(double value) {
        static_assert(sizeof(double)==sizeof(uint64_t),"requires binary64 double");
        uint64_t bits;
        std::memcpy(&bits,&value,sizeof(bits));
        Word(bits);
    }
    uint64_t Value() const { return hash_; }
};

uint64_t ReferenceFingerprint(const std::vector<Configuration>& refs)
{
    Fingerprint result;
    result.Word(refs.size());
    for (const Configuration& cfg:refs) {
        result.Word(cfg.size());
        for (int a=0;a<3;++a) for (int b=0;b<3;++b)
            result.Real(cfg.lattice[a][b]);
        for (int i=0;i<cfg.size();++i) {
            result.Word(cfg.type(i));
            for (int a=0;a<3;++a) result.Real(cfg.pos(i,a));
        }
        result.Word(cfg.has_energy());
        result.Word(cfg.has_forces());
        result.Word(cfg.has_stresses());
        if (cfg.has_energy()) result.Real(cfg.energy);
        if (cfg.has_forces())
            for (int i=0;i<cfg.size();++i) for (int a=0;a<3;++a)
                result.Real(cfg.force(i,a));
        if (cfg.has_stresses())
            for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                result.Real(cfg.stresses[a][b]);
    }
    return result.Value();
}

std::vector<Configuration> Load(const char* path, int maximum, bool gradient)
{
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("cannot open reference dataset");
    std::vector<Configuration> refs;
    Configuration cfg;
    while ((maximum==0 || refs.size()<static_cast<size_t>(maximum)) && cfg.Load(input)) {
        if (cfg.size()<=0) throw std::runtime_error("empty configuration in dataset");
        if (gradient && (!cfg.has_energy() || !cfg.has_forces() || !cfg.has_stresses()))
            throw std::runtime_error("gradient benchmark requires every E/F/virial label");
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            Finite(cfg.lattice[a][b],"reference lattice");
            if (cfg.has_stresses()) Finite(cfg.stresses[a][b],"reference virial");
        }
        if (cfg.has_energy()) Finite(cfg.energy,"reference energy");
        for (int i=0;i<cfg.size();++i) for (int a=0;a<3;++a) {
            Finite(cfg.pos(i,a),"reference position");
            if (cfg.has_forces()) Finite(cfg.force(i,a),"reference force");
        }
        refs.push_back(cfg);
    }
    if (refs.empty()) throw std::runtime_error("no configurations loaded");
    return refs;
}

class FixedObjective : public NonLinearRegression {
public:
    explicit FixedObjective(MLMTPR& model)
        : NonLinearRegression(&model,1.0,0.1,0.01,0.0,0.0) {
        weighting="vibrations";
        norm_by_forces=0;
        std_scaling=0.0;
        stdd_scaling=0.0;
        consistent_std_value_loss=true;
        force_loss_kind=ForceLossKind::L2;
        radial_smooth_regularization=0.0;
        fixed_atomic_energies.clear();
        fixed_atomic_energy_weight=0.0;
        scalar_head_l2_regularization=0.0;
        gate_scalar_l2_regularization=0.0;
        gate_mix_l2_regularization=0.0;
        gate_full_l2_regularization=0.0;
        gate_x_l2_regularization=0.0;
        guard_compression_floor_weight=0.0;
        guard_compression_monotonic_weight=0.0;
        guard_compression_curvature_weight=0.0;
        guard_compression_block_monotonic_weight=0.0;
        ClearGuardCompressionSamples();
        SetCollectErrorMetrics(false);
    }
    double Loss() const { return loss_; }
    const std::vector<double>& Gradient() const { return loss_grad_; }
};

struct Checksums {
    long double energy=0,site_energy=0,force_square=0,virial_square=0;
    long double loss=0,gradient=0,gradient_weighted=0,gradient_square=0;
    void Predictions(const std::vector<Configuration>& predictions, bool efs) {
        for (const Configuration& cfg:predictions) {
            if (!cfg.has_energy() || cfg.cal_se.size()!=static_cast<size_t>(cfg.size()))
                throw std::runtime_error("prediction lacks energy/site energies");
            Finite(cfg.energy,"predicted energy");
            energy+=cfg.energy;
            for (double value:cfg.cal_se) {
                Finite(value,"predicted site energy");
                site_energy+=value;
            }
            if (!efs) continue;
            if (!cfg.has_forces() || !cfg.has_stresses())
                throw std::runtime_error("EFS prediction lacks forces/virial");
            for (int i=0;i<cfg.size();++i) for (int a=0;a<3;++a) {
                const double value=cfg.force(i,a);
                Finite(value,"predicted force");
                force_square+=static_cast<long double>(value)*value;
            }
            for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
                const double value=cfg.stresses[a][b];
                Finite(value,"predicted virial");
                virial_square+=static_cast<long double>(value)*value;
            }
        }
    }
    void Objective(const FixedObjective& objective, int count) {
        Finite(objective.Loss(),"EFS loss");
        if (objective.Gradient().size()!=static_cast<size_t>(count))
            throw std::runtime_error("physical gradient cardinality mismatch");
        loss+=objective.Loss();
        for (int i=0;i<count;++i) {
            const double value=objective.Gradient()[i];
            Finite(value,"physical gradient");
            gradient+=value;
            gradient_weighted+=static_cast<long double>(1+i%97)*value;
            gradient_square+=static_cast<long double>(value)*value;
        }
    }
};

void SameModel(MLMTPR& model, const std::vector<double>& coefficients, double scale)
{
    if (model.CoeffCount()!=static_cast<int>(coefficients.size()) || model.scaling!=scale)
        throw std::runtime_error("fixed model changed during benchmark");
    if (std::memcmp(model.Coeff(),coefficients.data(),coefficients.size()*sizeof(double))!=0)
        throw std::runtime_error("physical coefficients changed during benchmark");
}

void Dump(const std::string& path, const std::string& mode,
          const std::vector<Configuration>& predictions, long long atoms,
          const FixedObjective& objective, int coefficients)
{
    if (path=="-") return;
    // Refuse overwrites, including accidentally naming the original CFG/MTP.
    struct stat info;
    if (::stat(path.c_str(),&info)==0)
        throw std::runtime_error("refusing to overwrite an existing dump path");
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("cannot create benchmark dump");
    out << std::setprecision(17)
        << "{\"schema\":\"rd_m0_benchmark_dump_v1\",\"mode\":\"" << mode
        << "\",\"configs\":" << predictions.size() << ",\"atoms\":" << atoms
        << ",\"coeff_count\":" << coefficients << "}\n";
    if (mode=="gradient") {
        out << "{\"loss\":" << objective.Loss() << ",\"gradient\":[";
        for (int i=0;i<coefficients;++i)
            out << (i ? "," : "") << objective.Gradient()[i];
        out << "]}\n";
    } else for (size_t k=0;k<predictions.size();++k) {
        const Configuration& cfg=predictions[k];
        out << "{\"index\":" << k << ",\"atoms\":" << cfg.size()
            << ",\"energy\":" << cfg.energy << ",\"site_energies\":[";
        for (int i=0;i<cfg.size();++i) out << (i ? "," : "") << cfg.cal_se[i];
        out << "]";
        if (mode=="efs" || mode=="efs-total") {
            out << ",\"forces\":[";
            for (int i=0;i<cfg.size();++i) for (int a=0;a<3;++a)
                out << (i || a ? "," : "") << cfg.force(i,a);
            out << "],\"virial\":[";
            for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                out << (a || b ? "," : "") << cfg.stresses[a][b];
            out << "]";
        }
        out << "}\n";
    }
    out.flush();
    if (!out) throw std::runtime_error("failed writing benchmark dump");
}

int AffinityCount()
{
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0,sizeof(affinity),&affinity)!=0)
        throw std::runtime_error("cannot inspect CPU affinity");
    return CPU_COUNT(&affinity);
#else
    throw std::runtime_error("benchmark requires Linux CPU-affinity verification");
#endif
}

void Run(int argc, char** argv)
{
    if (argc!=8)
        throw std::runtime_error("usage: rd_m0_benchmark model cfg energy|efs|gradient|efs-total max_configs repeats warmup_passes dump|-");
    int ranks=0;
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if (ranks!=1) throw std::runtime_error("benchmark requires exactly one MPI rank");
    const int cpus=AffinityCount();
    if (cpus!=1) throw std::runtime_error("bind benchmark to exactly one CPU, for example taskset -c CPU");
    const std::string mode(argv[3]);
    if (mode!="energy" && mode!="efs" && mode!="gradient" && mode!="efs-total")
        throw std::runtime_error("mode must be energy, efs, gradient, or efs-total");
    const int maximum=Integer(argv[4],"max_configs",0);
    const int repeats=Integer(argv[5],"repeats",1);
    const int warmup=Integer(argv[6],"warmup_passes",1);
    const bool gradient=mode=="gradient";
    // Enabled() denotes raw-D reduction only; the physical-Y ablations
    // ry-new/hybrid must retain their own labels in the timing evidence.
    const char* backend=RDM0Angular::BackendName();
    MLMTPR model(argv[1]);
    if (model.CoeffCount()<=0) throw std::runtime_error("model has no coefficients");
    const std::vector<double> coefficients(model.Coeff(),model.Coeff()+model.CoeffCount());
    const double physical_scale=model.scaling;
    for (double value:coefficients) Finite(value,"model coefficient");
    Finite(physical_scale,"model scale");
    std::vector<Configuration> refs=Load(argv[2],maximum,gradient);
    const uint64_t original_fingerprint=ReferenceFingerprint(refs);
    // Predictions never alias the reference labels used by the objective.
    std::vector<Configuration> predictions=refs;
    std::vector<Neighborhoods> neighborhoods;
    neighborhoods.reserve(refs.size());
    long long atoms=0;
    for (const Configuration& cfg:refs) {
        atoms+=cfg.size();
        neighborhoods.emplace_back(cfg,model.CutOff());
    }
    FixedObjective objective(model);
    auto pass=[&]() {
        if (gradient) objective.CalcObjectiveFunctionGrad(refs,&neighborhoods);
        else for (size_t i=0;i<predictions.size();++i) {
            if (mode=="energy") model.CalcEnergyAndSiteEnergies(predictions[i],neighborhoods[i]);
            else if (mode=="efs-total") model.CalcEFS(predictions[i]);
            else model.CalcEFS(predictions[i],neighborhoods[i]);
        }
    };
    auto inspect=[&](Checksums& checks) {
        if (gradient) checks.Objective(objective,model.CoeffCount());
        else checks.Predictions(predictions,mode!="energy");
        SameModel(model,coefficients,physical_scale);
        if (ReferenceFingerprint(refs)!=original_fingerprint)
            throw std::runtime_error("reference geometry/labels mutated during evaluation");
    };
    std::cerr << "RD_M0_BENCH warmup mode=" << mode << " backend=" << backend
              << " configs=" << refs.size() << " passes=" << warmup << std::endl;
    for (int i=0;i<warmup;++i) {
        pass();
        Checksums discarded;
        inspect(discarded);
    }
    Checksums checks;
    std::vector<double> pass_seconds;
    pass_seconds.reserve(repeats);
    double seconds=0.0;
    for (int i=0;i<repeats;++i) {
        const double start=Now();
        pass();
        const double elapsed=Now()-start;
        if (!(elapsed>0.0) || !std::isfinite(elapsed))
            throw std::runtime_error("invalid monotonic elapsed time");
        seconds+=elapsed;
        pass_seconds.push_back(elapsed);
        inspect(checks); // Deliberately outside the measured region.
        std::cerr << "RD_M0_BENCH measured_pass=" << i+1 << " seconds="
                  << std::setprecision(17) << elapsed << std::endl;
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF,&usage)!=0)
        throw std::runtime_error("cannot inspect peak resident memory");
    const double peak_rss_kib=static_cast<double>(usage.ru_maxrss); // Linux KiB.
    Dump(argv[7],mode,predictions,atoms,objective,model.CoeffCount());
    const long long calls=static_cast<long long>(refs.size())*repeats;
    const double atom_calls=static_cast<double>(atoms)*repeats;
    std::cout << std::setprecision(17)
        << "BENCH_JSON {\"mode\":\"" << mode << "\",\"backend\":\"" << backend
        << "\",\"ranks\":1,\"cpu_affinity_count\":" << cpus
        << ",\"dtype\":\"float64\",\"configs\":" << refs.size()
        << ",\"max_configs\":" << maximum << ",\"atoms\":" << atoms
        << ",\"repeats\":" << repeats << ",\"warmup_passes\":" << warmup
        << ",\"calls\":" << calls << ",\"seconds\":" << seconds
        << ",\"seconds_per_atom\":" << seconds/atom_calls
        << ",\"us_per_atom\":" << 1.e6*seconds/atom_calls
        << ",\"atoms_per_second\":" << atom_calls/seconds
        << ",\"peak_rss_kib\":" << peak_rss_kib
        << ",\"neighborhoods_prebuilt\":" << (mode=="efs-total" ? "false" : "true")
        << ",\"timing_scope\":\"" << (mode=="efs-total" ? "efs_with_neighborhood_build" : "kernel_only")
        << "\",\"checksums_timed\":false"
        << ",\"coeff_count\":" << model.CoeffCount()
        << ",\"physical_gradient\":" << (gradient ? "true" : "false")
        << ",\"objective_dataset_calls\":" << (gradient ? repeats : 0)
        << ",\"weighting\":\"vibrations\",\"efs_weights\":[1,0.1,0.01]"
        << ",\"regularization\":false,\"optimizer_steps\":0,\"finite\":true"
        << ",\"reference_unchanged\":true,\"model_unchanged\":true"
        << ",\"reference_fingerprint_fnv1a64\":\"" << original_fingerprint << "\""
        << ",\"energy_sum\":" << AsDouble(checks.energy,"energy checksum")
        << ",\"site_energy_sum\":" << AsDouble(checks.site_energy,"site-energy checksum")
        << ",\"force_square_sum\":" << AsDouble(checks.force_square,"force checksum")
        << ",\"virial_square_sum\":" << AsDouble(checks.virial_square,"virial checksum")
        << ",\"loss_sum\":" << AsDouble(checks.loss,"loss checksum")
        << ",\"loss_last\":" << (gradient ? objective.Loss() : 0.0)
        << ",\"gradient_sum\":" << AsDouble(checks.gradient,"gradient checksum")
        << ",\"gradient_weighted_sum\":" << AsDouble(checks.gradient_weighted,"weighted-gradient checksum")
        << ",\"gradient_square_sum\":" << AsDouble(checks.gradient_square,"gradient-square checksum")
        << ",\"pass_seconds\":[";
    for (size_t i=0;i<pass_seconds.size();++i)
        std::cout << (i ? "," : "") << pass_seconds[i];
    std::cout << "]}" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    // Set before MPI/model/BLAS initialization. The one-CPU affinity check is
    // an additional guard against accidental multicore benchmark execution.
    const char* single_thread[] = {"OMP_NUM_THREADS","MKL_NUM_THREADS",
                                  "OPENBLAS_NUM_THREADS","BLIS_NUM_THREADS"};
    for (const char* name:single_thread)
        if (setenv(name,"1",1)!=0) { std::cerr << "cannot set thread limit\n"; return 2; }
    if (setenv("OMP_DYNAMIC","FALSE",1)!=0 || setenv("MKL_DYNAMIC","FALSE",1)!=0) {
        std::cerr << "cannot disable dynamic threading\n";
        return 2;
    }
    MPI_Init(&argc,&argv);
    try { Run(argc,argv); }
    catch (const MlipException& error) {
        std::cerr << "RD_M0_BENCH_ERROR " << error.What() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,2);
        return 2;
    }
    catch (const std::exception& error) {
        std::cerr << "RD_M0_BENCH_ERROR " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,2);
        return 2;
    }
    MPI_Finalize();
    return 0;
}
