// Python glue only: all model mathematics remain in the audited C++ engine.
// Layout/API patterned after PySUS2SH's bindings/pysus2sh_core.cpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <climits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "dev_src/mtpr.h"

namespace py = pybind11;
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

static void check_finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("nonfinite geometry or model output");
}

class SuperlinearModel {
    std::string path_;
    std::vector<int> species_;
    std::unordered_map<int, int> types_;
    std::unique_ptr<MLMTPR> model_;
public:
    SuperlinearModel(const std::string& path, const std::vector<int>& species)
        : path_(path), species_(species) {
        if (species.empty()) throw std::invalid_argument("explicit species mapping is required");
        for (size_t i=0; i<species.size(); ++i) {
            if (species[i]<1 || species[i]>118 || !types_.emplace(species[i], i).second)
                throw std::invalid_argument("invalid or duplicate atomic number in species mapping");
        }
        model_.reset(new MLMTPR(path));
        if (model_->species_count != static_cast<int>(species.size()))
            throw std::invalid_argument("species mapping length differs from model species_count");
    }

    py::dict calculate(Array<int> numbers, Array<double> positions,
                       Array<double> cell, Array<bool> pbc) {
        // Validate shapes before unchecked access, including empty inputs.
        if (numbers.ndim()!=1 || positions.ndim()!=2 || positions.shape(1)!=3 ||
            numbers.shape(0)!=positions.shape(0) || numbers.shape(0)==0 || numbers.shape(0)>INT_MAX)
            throw std::invalid_argument("expected nonempty numbers[N] and positions[N,3]");
        if (cell.ndim()!=2 || cell.shape(0)!=3 || cell.shape(1)!=3 || pbc.ndim()!=1 || pbc.shape(0)!=3)
            throw std::invalid_argument("expected cell[3,3] and pbc[3]");
        auto num=numbers.unchecked<1>(); auto pos=positions.unchecked<2>();
        auto cel=cell.unchecked<2>(); auto periodic=pbc.unchecked<1>();
        const py::ssize_t n=numbers.shape(0);
        Configuration cfg;
        cfg.resize(static_cast<int>(n));
        Matrix3 original;
        int active_count=0;
        Vector3 basis[3];
        for (int a=0; a<3; ++a) {
            Vector3 v;
            for (int b=0; b<3; ++b) {
                check_finite(cel(a,b)); original[a][b]=cel(a,b);
                cfg.lattice[a][b]=periodic(a) ? cel(a,b) : 0.0;
                v[b]=cel(a,b);
            }
            if (periodic(a)) {
                // Gram-Schmidt prevents singular active periodic rows from
                // entering the native neighbor extension loop.
                for (int j=0; j<active_count; ++j) {
                    double dot=0; for(int b=0;b<3;++b) dot+=v[b]*basis[j][b];
                    for(int b=0;b<3;++b) v[b]-=dot*basis[j][b];
                }
                double norm=0; for(int b=0;b<3;++b) norm+=v[b]*v[b];
                norm=std::sqrt(norm);
                if(norm<0.5) throw std::invalid_argument("periodic cell is degenerate or thinner than 0.5 A");
                for(int b=0;b<3;++b) basis[active_count][b]=v[b]/norm;
                ++active_count;
            }
        }
        for (int i=0; i<n; ++i) {
            auto it=types_.find(num(i));
            if(it==types_.end()) throw std::invalid_argument("element is outside the model species mapping");
            cfg.type(i)=it->second;
            cfg.unique_elems.insert(it->second); cfg.type_num[it->second]++;
            for(int a=0;a<3;++a) {check_finite(pos(i,a)); cfg.pos(i,a)=pos(i,a);}
        }
        // Keep the GIL: the engine has mutable caches and process-global
        // diagnostics. Parallel throughput uses independent CPU processes.
        model_->CalcEFS(cfg);
        check_finite(cfg.energy);
        py::array_t<double> forces({n, py::ssize_t(3)});
        auto f=forces.mutable_unchecked<2>();
        for(int i=0;i<n;++i) for(int a=0;a<3;++a) {check_finite(cfg.force(i,a)); f(i,a)=cfg.force(i,a);}
        py::array_t<double> virial({py::ssize_t(3),py::ssize_t(3)});
        auto v=virial.mutable_unchecked<2>();
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) {check_finite(cfg.stresses[a][b]); v(a,b)=cfg.stresses[a][b];}
        double volume=std::abs(original.det());
        check_finite(volume);
        py::dict out;
        out["energy"]=cfg.energy; out["forces"]=forces; out["virial"]=virial;
        out["volume"]=volume; out["stress"]=py::none();
        if(volume>1e-6) {
            py::array_t<double> stress(6); auto s=stress.mutable_unchecked<1>();
            const int aa[6]={0,1,2,1,0,0}, bb[6]={0,1,2,2,2,1};
            for(int k=0;k<6;++k) {s(k)=-v(aa[k],bb[k])/volume; check_finite(s(k));}
            out["stress"]=stress;
        }
        return out;
    }

    py::dict metadata() const {
        py::dict d;
        d["model_path"]=path_; d["atomic_numbers"]=species_;
        d["source_revision"]=SUPERLINEAR_SOURCE_REVISION;
        d["precision"]="float64"; d["backend"]="superlinear-cpp-in-process";
        d["is_sh_potential"]=model_->IsSHPotential();
        d["species_count"]=model_->species_count; d["cutoff"]=model_->CutOff();
        // L stores the angular-order count; K_ is the number of warp blocks,
        // which is NOT the radial channel count for L/LK/pair sharing.
        d["L"]=model_->L-1;
        d["K"]=model_->L > 0 ? model_->radial_func_count/model_->L : 0;
        d["scaling_map"]=model_->ScalingMap();
        d["coefficient_count"]=model_->CoeffCount();
        d["basis_size"]=model_->p_RadialBasis->rb_size;
        d["basis_type"]=model_->p_RadialBasis->GetRBTypeString();
        d["gate_rank"]=model_->TwoLayerGateKRank(); d["outer_rank"]=model_->OuterKRank();
        d["readout"]=model_->ReadoutMode(); d["joint"]=model_->JointReadoutEnabled();
        d["adaptive_radial"]=model_->OuterUsesAdaptiveRadialCoefficients();
        d["projection"]=model_->AdaptiveRadialProjection();
        d["shared_n_rank"]=model_->AdaptiveRadialSharedNRank();
        d["activation"]=model_->AdaptiveRadialActivation();
        d["amplitude_mode"]=model_->AdaptiveRadialAmplitudeMode();
        d["tensor_message_lmax"]=model_->TensorMessageLMax();
        d["outer_coupling"]=model_->OuterUsesSameChannelGraph() ? "diagonal" : "cross";
        d["coordination_enabled"]=model_->TwoLayerGateEnabled();
        d["kind"]=model_->OuterUsesAdaptiveRadialCoefficients() ? "coordination-adaptive"
            : (model_->OuterUsesPureRadialMatrix() ? "pure-matrix"
               : (model_->TwoLayerGateEnabled() ? "coordination-effect" : "static-radial"));
        return d;
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc()="Persistent superlinear C++ inference for ASE; no subprocess or CFG I/O";
    py::register_exception_translator([](std::exception_ptr p) {
        try { if(p) std::rethrow_exception(p); }
        catch(const MlipException& e) { PyErr_SetString(PyExc_RuntimeError,e.What()); }
    });
    py::class_<SuperlinearModel>(m,"SuperlinearModel")
        .def(py::init<const std::string&,const std::vector<int>&>(),py::arg("model"),py::arg("atomic_numbers"))
        .def("calculate",&SuperlinearModel::calculate,py::arg("numbers"),py::arg("positions"),py::arg("cell"),py::arg("pbc"))
        .def_property_readonly("metadata",&SuperlinearModel::metadata);
}
