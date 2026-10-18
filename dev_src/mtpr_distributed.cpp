/* Native staged CPU inference for a one-cutoff owned/ghost domain.
 * Communication belongs to the host application; this file contains no MPI. */
#include "mtpr.h"
#include <algorithm>
#include <cmath>

namespace {
// Each native stage restores the prior flag even if the model rejects an
// input; ordinary training never inherits inference-only radial tables.
struct InferenceRadialScope {
    AnyRadialBasis& basis;
    bool previous;
    explicit InferenceRadialScope(AnyRadialBasis& rb)
        : basis(rb),previous(rb.SetInferenceRadialTableActive(true)) {}
    ~InferenceRadialScope() { basis.SetInferenceRadialTableActive(previous); }
};
void PackRow(const std::vector<double>& values, int width, int atom,
             double*& buffer) {
    if (!width) return;
    std::copy_n(values.data()+static_cast<size_t>(atom)*width,width,buffer);
    buffer+=width;
}
}
void MLMTPR::ConfigureDistributedRadialTable(RadialTableMode mode,
                                            double spacing,size_t max_bytes) {
    if (distributed_stage_!=0)
        ERROR("cannot change radial tables during distributed EFS evaluation");
    p_RadialBasis->ConfigureInferenceRadialTable(mode,spacing,max_bytes);
}

RadialTableStats MLMTPR::DistributedRadialTableStats() const {
    return p_RadialBasis->GetInferenceRadialTableStats();
}

void MLMTPR::AbortDistributedEFS() {
    distributed_stage_=0;
    active_two_layer_gate_values_=nullptr;
    active_two_layer_gate_adjoints_=nullptr;
    active_second_layer_gate_values_=nullptr;
    active_second_layer_gate_adjoints_=nullptr;
    active_two_layer_edge_cache_atom_index_=-1;
    two_layer_forward_final_moment_cache_ready_=false;
}
namespace {
void UnpackRow(std::vector<double>& values, int width, int atom,
               const double*& buffer, bool add) {
    if (!width) return;
    double* row=values.data()+static_cast<size_t>(atom)*width;
    for (int k=0;k<width;++k) {
        if (add) row[k]+=buffer[k]; else row[k]=buffer[k];
    }
    buffer+=width;
}
}

int MLMTPR::DistributedForwardWidth() const {
    if (OuterUsesAdaptiveRadialCoefficients())
        return AdaptiveRadialRowCount()
            +(TensorMessageEnabled() ? TensorMessageComponentCount() : 0);
    if (!RequiresTwoLayerGateEvaluation()) return 0;
    return radial_func_count
        +(HasNonzeroTwoLayerGateEdgeL1Weights() ? 3*TwoLayerGateEdgeL1SourceCount() : 0);
}

int MLMTPR::DistributedReverseWidth() const {
    return DistributedForwardWidth();
}

const Neighborhoods& MLMTPR::DistributedMLNeighborhoods(
        const Neighborhoods& neighborhoods) const {
    return distributed_has_filtered_neighborhoods_ ? distributed_ml_neighborhoods_ : neighborhoods;
}

void MLMTPR::ValidateDistributedDomain(const Configuration& cfg,
                                      const Neighborhoods& neighborhoods) const {
    if (cfg.size()!=distributed_atom_count_ || neighborhoods.size()!=cfg.size()
        || distributed_owned_count_<0 || distributed_owned_count_>cfg.size())
        ERROR("distributed EFS atom counts changed during an evaluation");
    // A complete list is required for each owner; ghosts carry source values,
    // never a second geometric halo or fabricated incomplete neighborhoods.
    for (int i=0;i<cfg.size();++i) {
        const Neighborhood& nbh=neighborhoods[i];
        if (nbh.my_type<0 || nbh.my_type>=species_count || cfg.type(i)!=nbh.my_type)
            ERROR("distributed EFS requires valid model types for owned and ghost atoms");
        if (i>=distributed_owned_count_ && nbh.count!=0)
            ERROR("distributed EFS ghost neighborhoods must be empty");
        if (nbh.count<0 || nbh.inds.size()!=static_cast<size_t>(nbh.count)
            || nbh.types.size()!=static_cast<size_t>(nbh.count)
            || nbh.vecs.size()!=static_cast<size_t>(nbh.count)
            || nbh.dists.size()!=static_cast<size_t>(nbh.count))
            ERROR("distributed EFS neighborhood arrays have inconsistent sizes");
        for (int j=0;j<nbh.count;++j) {
            const int source=nbh.inds[j];
            if (source<0 || source>=cfg.size() || nbh.types[j]!=cfg.type(source)
                || !(nbh.dists[j]>0.0) || !std::isfinite(nbh.dists[j]))
                ERROR("distributed EFS neighborhood contains an invalid source or distance");
        }
    }
}

void MLMTPR::BeginDistributedEFS(Configuration& cfg,
                                const Neighborhoods& neighborhoods,int nowned) {
    if (distributed_stage_!=0)
        ERROR("distributed EFS evaluation already in progress");
    if (multi_layer_gate_enabled_ || TwoLayerResidualEnabled())
        ERROR("distributed EFS does not support historical residual or multilayer gates");
    distributed_owned_count_=nowned;
    distributed_atom_count_=cfg.size();
    ValidateDistributedDomain(cfg,neighborhoods);
    if (neighborhoods.cutoff+1.0e-12<CutOff())
        ERROR("distributed EFS neighborhood cutoff is below the model cutoff");
    if (HasZBL() && ZBLEvaluationEnabled()
        && neighborhoods.cutoff+1.0e-12<ZBL().MaxOuterCutoff())
        ERROR("distributed EFS neighborhood cutoff is below the ZBL cutoff");
    if (HasZBL() && ZBLEvaluationEnabled()
        && ZBL().AtomicNumbers().size()!=static_cast<size_t>(species_count))
        ERROR("distributed EFS ZBL species map does not match the model");
    // A legal ZBL cutoff can exceed the learned model's geometric cutoff.
    // Several native legacy radial bases rely on neighborhood construction
    // for that bound and become nonzero again beyond max_dist. Keep their
    // complete owned environments at exactly the model cutoff while keeping
    // the original, wider directed rows for the once-only ZBL contribution.
    const double ml_cutoff=CutOff();
    distributed_has_filtered_neighborhoods_=neighborhoods.cutoff>ml_cutoff;
    if (distributed_has_filtered_neighborhoods_) {
        distributed_ml_neighborhoods_=neighborhoods;
        distributed_ml_neighborhoods_.cutoff=ml_cutoff;
        for (int i=0;i<nowned;++i) {
            Neighborhood& nbh=distributed_ml_neighborhoods_[i];
            int kept=0;
            for (int j=0;j<nbh.count;++j) {
                if (nbh.dists[j]>=ml_cutoff) continue;
                nbh.inds[kept]=nbh.inds[j]; nbh.types[kept]=nbh.types[j];
                nbh.vecs[kept]=nbh.vecs[j]; nbh.dists[kept]=nbh.dists[j];
                ++kept;
            }
            nbh.count=kept;
            nbh.inds.resize(kept); nbh.types.resize(kept);
            nbh.vecs.resize(kept); nbh.dists.resize(kept);
        }
    }
    const Neighborhoods& ml_neighborhoods=DistributedMLNeighborhoods(neighborhoods);
    ResetEFS(cfg);
    cfg.has_energy(true); cfg.has_forces(true); cfg.has_stresses(true);
    cfg.has_site_energies(true);
    cfg.cal_se.assign(cfg.size(),0.0); cfg.cal_se0.assign(cfg.size(),0.0);
    cfg.type_mean.assign(cfg.unique_elems.size(),0.0);
    two_layer_full_edge_cache_for_next_calc_=false;
    two_layer_reuse_full_edge_cache_once_=false;
    two_layer_forward_final_moment_cache_ready_=false;
    active_two_layer_gate_values_=nullptr;
    active_two_layer_gate_adjoints_=nullptr;
    active_two_layer_edge_cache_atom_index_=-1;
    InferenceRadialScope radial_scope(*p_RadialBasis);
    if (cfg.size()==0) {
        distributed_stage_=1;
        return;
    }
    if (OuterUsesAdaptiveRadialCoefficients()) {
        PrepareAdaptiveRadial(ml_neighborhoods,true,false);
    } else {
        PrepareEvalCaches();
        if (RequiresTwoLayerGateEvaluation()) {
            BuildTwoLayerEdgePrimitiveCache(ml_neighborhoods,true,false,true);
            PrepareTwoLayerGateValues(cfg,ml_neighborhoods);
            two_layer_gate_adjoints_.assign(static_cast<size_t>(cfg.size())*radial_func_count,0.0);
            if (HasNonzeroTwoLayerGateEdgeL1Weights())
                two_layer_gate_edge_l1_adjoints_.assign(static_cast<size_t>(cfg.size())
                    *TwoLayerGateEdgeL1SourceCount()*3,0.0);
            else two_layer_gate_edge_l1_adjoints_.clear();
        }
    }
    distributed_stage_=1;
}

void MLMTPR::PackDistributedForward(int atom,double* buffer) const {
    if (distributed_stage_!=1 || atom<0 || atom>=distributed_atom_count_)
        ERROR("invalid distributed EFS forward pack stage or atom");
    if (OuterUsesAdaptiveRadialCoefficients()) {
        PackRow(adaptive_radial_execution_.coefficients,AdaptiveRadialRowCount(),atom,buffer);
        PackRow(adaptive_radial_execution_.tensor_messages,
                TensorMessageEnabled() ? TensorMessageComponentCount() : 0,atom,buffer);
    } else if (RequiresTwoLayerGateEvaluation()) {
        PackRow(two_layer_gate_values_,radial_func_count,atom,buffer);
        PackRow(two_layer_gate_edge_l1_values_,HasNonzeroTwoLayerGateEdgeL1Weights()
                ? 3*TwoLayerGateEdgeL1SourceCount() : 0,atom,buffer);
    }
}

void MLMTPR::UnpackDistributedForward(int atom,const double* buffer) {
    if (distributed_stage_!=1 || atom<distributed_owned_count_ || atom>=distributed_atom_count_)
        ERROR("invalid distributed EFS forward unpack stage or ghost atom");
    if (OuterUsesAdaptiveRadialCoefficients()) {
        UnpackRow(adaptive_radial_execution_.coefficients,AdaptiveRadialRowCount(),atom,buffer,false);
        UnpackRow(adaptive_radial_execution_.tensor_messages,
                  TensorMessageEnabled() ? TensorMessageComponentCount() : 0,atom,buffer,false);
    } else if (RequiresTwoLayerGateEvaluation()) {
        UnpackRow(two_layer_gate_values_,radial_func_count,atom,buffer,false);
        UnpackRow(two_layer_gate_edge_l1_values_,HasNonzeroTwoLayerGateEdgeL1Weights()
                  ? 3*TwoLayerGateEdgeL1SourceCount() : 0,atom,buffer,false);
    }
}

void MLMTPR::EvaluateDistributedOuterEFS(Configuration& cfg,
                                         const Neighborhoods& neighborhoods) {
    if (distributed_stage_!=1) ERROR("distributed EFS outer stage requires forward preparation");
    ValidateDistributedDomain(cfg,neighborhoods);
    InferenceRadialScope radial_scope(*p_RadialBasis);
    if (cfg.size()==0) {
        distributed_stage_=2;
        return;
    }
    const Neighborhoods& ml_neighborhoods=DistributedMLNeighborhoods(neighborhoods);
    if (OuterUsesAdaptiveRadialCoefficients()) {
        EvaluateAdaptiveRadialSites(cfg,ml_neighborhoods,distributed_owned_count_,true);
    } else {
        if (RequiresTwoLayerGateEvaluation()) {
            // Pair caches were built before the host copied ghost source
            // signals. Rebuild their inexpensive signal-dependent factors.
            InvalidateTwoLayerGateTanhMuCache();
            if (HasNonzeroTwoLayerGateEdgeL1Weights())
                PrepareTwoLayerGateEdgeL1WeightedValues(cfg.size(),
                    two_layer_edge_offsets_cache_.empty() ? 0 : two_layer_edge_offsets_cache_.back());
            PrepareTwoLayerEdgeGateTypeScaleCache(ml_neighborhoods);
            active_two_layer_gate_values_=&two_layer_gate_values_;
            active_two_layer_gate_adjoints_=&two_layer_gate_adjoints_;
        }
        for (int i=0;i<distributed_owned_count_;++i) {
            const Neighborhood& nbh=ml_neighborhoods[i];
            buff_site_energy_0=0.0; buff_site_energy_=0.0;
            buff_site_energy_ders_.resize(nbh.count);
            active_two_layer_edge_cache_atom_index_=RequiresTwoLayerGateEvaluation() ? i : -1;
            CalcSiteEnergyDers(nbh);
            cfg.cal_se[i]=buff_site_energy_; cfg.cal_se0[i]=buff_site_energy_0;
            cfg.energy+=buff_site_energy_;
            for (int j=0;j<nbh.count;++j) {
                cfg.force(i)+=buff_site_energy_ders_[j];
                cfg.force(nbh.inds[j])-=buff_site_energy_ders_[j];
                for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                    cfg.stresses[a][b]-=buff_site_energy_ders_[j][a]*nbh.vecs[j][b];
            }
        }
        active_two_layer_gate_values_=nullptr;
        active_two_layer_gate_adjoints_=nullptr;
        active_two_layer_edge_cache_atom_index_=-1;
    }
    distributed_stage_=2;
}

void MLMTPR::PackDistributedReverse(int atom,double* buffer) const {
    if (distributed_stage_!=2 || atom<0 || atom>=distributed_atom_count_)
        ERROR("invalid distributed EFS reverse pack stage or atom");
    if (OuterUsesAdaptiveRadialCoefficients()) {
        PackRow(adaptive_radial_execution_.coefficient_adjoints,AdaptiveRadialRowCount(),atom,buffer);
        PackRow(adaptive_radial_execution_.tensor_message_adjoints,
                TensorMessageEnabled() ? TensorMessageComponentCount() : 0,atom,buffer);
    } else if (RequiresTwoLayerGateEvaluation()) {
        PackRow(two_layer_gate_adjoints_,radial_func_count,atom,buffer);
        PackRow(two_layer_gate_edge_l1_adjoints_,HasNonzeroTwoLayerGateEdgeL1Weights()
                ? 3*TwoLayerGateEdgeL1SourceCount() : 0,atom,buffer);
    }
}

void MLMTPR::AddDistributedReverse(int atom,const double* buffer) {
    // Multi-hop host reverse exchanges can add to an intermediate ghost.
    if (distributed_stage_!=2 || atom<0 || atom>=distributed_atom_count_)
        ERROR("invalid distributed EFS reverse add stage or atom");
    if (OuterUsesAdaptiveRadialCoefficients()) {
        UnpackRow(adaptive_radial_execution_.coefficient_adjoints,AdaptiveRadialRowCount(),atom,buffer,true);
        UnpackRow(adaptive_radial_execution_.tensor_message_adjoints,
                  TensorMessageEnabled() ? TensorMessageComponentCount() : 0,atom,buffer,true);
    } else if (RequiresTwoLayerGateEvaluation()) {
        UnpackRow(two_layer_gate_adjoints_,radial_func_count,atom,buffer,true);
        UnpackRow(two_layer_gate_edge_l1_adjoints_,HasNonzeroTwoLayerGateEdgeL1Weights()
                  ? 3*TwoLayerGateEdgeL1SourceCount() : 0,atom,buffer,true);
    }
}

void MLMTPR::FinishDistributedEFS(Configuration& cfg,
                                 const Neighborhoods& neighborhoods) {
    if (distributed_stage_!=2) ERROR("distributed EFS finish requires the outer and reverse stages");
    ValidateDistributedDomain(cfg,neighborhoods);
    InferenceRadialScope radial_scope(*p_RadialBasis);
    if (cfg.size()==0) {
        distributed_stage_=0;
        return;
    }
    const Neighborhoods& ml_neighborhoods=DistributedMLNeighborhoods(neighborhoods);
    if (OuterUsesAdaptiveRadialCoefficients()) {
        FinishAdaptiveRadialGateForces(cfg,ml_neighborhoods);
    } else if (RequiresTwoLayerGateEvaluation()) {
        AccumulateTwoLayerGateForceChain(cfg,ml_neighborhoods);
        AccumulateTwoLayerGateEdgeL1ForceChain(cfg,ml_neighborhoods);
    }
    if (HasZBL() && ZBLEvaluationEnabled()) {
        // Native ZBL traverses directed rows at half weight, so empty ghost
        // rows give exactly the owned-domain share without pair-ID rules.
        ZBL().AddTo(cfg,neighborhoods);
        // The established ZBL EFS API does not fill site energies. Use its
        // exact pair kernel and cutoff tables to supply that partition too.
        std::vector<double> inner,outer,outer_sq;
        const std::vector<int>& zs=ZBL().AtomicNumbers();
        FillZBLPairCutoffTables(zs,ZBL().InnerCutoff(),ZBL().OuterCutoff(),
            ZBL().TypewiseCutoffEnabled(),ZBL().TypewiseCutoffFactor(),inner,outer,outer_sq);
        for (int i=0;i<distributed_owned_count_;++i) {
            const Neighborhood& nbh=neighborhoods[i];
            for (int j=0;j<nbh.count;++j) {
                const int p=nbh.my_type*species_count+nbh.types[j];
                if (nbh.dists[j]>=outer[p]) continue;
                cfg.cal_se[i]+=0.5*ComputeZBLPairEnergyCached(
                    MakeZBLPairConstants(zs[nbh.my_type],zs[nbh.types[j]]),
                    nbh.dists[j],inner[p],outer[p]);
            }
        }
    }
    distributed_stage_=0;
}
