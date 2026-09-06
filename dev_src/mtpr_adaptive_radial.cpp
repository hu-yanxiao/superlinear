/* Environment-conditioned radial coefficients, with exact forward-over-reverse
 * differentiation of the E/F/virial training objective. */
#include "mtpr.h"
#include "adaptive_radial_execution.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace {
bool ReferenceComponents() {
    const char* value=std::getenv("SUS2_SH_ADAPTIVE_COMPONENTS");
    if (value==nullptr || std::strcmp(value,"batched")==0) return false;
    if (std::strcmp(value,"reference")==0) return true;
    ERROR("SUS2_SH_ADAPTIVE_COMPONENTS must be batched or reference");
    return false;
}
void AddEdgeDerivative(Configuration& cfg, int i, int j,
                       const Vector3& d, const Vector3& r) {
    cfg.force(i) += d;
    cfg.force(j) -= d;
    for (int a=0; a<3; ++a)
        for (int b=0; b<3; ++b) cfg.stresses[a][b] -= d[a]*r[b];
}
// Native combination-gradient callers pass the negative stress residual
// convention; the tensor weight is added here, exactly as in the established
// AccumulateEFSCombinationGrad API. It is not the weight of stored PlusStress.
void EdgeDirections(const Neighborhood& nbh, int atom,
                    const std::vector<Vector3>& forces, const Matrix3& stress,
                    std::vector<Vector3>& directions) {
    directions.assign(nbh.count, Vector3(0,0,0));
    for (int j=0; j<nbh.count; ++j) {
        if (!forces.empty()) directions[j] = forces[atom]-forces[nbh.inds[j]];
        for (int a=0; a<3; ++a)
            for (int b=0; b<3; ++b) directions[j][a] += stress[a][b]*nbh.vecs[j][b];
    }
}
int AdaptiveRadialDerStride(size_t cache_size, size_t edge_count,
                            int radial_stride) {
    if (edge_count==0 || radial_stride<=0) return radial_stride*5;
    const size_t compact=static_cast<size_t>(radial_stride)*edge_count;
    const size_t full=compact*5;
    if (cache_size==compact) return radial_stride;
    if (cache_size==full) return radial_stride*5;
    ERROR("adaptive radial derivative cache has inconsistent stride");
    return radial_stride*5;
}
bool AdaptiveRadialDersAreCompact(size_t cache_size, size_t edge_count,
                                  int radial_stride) {
    return edge_count>0 && radial_stride>0
        && cache_size==static_cast<size_t>(radial_stride)*edge_count;
}
int AdaptiveRadialCoordDerOffset(int rb_start, int n, bool compact) {
    return compact ? rb_start+n : 5*rb_start+n;
}
}

void MLMTPR::PrepareAdaptiveRadial(const Neighborhoods& neighborhoods,
                                  bool derivatives, bool parameters) {
    AdaptiveRadialExecution& ex = adaptive_radial_execution_;
    bool reuse=parameters && ex.reusable_full_cache && two_layer_edge_cache_ready_
        && two_layer_edge_cache_has_param_derivatives_
        && ex.cached_scaling==scaling && ex.cached_coefficients==regression_coeffs
        && two_layer_edge_cache_atom_count_==neighborhoods.size();
    size_t geometry_cursor=0,metadata_cursor=0;
    if (reuse) for (int i=0;i<neighborhoods.size() && reuse;++i) {
        const Neighborhood& nbh=neighborhoods[i];
        if (metadata_cursor+2>ex.cached_edge_metadata.size()
            || ex.cached_edge_metadata[metadata_cursor++]!=nbh.my_type
            || ex.cached_edge_metadata[metadata_cursor++]!=nbh.count) { reuse=false; break; }
        for (int j=0;j<nbh.count && reuse;++j) {
            if (metadata_cursor+2>ex.cached_edge_metadata.size() || geometry_cursor+4>ex.cached_edge_geometry.size()) { reuse=false; break; }
            if (ex.cached_edge_metadata[metadata_cursor++]!=nbh.inds[j]
                || ex.cached_edge_metadata[metadata_cursor++]!=nbh.types[j]) { reuse=false; break; }
            for (int a=0;a<3;++a)
                if (ex.cached_edge_geometry[geometry_cursor++]!=nbh.vecs[j][a]) reuse=false;
            if (ex.cached_edge_geometry[geometry_cursor++]!=nbh.dists[j]) reuse=false;
        }
    }
    reuse=reuse && geometry_cursor==ex.cached_edge_geometry.size() && metadata_cursor==ex.cached_edge_metadata.size();
    ex.reusable_full_cache=false;
    PrepareEvalCaches();
    active_two_layer_gate_values_ = nullptr;
    active_two_layer_gate_adjoints_ = nullptr;
    active_second_layer_gate_values_ = nullptr;
    active_second_layer_gate_adjoints_ = nullptr;
    active_two_layer_edge_cache_atom_index_ = -1;
    two_layer_full_edge_cache_for_next_calc_ = false;
    two_layer_reuse_full_edge_cache_once_ = false;
    two_layer_forward_final_moment_cache_ready_ = false;
    if (!reuse)
        BuildTwoLayerEdgePrimitiveCache(neighborhoods, derivatives, parameters, !parameters);
    if (!two_layer_edge_cache_ready_)
        ERROR("adaptive radial coefficients require the exact edge primitive cache; do not disable it");
    ex.derivatives = derivatives;
    ex.parameters = parameters;
    ex.rows = AdaptiveRadialRowCount();
    ex.scalars = TwoLayerGateScalarCount();
    ex.harmonics = (sh_l_max_+1)*(sh_l_max_+1);
    ex.radial_stride = static_cast<int>(radial_eval_to_scaling_block_.size())*p_RadialBasis->rb_size;
    const int atoms = neighborhoods.size(), D=ex.rows, Q=ex.scalars;
    const int R=p_RadialBasis->rb_size, B=(sh_l_max_+1)*outer_k_rank_;
    if (two_layer_gate_scalar_values_cache_.size()!=static_cast<size_t>(atoms)*Q)
        ERROR("adaptive radial gate scalar cache has inconsistent size");
    ex.signals.resize(static_cast<size_t>(atoms)*D);
    ex.coefficients.resize(ex.signals.size());
    if (atoms>0)
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,atoms,D,Q,1.0,
                    two_layer_gate_scalar_values_cache_.data(),Q,
                    regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,
                    0.0,ex.signals.data(),D);
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    for (int i=0; i<atoms; ++i)
        for (int l=0; l<=sh_l_max_; ++l)
            for (int o=0; o<outer_k_rank_; ++o) {
                const int row=(l*outer_k_rank_+o)*R;
                const int mu=o*(sh_l_max_+1)+l;
                const double* c=regression_coeffs.data()+RadialCoeffOffset()+mu*RadialCoeffBlockSize();
                for (int n=0; n<R; ++n) {
                    const size_t at=static_cast<size_t>(i)*D+row+n;
                    ex.coefficients[at]=c[n]+amplitude[row+n]*ex.signals[at];
                }
            }
    ex.outer_nodes.resize(static_cast<size_t>(outer_k_rank_)*ex.harmonics);
    for (int o=0; o<outer_k_rank_; ++o)
        for (int l=0; l<=sh_l_max_; ++l)
            for (int m=-l; m<=l; ++m)
                ex.outer_nodes[o*ex.harmonics+l*l+l+m]=OuterKRank1OuterBasicIndex(o,l,m);
    ex.values.resize(alpha_moments_count);
    ex.adjoints.resize(alpha_moments_count);
    if (parameters) {
        ex.tangents.resize(alpha_moments_count);
        ex.tangent_adjoints.resize(alpha_moments_count);
    }
    const size_t edges=two_layer_edge_offsets_cache_.back();
    ex.radial_values.resize(edges*B);
    if (derivatives) ex.radial_derivatives.resize(edges*B);
    if (parameters) ex.radial_tangents.resize(edges*B);
    ex.coefficient_adjoints.assign(static_cast<size_t>(atoms)*D,0.0);
    ex.coefficient_tangents.clear();
}

void MLMTPR::ForwardAdaptiveRadialSite(const Neighborhood& nbh, int atom,
                                     const std::vector<Vector3>* direction) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int H=ex.harmonics,D=ex.rows,R=p_RadialBasis->rb_size;
    const int B=(sh_l_max_+1)*outer_k_rank_;
    const bool tangent=direction!=nullptr;
    const size_t edge_count=two_layer_edge_offsets_cache_.empty()
        ? 0 : two_layer_edge_offsets_cache_.back();
    const int radial_der_stride=ex.derivatives
        ? AdaptiveRadialDerStride(two_layer_edge_radial_ders_cache_.size(),
                                  edge_count, ex.radial_stride)
        : 0;
    const bool compact_radial_ders=ex.derivatives
        && radial_der_stride==ex.radial_stride;
    std::fill(ex.values.begin(),ex.values.end(),0.0);
    if (tangent) std::fill(ex.tangents.begin(),ex.tangents.end(),0.0);
    for (int j=0; j<nbh.count; ++j) {
        const size_t edge=two_layer_edge_offsets_cache_[atom]+j;
        const double* y=two_layer_edge_sh_values_cache_.data()+edge*H;
        const double* dy=ex.derivatives ? two_layer_edge_sh_ders_cache_.data()+edge*3*H : nullptr;
        const double* rb=two_layer_edge_radial_vals_cache_.data()+edge*ex.radial_stride;
        const double* dr=ex.derivatives ? two_layer_edge_radial_ders_cache_.data()+edge*radial_der_stride : nullptr;
        const double* c=ex.coefficients.data()+static_cast<size_t>(nbh.inds[j])*D;
        const double* ct=tangent ? ex.coefficient_tangents.data()+static_cast<size_t>(nbh.inds[j])*D : nullptr;
        const double wr=tangent ? ((*direction)[j][0]*nbh.vecs[j][0]+(*direction)[j][1]*nbh.vecs[j][1]+(*direction)[j][2]*nbh.vecs[j][2])/nbh.dists[j] : 0.0;
        for (int l=0; l<=sh_l_max_; ++l)
            for (int o=0; o<outer_k_rank_; ++o) {
                const int block=l*outer_k_rank_+o,row=block*R,mu=o*(sh_l_max_+1)+l;
                const int rb_start=mu_to_radial_eval_block_[mu]*R;
                double value=0,radial_der=0,value_t=0;
                for (int n=0; n<R; ++n) {
                    value+=c[row+n]*rb[rb_start+n];
                    if (ex.derivatives)
                        radial_der+=c[row+n]*dr[AdaptiveRadialCoordDerOffset(rb_start,n,compact_radial_ders)];
                    if (tangent) value_t+=ct[row+n]*rb[rb_start+n];
                }
                if (tangent) value_t+=radial_der*wr;
                ex.radial_values[edge*B+block]=value;
                if (ex.derivatives) ex.radial_derivatives[edge*B+block]=radial_der;
                if (tangent) ex.radial_tangents[edge*B+block]=value_t;
                const double species=OuterTypeCoeff(nbh.my_type,mu)*OuterTypeCoeff(nbh.types[j],mu);
                for (int m=-l; m<=l; ++m) {
                    const int h=l*l+l+m,node=ex.outer_nodes[o*H+h];
                    ex.values[node]+=species*value*y[h];
                    if (tangent) {
                        double yt=0;
                        for (int a=0; a<3; ++a) yt+=dy[3*h+a]*(*direction)[j][a];
                        ex.tangents[node]+=species*(value_t*y[h]+value*yt);
                    }
                }
            }
    }
    for (size_t p=static_cast<size_t>(OuterKRank1ProductStart());p<sh_products_.size();++p) {
        const SHProduct& t=sh_products_[p];
        ex.values[t.target]+=t.coeff*ex.values[t.left]*ex.values[t.right];
        if (tangent) ex.tangents[t.target]+=t.coeff*(ex.tangents[t.left]*ex.values[t.right]+ex.values[t.left]*ex.tangents[t.right]);
    }
}

void MLMTPR::ReverseAdaptiveRadialGraph(bool tangent) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    for (int p=static_cast<int>(sh_products_.size())-1;p>=OuterKRank1ProductStart();--p) {
        const SHProduct& t=sh_products_[p];
        const double av=t.coeff*ex.adjoints[t.target];
        const double at=tangent ? t.coeff*ex.tangent_adjoints[t.target] : 0.0;
        ex.adjoints[t.left]+=av*ex.values[t.right]+(tangent ? at*ex.tangents[t.right] : 0.0);
        ex.adjoints[t.right]+=av*ex.values[t.left]+(tangent ? at*ex.tangents[t.left] : 0.0);
        if (tangent) {
            ex.tangent_adjoints[t.left]+=at*ex.values[t.right];
            ex.tangent_adjoints[t.right]+=at*ex.values[t.left];
        }
    }
}

void MLMTPR::AccumulateAdaptiveRadialEdgeForces(Configuration& cfg,
        const Neighborhood& nbh,int atom,int component,bool need_forces,bool need_stress) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int H=ex.harmonics,D=ex.rows,R=p_RadialBasis->rb_size;
    const int B=(sh_l_max_+1)*outer_k_rank_;
    for (int j=0;j<nbh.count;++j) {
        const size_t edge=two_layer_edge_offsets_cache_[atom]+j;
        const double* y=two_layer_edge_sh_values_cache_.data()+edge*H;
        const double* dy=two_layer_edge_sh_ders_cache_.data()+edge*3*H;
        const double* rb=two_layer_edge_radial_vals_cache_.data()+edge*ex.radial_stride;
        double* ca=ex.coefficient_adjoints.data()+static_cast<size_t>(nbh.inds[j])*D;
        Vector3 edge_der(0,0,0);
        for (int l=0;l<=sh_l_max_;++l)
            for (int o=0;o<outer_k_rank_;++o) {
                const int block=l*outer_k_rank_+o, row=block*R,mu=o*(sh_l_max_+1)+l;
                double radial_adjoint=0;
                Vector3 angular_adjoint(0,0,0);
                for (int m=-l;m<=l;++m) {
                    const int h=l*l+l+m;
                    const double adj=ex.adjoints[ex.outer_nodes[o*H+h]];
                    radial_adjoint+=adj*y[h];
                    for (int a=0;a<3;++a) angular_adjoint[a]+=adj*dy[3*h+a];
                }
                const double species=OuterTypeCoeff(nbh.my_type,mu)*OuterTypeCoeff(nbh.types[j],mu);
                const double r=ex.radial_values[edge*B+block], rd=ex.radial_derivatives[edge*B+block];
                for (int a=0;a<3;++a)
                    edge_der[a]+=species*(radial_adjoint*rd*nbh.vecs[j][a]/nbh.dists[j]+r*angular_adjoint[a]);
                const int rb_start=mu_to_radial_eval_block_[mu]*R;
                for (int n=0;n<R;++n) ca[row+n]+=species*radial_adjoint*rb[rb_start+n];
            }
        if (component<0) AddEdgeDerivative(cfg,atom,nbh.inds[j],edge_der,nbh.vecs[j]);
        else {
            for (int a=0;a<3;++a) {
                if (need_forces) {
                    forces_cmpnts(atom,component,a)+=edge_der[a];
                    forces_cmpnts(nbh.inds[j],component,a)-=edge_der[a];
                }
                if (need_stress)
                    for (int b=0;b<3;++b) stress_cmpnts[component][a][b]-=edge_der[a]*nbh.vecs[j][b];
            }
        }
    }
}

void MLMTPR::FinishAdaptiveRadialGateForces(Configuration& cfg,const Neighborhoods& neighborhoods,
                                          int component,bool need_forces,bool need_stress) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int atoms=neighborhoods.size(),D=ex.rows,Q=ex.scalars;
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    for (int i=0;i<atoms;++i)
        for (int row=0;row<D;++row) ex.coefficient_adjoints[static_cast<size_t>(i)*D+row]*=amplitude[row];
    ex.scalar_adjoints.resize(static_cast<size_t>(atoms)*Q);
    if (atoms>0)
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,D,1.0,
                    ex.coefficient_adjoints.data(),D,regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,
                    0.0,ex.scalar_adjoints.data(),Q);
    std::vector<Vector3> derivatives;
    for (int i=0;i<atoms;++i) {
        const Neighborhood& nbh=neighborhoods[i];
        CalcTwoLayerGateWeightedScalarDersForScalarSeeds(nbh,derivatives,i,ex.scalar_adjoints.data()+static_cast<size_t>(i)*Q);
        for (int j=0;j<nbh.count;++j) {
            if (component<0) AddEdgeDerivative(cfg,i,nbh.inds[j],derivatives[j],nbh.vecs[j]);
            else for (int a=0;a<3;++a) {
                if (need_forces) {
                    forces_cmpnts(i,component,a)+=derivatives[j][a];
                    forces_cmpnts(nbh.inds[j],component,a)-=derivatives[j][a];
                }
                if (need_stress)
                    for (int b=0;b<3;++b) stress_cmpnts[component][a][b]-=derivatives[j][a]*nbh.vecs[j][b];
            }
        }
    }
}

void MLMTPR::CalcAdaptiveRadialEFS(Configuration& cfg,const Neighborhoods& neighborhoods,bool need_derivatives) {
    const bool cache_for_gradient=need_derivatives && two_layer_full_edge_cache_for_next_calc_;
    ResetEFS(cfg);
    PrepareAdaptiveRadial(neighborhoods,need_derivatives,cache_for_gradient);
    cfg.has_energy(true); cfg.has_forces(need_derivatives); cfg.has_stresses(need_derivatives); cfg.has_site_energies(true);
    cfg.cal_se.resize(cfg.size()); cfg.cal_se0.resize(cfg.size());
    cfg.type_mean.assign(cfg.unique_elems.size(),0.0);
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    for (int i=0;i<cfg.size();++i) {
        ForwardAdaptiveRadialSite(neighborhoods[i],i,nullptr);
        const int type=neighborhoods[i].my_type;
        const double center=linear_coeffs[type];
        double scalar_energy=0;
        if (need_derivatives) std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
        for (int q=0;q<alpha_scalar_moments;++q) {
            const int node=alpha_moment_mapping[q];
            const double weight=center*linear_coeffs[species_count+q]*linear_mults[q];
            scalar_energy+=weight*ex.values[node];
            if (need_derivatives) ex.adjoints[node]+=weight;
            max_linear[q]=std::max(max_linear[q],std::abs(linear_coeffs[species_count+q]*ex.values[node]));
        }
        cfg.cal_se0[i]=scalar_energy;
        cfg.cal_se[i]=regression_coeffs[type]+center+scalar_energy;
        cfg.energy+=cfg.cal_se[i];
        if (need_derivatives) {
            ReverseAdaptiveRadialGraph(false);
            AccumulateAdaptiveRadialEdgeForces(cfg,neighborhoods[i],i);
        }
    }
    if (need_derivatives) FinishAdaptiveRadialGateForces(cfg,neighborhoods);
    if (HasZBL() && ZBLEvaluationEnabled()) {
        if (need_derivatives) ZBL().AddTo(cfg);
        else cfg.energy+=ZBL().ComputeEnergy(cfg);
    }
    if (cache_for_gradient) {
        ex.cached_scaling=scaling;
        ex.cached_coefficients=regression_coeffs;
        ex.cached_edge_geometry.clear(); ex.cached_edge_metadata.clear();
        for (int i=0;i<neighborhoods.size();++i) {
            const Neighborhood& nbh=neighborhoods[i];
            ex.cached_edge_metadata.push_back(nbh.my_type);
            ex.cached_edge_metadata.push_back(nbh.count);
            for (int j=0;j<nbh.count;++j) {
                ex.cached_edge_metadata.push_back(nbh.inds[j]);
                ex.cached_edge_metadata.push_back(nbh.types[j]);
                for (int a=0;a<3;++a) ex.cached_edge_geometry.push_back(nbh.vecs[j][a]);
                ex.cached_edge_geometry.push_back(nbh.dists[j]);
            }
        }
        ex.reusable_full_cache=true;
    }
}

void MLMTPR::CalcAdaptiveRadialComponents(Configuration& cfg,const Neighborhoods& neighborhoods,
                                         bool need_forces,bool need_stress) {
    const bool derivatives=need_forces||need_stress;
    PrepareAdaptiveRadial(neighborhoods,derivatives,false);
    const int columns=LinearEquationCount();
    std::fill(energy_cmpnts,energy_cmpnts+columns,0.0);
    if (need_forces) { forces_cmpnts.resize(cfg.size(),columns,3); forces_cmpnts.set(0); }
    if (need_stress) std::memset(stress_cmpnts,0,columns*sizeof(stress_cmpnts[0]));
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    ex.site_values.resize(static_cast<size_t>(cfg.size())*alpha_moments_count);
    for (int i=0;i<cfg.size();++i) {
        ForwardAdaptiveRadialSite(neighborhoods[i],i,nullptr);
        std::copy(ex.values.begin(),ex.values.end(),ex.site_values.data()+static_cast<size_t>(i)*alpha_moments_count);
        energy_cmpnts[neighborhoods[i].my_type]+=1.0;
        for (int q=0;q<alpha_scalar_moments;++q) energy_cmpnts[species_count+q]+=ex.values[alpha_moment_mapping[q]];
    }
    if (!derivatives) return;
    const bool reference=ReferenceComponents();
    const int D=ex.rows,P=alpha_scalar_moments;
    if (!reference)
        ex.component_coefficient_adjoints.resize(static_cast<size_t>(cfg.size())*P*D);
    // The linear solver's columns are raw scalar bases (linear_mults and
    // species prefactors are handled by its established coordinate chart).
    // Reuse graph values and radial contractions for all columns.
    for (int q=0;q<alpha_scalar_moments;++q) {
        std::fill(ex.coefficient_adjoints.begin(),ex.coefficient_adjoints.end(),0.0);
        for (int i=0;i<cfg.size();++i) {
            std::copy(ex.site_values.data()+static_cast<size_t>(i)*alpha_moments_count,
                      ex.site_values.data()+static_cast<size_t>(i+1)*alpha_moments_count,ex.values.begin());
            std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
            ex.adjoints[alpha_moment_mapping[q]]=1.0;
            ReverseAdaptiveRadialGraph(false);
            AccumulateAdaptiveRadialEdgeForces(cfg,neighborhoods[i],i,species_count+q,need_forces,need_stress);
        }
        if (reference)
            FinishAdaptiveRadialGateForces(cfg,neighborhoods,species_count+q,need_forces,need_stress);
        else for (int i=0;i<cfg.size();++i)
            std::copy(ex.coefficient_adjoints.data()+static_cast<size_t>(i)*D,
                      ex.coefficient_adjoints.data()+static_cast<size_t>(i+1)*D,
                      ex.component_coefficient_adjoints.data()+(static_cast<size_t>(i)*P+q)*D);
    }
    if (!reference) FinishAdaptiveRadialGateComponents(cfg,neighborhoods,need_forces,need_stress);
}

void MLMTPR::FinishAdaptiveRadialGateComponents(Configuration& cfg,const Neighborhoods& neighborhoods,
                                                bool need_forces,bool need_stress) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int P=alpha_scalar_moments,D=ex.rows,Q=ex.scalars,H=ex.harmonics;
    const int basic_count=static_cast<int>(two_layer_gate_required_basic_indices_.size());
    const int gate_mu_count=static_cast<int>(two_layer_gate_required_mu_indices_.size());
    const int cached_moments=static_cast<int>(two_layer_gate_required_moment_indices_.size());
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    std::vector<double> scalar_seeds(static_cast<size_t>(P)*Q);
    std::vector<double> adj(static_cast<size_t>(alpha_moments_count)*P);
    std::vector<double> gate_values(alpha_moments_count,0.0);
    std::vector<double> basic_seeds(static_cast<size_t>(P)*basic_count),jacobian,derivatives;
    for (int atom=0;atom<cfg.size();++atom) {
        const Neighborhood& nbh=neighborhoods[atom];
        const int coords=3*nbh.count;
        if (coords==0) continue;
        double* seeds=ex.component_coefficient_adjoints.data()+static_cast<size_t>(atom)*P*D;
        for (int p=0;p<P;++p)
            for (int row=0;row<D;++row) seeds[p*D+row]*=amplitude[row];
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,P,Q,D,1.0,seeds,D,
                    regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,0.0,scalar_seeds.data(),Q);
        std::fill(adj.begin(),adj.end(),0.0);
        for (int q=0;q<Q;++q) {
            double* target=adj.data()+static_cast<size_t>(TwoLayerGateScalarMomentIndexUnchecked(q))*P;
            for (int p=0;p<P;++p) target[p]+=scalar_seeds[p*Q+q];
        }
        const double* cached=two_layer_gate_moment_values_cache_.data()+static_cast<size_t>(atom)*cached_moments;
        for (int k=0;k<cached_moments;++k) gate_values[two_layer_gate_required_moment_indices_[k]]=cached[k];
        // All head seeds traverse the same gate graph together. Consecutive
        // head lanes are vectorizable and no scalar-coordinate Jacobian of
        // the entire product graph is materialized.
        for (int k=static_cast<int>(two_layer_gate_required_product_indices_.size())-1;k>=0;--k) {
            const SHProduct& term=sh_products_[two_layer_gate_required_product_indices_[k]];
            const double* target=adj.data()+static_cast<size_t>(term.target)*P;
            double* left=adj.data()+static_cast<size_t>(term.left)*P;
            double* right=adj.data()+static_cast<size_t>(term.right)*P;
            const double wl=term.coeff*gate_values[term.right],wr=term.coeff*gate_values[term.left];
            for (int p=0;p<P;++p) {left[p]+=wl*target[p];right[p]+=wr*target[p];}
        }
        if (TwoLayerGateKRankEnabled())
            for (int channel=0;channel<two_layer_gate_k_rank_;++channel)
                for (int l=0;l<=sh_l_max_;++l)
                    for (int m=-l;m<=l;++m) {
                        const double* contracted=adj.data()+static_cast<size_t>(TwoLayerGateKRankBasicIndex(channel,l,m))*P;
                        for (int k=0;k<sh_k_max_;++k) {
                            const int raw=TwoLayerGateKRankRawBasicIndex(l,m,k);
                            if (raw<0) continue;
                            double* target=adj.data()+static_cast<size_t>(raw)*P;
                            const double w=TwoLayerGateKRankWeight(channel,l,k);
                            for (int p=0;p<P;++p) target[p]+=w*contracted[p];
                        }
                    }
        jacobian.resize(static_cast<size_t>(basic_count)*coords);
        derivatives.resize(static_cast<size_t>(P)*coords);
        for (int b=0;b<basic_count;++b) {
            const int node=two_layer_gate_required_basic_indices_[b];
            for (int p=0;p<P;++p) basic_seeds[p*basic_count+b]=adj[static_cast<size_t>(node)*P+p];
        }
        for (int j=0;j<nbh.count;++j) {
            const size_t edge=two_layer_edge_offsets_cache_[atom]+j;
            const double* y=two_layer_edge_sh_values_cache_.data()+edge*H;
            const double* dy=two_layer_edge_sh_ders_cache_.data()+edge*3*H;
            const double* r=two_layer_edge_gate_mu_vals_cache_.data()+edge*gate_mu_count;
            const double* dr=two_layer_edge_gate_mu_ders_cache_.data()+edge*gate_mu_count;
            for (int b=0;b<basic_count;++b) {
                const int node=two_layer_gate_required_basic_indices_[b],mu=basic_mu_cache_[node];
                const int dense=two_layer_gate_required_basic_dense_mu_indices_[b],h=basic_sh_index_cache_[node];
                const double species=TwoLayerGateTypeCoeff(nbh.my_type,mu)*TwoLayerGateTypeCoeff(nbh.types[j],mu);
                for (int a=0;a<3;++a)
                    jacobian[static_cast<size_t>(b)*coords+3*j+a]=species*(dr[dense]*nbh.vecs[j][a]/nbh.dists[j]*y[h]+r[dense]*dy[3*h+a]);
            }
        }
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,P,coords,basic_count,1.0,
                    basic_seeds.data(),basic_count,jacobian.data(),coords,0.0,derivatives.data(),coords);
        for (int p=0;p<P;++p) for (int j=0;j<nbh.count;++j) for (int a=0;a<3;++a) {
            const double d=derivatives[static_cast<size_t>(p)*coords+3*j+a];
            if (need_forces) {
                forces_cmpnts(atom,species_count+p,a)+=d;
                forces_cmpnts(nbh.inds[j],species_count+p,a)-=d;
            }
            if (need_stress)
                for (int b=0;b<3;++b) stress_cmpnts[species_count+p][a][b]-=d*nbh.vecs[j][b];
        }
    }
}

void MLMTPR::AccumulateAdaptiveRadialCombinationGrad(Configuration& cfg,const Neighborhoods& neighborhoods,
        const std::vector<double>& ene_weight,const std::vector<Vector3>& frc_weights,const Matrix3& str_weights,
        Array1D& out) {
    if (ene_weight.size()!=static_cast<size_t>(cfg.size()) || (!frc_weights.empty() && frc_weights.size()!=static_cast<size_t>(cfg.size())))
        ERROR("adaptive radial combination weights have inconsistent sizes");
    bool tangent=false;
    for (const Vector3& f:frc_weights) if (f.NormSq()!=0.0) tangent=true;
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) if (str_weights[a][b]!=0.0) tangent=true;
    // Even energy-only parameter gradients require scaling derivatives.
    PrepareAdaptiveRadial(neighborhoods,true,true);
    out.resize(CoeffCount());
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int atoms=cfg.size(),D=ex.rows,Q=ex.scalars,H=ex.harmonics,R=p_RadialBasis->rb_size;
    const int B=(sh_l_max_+1)*outer_k_rank_;
    const size_t edge_count=two_layer_edge_offsets_cache_.empty()
        ? 0 : two_layer_edge_offsets_cache_.back();
    const int radial_der_stride=AdaptiveRadialDerStride(
        two_layer_edge_radial_ders_cache_.size(), edge_count, ex.radial_stride);
    if (AdaptiveRadialDersAreCompact(two_layer_edge_radial_ders_cache_.size(),
                                     edge_count, ex.radial_stride))
        ERROR("adaptive radial mixed parameter gradient requires full radial derivative cache");
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    const double* weights=regression_coeffs.data()+AdaptiveRadialWeightOffset();
    std::vector<Vector3> direction;
    std::vector<double> scalar_t, moment_t;
    if (tangent) {
        ex.scalar_tangents.resize(static_cast<size_t>(atoms)*Q);
        ex.gate_moment_tangents.resize(static_cast<size_t>(atoms)*alpha_moments_count);
        for (int i=0;i<atoms;++i) {
            EdgeDirections(neighborhoods[i],i,frc_weights,str_weights,direction);
            CalcTwoLayerGateScalarDirectionalDerivatives(neighborhoods[i],direction,scalar_t,&moment_t,i);
            std::copy(scalar_t.begin(),scalar_t.end(),ex.scalar_tangents.data()+static_cast<size_t>(i)*Q);
            std::copy(moment_t.begin(),moment_t.end(),ex.gate_moment_tangents.data()+static_cast<size_t>(i)*alpha_moments_count);
        }
        ex.coefficient_tangents.resize(static_cast<size_t>(atoms)*D);
        if (atoms>0)
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,atoms,D,Q,1.0,ex.scalar_tangents.data(),Q,weights,Q,0.0,ex.coefficient_tangents.data(),D);
        for (int i=0;i<atoms;++i) for (int row=0;row<D;++row) ex.coefficient_tangents[static_cast<size_t>(i)*D+row]*=amplitude[row];
        ex.coefficient_tangent_adjoints.assign(static_cast<size_t>(atoms)*D,0.0);
    }
    for (int i=0;i<atoms;++i) {
        const Neighborhood& nbh=neighborhoods[i];
        if (tangent) EdgeDirections(nbh,i,frc_weights,str_weights,direction);
        ForwardAdaptiveRadialSite(nbh,i,tangent ? &direction : nullptr);
        std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
        if (tangent) std::fill(ex.tangent_adjoints.begin(),ex.tangent_adjoints.end(),0.0);
        const int type=nbh.my_type;
        const double center=linear_coeffs[type];
        out[type]+=ene_weight[i];
        double center_grad=ene_weight[i];
        for (int q=0;q<alpha_scalar_moments;++q) {
            const int node=alpha_moment_mapping[q];
            const double head=linear_coeffs[species_count+q]*linear_mults[q];
            const double feature=ene_weight[i]*ex.values[node]+(tangent ? ex.tangents[node] : 0.0);
            center_grad+=head*feature;
            out[LinearCoeffOffset()+species_count+q]+=center*linear_mults[q]*feature;
            ex.adjoints[node]+=ene_weight[i]*center*head;
            if (tangent) ex.tangent_adjoints[node]+=center*head;
        }
        out[LinearCoeffOffset()+type]+=center_grad;
        ReverseAdaptiveRadialGraph(tangent);
        for (int j=0;j<nbh.count;++j) {
            const size_t edge=two_layer_edge_offsets_cache_[i]+j;
            const double* y=two_layer_edge_sh_values_cache_.data()+edge*H;
            const double* dy=two_layer_edge_sh_ders_cache_.data()+edge*3*H;
            const double* rb=two_layer_edge_radial_vals_cache_.data()+edge*ex.radial_stride;
            const double* dr=two_layer_edge_radial_ders_cache_.data()+edge*radial_der_stride;
            const size_t start=static_cast<size_t>(nbh.inds[j])*D;
            const double* c=ex.coefficients.data()+start;
            const double* ct=tangent ? ex.coefficient_tangents.data()+start : nullptr;
            double* ca=ex.coefficient_adjoints.data()+start;
            double* cat=tangent ? ex.coefficient_tangent_adjoints.data()+start : nullptr;
            const double wr=tangent ? (direction[j][0]*nbh.vecs[j][0]+direction[j][1]*nbh.vecs[j][1]+direction[j][2]*nbh.vecs[j][2])/nbh.dists[j] : 0.0;
            for (int l=0;l<=sh_l_max_;++l)
                for (int o=0;o<outer_k_rank_;++o) {
                    const int block=l*outer_k_rank_+o,row=block*R,mu=o*(sh_l_max_+1)+l;
                    double av=0,at=0,at_y=0;
                    for (int m=-l;m<=l;++m) {
                        const int h=l*l+l+m,node=ex.outer_nodes[o*H+h];
                        av+=ex.adjoints[node]*y[h];
                        if (tangent) {
                            at+=ex.tangent_adjoints[node]*y[h];
                            double yt=0; for (int a=0;a<3;++a) yt+=dy[3*h+a]*direction[j][a];
                            at_y+=ex.tangent_adjoints[node]*yt;
                        }
                    }
                    av+=at_y;
                    const double vi=OuterTypeCoeff(type,mu),vj=OuterTypeCoeff(nbh.types[j],mu),species=vi*vj;
                    const double species_grad=av*ex.radial_values[edge*B+block]+(tangent ? at*ex.radial_tangents[edge*B+block] : 0.0);
                    out[OuterTypeCoeffIndex(type,mu)]+=vj*species_grad;
                    out[OuterTypeCoeffIndex(nbh.types[j],mu)]+=vi*species_grad;
                    const int rb_start=mu_to_radial_eval_block_[mu]*R;
                    const int scaling_block=radial_eval_to_scaling_block_[mu_to_radial_eval_block_[mu]];
                    double slope=0,shift=0;
                    for (int n=0;n<R;++n) {
                        ca[row+n]+=species*(av*rb[rb_start+n]+at*wr*dr[5*rb_start+n]);
                        if (tangent) cat[row+n]+=species*at*rb[rb_start+n];
                        slope+=av*c[row+n]*dr[5*rb_start+R+n];
                        shift+=av*c[row+n]*dr[5*rb_start+3*R+n];
                        if (tangent) {
                            slope+=at*(ct[row+n]*dr[5*rb_start+R+n]+c[row+n]*wr*dr[5*rb_start+2*R+n]);
                            shift+=at*(ct[row+n]*dr[5*rb_start+3*R+n]+c[row+n]*wr*dr[5*rb_start+4*R+n]);
                        }
                    }
                    out[ScalingSlopeOffset(scaling_block,type,nbh.types[j])]+=species*slope;
                    out[ScalingShiftOffset(scaling_block,type,nbh.types[j])]+=species*shift;
                }
        }
    }
    // Aggregate every receiving edge before the readout pullback. No division
    // by a or w: zero amplitude/signal cases retain their correct derivatives.
    for (int i=0;i<atoms;++i)
        for (int l=0;l<=sh_l_max_;++l)
            for (int o=0;o<outer_k_rank_;++o) {
                const int mu=o*(sh_l_max_+1)+l,row0=(l*outer_k_rank_+o)*R;
                for (int n=0;n<R;++n) {
                    const int row=row0+n; const size_t at=static_cast<size_t>(i)*D+row;
                    const double ca=ex.coefficient_adjoints[at];
                    out[RadialCoeffOffset()+mu*RadialCoeffBlockSize()+n]+=ca;
                    double a_grad=ca*ex.signals[at];
                    if (tangent) {
                        double signal_t=0;
                        for (int q=0;q<Q;++q) signal_t+=weights[static_cast<size_t>(row)*Q+q]*ex.scalar_tangents[static_cast<size_t>(i)*Q+q];
                        a_grad+=ex.coefficient_tangent_adjoints[at]*signal_t;
                        ex.coefficient_tangent_adjoints[at]*=amplitude[row];
                    }
                    out[AdaptiveRadialAmplitudeOffset()+row]+=a_grad;
                    ex.coefficient_adjoints[at]*=amplitude[row];
                }
            }
    ex.scalar_adjoints.resize(static_cast<size_t>(atoms)*Q);
    if (atoms>0) {
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,Q,atoms,1.0,ex.coefficient_adjoints.data(),D,two_layer_gate_scalar_values_cache_.data(),Q,1.0,out.data()+AdaptiveRadialWeightOffset(),Q);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,D,1.0,ex.coefficient_adjoints.data(),D,weights,Q,0.0,ex.scalar_adjoints.data(),Q);
        if (tangent) {
            ex.scalar_tangent_adjoints.resize(static_cast<size_t>(atoms)*Q);
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,Q,atoms,1.0,ex.coefficient_tangent_adjoints.data(),D,ex.scalar_tangents.data(),Q,1.0,out.data()+AdaptiveRadialWeightOffset(),Q);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,D,1.0,ex.coefficient_tangent_adjoints.data(),D,weights,Q,0.0,ex.scalar_tangent_adjoints.data(),Q);
        }
    }
    for (int i=0;i<atoms;++i) {
        if (tangent) EdgeDirections(neighborhoods[i],i,frc_weights,str_weights,direction);
        AccumulateTwoLayerGateScalarParamGradForScalarSeeds(neighborhoods[i],out,
            ex.scalar_adjoints.data()+static_cast<size_t>(i)*Q,tangent ? direction.data() : nullptr,i,
            tangent ? ex.gate_moment_tangents.data()+static_cast<size_t>(i)*alpha_moments_count : nullptr,
            tangent ? ex.scalar_tangent_adjoints.data()+static_cast<size_t>(i)*Q : nullptr);
    }
}

void MLMTPR::AddAdaptiveRadialSmoothnessPenalty(double coeff,int grid_size,
                                                double& penalty,Array1D* gradient) {
    if (coeff==0.0) return;
    if (!(coeff>0.0) || grid_size<=0)
        ERROR("adaptive radial smoothness requires nonnegative weight and positive grid size");
    const int R=p_RadialBasis->rb_size,Q=TwoLayerGateScalarCount(),O=outer_k_rank_;
    const int blocks=(sh_l_max_+1)*O,C=species_count;
    const double cutoff=p_RadialBasis->max_dist;
    const double weight=coeff*cutoff/(static_cast<double>(grid_size)*blocks*C*C);
    if (gradient!=nullptr) gradient->resize(CoeffCount());
    std::vector<double> inverse_scale(Q);
    for (int q=0;q<Q;++q) inverse_scale[q]=1.0/AdaptiveRadialWeightOptimizerScale(q);
    // A fixed isotropic prior in normalized gate-feature coordinates:
    // |R_base'|² + sum_q |R_response,q'|². This is a physical a*U
    // regularizer, not a norm of the redundant optimization factors separately.
    // It is not claimed to equal the correlated empirical feature covariance.
    std::vector<double> response(static_cast<size_t>(Q+1)*R),der(Q+1),ds(Q+1),dh(Q+1);
    for (int l=0;l<=sh_l_max_;++l) for (int o=0;o<O;++o) {
        const int mu=o*(sh_l_max_+1)+l,row0=(l*O+o)*R;
        const int c0=RadialCoeffOffset()+mu*RadialCoeffBlockSize();
        for (int n=0;n<R;++n) {
            response[n]=regression_coeffs[c0+n];
            const double a=regression_coeffs[AdaptiveRadialAmplitudeOffset()+row0+n];
            for (int q=0;q<Q;++q)
                response[static_cast<size_t>(q+1)*R+n]=a*regression_coeffs[AdaptiveRadialWeightOffset()+static_cast<size_t>(row0+n)*Q+q]*inverse_scale[q];
        }
        for (int zi=0;zi<C;++zi) for (int zj=0;zj<C;++zj) {
            const int slope=ScalingSlopeOffset(mu_to_K[mu],zi,zj),shift=ScalingShiftOffset(mu_to_K[mu],zi,zj);
            for (int point=0;point<grid_size;++point) {
                const double r=cutoff*(point+0.5)/grid_size;
                p_RadialBasis->RB_Calc(r,regression_coeffs[slope],regression_coeffs[shift],RadialBasisKForMu(mu));
                for (int q=0;q<=Q;++q) {
                    der[q]=ds[q]=dh[q]=0.0;
                    for (int n=0;n<R;++n) {
                        const double b=response[static_cast<size_t>(q)*R+n]*scaling;
                        der[q]+=b*p_RadialBasis->rb_ders[n];
                        if (gradient!=nullptr) {
                            ds[q]+=b*p_RadialBasis->rb_ders[2*R+n];
                            dh[q]+=b*p_RadialBasis->rb_ders[4*R+n];
                        }
                    }
                    penalty+=weight*der[q]*der[q];
                    if (gradient!=nullptr) {
                        (*gradient)[slope]+=2.0*weight*der[q]*ds[q];
                        (*gradient)[shift]+=2.0*weight*der[q]*dh[q];
                    }
                }
                if (gradient==nullptr) continue;
                for (int n=0;n<R;++n) {
                    const double common=2.0*weight*scaling*p_RadialBasis->rb_ders[n];
                    (*gradient)[c0+n]+=common*der[0];
                    const int a_index=AdaptiveRadialAmplitudeOffset()+row0+n;
                    const int u0=AdaptiveRadialWeightOffset()+(row0+n)*Q;
                    double a_grad=0.0;
                    for (int q=0;q<Q;++q) {
                        const double g=common*der[q+1]*inverse_scale[q];
                        a_grad+=g*regression_coeffs[u0+q];
                        (*gradient)[u0+q]+=g*regression_coeffs[a_index];
                    }
                    (*gradient)[a_index]+=a_grad;
                }
            }
        }
    }
}
