/* Environment-conditioned radial coefficients, with exact forward-over-reverse
 * differentiation of the E/F/virial training objective. */
#include "mtpr.h"
#include "adaptive_radial_execution.h"
#include "rd_m0_angular.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#ifdef MLIP_MPI
#include <mpi.h>
#endif
#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace {
void ReportRDM0Angular(int lmax) {
    static const bool reported=[lmax]() {
#ifdef MLIP_MPI
        int rank=0;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (rank!=0) return true;
#endif
        std::cout << "[RDM0_ANGULAR] backend=rd-m0 lmax=" << lmax
                  << " gate_reduce=RD_then_Yl0 outer_reduce=RD_then_Yl0"
                  << " physical_aux_cache=1" << std::endl;
        return true;
    }();
    (void)reported;
}
void ReportPhysicalAngular(int lmax) {
    static const bool reported=[lmax]() {
#ifdef MLIP_MPI
        int rank=0;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (rank!=0) return true;
#endif
        std::cout << "[ANGULAR_BACKEND] backend=" << RDM0Angular::BackendName()
                  << " lmax=" << lmax
                  << " value_kernel="
                  << (RDM0Angular::PhysicalKernelEnabled(false) ? "rd-m0" : "ry")
                  << " derivative_kernel=rd-m0"
                  << " gate_reduce=RY outer_reduce=RY raw_d_cache=0"
                  << " physical_aux_cache=1" << std::endl;
        return true;
    }();
    (void)reported;
}
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
int TensorComponentLocal(int Lmsg, int outer_channel, int m, int outer_count) {
    return ((Lmsg * Lmsg - 1) * outer_count)
        + outer_channel * (2 * Lmsg + 1) + (m + Lmsg);
}
size_t TensorGateDenseLocal(int gate_channel, int Lmsg, int m, int tensor_lmax) {
    const int width = (tensor_lmax + 1) * 2 + 1;
    return (static_cast<size_t>(gate_channel) * tensor_lmax
            + static_cast<size_t>(Lmsg - 1)) * static_cast<size_t>(width)
        + static_cast<size_t>(m + tensor_lmax);
}
}

void MLMTPR::PrepareAdaptiveRadial(const Neighborhoods& neighborhoods,
                                  bool derivatives, bool parameters) {
    AdaptiveRadialExecution& ex = adaptive_radial_execution_;
    const bool rd_m0=RDM0Angular::Enabled();
    if (rd_m0) ReportRDM0Angular(sh_l_max_);
    else if (RDM0Angular::SelectedBackend()!=RDM0Angular::Backend::RY)
        ReportPhysicalAngular(sh_l_max_);
    bool reuse=parameters && ex.reusable_full_cache && two_layer_edge_cache_ready_
        && two_layer_edge_cache_has_param_derivatives_
        && ex.cached_scaling==scaling && ex.cached_coefficients==regression_coeffs
        && two_layer_edge_cache_atom_count_==neighborhoods.size();
    if (reuse && rd_m0) {
        // A copied/reused physical-Y cache alone does not supply raw D.
        const size_t raw_count=two_layer_edge_offsets_cache_.empty() ? 0
            : two_layer_edge_offsets_cache_.back()
                *static_cast<size_t>((sh_l_max_+1)*(sh_l_max_+1));
        reuse=!two_layer_edge_offsets_cache_.empty()
            && two_layer_edge_rd_m0_values_cache_.size()==raw_count
            && (!derivatives || two_layer_edge_rd_m0_ders_cache_.size()==3*raw_count);
    }
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
    ex.projection_rows = AdaptiveRadialProjectionRowCount();
    ex.scalars = TwoLayerGateScalarCount();
    ex.harmonics = (sh_l_max_+1)*(sh_l_max_+1);
    ex.radial_stride = static_cast<int>(radial_eval_to_scaling_block_.size())*p_RadialBasis->rb_size;
    const int atoms = neighborhoods.size(), D=ex.rows, Q=ex.scalars, P=ex.projection_rows;
    const int R=p_RadialBasis->rb_size, B=(sh_l_max_+1)*outer_k_rank_, T=AdaptiveRadialLatentCount();
    if (two_layer_gate_scalar_values_cache_.size()!=static_cast<size_t>(atoms)*Q)
        ERROR("adaptive radial gate scalar cache has inconsistent size");
    ex.signals.resize(static_cast<size_t>(atoms)*P);
    ex.coefficients.resize(static_cast<size_t>(atoms)*D);
    ex.amplitude_offsets.resize(atoms);
    for (int i=0; i<atoms; ++i) {
        if (neighborhoods[i].my_type<0 || neighborhoods[i].my_type>=species_count)
            ERROR("adaptive radial source species index is out of range");
        ex.amplitude_offsets[i]=AdaptiveRadialSpeciesAmplitude() ? neighborhoods[i].my_type*D*T : 0;
    }
    if (atoms>0)
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,atoms,P,Q,1.0,
                    two_layer_gate_scalar_values_cache_.data(),Q,
                    regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,
                    0.0,ex.signals.data(),P);
    // Store phi(h), not a broadcast [L,O,N,Q] projection. tanh derivatives
    // are recovered exactly from this compact cache: phi'=1-x*x.
    if (AdaptiveRadialTanh())
        for (double& x : ex.signals) x=std::tanh(x);
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    for (int i=0; i<atoms; ++i)
        for (int l=0; l<=sh_l_max_; ++l)
            for (int o=0; o<outer_k_rank_; ++o) {
                const int row=(l*outer_k_rank_+o)*R;
                const int mu=o*(sh_l_max_+1)+l;
                const double* c=regression_coeffs.data()+RadialCoeffOffset()+mu*RadialCoeffBlockSize();
                for (int n=0; n<R; ++n) {
                    const size_t at=static_cast<size_t>(i)*D+row+n;
                    const int projection_row=AdaptiveRadialSharedN() ? (l*outer_k_rank_+o)*T : row+n;
                    double correction=0.0;
                    for (int t=0;t<T;++t)
                        correction+=amplitude[ex.amplitude_offsets[i]+(row+n)*T+t]
                            *ex.signals[static_cast<size_t>(i)*P+projection_row+t];
                    ex.coefficients[at]=c[n]+correction;
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
    if (rd_m0
        && (two_layer_edge_rd_m0_values_cache_.size()!=edges*ex.harmonics
            || (derivatives
                && two_layer_edge_rd_m0_ders_cache_.size()!=edges*3*ex.harmonics)))
        ERROR("adaptive RD_m0 angular cache has inconsistent size");
    ex.radial_values.resize(edges*B);
    if (derivatives) ex.radial_derivatives.resize(edges*B);
    if (parameters) ex.radial_tangents.resize(edges*B);
    ex.coefficient_adjoints.assign(static_cast<size_t>(atoms)*D,0.0);
    ex.coefficient_tangents.clear();
    if (TensorMessageEnabled()) {
        const int TC = TensorMessageComponentCount();
        const int G = two_layer_gate_k_rank_;
        const int Lmsg = tensor_message_lmax_;
        const int cached_moments =
            static_cast<int>(two_layer_gate_required_moment_indices_.size());
        if (static_cast<int>(tensor_message_gate_moment_dense_indices_.size())
            != G * Lmsg * ((Lmsg + 1) * 2 + 1))
            ERROR("SUS2-SH tensor message dense gate moment map is inconsistent");
        if (two_layer_gate_moment_values_cache_.size()
            != static_cast<size_t>(atoms) * cached_moments)
            ERROR("SUS2-SH tensor message requires cached gate moment values");
        ex.tensor_messages.assign(static_cast<size_t>(atoms) * TC, 0.0);
        const double* Bmsg = regression_coeffs.data() + TensorMessageCoeffOffset();
        for (int atom = 0; atom < atoms; ++atom) {
            const double* z =
                two_layer_gate_moment_values_cache_.data()
                + static_cast<size_t>(atom) * cached_moments;
            double* hmsg = ex.tensor_messages.data()
                + static_cast<size_t>(atom) * TC;
            for (int L = 1; L <= Lmsg; ++L)
                for (int o = 0; o < outer_k_rank_; ++o)
                    for (int m = -L; m <= L; ++m) {
                        double value = 0.0;
                        for (int g = 0; g < G; ++g) {
                            const int dense =
                                tensor_message_gate_moment_dense_indices_[
                                    TensorGateDenseLocal(g, L, m, Lmsg)];
                            if (dense < 0 || dense >= cached_moments)
                                ERROR("SUS2-SH tensor message gate moment cache index is invalid");
                            const size_t b =
                                ((static_cast<size_t>(L - 1) * outer_k_rank_ + o)
                                 * G) + g;
                            value += Bmsg[b] * z[dense];
                        }
                        hmsg[TensorComponentLocal(L, o, m, outer_k_rank_)] =
                            value;
                    }
        }
        if (derivatives)
            ex.tensor_message_adjoints.assign(static_cast<size_t>(atoms) * TC, 0.0);
        else
            ex.tensor_message_adjoints.clear();
    } else {
        ex.tensor_messages.clear();
        ex.tensor_message_adjoints.clear();
    }
    if (JointReadoutEnabled()) {
        const int count = joint_readout_graph_.moments_count;
        const int cached = static_cast<int>(two_layer_gate_required_moment_indices_.size());
        if (two_layer_gate_moment_values_cache_.size() != static_cast<size_t>(atoms)*cached
            || joint_readout_gate_dense_indices_.size() != static_cast<size_t>(two_layer_gate_k_rank_)*ex.harmonics)
            ERROR("SUS2-SH joint readout requires all post-rank gate primitive values");
        ex.joint_values.resize(count); ex.joint_adjoints.resize(count);
        if (parameters) { ex.joint_tangents.resize(count); ex.joint_tangent_adjoints.resize(count); }
        if (derivatives) ex.joint_gate_adjoints.assign(static_cast<size_t>(atoms)*two_layer_gate_k_rank_*ex.harmonics, 0.0);
        else ex.joint_gate_adjoints.clear();
        if (parameters) ex.joint_gate_tangent_adjoints.assign(static_cast<size_t>(atoms)*two_layer_gate_k_rank_*ex.harmonics, 0.0);
        else ex.joint_gate_tangent_adjoints.clear();
    }
}

void MLMTPR::ForwardAdaptiveRadialSite(const Neighborhood& nbh, int atom,
                                     const std::vector<Vector3>* direction) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int H=ex.harmonics,D=ex.rows,R=p_RadialBasis->rb_size;
    const int B=(sh_l_max_+1)*outer_k_rank_;
    const bool tensor_enabled=TensorMessageEnabled();
    const int TC=tensor_enabled ? TensorMessageComponentCount() : 0;
    const int Lmsg=tensor_enabled ? tensor_message_lmax_ : 0;
    const bool tangent=direction!=nullptr;
    const bool rd_m0=RDM0Angular::Enabled();
    const std::vector<double>& angular_values=rd_m0
        ? two_layer_edge_rd_m0_values_cache_ : two_layer_edge_sh_values_cache_;
    const std::vector<double>& angular_ders=rd_m0
        ? two_layer_edge_rd_m0_ders_cache_ : two_layer_edge_sh_ders_cache_;
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
        const double* y=angular_values.data()+edge*H;
        const double* dy=ex.derivatives ? angular_ders.data()+edge*3*H : nullptr;
        const double* rb=two_layer_edge_radial_vals_cache_.data()+edge*ex.radial_stride;
        const double* dr=ex.derivatives ? two_layer_edge_radial_ders_cache_.data()+edge*radial_der_stride : nullptr;
        const double* c=ex.coefficients.data()+static_cast<size_t>(nbh.inds[j])*D;
        const double* ct=tangent ? ex.coefficient_tangents.data()+static_cast<size_t>(nbh.inds[j])*D : nullptr;
        const double* hmsg=tensor_enabled
            ? ex.tensor_messages.data()+static_cast<size_t>(nbh.inds[j])*TC
            : nullptr;
        const double* htmsg=(tensor_enabled && tangent
            && !ex.tensor_message_tangents.empty())
            ? ex.tensor_message_tangents.data()+static_cast<size_t>(nbh.inds[j])*TC
            : nullptr;
        const double wr=tangent ? ((*direction)[j][0]*nbh.vecs[j][0]+(*direction)[j][1]*nbh.vecs[j][1]+(*direction)[j][2]*nbh.vecs[j][2])/nbh.dists[j] : 0.0;
        for (int l=0; l<=sh_l_max_; ++l) {
            const double pole=(rd_m0 && tensor_enabled && l>=1 && l<=Lmsg)
                ? RDM0Angular::PoleValue(l) : 1.0;
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
                    double angular=y[h];
                    double angular_t=0.0;
                    if (tensor_enabled && l>=1 && l<=Lmsg) {
                        const int tc=TensorComponentLocal(l,o,m,outer_k_rank_);
                        if (rd_m0) {
                            // H is physical: p_l*(D + H/p_l) = Y + H.
                            // Raw D itself is evaluated independently, never Y/p_l.
                            angular+=hmsg[tc]/pole;
                            if (htmsg!=nullptr) angular_t+=htmsg[tc]/pole;
                        } else {
                            angular+=hmsg[tc];
                            if (htmsg!=nullptr) angular_t+=htmsg[tc];
                        }
                    }
                    ex.values[node]+=species*value*angular;
                    if (tangent) {
                        double yt=angular_t;
                        for (int a=0; a<3; ++a) yt+=dy[3*h+a]*(*direction)[j][a];
                        ex.tangents[node]+=species*(value_t*angular+value*yt);
                    }
                }
            }
        }
    }
    if (rd_m0) {
        // Restore physical primitive values and their directional tangents
        // after the neighbor reduction, before any CG products/joint readout.
        // All reverse consumers retain physical Y/dY and need no extra scale.
        for (int l=0; l<=sh_l_max_; ++l) {
            const double pole=RDM0Angular::PoleValue(l);
            for (int o=0; o<outer_k_rank_; ++o)
                for (int m=-l; m<=l; ++m) {
                    const int node=ex.outer_nodes[o*H+l*l+l+m];
                    ex.values[node]*=pole;
                    if (tangent) ex.tangents[node]*=pole;
                }
        }
    }
    for (size_t p=static_cast<size_t>(OuterKRank1ProductStart());p<sh_products_.size();++p) {
        const SHProduct& t=sh_products_[p];
        ex.values[t.target]+=t.coeff*ex.values[t.left]*ex.values[t.right];
        if (tangent) ex.tangents[t.target]+=t.coeff*(ex.tangents[t.left]*ex.values[t.right]+ex.values[t.left]*ex.tangents[t.right]);
    }
    if (JointReadoutEnabled()) ForwardJointReadout(atom,tangent);
}

void MLMTPR::ForwardJointReadout(int atom, bool tangent) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int H=ex.harmonics,G=two_layer_gate_k_rank_,O=outer_k_rank_;
    const int cached=static_cast<int>(two_layer_gate_required_moment_indices_.size());
    const double* z=two_layer_gate_moment_values_cache_.data()+static_cast<size_t>(atom)*cached;
    const double* zt=tangent ? ex.gate_moment_tangents.data()+static_cast<size_t>(atom)*alpha_moments_count : nullptr;
    std::fill(ex.joint_values.begin(),ex.joint_values.end(),0.0);
    if (tangent) std::fill(ex.joint_tangents.begin(),ex.joint_tangents.end(),0.0);
    for (size_t i=0;i<joint_readout_graph_.basic.size();++i) {
        const auto& b=joint_readout_graph_.basic[i];
        const int h=b[1]*b[1]+b[1]+b[2];
        double v=0.0,t=0.0;
        if (b[0]==0) {
            for (int o=0;o<O;++o) {
                const int node=ex.outer_nodes[o*H+h];
                v+=ex.values[node]; if(tangent) t+=ex.tangents[node];
            }
            v/=O; t/=O;
        } else {
            for (int g=0;g<G;++g) {
                const int dense=joint_readout_gate_dense_indices_[g*H+h];
                v+=z[dense]; if(tangent) t+=zt[two_layer_gate_required_moment_indices_[dense]];
            }
            v/=G; t/=G;
        }
        ex.joint_values[i]=v; if(tangent) ex.joint_tangents[i]=t;
    }
    for (const auto& p:joint_readout_graph_.products) {
        ex.joint_values[p.target]+=p.coeff*ex.joint_values[p.left]*ex.joint_values[p.right];
        if(tangent) ex.joint_tangents[p.target]+=p.coeff*(ex.joint_tangents[p.left]*ex.joint_values[p.right]+ex.joint_values[p.left]*ex.joint_tangents[p.right]);
    }
}

void MLMTPR::AccumulateJointReadoutScalarSquares(std::vector<double>& sums) const {
    if (!JointReadoutEnabled()) return;
    if (sums.empty()) sums.assign(JointReadoutScalarCount(),0.0);
    if (static_cast<int>(sums.size())!=JointReadoutScalarCount()) ERROR("Joint readout RMS size mismatch");
    for (int q=0;q<JointReadoutScalarCount();++q) {
        const double value=adaptive_radial_execution_.joint_values[joint_readout_graph_.scalar_mapping[q]];
        if(!std::isfinite(value) || !std::isfinite(value*value) || !std::isfinite(sums[q]+value*value))
            ERROR("SUS2-SH joint readout fixed-RMS calibration encountered a non-finite feature square");
        sums[q]+=value*value;
    }
}

void MLMTPR::ReverseJointReadout(int atom, bool tangent) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    for (int i=static_cast<int>(joint_readout_graph_.products.size())-1;i>=0;--i) {
        const auto& p=joint_readout_graph_.products[i];
        const double a=p.coeff*ex.joint_adjoints[p.target];
        const double at=tangent ? p.coeff*ex.joint_tangent_adjoints[p.target] : 0.0;
        ex.joint_adjoints[p.left]+=a*ex.joint_values[p.right]+(tangent ? at*ex.joint_tangents[p.right] : 0.0);
        ex.joint_adjoints[p.right]+=a*ex.joint_values[p.left]+(tangent ? at*ex.joint_tangents[p.left] : 0.0);
        if(tangent) {
            ex.joint_tangent_adjoints[p.left]+=at*ex.joint_values[p.right];
            ex.joint_tangent_adjoints[p.right]+=at*ex.joint_values[p.left];
        }
    }
    const int H=ex.harmonics,G=two_layer_gate_k_rank_,O=outer_k_rank_;
    const size_t start=static_cast<size_t>(atom)*G*H;
    for(size_t i=0;i<joint_readout_graph_.basic.size();++i) {
        const auto& b=joint_readout_graph_.basic[i]; const int h=b[1]*b[1]+b[1]+b[2];
        const double inv=1.0/(b[0]==0 ? O : G);
        const double a=ex.joint_adjoints[i]*inv;
        const double at=tangent ? ex.joint_tangent_adjoints[i]*inv : 0.0;
        if(b[0]==0) for(int o=0;o<O;++o) {
            const int node=ex.outer_nodes[o*H+h]; ex.adjoints[node]+=a;
            if(tangent) ex.tangent_adjoints[node]+=at;
        } else for(int g=0;g<G;++g) {
            ex.joint_gate_adjoints[start+g*H+h]+=a;
            if(tangent) ex.joint_gate_tangent_adjoints[start+g*H+h]+=at;
        }
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
    const bool tensor_enabled=TensorMessageEnabled();
    const int TC=tensor_enabled ? TensorMessageComponentCount() : 0;
    const int Lmsg=tensor_enabled ? tensor_message_lmax_ : 0;
    for (int j=0;j<nbh.count;++j) {
        const size_t edge=two_layer_edge_offsets_cache_[atom]+j;
        const double* y=two_layer_edge_sh_values_cache_.data()+edge*H;
        const double* dy=two_layer_edge_sh_ders_cache_.data()+edge*3*H;
        const double* rb=two_layer_edge_radial_vals_cache_.data()+edge*ex.radial_stride;
        double* ca=ex.coefficient_adjoints.data()+static_cast<size_t>(nbh.inds[j])*D;
        const double* hmsg=tensor_enabled
            ? ex.tensor_messages.data()+static_cast<size_t>(nbh.inds[j])*TC
            : nullptr;
        double* ha=tensor_enabled
            ? ex.tensor_message_adjoints.data()+static_cast<size_t>(nbh.inds[j])*TC
            : nullptr;
        Vector3 edge_der(0,0,0);
        for (int l=0;l<=sh_l_max_;++l)
            for (int o=0;o<outer_k_rank_;++o) {
                const int block=l*outer_k_rank_+o, row=block*R,mu=o*(sh_l_max_+1)+l;
                const double species=OuterTypeCoeff(nbh.my_type,mu)*OuterTypeCoeff(nbh.types[j],mu);
                double radial_adjoint=0;
                Vector3 angular_adjoint(0,0,0);
                for (int m=-l;m<=l;++m) {
                    const int h=l*l+l+m;
                    const double adj=ex.adjoints[ex.outer_nodes[o*H+h]];
                    double angular=y[h];
                    if (tensor_enabled && l>=1 && l<=Lmsg)
                        angular+=hmsg[TensorComponentLocal(l,o,m,outer_k_rank_)];
                    radial_adjoint+=adj*angular;
                    for (int a=0;a<3;++a) angular_adjoint[a]+=adj*dy[3*h+a];
                    if (tensor_enabled && l>=1 && l<=Lmsg && ha!=nullptr)
                        ha[TensorComponentLocal(l,o,m,outer_k_rank_)]
                            += species * adj * ex.radial_values[edge*B+block];
                }
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

void MLMTPR::ReduceAdaptiveRadialCoefficientAdjoints(bool tangent) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int D=ex.rows,P=ex.projection_rows,R=p_RadialBasis->rb_size,T=AdaptiveRadialLatentCount();
    const int atoms=static_cast<int>(ex.amplitude_offsets.size());
    const double* a=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    if (!AdaptiveRadialSharedN() && !AdaptiveRadialTanh()) {
        // Keep the original full/linear in-place pullback: no extra buffer,
        // zero-fill or copy is needed for already uncompressed rows.
        for (int i=0;i<atoms;++i) for (int row=0;row<D;++row) {
            const size_t at=static_cast<size_t>(i)*D+row;
            const double amp=a[ex.amplitude_offsets[i]+row];
            ex.coefficient_adjoints[at]*=amp;
            if (tangent) ex.coefficient_tangent_adjoints[at]*=amp;
        }
        return;
    }
    ex.signal_adjoints.assign(static_cast<size_t>(atoms)*P,0.0);
    if (tangent) ex.signal_tangent_adjoints.assign(static_cast<size_t>(atoms)*P,0.0);
    for (int i=0;i<atoms;++i) for (int row=0;row<D;++row) {
        const size_t at=static_cast<size_t>(i)*D+row;
        const int pr=AdaptiveRadialSharedN() ? (row/R)*T : row;
        for (int t=0;t<T;++t) {
            const double amp=a[ex.amplitude_offsets[i]+row*T+t];
            const size_t to=static_cast<size_t>(i)*P+pr+t;
            ex.signal_adjoints[to]+=amp*ex.coefficient_adjoints[at];
            if (tangent) ex.signal_tangent_adjoints[to]+=amp*ex.coefficient_tangent_adjoints[at];
        }
    }
    if (AdaptiveRadialTanh()) for (size_t at=0;at<ex.signal_adjoints.size();++at) {
        const double x=ex.signals[at], first=1.0-x*x;
        // Reverse over directional differentiation: bar_h includes phi''.
        // signal_tangents remains raw dot(h), not dot(phi(h)).
        ex.signal_adjoints[at]*=first;
        if (tangent) {
            ex.signal_adjoints[at]+=(-2.0*x*first)*ex.signal_tangents[at]*ex.signal_tangent_adjoints[at];
            ex.signal_tangent_adjoints[at]*=first;
        }
    }
}

void MLMTPR::FinishAdaptiveRadialGateForces(Configuration& cfg,const Neighborhoods& neighborhoods,
                                          int component,bool need_forces,bool need_stress) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int atoms=neighborhoods.size(),P=ex.projection_rows,Q=ex.scalars;
    const bool tensor_enabled=TensorMessageEnabled();
    const int TC=tensor_enabled ? TensorMessageComponentCount() : 0;
    const int G=two_layer_gate_k_rank_;
    const int Lmsg=tensor_message_lmax_;
    const double* Bmsg=tensor_enabled
        ? regression_coeffs.data()+TensorMessageCoeffOffset() : nullptr;
    ReduceAdaptiveRadialCoefficientAdjoints(false);
    const double* seeds=(!AdaptiveRadialSharedN() && !AdaptiveRadialTanh())
        ? ex.coefficient_adjoints.data() : ex.signal_adjoints.data();
    ex.scalar_adjoints.resize(static_cast<size_t>(atoms)*Q);
    if (atoms>0)
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,P,1.0,
                    seeds,P,regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,
                    0.0,ex.scalar_adjoints.data(),Q);
    if (tensor_enabled || JointReadoutEnabled())
        ex.tensor_gate_moment_adjoints.resize(alpha_moments_count);
    else
        ex.tensor_gate_moment_adjoints.clear();
    std::vector<Vector3> derivatives;
    for (int i=0;i<atoms;++i) {
        const Neighborhood& nbh=neighborhoods[i];
        if (tensor_enabled || JointReadoutEnabled()) {
            std::fill(ex.tensor_gate_moment_adjoints.begin(),ex.tensor_gate_moment_adjoints.end(),0.0);
        }
        if (JointReadoutEnabled()) {
            const size_t start=static_cast<size_t>(i)*G*ex.harmonics;
            for(int g=0;g<G;++g) for(int h=1;h<ex.harmonics;++h) {
                const int dense=joint_readout_gate_dense_indices_[g*ex.harmonics+h];
                ex.tensor_gate_moment_adjoints[two_layer_gate_required_moment_indices_[dense]]+=ex.joint_gate_adjoints[start+g*ex.harmonics+h];
            }
        }
        if (tensor_enabled) {
            const double* ha=ex.tensor_message_adjoints.data()
                + static_cast<size_t>(i)*TC;
            double* raw=ex.tensor_gate_moment_adjoints.data();
            for (int L=1; L<=Lmsg; ++L)
                for (int o=0; o<outer_k_rank_; ++o)
                    for (int m=-L; m<=L; ++m) {
                        const double adj=ha[TensorComponentLocal(L,o,m,outer_k_rank_)];
                        if (adj==0.0) continue;
                        for (int g=0; g<G; ++g) {
                            const int dense=tensor_message_gate_moment_dense_indices_[
                                TensorGateDenseLocal(g,L,m,Lmsg)];
                            const int moment=two_layer_gate_required_moment_indices_[dense];
                            const size_t b=
                                ((static_cast<size_t>(L-1)*outer_k_rank_+o)*G)+g;
                            raw[moment]+=Bmsg[b]*adj;
                        }
                    }
        }
        CalcTwoLayerGateWeightedScalarDersForScalarSeeds(
            nbh,derivatives,i,
            ex.scalar_adjoints.data()+static_cast<size_t>(i)*Q,
            (tensor_enabled || JointReadoutEnabled())
                ? ex.tensor_gate_moment_adjoints.data()
                : nullptr);
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
    const bool species_readout=UsesSpeciesReadout();
	    for (int i=0;i<cfg.size();++i) {
	        ForwardAdaptiveRadialSite(neighborhoods[i],i,nullptr);
	        const int type=neighborhoods[i].my_type;
	        const double center=linear_coeffs[type];
	        const int readout_offset=LinearCoeffOffset()+species_count
	            +(species_readout ? type*alpha_scalar_moments : 0);
	        const double* readout=regression_coeffs.data()+readout_offset;
	        double scalar_energy=0;
	        if (need_derivatives) std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
	        for (int q=0;q<alpha_scalar_moments;++q) {
	            const int node=alpha_moment_mapping[q];
	            const double coeff=readout[q];
	            const double weight=(species_readout ? coeff : center*coeff)*linear_mults[q];
	            scalar_energy+=weight*ex.values[node];
	            if (need_derivatives) ex.adjoints[node]+=weight;
            max_linear[q]=std::max(max_linear[q],std::abs(coeff*ex.values[node]));
        }
        if (JointReadoutEnabled()) {
            if (need_derivatives) std::fill(ex.joint_adjoints.begin(),ex.joint_adjoints.end(),0.0);
            for (int q=0;q<JointReadoutScalarCount();++q) {
                const int node=joint_readout_graph_.scalar_mapping[q];
                const double weight=(species_readout ? 1.0 : center)*regression_coeffs[JointReadoutCoeffIndex(type,q)];
                scalar_energy+=weight*ex.joint_values[node];
                if(need_derivatives) ex.joint_adjoints[node]+=weight;
            }
        }
        cfg.cal_se0[i]=scalar_energy;
        cfg.cal_se[i]=regression_coeffs[type]+center+scalar_energy;
        cfg.energy+=cfg.cal_se[i];
        if (need_derivatives) {
            if(JointReadoutEnabled()) ReverseJointReadout(i,false);
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
                                         bool need_forces,bool need_stress,
                                         bool fixed_center_readout_columns) {
    const bool derivatives=need_forces||need_stress;
    PrepareAdaptiveRadial(neighborhoods,derivatives,false);
    const int columns=LinearEquationCount();
    std::fill(energy_cmpnts,energy_cmpnts+columns,0.0);
    if (need_forces) { forces_cmpnts.resize(cfg.size(),columns,3); forces_cmpnts.set(0); }
    if (need_stress) std::memset(stress_cmpnts,0,columns*sizeof(stress_cmpnts[0]));
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const bool species_readout=UsesSpeciesReadout();
    // TrainLinear requests raw Phi columns and subsequently fixes its center
    // gauge. A physical readout diagnostic may instead hold unequal h_Z fixed
    // and request sum_i h_Zi Phi_i columns. Species readout has no such factor.
    const bool include_center=fixed_center_readout_columns && !species_readout;
    const int readout_count=ScalarReadoutCoeffCount();
    ex.site_values.resize(static_cast<size_t>(cfg.size())*alpha_moments_count);
    if(JointReadoutEnabled()) ex.joint_site_values.resize(static_cast<size_t>(cfg.size())*joint_readout_graph_.moments_count);
    for (int i=0;i<cfg.size();++i) {
        ForwardAdaptiveRadialSite(neighborhoods[i],i,nullptr);
        std::copy(ex.values.begin(),ex.values.end(),ex.site_values.data()+static_cast<size_t>(i)*alpha_moments_count);
        if(JointReadoutEnabled()) std::copy(ex.joint_values.begin(),ex.joint_values.end(),
            ex.joint_site_values.data()+static_cast<size_t>(i)*joint_readout_graph_.moments_count);
        const int type=neighborhoods[i].my_type;
        const double center=include_center ? linear_coeffs[type] : 1.0;
        energy_cmpnts[type]+=1.0;
        for (int q=0;q<alpha_scalar_moments;++q) {
            const int column=species_count+(species_readout ? type*alpha_scalar_moments+q : q);
            energy_cmpnts[column]+=center*ex.values[alpha_moment_mapping[q]];
        }
        for(int q=0;q<JointReadoutScalarCount();++q) {
            const int column=JointReadoutLinearColumnOffset()+(species_readout ? type*JointReadoutScalarCount() : 0)+q;
            energy_cmpnts[column]+=center*ex.joint_values[joint_readout_graph_.scalar_mapping[q]];
        }
    }
    if (!derivatives) return;
    const bool reference=ReferenceComponents();
    const int D=ex.rows,P=alpha_scalar_moments;
    const int TC=TensorMessageEnabled() ? TensorMessageComponentCount() : 0;
    if (!reference) {
        ex.component_coefficient_adjoints.resize(static_cast<size_t>(cfg.size())*readout_count*D);
        if (TensorMessageEnabled())
            ex.component_tensor_message_adjoints.assign(
                static_cast<size_t>(cfg.size())*readout_count*TC,0.0);
        else
            ex.component_tensor_message_adjoints.clear();
    }
    // Default linear-solver columns are raw scalar bases. In the optional
    // fixed-center chart h_Z is held fixed and scales each central atom's
    // complete direct and neighbor-environment derivative, including H.
    // Reuse graph values and radial contractions for all columns.
    for (int head=0;head<readout_count;++head) {
        const int type_filter=species_readout ? head/P : -1;
        const int q=species_readout ? head%P : head;
        std::fill(ex.coefficient_adjoints.begin(),ex.coefficient_adjoints.end(),0.0);
        if (TensorMessageEnabled())
            std::fill(ex.tensor_message_adjoints.begin(),
                      ex.tensor_message_adjoints.end(),0.0);
        for (int i=0;i<cfg.size();++i) {
            if (type_filter >= 0 && neighborhoods[i].my_type != type_filter)
                continue;
            std::copy(ex.site_values.data()+static_cast<size_t>(i)*alpha_moments_count,
                      ex.site_values.data()+static_cast<size_t>(i+1)*alpha_moments_count,ex.values.begin());
            std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
            ex.adjoints[alpha_moment_mapping[q]]=include_center
                ? linear_coeffs[neighborhoods[i].my_type] : 1.0;
            ReverseAdaptiveRadialGraph(false);
            AccumulateAdaptiveRadialEdgeForces(cfg,neighborhoods[i],i,species_count+head,need_forces,need_stress);
        }
        if (reference)
            FinishAdaptiveRadialGateForces(cfg,neighborhoods,species_count+head,need_forces,need_stress);
        else for (int i=0;i<cfg.size();++i) {
            std::copy(ex.coefficient_adjoints.data()+static_cast<size_t>(i)*D,
                      ex.coefficient_adjoints.data()+static_cast<size_t>(i+1)*D,
                      ex.component_coefficient_adjoints.data()+(static_cast<size_t>(i)*readout_count+head)*D);
            if (TensorMessageEnabled())
                std::copy(ex.tensor_message_adjoints.data()+static_cast<size_t>(i)*TC,
                          ex.tensor_message_adjoints.data()+static_cast<size_t>(i+1)*TC,
                          ex.component_tensor_message_adjoints.data()
                              +(static_cast<size_t>(i)*readout_count+head)*TC);
        }
    }
    if (!reference) FinishAdaptiveRadialGateComponents(cfg,neighborhoods,need_forces,need_stress);
    // Auxiliary columns share the cached physical edge contractions. Their
    // direct gate-primitive seed is kept independent of the main scalar GEMM.
    const int JP=JointReadoutScalarCount(),JN=joint_readout_graph_.moments_count;
    for(int head=0;head<JointReadoutCoeffCount();++head) {
        const int type_filter=species_readout ? head/JP : -1,q=head%JP;
        const int column=JointReadoutLinearColumnOffset()+head;
        std::fill(ex.coefficient_adjoints.begin(),ex.coefficient_adjoints.end(),0.0);
        std::fill(ex.tensor_message_adjoints.begin(),ex.tensor_message_adjoints.end(),0.0);
        std::fill(ex.joint_gate_adjoints.begin(),ex.joint_gate_adjoints.end(),0.0);
        for(int i=0;i<cfg.size();++i) {
            if(type_filter>=0 && neighborhoods[i].my_type!=type_filter) continue;
            std::copy(ex.site_values.data()+static_cast<size_t>(i)*alpha_moments_count,
                      ex.site_values.data()+static_cast<size_t>(i+1)*alpha_moments_count,ex.values.begin());
            std::copy(ex.joint_site_values.data()+static_cast<size_t>(i)*JN,
                      ex.joint_site_values.data()+static_cast<size_t>(i+1)*JN,ex.joint_values.begin());
            std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
            std::fill(ex.joint_adjoints.begin(),ex.joint_adjoints.end(),0.0);
            ex.joint_adjoints[joint_readout_graph_.scalar_mapping[q]]=include_center ? linear_coeffs[neighborhoods[i].my_type] : 1.0;
            ReverseJointReadout(i,false);
            ReverseAdaptiveRadialGraph(false);
            AccumulateAdaptiveRadialEdgeForces(cfg,neighborhoods[i],i,column,need_forces,need_stress);
        }
        FinishAdaptiveRadialGateForces(cfg,neighborhoods,column,need_forces,need_stress);
    }
}

void MLMTPR::FinishAdaptiveRadialGateComponents(Configuration& cfg,const Neighborhoods& neighborhoods,
                                                bool need_forces,bool need_stress) {
    AdaptiveRadialExecution& ex=adaptive_radial_execution_;
    const int P=ScalarReadoutCoeffCount(),D=ex.rows,Q=ex.scalars,H=ex.harmonics;
    const int projection_rows=ex.projection_rows,R=p_RadialBasis->rb_size,T=AdaptiveRadialLatentCount();
    const int TC=TensorMessageEnabled() ? TensorMessageComponentCount() : 0;
    const int G=two_layer_gate_k_rank_;
    const int Lmsg=TensorMessageEnabled() ? tensor_message_lmax_ : 0;
    const int basic_count=static_cast<int>(two_layer_gate_required_basic_indices_.size());
    const int gate_mu_count=static_cast<int>(two_layer_gate_required_mu_indices_.size());
    const int cached_moments=static_cast<int>(two_layer_gate_required_moment_indices_.size());
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    std::vector<double> scalar_seeds(static_cast<size_t>(P)*Q);
    const bool direct_linear=!AdaptiveRadialSharedN() && !AdaptiveRadialTanh();
    std::vector<double> compact_seeds(direct_linear ? 0 : static_cast<size_t>(P)*projection_rows);
    std::vector<double> adj(static_cast<size_t>(alpha_moments_count)*P);
    std::vector<double> gate_values(alpha_moments_count,0.0);
    std::vector<double> basic_seeds(static_cast<size_t>(P)*basic_count),jacobian,derivatives;
    for (int atom=0;atom<cfg.size();++atom) {
        const Neighborhood& nbh=neighborhoods[atom];
        const int coords=3*nbh.count;
        if (coords==0) continue;
        double* seeds=ex.component_coefficient_adjoints.data()+static_cast<size_t>(atom)*P*D;
        if (direct_linear) {
            for (int p=0;p<P;++p) for (int row=0;row<D;++row)
                seeds[p*D+row]*=amplitude[ex.amplitude_offsets[atom]+row];
        } else {
        std::fill(compact_seeds.begin(),compact_seeds.end(),0.0);
        for (int p=0;p<P;++p)
            for (int row=0;row<D;++row) {
                const int pr=AdaptiveRadialSharedN() ? (row/R)*T : row;
                for (int t=0;t<T;++t)
                    compact_seeds[p*projection_rows+pr+t]+=seeds[p*D+row]
                        *amplitude[ex.amplitude_offsets[atom]+row*T+t];
            }
        if (AdaptiveRadialTanh()) for (int p=0;p<P;++p) for (int pr=0;pr<projection_rows;++pr) {
            const double x=ex.signals[static_cast<size_t>(atom)*projection_rows+pr];
            compact_seeds[p*projection_rows+pr]*=1.0-x*x;
        }
        }
        const double* projected=direct_linear ? seeds : compact_seeds.data();
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,P,Q,projection_rows,1.0,projected,projection_rows,
                    regression_coeffs.data()+AdaptiveRadialWeightOffset(),Q,0.0,scalar_seeds.data(),Q);
        std::fill(adj.begin(),adj.end(),0.0);
        for (int q=0;q<Q;++q) {
            double* target=adj.data()+static_cast<size_t>(TwoLayerGateScalarMomentIndexUnchecked(q))*P;
            for (int p=0;p<P;++p) target[p]+=scalar_seeds[p*Q+q];
        }
        if (TensorMessageEnabled()) {
            const double* Bmsg=regression_coeffs.data()+TensorMessageCoeffOffset();
            const double* heads=ex.component_tensor_message_adjoints.data()
                + static_cast<size_t>(atom)*P*TC;
            for (int L=1; L<=Lmsg; ++L)
                for (int o=0; o<outer_k_rank_; ++o)
                    for (int m=-L; m<=L; ++m) {
                        const int tc=TensorComponentLocal(L,o,m,outer_k_rank_);
                        for (int g=0; g<G; ++g) {
                            const int dense=tensor_message_gate_moment_dense_indices_[
                                TensorGateDenseLocal(g,L,m,Lmsg)];
                            const int moment=two_layer_gate_required_moment_indices_[dense];
                            double* target=adj.data()+static_cast<size_t>(moment)*P;
                            const size_t b=
                                ((static_cast<size_t>(L-1)*outer_k_rank_+o)*G)+g;
                            const double coeff=Bmsg[b];
                            if (coeff==0.0) continue;
                            for (int p=0; p<P; ++p)
                                target[p]+=coeff*heads[p*TC+tc];
                        }
                    }
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
    const int B=(sh_l_max_+1)*outer_k_rank_, T=AdaptiveRadialLatentCount();
    const bool tensor_enabled=TensorMessageEnabled();
    const int TC=tensor_enabled ? TensorMessageComponentCount() : 0;
    const int G=tensor_enabled ? two_layer_gate_k_rank_ : 0;
    const int Lmsg=tensor_enabled ? tensor_message_lmax_ : 0;
    const bool species_readout=UsesSpeciesReadout();
    const size_t edge_count=two_layer_edge_offsets_cache_.empty()
        ? 0 : two_layer_edge_offsets_cache_.back();
    const int radial_der_stride=AdaptiveRadialDerStride(
        two_layer_edge_radial_ders_cache_.size(), edge_count, ex.radial_stride);
    if (AdaptiveRadialDersAreCompact(two_layer_edge_radial_ders_cache_.size(),
                                     edge_count, ex.radial_stride))
        ERROR("adaptive radial mixed parameter gradient requires full radial derivative cache");
    const double* amplitude=regression_coeffs.data()+AdaptiveRadialAmplitudeOffset();
    const double* weights=regression_coeffs.data()+AdaptiveRadialWeightOffset();
    if (tensor_enabled)
        std::fill(ex.tensor_message_adjoints.begin(),
                  ex.tensor_message_adjoints.end(),0.0);
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
        const int projection_rows=ex.projection_rows;
        ex.signal_tangents.resize(static_cast<size_t>(atoms)*projection_rows);
        ex.coefficient_tangents.resize(static_cast<size_t>(atoms)*D);
        if (atoms>0)
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,atoms,projection_rows,Q,1.0,ex.scalar_tangents.data(),Q,weights,Q,0.0,ex.signal_tangents.data(),projection_rows);
        for (int i=0;i<atoms;++i) for (int row=0;row<D;++row) {
            const int pr=AdaptiveRadialSharedN() ? (row/R)*T : row;
            double value=0.0;
            for (int t=0;t<T;++t) {
                const size_t at=static_cast<size_t>(i)*projection_rows+pr+t;
                const double first=AdaptiveRadialTanh() ? 1.0-ex.signals[at]*ex.signals[at] : 1.0;
                value+=amplitude[ex.amplitude_offsets[i]+row*T+t]*first*ex.signal_tangents[at];
            }
            ex.coefficient_tangents[static_cast<size_t>(i)*D+row]=value;
        }
        ex.coefficient_tangent_adjoints.assign(static_cast<size_t>(atoms)*D,0.0);
        if (tensor_enabled) {
            ex.tensor_message_tangents.assign(static_cast<size_t>(atoms)*TC,0.0);
            ex.tensor_message_tangent_adjoints.assign(static_cast<size_t>(atoms)*TC,0.0);
            const double* Bmsg=regression_coeffs.data()+TensorMessageCoeffOffset();
            for (int atom=0; atom<atoms; ++atom) {
                const double* zt=ex.gate_moment_tangents.data()
                    + static_cast<size_t>(atom)*alpha_moments_count;
                double* ht=ex.tensor_message_tangents.data()
                    + static_cast<size_t>(atom)*TC;
                for (int L=1; L<=Lmsg; ++L)
                    for (int o=0; o<outer_k_rank_; ++o)
                        for (int m=-L; m<=L; ++m) {
                            double value=0.0;
                            for (int g=0; g<G; ++g) {
                                const size_t b=
                                    ((static_cast<size_t>(L-1)*outer_k_rank_+o)*G)+g;
                                const int dense=tensor_message_gate_moment_dense_indices_[
                                    TensorGateDenseLocal(g,L,m,Lmsg)];
                                value+=Bmsg[b]*zt[two_layer_gate_required_moment_indices_[dense]];
                            }
                            ht[TensorComponentLocal(L,o,m,outer_k_rank_)] = value;
                        }
            }
        }
    } else if (tensor_enabled) {
        ex.tensor_message_tangents.clear();
        ex.tensor_message_tangent_adjoints.clear();
    }
    for (int i=0;i<atoms;++i) {
        const Neighborhood& nbh=neighborhoods[i];
        if (tangent) EdgeDirections(nbh,i,frc_weights,str_weights,direction);
        ForwardAdaptiveRadialSite(nbh,i,tangent ? &direction : nullptr);
        std::fill(ex.adjoints.begin(),ex.adjoints.end(),0.0);
        if (tangent) std::fill(ex.tangent_adjoints.begin(),ex.tangent_adjoints.end(),0.0);
	        const int type=nbh.my_type;
	        const double center=linear_coeffs[type];
	        const int readout_offset=LinearCoeffOffset()+species_count
	            +(species_readout ? type*alpha_scalar_moments : 0);
	        const double* readout=regression_coeffs.data()+readout_offset;
	        out[type]+=ene_weight[i];
	        double center_grad=ene_weight[i];
	        for (int q=0;q<alpha_scalar_moments;++q) {
	            const int node=alpha_moment_mapping[q];
	            const double coeff=readout[q];
	            const double head=coeff*linear_mults[q];
	            const double feature=ene_weight[i]*ex.values[node]+(tangent ? ex.tangents[node] : 0.0);
	            if (species_readout) {
	                out[readout_offset+q]+=linear_mults[q]*feature;
	                ex.adjoints[node]+=ene_weight[i]*head;
	                if (tangent) ex.tangent_adjoints[node]+=head;
	            } else {
	                center_grad+=head*feature;
	                out[readout_offset+q]+=center*linear_mults[q]*feature;
	                ex.adjoints[node]+=ene_weight[i]*center*head;
	                if (tangent) ex.tangent_adjoints[node]+=center*head;
            }
        }
        if(JointReadoutEnabled()) {
            std::fill(ex.joint_adjoints.begin(),ex.joint_adjoints.end(),0.0);
            if(tangent) std::fill(ex.joint_tangent_adjoints.begin(),ex.joint_tangent_adjoints.end(),0.0);
            for(int q=0;q<JointReadoutScalarCount();++q) {
                const int node=joint_readout_graph_.scalar_mapping[q],index=JointReadoutCoeffIndex(type,q);
                const double head=regression_coeffs[index];
                const double feature=ene_weight[i]*ex.joint_values[node]+(tangent ? ex.joint_tangents[node] : 0.0);
                const double factor=species_readout ? 1.0 : center;
                out[index]+=factor*feature;
                if(!species_readout) center_grad+=head*feature;
                ex.joint_adjoints[node]+=ene_weight[i]*factor*head;
                if(tangent) ex.joint_tangent_adjoints[node]+=factor*head;
            }
            ReverseJointReadout(i,tangent);
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
            const double* hmsg=tensor_enabled
                ? ex.tensor_messages.data()+static_cast<size_t>(nbh.inds[j])*TC
                : nullptr;
            const double* htmsg=(tensor_enabled && tangent
                && !ex.tensor_message_tangents.empty())
                ? ex.tensor_message_tangents.data()+static_cast<size_t>(nbh.inds[j])*TC
                : nullptr;
            double* ha=tensor_enabled
                ? ex.tensor_message_adjoints.data()+static_cast<size_t>(nbh.inds[j])*TC
                : nullptr;
            double* hta=(tensor_enabled && tangent)
                ? ex.tensor_message_tangent_adjoints.data()+static_cast<size_t>(nbh.inds[j])*TC
                : nullptr;
            const double wr=tangent ? (direction[j][0]*nbh.vecs[j][0]+direction[j][1]*nbh.vecs[j][1]+direction[j][2]*nbh.vecs[j][2])/nbh.dists[j] : 0.0;
            for (int l=0;l<=sh_l_max_;++l)
                for (int o=0;o<outer_k_rank_;++o) {
                    const int block=l*outer_k_rank_+o,row=block*R,mu=o*(sh_l_max_+1)+l;
                    double av=0,at=0,at_y=0;
                    for (int m=-l;m<=l;++m) {
                        const int h=l*l+l+m,node=ex.outer_nodes[o*H+h];
                        double angular=y[h];
                        double angular_t=0.0;
                        int tc=-1;
                        if (tensor_enabled && l>=1 && l<=Lmsg) {
                            tc=TensorComponentLocal(l,o,m,outer_k_rank_);
                            angular+=hmsg[tc];
                            if (htmsg!=nullptr) angular_t+=htmsg[tc];
                        }
                        av+=ex.adjoints[node]*angular;
                        if (tangent) {
                            at+=ex.tangent_adjoints[node]*angular;
                            double yt=angular_t;
                            for (int a=0;a<3;++a) yt+=dy[3*h+a]*direction[j][a];
                            at_y+=ex.tangent_adjoints[node]*yt;
                        }
                        if (tensor_enabled && l>=1 && l<=Lmsg) {
                            const double radial=ex.radial_values[edge*B+block];
                            const double radial_t=tangent
                                ? ex.radial_tangents[edge*B+block] : 0.0;
                            const double vi=OuterTypeCoeff(type,mu);
                            const double vj=OuterTypeCoeff(nbh.types[j],mu);
                            ha[tc]+=vi*vj*(ex.adjoints[node]*radial
                                           + (tangent ? ex.tangent_adjoints[node]*radial_t : 0.0));
                            if (tangent)
                                hta[tc]+=vi*vj*ex.tangent_adjoints[node]*radial;
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
                    const int pr=AdaptiveRadialSharedN() ? (row/R)*T : row;
                    for (int t=0;t<T;++t) {
                        const size_t signal_at=static_cast<size_t>(i)*ex.projection_rows+pr+t;
                        const double x=ex.signals[signal_at];
                        double a_grad=ca*x;
                        if (tangent) {
                            const double first=AdaptiveRadialTanh() ? 1.0-x*x : 1.0;
                            a_grad+=ex.coefficient_tangent_adjoints[at]*first*ex.signal_tangents[signal_at];
                        }
                        out[AdaptiveRadialAmplitudeOffset()+ex.amplitude_offsets[i]+row*T+t]+=a_grad;
                    }
                }
            }
    ReduceAdaptiveRadialCoefficientAdjoints(tangent);
    const int projection_rows=ex.projection_rows;
    const bool direct_linear=!AdaptiveRadialSharedN() && !AdaptiveRadialTanh();
    const double* seeds=direct_linear ? ex.coefficient_adjoints.data() : ex.signal_adjoints.data();
    const double* tangent_seeds=direct_linear ? ex.coefficient_tangent_adjoints.data() : ex.signal_tangent_adjoints.data();
    ex.scalar_adjoints.resize(static_cast<size_t>(atoms)*Q);
    if (atoms>0) {
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,projection_rows,Q,atoms,1.0,seeds,projection_rows,two_layer_gate_scalar_values_cache_.data(),Q,1.0,out.data()+AdaptiveRadialWeightOffset(),Q);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,projection_rows,1.0,seeds,projection_rows,weights,Q,0.0,ex.scalar_adjoints.data(),Q);
        if (tangent) {
            ex.scalar_tangent_adjoints.resize(static_cast<size_t>(atoms)*Q);
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,projection_rows,Q,atoms,1.0,tangent_seeds,projection_rows,ex.scalar_tangents.data(),Q,1.0,out.data()+AdaptiveRadialWeightOffset(),Q);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,atoms,Q,projection_rows,1.0,tangent_seeds,projection_rows,weights,Q,0.0,ex.scalar_tangent_adjoints.data(),Q);
        }
    }
    if (tensor_enabled || JointReadoutEnabled()) {
        ex.tensor_gate_moment_adjoints.resize(alpha_moments_count);
        if (tangent)
            ex.tensor_gate_moment_tangent_adjoints.resize(alpha_moments_count);
    }
    const int cached_moments=static_cast<int>(two_layer_gate_required_moment_indices_.size());
    const int tensor_offset=tensor_enabled ? TensorMessageCoeffOffset() : 0;
    const double* Bmsg=tensor_enabled ? regression_coeffs.data()+tensor_offset : nullptr;
    for (int i=0;i<atoms;++i) {
        if (tensor_enabled || JointReadoutEnabled()) {
            std::fill(ex.tensor_gate_moment_adjoints.begin(),
                      ex.tensor_gate_moment_adjoints.end(),0.0);
            if (tangent)
                std::fill(ex.tensor_gate_moment_tangent_adjoints.begin(),
                          ex.tensor_gate_moment_tangent_adjoints.end(),0.0);
        }
        if(JointReadoutEnabled()) {
            const int JG=two_layer_gate_k_rank_;
            const size_t start=static_cast<size_t>(i)*JG*H;
            for(int g=0;g<JG;++g) for(int h=1;h<H;++h) {
                const int dense=joint_readout_gate_dense_indices_[g*H+h];
                const int moment=two_layer_gate_required_moment_indices_[dense];
                ex.tensor_gate_moment_adjoints[moment]+=ex.joint_gate_adjoints[start+g*H+h];
                if(tangent) ex.tensor_gate_moment_tangent_adjoints[moment]+=ex.joint_gate_tangent_adjoints[start+g*H+h];
            }
        }
        if (tensor_enabled) {
            const double* z=two_layer_gate_moment_values_cache_.data()
                + static_cast<size_t>(i)*cached_moments;
            const double* zt=tangent ? ex.gate_moment_tangents.data()
                + static_cast<size_t>(i)*alpha_moments_count : nullptr;
            const double* ha=ex.tensor_message_adjoints.data()+static_cast<size_t>(i)*TC;
            const double* hta=tangent ? ex.tensor_message_tangent_adjoints.data()
                + static_cast<size_t>(i)*TC : nullptr;
            for (int L=1; L<=Lmsg; ++L)
                for (int o=0; o<outer_k_rank_; ++o)
                    for (int g=0; g<G; ++g) {
                        const int b=((L-1)*outer_k_rank_+o)*G+g;
                        double bgrad=0.0;
                        for (int m=-L; m<=L; ++m) {
                            const int tc=TensorComponentLocal(L,o,m,outer_k_rank_);
                            const int dense=tensor_message_gate_moment_dense_indices_[
                                TensorGateDenseLocal(g,L,m,Lmsg)];
                            const int moment=two_layer_gate_required_moment_indices_[dense];
                            bgrad+=ha[tc]*z[dense];
                            ex.tensor_gate_moment_adjoints[moment]+=Bmsg[b]*ha[tc];
                            if (tangent) {
                                bgrad+=hta[tc]*zt[moment];
                                ex.tensor_gate_moment_tangent_adjoints[moment]+=Bmsg[b]*hta[tc];
                            }
                        }
                        out[tensor_offset+b]+=bgrad;
                    }
        }
        if (tangent) EdgeDirections(neighborhoods[i],i,frc_weights,str_weights,direction);
        AccumulateTwoLayerGateScalarParamGradForScalarSeeds(neighborhoods[i],out,
            ex.scalar_adjoints.data()+static_cast<size_t>(i)*Q,tangent ? direction.data() : nullptr,i,
            tangent ? ex.gate_moment_tangents.data()+static_cast<size_t>(i)*alpha_moments_count : nullptr,
            tangent ? ex.scalar_tangent_adjoints.data()+static_cast<size_t>(i)*Q : nullptr,
            nullptr,
            (tensor_enabled || JointReadoutEnabled()) && tangent ? ex.tensor_gate_moment_tangent_adjoints.data() : nullptr,
            (tensor_enabled || JointReadoutEnabled()) ? ex.tensor_gate_moment_adjoints.data() : nullptr);
    }
}

void MLMTPR::AddAdaptiveRadialSmoothnessPenalty(double coeff,int grid_size,
                                                double& penalty,Array1D* gradient) {
    if (coeff==0.0) return;
    if (!(coeff>0.0) || grid_size<=0)
        ERROR("adaptive radial smoothness requires nonnegative weight and positive grid size");
    const int R=p_RadialBasis->rb_size,Q=TwoLayerGateScalarCount(),O=outer_k_rank_;
    const int blocks=(sh_l_max_+1)*O,C=species_count,T=AdaptiveRadialLatentCount();
    const double cutoff=p_RadialBasis->max_dist;
    const double weight=coeff*cutoff/(static_cast<double>(grid_size)*blocks*C*C);
    if (gradient!=nullptr) gradient->resize(CoeffCount());
    std::vector<double> inverse_scale(Q);
    for (int q=0;q<Q;++q) inverse_scale[q]=1.0/AdaptiveRadialWeightOptimizerScale(q);
    // A fixed isotropic prior in normalized gate-feature coordinates:
    // |R_base'|² + sum_q |R_response,q'|². This is a physical a*U
    // regularizer, not a norm of the redundant optimization factors separately.
    // With tanh this penalizes the response Jacobian at s=0 (phi'(0)=1),
    // not a nonexistent globally folded linear response.
    // It is not claimed to equal the correlated empirical feature covariance.
    std::vector<double> response(static_cast<size_t>(Q+1)*R),der(Q+1),ds(Q+1),dh(Q+1);
    for (int l=0;l<=sh_l_max_;++l) for (int o=0;o<O;++o) {
        const int mu=o*(sh_l_max_+1)+l,row0=(l*O+o)*R;
        const int c0=RadialCoeffOffset()+mu*RadialCoeffBlockSize();
        for (int n=0;n<R;++n) {
            response[n]=regression_coeffs[c0+n];
        }
        for (int zi=0;zi<C;++zi) for (int zj=0;zj<C;++zj) {
            const int a_offset=AdaptiveRadialAmplitudeOffset()+(AdaptiveRadialSpeciesAmplitude() ? zj*blocks*R*T : 0);
            for (int n=0;n<R;++n) {
                const int pr=AdaptiveRadialSharedN() ? (l*O+o)*T : row0+n;
                for (int q=0;q<Q;++q) {
                    double value=0.0;
                    for (int t=0;t<T;++t)
                        value+=regression_coeffs[a_offset+(row0+n)*T+t]
                            *regression_coeffs[AdaptiveRadialWeightOffset()+static_cast<size_t>(pr+t)*Q+q];
                    response[static_cast<size_t>(q+1)*R+n]=value*inverse_scale[q];
                }
            }
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
                    const int a_index=a_offset+(row0+n)*T;
                    const int u0=AdaptiveRadialWeightOffset()+(AdaptiveRadialSharedN() ? (l*O+o)*T : row0+n)*Q;
                    for (int t=0;t<T;++t) {
                        double a_grad=0.0;
                        for (int q=0;q<Q;++q) {
                            const double g=common*der[q+1]*inverse_scale[q];
                            a_grad+=g*regression_coeffs[u0+t*Q+q];
                            (*gradient)[u0+t*Q+q]+=g*regression_coeffs[a_index+t];
                        }
                        (*gradient)[a_index+t]+=a_grad;
                    }
                }
            }
        }
    }
}
