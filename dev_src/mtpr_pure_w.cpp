/* Exact basis-first pure-W EFS execution. No model or optimizer coordinates. */
#include "mtpr.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>

namespace {
double PureWNow() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

void MLMTPR::ReportPureWExecution() const
{
    const PureWExecution& ex = pure_w_execution_;
    if (!ex.profile) return;
    std::cout << "[PURE_W_EXECUTION] O=" << outer_k_rank_
        << " packed=" << ex.packed << " basis=" << ex.basis << " graph=" << ex.graph
        << " full_derivative_cache=" << two_layer_edge_cache_has_param_derivatives_
        << " W_entries=" << (ex.basis ? 0 : two_layer_edge_radial_mixing_kernel_cache_.size())
        << " W_storage_entries=" << two_layer_edge_radial_mixing_kernel_cache_.size()
        << " raw_dr_entries=" << (ex.basis ? ex.raw_dr.size() : 0)
        << " pack_s=" << ex.pack_seconds << " W_build_s=" << ex.kernel_seconds
        << " aggregate_s=" << ex.aggregate_seconds << " mix_s=" << ex.mix_seconds
        << " graph_s=" << ex.graph_seconds << " basis_force_s=" << ex.force_seconds
        << std::endl;
}
#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

void MLMTPR::PreparePureWGraph()
{
    PureWExecution& ex = pure_w_execution_;
    ex.graph = false;
    const char* requested = std::getenv("SUS2_SH_PURE_W_GRAPH");
    if (requested == nullptr || std::strcmp(requested,"legacy") == 0) return;
    if (std::strcmp(requested,"batched") != 0)
        ERROR("SUS2_SH_PURE_W_GRAPH must be legacy or batched");
    if (!OuterUsesPureRadialMatrix() || outer_k_rank_ < 2) return;
    const int O = outer_k_rank_, H = (sh_l_max_+1)*(sh_l_max_+1);
    const size_t start = static_cast<size_t>(OuterKRank1ProductStart());
    const size_t count = sh_products_.size()-start;
    if (count % O != 0) return;
    const size_t per_lane = count/O;
    // Model validation permits equivalent channelwise graphs that need not
    // have identical instruction ordering. Validate the execution mirror and
    // fall back if it cannot be represented by one instruction stream.
    std::vector<int> canonical(alpha_moments_count, -1);
    ex.nodes.assign(static_cast<size_t>(H)*O, -1);
    for (int o = 0; o < O; ++o)
        for (int l = 0; l <= sh_l_max_; ++l)
            for (int m = -l; m <= l; ++m) {
                const int h = l*l+l+m, node = OuterKRank1OuterBasicIndex(o,l,m);
                canonical[node] = h;
                ex.nodes[h*O+o] = node;
            }
    ex.terms.clear();
    int node_count = H;
    for (int o = O-1; o >= 0; --o) {
        for (size_t p = 0; p < per_lane; ++p) {
            const SHProduct& term = sh_products_[start+(O-1-o)*per_lane+p];
            if (canonical[term.left] < 0 || canonical[term.right] < 0) return;
            // The loader validates channel independence, but permits whole
            // channel blocks in a different order. Canonical angular labels
            // alone cannot establish physical output-lane ownership.
            if (ex.nodes[static_cast<size_t>(canonical[term.left])*O+o] != term.left
                || ex.nodes[static_cast<size_t>(canonical[term.right])*O+o] != term.right)
                return;
            if (o == O-1) {
                if (canonical[term.target] < 0) {
                    canonical[term.target] = node_count++;
                    ex.nodes.resize(static_cast<size_t>(node_count)*O, -1);
                }
                ex.terms.push_back({canonical[term.left],canonical[term.right],
                    canonical[term.target],term.coeff});
            } else {
                const PureWExecution::Term& ref = ex.terms[p];
                if (canonical[term.left] != ref.left || canonical[term.right] != ref.right
                    || term.coeff != ref.coeff) return;
                if (canonical[term.target] >= 0 && canonical[term.target] != ref.target) return;
                canonical[term.target] = ref.target;
            }
            const size_t mapped = static_cast<size_t>(canonical[term.target])*O+o;
            if (ex.nodes[mapped] >= 0 && ex.nodes[mapped] != term.target) return;
            ex.nodes[mapped] = term.target;
        }
    }
    for (int node : ex.nodes) if (node < 0) return;
    ex.graph_values.resize(ex.nodes.size());
    ex.graph_adjoints.resize(ex.nodes.size());
    ex.graph = true;
}

bool MLMTPR::ApplyPureWGraph(bool reverse)
{
    PureWExecution& ex = pure_w_execution_;
    if (!ex.graph || UseSHProductRows()) return false;
    const double started = ex.profile ? PureWNow() : 0.0;
    const int O = outer_k_rank_;
    for (size_t n = 0; n < ex.nodes.size(); ++n) {
        ex.graph_values[n] = moment_vals[ex.nodes[n]];
        if (reverse) ex.graph_adjoints[n] = site_energy_ders_wrt_moments_[ex.nodes[n]];
    }
    if (!reverse) {
        for (const PureWExecution::Term& t : ex.terms) {
            const double* left = ex.graph_values.data()+static_cast<size_t>(t.left)*O;
            const double* right = ex.graph_values.data()+static_cast<size_t>(t.right)*O;
            double* target = ex.graph_values.data()+static_cast<size_t>(t.target)*O;
            for (int o = 0; o < O; ++o) target[o] += t.coeff*left[o]*right[o];
        }
        for (size_t n = 0; n < ex.nodes.size(); ++n) moment_vals[ex.nodes[n]] = ex.graph_values[n];
    } else {
        for (size_t p = ex.terms.size(); p-- > 0;) {
            const PureWExecution::Term& t = ex.terms[p];
            for (int o = 0; o < O; ++o) {
                const double adj = ex.graph_adjoints[static_cast<size_t>(t.target)*O+o];
                if (adj == 0.0) continue;
                ex.graph_adjoints[static_cast<size_t>(t.left)*O+o] +=
                    t.coeff*ex.graph_values[static_cast<size_t>(t.right)*O+o]*adj;
                ex.graph_adjoints[static_cast<size_t>(t.right)*O+o] +=
                    t.coeff*ex.graph_values[static_cast<size_t>(t.left)*O+o]*adj;
            }
        }
        for (size_t n = 0; n < ex.nodes.size(); ++n)
            site_energy_ders_wrt_moments_[ex.nodes[n]] = ex.graph_adjoints[n];
    }
    if (ex.profile) ex.graph_seconds += PureWNow()-started;
    return true;
}

void MLMTPR::CalcPureWBasisFirstMoments(const Neighborhood& nbh)
{
    PureWExecution& ex = pure_w_execution_;
    const double started = ex.profile ? PureWNow() : 0.0;
    const int L = sh_l_max_+1, K = sh_k_max_, O = outer_k_rank_;
    const int R = p_RadialBasis->rb_size, H = L*L, KR = K*R;
    const size_t rb_stride = static_cast<size_t>(two_layer_edge_cache_eval_block_count_)*R;
    // Per-center scratch, not per-configuration-by-neighbor-by-output storage.
    ex.q.assign(static_cast<size_t>(KR)*H, 0.0);
    ex.outer.resize(static_cast<size_t>(O)*H);
    ex.input.resize(static_cast<size_t>(nbh.count)*radial_func_count);
    ex.gate_factor.resize(ex.input.size());
    const double* signals = active_two_layer_gate_values_ == nullptr ? nullptr
        : active_two_layer_gate_values_->data();
    std::fill(moment_vals, moment_vals+alpha_moments_count, 0.0);
    for (int j = 0; j < nbh.count; ++j) {
        const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
        const double* rb = two_layer_edge_radial_vals_cache_.data()+edge*rb_stride;
        const double* y = two_layer_edge_sh_values_cache_.data()+edge*H;
        const int neighbor = nbh.inds[j];
        PrepareTwoLayerGatePairMuBuffers(nbh.my_type, nbh.types[j], nullptr, -1,
            signals == nullptr ? nullptr : signals+static_cast<size_t>(neighbor)*radial_func_count,
            neighbor, kGateMuBufferTypeScale | kGateMuBufferAdditive);
        double* input = ex.input.data()+static_cast<size_t>(j)*radial_func_count;
        double* gate_factor = ex.gate_factor.data()+static_cast<size_t>(j)*radial_func_count;
        for (int mu = 0; mu < radial_func_count; ++mu) {
            input[mu] = two_layer_gate_type_scale_mu_buffer_[mu];
            const int type_index = RadialCoeffOffset()+mu*RadialCoeffBlockSize()+R+nbh.my_type;
            gate_factor[mu] = regression_coeffs[type_index]*two_layer_gate_additive_mu_buffer_[mu];
        }
        for (int l = 0; l < L; ++l) {
            const int M = 2*l+1;
            double* ql = ex.q.data()+static_cast<size_t>(KR)*l*l;
            for (int k = 0; k < K; ++k) {
                const int mu = k*L+l;
                const double* phi = rb+mu_to_radial_eval_block_[mu]*R;
                for (int b = 0; b < R; ++b) {
                    const double v = input[mu]*phi[b];
                    double* row = ql+(k*R+b)*M;
                    for (int m = 0; m < M; ++m)
                        row[m] += v*y[l*l+m];
                }
            }
        }
    }
    const double after_aggregate = ex.profile ? PureWNow() : 0.0;
    if (ex.profile) ex.aggregate_seconds += after_aggregate-started;
    const double* c = regression_coeffs.data()+TwoLayerGateRadialMixingCoeffOffset();
    for (int l = 0; l < L; ++l) {
        const int M = 2*l+1;
        // Physical coefficients already form [O,KR] matrices per l.
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, O, M, KR,
            1.0, c+static_cast<size_t>(l)*O*KR, KR,
            ex.q.data()+static_cast<size_t>(KR)*l*l, M,
            0.0, ex.outer.data(), M);
        for (int o = 0; o < O; ++o)
            for (int m = 0; m < M; ++m)
                moment_vals[OuterKRank1OuterBasicIndex(o,l,m-l)] = ex.outer[o*M+m];
    }
    if (ex.profile) ex.mix_seconds += PureWNow()-after_aggregate;
    if (ApplyPureWGraph(false)) {
        // The canonical instruction stream was executed across O lanes.
    } else if (UseSHProductRows()) {
        ApplySHProductRowsForward();
    } else {
        for (size_t p = static_cast<size_t>(OuterKRank1ProductStart()); p < sh_products_.size(); ++p) {
            const SHProduct& product = sh_products_[p];
            moment_vals[product.target] += product.coeff*moment_vals[product.left]*moment_vals[product.right];
        }
    }
    basis_vals[0] = 1.0;
    for (int i = 0; i < alpha_scalar_moments; ++i)
        basis_vals[1+i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcPureWBasisFirstForce(const Neighborhood& nbh, double center_linear)
{
    PureWExecution& ex = pure_w_execution_;
    const double started = ex.profile ? PureWNow() : 0.0;
    const int L = sh_l_max_+1, K = sh_k_max_, O = outer_k_rank_;
    const int R = p_RadialBasis->rb_size, H = L*L, KR = K*R;
    const size_t rb_stride = static_cast<size_t>(two_layer_edge_cache_eval_block_count_)*R;
    const double* c = regression_coeffs.data()+TwoLayerGateRadialMixingCoeffOffset();
    ex.q_adjoint.resize(static_cast<size_t>(KR)*H);
    // barQ = C^T barA. This consumes the same physical moment adjoints as the
    // edge-first EFS evaluator, including the species center head multiplier.
    for (int l = 0; l < L; ++l) {
        const int M = 2*l+1;
        for (int o = 0; o < O; ++o)
            for (int m = 0; m < M; ++m)
                ex.outer[o*M+m] = center_linear
                    * site_energy_ders_wrt_moments_[OuterKRank1OuterBasicIndex(o,l,m-l)];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, KR, M, O,
            1.0, c+static_cast<size_t>(l)*O*KR, KR, ex.outer.data(), M,
            0.0, ex.q_adjoint.data()+static_cast<size_t>(KR)*l*l, M);
    }
    for (int j = 0; j < nbh.count; ++j) {
        const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
        const double* rb = two_layer_edge_radial_vals_cache_.data()+edge*rb_stride;
        const double* dr = ex.raw_dr.data()+edge*rb_stride;
        const double* y = two_layer_edge_sh_values_cache_.data()+edge*H;
        const double* dy = two_layer_edge_sh_ders_cache_.data()+edge*3*H;
        const double* input = ex.input.data()+static_cast<size_t>(j)*radial_func_count;
        const double* gate_factor = ex.gate_factor.data()+static_cast<size_t>(j)*radial_func_count;
        double* gate_adj = active_two_layer_gate_adjoints_ == nullptr ? nullptr
            : active_two_layer_gate_adjoints_->data()+static_cast<size_t>(nbh.inds[j])*radial_func_count;
        const double inv_r = 1.0/nbh.dists[j];
        double radial_force = 0.0, angular_force[3] = {0.0,0.0,0.0};
        for (int l = 0; l < L; ++l) {
            const int M = 2*l+1;
            const double* adjl = ex.q_adjoint.data()+static_cast<size_t>(KR)*l*l;
            for (int k = 0; k < K; ++k) {
                const int mu = k*L+l;
                const int base = mu_to_radial_eval_block_[mu]*R;
                double input_adj = 0.0, phi_der = 0.0, angular[3] = {0.0,0.0,0.0};
                // Contract b before the four angular dot products. This is
                // the same chain rule, but costs 2*R*M rather than 4*R*M
                // products per (edge,l,k), plus small O(M) angular work.
                double phi_adj[13] = {}, radial_adj[13] = {}; // l_max <= 6
                for (int b = 0; b < R; ++b) {
                    const double* row = adjl+(k*R+b)*M;
                    for (int m = 0; m < M; ++m) {
                        phi_adj[m] += rb[base+b]*row[m];
                        radial_adj[m] += dr[base+b]*row[m];
                    }
                }
                for (int m = 0; m < M; ++m) {
                    input_adj += phi_adj[m]*y[l*l+m];
                    phi_der += radial_adj[m]*y[l*l+m];
                    for (int a = 0; a < 3; ++a)
                        angular[a] += phi_adj[m]*dy[3*(l*l+m)+a];
                }
                radial_force += input[mu]*phi_der;
                for (int a = 0; a < 3; ++a)
                    angular_force[a] += input[mu]*angular[a];
                if (gate_adj != nullptr) gate_adj[mu] += input_adj*gate_factor[mu];
            }
        }
        for (int a = 0; a < 3; ++a)
            buff_site_energy_ders_[j][a] += radial_force*nbh.vecs[j][a]*inv_r+angular_force[a];
    }
    // The caller's existing global gate-force chain propagates the accumulated
    // neighbor gate adjoints. Omitting it would silently lose three-body forces.
    if (ex.profile) ex.force_seconds += PureWNow()-started;
}
