#ifndef MLIP_PURE_W_EXECUTION_H
#define MLIP_PURE_W_EXECUTION_H

#include <algorithm>
#include <cstddef>
#include <vector>

// Execution-only storage. Physical coefficients/model serialization remain
// [l,o,k,b]. Repack from the current coefficients at each edge-cache build;
// never infer validity from a pointer (optimizers modify coefficients in place).
struct PureWExecution {
    int L = 0, O = 0, K = 0, R = 0;
    bool packed = false;
    bool basis = false;
    bool edge_force = false; // EFS-only fused pullback of cached W/Wprime
    bool graph = false;
    bool profile = false;
    double pack_seconds = 0.0, kernel_seconds = 0.0;
    double aggregate_seconds = 0.0, mix_seconds = 0.0, force_seconds = 0.0;
    double graph_seconds = 0.0;
    struct Term { int left, right, target; double coeff; };
    std::vector<Term> terms;
    std::vector<int> nodes; // canonical node-major, O independent lanes
    std::vector<double> graph_values, graph_adjoints;
    std::vector<double> coefficients; // [l,k,b,o]
    std::vector<double> edge_values, edge_ders;
    std::vector<double> raw_dr; // inference-only [edge,eval,b], no mixed derivatives
    std::vector<double> q, q_adjoint; // [l,k,b,m], all k/b/m retained
    std::vector<double> input, gate_factor; // one center's [neighbor,mu]
    std::vector<double> force_scratch; // one center's [neighbor,radial/x/y/z]
    std::vector<double> outer;

    // Conservative crossover measured on the VO2 CPU EFS workload. Small O,
    // large radial bases and sparse neighborhoods retain edge-first execution.
    // This is only dispatch: no rank/basis truncation changes the model.
    static bool PreferBasis(int outputs, int radial_size, size_t edges, int atoms) {
        return outputs >= 8 && radial_size <= 10 && atoms > 0
            && edges >= static_cast<size_t>(atoms)*16;
    }

    void Pack(int l_count, int outputs, int inputs, int basis_size,
              const double* physical) {
        L = l_count; O = outputs; K = inputs; R = basis_size;
        coefficients.resize(static_cast<size_t>(L) * O * K * R);
        edge_values.resize(O);
        edge_ders.resize(O);
        for (int l = 0; l < L; ++l)
            for (int k = 0; k < K; ++k)
                for (int b = 0; b < R; ++b)
                    for (int o = 0; o < O; ++o)
                        coefficients[((static_cast<size_t>(l)*K+k)*R+b)*O+o] =
                            physical[((static_cast<size_t>(l)*O+o)*K+k)*R+b];
    }

    // Out-of-line dispatch keeps specialized SIMD loops out of the already
    // large edge-cache builder's instruction footprint.
    void Kernels(const int* mu_to_eval, const double* values,
                 const double* derivatives, double* w, double* dw);

private:
    template<int Width, bool Derivatives>
    void FixedWidthKernels(const int* mu_to_eval, const double* values,
                           const double* derivatives, double* w, double* dw) const;
    template<bool Derivatives>
    bool DispatchFixedWidth(const int* mu_to_eval, const double* values,
                            const double* derivatives, double* w, double* dw) const;
};

#endif
