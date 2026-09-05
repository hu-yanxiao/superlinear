#include "pure_w_execution.h"

// Vectorize independent output lanes, retaining the b accumulation order
// for each physical W/W' entry. Output has the established [l,o,k] layout.
template<int Width, bool Derivatives>
void PureWExecution::FixedWidthKernels(const int* mu_to_eval, const double* values,
                       const double* derivatives, double* w, double* dw) const {
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < K; ++k) {
            const int base = mu_to_eval[k*L+l] * R;
            // Fixed-size independent accumulators stay in registers. Do
            // not use vector scratch here: alias checks and dynamic lane
            // bounds in the radial loop dominate the small-O cases.
            double v[Width] = {}, d[Width] = {};
            const double* c = coefficients.data()
                + ((static_cast<size_t>(l)*K+k)*R)*Width;
            for (int b = 0; b < R; ++b) {
                for (int o = 0; o < Width; ++o)
                    v[o] += c[b*Width+o]*values[base+b];
                if (Derivatives)
                    for (int o = 0; o < Width; ++o)
                        d[o] += c[b*Width+o]*derivatives[5*base+b];
            }
            for (int o = 0; o < Width; ++o) {
                const size_t idx = (static_cast<size_t>(l)*Width+o)*K+k;
                w[idx] = v[o];
                if (Derivatives) dw[idx] = d[o];
            }
        }
    }
}

template<bool Derivatives>
bool PureWExecution::DispatchFixedWidth(const int* mu_to_eval, const double* values,
                        const double* derivatives, double* w, double* dw) const {
    switch (O) {
    case 1: FixedWidthKernels<1,Derivatives>(mu_to_eval,values,derivatives,w,dw); break;
    case 2: FixedWidthKernels<2,Derivatives>(mu_to_eval,values,derivatives,w,dw); break;
    case 4: FixedWidthKernels<4,Derivatives>(mu_to_eval,values,derivatives,w,dw); break;
    case 8: FixedWidthKernels<8,Derivatives>(mu_to_eval,values,derivatives,w,dw); break;
    case 16: FixedWidthKernels<16,Derivatives>(mu_to_eval,values,derivatives,w,dw); break;
    default: return false;
    }
    return true;
}

void PureWExecution::Kernels(const int* mu_to_eval, const double* values,
             const double* derivatives, double* w, double* dw) {
    if (dw == nullptr) {
        if (DispatchFixedWidth<false>(mu_to_eval,values,derivatives,w,dw)) return;
    } else if (DispatchFixedWidth<true>(mu_to_eval,values,derivatives,w,dw)) return;
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < K; ++k) {
            const int base = mu_to_eval[k*L+l] * R;
            std::fill(edge_values.begin(), edge_values.end(), 0.0);
            std::fill(edge_ders.begin(), edge_ders.end(), 0.0);
            for (int b = 0; b < R; ++b) {
                const double* c = coefficients.data()
                    + ((static_cast<size_t>(l)*K+k)*R+b)*O;
                const double v = values[base+b];
                for (int o = 0; o < O; ++o)
                    edge_values[o] += c[o] * v;
                if (dw != nullptr) {
                    const double d = derivatives[5*base+b];
                    for (int o = 0; o < O; ++o)
                        edge_ders[o] += c[o] * d;
                }
            }
            for (int o = 0; o < O; ++o) {
                const size_t idx = (static_cast<size_t>(l)*O+o)*K+k;
                w[idx] = edge_values[o];
                if (dw != nullptr) dw[idx] = edge_ders[o];
            }
        }
    }
}
