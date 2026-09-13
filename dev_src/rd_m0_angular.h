#ifndef SUS2_RD_M0_ANGULAR_H
#define SUS2_RD_M0_ANGULAR_H

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Independent real m'=0 rotation column.  Our convention is
// Y_l(Q n) = D_l(Q) Y_l(n), with real +m = cos(m phi), -m = sin(m phi)
// and no additional Condon--Shortley sign.  If Q zhat = n, this returns
// D_l,m0(Q), not Y_lm(n).  Y_lm(n) = PoleValue(l) * D_l,m0(Q).
//
// Geometrically the column transports the fixed axial l-tensor from zhat to
// the edge direction.  It may be read as a passive angular/frame factor with
// the corresponding inverse convention for components.  The operational
// equation above fixes the sign/convention unambiguously.  No roll choice is
// needed: Q and Q R_z(phi) have the same m'=0 column.  This statement does not
// extend to transporting an arbitrary environment tensor without specifying
// its local-frame transformation.
namespace RDM0Angular {

constexpr int kMaxL = 6;
constexpr int kMaxComponents = (kMaxL + 1) * (kMaxL + 1);

inline int Index(int l, int m) { return l * l + l + m; }

// Select once per process, as for other SUS2 runtime kernels.  Callers may
// catch invalid_argument at their normal model/setup boundary to report it
// through the application's ERROR mechanism.  An invalid value never silently
// falls back to ry.  Environment changes after the first call are unsupported.
inline bool Enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("SUS2_SH_ANGULAR_BACKEND");
        if (value == nullptr || std::strcmp(value, "ry") == 0) return false;
        if (std::strcmp(value, "rd-m0") == 0) return true;
        throw std::invalid_argument("SUS2_SH_ANGULAR_BACKEND must be ry or rd-m0");
    }();
    return enabled;
}

namespace detail {

struct Coefficients {
    double pole[kMaxL + 1];
    double diagonal[kMaxL + 1];
    double first[kMaxComponents];
    double second[kMaxComponents];

    Coefficients() : pole{}, diagonal{}, first{}, second{}
    {
        const double inv_four_pi = 0.079577471545947667884441881686257181;
        for (int l = 0; l <= kMaxL; ++l)
            pole[l] = std::sqrt((2.0 * l + 1.0) * inv_four_pi);
        // The sqrt(2) real-basis factor enters at m=1, hence diagonal[1]=1.
        diagonal[1] = 1.0;
        for (int m = 2; m <= kMaxL; ++m)
            diagonal[m] = std::sqrt((2.0 * m - 1.0) / (2.0 * m));
        for (int m = 0; m <= kMaxL; ++m) {
            for (int l = m + 1; l <= kMaxL; ++l) {
                const int i = Index(l, m);
                const double denominator = static_cast<double>(l * l - m * m);
                first[i] = (2.0 * l - 1.0) / std::sqrt(denominator);
                second[i] = std::sqrt(
                    static_cast<double>((l - 1) * (l - 1) - m * m) / denominator);
            }
        }
    }
};

inline const Coefficients& Constants()
{
    static const Coefficients coefficients;
    return coefficients;
}

template <bool Derivatives>
inline void Evaluate(const double* xyz, double r, int lmax, double* d, double* dd)
{
    const Coefficients& c = Constants();
    const double inverse_r = 1.0 / r;
    const double n[3] = {xyz[0] * inverse_r, xyz[1] * inverse_r,
                         xyz[2] * inverse_r};
    d[0] = 1.0;
    if (Derivatives)
        dd[0] = dd[1] = dd[2] = 0.0;

    // Sectoral rotation columns are normalized powers of n_x+i*n_y.
    // This combines sin(theta)^m with exp(i*m*phi) before evaluation, avoiding
    // atan2, acos, division by sin(theta), and all special pole branches.
    for (int m = 1; m <= lmax; ++m) {
        const int p = Index(m, m), q = Index(m, -m);
        const int old_p = Index(m - 1, m - 1);
        const int old_q = Index(m - 1, -(m - 1));
        const double re = d[old_p];
        const double im = m == 1 ? 0.0 : d[old_q];
        const double factor = c.diagonal[m];
        d[p] = factor * (n[0] * re - n[1] * im);
        d[q] = factor * (n[1] * re + n[0] * im);
        if (Derivatives) {
            for (int a = 0; a < 3; ++a) {
                const double dre = dd[3 * old_p + a];
                const double dim = m == 1 ? 0.0 : dd[3 * old_q + a];
                dd[3 * p + a] = factor * ((a == 0 ? re : 0.0) + n[0] * dre
                                           - (a == 1 ? im : 0.0) - n[1] * dim);
                dd[3 * q + a] = factor * ((a == 1 ? re : 0.0) + n[1] * dre
                                           + (a == 0 ? im : 0.0) + n[0] * dim);
            }
        }
    }

    // Normalized m'=0 Wigner-column recurrence at fixed |m|:
    // d_lm = [(2l-1)/sqrt(l*l-m*m)] n_z d_(l-1)m
    //       - sqrt(((l-1)^2-m*m)/(l*l-m*m)) d_(l-2)m.
    // +m and -m share coefficients.  For m=0 there is only one real lane.
    for (int m = 0; m <= lmax; ++m) {
        for (int l = m + 1; l <= lmax; ++l) {
            const double acoef = c.first[Index(l, m)];
            const double bcoef = c.second[Index(l, m)];
            for (int lane = 0; lane < (m == 0 ? 1 : 2); ++lane) {
                const int signed_m = lane == 0 ? m : -m;
                const int current = Index(l, signed_m);
                const int previous = Index(l - 1, signed_m);
                d[current] = acoef * n[2] * d[previous];
                if (Derivatives) {
                    for (int a = 0; a < 3; ++a)
                        dd[3 * current + a] = acoef * ((a == 2 ? d[previous] : 0.0)
                                                       + n[2] * dd[3 * previous + a]);
                }
                if (l >= m + 2) {
                    const int previous2 = Index(l - 2, signed_m);
                    d[current] -= bcoef * d[previous2];
                    if (Derivatives) {
                        for (int a = 0; a < 3; ++a)
                            dd[3 * current + a] -= bcoef * dd[3 * previous2 + a];
                    }
                }
            }
        }
    }

    if (Derivatives) {
        // dd above differentiates a polynomial extension in n.  Apply the
        // complete normalization chain rule dn_b/dx_a=(delta_ab-n_a*n_b)/r.
        // This is also the exact finite tangent derivative at either pole.
        const int count = (lmax + 1) * (lmax + 1);
        for (int i = 0; i < count; ++i) {
            const double normal = n[0] * dd[3 * i] + n[1] * dd[3 * i + 1]
                                + n[2] * dd[3 * i + 2];
            for (int a = 0; a < 3; ++a)
                dd[3 * i + a] = (dd[3 * i + a] - n[a] * normal) * inverse_r;
        }
    }
}

} // namespace detail

inline double PoleValue(int l)
{
    if (l < 0 || l > kMaxL)
        throw std::invalid_argument("RDM0Angular::PoleValue requires 0 <= l <= 6");
    return detail::Constants().pole[l];
}

// xyz is a nonzero displacement and r is its Euclidean norm, already computed
// by the caller.  d has (lmax+1)^2 entries in l*l+l+m order.  Optional dd has
// three contiguous Cartesian derivatives per entry: dd[3*(l*l+l+m)+a].
// Input/output buffers must not alias.  Values-only specialization allocates
// no derivative workspace and executes no derivative arithmetic.
inline void Eval(const double* xyz, double r, int lmax, double* d, double* dd = nullptr)
{
    if (xyz == nullptr || d == nullptr || lmax < 0 || lmax > kMaxL
        || !(r > 0.0) || !std::isfinite(r))
        throw std::invalid_argument("RDM0Angular::Eval requires finite r>0, 0<=lmax<=6 and nonnull xyz/d");
    if (dd != nullptr)
        detail::Evaluate<true>(xyz, r, lmax, d, dd);
    else
        detail::Evaluate<false>(xyz, r, lmax, d, nullptr);
}

} // namespace RDM0Angular

#endif // SUS2_RD_M0_ANGULAR_H
