#include "src/radial_basis.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

double RelativeError(double a, double b)
{
    return std::abs(a-b)/(1.0+std::max(std::abs(a),std::abs(b)));
}

template<class Basis>
void CheckBasis()
{
    Basis basis(1.0, 6.0, 8);
    basis.ConfigureInferenceRadialTable(RadialTableMode::Table, 0.002, 8*1024*1024);
    double max_value = 0, max_der = 0, max_fd = 0;
    for (int channel : {0, 2}) {
        for (int cutoff : {0, 1}) {
            basis.SetRadialCutoff(cutoff ? "polynomial" : "legacy", 5);
            for (int sample = 0; sample < 101; ++sample) {
                const double r = sample == 0 ? 1.0 : sample == 100 ? 6.0 :
                    1.0 + 5.0*(sample+0.137)/101.0;
                basis.SetInferenceRadialTableActive(false);
                basis.RB_Calc(r, 1.7, 2.1, channel);
                const auto direct_vals = basis.rb_vals;
                const auto direct_ders = basis.rb_ders;
                basis.SetInferenceRadialTableActive(true);
                const auto hits_before = basis.GetInferenceRadialTableStats().hits;
                // Full, value-only, and r-derivative-only entry points must
                // share the same conservative interpolation polynomial.
                if (sample % 3 == 0) basis.RB_Calc(r, 1.7, 2.1, channel);
                if (sample % 3 == 1) basis.RB_CalcValsOnly(r, 1.7, 2.1, channel);
                if (sample % 3 == 2) basis.RB_CalcRDerOnly(r, 1.7, 2.1, channel);
                const auto vals = basis.rb_vals;
                const auto ders = basis.rb_ders;
                const bool table_used = basis.GetInferenceRadialTableStats().hits > hits_before;
                for (int b = 0; b < basis.rb_size; ++b) {
                    max_value = std::max(max_value, RelativeError(vals[b], direct_vals[b]));
                    max_der = std::max(max_der, RelativeError(ders[b], direct_ders[b]));
                    if (table_used && (sample == 0 || sample == 100))
                        Require(vals[b] == direct_vals[b] && ders[b] == direct_ders[b],
                                "table endpoint does not reproduce direct knot exactly");
                }
                if (table_used && sample > 0 && sample < 100) {
                    const double epsilon = 1e-6;
                    basis.RB_CalcValsOnly(r+epsilon, 1.7, 2.1, channel);
                    const auto plus = basis.rb_vals;
                    basis.RB_CalcValsOnly(r-epsilon, 1.7, 2.1, channel);
                    for (int b = 0; b < basis.rb_size; ++b)
                        max_fd = std::max(max_fd, RelativeError(
                            (plus[b]-basis.rb_vals[b])/(2*epsilon), ders[b]));
                }
            }
        }
    }
    std::cout << basis.GetRBTypeString() << " value=" << max_value
              << " derivative=" << max_der << " finite_difference=" << max_fd
              << " rejected_fallbacks=" << basis.GetInferenceRadialTableStats().invalid_fallbacks << '\n';
    Require(max_value < 2e-7, "table value differs from direct basis");
    Require(max_der < 2e-5, "table derivative differs from direct basis");
    Require(max_fd < 2e-6, "table value and derivative are not conservative");
}

class RejectedBasis : public AnyRadialBasis {
public:
    int direct_calls = 0;
    bool nonfinite = false;
    RejectedBasis() : AnyRadialBasis(1.0, 3.0, 2) {}
    void RB_Calc(double r, double scal=0.1, double s=0.1, int k=0) override {
        if (TryInferenceRadialTable(r,scal,s,k)) return;
        ++direct_calls;
        for (int b = 0; b < rb_size; ++b) {
            rb_vals[b] = nonfinite && r == 1.0 ?
                std::numeric_limits<double>::quiet_NaN() : r*r;
            rb_ders[b] = 2*r+1; // Deliberately inconsistent derivative.
        }
    }
};

void CheckRejectedTables()
{
    for (bool nonfinite : {false, true}) {
        RejectedBasis basis;
        basis.nonfinite = nonfinite;
        basis.ConfigureInferenceRadialTable(RadialTableMode::Table,0.01);
        basis.SetInferenceRadialTableActive(true);
        basis.RB_Calc(2.0);
        Require(basis.GetInferenceRadialTableStats().tables == 0, "invalid basis was tabled");
        Require(basis.GetInferenceRadialTableStats().invalid_fallbacks == 1,
                "invalid table fallback not reported");
        Require(basis.rb_vals[0] == 4.0 && basis.rb_ders[0] == 5.0,
                "rejected table contaminated direct scratch output");
        const int calls = basis.direct_calls;
        basis.RB_Calc(2.0);
        Require(basis.direct_calls == calls+1, "rejected table was rebuilt");
    }
}

void CheckSingleFunctionBasis()
{
    RadialBasis_Chebyshev_sss basis(1.0,3.0,1);
    basis.RB_CalcRDerOnly(2.1,1.7,2.1);
    const double direct_value = basis.rb_vals[0], direct_der = basis.rb_ders[0];
    basis.ConfigureInferenceRadialTable(RadialTableMode::Table,0.002);
    basis.SetInferenceRadialTableActive(true);
    basis.RB_CalcRDerOnly(2.1,1.7,2.1);
    Require(RelativeError(basis.rb_vals[0],direct_value) < 1e-12 &&
            RelativeError(basis.rb_ders[0],direct_der) < 1e-12,
            "single radial function table is incorrect");
}

void CheckKeysAndLimits()
{
    RadialBasis_Chebyshev_sssw_lmp basis(1.0, 6.0, 8);
    basis.ConfigureInferenceRadialTable(RadialTableMode::Table, 0.01, 1024*1024);
    // Configuring must not enable tables for training calls.
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().tables == 0, "training enabled tables");
    Require(!basis.SetInferenceRadialTableActive(true), "default inference flag is active");
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    basis.RB_Calc(2.2, std::nextafter(1.0, 2.0), 2.0, 0);
    basis.RB_Calc(2.2, 1.0, std::nextafter(2.0, 3.0), 0);
    basis.RB_Calc(2.2, 1.0, 2.0, 1);
    Require(basis.GetInferenceRadialTableStats().tables == 4, "parameter/channel key collision");
    const auto before = basis.GetInferenceRadialTableStats();
    basis.RB_Calc(0.5, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().domain_fallbacks == before.domain_fallbacks+1,
            "short distance was clamped or table extrapolated");
    const auto short_vals = basis.rb_vals;
    basis.SetInferenceRadialTableActive(false);
    basis.RB_Calc(0.5, 1.0, 2.0, 0);
    Require(short_vals == basis.rb_vals, "short-distance direct fallback changed values");
    basis.SetInferenceRadialTableActive(true);
    basis.scaling = 2.0;
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().tables == 1, "scaling mutation retained stale cache");
    basis.min_dist = 0.9;
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().tables == 1, "domain mutation retained stale cache");
    basis.SetRadialCutoff("polynomial", 6);
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().tables == 1, "cutoff mutation retained stale cache");
    basis.ConfigureInferenceRadialTable(RadialTableMode::Table,0.01,65536);
    basis.RB_Calc(2.2,1.0,2.0,0);
    Require(basis.GetInferenceRadialTableStats().tables == 1, "first table exceeds expected budget");
    basis.RB_Calc(2.2,1.0,2.0,1);
    Require(basis.GetInferenceRadialTableStats().tables == 1 &&
            basis.GetInferenceRadialTableStats().memory_fallbacks == 1 &&
            basis.GetInferenceRadialTableStats().bytes <= 65536,
            "multiple tables exceeded aggregate knot budget");
    basis.ConfigureInferenceRadialTable(RadialTableMode::Auto, 0.01, 1);
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().tables == 0, "memory budget exceeded");
    Require(basis.GetInferenceRadialTableStats().memory_fallbacks == 1, "memory fallback not exposed");
    const auto limited_vals = basis.rb_vals;
    basis.SetInferenceRadialTableActive(false);
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(limited_vals == basis.rb_vals, "memory fallback changed direct values");
    basis.SetInferenceRadialTableActive(true);
    basis.ConfigureInferenceRadialTable(RadialTableMode::Table,
        std::numeric_limits<double>::denorm_min(), 1024*1024);
    basis.RB_Calc(2.2, 1.0, 2.0, 0);
    Require(basis.GetInferenceRadialTableStats().memory_fallbacks == 1, "grid overflow not rejected");
    bool rejected = false;
    try { basis.ConfigureInferenceRadialTable(RadialTableMode::Table, 0.0); }
    catch (const std::invalid_argument&) { rejected = true; }
    Require(rejected, "invalid spacing accepted");
}
}

int main()
{
    try {
        CheckBasis<RadialBasis_Chebyshev_sss>();
        CheckBasis<RadialBasis_Chebyshev_sss_rational>();
        CheckBasis<RadialBasis_Chebyshev_sss_lmp>();
        CheckBasis<RadialBasis_Chebyshev_sss_rational_lmp>();
        CheckBasis<RadialBasis_Chebyshev_sssw>();
        CheckBasis<RadialBasis_Chebyshev_sssw_lmp>();
        CheckBasis<RadialBasis_Laguerre_log1p>();
        CheckBasis<RadialBasis_Laguerre_log1p_lmp>();
        CheckBasis<RadialBasis_Laguerre_log1p_pos>();
        CheckBasis<RadialBasis_Laguerre_log1p_pos_lmp>();
        CheckBasis<RadialBasis_Laguerre_log1p_noenv>();
        CheckBasis<RadialBasis_Laguerre_log1p_noenv_lmp>();
        CheckBasis<RadialBasis_Jacobi_sss>();
        CheckBasis<RadialBasis_Jacobi_sss_lmp>();
        CheckBasis<RadialBasis_Jacobi_sss_noweight>();
        CheckBasis<RadialBasis_Jacobi_sss_noweight_lmp>();
        CheckBasis<RadialBasis_Shapeev>();
        CheckBasis<RadialBasis_Chebyshev>();
        CheckBasis<RadialBasis_Chebyshev_s>();
        CheckBasis<RadialBasis_Chebyshev_ss>();
        CheckBasis<RadialBasis_Chebyshev_ssw>();
        CheckBasis<RadialBasis_Chebyshev_s_lmp>();
        CheckBasis<RadialBasis_Chebyshev_ss_lmp>();
        CheckBasis<RadialBasis_Chebyshev_ssw_lmp>();
        CheckBasis<RadialBasis_Chebyshev_ssss>();
        CheckBasis<RadialBasis_Chebyshev_sssss>();
        CheckBasis<RadialBasis_Chebyshev_sigma>();
        CheckBasis<RadialBasis_Bessel>();
        CheckBasis<RadialBasis_Besselw>();
        CheckBasis<RadialBasis_Chebyshev_Tri>();
        CheckBasis<RadialBasis_Bessel_sss>();
        CheckBasis<RadialBasis_Bessel_sssw>();
        CheckBasis<RadialBasis_Chebyshev_tanhexp>();
        CheckBasis<RadialBasis_Chebyshev_tanhexp_w>();
        CheckBasis<RadialBasis_Chebyshev_repuls>();
        CheckBasis<RadialBasis_Taylor>();
        CheckKeysAndLimits();
        CheckRejectedTables();
        CheckSingleFunctionBasis();
        std::cout << "radial Hermite table checks passed\n";
    } catch (const MlipException& error) {
        std::cerr << error.What() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
