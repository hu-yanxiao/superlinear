#include "src/radial_basis.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
double worst = 0.0, worst_fd = 0.0;
bool close(double a, double b, double tolerance) {
    const double error = std::abs(a-b)/std::max(1.0,std::max(std::abs(a),std::abs(b)));
    worst = std::max(worst,error);
    return std::isfinite(a) && std::isfinite(b) && error <= tolerance;
}
template<class Basis>
bool check(bool specialized) {
    for (int R : {2,3,10,16}) for (int p : {0,1,5,64}) {
        Basis ref(0.5,6.5,R), opt(0.5,6.5,R);
        ref.SetRadialCutoff(p ? "polynomial" : "legacy",p ? p : 5);
        opt.SetRadialCutoff(p ? "polynomial" : "legacy",p ? p : 5);
        for (double scale : {0.13,1.0,17.0}) for (double slope : {0.2,1.5,4.0})
        for (double shift : {0.0,2.5,6.0}) for (double r : {0.5,0.83,2.7,5.91,6.49999,6.5}) {
            ref.scaling = opt.scaling = scale;
            ref.RB_Calc(r,slope,shift,2);
            std::fill(opt.rb_vals.begin(),opt.rb_vals.end(),std::numeric_limits<double>::quiet_NaN());
            std::fill(opt.rb_ders.begin(),opt.rb_ders.end(),std::numeric_limits<double>::quiet_NaN());
            // Virtual dispatch and unspecified-derivative poisoning test.
            AnyRadialBasis& base = opt;
            base.RB_CalcRDerOnly(r,slope,shift,2);
            for (int b=0;b<R;++b)
                if (!close(ref.rb_vals[b],opt.rb_vals[b],3e-13)
                    || !close(ref.rb_ders[b],opt.rb_ders[b],3e-13)) return false;
            if (specialized) for (int b=R;b<5*R;++b)
                if (!std::isnan(opt.rb_ders[b])) return false;
            // Re-entering the full evaluator must refresh every mixed block.
            base.RB_Calc(r,slope,shift,2);
            for (int b=0;b<5*R;++b)
                if (!close(ref.rb_ders[b],opt.rb_ders[b],3e-13)) return false;
            // FD covers the new recurrences. Fallback bases are tested above
            // against every original derivative; saturated Jacobi endpoint
            // regularization is outside this optimization's FD contract.
            if (specialized && r>0.51 && r<6.49) {
                const double h=1e-6;
                base.RB_CalcRDerOnly(r+h,slope,shift,2);
                const std::vector<double> plus=opt.rb_vals;
                base.RB_CalcRDerOnly(r-h,slope,shift,2);
                for (int b=0;b<R;++b) {
                    const double fd=(plus[b]-opt.rb_vals[b])/(2*h);
                    const double err=std::abs(fd-ref.rb_ders[b])/std::max(1.0,std::abs(ref.rb_ders[b]));
                    worst_fd=std::max(worst_fd,err);
                    if (!std::isfinite(fd) || err>2e-6) {
                        std::cerr << ref.GetRBTypeString() << " R=" << R << " p=" << p
                            << " scale=" << scale << " slope=" << slope << " shift=" << shift
                            << " r=" << r << " b=" << b << " analytic=" << ref.rb_ders[b]
                            << " fd=" << fd << " error=" << err << '\n';
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
}
int main() {
    const bool ok=check<RadialBasis_Chebyshev_sss>(true)
        && check<RadialBasis_Chebyshev_sss_rational>(true)
        && check<RadialBasis_Laguerre_log1p>(false)
        && check<RadialBasis_Jacobi_sss>(false);
    std::cout << "radial_coordinate_only " << (ok?"PASS":"FAIL")
              << " max_scaled=" << worst << " fd_scaled=" << worst_fd << '\n';
    return ok ? 0 : 1;
}
