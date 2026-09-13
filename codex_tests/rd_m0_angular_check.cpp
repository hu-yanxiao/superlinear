// Standalone server-side qualification:
// c++ -O3 -std=c++11 -Idev_src codex_tests/rd_m0_angular_check.cpp -o rd_m0_check
// env -u SUS2_SH_ANGULAR_BACKEND ./rd_m0_check --expect-ry
// SUS2_SH_ANGULAR_BACKEND=ry ./rd_m0_check --expect-ry
// SUS2_SH_ANGULAR_BACKEND=ry-new ./rd_m0_check --expect-ry-new
// SUS2_SH_ANGULAR_BACKEND=hybrid ./rd_m0_check --expect-hybrid
// SUS2_SH_ANGULAR_BACKEND=rd-m0 ./rd_m0_check --expect-rd-m0
// SUS2_SH_ANGULAR_BACKEND=invalid ./rd_m0_check --expect-invalid-env
#include "rd_m0_angular.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const int kCount = RDM0Angular::kMaxComponents;
const double kPi = 3.141592653589793238462643383279502884;
int assertions = 0;
double max_fd_scaled_error = 0.0;
double max_reference_value_error = 0.0;
double max_reference_derivative_scaled_error = 0.0;
int physical_inplace_cases = 0;

void Require(bool condition, const std::string& name)
{
    ++assertions;
    if (!condition) throw std::runtime_error(name);
}

void Close(double actual, double expected, double absolute, const std::string& name)
{
    ++assertions;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > absolute) {
        std::cerr << std::setprecision(17) << name << " actual=" << actual
                  << " expected=" << expected << " tolerance=" << absolute << '\n';
        throw std::runtime_error(name);
    }
}

double Norm(const std::array<double, 3>& x)
{
    return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

long double Factorial(int n)
{
    long double f = 1.0L;
    for (int j = 2; j <= n; ++j) f *= j;
    return f;
}

long double Power(long double x, int n)
{
    long double p = 1.0L;
    for (int j = 0; j < n; ++j) p *= x;
    return p;
}

// Independent reference: explicit Rodrigues factorial sum for homogeneous
// regular solid harmonics, with analytic polynomial and r^-l derivatives.
// Unlike the implementation, this has no normalized angular recurrence.
// Its output is Y_lm itself, in the existing real SH convention.
void ReferenceRealSH(const std::array<double, 3>& x, int lmax,
                     double* y, double* dy)
{
    typedef std::complex<long double> Complex;
    const long double xx[3] = {x[0], x[1], x[2]};
    const long double r2 = xx[0] * xx[0] + xx[1] * xx[1] + xx[2] * xx[2];
    const long double r = std::sqrt(r2);
    const Complex u(xx[0], xx[1]);
    const Complex du[3] = {Complex(1.0L, 0.0L), Complex(0.0L, 1.0L), Complex(0.0L, 0.0L)};
    const long double pi = 3.141592653589793238462643383279502884L;
    Complex upow[7];
    upow[0] = Complex(1.0L, 0.0L);
    for (int m = 1; m <= lmax; ++m) upow[m] = upow[m - 1] * u;
    for (int l = 0; l <= lmax; ++l) {
        for (int m = 0; m <= l; ++m) {
            Complex solid(0.0L, 0.0L), derivative[3] = {};
            for (int s = 0; s <= (l - m) / 2; ++s) {
                const int zdegree = l - m - 2 * s;
                const long double coefficient = (s % 2 ? -1.0L : 1.0L)
                    * Factorial(2 * l - 2 * s)
                    / (Power(2.0L, l) * Factorial(s) * Factorial(l - s) * Factorial(zdegree));
                const long double zp = Power(xx[2], zdegree), rp = Power(r2, s);
                solid += coefficient * zp * rp * upow[m];
                for (int a = 0; a < 3; ++a) {
                    if (a == 2 && zdegree > 0)
                        derivative[a] += coefficient * zdegree * Power(xx[2], zdegree - 1) * rp * upow[m];
                    if (s > 0)
                        derivative[a] += coefficient * zp * s * Power(r2, s - 1) * 2.0L * xx[a] * upow[m];
                    if (m > 0)
                        derivative[a] += coefficient * zp * rp * m * upow[m - 1] * du[a];
                }
            }
            const long double normalization = std::sqrt((2.0L * l + 1.0L)
                / (4.0L * pi) * (m == 0 ? 1.0L : 2.0L) * Factorial(l - m) / Factorial(l + m));
            const long double factor = normalization / Power(r, l);
            const int p = RDM0Angular::Index(l, m), q = RDM0Angular::Index(l, -m);
            y[p] = static_cast<double>(factor * solid.real());
            if (m > 0) y[q] = static_cast<double>(factor * solid.imag());
            for (int a = 0; a < 3; ++a) {
                const Complex value = factor * (derivative[a] - static_cast<long double>(l) * solid * xx[a] / r2);
                dy[3 * p + a] = static_cast<double>(value.real());
                if (m > 0) dy[3 * q + a] = static_cast<double>(value.imag());
            }
        }
    }
}

void CheckClosedForms(const std::array<double, 3>& x)
{
    const double r = Norm(x), n[3] = {x[0] / r, x[1] / r, x[2] / r};
    double d[kCount], dd[3 * kCount];
    RDM0Angular::Eval(x.data(), r, 2, d, dd);
    Close(d[0], 1.0, 0.0, "D00");
    const int axes[3] = {1, 2, 0};
    for (int j = 0; j < 3; ++j) {
        const int i = RDM0Angular::Index(1, j - 1), axis = axes[j];
        Close(d[i], n[axis], 2.0e-15, "l1 value");
        for (int a = 0; a < 3; ++a)
            Close(dd[3 * i + a], ((a == axis ? 1.0 : 0.0) - n[axis] * n[a]) / r,
                  5.0e-15 / r, "l1 Cartesian derivative");
    }
    const double sqrt3 = std::sqrt(3.0);
    const double expected[5] = {sqrt3 * n[0] * n[1], sqrt3 * n[1] * n[2],
        (3.0 * n[2] * n[2] - 1.0) * 0.5, sqrt3 * n[0] * n[2],
        sqrt3 * 0.5 * (n[0] * n[0] - n[1] * n[1])};
    const double gradient[5][3] = {
        {sqrt3 * n[1], sqrt3 * n[0], 0.0},
        {0.0, sqrt3 * n[2], sqrt3 * n[1]},
        {0.0, 0.0, 3.0 * n[2]},
        {sqrt3 * n[2], 0.0, sqrt3 * n[0]},
        {sqrt3 * n[0], -sqrt3 * n[1], 0.0}
    };
    for (int j = 0; j < 5; ++j) {
        const int i = RDM0Angular::Index(2, j - 2);
        Close(d[i], expected[j], 5.0e-15, "l2 value");
        double dot = 0.0;
        for (int a = 0; a < 3; ++a) dot += n[a] * gradient[j][a];
        for (int a = 0; a < 3; ++a)
            Close(dd[3 * i + a], (gradient[j][a] - n[a] * dot) / r,
                  2.0e-14 / r, "l2 Cartesian derivative");
    }
}

void CheckPoles()
{
    for (int sign = -1; sign <= 1; sign += 2) {
        const double x[3] = {0.0, 0.0, sign * 2.7};
        double d[kCount], dd[3 * kCount];
        RDM0Angular::Eval(x, 2.7, 6, d, dd);
        for (int l = 0; l <= 6; ++l) {
            const double parity = sign < 0 && l % 2 ? -1.0 : 1.0;
            const double tangent = std::sqrt(l * (l + 1.0) / 2.0)
                * (sign < 0 && (l + 1) % 2 ? -1.0 : 1.0) / 2.7;
            for (int m = -l; m <= l; ++m) {
                const int i = RDM0Angular::Index(l, m);
                Close(d[i], m == 0 ? parity : 0.0, 3.0e-14, "exact pole value");
                for (int a = 0; a < 3; ++a) {
                    const double expected = (m == 1 && a == 0) || (m == -1 && a == 1) ? tangent : 0.0;
                    Close(dd[3 * i + a], expected, 2.0e-13, "exact pole derivative");
                }
            }
        }
    }
}

void CheckDirection(const std::array<double, 3>& x)
{
    const double r = Norm(x);
    double d[kCount], dd[3 * kCount], reference[kCount], reference_ders[3 * kCount];
    double only[kCount], opposite[kCount], opposite_ders[3 * kCount];
    RDM0Angular::Eval(x.data(), r, 6, d, dd);
    RDM0Angular::Eval(x.data(), r, 6, only, nullptr);
    ReferenceRealSH(x, 6, reference, reference_ders);
    const double negative[3] = {-x[0], -x[1], -x[2]};
    RDM0Angular::Eval(negative, r, 6, opposite, opposite_ders);
    for (int l = 0; l <= 6; ++l) {
        const double pole = RDM0Angular::PoleValue(l), parity = l % 2 ? -1.0 : 1.0;
        Close(pole, std::sqrt((2.0 * l + 1.0) / (4.0 * kPi)), 8.0e-16, "pole normalization");
        double norm2 = 0.0, norm_der[3] = {};
        for (int m = -l; m <= l; ++m) {
            const int i = RDM0Angular::Index(l, m);
            Close(only[i], d[i], 0.0, "values-only parity");
            Close(opposite[i], parity * d[i], 2.0e-14, "inversion value parity");
            const double error = std::abs(pole * d[i] - reference[i]);
            max_reference_value_error = std::max(max_reference_value_error, error);
            Close(pole * d[i], reference[i], 5.0e-14, "independent SH reference");
            double radial_derivative = 0.0;
            for (int a = 0; a < 3; ++a) {
                const double der_error = r * std::abs(pole * dd[3 * i + a] - reference_ders[3 * i + a]);
                max_reference_derivative_scaled_error = std::max(max_reference_derivative_scaled_error, der_error);
                Close(pole * dd[3 * i + a], reference_ders[3 * i + a], 5.0e-13 / r,
                      "independent SH derivative reference");
                Close(opposite_ders[3 * i + a], -parity * dd[3 * i + a],
                      3.0e-13 / r, "inversion derivative parity");
                radial_derivative += x[a] * dd[3 * i + a];
                norm_der[a] += d[i] * dd[3 * i + a];
            }
            Close(radial_derivative, 0.0, 3.0e-13, "radial tangent derivative");
            norm2 += d[i] * d[i];
        }
        Close(norm2, 1.0, 2.0e-13, "rotation column norm");
        for (int a = 0; a < 3; ++a)
            Close(norm_der[a], 0.0, 8.0e-13 / r, "rotation column norm derivative");
    }

    // Five-point Cartesian finite differences, including exact poles and axes.
    const double h = r * 3.0e-5;
    for (int a = 0; a < 3; ++a) {
        double shifted[4][kCount];
        const int offsets[4] = {-2, -1, 1, 2};
        for (int s = 0; s < 4; ++s) {
            std::array<double, 3> xx = x;
            xx[a] += offsets[s] * h;
            RDM0Angular::Eval(xx.data(), Norm(xx), 6, shifted[s], nullptr);
        }
        for (int i = 0; i < kCount; ++i) {
            const double fd = (shifted[0][i] - 8.0 * shifted[1][i]
                              + 8.0 * shifted[2][i] - shifted[3][i]) / (12.0 * h);
            max_fd_scaled_error = std::max(max_fd_scaled_error, r * std::abs(fd - dd[3 * i + a]));
            Close(fd, dd[3 * i + a], 3.0e-9 / r, "Cartesian five-point FD");
        }
    }

    // A real z-axis rotation must rotate each +/-m pair by m*angle.
    const double angle = 0.371, cs = std::cos(angle), sn = std::sin(angle);
    const double rotated[3] = {cs * x[0] - sn * x[1], sn * x[0] + cs * x[1], x[2]};
    double rotated_d[kCount];
    RDM0Angular::Eval(rotated, r, 6, rotated_d, nullptr);
    for (int l = 0; l <= 6; ++l) {
        Close(rotated_d[RDM0Angular::Index(l, 0)], d[RDM0Angular::Index(l, 0)],
              3.0e-14, "z rotation m0");
        for (int m = 1; m <= l; ++m) {
            const int p = RDM0Angular::Index(l, m), q = RDM0Angular::Index(l, -m);
            const double c = std::cos(m * angle), s = std::sin(m * angle);
            Close(rotated_d[p], c * d[p] - s * d[q], 6.0e-14, "real z rotation cos lane");
            Close(rotated_d[q], s * d[p] + c * d[q], 6.0e-14, "real z rotation sin lane");
        }
    }

    for (int lmax = 0; lmax <= 6; ++lmax) {
        double prefix[kCount + 1], prefix_ders[3 * kCount + 1], only_inplace[kCount + 1];
        std::fill(prefix, prefix + kCount + 1, 12345.0);
        std::fill(prefix_ders, prefix_ders + 3 * kCount + 1, 12345.0);
        std::fill(only_inplace, only_inplace + kCount + 1, 12345.0);
        RDM0Angular::Eval(x.data(), r, lmax, prefix, prefix_ders);
        RDM0Angular::Eval(x.data(), r, lmax, only_inplace, nullptr);
        const int count = (lmax + 1) * (lmax + 1);
        for (int i = 0; i < count; ++i) {
            Close(prefix[i], d[i], 0.0, "lmax prefix value");
            for (int a = 0; a < 3; ++a)
                Close(prefix_ders[3 * i + a], dd[3 * i + a], 0.0, "lmax prefix derivative");
        }
        Close(prefix[count], 12345.0, 0.0, "value buffer boundary");
        Close(prefix_ders[3 * count], 12345.0, 0.0, "derivative buffer boundary");

        // Caller-owned physical-Y buffers can be the D/dd output buffers.
        // Crucially ALL recurrence work finishes before ANY pole scaling:
        // scaling a lower-l slot during Eval would corrupt later recursion.
        // This validates that ownership/scheduling contract independently;
        // native CLI parity separately exercises actual EvalRDM0Physical.
        for (int l = 0; l <= lmax; ++l) {
            const double pole = RDM0Angular::PoleValue(l);
            for (int i = l * l; i < (l + 1) * (l + 1); ++i) {
                prefix[i] *= pole;
                only_inplace[i] *= pole;
                Close(prefix[i], pole * d[i], 0.0, "in-place physical value equals separate output");
                Close(only_inplace[i], prefix[i], 0.0, "in-place physical values-only parity");
                Close(prefix[i], reference[i], 5.0e-14, "in-place physical independent SH reference");
                for (int a = 0; a < 3; ++a) {
                    prefix_ders[3 * i + a] *= pole;
                    Close(prefix_ders[3 * i + a], pole * dd[3 * i + a], 0.0,
                          "in-place physical derivative equals separate output");
                    Close(prefix_ders[3 * i + a], reference_ders[3 * i + a], 5.0e-13 / r,
                          "in-place physical independent SH derivative reference");
                }
            }
        }
        Close(prefix[count], 12345.0, 0.0, "physical value buffer boundary");
        Close(only_inplace[count], 12345.0, 0.0, "physical values-only buffer boundary");
        Close(prefix_ders[3 * count], 12345.0, 0.0, "physical derivative buffer boundary");
        // Reuse already normalized buffers without clearing. Every active
        // value/derivative must be overwritten by the next complete Eval.
        RDM0Angular::Eval(x.data(), r, lmax, prefix, prefix_ders);
        for (int i = 0; i < count; ++i) {
            Close(prefix[i], d[i], 0.0, "reused physical buffer raw value");
            for (int a = 0; a < 3; ++a)
                Close(prefix_ders[3 * i + a], dd[3 * i + a], 0.0,
                      "reused physical buffer raw derivative");
        }
        ++physical_inplace_cases;
    }
}

void CheckBackendSelection(const std::string& expected_argument)
{
    using RDM0Angular::Backend;
    struct Case {
        const char* name;
        Backend backend;
        bool raw_reduction;
        bool new_values_kernel;
        bool new_derivative_kernel;
    };
    const Case cases[] = {
        {"ry", Backend::RY, false, false, false},
        {"ry-new", Backend::RYNew, false, true, true},
        {"hybrid", Backend::Hybrid, false, false, true},
        {"rd-m0", Backend::RDM0, true, true, true}
    };
    Require(RDM0Angular::ParseBackend(nullptr) == Backend::RY, "unset backend defaults to RY");
    for (const Case& row : cases)
        Require(RDM0Angular::ParseBackend(row.name) == row.backend, "exact backend parser mapping");
    for (const char* invalid : {"", "RY", "ry_new", "rd_m0", "hybrid ", "other"}) {
        bool rejected = false;
        try { RDM0Angular::ParseBackend(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "invalid backend token rejected");
    }
    const Backend selected = RDM0Angular::SelectedBackend();
    bool found = false;
    for (const Case& row : cases) {
        if (selected != row.backend) continue;
        found = true;
        if (!expected_argument.empty())
            Require(expected_argument == std::string("--expect-") + row.name, "selected backend matches requested process");
        Require(std::string(RDM0Angular::BackendName()) == row.name, "backend banner name");
        Require(RDM0Angular::Enabled() == row.raw_reduction, "only RD enables raw-D reduction/cache");
        Require(RDM0Angular::PhysicalKernelEnabled(false) == row.new_values_kernel,
                "values-only kernel dispatch truth table");
        Require(RDM0Angular::PhysicalKernelEnabled(true) == row.new_derivative_kernel,
                "derivative kernel dispatch truth table");
    }
    Require(found, "selected backend is known");
}

void CheckInvalidInput()
{
    const double x[3] = {1.0, 0.0, 0.0};
    double d[kCount];
    int failures = 0;
    const double radii[] = {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()};
    for (double r : radii) {
        try { RDM0Angular::Eval(x, r, 1, d, nullptr); }
        catch (const std::invalid_argument&) { ++failures; }
    }
    for (int l : {-1, 7}) {
        try { RDM0Angular::Eval(x, 1.0, l, d, nullptr); }
        catch (const std::invalid_argument&) { ++failures; }
        try { RDM0Angular::PoleValue(l); }
        catch (const std::invalid_argument&) { ++failures; }
    }
    try { RDM0Angular::Eval(nullptr, 1.0, 1, d, nullptr); }
    catch (const std::invalid_argument&) { ++failures; }
    try { RDM0Angular::Eval(x, 1.0, 1, nullptr, nullptr); }
    catch (const std::invalid_argument&) { ++failures; }
    Require(failures == 10, "invalid input must fail closed");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::string mode = argc == 2 ? argv[1] : "";
        Require(argc <= 2, "at most one expected-backend argument is accepted");
        if (mode == "--expect-invalid-env") {
            bool rejected = false;
            try { RDM0Angular::Enabled(); }
            catch (const std::invalid_argument&) { rejected = true; }
            Require(rejected, "invalid backend must not silently fall back");
            std::cout << "RD_M0_ENV_REJECTION_PASS\n";
            return 0;
        }
        Require(mode.empty() || mode == "--expect-ry" || mode == "--expect-rd-m0"
                || mode == "--expect-ry-new" || mode == "--expect-hybrid", "unknown test argument");
        CheckBackendSelection(mode);
        CheckInvalidInput();
        CheckPoles();
        std::vector<std::array<double, 3> > directions = {
            {{1.0, 0.0, 0.0}}, {{-1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}},
            {{0.0, -1.0, 0.0}}, {{0.0, 0.0, 1.0}}, {{0.0, 0.0, -1.0}},
            {{0.31, -0.72, 1.13}}, {{-1.1, 0.47, -0.29}}, {{1.0, 1.0, 1.0}},
            {{1.0e-14, -2.0e-14, 1.0}}, {{-2.0e-14, 1.0e-14, -1.0}},
            {{1.0e-7, 2.0e-7, 1.0}}, {{-2.0e-7, -1.0e-7, -1.0}}
        };
        for (int j = 0; j < 21; ++j)
            directions.push_back({{std::sin(0.71 * j + 0.17), std::cos(1.13 * j + 0.31),
                                    std::sin(0.47 * j - 0.62)}});
        for (const auto& x : directions) {
            for (double scale : {1.0e-8, 1.0, 1.0e8}) {
                const std::array<double, 3> scaled = {{scale * x[0], scale * x[1], scale * x[2]}};
                CheckClosedForms(scaled);
                CheckDirection(scaled);
            }
        }
        std::cout << std::setprecision(17) << "RD_M0_ANGULAR_PASS assertions=" << assertions
                  << " backend=" << RDM0Angular::BackendName()
                  << " direction_cases=" << directions.size() * 3
                  << " physical_inplace_cases=" << physical_inplace_cases
                  << " max_SH_value_error=" << max_reference_value_error
                  << " max_SH_derivative_error_times_r=" << max_reference_derivative_scaled_error
                  << " max_FD_error_times_r=" << max_fd_scaled_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RD_M0_ANGULAR_FAIL " << error.what() << '\n';
        return 1;
    }
}
