/* SUS2-SH real spherical-harmonic evaluation and analytic coefficient gradients. */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "mtpr.h"

#ifdef MLIP_INTEL_MKL
#	include <mkl_cblas.h>
#else
#	include <cblas.h>
#endif

using namespace std;

namespace {

// Keep the reference executable path available in the same binary. Auto uses
// a conservative measured EFS crossover, never a new mixed-gradient backend.
int PureWKernelMode()
{
    static const int mode = []() {
        const char* value = std::getenv("SUS2_SH_PURE_W_KERNEL");
        if (value == nullptr || std::strcmp(value, "auto") == 0) return -1;
        if (std::strcmp(value, "legacy") == 0) return 0;
        if (std::strcmp(value, "packed") == 0) return 1;
        if (std::strcmp(value, "basis") == 0) return 2;
        ERROR("SUS2_SH_PURE_W_KERNEL must be auto, legacy, packed or basis");
        return 0;
    }();
    return mode;
}

const double kPi = std::acos(-1.0);
const double kSqrt2 = std::sqrt(2.0);
const int kMaxSHL = 6;
const int kMaxSHComponents = (kMaxSHL + 1) * (kMaxSHL + 1);

const double kRealY00 = 0.5 / std::sqrt(kPi);
const double kRealY1 = 0.5 * std::sqrt(3.0 / kPi);
const double kRealY2A = 0.5 * std::sqrt(15.0 / kPi);
const double kRealY20 = 0.25 * std::sqrt(5.0 / kPi);
const double kRealY22 = 0.25 * std::sqrt(15.0 / kPi);
const double kRealY33 = 0.125 * std::sqrt(70.0 / kPi);
const double kRealY32 = 0.5 * std::sqrt(105.0 / kPi);
const double kRealY31 = 0.125 * std::sqrt(42.0 / kPi);
const double kRealY30 = 0.25 * std::sqrt(7.0 / kPi);
const double kRealY3p2 = 0.25 * std::sqrt(105.0 / kPi);
const double kRealY44m = 0.75 * std::sqrt(35.0 / kPi);
const double kRealY43 = 0.375 * std::sqrt(70.0 / kPi);
const double kRealY42m = 0.75 * std::sqrt(5.0 / kPi);
const double kRealY41 = 0.375 * std::sqrt(10.0 / kPi);
const double kRealY40 = 0.1875 / std::sqrt(kPi);
const double kRealY42 = 0.375 * std::sqrt(5.0 / kPi);
const double kRealY44 = 0.1875 * std::sqrt(35.0 / kPi);

inline int OuterTypeCoeffIndexFast(int radial_coeff_offset,
                                   int radial_coeff_block_size,
                                   int radial_basis_size,
                                   int type_outer,
                                   int mu)
{
	return radial_coeff_offset + mu * radial_coeff_block_size
		+ radial_basis_size + type_outer;
}

inline double OuterTypeCoeffFast(const std::vector<double>& coeffs,
                                 int radial_coeff_offset,
                                 int radial_coeff_block_size,
                                 int radial_basis_size,
                                 int type_outer,
                                 int mu)
{
	return coeffs[OuterTypeCoeffIndexFast(radial_coeff_offset,
	                                      radial_coeff_block_size,
	                                      radial_basis_size,
	                                      type_outer,
	                                      mu)];
}

inline int GateAdditiveCoeffIndexFast(int offset,
                                      int radial_func_count,
                                      int type_outer,
                                      int mu)
{
	return offset + type_outer * radial_func_count + mu;
}

// Pure W has shared input-channel scaling. Contract its output adjoints into
// coefficient space before applying the scaling chain rule. This replaces four
// separate (o,k,b) scaling-kernel dot products by two output contractions plus
// one (k,b) derivative pass. The scaling sum_o/sum_b and scalar factors are
// reassociated; each C coefficient retains the original edge accumulation order.
template <bool Coordinate, bool GateTangent>
void AccumulatePureWEdgeParameterGrad(
	int l, int L, int K, int O, int R, int species, int type_pair,
	const int* mu_to_eval, const int* eval_to_scaling,
	const double* matrix_coeffs, int matrix_offset,
	const double* rb_vals, const double* rb_ders,
	const double* value_weights, const double* coord_weights,
	const double* tangent_weights, const double* input, const double* input_t,
	const double* cached_kernel, const double* cached_kernel_der,
	double* grad, double* gate_value, double* gate_direct,
	double* scaling_scratch)
{
	for (int k = 0; k < K; ++k) {
		const int mu = k * L + l;
		const int eval = mu_to_eval[mu];
		const int rb_base = eval * R;
		const int der_base = 5 * rb_base;
		const double* phi = rb_vals + rb_base;
		const double* dphi = rb_ders + der_base;
		double* value_coeff_adjoint = scaling_scratch;
		double* coord_coeff_adjoint = scaling_scratch + R;
		std::fill(value_coeff_adjoint, value_coeff_adjoint + R, 0.0);
		if (Coordinate)
			std::fill(coord_coeff_adjoint, coord_coeff_adjoint + R, 0.0);
		double gate_value_sum = 0.0;
		double gate_direct_sum = 0.0;
		for (int o = 0; o < O; ++o) {
			const size_t row = (static_cast<size_t>(l) * O + o) * K + k;
			const double* coeff = matrix_coeffs + row * R;
			double* coeff_grad = grad + matrix_offset + row * R;
			const double value_factor = input[k] * value_weights[o]
				+ (GateTangent ? input_t[k] * tangent_weights[o] : 0.0);
			const double coord_factor = Coordinate ? input[k] * coord_weights[o] : 0.0;
			double kernel = cached_kernel == nullptr ? 0.0 : cached_kernel[row];
			double kernel_der = !Coordinate || cached_kernel_der == nullptr
				? 0.0 : cached_kernel_der[row];
			for (int b = 0; b < R; ++b) {
				const double c = coeff[b];
				coeff_grad[b] += phi[b] * value_factor
					+ (Coordinate ? dphi[b] * coord_factor : 0.0);
				value_coeff_adjoint[b] += c * value_factor;
				if (Coordinate)
					coord_coeff_adjoint[b] += c * coord_factor;
				if (cached_kernel == nullptr)
					kernel += c * phi[b];
				if (Coordinate && cached_kernel_der == nullptr)
					kernel_der += c * dphi[b];
			}
			gate_value_sum += kernel * value_weights[o]
				+ (Coordinate ? kernel_der * coord_weights[o] : 0.0);
			if (GateTangent)
				gate_direct_sum += kernel * tangent_weights[o];
		}
		double slope_grad = 0.0;
		double shift_grad = 0.0;
		for (int b = 0; b < R; ++b) {
			slope_grad += value_coeff_adjoint[b] * dphi[R + b]
				+ (Coordinate ? coord_coeff_adjoint[b] * dphi[2 * R + b] : 0.0);
			shift_grad += value_coeff_adjoint[b] * dphi[3 * R + b]
				+ (Coordinate ? coord_coeff_adjoint[b] * dphi[4 * R + b] : 0.0);
		}
		const int scale_index = species + 2 * species * species * eval_to_scaling[eval]
			+ type_pair;
		grad[scale_index] += slope_grad;
		grad[scale_index + species * species] += shift_grad;
		gate_value[mu] += gate_value_sum;
		if (GateTangent)
			gate_direct[mu] += gate_direct_sum;
	}
}

int SHFlatIndex(int l, int m)
{
	return l * l + (m + l);
}

double ParitySign(int m)
{
	return (std::abs(m) % 2) == 0 ? 1.0 : -1.0;
}

double RealSHValue(const std::complex<double>* values, int l, int m)
{
	if (m == 0)
		return values[SHFlatIndex(l, 0)].real();
	if (m > 0)
		return kSqrt2 * ParitySign(m) * values[SHFlatIndex(l, m)].real();
	return kSqrt2 * ParitySign(m) * values[SHFlatIndex(l, -m)].imag();
}

double RealSHDer(const std::complex<double>* ders, int l, int m, int a)
{
	if (m == 0)
		return ders[3 * SHFlatIndex(l, 0) + a].real();
	if (m > 0)
		return kSqrt2 * ParitySign(m) * ders[3 * SHFlatIndex(l, m) + a].real();
	return kSqrt2 * ParitySign(m) * ders[3 * SHFlatIndex(l, -m) + a].imag();
}

double RealSHValue(const double* values, int l, int m)
{
	return values[SHFlatIndex(l, m)];
}

double RealSHDer(const double* ders, int l, int m, int a)
{
	return ders[3 * SHFlatIndex(l, m) + a];
}

double SHCGFactorial(int n)
{
	if (n < 0)
		return 0.0;
	double value = 1.0;
	for (int i = 2; i <= n; ++i)
		value *= static_cast<double>(i);
	return value;
}

bool SHTriangle(int l1, int l2, int l3)
{
	return std::abs(l1 - l2) <= l3 && l3 <= l1 + l2;
}

double SHClebschGordan(int j1, int m1, int j2, int m2, int j, int m)
{
	if (m != m1 + m2)
		return 0.0;
	if (!SHTriangle(j1, j2, j))
		return 0.0;
	if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m) > j)
		return 0.0;

	const double pref1 = std::sqrt((2.0 * j + 1.0)
		* SHCGFactorial(j1 + j2 - j)
		* SHCGFactorial(j1 - j2 + j)
		* SHCGFactorial(-j1 + j2 + j)
		/ SHCGFactorial(j1 + j2 + j + 1));
	const double pref2 = std::sqrt(SHCGFactorial(j1 + m1)
		* SHCGFactorial(j1 - m1)
		* SHCGFactorial(j2 + m2)
		* SHCGFactorial(j2 - m2)
		* SHCGFactorial(j + m)
		* SHCGFactorial(j - m));

	double sum = 0.0;
	for (int k = 0; k <= 64; ++k) {
		const int a = j1 + j2 - j - k;
		const int b = j1 - m1 - k;
		const int c = j2 + m2 - k;
		const int d = j - j2 + m1 + k;
		const int e = j - j1 - m2 + k;
		if (a < 0 || b < 0 || c < 0 || d < 0 || e < 0)
			continue;
		const double term = ((k % 2) ? -1.0 : 1.0)
			/ (SHCGFactorial(k) * SHCGFactorial(a)
			   * SHCGFactorial(b) * SHCGFactorial(c)
			   * SHCGFactorial(d) * SHCGFactorial(e));
		sum += term;
	}
	return pref1 * pref2 * sum;
}

std::complex<double> SHRealFromComplexCoeff(int real_m, int complex_m)
{
	const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
	if (real_m == 0)
		return complex_m == 0 ? std::complex<double>(1.0, 0.0)
		                      : std::complex<double>(0.0, 0.0);
	const int a = std::abs(real_m);
	if (std::abs(complex_m) != a)
		return std::complex<double>(0.0, 0.0);
	if (real_m > 0) {
		if (complex_m == a)
			return inv_sqrt2 * ParitySign(a);
		return inv_sqrt2;
	}
	if (complex_m == a)
		return std::complex<double>(0.0, -inv_sqrt2 * ParitySign(a));
	return std::complex<double>(0.0, inv_sqrt2);
}

std::complex<double> SHRealCouplingPhase(int l1, int l2, int L)
{
	switch ((l1 + l2 - L) & 3) {
	case 0:
		return std::complex<double>(1.0, 0.0);
	case 1:
		return std::complex<double>(0.0, 1.0);
	case 2:
		return std::complex<double>(-1.0, 0.0);
	default:
		return std::complex<double>(0.0, -1.0);
	}
}

double SHRealCouplingCoeff(int l1, int rm1, int l2, int rm2, int L, int rM)
{
	std::complex<double> sum(0.0, 0.0);
	for (int M = -L; M <= L; ++M) {
		const std::complex<double> out_u =
			std::conj(SHRealFromComplexCoeff(rM, M));
		if (std::abs(out_u) == 0.0)
			continue;
		for (int m1 = -l1; m1 <= l1; ++m1) {
			const std::complex<double> in1 = SHRealFromComplexCoeff(rm1, m1);
			if (std::abs(in1) == 0.0)
				continue;
			for (int m2 = -l2; m2 <= l2; ++m2) {
				const std::complex<double> in2 =
					SHRealFromComplexCoeff(rm2, m2);
				if (std::abs(in2) == 0.0)
					continue;
				const double cg = SHClebschGordan(l1, m1, l2, m2, L, M);
				if (std::abs(cg) < 1.0e-14)
					continue;
				sum += out_u * cg * in1 * in2;
			}
		}
	}
	sum *= SHRealCouplingPhase(l1, l2, L);
	if (std::abs(sum.imag()) > 1.0e-10)
		ERROR("SUS2-SH effective-pair CG transform produced a non-real coefficient.");
	return sum.real();
}

std::vector<double> CoupleSHTensors(const std::vector<double>& left,
                                    int l_left,
                                    const std::vector<double>& right,
                                    int l_right,
                                    int l_out)
{
	std::vector<double> out(2 * l_out + 1, 0.0);
	for (int m_left = -l_left; m_left <= l_left; ++m_left) {
		const double left_value = left[m_left + l_left];
		if (left_value == 0.0)
			continue;
		for (int m_right = -l_right; m_right <= l_right; ++m_right) {
			const double right_value = right[m_right + l_right];
			if (right_value == 0.0)
				continue;
			const double product = left_value * right_value;
			for (int m_out = -l_out; m_out <= l_out; ++m_out) {
				const double coeff = SHRealCouplingCoeff(
					l_left, m_left, l_right, m_right, l_out, m_out);
				if (std::abs(coeff) < 1.0e-12)
					continue;
				out[m_out + l_out] += coeff * product;
			}
		}
	}
	return out;
}

void CoupleSHTensorArrays(const double* left,
                          int l_left,
                          const double* right,
                          int l_right,
                          int l_out,
                          double* out)
{
	std::fill(out, out + 2 * l_out + 1, 0.0);
	for (int m_left = -l_left; m_left <= l_left; ++m_left) {
		const double left_value = left[m_left + l_left];
		if (left_value == 0.0)
			continue;
		for (int m_right = -l_right; m_right <= l_right; ++m_right) {
			const double right_value = right[m_right + l_right];
			if (right_value == 0.0)
				continue;
			const double product = left_value * right_value;
			for (int m_out = -l_out; m_out <= l_out; ++m_out) {
				const double coeff = SHRealCouplingCoeff(
					l_left, m_left, l_right, m_right, l_out, m_out);
				if (std::abs(coeff) < 1.0e-12)
					continue;
				out[m_out + l_out] += coeff * product;
			}
		}
	}
}

bool SHUsesPrecomputedLmpTable(AnyRadialBasis* radial_basis)
{
	if (radial_basis == nullptr)
		return false;
	const std::string type = radial_basis->GetRBTypeString();
	return type == "RBChebyshev_sss_lmp"
	    || type == "RBChebyshev_sss_rational_lmp"
	    || type == "RBChebyshev_ss_lmp"
	    || type == "RBChebyshev_ssw_lmp"
	    || type == "RBChebyshev_sssw_lmp"
	    || type == "RBChebyshev_s_lmp"
	    || type == "RBLaguerre_log1p_lmp"
	    || type == "RBLaguerre_log1p_pos_lmp"
	    || type == "RBLaguerre_log1p_noenv_lmp"
	    || type == "RBJacobi_sss_lmp"
	    || type == "RBJacobi_sss_noweight_lmp";
}

bool DisableTwoLayerEdgePrimitiveCache()
{
	const char* value = std::getenv("SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool DisableTwoLayerGateTanhCache()
{
	static const bool disabled = []() {
		const char* value = std::getenv("SUS2_SH_DISABLE_TWO_LAYER_GATE_TANH_CACHE");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return disabled;
}

bool EnvFlagEnabled(const char* name)
{
	const char* env = std::getenv(name);
	if (env == nullptr)
		return false;
	const std::string value(env);
	return !(value == "0" || value == "false" || value == "False");
}

bool AdaptiveRadialFullDerivativeCacheEnabled()
{
	return EnvFlagEnabled("SUS2_SH_ADAPTIVE_RADIAL_DR_FULL_CACHE");
}

bool AdaptiveRadialEFSReferenceEnabled()
{
	return EnvFlagEnabled("SUS2_SH_ADAPTIVE_EFS_REFERENCE");
}

bool EdgeL1PairCacheEnabled()
{
	return EnvFlagEnabled("SUS2_SH_EDGE_L1_PAIR_CACHE")
		|| EnvFlagEnabled("SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE");
}

bool EdgeL1PairCacheDisabled()
{
	return EnvFlagEnabled("SUS2_SH_DISABLE_EDGE_L1_PAIR_CACHE");
}

bool RuntimeRankIsZero()
{
	const char* rank_envs[] = {
		"PMI_RANK",
		"OMPI_COMM_WORLD_RANK",
		"MV2_COMM_WORLD_RANK",
		"MPI_RANKID"
	};
	for (const char* rank_env : rank_envs) {
		const char* rank_value = std::getenv(rank_env);
		if (rank_value != nullptr)
			return std::atoi(rank_value) == 0;
	}
	return true;
}

bool EdgeL1PairCacheProfileEnabledOnRank0()
{
	return EnvFlagEnabled("SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE")
		&& RuntimeRankIsZero();
}

bool BasicMuForwardProfileEnabledOnRank0()
{
	return EnvFlagEnabled("SUS2_SH_BASIC_MU_FORWARD_PROFILE")
		&& RuntimeRankIsZero();
}

void InterpolateSHLmpMuTable(Array3D& value_table,
                             Array3D& der_table,
                             double inv_dr,
                             int species_count,
                             int type_central,
                             int type_outer,
                             double r,
                             int radial_func_count,
                             double* values,
                             double* ders)
{
	if (value_table.size1 != species_count * species_count
	    || value_table.size2 < 2
	    || value_table.size3 < radial_func_count)
		ERROR("SUS2-SH _lmp radial value table is not initialized");
	if (ders != nullptr
	    && (der_table.size1 != value_table.size1
	        || der_table.size2 != value_table.size2
	        || der_table.size3 < radial_func_count))
		ERROR("SUS2-SH _lmp radial derivative table is not initialized");
	if (type_central < 0 || type_central >= species_count
	    || type_outer < 0 || type_outer >= species_count)
		ERROR("SUS2-SH _lmp radial table type index is out of range");
	const int pair_index = species_count * type_central + type_outer;
	double grid_pos = r * inv_dr;
	int r_list = static_cast<int>(std::floor(grid_pos));
	const int last_interval = value_table.size2 - 2;
	if (r_list < 0)
		r_list = 0;
	if (r_list > last_interval)
		r_list = last_interval;
	const int r_next = r_list + 1;
	double ddr = grid_pos - r_list;
	if (ddr < 0.0)
		ddr = 0.0;
	if (ddr > 1.0)
		ddr = 1.0;
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double v1 = value_table(pair_index, r_list, mu);
		const double v2 = value_table(pair_index, r_next, mu);
		values[mu] = v1 + ddr * (v2 - v1);
		if (ders != nullptr) {
			const double d1 = der_table(pair_index, r_list, mu);
			const double d2 = der_table(pair_index, r_next, mu);
			ders[mu] = d1 + ddr * (d2 - d1);
		}
	}
}

double SHInvPower(int l, double r)
{
	const double inv_r = 1.0 / r;
	const double inv_r2 = inv_r * inv_r;
	if (l == 0)
		return 1.0;
	if (l == 1)
		return inv_r;
	if (l == 2)
		return inv_r2;
	if (l == 3)
		return inv_r2 * inv_r;
	if (l == 4)
		return inv_r2 * inv_r2;
	return std::pow(r, -l);
}

void AddSHPolynomial(int l,
	                     int m,
	                     double coeff,
	                     const std::complex<double>& poly,
	                     const std::complex<double> dpoly[3],
                     const Vector3& rvec,
                     double r,
                     std::complex<double>* values,
                     std::complex<double>* ders)
{
	const int idx = SHFlatIndex(l, m);
	const double inv_r = 1.0 / r;
	const double inv_pow = SHInvPower(l, r);
	const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow * inv_r * inv_r;
	values[idx] = coeff * poly * inv_pow;
	for (int a = 0; a < 3; ++a)
		ders[3 * idx + a] = coeff * (dpoly[a] * inv_pow + poly * inv_pow_der * rvec[a]);
}

void AddSHPolynomialValueOnly(int l,
                              int m,
                              double coeff,
                              const std::complex<double>& poly,
                              double r,
                              std::complex<double>* values)
{
	values[SHFlatIndex(l, m)] = coeff * poly * SHInvPower(l, r);
}

void AddRealSH(int l,
               int m,
               double coeff,
               double poly,
               double dpx,
               double dpy,
               double dpz,
               const Vector3& rvec,
               double r,
               double* values,
               double* ders)
{
	const int idx = SHFlatIndex(l, m);
	const double inv_r = 1.0 / r;
	const double inv_pow = SHInvPower(l, r);
	const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow * inv_r * inv_r;
	values[idx] = coeff * poly * inv_pow;
	ders[3 * idx + 0] = coeff * (dpx * inv_pow + poly * inv_pow_der * rvec[0]);
	ders[3 * idx + 1] = coeff * (dpy * inv_pow + poly * inv_pow_der * rvec[1]);
	ders[3 * idx + 2] = coeff * (dpz * inv_pow + poly * inv_pow_der * rvec[2]);
}

void AddRealSHValueOnly(int l, int m, double coeff, double poly, double r, double* values)
{
	values[SHFlatIndex(l, m)] = coeff * poly * SHInvPower(l, r);
}

double Factorial(int n)
{
	double value = 1.0;
	for (int i = 2; i <= n; ++i)
		value *= static_cast<double>(i);
	return value;
}

double ComplexSHNorm(int l, int m)
{
	return std::sqrt((2.0 * l + 1.0) / (4.0 * kPi) * Factorial(l - m) / Factorial(l + m));
}

void EvalRealSHFromSolid(const Vector3& rvec,
                         double r,
                         int lmax,
                         double* values,
                         double* ders)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = 0.0;
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = 0.0;
	}

	std::complex<double> solid[kMaxSHComponents];
	std::complex<double> solid_ders[3 * kMaxSHComponents];
	for (int i = 0; i < kMaxSHComponents; ++i) {
		solid[i] = std::complex<double>(0.0, 0.0);
		for (int a = 0; a < 3; ++a)
			solid_ders[3 * i + a] = std::complex<double>(0.0, 0.0);
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double r2 = r * r;
	const std::complex<double> u(x, y);
	const std::complex<double> du[3] = {
		std::complex<double>(1.0, 0.0),
		std::complex<double>(0.0, 1.0),
		std::complex<double>(0.0, 0.0)
	};

	solid[SHFlatIndex(0, 0)] = std::complex<double>(1.0, 0.0);
	for (int m = 1; m <= lmax; ++m) {
		const int prev = SHFlatIndex(m - 1, m - 1);
		const int idx = SHFlatIndex(m, m);
		const double coeff = -static_cast<double>(2 * m - 1);
		solid[idx] = coeff * u * solid[prev];
		for (int a = 0; a < 3; ++a)
			solid_ders[3 * idx + a] = coeff * (du[a] * solid[prev] + u * solid_ders[3 * prev + a]);
	}
	for (int m = 0; m <= lmax; ++m) {
		const int diag = SHFlatIndex(m, m);
		if (m + 1 <= lmax) {
			const int idx = SHFlatIndex(m + 1, m);
			const double coeff = static_cast<double>(2 * m + 1);
			solid[idx] = coeff * z * solid[diag];
			for (int a = 0; a < 3; ++a)
				solid_ders[3 * idx + a] = coeff * (
					(a == 2 ? solid[diag] : std::complex<double>(0.0, 0.0))
					+ z * solid_ders[3 * diag + a]);
		}
		for (int l = m + 2; l <= lmax; ++l) {
			const int idx = SHFlatIndex(l, m);
			const int prev1 = SHFlatIndex(l - 1, m);
			const int prev2 = SHFlatIndex(l - 2, m);
			const double acoef = static_cast<double>(2 * l - 1);
			const double bcoef = static_cast<double>(l + m - 1);
			const double denom = static_cast<double>(l - m);
			solid[idx] = (acoef * z * solid[prev1] - bcoef * r2 * solid[prev2]) / denom;
			for (int a = 0; a < 3; ++a) {
				const double dr2 = 2.0 * rvec[a];
				solid_ders[3 * idx + a] = (
					acoef * ((a == 2 ? solid[prev1] : std::complex<double>(0.0, 0.0))
					         + z * solid_ders[3 * prev1 + a])
					- bcoef * (dr2 * solid[prev2] + r2 * solid_ders[3 * prev2 + a])) / denom;
			}
		}
	}

	for (int l = 0; l <= lmax; ++l) {
		const double inv_pow = SHInvPower(l, r);
		const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow / (r * r);
		for (int m = 0; m <= l; ++m) {
			const int cidx = SHFlatIndex(l, m);
			const double norm = ComplexSHNorm(l, m);
			const std::complex<double> y_complex = norm * solid[cidx] * inv_pow;
			std::complex<double> dy_complex[3];
			for (int a = 0; a < 3; ++a)
				dy_complex[a] = norm * (solid_ders[3 * cidx + a] * inv_pow
				                         + solid[cidx] * inv_pow_der * rvec[a]);

			if (m == 0) {
				const int ridx = SHFlatIndex(l, 0);
				values[ridx] = y_complex.real();
				for (int a = 0; a < 3; ++a)
					ders[3 * ridx + a] = dy_complex[a].real();
			} else {
				const double sign = ParitySign(m);
				const int pidx = SHFlatIndex(l, m);
				const int nidx = SHFlatIndex(l, -m);
				values[pidx] = kSqrt2 * sign * y_complex.real();
				values[nidx] = kSqrt2 * sign * y_complex.imag();
				for (int a = 0; a < 3; ++a) {
					ders[3 * pidx + a] = kSqrt2 * sign * dy_complex[a].real();
					ders[3 * nidx + a] = kSqrt2 * sign * dy_complex[a].imag();
				}
				}
		}
	}
}

void EvalRealSHValuesOnlyFromSolid(const Vector3& rvec,
                                   double r,
                                   int lmax,
                                   double* values)
{
	double ders[3 * kMaxSHComponents];
	EvalRealSHFromSolid(rvec, r, lmax, values, ders);
}

void EvalComplexSH(const Vector3& rvec,
                   double r,
                   int lmax,
                   std::complex<double>* values,
                   std::complex<double>* ders)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = std::complex<double>(0.0, 0.0);
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = std::complex<double>(0.0, 0.0);
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const std::complex<double> I(0.0, 1.0);
	const std::complex<double> u(x, y);
	const std::complex<double> v(x, -y);
	std::complex<double> dpoly[3];

	dpoly[0] = dpoly[1] = dpoly[2] = 0.0;
	AddSHPolynomial(0, 0, 0.5 / std::sqrt(kPi), 1.0, dpoly, rvec, r, values, ders);
	if (lmax == 0)
		return;

	dpoly[0] = 1.0; dpoly[1] = -I; dpoly[2] = 0.0;
	AddSHPolynomial(1, -1, 0.5 * std::sqrt(3.0 / (2.0 * kPi)), v, dpoly, rvec, r, values, ders);
	dpoly[0] = 0.0; dpoly[1] = 0.0; dpoly[2] = 1.0;
	AddSHPolynomial(1, 0, 0.5 * std::sqrt(3.0 / kPi), z, dpoly, rvec, r, values, ders);
	dpoly[0] = 1.0; dpoly[1] = I; dpoly[2] = 0.0;
	AddSHPolynomial(1, 1, -0.5 * std::sqrt(3.0 / (2.0 * kPi)), u, dpoly, rvec, r, values, ders);
	if (lmax == 1)
		return;

	const std::complex<double> u2 = u * u;
	const std::complex<double> v2 = v * v;
	dpoly[0] = 2.0 * v; dpoly[1] = -2.0 * I * v; dpoly[2] = 0.0;
	AddSHPolynomial(2, -2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), v2, dpoly, rvec, r, values, ders);
	dpoly[0] = z; dpoly[1] = -I * z; dpoly[2] = v;
	AddSHPolynomial(2, -1, 0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * v, dpoly, rvec, r, values, ders);
	const std::complex<double> p20(2.0 * z * z - x * x - y * y, 0.0);
	dpoly[0] = -2.0 * x; dpoly[1] = -2.0 * y; dpoly[2] = 4.0 * z;
	AddSHPolynomial(2, 0, 0.25 * std::sqrt(5.0 / kPi), p20, dpoly, rvec, r, values, ders);
	dpoly[0] = z; dpoly[1] = I * z; dpoly[2] = u;
	AddSHPolynomial(2, 1, -0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * u, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * u; dpoly[1] = 2.0 * I * u; dpoly[2] = 0.0;
	AddSHPolynomial(2, 2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), u2, dpoly, rvec, r, values, ders);
	if (lmax == 2)
		return;

	const std::complex<double> u3 = u2 * u;
	const std::complex<double> v3 = v2 * v;
	dpoly[0] = 3.0 * v2; dpoly[1] = -3.0 * I * v2; dpoly[2] = 0.0;
	AddSHPolynomial(3, -3, 0.125 * std::sqrt(35.0 / kPi), v3, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * z * v; dpoly[1] = -2.0 * I * z * v; dpoly[2] = v2;
	AddSHPolynomial(3, -2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * v2, dpoly, rvec, r, values, ders);
	const std::complex<double> a31(4.0 * z * z - x * x - y * y, 0.0);
	dpoly[0] = a31 - 2.0 * x * v; dpoly[1] = -I * a31 - 2.0 * y * v; dpoly[2] = 8.0 * z * v;
	AddSHPolynomial(3, -1, 0.125 * std::sqrt(21.0 / kPi), v * a31, dpoly, rvec, r, values, ders);
	const std::complex<double> p30(z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y), 0.0);
	dpoly[0] = -6.0 * x * z; dpoly[1] = -6.0 * y * z; dpoly[2] = 6.0 * z * z - 3.0 * x * x - 3.0 * y * y;
	AddSHPolynomial(3, 0, 0.25 * std::sqrt(7.0 / kPi), p30, dpoly, rvec, r, values, ders);
	dpoly[0] = a31 - 2.0 * x * u; dpoly[1] = I * a31 - 2.0 * y * u; dpoly[2] = 8.0 * z * u;
	AddSHPolynomial(3, 1, -0.125 * std::sqrt(21.0 / kPi), u * a31, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * z * u; dpoly[1] = 2.0 * I * z * u; dpoly[2] = u2;
	AddSHPolynomial(3, 2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * u2, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * u2; dpoly[1] = 3.0 * I * u2; dpoly[2] = 0.0;
	AddSHPolynomial(3, 3, -0.125 * std::sqrt(35.0 / kPi), u3, dpoly, rvec, r, values, ders);
	if (lmax == 3)
		return;

	const double rho2 = x * x + y * y;
	const std::complex<double> u4 = u3 * u;
	const std::complex<double> v4 = v3 * v;

	dpoly[0] = 4.0 * v3; dpoly[1] = -4.0 * I * v3; dpoly[2] = 0.0;
	AddSHPolynomial(4, -4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), v4, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * z * v2; dpoly[1] = -3.0 * I * z * v2; dpoly[2] = v3;
	AddSHPolynomial(4, -3, 0.375 * std::sqrt(35.0 / kPi), z * v3, dpoly, rvec, r, values, ders);
	const std::complex<double> a42(6.0 * z * z - rho2, 0.0);
	dpoly[0] = 2.0 * v * a42 - 2.0 * x * v2;
	dpoly[1] = -2.0 * I * v * a42 - 2.0 * y * v2;
	dpoly[2] = 12.0 * z * v2;
	AddSHPolynomial(4, -2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), v2 * a42, dpoly, rvec, r, values, ders);
	const std::complex<double> a41(4.0 * z * z - 3.0 * rho2, 0.0);
	dpoly[0] = z * (a41 - 6.0 * x * v);
	dpoly[1] = z * (-I * a41 - 6.0 * y * v);
	dpoly[2] = v * (12.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, -1, 0.375 * std::sqrt(5.0 / kPi), z * v * a41, dpoly, rvec, r, values, ders);
	const std::complex<double> p40(8.0 * z * z * z * z - 24.0 * z * z * rho2 + 3.0 * rho2 * rho2, 0.0);
	dpoly[0] = 12.0 * x * (rho2 - 4.0 * z * z);
	dpoly[1] = 12.0 * y * (rho2 - 4.0 * z * z);
	dpoly[2] = 16.0 * z * (2.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, 0, 0.1875 / std::sqrt(kPi), p40, dpoly, rvec, r, values, ders);
	dpoly[0] = z * (a41 - 6.0 * x * u);
	dpoly[1] = z * (I * a41 - 6.0 * y * u);
	dpoly[2] = u * (12.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, 1, -0.375 * std::sqrt(5.0 / kPi), z * u * a41, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * u * a42 - 2.0 * x * u2;
	dpoly[1] = 2.0 * I * u * a42 - 2.0 * y * u2;
	dpoly[2] = 12.0 * z * u2;
	AddSHPolynomial(4, 2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), u2 * a42, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * z * u2; dpoly[1] = 3.0 * I * z * u2; dpoly[2] = u3;
	AddSHPolynomial(4, 3, -0.375 * std::sqrt(35.0 / kPi), z * u3, dpoly, rvec, r, values, ders);
	dpoly[0] = 4.0 * u3; dpoly[1] = 4.0 * I * u3; dpoly[2] = 0.0;
	AddSHPolynomial(4, 4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), u4, dpoly, rvec, r, values, ders);
}

void EvalComplexSHValuesOnly(const Vector3& rvec,
                             double r,
                             int lmax,
                             std::complex<double>* values)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i)
		values[i] = std::complex<double>(0.0, 0.0);

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const std::complex<double> u(x, y);
	const std::complex<double> v(x, -y);

	AddSHPolynomialValueOnly(0, 0, 0.5 / std::sqrt(kPi), 1.0, r, values);
	if (lmax == 0)
		return;

	AddSHPolynomialValueOnly(1, -1, 0.5 * std::sqrt(3.0 / (2.0 * kPi)), v, r, values);
	AddSHPolynomialValueOnly(1, 0, 0.5 * std::sqrt(3.0 / kPi), z, r, values);
	AddSHPolynomialValueOnly(1, 1, -0.5 * std::sqrt(3.0 / (2.0 * kPi)), u, r, values);
	if (lmax == 1)
		return;

	const std::complex<double> u2 = u * u;
	const std::complex<double> v2 = v * v;
	AddSHPolynomialValueOnly(2, -2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), v2, r, values);
	AddSHPolynomialValueOnly(2, -1, 0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * v, r, values);
	const std::complex<double> p20(2.0 * z * z - x * x - y * y, 0.0);
	AddSHPolynomialValueOnly(2, 0, 0.25 * std::sqrt(5.0 / kPi), p20, r, values);
	AddSHPolynomialValueOnly(2, 1, -0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * u, r, values);
	AddSHPolynomialValueOnly(2, 2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), u2, r, values);
	if (lmax == 2)
		return;

	const std::complex<double> u3 = u2 * u;
	const std::complex<double> v3 = v2 * v;
	AddSHPolynomialValueOnly(3, -3, 0.125 * std::sqrt(35.0 / kPi), v3, r, values);
	AddSHPolynomialValueOnly(3, -2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * v2, r, values);
	const std::complex<double> a31(4.0 * z * z - x * x - y * y, 0.0);
	AddSHPolynomialValueOnly(3, -1, 0.125 * std::sqrt(21.0 / kPi), v * a31, r, values);
	const std::complex<double> p30(z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y), 0.0);
	AddSHPolynomialValueOnly(3, 0, 0.25 * std::sqrt(7.0 / kPi), p30, r, values);
	AddSHPolynomialValueOnly(3, 1, -0.125 * std::sqrt(21.0 / kPi), u * a31, r, values);
	AddSHPolynomialValueOnly(3, 2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * u2, r, values);
	AddSHPolynomialValueOnly(3, 3, -0.125 * std::sqrt(35.0 / kPi), u3, r, values);
	if (lmax == 3)
		return;

	const double rho2 = x * x + y * y;
	const std::complex<double> u4 = u3 * u;
	const std::complex<double> v4 = v3 * v;
	AddSHPolynomialValueOnly(4, -4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), v4, r, values);
	AddSHPolynomialValueOnly(4, -3, 0.375 * std::sqrt(35.0 / kPi), z * v3, r, values);
	const std::complex<double> a42(6.0 * z * z - rho2, 0.0);
	AddSHPolynomialValueOnly(4, -2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), v2 * a42, r, values);
	const std::complex<double> a41(4.0 * z * z - 3.0 * rho2, 0.0);
	AddSHPolynomialValueOnly(4, -1, 0.375 * std::sqrt(5.0 / kPi), z * v * a41, r, values);
	const std::complex<double> p40(8.0 * z * z * z * z - 24.0 * z * z * rho2 + 3.0 * rho2 * rho2, 0.0);
	AddSHPolynomialValueOnly(4, 0, 0.1875 / std::sqrt(kPi), p40, r, values);
	AddSHPolynomialValueOnly(4, 1, -0.375 * std::sqrt(5.0 / kPi), z * u * a41, r, values);
	AddSHPolynomialValueOnly(4, 2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), u2 * a42, r, values);
	AddSHPolynomialValueOnly(4, 3, -0.375 * std::sqrt(35.0 / kPi), z * u3, r, values);
	AddSHPolynomialValueOnly(4, 4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), u4, r, values);
}

void EvalRealSH(const Vector3& rvec,
                double r,
                int lmax,
                double* values,
                double* ders)
{
	if (lmax > 4) {
		EvalRealSHFromSolid(rvec, r, lmax, values, ders);
		return;
	}

	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = 0.0;
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = 0.0;
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double x2 = x * x;
	const double y2 = y * y;
	const double z2 = z * z;

	AddRealSH(0, 0, kRealY00, 1.0, 0.0, 0.0, 0.0, rvec, r, values, ders);
	if (lmax == 0)
		return;

	AddRealSH(1, -1, kRealY1, y, 0.0, 1.0, 0.0, rvec, r, values, ders);
	AddRealSH(1, 0, kRealY1, z, 0.0, 0.0, 1.0, rvec, r, values, ders);
	AddRealSH(1, 1, kRealY1, x, 1.0, 0.0, 0.0, rvec, r, values, ders);
	if (lmax == 1)
		return;

	AddRealSH(2, -2, kRealY2A, x * y, y, x, 0.0, rvec, r, values, ders);
	AddRealSH(2, -1, kRealY2A, y * z, 0.0, z, y, rvec, r, values, ders);
	const double p20 = 2.0 * z2 - x2 - y2;
	AddRealSH(2, 0, kRealY20, p20, -2.0 * x, -2.0 * y, 4.0 * z, rvec, r, values, ders);
	AddRealSH(2, 1, kRealY2A, x * z, z, 0.0, x, rvec, r, values, ders);
	AddRealSH(2, 2, kRealY22, x2 - y2, 2.0 * x, -2.0 * y, 0.0, rvec, r, values, ders);
	if (lmax == 2)
		return;

	const double a31 = 4.0 * z2 - x2 - y2;
	const double p3m3 = 3.0 * x2 * y - y * y2;
	AddRealSH(3, -3, kRealY33, p3m3, 6.0 * x * y, 3.0 * x2 - 3.0 * y2, 0.0, rvec, r, values, ders);
	AddRealSH(3, -2, kRealY32, x * y * z, y * z, x * z, x * y, rvec, r, values, ders);
	AddRealSH(3, -1, kRealY31, y * a31, -2.0 * x * y, a31 - 2.0 * y2, 8.0 * y * z, rvec, r, values, ders);
	const double p30 = z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
	AddRealSH(3, 0, kRealY30, p30, -6.0 * x * z, -6.0 * y * z, 6.0 * z2 - 3.0 * x2 - 3.0 * y2, rvec, r, values, ders);
	AddRealSH(3, 1, kRealY31, x * a31, a31 - 2.0 * x2, -2.0 * x * y, 8.0 * x * z, rvec, r, values, ders);
	const double p32 = z * (x2 - y2);
	AddRealSH(3, 2, kRealY3p2, p32, 2.0 * x * z, -2.0 * y * z, x2 - y2, rvec, r, values, ders);
	const double p33 = x * x2 - 3.0 * x * y2;
	AddRealSH(3, 3, kRealY33, p33, 3.0 * x2 - 3.0 * y2, -6.0 * x * y, 0.0, rvec, r, values, ders);
	if (lmax == 3)
		return;

	const double rho2 = x2 + y2;
	const double a42 = 6.0 * z2 - rho2;
	const double a41 = 4.0 * z2 - 3.0 * rho2;
	const double p44base = x2 - y2;
	const double p4m4 = x * y * p44base;
	AddRealSH(4, -4, kRealY44m, p4m4, y * (3.0 * x2 - y2), x * (x2 - 3.0 * y2), 0.0, rvec, r, values, ders);
	AddRealSH(4, -3, kRealY43, z * p3m3, 6.0 * x * y * z, z * (3.0 * x2 - 3.0 * y2), p3m3, rvec, r, values, ders);
	AddRealSH(4, -2, kRealY42m, x * y * a42, y * a42 - 2.0 * x2 * y, x * a42 - 2.0 * x * y2, 12.0 * x * y * z, rvec, r, values, ders);
	AddRealSH(4, -1, kRealY41, y * z * a41, -6.0 * x * y * z, z * (a41 - 6.0 * y2), y * (12.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	const double p40 = 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2;
	AddRealSH(4, 0, kRealY40, p40, 12.0 * x * (rho2 - 4.0 * z2), 12.0 * y * (rho2 - 4.0 * z2), 16.0 * z * (2.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	AddRealSH(4, 1, kRealY41, x * z * a41, z * (a41 - 6.0 * x2), -6.0 * x * y * z, x * (12.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	AddRealSH(4, 2, kRealY42, p44base * a42, 2.0 * x * a42 - 2.0 * x * p44base, -2.0 * y * a42 - 2.0 * y * p44base, 12.0 * z * p44base, rvec, r, values, ders);
	AddRealSH(4, 3, kRealY43, z * p33, z * (3.0 * x2 - 3.0 * y2), -6.0 * x * y * z, p33, rvec, r, values, ders);
	const double p44 = x2 * x2 - 6.0 * x2 * y2 + y2 * y2;
	AddRealSH(4, 4, kRealY44, p44, 4.0 * x * x2 - 12.0 * x * y2, -12.0 * x2 * y + 4.0 * y * y2, 0.0, rvec, r, values, ders);
}

void EvalRealSHValuesOnly(const Vector3& rvec,
                          double r,
                          int lmax,
                          double* values)
{
	if (lmax > 4) {
		EvalRealSHValuesOnlyFromSolid(rvec, r, lmax, values);
		return;
	}

	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i)
		values[i] = 0.0;

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double x2 = x * x;
	const double y2 = y * y;
	const double z2 = z * z;

	AddRealSHValueOnly(0, 0, kRealY00, 1.0, r, values);
	if (lmax == 0)
		return;
	AddRealSHValueOnly(1, -1, kRealY1, y, r, values);
	AddRealSHValueOnly(1, 0, kRealY1, z, r, values);
	AddRealSHValueOnly(1, 1, kRealY1, x, r, values);
	if (lmax == 1)
		return;
	AddRealSHValueOnly(2, -2, kRealY2A, x * y, r, values);
	AddRealSHValueOnly(2, -1, kRealY2A, y * z, r, values);
	AddRealSHValueOnly(2, 0, kRealY20, 2.0 * z2 - x2 - y2, r, values);
	AddRealSHValueOnly(2, 1, kRealY2A, x * z, r, values);
	AddRealSHValueOnly(2, 2, kRealY22, x2 - y2, r, values);
	if (lmax == 2)
		return;
	const double a31 = 4.0 * z2 - x2 - y2;
	const double p3m3 = 3.0 * x2 * y - y * y2;
	const double p33 = x * x2 - 3.0 * x * y2;
	AddRealSHValueOnly(3, -3, kRealY33, p3m3, r, values);
	AddRealSHValueOnly(3, -2, kRealY32, x * y * z, r, values);
	AddRealSHValueOnly(3, -1, kRealY31, y * a31, r, values);
	AddRealSHValueOnly(3, 0, kRealY30, z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2), r, values);
	AddRealSHValueOnly(3, 1, kRealY31, x * a31, r, values);
	AddRealSHValueOnly(3, 2, kRealY3p2, z * (x2 - y2), r, values);
	AddRealSHValueOnly(3, 3, kRealY33, p33, r, values);
	if (lmax == 3)
		return;
	const double rho2 = x2 + y2;
	const double a42 = 6.0 * z2 - rho2;
	const double a41 = 4.0 * z2 - 3.0 * rho2;
	const double p44base = x2 - y2;
	AddRealSHValueOnly(4, -4, kRealY44m, x * y * p44base, r, values);
	AddRealSHValueOnly(4, -3, kRealY43, z * p3m3, r, values);
	AddRealSHValueOnly(4, -2, kRealY42m, x * y * a42, r, values);
	AddRealSHValueOnly(4, -1, kRealY41, y * z * a41, r, values);
	AddRealSHValueOnly(4, 0, kRealY40, 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2, r, values);
	AddRealSHValueOnly(4, 1, kRealY41, x * z * a41, r, values);
	AddRealSHValueOnly(4, 2, kRealY42, p44base * a42, r, values);
	AddRealSHValueOnly(4, 3, kRealY43, z * p33, r, values);
	AddRealSHValueOnly(4, 4, kRealY44, x2 * x2 - 6.0 * x2 * y2 + y2 * y2, r, values);
}

} // namespace

void MLMTPR::InvalidateTwoLayerGateTanhMuCache()
{
	if (!two_layer_gate_tanh_mu_cache_valid_.empty())
		std::fill(two_layer_gate_tanh_mu_cache_valid_.begin(),
		          two_layer_gate_tanh_mu_cache_valid_.end(), 0);
	two_layer_gate_center_mu_buffer_atom_index_ = -1;
	two_layer_gate_center_mu_buffer_atom_type_ = -1;
	two_layer_gate_center_mu_buffer_mask_ = 0;
	two_layer_gate_center_mu_buffer_signal_ = nullptr;
}

void MLMTPR::PrepareTwoLayerGateNeighborMuBuffers(int type_outer,
                                                  int type_center,
                                                  const double* gate_signal_by_mu,
                                                  int neighbor_atom_index,
                                                  int buffer_mask)
{
	const bool need_additive =
		(buffer_mask & kGateMuBufferAdditive) != 0;
	const bool need_type_scale =
		(buffer_mask & kGateMuBufferTypeScale) != 0;
	const bool need_multiplier =
		(buffer_mask & kGateMuBufferMultiplier) != 0;
	const bool need_outer =
		(buffer_mask & kGateMuBufferOuter) != 0;
	const bool need_additive_param =
		(buffer_mask & kGateMuBufferAdditiveParam) != 0;
	const bool need_gate_residual_param =
		(buffer_mask & kGateMuBufferGateResidualParam) != 0;
	const bool need_gate_outer_param =
		(buffer_mask & kGateMuBufferGateOuterParam) != 0;
	const bool need_gate_additive_param =
		(buffer_mask & kGateMuBufferGateAdditiveParam) != 0;

	if (need_additive
	    && static_cast<int>(two_layer_gate_additive_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_additive_mu_buffer_.resize(radial_func_count);
	if (need_type_scale
	    && static_cast<int>(two_layer_gate_type_scale_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_type_scale_mu_buffer_.resize(radial_func_count);
	if (need_multiplier
	    && static_cast<int>(two_layer_gate_multiplier_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_multiplier_mu_buffer_.resize(radial_func_count);
	if (need_outer
	    && static_cast<int>(two_layer_gate_outer_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_outer_mu_buffer_.resize(radial_func_count);
	if (need_additive_param
	    && static_cast<int>(two_layer_gate_additive_param_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_additive_param_mu_buffer_.resize(radial_func_count);
	if (need_gate_residual_param
	    && static_cast<int>(two_layer_gate_gate_residual_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_gate_residual_mu_buffer_.resize(radial_func_count);
	if (need_gate_outer_param
	    && static_cast<int>(two_layer_gate_gate_outer_param_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_gate_outer_param_mu_buffer_.resize(radial_func_count);
	if (need_gate_additive_param
	    && static_cast<int>(two_layer_gate_gate_additive_param_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_gate_additive_param_mu_buffer_.resize(radial_func_count);

	double* additive_by_mu = need_additive ? two_layer_gate_additive_mu_buffer_.data() : nullptr;
	double* type_scale_by_mu = need_type_scale ? two_layer_gate_type_scale_mu_buffer_.data() : nullptr;
	double* multiplier_by_mu = need_multiplier ? two_layer_gate_multiplier_mu_buffer_.data() : nullptr;
	double* outer_by_mu = need_outer ? two_layer_gate_outer_mu_buffer_.data() : nullptr;
	double* additive_param_by_mu =
		need_additive_param ? two_layer_gate_additive_param_mu_buffer_.data() : nullptr;
	double* gate_residual_param_by_mu =
		need_gate_residual_param ? two_layer_gate_gate_residual_mu_buffer_.data() : nullptr;
	double* gate_outer_param_by_mu =
		need_gate_outer_param ? two_layer_gate_gate_outer_param_mu_buffer_.data() : nullptr;
	double* gate_additive_param_by_mu =
		need_gate_additive_param ? two_layer_gate_gate_additive_param_mu_buffer_.data() : nullptr;
	if (type_center < 0 || type_center >= species_count
	    || type_outer < 0 || type_outer >= species_count)
		ERROR("SUS2-SH two-layer gate pair type index is out of range");
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	const int outer_type_coeff_rb_size = p_RadialBasis->rb_size;
	if (!two_layer_gate_enabled_) {
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double center_type_coeff =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   outer_type_coeff_rb_size,
				                   type_center,
				                   mu);
			const double outer_type_coeff =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   outer_type_coeff_rb_size,
				                   type_outer,
				                   mu);
			const double pair_type_scale = center_type_coeff * outer_type_coeff;
			if (need_additive)
				additive_by_mu[mu] = 0.0;
			if (need_type_scale)
				type_scale_by_mu[mu] = pair_type_scale;
			if (need_multiplier)
				multiplier_by_mu[mu] = 1.0;
			if (need_outer)
				outer_by_mu[mu] = outer_type_coeff;
			if (need_additive_param)
				additive_param_by_mu[mu] = 0.0;
			if (need_gate_residual_param)
				gate_residual_param_by_mu[mu] = 0.0;
			if (need_gate_outer_param)
				gate_outer_param_by_mu[mu] = 0.0;
			if (need_gate_additive_param)
				gate_additive_param_by_mu[mu] = 0.0;
		}
		return;
	}

	(void)neighbor_atom_index;
	const double* additive_coeff_row = TwoLayerGateAdditiveCoeffRowData(type_outer);
	double additive_coeff_scalar = 1.0;
	bool use_additive_coeff_scalar = false;
	if (additive_coeff_row == nullptr
	    && type_outer >= 0
	    && type_outer < static_cast<int>(two_layer_gate_additive_coeffs_.size())) {
		additive_coeff_scalar = two_layer_gate_additive_coeffs_[type_outer];
		use_additive_coeff_scalar = true;
	}

	if (buffer_mask == kGateMuBufferTypeScale) {
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double center_type_coeff =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   outer_type_coeff_rb_size,
				                   type_center,
				                   mu);
			const double outer_type_coeff =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   outer_type_coeff_rb_size,
				                   type_outer,
				                   mu);
			const double additive_coeff =
				additive_coeff_row != nullptr
					? additive_coeff_row[mu]
					: (use_additive_coeff_scalar
						   ? additive_coeff_scalar
						   : TwoLayerGateAdditiveCoeff(type_outer, mu));
			const double gate_signal =
				(gate_signal_by_mu == nullptr) ? 0.0 : gate_signal_by_mu[mu];
			type_scale_by_mu[mu] = center_type_coeff
				* (outer_type_coeff + additive_coeff * gate_signal);
		}
		return;
	}

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double center_type_coeff =
			OuterTypeCoeffFast(regression_coeffs,
			                   outer_type_coeff_offset,
			                   outer_type_coeff_stride,
			                   outer_type_coeff_rb_size,
			                   type_center,
			                   mu);
		const double outer_type_coeff =
			OuterTypeCoeffFast(regression_coeffs,
			                   outer_type_coeff_offset,
			                   outer_type_coeff_stride,
			                   outer_type_coeff_rb_size,
			                   type_outer,
			                   mu);
		const double additive_coeff =
			additive_coeff_row != nullptr
				? additive_coeff_row[mu]
				: (use_additive_coeff_scalar
					   ? additive_coeff_scalar
					   : TwoLayerGateAdditiveCoeff(type_outer, mu));
		const double gate_signal =
			(gate_signal_by_mu == nullptr) ? 0.0 : gate_signal_by_mu[mu];
		const double gate_outer = outer_type_coeff + additive_coeff * gate_signal;
		if (need_additive) {
			additive_by_mu[mu] = additive_coeff;
		}
		if (need_type_scale)
			type_scale_by_mu[mu] = center_type_coeff * gate_outer;
		if (need_multiplier)
			multiplier_by_mu[mu] = 1.0;
		if (need_outer)
			outer_by_mu[mu] = gate_outer;
		if (need_additive_param) {
			additive_param_by_mu[mu] = gate_signal;
		}
		if (need_gate_residual_param) {
			gate_residual_param_by_mu[mu] = 0.0;
		}
		if (need_gate_outer_param) {
			gate_outer_param_by_mu[mu] = 0.0;
		}
		if (need_gate_additive_param) {
			gate_additive_param_by_mu[mu] = 1.0;
		}
	}
}

void MLMTPR::PrepareTwoLayerGateAtomMuBuffers(int atom_type,
                                              const double* gate_signal_by_mu,
                                              int atom_index,
                                              int buffer_mask)
{
	const bool need_multiplier =
		(buffer_mask & kGateMuBufferMultiplier) != 0;
	const bool need_additive =
		(buffer_mask & kGateMuBufferAdditive) != 0;
	const bool need_additive_param =
		(buffer_mask & kGateMuBufferAdditiveParam) != 0;
	const bool need_gate_residual_param =
		(buffer_mask & kGateMuBufferGateResidualParam) != 0;
	const bool need_gate_additive_param =
		(buffer_mask & kGateMuBufferGateAdditiveParam) != 0;

	if (need_multiplier
	    && static_cast<int>(two_layer_gate_center_multiplier_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_center_multiplier_mu_buffer_.resize(radial_func_count);
	if (need_additive
	    && static_cast<int>(two_layer_gate_center_additive_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_center_additive_mu_buffer_.resize(radial_func_count);
	if (need_additive_param
	    && static_cast<int>(two_layer_gate_center_additive_param_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_center_additive_param_mu_buffer_.resize(radial_func_count);
	if (need_gate_residual_param
	    && static_cast<int>(two_layer_gate_center_residual_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_center_residual_mu_buffer_.resize(radial_func_count);
	if (need_gate_additive_param
	    && static_cast<int>(two_layer_gate_center_gate_additive_param_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_center_gate_additive_param_mu_buffer_.resize(radial_func_count);

	double* multiplier_by_mu = need_multiplier
		? two_layer_gate_center_multiplier_mu_buffer_.data() : nullptr;
	double* additive_by_mu = need_additive
		? two_layer_gate_center_additive_mu_buffer_.data() : nullptr;
	double* additive_param_by_mu = need_additive_param
		? two_layer_gate_center_additive_param_mu_buffer_.data() : nullptr;
	double* residual_by_mu = need_gate_residual_param
		? two_layer_gate_center_residual_mu_buffer_.data() : nullptr;
	double* gate_additive_param_by_mu = need_gate_additive_param
		? two_layer_gate_center_gate_additive_param_mu_buffer_.data() : nullptr;
	const bool need_gate_derivatives =
		need_additive || need_additive_param || need_gate_residual_param
		|| need_gate_additive_param;

	if (!two_layer_gate_enabled_) {
		for (int mu = 0; mu < radial_func_count; ++mu) {
			if (need_multiplier)
				multiplier_by_mu[mu] = 1.0;
			if (need_additive)
				additive_by_mu[mu] = 0.0;
			if (need_additive_param)
				additive_param_by_mu[mu] = 0.0;
			if (need_gate_residual_param)
				residual_by_mu[mu] = 0.0;
			if (need_gate_additive_param)
				gate_additive_param_by_mu[mu] = 0.0;
		}
		return;
	}

	const int coeff_index = TwoLayerGateAdditiveCoeffIndex(atom_type, 0);
	const bool use_regression_coeffs =
		coeff_index >= 0
		&& coeff_index < static_cast<int>(regression_coeffs.size());
	const bool use_saved_coeffs =
		atom_type >= 0
		&& atom_type < static_cast<int>(two_layer_gate_additive_coeffs_.size());
	const double additive_coeff = use_regression_coeffs
		? regression_coeffs[coeff_index]
		: (use_saved_coeffs ? two_layer_gate_additive_coeffs_[atom_type]
		                    : TwoLayerGateAdditiveCoeff(atom_type, 0));

	const bool use_tanh_cache =
		active_two_layer_gate_values_ != nullptr
		&& !DisableTwoLayerGateTanhCache()
		&& atom_index >= 0;
	double* tanh_cache = nullptr;
	bool fill_tanh_cache = false;
	if (use_tanh_cache) {
		if (radial_func_count <= 0
		    || active_two_layer_gate_values_->size() % radial_func_count != 0)
			ERROR("SUS2-SH two-layer gate mu signal cache has inconsistent size");
		const int atom_count =
			static_cast<int>(active_two_layer_gate_values_->size() / radial_func_count);
		if (atom_index >= atom_count)
			ERROR("SUS2-SH two-layer gate tanh cache atom index is out of range");
		const size_t expected_size =
			static_cast<size_t>(atom_count) * radial_func_count;
		if (two_layer_gate_tanh_mu_cache_atom_count_ != atom_count
		    || two_layer_gate_tanh_mu_cache_radial_func_count_ != radial_func_count
		    || two_layer_gate_tanh_mu_cache_.size() != expected_size
		    || static_cast<int>(two_layer_gate_tanh_mu_cache_valid_.size())
		        != atom_count) {
			two_layer_gate_tanh_mu_cache_.resize(expected_size);
			two_layer_gate_tanh_mu_cache_valid_.assign(atom_count, 0);
			two_layer_gate_tanh_mu_cache_atom_count_ = atom_count;
			two_layer_gate_tanh_mu_cache_radial_func_count_ = radial_func_count;
		}
		tanh_cache = two_layer_gate_tanh_mu_cache_.data()
			+ static_cast<size_t>(atom_index) * radial_func_count;
		fill_tanh_cache =
			two_layer_gate_tanh_mu_cache_valid_[atom_index] == 0;
	}

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double gate_signal =
			(gate_signal_by_mu == nullptr) ? 0.0 : gate_signal_by_mu[mu];
		const double arg = additive_coeff * gate_signal;
		const double tanh_arg = tanh_cache == nullptr
			? (arg == 0.0 ? 0.0 : std::tanh(arg))
			: (fill_tanh_cache
				? (tanh_cache[mu] = std::tanh(arg))
				: tanh_cache[mu]);
		const double sech2 = need_gate_derivatives
			? 1.0 - tanh_arg * tanh_arg
			: 0.0;
		if (need_multiplier)
			multiplier_by_mu[mu] =
				1.0 + two_layer_gate_tanh_amplitude_ * tanh_arg;
		if (need_additive)
			additive_by_mu[mu] =
				two_layer_gate_tanh_amplitude_ * additive_coeff * sech2;
		if (need_additive_param)
			additive_param_by_mu[mu] =
				two_layer_gate_tanh_amplitude_ * gate_signal * sech2;
		if (need_gate_residual_param)
			residual_by_mu[mu] =
				-2.0 * two_layer_gate_tanh_amplitude_
				* additive_coeff * additive_coeff * tanh_arg * sech2;
		if (need_gate_additive_param)
			gate_additive_param_by_mu[mu] =
				two_layer_gate_tanh_amplitude_ * sech2
				* (1.0 - 2.0 * arg * tanh_arg);
	}
	if (fill_tanh_cache)
		two_layer_gate_tanh_mu_cache_valid_[atom_index] = 1;
}

void MLMTPR::PrepareTwoLayerGatePairMuBuffers(int type_center,
                                              int type_outer,
                                              const double* center_gate_signal_by_mu,
                                              int center_atom_index,
                                              const double* neighbor_gate_signal_by_mu,
                                              int neighbor_atom_index,
                                              int buffer_mask)
{
	PrepareTwoLayerGateNeighborMuBuffers(
		type_outer, type_center,
		neighbor_gate_signal_by_mu, neighbor_atom_index, buffer_mask);
	if (!TwoLayerGateUsesCenterGate())
		return;

	const bool need_additive =
		(buffer_mask & kGateMuBufferAdditive) != 0;
	const bool need_type_scale =
		(buffer_mask & kGateMuBufferTypeScale) != 0;
	const bool need_multiplier =
		(buffer_mask & kGateMuBufferMultiplier) != 0;
	const bool need_outer =
		(buffer_mask & kGateMuBufferOuter) != 0;
	const bool need_additive_param =
		(buffer_mask & kGateMuBufferAdditiveParam) != 0;
	const bool need_gate_residual_param =
		(buffer_mask & kGateMuBufferGateResidualParam) != 0;
	const bool need_gate_outer_param =
		(buffer_mask & kGateMuBufferGateOuterParam) != 0;
	const bool need_gate_additive_param =
		(buffer_mask & kGateMuBufferGateAdditiveParam) != 0;
	int center_mask = kGateMuBufferMultiplier;
	if (need_additive)
		center_mask |= kGateMuBufferAdditive;
	if (need_additive_param)
		center_mask |= kGateMuBufferAdditiveParam;
	if (need_gate_residual_param)
		center_mask |= kGateMuBufferGateResidualParam;
	if (need_gate_additive_param)
		center_mask |= kGateMuBufferGateAdditiveParam;
	const bool center_cache_hit =
		two_layer_gate_center_mu_buffer_atom_index_ == center_atom_index
		&& two_layer_gate_center_mu_buffer_atom_type_ == type_center
		&& two_layer_gate_center_mu_buffer_signal_ == center_gate_signal_by_mu
		&& (two_layer_gate_center_mu_buffer_mask_ & center_mask) == center_mask;
	if (!center_cache_hit) {
		PrepareTwoLayerGateAtomMuBuffers(
			type_center, center_gate_signal_by_mu, center_atom_index, center_mask);
		two_layer_gate_center_mu_buffer_atom_index_ = center_atom_index;
		two_layer_gate_center_mu_buffer_atom_type_ = type_center;
		two_layer_gate_center_mu_buffer_mask_ = center_mask;
		two_layer_gate_center_mu_buffer_signal_ = center_gate_signal_by_mu;
	}

	if (need_multiplier
	    && static_cast<int>(two_layer_gate_neighbor_multiplier_mu_buffer_.size())
	    != radial_func_count)
		two_layer_gate_neighbor_multiplier_mu_buffer_.resize(radial_func_count);
	if (need_additive
	    && static_cast<int>(two_layer_gate_neighbor_additive_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_neighbor_additive_mu_buffer_.resize(radial_func_count);
	if (need_additive_param
	    && static_cast<int>(two_layer_gate_neighbor_additive_param_mu_buffer_.size())
	        != radial_func_count)
		two_layer_gate_neighbor_additive_param_mu_buffer_.resize(radial_func_count);

	const double* center_multiplier =
		two_layer_gate_center_multiplier_mu_buffer_.data();
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double center_mult = center_multiplier[mu];
		if (need_multiplier) {
			two_layer_gate_neighbor_multiplier_mu_buffer_[mu] =
				two_layer_gate_multiplier_mu_buffer_[mu];
			two_layer_gate_multiplier_mu_buffer_[mu] *= center_mult;
		}
		if (need_additive) {
			two_layer_gate_neighbor_additive_mu_buffer_[mu] =
				two_layer_gate_additive_mu_buffer_[mu];
			two_layer_gate_additive_mu_buffer_[mu] *= center_mult;
		}
		if (need_type_scale)
			two_layer_gate_type_scale_mu_buffer_[mu] *= center_mult;
		if (need_outer)
			two_layer_gate_outer_mu_buffer_[mu] *= center_mult;
		if (need_additive_param) {
			two_layer_gate_neighbor_additive_param_mu_buffer_[mu] =
				two_layer_gate_additive_param_mu_buffer_[mu];
			two_layer_gate_additive_param_mu_buffer_[mu] *= center_mult;
		}
		if (need_gate_residual_param)
			two_layer_gate_gate_residual_mu_buffer_[mu] *= center_mult;
		if (need_gate_outer_param)
			two_layer_gate_gate_outer_param_mu_buffer_[mu] *= center_mult;
		if (need_gate_additive_param)
			two_layer_gate_gate_additive_param_mu_buffer_[mu] *= center_mult;
	}
}

void MLMTPR::CalcSHBasisFuncs(const Neighborhood& nbh, double* bf_vals)
{
	CalcSHMomentValuesOnly(nbh);
	for (int i = 0; i < alpha_count; ++i)
		bf_vals[i] = basis_vals[i];
}

int MLMTPR::OuterKRank1ProductStart() const
{
	return OuterKRankEnabled() ? outer_k_rank1_product_begin_ : 0;
}

int MLMTPR::OuterKRank1OuterBasicIndex(int channel, int l, int m) const
{
	if (!OuterKRankEnabled())
		ERROR("SUS2-SH outer K-rank contraction is not enabled");
	if (channel < 0 || channel >= outer_k_rank_
	    || l < 0 || l > sh_l_max_ || std::abs(m) > l)
		ERROR("SUS2-SH outer K-rank (channel,l,m) index is out of range");
	// SHGraphBuilder orders l high-to-low, then contracted K high-to-low,
	// then m=-l..l.
	const int local =
		outer_k_rank_ * (
			(sh_l_max_ + 1) * (sh_l_max_ + 1) - (l + 1) * (l + 1))
		+ (outer_k_rank_ - 1 - channel) * (2 * l + 1)
		+ (m + l);
	if (local < 0 || local >= outer_k_rank1_basic_count_)
		ERROR("SUS2-SH outer K-rank primitive layout is inconsistent");
	return outer_k_rank1_basic_begin_ + local;
}

int MLMTPR::OuterKRank1RawBasicIndex(int l, int m, int k) const
{
	if (k < 0 || k >= sh_k_max_)
		ERROR("SUS2-SH outer K-rank raw k index is out of range");
	const int angular_local =
		(sh_l_max_ + 1) * (sh_l_max_ + 1)
		- (l + 1) * (l + 1) + (m + l);
	const size_t map_index =
		static_cast<size_t>(angular_local) * sh_k_max_ + k;
	if (map_index >= outer_k_rank1_raw_basic_indices_.size()
	    || outer_k_rank1_raw_basic_indices_[map_index] < 0)
		ERROR("SUS2-SH outer K-rank raw primitive cache is inconsistent");
	return outer_k_rank1_raw_basic_indices_[map_index];
}

void MLMTPR::ContractOuterKRank1Values(double* values) const
{
	if (!OuterKRankEnabled())
		return;
	if (OuterUsesPureRadialMatrix()) {
		// The complete outer primitive is supplied by W(r), not P*diag(R).
		// Clear this destination even for W=0; a zero pure matrix must never
		// recover the legacy baseline or retain a prior evaluation's values.
		std::fill(values + outer_k_rank1_basic_begin_,
		          values + outer_k_rank1_basic_begin_ + outer_k_rank1_basic_count_,
		          0.0);
		return;
	}
	if (OuterKRankUsesDirectRadial()) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				double contracted = 0.0;
				for (int k = 0; k < sh_k_max_; ++k)
					contracted +=
						values[OuterKRank1RawBasicIndex(l, m, k)];
				values[OuterKRank1OuterBasicIndex(0, l, m)] = contracted;
			}
		}
		return;
	}
	for (int channel = 0; channel < outer_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
			double contracted = 0.0;
			for (int k = 0; k < sh_k_max_; ++k)
				contracted += OuterKRank1Weight(channel, l, k)
					* values[OuterKRank1RawBasicIndex(l, m, k)];
			values[OuterKRank1OuterBasicIndex(channel, l, m)] = contracted;
			}
		}
	}
}

void MLMTPR::ContractOuterKRank1MomentDers(const Neighborhood& nbh)
{
	if (!OuterKRankEnabled())
		return;
	ContractOuterKRank1Values(moment_vals);
	if (OuterUsesPureRadialMatrix()) {
		for (int node = outer_k_rank1_basic_begin_;
		     node < outer_k_rank1_basic_begin_ + outer_k_rank1_basic_count_; ++node)
			for (int j = 0; j < nbh.count; ++j)
				for (int axis = 0; axis < 3; ++axis)
					moment_ders(node, j, axis) = 0.0;
		return;
	}
	if (OuterKRankUsesDirectRadial()) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				const int outer = OuterKRank1OuterBasicIndex(0, l, m);
				for (int j = 0; j < nbh.count; ++j) {
					for (int a = 0; a < 3; ++a) {
						double contracted = 0.0;
						for (int k = 0; k < sh_k_max_; ++k)
							contracted += moment_ders(
								OuterKRank1RawBasicIndex(l, m, k), j, a);
						moment_ders(outer, j, a) = contracted;
					}
				}
			}
		}
		return;
	}
	for (int channel = 0; channel < outer_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
			const int outer = OuterKRank1OuterBasicIndex(channel, l, m);
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					double contracted = 0.0;
					for (int k = 0; k < sh_k_max_; ++k)
						contracted += OuterKRank1Weight(channel, l, k)
							* moment_ders(
								OuterKRank1RawBasicIndex(l, m, k), j, a);
					moment_ders(outer, j, a) = contracted;
				}
			}
			}
		}
	}
}

void MLMTPR::BackpropOuterKRank1Adjoints(
	std::vector<double>& adjoints) const
{
	if (!OuterKRankEnabled())
		return;
	if (adjoints.size() < static_cast<size_t>(alpha_moments_count))
		ERROR("SUS2-SH outer K-rank1 adjoint buffer is too small");
	if (OuterUsesPureRadialMatrix())
		return; // W's adjoint is handled directly by the distance-matrix path.
	if (OuterKRankUsesDirectRadial()) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				const double outer_adjoint =
					adjoints[OuterKRank1OuterBasicIndex(0, l, m)];
				if (outer_adjoint == 0.0)
					continue;
				for (int k = 0; k < sh_k_max_; ++k)
					adjoints[OuterKRank1RawBasicIndex(l, m, k)] +=
						outer_adjoint;
			}
		}
		return;
	}
	for (int channel = 0; channel < outer_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
			const double outer_adjoint =
				adjoints[OuterKRank1OuterBasicIndex(channel, l, m)];
			if (outer_adjoint == 0.0)
				continue;
			for (int k = 0; k < sh_k_max_; ++k)
				adjoints[OuterKRank1RawBasicIndex(l, m, k)] +=
					OuterKRank1Weight(channel, l, k) * outer_adjoint;
			}
		}
	}
}

int MLMTPR::TwoLayerGateKRankBasicIndex(int channel, int l, int m) const
{
	if (!TwoLayerGateKRankEnabled())
		ERROR("SUS2-SH gate K-rank contraction is not enabled");
	if (channel < 0 || channel >= two_layer_gate_k_rank_
	    || l < 0 || l > sh_l_max_ || std::abs(m) > l)
		ERROR("SUS2-SH gate K-rank (channel,l,m) index is out of range");
	const int local =
		two_layer_gate_k_rank_ * (
			(sh_l_max_ + 1) * (sh_l_max_ + 1) - (l + 1) * (l + 1))
		+ (two_layer_gate_k_rank_ - 1 - channel) * (2 * l + 1)
		+ (m + l);
	if (local < 0 || local >= two_layer_gate_k_rank_basic_count_)
		ERROR("SUS2-SH gate K-rank primitive layout is inconsistent");
	return two_layer_gate_k_rank_basic_begin_ + local;
}

int MLMTPR::TwoLayerGateKRankRawBasicIndex(int l, int m, int k) const
{
	if (k < 0 || k >= sh_k_max_)
		ERROR("SUS2-SH gate K-rank raw k index is out of range");
	const int angular_local =
		(sh_l_max_ + 1) * (sh_l_max_ + 1)
		- (l + 1) * (l + 1) + (m + l);
	const size_t map_index =
		static_cast<size_t>(angular_local) * sh_k_max_ + k;
	if (map_index >= two_layer_gate_k_rank_raw_basic_indices_.size()
	    || two_layer_gate_k_rank_raw_basic_indices_[map_index] < 0)
		ERROR("SUS2-SH gate K-rank raw primitive cache is inconsistent");
	return two_layer_gate_k_rank_raw_basic_indices_[map_index];
}

void MLMTPR::ContractTwoLayerGateKRankValues(double* values) const
{
	if (!TwoLayerGateKRankEnabled())
		return;
	for (int channel = 0; channel < two_layer_gate_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				double contracted = 0.0;
				for (int k = 0; k < sh_k_max_; ++k)
					contracted += TwoLayerGateKRankWeight(channel, l, k)
						* values[TwoLayerGateKRankRawBasicIndex(l, m, k)];
				values[TwoLayerGateKRankBasicIndex(channel, l, m)] = contracted;
			}
		}
	}
}

void MLMTPR::BackpropTwoLayerGateKRankAdjoints(
	std::vector<double>& adjoints) const
{
	if (!TwoLayerGateKRankEnabled())
		return;
	if (adjoints.size() < static_cast<size_t>(alpha_moments_count))
		ERROR("SUS2-SH gate K-rank adjoint buffer is too small");
	for (int channel = 0; channel < two_layer_gate_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				const double contracted_adjoint =
					adjoints[TwoLayerGateKRankBasicIndex(channel, l, m)];
				if (contracted_adjoint == 0.0)
					continue;
				for (int k = 0; k < sh_k_max_; ++k)
					adjoints[TwoLayerGateKRankRawBasicIndex(l, m, k)] +=
						TwoLayerGateKRankWeight(channel, l, k)
						* contracted_adjoint;
			}
		}
	}
}

void MLMTPR::AccumulateTwoLayerGateKRankWeightGrad(
	std::vector<double>& out_grad,
	const std::vector<double>& total_adjoint,
	const std::vector<double>* energy_adjoint,
	const std::vector<double>* raw_tangent) const
{
	if (!TwoLayerGateKRankEnabled())
		return;
	if (total_adjoint.size() < static_cast<size_t>(alpha_moments_count)
	    || (energy_adjoint != nullptr
	        && energy_adjoint->size() < static_cast<size_t>(alpha_moments_count))
	    || (raw_tangent != nullptr
	        && raw_tangent->size() < static_cast<size_t>(alpha_moments_count)))
		ERROR("SUS2-SH gate K-rank weight-gradient buffer is too small");
	if ((energy_adjoint == nullptr) != (raw_tangent == nullptr))
		ERROR("SUS2-SH gate K-rank mixed-gradient inputs are inconsistent");
	if (out_grad.size() < regression_coeffs.size())
		out_grad.resize(regression_coeffs.size());
	for (int channel = 0; channel < two_layer_gate_k_rank_; ++channel) {
		for (int l = sh_l_max_; l >= 0; --l) {
			for (int m = -l; m <= l; ++m) {
				const int contracted =
					TwoLayerGateKRankBasicIndex(channel, l, m);
				for (int k = 0; k < sh_k_max_; ++k) {
					const int raw = TwoLayerGateKRankRawBasicIndex(l, m, k);
					double grad = total_adjoint[contracted] * moment_vals[raw];
					if (energy_adjoint != nullptr)
						grad += (*energy_adjoint)[contracted]
							* (*raw_tangent)[raw];
					out_grad[TwoLayerGateKRankWeightIndex(channel, l, k)] += grad;
				}
			}
		}
	}
}

bool MLMTPR::UseSHProductRows() const
{
	if (outer_k_rank1_enabled_)
		return false;
	if (sh_product_rows_.empty())
		return false;
	const char* env = std::getenv("SUS2_SH_PRODUCT_ROWS");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value == "1" || value == "true" || value == "True" || value == "on" || value == "ON";
}

bool MLMTPR::UseSHSiteDerivativeCache() const
{
	const char* env = std::getenv("SUS2_SH_SITE_DER_CACHE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

bool MLMTPR::UseSHAccumSkipSiteDers() const
{
	const char* env = std::getenv("SUS2_SH_ACCUM_SKIP_SITE_DERS");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

bool MLMTPR::UseSHProductHVTReverse() const
{
	const char* env = std::getenv("SUS2_SH_PRODUCT_HVT_REVERSE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

namespace {

bool SHAccumProfileEnabledOnRank0()
{
	const char* env = std::getenv("SUS2_SH_ACCUM_PROFILE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return false;
	const char* rank_envs[] = {
		"PMI_RANK",
		"OMPI_COMM_WORLD_RANK",
		"MV2_COMM_WORLD_RANK",
		"MPI_RANKID"
	};
	for (const char* rank_env : rank_envs) {
		const char* rank_value = std::getenv(rank_env);
		if (rank_value != nullptr)
			return std::atoi(rank_value) == 0;
	}
	return true;
}

int SHAccumProfileInterval()
{
	const char* env = std::getenv("SUS2_SH_ACCUM_PROFILE_INTERVAL");
	if (env == nullptr)
		return 10000;
	const int value = std::atoi(env);
	return value > 0 ? value : 10000;
}

double SHAccumProfileNow()
{
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct SHAccumProfileState {
	long long calls = 0;
	long long neighbors = 0;
	double total_s = 0.0;
	double moment_s = 0.0;
	double energy_backprop_s = 0.0;
	double force_product_s = 0.0;
	double force_cache_s = 0.0;
	double tangent_forward_s = 0.0;
	double mixed_seed_s = 0.0;
	double hvt_reverse_s = 0.0;
	double scalar_grad_s = 0.0;
	double coeff_grad_s = 0.0;
};

SHAccumProfileState& SHAccumProfile()
{
	static SHAccumProfileState state;
	return state;
}

void RecordSHAccumProfile(double total_s,
                          double moment_s,
                          double energy_backprop_s,
                          double force_product_s,
                          double force_cache_s,
                          double tangent_forward_s,
                          double mixed_seed_s,
                          double hvt_reverse_s,
                          double scalar_grad_s,
                          double coeff_grad_s,
                          int neighbor_count)
{
	SHAccumProfileState& state = SHAccumProfile();
	state.calls += 1;
	state.neighbors += neighbor_count;
	state.total_s += total_s;
	state.moment_s += moment_s;
	state.energy_backprop_s += energy_backprop_s;
	state.force_product_s += force_product_s;
	state.force_cache_s += force_cache_s;
	state.tangent_forward_s += tangent_forward_s;
	state.mixed_seed_s += mixed_seed_s;
	state.hvt_reverse_s += hvt_reverse_s;
	state.scalar_grad_s += scalar_grad_s;
	state.coeff_grad_s += coeff_grad_s;
	const int interval = SHAccumProfileInterval();
	if (state.calls % interval != 0)
		return;
	const double inv_calls = 1.0 / static_cast<double>(state.calls);
	std::cout << "[SUS2_SH_ACCUM_PROFILE] calls=" << state.calls
	          << " neighbors=" << state.neighbors
	          << " avg_us_total=" << 1.0e6 * state.total_s * inv_calls
	          << " avg_us_moment=" << 1.0e6 * state.moment_s * inv_calls
	          << " avg_us_energy_backprop=" << 1.0e6 * state.energy_backprop_s * inv_calls
	          << " avg_us_force_product=" << 1.0e6 * state.force_product_s * inv_calls
	          << " avg_us_force_cache=" << 1.0e6 * state.force_cache_s * inv_calls
	          << " avg_us_tangent_forward=" << 1.0e6 * state.tangent_forward_s * inv_calls
	          << " avg_us_mixed_seed=" << 1.0e6 * state.mixed_seed_s * inv_calls
	          << " avg_us_hvt_reverse=" << 1.0e6 * state.hvt_reverse_s * inv_calls
	          << " avg_us_scalar_grad=" << 1.0e6 * state.scalar_grad_s * inv_calls
	          << " avg_us_coeff_grad=" << 1.0e6 * state.coeff_grad_s * inv_calls
	          << std::endl;
}

} // namespace

void MLMTPR::TraceSHProductProgramOnce()
{
	if (sh_product_rows_trace_printed_)
		return;
	const char* env = std::getenv("SUS2_SH_PRODUCT_ROWS_TRACE");
	if (env == nullptr)
		return;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return;

	int terminal_rows = 0;
	int terminal_terms = 0;
	for (const SHProductRow& row : sh_product_rows_) {
		if (!row.terminal_scalar)
			continue;
		++terminal_rows;
		terminal_terms += row.term_count;
	}
	std::cout << "SUS2-SH product rows enabled: rows=" << sh_product_rows_.size()
	          << " terms=" << sh_product_row_terms_.size()
	          << " terminal_scalar_rows=" << terminal_rows
	          << " terminal_scalar_terms=" << terminal_terms << std::endl;
	sh_product_rows_trace_printed_ = true;
}

void MLMTPR::TraceSHSiteDerivativeCacheOnce(int neighbor_count,
                                            int sh_count,
                                            int radial_func_count)
{
	if (sh_site_der_cache_trace_printed_)
		return;
	const char* env = std::getenv("SUS2_SH_SITE_DER_CACHE_TRACE");
	if (env == nullptr)
		return;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return;

	std::cout << "SUS2-SH site derivative cache enabled: neighbors=" << neighbor_count
	          << " sh_components=" << sh_count
	          << " radial_functions=" << radial_func_count << std::endl;
	sh_site_der_cache_trace_printed_ = true;
}

void MLMTPR::ApplySHProductRowsForward()
{
	TraceSHProductProgramOnce();
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * moment_vals[term.left] * moment_vals[term.right];
		}
		moment_vals[row.target] += value;
	}
}

void MLMTPR::ApplySHProductRowsDers(const Neighborhood& nbh)
{
	TraceSHProductProgramOnce();
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * moment_vals[term.left] * moment_vals[term.right];
		}
		moment_vals[row.target] += value;

		if (row.terminal_scalar) {
			const int basis_index = 1 + row.scalar_index;
			basis_vals[basis_index] = moment_vals[row.target];
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					double der = 0.0;
					for (int t = row.term_begin; t < end; ++t) {
						const SHProductRowTerm& term = sh_product_row_terms_[t];
						der += term.coeff * (
							moment_ders(term.left, j, a) * moment_vals[term.right]
							+ moment_vals[term.left] * moment_ders(term.right, j, a));
					}
					basis_ders(basis_index, j, a) = der;
				}
			}
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					double der = 0.0;
					for (int t = row.term_begin; t < end; ++t) {
						const SHProductRowTerm& term = sh_product_row_terms_[t];
						der += term.coeff * (
							moment_ders(term.left, j, a) * moment_vals[term.right]
							+ moment_vals[term.left] * moment_ders(term.right, j, a));
					}
					moment_ders(row.target, j, a) += der;
				}
			}
		}
	}
}

void MLMTPR::AccumulateSHProductRowsForward(const std::vector<double>& input_values,
                                            std::vector<double>& output_values) const
{
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * (
				input_values[term.left] * moment_vals[term.right]
				+ moment_vals[term.left] * input_values[term.right]);
		}
		output_values[row.target] += value;
	}
}

void MLMTPR::BackpropSHProductRows(std::vector<double>& adjoints) const
{
	for (int r = static_cast<int>(sh_product_rows_.size()) - 1; r >= 0; --r) {
		const SHProductRow& row = sh_product_rows_[r];
		const double adj_target = adjoints[row.target];
		if (adj_target == 0.0)
			continue;
		for (int t = row.term_begin + row.term_count - 1; t >= row.term_begin; --t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			adjoints[term.left] += term.coeff * moment_vals[term.right] * adj_target;
			adjoints[term.right] += term.coeff * moment_vals[term.left] * adj_target;
		}
	}
}

void MLMTPR::ClearTwoLayerEdgePrimitiveCache()
{
	pure_w_execution_.basis = false;
	pure_w_execution_.edge_force = false;
	pure_w_execution_.packed = false;
	pure_w_execution_.graph = false;
	pure_w_execution_.profile = false;
	two_layer_edge_cache_ready_ = false;
	two_layer_edge_cache_has_derivatives_ = false;
	two_layer_edge_cache_has_param_derivatives_ = false;
	two_layer_full_edge_cache_for_next_calc_ = false;
	two_layer_reuse_full_edge_cache_once_ = false;
	two_layer_forward_final_moment_cache_ready_ = false;
	two_layer_edge_cache_atom_count_ = 0;
	two_layer_edge_cache_eval_block_count_ = 0;
	two_layer_edge_cache_sh_count_ = 0;
	two_layer_edge_cache_radial_func_count_ = 0;
	two_layer_edge_cache_gate_mu_count_ = 0;
	two_layer_edge_cache_rb_size_ = 0;
	active_two_layer_edge_cache_atom_index_ = -1;
	two_layer_gate_values_from_edge_cache_ready_ = false;
	two_layer_gate_edge_l1_values_.clear();
	two_layer_gate_edge_l1_weighted_values_.clear();
	two_layer_gate_edge_l1_weights_by_source_.clear();
	two_layer_edge_gate_type_scale_cache_.clear();
	two_layer_edge_gate_additive_cache_.clear();
	two_layer_edge_gate_neighbor_multiplier_cache_.clear();
	InvalidateTwoLayerGateTanhMuCache();
}

bool MLMTPR::HasTwoLayerEdgePrimitiveCache(int cache_atom_index,
                                           bool need_derivatives,
                                           bool need_param_derivatives) const
{
	if (cache_atom_index < 0)
		cache_atom_index = active_two_layer_edge_cache_atom_index_;
	if (!two_layer_edge_cache_ready_ || cache_atom_index < 0)
		return false;
	if (cache_atom_index >= two_layer_edge_cache_atom_count_)
		return false;
	if (need_derivatives && !two_layer_edge_cache_has_derivatives_)
		return false;
	if (need_derivatives
	    && need_param_derivatives
	    && !two_layer_edge_cache_has_param_derivatives_)
		return false;
	if (two_layer_edge_cache_eval_block_count_ !=
	    static_cast<int>(radial_eval_to_scaling_block_.size()))
		return false;
	if (two_layer_edge_cache_sh_count_ != (sh_l_max_ + 1) * (sh_l_max_ + 1))
		return false;
	if (two_layer_edge_cache_radial_func_count_ != radial_func_count)
		return false;
	if (two_layer_edge_cache_rb_size_ != p_RadialBasis->rb_size)
		return false;
	if (static_cast<size_t>(cache_atom_index + 1)
	    >= two_layer_edge_offsets_cache_.size())
		return false;
	const size_t edge_count = two_layer_edge_offsets_cache_.back();
	const size_t expected_sh_size =
		edge_count * static_cast<size_t>(two_layer_edge_cache_sh_count_);
	const bool adaptive_outer = OuterUsesAdaptiveRadialCoefficients();
	const bool adaptive_reduced_outer_cache =
		adaptive_outer && TwoLayerGateUsesSharedRadial() && !need_param_derivatives
		&& two_layer_edge_mu_vals_cache_.empty();
	const size_t expected_mu_size =
		adaptive_reduced_outer_cache
			? 0 : edge_count * static_cast<size_t>(radial_func_count);
	const size_t expected_gate_mu_size =
		edge_count * static_cast<size_t>(two_layer_gate_required_mu_indices_.size());
	if (two_layer_edge_sh_values_cache_.size() < expected_sh_size
	    || two_layer_edge_mu_vals_cache_.size() < expected_mu_size)
		return false;
	if (need_derivatives
	    && (two_layer_edge_sh_ders_cache_.size()
	            < expected_sh_size * static_cast<size_t>(3)
	        || two_layer_edge_mu_ders_cache_.size() < expected_mu_size))
		return false;
	if (OuterUsesPureRadialMatrix() || HasNonzeroTwoLayerGateRadialMixingCoeffs()) {
		const size_t expected_radial_val_size =
			edge_count
			* static_cast<size_t>(two_layer_edge_cache_eval_block_count_)
			* static_cast<size_t>(two_layer_edge_cache_rb_size_);
		if (two_layer_edge_radial_vals_cache_.size() < expected_radial_val_size)
			return false;
		const size_t expected_kernel_size =
			edge_count * static_cast<size_t>(sh_l_max_ + 1)
			* static_cast<size_t>(outer_k_rank_)
			* static_cast<size_t>(sh_k_max_);
		if (!pure_w_execution_.basis && two_layer_edge_radial_mixing_kernel_cache_.size()
		    < expected_kernel_size)
			return false;
		if (need_derivatives && !pure_w_execution_.basis
		    && two_layer_edge_radial_mixing_kernel_der_cache_.size()
		        < expected_kernel_size)
			return false;
		if (need_derivatives && pure_w_execution_.basis
		    && pure_w_execution_.raw_dr.size() < expected_radial_val_size)
			return false;
	}
	if (need_derivatives && need_param_derivatives) {
		const size_t expected_radial_val_size =
			edge_count
			* static_cast<size_t>(two_layer_edge_cache_eval_block_count_)
			* static_cast<size_t>(two_layer_edge_cache_rb_size_);
		const size_t expected_radial_der_size =
			expected_radial_val_size * static_cast<size_t>(5);
		if (two_layer_edge_radial_vals_cache_.size() < expected_radial_val_size
		    || two_layer_edge_radial_ders_cache_.size() < expected_radial_der_size
		    || two_layer_edge_mu_ders_s_cache_.size() < expected_mu_size
		    || two_layer_edge_mu_ders_ss_cache_.size() < expected_mu_size
		    || two_layer_edge_mu_coord_ders_s_cache_.size() < expected_mu_size
			    || two_layer_edge_mu_coord_ders_ss_cache_.size() < expected_mu_size)
			return false;
	}
	if (adaptive_outer) {
		const size_t expected_radial_val_size =
			edge_count
			* static_cast<size_t>(two_layer_edge_cache_eval_block_count_)
			* static_cast<size_t>(two_layer_edge_cache_rb_size_);
		if (two_layer_edge_radial_vals_cache_.size() < expected_radial_val_size)
			return false;
		if (need_derivatives && !need_param_derivatives) {
			const size_t full_radial_der_size =
				expected_radial_val_size * static_cast<size_t>(5);
			const size_t radial_der_size = two_layer_edge_radial_ders_cache_.size();
			if (radial_der_size != expected_radial_val_size
			    && radial_der_size != full_radial_der_size)
				return false;
		}
	}
	if (TwoLayerGateUsesSharedRadial()) {
		if (two_layer_edge_cache_gate_mu_count_
		    != static_cast<int>(two_layer_gate_required_mu_indices_.size()))
			return false;
		if (two_layer_edge_gate_mu_vals_cache_.size() < expected_gate_mu_size)
			return false;
		if (need_derivatives
		    && two_layer_edge_gate_mu_ders_cache_.size() < expected_gate_mu_size)
			return false;
		if (need_derivatives
		    && need_param_derivatives
		    && (two_layer_edge_gate_mu_ders_s_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_ders_ss_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_coord_ders_s_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_coord_ders_ss_cache_.size() < expected_gate_mu_size))
			return false;
		if (need_derivatives && need_param_derivatives
		    && TwoLayerGateUsesDecoupledScaling()) {
			const size_t expected_gate_radial_size =
				edge_count
				* static_cast<size_t>(two_layer_edge_cache_eval_block_count_)
				* static_cast<size_t>(two_layer_edge_cache_rb_size_);
			if (two_layer_edge_gate_radial_vals_cache_.size()
			        < expected_gate_radial_size
			    || two_layer_edge_gate_radial_dr_cache_.size()
			        < expected_gate_radial_size)
				return false;
		}
	}
	return true;
}

size_t MLMTPR::TwoLayerEdgePrimitiveOffset(int cache_atom_index,
                                           int neighbor_index) const
{
	if (cache_atom_index < 0)
		cache_atom_index = active_two_layer_edge_cache_atom_index_;
	return two_layer_edge_offsets_cache_[cache_atom_index]
		+ static_cast<size_t>(neighbor_index);
}

int MLMTPR::TwoLayerGateMuDenseIndex(int mu) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate mu index is out of range");
	if (static_cast<int>(two_layer_gate_mu_dense_index_.size()) != radial_func_count)
		ERROR("SUS2-SH two-layer gate mu dense map is not initialized");
	const int dense = two_layer_gate_mu_dense_index_[mu];
	if (dense < 0 || dense >= static_cast<int>(two_layer_gate_required_mu_indices_.size()))
		ERROR("SUS2-SH two-layer gate mu dense index is out of range");
	return dense;
}

void MLMTPR::BuildTwoLayerEdgePrimitiveCache(const Neighborhoods& neighborhoods,
                                             bool need_derivatives,
                                             bool need_param_derivatives,
                                             bool inference_only)
{
	ClearTwoLayerEdgePrimitiveCache();
	if (DisableTwoLayerEdgePrimitiveCache())
		return;
	if (!is_sh_potential_ || !two_layer_gate_enabled_)
		return;
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (use_lmp_table && need_param_derivatives)
		ERROR("SUS2-SH _lmp radial tables do not provide nonlinear parameter derivatives.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
		if ((TwoLayerGateScalarCount() > 0 || TwoLayerGateEdgeL1Enabled())
		    && (two_layer_gate_required_moments_.empty()
		        || two_layer_gate_required_moment_indices_.empty()))
			BuildTwoLayerGateProductProgram();
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const bool decoupled_gate_scaling =
		use_gate_radial && TwoLayerGateUsesDecoupledScaling();
	const int gate_count = TwoLayerGateScalarCount();
	const int cached_gate_moment_count =
		static_cast<int>(two_layer_gate_required_moment_indices_.size());
			const bool prepare_gate_values_from_cache =
				!multi_layer_gate_enabled_
				&& gate_count > 0 && cached_gate_moment_count > 0;
			const int edge_l1_source_count = TwoLayerGateEdgeL1Enabled()
				? TwoLayerGateEdgeL1SourceCount()
				: 0;
			const bool prepare_moment_values_from_cache =
				cached_gate_moment_count > 0
				&& (gate_count > 0 || edge_l1_source_count > 0);
			const bool prepare_edge_l1_values_from_cache =
				prepare_moment_values_from_cache && edge_l1_source_count > 0;

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const bool adaptive_outer = OuterUsesAdaptiveRadialCoefficients();
	const bool has_radial_mixing =
		TwoLayerGateRadialMixingEnabled()
		&& (OuterUsesPureRadialMatrix() || HasNonzeroTwoLayerGateRadialMixingCoeffs());
	// Keep the established block stride for shared W/gate consumers. EFS only
	// initializes the first R entries of each block; no mixed consumer may
	// reuse this cache (CanUseTwoLayerEdgePrimitiveCache checks that contract).
	const bool coordinate_only = inference_only && need_derivatives
		&& !need_param_derivatives && !EnvFlagEnabled("SUS2_SH_EFS_RADIAL_FULL");
	const int radial_der_copy_count = coordinate_only ? R : 5*R;
	const bool adaptive_reference = AdaptiveRadialEFSReferenceEnabled();
	const bool adaptive_compact_radial_dr =
		adaptive_outer && use_gate_radial && coordinate_only && !adaptive_reference
		&& !AdaptiveRadialFullDerivativeCacheEnabled()
		&& !has_radial_mixing;
	const int adaptive_radial_der_stride =
		adaptive_compact_radial_dr ? radial_val_stride : radial_der_stride;
	const bool store_outer_mu_cache =
		!(adaptive_outer && use_gate_radial && inference_only
		  && !need_param_derivatives && !adaptive_reference
		  && !has_radial_mixing);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;
	const int atom_count = neighborhoods.size();
	size_t edge_count = 0;
	two_layer_edge_offsets_cache_.assign(static_cast<size_t>(atom_count) + 1, 0);
	for (int ind = 0; ind < atom_count; ++ind) {
		two_layer_edge_offsets_cache_[ind] = edge_count;
		edge_count += static_cast<size_t>(neighborhoods[ind].count);
	}
	two_layer_edge_offsets_cache_[atom_count] = edge_count;
	pure_w_execution_.profile = OuterUsesPureRadialMatrix()
		&& EnvFlagEnabled("SUS2_SH_PURE_W_PROFILE") && RuntimeRankIsZero();
	pure_w_execution_.pack_seconds = pure_w_execution_.kernel_seconds = 0.0;
	pure_w_execution_.aggregate_seconds = pure_w_execution_.mix_seconds = 0.0;
	pure_w_execution_.force_seconds = pure_w_execution_.graph_seconds = 0.0;
	// The basis cache is deliberately not a full derivative cache. Other
	// consumers (linear components, mixed gradients, residuals and edge-L1)
	// continue to receive the original W/W' layout and full derivative data.
	const int pure_w_mode = OuterUsesPureRadialMatrix() ? PureWKernelMode() : 0;
	const bool prefer_basis = pure_w_mode == 2 || (pure_w_mode == -1
		&& PureWExecution::PreferBasis(outer_k_rank_, R, edge_count, atom_count));
	pure_w_execution_.basis = prefer_basis
		&& inference_only && !need_param_derivatives && !multi_layer_gate_enabled_
		&& !TwoLayerResidualEnabled() && !TwoLayerGateUsesCenterGate()
		&& !HasNonzeroTwoLayerGateEdgeL1Weights() && !use_lmp_table;
	pure_w_execution_.edge_force = OuterUsesPureRadialMatrix()
		&& inference_only && need_derivatives && !need_param_derivatives
		&& !pure_w_execution_.basis && !multi_layer_gate_enabled_
		&& !TwoLayerResidualEnabled() && !TwoLayerGateUsesCenterGate()
		&& !HasNonzeroTwoLayerGateEdgeL1Weights() && !use_lmp_table
		&& !EnvFlagEnabled("SUS2_SH_PURE_W_EDGE_REFERENCE");
	// Auto leaves full mixed-gradient consumers entirely on the legacy path.
	pure_w_execution_.packed = pure_w_mode > 0 || pure_w_execution_.basis;
	// M_l(r) needs the raw radial basis even for value-only line-search
	// evaluations. Cache it with the other edge primitives so every trial does
	// not repeat RB_CalcValsOnly for unchanged geometry.
	const bool cache_radial_mixing_kernels = has_radial_mixing && !pure_w_execution_.basis;
	const bool need_raw_radial_cache =
		(need_derivatives && need_param_derivatives)
		|| has_radial_mixing || adaptive_outer;
	const bool need_gate_raw_radial_cache =
		need_raw_radial_cache && decoupled_gate_scaling;
	const size_t radial_mixing_kernel_stride =
		static_cast<size_t>(sh_l_max_ + 1)
		* static_cast<size_t>(outer_k_rank_)
		* static_cast<size_t>(sh_k_max_);
	if (pure_w_execution_.basis && need_derivatives)
		pure_w_execution_.raw_dr.resize(edge_count * radial_val_stride);
	// Keep inactive scratch sizes as well as capacity. The execution flags,
	// not stale buffer contents, determine validity. Clearing here forces a
	// redundant zero-fill when an inference/full-gradient call alternates.
	const double pack_start = pure_w_execution_.profile ? SHAccumProfileNow() : 0.0;
	if (pure_w_execution_.packed)
		pure_w_execution_.Pack(sh_l_max_+1, outer_k_rank_, sh_k_max_, R,
			regression_coeffs.data() + TwoLayerGateRadialMixingCoeffOffset());
	PreparePureWGraph();
	if (pure_w_execution_.profile)
		pure_w_execution_.pack_seconds = SHAccumProfileNow()-pack_start;

	two_layer_edge_sh_values_cache_.resize(edge_count * sh_count);
	if (store_outer_mu_cache)
		two_layer_edge_mu_vals_cache_.resize(edge_count * mu_stride);
	else
		two_layer_edge_mu_vals_cache_.clear();
	if (need_raw_radial_cache) {
		two_layer_edge_radial_vals_cache_.resize(edge_count * radial_val_stride);
	} else {
		two_layer_edge_radial_vals_cache_.clear();
	}
	if (cache_radial_mixing_kernels) {
		two_layer_edge_radial_mixing_kernel_cache_.resize(
			edge_count * radial_mixing_kernel_stride);
		if (need_derivatives)
			two_layer_edge_radial_mixing_kernel_der_cache_.resize(
				edge_count * radial_mixing_kernel_stride);
		else
			two_layer_edge_radial_mixing_kernel_der_cache_.clear();
	} else if (!pure_w_execution_.basis) {
		two_layer_edge_radial_mixing_kernel_cache_.clear();
		two_layer_edge_radial_mixing_kernel_der_cache_.clear();
	}
	if (need_gate_raw_radial_cache) {
		two_layer_edge_gate_radial_vals_cache_.resize(
			edge_count * radial_val_stride);
		two_layer_edge_gate_radial_dr_cache_.resize(
			edge_count * radial_val_stride);
	} else {
		two_layer_edge_gate_radial_vals_cache_.clear();
		two_layer_edge_gate_radial_dr_cache_.clear();
	}
	if (need_derivatives) {
		two_layer_edge_sh_ders_cache_.resize(edge_count * 3 * sh_count);
		if (store_outer_mu_cache)
			two_layer_edge_mu_ders_cache_.resize(edge_count * mu_stride);
		else
			two_layer_edge_mu_ders_cache_.clear();
		// Adaptive coefficients contract the raw radial derivative, not the
		// obsolete environment-independent K-channel radial functions. Only
		// the first R entries of each block are valid in the EFS-only cache.
		if (adaptive_outer)
			two_layer_edge_radial_ders_cache_.resize(
				edge_count * static_cast<size_t>(adaptive_radial_der_stride));
		if (need_param_derivatives) {
			two_layer_edge_radial_ders_cache_.resize(edge_count * radial_der_stride);
			if (store_outer_mu_cache) {
				two_layer_edge_mu_ders_s_cache_.resize(edge_count * mu_stride);
				two_layer_edge_mu_ders_ss_cache_.resize(edge_count * mu_stride);
				two_layer_edge_mu_coord_ders_s_cache_.resize(edge_count * mu_stride);
				two_layer_edge_mu_coord_ders_ss_cache_.resize(edge_count * mu_stride);
			} else {
				two_layer_edge_mu_ders_s_cache_.clear();
				two_layer_edge_mu_ders_ss_cache_.clear();
				two_layer_edge_mu_coord_ders_s_cache_.clear();
				two_layer_edge_mu_coord_ders_ss_cache_.clear();
			}
		} else {
			if (!adaptive_outer)
				two_layer_edge_radial_ders_cache_.clear();
			two_layer_edge_mu_ders_s_cache_.clear();
			two_layer_edge_mu_ders_ss_cache_.clear();
			two_layer_edge_mu_coord_ders_s_cache_.clear();
			two_layer_edge_mu_coord_ders_ss_cache_.clear();
		}
	} else {
		two_layer_edge_sh_ders_cache_.clear();
		two_layer_edge_radial_ders_cache_.clear();
		two_layer_edge_mu_ders_cache_.clear();
		two_layer_edge_mu_ders_s_cache_.clear();
		two_layer_edge_mu_ders_ss_cache_.clear();
		two_layer_edge_mu_coord_ders_s_cache_.clear();
		two_layer_edge_mu_coord_ders_ss_cache_.clear();
	}
	if (use_gate_radial) {
		two_layer_edge_gate_mu_vals_cache_.resize(edge_count * gate_mu_stride);
		if (need_derivatives) {
			two_layer_edge_gate_mu_ders_cache_.resize(edge_count * gate_mu_stride);
			if (need_param_derivatives) {
				two_layer_edge_gate_mu_ders_s_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_ders_ss_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_coord_ders_s_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_coord_ders_ss_cache_.resize(edge_count * gate_mu_stride);
			} else {
				two_layer_edge_gate_mu_ders_s_cache_.clear();
				two_layer_edge_gate_mu_ders_ss_cache_.clear();
				two_layer_edge_gate_mu_coord_ders_s_cache_.clear();
				two_layer_edge_gate_mu_coord_ders_ss_cache_.clear();
			}
		}
	}
			if (prepare_moment_values_from_cache) {
				two_layer_gate_moment_values_cache_.resize(
					static_cast<size_t>(atom_count) * cached_gate_moment_count);
				if (prepare_edge_l1_values_from_cache)
					two_layer_gate_edge_l1_values_.assign(
						static_cast<size_t>(atom_count)
						* edge_l1_source_count * 3, 0.0);
			}
			if (prepare_gate_values_from_cache) {
				if (static_cast<int>(two_layer_gate_scalar_body_orders_.size()) != gate_count
				    || static_cast<int>(two_layer_gate_mu_body_orders_.size()) != radial_func_count)
					BuildTwoLayerGateBodyOrderBuckets();
				const bool body_linear_combo = TwoLayerGateUsesBodyLinearCombo();
			two_layer_gate_values_.assign(
			static_cast<size_t>(atom_count) * radial_func_count, 0.0);
		two_layer_gate_scalar_values_cache_.resize(
			static_cast<size_t>(atom_count) * gate_count);
		if (body_linear_combo)
			two_layer_gate_body_values_cache_.resize(
				static_cast<size_t>(atom_count) * TwoLayerGateBodyOrderCount());
			else
				two_layer_gate_body_values_cache_.clear();
			}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const double* radial_mixing_coeffs = cache_radial_mixing_kernels
		? regression_coeffs.data() + TwoLayerGateRadialMixingCoeffOffset()
		: nullptr;
	const double* gate_type_coeff_matrix = TwoLayerGateTypeCoeffMatrixData();
	std::vector<double> radial_vals_scratch;
	std::vector<double> radial_ders_scratch;
	if (!need_raw_radial_cache)
		radial_vals_scratch.resize(radial_val_stride);
	if (need_derivatives && !need_param_derivatives)
		radial_ders_scratch.resize(radial_der_stride);
	std::vector<double> gate_radial_vals_scratch;
	std::vector<double> gate_radial_ders_scratch;
	if (use_gate_radial && decoupled_gate_scaling) {
		if (!need_gate_raw_radial_cache)
			gate_radial_vals_scratch.resize(radial_val_stride);
		if (need_derivatives)
			gate_radial_ders_scratch.resize(radial_der_stride);
	}
	std::vector<double> gate_lmp_vals_scratch;
	std::vector<double> gate_lmp_ders_scratch;
	if (use_lmp_table && use_gate_radial) {
		gate_lmp_vals_scratch.resize(radial_func_count);
		if (need_derivatives)
			gate_lmp_ders_scratch.resize(radial_func_count);
	}

	for (int ind = 0; ind < atom_count; ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		const int type_central = nbh.my_type;
		if (type_central < 0 || type_central >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
		if (prepare_moment_values_from_cache)
			for (int moment_index : two_layer_gate_required_moment_indices_)
				moment_vals[moment_index] = 0.0;
		for (int j = 0; j < nbh.count; ++j) {
			const size_t edge = two_layer_edge_offsets_cache_[ind]
				+ static_cast<size_t>(j);
			const Vector3& rvec = nbh.vecs[j];
			const double r = nbh.dists[j];
			const int type_outer = nbh.types[j];
			if (type_outer < 0 || type_outer >= species_count)
				throw MlipException("Too few species count in the MTP potential!");
			double* sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			double* sh_ders = need_derivatives
				? two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count
				: nullptr;
			double* rb_vals = need_raw_radial_cache
				? two_layer_edge_radial_vals_cache_.data()
					+ edge * radial_val_stride
				: radial_vals_scratch.data();
			double* rb_ders = need_derivatives
				? (need_param_derivatives || adaptive_outer
					? two_layer_edge_radial_ders_cache_.data()
						+ edge * static_cast<size_t>(
							adaptive_outer ? adaptive_radial_der_stride : radial_der_stride)
					: radial_ders_scratch.data())
				: nullptr;
			double* mu_vals = store_outer_mu_cache
				? two_layer_edge_mu_vals_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_ders = need_derivatives
				? (store_outer_mu_cache
					? two_layer_edge_mu_ders_cache_.data() + edge * mu_stride
					: nullptr)
				: nullptr;
			double* mu_ders_s =
				store_outer_mu_cache && need_derivatives && need_param_derivatives
				? two_layer_edge_mu_ders_s_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_ders_ss =
				store_outer_mu_cache && need_derivatives && need_param_derivatives
				? two_layer_edge_mu_ders_ss_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_coord_ders_s =
				store_outer_mu_cache && need_derivatives && need_param_derivatives
				? two_layer_edge_mu_coord_ders_s_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_coord_ders_ss =
				store_outer_mu_cache && need_derivatives && need_param_derivatives
				? two_layer_edge_mu_coord_ders_ss_cache_.data() + edge * mu_stride
				: nullptr;
			double* gate_mu_vals = use_gate_radial
				? two_layer_edge_gate_mu_vals_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders = use_gate_radial && need_derivatives
				? two_layer_edge_gate_mu_ders_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders_s =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_ders_s_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders_ss =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_ders_ss_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_coord_ders_s =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_coord_ders_s_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_coord_ders_ss =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_coord_ders_ss_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_raw_vals_cache = need_gate_raw_radial_cache
				? two_layer_edge_gate_radial_vals_cache_.data()
					+ edge * radial_val_stride
				: nullptr;
			double* gate_raw_dr_cache = need_gate_raw_radial_cache
				? two_layer_edge_gate_radial_dr_cache_.data()
					+ edge * radial_val_stride
				: nullptr;
			double* radial_mixing_kernels = cache_radial_mixing_kernels
				? two_layer_edge_radial_mixing_kernel_cache_.data()
					+ edge * radial_mixing_kernel_stride
				: nullptr;
			double* radial_mixing_kernel_ders =
				cache_radial_mixing_kernels && need_derivatives
				? two_layer_edge_radial_mixing_kernel_der_cache_.data()
					+ edge * radial_mixing_kernel_stride
				: nullptr;

				if (need_derivatives)
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
			else
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

				if (use_lmp_table) {
					if (store_outer_mu_cache)
					InterpolateSHLmpMuTable(radial_list,
					                         radial_der_list,
					                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
					                         radial_func_count,
					                         mu_vals,
					                         need_derivatives ? mu_ders : nullptr);
				if (use_gate_radial) {
					InterpolateSHLmpMuTable(two_layer_gate_radial_list,
					                         two_layer_gate_radial_der_list,
					                         inv_dr,
					                         C,
					                         type_central,
					                         type_outer,
					                         r,
					                         radial_func_count,
					                         gate_lmp_vals_scratch.data(),
					                         need_derivatives ? gate_lmp_ders_scratch.data() : nullptr);
					for (int dense_mu = 0;
					     dense_mu < static_cast<int>(two_layer_gate_required_mu_indices_.size());
					     ++dense_mu) {
						const int mu = two_layer_gate_required_mu_indices_[dense_mu];
						gate_mu_vals[dense_mu] = gate_lmp_vals_scratch[mu];
						if (need_derivatives)
							gate_mu_ders[dense_mu] = gate_lmp_ders_scratch[mu];
					}
				}
			} else {
					for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					const double sigma =
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * type_central
							+ type_outer];
					const double shift =
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * C
							+ C * type_central + type_outer];
					if (coordinate_only)
						p_RadialBasis->RB_CalcRDerOnly(r, sigma, shift, basis_k);
					else if (need_derivatives)
						p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
					else
						p_RadialBasis->RB_CalcValsOnly(r, sigma, shift, basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] =
							p_RadialBasis->rb_vals[xi] * scaling;
						if (need_derivatives)
							for (int xi = 0; xi < radial_der_copy_count; ++xi)
								rb_ders[
									(adaptive_compact_radial_dr
									 ? eval_block * R : eval_block * R * 5)
									+ xi] =
									p_RadialBasis->rb_ders[xi] * scaling;
						if (adaptive_compact_radial_dr && use_gate_radial) {
							for (int xi = 0; xi < radial_der_copy_count; ++xi)
								radial_ders_scratch[eval_block * R * 5 + xi] =
									p_RadialBasis->rb_ders[xi] * scaling;
						}
					}

					for (int mu = 0; store_outer_mu_cache && mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int deriv_base = 5 * radial_base;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double dot_val = 0.0;
					double dot_der = 0.0;
					double dot_s = 0.0;
					double dot_coord_s = 0.0;
					double dot_ss = 0.0;
					double dot_coord_ss = 0.0;
					// W contracts the raw basis directly. Keep the legacy cache
					// slots zero, but do not contract unused outer R coefficients.
					const int outer_basis_count =
						(OuterUsesPureRadialMatrix() || OuterUsesAdaptiveRadialCoefficients()) ? 0 : R;
					for (int xi = 0; xi < outer_basis_count; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						dot_val += coeff * rb_vals[radial_base + xi];
						if (need_derivatives) {
							dot_der += coeff * rb_ders[deriv_base + xi];
							if (need_param_derivatives) {
								dot_s += coeff * rb_ders[deriv_base + xi + R];
								dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
								dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
								dot_coord_ss +=
									coeff * rb_ders[deriv_base + xi + 4 * R];
							}
						}
					}
					mu_vals[mu] = dot_val;
					if (need_derivatives) {
						mu_ders[mu] = dot_der;
						if (need_param_derivatives) {
							mu_ders_s[mu] = dot_s;
							mu_coord_ders_s[mu] = dot_coord_s;
							mu_ders_ss[mu] = dot_ss;
							mu_coord_ders_ss[mu] = dot_coord_ss;
						}
					}
					}
					if (use_gate_radial) {
						const double* gate_rb_vals = rb_vals;
						const double* gate_rb_ders = rb_ders;
						if (adaptive_compact_radial_dr)
							gate_rb_ders = radial_ders_scratch.data();
						if (decoupled_gate_scaling) {
							double* gate_rb_vals_write = need_gate_raw_radial_cache
								? gate_raw_vals_cache
								: gate_radial_vals_scratch.data();
							for (int eval_block :
							     two_layer_gate_required_radial_eval_blocks_) {
								const int scaling_block =
									radial_eval_to_scaling_block_[eval_block];
								const int basis_k =
									radial_eval_to_basis_k_[eval_block];
								const double sigma =
									TwoLayerGateScalingSlope(
										scaling_block,
										type_central,
										type_outer);
								const double shift =
									TwoLayerGateScalingShift(
										scaling_block,
										type_central,
										type_outer);
								if (coordinate_only)
									p_RadialBasis->RB_CalcRDerOnly(r, sigma, shift, basis_k);
								else if (need_derivatives)
									p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
								else
									p_RadialBasis->RB_CalcValsOnly(
										r, sigma, shift, basis_k);
								for (int xi = 0; xi < R; ++xi)
									gate_rb_vals_write[
										eval_block * R + xi] =
										p_RadialBasis->rb_vals[xi] * scaling;
								if (need_derivatives) {
									for (int xi = 0; xi < radial_der_copy_count; ++xi)
										gate_radial_ders_scratch[
											eval_block * R * 5 + xi] =
											p_RadialBasis->rb_ders[xi] * scaling;
									if (need_gate_raw_radial_cache)
										for (int xi = 0; xi < R; ++xi)
											gate_raw_dr_cache[eval_block * R + xi] =
												p_RadialBasis->rb_ders[xi] * scaling;
								}
							}
							gate_rb_vals = gate_rb_vals_write;
							gate_rb_ders = need_derivatives
								? gate_radial_ders_scratch.data()
								: nullptr;
						}
						for (int dense_mu = 0;
						     dense_mu < static_cast<int>(two_layer_gate_required_mu_indices_.size());
						     ++dense_mu) {
						const int mu = two_layer_gate_required_mu_indices_[dense_mu];
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int deriv_base = 5 * radial_base;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double dot_val = 0.0;
						double dot_der = 0.0;
						double dot_s = 0.0;
						double dot_coord_s = 0.0;
						double dot_ss = 0.0;
						double dot_coord_ss = 0.0;
							for (int xi = 0; xi < R; ++xi) {
								const double coeff = regression_coeffs[radial_offset + xi];
								dot_val += coeff * gate_rb_vals[radial_base + xi];
								if (need_derivatives) {
									dot_der += coeff * gate_rb_ders[deriv_base + xi];
									if (need_param_derivatives) {
										dot_s += coeff * gate_rb_ders[deriv_base + xi + R];
										dot_coord_s +=
											coeff * gate_rb_ders[deriv_base + xi + 2 * R];
										dot_ss += coeff * gate_rb_ders[deriv_base + xi + 3 * R];
										dot_coord_ss +=
											coeff * gate_rb_ders[deriv_base + xi + 4 * R];
									}
								}
							}
						gate_mu_vals[dense_mu] = dot_val;
						if (need_derivatives) {
							gate_mu_ders[dense_mu] = dot_der;
							if (need_param_derivatives) {
								gate_mu_ders_s[dense_mu] = dot_s;
								gate_mu_coord_ders_s[dense_mu] = dot_coord_s;
								gate_mu_ders_ss[dense_mu] = dot_ss;
								gate_mu_coord_ders_ss[dense_mu] = dot_coord_ss;
							}
						}
					}
					}
					}
					const double kernel_start = pure_w_execution_.profile ? SHAccumProfileNow() : 0.0;
					if (pure_w_execution_.basis && need_derivatives) {
						double* dr = pure_w_execution_.raw_dr.data() + edge * radial_val_stride;
						for (int eval = 0; eval < eval_block_count; ++eval)
							std::copy(rb_ders + eval*R*5, rb_ders + eval*R*5 + R, dr + eval*R);
					}
					if (cache_radial_mixing_kernels && pure_w_execution_.packed) {
						pure_w_execution_.Kernels(mu_to_radial_eval_block_.data(), rb_vals,
							rb_ders, radial_mixing_kernels,
							need_derivatives ? radial_mixing_kernel_ders : nullptr);
					} else if (cache_radial_mixing_kernels) {
							for (int l = 0; l <= sh_l_max_; ++l) {
							for (int a = 0; a < outer_k_rank_; ++a) {
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const int radial_base = mu_to_radial_eval_block_[mu] * R;
									const int radial_der_base = 5 * radial_base;
									const size_t lac =
										(static_cast<size_t>(l) * outer_k_rank_ + a)
										* sh_k_max_ + k;
									const double* coeff_row = radial_mixing_coeffs
										+ lac * static_cast<size_t>(R);
									double kernel = 0.0;
									double kernel_der = 0.0;
									for (int xi = 0; xi < R; ++xi) {
										const double coeff = coeff_row[xi];
										kernel += coeff * rb_vals[radial_base + xi];
										if (need_derivatives)
											kernel_der += coeff
												* rb_ders[radial_der_base + xi];
									}
									radial_mixing_kernels[lac] = kernel;
									if (need_derivatives)
										radial_mixing_kernel_ders[lac] = kernel_der;
								}
							}
						}
					}
					if (pure_w_execution_.profile)
						pure_w_execution_.kernel_seconds += SHAccumProfileNow()-kernel_start;
							if (prepare_moment_values_from_cache) {
						const double* gate_mu_values = use_gate_radial ? gate_mu_vals : mu_vals;
							for (int bi = 0;
							     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
							     ++bi) {
					const int basic_index = two_layer_gate_required_basic_indices_[bi];
					const int mu = basic_mu_cache_[basic_index];
						const double gate_center_type_coeff =
							gate_type_coeff_matrix != nullptr
								? gate_type_coeff_matrix[
									static_cast<size_t>(type_central) * radial_func_count + mu]
								: TwoLayerGateTypeCoeff(type_central, mu);
						const double gate_outer_type_coeff =
							gate_type_coeff_matrix != nullptr
								? gate_type_coeff_matrix[
									static_cast<size_t>(type_outer) * radial_func_count + mu]
								: TwoLayerGateTypeCoeff(type_outer, mu);
					const double type_scale =
						gate_center_type_coeff * gate_outer_type_coeff;
					const int mu_index =
						use_gate_radial
						? two_layer_gate_required_basic_dense_mu_indices_[bi]
						: mu;
				const int sh_index = basic_sh_index_cache_[basic_index];
				moment_vals[basic_index] +=
					type_scale * gate_mu_values[mu_index] * sh_values[sh_index];
				}
			}
			}
				if (prepare_moment_values_from_cache) {
					ContractTwoLayerGateKRankValues(moment_vals);
					for (int product_index : two_layer_gate_required_product_indices_) {
						const SHProduct& product = sh_products_[product_index];
					moment_vals[product.target] +=
						product.coeff * moment_vals[product.left]
						* moment_vals[product.right];
				}
				if (prepare_edge_l1_values_from_cache) {
					double* edge_l1_values =
						two_layer_gate_edge_l1_values_.data()
						+ static_cast<size_t>(ind) * edge_l1_source_count * 3;
					FillTwoLayerGateEdgeL1ValuesFromMoments(edge_l1_values);
				}
				double* cached_moments = two_layer_gate_moment_values_cache_.data()
					+ static_cast<size_t>(ind) * cached_gate_moment_count;
				for (int i = 0; i < cached_gate_moment_count; ++i)
					cached_moments[i] =
						moment_vals[two_layer_gate_required_moment_indices_[i]];
				if (!prepare_gate_values_from_cache)
					continue;
				double* cached_scalars = two_layer_gate_scalar_values_cache_.data()
					+ static_cast<size_t>(ind) * gate_count;
				for (int q = 0; q < gate_count; ++q) {
					const int moment = TwoLayerGateScalarMomentIndexUnchecked(q);
					cached_scalars[q] = moment_vals[moment];
				}
				double* gate_values = two_layer_gate_values_.data()
					+ static_cast<size_t>(ind) * radial_func_count;
				if (TwoLayerGateUsesBodyLinearCombo()) {
					double* body_values = two_layer_gate_body_values_cache_.data()
						+ static_cast<size_t>(ind) * TwoLayerGateBodyOrderCount();
					ComputeTwoLayerGateBodySignals(cached_scalars, body_values);
					ComputeTwoLayerGateMuSignals(body_values, gate_values);
				} else if (TwoLayerGateUsesFullScalarWeights()) {
					(void)gate_values;
				} else {
					ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
				}
			}
		}
		if (prepare_gate_values_from_cache && TwoLayerGateUsesFullScalarWeights()
		    && !OuterUsesAdaptiveRadialCoefficients())
			ComputeTwoLayerGateFullMuSignalsForAtoms(atom_count);
		if (prepare_edge_l1_values_from_cache)
			PrepareTwoLayerGateEdgeL1WeightedValues(atom_count, edge_count);
		if (prepare_gate_values_from_cache)
			two_layer_gate_values_from_edge_cache_ready_ = true;

	two_layer_edge_cache_ready_ = true;
	two_layer_edge_cache_has_derivatives_ = need_derivatives;
	two_layer_edge_cache_has_param_derivatives_ =
		need_derivatives && need_param_derivatives;
	two_layer_edge_cache_atom_count_ = atom_count;
	two_layer_edge_cache_eval_block_count_ = eval_block_count;
	two_layer_edge_cache_sh_count_ = sh_count;
	two_layer_edge_cache_radial_func_count_ = radial_func_count;
	two_layer_edge_cache_gate_mu_count_ = gate_mu_stride;
	two_layer_edge_cache_rb_size_ = R;
}

void MLMTPR::PrepareTwoLayerEdgeGateTypeScaleCache(
	const Neighborhoods& neighborhoods)
{
	two_layer_edge_gate_type_scale_cache_.clear();
	two_layer_edge_gate_additive_cache_.clear();
	two_layer_edge_gate_neighbor_multiplier_cache_.clear();
	if (!two_layer_gate_enabled_ || radial_func_count <= 0)
		return;
	if (EdgeL1PairCacheDisabled())
		return;
	if (TwoLayerGateUsesCenterGate())
		return;
	const bool has_edge_l1 = HasNonzeroTwoLayerGateEdgeL1Weights();
	if (!EdgeL1PairCacheEnabled() && !has_edge_l1)
		return;
	if (!two_layer_edge_cache_ready_ || two_layer_edge_offsets_cache_.empty())
		return;
	const int atom_count = neighborhoods.size();
	if (atom_count <= 0 || atom_count != two_layer_edge_cache_atom_count_)
		return;
	if (two_layer_gate_values_.size()
	    != static_cast<size_t>(atom_count) * radial_func_count)
		return;
	if (static_cast<int>(two_layer_edge_offsets_cache_.size()) != atom_count + 1)
		return;

	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	if (two_layer_edge_cache_sh_count_ != sh_count)
		return;
	const size_t edge_count = two_layer_edge_offsets_cache_.back();
	if (edge_count == 0)
		return;
	if (two_layer_edge_sh_values_cache_.size() < edge_count * sh_count)
		return;

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	std::vector<double> type_coeff_by_type_mu(
		static_cast<size_t>(C) * radial_func_count, 0.0);
	std::vector<double> additive_coeff_by_type_mu(
		static_cast<size_t>(C) * radial_func_count, 1.0);
	for (int type = 0; type < C; ++type) {
		const double* additive_coeff_row =
			TwoLayerGateAdditiveCoeffRowData(type);
		for (int mu = 0; mu < radial_func_count; ++mu) {
			type_coeff_by_type_mu[
				static_cast<size_t>(type) * radial_func_count + mu] =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   R,
				                   type,
				                   mu);
			const double additive_coeff =
				additive_coeff_row != nullptr
					? additive_coeff_row[mu]
					: TwoLayerGateAdditiveCoeff(type, mu);
			additive_coeff_by_type_mu[
				static_cast<size_t>(type) * radial_func_count + mu] =
				additive_coeff;
		}
	}
	two_layer_edge_gate_type_scale_cache_.resize(edge_count * radial_func_count);
	two_layer_edge_gate_additive_cache_.resize(edge_count * radial_func_count);
	two_layer_edge_gate_neighbor_multiplier_cache_.resize(
		edge_count * radial_func_count);

	const int source_count = has_edge_l1 ? TwoLayerGateEdgeL1SourceCount() : 0;
	const double* edge_l1_weights =
		has_edge_l1 ? TwoLayerGateEdgeL1WeightData() : nullptr;
	if (has_edge_l1) {
		if (sh_count < 4)
			ERROR("SUS2-SH edge L1 gate requires sh_l_max >= 1.");
		const size_t expected_l1_size =
			static_cast<size_t>(atom_count) * source_count * 3;
		if (two_layer_gate_edge_l1_values_.size() != expected_l1_size)
			ERROR("SUS2-SH edge L1 gate value cache has inconsistent size");
	}
	const size_t expected_weighted_size =
		static_cast<size_t>(atom_count) * radial_func_count * 3;
	const bool has_weighted_edge_l1 =
		has_edge_l1
		&& two_layer_gate_edge_l1_weighted_values_.size()
			== expected_weighted_size;
	const double* gate_values = two_layer_gate_values_.data();
	for (int ind = 0; ind < atom_count; ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		const int type_central = nbh.my_type;
		if (type_central >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
		for (int j = 0; j < nbh.count; ++j) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(ind, j);
			const int type_outer = nbh.types[j];
			if (type_outer >= species_count)
				throw MlipException("Too few species count in the MTP potential!");
			const int gate_neighbor_atom_index = nbh.inds[j];
			const double* sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			const double* base_gate_signal_by_mu = nullptr;
			const double* weighted_values = nullptr;
			const double* edge_l1_values = nullptr;
			if (gate_neighbor_atom_index >= 0) {
				if (gate_neighbor_atom_index >= atom_count)
					ERROR("SUS2-SH edge L1 gate neighbor index is out of range");
				base_gate_signal_by_mu =
					gate_values
					+ static_cast<size_t>(gate_neighbor_atom_index)
						* radial_func_count;
				if (has_weighted_edge_l1) {
					weighted_values =
						two_layer_gate_edge_l1_weighted_values_.data()
						+ static_cast<size_t>(gate_neighbor_atom_index)
							* radial_func_count * 3;
				} else if (has_edge_l1) {
					edge_l1_values =
						two_layer_gate_edge_l1_values_.data()
						+ static_cast<size_t>(gate_neighbor_atom_index)
							* source_count * 3;
				}
			}
			const double y1[3] = {sh_values[1], sh_values[2], sh_values[3]};
			double* type_scale_row =
				two_layer_edge_gate_type_scale_cache_.data()
				+ edge * radial_func_count;
			double* additive_row =
				two_layer_edge_gate_additive_cache_.data()
				+ edge * radial_func_count;
			double* multiplier_row =
				two_layer_edge_gate_neighbor_multiplier_cache_.data()
				+ edge * radial_func_count;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double center_type_coeff =
					type_coeff_by_type_mu[
						static_cast<size_t>(type_central) * radial_func_count + mu];
				const double outer_type_coeff =
					type_coeff_by_type_mu[
						static_cast<size_t>(type_outer) * radial_func_count + mu];
				const double additive_coeff =
					additive_coeff_by_type_mu[
						static_cast<size_t>(type_outer) * radial_func_count + mu];
				double gate_signal =
					base_gate_signal_by_mu == nullptr
						? 0.0
						: base_gate_signal_by_mu[mu];
				if (weighted_values != nullptr) {
					const double* u = weighted_values + mu * 3;
					gate_signal +=
						u[0] * y1[0] + u[1] * y1[1] + u[2] * y1[2];
					} else if (edge_l1_values != nullptr) {
						const double* weight_row =
							edge_l1_weights
							+ static_cast<size_t>(mu) * source_count;
						double extra = 0.0;
						for (int source = 0; source < source_count; ++source) {
							const double* h = edge_l1_values + source * 3;
							extra +=
								weight_row[source]
								* (h[0] * y1[0] + h[1] * y1[1] + h[2] * y1[2]);
						}
					gate_signal += extra;
				}
				type_scale_row[mu] =
					center_type_coeff
					* (outer_type_coeff + additive_coeff * gate_signal);
				additive_row[mu] = additive_coeff;
				multiplier_row[mu] = 1.0;
			}
		}
	}

	if (EdgeL1PairCacheProfileEnabledOnRank0()) {
		std::cout << "[SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE]"
		          << " pair_cache_edges=" << edge_count
		          << " pair_cache_mu=" << radial_func_count
		          << " pair_cache_center_gate=0"
		          << " pair_cache_fast_path=1"
		          << " pair_cache_weighted_l1=" << (has_weighted_edge_l1 ? 1 : 0)
		          << std::endl;
	}
}

void MLMTPR::AccumulateTwoLayerGateRadialMixingForwardEdge(
	const double* gate_input_by_mu,
	const double* radial_basis_values,
	const double* sh_values,
	double* outer_delta,
	const double* cached_mixing_kernels)
{
	if (!two_layer_gate_radial_mixing_active_eval_)
		return;
	const int R = p_RadialBasis->rb_size;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const double* mixing_coeffs =
		regression_coeffs.data() + TwoLayerGateRadialMixingCoeffOffset();
	two_layer_gate_radial_mixing_outer_rank_buffer_.resize(outer_k_rank_);
	double* outer_rank =
		two_layer_gate_radial_mixing_outer_rank_buffer_.data();

	// Direct residual: sum_k M_lak(r) * u_ij,kl. The baseline R_kl(r)
	// and the upstream gate-rank contraction do not belong to this branch.
	for (int l = 0; l <= sh_l_max_; ++l) {
		std::fill(outer_rank, outer_rank + outer_k_rank_, 0.0);
		for (int a = 0; a < outer_k_rank_; ++a) {
			for (int k = 0; k < sh_k_max_; ++k) {
				const int mu = k * (sh_l_max_ + 1) + l;
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int ak = a * sh_k_max_ + k;
				double kernel;
				if (cached_mixing_kernels != nullptr) {
					kernel = cached_mixing_kernels[
						static_cast<size_t>(l) * outer_k_rank_
						* sh_k_max_ + ak];
				} else {
					const double* coeff_row = mixing_coeffs
						+ (((l * outer_k_rank_ + a)
						    * sh_k_max_ + k) * R);
					kernel = 0.0;
					for (int xi = 0; xi < R; ++xi)
						kernel += coeff_row[xi]
							* radial_basis_values[radial_base + xi];
				}
				outer_rank[a] += kernel * gate_input_by_mu[mu];
			}
		}
		for (int m = -l; m <= l; ++m) {
			const int sh_index = SHFlatIndex(l, m);
			const double y = sh_values[sh_index];
			for (int a = 0; a < outer_k_rank_; ++a)
				outer_delta[static_cast<size_t>(a) * sh_count + sh_index]
					+= outer_rank[a] * y;
		}
	}
}

void MLMTPR::AccumulatePureWOuterBlockSecondMoments(
	const Configuration& cfg, std::vector<double>& sums, std::uint64_t& site_count)
{
	if (!OuterUsesPureRadialMatrix())
		ERROR("Outer block RMS statistics require a pure distance-matrix model");
	ValidateOuterModeConfiguration();
	const size_t blocks = static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_;
	if (!sums.empty() && sums.size() != blocks)
		ERROR("Pure W RMS accumulator must have (l_max+1)*outer_rank entries");
	for (double value : sums)
		if (!std::isfinite(value) || value < 0.0)
			ERROR("Pure W RMS accumulator must be non-negative and finite");
	if (site_count > std::numeric_limits<std::uint64_t>::max()
	                   - static_cast<std::uint64_t>(cfg.size()))
		ERROR("Pure W RMS site count overflow");
	for (int atom = 0; atom < cfg.size(); ++atom) {
		if (cfg.type(atom) < 0 || cfg.type(atom) >= species_count)
			ERROR("Pure W RMS configuration contains an unsupported atom type");
		for (int axis = 0; axis < 3; ++axis)
			if (!std::isfinite(cfg.pos(atom)[axis]))
				ERROR("Pure W RMS configuration contains non-finite coordinates");
	}
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			if (!std::isfinite(cfg.lattice[i][j]))
				ERROR("Pure W RMS configuration contains a non-finite lattice");
	std::vector<long double> added(blocks, 0.0L);
	// PrepareTwoLayerGateValues has a mutable legacy signature. A private
	// copy prevents the initialization diagnostic from changing input E/F/S.
	Configuration work_cfg(cfg);
	Neighborhoods neighborhoods(work_cfg, p_RadialBasis->max_dist);
	const auto clear_transient_state = [this]() {
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		active_second_layer_gate_values_ = nullptr;
		active_second_layer_gate_adjoints_ = nullptr;
		ClearTwoLayerEdgePrimitiveCache();
	};
	try {
		PrepareEvalCaches();
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, false, false);
		PrepareTwoLayerGateValues(work_cfg, neighborhoods);
		active_two_layer_gate_values_ = &two_layer_gate_values_;
		active_two_layer_gate_adjoints_ = nullptr;
		for (int atom = 0; atom < cfg.size(); ++atom) {
			active_two_layer_edge_cache_atom_index_ = atom;
			CalcSHMomentValuesOnly(neighborhoods[atom]);
			for (int l = 0; l <= sh_l_max_; ++l)
				for (int o = 0; o < outer_k_rank_; ++o) {
					long double square_norm = 0.0L;
					for (int m = -l; m <= l; ++m) {
						const double value = moment_vals[OuterKRank1OuterBasicIndex(o, l, m)];
						if (!std::isfinite(value))
							ERROR("Pure W RMS statistics encountered a non-finite outer primitive");
						square_norm += static_cast<long double>(value) * value;
					}
					added[static_cast<size_t>(l) * outer_k_rank_ + o] += square_norm / (2 * l + 1);
				}
		}
		clear_transient_state();
	} catch (...) {
		clear_transient_state();
		throw;
	}
	std::vector<double> updated(blocks, 0.0);
	for (size_t block = 0; block < blocks; ++block) {
		const long double value = added[block] + (sums.empty() ? 0.0L : sums[block]);
		if (!std::isfinite(value) || value > std::numeric_limits<double>::max())
			ERROR("Pure W RMS second-moment accumulation overflow");
		updated[block] = static_cast<double>(value);
	}
	sums.swap(updated);
	site_count += static_cast<std::uint64_t>(cfg.size());
}

void MLMTPR::CalcSHMomentValuesOnly(const Neighborhood& nbh)
{
	// Dispatch once so legacy models execute the release3 kernel without
	// radial-mixing branches or scratch-state traffic in the neighbor loop.
	if (TwoLayerGateRadialMixingEnabled()) {
		CalcSHMomentValuesOnlyRadialMixing(nbh);
		return;
	}
	two_layer_gate_radial_mixing_active_eval_ = false;
	CalcSHMomentValuesOnlyLegacy(nbh);
}

void MLMTPR::CalcSHMomentValuesOnlyLegacy(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;

	const int radial_coeff_base = C + 2 * C * C * K_;
	const double* active_gate_values_data =
		active_two_layer_gate_values_ == nullptr
			? nullptr
			: active_two_layer_gate_values_->data();
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;
	if (!use_basic_mu_groups
	    && static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
		       != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (BasicMuForwardProfileEnabledOnRank0()) {
		static bool profile_printed = false;
		if (!profile_printed) {
			std::cout << "[SUS2_SH_BASIC_MU_FORWARD_PROFILE]"
			          << " forward_grouped=" << (use_basic_mu_groups ? 1 : 0)
			          << " radial_func_count=" << radial_func_count
			          << " basic_count=" << alpha_index_basic_count
			          << std::endl;
			profile_printed = true;
		}
	}

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

				const double* gate_type_scale_by_mu = nullptr;
					if (use_edge_pair_gate_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
						gate_type_scale_by_mu =
							two_layer_edge_gate_type_scale_cache_.data()
						+ edge * radial_func_count;
					} else {
						const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
					const int gate_neighbor_atom_index = nbh.inds[j];
					const double* base_gate_signal_by_mu =
						active_gate_values_data == nullptr
							? nullptr
							: active_gate_values_data
								+ static_cast<size_t>(gate_neighbor_atom_index) * radial_func_count;
					int gate_neighbor_atom_index_for_tanh = gate_neighbor_atom_index;
					const double* gate_signal_by_mu =
						PrepareTwoLayerGateEdgeL1NeighborSignals(
							nbh, j, base_gate_signal_by_mu, sh_values_use,
							gate_neighbor_atom_index_for_tanh);
					const double* center_gate_signal_by_mu =
						(active_gate_values_data == nullptr || gate_center_atom_index < 0)
							? nullptr
							: active_gate_values_data
								+ static_cast<size_t>(gate_center_atom_index)
									* radial_func_count;
					PrepareTwoLayerGatePairMuBuffers(
						type_central, type_outer,
						center_gate_signal_by_mu, gate_center_atom_index,
						gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
						kGateMuBufferTypeScale);
					gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
				}
				if (use_basic_mu_groups) {
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double scaled_radial =
							gate_type_scale_by_mu[mu] * radial_values_use[mu];
						if (scaled_radial == 0.0)
							continue;
						const int begin = basic_mu_offsets_[mu];
						const int end = basic_mu_offsets_[mu + 1];
						for (int cursor = begin; cursor < end; ++cursor) {
							const int i = basic_indices_by_mu_[cursor];
							const int sh_index = basic_sh_index_cache_[i];
							moment_vals[i] +=
								scaled_radial * sh_values_use[sh_index];
						}
					}
				} else {
					double* gate_scaled_radial_by_mu =
						two_layer_gate_scaled_mu_vals_buffer_.data();
					for (int mu = 0; mu < radial_func_count; ++mu)
						gate_scaled_radial_by_mu[mu] =
							gate_type_scale_by_mu[mu] * radial_values_use[mu];

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
						const int sh_index = basic_sh_index_cache_[i];
						moment_vals[i] +=
							gate_scaled_radial_by_mu[mu] * sh_values_use[sh_index];
					}
				}
		}

	ContractOuterKRank1Values(moment_vals);
	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}


void MLMTPR::CalcSHMomentValuesOnlyRadialMixing(const Neighborhood& nbh)
{
	if (pure_w_execution_.basis && HasTwoLayerEdgePrimitiveCache(-1, false, false)) {
		CalcPureWBasisFirstMoments(nbh);
		return;
	}
	two_layer_gate_radial_mixing_active_eval_ =
		TwoLayerGateRadialMixingEnabled()
		&& (OuterUsesPureRadialMatrix() || HasNonzeroTwoLayerGateRadialMixingCoeffs());
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (TwoLayerGateRadialMixingEnabled() && use_lmp_table)
		ERROR("SUS2-SH two-layer gate radial mixing requires direct radial basis evaluation.");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;

	const int radial_coeff_base = C + 2 * C * C * K_;
	const double* active_gate_values_data =
		active_two_layer_gate_values_ == nullptr
			? nullptr
			: active_two_layer_gate_values_->data();
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;
	if (!use_basic_mu_groups
	    && static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
		       != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (BasicMuForwardProfileEnabledOnRank0()) {
		static bool profile_printed = false;
		if (!profile_printed) {
			std::cout << "[SUS2_SH_BASIC_MU_FORWARD_PROFILE]"
			          << " forward_grouped=" << (use_basic_mu_groups ? 1 : 0)
			          << " radial_func_count=" << radial_func_count
			          << " basic_count=" << alpha_index_basic_count
			          << std::endl;
			profile_printed = true;
		}
	}
	const bool radial_mixing_active =
		two_layer_gate_radial_mixing_active_eval_;
	const size_t radial_mixing_kernel_stride =
		static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_
		* sh_k_max_;
	// Exact zero-C fast path: the mixer contribution vanishes identically.
	std::vector<double>& radial_mixing_outer_delta =
		two_layer_gate_radial_mixing_outer_delta_buffer_;
	if (radial_mixing_active) {
		radial_mixing_outer_delta.assign(
			static_cast<size_t>(outer_k_rank_) * sh_count, 0.0);
		two_layer_gate_radial_mixing_outer_rank_buffer_.resize(outer_k_rank_);
	}
	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		const double* radial_mixing_rb_vals = rb_vals.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			if (radial_mixing_active) {
				const size_t radial_val_stride =
					static_cast<size_t>(two_layer_edge_cache_eval_block_count_) * R;
				const bool use_cached_raw_radial =
					two_layer_edge_radial_vals_cache_.size()
						>= (edge + 1) * radial_val_stride;
				if (use_cached_raw_radial) {
					radial_mixing_rb_vals =
						two_layer_edge_radial_vals_cache_.data()
						+ edge * radial_val_stride;
				} else {
					for (int eval_block = 0;
					     eval_block < static_cast<int>(radial_eval_to_scaling_block_.size());
					     ++eval_block) {
						const int scaling_block =
							radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_CalcValsOnly(
							r,
							regression_coeffs[
								C + 2 * scaling_block * C * C
								+ C * type_central + type_outer],
							regression_coeffs[
								C + 2 * scaling_block * C * C + C * C
								+ C * type_central + type_outer],
							basis_k);
						for (int xi = 0; xi < R; ++xi)
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
					}
				}
			}
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

				const double* gate_type_scale_by_mu = nullptr;
				const double* gate_signal_by_mu = nullptr;
					if (use_edge_pair_gate_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
						gate_type_scale_by_mu =
							two_layer_edge_gate_type_scale_cache_.data()
						+ edge * radial_func_count;
					} else {
						const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
					const int gate_neighbor_atom_index = nbh.inds[j];
					const double* base_gate_signal_by_mu =
						active_gate_values_data == nullptr
							? nullptr
							: active_gate_values_data
								+ static_cast<size_t>(gate_neighbor_atom_index) * radial_func_count;
					int gate_neighbor_atom_index_for_tanh = gate_neighbor_atom_index;
					gate_signal_by_mu =
						PrepareTwoLayerGateEdgeL1NeighborSignals(
							nbh, j, base_gate_signal_by_mu, sh_values_use,
							gate_neighbor_atom_index_for_tanh);
					const double* center_gate_signal_by_mu =
						(active_gate_values_data == nullptr || gate_center_atom_index < 0)
							? nullptr
							: active_gate_values_data
								+ static_cast<size_t>(gate_center_atom_index)
									* radial_func_count;
					PrepareTwoLayerGatePairMuBuffers(
						type_central, type_outer,
						center_gate_signal_by_mu, gate_center_atom_index,
						gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
						kGateMuBufferTypeScale);
					gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
				}
				const double* cached_mixing_kernels =
					use_edge_cache && radial_mixing_active
					? two_layer_edge_radial_mixing_kernel_cache_.data()
						+ TwoLayerEdgePrimitiveOffset(-1, j)
							* radial_mixing_kernel_stride
					: nullptr;
				if (radial_mixing_active)
					AccumulateTwoLayerGateRadialMixingForwardEdge(
						gate_type_scale_by_mu,
						radial_mixing_rb_vals,
						sh_values_use,
						radial_mixing_outer_delta.data(),
						cached_mixing_kernels);
				if (OuterUsesPureRadialMatrix())
					continue;
				if (use_basic_mu_groups) {
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double scaled_radial =
							gate_type_scale_by_mu[mu] * radial_values_use[mu];
						if (scaled_radial == 0.0)
							continue;
						const int begin = basic_mu_offsets_[mu];
						const int end = basic_mu_offsets_[mu + 1];
						for (int cursor = begin; cursor < end; ++cursor) {
							const int i = basic_indices_by_mu_[cursor];
							const int sh_index = basic_sh_index_cache_[i];
							moment_vals[i] +=
								scaled_radial * sh_values_use[sh_index];
						}
					}
				} else {
					double* gate_scaled_radial_by_mu =
						two_layer_gate_scaled_mu_vals_buffer_.data();
					for (int mu = 0; mu < radial_func_count; ++mu)
						gate_scaled_radial_by_mu[mu] =
							gate_type_scale_by_mu[mu] * radial_values_use[mu];

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
						const int sh_index = basic_sh_index_cache_[i];
						moment_vals[i] +=
							gate_scaled_radial_by_mu[mu] * sh_values_use[sh_index];
					}
				}
		}

	ContractOuterKRank1Values(moment_vals);
	if (radial_mixing_active)
		for (int a = 0; a < outer_k_rank_; ++a)
			for (int l = 0; l <= sh_l_max_; ++l)
				for (int m = -l; m <= l; ++m)
					moment_vals[OuterKRank1OuterBasicIndex(a, l, m)] +=
						radial_mixing_outer_delta[
							static_cast<size_t>(a) * sh_count
							+ SHFlatIndex(l, m)];
	if (ApplyPureWGraph(false)) {
		// One angular instruction stream, O independent value lanes.
	} else if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHEdgeBasicValues(const Neighborhood& nbh,
                                   int neighbor_index,
                                   std::vector<double>& edge_basic_values)
{
	if (!is_sh_potential_)
		ERROR("calc-eij currently supports SUS2-SH models only");
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		ERROR("calc-eij neighbor index is out of range");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	edge_basic_values.assign(alpha_moments_count, 0.0);

	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const Vector3& rvec = nbh.vecs[neighbor_index];
	const double r = nbh.dists[neighbor_index];
	const int type_outer = nbh.types[neighbor_index];
	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int radial_coeff_base = C + 2 * C * C * K_;

	const double* sh_values_use = sh_values;
	const double* radial_values_use = radial_values.data();
	if (HasTwoLayerEdgePrimitiveCache(-1, false, false)) {
		const size_t edge = TwoLayerEdgePrimitiveOffset(-1, neighbor_index);
		sh_values_use =
			two_layer_edge_sh_values_cache_.data() + edge * sh_count;
		radial_values_use =
			two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

		if (use_lmp_table) {
			InterpolateSHLmpMuTable(radial_list,
			                         radial_der_list,
			                         inv_dr,
			                         C,
			                         type_central,
			                         type_outer,
			                         r,
			                         radial_func_count,
			                         radial_values.data(),
			                         nullptr);
		} else {
			for (int eval_block = 0;
			     eval_block < static_cast<int>(radial_eval_to_scaling_block_.size());
			     ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_CalcValsOnly(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C
					                  + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C
					                  + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] =
						p_RadialBasis->rb_vals[xi] * scaling;
			}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double radial_value = 0.0;
				for (int xi = 0; xi < R; ++xi)
					radial_value +=
						regression_coeffs[radial_offset + xi]
						* rb_vals[radial_base + xi];
				radial_values[mu] = radial_value;
			}
		}
	}

	const double* active_gate_values_data =
		active_two_layer_gate_values_ == nullptr
			? nullptr
			: active_two_layer_gate_values_->data();
	const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
	const int gate_neighbor_atom_index = nbh.inds[neighbor_index];
	const double* base_gate_signal_by_mu =
		active_gate_values_data == nullptr
			? nullptr
			: active_gate_values_data
				+ static_cast<size_t>(gate_neighbor_atom_index) * radial_func_count;
	int gate_neighbor_atom_index_for_tanh = gate_neighbor_atom_index;
	const double* gate_signal_by_mu =
		PrepareTwoLayerGateEdgeL1NeighborSignals(
			nbh, neighbor_index, base_gate_signal_by_mu, sh_values_use,
			gate_neighbor_atom_index_for_tanh);
	const double* center_gate_signal_by_mu =
		(active_gate_values_data == nullptr || gate_center_atom_index < 0)
			? nullptr
			: active_gate_values_data
				+ static_cast<size_t>(gate_center_atom_index) * radial_func_count;
	PrepareTwoLayerGatePairMuBuffers(
		type_central, type_outer,
		center_gate_signal_by_mu, gate_center_atom_index,
		gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
		kGateMuBufferTypeScale);

	const double* gate_type_scale_by_mu =
		two_layer_gate_type_scale_mu_buffer_.data();
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;
	if (use_basic_mu_groups) {
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double scaled_radial =
				gate_type_scale_by_mu[mu] * radial_values_use[mu];
			for (int cursor = basic_mu_offsets_[mu];
			     cursor < basic_mu_offsets_[mu + 1];
			     ++cursor) {
				const int i = basic_indices_by_mu_[cursor];
				edge_basic_values[i] =
					scaled_radial * sh_values_use[basic_sh_index_cache_[i]];
			}
		}
	} else {
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const int mu = basic_mu_cache_[i];
			const int sh_index = basic_sh_index_cache_[i];
			edge_basic_values[i] = gate_type_scale_by_mu[mu]
				* radial_values_use[mu] * sh_values_use[sh_index];
		}
	}
	ContractOuterKRank1Values(edge_basic_values.data());
}

void MLMTPR::CalcTwoLayerGateEdgeBasicValues(
	const Neighborhood& nbh,
	int neighbor_index,
	int cache_atom_index,
	std::vector<double>& edge_basic_values)
{
	if (!is_sh_potential_ || !two_layer_gate_enabled_)
		ERROR("calc-eij gate readout projection requires a SUS2-SH gate model");
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		ERROR("calc-eij gate neighbor index is out of range");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	edge_basic_values.assign(alpha_moments_count, 0.0);

	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, false, false);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");
	const int type_outer = nbh.types[neighbor_index];
	const Vector3& rvec = nbh.vecs[neighbor_index];
	const double r = nbh.dists[neighbor_index];
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const double* sh_values_use = sh_values;
	const double* radial_values_use = radial_values.data();
	if (use_edge_cache) {
		const size_t edge =
			TwoLayerEdgePrimitiveOffset(cache_atom_index, neighbor_index);
		sh_values_use = two_layer_edge_sh_values_cache_.data() + edge * sh_count;
		radial_values_use =
			(use_gate_radial
				? two_layer_edge_gate_mu_vals_cache_.data()
				: two_layer_edge_mu_vals_cache_.data())
			+ edge * static_cast<size_t>(
				use_gate_radial ? gate_mu_stride : mu_stride);
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);
		if (use_lmp_table) {
			InterpolateSHLmpMuTable(
				use_gate_radial ? two_layer_gate_radial_list : radial_list,
				use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				inv_dr, C, type_central, type_outer, r,
				radial_func_count, radial_values.data(), nullptr);
		} else {
			for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block =
					radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_CalcValsOnly(
					r,
					TwoLayerGateScalingSlope(
						scaling_block, type_central, type_outer),
					TwoLayerGateScalingShift(
						scaling_block, type_central, type_outer),
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] =
						p_RadialBasis->rb_vals[xi] * scaling;
			}
			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				for (int xi = 0; xi < R; ++xi)
					radial_value += regression_coeffs[radial_offset + xi]
						* rb_vals[radial_base + xi];
				radial_values[mu] = radial_value;
			}
		}
	}

	for (int bi = 0;
	     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
	     ++bi) {
		const int basic_index = two_layer_gate_required_basic_indices_[bi];
		const int mu = basic_mu_cache_[basic_index];
		const int mu_index = use_edge_cache && use_gate_radial
			? two_layer_gate_required_basic_dense_mu_indices_[bi]
			: mu;
		const int sh_index = basic_sh_index_cache_[basic_index];
		const double type_scale =
			TwoLayerGateTypeCoeff(type_central, mu)
			* TwoLayerGateTypeCoeff(type_outer, mu);
		edge_basic_values[basic_index] =
			type_scale * radial_values_use[mu_index] * sh_values_use[sh_index];
	}
}

double MLMTPR::CalcSHScalarFirstFactorProjection(
	const SHScalarInfo& scalar_info,
	const std::vector<double>& edge_basic_values) const
{
	const int required_edge_size = outer_k_rank1_enabled_
		? alpha_moments_count
		: alpha_index_basic_count;
	if (edge_basic_values.size() < static_cast<size_t>(required_edge_size))
		ERROR("calc-eij internal edge basic buffer is too small");
	if (scalar_info.body_order < 2 || scalar_info.body_order > 6)
		ERROR("calc-eij found an invalid SH scalar body order");

	auto q_index_to_l_mu = [&](int q_index, int& l, int& mu) {
		if (OuterKRankEnabled()) {
			if (q_index < 0
			    || q_index >= outer_k_rank_ * (sh_l_max_ + 1))
				ERROR("calc-eij found an invalid outer K-rank scalar q index");
			l = sh_l_max_ - q_index / outer_k_rank_;
			mu = outer_k_rank_ - 1 - q_index % outer_k_rank_;
			return;
		}
		if (q_index < 0 || q_index >= sh_k_max_ * (sh_l_max_ + 1))
			ERROR("calc-eij found an invalid SH scalar q index");
		int cursor = 0;
		for (int l_iter = sh_l_max_; l_iter >= 0; --l_iter) {
			for (int k_iter = sh_k_max_ - 1; k_iter >= 0; --k_iter) {
				if (cursor == q_index) {
					l = l_iter;
					mu = k_iter * (sh_l_max_ + 1) + l_iter;
					return;
				}
				++cursor;
			}
		}
		ERROR("calc-eij failed to map SH scalar q index");
	};

	auto tensor_values = [&](int q_index, bool edge) {
		int l = 0;
		int mu = 0;
		q_index_to_l_mu(q_index, l, mu);
		std::vector<double> values(2 * l + 1, 0.0);
		for (int m = -l; m <= l; ++m) {
			int node = OuterKRankEnabled()
				? OuterKRank1OuterBasicIndex(mu, l, m)
				: -1;
			if (!OuterKRankEnabled()) {
				for (int i = 0; i < alpha_index_basic_count; ++i) {
					if (alpha_index_basic_.comp0[i] == mu
					    && alpha_index_basic_.comp1[i] == l
					    && alpha_index_basic_.comp2[i] == m) {
						node = i;
						break;
					}
				}
			}
			if (node < 0)
				ERROR("calc-eij could not find a required SH basic tensor component");
			values[m + l] = edge ? edge_basic_values[node] : moment_vals[node];
		}
		return values;
	};

	auto q_l = [&](int q_index) {
		int l = 0;
		int mu = 0;
		q_index_to_l_mu(q_index, l, mu);
		return l;
	};

	const int q0 = scalar_info.q[0];
	const int l0 = q_l(q0);
	const std::vector<double> t0_edge = tensor_values(q0, true);
	if (scalar_info.body_order == 2)
		return t0_edge[0];

	const int q1 = scalar_info.q[1];
	const int l1 = q_l(q1);
	const std::vector<double> t1_full = tensor_values(q1, false);
	if (scalar_info.body_order == 3) {
		const std::vector<double> scalar =
			CoupleSHTensors(t0_edge, l0, t1_full, l1, 0);
		return scalar[0];
	}

	const int q2 = scalar_info.q[2];
	const int l2 = q_l(q2);
	const std::vector<double> t2_full = tensor_values(q2, false);
	const int intermediate_l = scalar_info.intermediate_l;
	if (intermediate_l < 0 || intermediate_l > 2 * sh_l_max_)
		ERROR("calc-eij found an invalid SH intermediate l");

	const std::vector<double> left =
		CoupleSHTensors(t0_edge, l0, t1_full, l1, intermediate_l);
	if (scalar_info.body_order == 4) {
		const std::vector<double> scalar =
			CoupleSHTensors(left, intermediate_l, t2_full, l2, 0);
		return scalar[0];
	}

	const int q3 = scalar_info.q[3];
	const int l3 = q_l(q3);
	const std::vector<double> t3_full = tensor_values(q3, false);
	const std::vector<double> right_pair =
		CoupleSHTensors(t2_full, l2, t3_full, l3, intermediate_l);
	if (scalar_info.body_order == 5) {
		const std::vector<double> scalar =
			CoupleSHTensors(left, intermediate_l, right_pair, intermediate_l, 0);
		return scalar[0];
	}

	const int q4 = scalar_info.q[4];
	const int l4 = q_l(q4);
	const std::vector<double> t4_full = tensor_values(q4, false);
	const std::vector<double> right =
		CoupleSHTensors(right_pair, intermediate_l, t4_full, l4, intermediate_l);
	const std::vector<double> scalar =
		CoupleSHTensors(left, intermediate_l, right, intermediate_l, 0);
	return scalar[0];
}

bool MLMTPR::UseFastCalcEijProjection() const
{
	if (outer_k_rank1_enabled_)
		return false;
	const char* env = std::getenv("SUS2_SH_CALC_EIJ_FAST");
	return env == nullptr || std::strcmp(env, "0") != 0;
}

void MLMTPR::CalcSHDirectedEffectivePairWeights(
	const Neighborhood& nbh,
	std::vector<double>& edge_basic_weights,
	const std::vector<int>* scalar_indices,
	const std::vector<double>* scalar_coeffs) const
{
	if (!is_sh_potential_)
		ERROR("calc-eij currently supports SUS2-SH models only");
	if (!has_sh_scalar_info_)
		ERROR("calc-eij requires sh_scalar_info metadata; initialize or rewrite the model with --write-sh-scalar-info");
	if (TwoLayerResidualEnabled())
		ERROR("calc-eij does not support two-layer residual models");
	if ((scalar_indices == nullptr) != (scalar_coeffs == nullptr))
		ERROR("calc-eij scalar index and coefficient lists must be provided together");
	if (scalar_indices != nullptr
	    && scalar_indices->size() != scalar_coeffs->size())
		ERROR("calc-eij scalar index and coefficient lists have inconsistent sizes");

	edge_basic_weights.assign(alpha_index_basic_count, 0.0);

	const int q_count = sh_k_max_ * (sh_l_max_ + 1);
	std::vector<int> q_l(q_count, 0);
	std::vector<int> q_nodes(static_cast<size_t>(q_count) * kMaxSHComponents, -1);

	int cursor = 0;
	for (int l_iter = sh_l_max_; l_iter >= 0; --l_iter) {
		for (int k_iter = sh_k_max_ - 1; k_iter >= 0; --k_iter) {
			const int q_index = cursor++;
			const int mu = k_iter * (sh_l_max_ + 1) + l_iter;
			q_l[q_index] = l_iter;
			for (int m = -l_iter; m <= l_iter; ++m) {
				int node = -1;
				for (int i = 0; i < alpha_index_basic_count; ++i) {
					if (alpha_index_basic_.comp0[i] == mu
					    && alpha_index_basic_.comp1[i] == l_iter
					    && alpha_index_basic_.comp2[i] == m) {
						node = i;
						break;
					}
				}
				if (node < 0)
					ERROR("calc-eij could not find a required SH basic tensor component");
				q_nodes[static_cast<size_t>(q_index) * kMaxSHComponents
					+ static_cast<size_t>(m + l_iter)] = node;
			}
		}
	}

	auto require_q = [&](int q_index) {
		if (q_index < 0 || q_index >= q_count)
			ERROR("calc-eij found an invalid SH scalar q index");
		return q_index;
	};

	auto load_center_tensor = [&](int q_index, double* tensor) {
		std::fill(tensor, tensor + kMaxSHComponents, 0.0);
		const int l = q_l[q_index];
		for (int m = -l; m <= l; ++m) {
			const int node = q_nodes[static_cast<size_t>(q_index) * kMaxSHComponents
				+ static_cast<size_t>(m + l)];
			tensor[m + l] = moment_vals[node];
		}
	};

	const double center_linear = linear_coeffs[nbh.my_type];
	double t0[kMaxSHComponents];
	double t1[kMaxSHComponents];
	double t2[kMaxSHComponents];
	double t3[kMaxSHComponents];
	double t4[kMaxSHComponents];
	double left[kMaxSHComponents];
	double right_pair[kMaxSHComponents];
	double right[kMaxSHComponents];
	double scalar[kMaxSHComponents];

	const int selected_scalar_count = scalar_indices == nullptr
		? alpha_scalar_moments
		: static_cast<int>(scalar_indices->size());
	for (int selected = 0; selected < selected_scalar_count; ++selected) {
		const int scalar_index = scalar_indices == nullptr
			? selected
			: (*scalar_indices)[selected];
		if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
			ERROR("calc-eij found an invalid selected SH scalar index");
		const SHScalarInfo& scalar_info = sh_scalar_info_[scalar_index];
		if (scalar_info.body_order < 2 || scalar_info.body_order > 6)
			ERROR("calc-eij found an invalid SH scalar body order");
		const double scalar_coeff = scalar_coeffs == nullptr
			? center_linear
				* linear_coeffs[species_count + scalar_index]
				* linear_mults[scalar_index]
			: (*scalar_coeffs)[selected];
		if (scalar_coeff == 0.0)
			continue;

		const int q0 = require_q(scalar_info.q[0]);
		const int l0 = q_l[q0];
		int q1 = -1;
		int q2 = -1;
		int q3 = -1;
		int q4 = -1;
		int l1 = 0;
		int l2 = 0;
		int l3 = 0;
		int l4 = 0;
		int intermediate_l = 0;
		if (scalar_info.body_order >= 3) {
			q1 = require_q(scalar_info.q[1]);
			l1 = q_l[q1];
			load_center_tensor(q1, t1);
		}
		if (scalar_info.body_order >= 4) {
			q2 = require_q(scalar_info.q[2]);
			l2 = q_l[q2];
			load_center_tensor(q2, t2);
			intermediate_l = scalar_info.intermediate_l;
			if (intermediate_l < 0 || intermediate_l > 2 * sh_l_max_)
				ERROR("calc-eij found an invalid SH intermediate l");
		}
		if (scalar_info.body_order >= 5) {
			q3 = require_q(scalar_info.q[3]);
			l3 = q_l[q3];
			load_center_tensor(q3, t3);
			CoupleSHTensorArrays(
				t2, l2, t3, l3, intermediate_l, right_pair);
		}
		if (scalar_info.body_order >= 6) {
			q4 = require_q(scalar_info.q[4]);
			l4 = q_l[q4];
			load_center_tensor(q4, t4);
			CoupleSHTensorArrays(
				right_pair, intermediate_l, t4, l4, intermediate_l, right);
		}

		for (int m0 = -l0; m0 <= l0; ++m0) {
			const int node = q_nodes[static_cast<size_t>(q0) * kMaxSHComponents
				+ static_cast<size_t>(m0 + l0)];
			double projection = 0.0;
			if (scalar_info.body_order == 2) {
				projection = (m0 == -l0) ? 1.0 : 0.0;
			} else if (scalar_info.body_order == 3) {
				std::fill(t0, t0 + kMaxSHComponents, 0.0);
				t0[m0 + l0] = 1.0;
				CoupleSHTensorArrays(t0, l0, t1, l1, 0, scalar);
				projection = scalar[0];
			} else {
				std::fill(t0, t0 + kMaxSHComponents, 0.0);
				t0[m0 + l0] = 1.0;
				CoupleSHTensorArrays(
					t0, l0, t1, l1, intermediate_l, left);
				if (scalar_info.body_order == 4) {
					CoupleSHTensorArrays(
						left, intermediate_l, t2, l2, 0, scalar);
					projection = scalar[0];
				} else if (scalar_info.body_order == 5) {
					CoupleSHTensorArrays(
						left, intermediate_l, right_pair, intermediate_l, 0, scalar);
					projection = scalar[0];
				} else {
					CoupleSHTensorArrays(
						left, intermediate_l, right, intermediate_l, 0, scalar);
					projection = scalar[0];
				}
			}
			edge_basic_weights[node] += scalar_coeff * projection;
		}
	}
}

double MLMTPR::CalcSHDirectedEffectivePairEnergyFromWeights(
	const Neighborhood& nbh,
	int neighbor_index,
	std::vector<double>& edge_basic_values,
	const std::vector<double>& edge_basic_weights)
{
	if (edge_basic_weights.size() < static_cast<size_t>(alpha_index_basic_count))
		ERROR("calc-eij internal edge basic weight buffer is too small");

	CalcSHEdgeBasicValues(nbh, neighbor_index, edge_basic_values);
	double energy = 0.0;
	for (int i = 0; i < alpha_index_basic_count; ++i)
		energy += edge_basic_weights[i] * edge_basic_values[i];
	return energy;
}

double MLMTPR::CalcSHDirectedEffectivePairEnergy(
	const Neighborhood& nbh,
	int neighbor_index,
	std::vector<double>& edge_basic_values)
{
	if (!is_sh_potential_)
		ERROR("calc-eij currently supports SUS2-SH models only");
	if (!has_sh_scalar_info_)
		ERROR("calc-eij requires sh_scalar_info metadata; initialize or rewrite the model with --write-sh-scalar-info");
	if (TwoLayerResidualEnabled())
		ERROR("calc-eij does not support two-layer residual models");

	CalcSHEdgeBasicValues(nbh, neighbor_index, edge_basic_values);
	const double center_linear = linear_coeffs[nbh.my_type];
	double energy = 0.0;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double scalar_projection =
			CalcSHScalarFirstFactorProjection(sh_scalar_info_[i],
			                                  edge_basic_values);
		energy += center_linear
			* linear_coeffs[species_count + i]
			* linear_mults[i]
			* scalar_projection;
	}
	return energy;
}

void MLMTPR::CalcSHMomentValuesWithSiteDerivativeCache(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);

	TraceSHSiteDerivativeCacheOnce(nbh.count, sh_count, radial_func_count);

	grad_neighbor_sh_values_cache_.resize(neighbor_count * sh_count);
	grad_neighbor_sh_ders_cache_.resize(neighbor_count * 3 * sh_count);
	grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
	grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);

	const int radial_coeff_base = C + 2 * C * C * K_;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		double* cached_sh_values = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
		double* cached_sh_ders = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
		double* cached_radial_values = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
		double* cached_radial_derivatives = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			std::copy(two_layer_edge_sh_values_cache_.data() + edge * sh_count,
			          two_layer_edge_sh_values_cache_.data() + (edge + 1) * sh_count,
			          cached_sh_values);
			std::copy(two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count,
			          two_layer_edge_sh_ders_cache_.data() + (edge + 1) * 3 * sh_count,
			          cached_sh_ders);
			std::copy(two_layer_edge_mu_vals_cache_.data() + edge * mu_stride,
			          two_layer_edge_mu_vals_cache_.data() + (edge + 1) * mu_stride,
			          cached_radial_values);
			std::copy(two_layer_edge_mu_ders_cache_.data() + edge * mu_stride,
			          two_layer_edge_mu_ders_cache_.data() + (edge + 1) * mu_stride,
			          cached_radial_derivatives);
	} else {
		EvalRealSH(rvec, r, sh_l_max_, cached_sh_values, cached_sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         cached_radial_values,
				                         cached_radial_derivatives);
			} else {
				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					cached_radial_values[mu] = radial_value;
					cached_radial_derivatives[mu] = radial_der;
				}
			}
		}

				const double* gate_type_scale_by_mu = nullptr;
				if (use_edge_pair_gate_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
					gate_type_scale_by_mu =
						two_layer_edge_gate_type_scale_cache_.data()
						+ edge * radial_func_count;
					} else {
						int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
					const double* gate_signal_by_mu =
						PrepareTwoLayerGateEdgeL1NeighborSignals(
							nbh, j, TwoLayerGateNeighborSignals(nbh, j),
							cached_sh_values, gate_neighbor_atom_index_for_tanh);
					const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
					const double* center_gate_signal_by_mu =
						TwoLayerGateAtomSignals(gate_center_atom_index);
					PrepareTwoLayerGatePairMuBuffers(
						type_central, type_outer,
						center_gate_signal_by_mu, gate_center_atom_index,
						gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
						kGateMuBufferTypeScale);
					gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
				}
			double* gate_scaled_radial_by_mu =
				two_layer_gate_scaled_mu_vals_buffer_.data();
			for (int mu = 0; mu < radial_func_count; ++mu)
				gate_scaled_radial_by_mu[mu] =
					gate_type_scale_by_mu[mu] * cached_radial_values[mu];

			for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = basic_mu_cache_[i];
				const int sh_index = basic_sh_index_cache_[i];
				moment_vals[i] += gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
			}
	}

	ContractOuterKRank1Values(moment_vals);
	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHMomentValuesWithGradientCache(const Neighborhood& nbh)
{
	// Keep the feature test outside both specialized hot kernels.
	if (TwoLayerGateRadialMixingEnabled()) {
		CalcSHMomentValuesWithGradientCacheRadialMixing(nbh);
		return;
	}
	two_layer_gate_radial_mixing_active_eval_ = false;
	CalcSHMomentValuesWithGradientCacheLegacy(nbh);
}

void MLMTPR::CalcSHMomentValuesWithGradientCacheLegacy(const Neighborhood& nbh)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const int mu_stride = radial_func_count;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, true);
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (!use_edge_cache) {
		grad_neighbor_sh_values_cache_.resize(neighbor_count * sh_count);
		grad_neighbor_sh_ders_cache_.resize(neighbor_count * 3 * sh_count);
		grad_neighbor_radial_vals_cache_.resize(neighbor_count * radial_val_stride);
		grad_neighbor_radial_ders_cache_.resize(neighbor_count * radial_der_stride);
		grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_ss_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_ss_cache_.resize(neighbor_count * mu_stride);
	}

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	const int radial_coeff_base = C + 2 * C * C * K_;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* cached_sh_values = nullptr;
		const double* cached_radial_values = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			cached_radial_values =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
		} else {
			double* cached_sh_values_write =
				grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			double* cached_sh_ders =
				grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			double* cached_rb_vals =
				grad_neighbor_radial_vals_cache_.data()
				+ static_cast<size_t>(j) * radial_val_stride;
			double* cached_rb_ders =
				grad_neighbor_radial_ders_cache_.data()
				+ static_cast<size_t>(j) * radial_der_stride;
			double* cached_radial_values_write =
				grad_neighbor_mu_contract_vals_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_derivatives =
				grad_neighbor_mu_contract_ders_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_s_derivatives =
				grad_neighbor_mu_contract_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_ss_derivatives =
				grad_neighbor_mu_contract_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_s_derivatives =
				grad_neighbor_mu_contract_coord_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_ss_derivatives =
				grad_neighbor_mu_contract_coord_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			cached_sh_values = cached_sh_values_write;
			cached_radial_values = cached_radial_values_write;
			EvalRealSH(rvec, r, sh_l_max_, cached_sh_values_write, cached_sh_ders);

			for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					cached_rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					cached_rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_s = 0.0;
				double dot_coord_s = 0.0;
				double dot_ss = 0.0;
				double dot_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					dot_val += coeff * cached_rb_vals[radial_base + xi];
					dot_der += coeff * cached_rb_ders[deriv_base + xi];
					dot_s += coeff * cached_rb_ders[deriv_base + xi + R];
					dot_coord_s += coeff * cached_rb_ders[deriv_base + xi + 2 * R];
					dot_ss += coeff * cached_rb_ders[deriv_base + xi + 3 * R];
					dot_coord_ss += coeff * cached_rb_ders[deriv_base + xi + 4 * R];
				}
				cached_radial_values_write[mu] = dot_val;
				cached_radial_derivatives[mu] = dot_der;
				cached_radial_s_derivatives[mu] = dot_s;
				cached_radial_coord_s_derivatives[mu] = dot_coord_s;
				cached_radial_ss_derivatives[mu] = dot_ss;
				cached_radial_coord_ss_derivatives[mu] = dot_coord_ss;
			}
			}

					const double* gate_type_scale_by_mu = nullptr;
						if (use_edge_pair_gate_cache) {
							const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							gate_type_scale_by_mu =
								two_layer_edge_gate_type_scale_cache_.data()
							+ edge * radial_func_count;
						} else {
							int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
						const double* gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, TwoLayerGateNeighborSignals(nbh, j),
								cached_sh_values, gate_neighbor_atom_index_for_tanh);
						const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
						const double* center_gate_signal_by_mu =
							TwoLayerGateAtomSignals(gate_center_atom_index);
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferTypeScale);
						gate_type_scale_by_mu =
							two_layer_gate_type_scale_mu_buffer_.data();
					}
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
				for (int mu = 0; mu < radial_func_count; ++mu)
					gate_scaled_radial_by_mu[mu] =
						gate_type_scale_by_mu[mu] * cached_radial_values[mu];
				for (int i = 0; i < alpha_index_basic_count; ++i) {
					const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					moment_vals[i] += gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
				}
	}

	ContractOuterKRank1Values(moment_vals);
	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}


void MLMTPR::CalcSHMomentValuesWithGradientCacheRadialMixing(const Neighborhood& nbh)
{
	two_layer_gate_radial_mixing_active_eval_ =
		TwoLayerGateRadialMixingEnabled()
		&& (OuterUsesPureRadialMatrix() || HasNonzeroTwoLayerGateRadialMixingCoeffs());
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const int mu_stride = radial_func_count;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, true);
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (!use_edge_cache) {
		grad_neighbor_sh_values_cache_.resize(neighbor_count * sh_count);
		grad_neighbor_sh_ders_cache_.resize(neighbor_count * 3 * sh_count);
		grad_neighbor_radial_vals_cache_.resize(neighbor_count * radial_val_stride);
		grad_neighbor_radial_ders_cache_.resize(neighbor_count * radial_der_stride);
		grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_ss_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_ss_cache_.resize(neighbor_count * mu_stride);
	}

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	const int radial_coeff_base = C + 2 * C * C * K_;
	const bool radial_mixing_active =
		two_layer_gate_radial_mixing_active_eval_;
	const size_t radial_mixing_kernel_stride =
		static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_
		* sh_k_max_;
	std::vector<double>& radial_mixing_outer_delta =
		two_layer_gate_radial_mixing_outer_delta_buffer_;
	if (radial_mixing_active) {
		radial_mixing_outer_delta.assign(
			static_cast<size_t>(outer_k_rank_) * sh_count, 0.0);
		two_layer_gate_radial_mixing_outer_rank_buffer_.resize(outer_k_rank_);
	}

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* cached_sh_values = nullptr;
		const double* cached_radial_values = nullptr;
		const double* cached_rb_vals_read = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			cached_radial_values =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			cached_rb_vals_read =
				two_layer_edge_radial_vals_cache_.data() + edge * radial_val_stride;
		} else {
			double* cached_sh_values_write =
				grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			double* cached_sh_ders =
				grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			double* cached_rb_vals =
				grad_neighbor_radial_vals_cache_.data()
				+ static_cast<size_t>(j) * radial_val_stride;
			double* cached_rb_ders =
				grad_neighbor_radial_ders_cache_.data()
				+ static_cast<size_t>(j) * radial_der_stride;
			double* cached_radial_values_write =
				grad_neighbor_mu_contract_vals_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_derivatives =
				grad_neighbor_mu_contract_ders_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_s_derivatives =
				grad_neighbor_mu_contract_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_ss_derivatives =
				grad_neighbor_mu_contract_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_s_derivatives =
				grad_neighbor_mu_contract_coord_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_ss_derivatives =
				grad_neighbor_mu_contract_coord_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			cached_sh_values = cached_sh_values_write;
			cached_radial_values = cached_radial_values_write;
			cached_rb_vals_read = cached_rb_vals;
			EvalRealSH(rvec, r, sh_l_max_, cached_sh_values_write, cached_sh_ders);

			for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					cached_rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					cached_rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_s = 0.0;
				double dot_coord_s = 0.0;
				double dot_ss = 0.0;
				double dot_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					dot_val += coeff * cached_rb_vals[radial_base + xi];
					dot_der += coeff * cached_rb_ders[deriv_base + xi];
					dot_s += coeff * cached_rb_ders[deriv_base + xi + R];
					dot_coord_s += coeff * cached_rb_ders[deriv_base + xi + 2 * R];
					dot_ss += coeff * cached_rb_ders[deriv_base + xi + 3 * R];
					dot_coord_ss += coeff * cached_rb_ders[deriv_base + xi + 4 * R];
				}
				cached_radial_values_write[mu] = dot_val;
				cached_radial_derivatives[mu] = dot_der;
				cached_radial_s_derivatives[mu] = dot_s;
				cached_radial_coord_s_derivatives[mu] = dot_coord_s;
				cached_radial_ss_derivatives[mu] = dot_ss;
				cached_radial_coord_ss_derivatives[mu] = dot_coord_ss;
			}
			}

					const double* gate_type_scale_by_mu = nullptr;
						if (use_edge_pair_gate_cache) {
							const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							gate_type_scale_by_mu =
								two_layer_edge_gate_type_scale_cache_.data()
							+ edge * radial_func_count;
						} else {
							int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
						const double* gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, TwoLayerGateNeighborSignals(nbh, j),
								cached_sh_values, gate_neighbor_atom_index_for_tanh);
						const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
						const double* center_gate_signal_by_mu =
							TwoLayerGateAtomSignals(gate_center_atom_index);
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferTypeScale);
						gate_type_scale_by_mu =
							two_layer_gate_type_scale_mu_buffer_.data();
					}
				const double* cached_mixing_kernels =
					use_edge_cache && radial_mixing_active
					? two_layer_edge_radial_mixing_kernel_cache_.data()
						+ TwoLayerEdgePrimitiveOffset(-1, j)
							* radial_mixing_kernel_stride
					: nullptr;
				if (radial_mixing_active)
					AccumulateTwoLayerGateRadialMixingForwardEdge(
						gate_type_scale_by_mu,
						cached_rb_vals_read,
						cached_sh_values,
						radial_mixing_outer_delta.data(),
						cached_mixing_kernels);
				if (OuterUsesPureRadialMatrix())
					continue;
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
				for (int mu = 0; mu < radial_func_count; ++mu)
					gate_scaled_radial_by_mu[mu] =
						gate_type_scale_by_mu[mu] * cached_radial_values[mu];
				for (int i = 0; i < alpha_index_basic_count; ++i) {
					const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					moment_vals[i] += gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
				}
	}

	ContractOuterKRank1Values(moment_vals);
	if (radial_mixing_active)
		for (int a = 0; a < outer_k_rank_; ++a)
			for (int l = 0; l <= sh_l_max_; ++l)
				for (int m = -l; m <= l; ++m)
					moment_vals[OuterKRank1OuterBasicIndex(a, l, m)] +=
						radial_mixing_outer_delta[
							static_cast<size_t>(a) * sh_count
							+ SHFlatIndex(l, m)];
	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHBasisFuncsDers(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const bool pure_matrix = OuterUsesPureRadialMatrix();
	if (pure_matrix && use_lmp_table)
		ERROR("Pure distance-matrix linear components require direct radial basis evaluation");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	if (nbh.count != moment_ders.size2 || alpha_moments_count != moment_ders.size1)
		moment_ders.resize(alpha_moments_count, nbh.count, 3);
	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	moment_ders.set(0);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_scaled_mu_ders_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_ders_buffer_.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			radial_derivatives_use =
				two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

				const double* gate_type_scale_by_mu = nullptr;
				if (use_edge_pair_gate_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
					gate_type_scale_by_mu =
						two_layer_edge_gate_type_scale_cache_.data()
						+ edge * radial_func_count;
					} else {
						int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
					const double* gate_signal_by_mu =
						PrepareTwoLayerGateEdgeL1NeighborSignals(
						nbh, j, TwoLayerGateNeighborSignals(nbh, j),
						sh_values_use, gate_neighbor_atom_index_for_tanh);
				const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
				const double* center_gate_signal_by_mu =
					TwoLayerGateAtomSignals(gate_center_atom_index);
					PrepareTwoLayerGatePairMuBuffers(
						type_central, type_outer,
						center_gate_signal_by_mu, gate_center_atom_index,
						gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
						kGateMuBufferTypeScale);
					gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
				}
				if (pure_matrix) {
					// Partial coordinate derivative at fixed gate scalars x.
					// Their environment chain is added by AccumulateSHBasisGateDers.
					const size_t kernel_stride =
						static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_ * sh_k_max_;
					const size_t edge = use_edge_cache ? TwoLayerEdgePrimitiveOffset(-1, j) : 0;
					const double* kernels = use_edge_cache
						? two_layer_edge_radial_mixing_kernel_cache_.data() + edge * kernel_stride : nullptr;
					const double* kernel_ders = use_edge_cache
						? two_layer_edge_radial_mixing_kernel_der_cache_.data() + edge * kernel_stride : nullptr;
					for (int l = 0; l <= sh_l_max_; ++l) {
						for (int o = 0; o < outer_k_rank_; ++o) {
							double value = 0.0;
							double derivative = 0.0;
							for (int k = 0; k < sh_k_max_; ++k) {
								const int mu = k * (sh_l_max_ + 1) + l;
								const size_t lok = (static_cast<size_t>(l) * outer_k_rank_ + o) * sh_k_max_ + k;
								double kernel = kernels == nullptr ? 0.0 : kernels[lok];
								double kernel_der = kernel_ders == nullptr ? 0.0 : kernel_ders[lok];
								if (kernels == nullptr) {
									const int basis_begin = mu_to_radial_eval_block_[mu] * R;
									const int coeff_begin = TwoLayerGateRadialMixingCoeffIndex(l, o, k, 0);
									for (int b = 0; b < R; ++b) {
										kernel += regression_coeffs[coeff_begin + b] * rb_vals[basis_begin + b];
										kernel_der += regression_coeffs[coeff_begin + b] * rb_ders[basis_begin + b];
									}
								}
								value += gate_type_scale_by_mu[mu] * kernel;
								derivative += gate_type_scale_by_mu[mu] * kernel_der;
							}
							for (int m = -l; m <= l; ++m) {
								const int node = OuterKRank1OuterBasicIndex(o, l, m);
								const int sh_index = SHFlatIndex(l, m);
								const double y = sh_values_use[sh_index];
								moment_vals[node] += value * y;
								for (int axis = 0; axis < 3; ++axis)
									moment_ders(node, j, axis) +=
										derivative * rvec[axis] * inv_r * y
										+ value * sh_ders_use[3 * sh_index + axis];
							}
						}
					}
					continue;
				}
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
			double* gate_scaled_der_by_mu =
				two_layer_gate_scaled_mu_ders_buffer_.data();
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double type_scale = gate_type_scale_by_mu[mu];
				gate_scaled_radial_by_mu[mu] = type_scale * radial_values_use[mu];
				gate_scaled_der_by_mu[mu] = type_scale * radial_derivatives_use[mu];
			}

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					const int sh_der_index = basic_sh_der_index_cache_[i];
					const double radial_value = gate_scaled_radial_by_mu[mu];
					const double radial_der = gate_scaled_der_by_mu[mu];

					const double y = sh_values_use[sh_index];
					moment_vals[i] += radial_value * y;
					for (int a = 0; a < 3; ++a) {
						const double dy = sh_ders_use[sh_der_index + a];
						moment_ders(i, j, a) += radial_der * rvec[a] * inv_r * y + radial_value * dy;
			}
		}
	}

	basis_vals[0] = 1.0;
	if (basis_ders.size1 != alpha_count || basis_ders.size2 != nbh.count)
		basis_ders.resize(alpha_count, nbh.count, 3);
	if (active_two_layer_gate_values_ == nullptr)
		basis_ders.set(0);

	if (!pure_matrix)
		ContractOuterKRank1MomentDers(nbh);
	const bool use_product_rows = UseSHProductRows();
	if (use_product_rows) {
		ApplySHProductRowsDers(nbh);
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			const double left_value = moment_vals[product.left];
			const double right_value = moment_vals[product.right];
			moment_vals[product.target] += product.coeff * left_value * right_value;
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					moment_ders(product.target, j, a) += product.coeff * (
						moment_ders(product.left, j, a) * right_value + left_value * moment_ders(product.right, j, a));
		}
	}

	for (int i = 0; i < alpha_scalar_moments; ++i) {
		if (use_product_rows && sh_scalar_terminal_product_[i])
			continue;
		const int node = alpha_moment_mapping[i];
		basis_vals[1 + i] = moment_vals[node];
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				basis_ders(1 + i, j, a) = moment_ders(node, j, a);
	}
}

void MLMTPR::CalcTwoLayerGateScalarValuesOnly(
	const Neighborhood& nbh,
	std::vector<double>& gate_scalar_values,
	int cache_atom_index)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	for (int moment_index : two_layer_gate_required_moment_indices_)
		moment_vals[moment_index] = 0.0;
	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
				                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
					for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
						const int scaling_block = radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_CalcValsOnly(
							r,
							TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
							TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
							basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu : two_layer_gate_required_mu_indices_) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

			const double* second_gate_neighbor_signals =
				multi_layer_gate_enabled_
					? SecondLayerGateNeighborSignals(nbh, j)
					: nullptr;
			for (int bi = 0;
			     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
			     ++bi) {
				const int basic_index = two_layer_gate_required_basic_indices_[bi];
				const int mu = basic_mu_cache_[basic_index];
				const double gate_center_type_coeff =
					TwoLayerGateTypeCoeff(type_central, mu);
				const double gate_outer_type_coeff =
					TwoLayerGateTypeCoeff(type_outer, mu);
				const double second_gate_signal =
					second_gate_neighbor_signals != nullptr
						? second_gate_neighbor_signals[mu]
						: 0.0;
				const double second_gate_additive =
					multi_layer_gate_enabled_
						? SecondLayerGateAdditiveCoeff(type_outer, mu)
						: 0.0;
				const double gate_outer_eff =
					gate_outer_type_coeff + second_gate_additive * second_gate_signal;
				const double type_scale =
					gate_center_type_coeff * gate_outer_eff;
				const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			moment_vals[basic_index] +=
				type_scale * radial_values_use[mu_index] * sh_values_use[sh_index];
		}
	}

	ContractTwoLayerGateKRankValues(moment_vals);
	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		moment_vals[product.target] +=
			product.coeff * moment_vals[product.left] * moment_vals[product.right];
	}

	gate_scalar_values.assign(TwoLayerGateScalarCount(), 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q)
		gate_scalar_values[q] =
			moment_vals[TwoLayerGateScalarMomentIndexUnchecked(q)];
}

void MLMTPR::CalcSecondLayerGateScalarValuesOnly(
	const Neighborhood& nbh,
	std::vector<double>& gate_scalar_values,
	int cache_atom_index)
{
	(void)cache_atom_index;
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (second_layer_gate_required_moments_.empty()
	    || second_layer_gate_required_moment_indices_.empty())
		BuildSecondLayerGateProductProgram();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	for (int moment_index : second_layer_gate_required_moment_indices_)
		moment_vals[moment_index] = 0.0;
	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_gate_radial = SecondLayerGateUsesSharedRadial();
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	if (static_cast<int>(rb_vals.size()) < eval_block_count * R)
		rb_vals.resize(eval_block_count * R);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

		if (use_lmp_table) {
			InterpolateSHLmpMuTable(use_gate_radial ? second_layer_gate_radial_list : radial_list,
			                         use_gate_radial ? second_layer_gate_radial_der_list : radial_der_list,
			                         inv_dr,
			                         C,
			                         type_central,
			                         type_outer,
			                         r,
			                         radial_func_count,
			                         radial_values.data(),
			                         nullptr);
		} else {
			for (int eval_block : second_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_CalcValsOnly(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] =
						p_RadialBasis->rb_vals[xi] * scaling;
			}
			for (int mu : second_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				for (int xi = 0; xi < R; ++xi)
					radial_value += regression_coeffs[radial_offset + xi]
						* rb_vals[radial_base + xi];
				radial_values[mu] = radial_value;
			}
		}

		for (int basic_index : second_layer_gate_required_basic_indices_) {
			const int mu = basic_mu_cache_[basic_index];
			const double type_scale =
				SecondLayerGateTypeCoeff(type_central, mu)
				* SecondLayerGateTypeCoeff(type_outer, mu);
			const int sh_index = basic_sh_index_cache_[basic_index];
			moment_vals[basic_index] +=
				type_scale * radial_values[mu] * sh_values[sh_index];
		}
	}

	for (int product_index : second_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		moment_vals[product.target] +=
			product.coeff * moment_vals[product.left] * moment_vals[product.right];
	}

	gate_scalar_values.assign(SecondLayerGateScalarCount(), 0.0);
	for (int q = 0; q < SecondLayerGateScalarCount(); ++q) {
		const int scalar_index = second_layer_gate_scalar_indices_[q];
		gate_scalar_values[q] = moment_vals[alpha_moment_mapping[scalar_index]];
	}
}

void MLMTPR::FillTwoLayerGateEdgeL1ValuesFromMoments(double* edge_l1_values) const
{
	if (edge_l1_values == nullptr)
		ERROR("SUS2-SH edge L1 gate value buffer is missing");
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	for (int source = 0; source < source_count; ++source) {
		const int base = TwoLayerGateEdgeL1SourceMomentIndex(source);
		double* dst = edge_l1_values + source * 3;
		for (int m = 0; m < 3; ++m)
			dst[m] = moment_vals[base + m];
	}
}

void MLMTPR::CalcTwoLayerGateEdgeL1ValuesOnly(
	const Neighborhood& nbh,
	std::vector<double>& edge_l1_values,
	int cache_atom_index)
{
	if (!TwoLayerGateEdgeL1Enabled()) {
		edge_l1_values.clear();
		return;
	}
	if (sh_l_max_ < 1 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH edge L1 gate requires sh_l_max in [1,6].");
	if (TwoLayerGateEdgeL1SourceCount() == 0)
		BuildTwoLayerGateEdgeL1Sources();

	const int source_count = TwoLayerGateEdgeL1SourceCount();
	edge_l1_values.assign(static_cast<size_t>(source_count) * 3, 0.0);
	if (source_count == 0)
		return;

	CalcTwoLayerGateScalarValuesOnly(nbh, sh_gate_scalar_values_, cache_atom_index);
	FillTwoLayerGateEdgeL1ValuesFromMoments(edge_l1_values.data());
}

void MLMTPR::ComputeTwoLayerGateEdgeL1CoordLoss(
	const Neighborhoods& neighborhoods,
	const std::vector<Vector3>& frc_weights,
	const Matrix3& str_weights,
	std::vector<double>& coord_loss)
{
	if (!TwoLayerGateEdgeL1Enabled()) {
		coord_loss.clear();
		return;
	}
	if (sh_l_max_ < 1 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH edge L1 gate requires sh_l_max in [1,6].");
	if (TwoLayerGateEdgeL1SourceCount() == 0)
		BuildTwoLayerGateEdgeL1Sources();

	const int source_count = TwoLayerGateEdgeL1SourceCount();
	const int atom_count = neighborhoods.size();
	coord_loss.assign(
		static_cast<size_t>(atom_count) * source_count * 3, 0.0);
	if (source_count == 0 || atom_count == 0)
		return;

	std::vector<Vector3> direction_weights;
	std::vector<double> gate_scalar_tangents;
	std::vector<double> gate_moment_tangents;
	for (int ind = 0; ind < atom_count; ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		double* atom_coord_loss =
			coord_loss.data() + static_cast<size_t>(ind) * source_count * 3;
		direction_weights.assign(static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));

		for (int j = 0; j < nbh.count; ++j) {
			const Vector3& rvec = nbh.vecs[j];
			Vector3 edge_weight = frc_weights[ind];
			edge_weight -= frc_weights[nbh.inds[j]];
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b)
					edge_weight[a] += str_weights[a][b] * rvec[b];
			direction_weights[j] = edge_weight;
		}

		bool has_direction = false;
		for (const Vector3& direction : direction_weights)
			if (direction.NormSq() != 0.0) {
				has_direction = true;
				break;
			}
		if (!has_direction)
			continue;

		CalcTwoLayerGateScalarDirectionalDerivatives(
			nbh,
			direction_weights,
			gate_scalar_tangents,
			&gate_moment_tangents,
			ind);
		if (static_cast<int>(gate_moment_tangents.size()) < alpha_moments_count)
			ERROR("SUS2-SH edge L1 gate moment tangent cache is inconsistent");
		for (int source = 0; source < source_count; ++source) {
			const int base = TwoLayerGateEdgeL1SourceMomentIndex(source);
			double* dst = atom_coord_loss + source * 3;
			for (int m = 0; m < 3; ++m)
				dst[m] = gate_moment_tangents[base + m];
		}
	}
}

void MLMTPR::ComputeTwoLayerGateEdgeL1PairCoordTangents(
	const Neighborhood& nbh,
	int neighbor_index,
	const double* edge_sh_values,
	const double* edge_sh_ders,
	const Vector3* edge_der_weight,
	std::vector<double>& source_tangents) const
{
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	source_tangents.assign(source_count, 0.0);
	if (!TwoLayerGateEdgeL1Enabled() || source_count <= 0)
		return;
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return;
	const int atom_index = nbh.inds[neighbor_index];
	if (atom_index < 0)
		return;
	double local_sh_values[kMaxSHComponents];
	double local_sh_ders[3 * kMaxSHComponents];
	const double* sh_values_use = edge_sh_values;
	const double* sh_ders_use = edge_sh_ders;
	if (sh_values_use == nullptr
	    || (edge_der_weight != nullptr && sh_ders_use == nullptr)) {
		EvalRealSH(nbh.vecs[neighbor_index],
		           nbh.dists[neighbor_index],
		           sh_l_max_,
		           local_sh_values,
		           local_sh_ders);
		sh_values_use = local_sh_values;
		sh_ders_use = local_sh_ders;
	}
	const size_t value_stride = static_cast<size_t>(source_count) * 3;
	if (two_layer_gate_edge_l1_values_.size() % value_stride != 0)
		ERROR("SUS2-SH edge L1 gate value cache has inconsistent size");
	const int atom_count =
		static_cast<int>(two_layer_gate_edge_l1_values_.size() / value_stride);
	if (atom_index >= atom_count)
		ERROR("SUS2-SH edge L1 gate neighbor index is out of range");

	const double* edge_l1_values =
		two_layer_gate_edge_l1_values_.data()
		+ static_cast<size_t>(atom_index) * value_stride;
	const double* coord_loss = nullptr;
	if (active_two_layer_gate_edge_l1_coord_loss_ != nullptr) {
		if (active_two_layer_gate_edge_l1_coord_loss_->size()
		    != static_cast<size_t>(atom_count) * value_stride)
			ERROR("SUS2-SH edge L1 coordinate loss cache has inconsistent size");
		coord_loss =
			active_two_layer_gate_edge_l1_coord_loss_->data()
			+ static_cast<size_t>(atom_index) * value_stride;
	}

	const int y1_indices[3] = {
		SHFlatIndex(1, -1),
		SHFlatIndex(1, 0),
		SHFlatIndex(1, 1)
	};
	for (int source = 0; source < source_count; ++source) {
		const double* h = edge_l1_values + source * 3;
		for (int m = 0; m < 3; ++m) {
			const int y_index = y1_indices[m];
			if (coord_loss != nullptr)
				source_tangents[source] +=
					coord_loss[source * 3 + m] * sh_values_use[y_index];
			if (edge_der_weight != nullptr && sh_ders_use != nullptr) {
				const int dy_index = 3 * y_index;
				source_tangents[source] +=
					h[m] * (
						(*edge_der_weight)[0] * sh_ders_use[dy_index]
						+ (*edge_der_weight)[1] * sh_ders_use[dy_index + 1]
						+ (*edge_der_weight)[2] * sh_ders_use[dy_index + 2]);
			}
		}
	}
}

void MLMTPR::AccumulateTwoLayerGateEdgeL1ForceChain(
	Configuration& cfg,
	const Neighborhoods& neighborhoods)
{
	if (!HasNonzeroTwoLayerGateEdgeL1Weights())
		return;
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	if (source_count <= 0)
		return;
	if (two_layer_gate_edge_l1_adjoints_.size()
	    != static_cast<size_t>(cfg.size()) * source_count * 3)
		ERROR("SUS2-SH edge L1 gate adjoint cache has inconsistent size");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const bool use_edge_der_cache = HasTwoLayerEdgePrimitiveCache(0, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	std::vector<double> active_basic_adjoints;
	std::vector<int> active_basic_mu_indices;
	std::vector<int> active_basic_dense_mu_indices;
	std::vector<int> active_basic_sh_indices;
	std::vector<int> active_basic_sh_der_indices;
	active_basic_adjoints.reserve(two_layer_gate_required_basic_indices_.size());
	active_basic_mu_indices.reserve(two_layer_gate_required_basic_indices_.size());
	active_basic_dense_mu_indices.reserve(two_layer_gate_required_basic_indices_.size());
	active_basic_sh_indices.reserve(two_layer_gate_required_basic_indices_.size());
	active_basic_sh_der_indices.reserve(two_layer_gate_required_basic_indices_.size());

	for (int ind = 0; ind < cfg.size(); ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		const int type_central = nbh.my_type;
		if (type_central >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
			const double* atom_adjoints =
			two_layer_gate_edge_l1_adjoints_.data()
			+ static_cast<size_t>(ind) * source_count * 3;

		bool has_adjoint = false;
		for (int source = 0; source < source_count * 3; ++source)
			if (atom_adjoints[source] != 0.0) {
				has_adjoint = true;
				break;
			}
		if (!has_adjoint)
			continue;

		bool loaded_moment_cache = false;
		if (alpha_moments_count > 0) {
			const int cached_moment_count =
				static_cast<int>(two_layer_gate_required_moment_indices_.size());
			const size_t offset =
				static_cast<size_t>(ind) * cached_moment_count;
			if (cached_moment_count > 0
			    && offset + cached_moment_count <=
				two_layer_gate_moment_values_cache_.size()) {
				const double* cached_moments =
					two_layer_gate_moment_values_cache_.data() + offset;
				for (int i = 0; i < cached_moment_count; ++i)
					moment_vals[two_layer_gate_required_moment_indices_[i]] =
						cached_moments[i];
				loaded_moment_cache = true;
			}
		}
		if (!loaded_moment_cache)
			CalcTwoLayerGateScalarValuesOnly(nbh, sh_gate_scalar_values_, ind);
		site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
		for (int source = 0; source < source_count; ++source) {
			const int base = TwoLayerGateEdgeL1SourceMomentIndex(source);
			const double* h_adj = atom_adjoints + source * 3;
			for (int m = 0; m < 3; ++m)
				site_energy_ders_wrt_moments_[base + m] += h_adj[m];
		}
		for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[two_layer_gate_required_product_indices_[p]];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
		// Edge-L1 sources live in the contracted gate graph when gate K-rank is
		// enabled.  Product reverse reaches its contracted primitives; apply the
		// transpose contraction before consuming raw radial/SH basic adjoints.
		BackpropTwoLayerGateKRankAdjoints(site_energy_ders_wrt_moments_);

		active_basic_adjoints.clear();
		active_basic_mu_indices.clear();
		active_basic_dense_mu_indices.clear();
		active_basic_sh_indices.clear();
		active_basic_sh_der_indices.clear();
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double adjoint = site_energy_ders_wrt_moments_[basic_index];
			if (adjoint != 0.0) {
				active_basic_adjoints.push_back(adjoint);
				active_basic_mu_indices.push_back(basic_mu_cache_[basic_index]);
				active_basic_dense_mu_indices.push_back(
					bi < static_cast<int>(
						two_layer_gate_required_basic_dense_mu_indices_.size())
						? two_layer_gate_required_basic_dense_mu_indices_[bi]
						: -1);
				active_basic_sh_indices.push_back(
					basic_sh_index_cache_[basic_index]);
				active_basic_sh_der_indices.push_back(
					basic_sh_der_index_cache_[basic_index]);
			}
		}
		if (active_basic_adjoints.empty())
			continue;

		for (int j = 0; j < nbh.count; ++j) {
			const Vector3& rvec = nbh.vecs[j];
			const double r = nbh.dists[j];
			const double inv_r = 1.0 / r;
			const int type_outer = nbh.types[j];
			const double* sh_values_use = sh_values;
			const double* sh_ders_use = sh_ders;
			const double* radial_values_use = radial_values.data();
			const double* radial_derivatives_use = radial_derivatives.data();

			if (use_edge_der_cache) {
				const size_t edge = TwoLayerEdgePrimitiveOffset(ind, j);
				sh_values_use =
					two_layer_edge_sh_values_cache_.data() + edge * sh_count;
				sh_ders_use =
					two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
				radial_values_use =
					(use_gate_radial
						? two_layer_edge_gate_mu_vals_cache_.data()
						: two_layer_edge_mu_vals_cache_.data())
					+ edge * static_cast<size_t>(
						use_gate_radial ? gate_mu_stride : mu_stride);
				radial_derivatives_use =
					(use_gate_radial
						? two_layer_edge_gate_mu_ders_cache_.data()
						: two_layer_edge_mu_ders_cache_.data())
					+ edge * static_cast<size_t>(
						use_gate_radial ? gate_mu_stride : mu_stride);
			} else {
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
				if (use_lmp_table) {
					InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
					                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
					                         inv_dr,
					                         C,
					                         type_central,
					                         type_outer,
					                         r,
					                         radial_func_count,
					                         radial_values.data(),
					                         radial_derivatives.data());
					} else {
						for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
							const int scaling_block = radial_eval_to_scaling_block_[eval_block];
							const int basis_k = radial_eval_to_basis_k_[eval_block];
							p_RadialBasis->RB_Calc(
								r,
								TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
								TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
								basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
							rb_ders[eval_block * R + xi] =
								p_RadialBasis->rb_ders[xi] * scaling;
						}
					}
					for (int mu : two_layer_gate_required_mu_indices_) {
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double radial_value = 0.0;
						double radial_der = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[radial_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
				}
			}

				Vector3 edge_der(0.0, 0.0, 0.0);
				for (int active_pos = 0;
				     active_pos < static_cast<int>(active_basic_adjoints.size());
				     ++active_pos) {
					const double adjoint = active_basic_adjoints[active_pos];
					const int mu = active_basic_mu_indices[active_pos];
					const double gate_center_type_coeff =
						TwoLayerGateTypeCoeff(type_central, mu);
					const double gate_outer_type_coeff =
						TwoLayerGateTypeCoeff(type_outer, mu);
					const double type_scale =
						gate_center_type_coeff * gate_outer_type_coeff;
					const int mu_index =
						(use_edge_der_cache && use_gate_radial)
							? active_basic_dense_mu_indices[active_pos]
							: mu;
				const int sh_index = active_basic_sh_indices[active_pos];
				const int sh_der_index = active_basic_sh_der_indices[active_pos];
				const double radial_value = radial_values_use[mu_index];
				const double radial_der = radial_derivatives_use[mu_index];
				const double y = sh_values_use[sh_index];
				for (int a = 0; a < 3; ++a) {
					const double dy = sh_ders_use[sh_der_index + a];
					edge_der[a] += adjoint * type_scale
						* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
				}
			}
			cfg.force(ind) += edge_der;
			cfg.force(nbh.inds[j]) -= edge_der;
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b)
					cfg.stresses[a][b] -= edge_der[a] * nbh.vecs[j][b];
		}
	}
	}

void MLMTPR::AddTwoLayerGateX2Penalty(
	Configuration& cfg,
	const Neighborhoods& neighborhoods,
	double coeff,
	double& out_penalty_accumulator,
	Array1D* out_penalty_grad_accumulator)
{
	if (!two_layer_gate_enabled_ || coeff == 0.0)
		return;
	if (!std::isfinite(coeff) || coeff < 0.0)
		ERROR("--gate-x-l2 should be a finite non-negative value");
	if (OuterUsesAdaptiveRadialCoefficients())
		ERROR("--gate-x-l2 is not supported for radial-coefficient-gate models; use the adaptive radial regularization path instead");
	if (radial_func_count <= 0)
		return;

	std::vector<size_t> edge_count_by_type(species_count, 0);
	for (int ind = 0; ind < neighborhoods.size(); ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		for (int j = 0; j < nbh.count; ++j) {
			const int type_outer = nbh.types[j];
			if (type_outer >= species_count)
				throw MlipException("Too few species count in the MTP potential!");
			if (type_outer >= 0)
				++edge_count_by_type[type_outer];
		}
	}
	int active_type_count = 0;
	for (size_t count : edge_count_by_type)
		if (count > 0)
			++active_type_count;
	if (active_type_count == 0)
		return;

	PrepareEvalCaches();
	const bool edge_l1_enabled = TwoLayerGateEdgeL1Enabled();
	if (edge_l1_enabled)
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, false);
	PrepareTwoLayerGateValues(cfg, neighborhoods);

	const double channel_norm =
		coeff
		/ (static_cast<double>(active_type_count)
		   * static_cast<double>(radial_func_count));
	const bool need_grad = out_penalty_grad_accumulator != nullptr;
	if (need_grad && static_cast<int>(out_penalty_grad_accumulator->size()) < CoeffCount())
		out_penalty_grad_accumulator->resize(CoeffCount());

	two_layer_gate_adjoints_.assign(
		static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
	const int edge_l1_source_count =
		edge_l1_enabled ? TwoLayerGateEdgeL1SourceCount() : 0;
	if (need_grad && edge_l1_enabled && edge_l1_source_count > 0)
		two_layer_gate_edge_l1_adjoints_.assign(
			static_cast<size_t>(cfg.size()) * edge_l1_source_count * 3,
			0.0);
	else
		two_layer_gate_edge_l1_adjoints_.clear();

	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	active_two_layer_gate_values_ = &two_layer_gate_values_;
	active_two_layer_gate_adjoints_ = need_grad ? &two_layer_gate_adjoints_ : nullptr;

	const bool have_edge_cache =
		edge_l1_enabled && HasTwoLayerEdgePrimitiveCache(0, false, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const double* edge_l1_weights =
		(edge_l1_enabled && edge_l1_source_count > 0)
			? TwoLayerGateEdgeL1WeightData()
			: nullptr;
	const int additive_coeff_offset = TwoLayerGateAdditiveCoeffOffset();
	double sh_values[kMaxSHComponents];
	std::vector<double> gate_adjoint_by_mu(radial_func_count, 0.0);

	for (int ind = 0; ind < cfg.size(); ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		for (int j = 0; j < nbh.count; ++j) {
			const int type_outer = nbh.types[j];
			if (type_outer >= species_count)
				throw MlipException("Too few species count in the MTP potential!");
			const size_t type_edge_count =
				type_outer >= 0 ? edge_count_by_type[type_outer] : 0;
			if (type_edge_count == 0)
				continue;
			const double norm =
				channel_norm / static_cast<double>(type_edge_count);
			const double* edge_sh_values = nullptr;
			if (edge_l1_enabled) {
				if (have_edge_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(ind, j);
					edge_sh_values =
						two_layer_edge_sh_values_cache_.data() + edge * sh_count;
				} else {
					EvalRealSHValuesOnly(
						nbh.vecs[j], nbh.dists[j], sh_l_max_, sh_values);
					edge_sh_values = sh_values;
				}
			}

			int neighbor_atom_index_for_tanh = nbh.inds[j];
			const double* gate_signal_by_mu =
				edge_l1_enabled
					? PrepareTwoLayerGateEdgeL1NeighborSignals(
						nbh, j, TwoLayerGateNeighborSignals(nbh, j),
						edge_sh_values, neighbor_atom_index_for_tanh)
					: TwoLayerGateNeighborSignals(nbh, j);

			const double* additive_coeff_row =
				TwoLayerGateAdditiveCoeffRowData(type_outer);
			const int additive_coeff_index_base =
				need_grad
					? GateAdditiveCoeffIndexFast(additive_coeff_offset,
					                             radial_func_count,
					                             type_outer,
					                             0)
					: -1;
			std::fill(gate_adjoint_by_mu.begin(),
			          gate_adjoint_by_mu.end(),
			          0.0);
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double gate_signal =
					gate_signal_by_mu == nullptr ? 0.0 : gate_signal_by_mu[mu];
				const double additive_coeff =
					additive_coeff_row != nullptr
						? additive_coeff_row[mu]
						: TwoLayerGateAdditiveCoeff(type_outer, mu);
				const double x = additive_coeff * gate_signal;
				out_penalty_accumulator += norm * x * x;
				if (!need_grad)
					continue;
				const double x_adjoint = 2.0 * norm * x;
				(*out_penalty_grad_accumulator)[
					additive_coeff_index_base + mu] +=
					x_adjoint * gate_signal;
				const double gate_adjoint = x_adjoint * additive_coeff;
				gate_adjoint_by_mu[mu] = gate_adjoint;
				AddTwoLayerGateMuAdjoint(nbh, j, mu, gate_adjoint);
			}

			if (!need_grad || !edge_l1_enabled || edge_l1_source_count <= 0
			    || edge_sh_values == nullptr)
				continue;
			AccumulateTwoLayerGateEdgeL1WeightGradForPair(
				nbh, j, edge_sh_values, gate_adjoint_by_mu.data(),
				*out_penalty_grad_accumulator);

			const int atom_index = nbh.inds[j];
			if (atom_index < 0)
				continue;
			const size_t atom_count = static_cast<size_t>(cfg.size());
			if (static_cast<size_t>(atom_index) >= atom_count)
				ERROR("SUS2-SH edge L1 gate neighbor index is out of range");
			if (edge_l1_weights == nullptr)
				continue;
			double* edge_l1_adjoints =
				two_layer_gate_edge_l1_adjoints_.data()
				+ static_cast<size_t>(atom_index) * edge_l1_source_count * 3;
			const double y1[3] = {
				edge_sh_values[SHFlatIndex(1, -1)],
				edge_sh_values[SHFlatIndex(1, 0)],
				edge_sh_values[SHFlatIndex(1, 1)]
			};
			for (int source = 0; source < edge_l1_source_count; ++source) {
				double chi_adjoint = 0.0;
				for (int mu = 0; mu < radial_func_count; ++mu)
					chi_adjoint +=
						gate_adjoint_by_mu[mu]
						* edge_l1_weights[
							static_cast<size_t>(mu) * edge_l1_source_count
							+ source];
				if (chi_adjoint == 0.0)
					continue;
				double* h_adj = edge_l1_adjoints + source * 3;
				for (int m = 0; m < 3; ++m)
					h_adj[m] += chi_adjoint * y1[m];
			}
		}
	}

	if (need_grad) {
		AccumulateTwoLayerGateValueParamGradFromAdjoints(
			cfg, neighborhoods, *out_penalty_grad_accumulator);
		AccumulateTwoLayerGateEdgeL1SourceParamGradFromAdjoints(
			cfg, neighborhoods, *out_penalty_grad_accumulator);
	}

	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
}

void MLMTPR::AccumulateTwoLayerGateValueParamGradFromAdjoints(
	Configuration& cfg,
	const Neighborhoods& neighborhoods,
	Array1D& out_grad_accumulator)
{
	if (!two_layer_gate_enabled_)
		return;
	if (two_layer_gate_adjoints_.size()
	    != static_cast<size_t>(cfg.size()) * radial_func_count)
		ERROR("SUS2-SH two-layer gate adjoint cache has inconsistent size");
	if (static_cast<int>(out_grad_accumulator.size()) < CoeffCount())
		out_grad_accumulator.resize(CoeffCount());
	const int gate_count = TwoLayerGateScalarCount();
	const std::vector<double>* saved_second_gate_values =
		active_second_layer_gate_values_;
	std::vector<double>* saved_second_gate_adjoints =
		active_second_layer_gate_adjoints_;
	if (multi_layer_gate_enabled_) {
		if (second_layer_gate_values_.size()
		    != static_cast<size_t>(cfg.size()) * radial_func_count)
			PrepareSecondLayerGateValues(cfg, neighborhoods);
		second_layer_gate_adjoints_.assign(
			static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
		active_second_layer_gate_values_ = &second_layer_gate_values_;
		active_second_layer_gate_adjoints_ = &second_layer_gate_adjoints_;
	} else {
		active_second_layer_gate_values_ = nullptr;
		active_second_layer_gate_adjoints_ = nullptr;
	}
	if (gate_count <= 0) {
		active_second_layer_gate_values_ = saved_second_gate_values;
		active_second_layer_gate_adjoints_ = saved_second_gate_adjoints;
		return;
	}

	std::vector<double> gate_body_values_scratch(TwoLayerGateBodyOrderCount(), 0.0);
	std::vector<double> gate_body_adjoint_scratch(TwoLayerGateBodyOrderCount(), 0.0);
	const int body_count = TwoLayerGateBodyOrderCount();
	const bool full_gate_weight_blas =
		TwoLayerGateUsesFullScalarWeights()
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * gate_count;
	const bool body_combo_gate_weight_blas =
		TwoLayerGateUsesBodyLinearCombo()
		&& body_count > 0
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * gate_count
		&& two_layer_gate_body_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * body_count;

	if (full_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            gate_count,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_scalar_values_cache_.data(),
		            gate_count,
		            1.0,
		            out_grad_accumulator.data() + TwoLayerGateWeightOffset(),
		            gate_count);
	} else if (body_combo_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            body_count,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_body_values_cache_.data(),
		            body_count,
		            1.0,
		            out_grad_accumulator.data()
		                + TwoLayerGateBodyMixWeightOffset(),
		            body_count);
		std::vector<double> body_adjoint_cache(
			static_cast<size_t>(cfg.size()) * body_count, 0.0);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            body_count,
		            radial_func_count,
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            TwoLayerGateBodyMixWeightMatrixData(),
		            body_count,
		            0.0,
		            body_adjoint_cache.data(),
		            body_count);
		double* scalar_weight_grads =
			out_grad_accumulator.data() + TwoLayerGateWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* scalar_values =
				two_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(ind) * gate_count;
			const double* body_adjoint =
				body_adjoint_cache.data()
				+ static_cast<size_t>(ind) * body_count;
			for (int q = 0; q < gate_count; ++q) {
				const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
				scalar_weight_grads[q] +=
					body_adjoint[body_index] * scalar_values[q];
			}
		}
	} else {
		const double* body_mix_weights =
			TwoLayerGateUsesBodyLinearCombo()
				? TwoLayerGateBodyMixWeightMatrixData()
				: nullptr;
		const int gate_weight_offset = TwoLayerGateWeightOffset();
		const int body_mix_weight_offset = TwoLayerGateBodyMixWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* gate_adjoints_by_mu =
				two_layer_gate_adjoints_.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			bool has_gate_adjoint = false;
			for (int mu = 0; mu < radial_func_count; ++mu)
				if (gate_adjoints_by_mu[mu] != 0.0) {
					has_gate_adjoint = true;
					break;
				}
			if (!has_gate_adjoint)
				continue;
			const double* cached_scalars = nullptr;
			if (two_layer_gate_scalar_values_cache_.size()
			    == static_cast<size_t>(cfg.size()) * gate_count)
				cached_scalars =
					two_layer_gate_scalar_values_cache_.data()
					+ static_cast<size_t>(ind) * gate_count;
			if (cached_scalars == nullptr)
				CalcTwoLayerGateScalarValuesOnly(
					neighborhoods[ind], sh_gate_scalar_values_, ind);
			const double* scalar_values =
				cached_scalars == nullptr
					? sh_gate_scalar_values_.data()
					: cached_scalars;
			if (TwoLayerGateUsesBodyLinearCombo()) {
				const double* cached_body_values = nullptr;
				if (two_layer_gate_body_values_cache_.size()
				    == static_cast<size_t>(cfg.size()) * body_count)
					cached_body_values =
						two_layer_gate_body_values_cache_.data()
						+ static_cast<size_t>(ind) * body_count;
				const double* body_values = cached_body_values;
				if (body_values == nullptr) {
					ComputeTwoLayerGateBodySignals(
						scalar_values, gate_body_values_scratch.data());
					body_values = gate_body_values_scratch.data();
				}
				std::fill(gate_body_adjoint_scratch.begin(),
				          gate_body_adjoint_scratch.end(),
				          0.0);
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					const double* body_mix =
						body_mix_weights + static_cast<size_t>(mu) * body_count;
					for (int b = 0; b < body_count; ++b) {
						out_grad_accumulator[
							body_mix_weight_offset + mu * body_count + b] +=
							gate_adjoint * body_values[b];
						gate_body_adjoint_scratch[b] +=
							gate_adjoint * body_mix[b];
					}
				}
				for (int q = 0; q < gate_count; ++q) {
					const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
					const double body_adjoint =
						gate_body_adjoint_scratch[body_index];
					if (body_adjoint != 0.0)
						out_grad_accumulator[gate_weight_offset + q] +=
							body_adjoint * scalar_values[q];
				}
			} else if (TwoLayerGateUsesFullScalarWeights()) {
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					for (int q = 0; q < gate_count; ++q)
						out_grad_accumulator[TwoLayerGateWeightIndex(mu, q)] +=
							gate_adjoint * scalar_values[q];
				}
			} else {
				ERROR("SUS2-SH two-layer gate model has an unknown mode: "
				      + two_layer_gate_mode_);
			}
		}
	}

	if (!HasNonzeroTwoLayerGateWeights()) {
		active_second_layer_gate_values_ = saved_second_gate_values;
		active_second_layer_gate_adjoints_ = saved_second_gate_adjoints;
		return;
	}

	std::vector<double> scalar_gate_seeds(gate_count, 0.0);
	std::vector<double> scalar_seed_body_scratch;
	std::vector<double>& full_scalar_gate_seed_cache =
		two_layer_full_scalar_seed_scratch_;
	std::vector<double>& body_combo_gate_seed_cache =
		two_layer_body_combo_seed_scratch_;
	const bool full_seed_blas =
		TwoLayerGateUsesFullScalarWeights()
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const bool body_combo_seed_blas =
		TwoLayerGateUsesBodyLinearCombo()
		&& body_count > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const double* body_combo_scalar_weights =
		body_combo_seed_blas ? TwoLayerGateScalarWeightData() : nullptr;
	if (full_seed_blas) {
		full_scalar_gate_seed_cache.resize(
			static_cast<size_t>(cfg.size()) * gate_count);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            gate_count,
		            radial_func_count,
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            TwoLayerGateFullWeightMatrixData(),
		            gate_count,
		            0.0,
		            full_scalar_gate_seed_cache.data(),
		            gate_count);
	}
	if (body_combo_seed_blas) {
		body_combo_gate_seed_cache.resize(
			static_cast<size_t>(cfg.size()) * body_count);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            body_count,
		            radial_func_count,
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            TwoLayerGateBodyMixWeightMatrixData(),
		            body_count,
		            0.0,
		            body_combo_gate_seed_cache.data(),
		            body_count);
	}
	for (int ind = 0; ind < cfg.size(); ++ind) {
		const double* scalar_gate_seed_values = scalar_gate_seeds.data();
		if (full_seed_blas) {
			scalar_gate_seed_values =
				full_scalar_gate_seed_cache.data()
				+ static_cast<size_t>(ind) * gate_count;
		} else if (body_combo_seed_blas) {
			const double* body_gate_seeds =
				body_combo_gate_seed_cache.data()
				+ static_cast<size_t>(ind) * body_count;
			for (int q = 0; q < gate_count; ++q) {
				const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
				scalar_gate_seeds[q] =
					body_gate_seeds[body_index]
					* body_combo_scalar_weights[q];
			}
		} else {
			const double* gate_adjoints_by_mu =
				two_layer_gate_adjoints_.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
				gate_adjoints_by_mu,
				scalar_gate_seeds.data(),
				scalar_seed_body_scratch);
		}
			AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
				neighborhoods[ind],
				out_grad_accumulator,
				scalar_gate_seed_values,
				nullptr,
				ind,
				nullptr,
				nullptr);
		}
	if (multi_layer_gate_enabled_)
		AccumulateSecondLayerGateValueParamGradFromAdjoints(
			cfg, neighborhoods, out_grad_accumulator);
	active_second_layer_gate_values_ = saved_second_gate_values;
	active_second_layer_gate_adjoints_ = saved_second_gate_adjoints;
	}

void MLMTPR::AccumulateSecondLayerGateValueParamGradFromAdjoints(
	Configuration& cfg,
	const Neighborhoods& neighborhoods,
	Array1D& out_grad_accumulator,
	const std::vector<Vector3>* force_weights,
	const Matrix3* stress_weights,
	const std::vector<double>* energy_second_gate_adjoints)
{
	if (!multi_layer_gate_enabled_)
		return;
	if (second_layer_gate_adjoints_.size()
	    != static_cast<size_t>(cfg.size()) * radial_func_count)
		ERROR("SUS2-SH second-layer gate adjoint cache has inconsistent size");
	if (static_cast<int>(out_grad_accumulator.size()) < CoeffCount())
		out_grad_accumulator.resize(CoeffCount());
	const int gate_count = SecondLayerGateScalarCount();
	if (gate_count <= 0)
		return;

	const int body_count = SecondLayerGateBodyOrderCount();
	const bool use_mixed_derivative =
		force_weights != nullptr
		&& stress_weights != nullptr
		&& energy_second_gate_adjoints != nullptr
		&& energy_second_gate_adjoints->size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	if (SecondLayerGateUsesFullScalarWeights()
	    && second_layer_gate_scalar_values_cache_.size()
	       == static_cast<size_t>(cfg.size()) * gate_count) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            gate_count,
		            cfg.size(),
		            1.0,
		            second_layer_gate_adjoints_.data(),
		            radial_func_count,
		            second_layer_gate_scalar_values_cache_.data(),
		            gate_count,
		            1.0,
		            out_grad_accumulator.data() + SecondLayerGateWeightOffset(),
		            gate_count);
	} else if (SecondLayerGateUsesBodyLinearCombo()
	           && body_count > 0
	           && second_layer_gate_body_values_cache_.size()
	              == static_cast<size_t>(cfg.size()) * body_count
	           && second_layer_gate_scalar_values_cache_.size()
	              == static_cast<size_t>(cfg.size()) * gate_count) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            body_count,
		            cfg.size(),
		            1.0,
		            second_layer_gate_adjoints_.data(),
		            radial_func_count,
		            second_layer_gate_body_values_cache_.data(),
		            body_count,
		            1.0,
		            out_grad_accumulator.data()
		                + SecondLayerGateBodyMixWeightOffset(),
		            body_count);
		std::vector<double>& body_adjoint_cache =
			second_layer_body_adjoint_scratch_;
		body_adjoint_cache.resize(
			static_cast<size_t>(cfg.size()) * body_count);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            body_count,
		            radial_func_count,
		            1.0,
		            second_layer_gate_adjoints_.data(),
		            radial_func_count,
		            SecondLayerGateBodyMixWeightMatrixData(),
		            body_count,
		            0.0,
		            body_adjoint_cache.data(),
		            body_count);
		double* scalar_weight_grads =
			out_grad_accumulator.data() + SecondLayerGateWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* scalar_values =
				second_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(ind) * gate_count;
			const double* body_adjoint =
				body_adjoint_cache.data()
				+ static_cast<size_t>(ind) * body_count;
			for (int q = 0; q < gate_count; ++q) {
				const int body_index = second_layer_gate_scalar_body_orders_[q] - 2;
				scalar_weight_grads[q] +=
					body_adjoint[body_index] * scalar_values[q];
			}
		}
	}

	std::vector<double> scalar_gate_seeds(gate_count, 0.0);
	std::vector<double> scalar_energy_seeds(gate_count, 0.0);
	std::vector<double> scalar_seed_body_scratch;
	std::vector<double>& full_scalar_gate_seed_cache =
		second_layer_full_scalar_seed_scratch_;
	std::vector<double>& body_combo_gate_seed_cache =
		second_layer_body_combo_seed_scratch_;
	std::vector<double>& full_scalar_energy_seed_cache =
		two_layer_full_scalar_energy_seed_scratch_;
	std::vector<double>& body_combo_energy_seed_cache =
		two_layer_body_combo_energy_seed_scratch_;
	if (!use_mixed_derivative) {
		full_scalar_energy_seed_cache.clear();
		body_combo_energy_seed_cache.clear();
	}
	const bool full_seed_blas =
		SecondLayerGateUsesFullScalarWeights()
		&& second_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const bool body_combo_seed_blas =
		SecondLayerGateUsesBodyLinearCombo()
		&& body_count > 0
		&& second_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const double* body_combo_scalar_weights =
		body_combo_seed_blas ? SecondLayerGateScalarWeightData() : nullptr;
	if (full_seed_blas) {
		full_scalar_gate_seed_cache.resize(
			static_cast<size_t>(cfg.size()) * gate_count);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            gate_count,
		            radial_func_count,
		            1.0,
		            second_layer_gate_adjoints_.data(),
		            radial_func_count,
		            SecondLayerGateFullWeightMatrixData(),
		            gate_count,
		            0.0,
			            full_scalar_gate_seed_cache.data(),
			            gate_count);
		if (use_mixed_derivative) {
			full_scalar_energy_seed_cache.resize(
				static_cast<size_t>(cfg.size()) * gate_count);
			cblas_dgemm(CblasRowMajor,
			            CblasNoTrans,
			            CblasNoTrans,
			            cfg.size(),
			            gate_count,
			            radial_func_count,
			            1.0,
			            energy_second_gate_adjoints->data(),
			            radial_func_count,
			            SecondLayerGateFullWeightMatrixData(),
			            gate_count,
			            0.0,
			            full_scalar_energy_seed_cache.data(),
			            gate_count);
		}
	}
	if (body_combo_seed_blas) {
		body_combo_gate_seed_cache.resize(
			static_cast<size_t>(cfg.size()) * body_count);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            body_count,
		            radial_func_count,
		            1.0,
		            second_layer_gate_adjoints_.data(),
		            radial_func_count,
		            SecondLayerGateBodyMixWeightMatrixData(),
		            body_count,
			            0.0,
			            body_combo_gate_seed_cache.data(),
			            body_count);
		if (use_mixed_derivative) {
			body_combo_energy_seed_cache.resize(
				static_cast<size_t>(cfg.size()) * body_count);
			cblas_dgemm(CblasRowMajor,
			            CblasNoTrans,
			            CblasNoTrans,
			            cfg.size(),
			            body_count,
			            radial_func_count,
			            1.0,
			            energy_second_gate_adjoints->data(),
			            radial_func_count,
			            SecondLayerGateBodyMixWeightMatrixData(),
			            body_count,
			            0.0,
			            body_combo_energy_seed_cache.data(),
			            body_count);
		}
	}
	std::vector<Vector3> gate_der_weights;
	std::vector<double> gate_scalar_tangents;
	std::vector<double> gate_moment_tangents;
	std::vector<double> gate_body_tangents(body_count, 0.0);
	std::vector<double> gate_body_adjoint(body_count, 0.0);
	for (int ind = 0; ind < cfg.size(); ++ind) {
		const double* scalar_gate_seed_values = scalar_gate_seeds.data();
		const double* scalar_energy_seed_values = nullptr;
		if (full_seed_blas) {
			scalar_gate_seed_values =
				full_scalar_gate_seed_cache.data()
				+ static_cast<size_t>(ind) * gate_count;
			if (!full_scalar_energy_seed_cache.empty())
				scalar_energy_seed_values =
					full_scalar_energy_seed_cache.data()
					+ static_cast<size_t>(ind) * gate_count;
		} else if (body_combo_seed_blas) {
			const double* body_gate_seeds =
				body_combo_gate_seed_cache.data()
				+ static_cast<size_t>(ind) * body_count;
			for (int q = 0; q < gate_count; ++q) {
				const int body_index = second_layer_gate_scalar_body_orders_[q] - 2;
				scalar_gate_seeds[q] =
					body_gate_seeds[body_index] * body_combo_scalar_weights[q];
			}
			if (!body_combo_energy_seed_cache.empty()) {
				const double* body_energy_seeds =
					body_combo_energy_seed_cache.data()
					+ static_cast<size_t>(ind) * body_count;
				for (int q = 0; q < gate_count; ++q) {
					const int body_index =
						second_layer_gate_scalar_body_orders_[q] - 2;
					scalar_energy_seeds[q] =
						body_energy_seeds[body_index]
						* body_combo_scalar_weights[q];
				}
				scalar_energy_seed_values = scalar_energy_seeds.data();
			}
		} else {
			const double* gate_adjoints_by_mu =
				second_layer_gate_adjoints_.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			AccumulateSecondLayerGateScalarSeedsFromMuAdjoints(
				gate_adjoints_by_mu,
				scalar_gate_seeds.data(),
				scalar_seed_body_scratch);
			if (use_mixed_derivative) {
				const double* energy_gate_adjoints_by_mu =
					energy_second_gate_adjoints->data()
					+ static_cast<size_t>(ind) * radial_func_count;
				AccumulateSecondLayerGateScalarSeedsFromMuAdjoints(
					energy_gate_adjoints_by_mu,
					scalar_energy_seeds.data(),
					scalar_seed_body_scratch);
				scalar_energy_seed_values = scalar_energy_seeds.data();
			} else {
				std::fill(scalar_energy_seeds.begin(),
				          scalar_energy_seeds.end(),
				          0.0);
			}
		}
		const Neighborhood& nbh = neighborhoods[ind];
		const Vector3* gate_der_weights_ptr = nullptr;
		const double* gate_moment_tangents_ptr = nullptr;
		bool has_energy_gate_adjoint = false;
		if (scalar_energy_seed_values != nullptr)
			for (int q = 0; q < gate_count; ++q)
				if (scalar_energy_seed_values[q] != 0.0) {
					has_energy_gate_adjoint = true;
					break;
				}
		if (has_energy_gate_adjoint) {
			gate_der_weights.resize(nbh.count);
			FillWithZero(gate_der_weights);
			bool has_der_weight = false;
			for (int j = 0; j < nbh.count; ++j) {
				gate_der_weights[j] += (*force_weights)[ind];
				gate_der_weights[j] -= (*force_weights)[nbh.inds[j]];
			}
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					for (int b = 0; b < 3; ++b)
						gate_der_weights[j][a] +=
							(*stress_weights)[a][b] * nbh.vecs[j][b];
			for (int j = 0; j < nbh.count; ++j)
				if (gate_der_weights[j].NormSq() != 0.0) {
					has_der_weight = true;
					break;
				}
			if (has_der_weight) {
				gate_der_weights_ptr = gate_der_weights.data();
				CalcSecondLayerGateScalarDirectionalDerivatives(
					nbh,
					gate_der_weights,
					gate_scalar_tangents,
					&gate_moment_tangents);
				gate_moment_tangents_ptr = gate_moment_tangents.data();
				const double* energy_gate_adjoint_by_mu =
					energy_second_gate_adjoints->data()
					+ static_cast<size_t>(ind) * radial_func_count;
				if (SecondLayerGateUsesBodyLinearCombo()) {
					std::fill(gate_body_tangents.begin(), gate_body_tangents.end(), 0.0);
					for (int q = 0; q < gate_count; ++q) {
						const int body_index =
							second_layer_gate_scalar_body_orders_[q] - 2;
						gate_body_tangents[body_index] +=
							body_combo_scalar_weights[q] * gate_scalar_tangents[q];
					}
					std::fill(gate_body_adjoint.begin(), gate_body_adjoint.end(), 0.0);
					const double* body_mix_weights =
						SecondLayerGateBodyMixWeightMatrixData();
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double energy_adjoint = energy_gate_adjoint_by_mu[mu];
						if (energy_adjoint == 0.0)
							continue;
						const double* body_mix =
							body_mix_weights + static_cast<size_t>(mu) * body_count;
						for (int b = 0; b < body_count; ++b) {
							out_grad_accumulator[
								SecondLayerGateBodyMixWeightOffset()
								+ mu * body_count + b] +=
								energy_adjoint * gate_body_tangents[b];
							gate_body_adjoint[b] += energy_adjoint * body_mix[b];
						}
					}
					double* scalar_weight_grads =
						out_grad_accumulator.data() + SecondLayerGateWeightOffset();
					for (int q = 0; q < gate_count; ++q) {
						const int body_index =
							second_layer_gate_scalar_body_orders_[q] - 2;
						const double adjoint = gate_body_adjoint[body_index];
						if (adjoint != 0.0)
							scalar_weight_grads[q] +=
								adjoint * gate_scalar_tangents[q];
					}
				} else if (SecondLayerGateUsesFullScalarWeights()) {
					cblas_dger(CblasRowMajor,
					           radial_func_count,
					           gate_count,
					           1.0,
					           energy_gate_adjoint_by_mu,
					           1,
					           gate_scalar_tangents.data(),
					           1,
					           out_grad_accumulator.data()
					               + SecondLayerGateWeightOffset(),
					           gate_count);
				} else {
					ERROR("SUS2-SH second-layer gate model has an unknown mode: "
					      + second_layer_gate_mode_);
				}
			}
		}
		AccumulateSecondLayerGateScalarParamGradForScalarSeeds(
			neighborhoods[ind],
			out_grad_accumulator,
			scalar_gate_seed_values,
			gate_der_weights_ptr,
			ind,
			gate_moment_tangents_ptr,
			scalar_energy_seed_values);
	}
}

void MLMTPR::AccumulateTwoLayerGateEdgeL1SourceParamGradFromAdjoints(
	Configuration& cfg,
	const Neighborhoods& neighborhoods,
	Array1D& out_grad_accumulator)
{
	if (!TwoLayerGateEdgeL1Enabled() || two_layer_gate_edge_l1_adjoints_.empty())
		return;
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	if (source_count <= 0)
		return;
	if (two_layer_gate_edge_l1_adjoints_.size()
	    != static_cast<size_t>(cfg.size()) * source_count * 3)
		ERROR("SUS2-SH edge L1 gate adjoint cache has inconsistent size");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();
	if (static_cast<int>(out_grad_accumulator.size()) < CoeffCount())
		out_grad_accumulator.resize(CoeffCount());
	if (static_cast<int>(basic_mu_cache_.size()) < alpha_index_basic_count
	    || static_cast<int>(basic_sh_index_cache_.size()) < alpha_index_basic_count
	    || static_cast<int>(basic_scaling_block_cache_.size()) < alpha_index_basic_count)
		ERROR("SUS2-SH edge L1 source adjoint cache metadata is incomplete");
	if (static_cast<int>(mu_to_radial_eval_block_.size()) < radial_func_count)
		ERROR("SUS2-SH edge L1 source adjoint radial map is incomplete");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const bool have_edge_cache = HasTwoLayerEdgePrimitiveCache(0, false, false);
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
	std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	rb_vals.resize(static_cast<size_t>(eval_block_count) * R);
	rb_ders.resize(static_cast<size_t>(eval_block_count) * R * 5);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_s_derivatives.size()) < radial_func_count)
		radial_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_ss_derivatives.size()) < radial_func_count)
		radial_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	double sh_values[kMaxSHComponents];
	grad_dloss_dmom_.assign(alpha_moments_count, 0.0);

	for (int ind = 0; ind < cfg.size(); ++ind) {
		const double* atom_adjoints =
			two_layer_gate_edge_l1_adjoints_.data()
			+ static_cast<size_t>(ind) * source_count * 3;
		bool has_adjoint = false;
		for (int i = 0; i < source_count * 3; ++i)
			if (atom_adjoints[i] != 0.0) {
				has_adjoint = true;
				break;
			}
		if (!has_adjoint)
			continue;

		const Neighborhood& nbh = neighborhoods[ind];
		const int type_central = nbh.my_type;
		if (type_central >= species_count)
			throw MlipException("Too few species count in the MTP potential!");

		bool loaded_moment_cache = false;
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t moment_cache_offset =
			static_cast<size_t>(ind) * cached_moment_count;
		if (cached_moment_count > 0
		    && moment_cache_offset + cached_moment_count
		        <= two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data()
				+ moment_cache_offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
		if (!loaded_moment_cache)
			CalcTwoLayerGateScalarValuesOnly(
				nbh, sh_gate_scalar_values_, ind);

		std::fill(grad_dloss_dmom_.begin(), grad_dloss_dmom_.end(), 0.0);
		for (int source = 0; source < source_count; ++source) {
			const int base = TwoLayerGateEdgeL1SourceMomentIndex(source);
			if (base < 0 || base + 2 >= alpha_moments_count)
				ERROR("SUS2-SH edge L1 source adjoint moment index is out of range");
			for (int m = 0; m < 3; ++m)
				grad_dloss_dmom_[base + m] += atom_adjoints[source * 3 + m];
		}
		for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[two_layer_gate_required_product_indices_[p]];
			if (product.left < 0 || product.left >= alpha_moments_count
			    || product.right < 0 || product.right >= alpha_moments_count
			    || product.target < 0 || product.target >= alpha_moments_count)
				ERROR("SUS2-SH edge L1 source adjoint product index is out of range");
			const double adj_target = grad_dloss_dmom_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			grad_dloss_dmom_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
		if (TwoLayerGateKRankEnabled()) {
			// This path carries direct loss adjoints of the edge-L1 source tensors.
			// Accumulate dL/dW_g while raw moment values are still available, then
			// map the source adjoints back to the raw K channels for radial/type/scaling
			// parameter gradients below.
			AccumulateTwoLayerGateKRankWeightGrad(
				out_grad_accumulator, grad_dloss_dmom_, nullptr, nullptr);
			BackpropTwoLayerGateKRankAdjoints(grad_dloss_dmom_);
		}

		for (int j = 0; j < nbh.count; ++j) {
			const int type_outer = nbh.types[j];
			if (type_outer >= species_count)
				throw MlipException("Too few species count in the MTP potential!");
			const double* sh_values_use = sh_values;
			if (have_edge_cache) {
				const size_t edge = TwoLayerEdgePrimitiveOffset(ind, j);
				if ((edge + 1) * static_cast<size_t>(sh_count)
				    > two_layer_edge_sh_values_cache_.size())
					ERROR("SUS2-SH edge L1 source adjoint SH edge cache is out of range");
				sh_values_use =
					two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			} else {
				EvalRealSHValuesOnly(
					nbh.vecs[j], nbh.dists[j], sh_l_max_, sh_values);
			}

			for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block =
					radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					nbh.dists[j],
					TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
					TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
					basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] =
						p_RadialBasis->rb_vals[xi] * scaling;
					for (int d = 0; d < 5; ++d)
						rb_ders[eval_block * R * 5 + d * R + xi] =
							p_RadialBasis->rb_ders[d * R + xi] * scaling;
				}
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				if (mu < 0 || mu >= radial_func_count)
					ERROR("SUS2-SH edge L1 source adjoint mu index is out of range");
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				if (radial_base < 0
				    || radial_base + R > static_cast<int>(rb_vals.size())
				    || deriv_base < 0
				    || deriv_base + 5 * R > static_cast<int>(rb_ders.size()))
					ERROR("SUS2-SH edge L1 source adjoint radial buffer is out of range");
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				if (radial_offset < 0 || radial_offset + R > CoeffCount())
					ERROR("SUS2-SH edge L1 source adjoint radial coefficient index is out of range");
				double radial_value = 0.0;
				double radial_s_value = 0.0;
				double radial_ss_value = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double radial_coeff =
						regression_coeffs[radial_offset + xi];
					radial_value +=
						radial_coeff * rb_vals[radial_base + xi];
					radial_s_value +=
						radial_coeff * rb_ders[deriv_base + xi + R];
					radial_ss_value +=
						radial_coeff * rb_ders[deriv_base + xi + 3 * R];
				}
				radial_values[mu] = radial_value;
				radial_s_derivatives[mu] = radial_s_value;
				radial_ss_derivatives[mu] = radial_ss_value;
			}

			double* radial_coeff_accum =
				grad_radial_coeff_value_accum_.data();
			for (int mu : two_layer_gate_required_mu_indices_)
				radial_coeff_accum[mu] = 0.0;
			for (int basic_index : two_layer_gate_required_basic_indices_) {
				const double value_weight = grad_dloss_dmom_[basic_index];
				if (value_weight == 0.0)
					continue;
				if (basic_index < 0 || basic_index >= alpha_index_basic_count)
					ERROR("SUS2-SH edge L1 source adjoint basic index is out of range");
				const int mu = basic_mu_cache_[basic_index];
				const int sh_index = basic_sh_index_cache_[basic_index];
				const int scaling_block = basic_scaling_block_cache_[basic_index];
				if (mu < 0 || mu >= radial_func_count
				    || sh_index < 0 || sh_index >= sh_count
				    || scaling_block < 0 || scaling_block >= K_)
					ERROR("SUS2-SH edge L1 source adjoint basic metadata is out of range");
				const double y = sh_values_use[sh_index];
				const double gate_center_type_coeff =
					TwoLayerGateTypeCoeff(type_central, mu);
				const double gate_outer_type_coeff =
					TwoLayerGateTypeCoeff(type_outer, mu);
				const double radial_value = radial_values[mu];
				radial_coeff_accum[mu] += value_weight * y;
				out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_central, mu)] +=
					value_weight * gate_outer_type_coeff * radial_value * y;
				out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_outer, mu)] +=
					value_weight * gate_center_type_coeff * radial_value * y;
				const double type_scale =
					gate_center_type_coeff * gate_outer_type_coeff;
				const int sigma_coeff_offset =
					TwoLayerGateScalingSlopeOffset(
						scaling_block,
						type_central,
						type_outer);
				const int shift_coeff_offset =
					TwoLayerGateScalingShiftOffset(
						scaling_block,
						type_central,
						type_outer);
				if (sigma_coeff_offset < 0
				    || shift_coeff_offset < 0
				    || sigma_coeff_offset >= CoeffCount()
				    || shift_coeff_offset >= CoeffCount())
					ERROR("SUS2-SH edge L1 source adjoint radial scale coefficient index is out of range");
				out_grad_accumulator[sigma_coeff_offset] +=
					value_weight * type_scale * radial_s_derivatives[mu] * y;
				out_grad_accumulator[shift_coeff_offset] +=
					value_weight * type_scale * radial_ss_derivatives[mu] * y;
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				const double value_accum = radial_coeff_accum[mu];
				if (value_accum == 0.0)
					continue;
				const double gate_center_type_coeff =
					TwoLayerGateTypeCoeff(type_central, mu);
				const double gate_outer_type_coeff =
					TwoLayerGateTypeCoeff(type_outer, mu);
				const double type_scale =
					gate_center_type_coeff * gate_outer_type_coeff;
				const int eval_block = mu_to_radial_eval_block_[mu];
				const int radial_base = eval_block * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				if (eval_block < 0 || eval_block >= eval_block_count
				    || radial_base + R > static_cast<int>(rb_vals.size())
				    || radial_offset < 0 || radial_offset + R > CoeffCount())
					ERROR("SUS2-SH edge L1 source adjoint radial accumulation index is out of range");
				for (int xi = 0; xi < R; ++xi)
					out_grad_accumulator[radial_offset + xi] +=
						type_scale * rb_vals[radial_base + xi] * value_accum;
			}
		}
	}
}

	void MLMTPR::CalcTwoLayerGateScalarDers(
		const Neighborhood& nbh,
		std::vector<double>& gate_scalar_ders)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	const int coord_stride = 3 * moment_stride;
	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	sh_gate_moment_ders_.assign(
		static_cast<size_t>(nbh.count) * coord_stride, 0.0);
	gate_scalar_ders.assign(
		static_cast<size_t>(TwoLayerGateScalarCount()) * nbh.count * 3, 0.0);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

		if (use_lmp_table) {
			InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
			                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
			                         inv_dr,
			                         C,
			                         type_central,
			                         type_outer,
			                         r,
			                         radial_func_count,
			                         radial_values.data(),
			                         radial_derivatives.data());
		} else {
			for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
					TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
					basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			}
			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[radial_base + xi];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
			}
		}

			double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			const double* second_gate_neighbor_signals =
				multi_layer_gate_enabled_
					? SecondLayerGateNeighborSignals(nbh, j)
					: nullptr;
			for (int basic_index : two_layer_gate_required_basic_indices_) {
				const int mu = basic_mu_cache_[basic_index];
				const double gate_center_type_coeff =
					TwoLayerGateTypeCoeff(type_central, mu);
				const double gate_outer_type_coeff =
					TwoLayerGateTypeCoeff(type_outer, mu);
				const double second_gate_signal =
					second_gate_neighbor_signals != nullptr
						? second_gate_neighbor_signals[mu]
						: 0.0;
				const double second_gate_additive =
					multi_layer_gate_enabled_
						? SecondLayerGateAdditiveCoeff(type_outer, mu)
						: 0.0;
				const double gate_outer_eff =
					gate_outer_type_coeff + second_gate_additive * second_gate_signal;
				const double type_scale =
					gate_center_type_coeff * gate_outer_eff;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values[mu];
			const double radial_der = radial_derivatives[mu];
			const double y = sh_values[sh_index];
			moment_vals[basic_index] += type_scale * radial_value * y;
			for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders[sh_der_index + a];
				moment_der[a * moment_stride + basic_index] +=
					type_scale * (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
	}

	ContractTwoLayerGateKRankValues(moment_vals);
	if (TwoLayerGateKRankEnabled()) {
		for (int j = 0; j < nbh.count; ++j) {
			double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			for (int a = 0; a < 3; ++a)
				ContractTwoLayerGateKRankValues(
					moment_der + static_cast<size_t>(a) * moment_stride);
		}
	}
	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		moment_vals[product.target] += product.coeff * left_value * right_value;
		for (int j = 0; j < nbh.count; ++j) {
			double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			for (int a = 0; a < 3; ++a) {
				moment_der[a * moment_stride + product.target] += product.coeff * (
					moment_der[a * moment_stride + product.left] * right_value
					+ left_value * moment_der[a * moment_stride + product.right]);
			}
		}
	}

	for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
		const int moment = TwoLayerGateScalarMomentIndexUnchecked(q);
		for (int j = 0; j < nbh.count; ++j) {
			const double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			for (int a = 0; a < 3; ++a)
				gate_scalar_ders[(static_cast<size_t>(q) * nbh.count + j) * 3 + a] =
					moment_der[a * moment_stride + moment];
		}
	}
}

void MLMTPR::CalcTwoLayerGateWeightedScalarDers(
	const Neighborhood& nbh,
	std::vector<Vector3>& gate_scalar_ders,
	int cache_atom_index,
	int body_order_filter)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	gate_scalar_ders.assign(
		static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));
	if (TwoLayerGateScalarCount() == 0)
		return;

	bool has_weight = false;
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q)
		if (body_order_filter == 0
		    || TwoLayerGateWeightBodyOrder(q) == body_order_filter) {
			for (int mu = 0; mu < radial_func_count; ++mu)
				if (TwoLayerGateWeight(mu, q) != 0.0) {
					has_weight = true;
					break;
				}
			if (has_weight)
				break;
		}
	if (!has_weight)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central < 0 || type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");
	const double* gate_type_coeff_matrix = TwoLayerGateTypeCoeffMatrixData();

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
		if (body_order_filter != 0
		    && TwoLayerGateWeightBodyOrder(q) != body_order_filter)
			continue;
		double weight = 0.0;
		for (int mu = 0; mu < radial_func_count; ++mu)
			weight += TwoLayerGateWeight(mu, q);
		if (weight == 0.0)
			continue;
		site_energy_ders_wrt_moments_[
			TwoLayerGateScalarMomentIndexUnchecked(q)] += weight;
	}

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}
	BackpropTwoLayerGateKRankAdjoints(site_energy_ders_wrt_moments_);

	const int radial_coeff_base = C + 2 * C * C * K_;

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		if (type_outer < 0 || type_outer >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
					if (use_edge_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
						sh_values_use =
							two_layer_edge_sh_values_cache_.data() + edge * sh_count;
						sh_ders_use =
							two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
						radial_values_use =
							(use_gate_radial
								? two_layer_edge_gate_mu_vals_cache_.data()
								: two_layer_edge_mu_vals_cache_.data())
							+ edge * static_cast<size_t>(
								use_gate_radial ? gate_mu_stride : mu_stride);
						radial_derivatives_use =
							(use_gate_radial
								? two_layer_edge_gate_mu_ders_cache_.data()
								: two_layer_edge_mu_ders_cache_.data())
							+ edge * static_cast<size_t>(
								use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
				                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
				} else {
					for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
							const int scaling_block = radial_eval_to_scaling_block_[eval_block];
							const int basis_k = radial_eval_to_basis_k_[eval_block];
							p_RadialBasis->RB_Calc(
								r,
								TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
								TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
								basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
							rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
						}
					}

					for (int mu : two_layer_gate_required_mu_indices_) {
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double radial_value = 0.0;
						double radial_der = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[radial_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
			}
		}

				const double* second_gate_neighbor_signals =
					multi_layer_gate_enabled_
						? SecondLayerGateNeighborSignals(nbh, j)
						: nullptr;
					Vector3 gate_der(0.0, 0.0, 0.0);
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double adjoint = site_energy_ders_wrt_moments_[basic_index];
			if (adjoint == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const double gate_center_type_coeff =
				gate_type_coeff_matrix != nullptr
					? gate_type_coeff_matrix[
						static_cast<size_t>(type_central) * radial_func_count + mu]
					: TwoLayerGateTypeCoeff(type_central, mu);
			const double gate_outer_type_coeff =
				gate_type_coeff_matrix != nullptr
					? gate_type_coeff_matrix[
						static_cast<size_t>(type_outer) * radial_func_count + mu]
					: TwoLayerGateTypeCoeff(type_outer, mu);
			const double second_gate_signal =
					second_gate_neighbor_signals != nullptr
						? second_gate_neighbor_signals[mu]
						: 0.0;
				const double second_gate_additive =
					multi_layer_gate_enabled_
						? SecondLayerGateAdditiveCoeff(type_outer, mu)
						: 0.0;
				const double gate_outer_eff =
					gate_outer_type_coeff + second_gate_additive * second_gate_signal;
				const double type_scale =
					gate_center_type_coeff * gate_outer_eff;
				const int mu_index =
					(use_edge_cache && use_gate_radial)
						? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values_use[mu_index];
			const double radial_der = radial_derivatives_use[mu_index];
			const double y = sh_values_use[sh_index];
			for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders_use[sh_der_index + a];
				gate_der[a] += adjoint * type_scale
					* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
		gate_scalar_ders[j] = gate_der;
	}
}

void MLMTPR::CalcTwoLayerGateWeightedScalarDersForBodyOrderAdjoints(
	const Neighborhood& nbh,
	std::vector<Vector3>& gate_scalar_ders,
	int cache_atom_index,
	const double* body_order_adjoints)
{
	if (body_order_adjoints == nullptr)
		ERROR("SUS2-SH two-layer gate body-order adjoints are missing");
	std::vector<double> scalar_seeds(TwoLayerGateScalarCount(), 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q)
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const int body_order = two_layer_gate_mu_body_orders_[mu];
			scalar_seeds[q] += body_order_adjoints[body_order]
				* TwoLayerGateWeight(mu, q);
		}
	CalcTwoLayerGateWeightedScalarDersForScalarSeeds(
		nbh, gate_scalar_ders, cache_atom_index, scalar_seeds.data());
}

void MLMTPR::CalcTwoLayerGateWeightedScalarDersForScalarSeeds(
	const Neighborhood& nbh,
	std::vector<Vector3>& gate_scalar_ders,
	int cache_atom_index,
	const double* scalar_seeds)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();
	if (scalar_seeds == nullptr)
		ERROR("SUS2-SH two-layer gate scalar seeds are missing");

	gate_scalar_ders.assign(
		static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));
	const int gate_count = TwoLayerGateScalarCount();
	if (gate_count == 0)
		return;

	bool has_weighted_seed = false;
	for (int q = 0; q < gate_count; ++q)
		if (scalar_seeds[q] != 0.0) {
			has_weighted_seed = true;
			break;
		}
	if (!has_weighted_seed)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central < 0 || type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");
	const double* gate_type_coeff_matrix = TwoLayerGateTypeCoeffMatrixData();

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double seed = scalar_seeds[q];
		if (seed == 0.0)
			continue;
		site_energy_ders_wrt_moments_[
			TwoLayerGateScalarMomentIndexUnchecked(q)] += seed;
	}

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}
	BackpropTwoLayerGateKRankAdjoints(site_energy_ders_wrt_moments_);

	const int radial_coeff_base = C + 2 * C * C * K_;

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		if (type_outer < 0 || type_outer >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
				                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
				} else {
					for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
						const int scaling_block = radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_Calc(
							r,
							TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
							TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
							basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}

				for (int mu : two_layer_gate_required_mu_indices_) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

					const double* second_gate_neighbor_signals =
						multi_layer_gate_enabled_
							? SecondLayerGateNeighborSignals(nbh, j)
							: nullptr;
					const double* second_gate_neighbor_tangents = nullptr;
					double* second_gate_neighbor_adjoints =
					multi_layer_gate_enabled_
						? SecondLayerGateNeighborAdjoints(nbh, j)
						: nullptr;
				Vector3 gate_der(0.0, 0.0, 0.0);
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double adjoint = site_energy_ders_wrt_moments_[basic_index];
			if (adjoint == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values_use[mu_index];
			const double radial_der = radial_derivatives_use[mu_index];
			const double y = sh_values_use[sh_index];
			const double gate_center_type_coeff =
				gate_type_coeff_matrix != nullptr
					? gate_type_coeff_matrix[
						static_cast<size_t>(type_central) * radial_func_count + mu]
					: TwoLayerGateTypeCoeff(type_central, mu);
			const double gate_outer_type_coeff =
				gate_type_coeff_matrix != nullptr
					? gate_type_coeff_matrix[
						static_cast<size_t>(type_outer) * radial_func_count + mu]
					: TwoLayerGateTypeCoeff(type_outer, mu);
				const double second_gate_signal =
					second_gate_neighbor_signals != nullptr
						? second_gate_neighbor_signals[mu]
						: 0.0;
				const double second_gate_additive =
					multi_layer_gate_enabled_
						? SecondLayerGateAdditiveCoeff(type_outer, mu)
						: 0.0;
				const double gate_outer_eff =
					gate_outer_type_coeff + second_gate_additive * second_gate_signal;
				const double type_scale =
					gate_center_type_coeff * gate_outer_eff;
					if (second_gate_neighbor_adjoints != nullptr)
						second_gate_neighbor_adjoints[mu] +=
							adjoint * gate_center_type_coeff
							* second_gate_additive * radial_value * y;
				for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders_use[sh_der_index + a];
				gate_der[a] += adjoint * type_scale
					* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
		gate_scalar_ders[j] = gate_der;
	}
}

void MLMTPR::CalcSecondLayerGateWeightedScalarDersForScalarSeeds(
	const Neighborhood& nbh,
	std::vector<Vector3>& gate_scalar_ders,
	int cache_atom_index,
	const double* scalar_seeds)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	(void)cache_atom_index;
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (second_layer_gate_required_moments_.empty()
	    || second_layer_gate_required_moment_indices_.empty())
		BuildSecondLayerGateProductProgram();
	if (scalar_seeds == nullptr)
		ERROR("SUS2-SH second-layer gate scalar seeds are missing");

	gate_scalar_ders.assign(
		static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));
	const int gate_count = SecondLayerGateScalarCount();
	if (gate_count == 0)
		return;
	bool has_weighted_seed = false;
	for (int q = 0; q < gate_count; ++q)
		if (scalar_seeds[q] != 0.0) {
			has_weighted_seed = true;
			break;
		}
	if (!has_weighted_seed)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(second_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count
		       <= second_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				second_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[second_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcSecondLayerGateScalarValuesOnly(nbh,
		                                    sh_gate_scalar_values_,
		                                    cache_atom_index);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");
	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double seed = scalar_seeds[q];
		if (seed == 0.0)
			continue;
		const int scalar_index = second_layer_gate_scalar_indices_[q];
		site_energy_ders_wrt_moments_[alpha_moment_mapping[scalar_index]] += seed;
	}
	for (int p = static_cast<int>(second_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[second_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	if (static_cast<int>(rb_vals.size()) < eval_block_count * R)
		rb_vals.resize(eval_block_count * R);
	if (static_cast<int>(rb_ders.size()) < eval_block_count * R)
		rb_ders.resize(eval_block_count * R);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_derivatives.size()) < radial_func_count)
		radial_derivatives.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
		if (use_lmp_table) {
			InterpolateSHLmpMuTable(SecondLayerGateUsesSharedRadial()
					? second_layer_gate_radial_list : radial_list,
			                         SecondLayerGateUsesSharedRadial()
					? second_layer_gate_radial_der_list : radial_der_list,
			                         inv_dr,
			                         C,
			                         type_central,
			                         type_outer,
			                         r,
			                         radial_func_count,
			                         radial_values.data(),
			                         radial_derivatives.data());
		} else {
			for (int eval_block : second_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] =
						p_RadialBasis->rb_vals[xi] * scaling;
					rb_ders[eval_block * R + xi] =
						p_RadialBasis->rb_ders[xi] * scaling;
				}
			}
			for (int mu : second_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[radial_base + xi];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
			}
		}

		Vector3 gate_der(0.0, 0.0, 0.0);
		for (int basic_index : second_layer_gate_required_basic_indices_) {
			const double adjoint = site_energy_ders_wrt_moments_[basic_index];
			if (adjoint == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values[mu];
			const double radial_der = radial_derivatives[mu];
			const double y = sh_values[sh_index];
			const double type_scale =
				SecondLayerGateTypeCoeff(type_central, mu)
				* SecondLayerGateTypeCoeff(type_outer, mu);
			for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders[sh_der_index + a];
				gate_der[a] += adjoint * type_scale
					* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
		gate_scalar_ders[j] = gate_der;
	}
}

void MLMTPR::AccumulateTwoLayerGateScalarParamGrad(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	double gate_adjoint,
	const Vector3* gate_der_weights,
	int cache_atom_index,
	const double* gate_moment_tangents,
	double gate_moment_tangent_scale,
	int body_order_filter)
{
	bool has_der_weights = false;
	if (gate_der_weights != nullptr) {
		for (int j = 0; j < nbh.count; ++j)
			if (gate_der_weights[j].NormSq() != 0.0) {
				has_der_weights = true;
				break;
			}
	}
	if ((gate_adjoint == 0.0 && !has_der_weights)
	    || TwoLayerGateScalarCount() == 0)
		return;
	bool has_matching_weight = false;
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
		if (body_order_filter == 0
		    || TwoLayerGateWeightBodyOrder(q) == body_order_filter) {
			for (int mu = 0; mu < radial_func_count; ++mu)
				if (TwoLayerGateWeight(mu, q) != 0.0) {
					has_matching_weight = true;
					break;
				}
			if (has_matching_weight)
				break;
		}
	}
	if (!has_matching_weight)
		return;
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);
	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	grad_dloss_dmom_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
		if (body_order_filter != 0
		    && TwoLayerGateWeightBodyOrder(q) != body_order_filter)
			continue;
		double weight = 0.0;
		for (int mu = 0; mu < radial_func_count; ++mu)
			weight += TwoLayerGateWeight(mu, q);
		if (weight == 0.0)
			continue;
		const int moment = TwoLayerGateScalarMomentIndexUnchecked(q);
		site_energy_ders_wrt_moments_[moment] += weight;
	}

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
	std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
	std::vector<double>& radial_coord_s_derivatives = grad_mu_contract_coord_ders_s_;
	std::vector<double>& radial_coord_ss_derivatives = grad_mu_contract_coord_ders_ss_;
	if (static_cast<int>(rb_vals.size()) < radial_val_stride)
		rb_vals.resize(radial_val_stride);
	if (static_cast<int>(rb_ders.size()) < radial_der_stride)
		rb_ders.resize(radial_der_stride);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_derivatives.size()) < radial_func_count)
		radial_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_s_derivatives.size()) < radial_func_count)
		radial_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_ss_derivatives.size()) < radial_func_count)
		radial_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_s_derivatives.size()) < radial_func_count)
		radial_coord_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_ss_derivatives.size()) < radial_func_count)
		radial_coord_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, true);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const bool use_decoupled_gate_raw_cache =
		use_edge_cache && TwoLayerGateUsesDecoupledScaling();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;
	if (has_der_weights) {
		if (gate_moment_tangents != nullptr) {
			for (int moment_index : two_layer_gate_required_moment_indices_)
				grad_dloss_dsenders_[moment_index] =
					gate_moment_tangent_scale
					* gate_moment_tangents[moment_index];
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				const Vector3& dir = gate_der_weights[j];
				if (dir.NormSq() == 0.0)
					continue;
				const Vector3& rvec = nbh.vecs[j];
				const double r = nbh.dists[j];
				const double inv_r = 1.0 / r;
				const int type_outer = nbh.types[j];
				const double wr =
					(dir[0] * rvec[0] + dir[1] * rvec[1] + dir[2] * rvec[2])
					* inv_r;
				const double* sh_values_use = sh_values;
				const double* sh_ders_use = sh_ders;
				const double* radial_values_use = radial_values.data();
				const double* radial_derivatives_use = radial_derivatives.data();
				if (use_edge_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
					sh_values_use =
						two_layer_edge_sh_values_cache_.data() + edge * sh_count;
						sh_ders_use =
							two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							radial_values_use =
								(use_gate_radial
									? two_layer_edge_gate_mu_vals_cache_.data()
									: two_layer_edge_mu_vals_cache_.data())
								+ edge * static_cast<size_t>(
									use_gate_radial ? gate_mu_stride : mu_stride);
							radial_derivatives_use =
								(use_gate_radial
									? two_layer_edge_gate_mu_ders_cache_.data()
									: two_layer_edge_mu_ders_cache_.data())
								+ edge * static_cast<size_t>(
									use_gate_radial ? gate_mu_stride : mu_stride);
					} else {
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

						for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
							const int scaling_block = radial_eval_to_scaling_block_[eval_block];
							const int basis_k = radial_eval_to_basis_k_[eval_block];
							p_RadialBasis->RB_Calc(
								r,
								TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
								TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
								basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
								rb_ders[eval_block * R * 5 + xi] =
									p_RadialBasis->rb_ders[xi] * scaling;
							}
						}
						for (int mu : two_layer_gate_required_mu_indices_) {
							const int radial_base = mu_to_radial_eval_block_[mu] * R;
							const int deriv_base = 5 * radial_base;
							const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
							double radial_value = 0.0;
							double radial_der = 0.0;
							for (int xi = 0; xi < R; ++xi) {
								const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[deriv_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
					}

							const double* second_gate_neighbor_signals =
								multi_layer_gate_enabled_
									? SecondLayerGateNeighborSignals(nbh, j)
									: nullptr;
							for (int bi = 0;
						     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
						     ++bi) {
							const int basic_index = two_layer_gate_required_basic_indices_[bi];
							const int mu = basic_mu_cache_[basic_index];
							const double gate_center_type_coeff =
								TwoLayerGateTypeCoeff(type_central, mu);
							const double gate_outer_type_coeff =
								TwoLayerGateTypeCoeff(type_outer, mu);
								const double second_gate_signal =
									second_gate_neighbor_signals != nullptr
										? second_gate_neighbor_signals[mu]
										: 0.0;
							const double second_gate_additive =
								multi_layer_gate_enabled_
									? SecondLayerGateAdditiveCoeff(type_outer, mu)
									: 0.0;
							const double gate_outer_eff =
								gate_outer_type_coeff
								+ second_gate_additive * second_gate_signal;
							const double type_scale =
								gate_center_type_coeff * gate_outer_eff;
							const int mu_index =
								(use_edge_cache && use_gate_radial)
									? two_layer_gate_required_basic_dense_mu_indices_[bi]
								: mu;
						const int sh_index = basic_sh_index_cache_[basic_index];
						const int sh_der_index = basic_sh_der_index_cache_[basic_index];
						const double radial_value = radial_values_use[mu_index];
						const double radial_der = radial_derivatives_use[mu_index];
						const double y = sh_values_use[sh_index];
						const double wdy =
							dir[0] * sh_ders_use[sh_der_index]
							+ dir[1] * sh_ders_use[sh_der_index + 1]
						+ dir[2] * sh_ders_use[sh_der_index + 2];
					grad_dloss_dsenders_[basic_index] +=
						type_scale * (radial_der * wr * y + radial_value * wdy);
				}
			}

			ContractTwoLayerGateKRankValues(grad_dloss_dsenders_.data());
			for (int product_index : two_layer_gate_required_product_indices_) {
				const SHProduct& product = sh_products_[product_index];
				grad_dloss_dsenders_[product.target] += product.coeff * (
					grad_dloss_dsenders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
			}
		}

		for (int product_index : two_layer_gate_required_product_indices_) {
			const SHProduct& product = sh_products_[product_index];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
			grad_dloss_dmom_[product.right] +=
				grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
		}

		for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[two_layer_gate_required_product_indices_[p]];
			const double adj_target = grad_dloss_dmom_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			grad_dloss_dmom_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}

	if (gate_adjoint != 0.0) {
		if (TwoLayerGateKRankEnabled()) {
			for (int channel = 0; channel < two_layer_gate_k_rank_; ++channel)
				for (int l = sh_l_max_; l >= 0; --l)
					for (int m = -l; m <= l; ++m) {
						const int contracted =
							TwoLayerGateKRankBasicIndex(channel, l, m);
						grad_dloss_dmom_[contracted] += gate_adjoint
							* site_energy_ders_wrt_moments_[contracted];
					}
		} else {
			for (int basic_index : two_layer_gate_required_basic_indices_)
				grad_dloss_dmom_[basic_index] +=
					gate_adjoint * site_energy_ders_wrt_moments_[basic_index];
		}
	}
	if (TwoLayerGateKRankEnabled()) {
		AccumulateTwoLayerGateKRankWeightGrad(
			out_grad_accumulator,
			grad_dloss_dmom_,
			has_der_weights ? &site_energy_ders_wrt_moments_ : nullptr,
			has_der_weights ? &grad_dloss_dsenders_ : nullptr);
		BackpropTwoLayerGateKRankAdjoints(grad_dloss_dmom_);
		BackpropTwoLayerGateKRankAdjoints(site_energy_ders_wrt_moments_);
	}

	std::vector<double> radial_coeff_gate_tangent_accum(radial_func_count, 0.0);
	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double wx = has_der_weights ? gate_der_weights[j][0] : 0.0;
		const double wy = has_der_weights ? gate_der_weights[j][1] : 0.0;
		const double wz = has_der_weights ? gate_der_weights[j][2] : 0.0;
		const double wr = has_der_weights
			? (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r
			: 0.0;
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* rb_vals_use = rb_vals.data();
		const double* rb_ders_use = rb_ders.data();
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		const double* radial_s_derivatives_use = radial_s_derivatives.data();
		const double* radial_ss_derivatives_use = radial_ss_derivatives.data();
		const double* radial_coord_s_derivatives_use =
			radial_coord_s_derivatives.data();
		const double* radial_coord_ss_derivatives_use =
			radial_coord_ss_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			if (use_decoupled_gate_raw_cache) {
				rb_vals_use =
					two_layer_edge_gate_radial_vals_cache_.data()
					+ edge * radial_val_stride;
				rb_ders_use =
					two_layer_edge_gate_radial_dr_cache_.data()
					+ edge * radial_val_stride;
			} else {
				rb_vals_use =
					two_layer_edge_radial_vals_cache_.data()
					+ edge * radial_val_stride;
				rb_ders_use =
					two_layer_edge_radial_ders_cache_.data()
					+ edge * radial_der_stride;
			}
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_s_cache_.data()
					: two_layer_edge_mu_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_ss_cache_.data()
					: two_layer_edge_mu_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_s_cache_.data()
					: two_layer_edge_mu_coord_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_ss_cache_.data()
					: two_layer_edge_mu_coord_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

				for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
						TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
						basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] =
						p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				double radial_s = 0.0;
				double radial_coord_s = 0.0;
				double radial_ss = 0.0;
				double radial_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[deriv_base + xi];
					radial_s += coeff * rb_ders[deriv_base + xi + R];
					radial_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
					radial_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					radial_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
				radial_s_derivatives[mu] = radial_s;
				radial_ss_derivatives[mu] = radial_ss;
				radial_coord_s_derivatives[mu] = radial_coord_s;
				radial_coord_ss_derivatives[mu] = radial_coord_ss;
			}
		}

				double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
				double* radial_coeff_coord_accum = grad_radial_coeff_coord_accum_.data();
				for (int mu : two_layer_gate_required_mu_indices_) {
				radial_coeff_value_accum[mu] = 0.0;
				radial_coeff_coord_accum[mu] = 0.0;
				radial_coeff_gate_tangent_accum[mu] = 0.0;
			}

					const double* second_gate_neighbor_signals =
						multi_layer_gate_enabled_
							? SecondLayerGateNeighborSignals(nbh, j)
							: nullptr;
					const double* second_gate_neighbor_tangents = nullptr;
					double* second_gate_neighbor_adjoints =
						multi_layer_gate_enabled_
							? SecondLayerGateNeighborAdjoints(nbh, j)
					: nullptr;
			for (int bi = 0;
			     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
			     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double value_weight = grad_dloss_dmom_[basic_index];
			const double coord_weight = has_der_weights
				? site_energy_ders_wrt_moments_[basic_index]
				: 0.0;
			if (value_weight == 0.0 && coord_weight == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int scaling_block = basic_scaling_block_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double y = sh_values_use[sh_index];
			double dy[3];
			dy[0] = sh_ders_use[sh_der_index];
			dy[1] = sh_ders_use[sh_der_index + 1];
			dy[2] = sh_ders_use[sh_der_index + 2];
			const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];
			const double dot_val = radial_values_use[mu_index];
			const double dot_der = radial_derivatives_use[mu_index];
			const double dot_s = radial_s_derivatives_use[mu_index];
			const double dot_coord_s = radial_coord_s_derivatives_use[mu_index];
			const double dot_ss = radial_ss_derivatives_use[mu_index];
				const double dot_coord_ss = radial_coord_ss_derivatives_use[mu_index];
				const double base_direction = dot_der * wr * y + dot_val * wdy;
				const double sigma_direction = dot_coord_s * wr * y + dot_s * wdy;
				const double shift_direction = dot_coord_ss * wr * y + dot_ss * wdy;
					const double gate_center_type_coeff =
						TwoLayerGateTypeCoeff(type_central, mu);
					const double gate_outer_type_coeff =
						TwoLayerGateTypeCoeff(type_outer, mu);
						const double second_gate_signal =
							second_gate_neighbor_signals != nullptr
								? second_gate_neighbor_signals[mu]
								: 0.0;
					const double second_gate_additive =
						multi_layer_gate_enabled_
							? SecondLayerGateAdditiveCoeff(type_outer, mu)
							: 0.0;
					const double gate_outer_eff =
						gate_outer_type_coeff + second_gate_additive * second_gate_signal;
					const double type_scale =
						gate_center_type_coeff * gate_outer_eff;
					radial_coeff_value_accum[mu] +=
						value_weight * y + coord_weight * wdy;
					radial_coeff_coord_accum[mu] += coord_weight * y;

					const int sigma_coeff_offset =
						TwoLayerGateScalingSlopeOffset(
							scaling_block,
							type_central,
							type_outer);
					const int shift_coeff_offset =
						TwoLayerGateScalingShiftOffset(
							scaling_block,
							type_central,
							type_outer);
					out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_central, mu)] +=
						value_weight * gate_outer_eff * dot_val * y
						+ coord_weight * gate_outer_eff * base_direction;
					out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_outer, mu)] +=
						value_weight * gate_center_type_coeff * dot_val * y
						+ coord_weight * gate_center_type_coeff * base_direction;
						if (multi_layer_gate_enabled_) {
							const double endpoint_common =
								value_weight * dot_val * y
								+ coord_weight * base_direction;
							out_grad_accumulator[
								SecondLayerGateAdditiveCoeffIndex(type_outer, mu)] +=
								endpoint_common * gate_center_type_coeff
								* second_gate_signal;
							if (second_gate_neighbor_adjoints != nullptr)
								second_gate_neighbor_adjoints[mu] +=
									endpoint_common * gate_center_type_coeff
									* second_gate_additive;
						}
				out_grad_accumulator[sigma_coeff_offset] +=
					value_weight * type_scale * dot_s * y
					+ coord_weight * type_scale * sigma_direction;
			out_grad_accumulator[shift_coeff_offset] +=
				value_weight * type_scale * dot_ss * y
				+ coord_weight * type_scale * shift_direction;
		}

		for (int mu : two_layer_gate_required_mu_indices_) {
			const double value_accum = radial_coeff_value_accum[mu];
			const double coord_accum = radial_coeff_coord_accum[mu];
			if (value_accum == 0.0 && coord_accum == 0.0)
				continue;
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base =
					(use_decoupled_gate_raw_cache ? 1 : 5) * radial_base;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
					const double type_scale =
						TwoLayerGateTypeCoeff(type_central, mu)
							* (TwoLayerGateTypeCoeff(type_outer, mu)
							   + (multi_layer_gate_enabled_
							      ? SecondLayerGateAdditiveCoeff(type_outer, mu)
							        * (second_gate_neighbor_signals != nullptr
							           ? second_gate_neighbor_signals[mu]
							           : 0.0)
							      : 0.0));
				for (int xi = 0; xi < R; ++xi)
					out_grad_accumulator[radial_offset + xi] +=
					type_scale * (
						rb_vals_use[radial_base + xi] * value_accum
						+ rb_ders_use[deriv_base + xi] * wr * coord_accum);
		}
	}
}

void MLMTPR::AccumulateTwoLayerGateScalarParamGradForBodyOrderAdjoints(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const double* body_order_gate_adjoints,
	const Vector3* gate_der_weights,
	int cache_atom_index,
	const double* gate_moment_tangents,
	const double* body_order_energy_adjoints)
{
	if (body_order_gate_adjoints == nullptr)
		ERROR("SUS2-SH two-layer gate body-order adjoints are missing");
	std::vector<double> gate_scalar_seeds(TwoLayerGateScalarCount(), 0.0);
	std::vector<double> energy_scalar_seeds(TwoLayerGateScalarCount(), 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const int body_order = two_layer_gate_mu_body_orders_[mu];
			const double weight = TwoLayerGateWeight(mu, q);
			gate_scalar_seeds[q] += body_order_gate_adjoints[body_order] * weight;
			if (body_order_energy_adjoints != nullptr)
				energy_scalar_seeds[q] +=
					body_order_energy_adjoints[body_order] * weight;
		}
	}
	AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
		nbh,
		out_grad_accumulator,
		gate_scalar_seeds.data(),
		gate_der_weights,
		cache_atom_index,
		gate_moment_tangents,
		body_order_energy_adjoints == nullptr ? nullptr : energy_scalar_seeds.data());
}

void MLMTPR::AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const double* gate_scalar_seeds,
	const Vector3* gate_der_weights,
	int cache_atom_index,
	const double* gate_moment_tangents,
	const double* energy_scalar_seeds,
	const std::vector<double>* second_gate_directional_values,
	const double* raw_energy_moment_seeds)
{
	if (gate_scalar_seeds == nullptr)
		ERROR("SUS2-SH two-layer gate scalar seeds are missing");
	if (TwoLayerGateScalarCount() == 0)
		return;
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const int gate_count = TwoLayerGateScalarCount();

	bool has_gate_seed = false;
	bool has_energy_seed = false;
	for (int q = 0; q < gate_count; ++q) {
		if (gate_scalar_seeds[q] != 0.0)
			has_gate_seed = true;
		if (energy_scalar_seeds != nullptr && energy_scalar_seeds[q] != 0.0)
			has_energy_seed = true;
	}
	if (raw_energy_moment_seeds != nullptr)
		for (int moment = 0; moment < alpha_moments_count; ++moment)
			if (raw_energy_moment_seeds[moment] != 0.0) {
				has_energy_seed = true;
				break;
			}

	bool has_der_weights = false;
	if (has_energy_seed && gate_der_weights != nullptr) {
		for (int j = 0; j < nbh.count; ++j)
			if (gate_der_weights[j].NormSq() != 0.0) {
				has_der_weights = true;
				break;
			}
	}
	if (!has_gate_seed && !has_der_weights)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);
	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	sh_gate_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	grad_dloss_dmom_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double gate_adjoint = gate_scalar_seeds[q];
		const double energy_adjoint =
			(energy_scalar_seeds == nullptr) ? 0.0 : energy_scalar_seeds[q];
		if (gate_adjoint == 0.0 && (!has_der_weights || energy_adjoint == 0.0))
			continue;
		const int moment = TwoLayerGateScalarMomentIndexUnchecked(q);
		if (gate_adjoint != 0.0)
			site_energy_ders_wrt_moments_[moment] += gate_adjoint;
		if (has_der_weights && energy_adjoint != 0.0)
			sh_gate_energy_ders_wrt_moments_[moment] += energy_adjoint;
	}
	// This optional raw seed path is reserved for the edge-L1 source l=1
	// moment triplets.  It reuses the same reverse-over-forward machinery as
	// the scalar gate HVP without first projecting through a scalar readout.
	if (has_der_weights && raw_energy_moment_seeds != nullptr)
		for (int moment = 0; moment < alpha_moments_count; ++moment)
			sh_gate_energy_ders_wrt_moments_[moment] +=
				raw_energy_moment_seeds[moment];

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double gate_adj_target = site_energy_ders_wrt_moments_[product.target];
		if (gate_adj_target != 0.0) {
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * gate_adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * gate_adj_target;
		}
		const double energy_adj_target = sh_gate_energy_ders_wrt_moments_[product.target];
		if (energy_adj_target != 0.0) {
			sh_gate_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * energy_adj_target;
			sh_gate_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * energy_adj_target;
		}
	}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
	std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
	std::vector<double>& radial_coord_s_derivatives = grad_mu_contract_coord_ders_s_;
	std::vector<double>& radial_coord_ss_derivatives = grad_mu_contract_coord_ders_ss_;
	if (static_cast<int>(rb_vals.size()) < radial_val_stride)
		rb_vals.resize(radial_val_stride);
	if (static_cast<int>(rb_ders.size()) < radial_der_stride)
		rb_ders.resize(radial_der_stride);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_derivatives.size()) < radial_func_count)
		radial_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_s_derivatives.size()) < radial_func_count)
		radial_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_ss_derivatives.size()) < radial_func_count)
		radial_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_s_derivatives.size()) < radial_func_count)
		radial_coord_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_ss_derivatives.size()) < radial_func_count)
		radial_coord_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, true);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const bool use_decoupled_gate_raw_cache =
		use_edge_cache && TwoLayerGateUsesDecoupledScaling();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;
	if (has_der_weights) {
		if (gate_moment_tangents != nullptr) {
			for (int moment_index : two_layer_gate_required_moment_indices_)
				grad_dloss_dsenders_[moment_index] =
					gate_moment_tangents[moment_index];
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				const Vector3& dir = gate_der_weights[j];
				if (dir.NormSq() == 0.0)
					continue;
				const Vector3& rvec = nbh.vecs[j];
				const double r = nbh.dists[j];
				const double inv_r = 1.0 / r;
				const int type_outer = nbh.types[j];
				const double wr =
					(dir[0] * rvec[0] + dir[1] * rvec[1] + dir[2] * rvec[2])
					* inv_r;
				const double* sh_values_use = sh_values;
				const double* sh_ders_use = sh_ders;
				const double* radial_values_use = radial_values.data();
				const double* radial_derivatives_use = radial_derivatives.data();
				if (use_edge_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
					sh_values_use =
						two_layer_edge_sh_values_cache_.data() + edge * sh_count;
					sh_ders_use =
						two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
					radial_values_use =
						(use_gate_radial
							? two_layer_edge_gate_mu_vals_cache_.data()
							: two_layer_edge_mu_vals_cache_.data())
						+ edge * static_cast<size_t>(
							use_gate_radial ? gate_mu_stride : mu_stride);
					radial_derivatives_use =
						(use_gate_radial
							? two_layer_edge_gate_mu_ders_cache_.data()
							: two_layer_edge_mu_ders_cache_.data())
						+ edge * static_cast<size_t>(
							use_gate_radial ? gate_mu_stride : mu_stride);
				} else {
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

						for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
							const int scaling_block = radial_eval_to_scaling_block_[eval_block];
							const int basis_k = radial_eval_to_basis_k_[eval_block];
							p_RadialBasis->RB_Calc(
								r,
								TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
								TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
								basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
							rb_ders[eval_block * R * 5 + xi] =
								p_RadialBasis->rb_ders[xi] * scaling;
						}
					}
					for (int mu : two_layer_gate_required_mu_indices_) {
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int deriv_base = 5 * radial_base;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double radial_value = 0.0;
						double radial_der = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[deriv_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
				}

							const double* second_gate_neighbor_signals =
								multi_layer_gate_enabled_
									? SecondLayerGateNeighborSignals(nbh, j)
									: nullptr;
							for (int bi = 0;
							     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
							     ++bi) {
							const int basic_index = two_layer_gate_required_basic_indices_[bi];
							const int mu = basic_mu_cache_[basic_index];
							const double gate_center_type_coeff =
								TwoLayerGateTypeCoeff(type_central, mu);
							const double gate_outer_type_coeff =
								TwoLayerGateTypeCoeff(type_outer, mu);
							const double gate_outer_eff =
								gate_outer_type_coeff
									+ (multi_layer_gate_enabled_
										? SecondLayerGateAdditiveCoeff(type_outer, mu)
											* (second_gate_neighbor_signals != nullptr
											   ? second_gate_neighbor_signals[mu]
											   : 0.0)
										: 0.0);
							const double type_scale =
								gate_center_type_coeff * gate_outer_eff;
						const int mu_index =
							(use_edge_cache && use_gate_radial)
								? two_layer_gate_required_basic_dense_mu_indices_[bi]
							: mu;
					const int sh_index = basic_sh_index_cache_[basic_index];
					const int sh_der_index = basic_sh_der_index_cache_[basic_index];
					const double radial_value = radial_values_use[mu_index];
					const double radial_der = radial_derivatives_use[mu_index];
					const double y = sh_values_use[sh_index];
					const double wdy =
						dir[0] * sh_ders_use[sh_der_index]
						+ dir[1] * sh_ders_use[sh_der_index + 1]
						+ dir[2] * sh_ders_use[sh_der_index + 2];
					grad_dloss_dsenders_[basic_index] +=
						type_scale * (radial_der * wr * y + radial_value * wdy);
				}
			}

			ContractTwoLayerGateKRankValues(grad_dloss_dsenders_.data());
			for (int product_index : two_layer_gate_required_product_indices_) {
				const SHProduct& product = sh_products_[product_index];
				grad_dloss_dsenders_[product.target] += product.coeff * (
					grad_dloss_dsenders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
			}
		}

		for (int product_index : two_layer_gate_required_product_indices_) {
			const SHProduct& product = sh_products_[product_index];
			const double adj_target = sh_gate_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
			grad_dloss_dmom_[product.right] +=
				grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
		}

		for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[two_layer_gate_required_product_indices_[p]];
			const double adj_target = grad_dloss_dmom_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			grad_dloss_dmom_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}

	if (has_gate_seed) {
		if (TwoLayerGateKRankEnabled()) {
			for (int channel = 0; channel < two_layer_gate_k_rank_; ++channel)
				for (int l = sh_l_max_; l >= 0; --l)
					for (int m = -l; m <= l; ++m) {
						const int contracted =
							TwoLayerGateKRankBasicIndex(channel, l, m);
						grad_dloss_dmom_[contracted] +=
							site_energy_ders_wrt_moments_[contracted];
					}
		} else {
			for (int basic_index : two_layer_gate_required_basic_indices_)
				grad_dloss_dmom_[basic_index] +=
					site_energy_ders_wrt_moments_[basic_index];
		}
	}
	if (TwoLayerGateKRankEnabled()) {
		AccumulateTwoLayerGateKRankWeightGrad(
			out_grad_accumulator,
			grad_dloss_dmom_,
			has_der_weights ? &sh_gate_energy_ders_wrt_moments_ : nullptr,
			has_der_weights ? &grad_dloss_dsenders_ : nullptr);
		BackpropTwoLayerGateKRankAdjoints(grad_dloss_dmom_);
		BackpropTwoLayerGateKRankAdjoints(sh_gate_energy_ders_wrt_moments_);
	}

	std::vector<double> radial_coeff_gate_tangent_accum(radial_func_count, 0.0);
	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double wx = has_der_weights ? gate_der_weights[j][0] : 0.0;
		const double wy = has_der_weights ? gate_der_weights[j][1] : 0.0;
		const double wz = has_der_weights ? gate_der_weights[j][2] : 0.0;
		const double wr = has_der_weights
			? (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r
			: 0.0;
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* rb_vals_use = rb_vals.data();
		const double* rb_ders_use = rb_ders.data();
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		const double* radial_s_derivatives_use = radial_s_derivatives.data();
		const double* radial_ss_derivatives_use = radial_ss_derivatives.data();
		const double* radial_coord_s_derivatives_use =
			radial_coord_s_derivatives.data();
		const double* radial_coord_ss_derivatives_use =
			radial_coord_ss_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			if (use_decoupled_gate_raw_cache) {
				rb_vals_use =
					two_layer_edge_gate_radial_vals_cache_.data()
					+ edge * radial_val_stride;
				rb_ders_use =
					two_layer_edge_gate_radial_dr_cache_.data()
					+ edge * radial_val_stride;
			} else {
				rb_vals_use =
					two_layer_edge_radial_vals_cache_.data()
					+ edge * radial_val_stride;
				rb_ders_use =
					two_layer_edge_radial_ders_cache_.data()
					+ edge * radial_der_stride;
			}
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_s_cache_.data()
					: two_layer_edge_mu_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_ss_cache_.data()
					: two_layer_edge_mu_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_s_cache_.data()
					: two_layer_edge_mu_coord_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_ss_cache_.data()
					: two_layer_edge_mu_coord_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

				for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						TwoLayerGateScalingSlope(scaling_block, type_central, type_outer),
						TwoLayerGateScalingShift(scaling_block, type_central, type_outer),
						basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] =
						p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				double radial_s = 0.0;
				double radial_coord_s = 0.0;
				double radial_ss = 0.0;
				double radial_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[deriv_base + xi];
					radial_s += coeff * rb_ders[deriv_base + xi + R];
					radial_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
					radial_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					radial_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
				radial_s_derivatives[mu] = radial_s;
				radial_ss_derivatives[mu] = radial_ss;
				radial_coord_s_derivatives[mu] = radial_coord_s;
				radial_coord_ss_derivatives[mu] = radial_coord_ss;
			}
		}

			double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
			double* radial_coeff_coord_accum = grad_radial_coeff_coord_accum_.data();
				for (int mu : two_layer_gate_required_mu_indices_) {
				radial_coeff_value_accum[mu] = 0.0;
				radial_coeff_coord_accum[mu] = 0.0;
				radial_coeff_gate_tangent_accum[mu] = 0.0;
			}

				const double* second_gate_neighbor_signals =
					multi_layer_gate_enabled_
						? SecondLayerGateNeighborSignals(nbh, j)
						: nullptr;
				const double* second_gate_neighbor_tangents = nullptr;
				if (multi_layer_gate_enabled_
				    && second_gate_directional_values != nullptr) {
					const int atom_index = nbh.inds[j];
					if (atom_index >= 0
					    && second_gate_directional_values->size()
					        >= static_cast<size_t>(atom_index + 1)
					           * radial_func_count)
						second_gate_neighbor_tangents =
							second_gate_directional_values->data()
							+ static_cast<size_t>(atom_index) * radial_func_count;
				}
				double* second_gate_neighbor_adjoints =
					multi_layer_gate_enabled_
						? SecondLayerGateNeighborAdjoints(nbh, j)
					: nullptr;
			for (int bi = 0;
			     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
			     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double value_weight = grad_dloss_dmom_[basic_index];
			const double coord_weight = has_der_weights
				? sh_gate_energy_ders_wrt_moments_[basic_index]
				: 0.0;
			if (value_weight == 0.0 && coord_weight == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int scaling_block = basic_scaling_block_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double y = sh_values_use[sh_index];
			double dy[3];
			dy[0] = sh_ders_use[sh_der_index];
			dy[1] = sh_ders_use[sh_der_index + 1];
			dy[2] = sh_ders_use[sh_der_index + 2];
			const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];
			const double dot_val = radial_values_use[mu_index];
			const double dot_der = radial_derivatives_use[mu_index];
			const double dot_s = radial_s_derivatives_use[mu_index];
			const double dot_coord_s = radial_coord_s_derivatives_use[mu_index];
			const double dot_ss = radial_ss_derivatives_use[mu_index];
				const double dot_coord_ss = radial_coord_ss_derivatives_use[mu_index];
				const double base_direction = dot_der * wr * y + dot_val * wdy;
				const double sigma_direction = dot_coord_s * wr * y + dot_s * wdy;
				const double shift_direction = dot_coord_ss * wr * y + dot_ss * wdy;
					const double gate_center_type_coeff =
						TwoLayerGateTypeCoeff(type_central, mu);
					const double gate_outer_type_coeff =
						TwoLayerGateTypeCoeff(type_outer, mu);
							const double second_gate_signal =
								second_gate_neighbor_signals != nullptr
									? second_gate_neighbor_signals[mu]
									: 0.0;
							const double second_gate_tangent =
								second_gate_neighbor_tangents != nullptr
									? second_gate_neighbor_tangents[mu]
									: 0.0;
						const double second_gate_additive =
							multi_layer_gate_enabled_
								? SecondLayerGateAdditiveCoeff(type_outer, mu)
								: 0.0;
						const double gate_outer_eff_direction =
							second_gate_additive * second_gate_tangent;
						const double gate_outer_eff =
							gate_outer_type_coeff
							+ second_gate_additive * second_gate_signal;
					const double type_scale =
						gate_center_type_coeff * gate_outer_eff;
					radial_coeff_value_accum[mu] +=
						value_weight * y + coord_weight * wdy;
					radial_coeff_coord_accum[mu] += coord_weight * y;
					radial_coeff_gate_tangent_accum[mu] +=
						coord_weight * gate_outer_eff_direction * y;

					const int sigma_coeff_offset =
						TwoLayerGateScalingSlopeOffset(
							scaling_block,
							type_central,
							type_outer);
					const int shift_coeff_offset =
						TwoLayerGateScalingShiftOffset(
							scaling_block,
							type_central,
							type_outer);
							out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_central, mu)] +=
								value_weight * gate_outer_eff * dot_val * y
							+ coord_weight * (
								gate_outer_eff * base_direction
								+ gate_outer_eff_direction * dot_val * y);
						out_grad_accumulator[TwoLayerGateTypeCoeffIndex(type_outer, mu)] +=
							value_weight * gate_center_type_coeff * dot_val * y
							+ coord_weight * gate_center_type_coeff * base_direction;
						if (multi_layer_gate_enabled_) {
							const double endpoint_common =
								value_weight * dot_val * y
								+ coord_weight * base_direction;
							out_grad_accumulator[
									SecondLayerGateAdditiveCoeffIndex(type_outer, mu)] +=
									endpoint_common * gate_center_type_coeff
									* second_gate_signal
									+ coord_weight * gate_center_type_coeff
										* second_gate_tangent * dot_val * y;
								if (second_gate_neighbor_adjoints != nullptr)
									second_gate_neighbor_adjoints[mu] +=
										endpoint_common * gate_center_type_coeff
									* second_gate_additive;
						}
					out_grad_accumulator[sigma_coeff_offset] +=
						value_weight * type_scale * dot_s * y
						+ coord_weight * (
							type_scale * sigma_direction
							+ gate_center_type_coeff * gate_outer_eff_direction * dot_s * y);
					out_grad_accumulator[shift_coeff_offset] +=
						value_weight * type_scale * dot_ss * y
						+ coord_weight * (
						type_scale * shift_direction
						+ gate_center_type_coeff * gate_outer_eff_direction * dot_ss * y);
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				const double value_accum = radial_coeff_value_accum[mu];
				const double coord_accum = radial_coeff_coord_accum[mu];
				const double gate_tangent_accum = radial_coeff_gate_tangent_accum[mu];
				if (value_accum == 0.0 && coord_accum == 0.0
				    && gate_tangent_accum == 0.0)
					continue;
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base =
					(use_decoupled_gate_raw_cache ? 1 : 5) * radial_base;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				const double type_scale =
					TwoLayerGateTypeCoeff(type_central, mu)
						* (TwoLayerGateTypeCoeff(type_outer, mu)
						   + (multi_layer_gate_enabled_
						      ? SecondLayerGateAdditiveCoeff(type_outer, mu)
						        * (second_gate_neighbor_signals != nullptr
						           ? second_gate_neighbor_signals[mu]
						           : 0.0)
						      : 0.0));
					for (int xi = 0; xi < R; ++xi)
						out_grad_accumulator[radial_offset + xi] +=
						type_scale * (
							rb_vals_use[radial_base + xi] * value_accum
							+ rb_ders_use[deriv_base + xi] * wr * coord_accum)
						+ TwoLayerGateTypeCoeff(type_central, mu)
							* rb_vals_use[radial_base + xi] * gate_tangent_accum;
			}
	}
}

void MLMTPR::AccumulateSecondLayerGateScalarParamGradForScalarSeeds(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const double* gate_scalar_seeds,
	const Vector3* gate_der_weights,
	int cache_atom_index,
	const double* gate_moment_tangents,
	const double* energy_scalar_seeds)
{
	if (gate_scalar_seeds == nullptr)
		ERROR("SUS2-SH second-layer gate scalar seeds are missing");
	if (SecondLayerGateScalarCount() == 0)
		return;
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (second_layer_gate_required_moments_.empty()
	    || second_layer_gate_required_moment_indices_.empty())
		BuildSecondLayerGateProductProgram();
	const int gate_count = SecondLayerGateScalarCount();

	bool has_gate_seed = false;
	bool has_energy_seed = false;
	for (int q = 0; q < gate_count; ++q) {
		if (gate_scalar_seeds[q] != 0.0)
			has_gate_seed = true;
		if (energy_scalar_seeds != nullptr && energy_scalar_seeds[q] != 0.0)
			has_energy_seed = true;
	}

	bool has_der_weights = false;
	if (has_energy_seed && gate_der_weights != nullptr) {
		for (int j = 0; j < nbh.count; ++j)
			if (gate_der_weights[j].NormSq() != 0.0) {
				has_der_weights = true;
				break;
			}
	}
	if (!has_gate_seed && !has_der_weights)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(second_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count
		       <= second_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				second_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[second_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcSecondLayerGateScalarValuesOnly(nbh,
		                                    sh_gate_scalar_values_,
		                                    cache_atom_index);
	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	sh_gate_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	grad_dloss_dmom_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double gate_adjoint = gate_scalar_seeds[q];
		const double energy_adjoint =
			(energy_scalar_seeds == nullptr) ? 0.0 : energy_scalar_seeds[q];
		if (gate_adjoint == 0.0 && (!has_der_weights || energy_adjoint == 0.0))
			continue;
		const int scalar_index = second_layer_gate_scalar_indices_[q];
		const int moment = alpha_moment_mapping[scalar_index];
		if (gate_adjoint != 0.0)
			site_energy_ders_wrt_moments_[moment] += gate_adjoint;
		if (has_der_weights && energy_adjoint != 0.0)
			sh_gate_energy_ders_wrt_moments_[moment] += energy_adjoint;
	}
	for (int p = static_cast<int>(second_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[second_layer_gate_required_product_indices_[p]];
		const double gate_adj_target = site_energy_ders_wrt_moments_[product.target];
		if (gate_adj_target != 0.0) {
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * gate_adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * gate_adj_target;
		}
		const double energy_adj_target =
			sh_gate_energy_ders_wrt_moments_[product.target];
		if (energy_adj_target != 0.0) {
			sh_gate_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * energy_adj_target;
			sh_gate_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * energy_adj_target;
		}
	}

	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
	std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
	std::vector<double>& radial_coord_s_derivatives =
		grad_mu_contract_coord_ders_s_;
	std::vector<double>& radial_coord_ss_derivatives =
		grad_mu_contract_coord_ders_ss_;
	if (static_cast<int>(rb_vals.size()) < radial_val_stride)
		rb_vals.resize(radial_val_stride);
	if (static_cast<int>(rb_ders.size()) < radial_der_stride)
		rb_ders.resize(radial_der_stride);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_derivatives.size()) < radial_func_count)
		radial_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_s_derivatives.size()) < radial_func_count)
		radial_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_ss_derivatives.size()) < radial_func_count)
		radial_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_s_derivatives.size()) < radial_func_count)
		radial_coord_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_ss_derivatives.size()) < radial_func_count)
		radial_coord_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	if (has_der_weights) {
		if (gate_moment_tangents != nullptr) {
			for (int moment_index : second_layer_gate_required_moment_indices_)
				grad_dloss_dsenders_[moment_index] =
					gate_moment_tangents[moment_index];
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				const Vector3& dir = gate_der_weights[j];
				if (dir.NormSq() == 0.0)
					continue;
				const Vector3& rvec = nbh.vecs[j];
				const double r = nbh.dists[j];
				const double inv_r = 1.0 / r;
				const int type_outer = nbh.types[j];
				const double wr =
					(dir[0] * rvec[0] + dir[1] * rvec[1] + dir[2] * rvec[2])
					* inv_r;
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
				for (int eval_block : second_layer_gate_required_radial_eval_blocks_) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * C
							+ C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] =
							p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R * 5 + xi] =
							p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu : second_layer_gate_required_mu_indices_) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int deriv_base = 5 * radial_base;
					const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[deriv_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
				for (int basic_index : second_layer_gate_required_basic_indices_) {
					const int mu = basic_mu_cache_[basic_index];
					const double type_scale =
						SecondLayerGateTypeCoeff(type_central, mu)
						* SecondLayerGateTypeCoeff(type_outer, mu);
					const int sh_index = basic_sh_index_cache_[basic_index];
					const int sh_der_index = basic_sh_der_index_cache_[basic_index];
					const double radial_value = radial_values[mu];
					const double radial_der = radial_derivatives[mu];
					const double y = sh_values[sh_index];
					const double wdy =
						dir[0] * sh_ders[sh_der_index]
						+ dir[1] * sh_ders[sh_der_index + 1]
						+ dir[2] * sh_ders[sh_der_index + 2];
					grad_dloss_dsenders_[basic_index] +=
						type_scale * (radial_der * wr * y + radial_value * wdy);
				}
			}

			for (int product_index : second_layer_gate_required_product_indices_) {
				const SHProduct& product = sh_products_[product_index];
				grad_dloss_dsenders_[product.target] += product.coeff * (
					grad_dloss_dsenders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
			}
		}

		for (int product_index : second_layer_gate_required_product_indices_) {
			const SHProduct& product = sh_products_[product_index];
			const double adj_target =
				sh_gate_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
			grad_dloss_dmom_[product.right] +=
				grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
		}

		for (int p = static_cast<int>(second_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[second_layer_gate_required_product_indices_[p]];
			const double adj_target = grad_dloss_dmom_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			grad_dloss_dmom_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}

	if (has_gate_seed)
		for (int basic_index : second_layer_gate_required_basic_indices_)
			grad_dloss_dmom_[basic_index] +=
				site_energy_ders_wrt_moments_[basic_index];

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double wx = has_der_weights ? gate_der_weights[j][0] : 0.0;
		const double wy = has_der_weights ? gate_der_weights[j][1] : 0.0;
		const double wz = has_der_weights ? gate_der_weights[j][2] : 0.0;
		const double wr = has_der_weights
			? (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r
			: 0.0;
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
		for (int eval_block : second_layer_gate_required_radial_eval_blocks_) {
			const int scaling_block = radial_eval_to_scaling_block_[eval_block];
			const int basis_k = radial_eval_to_basis_k_[eval_block];
			p_RadialBasis->RB_Calc(
				r,
				regression_coeffs[
					C + 2 * scaling_block * C * C + C * type_central + type_outer],
				regression_coeffs[
					C + 2 * scaling_block * C * C + C * C
					+ C * type_central + type_outer],
				basis_k);
			for (int xi = 0; xi < R; ++xi)
				rb_vals[eval_block * R + xi] =
					p_RadialBasis->rb_vals[xi] * scaling;
			for (int xi = 0; xi < R * 5; ++xi)
				rb_ders[eval_block * R * 5 + xi] =
					p_RadialBasis->rb_ders[xi] * scaling;
		}
		for (int mu : second_layer_gate_required_mu_indices_) {
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int deriv_base = 5 * radial_base;
			const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
			double radial_value = 0.0;
			double radial_der = 0.0;
			double radial_s = 0.0;
			double radial_ss = 0.0;
			double radial_coord_s = 0.0;
			double radial_coord_ss = 0.0;
			for (int xi = 0; xi < R; ++xi) {
				const double coeff = regression_coeffs[radial_offset + xi];
				radial_value += coeff * rb_vals[radial_base + xi];
				radial_der += coeff * rb_ders[deriv_base + xi];
				radial_s += coeff * rb_ders[deriv_base + xi + R];
				radial_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
				radial_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
				radial_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
			}
			radial_values[mu] = radial_value;
			radial_derivatives[mu] = radial_der;
			radial_s_derivatives[mu] = radial_s;
			radial_ss_derivatives[mu] = radial_ss;
			radial_coord_s_derivatives[mu] = radial_coord_s;
			radial_coord_ss_derivatives[mu] = radial_coord_ss;
			grad_radial_coeff_value_accum_[mu] = 0.0;
			grad_radial_coeff_coord_accum_[mu] = 0.0;
		}
		for (int basic_index : second_layer_gate_required_basic_indices_) {
			const double value_weight = grad_dloss_dmom_[basic_index];
			const double coord_weight = has_der_weights
				? sh_gate_energy_ders_wrt_moments_[basic_index]
				: 0.0;
			if (value_weight == 0.0 && coord_weight == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int scaling_block = basic_scaling_block_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double y = sh_values[sh_index];
			double dy[3];
			dy[0] = sh_ders[sh_der_index];
			dy[1] = sh_ders[sh_der_index + 1];
			dy[2] = sh_ders[sh_der_index + 2];
			const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];
			const double dot_val = radial_values[mu];
			const double dot_der = radial_derivatives[mu];
			const double dot_s = radial_s_derivatives[mu];
			const double dot_ss = radial_ss_derivatives[mu];
			const double dot_coord_s = radial_coord_s_derivatives[mu];
			const double dot_coord_ss = radial_coord_ss_derivatives[mu];
			const double base_direction = dot_der * wr * y + dot_val * wdy;
			const double sigma_direction = dot_coord_s * wr * y + dot_s * wdy;
			const double shift_direction = dot_coord_ss * wr * y + dot_ss * wdy;
			const double gate_center_type_coeff =
				SecondLayerGateTypeCoeff(type_central, mu);
			const double gate_outer_type_coeff =
				SecondLayerGateTypeCoeff(type_outer, mu);
			const double type_scale =
				gate_center_type_coeff * gate_outer_type_coeff;
			grad_radial_coeff_value_accum_[mu] +=
				value_weight * y + coord_weight * wdy;
			grad_radial_coeff_coord_accum_[mu] += coord_weight * y;
			const int sigma_coeff_offset =
				C + 2 * C * C * scaling_block + type_central * C + type_outer;
			out_grad_accumulator[SecondLayerGateTypeCoeffIndex(type_central, mu)] +=
				value_weight * gate_outer_type_coeff * dot_val * y
				+ coord_weight * gate_outer_type_coeff * base_direction;
			out_grad_accumulator[SecondLayerGateTypeCoeffIndex(type_outer, mu)] +=
				value_weight * gate_center_type_coeff * dot_val * y
				+ coord_weight * gate_center_type_coeff * base_direction;
			out_grad_accumulator[sigma_coeff_offset] +=
				value_weight * type_scale * dot_s * y
				+ coord_weight * type_scale * sigma_direction;
			out_grad_accumulator[sigma_coeff_offset + C * C] +=
				value_weight * type_scale * dot_ss * y
				+ coord_weight * type_scale * shift_direction;
		}
		for (int mu : second_layer_gate_required_mu_indices_) {
			const double value_accum = grad_radial_coeff_value_accum_[mu];
			const double coord_accum = grad_radial_coeff_coord_accum_[mu];
			if (value_accum == 0.0 && coord_accum == 0.0)
				continue;
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int deriv_base = 5 * radial_base;
			const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
			const double type_scale =
				SecondLayerGateTypeCoeff(type_central, mu)
				* SecondLayerGateTypeCoeff(type_outer, mu);
			for (int xi = 0; xi < R; ++xi)
				out_grad_accumulator[radial_offset + xi] +=
					type_scale * (
						rb_vals[radial_base + xi] * value_accum
						+ rb_ders[deriv_base + xi] * wr * coord_accum);
		}
	}
}

void MLMTPR::CalcSHBasisGateDers(const Neighborhood& nbh,
                                 std::vector<double>& gate_basis_ders)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	const int scalar_stride = alpha_scalar_moments;
	sh_gate_moment_ders_.assign(
		static_cast<size_t>(nbh.count) * moment_stride, 0.0);
	gate_basis_ders.assign(
		static_cast<size_t>(nbh.count) * scalar_stride, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
		} else {
			EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

		double* gate_moment_der = sh_gate_moment_ders_.data()
			+ static_cast<size_t>(j) * moment_stride;
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const int mu = basic_mu_cache_[i];
			const int sh_index = basic_sh_index_cache_[i];
			const double pair_type_scale =
				OuterTypeCoeffFast(regression_coeffs,
				                   outer_type_coeff_offset,
				                   outer_type_coeff_stride,
				                   R,
				                   type_central,
				                   mu)
				* OuterTypeCoeffFast(regression_coeffs,
				                     outer_type_coeff_offset,
				                     outer_type_coeff_stride,
				                     R,
				                     type_outer,
				                     mu);
			gate_moment_der[i] +=
				pair_type_scale * radial_values_use[mu] * sh_values_use[sh_index];
		}
	}
	if (outer_k_rank1_enabled_)
		for (int j = 0; j < nbh.count; ++j)
			ContractOuterKRank1Values(
				sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * moment_stride);

	if (UseSHProductRows()) {
		for (const SHProductRow& row : sh_product_rows_) {
			const int end = row.term_begin + row.term_count;
			for (int t = row.term_begin; t < end; ++t) {
				const SHProductRowTerm& term = sh_product_row_terms_[t];
				for (int j = 0; j < nbh.count; ++j) {
					double* gate_moment_der = sh_gate_moment_ders_.data()
						+ static_cast<size_t>(j) * moment_stride;
					gate_moment_der[row.target] += term.coeff * (
						gate_moment_der[term.left] * moment_vals[term.right]
						+ moment_vals[term.left] * gate_moment_der[term.right]);
				}
			}
		}
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			for (int j = 0; j < nbh.count; ++j) {
				double* gate_moment_der = sh_gate_moment_ders_.data()
					+ static_cast<size_t>(j) * moment_stride;
				gate_moment_der[product.target] += product.coeff * (
					gate_moment_der[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * gate_moment_der[product.right]);
			}
		}
	}

	for (int j = 0; j < nbh.count; ++j) {
		const double* gate_moment_der = sh_gate_moment_ders_.data()
			+ static_cast<size_t>(j) * moment_stride;
		double* gate_basis_der = gate_basis_ders.data()
			+ static_cast<size_t>(j) * scalar_stride;
		for (int i = 0; i < alpha_scalar_moments; ++i)
			gate_basis_der[i] = gate_moment_der[alpha_moment_mapping[i]];
	}
}

void MLMTPR::AccumulateSHBasisGateDers(
	const Neighborhood& nbh,
	int center_atom_index,
	bool need_forces,
	bool need_stress,
	std::vector<double>& gate_linear_adjoints,
	std::vector<double>& edge_l1_projection_linear_adjoints)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const bool pure_matrix = OuterUsesPureRadialMatrix();
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (alpha_scalar_moments == 0)
		return;
	const size_t atom_stride =
		static_cast<size_t>(alpha_scalar_moments) * radial_func_count;
	if (atom_stride == 0 || gate_linear_adjoints.size() % atom_stride != 0)
		ERROR("SUS2-SH gate linear adjoint buffer has an incompatible size.");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	if (static_cast<int>(sh_gate_moment_ders_.size()) != moment_stride)
		sh_gate_moment_ders_.resize(moment_stride);
	const bool use_product_rows = UseSHProductRows();
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const bool has_edge_l1 = HasNonzeroTwoLayerGateEdgeL1Weights();
	const bool use_edge_der_cache =
		has_edge_l1 && HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const int edge_l1_source_count =
		has_edge_l1 ? TwoLayerGateEdgeL1SourceCount() : 0;
	if (has_edge_l1) {
		const size_t atom_count = gate_linear_adjoints.size() / atom_stride;
		const size_t expected_projection_adjoint_count =
			atom_count * 3 * alpha_scalar_moments * radial_func_count;
		if (edge_l1_source_count <= 0
		    || edge_l1_projection_linear_adjoints.size()
		        != expected_projection_adjoint_count)
			ERROR("SUS2-SH edge L1 linear projection adjoint buffer has an incompatible size");
		sh_gate_edge_linear_adjoints_.resize(
			static_cast<size_t>(alpha_scalar_moments) * radial_func_count);
		sh_gate_edge_l1_y_linear_adjoints_.resize(
			static_cast<size_t>(alpha_scalar_moments) * 3);
	}
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int center_type_coeff_offset = RadialCoeffOffset();
	const int center_type_coeff_stride = RadialCoeffBlockSize();

	for (int j = 0; j < nbh.count; ++j) {
		const int atom_index = nbh.inds[j];
		if (atom_index < 0
		    || static_cast<size_t>(atom_index + 1) * atom_stride
		        > gate_linear_adjoints.size())
			ERROR("SUS2-SH two-layer gate linear component atom index is out of range");

		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			if (use_edge_der_cache)
				sh_ders_use =
					two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			else if (has_edge_l1) {
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
				sh_values_use = sh_values;
				sh_ders_use = sh_ders;
			}
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
			if (has_edge_l1)
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
			else
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0;
				     eval_block < static_cast<int>(radial_eval_to_scaling_block_.size());
				     ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C
							+ C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C
							+ C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] =
							p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi]
							* rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

		int neighbor_atom_index_for_tanh = atom_index;
		const double* gate_signal_by_mu =
			has_edge_l1
				? PrepareTwoLayerGateEdgeL1NeighborSignals(
					nbh,
					j,
					TwoLayerGateNeighborSignals(nbh, j),
					sh_values_use,
					neighbor_atom_index_for_tanh)
				: TwoLayerGateNeighborSignals(nbh, j);
		PrepareTwoLayerGatePairMuBuffers(
			type_central,
			type_outer,
			nullptr,
			-1,
			gate_signal_by_mu,
			neighbor_atom_index_for_tanh,
			kGateMuBufferAdditive);
		double* gate_adjoints = gate_linear_adjoints.data()
			+ static_cast<size_t>(atom_index) * atom_stride;
		if (has_edge_l1)
			std::fill(sh_gate_edge_linear_adjoints_.begin(),
			          sh_gate_edge_linear_adjoints_.end(),
			          0.0);
		for (int mu = 0; mu < radial_func_count; ++mu) {
			std::fill(sh_gate_moment_ders_.begin(),
			          sh_gate_moment_ders_.end(),
			          0.0);
			auto set_basic_tangent = [&](int basic_index) {
				const int sh_index = basic_sh_index_cache_[basic_index];
				const double gate_signal_derivative =
					OuterTypeCoeffFast(regression_coeffs,
					                   center_type_coeff_offset,
					                   center_type_coeff_stride,
					                   R,
					                   type_central,
					                   mu)
					* two_layer_gate_additive_mu_buffer_[mu];
				sh_gate_moment_ders_[basic_index] =
					gate_signal_derivative
					* radial_values_use[mu] * sh_values_use[sh_index];
			};
			if (pure_matrix) {
				// d A[o,l,m] / d gate[j,k,l] = W[l,o,k](r) *
				// d x[j,k,l]/d gate[j,k,l] * Y[l,m].  Accumulating these
				// tangents through the existing outer DAG preserves the full
				// gate-environment force/stress chain in every linear column.
				const int l = mu % (sh_l_max_ + 1);
				const int k = mu / (sh_l_max_ + 1);
				const double gate_signal_derivative =
					OuterTypeCoeffFast(regression_coeffs,
						center_type_coeff_offset, center_type_coeff_stride,
						R, type_central, mu) * two_layer_gate_additive_mu_buffer_[mu];
				const size_t kernel_stride =
					static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_ * sh_k_max_;
				const double* kernels = use_edge_cache
					? two_layer_edge_radial_mixing_kernel_cache_.data()
						+ TwoLayerEdgePrimitiveOffset(-1, j) * kernel_stride
					: nullptr;
				for (int o = 0; o < outer_k_rank_; ++o) {
					const size_t lok = (static_cast<size_t>(l) * outer_k_rank_ + o) * sh_k_max_ + k;
					double kernel = kernels == nullptr ? 0.0 : kernels[lok];
					if (kernels == nullptr) {
						const int basis_begin = mu_to_radial_eval_block_[mu] * R;
						const int coeff_begin = TwoLayerGateRadialMixingCoeffIndex(l, o, k, 0);
						for (int b = 0; b < R; ++b)
							kernel += regression_coeffs[coeff_begin + b] * rb_vals[basis_begin + b];
					}
					for (int m = -l; m <= l; ++m)
						sh_gate_moment_ders_[OuterKRank1OuterBasicIndex(o, l, m)] =
							gate_signal_derivative * kernel * sh_values_use[SHFlatIndex(l, m)];
				}
			} else if (use_basic_mu_groups) {
				for (int cursor = basic_mu_offsets_[mu];
				     cursor < basic_mu_offsets_[mu + 1];
				     ++cursor)
					set_basic_tangent(basic_indices_by_mu_[cursor]);
			} else {
				for (int basic_index = 0;
				     basic_index < alpha_index_basic_count;
				     ++basic_index)
					if (basic_mu_cache_[basic_index] == mu)
						set_basic_tangent(basic_index);
			}
			if (!pure_matrix)
				ContractOuterKRank1Values(sh_gate_moment_ders_.data());

			if (use_product_rows) {
				for (const SHProductRow& row : sh_product_rows_) {
					double value = 0.0;
					const int end = row.term_begin + row.term_count;
					for (int t = row.term_begin; t < end; ++t) {
						const SHProductRowTerm& term = sh_product_row_terms_[t];
						value += term.coeff * (
							sh_gate_moment_ders_[term.left] * moment_vals[term.right]
							+ moment_vals[term.left] * sh_gate_moment_ders_[term.right]);
					}
					sh_gate_moment_ders_[row.target] = value;
				}
			} else {
				for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
				     p < sh_products_.size(); ++p) {
					const SHProduct& product = sh_products_[p];
					sh_gate_moment_ders_[product.target] += product.coeff * (
						sh_gate_moment_ders_[product.left] * moment_vals[product.right]
						+ moment_vals[product.left] * sh_gate_moment_ders_[product.right]);
				}
			}

			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const double gate_adjoint =
					sh_gate_moment_ders_[alpha_moment_mapping[i]];
				gate_adjoints[static_cast<size_t>(i) * radial_func_count + mu] +=
					gate_adjoint;
				if (has_edge_l1)
					sh_gate_edge_linear_adjoints_[
						static_cast<size_t>(i) * radial_func_count + mu] =
						gate_adjoint;
			}
		}

		if (!has_edge_l1)
			continue;

		const int y1_indices[3] = {
			SHFlatIndex(1, -1),
			SHFlatIndex(1, 0),
			SHFlatIndex(1, 1)
		};
		const double y1[3] = {
			sh_values_use[y1_indices[0]],
			sh_values_use[y1_indices[1]],
			sh_values_use[y1_indices[2]]
		};
		const size_t projection_stride =
			static_cast<size_t>(alpha_scalar_moments) * radial_func_count;
		double* projection_adjoints = edge_l1_projection_linear_adjoints.data()
			+ static_cast<size_t>(atom_index) * 3 * projection_stride;
		for (int m = 0; m < 3; ++m)
			cblas_daxpy(static_cast<int>(projection_stride),
			            y1[m],
			            sh_gate_edge_linear_adjoints_.data(),
			            1,
			            projection_adjoints + static_cast<size_t>(m) * projection_stride,
			            1);

		const size_t atom_count = gate_linear_adjoints.size() / atom_stride;
		const size_t expected_weighted_size =
			atom_count * static_cast<size_t>(radial_func_count) * 3;
		const double* weighted_values = nullptr;
		if (two_layer_gate_edge_l1_weighted_values_.size()
		    == expected_weighted_size) {
			weighted_values =
				two_layer_gate_edge_l1_weighted_values_.data()
				+ static_cast<size_t>(atom_index) * radial_func_count * 3;
		} else {
			const size_t expected_value_size =
				atom_count * static_cast<size_t>(edge_l1_source_count) * 3;
			if (two_layer_gate_edge_l1_values_.size() != expected_value_size)
				ERROR("SUS2-SH edge L1 value cache has an incompatible size");
			const double* edge_l1_values =
				two_layer_gate_edge_l1_values_.data()
				+ static_cast<size_t>(atom_index) * edge_l1_source_count * 3;
			two_layer_gate_edge_l1_value_buffer_.resize(
				static_cast<size_t>(radial_func_count) * 3);
			const double* edge_l1_weights = TwoLayerGateEdgeL1WeightData();
			cblas_dgemm(CblasRowMajor,
			            CblasNoTrans,
			            CblasNoTrans,
			            radial_func_count,
			            3,
			            edge_l1_source_count,
			            1.0,
			            edge_l1_weights,
			            edge_l1_source_count,
			            edge_l1_values,
			            3,
			            0.0,
			            two_layer_gate_edge_l1_value_buffer_.data(),
			            3);
			weighted_values = two_layer_gate_edge_l1_value_buffer_.data();
		}
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            alpha_scalar_moments,
		            3,
		            radial_func_count,
		            1.0,
		            sh_gate_edge_linear_adjoints_.data(),
		            radial_func_count,
		            weighted_values,
		            3,
		            0.0,
		            sh_gate_edge_l1_y_linear_adjoints_.data(),
		            3);

		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int column = species_count + i;
			const double* y_adjoints =
				sh_gate_edge_l1_y_linear_adjoints_.data()
				+ static_cast<size_t>(i) * 3;
			for (int a = 0; a < 3; ++a) {
				double component_der = 0.0;
				for (int m = 0; m < 3; ++m)
					component_der +=
						y_adjoints[m] * sh_ders_use[3 * y1_indices[m] + a];
				if (component_der == 0.0)
					continue;
				if (need_forces) {
					forces_cmpnts(center_atom_index, column, a) += component_der;
					forces_cmpnts(atom_index, column, a) -= component_der;
				}
				if (need_stress)
					for (int b = 0; b < 3; ++b)
						stress_cmpnts[column][a][b] -=
							component_der * rvec[b];
			}
		}
	}
}

void MLMTPR::CalcTwoLayerGateScalarDirectionalDerivatives(
	const Neighborhood& nbh,
	const std::vector<Vector3>& direction_weights,
	std::vector<double>& gate_scalar_tangents,
	std::vector<double>* gate_moment_tangents,
	int cache_atom_index,
	const std::vector<double>* second_gate_directional_values)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (static_cast<int>(direction_weights.size()) != nbh.count)
		ERROR("SUS2-SH gate directional derivative size does not match neighborhood size.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	bool loaded_moment_cache = false;
	const int cached_moment_count =
		static_cast<int>(two_layer_gate_required_moment_indices_.size());
	if (cache_atom_index >= 0 && cached_moment_count > 0) {
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (offset + cached_moment_count
		    <= two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int radial_coeff_base = C + 2 * C * C * K_;

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const bool has_direction = direction_weights[j].NormSq() != 0.0;
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			if (has_direction)
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
			else
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

				for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					const double sigma =
						TwoLayerGateScalingSlope(scaling_block, type_central, type_outer);
					const double shift =
						TwoLayerGateScalingShift(scaling_block, type_central, type_outer);
					if (has_direction)
					p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
				else
					p_RadialBasis->RB_CalcValsOnly(r, sigma, shift, basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					if (has_direction)
						rb_ders[eval_block * R + xi] =
							p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					if (has_direction)
						radial_der += coeff * rb_ders[radial_base + xi];
				}
				radial_values[mu] = radial_value;
				if (has_direction)
					radial_derivatives[mu] = radial_der;
			}
		}

			const double inv_r = has_direction ? 1.0 / r : 0.0;
				const double* second_gate_neighbor_signals =
					multi_layer_gate_enabled_
						? SecondLayerGateNeighborSignals(nbh, j)
						: nullptr;
				const double* second_gate_neighbor_tangents = nullptr;
				if (multi_layer_gate_enabled_
				    && second_gate_directional_values != nullptr) {
					const int atom_index = nbh.inds[j];
					if (atom_index >= 0
					    && second_gate_directional_values->size()
					        >= static_cast<size_t>(atom_index + 1)
					           * radial_func_count)
						second_gate_neighbor_tangents =
							second_gate_directional_values->data()
							+ static_cast<size_t>(atom_index) * radial_func_count;
				}
				for (int bi = 0;
				     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
				     ++bi) {
				const int basic_index = two_layer_gate_required_basic_indices_[bi];
				const int mu = basic_mu_cache_[basic_index];
				const double gate_center_type_coeff =
					TwoLayerGateTypeCoeff(type_central, mu);
				const double gate_outer_type_coeff =
					TwoLayerGateTypeCoeff(type_outer, mu);
				const double second_gate_additive =
					multi_layer_gate_enabled_
						? SecondLayerGateAdditiveCoeff(type_outer, mu)
						: 0.0;
					const double type_scale =
						gate_center_type_coeff
						* (gate_outer_type_coeff
							   + (multi_layer_gate_enabled_
							      ? second_gate_additive
							        * (second_gate_neighbor_signals != nullptr
							           ? second_gate_neighbor_signals[mu]
							           : 0.0)
							      : 0.0));
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values_use[mu_index];
			const double y = sh_values_use[sh_index];
				if (!loaded_moment_cache)
					moment_vals[basic_index] += type_scale * radial_value * y;
				if (has_direction) {
					const double radial_der = radial_derivatives_use[mu_index];
					double directional_derivative = 0.0;
				for (int a = 0; a < 3; ++a) {
					const double dy = sh_ders_use[sh_der_index + a];
					directional_derivative += direction_weights[j][a]
						* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
				}
					grad_dloss_dsenders_[basic_index] +=
						type_scale * directional_derivative;
				}
				if (second_gate_neighbor_tangents != nullptr
				    && second_gate_neighbor_tangents[mu] != 0.0) {
					grad_dloss_dsenders_[basic_index] +=
						gate_center_type_coeff * second_gate_additive
						* second_gate_neighbor_tangents[mu]
						* radial_value * y;
				}
			}
		}

	ContractTwoLayerGateKRankValues(moment_vals);
	ContractTwoLayerGateKRankValues(grad_dloss_dsenders_.data());
	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		if (!loaded_moment_cache)
			moment_vals[product.target] += product.coeff * left_value * right_value;
		grad_dloss_dsenders_[product.target] += product.coeff * (
			grad_dloss_dsenders_[product.left] * right_value
			+ left_value * grad_dloss_dsenders_[product.right]);
	}

	if (gate_moment_tangents != nullptr)
		gate_moment_tangents->assign(
			grad_dloss_dsenders_.begin(), grad_dloss_dsenders_.end());

	gate_scalar_tangents.assign(TwoLayerGateScalarCount(), 0.0);
	for (int q = 0; q < TwoLayerGateScalarCount(); ++q)
		gate_scalar_tangents[q] =
			grad_dloss_dsenders_[TwoLayerGateScalarMomentIndexUnchecked(q)];
}

void MLMTPR::CalcSecondLayerGateScalarDirectionalDerivatives(
	const Neighborhood& nbh,
	const std::vector<Vector3>& direction_weights,
	std::vector<double>& gate_scalar_tangents,
	std::vector<double>* gate_moment_tangents)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (static_cast<int>(direction_weights.size()) != nbh.count)
		ERROR("SUS2-SH second-layer gate directional derivative size does not match neighborhood size.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (second_layer_gate_required_moments_.empty()
	    || second_layer_gate_required_moment_indices_.empty())
		BuildSecondLayerGateProductProgram();

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;

	for (int j = 0; j < nbh.count; ++j) {
		const bool has_direction = direction_weights[j].NormSq() != 0.0;
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		if (has_direction)
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
		else
			EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

		for (int eval_block : second_layer_gate_required_radial_eval_blocks_) {
			const int scaling_block = radial_eval_to_scaling_block_[eval_block];
			const int basis_k = radial_eval_to_basis_k_[eval_block];
			const double sigma =
				regression_coeffs[
					C + 2 * scaling_block * C * C + C * type_central + type_outer];
			const double shift =
				regression_coeffs[
					C + 2 * scaling_block * C * C + C * C
					+ C * type_central + type_outer];
			if (has_direction)
				p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
			else
				p_RadialBasis->RB_CalcValsOnly(r, sigma, shift, basis_k);
			for (int xi = 0; xi < R; ++xi) {
				rb_vals[eval_block * R + xi] =
					p_RadialBasis->rb_vals[xi] * scaling;
				if (has_direction)
					rb_ders[eval_block * R + xi] =
						p_RadialBasis->rb_ders[xi] * scaling;
			}
		}
		for (int mu : second_layer_gate_required_mu_indices_) {
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int radial_offset = SecondLayerGateRadialCoeffIndex(mu, 0);
			double radial_value = 0.0;
			double radial_der = 0.0;
			for (int xi = 0; xi < R; ++xi) {
				const double coeff = regression_coeffs[radial_offset + xi];
				radial_value += coeff * rb_vals[radial_base + xi];
				if (has_direction)
					radial_der += coeff * rb_ders[radial_base + xi];
			}
			radial_values[mu] = radial_value;
			if (has_direction)
				radial_derivatives[mu] = radial_der;
		}

		const double inv_r = has_direction ? 1.0 / r : 0.0;
		for (int basic_index : second_layer_gate_required_basic_indices_) {
			const int mu = basic_mu_cache_[basic_index];
			const double type_scale =
				SecondLayerGateTypeCoeff(type_central, mu)
				* SecondLayerGateTypeCoeff(type_outer, mu);
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values[mu];
			const double y = sh_values[sh_index];
			moment_vals[basic_index] += type_scale * radial_value * y;
			if (has_direction) {
				const double radial_der = radial_derivatives[mu];
				double directional_derivative = 0.0;
				for (int a = 0; a < 3; ++a) {
					const double dy = sh_ders[sh_der_index + a];
					directional_derivative += direction_weights[j][a]
						* (radial_der * rvec[a] * inv_r * y
						   + radial_value * dy);
				}
				grad_dloss_dsenders_[basic_index] +=
					type_scale * directional_derivative;
			}
		}
	}

	for (int product_index : second_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		moment_vals[product.target] += product.coeff * left_value * right_value;
		grad_dloss_dsenders_[product.target] += product.coeff * (
			grad_dloss_dsenders_[product.left] * right_value
			+ left_value * grad_dloss_dsenders_[product.right]);
	}

	if (gate_moment_tangents != nullptr)
		gate_moment_tangents->assign(
			grad_dloss_dsenders_.begin(), grad_dloss_dsenders_.end());

	gate_scalar_tangents.assign(SecondLayerGateScalarCount(), 0.0);
	for (int q = 0; q < SecondLayerGateScalarCount(); ++q) {
		const int scalar_index = second_layer_gate_scalar_indices_[q];
		gate_scalar_tangents[q] =
			grad_dloss_dsenders_[alpha_moment_mapping[scalar_index]];
	}
}

void MLMTPR::CalcSHResidualSiteEnergyDers(const Neighborhood& nbh)
{
	if (!TwoLayerResidualEnabled())
		ERROR("CalcSHResidualSiteEnergyDers called for a non-residual model");
	if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != alpha_scalar_moments)
		two_layer_residual_e0_coeffs_.assign(alpha_scalar_moments, 0.0);

	const bool saved_residual = two_layer_residual_enabled_;
	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	const std::vector<double> saved_linear_coeffs = linear_coeffs;
	std::vector<double> saved_e0_coeffs(alpha_scalar_moments, 0.0);
	for (int i = 0; i < alpha_scalar_moments; ++i)
		saved_e0_coeffs[i] = TwoLayerResidualE0Coeff(i);

	// E0: one-cutoff unmodulated SH readout, including the one-body term.
	two_layer_residual_enabled_ = false;
	active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		linear_coeffs[species_count + i] = saved_e0_coeffs[i];
	for (int i = 0; i < species_count; ++i)
		linear_coeffs[i] = 1.0;
	CalcSHSiteEnergyDers(nbh);
	const double e0_energy =
		buff_site_energy_ + saved_linear_coeffs[nbh.my_type] - 1.0;
	const double e0_energy0 = buff_site_energy_0;
	std::vector<Vector3> e0_ders = buff_site_energy_ders_;

	// E1: direct-gated final SH readout. The one-body constant belongs to E0.
	linear_coeffs = saved_linear_coeffs;
	for (int i = 0; i < species_count; ++i)
		linear_coeffs[i] = 1.0;
	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
	CalcSHSiteEnergyDers(nbh);
	const double e1_one_body = regression_coeffs[nbh.my_type] + 1.0;
	buff_site_energy_ += e0_energy - e1_one_body;
	buff_site_energy_0 += e0_energy0;
	for (int j = 0; j < nbh.count; ++j)
		buff_site_energy_ders_[j] += e0_ders[j];

	two_layer_residual_enabled_ = saved_residual;
	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
	linear_coeffs = saved_linear_coeffs;
}

void MLMTPR::CalcSHSiteEnergyDers(const Neighborhood& nbh)
{
	// The legacy implementation below is intentionally the unmodified release3
	// force kernel; active models use the radial-mixing specialization.
	if (TwoLayerGateRadialMixingEnabled()) {
		CalcSHSiteEnergyDersRadialMixing(nbh);
		return;
	}
	two_layer_gate_radial_mixing_active_eval_ = false;
	CalcSHSiteEnergyDersLegacy(nbh);
}

void MLMTPR::CalcSHSiteEnergyDersLegacy(const Neighborhood& nbh)
{
	if (TwoLayerResidualEnabled()) {
		CalcSHResidualSiteEnergyDers(nbh);
		return;
	}
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const bool use_edge_der_cache =
		HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const bool use_site_der_cache =
		!use_edge_der_cache && UseSHSiteDerivativeCache();
	if (use_site_der_cache)
		CalcSHMomentValuesWithSiteDerivativeCache(nbh);
	else
		CalcSHMomentValuesOnly(nbh);
	buff_site_energy_ = 0.0;
	buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	const double center_linear = linear_coeffs[nbh.my_type];

	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0.0);
	buff_site_energy_ += regression_coeffs[nbh.my_type] + center_linear;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		const double coeff = center_linear * moment_coeff;
		const double scalar_value = moment_vals[alpha_moment_mapping[i]];
		buff_site_energy_ += coeff * scalar_value;
		buff_site_energy_0 += coeff * scalar_value;
		max_linear[i] = std::max(max_linear[i], std::abs(linear_coeffs[species_count + i] * scalar_value));
		site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] += moment_coeff;
	}

	if (UseSHProductRows()) {
		BackpropSHProductRows(site_energy_ders_wrt_moments_);
	} else {
		for (int p = static_cast<int>(sh_products_.size()) - 1;
		     p >= OuterKRank1ProductStart(); --p) {
			const SHProduct& product = sh_products_[p];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}
	BackpropOuterKRank1Adjoints(site_energy_ders_wrt_moments_);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	std::vector<double> gate_adjoint_by_mu;
	std::vector<double> center_gate_adjoint_by_mu;
	const double* active_gate_values_data =
		active_two_layer_gate_values_ == nullptr
			? nullptr
			: active_two_layer_gate_values_->data();
	double* active_gate_adjoints_data =
		active_two_layer_gate_adjoints_ == nullptr
			? nullptr
			: active_two_layer_gate_adjoints_->data();
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;
	const bool center_gate_enabled = TwoLayerGateUsesCenterGate();
	const bool has_nonzero_edge_l1_weights =
		HasNonzeroTwoLayerGateEdgeL1Weights();
	const int edge_l1_source_count =
		has_nonzero_edge_l1_weights
			? TwoLayerGateEdgeL1SourceCount()
			: 0;
	const double* edge_l1_weights =
		has_nonzero_edge_l1_weights
			? TwoLayerGateEdgeL1WeightData()
			: nullptr;
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_der_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& !center_gate_enabled
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size
		&& two_layer_edge_gate_additive_cache_.size() == edge_gate_cache_size
		&& two_layer_edge_gate_neighbor_multiplier_cache_.size()
			== edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_scaled_mu_ders_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_ders_buffer_.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_der_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			radial_derivatives_use =
				two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
		} else if (use_site_der_cache) {
			sh_values_use = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			sh_ders_use = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			radial_values_use = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
			radial_derivatives_use = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
	} else {
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

					const int gate_center_atom_index =
						active_two_layer_edge_cache_atom_index_;
					const int gate_neighbor_atom_index = nbh.inds[j];
					const double* gate_additive_by_mu = nullptr;
					const double* center_gate_additive_by_mu = nullptr;
					const double* neighbor_gate_multiplier_by_mu = nullptr;
					const double* gate_type_scale_by_mu = nullptr;
					if (use_edge_pair_gate_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
						gate_additive_by_mu =
							two_layer_edge_gate_additive_cache_.data()
							+ edge * radial_func_count;
						neighbor_gate_multiplier_by_mu =
							two_layer_edge_gate_neighbor_multiplier_cache_.data()
							+ edge * radial_func_count;
						gate_type_scale_by_mu =
							two_layer_edge_gate_type_scale_cache_.data()
							+ edge * radial_func_count;
					} else {
						const double* base_gate_signal_by_mu =
							active_gate_values_data == nullptr
								? nullptr
								: active_gate_values_data
									+ static_cast<size_t>(gate_neighbor_atom_index)
										* radial_func_count;
						int gate_neighbor_atom_index_for_tanh = gate_neighbor_atom_index;
						const double* gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, base_gate_signal_by_mu, sh_values_use,
								gate_neighbor_atom_index_for_tanh);
						const double* center_gate_signal_by_mu =
							(active_gate_values_data == nullptr
							 || gate_center_atom_index < 0)
								? nullptr
								: active_gate_values_data
									+ static_cast<size_t>(gate_center_atom_index)
										* radial_func_count;
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferAdditive | kGateMuBufferTypeScale
							| kGateMuBufferMultiplier);
						gate_additive_by_mu =
							two_layer_gate_additive_mu_buffer_.data();
						center_gate_additive_by_mu =
							two_layer_gate_center_additive_mu_buffer_.data();
						neighbor_gate_multiplier_by_mu =
							two_layer_gate_neighbor_multiplier_mu_buffer_.data();
						gate_type_scale_by_mu =
							two_layer_gate_type_scale_mu_buffer_.data();
					}
					double* gate_scaled_radial_by_mu =
						two_layer_gate_scaled_mu_vals_buffer_.data();
				double* gate_scaled_der_by_mu =
					two_layer_gate_scaled_mu_ders_buffer_.data();
				if (!use_basic_mu_groups) {
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double type_scale = gate_type_scale_by_mu[mu];
						gate_scaled_radial_by_mu[mu] =
							type_scale * radial_values_use[mu];
						gate_scaled_der_by_mu[mu] =
							type_scale * radial_derivatives_use[mu];
					}
				}
				if (static_cast<int>(gate_adjoint_by_mu.size())
				    != radial_func_count)
					gate_adjoint_by_mu.resize(radial_func_count);
				std::fill(gate_adjoint_by_mu.begin(),
				          gate_adjoint_by_mu.end(), 0.0);
				if (center_gate_enabled) {
					if (static_cast<int>(center_gate_adjoint_by_mu.size())
					    != radial_func_count)
						center_gate_adjoint_by_mu.resize(radial_func_count);
					std::fill(center_gate_adjoint_by_mu.begin(),
					          center_gate_adjoint_by_mu.end(), 0.0);
				}

				if (use_basic_mu_groups) {
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double radial_value = radial_values_use[mu];
						const double type_scale = gate_type_scale_by_mu[mu];
						const double scaled_radial_value =
							type_scale * radial_value;
							const double scaled_radial_der =
								type_scale * radial_derivatives_use[mu];
							const double additive_coeff = gate_additive_by_mu[mu];
							const double center_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_central,
								                   mu);
							const double outer_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_outer,
								                   mu);
							double gate_adjoint_mu = 0.0;
						double center_gate_adjoint_mu = 0.0;
						const int begin = basic_mu_offsets_[mu];
						const int end = basic_mu_offsets_[mu + 1];
						for (int cursor = begin; cursor < end; ++cursor) {
							const int i = basic_indices_by_mu_[cursor];
							const double adj =
								center_linear * site_energy_ders_wrt_moments_[i];
							if (adj == 0.0)
								continue;
							const int sh_index = basic_sh_index_cache_[i];
							const int sh_der_index = basic_sh_der_index_cache_[i];
							const double y = sh_values_use[sh_index];
							gate_adjoint_mu +=
								adj * center_type_coeff * additive_coeff
								* radial_value * y;
							if (center_gate_enabled)
								center_gate_adjoint_mu +=
									adj * center_type_coeff * outer_type_coeff
									* neighbor_gate_multiplier_by_mu[mu]
									* center_gate_additive_by_mu[mu]
									* radial_value * y;
							for (int a = 0; a < 3; ++a) {
								const double dy = sh_ders_use[sh_der_index + a];
								buff_site_energy_ders_[j][a] +=
									adj * (
										scaled_radial_der * rvec[a] * inv_r * y
										+ scaled_radial_value * dy);
							}
						}
						gate_adjoint_by_mu[mu] += gate_adjoint_mu;
						if (center_gate_enabled)
							center_gate_adjoint_by_mu[mu] += center_gate_adjoint_mu;
					}
				} else {
					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const double adj =
							center_linear * site_energy_ders_wrt_moments_[i];
							if (adj == 0.0)
								continue;
							const int mu = basic_mu_cache_[i];
							const double center_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_central,
								                   mu);
							const double outer_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_outer,
								                   mu);
							const int sh_index = basic_sh_index_cache_[i];
						const int sh_der_index = basic_sh_der_index_cache_[i];
						const double radial_value = radial_values_use[mu];
						const double scaled_radial_value =
							gate_scaled_radial_by_mu[mu];
						const double scaled_radial_der =
							gate_scaled_der_by_mu[mu];
						const double y = sh_values_use[sh_index];
						const double additive_coeff = gate_additive_by_mu[mu];
						gate_adjoint_by_mu[mu] +=
							adj * center_type_coeff * additive_coeff
							* radial_value * y;
						if (center_gate_enabled)
							center_gate_adjoint_by_mu[mu] +=
								adj * center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* radial_value * y;
						for (int a = 0; a < 3; ++a) {
							const double dy = sh_ders_use[sh_der_index + a];
							buff_site_energy_ders_[j][a] +=
								adj * (
									scaled_radial_der * rvec[a] * inv_r * y
									+ scaled_radial_value * dy);
						}
					}
				}
				double* neighbor_gate_adjoints =
					active_gate_adjoints_data == nullptr
						? nullptr
					: active_gate_adjoints_data
						+ static_cast<size_t>(gate_neighbor_atom_index)
							* radial_func_count;
				double* center_gate_adjoints =
					(active_gate_adjoints_data == nullptr || !center_gate_enabled
				 || gate_center_atom_index < 0)
					? nullptr
					: active_gate_adjoints_data
						+ static_cast<size_t>(gate_center_atom_index)
							* radial_func_count;
				for (int mu = 0; mu < radial_func_count; ++mu) {
					if (neighbor_gate_adjoints != nullptr)
						neighbor_gate_adjoints[mu] += gate_adjoint_by_mu[mu];
					if (center_gate_adjoints != nullptr)
						center_gate_adjoints[mu] += center_gate_adjoint_by_mu[mu];
				}
						if (active_gate_adjoints_data != nullptr
						    && has_nonzero_edge_l1_weights
						    && !two_layer_gate_edge_l1_adjoints_.empty()) {
							const int source_count = edge_l1_source_count;
					const size_t atom_count =
						two_layer_gate_edge_l1_adjoints_.size()
						/ (static_cast<size_t>(source_count) * 3);
					if (gate_neighbor_atom_index < 0
					    || gate_neighbor_atom_index >= static_cast<int>(atom_count))
						ERROR("SUS2-SH edge L1 gate adjoint atom index is out of range");
						const double* edge_l1_values =
							two_layer_gate_edge_l1_values_.data()
							+ static_cast<size_t>(gate_neighbor_atom_index)
								* source_count * 3;
						const size_t expected_weighted_size =
							atom_count * static_cast<size_t>(radial_func_count) * 3;
						const double* weighted_values = nullptr;
						if (two_layer_gate_edge_l1_weighted_values_.size()
						    == expected_weighted_size)
							weighted_values =
								two_layer_gate_edge_l1_weighted_values_.data()
								+ static_cast<size_t>(gate_neighbor_atom_index)
									* radial_func_count * 3;
						const size_t expected_weight_transpose_size =
							static_cast<size_t>(source_count) * radial_func_count;
						const double* weights_by_source =
							two_layer_gate_edge_l1_weights_by_source_.size()
								== expected_weight_transpose_size
								? two_layer_gate_edge_l1_weights_by_source_.data()
								: nullptr;
						double* edge_l1_adjoints =
							two_layer_gate_edge_l1_adjoints_.data()
							+ static_cast<size_t>(gate_neighbor_atom_index)
								* source_count * 3;
						const int y1_indices[3] = {
							SHFlatIndex(1, -1),
							SHFlatIndex(1, 0),
							SHFlatIndex(1, 1)
						};
						double y_adj[3] = {0.0, 0.0, 0.0};
						if (weighted_values != nullptr) {
							for (int mu = 0; mu < radial_func_count; ++mu) {
								const double adj = gate_adjoint_by_mu[mu];
								if (adj == 0.0)
									continue;
								const double* u = weighted_values + mu * 3;
								for (int m = 0; m < 3; ++m)
									y_adj[m] += adj * u[m];
							}
						}
						for (int source = 0; source < source_count; ++source) {
							double chi_adj = 0.0;
							if (weights_by_source != nullptr) {
								const double* weight_col =
									weights_by_source
									+ static_cast<size_t>(source) * radial_func_count;
								for (int mu = 0; mu < radial_func_count; ++mu) {
									const double adj = gate_adjoint_by_mu[mu];
									if (adj != 0.0)
										chi_adj += adj * weight_col[mu];
								}
							} else {
								for (int mu = 0; mu < radial_func_count; ++mu) {
									const double adj = gate_adjoint_by_mu[mu];
									if (adj != 0.0)
										chi_adj +=
											adj
											* edge_l1_weights[
												static_cast<size_t>(mu) * source_count + source];
								}
							}
						if (chi_adj == 0.0)
							continue;
						const double* h = edge_l1_values + source * 3;
						double* h_adj = edge_l1_adjoints + source * 3;
						for (int m = 0; m < 3; ++m) {
							const int y_index = y1_indices[m];
								const int dy_index = 3 * y_index;
								h_adj[m] += chi_adj * sh_values_use[y_index];
								if (weighted_values == nullptr)
									for (int a = 0; a < 3; ++a)
										buff_site_energy_ders_[j][a] +=
											chi_adj * h[m] * sh_ders_use[dy_index + a];
							}
						}
						if (weighted_values != nullptr) {
							for (int m = 0; m < 3; ++m) {
								const int y_index = y1_indices[m];
								const int dy_index = 3 * y_index;
								for (int a = 0; a < 3; ++a)
									buff_site_energy_ders_[j][a] +=
										y_adj[m] * sh_ders_use[dy_index + a];
							}
						}
					}
				}

	if (TwoLayerGateReadoutEnabled())
		AddTwoLayerGateReadoutSiteEnergyDers(nbh);
}


void MLMTPR::CalcSHSiteEnergyDersRadialMixing(const Neighborhood& nbh)
{
	const bool pure_matrix = OuterUsesPureRadialMatrix();
	if (TwoLayerResidualEnabled()) {
		CalcSHResidualSiteEnergyDers(nbh);
		return;
	}
	two_layer_gate_radial_mixing_active_eval_ =
		TwoLayerGateRadialMixingEnabled()
		&& (OuterUsesPureRadialMatrix() || HasNonzeroTwoLayerGateRadialMixingCoeffs());
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (TwoLayerGateRadialMixingEnabled() && use_lmp_table)
		ERROR("SUS2-SH two-layer gate radial mixing requires direct radial basis evaluation.");
	const bool radial_mixing_active =
		two_layer_gate_radial_mixing_active_eval_;
	// With C == 0, all current-C force chains vanish exactly. Direct
	// derivatives with respect to C are retained in the parameter path.
	const bool use_edge_der_cache =
		HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const size_t radial_mixing_kernel_stride =
		static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_
		* sh_k_max_;
	const bool use_radial_mixing_kernel_cache =
		radial_mixing_active && use_edge_der_cache;
	const bool use_site_der_cache =
		!use_edge_der_cache && !radial_mixing_active
		&& UseSHSiteDerivativeCache();
	if (use_site_der_cache)
		CalcSHMomentValuesWithSiteDerivativeCache(nbh);
	else
		CalcSHMomentValuesOnly(nbh);
	buff_site_energy_ = 0.0;
	buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	const double center_linear = linear_coeffs[nbh.my_type];

	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0.0);
	buff_site_energy_ += regression_coeffs[nbh.my_type] + center_linear;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		const double coeff = center_linear * moment_coeff;
		const double scalar_value = moment_vals[alpha_moment_mapping[i]];
		buff_site_energy_ += coeff * scalar_value;
		buff_site_energy_0 += coeff * scalar_value;
		max_linear[i] = std::max(max_linear[i], std::abs(linear_coeffs[species_count + i] * scalar_value));
		site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] += moment_coeff;
	}

	if (ApplyPureWGraph(true)) {
		// Adjoint operations retain their order within every output lane.
	} else if (UseSHProductRows()) {
		BackpropSHProductRows(site_energy_ders_wrt_moments_);
	} else {
		for (int p = static_cast<int>(sh_products_.size()) - 1;
		     p >= OuterKRank1ProductStart(); --p) {
			const SHProduct& product = sh_products_[p];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}
	BackpropOuterKRank1Adjoints(site_energy_ders_wrt_moments_);
	if (pure_w_execution_.basis && HasTwoLayerEdgePrimitiveCache(-1, true, false)) {
		CalcPureWBasisFirstForce(nbh, center_linear);
		return;
	}
	if (pure_w_execution_.edge_force && HasTwoLayerEdgePrimitiveCache(-1, true, false)) {
		CalcPureWEdgeFirstForce(nbh, center_linear);
		return;
	}

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	std::vector<double> gate_adjoint_by_mu;
	std::vector<double> center_gate_adjoint_by_mu;
	std::vector<double>& radial_mixing_input =
		sh_site_energy_radial_mixing_input_scratch_;
	std::vector<double>& radial_mixing_kernels =
		sh_site_energy_radial_mixing_kernel_scratch_;
	std::vector<double>& radial_mixing_kernel_ders =
		sh_site_energy_radial_mixing_kernel_der_scratch_;
	std::vector<double>& radial_mixing_outer =
		sh_site_energy_radial_mixing_outer_scratch_;
	std::vector<double>& radial_mixing_outer_ders =
		sh_site_energy_radial_mixing_outer_der_scratch_;
	std::vector<double>& radial_mixing_sum_adj_y =
		sh_site_energy_radial_mixing_sum_adj_y_scratch_;
	const double* radial_mixing_coeffs = nullptr;
	if (radial_mixing_active) {
		radial_mixing_input.resize(sh_k_max_);
		radial_mixing_kernels.resize(
			outer_k_rank_ * sh_k_max_);
		radial_mixing_kernel_ders.resize(
			outer_k_rank_ * sh_k_max_);
		radial_mixing_outer.resize(outer_k_rank_);
		radial_mixing_outer_ders.resize(3 * outer_k_rank_);
		radial_mixing_sum_adj_y.resize(outer_k_rank_);
		radial_mixing_coeffs =
			regression_coeffs.data() + TwoLayerGateRadialMixingCoeffOffset();
	}
	const double* active_gate_values_data =
		active_two_layer_gate_values_ == nullptr
			? nullptr
			: active_two_layer_gate_values_->data();
	double* active_gate_adjoints_data =
		active_two_layer_gate_adjoints_ == nullptr
			? nullptr
			: active_two_layer_gate_adjoints_->data();
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;
	const bool center_gate_enabled = TwoLayerGateUsesCenterGate();
	const bool has_nonzero_edge_l1_weights =
		HasNonzeroTwoLayerGateEdgeL1Weights();
	const int edge_l1_source_count =
		has_nonzero_edge_l1_weights
			? TwoLayerGateEdgeL1SourceCount()
			: 0;
	const double* edge_l1_weights =
		has_nonzero_edge_l1_weights
			? TwoLayerGateEdgeL1WeightData()
			: nullptr;
	const size_t edge_gate_cache_size =
		two_layer_edge_offsets_cache_.empty()
			? 0
			: two_layer_edge_offsets_cache_.back()
				* static_cast<size_t>(radial_func_count);
	const bool use_edge_pair_gate_cache =
		use_edge_der_cache && active_two_layer_gate_values_ == &two_layer_gate_values_
		&& !center_gate_enabled
		&& two_layer_edge_gate_type_scale_cache_.size() == edge_gate_cache_size
		&& two_layer_edge_gate_additive_cache_.size() == edge_gate_cache_size
		&& two_layer_edge_gate_neighbor_multiplier_cache_.size()
			== edge_gate_cache_size;
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_scaled_mu_ders_buffer_.size())
	    != radial_func_count)
		two_layer_gate_scaled_mu_ders_buffer_.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		const double* radial_mixing_rb_vals = rb_vals.data();
		const double* radial_mixing_rb_ders = rb_ders.data();
		const double* cached_radial_mixing_kernels = nullptr;
		const double* cached_radial_mixing_kernel_ders = nullptr;
		int radial_mixing_der_stride = 1;
		if (use_edge_der_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			if (use_radial_mixing_kernel_cache) {
				cached_radial_mixing_kernels =
					two_layer_edge_radial_mixing_kernel_cache_.data()
					+ edge * radial_mixing_kernel_stride;
				cached_radial_mixing_kernel_ders =
					two_layer_edge_radial_mixing_kernel_der_cache_.data()
					+ edge * radial_mixing_kernel_stride;
			}
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			radial_derivatives_use =
				two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
			// Pure W forces consume the cached W and W' directly. In the
			// prediction-only cache there may be no raw parameter derivatives;
			// rebuilding that unused radial basis would duplicate edge setup.
			if (radial_mixing_active && (!pure_matrix || !use_radial_mixing_kernel_cache)) {
				const size_t radial_val_stride =
					static_cast<size_t>(two_layer_edge_cache_eval_block_count_) * R;
				const size_t radial_der_stride = 5 * radial_val_stride;
				const bool use_cached_raw_radial =
					two_layer_edge_cache_has_param_derivatives_
					&& two_layer_edge_radial_vals_cache_.size()
						>= (edge + 1) * radial_val_stride
					&& two_layer_edge_radial_ders_cache_.size()
						>= (edge + 1) * radial_der_stride;
				if (use_cached_raw_radial) {
					radial_mixing_rb_vals =
						two_layer_edge_radial_vals_cache_.data()
						+ edge * radial_val_stride;
					radial_mixing_rb_ders =
						two_layer_edge_radial_ders_cache_.data()
						+ edge * radial_der_stride;
					radial_mixing_der_stride = 5;
				} else {
					for (int eval_block = 0;
					     eval_block < static_cast<int>(radial_eval_to_scaling_block_.size());
					     ++eval_block) {
						const int scaling_block =
							radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_Calc(
							r,
							regression_coeffs[
								C + 2 * scaling_block * C * C
								+ C * type_central + type_outer],
							regression_coeffs[
								C + 2 * scaling_block * C * C + C * C
								+ C * type_central + type_outer],
							basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
							rb_ders[eval_block * R + xi] =
								p_RadialBasis->rb_ders[xi] * scaling;
						}
					}
				}
			}
		} else if (use_site_der_cache) {
			sh_values_use = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			sh_ders_use = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			radial_values_use = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
			radial_derivatives_use = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
	} else {
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

					const int gate_center_atom_index =
						active_two_layer_edge_cache_atom_index_;
					const int gate_neighbor_atom_index = nbh.inds[j];
					const double* gate_additive_by_mu = nullptr;
					const double* center_gate_additive_by_mu = nullptr;
					const double* neighbor_gate_multiplier_by_mu = nullptr;
					const double* gate_type_scale_by_mu = nullptr;
					const double* gate_signal_by_mu = nullptr;
					if (use_edge_pair_gate_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
						gate_additive_by_mu =
							two_layer_edge_gate_additive_cache_.data()
							+ edge * radial_func_count;
						neighbor_gate_multiplier_by_mu =
							two_layer_edge_gate_neighbor_multiplier_cache_.data()
							+ edge * radial_func_count;
						gate_type_scale_by_mu =
							two_layer_edge_gate_type_scale_cache_.data()
							+ edge * radial_func_count;
					} else {
						const double* base_gate_signal_by_mu =
							active_gate_values_data == nullptr
								? nullptr
								: active_gate_values_data
									+ static_cast<size_t>(gate_neighbor_atom_index)
										* radial_func_count;
						int gate_neighbor_atom_index_for_tanh = gate_neighbor_atom_index;
						gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, base_gate_signal_by_mu, sh_values_use,
								gate_neighbor_atom_index_for_tanh);
						const double* center_gate_signal_by_mu =
							(active_gate_values_data == nullptr
							 || gate_center_atom_index < 0)
								? nullptr
								: active_gate_values_data
									+ static_cast<size_t>(gate_center_atom_index)
										* radial_func_count;
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferAdditive | kGateMuBufferTypeScale
							| kGateMuBufferMultiplier);
						gate_additive_by_mu =
							two_layer_gate_additive_mu_buffer_.data();
						center_gate_additive_by_mu =
							two_layer_gate_center_additive_mu_buffer_.data();
						neighbor_gate_multiplier_by_mu =
							two_layer_gate_neighbor_multiplier_mu_buffer_.data();
						gate_type_scale_by_mu =
							two_layer_gate_type_scale_mu_buffer_.data();
					}
					double* gate_scaled_radial_by_mu =
						two_layer_gate_scaled_mu_vals_buffer_.data();
				double* gate_scaled_der_by_mu =
					two_layer_gate_scaled_mu_ders_buffer_.data();
				if (!pure_matrix && !use_basic_mu_groups) {
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double type_scale = gate_type_scale_by_mu[mu];
						gate_scaled_radial_by_mu[mu] =
							type_scale * radial_values_use[mu];
						gate_scaled_der_by_mu[mu] =
							type_scale * radial_derivatives_use[mu];
					}
				}
				if (static_cast<int>(gate_adjoint_by_mu.size())
				    != radial_func_count)
					gate_adjoint_by_mu.resize(radial_func_count);
				std::fill(gate_adjoint_by_mu.begin(),
				          gate_adjoint_by_mu.end(), 0.0);
				if (center_gate_enabled) {
					if (static_cast<int>(center_gate_adjoint_by_mu.size())
					    != radial_func_count)
						center_gate_adjoint_by_mu.resize(radial_func_count);
					std::fill(center_gate_adjoint_by_mu.begin(),
					          center_gate_adjoint_by_mu.end(), 0.0);
				}

	if (radial_mixing_active) {
		for (int l = 0; l <= sh_l_max_; ++l) {
							double* gate_input = radial_mixing_input.data();
								std::fill(gate_input,
								          gate_input + sh_k_max_, 0.0);
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									gate_input[k] = gate_type_scale_by_mu[mu];
							}
							for (int a = 0; a < outer_k_rank_; ++a) {
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const int mixing_radial_base =
										mu_to_radial_eval_block_[mu] * R;
									const int ac =
										a * sh_k_max_ + k;
									const double* coeff_row = radial_mixing_coeffs
										+ (((l * outer_k_rank_ + a)
										    * sh_k_max_ + k) * R);
									double kernel;
									double kernel_der;
									if (cached_radial_mixing_kernels != nullptr) {
										const size_t lac =
											(static_cast<size_t>(l) * outer_k_rank_ + a)
											* sh_k_max_ + k;
										kernel = cached_radial_mixing_kernels[lac];
										kernel_der = cached_radial_mixing_kernel_ders[lac];
									} else {
										kernel = 0.0;
										kernel_der = 0.0;
										for (int xi = 0; xi < R; ++xi) {
											kernel += coeff_row[xi]
												* radial_mixing_rb_vals[mixing_radial_base + xi];
											kernel_der += coeff_row[xi]
												* radial_mixing_rb_ders[
													radial_mixing_der_stride
													* mixing_radial_base + xi];
										}
									}
									radial_mixing_kernels[ac] = kernel;
									radial_mixing_kernel_ders[ac] = kernel_der;
								}
							}
							// M_l(r) has no m index, so contract c once per (l,a)
							// and reuse the exact result for every spherical component.
								for (int a = 0; a < outer_k_rank_; ++a) {
									double outer_value = 0.0;
									double outer_der[3] = {0.0, 0.0, 0.0};
									for (int k = 0; k < sh_k_max_; ++k) {
										const int ac = a * sh_k_max_ + k;
										const double kernel = radial_mixing_kernels[ac];
										const double kernel_der = radial_mixing_kernel_ders[ac];
										outer_value += kernel * gate_input[k];
											for (int q = 0; q < 3; ++q)
												outer_der[q] +=
												kernel_der * rvec[q] * inv_r * gate_input[k];
									}
									radial_mixing_outer[a] = outer_value;
									for (int q = 0; q < 3; ++q)
										radial_mixing_outer_ders[3 * a + q] = outer_der[q];
								}
							std::fill(radial_mixing_sum_adj_y.begin(),
							          radial_mixing_sum_adj_y.end(), 0.0);
							for (int m = -l; m <= l; ++m) {
								const int sh_index = SHFlatIndex(l, m);
								const int sh_der_index = 3 * sh_index;
								const double y = sh_values_use[sh_index];
								for (int a = 0; a < outer_k_rank_; ++a) {
									const int outer =
										OuterKRank1OuterBasicIndex(a, l, m);
									const double adj =
										center_linear
										* site_energy_ders_wrt_moments_[outer];
									radial_mixing_sum_adj_y[a] += adj * y;
									if (adj == 0.0)
										continue;
									for (int q = 0; q < 3; ++q)
										buff_site_energy_ders_[j][q] +=
											adj * (
												radial_mixing_outer_ders[3 * a + q] * y
												+ radial_mixing_outer[a]
													* sh_ders_use[sh_der_index + q]);
								}
							}
							// Likewise contract the moment adjoints over m before
							// propagating back to the raw k channels.
							for (int a = 0; a < outer_k_rank_; ++a) {
								const double sum_adj_y = radial_mixing_sum_adj_y[a];
								if (sum_adj_y == 0.0)
									continue;
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const double center_type_coeff =
										OuterTypeCoeffFast(regression_coeffs,
										                   outer_type_coeff_offset,
										                   outer_type_coeff_stride,
										                   R,
										                   type_central,
										                   mu);
									gate_adjoint_by_mu[mu] +=
										sum_adj_y
										* radial_mixing_kernels[a * sh_k_max_ + k]
										* center_type_coeff * gate_additive_by_mu[mu];
								}
							}
						}
					}
					if (use_basic_mu_groups) {
					for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
						const double radial_value = radial_values_use[mu];
						const double type_scale = gate_type_scale_by_mu[mu];
						const double scaled_radial_value =
							type_scale * radial_value;
							const double scaled_radial_der =
								type_scale * radial_derivatives_use[mu];
							const double additive_coeff = gate_additive_by_mu[mu];
							const double center_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_central,
								                   mu);
							const double outer_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_outer,
								                   mu);
							double gate_adjoint_mu = 0.0;
						double center_gate_adjoint_mu = 0.0;
						const int begin = basic_mu_offsets_[mu];
						const int end = basic_mu_offsets_[mu + 1];
						for (int cursor = begin; cursor < end; ++cursor) {
							const int i = basic_indices_by_mu_[cursor];
							const double adj =
								center_linear * site_energy_ders_wrt_moments_[i];
							if (adj == 0.0)
								continue;
							const int sh_index = basic_sh_index_cache_[i];
							const int sh_der_index = basic_sh_der_index_cache_[i];
							const double y = sh_values_use[sh_index];
							gate_adjoint_mu +=
								adj * center_type_coeff * additive_coeff
								* radial_value * y;
							if (center_gate_enabled)
								center_gate_adjoint_mu +=
									adj * center_type_coeff * outer_type_coeff
									* neighbor_gate_multiplier_by_mu[mu]
									* center_gate_additive_by_mu[mu]
									* radial_value * y;
							for (int a = 0; a < 3; ++a) {
								const double dy = sh_ders_use[sh_der_index + a];
								buff_site_energy_ders_[j][a] +=
									adj * (
										scaled_radial_der * rvec[a] * inv_r * y
										+ scaled_radial_value * dy);
							}
						}
						gate_adjoint_by_mu[mu] += gate_adjoint_mu;
						if (center_gate_enabled)
							center_gate_adjoint_by_mu[mu] += center_gate_adjoint_mu;
					}
				} else {
					for (int i = 0; !pure_matrix && i < alpha_index_basic_count; ++i) {
						const double adj =
							center_linear * site_energy_ders_wrt_moments_[i];
							if (adj == 0.0)
								continue;
							const int mu = basic_mu_cache_[i];
							const double center_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_central,
								                   mu);
							const double outer_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_outer,
								                   mu);
							const int sh_index = basic_sh_index_cache_[i];
						const int sh_der_index = basic_sh_der_index_cache_[i];
						const double radial_value = radial_values_use[mu];
						const double scaled_radial_value =
							gate_scaled_radial_by_mu[mu];
						const double scaled_radial_der =
							gate_scaled_der_by_mu[mu];
						const double y = sh_values_use[sh_index];
						const double additive_coeff = gate_additive_by_mu[mu];
						gate_adjoint_by_mu[mu] +=
							adj * center_type_coeff * additive_coeff
							* radial_value * y;
						if (center_gate_enabled)
							center_gate_adjoint_by_mu[mu] +=
								adj * center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* radial_value * y;
						for (int a = 0; a < 3; ++a) {
							const double dy = sh_ders_use[sh_der_index + a];
							buff_site_energy_ders_[j][a] +=
								adj * (
									scaled_radial_der * rvec[a] * inv_r * y
									+ scaled_radial_value * dy);
						}
					}
				}
				double* neighbor_gate_adjoints =
					active_gate_adjoints_data == nullptr
						? nullptr
					: active_gate_adjoints_data
						+ static_cast<size_t>(gate_neighbor_atom_index)
							* radial_func_count;
				double* center_gate_adjoints =
					(active_gate_adjoints_data == nullptr || !center_gate_enabled
				 || gate_center_atom_index < 0)
					? nullptr
					: active_gate_adjoints_data
						+ static_cast<size_t>(gate_center_atom_index)
							* radial_func_count;
				for (int mu = 0; mu < radial_func_count; ++mu) {
					if (neighbor_gate_adjoints != nullptr)
						neighbor_gate_adjoints[mu] += gate_adjoint_by_mu[mu];
					if (center_gate_adjoints != nullptr)
						center_gate_adjoints[mu] += center_gate_adjoint_by_mu[mu];
				}
						if (active_gate_adjoints_data != nullptr
						    && has_nonzero_edge_l1_weights
						    && !two_layer_gate_edge_l1_adjoints_.empty()) {
							const int source_count = edge_l1_source_count;
					const size_t atom_count =
						two_layer_gate_edge_l1_adjoints_.size()
						/ (static_cast<size_t>(source_count) * 3);
					if (gate_neighbor_atom_index < 0
					    || gate_neighbor_atom_index >= static_cast<int>(atom_count))
						ERROR("SUS2-SH edge L1 gate adjoint atom index is out of range");
						const double* edge_l1_values =
							two_layer_gate_edge_l1_values_.data()
							+ static_cast<size_t>(gate_neighbor_atom_index)
								* source_count * 3;
						const size_t expected_weighted_size =
							atom_count * static_cast<size_t>(radial_func_count) * 3;
						const double* weighted_values = nullptr;
						if (two_layer_gate_edge_l1_weighted_values_.size()
						    == expected_weighted_size)
							weighted_values =
								two_layer_gate_edge_l1_weighted_values_.data()
								+ static_cast<size_t>(gate_neighbor_atom_index)
									* radial_func_count * 3;
						const size_t expected_weight_transpose_size =
							static_cast<size_t>(source_count) * radial_func_count;
						const double* weights_by_source =
							two_layer_gate_edge_l1_weights_by_source_.size()
								== expected_weight_transpose_size
								? two_layer_gate_edge_l1_weights_by_source_.data()
								: nullptr;
						double* edge_l1_adjoints =
							two_layer_gate_edge_l1_adjoints_.data()
							+ static_cast<size_t>(gate_neighbor_atom_index)
								* source_count * 3;
						const int y1_indices[3] = {
							SHFlatIndex(1, -1),
							SHFlatIndex(1, 0),
							SHFlatIndex(1, 1)
						};
						double y_adj[3] = {0.0, 0.0, 0.0};
						if (weighted_values != nullptr) {
							for (int mu = 0; mu < radial_func_count; ++mu) {
								const double adj = gate_adjoint_by_mu[mu];
								if (adj == 0.0)
									continue;
								const double* u = weighted_values + mu * 3;
								for (int m = 0; m < 3; ++m)
									y_adj[m] += adj * u[m];
							}
						}
						for (int source = 0; source < source_count; ++source) {
							double chi_adj = 0.0;
							if (weights_by_source != nullptr) {
								const double* weight_col =
									weights_by_source
									+ static_cast<size_t>(source) * radial_func_count;
								for (int mu = 0; mu < radial_func_count; ++mu) {
									const double adj = gate_adjoint_by_mu[mu];
									if (adj != 0.0)
										chi_adj += adj * weight_col[mu];
								}
							} else {
								for (int mu = 0; mu < radial_func_count; ++mu) {
									const double adj = gate_adjoint_by_mu[mu];
									if (adj != 0.0)
										chi_adj +=
											adj
											* edge_l1_weights[
												static_cast<size_t>(mu) * source_count + source];
								}
							}
						if (chi_adj == 0.0)
							continue;
						const double* h = edge_l1_values + source * 3;
						double* h_adj = edge_l1_adjoints + source * 3;
						for (int m = 0; m < 3; ++m) {
							const int y_index = y1_indices[m];
								const int dy_index = 3 * y_index;
								h_adj[m] += chi_adj * sh_values_use[y_index];
								if (weighted_values == nullptr)
									for (int a = 0; a < 3; ++a)
										buff_site_energy_ders_[j][a] +=
											chi_adj * h[m] * sh_ders_use[dy_index + a];
							}
						}
						if (weighted_values != nullptr) {
							for (int m = 0; m < 3; ++m) {
								const int y_index = y1_indices[m];
								const int dy_index = 3 * y_index;
								for (int a = 0; a < 3; ++a)
									buff_site_energy_ders_[j][a] +=
										y_adj[m] * sh_ders_use[dy_index + a];
							}
						}
					}
				}

	if (TwoLayerGateReadoutEnabled())
		AddTwoLayerGateReadoutSiteEnergyDers(nbh);
}

void MLMTPR::AddTwoLayerGateReadoutSiteEnergyDers(const Neighborhood& nbh)
{
	if (!TwoLayerGateReadoutEnabled())
		return;
	const int gate_count = TwoLayerGateReadoutCoeffCount();
	if (gate_count <= 0 || gate_count != TwoLayerGateScalarCount())
		ERROR("SUS2-SH gate readout has inconsistent scalar dimensions");

	const int cache_atom_index = active_two_layer_edge_cache_atom_index_;
	const size_t cache_offset =
		cache_atom_index >= 0
			? static_cast<size_t>(cache_atom_index) * gate_count
			: 0;
	const double* gate_scalars = nullptr;
	if (cache_atom_index >= 0
	    && cache_offset + gate_count <= two_layer_gate_scalar_values_cache_.size()) {
		gate_scalars = two_layer_gate_scalar_values_cache_.data() + cache_offset;
	} else {
		std::vector<double> saved_moment_vals(
			moment_vals, moment_vals + alpha_moments_count);
		CalcTwoLayerGateScalarValuesOnly(
			nbh, sh_gate_scalar_values_, cache_atom_index);
		std::copy(
			saved_moment_vals.begin(), saved_moment_vals.end(), moment_vals);
		gate_scalars = sh_gate_scalar_values_.data();
	}

	double readout_value = 0.0;
	for (int q = 0; q < gate_count; ++q)
		readout_value += TwoLayerGateReadoutCoeff(q) * gate_scalars[q];
	const double center_linear = linear_coeffs[nbh.my_type];
	buff_site_energy_ += center_linear * readout_value;
	buff_site_energy_0 += center_linear * readout_value;
}

void MLMTPR::AccumulateTwoLayerGateReadoutCombinationGrad(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	double se_weight,
	const Vector3* se_ders_weights)
{
	if (!TwoLayerGateReadoutEnabled())
		return;
	const int gate_count = TwoLayerGateReadoutCoeffCount();
	if (gate_count <= 0 || gate_count != TwoLayerGateScalarCount())
		ERROR("SUS2-SH gate readout has inconsistent scalar dimensions");
	out_grad_accumulator.resize(CoeffCount());

	const int cache_atom_index = active_two_layer_edge_cache_atom_index_;
	std::vector<double>& gate_tangents =
		two_layer_gate_readout_tangent_scratch_;
	std::vector<double>& gate_moment_tangents =
		two_layer_gate_readout_moment_tangent_scratch_;
	gate_tangents.assign(gate_count, 0.0);
	gate_moment_tangents.clear();
	const double* gate_scalars = nullptr;
	if (se_ders_weights != nullptr) {
		std::vector<Vector3>& direction_weights =
			two_layer_gate_readout_direction_scratch_;
		direction_weights.assign(
			se_ders_weights, se_ders_weights + nbh.count);
		CalcTwoLayerGateScalarDirectionalDerivatives(
			nbh,
			direction_weights,
			gate_tangents,
			&gate_moment_tangents,
			cache_atom_index);
		sh_gate_scalar_values_.resize(gate_count);
		for (int q = 0; q < gate_count; ++q)
			sh_gate_scalar_values_[q] =
				moment_vals[TwoLayerGateScalarMomentIndexUnchecked(q)];
		gate_scalars = sh_gate_scalar_values_.data();
	} else {
		const size_t cache_offset =
			cache_atom_index >= 0
				? static_cast<size_t>(cache_atom_index) * gate_count
				: 0;
		if (cache_atom_index >= 0
		    && cache_offset + gate_count <= two_layer_gate_scalar_values_cache_.size()) {
			gate_scalars = two_layer_gate_scalar_values_cache_.data() + cache_offset;
		} else {
			CalcTwoLayerGateScalarValuesOnly(
				nbh, sh_gate_scalar_values_, cache_atom_index);
			gate_scalars = sh_gate_scalar_values_.data();
		}
	}

	const double center_linear = linear_coeffs[nbh.my_type];
	double readout_value = 0.0;
	double readout_tangent = 0.0;
	std::vector<double>& gate_scalar_seeds =
		two_layer_gate_readout_value_seed_scratch_;
	std::vector<double>& energy_scalar_seeds =
		two_layer_gate_readout_energy_seed_scratch_;
	gate_scalar_seeds.assign(gate_count, 0.0);
	energy_scalar_seeds.assign(gate_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double phi = TwoLayerGateReadoutCoeff(q);
		readout_value += phi * gate_scalars[q];
		readout_tangent += phi * gate_tangents[q];
		out_grad_accumulator[TwoLayerGateReadoutCoeffOffset() + q] +=
			center_linear
			* (se_weight * gate_scalars[q] + gate_tangents[q]);
		gate_scalar_seeds[q] = center_linear * se_weight * phi;
		energy_scalar_seeds[q] = center_linear * phi;
	}
	out_grad_accumulator[LinearCoeffOffset() + nbh.my_type] +=
		se_weight * readout_value + readout_tangent;

	AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
		nbh,
		out_grad_accumulator,
		gate_scalar_seeds.data(),
		se_ders_weights,
		cache_atom_index,
		gate_moment_tangents.empty() ? nullptr : gate_moment_tangents.data(),
		energy_scalar_seeds.data());
}

void MLMTPR::AccumulateTwoLayerGateReadoutDirectionalGrad(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const Vector3* se_ders_weights,
	const double* gate_scalar_tangents,
	const double* gate_moment_tangents,
	int cache_atom_index)
{
	if (!TwoLayerGateReadoutEnabled())
		return;
	const int gate_count = TwoLayerGateReadoutCoeffCount();
	if (gate_count <= 0 || gate_count != TwoLayerGateScalarCount())
		ERROR("SUS2-SH gate readout has inconsistent scalar dimensions");
	if (se_ders_weights == nullptr
	    || gate_scalar_tangents == nullptr
	    || gate_moment_tangents == nullptr)
		ERROR("SUS2-SH gate readout directional gradient requires tangent data");
	out_grad_accumulator.resize(CoeffCount());

	const double center_linear = linear_coeffs[nbh.my_type];
	double readout_tangent = 0.0;
	std::vector<double>& gate_scalar_seeds =
		two_layer_gate_readout_value_seed_scratch_;
	std::vector<double>& energy_scalar_seeds =
		two_layer_gate_readout_energy_seed_scratch_;
	gate_scalar_seeds.assign(gate_count, 0.0);
	energy_scalar_seeds.assign(gate_count, 0.0);
	for (int q = 0; q < gate_count; ++q) {
		const double phi = TwoLayerGateReadoutCoeff(q);
		const double tangent = gate_scalar_tangents[q];
		readout_tangent += phi * tangent;
		out_grad_accumulator[TwoLayerGateReadoutCoeffOffset() + q] +=
			center_linear * tangent;
		energy_scalar_seeds[q] = center_linear * phi;
	}
	out_grad_accumulator[LinearCoeffOffset() + nbh.my_type] +=
		readout_tangent;

	AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
		nbh,
		out_grad_accumulator,
		gate_scalar_seeds.data(),
		se_ders_weights,
		cache_atom_index,
		gate_moment_tangents,
		energy_scalar_seeds.data());
}

void MLMTPR::AccumulateSHGateTangentGrad(const Neighborhood& nbh,
                                         std::vector<double>& out_grad_accumulator,
                                         const std::vector<double>& neighbor_gate_tangent,
                                         const double* center_gate_tangent,
                                         int cache_atom_index)
{
	// The template argument makes every radial-mixing branch compile-time
	// constant while retaining one source of truth for the gradient algebra.
	if (TwoLayerGateRadialMixingEnabled())
		AccumulateSHGateTangentGradImpl<true>(
			nbh, out_grad_accumulator, neighbor_gate_tangent,
			center_gate_tangent, cache_atom_index);
	else
		AccumulateSHGateTangentGradImpl<false>(
			nbh, out_grad_accumulator, neighbor_gate_tangent,
			center_gate_tangent, cache_atom_index);
}

template <bool RadialMixingEnabled>
void MLMTPR::AccumulateSHGateTangentGradImpl(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const std::vector<double>& neighbor_gate_tangent,
	const double* center_gate_tangent,
	int cache_atom_index)
{
	const bool pure_matrix = RadialMixingEnabled && OuterUsesPureRadialMatrix();
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	const size_t expected_tangent_size =
		static_cast<size_t>(nbh.count) * radial_func_count;
	if (neighbor_gate_tangent.size() != expected_tangent_size)
		ERROR("SUS2-SH gate tangent size does not match neighborhood size.");

		bool has_tangent = false;
		for (double value : neighbor_gate_tangent)
			if (value != 0.0) {
				has_tangent = true;
				break;
			}
		if (!has_tangent && center_gate_tangent != nullptr)
			for (int mu = 0; mu < radial_func_count; ++mu)
				if (center_gate_tangent[mu] != 0.0) {
					has_tangent = true;
					break;
				}
		if (!has_tangent)
			return;

	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	const int additive_coeff_offset = TwoLayerGateAdditiveCoeffOffset();
	const int coeff_count = LinearCoeffOffset();
	const double center_linear = linear_coeffs[type_central];
	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const bool center_gate_enabled = TwoLayerGateUsesCenterGate();
	const bool use_basic_mu_groups =
		static_cast<int>(basic_mu_offsets_.size()) == radial_func_count + 1
		&& static_cast<int>(basic_indices_by_mu_.size()) == alpha_index_basic_count;

	bool loaded_final_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * alpha_moments_count;
		if (offset + alpha_moments_count <=
		    two_layer_final_moment_values_cache_.size()
		    && offset + alpha_moments_count <=
		    two_layer_final_moment_ders_cache_.size()) {
			std::copy(two_layer_final_moment_values_cache_.data() + offset,
			          two_layer_final_moment_values_cache_.data() + offset
			              + alpha_moments_count,
			          moment_vals);
			site_energy_ders_wrt_moments_.assign(
				two_layer_final_moment_ders_cache_.data() + offset,
				two_layer_final_moment_ders_cache_.data() + offset
				    + alpha_moments_count);
			loaded_final_cache = true;
		}
	}
		if (!loaded_final_cache) {
			CalcSHMomentValuesOnly(nbh);
			site_energy_ders_wrt_moments_.resize(alpha_moments_count);
			std::fill(site_energy_ders_wrt_moments_.begin(),
		          site_energy_ders_wrt_moments_.end(), 0.0);
		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int node = alpha_moment_mapping[i];
			const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
			site_energy_ders_wrt_moments_[node] += moment_coeff;
		}
		if (UseSHProductRows()) {
			BackpropSHProductRows(site_energy_ders_wrt_moments_);
		} else {
			for (int p = static_cast<int>(sh_products_.size()) - 1;
			     p >= OuterKRank1ProductStart(); --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				if (adj_target == 0.0)
					continue;
				site_energy_ders_wrt_moments_[product.left] +=
					product.coeff * moment_vals[product.right] * adj_target;
				site_energy_ders_wrt_moments_[product.right] +=
					product.coeff * moment_vals[product.left] * adj_target;
			}
			}
			BackpropOuterKRank1Adjoints(site_energy_ders_wrt_moments_);
		}
		const bool radial_mixing_enabled = RadialMixingEnabled;
							const bool radial_mixing_active =
								radial_mixing_enabled
								&& two_layer_gate_radial_mixing_active_eval_;
		const int radial_mixing_coeff_offset =
			radial_mixing_enabled ? TwoLayerGateRadialMixingCoeffOffset() : 0;
		const double* radial_mixing_coeffs =
			radial_mixing_enabled
				? regression_coeffs.data() + radial_mixing_coeff_offset
				: nullptr;
		grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
		grad_dloss_dmom_.assign(alpha_moments_count, 0.0);

	const int radial_der_stride = radial_val_stride * 5;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);
		std::vector<double>& cached_sh_values_all = grad_neighbor_sh_values_cache_;
		std::vector<double>& cached_rb_vals_all = grad_neighbor_radial_vals_cache_;
		std::vector<double>& cached_rb_ders_all = grad_neighbor_radial_ders_cache_;
		std::vector<double>& cached_radial_values_all = grad_neighbor_mu_contract_vals_cache_;
		std::vector<double>& cached_radial_s_all = grad_neighbor_mu_contract_ders_s_cache_;
		std::vector<double>& cached_radial_ss_all = grad_neighbor_mu_contract_ders_ss_cache_;
			std::vector<double>& radial_mixing_gate_tangent_delta =
				sh_gate_tangent_radial_mixing_delta_scratch_;
			std::vector<double>& radial_mixing_gate_input_t =
				sh_gate_tangent_radial_mixing_input_t_scratch_;
			std::vector<double>& radial_mixing_gate_input_t_cache =
				sh_gate_tangent_radial_mixing_input_t_cache_scratch_;
				if (radial_mixing_active) {
					radial_mixing_gate_tangent_delta.assign(
						static_cast<size_t>(outer_k_rank_) * sh_count, 0.0);
					radial_mixing_gate_input_t.resize(radial_func_count);
					const size_t rank_cache_stride =
						static_cast<size_t>(sh_l_max_ + 1)
						* sh_k_max_;
					radial_mixing_gate_input_t_cache.assign(
						neighbor_count * rank_cache_stride, 0.0);
				}
		const bool use_edge_cache =
			HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, true);
			const size_t radial_mixing_kernel_stride =
				static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_
				* sh_k_max_;
		if (!use_edge_cache) {
			cached_sh_values_all.resize(neighbor_count * sh_count);
			cached_rb_vals_all.resize(neighbor_count * radial_val_stride);
			cached_rb_ders_all.resize(neighbor_count * radial_der_stride);
			cached_radial_values_all.resize(neighbor_count * radial_func_count);
			cached_radial_s_all.resize(neighbor_count * radial_func_count);
			cached_radial_ss_all.resize(neighbor_count * radial_func_count);
	}

		for (int j = 0; j < nbh.count; ++j) {
			const double* tangent_by_mu =
				neighbor_gate_tangent.data()
				+ static_cast<size_t>(j) * radial_func_count;
			const Vector3& rvec = nbh.vecs[j];
			const double r = nbh.dists[j];
			const int type_outer = nbh.types[j];
			const double* cached_sh_values = nullptr;
			const double* cached_rb_vals = nullptr;
			const double* cached_radial_values = nullptr;
			const double* cached_radial_mixing_kernels = nullptr;
			if (use_edge_cache) {
				const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
				cached_sh_values =
					two_layer_edge_sh_values_cache_.data() + edge * sh_count;
				cached_rb_vals =
					two_layer_edge_radial_vals_cache_.data() + edge * radial_val_stride;
				cached_radial_values =
					two_layer_edge_mu_vals_cache_.data() + edge * radial_func_count;
				if (radial_mixing_active)
					cached_radial_mixing_kernels =
						two_layer_edge_radial_mixing_kernel_cache_.data()
						+ edge * radial_mixing_kernel_stride;
			} else {
				double* cached_sh_values_write =
					cached_sh_values_all.data() + static_cast<size_t>(j) * sh_count;
				double* cached_rb_vals_write =
					cached_rb_vals_all.data() + static_cast<size_t>(j) * radial_val_stride;
				double* cached_rb_ders_write =
					cached_rb_ders_all.data() + static_cast<size_t>(j) * radial_der_stride;
				double* cached_radial_values_write =
					cached_radial_values_all.data()
					+ static_cast<size_t>(j) * radial_func_count;
			double* cached_radial_s =
				cached_radial_s_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			double* cached_radial_ss =
				cached_radial_ss_all.data()
					+ static_cast<size_t>(j) * radial_func_count;
				cached_sh_values = cached_sh_values_write;
				cached_rb_vals = cached_rb_vals_write;
				cached_radial_values = cached_radial_values_write;
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, cached_sh_values_write);
				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						cached_rb_vals_write[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					for (int xi = 0; xi < R * 5; ++xi)
						cached_rb_ders_write[eval_block * R * 5 + xi] =
							p_RadialBasis->rb_ders[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double radial_value = 0.0;
				double radial_s = 0.0;
					double radial_ss = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * cached_rb_vals[radial_base + xi];
							radial_s += coeff * cached_rb_ders_write[deriv_base + xi + R];
							radial_ss += coeff * cached_rb_ders_write[deriv_base + xi + 3 * R];
						}
					cached_radial_values_write[mu] = radial_value;
					cached_radial_s[mu] = radial_s;
					cached_radial_ss[mu] = radial_ss;
				}
			}
					bool neighbor_has_tangent = false;
					for (int mu = 0; mu < radial_func_count; ++mu)
						if (tangent_by_mu[mu] != 0.0) {
							neighbor_has_tangent = true;
							break;
						}
						bool center_has_tangent = false;
						if (center_gate_enabled && center_gate_tangent != nullptr)
							for (int mu = 0; mu < radial_func_count; ++mu)
								if (center_gate_tangent[mu] != 0.0) {
									center_has_tangent = true;
								break;
							}
					if (!neighbor_has_tangent && !center_has_tangent)
						continue;
						int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
						const double* gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, TwoLayerGateNeighborSignals(nbh, j),
								cached_sh_values, gate_neighbor_atom_index_for_tanh);
						const int gate_center_atom_index = cache_atom_index;
						const double* center_gate_signal_by_mu =
							TwoLayerGateAtomSignals(gate_center_atom_index);
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferAdditive | kGateMuBufferMultiplier);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* center_gate_additive_by_mu =
					two_layer_gate_center_additive_mu_buffer_.data();
				const double* neighbor_gate_multiplier_by_mu =
					two_layer_gate_neighbor_multiplier_mu_buffer_.data();
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
						for (int mu = 0; mu < radial_func_count; ++mu)
						{
							const double center_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_central,
								                   mu);
							const double outer_type_coeff =
								OuterTypeCoeffFast(regression_coeffs,
								                   outer_type_coeff_offset,
								                   outer_type_coeff_stride,
								                   R,
								                   type_outer,
								                   mu);
							double tangent_gate_derivative =
								tangent_by_mu[mu] * gate_additive_by_mu[mu];
						if (center_has_tangent)
							tangent_gate_derivative +=
								center_gate_tangent[mu] * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_by_mu[mu];
							const double gate_input_t =
								center_type_coeff * tangent_gate_derivative;
							gate_scaled_radial_by_mu[mu] =
								gate_input_t * cached_radial_values[mu];
							if (radial_mixing_active)
								radial_mixing_gate_input_t[mu] = gate_input_t;
					}

						if (use_basic_mu_groups) {
							for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
								const double scaled_radial =
									gate_scaled_radial_by_mu[mu];
								if (scaled_radial == 0.0)
									continue;
								const int begin = basic_mu_offsets_[mu];
								const int end = basic_mu_offsets_[mu + 1];
								for (int cursor = begin; cursor < end; ++cursor) {
									const int i = basic_indices_by_mu_[cursor];
									const int sh_index = basic_sh_index_cache_[i];
									grad_dloss_dsenders_[i] +=
										scaled_radial * cached_sh_values[sh_index];
								}
							}
						} else {
							for (int i = 0; !pure_matrix && i < alpha_index_basic_count; ++i) {
								const int mu = basic_mu_cache_[i];
								const int sh_index = basic_sh_index_cache_[i];
								grad_dloss_dsenders_[i] +=
									gate_scaled_radial_by_mu[mu]
									* cached_sh_values[sh_index];
							}
						}
							if (radial_mixing_active) {
								for (int l = 0; l <= sh_l_max_; ++l) {
										const size_t rank_cache_offset =
											(static_cast<size_t>(j)
											 * static_cast<size_t>(sh_l_max_ + 1)
											 + static_cast<size_t>(l))
											* static_cast<size_t>(sh_k_max_);
										double* gate_input_t =
											radial_mixing_gate_input_t_cache.data()
											+ rank_cache_offset;
									for (int k = 0; k < sh_k_max_; ++k) {
											const int mu = k * (sh_l_max_ + 1) + l;
										gate_input_t[k] = radial_mixing_gate_input_t[mu];
									}
								for (int a = 0; a < outer_k_rank_; ++a) {
									double outer_t = 0.0;
									for (int k = 0; k < sh_k_max_; ++k) {
										const int mu = k * (sh_l_max_ + 1) + l;
										const int mixing_radial_base =
											mu_to_radial_eval_block_[mu] * R;
									const size_t lac =
										(static_cast<size_t>(l) * outer_k_rank_ + a)
										* sh_k_max_ + k;
									double kernel;
									if (cached_radial_mixing_kernels != nullptr) {
										kernel = cached_radial_mixing_kernels[lac];
									} else {
										const double* coeff_row = radial_mixing_coeffs
											+ lac * static_cast<size_t>(R);
										kernel = 0.0;
										for (int xi = 0; xi < R; ++xi)
											kernel += coeff_row[xi]
												* cached_rb_vals[mixing_radial_base + xi];
									}
									outer_t += kernel * gate_input_t[k];
									}
									if (outer_t == 0.0)
										continue;
									for (int m = -l; m <= l; ++m) {
										const int sh_index = SHFlatIndex(l, m);
										radial_mixing_gate_tangent_delta[
											static_cast<size_t>(a) * sh_count
											+ SHFlatIndex(l, m)] +=
											outer_t * cached_sh_values[sh_index];
									}
								}
							}
						}
		}

	ContractOuterKRank1Values(grad_dloss_dsenders_.data());
	if (radial_mixing_active)
		for (int a = 0; a < outer_k_rank_; ++a)
			for (int l = 0; l <= sh_l_max_; ++l)
				for (int m = -l; m <= l; ++m)
					grad_dloss_dsenders_[
						OuterKRank1OuterBasicIndex(a, l, m)] +=
						radial_mixing_gate_tangent_delta[
							static_cast<size_t>(a) * sh_count
							+ SHFlatIndex(l, m)];
	const bool use_product_rows = UseSHProductRows();
	const bool use_product_hvt_reverse = !use_product_rows && UseSHProductHVTReverse();
	if (use_product_rows) {
		AccumulateSHProductRowsForward(grad_dloss_dsenders_, grad_dloss_dsenders_);
		for (const SHProductRow& row : sh_product_rows_) {
			const double adj_target = site_energy_ders_wrt_moments_[row.target];
			if (adj_target == 0.0)
				continue;
			const int end = row.term_begin + row.term_count;
			for (int t = row.term_begin; t < end; ++t) {
				const SHProductRowTerm& term = sh_product_row_terms_[t];
				grad_dloss_dmom_[term.left] +=
					grad_dloss_dsenders_[term.right] * adj_target * term.coeff;
				grad_dloss_dmom_[term.right] +=
					grad_dloss_dsenders_[term.left] * adj_target * term.coeff;
			}
		}
		BackpropSHProductRows(grad_dloss_dmom_);
	} else {
		for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
		     p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			grad_dloss_dsenders_[product.target] += product.coeff * (
				grad_dloss_dsenders_[product.left] * moment_vals[product.right]
				+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
		}
		if (use_product_hvt_reverse) {
			for (int p = static_cast<int>(sh_products_.size()) - 1;
			     p >= OuterKRank1ProductStart(); --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				const double hvt_target = grad_dloss_dmom_[product.target];
				if (adj_target == 0.0 && hvt_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] += product.coeff * (
					moment_vals[product.right] * hvt_target
					+ grad_dloss_dsenders_[product.right] * adj_target);
				grad_dloss_dmom_[product.right] += product.coeff * (
					moment_vals[product.left] * hvt_target
					+ grad_dloss_dsenders_[product.left] * adj_target);
			}
		} else {
			for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
			     p < sh_products_.size(); ++p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				if (adj_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] +=
					grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
				grad_dloss_dmom_[product.right] +=
					grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
			}
			for (int p = static_cast<int>(sh_products_.size()) - 1;
			     p >= OuterKRank1ProductStart(); --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = grad_dloss_dmom_[product.target];
				if (adj_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] +=
					product.coeff * moment_vals[product.right] * adj_target;
				grad_dloss_dmom_[product.right] +=
					product.coeff * moment_vals[product.left] * adj_target;
			}
		}
	}
	if (OuterKRankEnabled()) {
		if (OuterKRankHasTrainableWeights()) {
			for (int channel = 0; channel < outer_k_rank_; ++channel) {
				for (int l = sh_l_max_; l >= 0; --l) {
					for (int m = -l; m <= l; ++m) {
						const int outer =
							OuterKRank1OuterBasicIndex(channel, l, m);
						const double hvt_outer_adjoint = grad_dloss_dmom_[outer];
						const double energy_outer_adjoint =
							site_energy_ders_wrt_moments_[outer];
						for (int k = 0; k < sh_k_max_; ++k) {
							const int raw = OuterKRank1RawBasicIndex(l, m, k);
							out_grad_accumulator[
								OuterKRank1WeightIndex(channel, l, k)] +=
								center_linear * (
									hvt_outer_adjoint * moment_vals[raw]
									+ energy_outer_adjoint
										* grad_dloss_dsenders_[raw]);
							}
						}
					}
				}
			}
		BackpropOuterKRank1Adjoints(grad_dloss_dmom_);
	}

	double center_linear_grad = 0.0;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		center_linear_grad += moment_coeff * grad_dloss_dsenders_[node];
		out_grad_accumulator[coeff_count + species_count + i] +=
			center_linear * linear_mults[i] * grad_dloss_dsenders_[node];
	}
	out_grad_accumulator[coeff_count + type_central] += center_linear_grad;

	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);
		if (static_cast<int>(grad_radial_coeff_center_coord_accum_.size()) != radial_func_count)
			grad_radial_coeff_center_coord_accum_.resize(radial_func_count);
			std::vector<double> gate_adjoint_by_mu;
			std::vector<double> center_gate_adjoint_by_mu;
			std::vector<double>& radial_mixing_value_contribution_by_mu =
				sh_gate_tangent_radial_mixing_value_adj_scratch_;
			std::vector<double>& radial_mixing_direct_contribution_by_mu =
				sh_gate_tangent_radial_mixing_direct_adj_scratch_;
			std::vector<double>& radial_mixing_center_direct_contribution_by_mu =
				sh_gate_tangent_radial_mixing_center_direct_adj_scratch_;
			std::vector<double>& radial_mixing_input_by_k =
				sh_gate_tangent_radial_mixing_param_input_scratch_;
			std::vector<double>& radial_mixing_input_t_by_k =
				sh_gate_tangent_radial_mixing_param_input_t_scratch_;
			std::vector<double>& radial_mixing_param_kernel =
				sh_gate_tangent_radial_mixing_param_kernel_scratch_;
			std::vector<double>& radial_mixing_param_kernel_s =
				sh_gate_tangent_radial_mixing_param_kernel_s_scratch_;
			std::vector<double>& radial_mixing_param_kernel_ss =
				sh_gate_tangent_radial_mixing_param_kernel_ss_scratch_;
			if (radial_mixing_enabled) {
				radial_mixing_input_by_k.resize(sh_k_max_);
				radial_mixing_input_t_by_k.resize(sh_k_max_);
			}
			if (radial_mixing_active) {
				radial_mixing_value_contribution_by_mu.assign(
					radial_func_count, 0.0);
				radial_mixing_direct_contribution_by_mu.assign(
					radial_func_count, 0.0);
				radial_mixing_center_direct_contribution_by_mu.assign(
					radial_func_count, 0.0);
				const int ac_count = outer_k_rank_ * sh_k_max_;
				radial_mixing_param_kernel.resize(pure_matrix ? std::max(ac_count, 2 * outer_k_rank_) : ac_count);
				radial_mixing_param_kernel_s.resize(pure_matrix ? std::max(ac_count, 2 * R) : ac_count);
				radial_mixing_param_kernel_ss.resize(ac_count);
			}

		for (int j = 0; j < nbh.count; ++j) {
		const int type_outer = nbh.types[j];
		const int center_type_coeff_index_base =
			OuterTypeCoeffIndexFast(outer_type_coeff_offset,
			                        outer_type_coeff_stride,
			                        R,
			                        type_central,
			                        0);
		const int outer_type_coeff_index_base =
			OuterTypeCoeffIndexFast(outer_type_coeff_offset,
			                        outer_type_coeff_stride,
			                        R,
			                        type_outer,
			                        0);
		const int additive_outer_index_base =
			GateAdditiveCoeffIndexFast(additive_coeff_offset,
			                           radial_func_count,
			                           type_outer,
			                           0);
		const int additive_center_index_base =
			GateAdditiveCoeffIndexFast(additive_coeff_offset,
			                           radial_func_count,
			                           type_central,
			                           0);
			const double* cached_sh_values = nullptr;
			const double* cached_rb_vals = nullptr;
			const double* cached_rb_ders = nullptr;
			const double* cached_radial_values = nullptr;
			const double* cached_radial_s = nullptr;
			const double* cached_radial_ss = nullptr;
			const double* cached_radial_mixing_kernels = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
				cached_rb_vals =
					two_layer_edge_radial_vals_cache_.data() + edge * radial_val_stride;
				cached_rb_ders =
					two_layer_edge_radial_ders_cache_.data() + edge * radial_der_stride;
				cached_radial_values =
					two_layer_edge_mu_vals_cache_.data() + edge * radial_func_count;
			cached_radial_s =
				two_layer_edge_mu_ders_s_cache_.data() + edge * radial_func_count;
			cached_radial_ss =
				two_layer_edge_mu_ders_ss_cache_.data() + edge * radial_func_count;
			if (radial_mixing_active)
				cached_radial_mixing_kernels =
					two_layer_edge_radial_mixing_kernel_cache_.data()
					+ edge * radial_mixing_kernel_stride;
		} else {
			cached_sh_values =
				cached_sh_values_all.data() + static_cast<size_t>(j) * sh_count;
				cached_rb_vals =
					cached_rb_vals_all.data() + static_cast<size_t>(j) * radial_val_stride;
				cached_rb_ders =
					cached_rb_ders_all.data() + static_cast<size_t>(j) * radial_der_stride;
				cached_radial_values =
					cached_radial_values_all.data()
					+ static_cast<size_t>(j) * radial_func_count;
			cached_radial_s =
				cached_radial_s_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			cached_radial_ss =
				cached_radial_ss_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
		}
				int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
				const double* gate_signal_by_mu =
					PrepareTwoLayerGateEdgeL1NeighborSignals(
						nbh, j, TwoLayerGateNeighborSignals(nbh, j),
						cached_sh_values, gate_neighbor_atom_index_for_tanh);
				const double* tangent_by_mu =
						neighbor_gate_tangent.data()
						+ static_cast<size_t>(j) * radial_func_count;
					const int gate_center_atom_index = cache_atom_index;
					const double* center_gate_signal_by_mu =
						TwoLayerGateAtomSignals(gate_center_atom_index);
					PrepareTwoLayerGatePairMuBuffers(
						type_central, type_outer,
						center_gate_signal_by_mu, gate_center_atom_index,
						gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
						kGateMuBufferAll);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();
				const double* gate_multiplier_by_mu =
					two_layer_gate_multiplier_mu_buffer_.data();
				const double* gate_outer_by_mu =
					two_layer_gate_outer_mu_buffer_.data();
				const double* gate_additive_param_by_mu =
					two_layer_gate_additive_param_mu_buffer_.data();
				const double* gate_gate_residual_by_mu =
					two_layer_gate_gate_residual_mu_buffer_.data();
				const double* gate_gate_outer_param_by_mu =
					two_layer_gate_gate_outer_param_mu_buffer_.data();
					const double* gate_gate_additive_param_by_mu =
						two_layer_gate_gate_additive_param_mu_buffer_.data();
					const double* neighbor_gate_multiplier_by_mu =
						two_layer_gate_neighbor_multiplier_mu_buffer_.data();
					const double* neighbor_gate_additive_by_mu =
						two_layer_gate_neighbor_additive_mu_buffer_.data();
					const double* neighbor_gate_additive_param_by_mu =
						two_layer_gate_neighbor_additive_param_mu_buffer_.data();
					const double* center_gate_additive_by_mu =
						two_layer_gate_center_additive_mu_buffer_.data();
					const double* center_gate_additive_param_by_mu =
						two_layer_gate_center_additive_param_mu_buffer_.data();
					const double* center_gate_residual_by_mu =
						two_layer_gate_center_residual_mu_buffer_.data();
					const double* center_gate_gate_additive_param_by_mu =
						two_layer_gate_center_gate_additive_param_mu_buffer_.data();
						gate_adjoint_by_mu.assign(radial_func_count, 0.0);
						center_gate_adjoint_by_mu.assign(radial_func_count, 0.0);
			double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
			double* radial_coeff_direct_accum = grad_radial_coeff_coord_accum_.data();
			double* radial_coeff_center_direct_accum =
				grad_radial_coeff_center_coord_accum_.data();
			std::fill(radial_coeff_value_accum, radial_coeff_value_accum + radial_func_count, 0.0);
				std::fill(radial_coeff_direct_accum, radial_coeff_direct_accum + radial_func_count, 0.0);
				std::fill(radial_coeff_center_direct_accum,
				          radial_coeff_center_direct_accum + radial_func_count, 0.0);
				if (radial_mixing_active) {
					std::fill(radial_mixing_value_contribution_by_mu.begin(),
					          radial_mixing_value_contribution_by_mu.end(), 0.0);
					std::fill(radial_mixing_direct_contribution_by_mu.begin(),
					          radial_mixing_direct_contribution_by_mu.end(), 0.0);
					std::fill(radial_mixing_center_direct_contribution_by_mu.begin(),
					          radial_mixing_center_direct_contribution_by_mu.end(), 0.0);
				}

					if (radial_mixing_enabled) {
						for (int l = 0; l <= sh_l_max_; ++l) {
								double* gate_input_storage =
									radial_mixing_input_by_k.data();
								double* gate_input_t_storage =
									radial_mixing_input_t_by_k.data();
							const double* gate_input = gate_input_storage;
							const double* gate_input_t = gate_input_t_storage;
							const bool use_cached_gate_input_t =
								radial_mixing_active;
								if (use_cached_gate_input_t) {
									const size_t rank_cache_offset =
										(static_cast<size_t>(j)
										 * static_cast<size_t>(sh_l_max_ + 1)
										 + static_cast<size_t>(l))
										* static_cast<size_t>(sh_k_max_);
									gate_input_t =
									radial_mixing_gate_input_t_cache.data()
										+ rank_cache_offset;
								}
								std::fill(gate_input_storage,
								          gate_input_storage + sh_k_max_,
								          0.0);
								if (!use_cached_gate_input_t)
									std::fill(gate_input_t_storage,
									          gate_input_t_storage
									              + sh_k_max_,
									          0.0);
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
								const double center_type_coeff =
								regression_coeffs[
									center_type_coeff_index_base
									+ mu * outer_type_coeff_stride];
							const double outer_type_coeff =
								regression_coeffs[
									outer_type_coeff_index_base
									+ mu * outer_type_coeff_stride];
							double tangent_gate_derivative =
								tangent_by_mu[mu] * gate_additive_by_mu[mu];
							if (center_gate_enabled && center_gate_tangent != nullptr)
								tangent_gate_derivative +=
									center_gate_tangent[mu] * outer_type_coeff
									* neighbor_gate_multiplier_by_mu[mu]
									* center_gate_additive_by_mu[mu];
							gate_input_storage[k] = gate_type_scale_by_mu[mu];
								if (!use_cached_gate_input_t)
									gate_input_t_storage[k] =
										center_type_coeff * tangent_gate_derivative;
								}
							if (pure_matrix) {
								double* value_weights = radial_mixing_param_kernel.data();
								double* tangent_weights = value_weights + outer_k_rank_;
								for (int o = 0; o < outer_k_rank_; ++o) {
									double value_sum = 0.0;
									double tangent_sum = 0.0;
									for (int m = -l; m <= l; ++m) {
										const int outer = OuterKRank1OuterBasicIndex(o, l, m);
										const double y = cached_sh_values[SHFlatIndex(l, m)];
										value_sum += center_linear * grad_dloss_dmom_[outer] * y;
										tangent_sum += center_linear * site_energy_ders_wrt_moments_[outer] * y;
									}
									value_weights[o] = value_sum;
									tangent_weights[o] = tangent_sum;
								}
								AccumulatePureWEdgeParameterGrad<false, true>(
									l, sh_l_max_ + 1, sh_k_max_, outer_k_rank_, R, C,
									type_central * C + type_outer,
									mu_to_radial_eval_block_.data(), radial_eval_to_scaling_block_.data(),
									radial_mixing_coeffs, radial_mixing_coeff_offset,
									cached_rb_vals, cached_rb_ders, value_weights, nullptr,
									tangent_weights, gate_input, gate_input_t,
									cached_radial_mixing_kernels, nullptr, out_grad_accumulator.data(),
									radial_mixing_value_contribution_by_mu.data(),
									radial_mixing_direct_contribution_by_mu.data(),
									radial_mixing_param_kernel_s.data());
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const double direct = radial_mixing_direct_contribution_by_mu[mu];
									if (center_gate_enabled && center_gate_tangent != nullptr)
										radial_mixing_center_direct_contribution_by_mu[mu] = direct * center_gate_tangent[mu];
									radial_mixing_direct_contribution_by_mu[mu] = direct * tangent_by_mu[mu];
								}
								continue;
							}
							if (radial_mixing_active) {
								for (int a = 0; a < outer_k_rank_; ++a) {
									for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const int mixing_radial_base =
										mu_to_radial_eval_block_[mu] * R;
									const int mixing_der_base = 5 * mixing_radial_base;
									const int ac = a * sh_k_max_ + k;
									const size_t lac =
										(static_cast<size_t>(l) * outer_k_rank_ + a)
										* sh_k_max_ + k;
									const double* coeff_row = radial_mixing_coeffs
										+ lac * static_cast<size_t>(R);
								double kernel = cached_radial_mixing_kernels != nullptr
									? cached_radial_mixing_kernels[lac]
									: 0.0;
								double kernel_s = 0.0;
								double kernel_ss = 0.0;
								for (int xi = 0; xi < R; ++xi) {
									const double coeff = coeff_row[xi];
									if (cached_radial_mixing_kernels == nullptr)
										kernel += coeff
											* cached_rb_vals[mixing_radial_base + xi];
										kernel_s += coeff
											* cached_rb_ders[mixing_der_base + R + xi];
										kernel_ss += coeff
											* cached_rb_ders[mixing_der_base + 3 * R + xi];
									}
									radial_mixing_param_kernel[ac] = kernel;
									radial_mixing_param_kernel_s[ac] = kernel_s;
									radial_mixing_param_kernel_ss[ac] = kernel_ss;
								}
							}
						}
						for (int a = 0; a < outer_k_rank_; ++a) {
							double sum_adj_y = 0.0;
							double sum_energy_y = 0.0;
							for (int m = -l; m <= l; ++m) {
								const int sh_index = SHFlatIndex(l, m);
								const int outer =
									OuterKRank1OuterBasicIndex(a, l, m);
								const double y = cached_sh_values[sh_index];
								sum_adj_y +=
									center_linear * grad_dloss_dmom_[outer] * y;
								sum_energy_y +=
									center_linear
									* site_energy_ders_wrt_moments_[outer] * y;
							}
								for (int k = 0; k < sh_k_max_; ++k) {
									const int mu = k * (sh_l_max_ + 1) + l;
									const int mixing_radial_base =
										mu_to_radial_eval_block_[mu] * R;
									const int mixing_der_base = 5 * mixing_radial_base;
									const int mixing_scaling_block =
										radial_eval_to_scaling_block_[
											mu_to_radial_eval_block_[mu]];
									const int mixing_sigma_coeff_offset =
										C + 2 * C * C * mixing_scaling_block
										+ type_central * C + type_outer;
									const double input_value_factor =
									gate_input[k] * sum_adj_y;
								const double input_t_energy_factor =
									gate_input_t[k] * sum_energy_y;
								const int mixing_coeff_base =
									radial_mixing_coeff_offset
									+ (((l * outer_k_rank_ + a)
									    * sh_k_max_ + k) * R);
							for (int xi = 0; xi < R; ++xi) {
									const double rb =
										cached_rb_vals[mixing_radial_base + xi];
									out_grad_accumulator[mixing_coeff_base + xi] +=
										rb * (input_value_factor + input_t_energy_factor);
								}
									if (!radial_mixing_active)
										continue;
									const int ac = a * sh_k_max_ + k;
									out_grad_accumulator[mixing_sigma_coeff_offset] +=
										radial_mixing_param_kernel_s[ac]
										* (input_value_factor + input_t_energy_factor);
								out_grad_accumulator[mixing_sigma_coeff_offset + C * C] +=
									radial_mixing_param_kernel_ss[ac]
									* (input_value_factor + input_t_energy_factor);
								const double kernel = radial_mixing_param_kernel[ac];
								radial_mixing_value_contribution_by_mu[mu] +=
									kernel * sum_adj_y;
								radial_mixing_direct_contribution_by_mu[mu] +=
									kernel * sum_energy_y * tangent_by_mu[mu];
								if (center_gate_enabled && center_gate_tangent != nullptr)
									radial_mixing_center_direct_contribution_by_mu[mu] +=
										kernel * sum_energy_y * center_gate_tangent[mu];
								}
							}
					}
				}

						if (use_basic_mu_groups) {
						for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
						const double neighbor_tangent = tangent_by_mu[mu];
						const double center_tangent =
							(center_gate_enabled && center_gate_tangent != nullptr)
								? center_gate_tangent[mu]
								: 0.0;
						const int begin = basic_mu_offsets_[mu];
						const int end = basic_mu_offsets_[mu + 1];
						for (int cursor = begin; cursor < end; ++cursor) {
								const int i = basic_indices_by_mu_[cursor];
								const int sh_index = basic_sh_index_cache_[i];
								const double y = cached_sh_values[sh_index];
								const double value_weight =
									center_linear * grad_dloss_dmom_[i];
								const double site_adjoint =
									center_linear * site_energy_ders_wrt_moments_[i];
								const double direct_weight =
									site_adjoint * neighbor_tangent;
							const double center_direct_weight =
								site_adjoint * center_tangent;
							if (value_weight == 0.0 && direct_weight == 0.0
							    && center_direct_weight == 0.0)
								continue;
							if (value_weight != 0.0)
								radial_coeff_value_accum[mu] += value_weight * y;
							if (direct_weight != 0.0)
								radial_coeff_direct_accum[mu] += direct_weight * y;
							if (center_direct_weight != 0.0)
								radial_coeff_center_direct_accum[mu] +=
									center_direct_weight * y;
						}
					}
				} else {
					for (int i = 0; !pure_matrix && i < alpha_index_basic_count; ++i) {
							const int mu = basic_mu_cache_[i];
							const int sh_index = basic_sh_index_cache_[i];
							const double y = cached_sh_values[sh_index];
							const double value_weight =
								center_linear * grad_dloss_dmom_[i];
							const double site_adjoint =
								center_linear * site_energy_ders_wrt_moments_[i];
							const double direct_weight =
								site_adjoint * tangent_by_mu[mu];
							const double center_direct_weight =
								(center_gate_enabled && center_gate_tangent != nullptr)
								? site_adjoint * center_gate_tangent[mu]
								: 0.0;
						if (value_weight == 0.0 && direct_weight == 0.0
						    && center_direct_weight == 0.0)
							continue;
						if (value_weight != 0.0)
							radial_coeff_value_accum[mu] += value_weight * y;
						if (direct_weight != 0.0)
							radial_coeff_direct_accum[mu] += direct_weight * y;
						if (center_direct_weight != 0.0)
							radial_coeff_center_direct_accum[mu] +=
								center_direct_weight * y;
					}
				}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double value_accum = radial_coeff_value_accum[mu];
				const double direct_accum = radial_coeff_direct_accum[mu];
				const double center_direct_accum =
					radial_coeff_center_direct_accum[mu];
				const double mixing_value = radial_mixing_active
					? radial_mixing_value_contribution_by_mu[mu] : 0.0;
				const double mixing_direct = radial_mixing_active
					? radial_mixing_direct_contribution_by_mu[mu] : 0.0;
				const double mixing_center_direct = radial_mixing_active
					? radial_mixing_center_direct_contribution_by_mu[mu] : 0.0;
				if (value_accum == 0.0 && direct_accum == 0.0
				    && center_direct_accum == 0.0 && mixing_value == 0.0
				    && mixing_direct == 0.0 && mixing_center_direct == 0.0)
					continue;
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int scaling_block =
				radial_eval_to_scaling_block_[mu_to_radial_eval_block_[mu]];
			const int radial_offset = radial_coeff_base + mu * (R + C);
						const int sigma_coeff_offset =
							C + 2 * C * C * scaling_block + type_central * C + type_outer;
							const double dot_val = cached_radial_values[mu];
							const double value_contribution =
								dot_val * value_accum + mixing_value;
							const double direct_contribution =
								dot_val * direct_accum + mixing_direct;
							const double center_direct_contribution =
								dot_val * center_direct_accum + mixing_center_direct;
						const double center_type_coeff =
							regression_coeffs[
								center_type_coeff_index_base
								+ mu * outer_type_coeff_stride];
						const double outer_type_coeff =
							regression_coeffs[
								outer_type_coeff_index_base
								+ mu * outer_type_coeff_stride];
						const int center_type_coeff_index =
							center_type_coeff_index_base
							+ mu * outer_type_coeff_stride;
						const int outer_type_coeff_index =
							outer_type_coeff_index_base
							+ mu * outer_type_coeff_stride;
						const double gate_df = gate_additive_by_mu[mu];
					const double type_scale = gate_type_scale_by_mu[mu];
					const double gate_multiplier = gate_multiplier_by_mu[mu];
					const double gate_outer = gate_outer_by_mu[mu];
					const double gate_da = gate_additive_param_by_mu[mu];
					const double gate_df_df = gate_gate_residual_by_mu[mu];
						const double gate_df_douter = gate_gate_outer_param_by_mu[mu];
						const double gate_df_da = gate_gate_additive_param_by_mu[mu];
							const double center_gate_df = TwoLayerGateUsesCenterGate()
								? outer_type_coeff * neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_by_mu[mu]
							: 0.0;
								if (value_contribution != 0.0) {
									out_grad_accumulator[center_type_coeff_index] +=
										gate_outer * value_contribution;
								out_grad_accumulator[outer_type_coeff_index] +=
									center_type_coeff * value_contribution;
							}
							if (direct_contribution != 0.0) {
								out_grad_accumulator[center_type_coeff_index] +=
									gate_df * direct_contribution;
								out_grad_accumulator[outer_type_coeff_index] +=
									center_type_coeff * gate_df_douter * direct_contribution;
							}
							if (center_direct_contribution != 0.0) {
								out_grad_accumulator[center_type_coeff_index] +=
									center_gate_df * center_direct_contribution;
								out_grad_accumulator[outer_type_coeff_index] +=
									center_type_coeff * neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* center_direct_contribution;
						}
						if (value_contribution != 0.0)
							gate_adjoint_by_mu[mu] +=
								center_type_coeff * gate_df * value_contribution;
						if (value_contribution != 0.0 && TwoLayerGateUsesCenterGate())
							center_gate_adjoint_by_mu[mu] +=
								center_type_coeff * center_gate_df * value_contribution;
						if (direct_contribution != 0.0)
							gate_adjoint_by_mu[mu] +=
								center_type_coeff * gate_df_df * direct_contribution;
						if (direct_contribution != 0.0 && TwoLayerGateUsesCenterGate())
							center_gate_adjoint_by_mu[mu] +=
								center_type_coeff * neighbor_gate_additive_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* direct_contribution;
						if (center_direct_contribution != 0.0) {
							gate_adjoint_by_mu[mu] +=
								center_type_coeff * neighbor_gate_additive_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* center_direct_contribution;
							center_gate_adjoint_by_mu[mu] +=
								center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_residual_by_mu[mu]
								* center_direct_contribution;
						}
						const double additive_accum =
							center_type_coeff * (gate_da * value_contribution
								+ gate_df_da * direct_contribution);
						if (two_layer_gate_enabled_ && additive_accum != 0.0)
							out_grad_accumulator[additive_outer_index_base + mu] +=
								additive_accum;
						if (value_contribution != 0.0 && TwoLayerGateUsesCenterGate())
							out_grad_accumulator[additive_center_index_base + mu] +=
								center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_param_by_mu[mu]
								* value_contribution;
						if (direct_contribution != 0.0 && TwoLayerGateUsesCenterGate())
							out_grad_accumulator[additive_center_index_base + mu] +=
								center_type_coeff * neighbor_gate_additive_by_mu[mu]
								* center_gate_additive_param_by_mu[mu]
								* direct_contribution;
						if (center_direct_contribution != 0.0) {
							out_grad_accumulator[additive_outer_index_base + mu] +=
								center_type_coeff * neighbor_gate_additive_param_by_mu[mu]
								* center_gate_additive_by_mu[mu]
								* center_direct_contribution;
							out_grad_accumulator[additive_center_index_base + mu] +=
								center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_gate_additive_param_by_mu[mu]
								* center_direct_contribution;
					}
					// Pure W has already propagated shared scaling above; these
					// remaining terms belong only to the absent legacy radial factor.
					if (pure_matrix)
						continue;
					const double sigma_accum =
						type_scale * value_accum + center_type_coeff * gate_df * direct_accum
						+ center_type_coeff * center_gate_df * center_direct_accum;
					if (sigma_accum != 0.0) {
				out_grad_accumulator[sigma_coeff_offset] +=
					cached_radial_s[mu] * sigma_accum;
				out_grad_accumulator[sigma_coeff_offset + C * C] +=
					cached_radial_ss[mu] * sigma_accum;
			}
			for (int xi = 0; xi < R; ++xi) {
					const double rb_val = cached_rb_vals[radial_base + xi];
					if (value_accum != 0.0)
						out_grad_accumulator[radial_offset + xi] +=
							type_scale * rb_val * value_accum;
						if (direct_accum != 0.0)
							out_grad_accumulator[radial_offset + xi] +=
								center_type_coeff * gate_df * rb_val * direct_accum;
						if (center_direct_accum != 0.0)
							out_grad_accumulator[radial_offset + xi] +=
								center_type_coeff * center_gate_df * rb_val
								* center_direct_accum;
					}
			}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				AddTwoLayerGateMuAdjoint(nbh, j, mu, gate_adjoint_by_mu[mu]);
				if (TwoLayerGateUsesCenterGate())
					AddTwoLayerGateAtomMuAdjoint(
						gate_center_atom_index, mu,
						center_gate_adjoint_by_mu[mu]);
			}
			AccumulateTwoLayerGateEdgeL1WeightGradForPair(
				nbh, j, cached_sh_values, gate_adjoint_by_mu.data(),
				out_grad_accumulator);
			AccumulateTwoLayerGateEdgeL1SourceAdjointsForPair(
				nbh, j, cached_sh_values, gate_adjoint_by_mu.data(),
				two_layer_gate_edge_l1_adjoints_);
		}
}

void MLMTPR::AccumulateSHCombinationGrad(const Neighborhood& nbh,
                                         std::vector<double>& out_grad_accumulator,
                                         const double se_weight,
                                         const Vector3* se_ders_weights)
{
	// Select once per neighborhood; the disabled instantiation contains no
	// radial-mixing parameter or chain-rule work in its inner loops.
	if (TwoLayerGateRadialMixingEnabled())
		AccumulateSHCombinationGradImpl<true>(
			nbh, out_grad_accumulator, se_weight, se_ders_weights);
	else
		AccumulateSHCombinationGradImpl<false>(
			nbh, out_grad_accumulator, se_weight, se_ders_weights);
}

template <bool RadialMixingEnabled>
void MLMTPR::AccumulateSHCombinationGradImpl(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	const double se_weight,
	const Vector3* se_ders_weights)
{
	const bool pure_matrix = RadialMixingEnabled && OuterUsesPureRadialMatrix();
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");

	if (TwoLayerResidualEnabled()) {
		out_grad_accumulator.resize(CoeffCount());
		const bool need_e0_grad = (two_layer_residual_eval_stage_ != 2);
		const bool need_e1_grad = (two_layer_residual_eval_stage_ != 1);
		std::vector<double> e0_grad;
		std::vector<double> e1_grad;
		const bool saved_residual = two_layer_residual_enabled_;
		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
		const bool saved_skip_outer_param_grad =
			two_layer_residual_skip_outer_param_grad_;
		const std::vector<double> saved_linear_coeffs = linear_coeffs;
		const int residual_linear_offset = LinearCoeffOffset();
		const int residual_e0_offset = TwoLayerResidualE0CoeffOffset();
		const int component_linear_offset =
			TwoLayerGateWeightOffset() + TwoLayerGateWeightCount();
		const int type_central = nbh.my_type;
		std::vector<double> saved_e0_coeffs(alpha_scalar_moments, 0.0);
		for (int i = 0; i < alpha_scalar_moments; ++i)
			saved_e0_coeffs[i] = TwoLayerResidualE0Coeff(i);

		if (need_e0_grad) {
			e0_grad.assign(CoeffCount(), 0.0);
			two_layer_residual_enabled_ = false;
			active_two_layer_gate_values_ = nullptr;
			active_two_layer_gate_adjoints_ = nullptr;
			for (int i = 0; i < alpha_scalar_moments; ++i)
				linear_coeffs[species_count + i] = saved_e0_coeffs[i];
			for (int i = 0; i < species_count; ++i)
				linear_coeffs[i] = 1.0;
			AccumulateSHCombinationGrad(nbh, e0_grad, se_weight, se_ders_weights);
			for (int i = 0; i < component_linear_offset; ++i)
				out_grad_accumulator[i] += e0_grad[i];
			for (int i = 0; i < species_count; ++i)
				out_grad_accumulator[residual_linear_offset + i] +=
					(i == type_central ? se_weight : 0.0);
			for (int i = 0; i < alpha_scalar_moments; ++i)
				out_grad_accumulator[residual_e0_offset + i] +=
					e0_grad[component_linear_offset + species_count + i];
		}

		if (need_e1_grad) {
			e1_grad.assign(CoeffCount(), 0.0);
			two_layer_residual_enabled_ = false;
			linear_coeffs = saved_linear_coeffs;
			for (int i = 0; i < species_count; ++i)
				linear_coeffs[i] = 1.0;
			active_two_layer_gate_values_ = saved_gate_values;
			active_two_layer_gate_adjoints_ = saved_gate_adjoints;
			two_layer_residual_skip_outer_param_grad_ =
				(two_layer_residual_eval_stage_ == 2);
			AccumulateSHCombinationGrad(nbh, e1_grad, se_weight, se_ders_weights);
			two_layer_residual_skip_outer_param_grad_ =
				saved_skip_outer_param_grad;
			if (shift_)
				e1_grad[type_central] -= se_weight;
			e1_grad[component_linear_offset + type_central] -= se_weight;
			for (int i = 0; i < component_linear_offset; ++i)
				out_grad_accumulator[i] += e1_grad[i];
			for (int i = species_count; i < LinearCoeffCount(); ++i)
				out_grad_accumulator[residual_linear_offset + i] +=
					e1_grad[component_linear_offset + i];
		}

		two_layer_residual_enabled_ = saved_residual;
		active_two_layer_gate_values_ = saved_gate_values;
		active_two_layer_gate_adjoints_ = saved_gate_adjoints;
		two_layer_residual_skip_outer_param_grad_ =
			saved_skip_outer_param_grad;
		linear_coeffs = saved_linear_coeffs;
		return;
	}

	const bool skip_site_der_output = UseSHAccumSkipSiteDers();

	{
		buff_site_energy_ = 0.0;
		buff_site_energy_0 = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		if (!skip_site_der_output)
			FillWithZero(buff_site_energy_ders_);
		out_grad_accumulator.resize(CoeffCount());
		double* grad_out = out_grad_accumulator.data();

		const int C = species_count;
		const int R = p_RadialBasis->rb_size;
				const int K = radial_func_count;
				const int coeff_count = LinearCoeffOffset();
				const int type_central = nbh.my_type;
				const int radial_coeff_base = C + 2 * C * C * K_;
				const int outer_type_coeff_offset = RadialCoeffOffset();
				const int outer_type_coeff_stride = RadialCoeffBlockSize();
				const int additive_coeff_offset = TwoLayerGateAdditiveCoeffOffset();
				const double center_linear = linear_coeffs[type_central];

		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
		std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0.0);
		if (static_cast<int>(grad_dloss_dsenders_.size()) != alpha_moments_count)
			grad_dloss_dsenders_.resize(alpha_moments_count);
		if (static_cast<int>(grad_dloss_dmom_.size()) != alpha_moments_count)
			grad_dloss_dmom_.resize(alpha_moments_count);
		std::fill(grad_dloss_dsenders_.begin(), grad_dloss_dsenders_.end(), 0.0);
		std::fill(grad_dloss_dmom_.begin(), grad_dloss_dmom_.end(), 0.0);
		const bool use_product_rows = UseSHProductRows();
		const bool use_product_hvt_reverse = !use_product_rows && UseSHProductHVTReverse();
		const bool profile_accum = SHAccumProfileEnabledOnRank0();
		const double profile_start = profile_accum ? SHAccumProfileNow() : 0.0;
		if (se_ders_weights != nullptr)
			CalcSHMomentValuesWithGradientCache(nbh);
		else
			CalcSHMomentValuesOnly(nbh);
		const double profile_after_moment = profile_accum ? SHAccumProfileNow() : 0.0;

			double scalar_sum = 0.0;
			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const int node = alpha_moment_mapping[i];
				const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
				scalar_sum += moment_coeff * moment_vals[node];
				site_energy_ders_wrt_moments_[node] += moment_coeff;
			}

			if (use_product_rows) {
				BackpropSHProductRows(site_energy_ders_wrt_moments_);
			} else {
				for (int p = static_cast<int>(sh_products_.size()) - 1;
				     p >= OuterKRank1ProductStart(); --p) {
					const SHProduct& product = sh_products_[p];
					const double adj_target = site_energy_ders_wrt_moments_[product.target];
					if (adj_target == 0.0)
						continue;
					site_energy_ders_wrt_moments_[product.left] +=
						product.coeff * moment_vals[product.right] * adj_target;
					site_energy_ders_wrt_moments_[product.right] +=
						product.coeff * moment_vals[product.left] * adj_target;
				}
			}
			BackpropOuterKRank1Adjoints(site_energy_ders_wrt_moments_);
		const double profile_after_energy_backprop = profile_accum ? SHAccumProfileNow() : 0.0;

			double sh_values[kMaxSHComponents];
			double sh_ders[3 * kMaxSHComponents];
		std::vector<double>& rb_vals = radial_vals_buffer_;
		std::vector<double>& rb_ders = radial_ders_buffer_;
		std::vector<double>& radial_values = grad_mu_contract_vals_;
		std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
			std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
			std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
			std::vector<double>& radial_coord_s_derivatives = grad_mu_contract_coord_ders_s_;
			std::vector<double>& radial_coord_ss_derivatives = grad_mu_contract_coord_ders_ss_;
			const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
			const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
			const int radial_val_stride = eval_block_count * R;
				const int radial_der_stride = radial_val_stride * 5;
				const int mu_stride = radial_func_count;
						const bool use_edge_cache_for_gradient =
							se_ders_weights != nullptr
							&& HasTwoLayerEdgePrimitiveCache(-1, true, true);
						std::vector<double>& radial_mixing_tangent_delta =
							sh_combination_radial_mixing_tangent_delta_scratch_;
						std::vector<double>& radial_mixing_tangent_input_cache =
							sh_combination_radial_mixing_tangent_input_cache_scratch_;
						std::vector<double>& radial_mixing_tangent_input_t_cache =
							sh_combination_radial_mixing_tangent_input_t_cache_scratch_;
						std::vector<double>& radial_mixing_tangent_kernel =
							sh_combination_radial_mixing_tangent_kernel_scratch_;
						std::vector<double>& radial_mixing_tangent_kernel_t =
							sh_combination_radial_mixing_tangent_kernel_t_scratch_;
						std::vector<double>& radial_mixing_tangent_outer =
							sh_combination_radial_mixing_tangent_outer_scratch_;
						std::vector<double>& radial_mixing_tangent_outer_t =
							sh_combination_radial_mixing_tangent_outer_t_scratch_;
						const double* radial_mixing_coeffs = nullptr;
						const bool radial_mixing_enabled = RadialMixingEnabled;
						const bool radial_mixing_active =
							radial_mixing_enabled
							&& two_layer_gate_radial_mixing_active_eval_;
						const size_t radial_mixing_kernel_stride =
							static_cast<size_t>(sh_l_max_ + 1) * outer_k_rank_
							* sh_k_max_;
						const int radial_mixing_coeff_offset =
							radial_mixing_enabled ? TwoLayerGateRadialMixingCoeffOffset() : 0;
						if (radial_mixing_active) {
							radial_mixing_tangent_delta.assign(
								static_cast<size_t>(outer_k_rank_) * sh_count, 0.0);
						const size_t rank_cache_stride =
							static_cast<size_t>(sh_l_max_ + 1)
							* sh_k_max_;
						radial_mixing_tangent_input_cache.resize(
							static_cast<size_t>(nbh.count) * rank_cache_stride);
						radial_mixing_tangent_input_t_cache.resize(
							static_cast<size_t>(nbh.count) * rank_cache_stride);
						radial_mixing_tangent_kernel.resize(
							outer_k_rank_ * sh_k_max_);
						radial_mixing_tangent_kernel_t.resize(
							outer_k_rank_ * sh_k_max_);
							radial_mixing_tangent_outer.resize(outer_k_rank_);
							radial_mixing_tangent_outer_t.resize(outer_k_rank_);
							radial_mixing_coeffs = regression_coeffs.data()
								+ radial_mixing_coeff_offset;
						}
				double profile_after_force_cache = profile_after_energy_backprop;
				double profile_after_tangent_forward = profile_after_energy_backprop;
				double profile_after_mixed_seed = profile_after_energy_backprop;

			if (se_ders_weights != nullptr) {
				for (int j = 0; j < nbh.count; ++j) {
					const Vector3& rvec = nbh.vecs[j];
					const Vector3& se_weight_vec = se_ders_weights[j];
					const double r = nbh.dists[j];
					const double inv_r = 1.0 / r;
						const double wr = (se_weight_vec[0] * rvec[0]
							+ se_weight_vec[1] * rvec[1]
							+ se_weight_vec[2] * rvec[2]) * inv_r;
						const int type_outer = nbh.types[j];
						const double* cached_sh_values = nullptr;
							const double* cached_sh_ders = nullptr;
							const double* cached_radial_values = nullptr;
							const double* cached_radial_derivatives = nullptr;
							const double* cached_radial_mixing_kernels = nullptr;
							const double* cached_radial_mixing_kernel_ders = nullptr;
							if (use_edge_cache_for_gradient) {
								const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							cached_sh_values =
								two_layer_edge_sh_values_cache_.data() + edge * sh_count;
							cached_sh_ders =
								two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							cached_radial_values =
								two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
								cached_radial_derivatives =
									two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
								if (radial_mixing_active) {
									cached_radial_mixing_kernels =
										two_layer_edge_radial_mixing_kernel_cache_.data()
										+ edge * radial_mixing_kernel_stride;
									cached_radial_mixing_kernel_ders =
										two_layer_edge_radial_mixing_kernel_der_cache_.data()
										+ edge * radial_mixing_kernel_stride;
								}
						} else {
							cached_sh_values =
								grad_neighbor_sh_values_cache_.data()
								+ static_cast<size_t>(j) * sh_count;
							cached_sh_ders =
								grad_neighbor_sh_ders_cache_.data()
								+ static_cast<size_t>(j) * 3 * sh_count;
							cached_radial_values =
								grad_neighbor_mu_contract_vals_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							cached_radial_derivatives =
								grad_neighbor_mu_contract_ders_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
						}
								int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
								const double* gate_signal_by_mu =
									PrepareTwoLayerGateEdgeL1NeighborSignals(
										nbh, j, TwoLayerGateNeighborSignals(nbh, j),
										cached_sh_values,
										gate_neighbor_atom_index_for_tanh);
								const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
								const double* center_gate_signal_by_mu =
									TwoLayerGateAtomSignals(gate_center_atom_index);
								PrepareTwoLayerGatePairMuBuffers(
									type_central, type_outer,
									center_gate_signal_by_mu, gate_center_atom_index,
									gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
									kGateMuBufferTypeScale);
							const double* gate_type_scale_by_mu =
								two_layer_gate_type_scale_mu_buffer_.data();
							double* gate_scaled_radial_by_mu =
								two_layer_gate_scaled_mu_vals_buffer_.data();
							double* gate_scaled_der_by_mu =
								two_layer_gate_scaled_mu_ders_buffer_.data();
							for (int mu = 0; !pure_matrix && mu < radial_func_count; ++mu) {
								const double type_scale = gate_type_scale_by_mu[mu];
								gate_scaled_radial_by_mu[mu] =
									type_scale * cached_radial_values[mu];
								gate_scaled_der_by_mu[mu] =
									type_scale * cached_radial_derivatives[mu];
							}

							for (int i = 0; !pure_matrix && i < alpha_index_basic_count; ++i) {
								const int mu = basic_mu_cache_[i];
							const int sh_index = basic_sh_index_cache_[i];
							const int sh_der_index = basic_sh_der_index_cache_[i];
							const double radial_value = gate_scaled_radial_by_mu[mu];
							const double radial_der = gate_scaled_der_by_mu[mu];
							const double y = cached_sh_values[sh_index];
								const double wdy =
									se_weight_vec[0] * cached_sh_ders[sh_der_index]
									+ se_weight_vec[1] * cached_sh_ders[sh_der_index + 1]
									+ se_weight_vec[2] * cached_sh_ders[sh_der_index + 2];
									grad_dloss_dsenders_[i] += radial_der * wr * y + radial_value * wdy;
								}
								if (radial_mixing_active) {
									const double* cached_rb_vals = use_edge_cache_for_gradient
										? two_layer_edge_radial_vals_cache_.data()
											+ TwoLayerEdgePrimitiveOffset(-1, j) * radial_val_stride
										: grad_neighbor_radial_vals_cache_.data()
											+ static_cast<size_t>(j) * radial_val_stride;
									const double* cached_rb_ders = use_edge_cache_for_gradient
										? two_layer_edge_radial_ders_cache_.data()
											+ TwoLayerEdgePrimitiveOffset(-1, j) * radial_der_stride
									: grad_neighbor_radial_ders_cache_.data()
										+ static_cast<size_t>(j) * radial_der_stride;
								for (int l = 0; l <= sh_l_max_; ++l) {
										const size_t rank_cache_offset =
											(static_cast<size_t>(j)
											 * static_cast<size_t>(sh_l_max_ + 1)
											 + static_cast<size_t>(l))
											* static_cast<size_t>(sh_k_max_);
										double* gate_input =
											radial_mixing_tangent_input_cache.data()
											+ rank_cache_offset;
										double* gate_input_t =
											radial_mixing_tangent_input_t_cache.data()
											+ rank_cache_offset;
										std::fill(gate_input,
										          gate_input + sh_k_max_, 0.0);
										std::fill(gate_input_t,
										          gate_input_t + sh_k_max_, 0.0);
										for (int k = 0; k < sh_k_max_; ++k) {
											const int mu = k * (sh_l_max_ + 1) + l;
											const double type_scale = gate_type_scale_by_mu[mu];
											gate_input[k] = type_scale;
											gate_input_t[k] = 0.0;
										}
										for (int a = 0; a < outer_k_rank_; ++a) {
											for (int k = 0; k < sh_k_max_; ++k) {
											const int mu = k * (sh_l_max_ + 1) + l;
											const int mixing_radial_base =
												mu_to_radial_eval_block_[mu] * R;
											const int mixing_der_base = 5 * mixing_radial_base;
											const int ac = a * sh_k_max_ + k;
											const size_t lac =
												(static_cast<size_t>(l) * outer_k_rank_ + a)
												* sh_k_max_ + k;
											double kernel;
											double kernel_t;
											if (cached_radial_mixing_kernels != nullptr) {
												kernel = cached_radial_mixing_kernels[lac];
												kernel_t = cached_radial_mixing_kernel_ders[lac] * wr;
											} else {
												const double* coeff_row = radial_mixing_coeffs
													+ lac * static_cast<size_t>(R);
												kernel = 0.0;
												kernel_t = 0.0;
												for (int xi = 0; xi < R; ++xi) {
													kernel += coeff_row[xi]
														* cached_rb_vals[mixing_radial_base + xi];
													kernel_t += coeff_row[xi]
														* cached_rb_ders[mixing_der_base + xi] * wr;
												}
											}
												radial_mixing_tangent_kernel[ac] = kernel;
											radial_mixing_tangent_kernel_t[ac] = kernel_t;
										}
									}
									for (int a = 0; a < outer_k_rank_; ++a) {
										double outer_value = 0.0;
										double outer_t = 0.0;
										for (int k = 0; k < sh_k_max_; ++k) {
											const int ac = a * sh_k_max_ + k;
											outer_value +=
											radial_mixing_tangent_kernel[ac] * gate_input[k];
										outer_t +=
											radial_mixing_tangent_kernel_t[ac] * gate_input[k]
											+ radial_mixing_tangent_kernel[ac] * gate_input_t[k];
										}
										radial_mixing_tangent_outer[a] = outer_value;
										radial_mixing_tangent_outer_t[a] = outer_t;
									}
									for (int m = -l; m <= l; ++m) {
											const int sh_index = SHFlatIndex(l, m);
											const int sh_der_index = 3 * sh_index;
											const double y = cached_sh_values[sh_index];
											const double wdy =
												se_weight_vec[0] * cached_sh_ders[sh_der_index]
												+ se_weight_vec[1] * cached_sh_ders[sh_der_index + 1]
												+ se_weight_vec[2] * cached_sh_ders[sh_der_index + 2];
										for (int a = 0; a < outer_k_rank_; ++a) {
											const double delta_t =
												radial_mixing_tangent_outer_t[a] * y
												+ radial_mixing_tangent_outer[a] * wdy;
											radial_mixing_tangent_delta[
												static_cast<size_t>(a) * sh_count
												+ SHFlatIndex(l, m)] += delta_t;
											}
										}
									}
								}
					}
					profile_after_force_cache = profile_accum ? SHAccumProfileNow() : 0.0;
					ContractOuterKRank1Values(grad_dloss_dsenders_.data());
					if (radial_mixing_active)
						for (int a = 0; a < outer_k_rank_; ++a)
							for (int l = 0; l <= sh_l_max_; ++l)
								for (int m = -l; m <= l; ++m)
									grad_dloss_dsenders_[
										OuterKRank1OuterBasicIndex(a, l, m)] +=
										radial_mixing_tangent_delta[
											static_cast<size_t>(a) * sh_count
											+ SHFlatIndex(l, m)];

						if (use_product_rows) {
							AccumulateSHProductRowsForward(grad_dloss_dsenders_, grad_dloss_dsenders_);
				} else {
					for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
					     p < sh_products_.size(); ++p) {
						const SHProduct& product = sh_products_[p];
						grad_dloss_dsenders_[product.target] += product.coeff * (
							grad_dloss_dsenders_[product.left] * moment_vals[product.right]
							+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
						}
					}
					profile_after_tangent_forward = profile_accum ? SHAccumProfileNow() : 0.0;

					if (use_product_rows) {
						for (const SHProductRow& row : sh_product_rows_) {
						const double adj_target = site_energy_ders_wrt_moments_[row.target];
						if (adj_target == 0.0)
							continue;
						const int end = row.term_begin + row.term_count;
						for (int t = row.term_begin; t < end; ++t) {
							const SHProductRowTerm& term = sh_product_row_terms_[t];
							grad_dloss_dmom_[term.left] +=
								grad_dloss_dsenders_[term.right] * adj_target * term.coeff;
							grad_dloss_dmom_[term.right] +=
								grad_dloss_dsenders_[term.left] * adj_target * term.coeff;
						}
					}
				} else if (!use_product_hvt_reverse) {
					for (size_t p = static_cast<size_t>(OuterKRank1ProductStart());
					     p < sh_products_.size(); ++p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = site_energy_ders_wrt_moments_[product.target];
						if (adj_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] +=
							grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
						grad_dloss_dmom_[product.right] +=
							grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
					}
				}
				profile_after_mixed_seed = profile_accum ? SHAccumProfileNow() : 0.0;
			}

		if (outer_k_rank1_enabled_) {
			for (int i = 0; i < outer_k_rank1_basic_count_; ++i) {
				const int outer = outer_k_rank1_basic_begin_ + i;
				grad_dloss_dmom_[outer] +=
					se_weight * site_energy_ders_wrt_moments_[outer];
			}
		} else {
			for (int i = 0; i < alpha_index_basic_count; ++i)
				grad_dloss_dmom_[i] +=
					se_weight * site_energy_ders_wrt_moments_[i];
		}

			if (se_ders_weights != nullptr) {
				if (use_product_rows) {
					BackpropSHProductRows(grad_dloss_dmom_);
				} else if (use_product_hvt_reverse) {
					for (int p = static_cast<int>(sh_products_.size()) - 1;
					     p >= OuterKRank1ProductStart(); --p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = site_energy_ders_wrt_moments_[product.target];
						const double hvt_target = grad_dloss_dmom_[product.target];
						if (adj_target == 0.0 && hvt_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] += product.coeff * (
							moment_vals[product.right] * hvt_target
							+ grad_dloss_dsenders_[product.right] * adj_target);
						grad_dloss_dmom_[product.right] += product.coeff * (
							moment_vals[product.left] * hvt_target
							+ grad_dloss_dsenders_[product.left] * adj_target);
					}
				} else {
					for (int p = static_cast<int>(sh_products_.size()) - 1;
					     p >= OuterKRank1ProductStart(); --p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = grad_dloss_dmom_[product.target];
						if (adj_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] +=
							product.coeff * moment_vals[product.right] * adj_target;
						grad_dloss_dmom_[product.right] +=
							product.coeff * moment_vals[product.left] * adj_target;
					}
				}
			}
		if (OuterKRankEnabled()) {
			if (OuterKRankHasTrainableWeights()) {
				for (int channel = 0; channel < outer_k_rank_; ++channel) {
					for (int l = sh_l_max_; l >= 0; --l) {
						for (int m = -l; m <= l; ++m) {
							const int outer =
								OuterKRank1OuterBasicIndex(channel, l, m);
							const double total_outer_adjoint = grad_dloss_dmom_[outer];
							const double energy_outer_adjoint =
								site_energy_ders_wrt_moments_[outer];
							for (int k = 0; k < sh_k_max_; ++k) {
								const int raw = OuterKRank1RawBasicIndex(l, m, k);
								double weight_grad =
									total_outer_adjoint * moment_vals[raw];
								if (se_ders_weights != nullptr)
									weight_grad += energy_outer_adjoint
										* grad_dloss_dsenders_[raw];
								grad_out[
									OuterKRank1WeightIndex(channel, l, k)] +=
									center_linear * weight_grad;
							}
						}
					}
				}
			}
			BackpropOuterKRank1Adjoints(grad_dloss_dmom_);
		}
		const double profile_after_force_product = profile_accum ? SHAccumProfileNow() : 0.0;

		buff_site_energy_ += regression_coeffs[type_central] + center_linear + center_linear * scalar_sum;
		buff_site_energy_0 += center_linear * scalar_sum;
		if (shift_)
			grad_out[type_central] += se_weight;

		double center_linear_force_grad = 0.0;
		if (se_ders_weights != nullptr) {
			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const int node = alpha_moment_mapping[i];
				const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
				center_linear_force_grad += moment_coeff * grad_dloss_dsenders_[node];
			}
		}
		grad_out[coeff_count + type_central] +=
			se_weight * (1.0 + scalar_sum) + center_linear_force_grad;
		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int node = alpha_moment_mapping[i];
			const double sender = (se_ders_weights == nullptr) ? 0.0 : grad_dloss_dsenders_[node];
			grad_out[coeff_count + species_count + i] +=
				center_linear * linear_mults[i] * (se_weight * moment_vals[node] + sender);
		}
		const double profile_after_scalar_grad = profile_accum ? SHAccumProfileNow() : 0.0;

		if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
			grad_radial_coeff_value_accum_.resize(radial_func_count);
			if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
				grad_radial_coeff_coord_accum_.resize(radial_func_count);
			std::vector<double> gate_adjoint_by_mu;
			std::vector<double> center_gate_adjoint_by_mu;
			std::vector<double>& radial_mixing_contribution_by_mu =
				sh_combination_radial_mixing_value_adj_scratch_;
			std::vector<double>& radial_mixing_input_by_k =
				sh_combination_radial_mixing_param_input_scratch_;
			std::vector<double>& radial_mixing_param_kernel =
				sh_combination_radial_mixing_param_kernel_scratch_;
			std::vector<double>& radial_mixing_param_kernel_t =
				sh_combination_radial_mixing_param_kernel_t_scratch_;
			std::vector<double>& radial_mixing_param_kernel_s =
				sh_combination_radial_mixing_param_kernel_s_scratch_;
		std::vector<double>& radial_mixing_param_kernel_coord_s =
			sh_combination_radial_mixing_param_kernel_coord_s_scratch_;
			std::vector<double>& radial_mixing_param_kernel_ss =
				sh_combination_radial_mixing_param_kernel_ss_scratch_;
			std::vector<double>& radial_mixing_param_kernel_coord_ss =
				sh_combination_radial_mixing_param_kernel_coord_ss_scratch_;
			if (radial_mixing_enabled)
				radial_mixing_input_by_k.resize(sh_k_max_);
			if (radial_mixing_active) {
				radial_mixing_contribution_by_mu.assign(radial_func_count, 0.0);
				const int ac_count = outer_k_rank_ * sh_k_max_;
				radial_mixing_param_kernel.resize(pure_matrix ? std::max(ac_count, 2 * outer_k_rank_) : ac_count);
				radial_mixing_param_kernel_t.resize(ac_count);
				radial_mixing_param_kernel_s.resize(pure_matrix ? std::max(ac_count, 2 * R) : ac_count);
				radial_mixing_param_kernel_coord_s.resize(ac_count);
				radial_mixing_param_kernel_ss.resize(ac_count);
				radial_mixing_param_kernel_coord_ss.resize(ac_count);
			}

			for (int j = 0; j < nbh.count; ++j) {
				const Vector3& rvec = nbh.vecs[j];
				const double r = nbh.dists[j];
				const double inv_r = 1.0 / r;
				const int type_outer = nbh.types[j];
				const int center_type_coeff_index_base =
					OuterTypeCoeffIndexFast(outer_type_coeff_offset,
					                        outer_type_coeff_stride,
					                        R,
					                        type_central,
					                        0);
				const int outer_type_coeff_index_base =
					OuterTypeCoeffIndexFast(outer_type_coeff_offset,
					                        outer_type_coeff_stride,
					                        R,
					                        type_outer,
					                        0);
				const int additive_outer_index_base =
					GateAdditiveCoeffIndexFast(additive_coeff_offset,
					                           radial_func_count,
					                           type_outer,
					                           0);
				const int additive_center_index_base =
					GateAdditiveCoeffIndexFast(additive_coeff_offset,
					                           radial_func_count,
					                           type_central,
					                           0);
				const double wx = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][0];
				const double wy = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][1];
				const double wz = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][2];
				const double wr = (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r;
				const double* sh_values_use = sh_values;
				const double* sh_ders_use = sh_ders;
				const double* rb_vals_use = rb_vals.data();
				const double* rb_ders_use = rb_ders.data();
				const double* radial_values_use = radial_values.data();
				const double* radial_derivatives_use = radial_derivatives.data();
				const double* cached_radial_mixing_kernels = nullptr;
				const double* cached_radial_mixing_kernel_ders = nullptr;
					const double* radial_s_derivatives_use = radial_s_derivatives.data();
					const double* radial_ss_derivatives_use = radial_ss_derivatives.data();
					const double* radial_coord_s_derivatives_use = radial_coord_s_derivatives.data();
					const double* radial_coord_ss_derivatives_use = radial_coord_ss_derivatives.data();
					if (se_ders_weights != nullptr) {
						if (use_edge_cache_for_gradient) {
							const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							if (radial_mixing_active) {
								cached_radial_mixing_kernels =
									two_layer_edge_radial_mixing_kernel_cache_.data()
									+ edge * radial_mixing_kernel_stride;
								cached_radial_mixing_kernel_ders =
									two_layer_edge_radial_mixing_kernel_der_cache_.data()
									+ edge * radial_mixing_kernel_stride;
							}
							sh_values_use =
								two_layer_edge_sh_values_cache_.data() + edge * sh_count;
							sh_ders_use =
								two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							rb_vals_use =
								two_layer_edge_radial_vals_cache_.data()
								+ edge * radial_val_stride;
							rb_ders_use =
								two_layer_edge_radial_ders_cache_.data()
								+ edge * radial_der_stride;
							radial_values_use =
								two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
							radial_derivatives_use =
								two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
							radial_s_derivatives_use =
								two_layer_edge_mu_ders_s_cache_.data() + edge * mu_stride;
							radial_coord_s_derivatives_use =
								two_layer_edge_mu_coord_ders_s_cache_.data()
								+ edge * mu_stride;
							radial_ss_derivatives_use =
								two_layer_edge_mu_ders_ss_cache_.data() + edge * mu_stride;
							radial_coord_ss_derivatives_use =
								two_layer_edge_mu_coord_ders_ss_cache_.data()
								+ edge * mu_stride;
						} else {
							sh_values_use =
								grad_neighbor_sh_values_cache_.data()
								+ static_cast<size_t>(j) * sh_count;
							sh_ders_use =
								grad_neighbor_sh_ders_cache_.data()
								+ static_cast<size_t>(j) * 3 * sh_count;
							rb_vals_use =
								grad_neighbor_radial_vals_cache_.data()
								+ static_cast<size_t>(j) * radial_val_stride;
							rb_ders_use =
								grad_neighbor_radial_ders_cache_.data()
								+ static_cast<size_t>(j) * radial_der_stride;
							radial_values_use =
								grad_neighbor_mu_contract_vals_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_derivatives_use =
								grad_neighbor_mu_contract_ders_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_s_derivatives_use =
								grad_neighbor_mu_contract_ders_s_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_coord_s_derivatives_use =
								grad_neighbor_mu_contract_coord_ders_s_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_ss_derivatives_use =
								grad_neighbor_mu_contract_ders_ss_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_coord_ss_derivatives_use =
								grad_neighbor_mu_contract_coord_ders_ss_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
						}
					} else {
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_s = 0.0;
				double dot_coord_s = 0.0;
				double dot_ss = 0.0;
				double dot_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					dot_val += coeff * rb_vals[radial_base + xi];
					dot_der += coeff * rb_ders[deriv_base + xi];
					dot_s += coeff * rb_ders[deriv_base + xi + R];
					dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
					dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					dot_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
				}
				radial_values[mu] = dot_val;
				radial_derivatives[mu] = dot_der;
				radial_s_derivatives[mu] = dot_s;
				radial_coord_s_derivatives[mu] = dot_coord_s;
					radial_ss_derivatives[mu] = dot_ss;
					radial_coord_ss_derivatives[mu] = dot_coord_ss;
				}
				}
							int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
							const double* gate_signal_by_mu =
								PrepareTwoLayerGateEdgeL1NeighborSignals(
									nbh, j, TwoLayerGateNeighborSignals(nbh, j),
									sh_values_use, gate_neighbor_atom_index_for_tanh);
							const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
							const double* center_gate_signal_by_mu =
								TwoLayerGateAtomSignals(gate_center_atom_index);
							PrepareTwoLayerGatePairMuBuffers(
								type_central, type_outer,
								center_gate_signal_by_mu, gate_center_atom_index,
								gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
								kGateMuBufferAdditive | kGateMuBufferTypeScale
								| kGateMuBufferMultiplier | kGateMuBufferOuter
									| kGateMuBufferAdditiveParam);
						const double* gate_additive_by_mu =
							two_layer_gate_additive_mu_buffer_.data();
						const double* center_gate_additive_by_mu =
							two_layer_gate_center_additive_mu_buffer_.data();
						const double* center_gate_additive_param_by_mu =
							two_layer_gate_center_additive_param_mu_buffer_.data();
						const double* neighbor_gate_multiplier_by_mu =
							two_layer_gate_neighbor_multiplier_mu_buffer_.data();
					const double* gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
					const double* gate_multiplier_by_mu =
						two_layer_gate_multiplier_mu_buffer_.data();
					const double* gate_outer_by_mu =
						two_layer_gate_outer_mu_buffer_.data();
					const double* gate_additive_param_by_mu =
						two_layer_gate_additive_param_mu_buffer_.data();
							const bool skip_outer_param_grad =
									two_layer_residual_skip_outer_param_grad_;
									gate_adjoint_by_mu.assign(radial_func_count, 0.0);
									center_gate_adjoint_by_mu.assign(radial_func_count, 0.0);
							const bool need_edge_l1_coord_weight_grad =
								se_ders_weights != nullptr
								&& TwoLayerGateEdgeL1Enabled();
							if (need_edge_l1_coord_weight_grad)
								two_layer_gate_edge_l1_energy_adjoint_buffer_.assign(
									radial_func_count, 0.0);
				double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
			double* radial_coeff_coord_accum = grad_radial_coeff_coord_accum_.data();
				std::fill(radial_coeff_value_accum,
				          radial_coeff_value_accum + radial_func_count,
				          0.0);
				std::fill(radial_coeff_coord_accum,
				          radial_coeff_coord_accum + radial_func_count,
				          0.0);
					if (radial_mixing_active) {
					std::fill(radial_mixing_contribution_by_mu.begin(),
					          radial_mixing_contribution_by_mu.end(), 0.0);
					}

						if (radial_mixing_enabled) {
							for (int l = 0; l <= sh_l_max_; ++l) {
									double* gate_input = radial_mixing_input_by_k.data();
										for (int k = 0; k < sh_k_max_; ++k) {
											const int mu = k * (sh_l_max_ + 1) + l;
										gate_input[k] = gate_type_scale_by_mu[mu];
										}
								if (pure_matrix) {
									double* value_weights = radial_mixing_param_kernel.data();
									double* coord_weights = value_weights + outer_k_rank_;
									for (int o = 0; o < outer_k_rank_; ++o) {
										double value_sum = 0.0;
										double energy_sum = 0.0;
										double angular_sum = 0.0;
										for (int m = -l; m <= l; ++m) {
											const int sh_index = SHFlatIndex(l, m);
											const int outer = OuterKRank1OuterBasicIndex(o, l, m);
											const double y = sh_values_use[sh_index];
											value_sum += center_linear * grad_dloss_dmom_[outer] * y;
											if (se_ders_weights != nullptr) {
												const double energy_adj = center_linear * site_energy_ders_wrt_moments_[outer];
												energy_sum += energy_adj * y;
												angular_sum += energy_adj * (wx * sh_ders_use[3 * sh_index]
													+ wy * sh_ders_use[3 * sh_index + 1] + wz * sh_ders_use[3 * sh_index + 2]);
											}
										}
										value_weights[o] = value_sum + angular_sum;
										coord_weights[o] = wr * energy_sum;
									}
									if (se_ders_weights != nullptr)
										AccumulatePureWEdgeParameterGrad<true, false>(
											l, sh_l_max_ + 1, sh_k_max_, outer_k_rank_, R, C,
											type_central * C + type_outer, mu_to_radial_eval_block_.data(),
											radial_eval_to_scaling_block_.data(), radial_mixing_coeffs, radial_mixing_coeff_offset,
											rb_vals_use, rb_ders_use, value_weights, coord_weights, nullptr,
											gate_input, nullptr, cached_radial_mixing_kernels, cached_radial_mixing_kernel_ders,
											grad_out, radial_mixing_contribution_by_mu.data(), nullptr, radial_mixing_param_kernel_s.data());
									else
										AccumulatePureWEdgeParameterGrad<false, false>(
											l, sh_l_max_ + 1, sh_k_max_, outer_k_rank_, R, C,
											type_central * C + type_outer, mu_to_radial_eval_block_.data(),
											radial_eval_to_scaling_block_.data(), radial_mixing_coeffs, radial_mixing_coeff_offset,
											rb_vals_use, rb_ders_use, value_weights, nullptr, nullptr,
											gate_input, nullptr, cached_radial_mixing_kernels, nullptr,
											grad_out, radial_mixing_contribution_by_mu.data(), nullptr, radial_mixing_param_kernel_s.data());
									continue;
								}
									if (radial_mixing_active) {
										for (int a = 0; a < outer_k_rank_; ++a) {
											for (int k = 0; k < sh_k_max_; ++k) {
										const int mu = k * (sh_l_max_ + 1) + l;
										const int mixing_radial_base =
											mu_to_radial_eval_block_[mu] * R;
										const int mixing_der_base = 5 * mixing_radial_base;
										const int ac = a * sh_k_max_ + k;
										const size_t lac =
											(static_cast<size_t>(l) * outer_k_rank_ + a)
											* sh_k_max_ + k;
										const double* coeff_row = radial_mixing_coeffs
											+ lac * static_cast<size_t>(R);
										double kernel = cached_radial_mixing_kernels != nullptr
											? cached_radial_mixing_kernels[lac]
											: 0.0;
										double kernel_t =
											cached_radial_mixing_kernel_ders != nullptr
											? cached_radial_mixing_kernel_ders[lac] * wr
											: 0.0;
												double kernel_s = 0.0;
												double kernel_coord_s = 0.0;
												double kernel_ss = 0.0;
												double kernel_coord_ss = 0.0;
												for (int xi = 0; xi < R; ++xi) {
													const double coeff = coeff_row[xi];
											if (cached_radial_mixing_kernels == nullptr)
												kernel += coeff
													* rb_vals_use[mixing_radial_base + xi];
											if (se_ders_weights != nullptr
											    && cached_radial_mixing_kernel_ders == nullptr)
												kernel_t += coeff
															* rb_ders_use[mixing_der_base + xi] * wr;
													kernel_s += coeff
														* rb_ders_use[mixing_der_base + R + xi];
													kernel_ss += coeff
														* rb_ders_use[mixing_der_base + 3 * R + xi];
													if (se_ders_weights != nullptr) {
														kernel_coord_s += coeff
															* rb_ders_use[mixing_der_base + 2 * R + xi];
														kernel_coord_ss += coeff
															* rb_ders_use[mixing_der_base + 4 * R + xi];
													}
												}
												radial_mixing_param_kernel[ac] = kernel;
												radial_mixing_param_kernel_t[ac] = kernel_t;
												radial_mixing_param_kernel_s[ac] = kernel_s;
												radial_mixing_param_kernel_coord_s[ac] =
													kernel_coord_s;
												radial_mixing_param_kernel_ss[ac] = kernel_ss;
												radial_mixing_param_kernel_coord_ss[ac] =
													kernel_coord_ss;
											}
										}
								}

									// Contract the m adjoints once before radial and gate
									// parameter propagation; this is an exact reassociation.
									for (int a = 0; a < outer_k_rank_; ++a) {
										double sum_adj_y = 0.0;
										double sum_energy_y = 0.0;
										double sum_energy_wdy = 0.0;
										for (int m = -l; m <= l; ++m) {
											const int sh_index = SHFlatIndex(l, m);
											const int sh_der_index = 3 * sh_index;
											const int outer = OuterKRank1OuterBasicIndex(a, l, m);
											const double y = sh_values_use[sh_index];
											const double wdy =
												(se_ders_weights == nullptr)
												? 0.0
												: (wx * sh_ders_use[sh_der_index]
												   + wy * sh_ders_use[sh_der_index + 1]
												   + wz * sh_ders_use[sh_der_index + 2]);
											const double adj = center_linear * grad_dloss_dmom_[outer];
											const double energy_adj =
												(se_ders_weights == nullptr)
												? 0.0
												: center_linear * site_energy_ders_wrt_moments_[outer];
											sum_adj_y += adj * y;
											sum_energy_y += energy_adj * y;
											sum_energy_wdy += energy_adj * wdy;
										}
										for (int k = 0; k < sh_k_max_; ++k) {
											const int mu = k * (sh_l_max_ + 1) + l;
											const int mixing_radial_base =
												mu_to_radial_eval_block_[mu] * R;
											const int mixing_der_base = 5 * mixing_radial_base;
											const int mixing_scaling_block =
												radial_eval_to_scaling_block_[
													mu_to_radial_eval_block_[mu]];
											const int mixing_sigma_coeff_offset =
												C + 2 * C * C * mixing_scaling_block
												+ type_central * C + type_outer;
										const double input_value_factor =
										gate_input[k] * sum_adj_y;
									const double input_energy_factor =
										gate_input[k] * sum_energy_y;
					const double input_direction_factor =
						gate_input[k] * sum_energy_wdy;
							const int mixing_coeff_base =
								radial_mixing_coeff_offset
								+ (((l * outer_k_rank_ + a)
								    * sh_k_max_ + k) * R);
							for (int xi = 0; xi < R; ++xi) {
												const double rb = rb_vals_use[mixing_radial_base + xi];
												const double rb_t =
													(se_ders_weights == nullptr)
													? 0.0
													: rb_ders_use[mixing_der_base + xi] * wr;
											grad_out[mixing_coeff_base + xi] +=
												rb * (input_value_factor + input_direction_factor)
												+ rb_t * input_energy_factor;
											}
											if (!radial_mixing_active)
												continue;
											const int ac = a * sh_k_max_ + k;
											grad_out[mixing_sigma_coeff_offset] +=
												radial_mixing_param_kernel_s[ac]
													* (input_value_factor + input_direction_factor)
												+ radial_mixing_param_kernel_coord_s[ac] * wr
													* input_energy_factor;
						grad_out[mixing_sigma_coeff_offset + C * C] +=
							radial_mixing_param_kernel_ss[ac]
								* (input_value_factor + input_direction_factor)
							+ radial_mixing_param_kernel_coord_ss[ac] * wr
								* input_energy_factor;
										radial_mixing_contribution_by_mu[mu] +=
											radial_mixing_param_kernel[ac]
												* (sum_adj_y + sum_energy_wdy)
											+ radial_mixing_param_kernel_t[ac]
												* sum_energy_y;
										}
									}
								}
								}

						for (int i = 0; !pure_matrix && i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int sh_index = basic_sh_index_cache_[i];
						const int sh_der_index = basic_sh_der_index_cache_[i];
						const double y = sh_values_use[sh_index];
					double dy[3];
					dy[0] = sh_ders_use[sh_der_index];
					dy[1] = sh_ders_use[sh_der_index + 1];
					dy[2] = sh_ders_use[sh_der_index + 2];
					const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];

					const double dot_val = radial_values_use[mu];
					const double dot_der = radial_derivatives_use[mu];
					const double dot_s = radial_s_derivatives_use[mu];
						const double dot_coord_s = radial_coord_s_derivatives_use[mu];
							const double dot_ss = radial_ss_derivatives_use[mu];
							const double dot_coord_ss = radial_coord_ss_derivatives_use[mu];
							const double center_type_coeff =
								regression_coeffs[
									center_type_coeff_index_base
									+ mu * outer_type_coeff_stride];

							const double value_weight =
								center_linear * grad_dloss_dmom_[i];
							const double coord_weight = (se_ders_weights == nullptr)
								? 0.0
								: center_linear * site_energy_ders_wrt_moments_[i];
								const double energy_der_weight =
									center_linear * site_energy_ders_wrt_moments_[i];
						if (need_edge_l1_coord_weight_grad && energy_der_weight != 0.0)
							two_layer_gate_edge_l1_energy_adjoint_buffer_[mu] +=
								energy_der_weight * center_type_coeff
								* gate_additive_by_mu[mu] * dot_val * y;

						const double base_direction = dot_der * wr * y + dot_val * wdy;
					const double final_contribution_weight =
						value_weight * dot_val * y
						+ coord_weight * base_direction;
						radial_coeff_value_accum[mu] +=
							value_weight * y + coord_weight * wdy;
							radial_coeff_coord_accum[mu] += coord_weight * y;

						if (!skip_site_der_output && energy_der_weight != 0.0) {
						const double type_scale = gate_type_scale_by_mu[mu];
							for (int a = 0; a < 3; ++a) {
							const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
							buff_site_energy_ders_[j][a] += energy_der_weight * type_scale * base_coord;
								}
							}
						}
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double value_accum = radial_coeff_value_accum[mu];
						const double coord_accum = radial_coeff_coord_accum[mu];
						const double mixing_contribution = radial_mixing_active
							? radial_mixing_contribution_by_mu[mu] : 0.0;
									if (value_accum == 0.0 && coord_accum == 0.0
									    && mixing_contribution == 0.0)
										continue;
										const int radial_base = mu_to_radial_eval_block_[mu] * R;
										const int deriv_base = 5 * radial_base;
										const int radial_offset = radial_coeff_base + mu * (R + C);
										const double center_type_coeff =
											regression_coeffs[
												center_type_coeff_index_base
												+ mu * outer_type_coeff_stride];
										const double outer_type_coeff =
											regression_coeffs[
												outer_type_coeff_index_base
												+ mu * outer_type_coeff_stride];
										const int center_type_coeff_index =
											center_type_coeff_index_base
											+ mu * outer_type_coeff_stride;
										const int outer_type_coeff_index =
											outer_type_coeff_index_base
											+ mu * outer_type_coeff_stride;
									const double gate_df = gate_additive_by_mu[mu];
								const double gate_da = gate_additive_param_by_mu[mu];
										const double mu_contribution =
										radial_values_use[mu] * value_accum
										+ radial_derivatives_use[mu] * wr * coord_accum
										+ mixing_contribution;
								gate_adjoint_by_mu[mu] +=
									center_type_coeff * gate_df * mu_contribution;
								if (TwoLayerGateUsesCenterGate())
									center_gate_adjoint_by_mu[mu] +=
										center_type_coeff * outer_type_coeff
										* neighbor_gate_multiplier_by_mu[mu]
										* center_gate_additive_by_mu[mu]
										* mu_contribution;
									if (two_layer_gate_enabled_)
										grad_out[additive_outer_index_base + mu] +=
											center_type_coeff * gate_da * mu_contribution;
									if (TwoLayerGateUsesCenterGate())
										grad_out[additive_center_index_base + mu] +=
											center_type_coeff * outer_type_coeff
										* neighbor_gate_multiplier_by_mu[mu]
										* center_gate_additive_param_by_mu[mu]
										* mu_contribution;
								if (skip_outer_param_grad)
									continue;
								const double type_scale = gate_type_scale_by_mu[mu];
								const double gate_multiplier = gate_multiplier_by_mu[mu];
								const double gate_outer = gate_outer_by_mu[mu];
								const int scaling_block =
									radial_eval_to_scaling_block_[mu_to_radial_eval_block_[mu]];
								const int sigma_coeff_offset =
									C + 2 * C * C * scaling_block
									+ type_central * C + type_outer;
										grad_out[center_type_coeff_index] +=
											gate_outer * mu_contribution;
										grad_out[outer_type_coeff_index] +=
											center_type_coeff * mu_contribution;
								if (pure_matrix)
									continue;
								grad_out[sigma_coeff_offset] += type_scale * (
									radial_s_derivatives_use[mu] * value_accum
									+ radial_coord_s_derivatives_use[mu] * wr * coord_accum);
								grad_out[sigma_coeff_offset + C * C] += type_scale * (
									radial_ss_derivatives_use[mu] * value_accum
									+ radial_coord_ss_derivatives_use[mu] * wr * coord_accum);
								for (int xi = 0; xi < R; ++xi) {
								grad_out[radial_offset + xi] += type_scale * (
									rb_vals_use[radial_base + xi] * value_accum
								+ rb_ders_use[deriv_base + xi] * wr * coord_accum);
						}
					}
					for (int mu = 0; mu < radial_func_count; ++mu) {
						AddTwoLayerGateMuAdjoint(nbh, j, mu, gate_adjoint_by_mu[mu]);
						if (TwoLayerGateUsesCenterGate())
							AddTwoLayerGateAtomMuAdjoint(
								gate_center_atom_index, mu,
								center_gate_adjoint_by_mu[mu]);
					}
						AccumulateTwoLayerGateEdgeL1WeightGradForPair(
							nbh, j, sh_values_use, gate_adjoint_by_mu.data(),
							out_grad_accumulator);
						AccumulateTwoLayerGateEdgeL1SourceAdjointsForPair(
							nbh, j, sh_values_use, gate_adjoint_by_mu.data(),
							two_layer_gate_edge_l1_adjoints_);
						if (need_edge_l1_coord_weight_grad)
							AccumulateTwoLayerGateEdgeL1WeightCoordGradForPair(
								nbh,
								j,
								sh_values_use,
								sh_ders_use,
								two_layer_gate_edge_l1_energy_adjoint_buffer_.data(),
								&se_ders_weights[j],
								out_grad_accumulator);
						if (need_edge_l1_coord_weight_grad)
							AccumulateTwoLayerGateEdgeL1SourceCoordAdjointsForPair(
								nbh,
								j,
								sh_ders_use,
								two_layer_gate_edge_l1_energy_adjoint_buffer_.data(),
								&se_ders_weights[j],
								two_layer_gate_edge_l1_adjoints_);
						}
		if (profile_accum) {
			const double profile_end = SHAccumProfileNow();
			RecordSHAccumProfile(
				profile_end - profile_start,
				profile_after_moment - profile_start,
				profile_after_energy_backprop - profile_after_moment,
				profile_after_force_product - profile_after_energy_backprop,
				profile_after_force_cache - profile_after_energy_backprop,
				profile_after_tangent_forward - profile_after_force_cache,
				profile_after_mixed_seed - profile_after_tangent_forward,
				profile_after_force_product - profile_after_mixed_seed,
				profile_after_scalar_grad - profile_after_force_product,
				profile_end - profile_after_scalar_grad,
				nbh.count);
		}
		return;
	}

	CalcSHBasisFuncsDers(nbh);
	buff_site_energy_ = 0.0;
	buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);
	out_grad_accumulator.resize(CoeffCount());
	double* grad_out = out_grad_accumulator.data();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int K = radial_func_count;
	const int coeff_count = LinearCoeffOffset();
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int outer_type_coeff_offset = RadialCoeffOffset();
	const int outer_type_coeff_stride = RadialCoeffBlockSize();
	const int additive_coeff_offset = TwoLayerGateAdditiveCoeffOffset();
	const double center_linear = linear_coeffs[type_central];

	if (static_cast<int>(sh_adj_vals_.size()) != alpha_moments_count)
		sh_adj_vals_.resize(alpha_moments_count);
	std::fill(sh_adj_vals_.begin(), sh_adj_vals_.end(), 0.0);
	if (sh_adj_ders_.size1 != alpha_moments_count || sh_adj_ders_.size2 != nbh.count)
		sh_adj_ders_.resize(alpha_moments_count, nbh.count, 3);
	sh_adj_ders_.set(0);

	double scalar_sum = 0.0;
	double center_linear_force_grad = 0.0;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		scalar_sum += moment_coeff * moment_vals[node];
		sh_adj_vals_[node] += se_weight * center_linear * moment_coeff;
		if (se_ders_weights != nullptr) {
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a) {
					const double weighted_der = moment_coeff * basis_ders(1 + i, j, a);
					center_linear_force_grad += se_ders_weights[j][a] * weighted_der;
					sh_adj_ders_(node, j, a) += center_linear * moment_coeff * se_ders_weights[j][a];
				}
		}
	}

	buff_site_energy_ += regression_coeffs[type_central] + center_linear + center_linear * scalar_sum;
	buff_site_energy_0 += center_linear * scalar_sum;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double coeff = center_linear * linear_coeffs[species_count + i] * linear_mults[i];
		const int node = alpha_moment_mapping[i];
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				buff_site_energy_ders_[j][a] += coeff * moment_ders(node, j, a);
	}

	for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
		const SHProduct& product = sh_products_[p];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		const double adj_target = sh_adj_vals_[product.target];
		if (adj_target != 0.0) {
			sh_adj_vals_[product.left] += product.coeff * right_value * adj_target;
			sh_adj_vals_[product.right] += product.coeff * left_value * adj_target;
		}
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a) {
				const double adj_der = sh_adj_ders_(product.target, j, a);
				if (adj_der == 0.0)
					continue;
				sh_adj_ders_(product.left, j, a) += product.coeff * right_value * adj_der;
				sh_adj_ders_(product.right, j, a) += product.coeff * left_value * adj_der;
				sh_adj_vals_[product.left] += product.coeff * moment_ders(product.right, j, a) * adj_der;
				sh_adj_vals_[product.right] += product.coeff * moment_ders(product.left, j, a) * adj_der;
			}
	}

	if (shift_)
		grad_out[type_central] += se_weight;
	grad_out[coeff_count + type_central] += se_weight * (1.0 + scalar_sum) + center_linear_force_grad;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		double grad = se_weight * center_linear * linear_mults[i] * moment_vals[node];
		if (se_ders_weights != nullptr) {
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					grad += center_linear * linear_mults[i] * se_ders_weights[j][a] * moment_ders(node, j, a);
		}
		grad_out[coeff_count + species_count + i] += grad;
	}

		double sh_values[kMaxSHComponents];
		double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const int center_type_coeff_index_base =
			OuterTypeCoeffIndexFast(outer_type_coeff_offset,
			                        outer_type_coeff_stride,
			                        R,
			                        type_central,
			                        0);
		const int outer_type_coeff_index_base =
			OuterTypeCoeffIndexFast(outer_type_coeff_offset,
			                        outer_type_coeff_stride,
			                        R,
			                        type_outer,
			                        0);
		const int additive_outer_index_base =
			GateAdditiveCoeffIndexFast(additive_coeff_offset,
			                           radial_func_count,
			                           type_outer,
			                           0);
		const int additive_center_index_base =
			GateAdditiveCoeffIndexFast(additive_coeff_offset,
			                           radial_func_count,
			                           type_central,
			                           0);
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

		for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
			const int scaling_block = radial_eval_to_scaling_block_[eval_block];
			const int basis_k = radial_eval_to_basis_k_[eval_block];
			p_RadialBasis->RB_Calc(
				r,
				regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
				regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
				basis_k);
			for (int xi = 0; xi < R; ++xi)
				rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
			for (int xi = 0; xi < R * 5; ++xi)
				rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
		}
						int gate_neighbor_atom_index_for_tanh = nbh.inds[j];
						const double* gate_signal_by_mu =
							PrepareTwoLayerGateEdgeL1NeighborSignals(
								nbh, j, TwoLayerGateNeighborSignals(nbh, j),
								sh_values, gate_neighbor_atom_index_for_tanh);
						const int gate_center_atom_index = active_two_layer_edge_cache_atom_index_;
						const double* center_gate_signal_by_mu =
							TwoLayerGateAtomSignals(gate_center_atom_index);
						PrepareTwoLayerGatePairMuBuffers(
							type_central, type_outer,
							center_gate_signal_by_mu, gate_center_atom_index,
							gate_signal_by_mu, gate_neighbor_atom_index_for_tanh,
							kGateMuBufferAdditive | kGateMuBufferTypeScale
							| kGateMuBufferMultiplier | kGateMuBufferOuter
							| kGateMuBufferAdditiveParam);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();
				const double* gate_multiplier_by_mu =
					two_layer_gate_multiplier_mu_buffer_.data();
				const double* gate_outer_by_mu =
					two_layer_gate_outer_mu_buffer_.data();
					const double* gate_additive_param_by_mu =
						two_layer_gate_additive_param_mu_buffer_.data();
					const double* center_gate_additive_param_by_mu =
						two_layer_gate_center_additive_param_mu_buffer_.data();
					const double* neighbor_gate_multiplier_by_mu =
						two_layer_gate_neighbor_multiplier_mu_buffer_.data();

						for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = basic_mu_cache_[i];
					const double center_type_coeff =
						regression_coeffs[
							center_type_coeff_index_base
							+ mu * outer_type_coeff_stride];
					const double outer_type_coeff =
						regression_coeffs[
							outer_type_coeff_index_base
							+ mu * outer_type_coeff_stride];
					const int center_type_coeff_index =
						center_type_coeff_index_base
						+ mu * outer_type_coeff_stride;
					const int outer_type_coeff_index =
						outer_type_coeff_index_base
						+ mu * outer_type_coeff_stride;
						const int scaling_block = basic_scaling_block_cache_[i];
					const int radial_base = basic_radial_base_cache_[i];
					const int deriv_base = basic_radial_deriv_base_cache_[i];
					const int radial_offset = basic_radial_offset_cache_[i];
				const int sh_index = basic_sh_index_cache_[i];
				const int sh_der_index = basic_sh_der_index_cache_[i];
				const double y = sh_values[sh_index];
			double dy[3];
			for (int a = 0; a < 3; ++a)
				dy[a] = sh_ders[sh_der_index + a];

			double dot_val = 0.0;
			double dot_der = 0.0;
			double dot_s = 0.0;
			double dot_coord_s = 0.0;
			double dot_ss = 0.0;
			double dot_coord_ss = 0.0;
			for (int xi = 0; xi < R; ++xi) {
				const double coeff = regression_coeffs[radial_offset + xi];
				dot_val += coeff * rb_vals[radial_base + xi];
				dot_der += coeff * rb_ders[deriv_base + xi];
				dot_s += coeff * rb_ders[deriv_base + xi + R];
				dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
				dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
				dot_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
			}

						const double adj_value = sh_adj_vals_[i];
						double adj_der[3] = {sh_adj_ders_(i, j, 0), sh_adj_ders_(i, j, 1), sh_adj_ders_(i, j, 2)};
						const double gate_da = gate_additive_param_by_mu[mu];
						const double type_scale = gate_type_scale_by_mu[mu];
						const double gate_multiplier = gate_multiplier_by_mu[mu];
						const double gate_outer = gate_outer_by_mu[mu];
					for (int xi = 0; xi < R; ++xi) {
					const double rb_val = rb_vals[radial_base + xi];
				const double rb_der = rb_ders[deriv_base + xi];
				double grad = adj_value * type_scale * rb_val * y;
				for (int a = 0; a < 3; ++a)
					grad += adj_der[a] * type_scale * (rb_der * rvec[a] * inv_r * y + rb_val * dy[a]);
				grad_out[radial_offset + xi] += grad;
			}

						double center_grad =
							adj_value * gate_outer * dot_val * y;
						double gate_type_grad =
							adj_value * center_type_coeff * gate_multiplier * dot_val * y;
					double sigma_grad = adj_value * type_scale * dot_s * y;
					double shift_grad = adj_value * type_scale * dot_ss * y;
						double additive_grad =
							adj_value * center_type_coeff * gate_da * dot_val * y;
						double center_additive_grad = 0.0;
						if (TwoLayerGateUsesCenterGate())
							center_additive_grad =
								adj_value * center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_param_by_mu[mu]
								* dot_val * y;
					for (int a = 0; a < 3; ++a) {
						const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
					const double sigma_coord = dot_coord_s * rvec[a] * inv_r * y + dot_s * dy[a];
					const double shift_coord = dot_coord_ss * rvec[a] * inv_r * y + dot_ss * dy[a];
					center_grad +=
						adj_der[a] * gate_outer * base_coord;
						gate_type_grad +=
							adj_der[a] * center_type_coeff * gate_multiplier * base_coord;
					sigma_grad += adj_der[a] * type_scale * sigma_coord;
					shift_grad += adj_der[a] * type_scale * shift_coord;
						additive_grad +=
							adj_der[a] * center_type_coeff * gate_da * base_coord;
						if (TwoLayerGateUsesCenterGate())
							center_additive_grad +=
								adj_der[a] * center_type_coeff * outer_type_coeff
								* neighbor_gate_multiplier_by_mu[mu]
								* center_gate_additive_param_by_mu[mu]
								* base_coord;
					}

					const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
						grad_out[center_type_coeff_index] += center_grad;
						grad_out[outer_type_coeff_index] += gate_type_grad;
				grad_out[sigma_coeff_offset] += sigma_grad;
				grad_out[sigma_coeff_offset + C * C] += shift_grad;
			if (two_layer_gate_enabled_)
				grad_out[additive_outer_index_base + mu] +=
					additive_grad;
			if (TwoLayerGateUsesCenterGate())
				grad_out[additive_center_index_base + mu] +=
					center_additive_grad;
				}
		}
	}
