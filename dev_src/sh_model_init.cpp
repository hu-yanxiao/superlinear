#include "sh_model_init.h"

#include "../src/common/stdafx.h"
#include "../src/common/utils.h"
#include "../src/zbl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class SHFactorPruning {
	SameK,
	Legacy,
	QTotal
};

struct Product {
	int left;
	int right;
	int target;
	double coeff;
};

struct LocalProduct {
	int left;
	int right;
	int target_component;
	double coeff;
};

struct BasicIndex {
	int k;
	int l;
	int m;
	int mu;
};

struct QIndex {
	int l;
	int k;
	int mu;
	std::string key;
};

struct Tensor {
	std::string key;
	int l;
	std::vector<int> node;
	bool zero = false;
};

struct ScalarSpec {
	int body_order;
	int q0;
	int q1;
	int q2;
	int q3;
	int q4;
	int intermediate_l;
};

int CheckedGraphCount(std::initializer_list<size_t> factors, const char* name)
{
	size_t count = 1;
	const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
	for (size_t factor : factors) {
		if (factor == 0) {
			count = 0;
			continue;
		}
		if (factor > limit || count > limit / factor)
			ERROR(std::string(name) + " exceeds the supported 32-bit graph/parameter index range.");
		count *= factor;
	}
	return static_cast<int>(count);
}

int IntOpt(const std::map<std::string, std::string>& opts, const std::string& name, int default_value)
{
	std::map<std::string, std::string>::const_iterator it = opts.find(name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return std::stoi(it->second);
}

int StrictNonnegativeIntOpt(
	const std::map<std::string, std::string>& opts,
	const std::string& name,
	int default_value)
{
	const std::map<std::string, std::string>::const_iterator it = opts.find(name);
	if (it == opts.end())
		return default_value;
	const std::string& value = it->second;
	if (value.empty()
	    || !std::all_of(value.begin(), value.end(), [](char ch) {
		    return ch >= '0' && ch <= '9';
	    }))
		ERROR("--" + name + " should be a non-negative integer written as --"
		      + name + "=<R>.");
	try {
		return std::stoi(value);
	} catch (const std::exception&) {
		ERROR("--" + name + " is outside the supported integer range.");
	}
}

double DoubleOpt(const std::map<std::string, std::string>& opts, const std::string& name, double default_value)
{
	std::map<std::string, std::string>::const_iterator it = opts.find(name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return std::stod(it->second);
}

std::string StringOpt(const std::map<std::string, std::string>& opts, const std::string& name, const std::string& default_value)
{
	std::map<std::string, std::string>::const_iterator it = opts.find(name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return it->second;
}

std::string CanonicalRadialBasisType(const std::string& value)
{
	if (value == "jacobi_sss")
		return "RBJacobi_sss";
	if (value == "laguerre_log1p")
		return "RBLaguerre_log1p";
	return value;
}

bool IsSupportedSHRadialBasis(const std::string& value)
{
	return value == "RBChebyshev_sss"
	    || value == "RBChebyshev_sss_rational"
	    || value == "RBLaguerre_log1p"
	    || value == "RBJacobi_sss";
}

bool HasOpt(const std::map<std::string, std::string>& opts, const std::string& name)
{
	std::map<std::string, std::string>::const_iterator it = opts.find(name);
	return it != opts.end() && !it->second.empty();
}

bool BoolOpt(const std::map<std::string, std::string>& opts, const std::string& name, bool fallback)
{
	const auto found = opts.find(name);
	if (found == opts.end()) return fallback;
	if (found->second.empty() || found->second == "true" || found->second == "1") return true;
	if (found->second == "false" || found->second == "0") return false;
	ERROR("--" + name + " must be true or false");
	return fallback;
}

std::string CanonicalRadialCutoffType(const std::string& value)
{
	if (value == "legacy" || value == "quadratic")
		return "legacy";
	if (value == "polynomial")
		return "polynomial";
	ERROR("--radial-cutoff-type should be legacy or polynomial.");
	return "legacy";
}

double LerpLocal(double a, double b, double t)
{
	return a + (b - a) * t;
}

double Clamp01Local(double value)
{
	return std::max(0.0, std::min(1.0, value));
}

double OrderedPairStrengthLocal(int type_central, int type_outer)
{
	return std::sqrt(static_cast<double>(type_central + 1))
	     + std::sqrt(static_cast<double>(type_outer + 1));
}

double NormalizedOrderedPairStrengthLocal(
	int type_central,
	int type_outer,
	int species_count)
{
	if (species_count <= 1)
		return 0.5;
	const double min_strength = OrderedPairStrengthLocal(0, 0);
	const double max_strength =
		OrderedPairStrengthLocal(species_count - 1, species_count - 1);
	if (max_strength <= min_strength)
		return 0.5;
	return Clamp01Local(
		(OrderedPairStrengthLocal(type_central, type_outer) - min_strength)
		/ (max_strength - min_strength));
}

double DefaultScalingSlopeStart(const std::string& rbasis)
{
	return rbasis == "RBLaguerre_log1p" ? 3.0 : 4.0;
}

void WriteDefaultScalingCoeffList(std::ofstream& ofs,
                                  int species_count,
                                  int kmax,
                                  const std::string& rbasis)
{
	const double scal_start = DefaultScalingSlopeStart(rbasis);
	const double scal_end = 2.0;
	const double shift_start = 1.5;
	const double shift_end = 2.5;
	const int pair_count = species_count * species_count;
	for (int scaling_block = 0; scaling_block < kmax; ++scaling_block) {
		if (scaling_block != 0)
			ofs << ", ";
		for (int pair = 0; pair < pair_count; ++pair) {
			if (pair != 0)
				ofs << ", ";
			const int type_central = pair / species_count;
			const int type_outer = pair % species_count;
			const double strength =
				NormalizedOrderedPairStrengthLocal(
					type_central,
					type_outer,
					species_count);
			ofs << LerpLocal(scal_start, scal_end, strength);
		}
		for (int pair = 0; pair < pair_count; ++pair) {
			ofs << ", ";
			const int type_central = pair / species_count;
			const int type_outer = pair % species_count;
			const double strength =
				NormalizedOrderedPairStrengthLocal(
					type_central,
					type_outer,
					species_count);
			ofs << LerpLocal(shift_start, shift_end, strength);
		}
	}
}

double ParseStrictDoubleOption(const std::string& value,
                               const std::string& opt_name)
{
	std::size_t parsed = 0;
	double result = 0.0;
	try {
		result = std::stod(value, &parsed);
	} catch (const std::exception&) {
		ERROR(opt_name + " should be a finite double.");
	}
	if (parsed != value.size() || !std::isfinite(result))
		ERROR(opt_name + " should be a finite double.");
	return result;
}

struct ZBLInitOptions {
	bool enabled = false;
	double inner = DefaultZBLInnerCutoff();
	double outer = DefaultZBLOuterCutoff();
	bool typewise = false;
	double typewise_factor = DefaultZBLTypewiseCutoffFactor();
	std::vector<int> atomic_numbers;
};

bool HasAnyZBLOpt(const std::map<std::string, std::string>& opts)
{
	return HasOpt(opts, "zbl-elements")
	    || HasOpt(opts, "zbl-inner")
	    || HasOpt(opts, "zbl-outer")
	    || HasOpt(opts, "zbl-typewise-cutoff-factor");
}

ZBLInitOptions ParseZBLInitOptions(const std::map<std::string, std::string>& opts,
                                   int species_count)
{
	ZBLInitOptions zbl;
	if (!HasAnyZBLOpt(opts))
		return zbl;
	if (!HasOpt(opts, "zbl-elements"))
		ERROR("ZBL requires --zbl-elements=<comma-separated symbols or atomic numbers>.");
	zbl.enabled = true;
	zbl.atomic_numbers = ParseZBLAtomicNumbers(opts.find("zbl-elements")->second);
	if (static_cast<int>(zbl.atomic_numbers.size()) != species_count)
		ERROR("--zbl-elements count should match --species-count.");
	if (HasOpt(opts, "zbl-inner"))
		zbl.inner = ParseStrictDoubleOption(opts.find("zbl-inner")->second,
		                                    "--zbl-inner");
	if (HasOpt(opts, "zbl-outer"))
		zbl.outer = ParseStrictDoubleOption(opts.find("zbl-outer")->second,
		                                    "--zbl-outer");
	if (!std::isfinite(zbl.inner) || zbl.inner < 0.0)
		ERROR("--zbl-inner should be finite and non-negative.");
	if (!std::isfinite(zbl.outer) || zbl.outer <= 0.0)
		ERROR("--zbl-outer should be finite and positive.");
	zbl.typewise = HasOpt(opts, "zbl-typewise-cutoff-factor");
	if (zbl.typewise) {
		zbl.typewise_factor =
			ParseStrictDoubleOption(opts.find("zbl-typewise-cutoff-factor")->second,
			                        "--zbl-typewise-cutoff-factor");
		if (!std::isfinite(zbl.typewise_factor) || zbl.typewise_factor < 0.5)
			ERROR("--zbl-typewise-cutoff-factor should be finite and >= 0.5.");
	} else if (zbl.outer <= zbl.inner) {
		ERROR("--zbl-inner and --zbl-outer should satisfy 0 <= inner < outer.");
	}
	return zbl;
}

void WriteZBLSection(std::ofstream& ofs, const ZBLInitOptions& zbl)
{
	if (!zbl.enabled)
		return;
	ofs << "zbl_enabled = true\n";
	ofs << "zbl_inner = " << (zbl.typewise ? 0.0 : zbl.inner) << "\n";
	ofs << "zbl_outer = " << zbl.outer << "\n";
	ofs << "zbl_typewise_cutoff_enabled = "
	    << (zbl.typewise ? "true" : "false") << "\n";
	ofs << "zbl_typewise_cutoff_factor = " << zbl.typewise_factor << "\n";
	ofs << "zbl_atomic_numbers = {";
	for (size_t i = 0; i < zbl.atomic_numbers.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		ofs << zbl.atomic_numbers[i];
	}
	ofs << "}\n";
}

SHFactorPruning ParseFactorPruningValue(const std::string& value,
	                                    const std::string& option_name)
{
	if (value == "same-k")
		return SHFactorPruning::SameK;
	if (value == "legacy")
		return SHFactorPruning::Legacy;
	if (value == "q-total" || value == "total")
		return SHFactorPruning::QTotal;
	ERROR("--" + option_name + " should be same-k, legacy, or q-total.");
	return SHFactorPruning::QTotal;
}

SHFactorPruning ParseFactorPruning(const std::map<std::string, std::string>& opts,
	                               const std::string& option_name,
	                               SHFactorPruning default_value)
{
	std::map<std::string, std::string>::const_iterator it = opts.find(option_name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return ParseFactorPruningValue(it->second, option_name);
}

int FactorPruningRank(SHFactorPruning pruning)
{
	switch (pruning) {
	case SHFactorPruning::SameK:
		return 0;
	case SHFactorPruning::Legacy:
		return 1;
	case SHFactorPruning::QTotal:
		return 2;
	}
	return -1;
}

bool IsFactorPruningSubset(SHFactorPruning inner, SHFactorPruning outer)
{
	return FactorPruningRank(inner) <= FactorPruningRank(outer);
}

const char* FactorPruningName(SHFactorPruning pruning)
{
	if (pruning == SHFactorPruning::SameK)
		return "same-k";
	if (pruning == SHFactorPruning::Legacy)
		return "legacy";
	return "q-total";
}

std::vector<int> ParseBodyLMax(const std::map<std::string, std::string>& opts,
                               int global_lmax,
                               int body_order)
{
	std::vector<int> body_lmax(7, global_lmax);
	std::map<std::string, std::string>::const_iterator it = opts.find("body-l-max");
	if (it != opts.end() && !it->second.empty()) {
		std::vector<int> values;
		std::stringstream ss(it->second);
		std::string token;
		while (std::getline(ss, token, ',')) {
			if (!token.empty())
				values.push_back(std::stoi(token));
		}
		const size_t expected = body_order >= 6 ? 5 : 4;
		if (values.size() != expected)
			ERROR("--body-l-max should contain four values for body 2..5, or five values for body 2..6.");
		for (int body = 2; body <= 1 + static_cast<int>(values.size()); ++body)
			body_lmax[body] = values[body - 2];
	}
	for (int body = 2; body <= 6; ++body) {
		const std::string opt = "body" + std::to_string(body) + "-l-max";
		std::map<std::string, std::string>::const_iterator one = opts.find(opt);
		if (one != opts.end() && !one->second.empty())
			body_lmax[body] = std::stoi(one->second);
		if (body_lmax[body] < 0 || body_lmax[body] > global_lmax)
			ERROR("body-specific l cutoff should be between 0 and --l-max.");
	}
	return body_lmax;
}

double Fact(int n)
{
	if (n < 0)
		return 0.0;
	double value = 1.0;
	for (int i = 2; i <= n; ++i)
		value *= static_cast<double>(i);
	return value;
}

bool Triangle(int l1, int l2, int l3)
{
	return std::abs(l1 - l2) <= l3 && l3 <= l1 + l2;
}

double ClebschGordan(int j1, int m1, int j2, int m2, int j, int m)
{
	if (m != m1 + m2)
		return 0.0;
	if (!Triangle(j1, j2, j))
		return 0.0;
	if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m) > j)
		return 0.0;

	const double pref1 = std::sqrt((2.0 * j + 1.0)
		* Fact(j1 + j2 - j)
		* Fact(j1 - j2 + j)
		* Fact(-j1 + j2 + j)
		/ Fact(j1 + j2 + j + 1));
	const double pref2 = std::sqrt(Fact(j1 + m1) * Fact(j1 - m1)
		* Fact(j2 + m2) * Fact(j2 - m2)
		* Fact(j + m) * Fact(j - m));

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
			/ (Fact(k) * Fact(a) * Fact(b) * Fact(c) * Fact(d) * Fact(e));
		sum += term;
	}
	return pref1 * pref2 * sum;
}

double ParitySign(int m)
{
	return (std::abs(m) % 2) == 0 ? 1.0 : -1.0;
}

std::complex<double> RealFromComplexCoeff(int real_m, int complex_m)
{
	const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
	if (real_m == 0)
		return complex_m == 0 ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
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

std::complex<double> RealCouplingPhase(int l1, int l2, int L)
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

double RealCGCoeff(int l1, int rm1, int l2, int rm2, int L, int rM)
{
	std::complex<double> sum(0.0, 0.0);
	for (int M = -L; M <= L; ++M) {
		const std::complex<double> out_u = std::conj(RealFromComplexCoeff(rM, M));
		if (std::abs(out_u) == 0.0)
			continue;
		for (int m1 = -l1; m1 <= l1; ++m1) {
			const std::complex<double> in1 = RealFromComplexCoeff(rm1, m1);
			if (std::abs(in1) == 0.0)
				continue;
			for (int m2 = -l2; m2 <= l2; ++m2) {
				const std::complex<double> in2 = RealFromComplexCoeff(rm2, m2);
				if (std::abs(in2) == 0.0)
					continue;
				const double cg = ClebschGordan(l1, m1, l2, m2, L, M);
				if (std::abs(cg) < 1.0e-14)
					continue;
				sum += out_u * cg * in1 * in2;
			}
		}
	}
	sum *= RealCouplingPhase(l1, l2, L);
	if (std::abs(sum.imag()) > 1.0e-10)
		ERROR("real spherical harmonic CG transform produced a non-real coefficient.");
	return sum.real();
}

std::string TensorKey(const std::string& left, const std::string& right, int L)
{
	std::ostringstream oss;
	oss << "(" << left << "x" << right << ")->" << L;
	return oss.str();
}

class SHGraphBuilder {
public:
	using CGCache = std::map<std::array<int, 6>, double>;
	SHGraphBuilder(int lmax, int kmax, SHFactorPruning factor_pruning,
	               CGCache* cg_cache = nullptr)
		: lmax_(lmax), kmax_(kmax), factor_pruning_(factor_pruning),
		  cg_cache_(cg_cache)
	{
	}

	double CouplingCoeff(int l1, int m1, int l2, int m2, int L, int M)
	{
		if (cg_cache_ == nullptr)
			return RealCGCoeff(l1, m1, l2, m2, L, M);
		const std::array<int, 6> key = {{l1, m1, l2, m2, L, M}};
		const auto found = cg_cache_->find(key);
		if (found != cg_cache_->end())
			return found->second;
		const double value = RealCGCoeff(l1, m1, l2, m2, L, M);
		(*cg_cache_)[key] = value;
		return value;
	}

	const Tensor& BasicTensor(int q_index)
	{
		const QIndex& q = q_[q_index];
		std::map<std::string, int>::const_iterator found = tensor_lookup_.find(q.key);
		if (found != tensor_lookup_.end())
			return tensors_[found->second];

		Tensor tensor;
		tensor.key = q.key;
		tensor.l = q.l;
		tensor.zero = false;
		tensor.node.resize(2 * q.l + 1);
		for (int m = -q.l; m <= q.l; ++m) {
			const int node = AddNode();
			tensor.node[m + q.l] = node;
			BasicIndex basic = {q.k, q.l, m, q.mu};
			basic_.push_back(basic);
		}
		const int index = static_cast<int>(tensors_.size());
		tensors_.push_back(tensor);
		tensor_lookup_[q.key] = index;
		return tensors_.back();
	}

	const Tensor& Couple(const Tensor& left, const Tensor& right, int L)
	{
		const Tensor left_copy = left;
		const Tensor right_copy = right;
		const std::string key = TensorKey(left_copy.key, right_copy.key, L);
		std::map<std::string, int>::const_iterator found = tensor_lookup_.find(key);
		if (found != tensor_lookup_.end())
			return tensors_[found->second];

		Tensor out;
		out.key = key;
		out.l = L;

		if (left_copy.zero || right_copy.zero) {
			out.zero = true;
			const int index = static_cast<int>(tensors_.size());
			tensors_.push_back(out);
			tensor_lookup_[key] = index;
			return tensors_.back();
		}

		std::vector<LocalProduct> local_products;
		std::map<std::string, int> local_lookup;
		auto add_local_product = [&](int left_node, int right_node, int target_component, double coeff) {
			if (std::abs(coeff) < 1.0e-12)
				return;
			if (right_node < left_node)
				std::swap(left_node, right_node);
			std::ostringstream local_key;
			local_key << left_node << ',' << right_node << ',' << target_component;
			std::map<std::string, int>::iterator local_found = local_lookup.find(local_key.str());
			if (local_found != local_lookup.end()) {
				local_products[local_found->second].coeff += coeff;
				return;
			}
			LocalProduct product = {left_node, right_node, target_component, coeff};
			local_lookup[local_key.str()] = static_cast<int>(local_products.size());
			local_products.push_back(product);
		};

		for (int rm1 = -left_copy.l; rm1 <= left_copy.l; ++rm1) {
			for (int rm2 = -right_copy.l; rm2 <= right_copy.l; ++rm2) {
				for (int rM = -L; rM <= L; ++rM) {
					const double coeff = CouplingCoeff(left_copy.l, rm1, right_copy.l, rm2, L, rM);
					if (std::abs(coeff) < 1.0e-12)
						continue;
					add_local_product(left_copy.node[rm1 + left_copy.l],
					                  right_copy.node[rm2 + right_copy.l],
					                  rM + L,
					                  coeff);
				}
			}
		}
		std::vector<LocalProduct> compact_local;
		compact_local.reserve(local_products.size());
		for (size_t i = 0; i < local_products.size(); ++i)
			if (std::abs(local_products[i].coeff) >= 1.0e-12)
				compact_local.push_back(local_products[i]);
		if (compact_local.empty()) {
			out.zero = true;
			const int index = static_cast<int>(tensors_.size());
			tensors_.push_back(out);
			tensor_lookup_[key] = index;
			return tensors_.back();
		}

		out.zero = false;
		out.node.resize(2 * L + 1);
		for (int M = -L; M <= L; ++M)
			out.node[M + L] = AddNode();
		const int index = static_cast<int>(tensors_.size());
		tensors_.push_back(out);
		tensor_lookup_[key] = index;
		Tensor& stored = tensors_.back();

		for (size_t i = 0; i < compact_local.size(); ++i)
			AddProduct(compact_local[i].left,
			           compact_local[i].right,
			           stored.node[compact_local[i].target_component],
			           compact_local[i].coeff);
		return stored;
	}

	void Build(int minimum_l = 0)
	{
		for (int l = lmax_; l >= minimum_l; --l) {
			for (int k = kmax_ - 1; k >= 0; --k) {
				QIndex q;
				q.l = l;
				q.k = k;
				q.mu = k * (lmax_ + 1) + l;
				std::ostringstream key;
				key << "q" << q.mu << "_l" << l << "_k" << k;
				q.key = key.str();
				q_.push_back(q);
			}
		}
	}

	void AddAllBasics()
	{
		for (int q_index = 0; q_index < static_cast<int>(q_.size()); ++q_index)
			BasicTensor(q_index);
	}

	bool Allowed(int q_index, int body_order) const
	{
		return q_[q_index].l <= body_lmax_[body_order];
	}

	void EnumerateFactorTuples(int factor_count,
	                           int body_order,
	                           const std::function<void(const std::vector<int>&)>& callback)
	{
		std::vector<int> tuple;
		if (factor_pruning_ == SHFactorPruning::Legacy) {
			std::function<void(int, int, int)> rec = [&](int pos, int max_l, int max_k) {
				if (pos == factor_count) {
					callback(tuple);
					return;
				}
				for (int q_index = 0; q_index < static_cast<int>(q_.size()); ++q_index) {
					const QIndex& q = q_[q_index];
					if (!Allowed(q_index, body_order) || q.l > max_l || q.k > max_k)
						continue;
					tuple.push_back(q_index);
					rec(pos + 1, q.l, q.k);
					tuple.pop_back();
				}
			};
			rec(0, lmax_, kmax_ - 1);
		} else {
			const bool same_k = factor_pruning_ == SHFactorPruning::SameK;
			std::function<void(int, int, int)> rec =
				[&](int pos, int first_q_index, int selected_k) {
				if (pos == factor_count) {
					callback(tuple);
					return;
				}
				for (int q_index = first_q_index; q_index < static_cast<int>(q_.size()); ++q_index) {
					if (!Allowed(q_index, body_order)
					    || (same_k && selected_k >= 0 && q_[q_index].k != selected_k))
						continue;
					tuple.push_back(q_index);
					rec(pos + 1,
					    q_index,
					    selected_k >= 0 ? selected_k : q_[q_index].k);
					tuple.pop_back();
				}
			};
			rec(0, 0, -1);
		}
	}

	bool ScalarAllowedByFactorPruning(const ScalarSpec& spec,
	                                  SHFactorPruning pruning) const
	{
		if (pruning == SHFactorPruning::QTotal)
			return true;
		const int qids[] = {spec.q0, spec.q1, spec.q2, spec.q3, spec.q4};
		const int factor_count = spec.body_order - 1;
		for (int i = 1; i < factor_count; ++i) {
			const QIndex& previous = q_[qids[i - 1]];
			const QIndex& current = q_[qids[i]];
			if (current.l > previous.l || current.k > previous.k)
				return false;
		}
		if (pruning == SHFactorPruning::Legacy)
			return true;
		const int k = q_[qids[0]].k;
		for (int i = 1; i < factor_count; ++i)
			if (q_[qids[i]].k != k)
				return false;
		return true;
	}

	std::vector<int> ScalarIndices(int body_order_max,
	                              SHFactorPruning pruning,
	                              std::vector<int>& body_counts) const
	{
		std::vector<int> indices;
		body_counts.assign(body_order_max + 1, 0);
		for (size_t i = 0; i < scalar_infos_.size(); ++i) {
			const ScalarSpec& spec = scalar_infos_[i];
			if (spec.body_order < 2 || spec.body_order > body_order_max
			    || !ScalarAllowedByFactorPruning(spec, pruning))
				continue;
			indices.push_back(static_cast<int>(i));
			++body_counts[spec.body_order];
		}
		return indices;
	}

	void AddScalarSpec(int body_order, int q0, int q1, int q2, int q3, int q4, int intermediate_l)
	{
		ScalarSpec spec = {body_order, q0, q1, q2, q3, q4, intermediate_l};
		scalar_specs_.push_back(spec);
		required_q_[q0] = true;
		if (body_order >= 3)
			required_q_[q1] = true;
		if (body_order >= 4)
			required_q_[q2] = true;
		if (body_order >= 5)
			required_q_[q3] = true;
		if (body_order >= 6)
			required_q_[q4] = true;
	}

	void CollectScalarSpecs(int body_order)
	{
		required_q_.assign(q_.size(), false);
		scalar_specs_.clear();

		if (body_order >= 2) {
			EnumerateFactorTuples(1, 2, [&](const std::vector<int>& qids) {
				if (q_[qids[0]].l == 0)
					AddScalarSpec(2, qids[0], -1, -1, -1, -1, 0);
			});
		}
		if (body_order >= 3) {
			EnumerateFactorTuples(2, 3, [&](const std::vector<int>& qids) {
				if (q_[qids[0]].l == q_[qids[1]].l)
					AddScalarSpec(3, qids[0], qids[1], -1, -1, -1, 0);
			});
		}
		if (body_order >= 4) {
			EnumerateFactorTuples(3, 4, [&](const std::vector<int>& qids) {
				const int l0 = q_[qids[0]].l;
				const int l1 = q_[qids[1]].l;
				const int l2 = q_[qids[2]].l;
				if (((l0 + l1 + l2) % 2) == 0 && Triangle(l0, l1, l2))
					AddScalarSpec(4, qids[0], qids[1], qids[2], -1, -1, l2);
			});
		}
		if (body_order >= 5) {
			EnumerateFactorTuples(4, 5, [&](const std::vector<int>& qids) {
				const int l0 = q_[qids[0]].l;
				const int l1 = q_[qids[1]].l;
				const int l2 = q_[qids[2]].l;
				const int l3 = q_[qids[3]].l;
				if (((l0 + l1 + l2 + l3) % 2) != 0)
					return;
				const int lo = std::max(std::abs(l0 - l1), std::abs(l2 - l3));
				const int hi = std::min(l0 + l1, l2 + l3);
				for (int L = lo; L <= hi; ++L)
					AddScalarSpec(5, qids[0], qids[1], qids[2], qids[3], -1, L);
			});
		}
		if (body_order >= 6) {
			EnumerateFactorTuples(5, 6, [&](const std::vector<int>& qids) {
				const int l0 = q_[qids[0]].l;
				const int l1 = q_[qids[1]].l;
				const int l2 = q_[qids[2]].l;
				const int l3 = q_[qids[3]].l;
				const int l4 = q_[qids[4]].l;
				if (((l0 + l1 + l2 + l3 + l4) % 2) != 0)
					return;
				const int lo = std::max(std::abs(l0 - l1), std::abs(l2 - l3));
				const int hi = std::min(l0 + l1, l2 + l3);
				for (int L = lo; L <= hi; ++L)
					if (Triangle(L, l4, L))
						AddScalarSpec(6, qids[0], qids[1], qids[2], qids[3], qids[4], L);
			});
		}
	}

	int BuildScalar(const ScalarSpec& spec)
	{
		const Tensor t0 = BasicTensor(spec.q0);
		if (spec.body_order == 2)
			return t0.node[0];
		const Tensor t1 = BasicTensor(spec.q1);
		if (spec.body_order == 3) {
			const Tensor scalar = Couple(t0, t1, 0);
			if (scalar.zero)
				return -1;
			return scalar.node[0];
		}
		const Tensor t2 = BasicTensor(spec.q2);
		if (spec.body_order == 4) {
			const Tensor pair = Couple(t0, t1, spec.intermediate_l);
			if (pair.zero)
				return -1;
			const Tensor scalar = Couple(pair, t2, 0);
			if (scalar.zero)
				return -1;
			return scalar.node[0];
		}
		const Tensor t3 = BasicTensor(spec.q3);
		const Tensor left = Couple(t0, t1, spec.intermediate_l);
		const Tensor right_pair = Couple(t2, t3, spec.intermediate_l);
		if (spec.body_order == 6) {
			const Tensor t4 = BasicTensor(spec.q4);
			if (left.zero || right_pair.zero)
				return -1;
			const Tensor right = Couple(right_pair, t4, spec.intermediate_l);
			if (right.zero)
				return -1;
			const Tensor scalar = Couple(left, right, 0);
			if (scalar.zero)
				return -1;
			return scalar.node[0];
		}
		const Tensor right = right_pair;
		if (left.zero || right.zero)
			return -1;
		const Tensor scalar = Couple(left, right, 0);
		if (scalar.zero)
			return -1;
		return scalar.node[0];
	}

	void CompressProducts()
	{
		std::vector<Product> compact;
		compact.reserve(products_.size());
		for (size_t i = 0; i < products_.size(); ++i) {
			if (std::abs(products_[i].coeff) >= 1.0e-12)
				compact.push_back(products_[i]);
		}
		products_.swap(compact);
	}

	void AddScalars(int body_order, const std::vector<int>& body_lmax)
	{
		body_lmax_ = body_lmax;
		CollectScalarSpecs(body_order);
			for (int q_index = 0; q_index < static_cast<int>(q_.size()); ++q_index)
				if (required_q_[q_index])
					BasicTensor(q_index);
			for (size_t i = 0; i < scalar_specs_.size(); ++i) {
				const int scalar = BuildScalar(scalar_specs_[i]);
				if (scalar >= 0) {
					scalars_.push_back(scalar);
					scalar_infos_.push_back(scalar_specs_[i]);
				}
			}
			CompressProducts();
		}

	// Compile one K=1 angular plan onto independent output channels. Only
	// values and node labels are replicated: CG construction and single-channel
	// symmetry cancellation happen once, before channel binding. The expanded
	// instructions preserve the established evaluator/model-file interface.
	void BindSingleChannelPlan(const SHGraphBuilder& angular)
	{
		if (angular.kmax_ != 1 || angular.lmax_ != lmax_
		    || !products_.empty() || !scalars_.empty()
		    || node_count_ != static_cast<int>(basic_.size()))
			ERROR("Invalid single-channel outer plan binding.");
		body_lmax_ = angular.body_lmax_;
		CheckedGraphCount({static_cast<size_t>(kmax_),
			static_cast<size_t>(angular.node_count_)}, "Channelwise outer node count");
		CheckedGraphCount({static_cast<size_t>(kmax_), angular.products_.size()},
			"Channelwise outer product count");
		CheckedGraphCount({static_cast<size_t>(kmax_), angular.scalars_.size()},
			"Channelwise outer scalar count");
		std::vector<std::vector<int>> nodes(kmax_,
			std::vector<int>(angular.node_count_, -1));
		std::map<std::array<int, 3>, int> basics;
		for (int i = 0; i < static_cast<int>(basic_.size()); ++i) {
			const BasicIndex& b = basic_[i];
			basics[{{b.k, b.l, b.m}}] = i;
		}
		for (int channel = kmax_ - 1; channel >= 0; --channel) {
			for (int i = 0; i < static_cast<int>(angular.basic_.size()); ++i) {
				const BasicIndex& b = angular.basic_[i];
				nodes[channel][i] = basics.at({{channel, b.l, b.m}});
			}
			for (int i = static_cast<int>(angular.basic_.size());
			     i < angular.node_count_; ++i)
				nodes[channel][i] = AddNode();
			for (const Product& p : angular.products_)
				products_.push_back({nodes[channel][p.left],
					nodes[channel][p.right], nodes[channel][p.target], p.coeff});
		}
		for (size_t i = 0; i < angular.scalars_.size(); ++i) {
			for (int channel = kmax_ - 1; channel >= 0; --channel) {
				scalars_.push_back(nodes[channel][angular.scalars_[i]]);
				ScalarSpec spec = angular.scalar_infos_[i];
				int* qids[] = {&spec.q0, &spec.q1, &spec.q2, &spec.q3, &spec.q4};
				for (int* q : qids)
					if (*q >= 0)
						*q = *q * kmax_ + (kmax_ - 1 - channel);
				scalar_infos_.push_back(spec);
			}
		}
		if (scalars_.size() != static_cast<size_t>(kmax_) * angular.scalars_.size())
			ERROR("Channelwise outer scalar count must equal O times the angular scalar count.");
	}

	std::vector<int> L1SourceMomentBases() const
	{
		std::vector<int> bases;
		for (const Tensor& tensor : tensors_) {
			if (tensor.zero || tensor.l != 1 || tensor.node.size() != 3)
				continue;
			bases.push_back(tensor.node[0]);
		}
		std::sort(bases.begin(), bases.end());
		bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
		return bases;
	}

	int AddNode()
	{
		return node_count_++;
	}

	void AddProduct(int left, int right, int target, double coeff)
	{
		if (std::abs(coeff) < 1.0e-12)
			return;
		if (right < left)
			std::swap(left, right);
		std::ostringstream key;
		key << left << ',' << right << ',' << target;
		std::map<std::string, int>::iterator found = product_lookup_.find(key.str());
		if (found != product_lookup_.end()) {
			products_[found->second].coeff += coeff;
			return;
		}
		Product product = {left, right, target, coeff};
		product_lookup_[key.str()] = static_cast<int>(products_.size());
		products_.push_back(product);
	}

	int lmax_;
	int kmax_;
	SHFactorPruning factor_pruning_;
	CGCache* cg_cache_;
	int node_count_ = 0;
	std::vector<int> body_lmax_;
	std::vector<QIndex> q_;
	std::vector<Tensor> tensors_;
	std::map<std::string, int> tensor_lookup_;
	std::map<std::string, int> product_lookup_;
	std::vector<char> required_q_;
	std::vector<ScalarSpec> scalar_specs_;
	std::vector<ScalarSpec> scalar_infos_;
	std::vector<BasicIndex> basic_;
	std::vector<Product> products_;
	std::vector<int> scalars_;
};

} // namespace

JointReadoutTopology BuildJointReadoutTopology(int l_max, int body_order,
                                               const std::vector<int>& body_l_max)
{
    if (l_max < 1 || l_max > 6 || body_order < 2 || body_order > 6
        || body_l_max.size() != 7)
        ERROR("SUS2-SH joint readout requires l_max >= 1 and complete outer body metadata");
    for (int b = 2; b <= (body_order >= 6 ? 6 : 5); ++b)
        if (body_l_max[b] < 0 || body_l_max[b] > l_max)
            ERROR("SUS2-SH joint readout body_l_max is invalid");
    SHGraphBuilder::CGCache cache;
    SHGraphBuilder builder(l_max, 2, SHFactorPruning::QTotal, &cache);
    builder.Build(1); // Exclude l=0 leaves, not just their numerical values.
    builder.AddAllBasics();
    builder.AddScalars(body_order, body_l_max);
    JointReadoutTopology result;
    result.l_max = l_max; result.body_order = body_order;
    result.body_l_max = body_l_max; result.moments_count = builder.node_count_;
    for (const BasicIndex& b : builder.basic_) result.basic.push_back({{b.k, b.l, b.m}});
    for (const Product& p : builder.products_)
        result.products.push_back({p.left, p.right, p.target, p.coeff});
    result.scalar_mapping = builder.scalars_;
    for (const ScalarSpec& s : builder.scalar_infos_)
        result.scalar_info.push_back({{s.body_order, s.q0, s.q1, s.q2, s.q3, s.q4, s.intermediate_l}});
    return result;
}

void WriteJointReadoutSection(std::ostream& out, const JointReadoutTopology& g,
                             const std::vector<double>& coefficients,
                             const std::vector<double>& scales)
{
    out << "joint_readout_enabled = true\n"
        << "joint_readout_l_max = " << g.l_max << '\n'
        << "joint_readout_body_order = " << g.body_order << '\n'
        << "joint_readout_body_l_max = {";
    for (int b = 2; b <= (g.body_order >= 6 ? 6 : 5); ++b)
        out << (b == 2 ? "" : ", ") << g.body_l_max[b];
    out << "}\njoint_readout_alpha_moments_count = " << g.moments_count
        << "\njoint_readout_alpha_index_basic_count = " << g.basic.size()
        << "\njoint_readout_alpha_index_basic = {";
    for (size_t i = 0; i < g.basic.size(); ++i) {
        const auto& b = g.basic[i];
        out << (i ? ", " : "") << '{' << b[0] << ", " << b[1] << ", " << b[2] << '}';
    }
    out << "}\njoint_readout_sh_product_count = " << g.products.size()
        << "\njoint_readout_sh_products = {";
    for (size_t i = 0; i < g.products.size(); ++i) {
        const auto& p = g.products[i];
        out << (i ? ", " : "") << '{' << p.left << ", " << p.right << ", " << p.target << ", " << p.coeff << '}';
    }
    out << "}\njoint_readout_scalar_count = " << g.scalar_mapping.size()
        << "\njoint_readout_scalar_mapping = {";
    for (size_t i = 0; i < g.scalar_mapping.size(); ++i)
        out << (i ? ", " : "") << g.scalar_mapping[i];
    out << "}\njoint_readout_scalar_info_count = " << g.scalar_info.size()
        << "\njoint_readout_scalar_info = {";
    for (size_t i = 0; i < g.scalar_info.size(); ++i) {
        out << (i ? ", {" : "{");
        for (int j = 0; j < 7; ++j) out << (j ? ", " : "") << g.scalar_info[i][j];
        out << '}';
    }
    out << "}\njoint_readout_coeff_count = " << coefficients.size()
        << "\njoint_readout_coeffs = {";
    for (size_t i = 0; i < coefficients.size(); ++i) out << (i ? ", " : "") << coefficients[i];
    out << "}\njoint_readout_optimizer_scales = {";
    for (size_t i = 0; i < scales.size(); ++i) out << (i ? ", " : "") << scales[i];
    out << "}\n";
}

void ReadJointReadoutSection(std::istream& in, JointReadoutTopology& g,
                            std::vector<double>& coefficients, std::vector<double>& scales,
                            int species_count, bool species_readout,
                            int l_max, int body_order, const std::vector<int>& body_l_max)
{
    // Compare a serialized graph to the canonical graph. This prevents an
    // arbitrary/malformed invariant graph from silently changing semantics.
    const JointReadoutTopology expected = BuildJointReadoutTopology(l_max, body_order, body_l_max);
    auto character = [&](char wanted) {
        char found = 0; if (!(in >> found) || found != wanted)
            ERROR("SUS2-SH malformed joint readout list or assignment");
    };
    auto field = [&](const char* wanted) {
        std::string name; if (!(in >> name) || name != wanted)
            ERROR(std::string("SUS2-SH joint readout missing ordered field ") + wanted);
        character('=');
    };
    auto integer = [&](int wanted) {
        int value = -1; if (!(in >> value) || value != wanted)
            ERROR("SUS2-SH joint readout graph/count metadata differs from canonical graph");
    };
    auto finite = [&]() {
        double value = 0.0; if (!(in >> value) || !std::isfinite(value))
            ERROR("SUS2-SH joint readout coefficient is not finite");
        return value;
    };
    character('='); std::string enabled;
    if (!(in >> enabled) || enabled != "true") ERROR("SUS2-SH joint readout field must be true");
    field("joint_readout_l_max"); integer(l_max);
    field("joint_readout_body_order"); integer(body_order);
    field("joint_readout_body_l_max"); character('{');
    for (int b = 2; b <= (body_order >= 6 ? 6 : 5); ++b) {
        if (b != 2) character(','); integer(body_l_max[b]);
    }
    character('}');
    field("joint_readout_alpha_moments_count"); integer(expected.moments_count);
    field("joint_readout_alpha_index_basic_count"); integer(expected.basic.size());
    field("joint_readout_alpha_index_basic"); character('{');
    for (size_t i = 0; i < expected.basic.size(); ++i) {
        if (i) character(','); character('{');
        for (int j = 0; j < 3; ++j) { if (j) character(','); integer(expected.basic[i][j]); }
        character('}');
    }
    character('}');
    field("joint_readout_sh_product_count"); integer(expected.products.size());
    field("joint_readout_sh_products"); character('{');
    g = expected;
    for (size_t i = 0; i < expected.products.size(); ++i) {
        if (i) character(','); character('{'); const auto& p = expected.products[i];
        integer(p.left); character(','); integer(p.right); character(','); integer(p.target); character(',');
        const double c = finite();
        if (std::abs(c - p.coeff) > 2e-13 * std::max(1.0, std::abs(p.coeff)))
            ERROR("SUS2-SH joint readout coupling differs from the canonical CG coefficient");
        g.products[i].coeff = c; character('}');
    }
    character('}');
    field("joint_readout_scalar_count"); integer(expected.scalar_mapping.size());
    field("joint_readout_scalar_mapping"); character('{');
    for (size_t i = 0; i < expected.scalar_mapping.size(); ++i) {
        if (i) character(','); integer(expected.scalar_mapping[i]);
    }
    character('}');
    field("joint_readout_scalar_info_count"); integer(expected.scalar_info.size());
    field("joint_readout_scalar_info"); character('{');
    for (size_t i = 0; i < expected.scalar_info.size(); ++i) {
        if (i) character(','); character('{');
        for (int j = 0; j < 7; ++j) { if (j) character(','); integer(expected.scalar_info[i][j]); }
        character('}');
    }
    character('}');
    const size_t count = expected.scalar_mapping.size() * (species_readout ? species_count : 1);
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) ERROR("Joint readout count overflow");
    field("joint_readout_coeff_count"); integer(count);
    field("joint_readout_coeffs"); character('{'); coefficients.resize(count);
    for (size_t i = 0; i < count; ++i) { if (i) character(','); coefficients[i] = finite(); }
    character('}');
    field("joint_readout_optimizer_scales"); character('{'); scales.resize(expected.scalar_mapping.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        if (i) character(','); scales[i] = finite();
        if (scales[i] <= 0.0) ERROR("SUS2-SH joint readout optimizer scales must be positive");
    }
    character('}');
}

void WriteSphericalHarmonicModel(const std::string& filename,
                                 const std::map<std::string, std::string>& opts)
{
	const int lmax = IntOpt(opts, "l-max", 3);
	const int kmax = IntOpt(opts, "k-max", 3);
	const int body_order = IntOpt(opts, "body-order", 5);
	const int species_count = IntOpt(opts, "species-count", 2);
	const int rb_size = IntOpt(opts, "radial-basis-size", 10);
	const double min_dist = DoubleOpt(opts, "min-dist", 1.5);
	const double max_dist = DoubleOpt(opts, "cutoff", DoubleOpt(opts, "max-dist", 7.5));
	const double scaling = DoubleOpt(opts, "scaling", 0.01);
	const std::string scaling_map = StringOpt(opts, "scaling-map", "LK");
	const std::string rbasis = CanonicalRadialBasisType(StringOpt(opts, "radial-basis-type", "RBChebyshev_sss"));
	const std::string radial_cutoff_type =
		CanonicalRadialCutoffType(StringOpt(opts, "radial-cutoff-type", "legacy"));
	const bool explicit_radial_cutoff_p = HasOpt(opts, "radial-cutoff-p");
	const int radial_cutoff_p = explicit_radial_cutoff_p
		? IntOpt(opts, "radial-cutoff-p", 5)
		: 5;
	const std::string superlinear_mode = StringOpt(opts, "superlinear-mode", "legacy");
	if (superlinear_mode != "legacy"
	    && superlinear_mode != "pure-w-channelwise"
	    && superlinear_mode != "radial-coefficient-gate")
		ERROR("--superlinear-mode should be legacy, pure-w-channelwise, or radial-coefficient-gate.");
	const bool pure_w_channelwise = superlinear_mode == "pure-w-channelwise";
	const bool radial_coefficient_gate =
		superlinear_mode == "radial-coefficient-gate";
	const std::string readout_mode = StringOpt(opts, "readout-mode", "shared");
	const bool joint_readout = BoolOpt(opts, "joint-readout", false);
	if (joint_readout && (!radial_coefficient_gate || lmax < 1))
		ERROR("--joint-readout requires adaptive radial coefficients with l-max >= 1");
	if (readout_mode != "shared" && readout_mode != "species")
		ERROR("--readout-mode should be shared or species.");
	if (readout_mode == "species" && pure_w_channelwise)
		ERROR("--readout-mode=species supports Legacy and adaptive radial coefficients, not pure-W.");
	const std::string adaptive_projection = StringOpt(opts, "adaptive-radial-projection", "full");
	const std::string adaptive_amplitude_mode = StringOpt(opts, "adaptive-radial-amplitude-mode", "shared");
	const int adaptive_shared_n_rank = StrictNonnegativeIntOpt(opts, "adaptive-radial-shared-n-rank", 1);
	const std::string adaptive_activation = StringOpt(opts, "adaptive-radial-activation", "none");
	if (adaptive_projection != "full" && adaptive_projection != "shared-n")
		ERROR("--adaptive-radial-projection must be full or shared-n.");
	if (adaptive_amplitude_mode != "shared" && adaptive_amplitude_mode != "species")
		ERROR("--adaptive-radial-amplitude-mode must be shared or species.");
	if (adaptive_shared_n_rank < 1 || adaptive_shared_n_rank > rb_size
	    || (adaptive_projection == "full" && adaptive_shared_n_rank != 1))
		ERROR("--adaptive-radial-shared-n-rank must be in [1,radial-basis-size]; full projection requires rank 1.");
	if (adaptive_activation != "none" && adaptive_activation != "tanh")
		ERROR("--adaptive-radial-activation must be none or tanh.");
	const bool adaptive_v17 = adaptive_shared_n_rank != 1 || adaptive_activation != "none";
	const bool adaptive_extended = adaptive_projection != "full" || adaptive_amplitude_mode != "shared" || adaptive_v17;
	if ((HasOpt(opts,"adaptive-radial-projection") || HasOpt(opts,"adaptive-radial-amplitude-mode")
	    || HasOpt(opts,"adaptive-radial-shared-n-rank") || HasOpt(opts,"adaptive-radial-activation")) && !radial_coefficient_gate)
		ERROR("Adaptive radial layout options require --superlinear-mode=radial-coefficient-gate.");
	const int tensor_message_lmax =
		StrictNonnegativeIntOpt(opts, "tensor-message-lmax", 0);
	if (tensor_message_lmax > 2)
		ERROR("--tensor-message-lmax currently supports only 0, 1, or 2.");
	if (tensor_message_lmax > 0 && !radial_coefficient_gate)
		ERROR("--tensor-message-lmax currently requires --superlinear-mode=radial-coefficient-gate.");
	if (tensor_message_lmax > lmax)
		ERROR("--tensor-message-lmax cannot exceed --l-max.");
	const SHFactorPruning factor_pruning = ParseFactorPruning(opts,
		"sh-factor-pruning", (pure_w_channelwise || radial_coefficient_gate)
			? SHFactorPruning::SameK : SHFactorPruning::QTotal);
	const bool outer_same_channel_mode =
		pure_w_channelwise
		|| factor_pruning == SHFactorPruning::SameK;
	const auto outer_k_rank1_alias_it = opts.find("outer-k-rank1");
	const bool outer_k_rank1_alias = outer_k_rank1_alias_it != opts.end();
	if (outer_k_rank1_alias && outer_k_rank1_alias_it->second != "true")
		ERROR("--outer-k-rank1 is a boolean compatibility flag; use it without '=value'.");
	const bool has_outer_k_rank = opts.find("outer-k-rank") != opts.end();
	if (outer_k_rank1_alias && has_outer_k_rank)
		ERROR("Use either --outer-k-rank1 or --outer-k-rank=<int>, not both.");
	const int outer_k_rank = has_outer_k_rank
		? StrictNonnegativeIntOpt(opts, "outer-k-rank", 0)
		: (outer_k_rank1_alias ? 1 : 0);
	const bool outer_k_rank_enabled = outer_k_rank > 0;
	const int two_layer_gate_k_rank =
		StrictNonnegativeIntOpt(opts, "two-layer-gate-k-rank",
			(pure_w_channelwise || radial_coefficient_gate) ? kmax : 0);
	const bool two_layer_gate_k_rank_enabled = two_layer_gate_k_rank > 0;
	const bool multi_layer_gate = HasOpt(opts, "multi-layer-gate");
	const bool two_layer_gate_readout = HasOpt(opts, "two-layer-gate-readout");
	const std::string two_layer_gate_radial_mixing =
		StringOpt(opts, "two-layer-gate-radial-mixing", pure_w_channelwise ? "full" : "none");
	const bool two_layer_gate_radial_mixing_full =
		two_layer_gate_radial_mixing == "full";
	const bool two_layer_gate = HasOpt(opts, "two-layer-gate") || multi_layer_gate
		|| two_layer_gate_readout || two_layer_gate_radial_mixing_full
		|| radial_coefficient_gate;
	const SHFactorPruning two_layer_gate_factor_pruning =
		ParseFactorPruning(
			opts, "two-layer-gate-factor-pruning",
			(pure_w_channelwise || radial_coefficient_gate)
				? SHFactorPruning::QTotal : factor_pruning);
	const SHFactorPruning second_layer_gate_factor_pruning =
		ParseFactorPruning(
			opts, "second-layer-gate-factor-pruning", two_layer_gate_factor_pruning);
	const bool two_layer_gate_edge_l1 = HasOpt(opts, "two-layer-gate-edge-l1");
	const bool requested_two_layer_gate_shared_radial =
		HasOpt(opts, "two-layer-gate-shared-radial");
	const bool two_layer_gate_decoupled_scaling =
		HasOpt(opts, "two-layer-gate-decoupled-scaling")
		|| HasOpt(opts, "two-layer-gate-full-decoupled");
	const bool two_layer_gate_shared_radial = two_layer_gate;
	const bool two_layer_residual = HasOpt(opts, "two-layer-residual");
	if (superlinear_mode == "legacy" && readout_mode == "species"
	    && (two_layer_residual || multi_layer_gate || two_layer_gate_readout || two_layer_gate_edge_l1))
		ERROR("Legacy species readout does not support residual, multi-layer, auxiliary gate readout, or edge-L1 modes.");
	const int default_two_layer_gate_body_order = body_order;
	const int two_layer_gate_body_order =
		IntOpt(opts, "two-layer-gate-body-order", default_two_layer_gate_body_order);
	std::string two_layer_gate_mode =
		StringOpt(opts, "two-layer-gate-mode",
			radial_coefficient_gate ? "mu-scalar-full" : "mu-body-linear-combo");
	if (two_layer_gate_mode == "mu-body-order")
		two_layer_gate_mode = "mu-scalar-full";
	std::string second_layer_gate_mode =
		StringOpt(opts, "second-layer-gate-mode", two_layer_gate_mode);
	if (second_layer_gate_mode == "mu-body-order")
		second_layer_gate_mode = "mu-scalar-full";
	const int second_layer_gate_body_order =
		IntOpt(opts, "second-layer-gate-body-order", body_order);
	const std::string two_layer_gate_site_mode =
		StringOpt(opts, "two-layer-gate-site-mode", "neighbor");
	const bool write_scalar_info =
		HasOpt(opts, "write-sh-scalar-info") || two_layer_gate
		|| multi_layer_gate || outer_k_rank_enabled;
	std::ostringstream default_name;
	default_name << "sus2sh_l" << lmax << "k" << kmax << "_b" << body_order;
	const std::string name = StringOpt(opts, "potential-name", default_name.str());
	const std::vector<int> body_lmax = ParseBodyLMax(opts, lmax, body_order);
	const ZBLInitOptions zbl_options = ParseZBLInitOptions(opts, species_count);

	if (lmax < 0 || lmax > 6)
		ERROR("init-sh currently supports --l-max from 0 to 6.");
	if (kmax <= 0)
		ERROR("init-sh requires --k-max > 0.");
	if (body_order < 2 || body_order > 6)
		ERROR("init-sh supports --body-order from 2 to 6.");
	if (scaling_map != "K" && scaling_map != "L" && scaling_map != "LK" && scaling_map != "pair")
		ERROR("init-sh --scaling-map should be K, L, LK, or pair.");
	if (pure_w_channelwise && !outer_k_rank_enabled)
		ERROR("--superlinear-mode=pure-w-channelwise requires --outer-k-rank=<O> with O > 0 (O may exceed K).");
	if (radial_coefficient_gate && !outer_k_rank_enabled)
		ERROR("--superlinear-mode=radial-coefficient-gate requires --outer-k-rank=<O> with 0 < O <= K.");
	if (pure_w_channelwise && factor_pruning != SHFactorPruning::SameK)
		ERROR("--superlinear-mode=pure-w-channelwise requires --sh-factor-pruning=same-k or omission.");
	if (radial_coefficient_gate
	    && factor_pruning != SHFactorPruning::SameK
	    && factor_pruning != SHFactorPruning::QTotal)
		ERROR("--superlinear-mode=radial-coefficient-gate supports --sh-factor-pruning=same-k or q-total.");
	if (outer_k_rank < 0
	    || ((!pure_w_channelwise && !radial_coefficient_gate)
	        && outer_k_rank > kmax))
		ERROR("--outer-k-rank must be in [1, --k-max] when enabled.");
	if (radial_coefficient_gate && outer_k_rank > kmax)
		ERROR("--superlinear-mode=radial-coefficient-gate requires O <= raw K.");
	if (two_layer_gate_k_rank < 0 || two_layer_gate_k_rank > kmax)
		ERROR("--two-layer-gate-k-rank must be in [1, --k-max] when enabled.");
	if (two_layer_gate_k_rank_enabled && !HasOpt(opts, "two-layer-gate")
	    && !pure_w_channelwise && !radial_coefficient_gate)
		ERROR("--two-layer-gate-k-rank requires an explicit --two-layer-gate model.");
	if (two_layer_gate_k_rank_enabled && !outer_k_rank_enabled)
		ERROR("--two-layer-gate-k-rank currently requires --outer-k-rank=<R>.");
	if (outer_k_rank_enabled && !pure_w_channelwise && !radial_coefficient_gate
	    && factor_pruning != SHFactorPruning::QTotal
	    && factor_pruning != SHFactorPruning::SameK)
		ERROR("--outer-k-rank supports --sh-factor-pruning=q-total or same-k.");
	if (two_layer_gate_k_rank_enabled
	    && two_layer_gate_factor_pruning != SHFactorPruning::QTotal
	    && two_layer_gate_factor_pruning != SHFactorPruning::SameK)
		ERROR("--two-layer-gate-k-rank supports q-total or same-k.");
	if (HasOpt(opts, "two-layer-gate-body-order") && !two_layer_gate)
		ERROR("--two-layer-gate-body-order requires --two-layer-gate.");
	if (HasOpt(opts, "two-layer-gate-factor-pruning") && !two_layer_gate)
		ERROR("--two-layer-gate-factor-pruning requires --two-layer-gate.");
	if (two_layer_gate && !pure_w_channelwise && !radial_coefficient_gate
	    && !two_layer_gate_k_rank_enabled
	    && !IsFactorPruningSubset(two_layer_gate_factor_pruning, factor_pruning))
		ERROR("--two-layer-gate-factor-pruning must select a subgraph of --sh-factor-pruning.");
	if (two_layer_gate
	    && (two_layer_gate_body_order < 2
	        || two_layer_gate_body_order
	           > ((pure_w_channelwise || radial_coefficient_gate) ? 6 : body_order)))
		ERROR(((pure_w_channelwise || radial_coefficient_gate)
			? "--two-layer-gate-body-order must be in [2, 6]."
			: "--two-layer-gate-body-order must be in [2, --body-order]."));
	if (HasOpt(opts, "second-layer-gate-body-order") && !multi_layer_gate)
		ERROR("--second-layer-gate-body-order requires --multi-layer-gate.");
	if (HasOpt(opts, "second-layer-gate-factor-pruning") && !multi_layer_gate)
		ERROR("--second-layer-gate-factor-pruning requires --multi-layer-gate.");
	if (multi_layer_gate
	    && !IsFactorPruningSubset(second_layer_gate_factor_pruning, factor_pruning))
		ERROR("--second-layer-gate-factor-pruning must select a subgraph of --sh-factor-pruning.");
	if (multi_layer_gate
	    && (second_layer_gate_body_order < 2
	        || second_layer_gate_body_order > body_order))
		ERROR("--second-layer-gate-body-order must be in [2, --body-order].");
	if (HasOpt(opts, "second-layer-gate-mode") && !multi_layer_gate)
		ERROR("--second-layer-gate-mode requires --multi-layer-gate.");
	if (two_layer_gate_edge_l1 && !two_layer_gate)
		ERROR("--two-layer-gate-edge-l1 requires --two-layer-gate.");
	if (two_layer_gate_edge_l1 && lmax < 1)
		ERROR("--two-layer-gate-edge-l1 requires --l-max >= 1.");
	if (two_layer_gate_radial_mixing != "none"
	    && two_layer_gate_radial_mixing != "full")
		ERROR("--two-layer-gate-radial-mixing should be 'none' or 'full'.");
	if (pure_w_channelwise
	    && (!two_layer_gate_radial_mixing_full || !two_layer_gate_k_rank_enabled))
		ERROR("pure-w-channelwise requires full radial mixing and a positive gate rank.");
	if (radial_coefficient_gate
	    && (two_layer_gate_radial_mixing_full || !two_layer_gate_k_rank_enabled))
		ERROR("radial-coefficient-gate requires adaptive coefficients, no legacy radial mixing, and a positive gate rank.");
	if (HasOpt(opts, "two-layer-gate-radial-mixing")
	    && !two_layer_gate_radial_mixing_full)
		ERROR("--two-layer-gate-radial-mixing currently supports only explicit '=full' or omission.");
	if (two_layer_gate_mode != "mu-body-linear-combo"
	    && two_layer_gate_mode != "mu-scalar-full")
		ERROR("--two-layer-gate-mode should be mu-body-linear-combo or mu-scalar-full.");
	if (radial_coefficient_gate && two_layer_gate_mode != "mu-scalar-full")
		ERROR("--superlinear-mode=radial-coefficient-gate requires --two-layer-gate-mode=mu-scalar-full.");
	if (second_layer_gate_mode != "mu-body-linear-combo"
	    && second_layer_gate_mode != "mu-scalar-full")
		ERROR("--second-layer-gate-mode should be mu-body-linear-combo or mu-scalar-full.");
	if (HasOpt(opts, "two-layer-gate-mode") && !two_layer_gate)
		ERROR("--two-layer-gate-mode requires --two-layer-gate.");
	if (HasOpt(opts, "two-layer-gate-tanh-amplitude"))
		ERROR("--two-layer-gate-tanh-amplitude is obsolete in the release additive-node gate.");
	if (two_layer_gate_site_mode != "neighbor")
		ERROR("--two-layer-gate-site-mode=double is obsolete; release gate supports 'neighbor' only.");
	if (HasOpt(opts, "two-layer-gate-site-mode") && !two_layer_gate)
		ERROR("--two-layer-gate-site-mode requires --two-layer-gate.");
	if (two_layer_gate_edge_l1 && two_layer_gate_site_mode != "neighbor")
		ERROR("--two-layer-gate-edge-l1 first version supports neighbor site mode only.");
	if (multi_layer_gate && two_layer_gate_edge_l1)
		ERROR("--multi-layer-gate is not compatible with --two-layer-gate-edge-l1.");
	if (two_layer_gate_radial_mixing_full && two_layer_gate_edge_l1)
		ERROR("--two-layer-gate-radial-mixing=full is not compatible with --two-layer-gate-edge-l1.");
	if (two_layer_gate_radial_mixing_full && multi_layer_gate)
		ERROR("--two-layer-gate-radial-mixing=full is not compatible with --multi-layer-gate.");
	if (two_layer_gate_radial_mixing_full
	    && (!outer_k_rank_enabled || !two_layer_gate_k_rank_enabled))
		ERROR("--two-layer-gate-radial-mixing=full requires --outer-k-rank=<Ro> and --two-layer-gate-k-rank=<Rg>.");
	if (two_layer_gate_readout && multi_layer_gate)
		ERROR("--two-layer-gate-readout first version supports one gate only.");
	if (two_layer_gate_readout && two_layer_gate_edge_l1)
		ERROR("--two-layer-gate-readout first version does not include edge-L1.");
	if (requested_two_layer_gate_shared_radial && !two_layer_gate)
		ERROR("--two-layer-gate-shared-radial requires --two-layer-gate.");
	if (two_layer_gate_decoupled_scaling && !two_layer_gate)
		ERROR("--two-layer-gate-decoupled-scaling requires --two-layer-gate.");
	if (two_layer_gate_decoupled_scaling && multi_layer_gate)
		ERROR("--two-layer-gate-decoupled-scaling is only supported for one-layer gate models.");
	if (two_layer_residual && !two_layer_gate)
		ERROR("--two-layer-residual requires --two-layer-gate.");
	if (two_layer_residual && two_layer_gate)
		ERROR("--two-layer-residual is not supported by mu-body-order gate models.");
	if (two_layer_gate_readout && two_layer_residual)
		ERROR("--two-layer-gate-readout is not compatible with --two-layer-residual.");
	if (radial_coefficient_gate
	    && (two_layer_gate_readout || two_layer_residual
	        || multi_layer_gate || two_layer_gate_edge_l1))
		ERROR("--superlinear-mode=radial-coefficient-gate does not support readout, residual, multi-layer, or edge-L1.");
	if (outer_k_rank_enabled && two_layer_gate_readout)
		ERROR("--outer-k-rank is not compatible with --two-layer-gate-readout.");
	if (outer_k_rank_enabled && two_layer_residual)
		ERROR("--outer-k-rank is not compatible with --two-layer-residual.");
	if (outer_k_rank_enabled && multi_layer_gate)
		ERROR("--outer-k-rank currently supports one gate only.");
	if (two_layer_gate_k_rank_enabled
	    && (two_layer_gate_readout || two_layer_residual
	        || multi_layer_gate))
		ERROR("--two-layer-gate-k-rank is not compatible with readout, residual, or multi-layer gate.");
	if (!IsSupportedSHRadialBasis(rbasis))
		ERROR("init-sh currently writes RBChebyshev_sss, RBChebyshev_sss_rational, RBLaguerre_log1p, or RBJacobi_sss models.");
	if (radial_cutoff_p < 1 || radial_cutoff_p > 64)
		ERROR("--radial-cutoff-p should be in [1, 64].");
	if (radial_cutoff_type == "legacy" && explicit_radial_cutoff_p)
		ERROR("--radial-cutoff-p only applies to --radial-cutoff-type=polynomial.");
	if (rbasis == "RBJacobi_sss" && kmax > 6)
		ERROR("init-sh with RBJacobi_sss supports --k-max up to 6 because Jacobi blocks are indexed as k=0..5.");
	if (pure_w_channelwise || radial_coefficient_gate) {
		if (rb_size <= 0)
			ERROR("same-channel superlinear modes require --radial-basis-size > 0.");
		const size_t angular_count = static_cast<size_t>(lmax + 1);
		CheckedGraphCount({static_cast<size_t>(kmax), angular_count, angular_count},
			"Raw basic count");
		CheckedGraphCount({static_cast<size_t>(outer_k_rank), angular_count, angular_count},
			"Outer basic count");
		if (pure_w_channelwise)
			CheckedGraphCount({angular_count, static_cast<size_t>(outer_k_rank),
				static_cast<size_t>(kmax), static_cast<size_t>(rb_size)},
				"Full W coefficient count");
	}

	SHGraphBuilder::CGCache cg_cache;
	SHGraphBuilder graph(lmax, kmax, factor_pruning, &cg_cache);
	graph.Build();
	if (outer_k_rank_enabled || two_layer_gate_k_rank_enabled)
		graph.AddAllBasics();
	// A gate-free outer-rank model needs only the raw SH primitives before the
	// K contraction.  Preserve the full raw product graph only when it is the
	// energy graph, or when an unranked gate (including edge-L1 sources) uses it.
	if (!outer_k_rank_enabled
	    || (two_layer_gate && !two_layer_gate_k_rank_enabled))
		graph.AddScalars(body_order, body_lmax);
	SHGraphBuilder gate_rank_graph(
		lmax, std::max(1, two_layer_gate_k_rank),
		two_layer_gate_factor_pruning, &cg_cache);
	if (two_layer_gate_k_rank_enabled) {
		gate_rank_graph.Build();
		gate_rank_graph.AddAllBasics();
		if (two_layer_gate_factor_pruning == SHFactorPruning::SameK) {
			SHGraphBuilder gate_angular(
				lmax, 1, SHFactorPruning::SameK, &cg_cache);
			gate_angular.Build();
			gate_angular.AddAllBasics();
			gate_angular.AddScalars(
				two_layer_gate_edge_l1 ? body_order : two_layer_gate_body_order,
				body_lmax);
			gate_rank_graph.BindSingleChannelPlan(gate_angular);
		} else {
			gate_rank_graph.AddScalars(
				two_layer_gate_edge_l1 ? body_order : two_layer_gate_body_order,
				body_lmax);
		}
		if (gate_rank_graph.basic_.size()
		    != static_cast<size_t>(
			   two_layer_gate_k_rank * (lmax + 1) * (lmax + 1)))
			ERROR("--two-layer-gate-k-rank generated an unexpected primitive layout.");
		if (gate_rank_graph.scalars_.empty())
			ERROR("--two-layer-gate-k-rank generated no gate scalar basis functions.");
	}
	SHGraphBuilder outer_rank_graph(
		lmax, std::max(1, outer_k_rank), factor_pruning, &cg_cache);
	if (outer_k_rank_enabled) {
		outer_rank_graph.Build();
		outer_rank_graph.AddAllBasics();
		if (outer_same_channel_mode) {
			SHGraphBuilder angular(lmax, 1, SHFactorPruning::SameK, &cg_cache);
			angular.Build();
			angular.AddAllBasics();
			angular.AddScalars(body_order, body_lmax);
			outer_rank_graph.BindSingleChannelPlan(angular);
			std::cout << superlinear_mode << ": outer angular plan built once; O="
			          << outer_k_rank << " angular_scalars=" << angular.scalars_.size()
			          << " outer_scalars=" << outer_rank_graph.scalars_.size()
			          << " gate_rank=" << two_layer_gate_k_rank
			          << (HasOpt(opts, "two-layer-gate-k-rank") ? " (explicit)" : " (default=K)")
			          << std::endl;
		} else {
			outer_rank_graph.AddScalars(body_order, body_lmax);
		}
		if (outer_rank_graph.basic_.size()
		    != static_cast<size_t>(outer_k_rank * (lmax + 1) * (lmax + 1)))
			ERROR("--outer-k-rank generated an unexpected contracted basic-moment layout.");
		if (outer_rank_graph.scalars_.empty())
			ERROR("--outer-k-rank generated no outer scalar basis functions.");
	}

	std::ofstream ofs(filename.c_str());
	if (!ofs.is_open())
		ERROR("Cannot open output model for writing: " + filename);
	ofs.setf(std::ios::scientific);
	ofs.precision(15);

	ofs << "MTP\n";
	const bool uses_extended_sh_format =
		outer_k_rank_enabled || two_layer_gate_k_rank_enabled
		|| two_layer_gate_readout;
	// Ranked gate and energy graphs are serialized independently below. New
	// Legacy channel/readout layouts must fail closed in historical readers.
	const bool legacy_v18 = superlinear_mode == "legacy"
		&& (readout_mode == "species"
		    || (outer_k_rank_enabled && outer_same_channel_mode)
		    || (two_layer_gate_k_rank_enabled
		        && two_layer_gate_factor_pruning == SHFactorPruning::SameK));
	ofs << "version = "
	    << (legacy_v18 ? "1.8.0" : (adaptive_v17 ? "1.7.0" : (joint_readout ? "1.6.0" : (adaptive_extended ? "1.5.0" : (tensor_message_lmax > 0 ? "1.4.0" : (readout_mode == "species" ? "1.3.0"
	        : (uses_extended_sh_format ? "1.2.0" : "1.1.0"))))))) << "\n";
	ofs << "potential_name = " << name << "\n";
	ofs << "scaling = " << scaling << "\n";
	ofs << "L = " << lmax << "\n";
	ofs << "scaling_map = " << scaling_map << "\n";
	ofs << "species_count = " << species_count << "\n";
	WriteZBLSection(ofs, zbl_options);
	ofs << "potential_tag = SUS2-SH\n";
	ofs << "sh_l_max = " << lmax << "\n";
	ofs << "sh_k_max = " << kmax << "\n";
	ofs << "sh_body_order = " << body_order << "\n";
	ofs << "sh_parity = even\n";
	ofs << "sh_body_l_max = {" << body_lmax[2] << ", " << body_lmax[3]
	    << ", " << body_lmax[4] << ", " << body_lmax[5];
	if (body_order >= 6)
		ofs << ", " << body_lmax[6];
	ofs << "}\n";
	ofs << "radial_basis_type = " << rbasis << "\n";
	ofs << "\tmin_dist = " << min_dist << "\n";
	ofs << "\tmax_dist = " << max_dist << "\n";
	ofs << "\tradial_basis_size = " << rb_size << "\n";
	if (radial_cutoff_type != "legacy") {
		ofs << "\tradial_cutoff_type = " << radial_cutoff_type << "\n";
		ofs << "\tradial_cutoff_p = " << radial_cutoff_p << "\n";
	}
	const int radial_func_count = kmax * (lmax + 1);
	const int scaling_block_count =
		(scaling_map == "pair") ? 1
		: (scaling_map == "K") ? kmax
		: (scaling_map == "L") ? (lmax + 1)
		: radial_func_count;
	ofs << "\tradial_funcs_count = " << radial_func_count << "\n";

	const SHGraphBuilder& energy_graph =
		outer_k_rank_enabled ? outer_rank_graph : graph;
	const int gate_rank_node_offset = graph.node_count_;
	const size_t outer_offset_count = static_cast<size_t>(graph.node_count_)
		+ (two_layer_gate_k_rank_enabled ? static_cast<size_t>(gate_rank_graph.node_count_) : 0);
	const int outer_rank_node_offset = CheckedGraphCount({outer_offset_count}, "Outer graph offset");
	const int alpha_moments_count = CheckedGraphCount({outer_offset_count
		+ (outer_k_rank_enabled ? static_cast<size_t>(outer_rank_graph.node_count_) : 0)},
		"Combined graph node count");
	ofs << "alpha_moments_count = " << alpha_moments_count << "\n";
	ofs << "alpha_index_basic_count = " << graph.basic_.size() << "\n";
	ofs << "alpha_index_basic = {";
	for (size_t i = 0; i < graph.basic_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		const BasicIndex& b = graph.basic_[i];
		ofs << "{" << b.k << ", " << b.l << ", " << b.m << "}";
	}
	ofs << "}\n";
	ofs << "alpha_index_times_count = 0\n";
	ofs << "alpha_index_times = {}\n";
	const size_t sh_product_count = graph.products_.size()
		+ (two_layer_gate_k_rank_enabled ? gate_rank_graph.products_.size() : 0)
		+ (outer_k_rank_enabled ? outer_rank_graph.products_.size() : 0);
	CheckedGraphCount({sh_product_count}, "Combined graph product count");
	ofs << "sh_product_count = " << sh_product_count << "\n";
	ofs << "sh_products = {";
	for (size_t i = 0; i < graph.products_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		const Product& p = graph.products_[i];
		ofs << "{" << p.left << ", " << p.right << ", " << p.target << ", " << p.coeff << "}";
	}
	if (two_layer_gate_k_rank_enabled) {
		for (size_t i = 0; i < gate_rank_graph.products_.size(); ++i) {
			if (!graph.products_.empty() || i != 0)
				ofs << ", ";
			const Product& p = gate_rank_graph.products_[i];
			ofs << "{" << p.left + gate_rank_node_offset << ", "
			    << p.right + gate_rank_node_offset << ", "
			    << p.target + gate_rank_node_offset << ", "
			    << p.coeff << "}";
		}
	}
	if (outer_k_rank_enabled) {
		for (size_t i = 0; i < outer_rank_graph.products_.size(); ++i) {
			if (!graph.products_.empty()
			    || (two_layer_gate_k_rank_enabled
			        && !gate_rank_graph.products_.empty())
			    || i != 0)
				ofs << ", ";
			const Product& p = outer_rank_graph.products_[i];
			ofs << "{" << p.left + outer_rank_node_offset << ", "
			    << p.right + outer_rank_node_offset << ", "
			    << p.target + outer_rank_node_offset << ", "
			    << p.coeff << "}";
		}
	}
	ofs << "}\n";
	ofs << "alpha_scalar_moments = " << energy_graph.scalars_.size() << "\n";
	ofs << "alpha_moment_mapping = {";
	for (size_t i = 0; i < energy_graph.scalars_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		ofs << energy_graph.scalars_[i]
		    + (outer_k_rank_enabled ? outer_rank_node_offset : 0);
	}
	ofs << "}\n";
	if (write_scalar_info) {
		ofs << "sh_scalar_info_count = " << energy_graph.scalar_infos_.size() << "\n";
		ofs << "sh_scalar_info = {";
		for (size_t i = 0; i < energy_graph.scalar_infos_.size(); ++i) {
			if (i != 0)
				ofs << ", ";
			const ScalarSpec& spec = energy_graph.scalar_infos_[i];
			ofs << "{" << spec.body_order << ", "
			    << spec.q0 << ", " << spec.q1 << ", " << spec.q2 << ", "
			    << spec.q3 << ", " << spec.q4 << ", " << spec.intermediate_l << "}";
		}
		ofs << "}\n";
	}
	if (outer_k_rank_enabled) {
		const int outer_k_weight_count =
			(pure_w_channelwise || radial_coefficient_gate)
				? 0 : outer_k_rank * kmax * (lmax + 1);
		ofs << "outer_k_rank1_enabled = true\n";
		// Preserve bidirectional compatibility with the original rank-1
		// prototype, whose reader does not know the generic rank field.
		if (outer_k_rank != 1)
			ofs << "outer_k_rank = " << outer_k_rank << "\n";
		if (legacy_v18) {
			ofs << "outer_channel_mode = "
			    << (outer_same_channel_mode ? "same-channel" : "cross-channel") << "\n";
			ofs << "outer_radial_mode = legacy-plus-mixer\n";
			ofs << "readout_mode = " << readout_mode << "\n";
		}
		if (pure_w_channelwise) {
			ofs << "outer_channel_mode = same-channel\n";
			ofs << "outer_radial_mode = pure-matrix\n";
		}
			if (radial_coefficient_gate) {
				ofs << "outer_channel_mode = "
				    << (outer_same_channel_mode ? "same-channel" : "cross-channel")
				    << "\n";
				ofs << "outer_radial_mode = adaptive-coefficients\n";
				if (readout_mode == "species" || tensor_message_lmax > 0 || adaptive_extended || joint_readout)
					ofs << "readout_mode = " << readout_mode << "\n";
				if (adaptive_extended || joint_readout) {
					ofs << "adaptive_radial_projection = " << adaptive_projection << "\n";
					ofs << "adaptive_radial_amplitude_mode = " << adaptive_amplitude_mode << "\n";
				}
				if (adaptive_v17) {
					ofs << "adaptive_radial_shared_n_rank = " << adaptive_shared_n_rank << "\n";
					ofs << "adaptive_radial_activation = " << adaptive_activation << "\n";
				}
				if (tensor_message_lmax > 0 || adaptive_extended || joint_readout)
					ofs << "tensor_message_lmax = " << tensor_message_lmax << "\n";
				if (joint_readout) {
					const auto joint = BuildJointReadoutTopology(lmax, body_order, body_lmax);
					WriteJointReadoutSection(ofs, joint,
						std::vector<double>(joint.scalar_mapping.size() * (readout_mode == "species" ? species_count : 1), 0.0),
						std::vector<double>(joint.scalar_mapping.size(), 1.0));
				}
				else if (adaptive_v17) ofs << "joint_readout_enabled = false\n";
			}
		ofs << "outer_k_rank1_basic_begin = "
		    << outer_rank_node_offset << "\n";
		ofs << "outer_k_rank1_basic_count = "
		    << outer_rank_graph.basic_.size() << "\n";
		ofs << "outer_k_rank1_product_begin = "
		    << graph.products_.size()
			   + (two_layer_gate_k_rank_enabled
			      ? gate_rank_graph.products_.size() : 0) << "\n";
		ofs << "outer_k_rank1_weight_count = "
		    << outer_k_weight_count << "\n";
		ofs << "outer_k_rank1_weights = {";
		for (int l = 0; outer_k_weight_count > 0 && l <= lmax; ++l) {
			for (int channel = 0; channel < outer_k_rank; ++channel) {
				for (int k = 0; k < kmax; ++k) {
					const int index =
						(l * outer_k_rank + channel) * kmax + k;
					if (index != 0)
						ofs << ", ";
					// For rank 1, W is only a serialization compatibility shell:
					// the loader absorbs it into the freely trainable radial rows.
					// Use all ones so every original K channel is active from the
					// start. Higher ranks retain the learned identity initialization.
					ofs << (outer_k_rank == 1
						? 1.0 : (k == channel ? 1.0 : 0.0));
				}
			}
		}
		ofs << "}\n";
	}
	if (legacy_v18 && !outer_k_rank_enabled) {
		ofs << "readout_mode = " << readout_mode << "\n";
		ofs << "outer_channel_mode = cross-channel\n";
		ofs << "outer_radial_mode = legacy-plus-mixer\n";
	}
	if (two_layer_gate_k_rank_enabled) {
		const int gate_k_weight_count =
			two_layer_gate_k_rank * kmax * (lmax + 1);
		ofs << "two_layer_gate_k_rank = " << two_layer_gate_k_rank << "\n";
		ofs << "two_layer_gate_k_rank_basic_begin = "
		    << gate_rank_node_offset << "\n";
		ofs << "two_layer_gate_k_rank_basic_count = "
		    << gate_rank_graph.basic_.size() << "\n";
		ofs << "two_layer_gate_k_rank_product_begin = "
		    << graph.products_.size() << "\n";
		ofs << "two_layer_gate_k_rank_product_end = "
		    << graph.products_.size() + gate_rank_graph.products_.size() << "\n";
		ofs << "two_layer_gate_k_rank_weight_count = "
		    << gate_k_weight_count << "\n";
		ofs << "two_layer_gate_k_rank_weights = {";
		for (int l = 0; l <= lmax; ++l) {
			for (int channel = 0; channel < two_layer_gate_k_rank; ++channel) {
				for (int k = 0; k < kmax; ++k) {
					const int index =
						(l * two_layer_gate_k_rank + channel) * kmax + k;
					if (index != 0)
						ofs << ", ";
					ofs << (k == channel ? 1.0 : 0.0);
				}
			}
		}
		ofs << "}\n";
	}
	if (two_layer_gate) {
		std::vector<int> gate_body_counts;
		std::vector<int> gate_scalar_indices;
		if (two_layer_gate_k_rank_enabled) {
			if (two_layer_gate_edge_l1) {
				gate_scalar_indices = gate_rank_graph.ScalarIndices(
					two_layer_gate_body_order,
					two_layer_gate_factor_pruning,
					gate_body_counts);
				for (int q = 0; q < static_cast<int>(gate_scalar_indices.size()); ++q)
					if (gate_scalar_indices[q] != q)
						ERROR("--two-layer-gate-k-rank edge-L1 requires canonical gate scalar ordinals.");
			} else {
				gate_body_counts.assign(two_layer_gate_body_order + 1, 0);
				gate_scalar_indices.resize(gate_rank_graph.scalars_.size());
				for (int q = 0; q < static_cast<int>(gate_scalar_indices.size()); ++q) {
					gate_scalar_indices[q] = q;
					++gate_body_counts[gate_rank_graph.scalar_infos_[q].body_order];
				}
			}
		} else {
			gate_scalar_indices = graph.ScalarIndices(
				two_layer_gate_body_order,
				two_layer_gate_factor_pruning,
				gate_body_counts);
		}
		if (gate_scalar_indices.empty())
			ERROR("--two-layer-gate selected no SH scalar basis functions.");
		for (int bo = 2; bo <= two_layer_gate_body_order; ++bo)
			if (gate_body_counts[bo] == 0)
				ERROR("--two-layer-gate selected no SH scalar basis functions for one required mu body-order bucket.");

			ofs << "two_layer_gate_enabled = true\n";
			ofs << "two_layer_gate_mode = " << two_layer_gate_mode << "\n";
			ofs << "two_layer_gate_body_order_max = " << two_layer_gate_body_order << "\n";
			ofs << "two_layer_gate_include_one_body = false\n";
			ofs << "two_layer_gate_site_mode = " << two_layer_gate_site_mode << "\n";
			ofs << "two_layer_gate_factor_pruning = "
			    << FactorPruningName(two_layer_gate_factor_pruning) << "\n";
			ofs << "two_layer_gate_scale_mode = additive-node\n";
			if (two_layer_gate_shared_radial) {
				const int gate_radial_count = kmax * (lmax + 1) * rb_size;
				ofs << "two_layer_gate_radial_mode = shared-radial\n";
				ofs << "two_layer_gate_radial_coeff_count = " << gate_radial_count << "\n";
			ofs << "two_layer_gate_radial_coeffs = {";
			for (int i = 0; i < gate_radial_count; ++i) {
				if (i != 0)
					ofs << ", ";
				ofs << 1.0e-2;
			}
			ofs << "}\n";
			}
			if (two_layer_gate_decoupled_scaling) {
				const int gate_scaling_count =
					2 * species_count * species_count * scaling_block_count;
				ofs << "two_layer_gate_scaling_mode = decoupled\n";
				ofs << "two_layer_gate_scaling_coeff_count = "
				    << gate_scaling_count << "\n";
				ofs << "two_layer_gate_scaling_coeffs = {";
				WriteDefaultScalingCoeffList(
					ofs, species_count, scaling_block_count, rbasis);
				ofs << "}\n";
			}
			if (!radial_coefficient_gate) {
				const int gate_additive_count = species_count * radial_func_count;
				ofs << "two_layer_gate_additive_coeff_count = "
				    << gate_additive_count << "\n";
				ofs << "two_layer_gate_additive_coeffs = {";
				for (int i = 0; i < gate_additive_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << (1.0 / 12.0);
				}
				ofs << "}\n";
			}
			const int gate_type_count = species_count * radial_func_count;
			ofs << "two_layer_gate_type_coeff_count = "
			    << gate_type_count << "\n";
			ofs << "two_layer_gate_type_coeffs = {";
			for (int i = 0; i < gate_type_count; ++i) {
				if (i != 0)
					ofs << ", ";
				ofs << 1.0;
			}
			ofs << "}\n";
			const int gate_weight_count =
				radial_coefficient_gate ? 0
				: (two_layer_gate_mode == "mu-scalar-full")
					? radial_func_count * static_cast<int>(gate_scalar_indices.size())
				: static_cast<int>(gate_scalar_indices.size());
		if (radial_coefficient_gate)
			ofs << "two_layer_gate_scalar_count = "
			    << gate_scalar_indices.size() << "\n";
		ofs << "two_layer_gate_weight_count = " << gate_weight_count << "\n";
		ofs << "two_layer_gate_scalar_indices = {";
		for (size_t i = 0; i < gate_scalar_indices.size(); ++i) {
			if (i != 0)
				ofs << ", ";
			ofs << gate_scalar_indices[i];
		}
		ofs << "}\n";
		if (outer_k_rank_enabled || two_layer_gate_k_rank_enabled) {
			ofs << "two_layer_gate_scalar_moment_indices = {";
			for (size_t i = 0; i < gate_scalar_indices.size(); ++i) {
				if (i != 0)
					ofs << ", ";
				const int scalar_index = gate_scalar_indices[i];
				if (two_layer_gate_k_rank_enabled) {
					if (scalar_index < 0
					    || scalar_index >= static_cast<int>(gate_rank_graph.scalars_.size()))
						ERROR("--two-layer-gate-k-rank scalar index is out of range.");
					ofs << gate_rank_graph.scalars_[scalar_index]
					    + gate_rank_node_offset;
				} else {
					if (scalar_index < 0
					    || scalar_index >= static_cast<int>(graph.scalars_.size()))
						ERROR("--outer-k-rank gate scalar index is out of range.");
					ofs << graph.scalars_[scalar_index];
				}
			}
			ofs << "}\n";
			ofs << "two_layer_gate_scalar_body_orders = {";
			for (size_t i = 0; i < gate_scalar_indices.size(); ++i) {
				if (i != 0)
					ofs << ", ";
				const int scalar_index = gate_scalar_indices[i];
				ofs << (two_layer_gate_k_rank_enabled
					? gate_rank_graph.scalar_infos_[scalar_index].body_order
					: graph.scalar_infos_[scalar_index].body_order);
			}
			ofs << "}\n";
		}
		ofs << "two_layer_gate_weights = {";
		for (int i = 0; i < gate_weight_count; ++i) {
			if (i != 0)
				ofs << ", ";
			ofs << 1.0e-1;
		}
		ofs << "}\n";
		if (!radial_coefficient_gate
		    && two_layer_gate_mode == "mu-body-linear-combo") {
			const int gate_body_mix_weight_count =
				radial_func_count * (two_layer_gate_body_order - 1);
			ofs << "two_layer_gate_body_mix_weight_count = "
			    << gate_body_mix_weight_count << "\n";
			ofs << "two_layer_gate_body_mix_weights = {";
			for (int i = 0; i < gate_body_mix_weight_count; ++i) {
				if (i != 0)
					ofs << ", ";
				ofs << 1.0;
			}
			ofs << "}\n";
		}
		if (two_layer_gate_radial_mixing_full) {
			const int mixing_count =
				(lmax + 1) * outer_k_rank * kmax * rb_size;
			ofs << "two_layer_gate_radial_mixing = full\n";
			ofs << "two_layer_gate_radial_mixing_layout = direct-k\n";
			ofs << "two_layer_gate_radial_mixing_coeff_count = "
			    << mixing_count << "\n";
			ofs << "two_layer_gate_radial_mixing_coeffs = {";
			for (int i = 0; i < mixing_count; ++i) {
				if (i != 0)
					ofs << ", ";
				// Internal storage is [l,o,k,b], although physical W_l is K x O.
				// Deterministic, nonzero rows preserve a meaningful initial outer
				// function. Output-channel variation is not random reinitialization.
				const int o = (i / (kmax * rb_size)) % outer_k_rank;
				const int k = (i / rb_size) % kmax;
				const int b = i % rb_size;
				const double channel_phase =
					static_cast<double>(o + 1) * (k * rb_size + b + 1);
				ofs << (pure_w_channelwise
					? 1.0e-2 * (1.0 + 0.25 * std::cos(channel_phase))
					  / std::sqrt(static_cast<double>(kmax))
					: 0.0);
			}
			ofs << "}\n";
		}
		if (radial_coefficient_gate) {
			const int adaptive_rows = CheckedGraphCount(
				{static_cast<size_t>(lmax + 1), static_cast<size_t>(outer_k_rank),
				 static_cast<size_t>(rb_size)}, "Adaptive radial row count");
			const int adaptive_weights = CheckedGraphCount(
				{static_cast<size_t>(adaptive_projection == "shared-n" ? adaptive_rows/rb_size*adaptive_shared_n_rank : adaptive_rows), gate_scalar_indices.size()},
				"Adaptive radial readout count");
			const int adaptive_amplitudes = CheckedGraphCount(
				{static_cast<size_t>(adaptive_rows), static_cast<size_t>(adaptive_shared_n_rank), static_cast<size_t>(adaptive_amplitude_mode == "species" ? species_count : 1)},
				"Adaptive radial amplitude count");
			ofs << "adaptive_radial_amplitude_count = "
			    << adaptive_amplitudes << "\n";
			ofs << "adaptive_radial_amplitudes = {";
			for (int i = 0; i < adaptive_amplitudes; ++i) {
				if (i != 0)
					ofs << ", ";
				ofs << 1.0e-2 / std::sqrt(static_cast<double>(adaptive_shared_n_rank));
			}
			ofs << "}\n";
			ofs << "adaptive_radial_weight_count = "
			    << adaptive_weights << "\n";
				ofs << "adaptive_radial_weights = {";
			const double amplitude =
				1.0 / std::sqrt(static_cast<double>(
					std::max<size_t>(1, gate_scalar_indices.size())));
			for (int i = 0; i < adaptive_weights; ++i) {
				if (i != 0)
					ofs << ", ";
				if (adaptive_shared_n_rank > 1) {
					const int q = i % static_cast<int>(gate_scalar_indices.size());
					const int t = (i / static_cast<int>(gate_scalar_indices.size())) % adaptive_shared_n_rank;
					// Distinct latent directions even when Q is a multiple of the
					// old 34-entry wave period; same across output channels.
					ofs << amplitude * (std::sin((q+1)*(t+1)*0.6180339887498949)
						+ 0.5*std::cos((q+1)*(t+2)*0.4142135623730951));
					continue;
				}
				const double sign = (i % 2 == 0) ? 1.0 : -1.0;
				const double ramp = 1.0
					+ static_cast<double>((i / 2) % 17) / 32.0;
				ofs << sign * amplitude * ramp;
			}
			ofs << "}\n";
			if (tensor_message_lmax > 0) {
				const int tensor_count = CheckedGraphCount(
					{static_cast<size_t>(tensor_message_lmax),
					 static_cast<size_t>(outer_k_rank),
					 static_cast<size_t>(two_layer_gate_k_rank)},
					"Tensor message coefficient count");
				ofs << "tensor_message_coeff_count = "
				    << tensor_count << "\n";
				ofs << "tensor_message_coeffs = {";
				for (int i = 0; i < tensor_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 0.0;
				}
				ofs << "}\n";
				ofs << "tensor_message_optimizer_scales = {";
				for (int i = 0; i < tensor_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 1.0;
				}
				ofs << "}\n";
			}
		}
		if (two_layer_gate_readout) {
				const int readout_count =
					static_cast<int>(gate_scalar_indices.size());
				ofs << "two_layer_gate_readout_enabled = true\n";
				ofs << "two_layer_gate_readout_coeff_count = "
				    << readout_count << "\n";
				ofs << "two_layer_gate_readout_coeffs = {";
				for (int q = 0; q < readout_count; ++q) {
					if (q != 0)
						ofs << ", ";
					ofs << 0.0;
				}
				ofs << "}\n";
			}
				if (multi_layer_gate) {
				std::vector<int> second_gate_body_counts;
				const std::vector<int> second_gate_scalar_indices =
					graph.ScalarIndices(
						second_layer_gate_body_order,
						second_layer_gate_factor_pruning,
						second_gate_body_counts);
				if (second_gate_scalar_indices.empty())
					ERROR("--multi-layer-gate selected no second-layer SH scalar basis functions.");
				for (int bo = 2; bo <= second_layer_gate_body_order; ++bo)
					if (second_gate_body_counts[bo] == 0)
						ERROR("--multi-layer-gate selected no second-layer scalar basis functions for one required mu body-order bucket.");

				ofs << "multi_layer_gate_enabled = true\n";
				ofs << "second_layer_gate_mode = " << second_layer_gate_mode << "\n";
				ofs << "second_layer_gate_body_order_max = "
				    << second_layer_gate_body_order << "\n";
				ofs << "second_layer_gate_radial_mode = shared-radial\n";
				const int second_gate_radial_count =
					radial_func_count * rb_size;
				ofs << "second_layer_gate_radial_coeff_count = "
				    << second_gate_radial_count << "\n";
				ofs << "second_layer_gate_radial_coeffs = {";
				for (int i = 0; i < second_gate_radial_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 1.0e-2;
				}
				ofs << "}\n";
				const int second_gate_additive_count =
					species_count * radial_func_count;
				ofs << "second_layer_gate_additive_coeff_count = "
				    << second_gate_additive_count << "\n";
				ofs << "second_layer_gate_additive_coeffs = {";
				for (int i = 0; i < second_gate_additive_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << (1.0 / 12.0);
				}
				ofs << "}\n";
				const int second_gate_type_count =
					species_count * radial_func_count;
				ofs << "second_layer_gate_type_coeff_count = "
				    << second_gate_type_count << "\n";
				ofs << "second_layer_gate_type_coeffs = {";
				for (int i = 0; i < second_gate_type_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 1.0;
				}
				ofs << "}\n";
				const int second_gate_weight_count =
					(second_layer_gate_mode == "mu-scalar-full")
						? radial_func_count
							* static_cast<int>(second_gate_scalar_indices.size())
						: static_cast<int>(second_gate_scalar_indices.size());
				ofs << "second_layer_gate_weight_count = "
				    << second_gate_weight_count << "\n";
				ofs << "second_layer_gate_scalar_indices = {";
				for (size_t i = 0; i < second_gate_scalar_indices.size(); ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << second_gate_scalar_indices[i];
				}
				ofs << "}\n";
				ofs << "second_layer_gate_weights = {";
				for (int i = 0; i < second_gate_weight_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 1.0e-1;
				}
				ofs << "}\n";
				if (second_layer_gate_mode == "mu-body-linear-combo") {
					const int second_gate_body_mix_weight_count =
						radial_func_count
						* (second_layer_gate_body_order - 1);
					ofs << "second_layer_gate_body_mix_weight_count = "
					    << second_gate_body_mix_weight_count << "\n";
					ofs << "second_layer_gate_body_mix_weights = {";
					for (int i = 0; i < second_gate_body_mix_weight_count; ++i) {
						if (i != 0)
							ofs << ", ";
						ofs << 1.0;
					}
					ofs << "}\n";
				}
			}
			if (two_layer_gate_edge_l1) {
				std::vector<int> edge_l1_source_moment_indices =
					two_layer_gate_k_rank_enabled
						? gate_rank_graph.L1SourceMomentBases()
						: graph.L1SourceMomentBases();
				if (two_layer_gate_k_rank_enabled)
					for (int& base : edge_l1_source_moment_indices)
						base += gate_rank_node_offset;
				const int edge_l1_source_count =
					static_cast<int>(edge_l1_source_moment_indices.size());
				if (edge_l1_source_count <= 0)
					ERROR("--two-layer-gate-edge-l1 selected no L=1 source moments.");
				const int edge_l1_weight_count =
					radial_func_count * edge_l1_source_count;
				ofs << "two_layer_gate_edge_l1_enabled = true\n";
				ofs << "two_layer_gate_edge_l1_source_moment_count = "
				    << edge_l1_source_count << "\n";
				ofs << "two_layer_gate_edge_l1_source_moment_indices = {";
				for (int i = 0; i < edge_l1_source_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << edge_l1_source_moment_indices[i];
				}
				ofs << "}\n";
				ofs << "two_layer_gate_edge_l1_weight_count = "
				    << edge_l1_weight_count << "\n";
				ofs << "two_layer_gate_edge_l1_weights = {";
				for (int i = 0; i < edge_l1_weight_count; ++i) {
					if (i != 0)
						ofs << ", ";
					ofs << 0.0;
				}
				ofs << "}\n";
			}
			if (two_layer_residual) {
			ofs << "two_layer_residual_e0_coeff_count = "
			    << graph.scalars_.size() << "\n";
			ofs << "two_layer_residual_e0_coeffs = {";
			for (size_t i = 0; i < graph.scalars_.size(); ++i) {
				if (i != 0)
					ofs << ", ";
				ofs << 0.0;
			}
			ofs << "}\n";
		}
	}
	}
