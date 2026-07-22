#include "sh_model_init.h"

#include "../src/common/stdafx.h"
#include "../src/common/utils.h"
#include "../src/zbl.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <fstream>
#include <iomanip>
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
	SHGraphBuilder(int lmax, int kmax, SHFactorPruning factor_pruning)
		: lmax_(lmax), kmax_(kmax), factor_pruning_(factor_pruning)
	{
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
					const double coeff = RealCGCoeff(left_copy.l, rm1, right_copy.l, rm2, L, rM);
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

	void Build()
	{
		for (int l = lmax_; l >= 0; --l) {
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
	const std::string rbasis = CanonicalRadialBasisType(StringOpt(opts, "radial-basis-type", "RBChebyshev_sss"));
	const SHFactorPruning factor_pruning =
		ParseFactorPruning(opts, "sh-factor-pruning", SHFactorPruning::QTotal);
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
		StrictNonnegativeIntOpt(opts, "two-layer-gate-k-rank", 0);
	const bool two_layer_gate_k_rank_enabled = two_layer_gate_k_rank > 0;
	const bool multi_layer_gate = HasOpt(opts, "multi-layer-gate");
	const bool two_layer_gate_readout = HasOpt(opts, "two-layer-gate-readout");
	const bool two_layer_gate = HasOpt(opts, "two-layer-gate") || multi_layer_gate
		|| two_layer_gate_readout;
	const SHFactorPruning two_layer_gate_factor_pruning =
		ParseFactorPruning(
			opts, "two-layer-gate-factor-pruning", factor_pruning);
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
	const int default_two_layer_gate_body_order = body_order;
	const int two_layer_gate_body_order =
		IntOpt(opts, "two-layer-gate-body-order", default_two_layer_gate_body_order);
	std::string two_layer_gate_mode =
		StringOpt(opts, "two-layer-gate-mode", "mu-body-linear-combo");
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
		HasOpt(opts, "write-sh-scalar-info") || two_layer_gate || multi_layer_gate;
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
	if (outer_k_rank < 0 || outer_k_rank > kmax)
		ERROR("--outer-k-rank must be in [1, --k-max] when enabled.");
	if (two_layer_gate_k_rank < 0 || two_layer_gate_k_rank > kmax)
		ERROR("--two-layer-gate-k-rank must be in [1, --k-max] when enabled.");
	if (outer_k_rank_enabled && !HasOpt(opts, "two-layer-gate"))
		ERROR("--outer-k-rank requires an explicit one-layer --two-layer-gate model.");
	if (two_layer_gate_k_rank_enabled && !HasOpt(opts, "two-layer-gate"))
		ERROR("--two-layer-gate-k-rank requires an explicit --two-layer-gate model.");
	if (two_layer_gate_k_rank_enabled && !outer_k_rank_enabled)
		ERROR("--two-layer-gate-k-rank currently requires --outer-k-rank=<R>.");
	if (outer_k_rank_enabled && factor_pruning != SHFactorPruning::QTotal)
		ERROR("--outer-k-rank currently requires --sh-factor-pruning=q-total so contracted channels retain their cross products.");
	if (two_layer_gate_k_rank_enabled
	    && two_layer_gate_factor_pruning != SHFactorPruning::QTotal)
		ERROR("--two-layer-gate-k-rank currently requires --two-layer-gate-factor-pruning=q-total so contracted channels retain their cross products.");
	if (HasOpt(opts, "two-layer-gate-body-order") && !two_layer_gate)
		ERROR("--two-layer-gate-body-order requires --two-layer-gate.");
	if (HasOpt(opts, "two-layer-gate-factor-pruning") && !two_layer_gate)
		ERROR("--two-layer-gate-factor-pruning requires --two-layer-gate.");
	if (two_layer_gate
	    && !IsFactorPruningSubset(two_layer_gate_factor_pruning, factor_pruning))
		ERROR("--two-layer-gate-factor-pruning must select a subgraph of --sh-factor-pruning.");
	if (two_layer_gate
	    && (two_layer_gate_body_order < 2 || two_layer_gate_body_order > body_order))
		ERROR("--two-layer-gate-body-order must be in [2, --body-order].");
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
	if (two_layer_gate_mode != "mu-body-linear-combo"
	    && two_layer_gate_mode != "mu-scalar-full")
		ERROR("--two-layer-gate-mode should be mu-body-linear-combo or mu-scalar-full.");
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
	if (outer_k_rank_enabled && two_layer_gate_readout)
		ERROR("--outer-k-rank is not compatible with --two-layer-gate-readout.");
	if (outer_k_rank_enabled && two_layer_residual)
		ERROR("--outer-k-rank is not compatible with --two-layer-residual.");
	if (outer_k_rank_enabled && multi_layer_gate)
		ERROR("--outer-k-rank currently supports one gate only.");
	if (outer_k_rank_enabled && two_layer_gate_edge_l1)
		ERROR("--outer-k-rank currently does not include edge-L1.");
	if (two_layer_gate_k_rank_enabled
	    && (two_layer_gate_readout || two_layer_residual
	        || multi_layer_gate || two_layer_gate_edge_l1))
		ERROR("--two-layer-gate-k-rank is not compatible with readout, residual, multi-layer gate, or edge-L1.");
	if (!IsSupportedSHRadialBasis(rbasis))
		ERROR("init-sh currently writes RBChebyshev_sss, RBChebyshev_sss_rational, RBLaguerre_log1p, or RBJacobi_sss models.");
	if (rbasis == "RBJacobi_sss" && kmax > 6)
		ERROR("init-sh with RBJacobi_sss supports --k-max up to 6 because Jacobi blocks are indexed as k=0..5.");

	SHGraphBuilder graph(lmax, kmax, factor_pruning);
	graph.Build();
	if (outer_k_rank_enabled || two_layer_gate_k_rank_enabled)
		graph.AddAllBasics();
	if (!two_layer_gate_k_rank_enabled)
		graph.AddScalars(body_order, body_lmax);
	SHGraphBuilder gate_rank_graph(
		lmax, std::max(1, two_layer_gate_k_rank),
		two_layer_gate_factor_pruning);
	if (two_layer_gate_k_rank_enabled) {
		gate_rank_graph.Build();
		gate_rank_graph.AddAllBasics();
		gate_rank_graph.AddScalars(two_layer_gate_body_order, body_lmax);
		if (gate_rank_graph.basic_.size()
		    != static_cast<size_t>(
			   two_layer_gate_k_rank * (lmax + 1) * (lmax + 1)))
			ERROR("--two-layer-gate-k-rank generated an unexpected primitive layout.");
		if (gate_rank_graph.scalars_.empty())
			ERROR("--two-layer-gate-k-rank generated no gate scalar basis functions.");
	}
	SHGraphBuilder outer_rank_graph(
		lmax, std::max(1, outer_k_rank), factor_pruning);
	if (outer_k_rank_enabled) {
		outer_rank_graph.Build();
		outer_rank_graph.AddAllBasics();
		outer_rank_graph.AddScalars(body_order, body_lmax);
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
	ofs << "version = "
	    << (uses_extended_sh_format ? "1.2.0" : "1.1.0") << "\n";
	ofs << "potential_name = " << name << "\n";
	ofs << "scaling = " << scaling << "\n";
	ofs << "L = " << lmax << "\n";
	ofs << "scaling_map = LK\n";
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
	const int radial_func_count = kmax * (lmax + 1);
	ofs << "\tradial_funcs_count = " << radial_func_count << "\n";

	const SHGraphBuilder& energy_graph =
		outer_k_rank_enabled ? outer_rank_graph : graph;
	const int gate_rank_node_offset = graph.node_count_;
	const int outer_rank_node_offset = graph.node_count_
		+ (two_layer_gate_k_rank_enabled ? gate_rank_graph.node_count_ : 0);
	const int alpha_moments_count = graph.node_count_
		+ (two_layer_gate_k_rank_enabled ? gate_rank_graph.node_count_ : 0)
		+ (outer_k_rank_enabled ? outer_rank_graph.node_count_ : 0);
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
			outer_k_rank * kmax * (lmax + 1);
		ofs << "outer_k_rank1_enabled = true\n";
		// Preserve bidirectional compatibility with the original rank-1
		// prototype, whose reader does not know the generic rank field.
		if (outer_k_rank != 1)
			ofs << "outer_k_rank = " << outer_k_rank << "\n";
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
		for (int l = 0; l <= lmax; ++l) {
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
			gate_body_counts.assign(two_layer_gate_body_order + 1, 0);
			gate_scalar_indices.resize(gate_rank_graph.scalars_.size());
			for (int q = 0; q < static_cast<int>(gate_scalar_indices.size()); ++q) {
				gate_scalar_indices[q] = q;
				++gate_body_counts[gate_rank_graph.scalar_infos_[q].body_order];
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
					2 * species_count * species_count * radial_func_count;
				ofs << "two_layer_gate_scaling_mode = decoupled\n";
				ofs << "two_layer_gate_scaling_coeff_count = "
				    << gate_scaling_count << "\n";
				ofs << "two_layer_gate_scaling_coeffs = {";
				WriteDefaultScalingCoeffList(
					ofs, species_count, radial_func_count, rbasis);
				ofs << "}\n";
			}
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
				(two_layer_gate_mode == "mu-scalar-full")
					? radial_func_count * static_cast<int>(gate_scalar_indices.size())
				: static_cast<int>(gate_scalar_indices.size());
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
			if (two_layer_gate_mode == "mu-body-linear-combo") {
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
				const std::vector<int> edge_l1_source_moment_indices =
					graph.L1SourceMomentBases();
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
