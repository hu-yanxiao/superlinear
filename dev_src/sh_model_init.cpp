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

SHFactorPruning ParseFactorPruning(const std::map<std::string, std::string>& opts)
{
	const std::string value = StringOpt(opts, "sh-factor-pruning", "legacy");
	if (value == "legacy")
		return SHFactorPruning::Legacy;
	if (value == "q-total" || value == "total")
		return SHFactorPruning::QTotal;
	ERROR("--sh-factor-pruning should be legacy or q-total.");
	return SHFactorPruning::Legacy;
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
			std::function<void(int, int)> rec = [&](int pos, int first_q_index) {
				if (pos == factor_count) {
					callback(tuple);
					return;
				}
				for (int q_index = first_q_index; q_index < static_cast<int>(q_.size()); ++q_index) {
					if (!Allowed(q_index, body_order))
						continue;
					tuple.push_back(q_index);
					rec(pos + 1, q_index);
					tuple.pop_back();
				}
			};
			rec(0, 0);
		}
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
	const SHFactorPruning factor_pruning = ParseFactorPruning(opts);
	const bool two_layer_gate = HasOpt(opts, "two-layer-gate");
	const bool two_layer_gate_edge_l1 = HasOpt(opts, "two-layer-gate-edge-l1");
	const bool requested_two_layer_gate_shared_radial =
		HasOpt(opts, "two-layer-gate-shared-radial");
	const bool two_layer_gate_shared_radial = two_layer_gate;
	const bool two_layer_residual = HasOpt(opts, "two-layer-residual");
	const int two_layer_gate_body_order = kmax + 1;
	const double two_layer_gate_tanh_amplitude =
		DoubleOpt(opts, "two-layer-gate-tanh-amplitude", 0.8);
	std::string two_layer_gate_mode =
		StringOpt(opts, "two-layer-gate-mode", "mu-body-linear-combo");
	if (two_layer_gate_mode == "mu-body-order")
		two_layer_gate_mode = "mu-scalar-full";
	const std::string two_layer_gate_site_mode =
		StringOpt(opts, "two-layer-gate-site-mode", "neighbor");
	const bool write_scalar_info = HasOpt(opts, "write-sh-scalar-info") || two_layer_gate;
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
	if (two_layer_gate && HasOpt(opts, "two-layer-gate-body-order"))
		ERROR("--two-layer-gate-body-order is not used by mu-body-order gate models; use --body-order >= --k-max + 1.");
	if (two_layer_gate && body_order < two_layer_gate_body_order)
		ERROR("--two-layer-gate requires --body-order >= --k-max + 1 for mu-body-order scalar buckets.");
	if (two_layer_gate_edge_l1 && !two_layer_gate)
		ERROR("--two-layer-gate-edge-l1 requires --two-layer-gate.");
	if (two_layer_gate_edge_l1 && lmax < 1)
		ERROR("--two-layer-gate-edge-l1 requires --l-max >= 1.");
	if (two_layer_gate_mode != "mu-body-linear-combo"
	    && two_layer_gate_mode != "mu-scalar-full")
		ERROR("--two-layer-gate-mode should be mu-body-linear-combo or mu-scalar-full.");
	if (HasOpt(opts, "two-layer-gate-mode") && !two_layer_gate)
		ERROR("--two-layer-gate-mode requires --two-layer-gate.");
	if (!std::isfinite(two_layer_gate_tanh_amplitude)
	    || two_layer_gate_tanh_amplitude < 0.0
	    || two_layer_gate_tanh_amplitude > 1.0)
		ERROR("--two-layer-gate-tanh-amplitude should be finite and in [0, 1].");
	if (two_layer_gate_site_mode != "neighbor"
	    && two_layer_gate_site_mode != "double")
		ERROR("--two-layer-gate-site-mode should be 'neighbor' or 'double'.");
	if (HasOpt(opts, "two-layer-gate-site-mode") && !two_layer_gate)
		ERROR("--two-layer-gate-site-mode requires --two-layer-gate.");
	if (two_layer_gate_edge_l1 && two_layer_gate_site_mode != "neighbor")
		ERROR("--two-layer-gate-edge-l1 first version supports neighbor site mode only.");
	if (requested_two_layer_gate_shared_radial && !two_layer_gate)
		ERROR("--two-layer-gate-shared-radial requires --two-layer-gate.");
	if (two_layer_residual && !two_layer_gate)
		ERROR("--two-layer-residual requires --two-layer-gate.");
	if (two_layer_residual && two_layer_gate)
		ERROR("--two-layer-residual is not supported by mu-body-order gate models.");
	if (!IsSupportedSHRadialBasis(rbasis))
		ERROR("init-sh currently writes RBChebyshev_sss, RBChebyshev_sss_rational, RBLaguerre_log1p, or RBJacobi_sss models.");
	if (rbasis == "RBJacobi_sss" && kmax > 6)
		ERROR("init-sh with RBJacobi_sss supports --k-max up to 6 because Jacobi blocks are indexed as k=0..5.");

	SHGraphBuilder graph(lmax, kmax, factor_pruning);
	graph.Build();
	graph.AddScalars(body_order, body_lmax);

	std::ofstream ofs(filename.c_str());
	if (!ofs.is_open())
		ERROR("Cannot open output model for writing: " + filename);
	ofs.setf(std::ios::scientific);
	ofs.precision(15);

	ofs << "MTP\n";
	ofs << "version = 1.1.0\n";
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
	ofs << "\tradial_funcs_count = " << kmax * (lmax + 1) << "\n";

	ofs << "alpha_moments_count = " << graph.node_count_ << "\n";
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
	ofs << "sh_product_count = " << graph.products_.size() << "\n";
	ofs << "sh_products = {";
	for (size_t i = 0; i < graph.products_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		const Product& p = graph.products_[i];
		ofs << "{" << p.left << ", " << p.right << ", " << p.target << ", " << p.coeff << "}";
	}
	ofs << "}\n";
	ofs << "alpha_scalar_moments = " << graph.scalars_.size() << "\n";
	ofs << "alpha_moment_mapping = {";
	for (size_t i = 0; i < graph.scalars_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		ofs << graph.scalars_[i];
	}
	ofs << "}\n";
	if (write_scalar_info) {
		ofs << "sh_scalar_info_count = " << graph.scalar_infos_.size() << "\n";
		ofs << "sh_scalar_info = {";
		for (size_t i = 0; i < graph.scalar_infos_.size(); ++i) {
			if (i != 0)
				ofs << ", ";
			const ScalarSpec& spec = graph.scalar_infos_[i];
			ofs << "{" << spec.body_order << ", "
			    << spec.q0 << ", " << spec.q1 << ", " << spec.q2 << ", "
			    << spec.q3 << ", " << spec.q4 << ", " << spec.intermediate_l << "}";
		}
		ofs << "}\n";
	}
	if (two_layer_gate) {
		std::vector<int> gate_scalar_indices;
		std::vector<int> gate_body_counts(two_layer_gate_body_order + 1, 0);
		for (size_t i = 0; i < graph.scalar_infos_.size(); ++i)
			if (graph.scalar_infos_[i].body_order >= 2
			    && graph.scalar_infos_[i].body_order <= two_layer_gate_body_order) {
				gate_scalar_indices.push_back(static_cast<int>(i));
				++gate_body_counts[graph.scalar_infos_[i].body_order];
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
		ofs << "two_layer_gate_tanh_amplitude = "
		    << two_layer_gate_tanh_amplitude << "\n";
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
		const int radial_func_count = kmax * (lmax + 1);
		const int gate_additive_count = species_count;
		ofs << "two_layer_gate_additive_coeff_count = "
		    << gate_additive_count << "\n";
		ofs << "two_layer_gate_additive_coeffs = {";
		for (int i = 0; i < gate_additive_count; ++i) {
			if (i != 0)
				ofs << ", ";
			ofs << (1.0 / 12.0);
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
