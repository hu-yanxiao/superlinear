/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <sstream>
#include "mtpr.h"

#ifdef MLIP_INTEL_MKL
#	include <mkl_cblas.h>
#else
#	include <cblas.h>
#endif

#ifdef MLIP_MPI
#	include "mpi.h"
#endif

using namespace std;

namespace {

constexpr double kRandomScalMin = 2.0;
constexpr double kRandomScalMax = 4.0;
constexpr double kLaguerreRandomScalMax = 3.0;
constexpr double kRandomShiftMin = 1.5;
constexpr double kRandomShiftMax = 2.5;
constexpr double kRadialFirstCoeffPositiveFloor = 1.0e-12;

double Clamp01(double value)
{
	return std::max(0.0, std::min(1.0, value));
}

struct GateUpgradeQIndex {
	int l;
	int k;
	int mu;
	std::string key;
};

struct GateUpgradeTensor {
	std::string key;
	int l = 0;
	std::vector<int> node;
	bool zero = false;
};

double GateUpgradeFact(int n)
{
	if (n < 0)
		return 0.0;
	double value = 1.0;
	for (int i = 2; i <= n; ++i)
		value *= static_cast<double>(i);
	return value;
}

bool GateUpgradeTriangle(int l1, int l2, int l3)
{
	return std::abs(l1 - l2) <= l3 && l3 <= l1 + l2;
}

double GateUpgradeClebschGordan(int j1, int m1, int j2, int m2, int j, int m)
{
	if (m != m1 + m2)
		return 0.0;
	if (!GateUpgradeTriangle(j1, j2, j))
		return 0.0;
	if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m) > j)
		return 0.0;

	const double pref1 = std::sqrt((2.0 * j + 1.0)
		* GateUpgradeFact(j1 + j2 - j)
		* GateUpgradeFact(j1 - j2 + j)
		* GateUpgradeFact(-j1 + j2 + j)
		/ GateUpgradeFact(j1 + j2 + j + 1));
	const double pref2 = std::sqrt(GateUpgradeFact(j1 + m1)
		* GateUpgradeFact(j1 - m1)
		* GateUpgradeFact(j2 + m2)
		* GateUpgradeFact(j2 - m2)
		* GateUpgradeFact(j + m)
		* GateUpgradeFact(j - m));

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
			/ (GateUpgradeFact(k) * GateUpgradeFact(a)
			   * GateUpgradeFact(b) * GateUpgradeFact(c)
			   * GateUpgradeFact(d) * GateUpgradeFact(e));
		sum += term;
	}
	return pref1 * pref2 * sum;
}

double GateUpgradeParitySign(int m)
{
	return (std::abs(m) % 2) == 0 ? 1.0 : -1.0;
}

std::complex<double> GateUpgradeRealFromComplexCoeff(int real_m, int complex_m)
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
			return inv_sqrt2 * GateUpgradeParitySign(a);
		return inv_sqrt2;
	}
	if (complex_m == a)
		return std::complex<double>(0.0, -inv_sqrt2 * GateUpgradeParitySign(a));
	return std::complex<double>(0.0, inv_sqrt2);
}

std::complex<double> GateUpgradeRealCouplingPhase(int l1, int l2, int L)
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

double GateUpgradeRealCGCoeff(int l1, int rm1, int l2, int rm2, int L, int rM)
{
	std::complex<double> sum(0.0, 0.0);
	for (int M = -L; M <= L; ++M) {
		const std::complex<double> out_u =
			std::conj(GateUpgradeRealFromComplexCoeff(rM, M));
		if (std::abs(out_u) == 0.0)
			continue;
		for (int m1 = -l1; m1 <= l1; ++m1) {
			const std::complex<double> in1 =
				GateUpgradeRealFromComplexCoeff(rm1, m1);
			if (std::abs(in1) == 0.0)
				continue;
			for (int m2 = -l2; m2 <= l2; ++m2) {
				const std::complex<double> in2 =
					GateUpgradeRealFromComplexCoeff(rm2, m2);
				if (std::abs(in2) == 0.0)
					continue;
				const double cg =
					GateUpgradeClebschGordan(l1, m1, l2, m2, L, M);
				if (std::abs(cg) < 1.0e-14)
					continue;
				sum += out_u * cg * in1 * in2;
			}
		}
	}
	sum *= GateUpgradeRealCouplingPhase(l1, l2, L);
	if (std::abs(sum.imag()) > 1.0e-10)
		ERROR("SUS2-SH edge-L1 upgrade CG transform produced a non-real coefficient.");
	return sum.real();
}

std::string GateUpgradeTensorKey(const std::string& left,
                                 const std::string& right,
                                 int L)
{
	std::ostringstream oss;
	oss << "(" << left << "x" << right << ")->" << L;
	return oss.str();
}

class GateUpgradeGraphBuilder {
public:
	GateUpgradeGraphBuilder(int lmax, int kmax)
		: lmax_(lmax), kmax_(kmax)
	{
		for (int l = lmax_; l >= 0; --l) {
			for (int k = kmax_ - 1; k >= 0; --k) {
				GateUpgradeQIndex q;
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

	void AddScalarsFromMetadata(const std::vector<SHScalarInfo>& scalar_infos)
	{
		std::vector<char> required_q(q_.size(), 0);
		for (const SHScalarInfo& spec : scalar_infos) {
			const int factor_count = spec.body_order - 1;
			if (factor_count < 1 || factor_count > 5)
				ERROR("SUS2-SH edge-L1 upgrade found invalid sh_scalar_info body order");
			for (int factor = 0; factor < factor_count; ++factor) {
				const int q = spec.q[factor];
				if (q < 0 || q >= static_cast<int>(q_.size()))
					ERROR("SUS2-SH edge-L1 upgrade found invalid q index");
				required_q[q] = 1;
			}
		}
		for (int q = 0; q < static_cast<int>(required_q.size()); ++q)
			if (required_q[q])
				BasicTensor(q);
		for (const SHScalarInfo& spec : scalar_infos)
			BuildScalar(spec);
	}

	std::vector<int> L1SourceMomentBases() const
	{
		std::vector<int> bases;
		for (const GateUpgradeTensor& tensor : tensors_) {
			if (!tensor.zero && tensor.l == 1
			    && static_cast<int>(tensor.node.size()) == 3)
				bases.push_back(tensor.node[0]);
		}
		std::sort(bases.begin(), bases.end());
		bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
		return bases;
	}

	int NodeCount() const { return node_count_; }

private:
	GateUpgradeTensor BasicTensor(int q_index)
	{
		if (q_index < 0 || q_index >= static_cast<int>(q_.size()))
			ERROR("SUS2-SH edge-L1 upgrade q index is out of range");
		const GateUpgradeQIndex& q = q_[q_index];
		std::map<std::string, int>::const_iterator found =
			tensor_lookup_.find(q.key);
		if (found != tensor_lookup_.end())
			return tensors_[found->second];

		GateUpgradeTensor tensor;
		tensor.key = q.key;
		tensor.l = q.l;
		tensor.zero = false;
		tensor.node.resize(2 * q.l + 1);
		for (int m = -q.l; m <= q.l; ++m)
			tensor.node[m + q.l] = AddNode();
		const int index = static_cast<int>(tensors_.size());
		tensors_.push_back(tensor);
		tensor_lookup_[q.key] = index;
		return tensors_.back();
	}

	GateUpgradeTensor Couple(const GateUpgradeTensor& left,
	                         const GateUpgradeTensor& right,
	                         int L)
	{
		const GateUpgradeTensor left_copy = left;
		const GateUpgradeTensor right_copy = right;
		const std::string key = GateUpgradeTensorKey(left_copy.key,
		                                             right_copy.key,
		                                             L);
		std::map<std::string, int>::const_iterator found =
			tensor_lookup_.find(key);
		if (found != tensor_lookup_.end())
			return tensors_[found->second];

		GateUpgradeTensor out;
		out.key = key;
		out.l = L;
		if (left_copy.zero || right_copy.zero) {
			out.zero = true;
			const int index = static_cast<int>(tensors_.size());
			tensors_.push_back(out);
			tensor_lookup_[key] = index;
			return tensors_.back();
		}

		std::map<std::string, double> local_coeffs;
		for (int rm1 = -left_copy.l; rm1 <= left_copy.l; ++rm1) {
			for (int rm2 = -right_copy.l; rm2 <= right_copy.l; ++rm2) {
				for (int rM = -L; rM <= L; ++rM) {
					const double coeff = GateUpgradeRealCGCoeff(
						left_copy.l, rm1, right_copy.l, rm2, L, rM);
					if (std::abs(coeff) < 1.0e-12)
						continue;
					int left_node = left_copy.node[rm1 + left_copy.l];
					int right_node = right_copy.node[rm2 + right_copy.l];
					if (right_node < left_node)
						std::swap(left_node, right_node);
					std::ostringstream local_key;
					local_key << left_node << ',' << right_node << ',' << (rM + L);
					local_coeffs[local_key.str()] += coeff;
				}
			}
		}
		bool nonzero = false;
		for (std::map<std::string, double>::const_iterator it = local_coeffs.begin();
		     it != local_coeffs.end(); ++it) {
			if (std::abs(it->second) >= 1.0e-12) {
				nonzero = true;
				break;
			}
		}
		if (!nonzero) {
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
		return tensors_.back();
	}

	int BuildScalar(const SHScalarInfo& spec)
	{
		const GateUpgradeTensor t0 = BasicTensor(spec.q[0]);
		if (spec.body_order == 2)
			return t0.node[0];
		const GateUpgradeTensor t1 = BasicTensor(spec.q[1]);
		if (spec.body_order == 3) {
			const GateUpgradeTensor scalar = Couple(t0, t1, 0);
			return scalar.zero ? -1 : scalar.node[0];
		}
		const GateUpgradeTensor t2 = BasicTensor(spec.q[2]);
		if (spec.body_order == 4) {
			const GateUpgradeTensor pair = Couple(t0, t1, spec.intermediate_l);
			if (pair.zero)
				return -1;
			const GateUpgradeTensor scalar = Couple(pair, t2, 0);
			return scalar.zero ? -1 : scalar.node[0];
		}
		const GateUpgradeTensor t3 = BasicTensor(spec.q[3]);
		const GateUpgradeTensor left = Couple(t0, t1, spec.intermediate_l);
		const GateUpgradeTensor right_pair = Couple(t2, t3, spec.intermediate_l);
		if (spec.body_order == 6) {
			const GateUpgradeTensor t4 = BasicTensor(spec.q[4]);
			if (left.zero || right_pair.zero)
				return -1;
			const GateUpgradeTensor right =
				Couple(right_pair, t4, spec.intermediate_l);
			if (right.zero)
				return -1;
			const GateUpgradeTensor scalar = Couple(left, right, 0);
			return scalar.zero ? -1 : scalar.node[0];
		}
		if (left.zero || right_pair.zero)
			return -1;
		const GateUpgradeTensor scalar = Couple(left, right_pair, 0);
		return scalar.zero ? -1 : scalar.node[0];
	}

	int AddNode()
	{
		return node_count_++;
	}

	int lmax_;
	int kmax_;
	int node_count_ = 0;
	std::vector<GateUpgradeQIndex> q_;
	std::vector<GateUpgradeTensor> tensors_;
	std::map<std::string, int> tensor_lookup_;
};

std::vector<int> BuildFullEdgeL1SourceMomentBasesFromScalarInfo(
	int lmax,
	int kmax,
	const std::vector<SHScalarInfo>& scalar_infos)
{
	GateUpgradeGraphBuilder builder(lmax, kmax);
	builder.AddScalarsFromMetadata(scalar_infos);
	return builder.L1SourceMomentBases();
}

bool TwoLayerProfileEnabledOnRank0()
{
	const char* env = std::getenv("SUS2_SH_TWO_LAYER_PROFILE");
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

bool EdgeL1CacheProfileEnabledOnRank0()
{
	const char* env = std::getenv("SUS2_SH_EDGE_L1_CACHE_PROFILE");
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

int TwoLayerProfileInterval()
{
	const char* env = std::getenv("SUS2_SH_TWO_LAYER_PROFILE_INTERVAL");
	if (env == nullptr)
		return 20;
	const int value = std::atoi(env);
	return value > 0 ? value : 20;
}

double TwoLayerProfileNow()
{
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct TwoLayerGradProfileState {
	long long calls = 0;
	long long atoms = 0;
	double total_s = 0.0;
	double prepare_s = 0.0;
	double main_grad_s = 0.0;
	double energy_adjoint_s = 0.0;
	double directional_s = 0.0;
	double tangent_grad_s = 0.0;
	double gate_weight_s = 0.0;
	double scalar_param_s = 0.0;
};

struct TwoLayerForwardProfileState {
	long long calls = 0;
	long long atoms = 0;
	double total_s = 0.0;
	double prepare_s = 0.0;
	double main_forward_s = 0.0;
	double force_chain_s = 0.0;
};

struct TwoLayerLinearProfileState {
	long long calls = 0;
	long long atoms = 0;
	double total_s = 0.0;
	double prepare_s = 0.0;
	double main_components_s = 0.0;
	double gate_adjoints_s = 0.0;
	double chain_apply_s = 0.0;
	double basis_calc_s = 0.0;
	double component_fill_s = 0.0;
};

TwoLayerGradProfileState& TwoLayerGradProfile()
{
	static TwoLayerGradProfileState state;
	return state;
}

TwoLayerForwardProfileState& TwoLayerForwardProfile()
{
	static TwoLayerForwardProfileState state;
	return state;
}

TwoLayerLinearProfileState& TwoLayerLinearProfile()
{
	static TwoLayerLinearProfileState state;
	return state;
}

void RecordTwoLayerGradProfile(int atom_count,
                               double total_s,
                               double prepare_s,
                               double main_grad_s,
                               double energy_adjoint_s,
                               double directional_s,
                               double tangent_grad_s,
                               double gate_weight_s,
                               double scalar_param_s)
{
	TwoLayerGradProfileState& state = TwoLayerGradProfile();
	state.calls += 1;
	state.atoms += atom_count;
	state.total_s += total_s;
	state.prepare_s += prepare_s;
	state.main_grad_s += main_grad_s;
	state.energy_adjoint_s += energy_adjoint_s;
	state.directional_s += directional_s;
	state.tangent_grad_s += tangent_grad_s;
	state.gate_weight_s += gate_weight_s;
	state.scalar_param_s += scalar_param_s;
	const int interval = TwoLayerProfileInterval();
	if (state.calls % interval != 0)
		return;
	const double inv_calls = 1.0 / static_cast<double>(state.calls);
	std::cout << "[SUS2_SH_TWO_LAYER_GRAD_PROFILE] calls=" << state.calls
	          << " atoms=" << state.atoms
	          << " avg_us_total=" << 1.0e6 * state.total_s * inv_calls
	          << " avg_us_prepare_gate=" << 1.0e6 * state.prepare_s * inv_calls
	          << " avg_us_main_gated_grad=" << 1.0e6 * state.main_grad_s * inv_calls
	          << " avg_us_energy_adjoint=" << 1.0e6 * state.energy_adjoint_s * inv_calls
	          << " avg_us_directional=" << 1.0e6 * state.directional_s * inv_calls
	          << " avg_us_tangent_grad=" << 1.0e6 * state.tangent_grad_s * inv_calls
	          << " avg_us_gate_weight=" << 1.0e6 * state.gate_weight_s * inv_calls
	          << " avg_us_scalar_param=" << 1.0e6 * state.scalar_param_s * inv_calls
	          << std::endl;
}

void RecordTwoLayerForwardProfile(int atom_count,
                                  double total_s,
                                  double prepare_s,
                                  double main_forward_s,
                                  double force_chain_s)
{
	TwoLayerForwardProfileState& state = TwoLayerForwardProfile();
	state.calls += 1;
	state.atoms += atom_count;
	state.total_s += total_s;
	state.prepare_s += prepare_s;
	state.main_forward_s += main_forward_s;
	state.force_chain_s += force_chain_s;
	const int interval = TwoLayerProfileInterval();
	if (state.calls % interval != 0)
		return;
	const double inv_calls = 1.0 / static_cast<double>(state.calls);
	std::cout << "[SUS2_SH_TWO_LAYER_FORWARD_PROFILE] calls=" << state.calls
	          << " atoms=" << state.atoms
	          << " avg_us_total=" << 1.0e6 * state.total_s * inv_calls
	          << " avg_us_prepare_gate=" << 1.0e6 * state.prepare_s * inv_calls
	          << " avg_us_main_forward=" << 1.0e6 * state.main_forward_s * inv_calls
	          << " avg_us_force_chain=" << 1.0e6 * state.force_chain_s * inv_calls
	          << std::endl;
}

void RecordTwoLayerLinearProfile(int atom_count,
                                 double total_s,
                                 double prepare_s,
                                 double main_components_s,
                                 double gate_adjoints_s,
                                 double chain_apply_s,
                                 double basis_calc_s,
                                 double component_fill_s)
{
	TwoLayerLinearProfileState& state = TwoLayerLinearProfile();
	state.calls += 1;
	state.atoms += atom_count;
	state.total_s += total_s;
	state.prepare_s += prepare_s;
	state.main_components_s += main_components_s;
	state.gate_adjoints_s += gate_adjoints_s;
	state.chain_apply_s += chain_apply_s;
	state.basis_calc_s += basis_calc_s;
	state.component_fill_s += component_fill_s;
	const int interval = TwoLayerProfileInterval();
	if (state.calls % interval != 0)
		return;
	const double inv_calls = 1.0 / static_cast<double>(state.calls);
	std::cout << "[SUS2_SH_TWO_LAYER_LINEAR_PROFILE] calls=" << state.calls
	          << " atoms=" << state.atoms
	          << " avg_us_total=" << 1.0e6 * state.total_s * inv_calls
	          << " avg_us_prepare_gate=" << 1.0e6 * state.prepare_s * inv_calls
	          << " avg_us_main_components=" << 1.0e6 * state.main_components_s * inv_calls
	          << " avg_us_gate_adjoints=" << 1.0e6 * state.gate_adjoints_s * inv_calls
	          << " avg_us_chain_apply=" << 1.0e6 * state.chain_apply_s * inv_calls
	          << " avg_us_basis_calc=" << 1.0e6 * state.basis_calc_s * inv_calls
	          << " avg_us_component_fill=" << 1.0e6 * state.component_fill_s * inv_calls
	          << std::endl;
}

double StableSoftplus(double x)
{
	if (x > 40.0)
		return x;
	if (x < -40.0)
		return std::exp(x);
	return std::log1p(std::exp(x));
}

double StableInvSoftplus(double y)
{
	if (y > 40.0)
		return y;
	return std::log(std::expm1(y));
}

double Lerp(double start, double end, double t)
{
	return start + (end - start) * t;
}

std::string TrimCopy(const std::string& value)
{
	std::size_t begin = 0;
	std::size_t end = value.size();
	while (begin < end && std::isspace(static_cast<unsigned char>(value[begin])))
		++begin;
	while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
		--end;
	return value.substr(begin, end - begin);
}

std::string ReadAssignmentTail(const std::string& line)
{
	const std::size_t pos = line.find('=');
	if (pos == std::string::npos)
		return TrimCopy(line);
	return TrimCopy(line.substr(pos + 1));
}

void ReadIntList(std::ifstream& ifs, std::vector<int>& values, int count)
{
	values.assign(count, 0);
	char comma = ' ';
	ifs.ignore(1000, '{');
	for (int i = 0; i < count; ++i)
		ifs >> values[i] >> comma;
	if (ifs.fail())
		ERROR("Error reading integer list from .mtp file");
}

void ReadDoubleList(std::ifstream& ifs, std::vector<double>& values, int count)
{
	values.assign(count, 0.0);
	char comma = ' ';
	ifs.ignore(1000, '{');
	for (int i = 0; i < count; ++i)
		ifs >> values[i] >> comma;
	if (ifs.fail())
		ERROR("Error reading floating-point list from .mtp file");
}

bool ReadBoolToken(const std::string& value)
{
	return value == "true" || value == "1" || value == "yes";
}

bool IsLaguerreLog1pBasisType(const std::string& basis_type)
{
	return basis_type == "RBLaguerre_log1p"
	    || basis_type == "RBLaguerre_log1p_lmp"
	    || basis_type == "RBLaguerre_log1p_pos"
	    || basis_type == "RBLaguerre_log1p_pos_lmp"
	    || basis_type == "RBLaguerre_log1p_noenv"
	    || basis_type == "RBLaguerre_log1p_noenv_lmp";
}

bool IsLaguerreLog1pBasisType(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && IsLaguerreLog1pBasisType(radial_basis->GetRBTypeString());
}

bool IsJacobiIndexedBasisType(const std::string& basis_type)
{
	return basis_type == "RBJacobi_sss"
	    || basis_type == "RBJacobi_sss_lmp"
	    || basis_type == "RBJacobi_sss_noweight"
	    || basis_type == "RBJacobi_sss_noweight_lmp";
}

bool IsJacobiIndexedBasisType(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && IsJacobiIndexedBasisType(radial_basis->GetRBTypeString());
}

bool UsesPrecomputedLmpTable(const std::string& basis_type)
{
	return basis_type == "RBChebyshev_sss_lmp"
	    || basis_type == "RBLaguerre_log1p_lmp"
	    || basis_type == "RBLaguerre_log1p_pos_lmp"
	    || basis_type == "RBLaguerre_log1p_noenv_lmp"
	    || basis_type == "RBJacobi_sss_lmp"
	    || basis_type == "RBJacobi_sss_noweight_lmp";
}

bool UsesPrecomputedLmpTable(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && UsesPrecomputedLmpTable(radial_basis->GetRBTypeString());
}

double ScalingSlopeUpperBound(AnyRadialBasis* radial_basis)
{
	if (IsLaguerreLog1pBasisType(radial_basis))
		return kLaguerreRandomScalMax;
	return kRandomScalMax;
}

double ScalingSlopeRangeStart(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_slope_range)
		return mtpr.scaling_slope_range_start;
	return ScalingSlopeUpperBound(mtpr.p_RadialBasis);
}

double ScalingSlopeRangeEnd(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_slope_range)
		return mtpr.scaling_slope_range_end;
	return kRandomScalMin;
}

double ScalingShiftRangeStart(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_shift_range)
		return mtpr.scaling_shift_range_start;
	return kRandomShiftMin;
}

double ScalingShiftRangeEnd(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_shift_range)
		return mtpr.scaling_shift_range_end;
	return kRandomShiftMax;
}

}  // namespace

int MLMTPR::ScalingCoeffCount() const
{
	return 2 * species_count * species_count * K_;
}

bool MLMTPR::HasCompleteParameters() const
{
	return inited
	    && has_shift_coeffs
	    && has_scal_coeffs
	    && has_radial_coeffs
	    && has_linear_coeffs;
}

int MLMTPR::RadialCoeffOffset() const
{
	return species_count + ScalingCoeffCount();
}

int MLMTPR::RadialCoeffBlockSize() const
{
	return p_RadialBasis->rb_size + species_count;
}

int MLMTPR::BaseNonlinearCoeffCount() const
{
	return RadialCoeffOffset() + radial_func_count * RadialCoeffBlockSize();
}

bool MLMTPR::TwoLayerGateUsesSharedRadial() const
{
	return two_layer_gate_enabled_ && two_layer_gate_shared_radial_;
}

bool MLMTPR::TwoLayerGateUsesBodyLinearCombo() const
{
	return two_layer_gate_enabled_
	    && two_layer_gate_mode_ == "mu-body-linear-combo";
}

bool MLMTPR::TwoLayerGateUsesFullScalarWeights() const
{
	return two_layer_gate_enabled_ && two_layer_gate_mode_ == "mu-scalar-full";
}

bool MLMTPR::TwoLayerResidualEnabled() const
{
	return is_sh_potential_ && two_layer_gate_enabled_ && two_layer_residual_enabled_;
}

bool MLMTPR::TwoLayerGateUsesDirectScale() const
{
	return two_layer_gate_scale_mode_ == "direct";
}

bool MLMTPR::TwoLayerGateUsesCenterGate() const
{
	return two_layer_gate_enabled_ && two_layer_gate_site_mode_ == "double";
}

void MLMTPR::SetTwoLayerGateMode(const std::string& mode)
{
	if (mode == "mu-body-linear-combo" || mode == "mu-scalar-full") {
		two_layer_gate_mode_ = mode;
		return;
	}
	if (mode == "mu-body-order") {
		two_layer_gate_mode_ = "mu-scalar-full";
		return;
	}
	ERROR("SUS2-SH two-layer gate mode should be 'mu-body-linear-combo' or 'mu-scalar-full'");
}

double MLMTPR::TwoLayerGateTanhAmplitude() const
{
	return two_layer_gate_tanh_amplitude_;
}

void MLMTPR::SetTwoLayerGateTanhAmplitude(double amplitude)
{
	if (!std::isfinite(amplitude) || amplitude < 0.0 || amplitude > 1.0)
		ERROR("--two-layer-gate-tanh-amplitude should be finite and in [0, 1]");
	two_layer_gate_tanh_amplitude_ = amplitude;
}

void MLMTPR::SetTwoLayerGateSiteMode(const std::string& mode)
{
	if (mode != "neighbor" && mode != "double")
		ERROR("--two-layer-gate-site-mode should be 'neighbor' or 'double'");
	two_layer_gate_site_mode_ = mode;
}

void MLMTPR::EnsureSHScalarInfoForGateUpgrade()
{
	if (has_sh_scalar_info_)
		return;
	if (!is_sh_potential_)
		ERROR("SUS2-SH gate upgrade requires a SUS2-SH model");
	if (alpha_scalar_moments <= 0)
		ERROR("SUS2-SH gate upgrade found no scalar basis functions");
	if (alpha_moments_count <= 0)
		ERROR("SUS2-SH gate upgrade found no moment graph");

	std::vector<int> body_order(alpha_moments_count, -1);
	for (int i = 0; i < alpha_index_basic_count && i < alpha_moments_count; ++i)
		body_order[i] = 2;

	for (const SHProduct& product : sh_products_) {
		if (product.left < 0 || product.left >= alpha_moments_count
		    || product.right < 0 || product.right >= alpha_moments_count
		    || product.target < 0 || product.target >= alpha_moments_count)
			ERROR("SUS2-SH gate upgrade found an invalid product graph index");
		if (body_order[product.left] < 0 || body_order[product.right] < 0)
			ERROR("SUS2-SH gate upgrade cannot infer scalar body orders from this product graph");
		const int inferred = body_order[product.left] + body_order[product.right] - 1;
		if (inferred < 2 || inferred > sh_body_order_)
			ERROR("SUS2-SH gate upgrade inferred an out-of-range scalar body order");
		if (body_order[product.target] >= 0 && body_order[product.target] != inferred)
			ERROR("SUS2-SH gate upgrade found inconsistent body orders for one product target");
		body_order[product.target] = inferred;
	}

	sh_scalar_info_.assign(alpha_scalar_moments, SHScalarInfo());
	for (int scalar = 0; scalar < alpha_scalar_moments; ++scalar) {
		const int moment = alpha_moment_mapping[scalar];
		if (moment < 0 || moment >= alpha_moments_count)
			ERROR("SUS2-SH gate upgrade found an invalid scalar moment mapping");
		if (body_order[moment] < 0)
			ERROR("SUS2-SH gate upgrade cannot infer one scalar body order");
		sh_scalar_info_[scalar].body_order = body_order[moment];
	}
	has_sh_scalar_info_ = true;
}

void MLMTPR::UpgradePlainSHToTwoLayerGate(int gate_body_order,
                                          bool independent_gate_radial_coeffs,
                                          const std::string& gate_site_mode,
                                          const std::string& gate_mode)
{
	if (!is_sh_potential_)
		ERROR("--two-layer-gate can only upgrade a SUS2-SH model");
	if (two_layer_gate_enabled_)
		ERROR("SUS2-SH model is already initialized with two-layer gate metadata");
	(void)gate_body_order;
	const int required_gate_body_order = sh_k_max_ + 1;
	if (sh_body_order_ < required_gate_body_order)
		ERROR("--two-layer-gate requires SH body_order >= sh_k_max + 1 for mu-body-order scalar buckets");
	if (p_RadialBasis == nullptr)
		ERROR("SUS2-SH gate upgrade requires an initialized radial basis");

	EnsureSHScalarInfoForGateUpgrade();

	two_layer_gate_scalar_indices_.clear();
	std::vector<int> gate_body_counts(required_gate_body_order + 1, 0);
	for (int scalar = 0; scalar < alpha_scalar_moments; ++scalar) {
		const int body_order = sh_scalar_info_[scalar].body_order;
		if (body_order >= 2 && body_order <= required_gate_body_order) {
			two_layer_gate_scalar_indices_.push_back(scalar);
			++gate_body_counts[body_order];
		}
	}
	if (two_layer_gate_scalar_indices_.empty())
		ERROR("--two-layer-gate selected no SH scalar basis functions");
	for (int bo = 2; bo <= required_gate_body_order; ++bo)
		if (gate_body_counts[bo] == 0)
			ERROR("--two-layer-gate selected no SH scalar basis functions for one required mu body-order bucket");

	two_layer_gate_enabled_ = true;
	SetTwoLayerGateMode(gate_mode);
	SetTwoLayerGateSiteMode(gate_site_mode);
	two_layer_gate_body_order_max_ = required_gate_body_order;
	two_layer_gate_include_one_body_ = false;
	two_layer_gate_shared_radial_ = independent_gate_radial_coeffs;
	two_layer_residual_enabled_ = false;
	two_layer_gate_scale_mode_ = "legacy";
	two_layer_gate_bias_ = 1.0;
	two_layer_gate_tanh_amplitude_ = 0.8;
	InitializeTwoLayerGateAdditiveCoeffs();
	two_layer_gate_weights_.assign(TwoLayerGateWeightCount(), 1.0e-1);
	two_layer_gate_body_mix_weights_.assign(TwoLayerGateBodyMixWeightCount(), 1.0);
	two_layer_residual_e0_coeffs_.clear();

	if (two_layer_gate_shared_radial_)
		InitializeTwoLayerGateRadialCoeffsFromBase();
	else
		two_layer_gate_radial_coeffs_.clear();

	DistributeCoeffs();
	BuildTwoLayerGateProductProgram();
	BuildTwoLayerGateBodyOrderBuckets();
	ClearTwoLayerEdgePrimitiveCache();
}

void MLMTPR::UpgradeTwoLayerGateToEdgeL1()
{
	if (!is_sh_potential_)
		ERROR("--two-layer-gate-edge-l1 can only upgrade a SUS2-SH model");
	if (!two_layer_gate_enabled_)
		ERROR("--two-layer-gate-edge-l1 requires a two-layer gate model or --two-layer-gate");
	if (two_layer_gate_edge_l1_enabled_)
		return;
	if (sh_l_max_ < 1)
		ERROR("--two-layer-gate-edge-l1 requires --l-max >= 1");
	if (two_layer_gate_site_mode_ != "neighbor")
		ERROR("SUS2-SH edge L1 gate first version supports neighbor site mode only");
	if (!has_sh_scalar_info_)
		ERROR("SUS2-SH edge-L1 gate upgrade requires sh_scalar_info metadata");
	if (static_cast<int>(sh_scalar_info_.size()) != alpha_scalar_moments)
		ERROR("SUS2-SH edge-L1 gate upgrade found inconsistent sh_scalar_info metadata");

	std::vector<int> source_moment_indices =
		BuildFullEdgeL1SourceMomentBasesFromScalarInfo(
			sh_l_max_, sh_k_max_, sh_scalar_info_);
	if (source_moment_indices.empty())
		ERROR("--two-layer-gate-edge-l1 selected no L=1 source moments");
	for (int base : source_moment_indices) {
		if (base < 0 || base + 2 >= alpha_moments_count)
			ERROR("SUS2-SH edge-L1 gate upgrade produced an invalid source moment index");
	}

	two_layer_gate_edge_l1_enabled_ = true;
	two_layer_gate_edge_l1_source_mu_indices_.clear();
	two_layer_gate_edge_l1_source_moment_indices_.swap(source_moment_indices);
	two_layer_gate_edge_l1_weights_.assign(TwoLayerGateEdgeL1WeightCount(), 0.0);
	two_layer_gate_edge_l1_values_.clear();
	two_layer_gate_edge_l1_weighted_values_.clear();
	two_layer_gate_edge_l1_adjoints_.clear();
	two_layer_gate_edge_l1_value_buffer_.clear();
	two_layer_gate_edge_l1_signal_buffer_.clear();
	two_layer_gate_edge_l1_chi_buffer_.clear();
	two_layer_gate_edge_l1_chi_adj_buffer_.clear();
	two_layer_gate_edge_l1_weights_by_source_.clear();
	two_layer_gate_edge_l1_coord_loss_.clear();
	two_layer_gate_edge_l1_energy_adjoint_buffer_.clear();
	active_two_layer_gate_edge_l1_coord_loss_ = nullptr;

	DistributeCoeffs();
	BuildTwoLayerGateProductProgram();
	ClearTwoLayerEdgePrimitiveCache();
}

int MLMTPR::TwoLayerGateMuBodyOrder(int mu) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate mu index is out of range");
	if (sh_l_max_ < 0)
		ERROR("SUS2-SH two-layer gate has invalid sh_l_max");
	const int k_internal = mu / (sh_l_max_ + 1);
	return k_internal + 2;
}

int MLMTPR::TwoLayerGateWeightBodyOrder(int weight_index) const
{
	if (weight_index < 0 || weight_index >= TwoLayerGateScalarCount())
		ERROR("SUS2-SH two-layer gate scalar weight index is out of range");
	if (static_cast<int>(two_layer_gate_scalar_body_orders_.size())
	    != TwoLayerGateScalarCount())
		const_cast<MLMTPR*>(this)->BuildTwoLayerGateBodyOrderBuckets();
	return two_layer_gate_scalar_body_orders_[weight_index];
}

void MLMTPR::BuildTwoLayerGateBodyOrderBuckets()
{
	two_layer_gate_scalar_body_orders_.clear();
	two_layer_gate_body_order_weight_offsets_.clear();
	two_layer_gate_body_order_weight_indices_.clear();
	two_layer_gate_mu_body_orders_.clear();
	if (!is_sh_potential_ || !two_layer_gate_enabled_)
		return;
	if (!TwoLayerGateUsesBodyLinearCombo() && !TwoLayerGateUsesFullScalarWeights())
		ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
	if (!has_sh_scalar_info_)
		ERROR("SUS2-SH two-layer gate requires sh_scalar_info metadata");
	if (static_cast<int>(two_layer_gate_weights_.size()) != TwoLayerGateWeightCount())
		ERROR("SUS2-SH two-layer gate metadata has inconsistent sizes");
	if (TwoLayerGateUsesBodyLinearCombo()
	    && static_cast<int>(two_layer_gate_body_mix_weights_.size())
	    != TwoLayerGateBodyMixWeightCount())
		ERROR("SUS2-SH two-layer gate body mix metadata has inconsistent sizes");
	const int required_gate_body_order = sh_k_max_ + 1;
	if (sh_body_order_ < required_gate_body_order)
		ERROR("SUS2-SH two-layer gate requires sh_body_order >= sh_k_max + 1");
	if (two_layer_gate_body_order_max_ != required_gate_body_order)
		ERROR("SUS2-SH mu-body-order gate has inconsistent two_layer_gate_body_order_max");

	const int gate_count = static_cast<int>(two_layer_gate_scalar_indices_.size());
	two_layer_gate_scalar_body_orders_.assign(gate_count, 0);
	std::vector<int> counts(required_gate_body_order + 2, 0);
	for (int q = 0; q < gate_count; ++q) {
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
			ERROR("SUS2-SH two-layer gate scalar index is out of range");
		const int body_order = sh_scalar_info_[scalar_index].body_order;
		if (body_order < 2 || body_order > required_gate_body_order)
			ERROR("SUS2-SH mu-body-order gate scalar index has wrong body order");
		two_layer_gate_scalar_body_orders_[q] = body_order;
		++counts[body_order];
	}
	for (int bo = 2; bo <= required_gate_body_order; ++bo)
		if (counts[bo] == 0)
			ERROR("SUS2-SH mu-body-order gate is missing one scalar body-order bucket");

	two_layer_gate_body_order_weight_offsets_.assign(required_gate_body_order + 2, 0);
	for (int bo = 0; bo <= required_gate_body_order; ++bo)
		two_layer_gate_body_order_weight_offsets_[bo + 1] =
			two_layer_gate_body_order_weight_offsets_[bo] + counts[bo];
	two_layer_gate_body_order_weight_indices_.assign(gate_count, 0);
	std::vector<int> cursor = two_layer_gate_body_order_weight_offsets_;
	for (int q = 0; q < gate_count; ++q) {
		const int body_order = two_layer_gate_scalar_body_orders_[q];
		two_layer_gate_body_order_weight_indices_[cursor[body_order]++] = q;
	}

	two_layer_gate_mu_body_orders_.assign(radial_func_count, 0);
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int body_order = TwoLayerGateMuBodyOrder(mu);
		if (body_order < 2 || body_order > required_gate_body_order)
			ERROR("SUS2-SH mu-body-order gate has invalid mu body-order bucket");
		two_layer_gate_mu_body_orders_[mu] = body_order;
	}
}

void MLMTPR::RequestTwoLayerFullEdgeCacheForNextCalcEFS()
{
	two_layer_full_edge_cache_for_next_calc_ = true;
	two_layer_reuse_full_edge_cache_once_ = false;
}

int MLMTPR::TwoLayerGateRadialCoeffCount() const
{
	if (!TwoLayerGateUsesSharedRadial() || p_RadialBasis == nullptr)
		return 0;
	return radial_func_count * p_RadialBasis->rb_size;
}

int MLMTPR::TwoLayerGateRadialCoeffOffset() const
{
	return BaseNonlinearCoeffCount();
}

int MLMTPR::TwoLayerGateWeightCount() const
{
	if (!two_layer_gate_enabled_)
		return 0;
	if (two_layer_gate_mode_ == "mu-scalar-full")
		return radial_func_count * TwoLayerGateScalarCount();
	if (two_layer_gate_mode_ == "mu-body-linear-combo")
		return TwoLayerGateScalarCount();
	ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
	return 0;
}

int MLMTPR::TwoLayerGateAdditiveCoeffCount() const
{
	return two_layer_gate_enabled_ ? species_count : 0;
}

int MLMTPR::TwoLayerGateScalarCount() const
{
	return two_layer_gate_enabled_
		? static_cast<int>(two_layer_gate_scalar_indices_.size())
		: 0;
}

int MLMTPR::TwoLayerGateBodyOrderCount() const
{
	return two_layer_gate_enabled_ ? two_layer_gate_body_order_max_ - 1 : 0;
}

int MLMTPR::TwoLayerGateAdditiveCoeffOffset() const
{
	return BaseNonlinearCoeffCount() + TwoLayerGateRadialCoeffCount();
}

int MLMTPR::TwoLayerGateWeightOffset() const
{
	return TwoLayerGateAdditiveCoeffOffset() + TwoLayerGateAdditiveCoeffCount();
}

int MLMTPR::TwoLayerGateBodyMixWeightCount() const
{
	return TwoLayerGateUsesBodyLinearCombo()
		? radial_func_count * TwoLayerGateBodyOrderCount()
		: 0;
}

int MLMTPR::TwoLayerGateBodyMixWeightOffset() const
{
	return TwoLayerGateWeightOffset() + TwoLayerGateWeightCount();
}

bool MLMTPR::TwoLayerGateEdgeL1Enabled() const
{
	return two_layer_gate_enabled_ && two_layer_gate_edge_l1_enabled_;
}

int MLMTPR::TwoLayerGateEdgeL1SourceCount() const
{
	return TwoLayerGateEdgeL1Enabled()
		? static_cast<int>(two_layer_gate_edge_l1_source_moment_indices_.size())
		: 0;
}

int MLMTPR::TwoLayerGateEdgeL1WeightCount() const
{
	return TwoLayerGateEdgeL1Enabled()
		? radial_func_count * TwoLayerGateEdgeL1SourceCount()
		: 0;
}

int MLMTPR::TwoLayerGateEdgeL1WeightOffset() const
{
	return TwoLayerGateBodyMixWeightOffset() + TwoLayerGateBodyMixWeightCount();
}

int MLMTPR::TwoLayerGateEdgeL1SourceMomentIndex(int source_index) const
{
	if (!TwoLayerGateEdgeL1Enabled())
		return -1;
	if (source_index < 0 || source_index >= TwoLayerGateEdgeL1SourceCount())
		ERROR("SUS2-SH edge L1 gate source index is out of range");
	const int base = two_layer_gate_edge_l1_source_moment_indices_[source_index];
	if (base < 0 || base + 2 >= alpha_moments_count)
		ERROR("SUS2-SH edge L1 gate source moment index is out of range");
	return base;
}

int MLMTPR::TwoLayerResidualE0CoeffCount() const
{
	return TwoLayerResidualEnabled() ? alpha_scalar_moments : 0;
}

int MLMTPR::TwoLayerResidualE0CoeffOffset() const
{
	return TwoLayerGateEdgeL1WeightOffset() + TwoLayerGateEdgeL1WeightCount();
}

int MLMTPR::LinearCoeffOffset() const
{
	return TwoLayerResidualE0CoeffOffset() + TwoLayerResidualE0CoeffCount();
}

int MLMTPR::LinearCoeffCount() const
{
	return alpha_count + species_count - 1;
}

int MLMTPR::LinearEquationCount() const
{
	return LinearCoeffCount() + TwoLayerResidualE0CoeffCount();
}

int MLMTPR::TwoLayerGateRadialCoeffIndex(int mu, int xi) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate radial mu index is out of range");
	if (p_RadialBasis == nullptr || xi < 0 || xi >= p_RadialBasis->rb_size)
		ERROR("SUS2-SH two-layer gate radial basis index is out of range");
	if (TwoLayerGateUsesSharedRadial())
		return TwoLayerGateRadialCoeffOffset() + mu * p_RadialBasis->rb_size + xi;
	return RadialCoeffOffset() + mu * RadialCoeffBlockSize() + xi;
}

double MLMTPR::TwoLayerGateRadialCoeff(int mu, int xi) const
{
	const int coeff_index = TwoLayerGateRadialCoeffIndex(mu, xi);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	if (TwoLayerGateUsesSharedRadial()) {
		const int local_index = mu * p_RadialBasis->rb_size + xi;
		if (local_index >= 0 && local_index < static_cast<int>(two_layer_gate_radial_coeffs_.size()))
			return two_layer_gate_radial_coeffs_[local_index];
	}
	const int base_index = RadialCoeffOffset() + mu * RadialCoeffBlockSize() + xi;
	if (base_index >= 0 && base_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[base_index];
	ERROR("SUS2-SH two-layer gate radial coefficient storage is inconsistent");
	return 0.0;
}

int MLMTPR::TwoLayerGateAdditiveCoeffIndex(int type_outer, int mu) const
{
	if (!two_layer_gate_enabled_)
		return -1;
	if (type_outer < 0 || type_outer >= species_count)
		ERROR("SUS2-SH two-layer gate additive type index is out of range");
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate additive mu index is out of range");
	return TwoLayerGateAdditiveCoeffOffset() + type_outer;
}

double MLMTPR::TwoLayerGateAdditiveCoeff(int type_outer, int mu) const
{
	if (!two_layer_gate_enabled_)
		return 0.0;
	const int coeff_index = TwoLayerGateAdditiveCoeffIndex(type_outer, mu);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	const int local_index = type_outer;
	if (local_index >= 0
	    && local_index < static_cast<int>(two_layer_gate_additive_coeffs_.size()))
		return two_layer_gate_additive_coeffs_[local_index];
	return 1.0;
}

int MLMTPR::TwoLayerGateWeightIndex(int scalar_weight_index) const
{
	if (!two_layer_gate_enabled_)
		return -1;
	if (scalar_weight_index < 0 || scalar_weight_index >= TwoLayerGateWeightCount())
		ERROR("SUS2-SH two-layer gate scalar weight index is out of range");
	return TwoLayerGateWeightOffset() + scalar_weight_index;
}

int MLMTPR::TwoLayerGateWeightIndex(int mu, int scalar_weight_index) const
{
	if (!two_layer_gate_enabled_)
		return -1;
	if (scalar_weight_index < 0 || scalar_weight_index >= TwoLayerGateScalarCount())
		ERROR("SUS2-SH two-layer gate scalar weight index is out of range");
	if (TwoLayerGateUsesBodyLinearCombo()) {
		(void)mu;
		return TwoLayerGateWeightIndex(scalar_weight_index);
	}
	if (TwoLayerGateUsesFullScalarWeights()) {
		if (mu < 0 || mu >= radial_func_count)
			ERROR("SUS2-SH two-layer gate weight mu index is out of range");
		return TwoLayerGateWeightOffset()
			+ mu * TwoLayerGateScalarCount()
			+ scalar_weight_index;
	}
	ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
	return -1;
}

int MLMTPR::TwoLayerGateBodyMixWeightIndex(int mu, int body_order) const
{
	if (!two_layer_gate_enabled_)
		return -1;
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body mix weights are only defined for mu-body-linear-combo mode");
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate body mix mu index is out of range");
	if (body_order < 2 || body_order > two_layer_gate_body_order_max_)
		ERROR("SUS2-SH two-layer gate body mix body order is out of range");
	return TwoLayerGateBodyMixWeightOffset()
		+ mu * TwoLayerGateBodyOrderCount()
			+ (body_order - 2);
}

int MLMTPR::TwoLayerGateEdgeL1WeightIndex(int mu, int source_index) const
{
	if (!TwoLayerGateEdgeL1Enabled())
		return -1;
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH edge L1 gate target mu index is out of range");
	if (source_index < 0 || source_index >= TwoLayerGateEdgeL1SourceCount())
		ERROR("SUS2-SH edge L1 gate source index is out of range");
	return TwoLayerGateEdgeL1WeightOffset()
		+ mu * TwoLayerGateEdgeL1SourceCount() + source_index;
}

double MLMTPR::TwoLayerGateScalarWeight(int scalar_weight_index) const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate scalar weights are only defined for mu-body-linear-combo mode");
	const int coeff_index = TwoLayerGateWeightIndex(scalar_weight_index);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	if (scalar_weight_index >= 0
	    && scalar_weight_index < static_cast<int>(two_layer_gate_weights_.size()))
		return two_layer_gate_weights_[scalar_weight_index];
	ERROR("SUS2-SH two-layer gate scalar weight storage is inconsistent");
	return 0.0;
}

const double* MLMTPR::TwoLayerGateScalarWeightData() const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate scalar weights are only defined for mu-body-linear-combo mode");
	const int weight_count = TwoLayerGateWeightCount();
	const int coeff_offset = TwoLayerGateWeightOffset();
	if (coeff_offset >= 0
	    && coeff_offset + weight_count <= static_cast<int>(regression_coeffs.size()))
		return regression_coeffs.data() + coeff_offset;
	if (weight_count <= static_cast<int>(two_layer_gate_weights_.size()))
		return two_layer_gate_weights_.data();
	ERROR("SUS2-SH two-layer gate scalar weight storage is inconsistent");
	return nullptr;
}

double MLMTPR::TwoLayerGateWeight(int mu, int scalar_weight_index) const
{
	if (TwoLayerGateUsesFullScalarWeights()) {
		const int coeff_index = TwoLayerGateWeightIndex(mu, scalar_weight_index);
		if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
			return regression_coeffs[coeff_index];
		const int local_index = mu * TwoLayerGateScalarCount() + scalar_weight_index;
		if (local_index >= 0
		    && local_index < static_cast<int>(two_layer_gate_weights_.size()))
			return two_layer_gate_weights_[local_index];
		ERROR("SUS2-SH two-layer gate weight storage is inconsistent");
		return 0.0;
	}
	if (TwoLayerGateUsesBodyLinearCombo()) {
		const int body_order = TwoLayerGateWeightBodyOrder(scalar_weight_index);
		return TwoLayerGateScalarWeight(scalar_weight_index)
			* TwoLayerGateBodyMixWeight(mu, body_order);
	}
	ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
	return 0.0;
}

double MLMTPR::TwoLayerGateWeight(int weight_index) const
{
	if (!two_layer_gate_enabled_)
		return 0.0;
	if (weight_index < 0 || weight_index >= TwoLayerGateWeightCount())
		ERROR("SUS2-SH two-layer gate weight index is out of range");
	const int coeff_index = TwoLayerGateWeightIndex(weight_index);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	if (weight_index < static_cast<int>(two_layer_gate_weights_.size()))
		return two_layer_gate_weights_[weight_index];
	ERROR("SUS2-SH two-layer gate weight storage is inconsistent");
	return 0.0;
}

double MLMTPR::TwoLayerGateBodyMixWeight(int mu, int body_order) const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body mix weights are only defined for mu-body-linear-combo mode");
	const int coeff_index = TwoLayerGateBodyMixWeightIndex(mu, body_order);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	const int local_index =
		mu * TwoLayerGateBodyOrderCount() + (body_order - 2);
	if (local_index >= 0
	    && local_index < static_cast<int>(two_layer_gate_body_mix_weights_.size()))
		return two_layer_gate_body_mix_weights_[local_index];
	ERROR("SUS2-SH two-layer gate body mix weight storage is inconsistent");
	return 0.0;
}

double MLMTPR::TwoLayerGateBodyMixWeight(int weight_index) const
{
	if (!two_layer_gate_enabled_)
		return 0.0;
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body mix weights are only defined for mu-body-linear-combo mode");
	if (weight_index < 0 || weight_index >= TwoLayerGateBodyMixWeightCount())
		ERROR("SUS2-SH two-layer gate body mix weight index is out of range");
	const int coeff_index = TwoLayerGateBodyMixWeightOffset() + weight_index;
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	if (weight_index < static_cast<int>(two_layer_gate_body_mix_weights_.size()))
		return two_layer_gate_body_mix_weights_[weight_index];
	ERROR("SUS2-SH two-layer gate body mix weight storage is inconsistent");
	return 0.0;
}

double MLMTPR::TwoLayerGateEdgeL1Weight(int mu, int source_index) const
{
	if (!TwoLayerGateEdgeL1Enabled())
		return 0.0;
	const int coeff_index = TwoLayerGateEdgeL1WeightIndex(mu, source_index);
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	const int local_index = mu * TwoLayerGateEdgeL1SourceCount() + source_index;
	if (local_index >= 0
	    && local_index < static_cast<int>(two_layer_gate_edge_l1_weights_.size()))
		return two_layer_gate_edge_l1_weights_[local_index];
	ERROR("SUS2-SH edge L1 gate weight storage is inconsistent");
	return 0.0;
}

const double* MLMTPR::TwoLayerGateEdgeL1WeightData() const
{
	if (!TwoLayerGateEdgeL1Enabled())
		return nullptr;
	const int weight_count = TwoLayerGateEdgeL1WeightCount();
	const int coeff_offset = TwoLayerGateEdgeL1WeightOffset();
	if (coeff_offset >= 0
	    && coeff_offset + weight_count <= static_cast<int>(regression_coeffs.size()))
		return regression_coeffs.data() + coeff_offset;
	if (weight_count <= static_cast<int>(two_layer_gate_edge_l1_weights_.size()))
		return two_layer_gate_edge_l1_weights_.data();
	ERROR("SUS2-SH edge L1 gate weight storage is inconsistent");
	return nullptr;
}

int MLMTPR::FindTwoLayerGateEdgeL1RawSourceMomentIndex(int mu) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH edge L1 gate source mu index is out of range");
	for (int i = 0; i + 2 < alpha_index_basic_count; ++i) {
		if (alpha_index_basic_.comp0[i] == mu
		    && alpha_index_basic_.comp1[i] == 1
		    && alpha_index_basic_.comp2[i] == -1
		    && alpha_index_basic_.comp0[i + 1] == mu
		    && alpha_index_basic_.comp1[i + 1] == 1
		    && alpha_index_basic_.comp2[i + 1] == 0
		    && alpha_index_basic_.comp0[i + 2] == mu
		    && alpha_index_basic_.comp1[i + 2] == 1
		    && alpha_index_basic_.comp2[i + 2] == 1)
			return i;
	}
	ERROR("SUS2-SH edge L1 gate source mu does not map to a basic l=1 moment");
	return -1;
}

void MLMTPR::BuildTwoLayerGateEdgeL1Sources()
{
	two_layer_gate_edge_l1_source_mu_indices_.clear();
	two_layer_gate_edge_l1_source_moment_indices_.clear();
	if (!TwoLayerGateEdgeL1Enabled())
		return;
	if (sh_l_max_ < 1)
		ERROR("SUS2-SH edge L1 gate requires sh_l_max >= 1");
	if (two_layer_gate_site_mode_ != "neighbor")
		ERROR("SUS2-SH edge L1 gate first version supports neighbor site mode only");
	if (sh_k_max_ <= 0)
		ERROR("SUS2-SH edge L1 gate requires positive sh_k_max");
	for (int k = 0; k < sh_k_max_; ++k) {
		const int mu = k * (sh_l_max_ + 1) + 1;
		if (mu < 0 || mu >= radial_func_count)
			ERROR("SUS2-SH edge L1 gate source mu index is out of range");
		two_layer_gate_edge_l1_source_mu_indices_.push_back(mu);
		two_layer_gate_edge_l1_source_moment_indices_.push_back(
			FindTwoLayerGateEdgeL1RawSourceMomentIndex(mu));
	}
}

const double* MLMTPR::TwoLayerGateBodyMixWeightMatrixData() const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body mix weights are only defined for mu-body-linear-combo mode");
	const int weight_count = TwoLayerGateBodyMixWeightCount();
	const int coeff_offset = TwoLayerGateBodyMixWeightOffset();
	if (coeff_offset >= 0
	    && coeff_offset + weight_count <= static_cast<int>(regression_coeffs.size()))
		return regression_coeffs.data() + coeff_offset;
	if (weight_count <= static_cast<int>(two_layer_gate_body_mix_weights_.size()))
		return two_layer_gate_body_mix_weights_.data();
	ERROR("SUS2-SH two-layer gate body mix weight storage is inconsistent");
	return nullptr;
}

void MLMTPR::ComputeTwoLayerGateBodySignals(const double* scalar_values,
                                            double* body_values) const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body signals are only defined for mu-body-linear-combo mode");
	const int body_count = TwoLayerGateBodyOrderCount();
	const double* scalar_weights = TwoLayerGateScalarWeightData();
	for (int b = 0; b < body_count; ++b)
		body_values[b] = 0.0;
	for (int body_order = 2; body_order <= two_layer_gate_body_order_max_;
	     ++body_order) {
		double value = 0.0;
		for (int cursor = two_layer_gate_body_order_weight_offsets_[body_order];
		     cursor < two_layer_gate_body_order_weight_offsets_[body_order + 1];
		     ++cursor) {
			const int q = two_layer_gate_body_order_weight_indices_[cursor];
			value += scalar_weights[q] * scalar_values[q];
		}
		body_values[body_order - 2] = value;
	}
}

void MLMTPR::ComputeTwoLayerGateMuSignals(const double* body_values,
                                          double* mu_values) const
{
	if (!TwoLayerGateUsesBodyLinearCombo())
		ERROR("SUS2-SH two-layer gate body mix signals are only defined for mu-body-linear-combo mode");
	const int body_count = TwoLayerGateBodyOrderCount();
	const double* body_mix_weights = TwoLayerGateBodyMixWeightMatrixData();
	for (int mu = 0; mu < radial_func_count; ++mu) {
		double signal = 0.0;
		const double* body_mix = body_mix_weights
			+ static_cast<size_t>(mu) * body_count;
		for (int b = 0; b < body_count; ++b)
			signal += body_mix[b] * body_values[b];
		mu_values[mu] = signal;
	}
}

const double* MLMTPR::TwoLayerGateFullWeightMatrixData() const
{
	if (!TwoLayerGateUsesFullScalarWeights())
		ERROR("SUS2-SH full gate weight matrix is only defined for mu-scalar-full mode");
	const int weight_count = TwoLayerGateWeightCount();
	const int coeff_offset = TwoLayerGateWeightOffset();
	if (coeff_offset >= 0
	    && coeff_offset + weight_count <= static_cast<int>(regression_coeffs.size()))
		return regression_coeffs.data() + coeff_offset;
	if (weight_count <= static_cast<int>(two_layer_gate_weights_.size()))
		return two_layer_gate_weights_.data();
	ERROR("SUS2-SH full gate weight storage is inconsistent");
	return nullptr;
}

void MLMTPR::ComputeTwoLayerGateFullMuSignalsForAtoms(int atom_count)
{
	if (!TwoLayerGateUsesFullScalarWeights())
		ERROR("SUS2-SH full gate mu signals are only defined for mu-scalar-full mode");
	const int gate_count = TwoLayerGateScalarCount();
	if (atom_count <= 0 || gate_count <= 0 || radial_func_count <= 0)
		return;
	const size_t scalar_size = static_cast<size_t>(atom_count) * gate_count;
	const size_t gate_size = static_cast<size_t>(atom_count) * radial_func_count;
	if (two_layer_gate_scalar_values_cache_.size() != scalar_size
	    || two_layer_gate_values_.size() != gate_size)
		ERROR("SUS2-SH full gate projection cache has inconsistent sizes");
	const double* weights = TwoLayerGateFullWeightMatrixData();
	cblas_dgemm(CblasRowMajor,
	            CblasNoTrans,
	            CblasTrans,
	            atom_count,
	            radial_func_count,
	            gate_count,
	            1.0,
	            two_layer_gate_scalar_values_cache_.data(),
	            gate_count,
	            weights,
	            gate_count,
	            0.0,
	            two_layer_gate_values_.data(),
	            radial_func_count);
}

void MLMTPR::AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
	const double* mu_adjoints,
	double* scalar_seeds,
	std::vector<double>& body_scratch) const
{
	const int gate_count = TwoLayerGateScalarCount();
	std::fill(scalar_seeds, scalar_seeds + gate_count, 0.0);
	if (TwoLayerGateUsesBodyLinearCombo()) {
		const int body_count = TwoLayerGateBodyOrderCount();
		const double* body_mix_weights = TwoLayerGateBodyMixWeightMatrixData();
		const double* scalar_weights = TwoLayerGateScalarWeightData();
		body_scratch.assign(body_count, 0.0);
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double adjoint = mu_adjoints[mu];
			if (adjoint == 0.0)
				continue;
			const double* body_mix = body_mix_weights
				+ static_cast<size_t>(mu) * body_count;
			for (int b = 0; b < body_count; ++b)
				body_scratch[b] += adjoint * body_mix[b];
		}
		for (int q = 0; q < gate_count; ++q) {
			const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
			scalar_seeds[q] = body_scratch[body_index] * scalar_weights[q];
		}
		return;
	}
	if (TwoLayerGateUsesFullScalarWeights()) {
		cblas_dgemv(CblasRowMajor,
		            CblasTrans,
		            radial_func_count,
		            gate_count,
		            1.0,
		            TwoLayerGateFullWeightMatrixData(),
		            gate_count,
		            mu_adjoints,
		            1,
		            0.0,
		            scalar_seeds,
		            1);
		return;
	}
	ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
}

double MLMTPR::TwoLayerResidualE0Coeff(int scalar_index) const
{
	if (!TwoLayerResidualEnabled())
		return 0.0;
	if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
		ERROR("SUS2-SH residual E0 scalar index is out of range");
	const int coeff_index = TwoLayerResidualE0CoeffOffset() + scalar_index;
	if (coeff_index >= 0 && coeff_index < static_cast<int>(regression_coeffs.size()))
		return regression_coeffs[coeff_index];
	if (scalar_index < static_cast<int>(two_layer_residual_e0_coeffs_.size()))
		return two_layer_residual_e0_coeffs_[scalar_index];
	ERROR("SUS2-SH residual E0 coefficient storage is inconsistent");
	return 0.0;
}

void MLMTPR::InitializeTwoLayerGateRadialCoeffsFromBase()
{
	const int gate_radial_count = TwoLayerGateRadialCoeffCount();
	if (gate_radial_count == 0)
		return;
	const int R = p_RadialBasis->rb_size;
	const int base_offset = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	two_layer_gate_radial_coeffs_.assign(gate_radial_count, 0.0);
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int source_offset = base_offset + mu * block_size;
		const int target_offset = mu * R;
		if (source_offset + R > static_cast<int>(regression_coeffs.size()))
			ERROR("SUS2-SH cannot initialize gate radial coefficients from base radial block");
		for (int xi = 0; xi < R; ++xi)
			two_layer_gate_radial_coeffs_[target_offset + xi] =
				regression_coeffs[source_offset + xi];
	}
}

void MLMTPR::InitializeTwoLayerGateAdditiveCoeffs(double value)
{
	const int gate_additive_count = TwoLayerGateAdditiveCoeffCount();
	if (gate_additive_count == 0) {
		two_layer_gate_additive_coeffs_.clear();
		return;
	}
	two_layer_gate_additive_coeffs_.assign(gate_additive_count, value);
}

int MLMTPR::EnforcePositiveRadialFirstCoeffs(double min_value)
{
	const double positive_floor = min_value > 0.0 ? min_value : kRadialFirstCoeffPositiveFloor;
	if (p_RadialBasis == nullptr)
		return 0;
	const int radial_begin = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int radial_end = radial_begin + radial_func_count * block_size;
	if (block_size <= 0 ||
	    radial_func_count <= 0 ||
	    radial_begin < 0 ||
	    radial_end > static_cast<int>(regression_coeffs.size()))
		return 0;

	int changed_count = 0;
	for (int mu = 0; mu < radial_func_count; ++mu) {
		double& first_coeff = regression_coeffs[radial_begin + mu * block_size];
		const double old_value = first_coeff;
		if (first_coeff < 0.0)
			first_coeff = -first_coeff;
		if (first_coeff <= positive_floor)
			first_coeff = positive_floor;
		if (first_coeff != old_value)
			++changed_count;
	}
	return changed_count;
}

bool MLMTPR::IsRadialFirstCoeff(int coeff_index) const
{
	if (p_RadialBasis == nullptr)
		return false;
	const int radial_begin = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int radial_end = radial_begin + radial_func_count * block_size;
	if (block_size <= 0 || coeff_index < radial_begin || coeff_index >= radial_end)
		return false;
	return (coeff_index - radial_begin) % block_size == 0;
}

double MLMTPR::RadialFirstCoeffRawToValue(double raw_value) const
{
	return kRadialFirstCoeffPositiveFloor + StableSoftplus(raw_value);
}

double MLMTPR::RadialFirstCoeffValueToRaw(double coeff_value) const
{
	const double softplus_value =
		std::max(kRadialFirstCoeffPositiveFloor, coeff_value - kRadialFirstCoeffPositiveFloor);
	return StableInvSoftplus(softplus_value);
}

double MLMTPR::RadialFirstCoeffDerivativeFromValue(double coeff_value) const
{
	const double softplus_value =
		std::max(kRadialFirstCoeffPositiveFloor, coeff_value - kRadialFirstCoeffPositiveFloor);
	return 1.0 - std::exp(-softplus_value);
}

bool MLMTPR::IsRedundantRadialSpeciesCoeff(int coeff_index) const
{
	const int radial_begin = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int rb_size = p_RadialBasis->rb_size;
	if (coeff_index < radial_begin)
		return false;
	const int radial_end = radial_begin + radial_func_count * block_size;
	if (coeff_index >= radial_end)
		return false;

	const int relative_index = coeff_index - radial_begin;
	const int mu = relative_index / block_size;
	const int in_block = relative_index % block_size;
	return mu > 0 && in_block >= rb_size;
}

void MLMTPR::BuildActiveCoeffIndices(std::vector<int>& active_coeff_indices, bool exclude_scal_coeffs) const
{
	active_coeff_indices.clear();
	active_coeff_indices.reserve(regression_coeffs.size());
	const int scal_begin = species_count;
	const int scal_end = RadialCoeffOffset();
	for (int coeff_index = 0; coeff_index < static_cast<int>(regression_coeffs.size()); ++coeff_index) {
		if (exclude_scal_coeffs && coeff_index >= scal_begin && coeff_index < scal_end)
			continue;
		if (IsRedundantRadialSpeciesCoeff(coeff_index))
			continue;
		active_coeff_indices.push_back(coeff_index);
	}
}

int MLMTPR::ActiveCoeffCount(bool exclude_scal_coeffs) const
{
	std::vector<int> active_coeff_indices;
	BuildActiveCoeffIndices(active_coeff_indices, exclude_scal_coeffs);
	return static_cast<int>(active_coeff_indices.size());
}

int MLMTPR::ScalingSlopeOffset(int scaling_block, int type_central, int type_outer) const
{
	const int pair_count = species_count * species_count;
	return species_count + 2 * scaling_block * pair_count + type_central * species_count + type_outer;
}

int MLMTPR::ScalingShiftOffset(int scaling_block, int type_central, int type_outer) const
{
	const int pair_count = species_count * species_count;
	return ScalingSlopeOffset(scaling_block, type_central, type_outer) + pair_count;
}

double MLMTPR::OrderedPairStrength(int type_central, int type_outer) const
{
	return std::sqrt(static_cast<double>(type_central + 1))
	     + std::sqrt(static_cast<double>(type_outer + 1));
}

double MLMTPR::NormalizedOrderedPairStrength(int type_central, int type_outer) const
{
	if (species_count <= 1)
		return 0.5;

	const double min_strength = OrderedPairStrength(0, 0);
	const double max_strength = OrderedPairStrength(species_count - 1, species_count - 1);
	if (max_strength <= min_strength)
		return 0.5;

	return Clamp01((OrderedPairStrength(type_central, type_outer) - min_strength) /
	               (max_strength - min_strength));
}

void MLMTPR::SetScalingSlopeRange(double start, double end)
{
	custom_scaling_slope_range = true;
	scaling_slope_range_start = start;
	scaling_slope_range_end = end;
}

void MLMTPR::SetScalingShiftRange(double start, double end)
{
	custom_scaling_shift_range = true;
	scaling_shift_range_start = start;
	scaling_shift_range_end = end;
}

void MLMTPR::InitializeDefaultScalingCoeffs()
{
	const double scal_start = ScalingSlopeRangeStart(*this);
	const double scal_end = ScalingSlopeRangeEnd(*this);
	const double shift_start = ScalingShiftRangeStart(*this);
	const double shift_end = ScalingShiftRangeEnd(*this);
	for (int type_central = 0; type_central < species_count; ++type_central) {
		for (int type_outer = 0; type_outer < species_count; ++type_outer) {
			for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
				const double strength = NormalizedOrderedPairStrength(type_central, type_outer);
				regression_coeffs[ScalingSlopeOffset(scaling_block, type_central, type_outer)] =
					Lerp(scal_start, scal_end, strength);
				regression_coeffs[ScalingShiftOffset(scaling_block, type_central, type_outer)] =
					Lerp(shift_start, shift_end, strength);
			}
		}
	}
}

void MLMTPR::RandomizeScalingCoeffs(std::mt19937_64& generator, double strength_jitter)
{
	const double scal_start = ScalingSlopeRangeStart(*this);
	const double scal_end = ScalingSlopeRangeEnd(*this);
	const double shift_start = ScalingShiftRangeStart(*this);
	const double shift_end = ScalingShiftRangeEnd(*this);
	std::uniform_real_distribution<double> jitter(-strength_jitter, strength_jitter);
	for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
		for (int type_central = 0; type_central < species_count; ++type_central) {
			for (int type_outer = 0; type_outer < species_count; ++type_outer) {
				const double strength = NormalizedOrderedPairStrength(type_central, type_outer);
				const double sample = Clamp01(strength + jitter(generator));
				regression_coeffs[ScalingSlopeOffset(scaling_block, type_central, type_outer)] =
					Lerp(scal_start, scal_end, sample);
				regression_coeffs[ScalingShiftOffset(scaling_block, type_central, type_outer)] =
					Lerp(shift_start, shift_end, sample);
			}
		}
	}
}

void MLMTPR::RandomizeRadialCoeffs(std::mt19937_64& generator, double radial_scale)
{
	std::uniform_real_distribution<double> uniform(-1.0, 1.0);
	const int radial_coeff_offset = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int rb_size = p_RadialBasis->rb_size;

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int block_offset = radial_coeff_offset + mu * block_size;
		for (int xi = 0; xi < rb_size; ++xi)
			regression_coeffs[block_offset + xi] = radial_scale * uniform(generator);
		for (int type = 0; type < species_count; ++type)
			regression_coeffs[block_offset + rb_size + type] = 1.0;
	}
	EnforcePositiveRadialFirstCoeffs();
	if (TwoLayerGateUsesSharedRadial()) {
		InitializeTwoLayerGateRadialCoeffsFromBase();
		const int gate_radial_offset = TwoLayerGateRadialCoeffOffset();
		const int gate_radial_count = TwoLayerGateRadialCoeffCount();
		if (gate_radial_offset + gate_radial_count <=
		    static_cast<int>(regression_coeffs.size())) {
			for (int i = 0; i < gate_radial_count; ++i)
				regression_coeffs[gate_radial_offset + i] =
					two_layer_gate_radial_coeffs_[i];
		}
	}
}

void MLMTPR::RandomizeNonlinearCoeffs(std::mt19937_64& generator, double radial_scale, bool include_scaling, double scaling_strength_jitter)
{
	if (include_scaling)
		RandomizeScalingCoeffs(generator, scaling_strength_jitter);
	RandomizeRadialCoeffs(generator, radial_scale);
}

void MLMTPR::PruneSpecies(const std::vector<int>& old_species_indices)
{
	if (!HasCompleteParameters())
		ERROR("PruneSpecies requires a complete trained model with shift/scal/radial/linear coefficients.");
	if (old_species_indices.empty())
		ERROR("PruneSpecies requires at least one species index.");

	const int old_species_count = species_count;
	const int new_species_count = static_cast<int>(old_species_indices.size());
	std::vector<int> sorted_indices = old_species_indices;
	std::sort(sorted_indices.begin(), sorted_indices.end());
	for (int old_type : old_species_indices) {
		if (old_type < 0 || old_type >= old_species_count)
			ERROR("PruneSpecies species index is out of range.");
	}
	for (int i = 1; i < new_species_count; ++i) {
		if (sorted_indices[i] == sorted_indices[i - 1])
			ERROR("PruneSpecies species indices should be unique.");
	}

	const int rb_size = p_RadialBasis->rb_size;
	const int old_pair_count = old_species_count * old_species_count;
	const int new_pair_count = new_species_count * new_species_count;
	const int old_scaling_offset = old_species_count;
	const int new_scaling_offset = new_species_count;
	const int old_radial_offset = old_species_count + 2 * old_pair_count * K_;
	const int new_radial_offset = new_species_count + 2 * new_pair_count * K_;
	const int old_radial_block_size = rb_size + old_species_count;
	const int new_radial_block_size = rb_size + new_species_count;
	const int gate_radial_count = TwoLayerGateRadialCoeffCount();
	const int old_gate_additive_count = two_layer_gate_enabled_
		? old_species_count
		: 0;
	const int new_gate_additive_count = two_layer_gate_enabled_
		? new_species_count
		: 0;
	const int gate_weight_count = two_layer_gate_enabled_
		? static_cast<int>(two_layer_gate_scalar_indices_.size())
		: 0;
	const int gate_body_mix_weight_count = two_layer_gate_enabled_
		? radial_func_count * (two_layer_gate_body_order_max_ - 1)
		: 0;
	const int old_gate_radial_offset =
		old_radial_offset + radial_func_count * old_radial_block_size;
	const int new_gate_radial_offset =
		new_radial_offset + radial_func_count * new_radial_block_size;
	const int old_gate_additive_offset = old_gate_radial_offset + gate_radial_count;
	const int new_gate_additive_offset = new_gate_radial_offset + gate_radial_count;
	const int old_gate_offset = old_gate_additive_offset + old_gate_additive_count;
	const int new_gate_offset = new_gate_additive_offset + new_gate_additive_count;
	const int old_gate_body_mix_offset = old_gate_offset + gate_weight_count;
	const int new_gate_body_mix_offset = new_gate_offset + gate_weight_count;
	const int old_linear_offset = old_gate_body_mix_offset + gate_body_mix_weight_count;
	const int new_linear_offset = new_gate_body_mix_offset + gate_body_mix_weight_count;
	const int old_linear_count = alpha_count + old_species_count - 1;
	const int new_linear_count = alpha_count + new_species_count - 1;

	if (static_cast<int>(regression_coeffs.size()) < old_linear_offset + old_linear_count)
		ERROR("PruneSpecies found an inconsistent regression coefficient layout.");

	const std::vector<double> old_coeffs = regression_coeffs;
	std::vector<double> new_coeffs(new_linear_offset + new_linear_count, 0.0);

	for (int new_type = 0; new_type < new_species_count; ++new_type) {
		const int old_type = old_species_indices[new_type];
		new_coeffs[new_type] = old_coeffs[old_type];
	}

	for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
		for (int new_center = 0; new_center < new_species_count; ++new_center) {
			const int old_center = old_species_indices[new_center];
			for (int new_outer = 0; new_outer < new_species_count; ++new_outer) {
				const int old_outer = old_species_indices[new_outer];
				const int old_slope_offset = old_scaling_offset
					+ 2 * scaling_block * old_pair_count
					+ old_center * old_species_count
					+ old_outer;
				const int old_shift_offset = old_slope_offset + old_pair_count;
				const int new_slope_offset = new_scaling_offset
					+ 2 * scaling_block * new_pair_count
					+ new_center * new_species_count
					+ new_outer;
				const int new_shift_offset = new_slope_offset + new_pair_count;
				new_coeffs[new_slope_offset] = old_coeffs[old_slope_offset];
				new_coeffs[new_shift_offset] = old_coeffs[old_shift_offset];
			}
		}
	}

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int old_block_offset = old_radial_offset + mu * old_radial_block_size;
		const int new_block_offset = new_radial_offset + mu * new_radial_block_size;
		for (int xi = 0; xi < rb_size; ++xi)
			new_coeffs[new_block_offset + xi] = old_coeffs[old_block_offset + xi];
		for (int new_type = 0; new_type < new_species_count; ++new_type) {
			const int old_type = old_species_indices[new_type];
			new_coeffs[new_block_offset + rb_size + new_type] =
				old_coeffs[old_block_offset + rb_size + old_type];
		}
	}

	for (int i = 0; i < gate_radial_count; ++i)
		new_coeffs[new_gate_radial_offset + i] =
			old_coeffs[old_gate_radial_offset + i];

	for (int new_type = 0; new_type < new_species_count; ++new_type) {
		const int old_type = old_species_indices[new_type];
		new_coeffs[new_gate_additive_offset + new_type] =
			old_coeffs[old_gate_additive_offset + old_type];
	}

	for (int i = 0; i < gate_weight_count; ++i)
		new_coeffs[new_gate_offset + i] = old_coeffs[old_gate_offset + i];

	for (int i = 0; i < gate_body_mix_weight_count; ++i)
		new_coeffs[new_gate_body_mix_offset + i] =
			old_coeffs[old_gate_body_mix_offset + i];

	for (int new_type = 0; new_type < new_species_count; ++new_type) {
		const int old_type = old_species_indices[new_type];
		new_coeffs[new_linear_offset + new_type] = old_coeffs[old_linear_offset + old_type];
	}
	for (int moment = 0; moment < alpha_count - 1; ++moment) {
		new_coeffs[new_linear_offset + new_species_count + moment] =
			old_coeffs[old_linear_offset + old_species_count + moment];
	}

	species_count = new_species_count;
	regression_coeffs.swap(new_coeffs);
	linear_coeffs.assign(new_linear_count, 0.0);
	for (int i = 0; i < new_linear_count; ++i)
		linear_coeffs[i] = regression_coeffs[new_linear_offset + i];
	if (gate_radial_count > 0) {
		two_layer_gate_radial_coeffs_.assign(gate_radial_count, 0.0);
		for (int i = 0; i < gate_radial_count; ++i)
			two_layer_gate_radial_coeffs_[i] =
				regression_coeffs[new_gate_radial_offset + i];
	}
	if (gate_weight_count > 0) {
		two_layer_gate_weights_.assign(gate_weight_count, 0.0);
		for (int i = 0; i < gate_weight_count; ++i)
			two_layer_gate_weights_[i] = regression_coeffs[new_gate_offset + i];
	}
	if (gate_body_mix_weight_count > 0) {
		two_layer_gate_body_mix_weights_.assign(gate_body_mix_weight_count, 0.0);
		for (int i = 0; i < gate_body_mix_weight_count; ++i)
			two_layer_gate_body_mix_weights_[i] =
				regression_coeffs[new_gate_body_mix_offset + i];
	}
	if (new_gate_additive_count > 0) {
		two_layer_gate_additive_coeffs_.assign(new_gate_additive_count, 0.0);
		for (int i = 0; i < new_gate_additive_count; ++i)
			two_layer_gate_additive_coeffs_[i] =
				regression_coeffs[new_gate_additive_offset + i];
	}

	radial_list.resize(0, 0, 0);
	radial_der_list.resize(0, 0, 0);
	two_layer_gate_radial_list.resize(0, 0, 0);
	two_layer_gate_radial_der_list.resize(0, 0, 0);
	max_radial.clear();
	inited = true;
	has_shift_coeffs = true;
	has_scal_coeffs = true;
	has_radial_coeffs = true;
	has_linear_coeffs = true;
}


void MLMTPR::Load(const string& filename)
{
	alpha_count = 0;
	energy_cmpnts = NULL;
	stress_cmpnts = NULL;
	has_shift_coeffs = false;
	has_scal_coeffs = false;
	has_radial_coeffs = false;
	has_linear_coeffs = false;

	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Cannot open " + filename);


	char tmpline[1000];
	string tmpstr;

	ifs.getline(tmpline, 1000);
	int len = (int)((string)tmpline).length();
	if (len > 0 && tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
		tmpline[len - 1] = '\0';

	if ((string)tmpline == "MTP" ||
		((string)tmpline).compare(0, 10, "version = ") != 0)
	{
		// version reading block
		ifs.getline(tmpline, 1000);
		len = (int)((string)tmpline).length();
		if (len > 0 && tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
			tmpline[len - 1] = '\0';
	}
	if (((string)tmpline).compare(0, 10, "version = ") != 0)
		ERROR("MTP file must contain a version header");

		// name/description reading block
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		if (tmpstr == "scaling") // optional 
		{
			ifs.ignore(2);
			ifs >> scaling;
			ifs >> tmpstr;
		}
		if (tmpstr != "L") // optional 
		{
			ERROR("Error reading .mtp file");
		}
		ifs.ignore(2);
		ifs >> L;
		ifs >> tmpstr;
		if (tmpstr != "scaling_map")
		{
			ERROR("Error reading .mtp file");
		}
			ifs.ignore(2);
			ifs >> scaling_map;
			ifs >> tmpstr;
		L += 1;   // L=0,1 ...... L_max

	if (tmpstr != "species_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> species_count;

			ifs >> tmpstr;
			potential_tag = "";
			is_sh_potential_ = false;
			ClearZBL();
			sh_products_.clear();
		sh_product_rows_.clear();
		sh_product_row_terms_.clear();
		sh_scalar_index_by_moment_.clear();
		sh_scalar_terminal_product_.clear();
	sh_product_rows_trace_printed_ = false;
	sh_site_der_cache_trace_printed_ = false;
	has_sh_scalar_info_ = false;
	sh_scalar_info_.clear();
	two_layer_gate_enabled_ = false;
	two_layer_gate_body_order_max_ = 0;
		two_layer_gate_include_one_body_ = false;
		two_layer_gate_shared_radial_ = false;
		two_layer_gate_mode_ = "mu-body-linear-combo";
		two_layer_gate_site_mode_ = "neighbor";
		two_layer_residual_enabled_ = false;
	two_layer_gate_scale_mode_ = "legacy";
		two_layer_gate_bias_ = 1.0;
		two_layer_gate_tanh_amplitude_ = 0.8;
			two_layer_gate_scalar_indices_.clear();
			two_layer_gate_scalar_body_orders_.clear();
			two_layer_gate_body_order_weight_offsets_.clear();
			two_layer_gate_body_order_weight_indices_.clear();
			two_layer_gate_mu_body_orders_.clear();
			two_layer_gate_radial_coeffs_.clear();
				two_layer_gate_additive_coeffs_.clear();
				two_layer_gate_weights_.clear();
				two_layer_gate_body_mix_weights_.clear();
			two_layer_gate_edge_l1_enabled_ = false;
				two_layer_gate_edge_l1_source_mu_indices_.clear();
				two_layer_gate_edge_l1_source_moment_indices_.clear();
				two_layer_gate_edge_l1_weights_.clear();
				two_layer_gate_edge_l1_values_.clear();
				two_layer_gate_edge_l1_weighted_values_.clear();
				two_layer_gate_edge_l1_adjoints_.clear();
			two_layer_gate_edge_l1_value_buffer_.clear();
			two_layer_gate_edge_l1_signal_buffer_.clear();
			two_layer_gate_edge_l1_chi_buffer_.clear();
			two_layer_gate_edge_l1_chi_adj_buffer_.clear();
			two_layer_residual_e0_coeffs_.clear();
	two_layer_gate_required_moments_.clear();
	two_layer_gate_required_moment_indices_.clear();
	two_layer_gate_required_basic_indices_.clear();
	two_layer_gate_required_basic_dense_mu_indices_.clear();
	two_layer_gate_required_product_indices_.clear();
	two_layer_gate_required_mu_indices_.clear();
	two_layer_gate_mu_dense_index_.clear();
	two_layer_gate_required_radial_eval_blocks_.clear();
	two_layer_gate_values_.clear();
	two_layer_gate_scalar_values_cache_.clear();
	two_layer_gate_body_values_cache_.clear();
		two_layer_gate_moment_values_cache_.clear();
		two_layer_final_moment_values_cache_.clear();
		two_layer_final_moment_ders_cache_.clear();
		two_layer_forward_energy_gate_adjoints_cache_.clear();
		two_layer_gate_adjoints_.clear();
		two_layer_gate_values_from_edge_cache_ready_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		ClearTwoLayerEdgePrimitiveCache();
		sh_body_l_max_.assign(7, 0);
			if (tmpstr == "potential_tag")
			{
				getline(ifs, tmpstr);
				potential_tag = ReadAssignmentTail(tmpstr);
				is_sh_potential_ = (potential_tag == "SUS2-SH");
				ifs >> tmpstr;
			}
		if (tmpstr == "zbl_enabled")
		{
			std::string bool_token;
			ifs.ignore(2);
			ifs >> bool_token;
			const bool zbl_enabled = ReadBoolToken(bool_token);
			ifs >> tmpstr;
			if (zbl_enabled) {
				double zbl_inner = DefaultZBLInnerCutoff();
				double zbl_outer = DefaultZBLOuterCutoff();
				bool zbl_typewise = false;
				double zbl_typewise_factor = DefaultZBLTypewiseCutoffFactor();
				std::vector<int> zbl_atomic_numbers;

				if (tmpstr != "zbl_inner")
					ERROR("ZBL section is missing zbl_inner");
				ifs.ignore(2);
				ifs >> zbl_inner;
				ifs >> tmpstr;
				if (tmpstr != "zbl_outer")
					ERROR("ZBL section is missing zbl_outer");
				ifs.ignore(2);
				ifs >> zbl_outer;
				ifs >> tmpstr;
				if (tmpstr != "zbl_typewise_cutoff_enabled")
					ERROR("ZBL section is missing zbl_typewise_cutoff_enabled");
				ifs.ignore(2);
				ifs >> bool_token;
				zbl_typewise = ReadBoolToken(bool_token);
				ifs >> tmpstr;
				if (tmpstr != "zbl_typewise_cutoff_factor")
					ERROR("ZBL section is missing zbl_typewise_cutoff_factor");
				ifs.ignore(2);
				ifs >> zbl_typewise_factor;
				ifs >> tmpstr;
				if (tmpstr != "zbl_atomic_numbers")
					ERROR("ZBL section is missing zbl_atomic_numbers");
				ReadIntList(ifs, zbl_atomic_numbers, species_count);
				ifs.ignore(1000, '\n');
				ConfigureZBL(zbl_atomic_numbers, zbl_inner, zbl_outer,
				             zbl_typewise, zbl_typewise_factor);
				ifs >> tmpstr;
			}
		}
		if (tmpstr == "potential_tag")
		{
			getline(ifs, tmpstr);
			potential_tag = ReadAssignmentTail(tmpstr);
			is_sh_potential_ = (potential_tag == "SUS2-SH");
			ifs >> tmpstr;
		}
			if (is_sh_potential_)
			{
		if (tmpstr != "sh_l_max")
			ERROR("SUS2-SH model is missing sh_l_max");
		ifs.ignore(2);
		ifs >> sh_l_max_;
		ifs >> tmpstr;
		if (tmpstr != "sh_k_max")
			ERROR("SUS2-SH model is missing sh_k_max");
		ifs.ignore(2);
		ifs >> sh_k_max_;
		ifs >> tmpstr;
		if (tmpstr != "sh_body_order")
			ERROR("SUS2-SH model is missing sh_body_order");
		ifs.ignore(2);
		ifs >> sh_body_order_;
		ifs >> tmpstr;
		if (tmpstr != "sh_parity")
			ERROR("SUS2-SH model is missing sh_parity");
		ifs.ignore(2);
		ifs >> sh_parity_;
		ifs >> tmpstr;
			if (tmpstr == "sh_body_l_max") {
				std::vector<int> body_values;
				const int body_lmax_count = sh_body_order_ >= 6 ? 5 : 4;
				ReadIntList(ifs, body_values, body_lmax_count);
			sh_body_l_max_.assign(7, sh_l_max_);
			for (int body = 2; body <= 1 + body_lmax_count; ++body)
				sh_body_l_max_[body] = body_values[body - 2];
				ifs.ignore(1000, '\n');
				ifs >> tmpstr;
			} else {
				sh_body_l_max_.assign(7, sh_l_max_);
			}
		}

		if (tmpstr != "radial_basis_type")
			ERROR("Error reading .mtp file");
		ifs.ignore(2);
		ifs >> rbasis_type;
	
	if (rbasis_type == "RBChebyshev")
		p_RadialBasis = new RadialBasis_Chebyshev(ifs);
	else if (rbasis_type == "RBChebyshev_repuls")
		p_RadialBasis = new RadialBasis_Chebyshev_repuls(ifs);
        else if (rbasis_type == "RBChebyshev_s")
                p_RadialBasis = new RadialBasis_Chebyshev_s(ifs);
        else if (rbasis_type == "RBChebyshev_ss")
                p_RadialBasis = new RadialBasis_Chebyshev_ss(ifs);
        else if (rbasis_type == "RBChebyshev_ssw")
                p_RadialBasis = new RadialBasis_Chebyshev_ssw(ifs);
        else if (rbasis_type == "RBChebyshev_sss")
                p_RadialBasis = new RadialBasis_Chebyshev_sss(ifs);
        else if (rbasis_type == "RBChebyshev_sss_rational")
                p_RadialBasis = new RadialBasis_Chebyshev_sss_rational(ifs);
        else if (rbasis_type == "RBChebyshev_sssw")
                p_RadialBasis = new RadialBasis_Chebyshev_sssw(ifs);
		else if (rbasis_type == "RBChebyshev_sss_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sss_lmp(ifs);
		else if (rbasis_type == "RBChebyshev_sss_rational_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sss_rational_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ss_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_ss_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ssw_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_ssw_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_sssw_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sssw_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_s_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_s_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ssss")
                p_RadialBasis = new RadialBasis_Chebyshev_ssss(ifs);
        else if (rbasis_type == "RBChebyshev_sssss")
                p_RadialBasis = new RadialBasis_Chebyshev_sssss(ifs);
        else if (rbasis_type == "RBChebyshev_tanhexp")
                p_RadialBasis = new RadialBasis_Chebyshev_tanhexp(ifs);
        else if (rbasis_type == "RBChebyshev_tanhexp_w")
                p_RadialBasis = new RadialBasis_Chebyshev_tanhexp_w(ifs);
	        else if (rbasis_type == "RBChebyshev_sigma")
	                p_RadialBasis = new RadialBasis_Chebyshev_sigma(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_lmp(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_pos")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_pos(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_pos_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_pos_lmp(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_noenv")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_noenv(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_noenv_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_noenv_lmp(ifs);
	        else if (rbasis_type == "RBJacobi_sss")
	                p_RadialBasis = new RadialBasis_Jacobi_sss(ifs);
	        else if (rbasis_type == "RBJacobi_sss_lmp")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_lmp(ifs);
	        else if (rbasis_type == "RBJacobi_sss_noweight")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_noweight(ifs);
	        else if (rbasis_type == "RBJacobi_sss_noweight_lmp")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_noweight_lmp(ifs);
	        else if (rbasis_type == "RBBessel")
	                p_RadialBasis = new RadialBasis_Bessel(ifs);
	        else if (rbasis_type == "RBBessel_sss")
	                p_RadialBasis = new RadialBasis_Bessel_sss(ifs);
        else if (rbasis_type == "RBBesselw")
                p_RadialBasis = new RadialBasis_Besselw(ifs);
        else if (rbasis_type == "RBBessel_sssw")
                p_RadialBasis = new RadialBasis_Bessel_sssw(ifs);
        else if (rbasis_type == "RBChebyshev_Tri")
                p_RadialBasis = new RadialBasis_Chebyshev_Tri(ifs);
	else if (rbasis_type == "RBShapeev")
		p_RadialBasis = new RadialBasis_Shapeev(ifs);
	else if (rbasis_type == "RBTaylor")
		p_RadialBasis = new RadialBasis_Taylor(ifs);
		else
			ERROR("Wrong radial basis type");

	// We do not need double scaling
	if (p_RadialBasis->scaling != 1.0) {
		scaling *= p_RadialBasis->scaling;
		p_RadialBasis->scaling = 1.0;
	}


	ifs >> tmpstr;
	if (tmpstr != "radial_funcs_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
		ifs >> radial_func_count;
	mu_to_K.resize(radial_func_count);
	mu_to_sigma.resize(radial_func_count);
	for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_sigma[n] = n / L;
	}
	if (scaling_map=="K")
	{
		K_ = radial_func_count / L;
	
	for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n / L;
	}
	}
	else if (scaling_map=="L")
	{
		K_ = L;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n % L;
	}

	}
	else if (scaling_map=="LK")
	{
		K_ = radial_func_count;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n;
	}

	}
	else 
	{
		ERROR("Wrong scaling map");
	}

	if (is_sh_potential_)
	{
		if (scaling_map != "LK")
			ERROR("SUS2-SH requires scaling_map = LK");
		if (sh_l_max_ < 0 || sh_l_max_ > 4)
			ERROR("SUS2-SH runtime currently supports sh_l_max in [0,4]");
		if (sh_k_max_ <= 0)
			ERROR("SUS2-SH requires positive sh_k_max");
		if (L != sh_l_max_ + 1)
			ERROR("SUS2-SH L header is inconsistent with sh_l_max");
		if (radial_func_count != sh_k_max_ * L)
			ERROR("SUS2-SH radial_funcs_count must equal sh_k_max * (sh_l_max + 1)");
		if (sh_body_order_ < 2 || sh_body_order_ > 6)
			ERROR("SUS2-SH sh_body_order should be in [2,6]");
		if (sh_parity_ != "even")
			ERROR("SUS2-SH currently supports even parity only");
		if (sh_body_l_max_.size() < 7)
			sh_body_l_max_.assign(7, sh_l_max_);
		for (int body = 2; body <= sh_body_order_; ++body)
			if (sh_body_l_max_[body] < 0 || sh_body_l_max_[body] > sh_l_max_)
				ERROR("SUS2-SH sh_body_l_max entry is out of range");
	}



	//Radial coeffs initialization
	int pairs_count = species_count*species_count;           //number of species pairs

	char foo = ' ';
	int foo_int = 0;

	regression_coeffs.resize(species_count +2*pairs_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));

	inited = true;
        
        ifs >> tmpstr;
        if (tmpstr != "shift_coeffs")
        {

                for (int i = 0; i <species_count; i++)
                regression_coeffs[i] = -1.0;

                }
        else
        {has_shift_coeffs = true;
		ifs.ignore(4);

                for (int i = 0; i < species_count; i++){
                        ifs >> regression_coeffs[i] >> foo;}
                        ifs >> tmpstr; }
    
        
//	ifs >> tmpstr;
        if (tmpstr != "scal_coeffs")
        {
           InitializeDefaultScalingCoeffs();
                }
        else
        {has_scal_coeffs = true;
		ifs.ignore(4);
                
                for (int i = 0; i < 2*pairs_count* K_; i++){
                        ifs >> regression_coeffs[species_count+i] >> foo;}
                        ifs >> tmpstr; }
                
	if (tmpstr == "radial_coeffs")
	{
		has_radial_coeffs = true;

		for (int s1 = 0; s1 < 1; s1++)
			for (int s2 = 0; s2 < 1; s2++)
			{
				ifs >> foo_int >> foo >> foo_int;

				double t;

				for (int i = 0; i < radial_func_count; i++)
				{
					ifs >> foo;
					for (int j = 0; j < p_RadialBasis->rb_size + species_count ; j++)
					{
						ifs >> t >> foo;
						regression_coeffs[species_count+2*pairs_count* K_ +i*(p_RadialBasis->rb_size + species_count) + j] = t;

					}

				}

			}

		ifs >> tmpstr;

	}
	else
	{
		//cout << "Radial coeffs not found, initializing defaults" << endl;
		inited = false;
		has_radial_coeffs = false;

		regression_coeffs.resize(species_count+2*species_count*species_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));


		for (pairs_count = 0; pairs_count < 1; pairs_count++)
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size + species_count; j++)
				{	regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
					i*(p_RadialBasis->rb_size + species_count) + j] = 1e-2;
                                        if (j >= p_RadialBasis->rb_size) {regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
                                        i*(p_RadialBasis->rb_size + species_count) + j] = 1.1; }
                                }             
			//	regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
			//		i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size)] = 1e-3;


			}
	}

	if (tmpstr != "alpha_moments_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_moments_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_basic_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_basic = new int[alpha_index_basic_count][4];
	alpha_index_basic_.comp0.resize(alpha_index_basic_count);
	alpha_index_basic_.comp1.resize(alpha_index_basic_count);
	alpha_index_basic_.comp2.resize(alpha_index_basic_count);
	alpha_index_basic_.comp3.resize(alpha_index_basic_count);
	if (alpha_index_basic == nullptr)
		ERROR("Memory allocation error");

	int radial_func_max = -1;
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		if (is_sh_potential_) {
			int k = 0, l = 0, m = 0;
			ifs >> k >> tmpch >> l >> tmpch >> m;
			if (k < 0 || k >= sh_k_max_ || l < 0 || l > sh_l_max_ || std::abs(m) > l)
				ERROR("Invalid SUS2-SH alpha_index_basic entry");
			alpha_index_basic[i][0] = k * L + l;
			alpha_index_basic[i][1] = l;
			alpha_index_basic[i][2] = m;
			alpha_index_basic[i][3] = 0;
		} else {
			ifs >> alpha_index_basic[i][0]  >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
		}
		alpha_index_basic_.comp0[i]=alpha_index_basic[i][0];
		alpha_index_basic_.comp1[i]=alpha_index_basic[i][1];
		alpha_index_basic_.comp2[i]=alpha_index_basic[i][2];
		alpha_index_basic_.comp3[i]=alpha_index_basic[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");

		if (alpha_index_basic[i][0]>radial_func_max)
			radial_func_max = alpha_index_basic[i][0];
	}
	

	//cout << radial_func_count << endl;

	if (!is_sh_potential_ && radial_func_max!=radial_func_count-1)
		ERROR("Wrong number of radial functions specified");
	if (is_sh_potential_ && radial_func_max >= radial_func_count)
		ERROR("SUS2-SH alpha_index_basic references an out-of-range radial function");

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_times_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_times = new int[alpha_index_times_count][4];
	alpha_index_times_.comp0.resize(alpha_index_times_count);
	alpha_index_times_.comp1.resize(alpha_index_times_count);
	alpha_index_times_.comp2.resize(alpha_index_times_count);
	alpha_index_times_.comp3.resize(alpha_index_times_count);
	if (alpha_index_times == nullptr)
		ERROR("Memory allocation error");

	for (int i = 0; i < alpha_index_times_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
		alpha_index_times_.comp0[i]=alpha_index_times[i][0];
		alpha_index_times_.comp1[i]=alpha_index_times[i][1];
		alpha_index_times_.comp2[i]=alpha_index_times[i][2];
		alpha_index_times_.comp3[i]=alpha_index_times[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (is_sh_potential_ && tmpstr == "sh_product_count")
		ReadSHProductGraph(ifs, tmpstr);
	if (tmpstr != "alpha_scalar_moments")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_scalar_moments;
	if (alpha_scalar_moments < 0)
		ERROR("Error reading .mtp file");

	alpha_moment_mapping = new int[alpha_scalar_moments];
	if (alpha_moment_mapping == nullptr)
		ERROR("Memory allocation error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_moment_mapping")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
		for (int i = 0; i < alpha_scalar_moments; i++)
		{
			char tmpch = ' ';
			ifs >> alpha_moment_mapping[i] >> tmpch;
			if (ifs.fail())
				ERROR("Error reading .mtp file");
		}
		if (is_sh_potential_) {
			std::vector<char> defined(alpha_moments_count, 0);
			for (int i = 0; i < alpha_index_basic_count && i < alpha_moments_count; ++i)
				defined[i] = 1;
			for (size_t i = 0; i < sh_products_.size(); ++i)
				defined[sh_products_[i].target] = 1;
			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const int node = alpha_moment_mapping[i];
				if (node < 0 || node >= alpha_moments_count)
					ERROR("SUS2-SH alpha_moment_mapping index out of range");
				if (!defined[node])
					ERROR("SUS2-SH alpha_moment_mapping references an undefined tensor");
			}
		}
		ifs.ignore(1000, '\n');

	alpha_count = alpha_scalar_moments + 1;


	//Reading linear coeffs
	ifs >> tmpstr;
	if (is_sh_potential_ && tmpstr == "sh_scalar_info_count") {
		int sh_scalar_info_count = 0;
		ifs.ignore(2);
		ifs >> sh_scalar_info_count;
		if (sh_scalar_info_count != alpha_scalar_moments)
			ERROR("SUS2-SH sh_scalar_info_count should match alpha_scalar_moments");
		ifs >> tmpstr;
		if (tmpstr != "sh_scalar_info")
			ERROR("Cannot read SUS2-SH scalar metadata");
		ifs.ignore(4);
		sh_scalar_info_.resize(sh_scalar_info_count);
		for (int i = 0; i < sh_scalar_info_count; ++i) {
			char tmpch = ' ';
			ifs.ignore(1000, '{');
			SHScalarInfo& info = sh_scalar_info_[i];
			ifs >> info.body_order >> tmpch
			    >> info.q[0] >> tmpch
			    >> info.q[1] >> tmpch
			    >> info.q[2] >> tmpch
			    >> info.q[3] >> tmpch
			    >> info.q[4] >> tmpch
			    >> info.intermediate_l;
			if (ifs.fail())
				ERROR("Error reading SUS2-SH scalar metadata");
		}
		ifs.ignore(1000, '\n');
		has_sh_scalar_info_ = true;
		ifs >> tmpstr;
	}
	if (is_sh_potential_ && sh_body_order_ > 2 && sh_products_.empty() && alpha_scalar_moments > 0)
		ERROR("SUS2-SH model has scalar moments but no sh_products graph");
	if (is_sh_potential_)
		BuildSHProductProgram();

	if (is_sh_potential_ && tmpstr == "two_layer_gate_enabled") {
		std::string bool_token;
		ifs.ignore(2);
		ifs >> bool_token;
		two_layer_gate_enabled_ = ReadBoolToken(bool_token);

		ifs >> tmpstr;
			if (tmpstr != "two_layer_gate_mode")
				ERROR("SUS2-SH legacy two-layer gate models are not compatible with mu-body-order gate mode");
			ifs.ignore(2);
			std::string loaded_gate_mode;
			ifs >> loaded_gate_mode;
			SetTwoLayerGateMode(loaded_gate_mode);

		ifs >> tmpstr;
		if (tmpstr != "two_layer_gate_body_order_max")
			ERROR("SUS2-SH two-layer gate is missing two_layer_gate_body_order_max");
		ifs.ignore(2);
		ifs >> two_layer_gate_body_order_max_;

		ifs >> tmpstr;
		if (tmpstr != "two_layer_gate_include_one_body")
			ERROR("SUS2-SH two-layer gate is missing two_layer_gate_include_one_body");
		ifs.ignore(2);
			ifs >> bool_token;
			two_layer_gate_include_one_body_ = ReadBoolToken(bool_token);

			ifs >> tmpstr;
			if (tmpstr == "two_layer_gate_site_mode") {
				ifs.ignore(2);
				ifs >> two_layer_gate_site_mode_;
				if (two_layer_gate_site_mode_ != "neighbor"
				    && two_layer_gate_site_mode_ != "double")
					ERROR("SUS2-SH two-layer gate has an unknown site mode: " + two_layer_gate_site_mode_);
				ifs >> tmpstr;
			} else {
				two_layer_gate_site_mode_ = "neighbor";
			}
			if (tmpstr == "two_layer_residual_enabled") {
				ifs.ignore(2);
			ifs >> bool_token;
			two_layer_residual_enabled_ = ReadBoolToken(bool_token);
			if (two_layer_residual_enabled_)
				ERROR("SUS2-SH residual two-layer mode is not supported by mu-body-order gate models");
			ifs >> tmpstr;
		}
		if (tmpstr == "two_layer_gate_scale_mode") {
			ERROR("SUS2-SH mu-body-order gate models do not use two_layer_gate_scale_mode");
		}
		if (tmpstr == "two_layer_gate_bias") {
			ERROR("SUS2-SH mu-body-order gate models do not use two_layer_gate_bias");
		}
		if (tmpstr == "two_layer_gate_tanh_amplitude") {
			ifs.ignore(2);
			ifs >> two_layer_gate_tanh_amplitude_;
			if (!std::isfinite(two_layer_gate_tanh_amplitude_)
			    || two_layer_gate_tanh_amplitude_ < 0.0
			    || two_layer_gate_tanh_amplitude_ > 1.0)
				ERROR("SUS2-SH two_layer_gate_tanh_amplitude should be finite and in [0, 1]");
			ifs >> tmpstr;
		}
		if (tmpstr == "two_layer_gate_radial_mode") {
			std::string radial_mode;
			ifs.ignore(2);
				ifs >> radial_mode;
				if (radial_mode == "shared-radial") {
					two_layer_gate_shared_radial_ = true;
				} else if (radial_mode == "base-radial" || radial_mode == "legacy") {
					two_layer_gate_shared_radial_ = false;
				} else {
					ERROR("SUS2-SH two-layer gate has an unknown radial mode: " + radial_mode);
				}

				ifs >> tmpstr;
				if (tmpstr != "two_layer_gate_radial_coeff_count")
					ERROR("SUS2-SH two-layer shared-radial gate is missing two_layer_gate_radial_coeff_count");
				ifs.ignore(2);
				int gate_radial_count = 0;
				ifs >> gate_radial_count;
				const int expected_gate_radial_count = TwoLayerGateRadialCoeffCount();
				if (gate_radial_count != expected_gate_radial_count)
					ERROR("SUS2-SH two-layer shared-radial gate radial coefficient count is inconsistent");

				ifs >> tmpstr;
				if (tmpstr != "two_layer_gate_radial_coeffs")
					ERROR("SUS2-SH two-layer shared-radial gate is missing two_layer_gate_radial_coeffs");
					ReadDoubleList(ifs, two_layer_gate_radial_coeffs_, gate_radial_count);
					ifs.ignore(1000, '\n');
					ifs >> tmpstr;
				}
				if (tmpstr == "two_layer_gate_additive_coeff_count") {
					ifs.ignore(2);
					int gate_additive_count = 0;
					ifs >> gate_additive_count;
					const int expected_gate_additive_count =
						TwoLayerGateAdditiveCoeffCount();
					if (gate_additive_count != expected_gate_additive_count)
						ERROR("SUS2-SH two-layer gate additive coefficient count is inconsistent");
					ifs >> tmpstr;
					if (tmpstr != "two_layer_gate_additive_coeffs")
						ERROR("SUS2-SH two-layer gate is missing two_layer_gate_additive_coeffs");
					ReadDoubleList(ifs, two_layer_gate_additive_coeffs_, gate_additive_count);
					ifs.ignore(1000, '\n');
					ifs >> tmpstr;
				} else {
					InitializeTwoLayerGateAdditiveCoeffs();
				}
				if (tmpstr != "two_layer_gate_weight_count")
					ERROR("SUS2-SH two-layer gate is missing two_layer_gate_weight_count");
		ifs.ignore(2);
			int gate_weight_count = 0;
			ifs >> gate_weight_count;
			if (gate_weight_count <= 0)
				ERROR("SUS2-SH two-layer gate should contain at least one scalar weight");
			int gate_scalar_count = gate_weight_count;
			if (TwoLayerGateUsesFullScalarWeights()) {
				if (radial_func_count <= 0
				    || gate_weight_count % radial_func_count != 0)
					ERROR("SUS2-SH full mu/scalar gate weight count is inconsistent");
				gate_scalar_count = gate_weight_count / radial_func_count;
			}

		ifs >> tmpstr;
		if (tmpstr != "two_layer_gate_scalar_indices")
			ERROR("SUS2-SH two-layer gate is missing two_layer_gate_scalar_indices");
		ReadIntList(ifs, two_layer_gate_scalar_indices_, gate_scalar_count);
		ifs.ignore(1000, '\n');

		ifs >> tmpstr;
		if (tmpstr != "two_layer_gate_weights")
			ERROR("SUS2-SH two-layer gate is missing two_layer_gate_weights");
		ReadDoubleList(ifs, two_layer_gate_weights_, gate_weight_count);
		ifs.ignore(1000, '\n');

			if (!(ifs >> tmpstr))
				tmpstr = "";
			if (TwoLayerGateUsesBodyLinearCombo()) {
				if (tmpstr != "two_layer_gate_body_mix_weight_count")
					ERROR("SUS2-SH two-layer gate is missing two_layer_gate_body_mix_weight_count");
				ifs.ignore(2);
				int gate_body_mix_weight_count = 0;
				ifs >> gate_body_mix_weight_count;
				const int expected_gate_body_mix_weight_count =
					radial_func_count * (two_layer_gate_body_order_max_ - 1);
				if (gate_body_mix_weight_count != expected_gate_body_mix_weight_count)
					ERROR("SUS2-SH two-layer gate body mix weight count is inconsistent");

				ifs >> tmpstr;
				if (tmpstr != "two_layer_gate_body_mix_weights")
					ERROR("SUS2-SH two-layer gate is missing two_layer_gate_body_mix_weights");
				ReadDoubleList(ifs,
				               two_layer_gate_body_mix_weights_,
				               gate_body_mix_weight_count);
				ifs.ignore(1000, '\n');
				if (!(ifs >> tmpstr))
					tmpstr = "";
				} else {
					two_layer_gate_body_mix_weights_.clear();
					if (tmpstr == "two_layer_gate_body_mix_weight_count")
						ERROR("SUS2-SH mu-scalar-full gate models should not contain body mix weights");
				}
			if (tmpstr == "two_layer_gate_edge_l1_enabled") {
				ifs.ignore(2);
				ifs >> bool_token;
				two_layer_gate_edge_l1_enabled_ = ReadBoolToken(bool_token);
				if (!(ifs >> tmpstr))
					tmpstr = "";
				if (two_layer_gate_edge_l1_enabled_) {
					if (two_layer_gate_site_mode_ != "neighbor")
						ERROR("SUS2-SH edge L1 gate first version supports neighbor site mode only");
					if (sh_l_max_ < 1)
						ERROR("SUS2-SH edge L1 gate requires sh_l_max >= 1");

					const bool has_full_l1_sources =
						tmpstr == "two_layer_gate_edge_l1_source_moment_count";
					const bool has_legacy_raw_sources =
						tmpstr == "two_layer_gate_edge_l1_source_mu_count";
					if (!has_full_l1_sources && !has_legacy_raw_sources)
						ERROR("SUS2-SH edge L1 gate is missing source count metadata");
					ifs.ignore(2);
					int edge_l1_source_count = 0;
					ifs >> edge_l1_source_count;
					if (edge_l1_source_count <= 0)
						ERROR("SUS2-SH edge L1 gate source count should be positive");

					ifs >> tmpstr;
					std::vector<int> loaded_source_indices;
					if (has_full_l1_sources) {
						if (tmpstr != "two_layer_gate_edge_l1_source_moment_indices")
							ERROR("SUS2-SH edge L1 gate is missing two_layer_gate_edge_l1_source_moment_indices");
					} else if (tmpstr != "two_layer_gate_edge_l1_source_mu_indices") {
						ERROR("SUS2-SH edge L1 gate is missing two_layer_gate_edge_l1_source_mu_indices");
					}
					ReadIntList(ifs, loaded_source_indices, edge_l1_source_count);
					ifs.ignore(1000, '\n');

					ifs >> tmpstr;
					if (tmpstr != "two_layer_gate_edge_l1_weight_count")
						ERROR("SUS2-SH edge L1 gate is missing two_layer_gate_edge_l1_weight_count");
					ifs.ignore(2);
					int edge_l1_weight_count = 0;
					ifs >> edge_l1_weight_count;
					if (edge_l1_weight_count != radial_func_count * edge_l1_source_count)
						ERROR("SUS2-SH edge L1 gate weight count is inconsistent");

					ifs >> tmpstr;
					if (tmpstr != "two_layer_gate_edge_l1_weights")
						ERROR("SUS2-SH edge L1 gate is missing two_layer_gate_edge_l1_weights");
					ReadDoubleList(ifs, two_layer_gate_edge_l1_weights_, edge_l1_weight_count);
					ifs.ignore(1000, '\n');

					two_layer_gate_edge_l1_source_mu_indices_.clear();
					two_layer_gate_edge_l1_source_moment_indices_.clear();
					if (has_full_l1_sources) {
						for (int base : loaded_source_indices) {
							if (base < 0 || base + 2 >= alpha_moments_count)
								ERROR("SUS2-SH edge L1 gate source moment index is out of range");
							two_layer_gate_edge_l1_source_moment_indices_.push_back(base);
						}
					} else {
						two_layer_gate_edge_l1_source_mu_indices_ = loaded_source_indices;
						for (int mu : two_layer_gate_edge_l1_source_mu_indices_)
							two_layer_gate_edge_l1_source_moment_indices_.push_back(
								FindTwoLayerGateEdgeL1RawSourceMomentIndex(mu));
					}
					if (!(ifs >> tmpstr))
						tmpstr = "";
				} else {
					two_layer_gate_edge_l1_source_mu_indices_.clear();
					two_layer_gate_edge_l1_source_moment_indices_.clear();
					two_layer_gate_edge_l1_weights_.clear();
				}
			}

			if (!has_sh_scalar_info_)
				ERROR("SUS2-SH two-layer gate requires sh_scalar_info metadata");
		for (int index : two_layer_gate_scalar_indices_) {
			if (index < 0 || index >= alpha_scalar_moments)
				ERROR("SUS2-SH two-layer gate scalar index is out of range");
			if (sh_scalar_info_[index].body_order < 2
			    || sh_scalar_info_[index].body_order > two_layer_gate_body_order_max_)
				ERROR("SUS2-SH mu-body-order gate scalar index has wrong body order");
		}
		if (two_layer_gate_include_one_body_)
			ERROR("SUS2-SH two-layer gate currently requires two_layer_gate_include_one_body = false");
		if (two_layer_gate_body_order_max_ != sh_k_max_ + 1)
			ERROR("SUS2-SH mu-body-order gate body order should be sh_k_max + 1");
			if (two_layer_gate_body_order_max_ < 2 || two_layer_gate_body_order_max_ > sh_body_order_)
				ERROR("SUS2-SH mu-body-order gate body order is out of range");
				BuildTwoLayerGateBodyOrderBuckets();
		if (tmpstr == "two_layer_residual_e0_coeff_count") {
			ifs.ignore(2);
			int e0_count = 0;
			ifs >> e0_count;
			if (!two_layer_residual_enabled_)
				ERROR("SUS2-SH found residual E0 coefficients without two_layer_residual_enabled");
			if (e0_count != alpha_scalar_moments)
				ERROR("SUS2-SH residual E0 coefficient count should match alpha_scalar_moments");
			ifs >> tmpstr;
			if (tmpstr != "two_layer_residual_e0_coeffs")
				ERROR("SUS2-SH residual mode is missing two_layer_residual_e0_coeffs");
			ReadDoubleList(ifs, two_layer_residual_e0_coeffs_, e0_count);
			ifs.ignore(1000, '\n');
			if (!(ifs >> tmpstr))
				tmpstr = "";
		}
		if (two_layer_residual_enabled_) {
			if (two_layer_gate_scale_mode_ != "direct")
				ERROR("SUS2-SH residual two-layer mode requires two_layer_gate_scale_mode = direct");
			if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != alpha_scalar_moments)
				two_layer_residual_e0_coeffs_.assign(alpha_scalar_moments, 0.0);
		}
		}

		if (tmpstr != "species_coeffs")
	{
		inited = false;
		//cout << "Linear coeffs not found, initializing defaults, species_count = " << species_count << endl;
		linear_coeffs.resize(LinearCoeffCount());
		for (int i = 0; i < LinearCoeffCount(); i++)
			linear_coeffs[i] = 1e-3;
	}
	else
	{
		has_linear_coeffs = true;
		ifs.ignore(4);

		linear_coeffs.resize(species_count);
		for (int i = 0; i < species_count; i++)
			ifs >> linear_coeffs[i] >> foo;


		ifs >> tmpstr;

		if (tmpstr != "moment_coeffs")
			ERROR("Cannot read linear coeffs");

		ifs.ignore(2);

		linear_coeffs.resize(LinearCoeffCount());

		ifs.ignore(10, '{');


		for (int i = 0; i < alpha_count - 1; i++)
			ifs >> linear_coeffs[i + species_count] >> foo;

	}
	if (UsesPrecomputedLmpTable(rbasis_type))
	{
		radial_list.resize(species_count * species_count, 200002, radial_func_count);
		radial_der_list.resize(species_count * species_count, 200002, radial_func_count);
		radial_list.set(0);
		radial_der_list.set(0);
		if (is_sh_potential_ && TwoLayerGateUsesSharedRadial()) {
			two_layer_gate_radial_list.resize(species_count * species_count, 200002, radial_func_count);
			two_layer_gate_radial_der_list.resize(species_count * species_count, 200002, radial_func_count);
			two_layer_gate_radial_list.set(0);
			two_layer_gate_radial_der_list.set(0);
		} else {
			two_layer_gate_radial_list.resize(0, 0, 0);
			two_layer_gate_radial_der_list.resize(0, 0, 0);
		}

		inv_dr = 200000 / p_RadialBasis->max_dist;
		double dr = 1 / inv_dr;
		const int C = species_count;
		const int R = p_RadialBasis->rb_size;
		int k_;
		double factor;
		for (int i = 0; i < C; i++)
		{
			for (int j = 0; j < C; j++)
			{
					for (int n = 0; n < 200001; n++)
					{
						for (int mu = 0; mu < radial_func_count; mu++)
						{
							k_ = mu_to_K[mu];
							const int basis_k = UsesJacobiIndexedBasis()
								? JacobiIndexedBlockForMu(mu)
								: mu_to_sigma[k_];

							p_RadialBasis->RB_Calc(dr * n,
								regression_coeffs[C + 2 * k_ * C * C + C * i + j],
								1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * i + j],
								basis_k);
						for (int xi = 0; xi < R; xi++)
						{
							factor = regression_coeffs[C + 2 * C * C * K_ + mu * (R + C) + xi];
							if (!is_sh_potential_) {
								factor *= regression_coeffs[C + 2 * C * C * K_ + R + i]
								       * regression_coeffs[C + 2 * C * C * K_ + R + j];
							}
							radial_list(i * C + j, n, mu) += p_RadialBasis->rb_vals[xi] * scaling * factor;
							radial_der_list(i * C + j, n, mu) += p_RadialBasis->rb_ders[xi] * scaling * factor;
							if (is_sh_potential_ && TwoLayerGateUsesSharedRadial()) {
								const double gate_factor = TwoLayerGateRadialCoeff(mu, xi);
								two_layer_gate_radial_list(i * C + j, n, mu) +=
									p_RadialBasis->rb_vals[xi] * scaling * gate_factor;
								two_layer_gate_radial_der_list(i * C + j, n, mu) +=
									p_RadialBasis->rb_ders[xi] * scaling * gate_factor;
							}
						}
					}
				}
			}
		}
	}
	else
	{
		radial_list.resize(0, 0, 0);
		radial_der_list.resize(0, 0, 0);
		two_layer_gate_radial_list.resize(0, 0, 0);
		two_layer_gate_radial_der_list.resize(0, 0, 0);
	}
		MemAlloc();
		DistributeCoeffs();
}



void MLMTPR::CalcDescriptors(Configuration& cfg, ofstream& ofs)
{
	int n = LinearCoeffCount();

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << nbh.my_type;
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << basis_vals[1 + i] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}

void MLMTPR::CalcpartialE(Configuration& cfg, ofstream& ofs)
{
	int n = LinearCoeffCount();

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << linear_coeffs[species_count + i]*basis_vals[1 + i]*linear_coeffs[nbh.my_type] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}

void MLMTPR::CalcEij(Configuration& cfg, ofstream& ofs)
{
	if (!is_sh_potential_)
		ERROR("calc-eij currently supports SUS2-SH models only");
	if (!has_sh_scalar_info_)
		ERROR("calc-eij requires sh_scalar_info metadata; initialize or rewrite the model with --write-sh-scalar-info");
	if (TwoLayerResidualEnabled())
		ERROR("calc-eij does not support two-layer residual models");

	struct DirectedPair {
		int center = -1;
		int neighbor = -1;
		int type_center = -1;
		int type_neighbor = -1;
		Vector3 vec;
		double dist = 0.0;
		double energy = 0.0;
	};

	Neighborhoods neighborhoods(cfg, CutOff());
	PrepareEvalCaches();

	const bool use_two_layer_gate = RequiresTwoLayerGateEvaluation();
	if (!use_two_layer_gate) {
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		active_two_layer_edge_cache_atom_index_ = -1;
	} else {
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, false, false);
		PrepareTwoLayerGateValues(cfg, neighborhoods);
		active_two_layer_gate_values_ = &two_layer_gate_values_;
		active_two_layer_gate_adjoints_ = nullptr;
	}

	std::vector<DirectedPair> directed_pairs;
	std::vector<double> edge_basic_values;
	std::vector<double> edge_basic_weights;
	const bool use_fast_projection = UseFastCalcEijProjection();
	for (int ind = 0; ind < cfg.size(); ++ind) {
		if (use_two_layer_gate)
			active_two_layer_edge_cache_atom_index_ = ind;
		const Neighborhood& nbh = neighborhoods[ind];
		CalcSHMomentValuesOnly(nbh);
		if (use_fast_projection)
			CalcSHDirectedEffectivePairWeights(nbh, edge_basic_weights);
		for (int j = 0; j < nbh.count; ++j) {
			DirectedPair pair;
			pair.center = ind;
			pair.neighbor = nbh.inds[j];
			pair.type_center = nbh.my_type;
			pair.type_neighbor = nbh.types[j];
			pair.vec = nbh.vecs[j];
			pair.dist = nbh.dists[j];
			pair.energy = use_fast_projection
				? CalcSHDirectedEffectivePairEnergyFromWeights(
					nbh, j, edge_basic_values, edge_basic_weights)
				: CalcSHDirectedEffectivePairEnergy(
					nbh, j, edge_basic_values);
			directed_pairs.push_back(pair);
		}
	}

	active_two_layer_edge_cache_atom_index_ = -1;
	if (use_two_layer_gate)
		active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;

	std::vector<char> used(directed_pairs.size(), 0);
	const double base_tol = 1.0e-8;
	for (size_t i = 0; i < directed_pairs.size(); ++i) {
		if (used[i])
			continue;
		used[i] = 1;
		const DirectedPair& first = directed_pairs[i];
		double pair_energy = first.energy;

		if (first.center != first.neighbor) {
			int reverse_index = -1;
			for (size_t k = i + 1; k < directed_pairs.size(); ++k) {
				if (used[k])
					continue;
				const DirectedPair& candidate = directed_pairs[k];
				if (candidate.center != first.neighbor
				    || candidate.neighbor != first.center)
					continue;
				const double dist_tol =
					base_tol * std::max(1.0, std::max(first.dist, candidate.dist));
				if (std::abs(candidate.dist - first.dist) > dist_tol)
					continue;
				double vec_err = 0.0;
				for (int a = 0; a < 3; ++a) {
					const double sum = candidate.vec[a] + first.vec[a];
					vec_err += sum * sum;
				}
				if (std::sqrt(vec_err) > dist_tol)
					continue;
				reverse_index = static_cast<int>(k);
				break;
			}
			if (reverse_index >= 0) {
				used[reverse_index] = 1;
				pair_energy += directed_pairs[reverse_index].energy;
			}
		}

		const int type0 = std::min(first.type_center, first.type_neighbor);
		const int type1 = std::max(first.type_center, first.type_neighbor);
		ofs << type0 << ' ' << type1 << ' '
		    << first.dist << ' ' << pair_energy << '\n';
	}
}






void MLMTPR::Save(const string& filename)
{
	ofstream ofs(filename);
	ofs.setf(ios::scientific);
	ofs.precision(15);

	ofs << "MTP\n";
	ofs << "version = 1.1.0\n";
	ofs << "potential_name = " << pot_desc << endl;
	if(scaling != 1.0)
		ofs << "scaling = " << scaling << endl;
	ofs << "L = " << L-1 << endl;
	ofs << "scaling_map = " << scaling_map << endl;
	ofs << "species_count = " << species_count << endl;
	if (HasZBL()) {
		const ZBLPotential& zbl = ZBL();
		ofs << "zbl_enabled = true\n";
		ofs << "zbl_inner = " << zbl.InnerCutoff() << '\n';
		ofs << "zbl_outer = " << zbl.OuterCutoff() << '\n';
		ofs << "zbl_typewise_cutoff_enabled = "
		    << (zbl.TypewiseCutoffEnabled() ? "true" : "false") << '\n';
		ofs << "zbl_typewise_cutoff_factor = "
		    << zbl.TypewiseCutoffFactor() << '\n';
		ofs << "zbl_atomic_numbers = {";
		const std::vector<int>& atomic_numbers = zbl.AtomicNumbers();
		for (size_t i = 0; i < atomic_numbers.size(); ++i) {
			if (i != 0)
				ofs << ", ";
			ofs << atomic_numbers[i];
		}
		ofs << "}\n";
	}
	if (is_sh_potential_)
	{
		ofs << "potential_tag = SUS2-SH" << endl;
		ofs << "sh_l_max = " << sh_l_max_ << endl;
		ofs << "sh_k_max = " << sh_k_max_ << endl;
		ofs << "sh_body_order = " << sh_body_order_ << endl;
		ofs << "sh_parity = " << sh_parity_ << endl;
		std::vector<int> body_lmax = sh_body_l_max_;
		if (body_lmax.size() < 7)
			body_lmax.assign(7, sh_l_max_);
		ofs << "sh_body_l_max = {" << body_lmax[2] << ", " << body_lmax[3]
		    << ", " << body_lmax[4] << ", " << body_lmax[5];
		if (sh_body_order_ >= 6)
			ofs << ", " << body_lmax[6];
		ofs << "}" << endl;
	}
	else
		ofs << "potential_tag = " << "" << endl;
	p_RadialBasis->WriteRadialBasis(ofs);
	ofs << "\tradial_funcs_count = " << radial_func_count << '\n';
             
        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;


//        ofs << "\tradial_coeffs" << '\n';


        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count* K_; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[species_count+i];
        }
        ofs << '}' << endl;


	ofs << "\tradial_coeffs" << '\n';

	int q = species_count+2*species_count*species_count* K_;
	for (int i = 0; i < 1; i++)
		for (int j = 0; j < 1; j++)
		{
			ofs <<"\t\t"<< i << "-" << j << "\n";

			for (int k = 0; k < radial_func_count; k++)
			{
				ofs << "\t\t\t{";
				for (int l = 0; l < p_RadialBasis->rb_size + species_count; l++)
				{
					ofs << regression_coeffs[q++];
					if (l != p_RadialBasis->rb_size - 1 + species_count)
						ofs << ", ";
					else
						ofs << "}" << endl;
				}
			}
		}

	ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
	ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';

	ofs << "alpha_index_basic = {";
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		if (is_sh_potential_) {
			const int mu = alpha_index_basic[i][0];
			const int k = mu / L;
			ofs << '{' << k << ", "
				<< alpha_index_basic[i][1] << ", "
				<< alpha_index_basic[i][2] << "}";
		} else {
			ofs << '{'
				<< alpha_index_basic[i][0] << ", "
				<< alpha_index_basic[i][1] << ", "
				<< alpha_index_basic[i][2] << ", "
				<< alpha_index_basic[i][3] << "}";
		}
		if (i < alpha_index_basic_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';

	ofs << "alpha_index_times = {";
	for (int i = 0; i < alpha_index_times_count; i++)
	{
		ofs << '{'
			<< alpha_index_times[i][0] << ", "
			<< alpha_index_times[i][1] << ", "
			<< alpha_index_times[i][2] << ", "
			<< alpha_index_times[i][3] << "}";
		if (i < alpha_index_times_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	if (is_sh_potential_)
		WriteSHProductGraph(ofs);

	ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

	ofs << "alpha_moment_mapping = {";
	for (int i = 0; i < alpha_scalar_moments; i++) {
		if (i > 0)
			ofs << ", ";
		ofs << alpha_moment_mapping[i];
	}
	ofs << "}\n";
	if (is_sh_potential_ && has_sh_scalar_info_) {
		ofs << "sh_scalar_info_count = " << sh_scalar_info_.size() << '\n';
		ofs << "sh_scalar_info = {";
		for (int i = 0; i < static_cast<int>(sh_scalar_info_.size()); ++i) {
			if (i > 0)
				ofs << ", ";
			const SHScalarInfo& info = sh_scalar_info_[i];
			ofs << "{" << info.body_order << ", "
			    << info.q[0] << ", " << info.q[1] << ", " << info.q[2] << ", "
			    << info.q[3] << ", " << info.q[4] << ", " << info.intermediate_l << "}";
		}
		ofs << "}\n";
	}
	if (is_sh_potential_ && two_layer_gate_enabled_) {
		if (!has_sh_scalar_info_)
			ERROR("SUS2-SH two-layer gate requires sh_scalar_info metadata");
		if (static_cast<int>(two_layer_gate_weights_.size()) != TwoLayerGateWeightCount())
			ERROR("SUS2-SH two-layer gate metadata has inconsistent sizes");
		for (int index : two_layer_gate_scalar_indices_) {
			if (index < 0 || index >= static_cast<int>(sh_scalar_info_.size()))
				ERROR("SUS2-SH two-layer gate scalar index is out of range");
			if (sh_scalar_info_[index].body_order < 2
			    || sh_scalar_info_[index].body_order > two_layer_gate_body_order_max_)
				ERROR("SUS2-SH mu-body-order gate scalar index has wrong body order");
		}
		ofs << "two_layer_gate_enabled = true\n";
		ofs << "two_layer_gate_mode = " << two_layer_gate_mode_ << '\n';
		ofs << "two_layer_gate_body_order_max = " << two_layer_gate_body_order_max_ << '\n';
		ofs << "two_layer_gate_include_one_body = false\n";
		ofs << "two_layer_gate_site_mode = " << two_layer_gate_site_mode_ << '\n';
		ofs << "two_layer_gate_tanh_amplitude = "
		    << two_layer_gate_tanh_amplitude_ << '\n';
		if (TwoLayerGateUsesSharedRadial()) {
			const int gate_radial_count = TwoLayerGateRadialCoeffCount();
			if (gate_radial_count <= 0)
				ERROR("SUS2-SH two-layer shared-radial gate has no radial coefficients");
			ofs << "two_layer_gate_radial_mode = shared-radial\n";
			ofs << "two_layer_gate_radial_coeff_count = " << gate_radial_count << '\n';
			ofs << "two_layer_gate_radial_coeffs = {";
			const int R = p_RadialBasis->rb_size;
			for (int i = 0; i < gate_radial_count; ++i) {
				if (i > 0)
					ofs << ", ";
				const int mu = i / R;
				const int xi = i % R;
				ofs << TwoLayerGateRadialCoeff(mu, xi);
			}
			ofs << "}\n";
		}
		const int gate_additive_count = TwoLayerGateAdditiveCoeffCount();
		if (gate_additive_count <= 0)
			ERROR("SUS2-SH two-layer gate has no additive coefficients");
		ofs << "two_layer_gate_additive_coeff_count = "
		    << gate_additive_count << '\n';
		ofs << "two_layer_gate_additive_coeffs = {";
		for (int i = 0; i < gate_additive_count; ++i) {
			if (i > 0)
				ofs << ", ";
			ofs << TwoLayerGateAdditiveCoeff(i, 0);
		}
		ofs << "}\n";
		ofs << "two_layer_gate_weight_count = " << TwoLayerGateWeightCount() << '\n';
		ofs << "two_layer_gate_scalar_indices = {";
		for (int i = 0; i < static_cast<int>(two_layer_gate_scalar_indices_.size()); ++i) {
			if (i > 0)
				ofs << ", ";
			ofs << two_layer_gate_scalar_indices_[i];
		}
		ofs << "}\n";
		ofs << "two_layer_gate_weights = {";
		for (int i = 0; i < TwoLayerGateWeightCount(); ++i) {
			if (i > 0)
				ofs << ", ";
			ofs << TwoLayerGateWeight(i);
		}
		ofs << "}\n";
			if (TwoLayerGateUsesBodyLinearCombo()) {
				ofs << "two_layer_gate_body_mix_weight_count = "
				    << TwoLayerGateBodyMixWeightCount() << '\n';
			ofs << "two_layer_gate_body_mix_weights = {";
			for (int i = 0; i < TwoLayerGateBodyMixWeightCount(); ++i) {
				if (i > 0)
					ofs << ", ";
				ofs << TwoLayerGateBodyMixWeight(i);
				}
				ofs << "}\n";
			}
			if (TwoLayerGateEdgeL1Enabled()) {
				if (two_layer_gate_site_mode_ != "neighbor")
					ERROR("SUS2-SH edge L1 gate first version supports neighbor site mode only");
				if (static_cast<int>(two_layer_gate_edge_l1_weights_.size())
				    != TwoLayerGateEdgeL1WeightCount())
					ERROR("SUS2-SH edge L1 gate metadata has inconsistent sizes");
				ofs << "two_layer_gate_edge_l1_enabled = true\n";
				ofs << "two_layer_gate_edge_l1_source_moment_count = "
				    << TwoLayerGateEdgeL1SourceCount() << '\n';
				ofs << "two_layer_gate_edge_l1_source_moment_indices = {";
				for (int i = 0; i < TwoLayerGateEdgeL1SourceCount(); ++i) {
					if (i > 0)
						ofs << ", ";
					ofs << TwoLayerGateEdgeL1SourceMomentIndex(i);
				}
				ofs << "}\n";
				ofs << "two_layer_gate_edge_l1_weight_count = "
				    << TwoLayerGateEdgeL1WeightCount() << '\n';
				ofs << "two_layer_gate_edge_l1_weights = {";
				for (int mu = 0; mu < radial_func_count; ++mu) {
					for (int source = 0; source < TwoLayerGateEdgeL1SourceCount(); ++source) {
						const int flat = mu * TwoLayerGateEdgeL1SourceCount() + source;
						if (flat > 0)
							ofs << ", ";
						ofs << TwoLayerGateEdgeL1Weight(mu, source);
					}
				}
				ofs << "}\n";
			}
			if (two_layer_residual_enabled_) {
				const int e0_count = TwoLayerResidualE0CoeffCount();
			if (e0_count != alpha_scalar_moments)
				ERROR("SUS2-SH residual E0 coefficient count is inconsistent");
			ofs << "two_layer_residual_e0_coeff_count = " << e0_count << '\n';
			ofs << "two_layer_residual_e0_coeffs = {";
			for (int i = 0; i < e0_count; ++i) {
				if (i > 0)
					ofs << ", ";
				ofs << TwoLayerResidualE0Coeff(i);
			}
			ofs << "}\n";
		}
	}

	ofs << "species_coeffs = {";
	for (int i = 0; i < species_count; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i];
	}
	ofs << '}' << endl;

	ofs << "moment_coeffs = {";
	for (int i = 0; i < alpha_count - 1; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i + species_count];
	}
	ofs << '}';

	ofs.close();
}


void MLMTPR::Save_2(const string& filename)
{
	if (is_sh_potential_)
		ERROR("SUS2-SH models should be saved with Save(), not Save_2().");
        ofstream ofs(filename);
        ofs.setf(ios::scientific);
        ofs.precision(15);

        ofs << "MTP\n";
        ofs << "version = 1.1.0\n";
        ofs << "potential_name = " << pot_desc << endl;
        if(scaling != 1.0)
                ofs << "scaling = " << scaling << endl;
        ofs << "species_count = " << species_count << endl;
        ofs << "potential_tag = " << "" << endl;
        p_RadialBasis->WriteRadialBasis(ofs);
        ofs << "\tradial_funcs_count = " << radial_func_count << '\n';

        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;
  

        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i+species_count];
        }
        ofs << '}' << endl;


        ofs << "\tradial_coeffs" << '\n';

        int q = species_count*species_count;

        for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                {
                        ofs <<"\t\t"<< i << "-" << j << "\n";

                        for (int k = 0; k < radial_func_count; k++)
                        {
                                ofs << "\t\t\t{";
                                for (int l = 0; l < p_RadialBasis->rb_size ; l++)
                                {
                                        ofs << regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+l]*regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+i]
                                               *regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+j] ;
                                        if (l != p_RadialBasis->rb_size - 1)
                                                ofs << ", ";
                                        else
                                                ofs << "}" << endl;
                                }
                        }
                }

        ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
        ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';

        ofs << "alpha_index_basic = {";
        for (int i = 0; i < alpha_index_basic_count; i++)
        {
                ofs << '{'
                        << alpha_index_basic[i][0] << ", "
                        << alpha_index_basic[i][1] << ", "
                        << alpha_index_basic[i][2] << ", "
                        << alpha_index_basic[i][3] << "}";
                if (i < alpha_index_basic_count - 1)
                        ofs << ", ";
        }
        ofs << "}\n";

        ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';

        ofs << "alpha_index_times = {";
        for (int i = 0; i < alpha_index_times_count; i++)
        {
                ofs << '{'
                        << alpha_index_times[i][0] << ", "
                        << alpha_index_times[i][1] << ", "
                        << alpha_index_times[i][2] << ", "
                        << alpha_index_times[i][3] << "}";
                if (i < alpha_index_times_count - 1)
                        ofs << ", ";
        }
        ofs << "}\n";

        ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

        ofs << "alpha_moment_mapping = {";
        for (int i = 0; i < alpha_scalar_moments; i++) {
                if (i > 0)
                        ofs << ", ";
                ofs << alpha_moment_mapping[i];
        }
        ofs << "}\n";

        ofs << "species_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i];
        }
        ofs << '}' << endl;

        ofs << "moment_coeffs = {";
        for (int i = 0; i < alpha_count - 1; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i + species_count];
        }
        ofs << '}';

	ofs.close();
}

bool MLMTPR::HasNonzeroTwoLayerGateWeights() const
{
	if (!two_layer_gate_enabled_)
		return false;
	for (int i = 0; i < TwoLayerGateWeightCount(); ++i)
		if (std::abs(TwoLayerGateWeight(i)) > 0.0)
			return true;
	return false;
}

bool MLMTPR::HasNonzeroTwoLayerGateEdgeL1Weights() const
{
	if (!TwoLayerGateEdgeL1Enabled())
		return false;
	const int weight_count = TwoLayerGateEdgeL1WeightCount();
	const double* weights = TwoLayerGateEdgeL1WeightData();
	if (weights == nullptr)
		return false;
	for (int i = 0; i < weight_count; ++i) {
		if (std::abs(weights[i]) > 0.0)
			return true;
	}
	return false;
}

bool MLMTPR::RequiresTwoLayerGateEvaluation() const
{
	return two_layer_gate_enabled_
	    && (HasNonzeroTwoLayerGateWeights()
	        || HasNonzeroTwoLayerGateEdgeL1Weights()
	        || TwoLayerResidualEnabled());
}

void MLMTPR::PrepareTwoLayerGateValues(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	if (static_cast<int>(two_layer_gate_weights_.size()) != TwoLayerGateWeightCount())
		ERROR("SUS2-SH two-layer gate metadata has inconsistent sizes");
	if (static_cast<int>(two_layer_gate_scalar_body_orders_.size())
	    != static_cast<int>(two_layer_gate_scalar_indices_.size())
	    || static_cast<int>(two_layer_gate_mu_body_orders_.size()) != radial_func_count)
		BuildTwoLayerGateBodyOrderBuckets();
		const bool body_linear_combo = TwoLayerGateUsesBodyLinearCombo();
		const bool edge_l1_active = TwoLayerGateEdgeL1Enabled();
		size_t edge_l1_edge_use_count = 0;
		if (edge_l1_active)
			for (int ind = 0; ind < neighborhoods.size(); ++ind)
				edge_l1_edge_use_count +=
					static_cast<size_t>(neighborhoods[ind].count);
		const int gate_count = TwoLayerGateScalarCount();
		if (gate_count == 0) {
			two_layer_gate_values_.assign(
			static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
		two_layer_gate_scalar_values_cache_.clear();
		two_layer_gate_body_values_cache_.clear();
		two_layer_gate_moment_values_cache_.clear();
		if (edge_l1_active) {
			two_layer_gate_edge_l1_values_.assign(
				static_cast<size_t>(cfg.size())
				* TwoLayerGateEdgeL1SourceCount() * 3, 0.0);
			for (int ind = 0; ind < cfg.size(); ++ind) {
				CalcTwoLayerGateEdgeL1ValuesOnly(
					neighborhoods[ind],
					two_layer_gate_edge_l1_value_buffer_,
					ind);
				double* dst = two_layer_gate_edge_l1_values_.data()
					+ static_cast<size_t>(ind)
						* TwoLayerGateEdgeL1SourceCount() * 3;
					std::copy(two_layer_gate_edge_l1_value_buffer_.begin(),
					          two_layer_gate_edge_l1_value_buffer_.end(),
					          dst);
				}
				PrepareTwoLayerGateEdgeL1WeightedValues(
					cfg.size(), edge_l1_edge_use_count);
				PrepareTwoLayerEdgeGateTypeScaleCache(neighborhoods);
			} else {
				two_layer_gate_edge_l1_values_.clear();
				two_layer_gate_edge_l1_weighted_values_.clear();
				two_layer_gate_edge_l1_weights_by_source_.clear();
				PrepareTwoLayerEdgeGateTypeScaleCache(neighborhoods);
			}
			two_layer_gate_values_from_edge_cache_ready_ = false;
			InvalidateTwoLayerGateTanhMuCache();
		return;
	}
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();
	const int cached_moment_count =
		static_cast<int>(two_layer_gate_required_moment_indices_.size());
	const size_t expected_scalar_size =
		static_cast<size_t>(cfg.size()) * gate_count;
	const size_t expected_body_size =
		body_linear_combo
			? static_cast<size_t>(cfg.size()) * TwoLayerGateBodyOrderCount()
			: 0;
	const size_t expected_moment_size =
		static_cast<size_t>(cfg.size()) * cached_moment_count;
	const size_t expected_gate_value_size =
		static_cast<size_t>(cfg.size()) * radial_func_count;
	const size_t expected_edge_l1_size =
		edge_l1_active
			? static_cast<size_t>(cfg.size())
				* TwoLayerGateEdgeL1SourceCount() * 3
			: 0;
	if (two_layer_gate_values_from_edge_cache_ready_
	    && two_layer_gate_values_.size() == expected_gate_value_size
	    && two_layer_gate_scalar_values_cache_.size() == expected_scalar_size
		    && two_layer_gate_body_values_cache_.size() == expected_body_size
		    && two_layer_gate_moment_values_cache_.size() == expected_moment_size
		    && two_layer_gate_edge_l1_values_.size() == expected_edge_l1_size) {
			if (edge_l1_active)
				PrepareTwoLayerGateEdgeL1WeightedValues(
					cfg.size(), edge_l1_edge_use_count);
			else
				two_layer_gate_edge_l1_weighted_values_.clear();
			PrepareTwoLayerEdgeGateTypeScaleCache(neighborhoods);
			InvalidateTwoLayerGateTanhMuCache();
			return;
		}
	two_layer_gate_values_from_edge_cache_ready_ = false;
	InvalidateTwoLayerGateTanhMuCache();
	two_layer_gate_values_.assign(expected_gate_value_size, 0.0);
	two_layer_gate_scalar_values_cache_.resize(expected_scalar_size);
	if (body_linear_combo)
		two_layer_gate_body_values_cache_.resize(expected_body_size);
	else
		two_layer_gate_body_values_cache_.clear();
	two_layer_gate_moment_values_cache_.resize(expected_moment_size);
		if (edge_l1_active)
			two_layer_gate_edge_l1_values_.assign(expected_edge_l1_size, 0.0);
		else {
			two_layer_gate_edge_l1_values_.clear();
			two_layer_gate_edge_l1_weighted_values_.clear();
			two_layer_gate_edge_l1_weights_by_source_.clear();
		}

	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	active_two_layer_gate_values_ = nullptr;
	for (int ind = 0; ind < cfg.size(); ++ind) {
		CalcTwoLayerGateScalarValuesOnly(neighborhoods[ind],
		                                  sh_gate_scalar_values_,
		                                  ind);
		double* cached_scalars = two_layer_gate_scalar_values_cache_.data()
			+ static_cast<size_t>(ind) * gate_count;
		double* cached_moments = two_layer_gate_moment_values_cache_.data()
			+ static_cast<size_t>(ind) * cached_moment_count;
		for (int i = 0; i < cached_moment_count; ++i)
			cached_moments[i] =
				moment_vals[two_layer_gate_required_moment_indices_[i]];
		for (int q = 0; q < gate_count; ++q) {
			const int scalar_index = two_layer_gate_scalar_indices_[q];
			if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
				ERROR("SUS2-SH two-layer gate scalar index is out of range");
			cached_scalars[q] = sh_gate_scalar_values_[q];
		}
		double* cached_gate_values = two_layer_gate_values_.data()
			+ static_cast<size_t>(ind) * radial_func_count;
		if (body_linear_combo) {
			double* cached_body_values = two_layer_gate_body_values_cache_.data()
				+ static_cast<size_t>(ind) * TwoLayerGateBodyOrderCount();
			ComputeTwoLayerGateBodySignals(cached_scalars, cached_body_values);
			ComputeTwoLayerGateMuSignals(cached_body_values, cached_gate_values);
		} else if (TwoLayerGateUsesFullScalarWeights()) {
			(void)cached_gate_values;
		} else {
			ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
		}
		if (edge_l1_active) {
			CalcTwoLayerGateEdgeL1ValuesOnly(
				neighborhoods[ind],
				two_layer_gate_edge_l1_value_buffer_,
				ind);
			double* edge_l1_values = two_layer_gate_edge_l1_values_.data()
				+ static_cast<size_t>(ind)
					* TwoLayerGateEdgeL1SourceCount() * 3;
			std::copy(two_layer_gate_edge_l1_value_buffer_.begin(),
			          two_layer_gate_edge_l1_value_buffer_.end(),
			          edge_l1_values);
		}
		}
		if (TwoLayerGateUsesFullScalarWeights())
			ComputeTwoLayerGateFullMuSignalsForAtoms(cfg.size());
		if (edge_l1_active)
			PrepareTwoLayerGateEdgeL1WeightedValues(
				cfg.size(), edge_l1_edge_use_count);
		else
			two_layer_gate_edge_l1_weighted_values_.clear();
		if (!edge_l1_active)
			two_layer_gate_edge_l1_weights_by_source_.clear();
		PrepareTwoLayerEdgeGateTypeScaleCache(neighborhoods);
		active_two_layer_gate_values_ = saved_gate_values;
	}

	void MLMTPR::PrepareTwoLayerGateEdgeL1WeightedValues(
		int atom_count,
		size_t edge_use_count)
	{
	if (!HasNonzeroTwoLayerGateEdgeL1Weights()) {
		two_layer_gate_edge_l1_weighted_values_.clear();
		two_layer_gate_edge_l1_weights_by_source_.clear();
		return;
	}
	const int source_count = TwoLayerGateEdgeL1SourceCount();
		if (atom_count <= 0 || source_count <= 0 || radial_func_count <= 0) {
			two_layer_gate_edge_l1_weighted_values_.clear();
			two_layer_gate_edge_l1_weights_by_source_.clear();
			return;
		}
		const double* edge_l1_weights = TwoLayerGateEdgeL1WeightData();
		const size_t expected_h_size =
			static_cast<size_t>(atom_count) * source_count * 3;
		if (two_layer_gate_edge_l1_values_.size() != expected_h_size)
			ERROR("SUS2-SH edge L1 gate value cache has inconsistent size");
		two_layer_gate_edge_l1_weights_by_source_.resize(
			static_cast<size_t>(source_count) * radial_func_count);
		for (int source = 0; source < source_count; ++source) {
			double* weight_col =
				two_layer_gate_edge_l1_weights_by_source_.data()
				+ static_cast<size_t>(source) * radial_func_count;
			for (int mu = 0; mu < radial_func_count; ++mu)
				weight_col[mu] =
					edge_l1_weights[static_cast<size_t>(mu) * source_count + source];
		}

		two_layer_gate_edge_l1_weighted_values_.assign(
			static_cast<size_t>(atom_count) * radial_func_count * 3, 0.0);
		for (int atom = 0; atom < atom_count; ++atom) {
			const double* h_atom = two_layer_gate_edge_l1_values_.data()
				+ static_cast<size_t>(atom) * source_count * 3;
				double* u_atom = two_layer_gate_edge_l1_weighted_values_.data()
					+ static_cast<size_t>(atom) * radial_func_count * 3;
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double* weight_row =
						edge_l1_weights + static_cast<size_t>(mu) * source_count;
					double u0 = 0.0;
					double u1 = 0.0;
					double u2 = 0.0;
					for (int source = 0; source < source_count; ++source) {
						const double w = weight_row[source];
						const double* h = h_atom + source * 3;
						u0 += w * h[0];
						u1 += w * h[1];
						u2 += w * h[2];
				}
				double* u = u_atom + mu * 3;
				u[0] = u0;
				u[1] = u1;
				u[2] = u2;
			}
		}

		if (EdgeL1CacheProfileEnabledOnRank0()) {
			std::cout << "[SUS2_SH_EDGE_L1_CACHE_PROFILE]"
			          << " weighted_cache_atoms=" << atom_count
			          << " weighted_cache_sources=" << source_count
			          << " weighted_cache_mu=" << radial_func_count
			          << " weighted_cache_edge_uses=" << edge_use_count
			          << std::endl;
		}
	}

const double* MLMTPR::TwoLayerGateNeighborSignals(const Neighborhood& nbh, int neighbor_index) const
{
	if (active_two_layer_gate_values_ == nullptr)
		return nullptr;
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return nullptr;
	const int atom_index = nbh.inds[neighbor_index];
	return TwoLayerGateAtomSignals(atom_index);
}

const double* MLMTPR::PrepareTwoLayerGateEdgeL1NeighborSignals(
	const Neighborhood& nbh,
	int neighbor_index,
	const double* base_signal_by_mu,
	const double* edge_sh_values,
	int& neighbor_atom_index_for_tanh)
{
	if (!HasNonzeroTwoLayerGateEdgeL1Weights())
		return base_signal_by_mu;
	if (active_two_layer_gate_values_ == nullptr)
		return base_signal_by_mu;
	if (edge_sh_values == nullptr)
		return base_signal_by_mu;
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return base_signal_by_mu;

	const int source_count = TwoLayerGateEdgeL1SourceCount();
	if (source_count <= 0)
		return base_signal_by_mu;
	const int atom_index = nbh.inds[neighbor_index];
	if (atom_index < 0)
		return base_signal_by_mu;
	if (radial_func_count <= 0
	    || active_two_layer_gate_values_->size() % radial_func_count != 0)
		ERROR("SUS2-SH two-layer gate mu signal cache has inconsistent size");
	const size_t expected_size =
		(active_two_layer_gate_values_->size() / radial_func_count)
		* static_cast<size_t>(source_count) * 3;
	if (two_layer_gate_edge_l1_values_.size() != expected_size)
		ERROR("SUS2-SH edge L1 gate value cache has inconsistent size");
	const int atom_count =
		static_cast<int>(active_two_layer_gate_values_->size() / radial_func_count);
	if (atom_index >= atom_count)
		ERROR("SUS2-SH edge L1 gate neighbor index is out of range");

		if (static_cast<int>(two_layer_gate_edge_l1_signal_buffer_.size())
		    != radial_func_count)
			two_layer_gate_edge_l1_signal_buffer_.resize(radial_func_count);

		for (int mu = 0; mu < radial_func_count; ++mu)
			two_layer_gate_edge_l1_signal_buffer_[mu] =
				base_signal_by_mu == nullptr ? 0.0 : base_signal_by_mu[mu];

			const double y1[3] = {
				edge_sh_values[1],
				edge_sh_values[2],
				edge_sh_values[3]
			};
			const double* edge_l1_weights = TwoLayerGateEdgeL1WeightData();
			const size_t expected_weighted_size =
				static_cast<size_t>(atom_count) * radial_func_count * 3;
			if (two_layer_gate_edge_l1_weighted_values_.size()
			    == expected_weighted_size) {
			const double* weighted_values =
				two_layer_gate_edge_l1_weighted_values_.data()
				+ static_cast<size_t>(atom_index) * radial_func_count * 3;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double* u = weighted_values + mu * 3;
				two_layer_gate_edge_l1_signal_buffer_[mu] +=
					u[0] * y1[0] + u[1] * y1[1] + u[2] * y1[2];
			}
		} else {
			if (static_cast<int>(two_layer_gate_edge_l1_chi_buffer_.size())
			    != source_count)
				two_layer_gate_edge_l1_chi_buffer_.resize(source_count);
			const double* edge_l1_values =
				two_layer_gate_edge_l1_values_.data()
				+ static_cast<size_t>(atom_index) * source_count * 3;
			for (int source = 0; source < source_count; ++source) {
				const double* h = edge_l1_values + source * 3;
				two_layer_gate_edge_l1_chi_buffer_[source] =
					h[0] * y1[0] + h[1] * y1[1] + h[2] * y1[2];
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double* weight_row =
						edge_l1_weights + static_cast<size_t>(mu) * source_count;
					double extra = 0.0;
					for (int source = 0; source < source_count; ++source)
						extra +=
							weight_row[source]
							* two_layer_gate_edge_l1_chi_buffer_[source];
					two_layer_gate_edge_l1_signal_buffer_[mu] += extra;
				}
			}
		neighbor_atom_index_for_tanh = -1;
		return two_layer_gate_edge_l1_signal_buffer_.data();
	}

void MLMTPR::AccumulateTwoLayerGateEdgeL1WeightGradForPair(
	const Neighborhood& nbh,
	int neighbor_index,
	const double* edge_sh_values,
	const double* gate_adjoint_by_mu,
	std::vector<double>& out_grad_accumulator)
{
	if (!TwoLayerGateEdgeL1Enabled())
		return;
	if (edge_sh_values == nullptr || gate_adjoint_by_mu == nullptr)
		return;
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return;
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	if (source_count <= 0)
		return;
	const int atom_index = nbh.inds[neighbor_index];
	if (atom_index < 0)
		return;
	const size_t expected_size =
		static_cast<size_t>(source_count) * 3
		* (two_layer_gate_values_.empty()
			? 0
			: two_layer_gate_values_.size() / radial_func_count);
	if (two_layer_gate_edge_l1_values_.size() != expected_size)
		ERROR("SUS2-SH edge L1 gate value cache has inconsistent size");
	const int atom_count =
		static_cast<int>(two_layer_gate_edge_l1_values_.size()
			/ (static_cast<size_t>(source_count) * 3));
	if (atom_index >= atom_count)
		ERROR("SUS2-SH edge L1 gate neighbor index is out of range");
		if (static_cast<int>(out_grad_accumulator.size()) < CoeffCount())
			out_grad_accumulator.resize(CoeffCount());

		const double* edge_l1_values = two_layer_gate_edge_l1_values_.data()
			+ static_cast<size_t>(atom_index) * source_count * 3;
		if (static_cast<int>(two_layer_gate_edge_l1_chi_buffer_.size())
		    != source_count)
			two_layer_gate_edge_l1_chi_buffer_.resize(source_count);
		const double y1[3] = {
			edge_sh_values[1],
			edge_sh_values[2],
			edge_sh_values[3]
		};
		for (int source = 0; source < source_count; ++source) {
			const double* h = edge_l1_values + source * 3;
			two_layer_gate_edge_l1_chi_buffer_[source] =
				h[0] * y1[0] + h[1] * y1[1] + h[2] * y1[2];
		}

		double* grad_edge_l1 =
			out_grad_accumulator.data() + TwoLayerGateEdgeL1WeightOffset();
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double adjoint = gate_adjoint_by_mu[mu];
			if (adjoint == 0.0)
				continue;
			double* grad_row = grad_edge_l1 + mu * source_count;
			for (int source = 0; source < source_count; ++source) {
				const double chi = two_layer_gate_edge_l1_chi_buffer_[source];
				if (chi == 0.0)
					continue;
				grad_row[source] += adjoint * chi;
			}
		}
	}

void MLMTPR::AccumulateTwoLayerGateEdgeL1WeightCoordGradForPair(
	const Neighborhood& nbh,
	int neighbor_index,
	const double* edge_sh_values,
	const double* edge_sh_ders,
	const double* energy_gate_adjoint_by_mu,
	const Vector3* edge_der_weight,
	std::vector<double>& out_grad_accumulator)
{
	if (!TwoLayerGateEdgeL1Enabled())
		return;
	if (edge_sh_values == nullptr || energy_gate_adjoint_by_mu == nullptr)
		return;
	const int source_count = TwoLayerGateEdgeL1SourceCount();
	if (source_count <= 0)
		return;
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return;

	ComputeTwoLayerGateEdgeL1PairCoordTangents(
		nbh,
		neighbor_index,
		edge_sh_values,
		edge_sh_ders,
		edge_der_weight,
		two_layer_gate_edge_l1_chi_adj_buffer_);

	bool has_tangent = false;
	for (double tangent : two_layer_gate_edge_l1_chi_adj_buffer_)
		if (tangent != 0.0) {
			has_tangent = true;
			break;
		}
	if (!has_tangent)
		return;
		if (static_cast<int>(out_grad_accumulator.size()) < CoeffCount())
			out_grad_accumulator.resize(CoeffCount());

		double* grad_edge_l1 =
			out_grad_accumulator.data() + TwoLayerGateEdgeL1WeightOffset();
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double adjoint = energy_gate_adjoint_by_mu[mu];
			if (adjoint == 0.0)
				continue;
			double* grad_row = grad_edge_l1 + mu * source_count;
			for (int source = 0; source < source_count; ++source) {
				const double tangent =
					two_layer_gate_edge_l1_chi_adj_buffer_[source];
				if (tangent == 0.0)
					continue;
				grad_row[source] += adjoint * tangent;
			}
		}
	}

const double* MLMTPR::TwoLayerGateAtomSignals(int atom_index) const
{
	if (active_two_layer_gate_values_ == nullptr)
		return nullptr;
	if (atom_index < 0)
		return nullptr;
	if (radial_func_count <= 0
	    || active_two_layer_gate_values_->size() % radial_func_count != 0)
		ERROR("SUS2-SH two-layer gate mu signal cache has inconsistent size");
	const int atom_count =
		static_cast<int>(active_two_layer_gate_values_->size() / radial_func_count);
	if (atom_index < 0 || atom_index >= atom_count)
		ERROR("SUS2-SH two-layer gate neighbor index is out of range");
	return active_two_layer_gate_values_->data()
		+ static_cast<size_t>(atom_index) * radial_func_count;
}

double MLMTPR::TwoLayerGateNeighborSignal(const Neighborhood& nbh, int neighbor_index, int mu) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate mu index is out of range");
	const double* signals = TwoLayerGateNeighborSignals(nbh, neighbor_index);
	if (signals == nullptr)
		return 0.0;
	return signals[mu];
}

void MLMTPR::AddTwoLayerGateMuAdjoint(const Neighborhood& nbh, int neighbor_index, int mu, double adjoint)
{
	if (active_two_layer_gate_adjoints_ == nullptr || adjoint == 0.0)
		return;
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate adjoint mu index is out of range");
	if (neighbor_index < 0 || neighbor_index >= nbh.count)
		return;
	const int atom_index = nbh.inds[neighbor_index];
	AddTwoLayerGateAtomMuAdjoint(atom_index, mu, adjoint);
}

void MLMTPR::AddTwoLayerGateAtomMuAdjoint(int atom_index, int mu, double adjoint)
{
	if (active_two_layer_gate_adjoints_ == nullptr || adjoint == 0.0)
		return;
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate adjoint mu index is out of range");
	if (atom_index < 0)
		return;
	if (radial_func_count <= 0
	    || active_two_layer_gate_adjoints_->size() % radial_func_count != 0)
		ERROR("SUS2-SH two-layer gate adjoint cache has inconsistent size");
	const int atom_count =
		static_cast<int>(active_two_layer_gate_adjoints_->size() / radial_func_count);
	if (atom_index < 0 || atom_index >= atom_count)
		ERROR("SUS2-SH two-layer gate adjoint neighbor index is out of range");
	(*active_two_layer_gate_adjoints_)[
		static_cast<size_t>(atom_index) * radial_func_count + mu] += adjoint;
}

void MLMTPR::AccumulateTwoLayerGateForceChain(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	if (!HasNonzeroTwoLayerGateWeights())
		return;
	if (two_layer_gate_adjoints_.size()
	    != static_cast<size_t>(cfg.size()) * radial_func_count)
		ERROR("SUS2-SH two-layer gate adjoint cache has inconsistent size");

	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;

	std::vector<double> scalar_seeds(TwoLayerGateScalarCount(), 0.0);
	std::vector<double> body_seed_scratch;
	std::vector<double> full_scalar_seed_cache;
	std::vector<double> body_combo_seed_cache;
	const int gate_count = TwoLayerGateScalarCount();
	const int body_count = TwoLayerGateBodyOrderCount();
	const bool full_seed_blas =
		TwoLayerGateUsesFullScalarWeights()
		&& gate_count > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const bool body_combo_seed_blas =
		TwoLayerGateUsesBodyLinearCombo()
		&& gate_count > 0
		&& body_count > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count;
	const double* body_combo_scalar_weights =
		body_combo_seed_blas ? TwoLayerGateScalarWeightData() : nullptr;
	if (full_seed_blas) {
		full_scalar_seed_cache.assign(
			static_cast<size_t>(cfg.size()) * gate_count, 0.0);
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
		            full_scalar_seed_cache.data(),
		            gate_count);
	}
	if (body_combo_seed_blas) {
		body_combo_seed_cache.assign(
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
		            body_combo_seed_cache.data(),
		            body_count);
	}
	for (int ind = 0; ind < cfg.size(); ++ind) {
		const double* gate_adjoints = two_layer_gate_adjoints_.data()
			+ static_cast<size_t>(ind) * radial_func_count;
		bool has_adjoint = false;
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double adjoint = gate_adjoints[mu];
			if (adjoint == 0.0)
				continue;
			has_adjoint = true;
		}
		if (!has_adjoint)
			continue;
		const double* scalar_seed_values = scalar_seeds.data();
		if (full_seed_blas) {
			scalar_seed_values = full_scalar_seed_cache.data()
				+ static_cast<size_t>(ind) * gate_count;
		} else if (body_combo_seed_blas) {
			const double* body_seeds = body_combo_seed_cache.data()
				+ static_cast<size_t>(ind) * body_count;
			for (int q = 0; q < gate_count; ++q) {
				const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
				scalar_seeds[q] =
					body_seeds[body_index] * body_combo_scalar_weights[q];
			}
		} else {
			AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
				gate_adjoints, scalar_seeds.data(), body_seed_scratch);
		}
		const Neighborhood& nbh = neighborhoods[ind];
		CalcTwoLayerGateWeightedScalarDersForScalarSeeds(
			nbh, sh_gate_component_ders_, ind, scalar_seed_values);
		for (int j = 0; j < nbh.count; ++j) {
			const Vector3& gate_der = sh_gate_component_ders_[j];
			cfg.force(ind) += gate_der;
			cfg.force(nbh.inds[j]) -= gate_der;
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b)
					cfg.stresses[a][b] -= gate_der[a] * nbh.vecs[j][b];
		}
	}

	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
}

void MLMTPR::CalcEFS(Configuration& cfg)
{
	Neighborhoods neighborhoods(cfg, CutOff());
	CalcEFS(cfg, neighborhoods);
}

void MLMTPR::CalcEFS(Configuration& cfg, const Neighborhoods& neighborhoods)
{
			if (!RequiresTwoLayerGateEvaluation()) {
				two_layer_full_edge_cache_for_next_calc_ = false;
				two_layer_reuse_full_edge_cache_once_ = false;
				two_layer_forward_final_moment_cache_ready_ = false;
				active_two_layer_gate_values_ = nullptr;
				active_two_layer_gate_adjoints_ = nullptr;
				active_two_layer_edge_cache_atom_index_ = -1;

				ResetEFS(cfg);
				PrepareEvalCaches();
				cfg.has_energy(true);
				cfg.has_forces(true);
				cfg.has_stresses(true);
				cfg.has_site_energies(true);
				cfg.cal_se.resize(cfg.size());
				cfg.cal_se0.resize(cfg.size());
				cfg.type_mean.resize(cfg.unique_elems.size());
				FillWithZero(cfg.type_mean);

				for (int ind = 0; ind < cfg.size(); ++ind) {
					const Neighborhood& nbh = neighborhoods[ind];
					if (ind < 2)
					buff_site_energy_0 = 0.0;
					buff_site_energy_ = 0.0;
					buff_site_energy_ders_.resize(nbh.count);
					MLMTPR::CalcSiteEnergyDers(nbh);
					if (ind < 2)
					cfg.cal_se[ind] = buff_site_energy_;
					cfg.cal_se0[ind] = buff_site_energy_0;
					cfg.energy += buff_site_energy_;

					for (int j = 0; j < nbh.count; ++j) {
						cfg.force(ind) += buff_site_energy_ders_[j];
						cfg.force(nbh.inds[j]) -= buff_site_energy_ders_[j];
					}
					for (int j = 0; j < nbh.count; ++j)
						for (int a = 0; a < 3; ++a)
							for (int b = 0; b < 3; ++b)
								cfg.stresses[a][b] -=
									buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
				}
				if (HasZBL() && ZBLEvaluationEnabled()) {
					if (neighborhoods.cutoff + 1.0e-12 >= ZBL().MaxOuterCutoff())
						ZBL().AddTo(cfg, neighborhoods);
					else
						ZBL().AddTo(cfg);
				}
				return;
			}

	ResetEFS(cfg);
	PrepareEvalCaches();
		const bool build_full_edge_cache =
			two_layer_full_edge_cache_for_next_calc_;
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
			const bool profile_two_layer = TwoLayerProfileEnabledOnRank0();
			const double profile_start = profile_two_layer ? TwoLayerProfileNow() : 0.0;
			BuildTwoLayerEdgePrimitiveCache(neighborhoods, true, build_full_edge_cache);
			PrepareTwoLayerGateValues(cfg, neighborhoods);
			const bool has_nonzero_edge_l1_weights =
				HasNonzeroTwoLayerGateEdgeL1Weights();
		const double profile_after_prepare =
			profile_two_layer ? TwoLayerProfileNow() : 0.0;
				two_layer_gate_adjoints_.assign(
					static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
				if (has_nonzero_edge_l1_weights)
					two_layer_gate_edge_l1_adjoints_.assign(
						static_cast<size_t>(cfg.size())
						* TwoLayerGateEdgeL1SourceCount() * 3, 0.0);
			else
				two_layer_gate_edge_l1_adjoints_.clear();
			active_two_layer_gate_values_ = &two_layer_gate_values_;
			active_two_layer_gate_adjoints_ = &two_layer_gate_adjoints_;
		const bool cache_forward_final_moments =
			build_full_edge_cache && alpha_moments_count > 0;
		if (cache_forward_final_moments) {
			two_layer_final_moment_values_cache_.resize(
				static_cast<size_t>(cfg.size()) * alpha_moments_count);
			two_layer_final_moment_ders_cache_.resize(
				static_cast<size_t>(cfg.size()) * alpha_moments_count);
			two_layer_forward_energy_gate_adjoints_cache_.assign(
				static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
		}

	cfg.has_energy(true);
	cfg.has_forces(true);
	cfg.has_stresses(true);
	cfg.has_site_energies(true);
	cfg.cal_se.resize(cfg.size());
	cfg.cal_se0.resize(cfg.size());
	cfg.type_mean.resize(cfg.unique_elems.size());
	FillWithZero(cfg.type_mean);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];
		buff_site_energy_0 = 0;
		buff_site_energy_ = 0;
		buff_site_energy_ders_.resize(nbh.count);
			active_two_layer_edge_cache_atom_index_ = ind;
			CalcSiteEnergyDers(nbh);
			if (cache_forward_final_moments) {
				const size_t offset =
					static_cast<size_t>(ind) * alpha_moments_count;
				std::copy(moment_vals,
				          moment_vals + alpha_moments_count,
				          two_layer_final_moment_values_cache_.data() + offset);
				std::copy(site_energy_ders_wrt_moments_.begin(),
				          site_energy_ders_wrt_moments_.end(),
				          two_layer_final_moment_ders_cache_.data() + offset);
			}
			cfg.cal_se[ind] = buff_site_energy_;
			cfg.cal_se0[ind] = buff_site_energy_0;
			cfg.energy += buff_site_energy_;

		for (int j = 0; j < nbh.count; j++) {
			cfg.force(ind) += buff_site_energy_ders_[j];
			cfg.force(nbh.inds[j]) -= buff_site_energy_ders_[j];
		}

		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					cfg.stresses[a][b] -= buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
	}
		const double profile_after_main =
			profile_two_layer ? TwoLayerProfileNow() : 0.0;
		if (cache_forward_final_moments) {
			two_layer_forward_energy_gate_adjoints_cache_ =
				two_layer_gate_adjoints_;
			two_layer_forward_final_moment_cache_ready_ =
				cfg.size() > 0
				&& HasTwoLayerEdgePrimitiveCache(0, true, true)
				&& two_layer_final_moment_values_cache_.size()
					== static_cast<size_t>(cfg.size()) * alpha_moments_count
				&& two_layer_final_moment_ders_cache_.size()
					== static_cast<size_t>(cfg.size()) * alpha_moments_count
				&& two_layer_forward_energy_gate_adjoints_cache_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
		}
		active_two_layer_edge_cache_atom_index_ = -1;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		AccumulateTwoLayerGateForceChain(cfg, neighborhoods);
		AccumulateTwoLayerGateEdgeL1ForceChain(cfg, neighborhoods);
	if (profile_two_layer) {
		const double profile_end = TwoLayerProfileNow();
		RecordTwoLayerForwardProfile(
			cfg.size(),
			profile_end - profile_start,
			profile_after_prepare - profile_start,
			profile_after_main - profile_after_prepare,
			profile_end - profile_after_main);
	}
		two_layer_reuse_full_edge_cache_once_ =
			build_full_edge_cache && cfg.size() > 0
			&& HasTwoLayerEdgePrimitiveCache(0, true, true);
		if (HasZBL() && ZBLEvaluationEnabled()) {
			if (neighborhoods.cutoff + 1.0e-12 >= ZBL().MaxOuterCutoff())
				ZBL().AddTo(cfg, neighborhoods);
			else
				ZBL().AddTo(cfg);
			}
		}

double MLMTPR::CalcSiteEnergyValue(const Neighborhood& nbh,
                                   double* out_site_energy0)
{
	if (nbh.my_type >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	if (is_sh_potential_ && TwoLayerResidualEnabled()) {
		if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != alpha_scalar_moments)
			two_layer_residual_e0_coeffs_.assign(alpha_scalar_moments, 0.0);

		const bool saved_residual = two_layer_residual_enabled_;
		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
		const std::vector<double> saved_linear_coeffs = linear_coeffs;
		std::vector<double> saved_e0_coeffs(alpha_scalar_moments, 0.0);
		for (int i = 0; i < alpha_scalar_moments; ++i)
			saved_e0_coeffs[i] = TwoLayerResidualE0Coeff(i);

		two_layer_residual_enabled_ = false;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		for (int i = 0; i < alpha_scalar_moments; ++i)
			linear_coeffs[species_count + i] = saved_e0_coeffs[i];
		for (int i = 0; i < species_count; ++i)
			linear_coeffs[i] = 1.0;
		double e0_site_energy0 = 0.0;
		const double e0_energy =
			CalcSiteEnergyValue(nbh, &e0_site_energy0)
			+ saved_linear_coeffs[nbh.my_type] - 1.0;

		linear_coeffs = saved_linear_coeffs;
		for (int i = 0; i < species_count; ++i)
			linear_coeffs[i] = 1.0;
		active_two_layer_gate_values_ = saved_gate_values;
		active_two_layer_gate_adjoints_ = saved_gate_adjoints;
		double e1_site_energy0 = 0.0;
		const double e1_energy = CalcSiteEnergyValue(nbh, &e1_site_energy0);
		const double e1_one_body = regression_coeffs[nbh.my_type] + 1.0;

		two_layer_residual_enabled_ = saved_residual;
		active_two_layer_gate_values_ = saved_gate_values;
		active_two_layer_gate_adjoints_ = saved_gate_adjoints;
		linear_coeffs = saved_linear_coeffs;

		if (out_site_energy0 != nullptr)
			*out_site_energy0 = e1_site_energy0 + e0_site_energy0;
		return e1_energy + e0_energy - e1_one_body;
	}

	CalcBasisFuncs(nbh, basis_vals);
	double site_energy0 = 0.0;
	double site_energy = regression_coeffs[nbh.my_type] + linear_coeffs[nbh.my_type];
	const double center_linear = linear_coeffs[nbh.my_type];
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double scalar_value = basis_vals[1 + i];
		const double term = linear_coeffs[species_count + i]
		                  * linear_mults[i]
		                  * scalar_value
		                  * center_linear;
		site_energy += term;
		site_energy0 += term;
		max_linear[i] = std::max(max_linear[i],
		                         std::abs(linear_coeffs[species_count + i]
		                                  * scalar_value));
	}

	if (!is_sh_potential_
	    && p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls") {
		const double multiplier = 10000.0;
		for (int j = 0; j < nbh.count; ++j)
			if (nbh.dists[j] < p_RadialBasis->min_dist)
				site_energy += multiplier
					* (exp(-10 * (nbh.dists[j] - 1))
					   - exp(-10 * (p_RadialBasis->min_dist - 1)));
	}

	if (out_site_energy0 != nullptr)
		*out_site_energy0 = site_energy0;
	return site_energy;
}

void MLMTPR::CalcEnergyAndSiteEnergies(Configuration& cfg)
{
	Neighborhoods neighborhoods(cfg, CutOff());
	CalcEnergyAndSiteEnergies(cfg, neighborhoods);
}

void MLMTPR::CalcEnergyAndSiteEnergies(Configuration& cfg,
                                       const Neighborhoods& neighborhoods)
{
	cfg.energy = 0.0;
	memset(&cfg.stresses[0][0], 0, sizeof(Matrix3));
	cfg.has_energy(true);
	cfg.has_forces(false);
	cfg.has_stresses(false);
	cfg.has_site_energies(true);
	cfg.cal_se.resize(cfg.size());
	cfg.cal_se0.resize(cfg.size());
	cfg.type_mean.resize(cfg.unique_elems.size());
	FillWithZero(cfg.type_mean);

	PrepareEvalCaches();
	const bool use_two_layer_gate = RequiresTwoLayerGateEvaluation();
	if (!use_two_layer_gate) {
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		active_two_layer_edge_cache_atom_index_ = -1;
	} else {
		two_layer_full_edge_cache_for_next_calc_ = false;
		two_layer_reuse_full_edge_cache_once_ = false;
		two_layer_forward_final_moment_cache_ready_ = false;
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, false);
		PrepareTwoLayerGateValues(cfg, neighborhoods);
		active_two_layer_gate_values_ = &two_layer_gate_values_;
		active_two_layer_gate_adjoints_ = nullptr;
	}

	for (int ind = 0; ind < cfg.size(); ++ind) {
		if (use_two_layer_gate)
			active_two_layer_edge_cache_atom_index_ = ind;
		double site_energy0 = 0.0;
		const double site_energy =
			CalcSiteEnergyValue(neighborhoods[ind], &site_energy0);
		cfg.cal_se[ind] = site_energy;
		cfg.cal_se0[ind] = site_energy0;
		cfg.energy += site_energy;
	}

	active_two_layer_edge_cache_atom_index_ = -1;
	if (use_two_layer_gate)
		active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;

	if (HasZBL() && ZBLEvaluationEnabled()) {
		if (neighborhoods.cutoff + 1.0e-12 >= ZBL().MaxOuterCutoff())
			cfg.energy += ZBL().ComputeEnergy(cfg, neighborhoods);
		else
			cfg.energy += ZBL().ComputeEnergy(cfg);
	}
}

void MLMTPR::AccumulateEFSCombinationGrad(Configuration& cfg,
	                                          std::vector<double>& ene_weight,
	                                          const std::vector<Vector3>& frc_weights,
                                          const Matrix3& str_weights,
                                          Array1D& out_grads_accumulator)
{
	Neighborhoods neighborhoods(cfg, CutOff());
	AccumulateEFSCombinationGrad(cfg, ene_weight, frc_weights, str_weights,
	                             out_grads_accumulator, neighborhoods);
}

void MLMTPR::AccumulateEFSCombinationGrad(Configuration& cfg,
                                          std::vector<double>& ene_weight,
                                          const std::vector<Vector3>& frc_weights,
                                          const Matrix3& str_weights,
                                          Array1D& out_grads_accumulator,
                                          const Neighborhoods& neighborhoods)
{
	if (!is_sh_potential_ || !two_layer_gate_enabled_) {
		AnyLocalMLIP::AccumulateEFSCombinationGrad(cfg, ene_weight, frc_weights,
		                                           str_weights, out_grads_accumulator,
		                                           neighborhoods);
		return;
	}

	out_grads_accumulator.resize(CoeffCount());
	PrepareEvalCaches();
	const bool profile_two_layer = TwoLayerProfileEnabledOnRank0();
	const double profile_start = profile_two_layer ? TwoLayerProfileNow() : 0.0;
	const bool reuse_full_edge_cache =
		two_layer_reuse_full_edge_cache_once_
		&& neighborhoods.size() == two_layer_edge_cache_atom_count_
		&& neighborhoods.size() > 0
		&& HasTwoLayerEdgePrimitiveCache(0, true, true);
	two_layer_reuse_full_edge_cache_once_ = false;
	if (!reuse_full_edge_cache)
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, true);
	PrepareTwoLayerGateValues(cfg, neighborhoods);
	const double profile_after_prepare =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;
	two_layer_gate_adjoints_.assign(
		static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);

	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	const std::vector<double>* saved_edge_l1_coord_loss =
		active_two_layer_gate_edge_l1_coord_loss_;
	active_two_layer_gate_values_ = &two_layer_gate_values_;
	active_two_layer_gate_adjoints_ = &two_layer_gate_adjoints_;

	bool need_gate_chain_force_weight_grad = false;
	for (int ind = 0; ind < cfg.size() && !need_gate_chain_force_weight_grad; ++ind) {
		if (frc_weights[ind].NormSq() != 0.0)
			need_gate_chain_force_weight_grad = true;
	}
	if (!need_gate_chain_force_weight_grad)
		for (int a = 0; a < 3 && !need_gate_chain_force_weight_grad; ++a)
			for (int b = 0; b < 3; ++b)
				if (str_weights[a][b] != 0.0) {
					need_gate_chain_force_weight_grad = true;
					break;
				}
	if (need_gate_chain_force_weight_grad && TwoLayerGateEdgeL1Enabled()) {
		ComputeTwoLayerGateEdgeL1CoordLoss(
			neighborhoods,
			frc_weights,
			str_weights,
			two_layer_gate_edge_l1_coord_loss_);
		active_two_layer_gate_edge_l1_coord_loss_ =
			&two_layer_gate_edge_l1_coord_loss_;
	} else {
		two_layer_gate_edge_l1_coord_loss_.clear();
		active_two_layer_gate_edge_l1_coord_loss_ = nullptr;
	}

	std::vector<Vector3> se_ders_weights;
	for (int ind = 0; ind < cfg.size(); ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		se_ders_weights.resize(nbh.count);
		FillWithZero(se_ders_weights);
		for (int j = 0; j < nbh.count; ++j) {
			se_ders_weights[j] += frc_weights[ind];
			se_ders_weights[j] -= frc_weights[nbh.inds[j]];
		}
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b)
					se_ders_weights[j][a] += str_weights[a][b] * nbh.vecs[j][b];

		bool grad_zero = true;
		for (int j = 0; j < nbh.count && grad_zero; ++j)
			for (int a = 0; a < 3; ++a)
				if (se_ders_weights[j][a] != 0.0) {
					grad_zero = false;
					break;
				}

		active_two_layer_edge_cache_atom_index_ = ind;
		AccumulateCombinationGrad(nbh,
		                           out_grads_accumulator,
		                           ene_weight[ind],
		                           grad_zero ? nullptr : se_ders_weights.data());
	}
	const double profile_after_main_grad =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;
	active_two_layer_edge_cache_atom_index_ = -1;

	active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;

	std::vector<double> energy_gate_adjoints;
			if (need_gate_chain_force_weight_grad && TwoLayerGateWeightCount() > 0) {
			const size_t final_moment_cache_size =
				static_cast<size_t>(cfg.size()) * alpha_moments_count;
			const bool reuse_forward_final_moment_cache =
				reuse_full_edge_cache
				&& two_layer_forward_final_moment_cache_ready_
				&& two_layer_final_moment_values_cache_.size()
					== final_moment_cache_size
				&& two_layer_final_moment_ders_cache_.size()
					== final_moment_cache_size
				&& two_layer_forward_energy_gate_adjoints_cache_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
			if (reuse_forward_final_moment_cache) {
				energy_gate_adjoints =
					two_layer_forward_energy_gate_adjoints_cache_;
			} else {
				energy_gate_adjoints.assign(
					static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
				two_layer_final_moment_values_cache_.assign(
					final_moment_cache_size, 0.0);
				two_layer_final_moment_ders_cache_.assign(
					final_moment_cache_size, 0.0);
				active_two_layer_gate_values_ = &two_layer_gate_values_;
				active_two_layer_gate_adjoints_ = &energy_gate_adjoints;
				for (int ind = 0; ind < cfg.size(); ++ind) {
					active_two_layer_edge_cache_atom_index_ = ind;
					CalcSHSiteEnergyDers(neighborhoods[ind]);
					const size_t offset =
						static_cast<size_t>(ind) * alpha_moments_count;
					std::copy(moment_vals,
					          moment_vals + alpha_moments_count,
					          two_layer_final_moment_values_cache_.data() + offset);
					std::copy(site_energy_ders_wrt_moments_.begin(),
					          site_energy_ders_wrt_moments_.end(),
					          two_layer_final_moment_ders_cache_.data() + offset);
				}
				active_two_layer_edge_cache_atom_index_ = -1;
				active_two_layer_gate_values_ = nullptr;
				active_two_layer_gate_adjoints_ = nullptr;
			}
		} else {
			two_layer_final_moment_values_cache_.clear();
			two_layer_final_moment_ders_cache_.clear();
		}
		two_layer_forward_final_moment_cache_ready_ = false;
		const double profile_after_energy_adjoint =
			profile_two_layer ? TwoLayerProfileNow() : 0.0;

	double profile_directional_s = 0.0;
	double profile_tangent_grad_s = 0.0;
	std::vector<double> gate_directional_moment_tangent_cache;
	std::vector<char> gate_directional_moment_tangent_valid;
	if (!energy_gate_adjoints.empty()) {
		std::vector<Vector3> gate_chain_weights;
		std::vector<double> gate_scalar_tangents;
			std::vector<double> gate_body_tangents(TwoLayerGateBodyOrderCount(), 0.0);
			std::vector<double> gate_body_adjoint(TwoLayerGateBodyOrderCount(), 0.0);
		std::vector<double> gate_moment_tangents;
		std::vector<double> gate_chain_directional_values(
			static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);
			if (alpha_moments_count > 0) {
				gate_directional_moment_tangent_cache.assign(
					static_cast<size_t>(cfg.size()) * alpha_moments_count, 0.0);
				gate_directional_moment_tangent_valid.assign(cfg.size(), 0);
			}
		const double profile_directional_start =
			profile_two_layer ? TwoLayerProfileNow() : 0.0;
		const int directional_gate_count = TwoLayerGateScalarCount();
		const int directional_body_count = TwoLayerGateBodyOrderCount();
		const double* directional_scalar_weights =
			TwoLayerGateUsesBodyLinearCombo()
				? TwoLayerGateScalarWeightData()
				: nullptr;
		const double* directional_body_mix_weights =
			TwoLayerGateUsesBodyLinearCombo()
				? TwoLayerGateBodyMixWeightMatrixData()
				: nullptr;
		const int directional_gate_weight_offset = TwoLayerGateWeightOffset();
		const int directional_body_mix_weight_offset = TwoLayerGateBodyMixWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* energy_gate_adjoint_by_mu =
				energy_gate_adjoints.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			bool has_energy_gate_adjoint = false;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double adjoint = energy_gate_adjoint_by_mu[mu];
				if (adjoint == 0.0)
					continue;
				has_energy_gate_adjoint = true;
			}
			const Neighborhood& nbh = neighborhoods[ind];
			gate_chain_weights.resize(nbh.count);
			FillWithZero(gate_chain_weights);
			bool grad_zero = true;
			for (int j = 0; j < nbh.count; ++j) {
				gate_chain_weights[j] += frc_weights[ind];
				gate_chain_weights[j] -= frc_weights[nbh.inds[j]];
			}
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					for (int b = 0; b < 3; ++b)
						gate_chain_weights[j][a] += str_weights[a][b] * nbh.vecs[j][b];
			for (int j = 0; j < nbh.count && grad_zero; ++j) {
				for (int a = 0; a < 3; ++a) {
					if (gate_chain_weights[j][a] != 0.0) {
						grad_zero = false;
						break;
					}
				}
			}
			if (grad_zero)
				continue;

			std::vector<double>* gate_moment_tangents_ptr =
				(has_energy_gate_adjoint
				 && !gate_directional_moment_tangent_cache.empty())
					? &gate_moment_tangents
					: nullptr;
			CalcTwoLayerGateScalarDirectionalDerivatives(nbh,
			                                             gate_chain_weights,
			                                             gate_scalar_tangents,
			                                             gate_moment_tangents_ptr,
			                                             ind);
			if (gate_moment_tangents_ptr != nullptr
			    && static_cast<int>(gate_moment_tangents.size())
			        == alpha_moments_count) {
				const size_t offset =
					static_cast<size_t>(ind) * alpha_moments_count;
				std::copy(gate_moment_tangents.begin(),
				          gate_moment_tangents.end(),
				          gate_directional_moment_tangent_cache.begin()
				              + offset);
				gate_directional_moment_tangent_valid[ind] = 1;
			}
			double* directional_by_mu = gate_chain_directional_values.data()
				+ static_cast<size_t>(ind) * radial_func_count;
				if (TwoLayerGateUsesBodyLinearCombo()) {
					std::fill(gate_body_tangents.begin(), gate_body_tangents.end(), 0.0);
					for (int q = 0; q < directional_gate_count; ++q) {
						const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
						gate_body_tangents[body_index] +=
							directional_scalar_weights[q] * gate_scalar_tangents[q];
					}
					std::fill(gate_body_adjoint.begin(), gate_body_adjoint.end(), 0.0);
					for (int mu = 0; mu < radial_func_count; ++mu) {
						double directional = 0.0;
						const double energy_adjoint = energy_gate_adjoint_by_mu[mu];
						const double* body_mix = directional_body_mix_weights
							+ static_cast<size_t>(mu) * directional_body_count;
						for (int b = 0; b < directional_body_count; ++b) {
							const double tangent = gate_body_tangents[b];
							const double mix = body_mix[b];
							directional += mix * tangent;
							if (energy_adjoint != 0.0) {
								out_grads_accumulator[
									directional_body_mix_weight_offset
									+ mu * directional_body_count + b] +=
									energy_adjoint * tangent;
								gate_body_adjoint[b] += energy_adjoint * mix;
							}
						}
						directional_by_mu[mu] = directional;
					}
					for (int q = 0; q < directional_gate_count; ++q) {
						const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
						const double tangent = gate_scalar_tangents[q];
						const double adjoint = gate_body_adjoint[body_index];
						if (adjoint != 0.0)
							out_grads_accumulator[directional_gate_weight_offset + q] +=
								adjoint * tangent;
					}
				} else if (TwoLayerGateUsesFullScalarWeights()) {
					const int gate_count = TwoLayerGateScalarCount();
					cblas_dgemv(CblasRowMajor,
					            CblasNoTrans,
					            radial_func_count,
					            gate_count,
					            1.0,
					            TwoLayerGateFullWeightMatrixData(),
					            gate_count,
					            gate_scalar_tangents.data(),
					            1,
					            0.0,
					            directional_by_mu,
					            1);
					if (has_energy_gate_adjoint)
						cblas_dger(CblasRowMajor,
						           radial_func_count,
						           gate_count,
						           1.0,
						           energy_gate_adjoint_by_mu,
						           1,
						           gate_scalar_tangents.data(),
						           1,
						           out_grads_accumulator.data() + TwoLayerGateWeightOffset(),
						           gate_count);
				} else {
					ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
				}
		}
		if (profile_two_layer) {
			const double profile_after_directional = TwoLayerProfileNow();
			profile_directional_s +=
				profile_after_directional - profile_directional_start;
		}
			active_two_layer_gate_values_ = &two_layer_gate_values_;
			active_two_layer_gate_adjoints_ = &two_layer_gate_adjoints_;
			std::vector<double> neighbor_gate_tangent;
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
			const double profile_tangent_start =
				profile_two_layer ? TwoLayerProfileNow() : 0.0;
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const Neighborhood& nbh = neighborhoods[ind];
			neighbor_gate_tangent.resize(
				static_cast<size_t>(nbh.count) * radial_func_count);
			bool has_tangent = false;
				for (int j = 0; j < nbh.count; ++j) {
					const int atom_index = nbh.inds[j];
					const double* tangent_by_mu =
						gate_chain_directional_values.data()
						+ static_cast<size_t>(atom_index) * radial_func_count;
				double* neighbor_tangent_by_mu =
					neighbor_gate_tangent.data()
					+ static_cast<size_t>(j) * radial_func_count;
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double tangent = tangent_by_mu[mu];
						neighbor_tangent_by_mu[mu] = tangent;
						if (tangent != 0.0)
							has_tangent = true;
					}
						if (has_nonzero_edge_l1_weights) {
							Vector3 edge_der_weight = frc_weights[ind];
							edge_der_weight -= frc_weights[atom_index];
						for (int a = 0; a < 3; ++a)
							for (int b = 0; b < 3; ++b)
								edge_der_weight[a] +=
									str_weights[a][b] * nbh.vecs[j][b];
						ComputeTwoLayerGateEdgeL1PairCoordTangents(
							nbh,
							j,
							nullptr,
							nullptr,
							&edge_der_weight,
							two_layer_gate_edge_l1_chi_adj_buffer_);
							for (int mu = 0; mu < radial_func_count; ++mu) {
								const double* weight_row =
									edge_l1_weights
									+ static_cast<size_t>(mu) * edge_l1_source_count;
								double edge_tangent = 0.0;
								for (int source = 0;
								     source < edge_l1_source_count;
								     ++source)
									edge_tangent +=
										weight_row[source]
										* two_layer_gate_edge_l1_chi_adj_buffer_[source];
							neighbor_tangent_by_mu[mu] += edge_tangent;
							if (edge_tangent != 0.0)
								has_tangent = true;
						}
					}
				}
			const double* center_gate_tangent = TwoLayerGateUsesCenterGate()
				? gate_chain_directional_values.data()
					+ static_cast<size_t>(ind) * radial_func_count
				: nullptr;
			if (!has_tangent && center_gate_tangent != nullptr)
				for (int mu = 0; mu < radial_func_count; ++mu)
					if (center_gate_tangent[mu] != 0.0) {
						has_tangent = true;
						break;
					}
			if (has_tangent) {
				AccumulateSHGateTangentGrad(nbh,
				                            out_grads_accumulator,
				                            neighbor_gate_tangent,
				                            center_gate_tangent,
				                            ind);
			}
		}
		if (profile_two_layer)
			profile_tangent_grad_s += TwoLayerProfileNow() - profile_tangent_start;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
	}

	const double profile_gate_weight_start =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;
	std::vector<double> gate_body_values_scratch(TwoLayerGateBodyOrderCount(), 0.0);
	std::vector<double> gate_body_adjoint_scratch(TwoLayerGateBodyOrderCount(), 0.0);
	const int gate_count_for_weights = TwoLayerGateScalarCount();
	const int body_count_for_weights = TwoLayerGateBodyOrderCount();
	const bool full_gate_weight_blas =
		TwoLayerGateUsesFullScalarWeights()
		&& gate_count_for_weights > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * gate_count_for_weights;
	const bool body_combo_gate_weight_blas =
		TwoLayerGateUsesBodyLinearCombo()
		&& gate_count_for_weights > 0
		&& body_count_for_weights > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * gate_count_for_weights
		&& two_layer_gate_body_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * body_count_for_weights;
	if (full_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            gate_count_for_weights,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_scalar_values_cache_.data(),
		            gate_count_for_weights,
		            1.0,
		            out_grads_accumulator.data() + TwoLayerGateWeightOffset(),
		            gate_count_for_weights);
	} else if (body_combo_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            body_count_for_weights,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_body_values_cache_.data(),
		            body_count_for_weights,
		            1.0,
		            out_grads_accumulator.data()
		                + TwoLayerGateBodyMixWeightOffset(),
		            body_count_for_weights);
		std::vector<double> body_adjoint_cache(
			static_cast<size_t>(cfg.size()) * body_count_for_weights, 0.0);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            body_count_for_weights,
		            radial_func_count,
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            TwoLayerGateBodyMixWeightMatrixData(),
		            body_count_for_weights,
		            0.0,
		            body_adjoint_cache.data(),
		            body_count_for_weights);
		double* scalar_weight_grads =
			out_grads_accumulator.data() + TwoLayerGateWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* scalar_values =
				two_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(ind) * gate_count_for_weights;
			const double* body_adjoint =
				body_adjoint_cache.data()
				+ static_cast<size_t>(ind) * body_count_for_weights;
			for (int q = 0; q < gate_count_for_weights; ++q) {
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
			const double* gate_adjoints_by_mu = two_layer_gate_adjoints_.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			bool has_gate_adjoint = false;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double adjoint = gate_adjoints_by_mu[mu];
				if (adjoint == 0.0)
					continue;
				has_gate_adjoint = true;
			}
			if (!has_gate_adjoint)
				continue;
			const int gate_count = TwoLayerGateScalarCount();
			const double* cached_scalars = nullptr;
			if (gate_count > 0 &&
				two_layer_gate_scalar_values_cache_.size() ==
					static_cast<size_t>(cfg.size()) * gate_count) {
				cached_scalars = two_layer_gate_scalar_values_cache_.data()
					+ static_cast<size_t>(ind) * gate_count;
			}
			if (cached_scalars == nullptr)
				CalcTwoLayerGateScalarValuesOnly(neighborhoods[ind],
				                                  sh_gate_scalar_values_,
				                                  ind);
			const double* scalar_values =
				(cached_scalars == nullptr) ? sh_gate_scalar_values_.data() : cached_scalars;
			if (TwoLayerGateUsesBodyLinearCombo()) {
				const double* cached_body_values = nullptr;
				if (two_layer_gate_body_values_cache_.size() ==
				    static_cast<size_t>(cfg.size()) * TwoLayerGateBodyOrderCount())
					cached_body_values = two_layer_gate_body_values_cache_.data()
						+ static_cast<size_t>(ind) * TwoLayerGateBodyOrderCount();
				const double* body_values = cached_body_values;
				if (body_values == nullptr) {
					ComputeTwoLayerGateBodySignals(scalar_values,
					                               gate_body_values_scratch.data());
					body_values = gate_body_values_scratch.data();
				}
				std::fill(gate_body_adjoint_scratch.begin(),
				          gate_body_adjoint_scratch.end(),
				          0.0);
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					const double* body_mix = body_mix_weights
						+ static_cast<size_t>(mu) * body_count_for_weights;
					for (int b = 0; b < body_count_for_weights; ++b) {
						const double body_value = body_values[b];
						out_grads_accumulator[
							body_mix_weight_offset
							+ mu * body_count_for_weights + b] +=
							gate_adjoint * body_value;
						gate_body_adjoint_scratch[b] +=
							gate_adjoint * body_mix[b];
					}
				}
				for (int q = 0; q < gate_count; ++q) {
					const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
					const double body_adjoint =
						gate_body_adjoint_scratch[body_index];
					if (body_adjoint != 0.0)
						out_grads_accumulator[gate_weight_offset + q] +=
							body_adjoint * scalar_values[q];
				}
			} else if (TwoLayerGateUsesFullScalarWeights()) {
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					for (int q = 0; q < gate_count; ++q)
						out_grads_accumulator[TwoLayerGateWeightIndex(mu, q)] +=
							gate_adjoint * scalar_values[q];
				}
			} else {
				ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
			}
		}
	}
	const double profile_after_gate_weight =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;

	const double profile_scalar_param_start =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;
		if (HasNonzeroTwoLayerGateWeights()) {
			std::vector<Vector3> gate_der_weights;
			std::vector<double> scalar_gate_seeds(TwoLayerGateScalarCount(), 0.0);
			std::vector<double> scalar_energy_seeds(TwoLayerGateScalarCount(), 0.0);
			std::vector<double> scalar_seed_body_scratch;
			std::vector<double> full_scalar_gate_seed_cache;
			std::vector<double> full_scalar_energy_seed_cache;
			std::vector<double> body_combo_gate_seed_cache;
			std::vector<double> body_combo_energy_seed_cache;
			const int gate_count = TwoLayerGateScalarCount();
			const int body_count = TwoLayerGateBodyOrderCount();
			const bool full_seed_blas =
				TwoLayerGateUsesFullScalarWeights()
				&& gate_count > 0
				&& two_layer_gate_adjoints_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
			const bool body_combo_seed_blas =
				TwoLayerGateUsesBodyLinearCombo()
				&& gate_count > 0
				&& body_count > 0
				&& two_layer_gate_adjoints_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
			const double* body_combo_scalar_weights =
				body_combo_seed_blas ? TwoLayerGateScalarWeightData() : nullptr;
			if (full_seed_blas) {
				full_scalar_gate_seed_cache.assign(
					static_cast<size_t>(cfg.size()) * gate_count, 0.0);
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
				if (!energy_gate_adjoints.empty()) {
					full_scalar_energy_seed_cache.assign(
						static_cast<size_t>(cfg.size()) * gate_count, 0.0);
					cblas_dgemm(CblasRowMajor,
					            CblasNoTrans,
					            CblasNoTrans,
					            cfg.size(),
					            gate_count,
					            radial_func_count,
					            1.0,
					            energy_gate_adjoints.data(),
					            radial_func_count,
					            TwoLayerGateFullWeightMatrixData(),
					            gate_count,
					            0.0,
					            full_scalar_energy_seed_cache.data(),
					            gate_count);
				}
			}
			if (body_combo_seed_blas) {
				body_combo_gate_seed_cache.assign(
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
				            body_combo_gate_seed_cache.data(),
				            body_count);
				if (!energy_gate_adjoints.empty()) {
					body_combo_energy_seed_cache.assign(
						static_cast<size_t>(cfg.size()) * body_count, 0.0);
					cblas_dgemm(CblasRowMajor,
					            CblasNoTrans,
					            CblasNoTrans,
					            cfg.size(),
					            body_count,
					            radial_func_count,
					            1.0,
					            energy_gate_adjoints.data(),
					            radial_func_count,
					            TwoLayerGateBodyMixWeightMatrixData(),
					            body_count,
					            0.0,
					            body_combo_energy_seed_cache.data(),
					            body_count);
				}
			}
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
						const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
						scalar_gate_seeds[q] =
							body_gate_seeds[body_index]
							* body_combo_scalar_weights[q];
					}
					if (!body_combo_energy_seed_cache.empty()) {
						const double* body_energy_seeds =
							body_combo_energy_seed_cache.data()
							+ static_cast<size_t>(ind) * body_count;
						for (int q = 0; q < gate_count; ++q) {
							const int body_index =
								two_layer_gate_scalar_body_orders_[q] - 2;
							scalar_energy_seeds[q] =
								body_energy_seeds[body_index]
								* body_combo_scalar_weights[q];
						}
						scalar_energy_seed_values = scalar_energy_seeds.data();
					}
				} else {
					const double* gate_adjoints_by_mu = two_layer_gate_adjoints_.data()
						+ static_cast<size_t>(ind) * radial_func_count;
					AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
						gate_adjoints_by_mu,
						scalar_gate_seeds.data(),
						scalar_seed_body_scratch);
					if (!energy_gate_adjoints.empty()) {
						const double* energy_gate_adjoints_by_mu =
							energy_gate_adjoints.data()
							+ static_cast<size_t>(ind) * radial_func_count;
						AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
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
				if (!full_seed_blas && !energy_gate_adjoints.empty()) {
					scalar_energy_seed_values = scalar_energy_seeds.data();
				}
				if (!full_seed_blas && energy_gate_adjoints.empty()) {
					scalar_energy_seed_values = nullptr;
				}
				const Neighborhood& nbh = neighborhoods[ind];
				const Vector3* gate_der_weights_ptr = nullptr;
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
					gate_der_weights[j] += frc_weights[ind];
					gate_der_weights[j] -= frc_weights[nbh.inds[j]];
				}
				for (int j = 0; j < nbh.count; ++j)
					for (int a = 0; a < 3; ++a)
						for (int b = 0; b < 3; ++b)
							gate_der_weights[j][a] +=
								str_weights[a][b] * nbh.vecs[j][b];
				for (int j = 0; j < nbh.count; ++j) {
					if (gate_der_weights[j].NormSq() != 0.0)
						has_der_weight = true;
				}
				if (has_der_weight)
					gate_der_weights_ptr = gate_der_weights.data();
			}
			const double* gate_moment_tangents_ptr = nullptr;
			if (gate_der_weights_ptr != nullptr
			    && ind < static_cast<int>(
			        gate_directional_moment_tangent_valid.size())
			    && gate_directional_moment_tangent_valid[ind]) {
				gate_moment_tangents_ptr =
					gate_directional_moment_tangent_cache.data()
					+ static_cast<size_t>(ind) * alpha_moments_count;
			}
				AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
					nbh,
					out_grads_accumulator,
					scalar_gate_seed_values,
					gate_der_weights_ptr,
					ind,
					gate_moment_tangents_ptr,
					scalar_energy_seed_values);
			}
		}
	const double profile_after_scalar_param =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;

	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
	active_two_layer_gate_edge_l1_coord_loss_ = saved_edge_l1_coord_loss;
	if (profile_two_layer) {
		const double profile_end = TwoLayerProfileNow();
		RecordTwoLayerGradProfile(
			cfg.size(),
			profile_end - profile_start,
			profile_after_prepare - profile_start,
			profile_after_main_grad - profile_after_prepare,
			profile_after_energy_adjoint - profile_after_main_grad,
			profile_directional_s,
			profile_tangent_grad_s,
			profile_after_gate_weight - profile_gate_weight_start,
			profile_after_scalar_param - profile_scalar_param_start);
		}
	}

void MLMTPR::AccumulateEnergyCombinationGrad(Configuration& cfg,
                                             std::vector<double>& ene_weight,
                                             Array1D& out_grads_accumulator)
{
	Neighborhoods neighborhoods(cfg, CutOff());
	AccumulateEnergyCombinationGrad(cfg, ene_weight, out_grads_accumulator,
	                                neighborhoods);
}

void MLMTPR::AccumulateEnergyCombinationGrad(Configuration& cfg,
                                             std::vector<double>& ene_weight,
                                             Array1D& out_grads_accumulator,
                                             const Neighborhoods& neighborhoods)
{
	if (!is_sh_potential_ || !two_layer_gate_enabled_) {
		AnyLocalMLIP::AccumulateEnergyCombinationGrad(cfg, ene_weight,
		                                              out_grads_accumulator,
		                                              neighborhoods);
		return;
	}

	out_grads_accumulator.resize(CoeffCount());
	PrepareEvalCaches();
	two_layer_full_edge_cache_for_next_calc_ = false;
	two_layer_reuse_full_edge_cache_once_ = false;
	two_layer_forward_final_moment_cache_ready_ = false;
	BuildTwoLayerEdgePrimitiveCache(neighborhoods, false);
	PrepareTwoLayerGateValues(cfg, neighborhoods);
	two_layer_gate_adjoints_.assign(
		static_cast<size_t>(cfg.size()) * radial_func_count, 0.0);

	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	active_two_layer_gate_values_ = &two_layer_gate_values_;
	active_two_layer_gate_adjoints_ = &two_layer_gate_adjoints_;

	for (int ind = 0; ind < cfg.size(); ++ind) {
		active_two_layer_edge_cache_atom_index_ = ind;
		AccumulateCombinationGrad(neighborhoods[ind],
		                           out_grads_accumulator,
		                           ene_weight[ind],
		                           nullptr);
	}
	active_two_layer_edge_cache_atom_index_ = -1;

	const int energy_gate_count_for_weights = TwoLayerGateScalarCount();
	const int energy_body_count_for_weights = TwoLayerGateBodyOrderCount();
	const bool energy_full_gate_weight_blas =
		TwoLayerGateUsesFullScalarWeights()
		&& energy_gate_count_for_weights > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * energy_gate_count_for_weights;
	const bool energy_body_combo_gate_weight_blas =
		TwoLayerGateUsesBodyLinearCombo()
		&& energy_gate_count_for_weights > 0
		&& energy_body_count_for_weights > 0
		&& two_layer_gate_adjoints_.size()
			== static_cast<size_t>(cfg.size()) * radial_func_count
		&& two_layer_gate_scalar_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * energy_gate_count_for_weights
		&& two_layer_gate_body_values_cache_.size()
			== static_cast<size_t>(cfg.size()) * energy_body_count_for_weights;
	if (energy_full_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            energy_gate_count_for_weights,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_scalar_values_cache_.data(),
		            energy_gate_count_for_weights,
		            1.0,
		            out_grads_accumulator.data() + TwoLayerGateWeightOffset(),
		            energy_gate_count_for_weights);
	} else if (energy_body_combo_gate_weight_blas) {
		cblas_dgemm(CblasRowMajor,
		            CblasTrans,
		            CblasNoTrans,
		            radial_func_count,
		            energy_body_count_for_weights,
		            cfg.size(),
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            two_layer_gate_body_values_cache_.data(),
		            energy_body_count_for_weights,
		            1.0,
		            out_grads_accumulator.data()
		                + TwoLayerGateBodyMixWeightOffset(),
		            energy_body_count_for_weights);
		std::vector<double> body_adjoint_cache(
			static_cast<size_t>(cfg.size()) * energy_body_count_for_weights, 0.0);
		cblas_dgemm(CblasRowMajor,
		            CblasNoTrans,
		            CblasNoTrans,
		            cfg.size(),
		            energy_body_count_for_weights,
		            radial_func_count,
		            1.0,
		            two_layer_gate_adjoints_.data(),
		            radial_func_count,
		            TwoLayerGateBodyMixWeightMatrixData(),
		            energy_body_count_for_weights,
		            0.0,
		            body_adjoint_cache.data(),
		            energy_body_count_for_weights);
		double* scalar_weight_grads =
			out_grads_accumulator.data() + TwoLayerGateWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* scalar_values =
				two_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(ind) * energy_gate_count_for_weights;
			const double* body_adjoint =
				body_adjoint_cache.data()
				+ static_cast<size_t>(ind) * energy_body_count_for_weights;
			for (int q = 0; q < energy_gate_count_for_weights; ++q) {
				const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
				scalar_weight_grads[q] +=
					body_adjoint[body_index] * scalar_values[q];
			}
		}
	} else {
		std::vector<double> energy_gate_body_values(
			TwoLayerGateBodyOrderCount(), 0.0);
		std::vector<double> energy_gate_body_adjoint(
			TwoLayerGateBodyOrderCount(), 0.0);
		const double* body_mix_weights =
			TwoLayerGateUsesBodyLinearCombo()
				? TwoLayerGateBodyMixWeightMatrixData()
				: nullptr;
		const int gate_weight_offset = TwoLayerGateWeightOffset();
		const int body_mix_weight_offset = TwoLayerGateBodyMixWeightOffset();
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* gate_adjoints_by_mu = two_layer_gate_adjoints_.data()
				+ static_cast<size_t>(ind) * radial_func_count;
			bool has_gate_adjoint = false;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double adjoint = gate_adjoints_by_mu[mu];
				if (adjoint == 0.0)
					continue;
				has_gate_adjoint = true;
			}
			if (!has_gate_adjoint)
				continue;
			const int gate_count = TwoLayerGateScalarCount();
			const double* cached_scalars = nullptr;
			if (gate_count > 0 &&
				two_layer_gate_scalar_values_cache_.size() ==
					static_cast<size_t>(cfg.size()) * gate_count) {
				cached_scalars = two_layer_gate_scalar_values_cache_.data()
					+ static_cast<size_t>(ind) * gate_count;
			}
			if (cached_scalars == nullptr)
				CalcTwoLayerGateScalarValuesOnly(neighborhoods[ind],
				                                  sh_gate_scalar_values_,
				                                  ind);
			const double* scalar_values =
				(cached_scalars == nullptr) ? sh_gate_scalar_values_.data() : cached_scalars;
			if (TwoLayerGateUsesBodyLinearCombo()) {
				const double* cached_body_values = nullptr;
				if (two_layer_gate_body_values_cache_.size() ==
				    static_cast<size_t>(cfg.size()) * TwoLayerGateBodyOrderCount())
					cached_body_values = two_layer_gate_body_values_cache_.data()
						+ static_cast<size_t>(ind) * TwoLayerGateBodyOrderCount();
				const double* body_values = cached_body_values;
				if (body_values == nullptr) {
					ComputeTwoLayerGateBodySignals(scalar_values,
					                               energy_gate_body_values.data());
					body_values = energy_gate_body_values.data();
				}
				std::fill(energy_gate_body_adjoint.begin(),
				          energy_gate_body_adjoint.end(),
				          0.0);
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					const double* body_mix = body_mix_weights
						+ static_cast<size_t>(mu) * energy_body_count_for_weights;
					for (int b = 0; b < energy_body_count_for_weights; ++b) {
						out_grads_accumulator[
							body_mix_weight_offset
							+ mu * energy_body_count_for_weights + b] +=
							gate_adjoint * body_values[b];
						energy_gate_body_adjoint[b] +=
							gate_adjoint * body_mix[b];
					}
				}
				for (int q = 0; q < gate_count; ++q) {
					const int body_index = two_layer_gate_scalar_body_orders_[q] - 2;
					const double body_adjoint =
						energy_gate_body_adjoint[body_index];
					if (body_adjoint != 0.0)
						out_grads_accumulator[gate_weight_offset + q] +=
							body_adjoint * scalar_values[q];
				}
			} else if (TwoLayerGateUsesFullScalarWeights()) {
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double gate_adjoint = gate_adjoints_by_mu[mu];
					if (gate_adjoint == 0.0)
						continue;
					for (int q = 0; q < gate_count; ++q)
						out_grads_accumulator[TwoLayerGateWeightIndex(mu, q)] +=
							gate_adjoint * scalar_values[q];
				}
			} else {
				ERROR("SUS2-SH two-layer gate model has an unknown mode: " + two_layer_gate_mode_);
			}
		}
	}

		if (HasNonzeroTwoLayerGateWeights()) {
			std::vector<double> scalar_gate_seeds(TwoLayerGateScalarCount(), 0.0);
			std::vector<double> scalar_seed_body_scratch;
			std::vector<double> full_scalar_gate_seed_cache;
			std::vector<double> body_combo_gate_seed_cache;
			const int gate_count = TwoLayerGateScalarCount();
			const int body_count = TwoLayerGateBodyOrderCount();
			const bool full_seed_blas =
				TwoLayerGateUsesFullScalarWeights()
				&& gate_count > 0
				&& two_layer_gate_adjoints_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
			const bool body_combo_seed_blas =
				TwoLayerGateUsesBodyLinearCombo()
				&& gate_count > 0
				&& body_count > 0
				&& two_layer_gate_adjoints_.size()
					== static_cast<size_t>(cfg.size()) * radial_func_count;
			const double* body_combo_scalar_weights =
				body_combo_seed_blas ? TwoLayerGateScalarWeightData() : nullptr;
			if (full_seed_blas) {
				full_scalar_gate_seed_cache.assign(
					static_cast<size_t>(cfg.size()) * gate_count, 0.0);
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
				body_combo_gate_seed_cache.assign(
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
					const double* gate_adjoints_by_mu = two_layer_gate_adjoints_.data()
						+ static_cast<size_t>(ind) * radial_func_count;
					AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
						gate_adjoints_by_mu,
						scalar_gate_seeds.data(),
						scalar_seed_body_scratch);
				}
				AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
				neighborhoods[ind],
				out_grads_accumulator,
				scalar_gate_seed_values,
				nullptr,
				ind,
				nullptr,
				nullptr);
		}
	}

	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
}





void MLMTPR::CalcEFSComponents(Configuration& cfg)
{
	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);
	CalcEFSComponents(cfg, neighborhoods);
}

void MLMTPR::CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	CalcEFSComponents(cfg, neighborhoods, true, true);
}

void MLMTPR::CalcEFSComponents(Configuration& cfg, bool need_forces, bool need_stress)
{
	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);
	CalcEFSComponents(cfg, neighborhoods, need_forces, need_stress);
}

void MLMTPR::CalcEFSComponents(Configuration& cfg,
                               const Neighborhoods& neighborhoods,
                               bool need_forces,
                               bool need_stress)
{
	int n = LinearEquationCount();

	if (!need_forces && !need_stress) {
		CalcEComponents(cfg, neighborhoods);
		return;
	}

	if (cfg.size() != forces_cmpnts.size1 || n != forces_cmpnts.size2)
		forces_cmpnts.resize(cfg.size(), n, 3);

	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));
	if (need_forces)
		forces_cmpnts.set(0);
	if (need_stress)
		memset(stress_cmpnts, 0, n * sizeof(stress_cmpnts[0]));

	const bool use_two_layer_gate = RequiresTwoLayerGateEvaluation();
	if (!use_two_layer_gate) {
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		active_two_layer_edge_cache_atom_index_ = -1;
	}
	const bool profile_two_layer =
		use_two_layer_gate && TwoLayerProfileEnabledOnRank0();
	const double profile_start = profile_two_layer ? TwoLayerProfileNow() : 0.0;
	if (use_two_layer_gate) {
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, true, false);
		PrepareTwoLayerGateValues(cfg, neighborhoods);
		active_two_layer_gate_values_ = &two_layer_gate_values_;
	}
	const double profile_after_prepare =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;
	const bool need_gate_linear_chain =
		use_two_layer_gate && (need_forces || need_stress);
	if (need_gate_linear_chain) {
		sh_gate_linear_adjoints_.assign(
			static_cast<size_t>(cfg.size()) * alpha_scalar_moments, 0.0);
	}

		if (TwoLayerResidualEnabled()) {
			const int e1_linear_count = LinearCoeffCount();
			const int e0_col_offset = e1_linear_count;
			double profile_gate_adjoints_s = 0.0;
			double profile_basis_calc_s = 0.0;
			double profile_component_fill_s = 0.0;
			for (int ind = 0; ind < cfg.size(); ind++) {
				const Neighborhood& nbh = neighborhoods[ind];
				if (nbh.my_type >= species_count)
					throw MlipException("Too few species count in the MTP potential!");

				// E1 columns: final direct-gated scalar basis, no one-body column.
				active_two_layer_gate_values_ = &two_layer_gate_values_;
				active_two_layer_edge_cache_atom_index_ = ind;
				const double profile_basis_start =
					profile_two_layer ? TwoLayerProfileNow() : 0.0;
				CalcBasisFuncsDers(nbh);
				const double profile_after_basis =
					profile_two_layer ? TwoLayerProfileNow() : 0.0;
				if (profile_two_layer)
					profile_basis_calc_s += profile_after_basis - profile_basis_start;
				for (int k = species_count; k < e1_linear_count; k++) {
					const int i = k - species_count + 1;
					energy_cmpnts[k] += basis_vals[i];
					if (need_forces)
						for (int j = 0; j < nbh.count; j++)
							for (int a = 0; a < 3; a++) {
								forces_cmpnts(ind, k, a) += basis_ders(i, j, a);
								forces_cmpnts(nbh.inds[j], k, a) -= basis_ders(i, j, a);
							}
					if (need_stress)
						for (int j = 0; j < nbh.count; j++)
							for (int a = 0; a < 3; a++)
									for (int b = 0; b < 3; b++)
											stress_cmpnts[k][a][b] -= basis_ders(i, j, a) * nbh.vecs[j][b];
						}
				if (profile_two_layer)
					profile_component_fill_s += TwoLayerProfileNow() - profile_after_basis;
				if (need_gate_linear_chain) {
					const double profile_gate_adjoints_start =
						profile_two_layer ? TwoLayerProfileNow() : 0.0;
					AccumulateSHBasisGateDers(nbh, sh_gate_linear_adjoints_);
					if (profile_two_layer)
						profile_gate_adjoints_s +=
							TwoLayerProfileNow() - profile_gate_adjoints_start;
				}

				// E0 columns: one-cutoff unmodulated scalar basis plus one-body.
				active_two_layer_gate_values_ = nullptr;
				const double profile_e0_basis_start =
					profile_two_layer ? TwoLayerProfileNow() : 0.0;
				CalcBasisFuncsDers(nbh);
				const double profile_after_e0_basis =
					profile_two_layer ? TwoLayerProfileNow() : 0.0;
				if (profile_two_layer)
					profile_basis_calc_s += profile_after_e0_basis - profile_e0_basis_start;
				energy_cmpnts[nbh.my_type] += basis_vals[0];
				for (int i = 0; i < alpha_scalar_moments; ++i) {
					const int col = e0_col_offset + i;
					energy_cmpnts[col] += basis_vals[1 + i];
					if (need_forces)
						for (int j = 0; j < nbh.count; j++)
							for (int a = 0; a < 3; a++) {
								forces_cmpnts(ind, col, a) += basis_ders(1 + i, j, a);
								forces_cmpnts(nbh.inds[j], col, a) -= basis_ders(1 + i, j, a);
							}
					if (need_stress)
						for (int j = 0; j < nbh.count; j++)
							for (int a = 0; a < 3; a++)
								for (int b = 0; b < 3; b++)
									stress_cmpnts[col][a][b] -= basis_ders(1 + i, j, a) * nbh.vecs[j][b];
				}
				if (profile_two_layer)
					profile_component_fill_s += TwoLayerProfileNow() - profile_after_e0_basis;
			}
			active_two_layer_edge_cache_atom_index_ = -1;

			if (need_gate_linear_chain) {
				const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
				active_two_layer_gate_values_ = nullptr;
				for (int ind = 0; ind < cfg.size(); ++ind) {
					const double* gate_adjoints = sh_gate_linear_adjoints_.data()
						+ static_cast<size_t>(ind) * alpha_scalar_moments;
					bool has_gate_adjoint = false;
					for (int i = 0; i < alpha_scalar_moments; ++i)
						if (gate_adjoints[i] != 0.0) {
							has_gate_adjoint = true;
							break;
						}
					if (!has_gate_adjoint)
						continue;
					const Neighborhood& nbh = neighborhoods[ind];
					CalcTwoLayerGateWeightedScalarDers(nbh, sh_gate_component_ders_, ind);
					for (int j = 0; j < nbh.count; ++j) {
						for (int a = 0; a < 3; ++a) {
							const double component_der = sh_gate_component_ders_[j][a];
							if (component_der == 0.0)
								continue;
							if (need_forces) {
								cblas_daxpy(alpha_scalar_moments,
								            component_der,
								            gate_adjoints,
								            1,
								            &forces_cmpnts(ind, species_count, a),
								            3);
								cblas_daxpy(alpha_scalar_moments,
								            -component_der,
								            gate_adjoints,
								            1,
								            &forces_cmpnts(nbh.inds[j], species_count, a),
								            3);
							}
							if (need_stress) {
								for (int b = 0; b < 3; ++b)
									cblas_daxpy(alpha_scalar_moments,
									            -component_der * nbh.vecs[j][b],
									            gate_adjoints,
									            1,
									            &stress_cmpnts[species_count][a][b],
									            9);
							}
						}
					}
				}
				active_two_layer_gate_values_ = saved_gate_values;
			}
			if (use_two_layer_gate)
				active_two_layer_gate_values_ = nullptr;
			return;
		}

		double profile_gate_adjoints_s = 0.0;
		double profile_basis_calc_s = 0.0;
		double profile_component_fill_s = 0.0;
		for (int ind = 0; ind < cfg.size(); ind++) {
			const Neighborhood& nbh = neighborhoods[ind];
			if (use_two_layer_gate)
				active_two_layer_edge_cache_atom_index_ = ind;
			const double profile_basis_start =
				profile_two_layer ? TwoLayerProfileNow() : 0.0;
			CalcBasisFuncsDers(nbh);
			const double profile_after_basis =
				profile_two_layer ? TwoLayerProfileNow() : 0.0;
			if (profile_two_layer)
				profile_basis_calc_s += profile_after_basis - profile_basis_start;

			if (nbh.my_type>=species_count)
				throw MlipException("Too few species count in the MTP potential!");

			energy_cmpnts[nbh.my_type] += basis_vals[0];
			for (int k = species_count; k < n; k++) {
				int i = k - species_count + 1;
				energy_cmpnts[k] += basis_vals[i];
				if (need_forces)
					for (int j = 0; j < nbh.count; j++)
						for (int a = 0; a < 3; a++) {
							forces_cmpnts(ind, k, a) += basis_ders(i, j, a);
							forces_cmpnts(nbh.inds[j], k, a) -= basis_ders(i, j, a);
						}
				if (need_stress)
					for (int j = 0; j < nbh.count; j++)
						for (int a = 0; a < 3; a++)
							for (int b = 0; b < 3; b++)
								stress_cmpnts[k][a][b] -= basis_ders(i, j, a) * nbh.vecs[j][b];
			}
			if (profile_two_layer)
				profile_component_fill_s += TwoLayerProfileNow() - profile_after_basis;
			if (need_gate_linear_chain) {
				const double profile_gate_adjoints_start =
					profile_two_layer ? TwoLayerProfileNow() : 0.0;
				AccumulateSHBasisGateDers(nbh, sh_gate_linear_adjoints_);
				if (profile_two_layer)
					profile_gate_adjoints_s +=
						TwoLayerProfileNow() - profile_gate_adjoints_start;
			}
		}
	if (use_two_layer_gate)
		active_two_layer_edge_cache_atom_index_ = -1;
	const double profile_after_main_components =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;

	if (need_gate_linear_chain) {
		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		active_two_layer_gate_values_ = nullptr;
		for (int ind = 0; ind < cfg.size(); ++ind) {
			const double* gate_adjoints = sh_gate_linear_adjoints_.data()
				+ static_cast<size_t>(ind) * alpha_scalar_moments;
			bool has_gate_adjoint = false;
			for (int i = 0; i < alpha_scalar_moments; ++i)
				if (gate_adjoints[i] != 0.0) {
					has_gate_adjoint = true;
					break;
				}
			if (!has_gate_adjoint)
				continue;

			const Neighborhood& nbh = neighborhoods[ind];
			CalcTwoLayerGateWeightedScalarDers(nbh, sh_gate_component_ders_, ind);

			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					const double component_der = sh_gate_component_ders_[j][a];
					if (component_der == 0.0)
						continue;
					if (need_forces) {
						cblas_daxpy(alpha_scalar_moments,
						            component_der,
						            gate_adjoints,
						            1,
						            &forces_cmpnts(ind, species_count, a),
						            3);
						cblas_daxpy(alpha_scalar_moments,
						            -component_der,
						            gate_adjoints,
						            1,
						            &forces_cmpnts(nbh.inds[j], species_count, a),
						            3);
					}
					if (need_stress) {
						for (int b = 0; b < 3; ++b)
							cblas_daxpy(alpha_scalar_moments,
							            -component_der * nbh.vecs[j][b],
							            gate_adjoints,
							            1,
							            &stress_cmpnts[species_count][a][b],
							            9);
					}
				}
			}
		}
		active_two_layer_gate_values_ = saved_gate_values;
	}
	const double profile_after_chain_apply =
		profile_two_layer ? TwoLayerProfileNow() : 0.0;

	if (use_two_layer_gate)
		active_two_layer_gate_values_ = nullptr;
	if (profile_two_layer) {
		const double profile_end = TwoLayerProfileNow();
		RecordTwoLayerLinearProfile(
			cfg.size(),
			profile_end - profile_start,
			profile_after_prepare - profile_start,
			(profile_after_main_components - profile_after_prepare)
				- profile_gate_adjoints_s,
			profile_gate_adjoints_s,
			profile_after_chain_apply - profile_after_main_components,
			profile_basis_calc_s,
			profile_component_fill_s);
	}
}

void MLMTPR::CalcEComponents(Configuration& cfg)
{
	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist); 
	CalcEComponents(cfg, neighborhoods);
}

void MLMTPR::CalcEComponents(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	int n = LinearEquationCount();
	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));

	const bool use_two_layer_gate = RequiresTwoLayerGateEvaluation();
	if (!use_two_layer_gate) {
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;
		active_two_layer_edge_cache_atom_index_ = -1;
	}
	if (use_two_layer_gate) {
		BuildTwoLayerEdgePrimitiveCache(neighborhoods, false);
		PrepareTwoLayerGateValues(cfg, neighborhoods);
		active_two_layer_gate_values_ = &two_layer_gate_values_;
	}

	if (TwoLayerResidualEnabled()) {
		const int e1_linear_count = LinearCoeffCount();
		const int e0_col_offset = e1_linear_count;
		for (int ind = 0; ind < cfg.size(); ind++) {
			const Neighborhood& nbh = neighborhoods[ind];
			if (nbh.my_type >= species_count)
				throw MlipException("Too few species count in the MTP potential!");

			active_two_layer_gate_values_ = &two_layer_gate_values_;
			active_two_layer_edge_cache_atom_index_ = ind;
			CalcBasisFuncs(nbh, basis_vals);
			for (int k = species_count; k < e1_linear_count; k++) {
				const int i = k - species_count + 1;
				energy_cmpnts[k] += basis_vals[i];
			}

			active_two_layer_gate_values_ = nullptr;
			CalcBasisFuncs(nbh, basis_vals);
			energy_cmpnts[nbh.my_type] += basis_vals[0];
			for (int i = 0; i < alpha_scalar_moments; ++i)
				energy_cmpnts[e0_col_offset + i] += basis_vals[1 + i];
		}
		active_two_layer_edge_cache_atom_index_ = -1;
		if (use_two_layer_gate)
			active_two_layer_gate_values_ = nullptr;
		return;
	}

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

		energy_cmpnts[nbh.my_type] += basis_vals[0];

		for (int k = species_count; k < n; k++)
		{
			int i = k - species_count + 1;
			energy_cmpnts[k] += basis_vals[i];
		}
	}

	if (use_two_layer_gate)
		active_two_layer_gate_values_ = nullptr;
}

void MLMTPR::CalcBasisFuncs(const Neighborhood& Neighborhood, double* bf_vals)
{
	if (is_sh_potential_) {
		CalcSHBasisFuncs(Neighborhood, bf_vals);
		return;
	}
	int C = species_count;						//number of different species in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	int type_central = Neighborhood.my_type;
	std::vector<double>& val_ = radial_vals_buffer_;

	if (type_central>=species_count)
				throw MlipException("Too few species count in the MTP potential!");

	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
                int type_outer = Neighborhood.types[j];
		
                
//		int type_outer = Neighborhood.types[j];/

			dist_powers_[0] = 1;
			//coords_powers_[0] = Vector3(1, 1, 1);
			coords_powers_x[0]=1;
			coords_powers_y[0]=1;
			coords_powers_z[0]=1;
			for (int k = 1; k < max_alpha_index_basic_; k++) {
				dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
				coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
				coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
				coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];

			}
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					Neighborhood.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					val_[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
			}

		for (int i = 0; i < alpha_index_basic_count; i++) {
			double val = 0;
			int mu = alpha_index_basic_.comp0[i];
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				p_RadialBasis->rb_vals[xi] *= scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				p_RadialBasis->rb_ders[xi] *= scaling; */


			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val += regression_coeffs[C+2*C*C*K_+ mu * (R+C) + xi] *regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_outer]* val_[radial_base + xi];
		
			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;		

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	bf_vals[0] = 1.0;  // setting the constant basis function

	for (int i = 0; i < alpha_scalar_moments; i++) 
		bf_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
	
}

void MLMTPR::CalcBasisFuncsDers(const Neighborhood& Neighborhood)
{
	if (is_sh_potential_) {
		CalcSHBasisFuncsDers(Neighborhood);
		return;
	}
	int C = species_count;						//number of different species in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	if (Neighborhood.count != moment_ders.size2)
		moment_ders.resize(alpha_moments_count, Neighborhood.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_ders.set(0);
	int type_central = Neighborhood.my_type;
	std::vector<double>& val_ = radial_vals_buffer_;
	std::vector<double>& der_ = basis_radial_ders_buffer_;
	const auto checked_eval_base = [&](int eval_block, const char* stage) -> int {
		if (eval_block < 0 || eval_block >= static_cast<int>(radial_eval_to_scaling_block_.size()))
			ERROR(std::string("CalcBasisFuncsDers: eval_block out of range at ") + stage);
		const int radial_base = eval_block * R;
		if (radial_base < 0 || radial_base + R > static_cast<int>(val_.size())
			|| radial_base + R > static_cast<int>(der_.size()))
			ERROR(std::string("CalcBasisFuncsDers: radial buffer out of range at ") + stage);
		return radial_base;
	};

	if (type_central>=species_count)
				throw MlipException("Too few species count in the MTP potential!");


	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
               // int type_outer = Neighborhood.types[j];
		// calculates vals and ders for j-th atom in the neighborhood
	//	p_RadialBasis->RB_Calc(Neighborhood.dists[j]);
	        int type_outer = Neighborhood.types[j];
	        
                
	//	int type_outer = Neighborhood.types[j];

			dist_powers_[0] = 1;
			coords_powers_x[0]=1;
			coords_powers_y[0]=1;
			coords_powers_z[0]=1;
			for (int k = 1; k < max_alpha_index_basic_; k++) {
				dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
				coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
				coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
				coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];
			}
			
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int radial_base = checked_eval_base(eval_block, "buffer fill");
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					Neighborhood.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++) {
					val_[radial_base + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					der_[radial_base + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			}
			for (int i = 0; i < alpha_index_basic_count; i++) {

				double val = 0;
				double der = 0;
				int mu = alpha_index_basic_.comp0[i];
				if (mu < 0 || mu >= static_cast<int>(mu_to_radial_eval_block_.size()))
					ERROR("CalcBasisFuncsDers: mu out of range");
				const int radial_base = checked_eval_base(mu_to_radial_eval_block_[mu], "basic accumulation");
				/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					p_RadialBasis->rb_vals[xi] *= scaling;
				for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
					p_RadialBasis->rb_ders[xi] *= scaling; */

			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
			{

				// here \phi_xi(r) is RadialBasis::vals[xi]
					val += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * val_[radial_base + xi];
					der += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * der_[radial_base + xi];


			}

			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / Neighborhood.dists[j];

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;

			mult0 *= der / Neighborhood.dists[j];
			moment_ders(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_ders(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_ders(i, j, 2) += mult0 * NeighbVect_j[2];



			if (alpha_index_basic_.comp1[i] != 0) {
				moment_ders(i, j, 0) += val * alpha_index_basic_.comp1[i]
					* coords_powers_x[alpha_index_basic_.comp1[i] - 1]
					* pow1
					* pow2;
			}
			if (alpha_index_basic_.comp2[i] != 0) {
				moment_ders(i, j, 1) += val * alpha_index_basic_.comp2[i]
					* pow0
					* coords_powers_y[alpha_index_basic_.comp2[i] - 1]
					* pow2;
			}
			if (alpha_index_basic_.comp3[i] != 0) {
				moment_ders(i, j, 2) += val * alpha_index_basic_.comp3[i]
					* pow0
					* pow1
					* coords_powers_z[alpha_index_basic_.comp3[i] - 1];
			}
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;

		for (int j = 0; j < Neighborhood.count; j++) {
			for (int a = 0; a < 3; a++) {
				moment_ders(alpha_index_times_.comp3[i], j, a) += val2 * (moment_ders(alpha_index_times_.comp0[i], j, a) * val1 + val0 * moment_ders(alpha_index_times_.comp1[i], j, a));
			}
		}
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	basis_vals[0] = 1.0;  // setting the constant basis function

	if (basis_ders.size2 != Neighborhood.count) // TODO: remove this check?
		basis_ders.resize(alpha_count, Neighborhood.count, 3);
	memset(&basis_ders(0, 0, 0), 0, 3 * Neighborhood.count*sizeof(double));

	for (int i = 0; i < alpha_scalar_moments; i++) {
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
		memcpy(&basis_ders(1 + i, 0, 0), &moment_ders(alpha_moment_mapping[i], 0, 0), 3 * Neighborhood.count*sizeof(double));
	}
}

bool MLMTPR::UsesJacobiIndexedBasis() const
{
	return IsJacobiIndexedBasisType(p_RadialBasis);
}

int MLMTPR::JacobiIndexedBlockForMu(int mu) const
{
	if (!UsesJacobiIndexedBasis())
		return 0;
	if (L < 0)
		ERROR("RBJacobi indexed basis requires non-negative L");
	// Internal L stores the number of angular channels, i.e. (L_max + 1).
	const int block_span = L;
	if (block_span <= 0)
		ERROR("RBJacobi indexed basis requires positive Jacobi block span");
	if (radial_func_count % block_span != 0)
		ERROR("RBJacobi indexed basis requires radial_funcs_count to be divisible by internal L-channel count");
	const int jacobi_block = mu / block_span;
	if (jacobi_block > 5)
		ERROR("RBJacobi indexed basis supports at most 6 Jacobi blocks: k=0..5");
	return jacobi_block;
}

void MLMTPR::MemAlloc()
{
	int n = LinearEquationCount();

	energy_cmpnts = new double[n];
	forces_cmpnts.reserve(n * 3);

	stress_cmpnts = (double(*)[3][3])malloc(n * sizeof(stress_cmpnts[0]));

	moment_vals = new double[alpha_moments_count];
	basis_vals = new double[alpha_count];
	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	grad_mom_vals_.resize(alpha_moments_count);
	grad_dloss_dsenders_.resize(alpha_moments_count);
	grad_dloss_dmom_.resize(alpha_moments_count);
	sh_adj_vals_.resize(alpha_moments_count);

	max_alpha_index_basic_ = 0;
	if (is_sh_potential_)
		max_alpha_index_basic_ = std::max(1, sh_l_max_ + 1);
	else {
		for (int i = 0; i < alpha_index_basic_count; i++)
			max_alpha_index_basic_ = max(max_alpha_index_basic_,
				alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i]);
		max_alpha_index_basic_++;
	}
	dist_powers_.resize(max_alpha_index_basic_);
	coords_powers_.resize(max_alpha_index_basic_);
	coords_powers_x.resize(max_alpha_index_basic_);
	coords_powers_y.resize(max_alpha_index_basic_);
	coords_powers_z.resize(max_alpha_index_basic_);
	grad_dist_powers_.resize(max_alpha_index_basic_);
	grad_coords_powers_x_.resize(max_alpha_index_basic_);
	grad_coords_powers_y_.resize(max_alpha_index_basic_);
	grad_coords_powers_z_.resize(max_alpha_index_basic_);
	grad_mu_contract_vals_.resize(radial_func_count);
	grad_mu_contract_ders_.resize(radial_func_count);
	grad_mu_contract_ders_s_.resize(radial_func_count);
	grad_mu_contract_ders_ss_.resize(radial_func_count);
	grad_mu_contract_coord_ders_s_.resize(radial_func_count);
	grad_mu_contract_coord_ders_ss_.resize(radial_func_count);
	lmp_radial_vals_buffer_.resize(radial_func_count);
	lmp_radial_ders_buffer_.resize(radial_func_count);
	mu_to_radial_eval_block_.resize(radial_func_count);
	radial_eval_to_scaling_block_.clear();
	radial_eval_to_basis_k_.clear();
	std::map<std::pair<int, int>, int> radial_eval_lookup;
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int scaling_block = mu_to_K[mu];
		const int basis_k = UsesJacobiIndexedBasis()
			? JacobiIndexedBlockForMu(mu)
			: mu_to_sigma[scaling_block];
		const std::pair<int, int> eval_key(scaling_block, basis_k);
		auto insert_result = radial_eval_lookup.emplace(eval_key, static_cast<int>(radial_eval_lookup.size()));
		const int eval_block = insert_result.first->second;
		mu_to_radial_eval_block_[mu] = eval_block;
		if (insert_result.second) {
			radial_eval_to_scaling_block_.push_back(scaling_block);
			radial_eval_to_basis_k_.push_back(basis_k);
		}
	}
	const int radial_eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	radial_vals_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size);
	radial_ders_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size * 5);
	basis_radial_ders_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size);
	basic_total_degree_cache_.resize(alpha_index_basic_count);
	basic_scaling_block_cache_.resize(alpha_index_basic_count);
	basic_radial_eval_block_cache_.resize(alpha_index_basic_count);
	basic_radial_base_cache_.resize(alpha_index_basic_count);
	basic_radial_deriv_base_cache_.resize(alpha_index_basic_count);
	basic_radial_offset_cache_.resize(alpha_index_basic_count);
	basic_mu_cache_.resize(alpha_index_basic_count);
	basic_sh_index_cache_.resize(alpha_index_basic_count);
	basic_sh_der_index_cache_.resize(alpha_index_basic_count);
	basic_indices_by_mu_.assign(alpha_index_basic_count, 0);
	basic_mu_offsets_.assign(radial_func_count + 1, 0);
	const int radial_coeff_base = species_count + 2 * species_count * species_count * K_;
	const int radial_stride = p_RadialBasis->rb_size + species_count;
	for (int i = 0; i < alpha_index_basic_count; i++) {
		const int mu = alpha_index_basic_.comp0[i];
		const int l = alpha_index_basic_.comp1[i];
		const int m = alpha_index_basic_.comp2[i];
		basic_total_degree_cache_[i] =
			alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
		basic_scaling_block_cache_[i] = mu_to_K[mu];
		basic_radial_eval_block_cache_[i] = mu_to_radial_eval_block_[mu];
		basic_radial_base_cache_[i] = basic_radial_eval_block_cache_[i] * p_RadialBasis->rb_size;
		basic_radial_deriv_base_cache_[i] = 5 * basic_radial_base_cache_[i];
		basic_radial_offset_cache_[i] = radial_coeff_base + mu * radial_stride;
		basic_mu_cache_[i] = mu;
		basic_sh_index_cache_[i] = l * l + m + l;
		basic_sh_der_index_cache_[i] = 3 * basic_sh_index_cache_[i];
		if (mu < 0 || mu >= radial_func_count)
			ERROR("SUS2-SH basic mu index is out of range");
		++basic_mu_offsets_[mu + 1];
	}
	for (int mu = 0; mu < radial_func_count; ++mu)
		basic_mu_offsets_[mu + 1] += basic_mu_offsets_[mu];
	std::vector<int> basic_mu_cursors = basic_mu_offsets_;
	for (int i = 0; i < alpha_index_basic_count; ++i) {
		const int mu = basic_mu_cache_[i];
		basic_indices_by_mu_[basic_mu_cursors[mu]++] = i;
	}

	if (two_layer_gate_enabled_)
		BuildTwoLayerGateProductProgram();
}


void MLMTPR::ReadSHProductGraph(std::ifstream& ifs, std::string& next_token)
{
	ifs.ignore(2);
	int product_count = 0;
	ifs >> product_count;
	if (product_count < 0)
		ERROR("SUS2-SH sh_product_count is negative");
	ifs >> next_token;
	if (next_token != "sh_products")
		ERROR("SUS2-SH model is missing sh_products");
	ifs.ignore(4);
	sh_products_.resize(product_count);
	std::vector<int> last_definition(alpha_moments_count, -1);
	for (int i = 0; i < product_count; ++i) {
		char comma;
		ifs.ignore(1000, '{');
		ifs >> sh_products_[i].left >> comma
		    >> sh_products_[i].right >> comma
		    >> sh_products_[i].target >> comma
		    >> sh_products_[i].coeff;
		if (ifs.fail())
			ERROR("Error reading SUS2-SH product graph");
		if (sh_products_[i].left < 0 || sh_products_[i].left >= alpha_moments_count
		    || sh_products_[i].right < 0 || sh_products_[i].right >= alpha_moments_count
		    || sh_products_[i].target < 0 || sh_products_[i].target >= alpha_moments_count)
			ERROR("SUS2-SH product graph index out of range");
		if (sh_products_[i].target < alpha_index_basic_count)
			ERROR("SUS2-SH product graph overwrites a basic moment");
		if (sh_products_[i].left >= sh_products_[i].target || sh_products_[i].right >= sh_products_[i].target)
			ERROR("SUS2-SH product graph is not in topological order");
		if (!std::isfinite(sh_products_[i].coeff))
			ERROR("SUS2-SH product graph contains a non-finite coefficient");
		last_definition[sh_products_[i].target] = i;
	}
		for (int i = 0; i < product_count; ++i) {
			const auto& product = sh_products_[i];
			if ((product.left >= alpha_index_basic_count && last_definition[product.left] < 0)
			    || (product.right >= alpha_index_basic_count && last_definition[product.right] < 0))
				ERROR("SUS2-SH product graph uses an undefined tensor");
			if ((product.left >= alpha_index_basic_count && last_definition[product.left] >= i)
			    || (product.right >= alpha_index_basic_count && last_definition[product.right] >= i))
				ERROR("SUS2-SH product graph execution order uses an unfinished tensor");
		}
	ifs.ignore(1000, '\n');
	ifs >> next_token;
}

void MLMTPR::BuildSHProductProgram()
{
	sh_product_rows_.clear();
	sh_product_row_terms_.clear();
	sh_scalar_index_by_moment_.assign(alpha_moments_count, -1);
	sh_scalar_terminal_product_.assign(alpha_scalar_moments, 0);
	sh_product_rows_trace_printed_ = false;
	sh_site_der_cache_trace_printed_ = false;

	if (!is_sh_potential_ || sh_products_.empty())
		return;

	for (int scalar = 0; scalar < alpha_scalar_moments; ++scalar) {
		const int moment = alpha_moment_mapping[scalar];
		if (moment >= 0 && moment < alpha_moments_count)
			sh_scalar_index_by_moment_[moment] = scalar;
	}

	std::vector<int> source_count(alpha_moments_count, 0);
	std::vector<int> row_term_count(alpha_moments_count, 0);
	std::vector<int> row_last_product(alpha_moments_count, -1);
	std::vector<std::vector<int>> product_indices_by_target(alpha_moments_count);
	for (int p = 0; p < static_cast<int>(sh_products_.size()); ++p) {
		const SHProduct& product = sh_products_[p];
		++source_count[product.left];
		++source_count[product.right];
		++row_term_count[product.target];
		row_last_product[product.target] = p;
		product_indices_by_target[product.target].push_back(p);
	}

	std::vector<int> row_targets;
	row_targets.reserve(sh_products_.size());
	for (int target = 0; target < alpha_moments_count; ++target) {
		if (row_term_count[target] > 0)
			row_targets.push_back(target);
	}
	std::sort(row_targets.begin(), row_targets.end(),
	          [&](int left, int right) {
		          return row_last_product[left] < row_last_product[right];
	          });

	sh_product_row_terms_.reserve(sh_products_.size());
	sh_product_rows_.reserve(row_targets.size());
	for (int target : row_targets) {
		SHProductRow row;
		row.target = target;
		row.term_begin = static_cast<int>(sh_product_row_terms_.size());
		row.term_count = 0;
		row.scalar_index = sh_scalar_index_by_moment_[target];
		row.terminal_scalar = row.scalar_index >= 0 && source_count[target] == 0;
		for (int product_index : product_indices_by_target[target]) {
			const SHProduct& product = sh_products_[product_index];
			SHProductRowTerm term;
			term.left = product.left;
			term.right = product.right;
			term.coeff = product.coeff;
			sh_product_row_terms_.push_back(term);
			++row.term_count;
		}
		if (row.terminal_scalar)
			sh_scalar_terminal_product_[row.scalar_index] = 1;
		sh_product_rows_.push_back(row);
	}
}

void MLMTPR::BuildTwoLayerGateProductProgram()
{
	two_layer_gate_required_moments_.clear();
	two_layer_gate_required_moment_indices_.clear();
	two_layer_gate_required_basic_indices_.clear();
	two_layer_gate_required_basic_dense_mu_indices_.clear();
	two_layer_gate_required_product_indices_.clear();
	two_layer_gate_required_mu_indices_.clear();
	two_layer_gate_mu_dense_index_.clear();
	two_layer_gate_required_radial_eval_blocks_.clear();
	if (!is_sh_potential_ || !two_layer_gate_enabled_)
		return;
	if (TwoLayerGateEdgeL1Enabled() && TwoLayerGateEdgeL1SourceCount() == 0)
		BuildTwoLayerGateEdgeL1Sources();
	if (two_layer_gate_scalar_indices_.empty()
	    && (!TwoLayerGateEdgeL1Enabled() || TwoLayerGateEdgeL1SourceCount() == 0))
		return;

	two_layer_gate_required_moments_.assign(alpha_moments_count, 0);
	for (int scalar_index : two_layer_gate_scalar_indices_) {
		if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
			ERROR("SUS2-SH two-layer gate scalar index is out of range");
		const int moment = alpha_moment_mapping[scalar_index];
		if (moment < 0 || moment >= alpha_moments_count)
			ERROR("SUS2-SH two-layer gate scalar moment is out of range");
		two_layer_gate_required_moments_[moment] = 1;
	}
	if (TwoLayerGateEdgeL1Enabled()) {
		for (int source = 0; source < TwoLayerGateEdgeL1SourceCount(); ++source) {
			const int base = TwoLayerGateEdgeL1SourceMomentIndex(source);
			for (int m = 0; m < 3; ++m)
				two_layer_gate_required_moments_[base + m] = 1;
		}
	}

	std::vector<char> required_products(sh_products_.size(), 0);
	for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
		const SHProduct& product = sh_products_[p];
		if (!two_layer_gate_required_moments_[product.target])
			continue;
		required_products[p] = 1;
		two_layer_gate_required_moments_[product.left] = 1;
		two_layer_gate_required_moments_[product.right] = 1;
	}

	const bool have_radial_caches =
		static_cast<int>(basic_mu_cache_.size()) == alpha_index_basic_count
		&& static_cast<int>(mu_to_radial_eval_block_.size()) == radial_func_count
		&& !radial_eval_to_scaling_block_.empty();
	std::vector<char> required_mu(have_radial_caches ? radial_func_count : 0, 0);
	std::vector<char> required_eval_blocks(
		have_radial_caches ? radial_eval_to_scaling_block_.size() : 0, 0);
	if (have_radial_caches)
		two_layer_gate_mu_dense_index_.assign(radial_func_count, -1);
	for (int i = 0; i < alpha_index_basic_count; ++i) {
		if (two_layer_gate_required_moments_[i]) {
			two_layer_gate_required_basic_indices_.push_back(i);
			if (have_radial_caches) {
				const int mu = basic_mu_cache_[i];
				if (mu < 0 || mu >= radial_func_count)
					ERROR("SUS2-SH two-layer gate basic mu index is out of range");
				if (!required_mu[mu]) {
					required_mu[mu] = 1;
					two_layer_gate_mu_dense_index_[mu] =
						static_cast<int>(two_layer_gate_required_mu_indices_.size());
					two_layer_gate_required_mu_indices_.push_back(mu);
				}
				two_layer_gate_required_basic_dense_mu_indices_.push_back(
					two_layer_gate_mu_dense_index_[mu]);
				const int eval_block = mu_to_radial_eval_block_[mu];
				if (eval_block < 0 || eval_block >= static_cast<int>(required_eval_blocks.size()))
					ERROR("SUS2-SH two-layer gate radial eval block is out of range");
				if (!required_eval_blocks[eval_block]) {
					required_eval_blocks[eval_block] = 1;
					two_layer_gate_required_radial_eval_blocks_.push_back(eval_block);
				}
			}
		}
	}
	for (int p = 0; p < static_cast<int>(sh_products_.size()); ++p)
		if (required_products[p])
			two_layer_gate_required_product_indices_.push_back(p);
	for (int i = 0; i < alpha_moments_count; ++i)
		if (two_layer_gate_required_moments_[i])
			two_layer_gate_required_moment_indices_.push_back(i);
}

void MLMTPR::WriteSHProductGraph(std::ofstream& ofs)
{
	ofs << "sh_product_count = " << sh_products_.size() << '\n';
	ofs << "sh_products = {";
	for (size_t i = 0; i < sh_products_.size(); ++i) {
		if (i != 0)
			ofs << ", ";
		ofs << '{' << sh_products_[i].left << ", "
		    << sh_products_[i].right << ", "
		    << sh_products_[i].target << ", "
		    << sh_products_[i].coeff << '}';
	}
	ofs << "}\n";
}

MLMTPR::~MLMTPR()
{
	if (moment_vals != NULL) delete[] moment_vals;
	if (basis_vals != NULL) delete[] basis_vals;

	if (alpha_moment_mapping != NULL) delete[] alpha_moment_mapping;
	if (alpha_index_times != NULL) delete[] alpha_index_times;
	if (alpha_index_basic != NULL) delete[] alpha_index_basic;

	moment_vals = NULL;
	basis_vals = NULL;
	alpha_moment_mapping = NULL;
	alpha_index_times = NULL;
	alpha_index_basic = NULL;

	if (energy_cmpnts != NULL) delete[] energy_cmpnts;
	if (stress_cmpnts != NULL) free(stress_cmpnts);

	if (p_RadialBasis!=NULL)
	delete p_RadialBasis;


}

void MLMTPR::RadialCoeffsInit(ifstream& ifs_rad)
{
	if (!ifs_rad.is_open())
		ERROR("Radial functions file is not open");

	string types, type1, type2;

	int pairs_count = 0;           //number of species pairs

	types = "";
	type1 = "";
	type2 = "";
	char foo = ' ';

	ifs_rad >> types;

	if (types == "radial_coeffs")
	{

		ifs_rad >> types;

		while (!ifs_rad.eof())
		{
			int i = 0;
			pairs_count++;
			regression_coeffs.resize(pairs_count*radial_func_count*(p_RadialBasis->rb_size));

			while (types.at(i) != '-')
				type1 += types.at(i++);

			while (i != types.length() - 1)
			{
				if (types.at(i + 1) != '+')
					type2 += types.at(++i);
				else
					i++;
			}


			double t;
			for (i = 0; i < radial_func_count; i++) 
			{
				ifs_rad >> foo;
				for (int j = 0; j <= p_RadialBasis->rb_size; j++)
				{
					ifs_rad >> t >> foo;
					regression_coeffs[(pairs_count - 1)*radial_func_count*(p_RadialBasis->rb_size) +
						i*(p_RadialBasis->rb_size) + j] = t;


				}


			}

			types = "";
			type1 = "";
			type2 = "";

			ifs_rad >> types;

		};

		if (species_count == 0)
			species_count = (int)sqrt(pairs_count);

		if (abs(species_count*species_count - pairs_count) > 1e-8)
			ERROR("WRONG NUMBER OF RADIAL FUNCTION BLOCKS, MUST BE species_count^2");
	}
	else
	{

		if (species_count == 0)
			species_count = 2;


		//cout << "Radial coeffs not found, initializing defaults" << endl;

		regression_coeffs.resize(species_count*species_count*radial_func_count*(p_RadialBasis->rb_size));


		for (pairs_count = 0; pairs_count < species_count*species_count; pairs_count++)//!!! pairs_count \xEF\xE5\xF0\xE5\xEE\xEF\xF0\xE5\xE4\xE5\xEB\xB8\xED
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size; j++)
					regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + j] = 1e-6;

				regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size-1)] = 1e-3;


			}

	}


}

void MLMTPR::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	if (is_sh_potential_) {
		CalcSHSiteEnergyDers(nbh);
		return;
	}
	buff_site_energy_ = 0.0;
        buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);
    //const auto& radial_list_ref = get_radial_list();
    //const auto& radial_der_list_ref = get_radial_der_list();
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function
    std::vector<double>& val_ = lmp_radial_vals_buffer_;
		std::vector<double>& der_ = lmp_radial_ders_buffer_;
	std::vector<double>& mu_contract_vals = grad_mu_contract_vals_;
	std::vector<double>& mu_contract_ders = grad_mu_contract_ders_;

	if (nbh.count != moment_jacobian_.size2)
		moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_jacobian_.set(0);


	int type_central = nbh.my_type;

	if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");


	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
//		p_RadialBasis->RB_Calc(nbh.dists[j]);
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_vals[xi] *= scaling;
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_ders[xi] *= scaling;
                int type_outer = nbh.types[j];
                
        const double r=nbh.dists[j];
		dist_powers_[0] = 1;
		coords_powers_x[0]=1;
		coords_powers_y[0]=1;
		coords_powers_z[0]=1;
		for (int k = 1; k < max_alpha_index_basic_; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * r;
			coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
			coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
			coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];
		}
		if (UsesPrecomputedLmpTable(p_RadialBasis))
		{
		int r_list;
		r_list=std::floor(r*inv_dr);
		const int r_next=r_list+1;
		const int shift=C * type_central + type_outer;
		const double ddr=r*inv_dr-r_list;



		for (int m = 0; m < K; m++)
		{
		const double v1=radial_list(shift,r_list,m);
		const double v2=radial_list(shift,r_next,m);
		const double d1=radial_der_list(shift,r_list,m);
		const double d2=radial_der_list(shift,r_next,m);

		val_[m] =v1+ddr*(v2-v1);
		der_[m] = d1+ddr*(d2-d1);

		}
		for (int i = 0; i < alpha_index_basic_count; i++)
		{
		double val = 0, der = 0;
        const int comp1=alpha_index_basic_.comp1[i];
        const int comp2=alpha_index_basic_.comp2[i];
        const int comp3=alpha_index_basic_.comp3[i];
		const int mu = alpha_index_basic_.comp0[i];
        val=val_[mu];
        der=der_[mu];
		//val =radial_list(C * type_central + type_outer,r_list,mu);
		//der = radial_der_list(C * type_central + type_outer,r_list,mu);
        const int k = comp1+comp2+comp3;
        //int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / r;

			double pow0 = coords_powers_x[comp1];
			double pow1 = coords_powers_y[comp2];
			double pow2 = coords_powers_z[comp3];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / r;
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];


            moment_jacobian_(i, j, 0) += (comp1 != 0) ?
            val * comp1 * coords_powers_x[comp1 - 1] * pow1 * pow2 : 0.0;

        moment_jacobian_(i, j, 1) += (comp2 != 0) ?
            val * comp2 * pow0 * coords_powers_y[alpha_index_basic_.comp2[i] - 1] * pow2 : 0.0;

        moment_jacobian_(i, j, 2) += (comp3 != 0) ?
            val * comp3 * pow0 * pow1 * coords_powers_z[comp3 - 1] : 0.0;



        }


		}
		else{
			std::vector<double>& val_ = radial_vals_buffer_;
			std::vector<double>& der_ = radial_ders_buffer_;
			FillWithZero(val_);
			FillWithZero(der_);
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					nbh.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					val_[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
					der_[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}
			const int shared_type_offset = C + 2 * C * C * K_ + R;
			const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
			const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double type_scale = center_type_coeff * outer_type_coeff;

			for (int mu = 0; mu < K; mu++) {
				const int eval_block = mu_to_radial_eval_block_[mu];
				const int radial_offset = C + 2 * C * C * K_ + mu * (R + C);
				const int radial_base = eval_block * R;
				const int deriv_base = 5 * radial_base;
				double dot_val = 0.0;
				double dot_der = 0.0;

			for (int xi = 0; xi < R; xi++) {
				const double radial_coeff = regression_coeffs[radial_offset + xi];
				dot_val += radial_coeff * val_[radial_base + xi];
				dot_der += radial_coeff * der_[deriv_base + xi];
			}

			mu_contract_vals[mu] = dot_val;
			mu_contract_ders[mu] = dot_der;
		}

		for (int i = 0; i < alpha_index_basic_count; i++) {
			double val = 0, der = 0;
			int mu = alpha_index_basic_.comp0[i];
			val = type_scale * mu_contract_vals[mu];
			der = type_scale * mu_contract_ders[mu];
			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_jacobian_(i, j, 0) += val * alpha_index_basic_.comp1[i]
					* coords_powers_x[alpha_index_basic_.comp1[i] - 1]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_jacobian_(i, j, 1) += val * alpha_index_basic_.comp2[i]
					* pow0
					* coords_powers_y[alpha_index_basic_.comp2[i] - 1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_jacobian_(i, j, 2) += val * alpha_index_basic_.comp3[i]
					* pow0
					* pow1
					* coords_powers_z[alpha_index_basic_.comp3[i] - 1];
			}
		}

		}

	
		//Repulsive term
		if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
		if (nbh.dists[j] < p_RadialBasis->min_dist)
		{
			double multiplier = 10000;
			buff_site_energy_ += multiplier*(exp(-10*(nbh.dists[j]-1)) - exp(-10*(p_RadialBasis->min_dist-1)));
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1))/ nbh.dists[j])*nbh.vecs[j][a];
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
	}

	// renewing maximum absolute values
		for (int i = 0; i < alpha_scalar_moments; i++)
			max_linear[i] = max(max_linear[i],abs(linear_coeffs[species_count + i]*moment_vals[alpha_moment_mapping[i]]));


	// convolving with coefficients
	buff_site_energy_ +=  regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];


	for (int i = 0; i < alpha_scalar_moments; i++)
	{	buff_site_energy_ +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
         buff_site_energy_0 +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
}

	//if (wgt_eqtn_forces != 0) //!!! CHECK IT!!!
	{
		// Backpropagation starts

		// Backpropagation step 1: site energy derivative is the corresponding linear combination
		memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

		for (int i = 0; i < alpha_scalar_moments; i++)
			site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i]*linear_mults[i];

		// SAME BUT UNSAFE:
		// memcpy(&site_energy_ders_wrt_moments_[0], &basis_coeffs[1],
		//		alpha_scalar_moments * sizeof(site_energy_ders_wrt_moments_[0]));

		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = moment_vals[alpha_index_times_.comp0[i]];
			double val1 = moment_vals[alpha_index_times_.comp1[i]];
			int val2 = alpha_index_times_.comp2[i];

			site_energy_ders_wrt_moments_[alpha_index_times_.comp1[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times_.comp0[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val1;
		}

		// Backpropagation step 3: multiply by the Jacobian:
		for (int i = 0; i < alpha_index_basic_count; i++)
			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a) * linear_coeffs[nbh.my_type];

	}
}

void MLMTPR::PrepareEvalCaches()
{
	LinCoeff();
}

void MLMTPR::AccumulateCombinationGrad(	const Neighborhood& nbh,
										std::vector<double>& out_grad_accumulator,
										const double se_weight,
										const Vector3* se_ders_weights) 
{
	if (is_sh_potential_) {
		AccumulateSHCombinationGrad(nbh, out_grad_accumulator, se_weight, se_ders_weights);
		return;
	}
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function

	int coeff_count = C+2*C*C * K_ +(R+C)*K;

	buff_site_energy_ders_.resize(nbh.count);
	out_grad_accumulator.resize(CoeffCount());
	double* grad_out = out_grad_accumulator.data();

	buff_site_energy_ = 0.0;
	FillWithZero(buff_site_energy_ders_);

	{
		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
		std::vector<double>& mom_val = grad_mom_vals_;
		std::vector<double>& dloss_dsenders = grad_dloss_dsenders_;
		std::vector<double>& dloss_dmom = grad_dloss_dmom_;
		std::vector<double>& mu_contract_vals = grad_mu_contract_vals_;
		std::vector<double>& mu_contract_ders = grad_mu_contract_ders_;
			std::vector<double>& mu_contract_ders_s = grad_mu_contract_ders_s_;
			std::vector<double>& mu_contract_ders_ss = grad_mu_contract_ders_ss_;
			std::vector<double>& mu_contract_coord_ders_s = grad_mu_contract_coord_ders_s_;
			std::vector<double>& mu_contract_coord_ders_ss = grad_mu_contract_coord_ders_ss_;
			const double* linear_scalar_coeffs = linear_coeffs.data() + species_count;
			const double* linear_mults_data = linear_mults.data();
			FillWithZero(mom_val);
			FillWithZero(site_energy_ders_wrt_moments_);
			FillWithZero(dloss_dsenders);
		FillWithZero(dloss_dmom);
		int type_central = nbh.my_type;

		if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

			const int radial_coeff_base = C + 2 * C * C * K_;
			const int shared_type_offset = radial_coeff_base + R;
			const double site_linear_coeff = linear_coeffs[nbh.my_type];
			const size_t neighbor_count = static_cast<size_t>(nbh.count);
			const size_t power_stride = static_cast<size_t>(max_alpha_index_basic_);
			const size_t radial_val_stride = static_cast<size_t>(K_) * R;
			const size_t radial_der_stride = radial_val_stride * 5;
			const size_t mu_stride = static_cast<size_t>(K);

			grad_neighbor_dist_powers_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_x_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_y_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_z_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_radial_vals_cache_.resize(neighbor_count * radial_val_stride);
			grad_neighbor_radial_ders_cache_.resize(neighbor_count * radial_der_stride);
			grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_s_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_ss_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_coord_ders_s_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_coord_ders_ss_cache_.resize(neighbor_count * mu_stride);

			struct NeighborGradCache {
				double* dist_powers;
				double* coords_powers_x;
				double* coords_powers_y;
				double* coords_powers_z;
				double* radial_vals;
				double* radial_ders;
				double* mu_contract_vals;
				double* mu_contract_ders;
				double* mu_contract_ders_s;
				double* mu_contract_ders_ss;
				double* mu_contract_coord_ders_s;
				double* mu_contract_coord_ders_ss;
			};

			auto neighbor_cache = [&](int neighbor_index) {
				const size_t offset = static_cast<size_t>(neighbor_index);
				return NeighborGradCache{
					grad_neighbor_dist_powers_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_x_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_y_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_z_cache_.data() + offset * power_stride,
					grad_neighbor_radial_vals_cache_.data() + offset * radial_val_stride,
					grad_neighbor_radial_ders_cache_.data() + offset * radial_der_stride,
					grad_neighbor_mu_contract_vals_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_s_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_ss_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_coord_ders_s_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_coord_ders_ss_cache_.data() + offset * mu_stride
				};
			};

			auto fill_neighbor_buffers = [&](int type_outer, double r, const NeighborGradCache& cache) {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; xi++)
						cache.radial_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					for (int xi = 0; xi < R * 5; xi++)
						cache.radial_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			};

			auto prepare_neighbor_state = [&](const Vector3& neighb_vec, int type_outer, double r, const NeighborGradCache& cache) {
				cache.dist_powers[0] = 1.0;
				cache.coords_powers_x[0] = 1.0;
				cache.coords_powers_y[0] = 1.0;
				cache.coords_powers_z[0] = 1.0;
				for (int k = 1; k < max_alpha_index_basic_; k++) {
					cache.dist_powers[k] = cache.dist_powers[k - 1] * r;
					cache.coords_powers_x[k] = cache.coords_powers_x[k - 1] * neighb_vec[0];
					cache.coords_powers_y[k] = cache.coords_powers_y[k - 1] * neighb_vec[1];
					cache.coords_powers_z[k] = cache.coords_powers_z[k - 1] * neighb_vec[2];
				}

				fill_neighbor_buffers(type_outer, r, cache);

					for (int mu = 0; mu < K; mu++) {
						const int eval_block = mu_to_radial_eval_block_[mu];
						const int radial_offset = radial_coeff_base + mu * (R + C);
						const int radial_base = eval_block * R;
						const int deriv_base = 5 * radial_base;
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_der_s = 0.0;
				double dot_der_ss = 0.0;
				double dot_coord_der_s = 0.0;
				double dot_coord_der_ss = 0.0;

					for (int xi = 0; xi < R; xi++) {
						const double radial_coeff = regression_coeffs[radial_offset + xi];
						dot_val += radial_coeff * cache.radial_vals[radial_base + xi];
						dot_der += radial_coeff * cache.radial_ders[deriv_base + xi];
						dot_der_s += radial_coeff * cache.radial_ders[deriv_base + xi + R];
						dot_coord_der_s += radial_coeff * cache.radial_ders[deriv_base + xi + 2 * R];
						dot_der_ss += radial_coeff * cache.radial_ders[deriv_base + xi + 3 * R];
						dot_coord_der_ss += radial_coeff * cache.radial_ders[deriv_base + xi + 4 * R];
					}

					cache.mu_contract_vals[mu] = dot_val;
					cache.mu_contract_ders[mu] = dot_der;
					cache.mu_contract_ders_s[mu] = dot_der_s;
					cache.mu_contract_ders_ss[mu] = dot_der_ss;
					cache.mu_contract_coord_ders_s[mu] = dot_coord_der_s;
					cache.mu_contract_coord_ders_ss[mu] = dot_coord_der_ss;
				}
			};

			for (int j = 0; j < nbh.count; j++) {
				const Vector3& neighb_vec = nbh.vecs[j];
				const int type_outer = nbh.types[j];
				const double r = nbh.dists[j];
				const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double type_scale = center_type_coeff * outer_type_coeff;
				const NeighborGradCache cache = neighbor_cache(j);

				prepare_neighbor_state(neighb_vec, type_outer, r, cache);

				for (int i = 0; i < alpha_index_basic_count; i++) {
					const int mu = alpha_index_basic_.comp0[i];
					const int k = basic_total_degree_cache_[i];
					const double powk = 1.0 / cache.dist_powers[k];
					const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
					const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
					const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
					const double mult0 = pow0 * pow1 * pow2;
					const double val = type_scale * cache.mu_contract_vals[mu] * powk;
					const double der = type_scale * cache.mu_contract_ders[mu];

					mom_val[i] += val * mult0;

					if (se_ders_weights != nullptr) {
					double local_der = der * powk - k * val / r;
					double jac_x = mult0 * local_der * neighb_vec[0] / r;
					double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

					dloss_dsenders[i] += se_ders_weights[j][0] * jac_x;
					dloss_dsenders[i] += se_ders_weights[j][1] * jac_y;
					dloss_dsenders[i] += se_ders_weights[j][2] * jac_z;
				}
			}

			if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
				const double multiplier = 10000.0;
				buff_site_energy_ += multiplier * (exp(-10 * (r - 1)) - exp(-10 * (p_RadialBasis->min_dist - 1)));
			}
		}

		// Next: calculating non-elementary b_i
		for (int i = 0; i < alpha_index_times_count; i++) {
			double val0 = mom_val[alpha_index_times_.comp0[i]];
			double val1 = mom_val[alpha_index_times_.comp1[i]];
			double val2 = alpha_index_times_.comp2[i];
			mom_val[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
		}

			// renewing maximum absolute values
			for (int i = 0; i < alpha_scalar_moments; i++)
				max_linear[i] = max(max_linear[i],abs(linear_scalar_coeffs[i] * mom_val[alpha_moment_mapping[i]]));


			// convolving with coefficients
			buff_site_energy_ +=   regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];
			for (int i = 0; i < alpha_scalar_moments; i++)
				buff_site_energy_ += linear_scalar_coeffs[i] * linear_mults_data[i] * mom_val[alpha_moment_mapping[i]] * linear_coeffs[nbh.my_type];

		// Backpropagation starts

			// Backpropagation step 1: site energy derivative is the corresponding linear combination
			for (int i = 0; i < alpha_scalar_moments; i++)
				site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_scalar_coeffs[i] * linear_mults_data[i];


		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = mom_val[alpha_index_times_.comp0[i]];
			double val1 = mom_val[alpha_index_times_.comp1[i]];
			double val2 = alpha_index_times_.comp2[i];

			site_energy_ders_wrt_moments_[alpha_index_times_.comp1[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times_.comp0[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val1;
		}

			for (int i = 0; i < alpha_index_basic_count; i++)
				dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];


			if (se_ders_weights)
			{

				for (int i = 0; i < alpha_index_times_count; i++) {
					const double val0 = mom_val[alpha_index_times_.comp0[i]];
				const double val1 = mom_val[alpha_index_times_.comp1[i]];
				const double val2 = alpha_index_times_.comp2[i];

				dloss_dsenders[alpha_index_times_.comp3[i]] += dloss_dsenders[alpha_index_times_.comp1[i]] * val2 * val0;
				dloss_dsenders[alpha_index_times_.comp3[i]] += dloss_dsenders[alpha_index_times_.comp0[i]] * val2 * val1;


				}

				for (int i = 0; i < alpha_index_times_count; i++) {
					const double val2 = alpha_index_times_.comp2[i];

					dloss_dmom[alpha_index_times_.comp1[i]] += dloss_dsenders[alpha_index_times_.comp0[i]] *
						site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]] * val2;

					dloss_dmom[alpha_index_times_.comp0[i]] += dloss_dsenders[alpha_index_times_.comp1[i]] *
						site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]] * val2;
				}

				for (int i = alpha_index_times_count - 1; i >= 0; i--) {
					double val0 = mom_val[alpha_index_times_.comp0[i]];
					double val1 = mom_val[alpha_index_times_.comp1[i]];
					const double val2 = alpha_index_times_.comp2[i];

					dloss_dmom[alpha_index_times_.comp0[i]] += dloss_dmom[alpha_index_times_.comp3[i]] * val2 * val1;
					dloss_dmom[alpha_index_times_.comp1[i]] += dloss_dmom[alpha_index_times_.comp3[i]] * val2 * val0;
				}

				}

			FillWithZero(buff_site_energy_ders_);
			if (se_ders_weights == nullptr) {
				for (int j = 0; j < nbh.count; j++) {
					const Vector3& neighb_vec = nbh.vecs[j];
					const int type_outer = nbh.types[j];
					const double r = nbh.dists[j];
					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const NeighborGradCache cache = neighbor_cache(j);

					for (int i = 0; i < alpha_index_basic_count; i++) {
						const int mu = alpha_index_basic_.comp0[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int radial_eval_block = basic_radial_eval_block_cache_[i];
						const int k = basic_total_degree_cache_[i];
						const int radial_offset = basic_radial_offset_cache_[i];
						const double powk = 1.0 / cache.dist_powers[k];
						const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
						const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
						const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
						const double mult0 = pow0 * pow1 * pow2;
						const double dloss_weight = site_linear_coeff * dloss_dmom[i];
						const int radial_base = radial_eval_block * R;
						const double mu_dot_val = cache.mu_contract_vals[mu];
						const double mu_dot_der = cache.mu_contract_ders[mu];
						const double mu_dot_der_s = cache.mu_contract_ders_s[mu];
						const double mu_dot_der_ss = cache.mu_contract_ders_ss[mu];
						const double radial_basis_scale = powk * mult0;
						const double val = type_scale * mu_dot_val * powk;
						const double der = type_scale * mu_dot_der;
						const double center_grad = outer_type_coeff * mu_dot_val * radial_basis_scale;
						const double outer_grad = center_type_coeff * mu_dot_val * radial_basis_scale;
						const double sigma_grad = type_scale * mu_dot_der_s * radial_basis_scale;
						const double sigma_ss_grad = type_scale * mu_dot_der_ss * radial_basis_scale;

						for (int xi = 0; xi < R; xi++) {
							const double rb_val = cache.radial_vals[radial_base + xi];
							const double basic_grad = rb_val * radial_basis_scale;
							grad_out[radial_offset + xi] += dloss_weight * basic_grad * type_scale;
						}

						double local_der = der * powk - k * val / r;
						double jac_x = mult0 * local_der * neighb_vec[0] / r;
						double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

						buff_site_energy_ders_[j][0] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_x;
						buff_site_energy_ders_[j][1] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_y;
						buff_site_energy_ders_[j][2] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_z;

						const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
						grad_out[shared_type_offset + type_central] += dloss_weight * center_grad;
						grad_out[shared_type_offset + type_outer] += dloss_weight * outer_grad;
						grad_out[sigma_coeff_offset] += dloss_weight * sigma_grad;
						grad_out[sigma_coeff_offset + C * C] += dloss_weight * sigma_ss_grad;
					}

					if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
						const double multiplier = 10000.0;
						for (int a = 0; a < 3; a++)
							buff_site_energy_ders_[j][a] += -10.0 * multiplier * (exp(-10.0 * (r - 1.0)) / r) * neighb_vec[a];
					}
				}

				if (shift_)
					grad_out[type_central] += se_weight;
				grad_out[coeff_count + type_central] += se_weight * ((buff_site_energy_ - regression_coeffs[nbh.my_type]) / linear_coeffs[nbh.my_type]);

				for (int i = 0; i < alpha_scalar_moments; i++) {
					const int moment_index = alpha_moment_mapping[i];
					grad_out[i + coeff_count + species_count] +=
						linear_coeffs[nbh.my_type] * se_weight * mom_val[moment_index] * linear_mults_data[i];
				}
			} else {
				for (int j = 0; j < nbh.count; j++) {
					const Vector3& neighb_vec = nbh.vecs[j];
					const Vector3& se_weight_vec = se_ders_weights[j];
					const int type_outer = nbh.types[j];
					const double r = nbh.dists[j];
					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const double inv_r = 1.0 / r;
					const NeighborGradCache cache = neighbor_cache(j);

					for (int i = 0; i < alpha_index_basic_count; i++) {
						const int mu = alpha_index_basic_.comp0[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int radial_eval_block = basic_radial_eval_block_cache_[i];
						const int k = basic_total_degree_cache_[i];
						const int radial_offset = basic_radial_offset_cache_[i];
						const double powk = 1.0 / cache.dist_powers[k];
						const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
						const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
						const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
						const double mult0 = pow0 * pow1 * pow2;
						const double dloss_weight = site_linear_coeff * dloss_dmom[i];
						const double coord_weight = site_linear_coeff * site_energy_ders_wrt_moments_[i];
						const int radial_base = radial_eval_block * R;
						const int deriv_base = 5 * radial_base;
						const double mu_dot_val = cache.mu_contract_vals[mu];
						const double mu_dot_der = cache.mu_contract_ders[mu];
						const double mu_dot_der_s = cache.mu_contract_ders_s[mu];
						const double mu_dot_der_ss = cache.mu_contract_ders_ss[mu];
						const double mu_dot_coord_der_s = cache.mu_contract_coord_ders_s[mu];
						const double mu_dot_coord_der_ss = cache.mu_contract_coord_ders_ss[mu];
						const double radial_basis_scale = powk * mult0;
						const double val = type_scale * mu_dot_val * powk;
						const double der = type_scale * mu_dot_der;
						double center_grad = outer_type_coeff * mu_dot_val * radial_basis_scale;
						double outer_grad = center_type_coeff * mu_dot_val * radial_basis_scale;
						double sigma_grad = type_scale * mu_dot_der_s * radial_basis_scale;
						double sigma_ss_grad = type_scale * mu_dot_der_ss * radial_basis_scale;
						double coord_center_grad = 0.0;
						double coord_outer_grad = 0.0;
						double coord_sigma_grad = 0.0;
						double coord_sigma_ss_grad = 0.0;

						for (int xi = 0; xi < R; xi++) {
							const double rb_val = cache.radial_vals[radial_base + xi];
							const double rb_der = cache.radial_ders[deriv_base + xi];
							const double rb_der_s = cache.radial_ders[deriv_base + xi + R];
							const double rb_der_coord_s = cache.radial_ders[deriv_base + xi + 2 * R];
							const double rb_der_ss = cache.radial_ders[deriv_base + xi + 3 * R];
							const double rb_der_coord_ss = cache.radial_ders[deriv_base + xi + 4 * R];
							const double basic_grad = rb_val * radial_basis_scale;
							double radial_grad = dloss_weight * basic_grad;
							double derx = neighb_vec[0] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double dery = neighb_vec[1] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double derz = neighb_vec[2] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double derx_s = neighb_vec[0] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double dery_s = neighb_vec[1] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double derz_s = neighb_vec[2] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double derx_ss = neighb_vec[0] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);
							double dery_ss = neighb_vec[1] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);
							double derz_ss = neighb_vec[2] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);

							if (alpha_index_basic_.comp1[i] != 0) {
								const double coord_term = alpha_index_basic_.comp1[i]
									* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
									* pow1 * pow2;
								derx += rb_val * powk * coord_term;
								derx_s += rb_der_s * powk * coord_term;
								derx_ss += rb_der_ss * powk * coord_term;
							}
							if (alpha_index_basic_.comp2[i] != 0) {
								const double coord_term = alpha_index_basic_.comp2[i]
									* pow0
									* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
									* pow2;
								dery += rb_val * powk * coord_term;
								dery_s += rb_der_s * powk * coord_term;
								dery_ss += rb_der_ss * powk * coord_term;
							}
							if (alpha_index_basic_.comp3[i] != 0) {
								const double coord_term = alpha_index_basic_.comp3[i]
									* pow0 * pow1
									* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];
								derz += rb_val * powk * coord_term;
								derz_s += rb_der_s * powk * coord_term;
								derz_ss += rb_der_ss * powk * coord_term;
							}

							const double coord_grad = se_weight_vec[0] * derx
								+ se_weight_vec[1] * dery
								+ se_weight_vec[2] * derz;
							const double coord_grad_s = se_weight_vec[0] * derx_s
								+ se_weight_vec[1] * dery_s
								+ se_weight_vec[2] * derz_s;
							const double coord_grad_ss = se_weight_vec[0] * derx_ss
								+ se_weight_vec[1] * dery_ss
								+ se_weight_vec[2] * derz_ss;

							radial_grad += coord_weight * coord_grad;
							grad_out[radial_offset + xi] += radial_grad * type_scale;
							coord_center_grad += outer_type_coeff * coord_grad;
							coord_outer_grad += center_type_coeff * coord_grad;
							coord_sigma_grad += type_scale * coord_grad_s;
							coord_sigma_ss_grad += type_scale * coord_grad_ss;
						}

						double local_der = der * powk - k * val / r;
						double jac_x = mult0 * local_der * neighb_vec[0] / r;
						double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

						buff_site_energy_ders_[j][0] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_x;
						buff_site_energy_ders_[j][1] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_y;
						buff_site_energy_ders_[j][2] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_z;

						double derx = neighb_vec[0] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double dery = neighb_vec[1] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double derz = neighb_vec[2] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double derx_s = neighb_vec[0] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double dery_s = neighb_vec[1] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double derz_s = neighb_vec[2] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double derx_ss = neighb_vec[0] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);
						double dery_ss = neighb_vec[1] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);
						double derz_ss = neighb_vec[2] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);

						if (alpha_index_basic_.comp1[i] != 0) {
							const double coord_term = alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
							derx += mu_dot_val * powk * coord_term;
							derx_s += mu_dot_der_s * powk * coord_term;
							derx_ss += mu_dot_der_ss * powk * coord_term;
						}
						if (alpha_index_basic_.comp2[i] != 0) {
							const double coord_term = alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
							dery += mu_dot_val * powk * coord_term;
							dery_s += mu_dot_der_s * powk * coord_term;
							dery_ss += mu_dot_der_ss * powk * coord_term;
						}
						if (alpha_index_basic_.comp3[i] != 0) {
							const double coord_term = alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];
							derz += mu_dot_val * powk * coord_term;
							derz_s += mu_dot_der_s * powk * coord_term;
							derz_ss += mu_dot_der_ss * powk * coord_term;
						}

						const double coord_grad = se_weight_vec[0] * derx
							+ se_weight_vec[1] * dery
							+ se_weight_vec[2] * derz;
						const double coord_grad_s = se_weight_vec[0] * derx_s
							+ se_weight_vec[1] * dery_s
							+ se_weight_vec[2] * derz_s;
						const double coord_grad_ss = se_weight_vec[0] * derx_ss
							+ se_weight_vec[1] * dery_ss
							+ se_weight_vec[2] * derz_ss;

						const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
						grad_out[shared_type_offset + type_central] += dloss_weight * center_grad + coord_weight * outer_type_coeff * coord_grad;
						grad_out[shared_type_offset + type_outer] += dloss_weight * outer_grad + coord_weight * center_type_coeff * coord_grad;
						grad_out[sigma_coeff_offset] += dloss_weight * sigma_grad + coord_weight * type_scale * coord_grad_s;
						grad_out[sigma_coeff_offset + C * C] += dloss_weight * sigma_ss_grad + coord_weight * type_scale * coord_grad_ss;
					}

					if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
						const double multiplier = 10000.0;
						for (int a = 0; a < 3; a++)
							buff_site_energy_ders_[j][a] += -10.0 * multiplier * (exp(-10.0 * (r - 1.0)) / r) * neighb_vec[a];
					}
				}

				if (shift_)
					grad_out[type_central] += se_weight;
				double center_linear_grad = se_weight * ((buff_site_energy_ - regression_coeffs[nbh.my_type]) / linear_coeffs[nbh.my_type]);
				for (int i = 0; i < alpha_scalar_moments; i++)
					center_linear_grad += dloss_dsenders[alpha_moment_mapping[i]] * linear_mults_data[i] * linear_scalar_coeffs[i];
				grad_out[coeff_count + type_central] += center_linear_grad;

				for (int i = 0; i < alpha_scalar_moments; i++) {
					const int moment_index = alpha_moment_mapping[i];
					const double scalar_grad = se_weight * mom_val[moment_index] + dloss_dsenders[moment_index];
					grad_out[i + coeff_count + species_count] +=
						linear_coeffs[nbh.my_type] * scalar_grad * linear_mults_data[i];
				}
			}

	}

}

void MLMTPR::AddRadialSmoothnessPenalty(const double coeff,
										const int grid_size,
										double& out_penalty_accumulator,
										Array1D* out_penalty_grad_accumulator)
{
	if (coeff == 0.0)
		return;
	if (coeff < 0.0)
		ERROR("Radial smoothness penalty coefficient should be >= 0");
	if (grid_size <= 0)
		ERROR("Radial smoothness grid size should be > 0");
	if (!is_sh_potential_)
		return;
	if (p_RadialBasis == nullptr || radial_func_count <= 0 || species_count <= 0)
		return;
	const std::string rb_type = p_RadialBasis->GetRBTypeString();
	if (rb_type.find("_lmp") != std::string::npos)
		ERROR("Radial smoothness penalty requires a training radial basis with scal/s derivatives; set --radial-smooth=0 for *_lmp radial bases");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int radial_begin = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int radial_end = radial_begin + radial_func_count * block_size;
	if (R <= 0 || block_size <= 0 || radial_begin < 0 ||
	    radial_end > static_cast<int>(regression_coeffs.size()))
		return;

	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	const double cutoff = p_RadialBasis->max_dist;
	if (!(cutoff > 0.0))
		ERROR("Radial smoothness penalty requires a positive cutoff");
	const double weight = coeff * cutoff
	                    / (static_cast<double>(grid_size)
	                       * static_cast<double>(radial_func_count)
	                       * static_cast<double>(C)
	                       * static_cast<double>(C));
	const int shared_type_offset = radial_begin + R;

	for (int type_central = 0; type_central < C; ++type_central) {
		for (int type_outer = 0; type_outer < C; ++type_outer) {
			for (int q = 0; q < grid_size; ++q) {
				const double r = cutoff * (static_cast<double>(q) + 0.5)
				               / static_cast<double>(grid_size);
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int scaling_block = mu_to_K[mu];
					const int basis_k = UsesJacobiIndexedBasis()
						? JacobiIndexedBlockForMu(mu)
						: mu_to_sigma[scaling_block];
					const int slope_offset = ScalingSlopeOffset(scaling_block, type_central, type_outer);
					const int shift_offset = ScalingShiftOffset(scaling_block, type_central, type_outer);
					const int radial_offset = radial_begin + mu * block_size;

					p_RadialBasis->RB_Calc(r,
						regression_coeffs[slope_offset],
						regression_coeffs[shift_offset],
						basis_k);

					double dot_der = 0.0;
					double dot_der_slope = 0.0;
					double dot_der_shift = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double radial_coeff = regression_coeffs[radial_offset + xi];
						dot_der += radial_coeff * p_RadialBasis->rb_ders[xi] * scaling;
						dot_der_slope += radial_coeff * p_RadialBasis->rb_ders[2 * R + xi] * scaling;
						dot_der_shift += radial_coeff * p_RadialBasis->rb_ders[4 * R + xi] * scaling;
					}

					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const double radial_der = type_scale * dot_der;

					out_penalty_accumulator += weight * radial_der * radial_der;
					if (out_penalty_grad_accumulator == nullptr)
						continue;

					const double common_grad = 2.0 * weight * radial_der;
					for (int xi = 0; xi < R; ++xi)
						(*out_penalty_grad_accumulator)[radial_offset + xi] +=
							common_grad * type_scale * p_RadialBasis->rb_ders[xi] * scaling;
					(*out_penalty_grad_accumulator)[shared_type_offset + type_central] +=
						common_grad * outer_type_coeff * dot_der;
					(*out_penalty_grad_accumulator)[shared_type_offset + type_outer] +=
						common_grad * center_type_coeff * dot_der;
					(*out_penalty_grad_accumulator)[slope_offset] +=
						common_grad * type_scale * dot_der_slope;
					(*out_penalty_grad_accumulator)[shift_offset] +=
						common_grad * type_scale * dot_der_shift;
				}
			}
		}
	}
}

void MLMTPR::AddFixedAtomicEnergyPenalty(const std::vector<double>& atomic_energies,
										const double coeff,
										double& out_penalty_accumulator,
										Array1D* out_penalty_grad_accumulator)
{
	if (atomic_energies.empty() || coeff == 0.0)
		return;
	if (coeff < 0.0)
		ERROR("Fixed atomic energy penalty coefficient should be >= 0");
	if (static_cast<int>(atomic_energies.size()) != species_count)
		ERROR("Fixed atomic energy count should match species_count");

	const int linear_begin = LinearCoeffOffset();
	if (linear_begin < 0 || linear_begin + species_count > CoeffCount())
		ERROR("Invalid MTPR linear coefficient layout for fixed atomic energy penalty");

	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	const double weight = coeff / static_cast<double>(species_count);
	for (int type = 0; type < species_count; ++type) {
		const int species_coeff_idx = linear_begin + type;
		const double diff =
			regression_coeffs[type] + regression_coeffs[species_coeff_idx] - atomic_energies[type];
		out_penalty_accumulator += weight * diff * diff;
		if (out_penalty_grad_accumulator != nullptr) {
			const double grad = 2.0 * weight * diff;
			(*out_penalty_grad_accumulator)[type] += grad;
			(*out_penalty_grad_accumulator)[species_coeff_idx] += grad;
		}
	}
}

void MLMTPR::AddScalarWeightL2Penalty(const double head_coeff,
									  const double gate_scalar_coeff,
									  const double gate_mix_coeff,
									  const double gate_full_coeff,
									  double& out_penalty_accumulator,
									  Array1D* out_penalty_grad_accumulator)
{
	if (!std::isfinite(head_coeff) || head_coeff < 0.0)
		ERROR("Scalar head L2 penalty coefficient should be finite and >= 0");
	if (!std::isfinite(gate_scalar_coeff) || gate_scalar_coeff < 0.0)
		ERROR("Gate scalar L2 penalty coefficient should be finite and >= 0");
	if (!std::isfinite(gate_mix_coeff) || gate_mix_coeff < 0.0)
		ERROR("Gate mix L2 penalty coefficient should be finite and >= 0");
	if (!std::isfinite(gate_full_coeff) || gate_full_coeff < 0.0)
		ERROR("Gate full L2 penalty coefficient should be finite and >= 0");
	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	if (head_coeff != 0.0 && alpha_scalar_moments > 0) {
		const int linear_begin = LinearCoeffOffset();
		const int scalar_begin = linear_begin + species_count;
		if (scalar_begin < 0
		    || scalar_begin + alpha_scalar_moments > static_cast<int>(regression_coeffs.size()))
			ERROR("Invalid MTPR scalar head coefficient layout for L2 penalty");
		const double weight = head_coeff / static_cast<double>(alpha_scalar_moments);
		for (int q = 0; q < alpha_scalar_moments; ++q) {
			const int coeff_index = scalar_begin + q;
			const double mult = q < static_cast<int>(linear_mults.size())
				? linear_mults[q]
				: 1.0;
			const double value = regression_coeffs[coeff_index] * mult;
			out_penalty_accumulator += weight * value * value;
			if (out_penalty_grad_accumulator != nullptr)
				(*out_penalty_grad_accumulator)[coeff_index] +=
					2.0 * weight * value * mult;
		}
	}

	if (TwoLayerGateUsesBodyLinearCombo()) {
		const int gate_weight_count = TwoLayerGateWeightCount();
		if (gate_scalar_coeff != 0.0 && gate_weight_count > 0) {
			const int gate_weight_offset = TwoLayerGateWeightOffset();
			if (gate_weight_offset < 0
			    || gate_weight_offset + gate_weight_count > static_cast<int>(regression_coeffs.size()))
				ERROR("Invalid SUS2-SH gate scalar weight layout for L2 penalty");
			const double weight = gate_scalar_coeff / static_cast<double>(gate_weight_count);
			for (int i = 0; i < gate_weight_count; ++i) {
				const int coeff_index = gate_weight_offset + i;
				const double value = regression_coeffs[coeff_index];
				out_penalty_accumulator += weight * value * value;
				if (out_penalty_grad_accumulator != nullptr)
					(*out_penalty_grad_accumulator)[coeff_index] +=
						2.0 * weight * value;
			}
		}

		const int gate_mix_count = TwoLayerGateBodyMixWeightCount();
		if (gate_mix_coeff != 0.0 && gate_mix_count > 0) {
			const int gate_mix_offset = TwoLayerGateBodyMixWeightOffset();
			if (gate_mix_offset < 0
			    || gate_mix_offset + gate_mix_count > static_cast<int>(regression_coeffs.size()))
				ERROR("Invalid SUS2-SH gate body-mix weight layout for L2 penalty");
			const double weight = gate_mix_coeff / static_cast<double>(gate_mix_count);
			for (int i = 0; i < gate_mix_count; ++i) {
				const int coeff_index = gate_mix_offset + i;
				const double diff = regression_coeffs[coeff_index] - 1.0;
				out_penalty_accumulator += weight * diff * diff;
				if (out_penalty_grad_accumulator != nullptr)
					(*out_penalty_grad_accumulator)[coeff_index] +=
						2.0 * weight * diff;
			}
		}
	} else if (TwoLayerGateUsesFullScalarWeights()) {
		const int gate_weight_count = TwoLayerGateWeightCount();
		if (gate_full_coeff != 0.0 && gate_weight_count > 0) {
			const int gate_weight_offset = TwoLayerGateWeightOffset();
			if (gate_weight_offset < 0
			    || gate_weight_offset + gate_weight_count > static_cast<int>(regression_coeffs.size()))
				ERROR("Invalid SUS2-SH full gate weight layout for L2 penalty");
			const double weight = gate_full_coeff / static_cast<double>(gate_weight_count);
			for (int i = 0; i < gate_weight_count; ++i) {
				const int coeff_index = gate_weight_offset + i;
				const double value = regression_coeffs[coeff_index];
				out_penalty_accumulator += weight * value * value;
				if (out_penalty_grad_accumulator != nullptr)
					(*out_penalty_grad_accumulator)[coeff_index] +=
						2.0 * weight * value;
			}
		}
	}
}

void MLMTPR::AddPenaltyGrad(const double coeff,													// Must calculate add penalty and optionally (if out_penalty_grad_accumulator != nullptr) its gradient w.r.t. coefficients multiplied by coeff to out_penalty
							double& out_penalty_accumulator,
							Array1D* out_penalty_grad_accumulator)
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	for (int k = 0; k < K; k++)
	{
		double norm = 0;

		for (int i = 0; i < 1; i++)
			for (int j = 0; j < 1; j++)
				for (int l = 0; l < R; l++)
					norm += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];


		out_penalty_accumulator += coeff * (norm - 1) * (norm - 1);

		if (out_penalty_grad_accumulator != nullptr)
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] += coeff * 4 * (norm - 1) * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];
	}



	for (int k1 = 0; k1 < K; k1++)
		for (int k2 = k1 + 1; k2 < K; k2++)
		{
			double scal = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						scal += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*(R+C) + k2*(R+C) + l];

			out_penalty_accumulator += coeff * scal*scal;

			if (out_penalty_grad_accumulator != nullptr)
				for (int i = 0; i < 1; i++)
					for (int j = 0; j < 1; j++)
						for (int l = 0; l < R; l++)
						{
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l];
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l];
						}
		}
}

void MLMTPR::Orthogonalize()
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	for (int k = 0; k < K; k++)
	{
//		for (int k2 = 0; k2 < k; k2++) {
//			double norm = 0;
//
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						norm += regression_coeffs[  k*(R+C) + l] * regression_coeffs[ k*(R+C) + l];
//
//			double scal = 0;
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						scal += regression_coeffs[ k*(R+C) + l] * regression_coeffs[ k2*(R+C) + l];
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						regression_coeffs[ k*(R+C) + l] -= regression_coeffs[ k2*(R+C) + l] * scal;
//		}


		{
			double norm = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						norm += regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l] * regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l]+1e-30;

			norm = sqrt(norm);
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						regression_coeffs[C+2*C*C * K_ + k*(R+C) + l] /= norm;
		}
	}
}

void MLMTPR::Perform_scaling()
{
	//#ifdef MLIP_MPI
	//#else
	//#endif

	int C = species_count;
	int K = radial_func_count;
	int R = p_RadialBasis->rb_size;

	std::vector<double> total_max(alpha_scalar_moments);
	total_max.clear();

	for (int j = 0; j < alpha_scalar_moments; j++)
	{

#ifdef MLIP_MPI
		MPI_Reduce(&max_linear[j], &total_max[j], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&total_max[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		max_linear[j] = total_max[j];
#endif
		regression_coeffs[(R+C)*K + j + species_count] *= linear_mults[j];

		if (max_linear[j] > 0)
			linear_mults[j] = 1e10 / max_linear[j];
		else
			linear_mults[j] = 1e10;

		linear_mults[j] = max(linear_mults[j], 1e-14);
		linear_mults[j] = min(linear_mults[j], 1e14);

		//cout << "mult=" << linear_mults[j] << endl;

		regression_coeffs[R*K + j + species_count] /= linear_mults[j];
		max_linear[j] = 1e-10;
	}
}
