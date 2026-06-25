/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../src/mlp/mlp.h"

#include "../../src/common/stdafx.h"
#include "../../src/mlip_wrapper.h"
#include "../../dev_src/mlp/dev_self_test.h"
#include "../../src/mlp/self_test.h"
#include "../../src/mlp/mlp.h"
#include "../../src/mlp/train.h"
#include "../../src/mlp/calc_errors.h"
#include "../mtpr.h" 
#include "../mtpr_trainer.h" 
#include "../sh_model_init.h"
#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace std;

namespace {

double ParseDevDoubleOption(const std::map<std::string, std::string>& opts,
                            const std::string& name,
                            double default_value)
{
	const auto it = opts.find(name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return std::stod(it->second);
}

int ParseDevIntOption(const std::map<std::string, std::string>& opts,
                      const std::string& name,
                      int default_value)
{
	const auto it = opts.find(name);
	if (it == opts.end() || it->second.empty())
		return default_value;
	return std::stoi(it->second);
}

int ParseDevResidualStageOption(const std::map<std::string, std::string>& opts)
{
	const auto it = opts.find("residual-stage");
	if (it == opts.end() || it->second.empty() || it->second == "full")
		return MTPR_trainer::kResidualStageFull;
	if (it->second == "e0" || it->second == "E0")
		return MTPR_trainer::kResidualStageE0;
	if (it->second == "e1" || it->second == "E1")
		return MTPR_trainer::kResidualStageE1;
	ERROR("--residual-stage should be one of: full, e0, e1");
}

const char* DevResidualStageName(int stage)
{
	switch (stage) {
	case MTPR_trainer::kResidualStageE0:
		return "e0";
	case MTPR_trainer::kResidualStageE1:
		return "e1";
	case MTPR_trainer::kResidualStageFull:
	default:
		return "full";
	}
}

class DevLossGradientProbe : public MTPR_trainer {
public:
	struct Result {
		int checked_count = 0;
		int worst_index = -1;
		double worst_rel_err = 0.0;
		double worst_abs_err = 0.0;
		double worst_analytic = 0.0;
		double worst_fd = 0.0;
		double base_loss = 0.0;
	};

	struct DirectionalResult {
		int checked_count = 0;
		double analytic = 0.0;
		double finite_difference = 0.0;
		double abs_err = 0.0;
		double rel_err = 0.0;
		double base_loss = 0.0;
	};

	int residual_stage = kResidualStageFull;
	bool stage_active_only = false;

	DevLossGradientProbe(MLMTPR* mtpr,
	                     double energy_weight,
	                     double force_weight,
	                     double stress_weight)
		: MTPR_trainer(mtpr, energy_weight, force_weight, stress_weight,
		               0.0, 0.0)
	{
		std_scaling = 0.0;
		stdd_scaling = 0.0;
		radial_smooth_regularization = 0.0;
		scalar_head_l2_regularization = 0.0;
		gate_scalar_l2_regularization = 0.0;
		gate_mix_l2_regularization = 0.0;
		gate_full_l2_regularization = 0.0;
		fixed_atomic_energies.clear();
	}

	bool CoeffActiveForProbeStage(int coeff_index) const
	{
		MLMTPR* mtpr = dynamic_cast<MLMTPR*>(p_mlip);
		if (mtpr == nullptr
		    || !mtpr->TwoLayerResidualEnabled()
		    || residual_stage == kResidualStageFull)
			return true;
		switch (residual_stage) {
		case kResidualStageE0: {
			const int base_end = mtpr->BaseNonlinearCoeffCount();
			const int e0_begin = mtpr->TwoLayerResidualE0CoeffOffset();
			const int e0_end = e0_begin + mtpr->TwoLayerResidualE0CoeffCount();
			const int linear_begin = mtpr->LinearCoeffOffset();
			const int species_linear_end = linear_begin + mtpr->species_count;
			return coeff_index < base_end
			    || (coeff_index >= e0_begin && coeff_index < e0_end)
			    || (coeff_index >= linear_begin && coeff_index < species_linear_end);
		}
			case kResidualStageE1: {
				const int gate_radial_begin = mtpr->TwoLayerGateRadialCoeffOffset();
				const int gate_radial_end =
					gate_radial_begin + mtpr->TwoLayerGateRadialCoeffCount();
				const int gate_additive_begin = mtpr->TwoLayerGateAdditiveCoeffOffset();
				const int gate_additive_end =
					gate_additive_begin + mtpr->TwoLayerGateAdditiveCoeffCount();
				const int gate_weight_begin = mtpr->TwoLayerGateWeightOffset();
				const int gate_weight_end =
					gate_weight_begin + mtpr->TwoLayerGateWeightCount();
				const int gate_edge_l1_begin = mtpr->TwoLayerGateEdgeL1WeightOffset();
				const int gate_edge_l1_end =
					gate_edge_l1_begin + mtpr->TwoLayerGateEdgeL1WeightCount();
				const int linear_begin = mtpr->LinearCoeffOffset();
				const int linear_end = linear_begin + mtpr->LinearCoeffCount();
				return (coeff_index >= gate_radial_begin && coeff_index < gate_radial_end)
				    || (coeff_index >= gate_additive_begin && coeff_index < gate_additive_end)
				    || (coeff_index >= gate_weight_begin && coeff_index < gate_weight_end)
				    || (coeff_index >= gate_edge_l1_begin && coeff_index < gate_edge_l1_end)
				    || (coeff_index >= linear_begin + mtpr->species_count
				        && coeff_index < linear_end);
		}
		default:
			return true;
		}
	}

	Result Check(std::vector<Configuration>& configs,
	             double displacement,
	             double abs_tolerance,
	             double rel_tolerance,
	             int coeff_begin,
	             int coeff_end)
	{
		Result result;
		MLMTPR* mtpr = dynamic_cast<MLMTPR*>(p_mlip);
		const int saved_stage =
			(mtpr == nullptr) ? 0 : mtpr->two_layer_residual_eval_stage_;
		if (mtpr != nullptr)
			mtpr->two_layer_residual_eval_stage_ =
				mtpr->TwoLayerResidualEnabled() ? residual_stage : 0;
		CalcObjectiveFunctionGrad(configs);
		std::vector<double> analytic = loss_grad_;
		result.base_loss = ObjectiveFunction(configs);
		if (mtpr != nullptr)
			mtpr->two_layer_residual_eval_stage_ = saved_stage;
		const int coeff_count = p_mlip->CoeffCount();
		coeff_begin = std::max(0, coeff_begin);
		coeff_end = coeff_end <= 0 ? coeff_count : std::min(coeff_end, coeff_count);
		if (coeff_begin >= coeff_end)
			ERROR("Invalid coefficient range for loss-gradient check.");
		for (int i = coeff_begin; i < coeff_end; ++i) {
			if (stage_active_only && !CoeffActiveForProbeStage(i))
				continue;
			const double original = p_mlip->Coeff()[i];
			p_mlip->Coeff()[i] = original + displacement;
			const double loss_plus = ObjectiveFunction(configs);
			p_mlip->Coeff()[i] = original - displacement;
			const double loss_minus = ObjectiveFunction(configs);
			p_mlip->Coeff()[i] = original;

			const double fd = (loss_plus - loss_minus) / (2.0 * displacement);
			const double abs_err = std::abs(analytic[i] - fd);
			const double scale = std::max(1.0e-14,
				std::max(std::abs(analytic[i]), std::abs(fd)));
			const double rel_err = abs_err / scale;
			++result.checked_count;
			if (abs_err > abs_tolerance && rel_err > result.worst_rel_err) {
				result.worst_rel_err = rel_err;
				result.worst_abs_err = abs_err;
				result.worst_index = i;
				result.worst_analytic = analytic[i];
				result.worst_fd = fd;
			}
		}
		if (result.checked_count == 0)
			ERROR("No coefficients selected for loss-gradient check.");
		if (result.worst_abs_err <= abs_tolerance)
			result.worst_rel_err = 0.0;
		return result;
	}

	DirectionalResult CheckDirectional(std::vector<Configuration>& configs,
	                                   double displacement,
	                                   int coeff_begin,
	                                   int coeff_end)
	{
		DirectionalResult result;
		MLMTPR* mtpr = dynamic_cast<MLMTPR*>(p_mlip);
		const int saved_stage =
			(mtpr == nullptr) ? 0 : mtpr->two_layer_residual_eval_stage_;
		if (mtpr != nullptr)
			mtpr->two_layer_residual_eval_stage_ =
				mtpr->TwoLayerResidualEnabled() ? residual_stage : 0;
		CalcObjectiveFunctionGrad(configs);
		std::vector<double> analytic = loss_grad_;
		result.base_loss = ObjectiveFunction(configs);
		if (mtpr != nullptr)
			mtpr->two_layer_residual_eval_stage_ = saved_stage;

		const int coeff_count = p_mlip->CoeffCount();
		coeff_begin = std::max(0, coeff_begin);
		coeff_end = coeff_end <= 0 ? coeff_count : std::min(coeff_end, coeff_count);
		if (coeff_begin >= coeff_end)
			ERROR("Invalid coefficient range for directional loss-gradient check.");

		std::vector<double> direction(coeff_count, 0.0);
		for (int i = coeff_begin; i < coeff_end; ++i) {
			if (stage_active_only && !CoeffActiveForProbeStage(i))
				continue;
			const int phase = (i - coeff_begin) % 5;
			const double value = static_cast<double>(phase - 2);
			direction[i] = (value == 0.0) ? 1.0 : value;
			result.analytic += analytic[i] * direction[i];
			++result.checked_count;
		}
		if (result.checked_count == 0)
			ERROR("Directional loss-gradient check found no active coefficients.");

		for (int i = coeff_begin; i < coeff_end; ++i)
			p_mlip->Coeff()[i] += displacement * direction[i];
		const double loss_plus = ObjectiveFunction(configs);
		for (int i = coeff_begin; i < coeff_end; ++i)
			p_mlip->Coeff()[i] -= 2.0 * displacement * direction[i];
		const double loss_minus = ObjectiveFunction(configs);
		for (int i = coeff_begin; i < coeff_end; ++i)
			p_mlip->Coeff()[i] += displacement * direction[i];

		result.finite_difference = (loss_plus - loss_minus) / (2.0 * displacement);
		result.abs_err = std::abs(result.analytic - result.finite_difference);
		const double scale = std::max(1.0e-14,
			std::max(std::abs(result.analytic), std::abs(result.finite_difference)));
		result.rel_err = result.abs_err / scale;
		return result;
	}
};

struct DevEFSFDResult {
	int force_components = 0;
	int stress_components = 0;
	int worst_force_config = -1;
	int worst_force_atom = -1;
	int worst_force_component = -1;
	int worst_stress_config = -1;
	int worst_stress_a = -1;
	int worst_stress_b = -1;
	double worst_force_abs_err = 0.0;
	double worst_force_rel_err = 0.0;
	double worst_force_analytic = 0.0;
	double worst_force_fd = 0.0;
	double worst_stress_abs_err = 0.0;
	double worst_stress_rel_err = 0.0;
	double worst_stress_analytic = 0.0;
	double worst_stress_fd = 0.0;
};

struct DevLinearComponentsFDResult {
	int checked_components = 0;
	int worst_config = -1;
	int worst_atom = -1;
	int worst_component = -1;
	int worst_coeff = -1;
	double worst_abs_err = 0.0;
	double worst_rel_err = 0.0;
	double worst_analytic = 0.0;
	double worst_fd = 0.0;
};

struct DevLinearReadoutResult {
	int energy_components = 0;
	int force_components = 0;
	int worst_energy_config = -1;
	int worst_force_config = -1;
	int worst_force_atom = -1;
	int worst_force_component = -1;
	double worst_energy_abs_err = 0.0;
	double worst_energy_rel_err = 0.0;
	double worst_energy_direct = 0.0;
	double worst_energy_linear = 0.0;
	double worst_force_abs_err = 0.0;
	double worst_force_rel_err = 0.0;
	double worst_force_direct = 0.0;
	double worst_force_linear = 0.0;
};

struct DevGateFastPathResult {
	int checked_values = 0;
	int checked_derivatives = 0;
	int worst_value_config = -1;
	int worst_value_atom = -1;
	int worst_value_q = -1;
	int worst_der_config = -1;
	int worst_der_atom = -1;
	int worst_der_q = -1;
	int worst_der_neighbor = -1;
	int worst_der_component = -1;
	double worst_value_abs_err = 0.0;
	double worst_value_rel_err = 0.0;
	double worst_value_full = 0.0;
	double worst_value_fast = 0.0;
	double worst_der_abs_err = 0.0;
	double worst_der_rel_err = 0.0;
	double worst_der_full = 0.0;
	double worst_der_fast = 0.0;
	double worst_weighted_der_abs_err = 0.0;
	double worst_weighted_der_rel_err = 0.0;
	double worst_weighted_der_full = 0.0;
	double worst_weighted_der_fast = 0.0;
};

struct DevGateMuBodyOrderResult {
	int checked_values = 0;
	int checked_body_orders = 0;
	int active_body_orders = 0;
	int worst_body_order = -1;
	int worst_config = -1;
	int worst_atom = -1;
	int worst_mu = -1;
	double worst_abs_err = 0.0;
	double worst_rel_err = 0.0;
	double worst_expected = 0.0;
	double worst_actual = 0.0;
};

void UpdateWorst(double analytic,
                 double finite_difference,
                 double& worst_abs_err,
                 double& worst_rel_err,
                 double& worst_analytic,
                 double& worst_fd)
{
	const double abs_err = std::abs(analytic - finite_difference);
	const double scale = std::max(1.0e-14,
		std::max(std::abs(analytic), std::abs(finite_difference)));
	const double rel_err = abs_err / scale;
	if (abs_err > worst_abs_err) {
		worst_abs_err = abs_err;
		worst_rel_err = rel_err;
		worst_analytic = analytic;
		worst_fd = finite_difference;
	}
}

class DevGateFastPathProbe : public MLMTPR {
public:
	DevGateFastPathProbe(const std::string& mtp_filename)
		: MLMTPR(mtp_filename)
	{
	}

	DevGateFastPathResult Check(std::vector<Configuration>& configs,
	                            int max_atoms)
	{
		if (!two_layer_gate_enabled_)
			ERROR("Model does not have two-layer gate enabled.");
		if (TwoLayerGateWeightCount() <= 0)
			ERROR("Model has no two-layer gate weights.");
		if (TwoLayerGateUsesSharedRadial())
			ERROR("check-two-layer-gate-fastpath-dev compares gate scalars to the full outer SH path and is only valid for legacy shared radial coefficients.");

		DevGateFastPathResult result;
		std::vector<double> fast_values;
		std::vector<double> fast_derivatives;
		std::vector<Vector3> fast_weighted_derivatives;
		std::vector<double> full_values(TwoLayerGateScalarCount(), 0.0);
		std::vector<double> full_derivatives;
		std::vector<Vector3> full_weighted_derivatives;

		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
		active_two_layer_gate_values_ = nullptr;
		active_two_layer_gate_adjoints_ = nullptr;

		for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size()); ++cfg_index) {
			Neighborhoods neighborhoods(configs[cfg_index], CutOff());
			const int atom_limit = max_atoms <= 0
				? configs[cfg_index].size()
				: std::min(max_atoms, configs[cfg_index].size());
			for (int atom_index = 0; atom_index < atom_limit; ++atom_index) {
				const Neighborhood& nbh = neighborhoods[atom_index];

				CalcSHBasisFuncs(nbh, basis_vals);
				for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
					const int scalar_index = two_layer_gate_scalar_indices_[q];
					full_values[q] = basis_vals[1 + scalar_index];
				}
				CalcTwoLayerGateScalarValuesOnly(nbh, fast_values);
				for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
					const double old_abs = result.worst_value_abs_err;
					UpdateWorst(full_values[q], fast_values[q],
					            result.worst_value_abs_err,
					            result.worst_value_rel_err,
					            result.worst_value_full,
					            result.worst_value_fast);
					if (result.worst_value_abs_err > old_abs) {
						result.worst_value_config = cfg_index;
						result.worst_value_atom = atom_index;
						result.worst_value_q = q;
					}
					++result.checked_values;
				}

				CalcSHBasisFuncsDers(nbh);
				full_derivatives.assign(
					static_cast<size_t>(TwoLayerGateScalarCount()) * nbh.count * 3,
					0.0);
				full_weighted_derivatives.assign(
					static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));
				for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
					const int scalar_index = two_layer_gate_scalar_indices_[q];
					double weight = 0.0;
					for (int mu = 0; mu < radial_func_count; ++mu)
						weight += TwoLayerGateWeight(mu, q);
					for (int j = 0; j < nbh.count; ++j) {
						for (int a = 0; a < 3; ++a) {
							const double full_der = basis_ders(1 + scalar_index, j, a);
							full_derivatives[(static_cast<size_t>(q) * nbh.count + j) * 3 + a] =
								full_der;
							full_weighted_derivatives[j][a] += weight * full_der;
						}
					}
				}

				CalcTwoLayerGateScalarDers(nbh, fast_derivatives);
				CalcTwoLayerGateWeightedScalarDers(nbh, fast_weighted_derivatives);
				for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
					for (int j = 0; j < nbh.count; ++j) {
						for (int a = 0; a < 3; ++a) {
							const size_t index =
								(static_cast<size_t>(q) * nbh.count + j) * 3 + a;
							const double old_abs = result.worst_der_abs_err;
							UpdateWorst(full_derivatives[index],
							            fast_derivatives[index],
							            result.worst_der_abs_err,
							            result.worst_der_rel_err,
							            result.worst_der_full,
							            result.worst_der_fast);
							if (result.worst_der_abs_err > old_abs) {
								result.worst_der_config = cfg_index;
								result.worst_der_atom = atom_index;
								result.worst_der_q = q;
								result.worst_der_neighbor = j;
								result.worst_der_component = a;
							}
							++result.checked_derivatives;

							UpdateWorst(full_weighted_derivatives[j][a],
							            fast_weighted_derivatives[j][a],
							            result.worst_weighted_der_abs_err,
							            result.worst_weighted_der_rel_err,
							            result.worst_weighted_der_full,
							            result.worst_weighted_der_fast);
						}
					}
				}
			}
		}

		active_two_layer_gate_values_ = saved_gate_values;
		active_two_layer_gate_adjoints_ = saved_gate_adjoints;
		return result;
	}
};

class DevGateMuBodyOrderProbe : public MLMTPR {
public:
	DevGateMuBodyOrderProbe(const std::string& mtp_filename)
		: MLMTPR(mtp_filename)
	{
	}

	DevGateMuBodyOrderResult Check(std::vector<Configuration>& configs,
	                               int max_atoms,
	                               double probe_weight)
	{
		if (!two_layer_gate_enabled_)
			ERROR("Model does not have two-layer gate enabled.");
		if (TwoLayerGateWeightCount() <= 0)
			ERROR("Model has no two-layer gate weights.");
		BuildTwoLayerGateBodyOrderBuckets();

		std::vector<int> probe_weight_by_body_order(
			two_layer_gate_body_order_max_ + 1, -1);
		std::vector<double> probe_weight_abs_by_body_order(
			two_layer_gate_body_order_max_ + 1, 0.0);
		for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
			const int body_order = TwoLayerGateWeightBodyOrder(q);
			if (body_order >= 2 && body_order <= two_layer_gate_body_order_max_
			    && probe_weight_by_body_order[body_order] < 0)
				probe_weight_by_body_order[body_order] = q;
		}
		for (int body_order = 2; body_order <= two_layer_gate_body_order_max_;
		     ++body_order) {
			if (probe_weight_by_body_order[body_order] < 0)
				ERROR("SUS2-SH mu-body-order gate is missing a scalar body-order bucket");
		}

		DevGateMuBodyOrderResult result;
		std::vector<double> scalar_values;
		for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size());
		     ++cfg_index) {
			Neighborhoods neighborhoods(configs[cfg_index], CutOff());
			const int atom_limit = max_atoms <= 0
				? configs[cfg_index].size()
				: std::min(max_atoms, configs[cfg_index].size());
			for (int atom_index = 0; atom_index < atom_limit; ++atom_index) {
				CalcTwoLayerGateScalarValuesOnly(neighborhoods[atom_index],
				                                  scalar_values);
				for (int q = 0; q < TwoLayerGateScalarCount(); ++q) {
					const int body_order = TwoLayerGateWeightBodyOrder(q);
					const double abs_value = std::abs(scalar_values[q]);
					if (abs_value > probe_weight_abs_by_body_order[body_order]) {
						probe_weight_abs_by_body_order[body_order] = abs_value;
						probe_weight_by_body_order[body_order] = q;
					}
				}
			}
		}
		for (int body_order = 2; body_order <= two_layer_gate_body_order_max_;
		     ++body_order) {
			if (probe_weight_abs_by_body_order[body_order] <= 1.0e-14)
				ERROR("SUS2-SH mu-body-order gate check could not find a nonzero scalar for one body-order bucket");
		}

		const std::vector<double> saved_weights = two_layer_gate_weights_;
		const std::vector<double> saved_body_mix_weights =
			two_layer_gate_body_mix_weights_;
		const std::vector<double> saved_regression_coeffs = regression_coeffs;
		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		active_two_layer_gate_values_ = nullptr;

		for (int body_order = 2; body_order <= two_layer_gate_body_order_max_;
		     ++body_order) {
			++result.checked_body_orders;
			const int q_probe = probe_weight_by_body_order[body_order];
			std::fill(two_layer_gate_weights_.begin(),
			          two_layer_gate_weights_.end(), 0.0);
			std::fill(two_layer_gate_body_mix_weights_.begin(),
			          two_layer_gate_body_mix_weights_.end(), 0.0);
			for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
				const int coeff_index = TwoLayerGateWeightOffset() + q;
				if (coeff_index < 0
				    || coeff_index >= static_cast<int>(regression_coeffs.size()))
					ERROR("SUS2-SH two-layer gate regression coefficient index is out of range");
				regression_coeffs[coeff_index] = 0.0;
			}
			for (int i = 0; i < TwoLayerGateBodyMixWeightCount(); ++i) {
				const int coeff_index = TwoLayerGateBodyMixWeightOffset() + i;
				if (coeff_index < 0
				    || coeff_index >= static_cast<int>(regression_coeffs.size()))
					ERROR("SUS2-SH two-layer gate body mix regression coefficient index is out of range");
				regression_coeffs[coeff_index] = 0.0;
			}
			two_layer_gate_weights_[q_probe] = probe_weight;
			regression_coeffs[TwoLayerGateWeightIndex(q_probe)] = probe_weight;
			for (int mu = 0; mu < radial_func_count; ++mu) {
				if (TwoLayerGateMuBodyOrder(mu) != body_order)
					continue;
				const int local_index =
					mu * TwoLayerGateBodyOrderCount() + (body_order - 2);
				two_layer_gate_body_mix_weights_[local_index] = 1.0;
				const int coeff_index =
					TwoLayerGateBodyMixWeightIndex(mu, body_order);
				if (coeff_index < 0
				    || coeff_index >= static_cast<int>(regression_coeffs.size()))
					ERROR("SUS2-SH two-layer gate regression coefficient index is out of range");
				regression_coeffs[coeff_index] = 1.0;
			}
			two_layer_gate_values_from_edge_cache_ready_ = false;

			bool saw_nonzero_matching_channel = false;
			for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size());
			     ++cfg_index) {
				Neighborhoods neighborhoods(configs[cfg_index], CutOff());
				PrepareTwoLayerGateValues(configs[cfg_index], neighborhoods);
				const int atom_limit = max_atoms <= 0
					? configs[cfg_index].size()
					: std::min(max_atoms, configs[cfg_index].size());
				for (int atom_index = 0; atom_index < atom_limit; ++atom_index) {
					const Neighborhood& nbh = neighborhoods[atom_index];
					CalcTwoLayerGateScalarValuesOnly(nbh, scalar_values);
					const double selected_scalar = scalar_values[q_probe];
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const int mu_body_order = TwoLayerGateMuBodyOrder(mu);
						const double expected =
							mu_body_order == body_order ? probe_weight * selected_scalar : 0.0;
						const double actual =
							two_layer_gate_values_[static_cast<size_t>(atom_index) *
							                       radial_func_count + mu];
						const double old_abs = result.worst_abs_err;
						UpdateWorst(expected, actual,
						            result.worst_abs_err,
						            result.worst_rel_err,
						            result.worst_expected,
						            result.worst_actual);
						if (result.worst_abs_err > old_abs) {
							result.worst_body_order = body_order;
							result.worst_config = cfg_index;
							result.worst_atom = atom_index;
							result.worst_mu = mu;
						}
						if (mu_body_order == body_order
						    && std::abs(expected) > 1.0e-14)
							saw_nonzero_matching_channel = true;
						++result.checked_values;
					}
				}
			}
			if (saw_nonzero_matching_channel)
				++result.active_body_orders;
		}

		two_layer_gate_weights_ = saved_weights;
		two_layer_gate_body_mix_weights_ = saved_body_mix_weights;
		regression_coeffs = saved_regression_coeffs;
		two_layer_gate_values_from_edge_cache_ready_ = false;
		active_two_layer_gate_values_ = saved_gate_values;
		return result;
	}
};

DevEFSFDResult CheckEFSFiniteDifference(MLMTPR& mtpr,
                                        const std::vector<Configuration>& configs,
                                        double displacement,
                                        int max_atoms)
{
	DevEFSFDResult result;
	for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size()); ++cfg_index) {
		Configuration base = configs[cfg_index];
		mtpr.CalcEFS(base);

		const int atom_limit = max_atoms <= 0
			? base.size()
			: std::min(max_atoms, base.size());
		for (int i = 0; i < atom_limit; ++i) {
			for (int a = 0; a < 3; ++a) {
				Configuration plus = configs[cfg_index];
				Configuration minus = configs[cfg_index];
				plus.pos(i, a) += displacement;
				minus.pos(i, a) -= displacement;
				mtpr.CalcEFS(plus);
				mtpr.CalcEFS(minus);
				const double fd_force =
					(minus.energy - plus.energy) / (2.0 * displacement);
				const double old_abs = result.worst_force_abs_err;
				UpdateWorst(base.force(i, a), fd_force,
				            result.worst_force_abs_err,
				            result.worst_force_rel_err,
				            result.worst_force_analytic,
				            result.worst_force_fd);
				if (result.worst_force_abs_err > old_abs) {
					result.worst_force_config = cfg_index;
					result.worst_force_atom = i;
					result.worst_force_component = a;
				}
				++result.force_components;
			}
		}

		Matrix3 fd_dedl(0, 0, 0, 0, 0, 0, 0, 0, 0);
		for (int a = 0; a < 3; ++a) {
			for (int b = 0; b < 3; ++b) {
				Configuration plus = configs[cfg_index];
				Configuration minus = configs[cfg_index];
				Matrix3 deform = configs[cfg_index].lattice;
				deform[a][b] += displacement;
				plus.Deform(plus.lattice.inverse() * deform);
				deform[a][b] -= 2.0 * displacement;
				minus.Deform(minus.lattice.inverse() * deform);
				mtpr.CalcEFS(plus);
				mtpr.CalcEFS(minus);
				const double denom = plus.lattice[a][b] - minus.lattice[a][b];
				fd_dedl[a][b] = (plus.energy - minus.energy) / denom;
			}
		}
		const Matrix3 analytic_dedl =
			-base.lattice.inverse().transpose() * base.stresses;
		for (int a = 0; a < 3; ++a) {
			for (int b = 0; b < 3; ++b) {
				const double old_abs = result.worst_stress_abs_err;
				UpdateWorst(analytic_dedl[a][b], fd_dedl[a][b],
				            result.worst_stress_abs_err,
				            result.worst_stress_rel_err,
				            result.worst_stress_analytic,
				            result.worst_stress_fd);
				if (result.worst_stress_abs_err > old_abs) {
					result.worst_stress_config = cfg_index;
					result.worst_stress_a = a;
					result.worst_stress_b = b;
				}
				++result.stress_components;
			}
		}
	}
	return result;
}

DevLinearComponentsFDResult CheckLinearComponentsFiniteDifference(
	MLMTPR& mtpr,
	const std::vector<Configuration>& configs,
	double displacement,
	int max_atoms,
	int coeff_begin,
	int coeff_end)
{
	DevLinearComponentsFDResult result;
	const int coeff_count = mtpr.LinearEquationCount();
	coeff_begin = std::max(0, coeff_begin);
	coeff_end = coeff_end <= 0 ? coeff_count : std::min(coeff_end, coeff_count);
	if (coeff_begin >= coeff_end)
		ERROR("Invalid coefficient range for linear-component FD check.");

	for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size()); ++cfg_index) {
		Configuration base = configs[cfg_index];
		Neighborhoods base_neighborhoods(base, mtpr.CutOff());
		mtpr.CalcEFSComponents(base, base_neighborhoods, true, false);

		const int atom_limit = max_atoms <= 0
			? base.size()
			: std::min(max_atoms, base.size());
		std::vector<double> base_force_components(
			static_cast<size_t>(atom_limit) * 3 * coeff_count, 0.0);
		for (int i = 0; i < atom_limit; ++i)
			for (int a = 0; a < 3; ++a)
				for (int k = coeff_begin; k < coeff_end; ++k)
					base_force_components[(static_cast<size_t>(i) * 3 + a) * coeff_count + k] =
						mtpr.forces_cmpnts(i, k, a);

		for (int i = 0; i < atom_limit; ++i) {
			for (int a = 0; a < 3; ++a) {
				Configuration plus = configs[cfg_index];
				Configuration minus = configs[cfg_index];
				plus.pos(i, a) += displacement;
				minus.pos(i, a) -= displacement;
				Neighborhoods plus_neighborhoods(plus, mtpr.CutOff());
				Neighborhoods minus_neighborhoods(minus, mtpr.CutOff());
				mtpr.CalcEFSComponents(plus, plus_neighborhoods, false, false);
				std::vector<double> plus_energy_components(
					mtpr.energy_cmpnts, mtpr.energy_cmpnts + coeff_count);
				mtpr.CalcEFSComponents(minus, minus_neighborhoods, false, false);
				for (int k = coeff_begin; k < coeff_end; ++k) {
					const double fd_force =
						-(plus_energy_components[k] - mtpr.energy_cmpnts[k])
						/ (2.0 * displacement);
					const double analytic =
						base_force_components[(static_cast<size_t>(i) * 3 + a) * coeff_count + k];
					const double old_abs = result.worst_abs_err;
					UpdateWorst(analytic, fd_force,
					            result.worst_abs_err,
					            result.worst_rel_err,
					            result.worst_analytic,
					            result.worst_fd);
					if (result.worst_abs_err > old_abs) {
						result.worst_config = cfg_index;
						result.worst_atom = i;
						result.worst_component = a;
						result.worst_coeff = k;
					}
					++result.checked_components;
				}
			}
		}
	}
	return result;
}

double LinearEquationCurrentCoeff(const MLMTPR& mtpr, int column)
{
	const int linear_count = mtpr.LinearCoeffCount();
	const int linear_begin = mtpr.LinearCoeffOffset();
	if (column < linear_count) {
		if (column < mtpr.species_count)
			return mtpr.regression_coeffs[column]
			     + mtpr.regression_coeffs[linear_begin + column];
		const int scalar_index = column - mtpr.species_count;
		double value = mtpr.regression_coeffs[linear_begin + column];
		if (scalar_index >= 0 && scalar_index < static_cast<int>(mtpr.linear_mults.size()))
			value *= mtpr.linear_mults[scalar_index];
		return value;
	}
	const int e0_index = column - linear_count;
	const int e0_begin = mtpr.TwoLayerResidualE0CoeffOffset();
	double value = 0.0;
	if (e0_index >= 0 && e0_index < mtpr.TwoLayerResidualE0CoeffCount())
		value = mtpr.regression_coeffs[e0_begin + e0_index];
	if (e0_index >= 0 && e0_index < static_cast<int>(mtpr.linear_mults.size()))
		value *= mtpr.linear_mults[e0_index];
	return value;
}

DevLinearReadoutResult CheckLinearReadout(
	MLMTPR& mtpr,
	const std::vector<Configuration>& configs,
	int max_atoms)
{
	DevLinearReadoutResult result;
	const int coeff_count = mtpr.LinearEquationCount();
	std::vector<double> coeffs(coeff_count, 0.0);
	for (int k = 0; k < coeff_count; ++k)
		coeffs[k] = LinearEquationCurrentCoeff(mtpr, k);

	for (int cfg_index = 0; cfg_index < static_cast<int>(configs.size()); ++cfg_index) {
		Configuration direct = configs[cfg_index];
		Neighborhoods neighborhoods(direct, mtpr.CutOff());
		mtpr.CalcEFS(direct, neighborhoods);

		Configuration component_cfg = configs[cfg_index];
		mtpr.CalcEFSComponents(component_cfg, neighborhoods, true, false);
		double linear_energy = 0.0;
		for (int k = 0; k < coeff_count; ++k)
			linear_energy += mtpr.energy_cmpnts[k] * coeffs[k];
		const double old_energy_abs = result.worst_energy_abs_err;
		UpdateWorst(direct.energy, linear_energy,
		            result.worst_energy_abs_err,
		            result.worst_energy_rel_err,
		            result.worst_energy_direct,
		            result.worst_energy_linear);
		if (result.worst_energy_abs_err > old_energy_abs)
			result.worst_energy_config = cfg_index;
		++result.energy_components;

		const int atom_limit = max_atoms <= 0
			? direct.size()
			: std::min(max_atoms, direct.size());
		for (int i = 0; i < atom_limit; ++i) {
			for (int a = 0; a < 3; ++a) {
				double linear_force = 0.0;
				for (int k = 0; k < coeff_count; ++k)
					linear_force += mtpr.forces_cmpnts(i, k, a) * coeffs[k];
				const double old_force_abs = result.worst_force_abs_err;
				UpdateWorst(direct.force(i, a), linear_force,
				            result.worst_force_abs_err,
				            result.worst_force_rel_err,
				            result.worst_force_direct,
				            result.worst_force_linear);
				if (result.worst_force_abs_err > old_force_abs) {
					result.worst_force_config = cfg_index;
					result.worst_force_atom = i;
					result.worst_force_component = a;
				}
				++result.force_components;
			}
		}
	}
	return result;
}

std::vector<int> ParseSpeciesIndexList(const std::string& value)
{
	std::vector<int> indices;
	std::stringstream stream(value);
	std::string token;
	while (std::getline(stream, token, ',')) {
		token.erase(std::remove_if(token.begin(), token.end(),
			[](unsigned char ch) { return std::isspace(ch) != 0; }),
			token.end());
		if (token.empty())
			ERROR("--species should be a comma-separated list of integer species indices.");
		std::size_t parsed_chars = 0;
		int index = 0;
		try {
			index = std::stoi(token, &parsed_chars);
		} catch (const std::exception&) {
			ERROR("--species should be a comma-separated list of integer species indices.");
		}
		if (parsed_chars != token.size())
			ERROR("--species should be a comma-separated list of integer species indices.");
		indices.push_back(index);
	}
	if (indices.empty())
		ERROR("--species should contain at least one species index.");
	return indices;
}

std::string FormatSpeciesMapping(const std::vector<int>& old_species_indices)
{
	std::ostringstream oss;
	for (int i = 0; i < static_cast<int>(old_species_indices.size()); ++i) {
		if (i != 0)
			oss << ", ";
		oss << old_species_indices[i] << "->" << i;
	}
	return oss.str();
}

std::string DescribeMTPRCoeffIndex(const MLMTPR& mtpr, int index)
{
	if (index < 0)
		return "none";
	if (index < mtpr.BaseNonlinearCoeffCount())
		return "nonlinear";
	if (index >= mtpr.TwoLayerGateRadialCoeffOffset()
	    && index < mtpr.TwoLayerGateWeightOffset())
		return "two_layer_gate_radial_coeff";
	if (index >= mtpr.TwoLayerGateWeightOffset()
	    && index < mtpr.TwoLayerGateWeightOffset() + mtpr.TwoLayerGateWeightCount())
		return "two_layer_gate_weight";
	if (index >= mtpr.TwoLayerGateEdgeL1WeightOffset()
	    && index < mtpr.TwoLayerGateEdgeL1WeightOffset()
	        + mtpr.TwoLayerGateEdgeL1WeightCount())
		return "two_layer_gate_edge_l1_weight";
	if (index >= mtpr.LinearCoeffOffset())
		return "linear";
	return "unknown";
}

}

// does a number of developer unit tests
// returns true if all tests are successful
// otherwise returns false and stops further tests
bool self_test_dev()
{
	ofstream logstream("temp/log");
	SetStreamForOutput(&logstream);

	int mpi_rank = 0;
	int mpi_size = 1;

#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif // MLIP_MPI

#ifndef MLIP_DEBUG
	if (mpi_rank == 0) {
		std::cout << "Note: self-test is running without #define MLIP_DEBUG;\n"
			<< "      build with -DMLIP_DEBUG and run if troubles encountered" << std::endl;
	}
#endif

	if (mpi_size == 1) {
		if (mpi_rank == 0) cout << "Serial pub tests:" << endl;
		if (!RunAllTests(false)) return false;
		if (mpi_rank == 0) cout << "Serial dev tests:" << endl;
		if (!RunAllTestsDev(false)) return false;
	}
#ifdef MLIP_MPI
	if (mpi_rank == 0) cout << "MPI pub tests (" << mpi_size << " cores):" << endl;
	if (!RunAllTests(true)) return false;
	if (mpi_rank == 0) cout << "MPI dev tests (" << mpi_size << " cores):" << endl;
	if (!RunAllTestsDev(true)) return false;
#endif // MLIP_MPI

	logstream.close();
	return true;
}

bool DevCommands(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	bool is_command_found = false;
	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	BEGIN_COMMAND("self-test-dev",
		"performs a number of unit tests",
		"mlp-sus2 self-test-dev\n"
	) {
		if(!self_test_dev()) exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-loss-gradient-dev",
		"checks MTPR loss gradient against finite differences",
		"mlp-sus2 check-loss-gradient-dev model.mtp train.cfg --max-configs=1 --energy-weight=1 --force-weight=1 --stress-weight=0 --radial-smooth=0 --radial-smooth-grid=128 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --displacement=1e-7 --abs-tolerance=1e-5 --rel-tolerance=1e-4 --coeff-start=0 --coeff-end=0 --residual-stage=full --stage-active-only\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-loss-gradient-dev: model and cfg arguments are required\n";
			return 1;
		}
		MLMTPR mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for loss-gradient check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for loss-gradient check.");

		const double energy_weight = ParseDevDoubleOption(opts, "energy-weight", 1.0);
		const double force_weight = ParseDevDoubleOption(opts, "force-weight", 1.0);
		const double stress_weight = ParseDevDoubleOption(opts, "stress-weight", 0.0);
		const double radial_smooth = ParseDevDoubleOption(opts, "radial-smooth", 0.0);
		const int radial_smooth_grid = ParseDevIntOption(opts, "radial-smooth-grid", 128);
		const double scalar_head_l2 = ParseDevDoubleOption(opts, "scalar-head-l2", 0.0);
		const double gate_scalar_l2 = ParseDevDoubleOption(opts, "gate-scalar-l2", 0.0);
		const double gate_mix_l2 = ParseDevDoubleOption(opts, "gate-mix-l2", 0.0);
		const double gate_full_l2 = ParseDevDoubleOption(opts, "gate-full-l2", 0.0);
		const double displacement = ParseDevDoubleOption(opts, "displacement", 1.0e-7);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-5);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-4);
		const int coeff_start = ParseDevIntOption(opts, "coeff-start", 0);
		const int coeff_end = ParseDevIntOption(opts, "coeff-end", 0);
		const int residual_stage = ParseDevResidualStageOption(opts);
		const bool stage_active_only = opts["stage-active-only"] != "";
		if (displacement <= 0.0)
			ERROR("--displacement should be positive.");

		DevLossGradientProbe probe(&mtpr, energy_weight, force_weight, stress_weight);
		probe.radial_smooth_regularization = radial_smooth;
		probe.radial_smooth_grid = radial_smooth_grid;
		probe.scalar_head_l2_regularization = scalar_head_l2;
		probe.gate_scalar_l2_regularization = gate_scalar_l2;
		probe.gate_mix_l2_regularization = gate_mix_l2;
		probe.gate_full_l2_regularization = gate_full_l2;
		probe.residual_stage = residual_stage;
		probe.stage_active_only = stage_active_only;
		DevLossGradientProbe::Result result =
			probe.Check(configs, displacement, abs_tolerance, rel_tolerance,
			            coeff_start, coeff_end);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "checked_coeffs=" << result.checked_count
				          << " coeff_count=" << mtpr.CoeffCount()
				          << " base_nonlinear_end=" << mtpr.BaseNonlinearCoeffCount()
				          << " gate_radial_begin=" << mtpr.TwoLayerGateRadialCoeffOffset()
				          << " gate_radial_end=" << mtpr.TwoLayerGateAdditiveCoeffOffset()
				          << " gate_additive_begin=" << mtpr.TwoLayerGateAdditiveCoeffOffset()
				          << " gate_additive_end=" << mtpr.TwoLayerGateWeightOffset()
				          << " gate_weight_begin=" << mtpr.TwoLayerGateWeightOffset()
				          << " gate_weight_end=" << mtpr.TwoLayerGateWeightOffset() + mtpr.TwoLayerGateWeightCount()
				          << " gate_edge_l1_begin=" << mtpr.TwoLayerGateEdgeL1WeightOffset()
				          << " gate_edge_l1_end=" << mtpr.TwoLayerGateEdgeL1WeightOffset() + mtpr.TwoLayerGateEdgeL1WeightCount()
				          << " linear_begin=" << mtpr.LinearCoeffOffset()
			          << " residual_stage=" << DevResidualStageName(residual_stage)
			          << " stage_active_only=" << (stage_active_only ? 1 : 0)
			          << " base_loss=" << result.base_loss
			          << " worst_index=" << result.worst_index
			          << " worst_group=" << DescribeMTPRCoeffIndex(mtpr, result.worst_index)
			          << " analytic=" << result.worst_analytic
			          << " finite_difference=" << result.worst_fd
			          << " abs_err=" << result.worst_abs_err
			          << " rel_err=" << result.worst_rel_err
			          << std::endl;
		}
		if (result.worst_abs_err > abs_tolerance
		    && result.worst_rel_err > rel_tolerance)
			exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-loss-gradient-direction-dev",
		"checks a directional MTPR loss gradient against finite differences",
		"mlp-sus2 check-loss-gradient-direction-dev model.mtp train.cfg --max-configs=1 --energy-weight=1 --force-weight=1 --stress-weight=0 --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --displacement=1e-4 --abs-tolerance=1e-5 --rel-tolerance=1e-4 --coeff-start=0 --coeff-end=0 --residual-stage=full --stage-active-only\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-loss-gradient-direction-dev: model and cfg arguments are required\n";
			return 1;
		}
		MLMTPR mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for directional loss-gradient check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for directional loss-gradient check.");

		const double energy_weight = ParseDevDoubleOption(opts, "energy-weight", 1.0);
		const double force_weight = ParseDevDoubleOption(opts, "force-weight", 1.0);
		const double stress_weight = ParseDevDoubleOption(opts, "stress-weight", 0.0);
		const double radial_smooth = ParseDevDoubleOption(opts, "radial-smooth", 0.0);
		const int radial_smooth_grid = ParseDevIntOption(opts, "radial-smooth-grid", 128);
		const double scalar_head_l2 = ParseDevDoubleOption(opts, "scalar-head-l2", 0.0);
		const double gate_scalar_l2 = ParseDevDoubleOption(opts, "gate-scalar-l2", 0.0);
		const double gate_mix_l2 = ParseDevDoubleOption(opts, "gate-mix-l2", 0.0);
		const double gate_full_l2 = ParseDevDoubleOption(opts, "gate-full-l2", 0.0);
		const double displacement = ParseDevDoubleOption(opts, "displacement", 1.0e-4);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-5);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-4);
		const int coeff_start = ParseDevIntOption(opts, "coeff-start", 0);
		const int coeff_end = ParseDevIntOption(opts, "coeff-end", 0);
		const int residual_stage = ParseDevResidualStageOption(opts);
		const bool stage_active_only = opts["stage-active-only"] != "";
		if (displacement <= 0.0)
			ERROR("--displacement should be positive.");

		DevLossGradientProbe probe(&mtpr, energy_weight, force_weight, stress_weight);
		probe.radial_smooth_regularization = radial_smooth;
		probe.radial_smooth_grid = radial_smooth_grid;
		probe.scalar_head_l2_regularization = scalar_head_l2;
		probe.gate_scalar_l2_regularization = gate_scalar_l2;
		probe.gate_mix_l2_regularization = gate_mix_l2;
		probe.gate_full_l2_regularization = gate_full_l2;
		probe.residual_stage = residual_stage;
		probe.stage_active_only = stage_active_only;
		DevLossGradientProbe::DirectionalResult result =
			probe.CheckDirectional(configs, displacement, coeff_start, coeff_end);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "checked_coeffs=" << result.checked_count
			          << " coeff_count=" << mtpr.CoeffCount()
			          << " gate_edge_l1_begin=" << mtpr.TwoLayerGateEdgeL1WeightOffset()
			          << " gate_edge_l1_end=" << mtpr.TwoLayerGateEdgeL1WeightOffset() + mtpr.TwoLayerGateEdgeL1WeightCount()
			          << " residual_stage=" << DevResidualStageName(residual_stage)
			          << " stage_active_only=" << (stage_active_only ? 1 : 0)
			          << " base_loss=" << result.base_loss
			          << " analytic_directional=" << result.analytic
			          << " finite_difference=" << result.finite_difference
			          << " abs_err=" << result.abs_err
			          << " rel_err=" << result.rel_err
			          << std::endl;
		}
		if (result.abs_err > abs_tolerance
		    && result.rel_err > rel_tolerance)
			exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-efs-fd-dev",
		"checks forces and stresses against energy finite differences",
		"mlp-sus2 check-efs-fd-dev model.mtp cfg --max-configs=1 --max-atoms=0 --displacement=1e-4 --abs-tolerance=1e-4 --rel-tolerance=1e-3\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-efs-fd-dev: model and cfg arguments are required\n";
			return 1;
		}
		MLMTPR mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for EFS finite-difference check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for EFS finite-difference check.");

		const int max_atoms = ParseDevIntOption(opts, "max-atoms", 0);
		const double displacement = ParseDevDoubleOption(opts, "displacement", 1.0e-4);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-4);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-3);
		if (displacement <= 0.0)
			ERROR("--displacement should be positive.");

		const DevEFSFDResult result =
			CheckEFSFiniteDifference(mtpr, configs, displacement, max_atoms);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "force_components=" << result.force_components
			          << " worst_force_config=" << result.worst_force_config
			          << " worst_force_atom=" << result.worst_force_atom
			          << " worst_force_component=" << result.worst_force_component
			          << " force_analytic=" << result.worst_force_analytic
			          << " force_fd=" << result.worst_force_fd
			          << " force_abs_err=" << result.worst_force_abs_err
			          << " force_rel_err=" << result.worst_force_rel_err
			          << " stress_components=" << result.stress_components
			          << " worst_stress_config=" << result.worst_stress_config
			          << " worst_stress_a=" << result.worst_stress_a
			          << " worst_stress_b=" << result.worst_stress_b
			          << " stress_dedl_analytic=" << result.worst_stress_analytic
			          << " stress_dedl_fd=" << result.worst_stress_fd
			          << " stress_abs_err=" << result.worst_stress_abs_err
			          << " stress_rel_err=" << result.worst_stress_rel_err
			          << std::endl;
		}
		const bool force_failed =
			result.worst_force_abs_err > abs_tolerance
			&& result.worst_force_rel_err > rel_tolerance;
		const bool stress_failed =
			result.worst_stress_abs_err > abs_tolerance
			&& result.worst_stress_rel_err > rel_tolerance;
		if (force_failed || stress_failed)
			exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-linear-components-fd-dev",
		"checks linear-regression force components against energy-component finite differences",
		"mlp-sus2 check-linear-components-fd-dev model.mtp cfg --max-configs=1 --max-atoms=1 --coeff-start=0 --coeff-end=0 --displacement=1e-5 --abs-tolerance=1e-5 --rel-tolerance=1e-4\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-linear-components-fd-dev: model and cfg arguments are required\n";
			return 1;
		}
		MLMTPR mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for linear-component finite-difference check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for linear-component finite-difference check.");

		const int max_atoms = ParseDevIntOption(opts, "max-atoms", 1);
		const int coeff_start = ParseDevIntOption(opts, "coeff-start", 0);
		const int coeff_end = ParseDevIntOption(opts, "coeff-end", 0);
		const double displacement = ParseDevDoubleOption(opts, "displacement", 1.0e-5);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-5);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-4);
		if (displacement <= 0.0)
			ERROR("--displacement should be positive.");

		const DevLinearComponentsFDResult result =
			CheckLinearComponentsFiniteDifference(mtpr, configs, displacement,
			                                      max_atoms, coeff_start, coeff_end);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "checked_components=" << result.checked_components
			          << " coeff_count=" << mtpr.LinearEquationCount()
			          << " worst_config=" << result.worst_config
			          << " worst_atom=" << result.worst_atom
			          << " worst_component=" << result.worst_component
			          << " worst_coeff=" << result.worst_coeff
			          << " analytic=" << result.worst_analytic
			          << " finite_difference=" << result.worst_fd
			          << " abs_err=" << result.worst_abs_err
			          << " rel_err=" << result.worst_rel_err
			          << std::endl;
		}
		if (result.worst_abs_err > abs_tolerance
		    && result.worst_rel_err > rel_tolerance)
			exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-linear-readout-dev",
		"checks direct E/F prediction against the current linear component readout",
		"mlp-sus2 check-linear-readout-dev model.mtp cfg --max-configs=1 --max-atoms=1 --abs-tolerance=1e-8 --rel-tolerance=1e-8\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-linear-readout-dev: model and cfg arguments are required\n";
			return 1;
		}
		MLMTPR mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for linear-readout check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for linear-readout check.");

		const int max_atoms = ParseDevIntOption(opts, "max-atoms", 1);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-8);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-8);

		const DevLinearReadoutResult result =
			CheckLinearReadout(mtpr, configs, max_atoms);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "energy_components=" << result.energy_components
			          << " force_components=" << result.force_components
			          << " coeff_count=" << mtpr.LinearEquationCount()
			          << " energy_worst_config=" << result.worst_energy_config
			          << " energy_direct=" << result.worst_energy_direct
			          << " energy_linear=" << result.worst_energy_linear
			          << " energy_abs_err=" << result.worst_energy_abs_err
			          << " energy_rel_err=" << result.worst_energy_rel_err
			          << " force_worst_config=" << result.worst_force_config
			          << " force_worst_atom=" << result.worst_force_atom
			          << " force_worst_component=" << result.worst_force_component
			          << " force_direct=" << result.worst_force_direct
			          << " force_linear=" << result.worst_force_linear
			          << " force_abs_err=" << result.worst_force_abs_err
			          << " force_rel_err=" << result.worst_force_rel_err
			          << std::endl;
		}
		const bool energy_failed =
			result.worst_energy_abs_err > abs_tolerance
			&& result.worst_energy_rel_err > rel_tolerance;
		const bool force_failed =
			result.worst_force_abs_err > abs_tolerance
			&& result.worst_force_rel_err > rel_tolerance;
		if (energy_failed || force_failed)
			exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("check-two-layer-gate-fastpath-dev",
		"checks two-layer gate pruned SH value/derivative paths against full SH",
		"mlp-sus2 check-two-layer-gate-fastpath-dev model.mtp cfg --max-configs=1 --max-atoms=1 --abs-tolerance=1e-10 --rel-tolerance=1e-9\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-two-layer-gate-fastpath-dev: model and cfg arguments are required\n";
			return 1;
		}
		DevGateFastPathProbe mtpr(args[0]);
		std::ifstream ifs(args[1], std::ios::binary);
		if (!ifs)
			ERROR("Cannot open configuration file for two-layer gate fast-path check.");

		const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
		std::vector<Configuration> configs;
		Configuration cfg;
		while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
		       && cfg.Load(ifs)) {
			configs.push_back(cfg);
		}
		if (configs.empty())
			ERROR("No configurations loaded for two-layer gate fast-path check.");

		const int max_atoms = ParseDevIntOption(opts, "max-atoms", 1);
		const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-10);
		const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-9);

		const DevGateFastPathResult result = mtpr.Check(configs, max_atoms);
		if (mpi_rank == 0) {
			std::cout << std::setprecision(12)
			          << "checked_values=" << result.checked_values
			          << " checked_derivatives=" << result.checked_derivatives
			          << " value_worst_config=" << result.worst_value_config
			          << " value_worst_atom=" << result.worst_value_atom
			          << " value_worst_q=" << result.worst_value_q
			          << " value_full=" << result.worst_value_full
			          << " value_fast=" << result.worst_value_fast
			          << " value_abs_err=" << result.worst_value_abs_err
			          << " value_rel_err=" << result.worst_value_rel_err
			          << " der_worst_config=" << result.worst_der_config
			          << " der_worst_atom=" << result.worst_der_atom
			          << " der_worst_q=" << result.worst_der_q
			          << " der_worst_neighbor=" << result.worst_der_neighbor
			          << " der_worst_component=" << result.worst_der_component
			          << " der_full=" << result.worst_der_full
			          << " der_fast=" << result.worst_der_fast
			          << " der_abs_err=" << result.worst_der_abs_err
			          << " der_rel_err=" << result.worst_der_rel_err
			          << " weighted_der_full=" << result.worst_weighted_der_full
			          << " weighted_der_fast=" << result.worst_weighted_der_fast
			          << " weighted_der_abs_err=" << result.worst_weighted_der_abs_err
			          << " weighted_der_rel_err=" << result.worst_weighted_der_rel_err
			          << std::endl;
		}
		const bool value_failed =
			result.worst_value_abs_err > abs_tolerance
			&& result.worst_value_rel_err > rel_tolerance;
		const bool der_failed =
			result.worst_der_abs_err > abs_tolerance
			&& result.worst_der_rel_err > rel_tolerance;
		const bool weighted_der_failed =
			result.worst_weighted_der_abs_err > abs_tolerance
			&& result.worst_weighted_der_rel_err > rel_tolerance;
		if (value_failed || der_failed || weighted_der_failed)
			exit(1);
		} END_COMMAND;

		BEGIN_COMMAND("check-two-layer-gate-mu-body-order-dev",
			"checks exact body-order routing for mu-dependent two-layer gate values",
			"mlp-sus2 check-two-layer-gate-mu-body-order-dev model.mtp cfg --max-configs=1 --max-atoms=1 --probe-weight=0.5 --abs-tolerance=1e-12 --rel-tolerance=1e-10\n"
		) {
			if (args.size() != 2) {
				std::cout << "mlp-sus2 check-two-layer-gate-mu-body-order-dev: model and cfg arguments are required\n";
				return 1;
			}
			DevGateMuBodyOrderProbe mtpr(args[0]);
			std::ifstream ifs(args[1], std::ios::binary);
			if (!ifs)
				ERROR("Cannot open configuration file for mu-body-order gate check.");

			const int max_configs = ParseDevIntOption(opts, "max-configs", 1);
			std::vector<Configuration> configs;
			Configuration cfg;
			while ((max_configs <= 0 || static_cast<int>(configs.size()) < max_configs)
			       && cfg.Load(ifs)) {
				configs.push_back(cfg);
			}
			if (configs.empty())
				ERROR("No configurations loaded for mu-body-order gate check.");

			const int max_atoms = ParseDevIntOption(opts, "max-atoms", 1);
			const double probe_weight = ParseDevDoubleOption(opts, "probe-weight", 0.5);
			const double abs_tolerance = ParseDevDoubleOption(opts, "abs-tolerance", 1.0e-12);
			const double rel_tolerance = ParseDevDoubleOption(opts, "rel-tolerance", 1.0e-10);

			const DevGateMuBodyOrderResult result =
				mtpr.Check(configs, max_atoms, probe_weight);
			if (mpi_rank == 0) {
				std::cout << std::setprecision(12)
				          << "checked_values=" << result.checked_values
				          << " checked_body_orders=" << result.checked_body_orders
				          << " active_body_orders=" << result.active_body_orders
				          << " worst_body_order=" << result.worst_body_order
				          << " worst_config=" << result.worst_config
				          << " worst_atom=" << result.worst_atom
				          << " worst_mu=" << result.worst_mu
				          << " expected=" << result.worst_expected
				          << " actual=" << result.worst_actual
				          << " abs_err=" << result.worst_abs_err
				          << " rel_err=" << result.worst_rel_err
				          << std::endl;
			}
			if (result.active_body_orders != result.checked_body_orders)
				exit(1);
			const bool failed =
				result.worst_abs_err > abs_tolerance
				&& result.worst_rel_err > rel_tolerance;
			if (failed)
				exit(1);
		} END_COMMAND;

			BEGIN_COMMAND("init-sh",
				"writes an untrained SUS2-SH model",
				"mlp-sus2 init-sh output.mtp --species-count=2 --l-max=3 --k-max=3 --body-order=6 --body-l-max=3,3,2,2,2 --cutoff=7.5 --radial-basis-size=10 --radial-basis-type=RBChebyshev_sss\n"
			"Supported SH radial basis types: RBChebyshev_sss, RBChebyshev_sss_rational, RBLaguerre_log1p, RBJacobi_sss\n"
				"Options: --sh-factor-pruning=legacy|q-total (default=legacy), --write-sh-scalar-info,\n"
				"         --two-layer-gate (uses exact body-order k+1 scalar buckets),\n"
				"         --two-layer-gate-mode=mu-body-linear-combo|mu-scalar-full (default=mu-body-linear-combo),\n"
				"         --two-layer-gate-tanh-amplitude=<double> (default=0.8),\n"
			"         --two-layer-gate-site-mode=neighbor|double (default=neighbor),\n"
			"         --two-layer-gate-shared-radial (default for gate models),\n"
			"         --two-layer-gate-edge-l1 (neighbor-only first-version edge-projected raw l=1 gate),\n"
			"         --two-layer-residual (rejected by mu-body-order gate models),\n"
			"         --zbl-elements=<...>, --zbl-inner=<r>, --zbl-outer=<r>,\n"
			"         --zbl-typewise-cutoff-factor=<factor>\n"
		) {
		if (args.size() != 1) {
			std::cout << "mlp-sus2 init-sh: 1 output .mtp argument is required\n";
			return 1;
		}
		if (mpi_rank == 0) {
			WriteSphericalHarmonicModel(args[0], opts);
			std::cout << "SUS2-SH untrained model written to " << args[0] << std::endl;
		}
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	} END_COMMAND;

	BEGIN_COMMAND("prune-model",
		"writes a species-pruned SUS2 model",
		"mlp-sus2 prune-model input.mtp output.mtp --species=2,4,6\n"
		"  Keeps the selected old species, preserves shared parameters, and remaps\n"
		"  them in the output model as 0,1,2,... in the order provided.\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 prune-model: 2 arguments are required\n";
			return 1;
		}
		if (opts["species"] == "") {
			std::cout << "mlp-sus2 prune-model: --species=<comma-separated indices> is required\n";
			return 1;
		}

		const std::vector<int> old_species_indices = ParseSpeciesIndexList(opts["species"]);
		if (mpi_rank == 0) {
			MLMTPR mtpr(args[0]);
			if (!mtpr.HasCompleteParameters())
				ERROR("prune-model requires a complete trained model with shift/scal/radial/linear coefficients.");
			const int old_species_count = mtpr.species_count;
			mtpr.PruneSpecies(old_species_indices);
			mtpr.Save(args[1]);
			std::cout << "Pruned model written to " << args[1]
			          << " species_count " << old_species_count
			          << " -> " << mtpr.species_count
			          << " mapping: " << FormatSpeciesMapping(old_species_indices)
			          << std::endl;
		}
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	} END_COMMAND;

	BEGIN_COMMAND("select-add",
		"actively selects configurations to be added to the current training set",
		"mlp-sus2 select-add pot.mtp train.cfg new.cfg diff.cfg:\n"
		"actively selects configurations from new.cfg and save those\n"
		"that need to be added to train.cfg to diff.cfg\n"
		"  Options:\n"
		"  --init-threshold=<num>: set the initial threshold to num, default=1e-5\n"
		"  --select-threshold=<num>: set the select threshold to num, default=1.1\n"
		"  --swap-threshold=<num>: set the swap threshold to num, default=1.0000001\n"
//		"  --energy-weight=<num>: set the weight for energy equation, default=1\n"
//		"  --force-weight=<num>: set the weight for force equations, default=0\n"
//		"  --stress-weight=<num>: set the weight for stress equations, default=0\n"
//		"  --nbh-weight=<num>: set the weight for site energy equations, default=0\n"
		"  --als-filename=<filename>: active learning state (ALS) filename\n"
		"  --selected-filename=<filename>: file with selected configurations\n"
		"  --selection-limit=<num>: swap limit for multiple selection, default=0 (disabled)\n"
		"  --weighting=<string>: way of weighting configurations with different number of atoms,\n"
                "                        default=vibrations, other=molecules, structures.\n"
		) {

		if (args.size() != 4) {
			std::cout << "\tError: 4 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string train_filename = args[1];
		const string new_cfg_filename = args[2];
		const string diff_filename = args[3];

		int selection_limit=0;						//limits the number of swaps in MaxVol

		cout << "Potential from " << mtp_filename
			<< ", traning set: " << train_filename
			<< ", add from set: " << new_cfg_filename
			<< endl;
		MLMTPR mtpr(mtp_filename);

		double init_threshold = 1e-5;
		if (opts["init-threshold"] != "")
			init_threshold = std::stod(opts["init-threshold"]);
		double select_threshold = 1.1;
		if (opts["select-threshold"] != "")
			select_threshold = std::stod(opts["select-threshold"]);
		double swap_threshold = 1.0000001;
		if (opts["swap-threshold"] != "")
			swap_threshold = std::stod(opts["swap-threshold"]);
		double nbh_cmpnts_weight = 0;
		if (opts["nbh-weight"] != "")
			nbh_cmpnts_weight = std::stod(opts["nbh-weight"]);
		double ene_cmpnts_weight = 1;
		if (opts["energy-weight"] != "")
			ene_cmpnts_weight = std::stod(opts["energy-weight"]);
		double frc_cmpnts_weight = 0;
		if (opts["force-weight"] != "")
			frc_cmpnts_weight = std::stod(opts["force-weight"]);
		double str_cmpnts_weight = 0;
		if (opts["stress-weight"] != "")
			str_cmpnts_weight = std::stod(opts["stress-weight"]);
		string als_filename = "state.als";
		if (opts["mvs-filename"] != "")
			als_filename = opts["mvs-filename"];
		if (opts["als-filename"] != "")
			als_filename = opts["als-filename"];
		string selected_filename = "selected.cfg";
		if (opts["selected-filename"] != "")
			selected_filename = opts["selected-filename"];
		if (opts["selection-limit"] != "")
			selection_limit = std::stoi(opts["selection-limit"]);
	
		MaxvolSelection selector(&mtpr, init_threshold, swap_threshold, swap_threshold,
						nbh_cmpnts_weight, ene_cmpnts_weight, frc_cmpnts_weight, str_cmpnts_weight);

		if (opts["weighting"] != "")
			selector.weighting = opts["weighting"];

		int count = 0;

		{
			cout << "loading training set... " << std::flush;

			ifstream ifs(train_filename, std::ios::binary);
			Configuration cfg;
			while (cfg.Load(ifs)) {
				cfg.features["ID"] = to_string(-1);
				selector.AddForSelection(cfg);
				count ++;
			}

			cout << "done" << endl;

		}
		cout << count << " configurations found in the training set\n" << std::flush;
		selector.Select();
		for (Configuration& x : selector.selected_cfgs)
			x.features["ID"] = "-1";

		selector.threshold_select = select_threshold;
	
		count = 1;

		ifstream ifs(new_cfg_filename, std::ios::binary);
		vector<Configuration> new_cfg_set;
		for (Configuration cfg; cfg.Load(ifs); count++) {
			cfg.features["ID"] = to_string(count);
			new_cfg_set.push_back(cfg);
			selector.AddForSelection(cfg);
		}

		if (selection_limit==0)
			selector.Select();
		else
			{
			selector.Select(selection_limit);
			cout << "Swap limit = " << selection_limit << endl;
			}

		{
			int count = 0;
			for (Configuration& cfg : selector.selected_cfgs)
				if (cfg.size() > 0) count++;

			cout << count << " configurations selected from both sets\n" << std::flush;
		}
		cout << "new configuration count = " << new_cfg_set.size() << endl;

		

		ofstream ofs(diff_filename, ios::binary);
		vector<int> valid_to_train;
		std::set<int> unique_cfg;
		count = 0;
		for (Configuration& x : selector.selected_cfgs) {
			if (stoi(x.features["ID"]) > 0 && unique_cfg.find(x.id()) == unique_cfg.end()) {
				valid_to_train.push_back(stoi(x.features["ID"]));
				x.features.erase("ID");
				x.Save(ofs);
				unique_cfg.insert(x.id());
				count++;
			}
		}
		ofs.close();

		cout << "TS increased by " << count << " configs" << endl;
		
		//further selection till selection limit
		int delta = selection_limit - count;
		count = 0;
		vector<double> grades;
		vector<int> inds;
		if (delta>99990) //disabled further selection
		{
			for (int j=0; j< new_cfg_set.size();j++)
			{
				if (unique_cfg.find(new_cfg_set[j].id()) == unique_cfg.end())
				{
					double gr = selector.Grade(new_cfg_set[j]);

					if (count == 0)
					{
						grades.push_back(gr);
						inds.push_back(j);
					}
					for (int i = 0; i < min(delta, count); i++)
					{

						if (grades[i] <= gr) 
						{
							grades.insert(grades.begin() + i, gr);
							inds.insert(inds.begin() + i, j);
							break;
						}
						else if (i == count - 1)
						{
							grades.push_back(gr);
							inds.push_back(j);
							break;
						}
					}
					count++;
				}
			}
		cout << "TS increased by " << delta << " configs" << endl;
		}
		
		selector.Save(als_filename);
		selector.SaveSelected(selected_filename);
		ofs.open(diff_filename,ios::app);
		for (int i = 0; i < delta; i++)
			new_cfg_set[inds[i]].Save(ofs);

		ofs.close();
	} END_COMMAND;

	BEGIN_COMMAND("calc-grade",
		"calculates and saves maxvol grades of input configurations",
		"mlp-sus2 calc-grade pot.mtp train.cfg in.cfg out.cfg:\n"
		"actively selects from train.cfg, generates the ALS file from train.cfg, and\n"
		"calculates maxvol grades of configurations located in in.cfg\n"
		"and writes them to out.cfg\n"
		"  Options:\n"
		"  --init-threshold=<num>: set the initial threshold to 1+num, default=1e-5\n"
		"  --select-threshold=<num>: set the select threshold to num, default=1.1\n"
		"  --swap-threshold=<num>: set the swap threshold to num, default=1.0000001\n"
//		"  --energy-weight=<num>: set the weight for energy equation, default=1\n"
//		"  --force-weight=<num>: set the weight for force equations, default=0\n"
//		"  --stress-weight=<num>: set the weight for stress equations, default=0\n"
//		"  --nbh-weight=<num>: set the weight for site energy equations, default=0\n"
		"  --als-filename=<filename>: active learning state (ALS) filename\n"
		) {

		if (args.size() != 4) {
			std::cout << "\tError: 4 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string train_filename = args[1];
		const string input_filename = args[2];
		const string output_filename = args[3];

		cout << "Potential from " << mtp_filename
			<< ", train: " << train_filename
			<< ", input: " << input_filename
			<< endl;
		MLMTPR mtpr(mtp_filename);

		double init_threshold = 1e-5;
		if (opts["init-threshold"] != "")
			init_threshold = std::stod(opts["init-threshold"]);
		double select_threshold = 1.1;
		if (opts["select-threshold"] != "")
			select_threshold = std::stod(opts["select-threshold"]);
		double swap_threshold = 1.0000001;
		if (opts["swap-threshold"] != "")
			swap_threshold = std::stod(opts["swap-threshold"]);
		double nbh_cmpnts_weight = 0;
		if (opts["nbh-weight"] != "")
			nbh_cmpnts_weight = std::stod(opts["nbh-weight"]);
		double ene_cmpnts_weight = 1;
		if (opts["energy-weight"] != "")
			ene_cmpnts_weight = std::stod(opts["energy-weight"]);
		double frc_cmpnts_weight = 0;
		if (opts["force-weight"] != "")
			frc_cmpnts_weight = std::stod(opts["force-weight"]);
		double str_cmpnts_weight = 0;
		if (opts["stress-weight"] != "")
			str_cmpnts_weight = std::stod(opts["stress-weight"]);
		string als_filename = "state.als";
		if (opts["mvs-filename"] != "")
			als_filename = opts["mvs-filename"];		
		if (opts["als-filename"] != "")
			als_filename = opts["als-filename"];		

		MaxvolSelection selector(&mtpr, init_threshold, select_threshold, swap_threshold, 
						nbh_cmpnts_weight, ene_cmpnts_weight, frc_cmpnts_weight, str_cmpnts_weight);

		ifstream ifs_train(train_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs_train)) {
			selector.AddForSelection(cfg);
		}
		selector.Select();

		ifs_train.close();
		selector.Save(als_filename);

		ifstream ifs_input(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		while (cfg.Load(ifs_input)) {
			selector.Grade(cfg);
			cfg.Save(ofs);	
		}
		ifs_input.close();
		ofs.close();

	} END_COMMAND;


	BEGIN_COMMAND("calc-descriptors",
		"calculates descriptors in each neighborhood of configurations",
		"mlp-sus2 calc-descriptors pot.mtp in.cfg out.cfg:\n"
		"calculates descriptors in each neighborhood of configurations from in.cfg and\n"
		"writes configurations with calculated descriptors to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcDescriptors(cfg, ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_COMMAND("calc-partialE",
		"calculates body-order E of each atom",
		"mlp-sus2 calc-partialE pot.mtp in.cfg out.cfg:\n"
		"calculates body-order E of atoms from in.cfg and\n"
		"writes results to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcpartialE(cfg, ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_COMMAND("calc-eij",
		"calculates effective-pair (EP) energies of configurations",
		"mlp-sus2 calc-eij pot.mtp in.cfg out.eij:\n"
		"calculates effective-pair energies from in.cfg and writes rows\n"
		"type_i type_j distance EP_energy to out.eij\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		ofs.setf(std::ios::scientific);
		ofs.precision(15);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcEij(cfg, ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_COMMAND("calc-efs",
		"calculates energies, forces, and stresses (efs) of configurations",
		"mlp-sus2 calc-efs pot.mtp in.cfg out.cfg:\n"
		"calculates energies, forces, and stresses (efs) of configurations from in.cfg and\n"
		"writes configurations with calculated efs to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcEFS(cfg);
			cfg.Save(ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("invert-stress",
		"changes sign of stress in a configuration database",
		"mlp-sus2 invert-stress db.cfg:\n"
		"changes sign of stress in all configurations of db.cfg"
	) {

		if (args.size() != 1) {
			std::cout << "\tError: 1 arguments required\n";
			return 1;
		}

		const string cfg_filename = args[0];

		auto cfgs = LoadCfgs(cfg_filename);
		
		{ ofstream ofs(cfg_filename); }

		int counter=0;
		for (auto cfg : cfgs)
		{
			cfg.stresses *= -1;
			cfg.AppendToFile(cfg_filename);
			cout << ++counter << endl;
		}

	} END_COMMAND;

	return is_command_found;
}
