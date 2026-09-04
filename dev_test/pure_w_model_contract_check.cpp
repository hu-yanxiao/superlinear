#include "../dev_src/mtpr.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool condition, const std::string& message)
{
	if (!condition)
		throw std::runtime_error(message);
}

void CheckPureActiveLayout(MLMTPR& model)
{
	Require(model.OuterUsesPureRadialMatrix(), "missing pure-matrix mode");
	Require(model.OuterUsesSameChannelGraph(), "missing same-channel mode");
	Require(!model.OuterKRankUsesDirectRadial(), "pure rank-1 became direct radial");
	Require(!model.OuterKRankHasTrainableWeights(), "pure P is trainable");
	Require(model.OuterKRank1WeightCount() == 0, "pure P block is not empty");
	Require(model.LinearCoeffOffset() == model.OuterKRank1WeightOffset(),
	        "linear block did not follow the empty P block");
	Require(model.TwoLayerGateUsesSharedRadial(), "independent gate radial block lost");
	const int row_size = model.p_RadialBasis->rb_size;
	const int matrix_count = (model.L) * model.OuterKRank()
		* (model.radial_func_count / model.L) * row_size;
	Require(model.TwoLayerGateRadialMixingCoeffCount() == matrix_count,
	        "incorrect matrix coefficient count");
	std::vector<int> active;
	model.BuildActiveCoeffIndices(active);
	std::vector<char> selected(model.CoeffCount(), 0);
	for (int index : active) {
		Require(index >= 0 && index < model.CoeffCount(), "active index out of range");
		Require(!selected[index], "duplicate active coefficient");
		selected[index] = 1;
	}
	int frozen = 0;
	for (int index = 0; index < model.CoeffCount(); ++index) {
		const int local = index - model.RadialCoeffOffset();
		const bool obsolete = local >= 0
			&& local < model.radial_func_count * model.RadialCoeffBlockSize()
			&& local % model.RadialCoeffBlockSize() < row_size;
		Require(model.IsUnusedOuterRadialCoeff(index) == obsolete,
		        "unused radial predicate froze scaling/type/gate or missed old R");
		Require(static_cast<bool>(selected[index]) == !obsolete,
		        "active coefficient layout differs from exact obsolete-R range");
		if (obsolete) {
			++frozen;
			Require(!model.IsRadialFirstCoeff(index), "obsolete R received a positivity chart");
		}
	}
	Require(frozen == model.radial_func_count * row_size, "wrong frozen coefficient count");
	Require(model.ActiveCoeffCount() + frozen == model.CoeffCount(), "wrong active count");
	model.BuildActiveCoeffIndices(active, true);
	for (int index : active) {
		Require(index < model.species_count || index >= model.RadialCoeffOffset(),
		        "exclude_scal_coeffs retained base scaling");
		Require(index < model.TwoLayerGateScalingCoeffOffset()
		        || index >= model.TwoLayerGateScalingCoeffOffset()
		                    + model.TwoLayerGateScalingCoeffCount(),
		        "exclude_scal_coeffs retained gate scaling");
	}
}
} // namespace

int main(int argc, char** argv)
{
	try {
		Require(argc == 5, "usage: core-contract pure.mtp legacy.mtp rank1.mtp output.mtp");
		MLMTPR model(argv[1]);
		CheckPureActiveLayout(model);
		const int matrix_offset = model.TwoLayerGateRadialMixingCoeffOffset();
		const int matrix_count = model.TwoLayerGateRadialMixingCoeffCount();
		for (int i = 0; i < matrix_count; ++i)
			model.Coeff()[matrix_offset + i] = 0.01 * (i + 1);
		const std::vector<double> expected(model.Coeff(), model.Coeff() + model.CoeffCount());
		model.DistributeCoeffs();
		Require(std::equal(expected.begin(), expected.end(), model.Coeff()),
		        "redistribution changed current physical coefficients");
		model.Orthogonalize();
		Require(std::equal(expected.begin(), expected.end(), model.Coeff()),
		        "legacy orthogonalization modified pure coefficients");
		model.Save(argv[4]);
		MLMTPR roundtrip(argv[4]);
		Require(roundtrip.CoeffCount() == model.CoeffCount(), "roundtrip changed coefficient count");
		for (int i = 0; i < model.CoeffCount(); ++i)
			Require(std::abs(model.Coeff()[i] - roundtrip.Coeff()[i])
			        <= 1e-14 * std::max(1.0, std::abs(model.Coeff()[i])),
			        "roundtrip changed a physical coefficient");
		double penalty = 7.0;
		model.AddPenaltyGrad(0.0, penalty, nullptr);
		Require(penalty == 7.0, "zero legacy penalty changed accumulator");
		bool rejected = false;
		try { model.AddPenaltyGrad(1e-9, penalty, nullptr); }
		catch (...) { rejected = true; }
		Require(rejected, "pure mode accepted obsolete radial orthogonality penalty");

		model.Load(argv[2]);
		Require(!model.OuterUsesPureRadialMatrix() && !model.OuterUsesSameChannelGraph(),
		        "repeated Load retained pure mode in a legacy model");
		Require(model.OuterKRankHasTrainableWeights(), "legacy P disappeared after repeated Load");
		Require(model.ActiveCoeffCount() == model.CoeffCount(), "pure frozen rows leaked into legacy Load");
		model.Load(argv[3]);
		Require(model.OuterKRank() == 1, "implicit rank-1 metadata not understood");
		CheckPureActiveLayout(model);
		model.Load(argv[1]);
		CheckPureActiveLayout(model);
		std::cout << "pure_w_model_core_contract_ok matrix_count=" << matrix_count << '\n';
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	} catch (...) {
		std::cerr << "model contract test failed with an MLIP exception\n";
		return 1;
	}
}
