#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dev_src/mtpr.h"
#include "dev_src/non_linear_regression.h"
#include "dev_src/diagonal_coordinate_scaling.h"
#include "dev_src/efs_head_coordinate_scaling.h"

struct InspectRegression : NonLinearRegression {
	explicit InspectRegression(MLMTPR* model)
		: NonLinearRegression(model, 1.0, 0.3, 0.1, 0.2, 0.0) {}
	double Loss() const { return loss_; }
	const std::vector<double>& Gradient() const { return loss_grad_; }
};

// Link against the ordinary serial model objects. No training/model files are
// changed: all altered center prefactors, multipliers and heads stay in memory.
int main(int argc, char** argv)
{
	if (argc != 3 && argc != 4) return 2;
	MLMTPR model(argv[1]);
	Configuration cfg;
	std::ifstream input(argv[2]);
	if (!cfg.Load(input) || cfg.size() == 0) return 3;
    if(argc==4) {
        Configuration original=cfg;model.CalcEFS(original);model.Save(argv[3]);
        MLMTPR restored(argv[3]);Configuration copy=cfg;restored.CalcEFS(copy);
        if(std::abs(copy.energy-original.energy)>1e-10) return 4;
        for(int i=0;i<cfg.size();++i) for(int a=0;a<3;++a)
            if(std::abs(copy.force(i,a)-original.force(i,a))>1e-10) return 5;
    }
	const int count = model.alpha_scalar_moments;
	const int head_begin = model.LinearCoeffOffset() + model.species_count;
	const double center = 1.7;
	for (int t = 0; t < model.species_count; ++t) {
		model.Coeff()[t] = 0.0;
		model.Coeff()[model.LinearCoeffOffset() + t] = center;
	}
	model.scaling = 0.4;
	for (int q = 0; q < count; ++q) {
		model.linear_mults[q] = (q % 2 ? -1.0 : 1.0) * (0.6 + 0.13 * q);
		model.Coeff()[head_begin + q] = 0.03 * std::sin(q + 1.0);
	}
	model.LinCoeff();
	model.CalcEFSComponents(cfg, true, true);
	const int outputs = 1 + 3 * cfg.size() + 9;
	std::vector<double> columns(static_cast<size_t>(count) * outputs, 0.0);
	for (int q = 0; q < count; ++q) {
		const int col = model.species_count + q;
		const double mult = center * model.linear_mults[q];
		double* values = columns.data() + static_cast<size_t>(q) * outputs;
		values[0] = mult * model.energy_cmpnts[col];
		for (int atom = 0; atom < cfg.size(); ++atom)
			for (int a = 0; a < 3; ++a)
				values[1 + 3 * atom + a] = mult * model.forces_cmpnts(atom, col, a);
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				values[1 + 3 * cfg.size() + 3 * a + b] = mult * model.stress_cmpnts[col][a][b];
	}
	std::vector<double> energy_weights(cfg.size(), 0.43);
	std::vector<Vector3> force_weights(cfg.size(), Vector3(0.0, 0.0, 0.0));
	Matrix3 stress_weights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	std::vector<double> weights(outputs, 0.0);
	weights[0] = energy_weights[0];
	for (int atom = 0; atom < cfg.size(); ++atom)
		for (int a = 0; a < 3; ++a)
			weights[1 + 3 * atom + a] = force_weights[atom][a] = std::sin(3 * atom + a + 0.2);
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			// The combination-gradient API takes the opposite of the saved
			// PlusStress coefficient (it adds +weight * edge displacement).
			weights[1 + 3 * cfg.size() + 3 * a + b] =
				-(stress_weights[a][b] = 0.1 * std::cos(3 * a + b + 0.3));
	Array1D gradient(model.CoeffCount());
	std::fill(&gradient[0], &gradient[0] + model.CoeffCount(), 0.0);
	Configuration evaluated = cfg;
	model.CalcEFS(evaluated);
	model.AccumulateEFSCombinationGrad(evaluated, energy_weights, force_weights, stress_weights, gradient);
	double max_vjp_error = 0.0, max_fd_error = 0.0;
	for (int q = 0; q < count; ++q) {
		const double* values = columns.data() + static_cast<size_t>(q) * outputs;
		double expected = 0.0;
		for (int j = 0; j < outputs; ++j) expected += weights[j] * values[j];
		max_vjp_error = std::max(max_vjp_error,
			std::abs(expected - gradient[head_begin + q]) / std::max(1.0e-12, std::abs(expected)));
		const double base = model.Coeff()[head_begin + q], step = 0.02;
		Configuration plus = cfg, minus = cfg;
		model.Coeff()[head_begin + q] = base + step;
		model.CalcEFS(plus);
		model.Coeff()[head_begin + q] = base - step;
		model.CalcEFS(minus);
		model.Coeff()[head_begin + q] = base;
		std::vector<double> differences(outputs, 0.0);
		differences[0] = (plus.energy - minus.energy) / (2.0 * step);
		for (int atom = 0; atom < cfg.size(); ++atom)
			for (int a = 0; a < 3; ++a)
				differences[1 + 3 * atom + a] = (plus.force(atom, a) - minus.force(atom, a)) / (2.0 * step);
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				differences[1 + 3 * cfg.size() + 3 * a + b] = (plus.stresses[a][b] - minus.stresses[a][b]) / (2.0 * step);
		for (int j = 0; j < outputs; ++j)
			max_fd_error = std::max(max_fd_error,
				std::abs(differences[j] - values[j]) / std::max(1.0e-10, std::abs(values[j])));
	}

    double force_error=0;
    model.CalcEFS(evaluated);
    for(int i=0;i<cfg.size();++i) for(int a=0;a<3;++a) {
        Configuration plus=cfg,minus=cfg;const double h=1e-5;
        plus.pos(i,a)+=h;minus.pos(i,a)-=h;
        model.CalcEFS(plus);model.CalcEFS(minus);
        force_error=std::max(force_error,std::abs((minus.energy-plus.energy)/(2*h)-evaluated.force(i,a)));
    }
    double cross_gram=0;
    for(int i=0;i<count/2;++i) for(int j=count/2;j<count;++j) {
        double dot=0;for(int a=0;a<outputs;++a) dot+=columns[i*outputs+a]*columns[j*outputs+a];
        cross_gram=std::max(cross_gram,std::abs(dot));
    }
    std::cout << "grouped_cross columns="<<count<<" vjp="<<max_vjp_error
        <<" cross_group_design_gram="<<cross_gram<<" mixed_efs_fd="<<max_fd_error<<" force_fd="<<force_error<<std::endl;
    return max_vjp_error<2e-8 && max_fd_error<2e-6 && force_error<2e-6 ? 0:1;
}
