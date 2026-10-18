#include "../dev_src/adaptive_pytorch_composition.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace comp = sus2::adaptive_pytorch_composition;

namespace {

void RequireClose(double actual, double expected, double tol, const char* label)
{
	if (std::abs(actual - expected) > tol) {
		std::cerr << label << " expected=" << expected
		          << " actual=" << actual << " tol=" << tol << std::endl;
		std::exit(1);
	}
}

void CheckRankDeficientVO2()
{
	comp::StreamingQR qr(2);
	qr.AddConfigurationRow(std::vector<double>{64.0, 32.0}, 128.0);
	qr.AddConfigurationRow(std::vector<double>{128.0, 64.0}, 256.0);
	const comp::SolveResult result = comp::SolveStacked(
		std::vector<std::vector<double> >{qr.Matrix()}, 2, 2);
	if (result.rank != 1) {
		std::cerr << "rank-deficient VO2 rank expected 1 actual "
		          << result.rank << std::endl;
		std::exit(1);
	}
	// Minimum-norm solution of [64,32] dot x = 128.
	RequireClose(result.e0[0], 1.6, 1.0e-11, "VO2 e0[0]");
	RequireClose(result.e0[1], 0.8, 1.0e-11, "VO2 e0[1]");
}

void CheckFullRankExact()
{
	comp::StreamingQR qr0(3);
	qr0.AddConfigurationRow(std::vector<double>{1.0, 0.0, 0.0}, 3.0);
	qr0.AddConfigurationRow(std::vector<double>{0.0, 1.0, 0.0}, -2.0);
	comp::StreamingQR qr1(3);
	qr1.AddConfigurationRow(std::vector<double>{0.0, 0.0, 1.0}, 5.5);
	qr1.AddConfigurationRow(std::vector<double>{1.0, 1.0, 1.0}, 6.5);
	const comp::SolveResult result = comp::SolveStacked(
		std::vector<std::vector<double> >{qr0.Matrix(), qr1.Matrix()}, 3, 4);
	if (result.rank != 3) {
		std::cerr << "full-rank rank expected 3 actual "
		          << result.rank << std::endl;
		std::exit(1);
	}
	RequireClose(result.e0[0], 3.0, 1.0e-11, "full e0[0]");
	RequireClose(result.e0[1], -2.0, 1.0e-11, "full e0[1]");
	RequireClose(result.e0[2], 5.5, 1.0e-11, "full e0[2]");
}

} // namespace

int main()
{
	try {
		CheckRankDeficientVO2();
		CheckFullRankExact();
	} catch (const std::exception& exc) {
		std::cerr << exc.what() << std::endl;
		return 1;
	}
	std::cout << "adaptive_pytorch_composition_check PASS" << std::endl;
	return 0;
}
