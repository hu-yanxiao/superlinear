#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/zbl_pair_cache_check"
mkdir -p "$tmp_dir"

src="$tmp_dir/zbl_pair_cache_check.cpp"
bin="$tmp_dir/zbl_pair_cache_check"

cat > "$src" <<'CPP'
#include "src/zbl.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void CheckClose(double expected, double actual, double tol, const char* label)
{
	if (std::fabs(expected - actual) > tol) {
		std::cerr << label << " mismatch: expected=" << expected
		          << " actual=" << actual
		          << " diff=" << std::fabs(expected - actual) << "\n";
		std::exit(1);
	}
}

void CheckPair(int zi, int zj, double inner, double outer)
{
	const ZBLPairConstants constants = MakeZBLPairConstants(zi, zj);
	const double distances[] = {0.35, 0.50, 0.70, 0.79, 0.81};
	for (double r : distances) {
		const ZBLPairValue direct = ComputeZBLPair(zi, zj, r, inner, outer);
		const ZBLPairValue cached =
			ComputeZBLPairCached(constants, r, inner, outer);
		CheckClose(direct.energy, cached.energy, 1.0e-12, "energy");
		CheckClose(direct.dEdr, cached.dEdr, 1.0e-12, "dEdr");
	}
}

} // namespace

int main()
{
	CheckPair(1, 8, 0.0, 0.8);
	CheckPair(27, 28, 0.4, 0.8);
	return 0;
}
CPP

${CXX:-g++} -std=c++17 -O2 -I. "$src" src/zbl_pair.cpp -o "$bin"
"$bin"
