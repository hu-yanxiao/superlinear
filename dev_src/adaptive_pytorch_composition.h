#ifndef SUS2_ADAPTIVE_PYTORCH_COMPOSITION_H
#define SUS2_ADAPTIVE_PYTORCH_COMPOSITION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef MLIP_INTEL_MKL
#	include <mkl_lapacke.h>
#endif

namespace sus2 {
namespace adaptive_pytorch_composition {

struct SolveResult {
	std::vector<double> e0;
	std::vector<double> singular_values;
	int rank = 0;
	int info = 0;
};

class StreamingQR {
public:
	explicit StreamingQR(int species_count)
		: ns_(species_count),
		  cols_(species_count + 1),
		  r_(static_cast<std::size_t>(species_count + 1)
		     * static_cast<std::size_t>(species_count + 1),
		     0.0)
	{
		if (species_count <= 0)
			throw std::invalid_argument("species_count should be positive");
	}

	void AddConfigurationRow(const std::vector<double>& counts, double energy)
	{
		if (static_cast<int>(counts.size()) != ns_)
			throw std::invalid_argument("composition row has wrong species count");
		std::vector<double> row(static_cast<std::size_t>(cols_), 0.0);
		for (int i = 0; i < ns_; ++i) {
			if (!std::isfinite(counts[static_cast<std::size_t>(i)]))
				throw std::invalid_argument("composition count is non-finite");
			row[static_cast<std::size_t>(i)] = counts[static_cast<std::size_t>(i)];
		}
		if (!std::isfinite(energy))
			throw std::invalid_argument("composition energy is non-finite");
		row[static_cast<std::size_t>(ns_)] = energy;
		AddAugmentedRow(row);
	}

	const std::vector<double>& Matrix() const { return r_; }
	int SpeciesCount() const { return ns_; }
	int ColumnCount() const { return cols_; }

private:
	void AddAugmentedRow(std::vector<double>& row)
	{
		for (int k = 0; k < ns_; ++k) {
			const double a = r_[static_cast<std::size_t>(k) * cols_ + k];
			const double b = row[static_cast<std::size_t>(k)];
			if (b == 0.0)
				continue;
			const double hypot = std::hypot(a, b);
			if (hypot == 0.0)
				continue;
			const double c = a / hypot;
			const double s = b / hypot;
			for (int j = k; j < cols_; ++j) {
				const std::size_t idx =
					static_cast<std::size_t>(k) * cols_ + static_cast<std::size_t>(j);
				const double rkj = r_[idx];
				const double xj = row[static_cast<std::size_t>(j)];
				r_[idx] = c * rkj + s * xj;
				row[static_cast<std::size_t>(j)] = -s * rkj + c * xj;
			}
		}
	}

	int ns_;
	int cols_;
	std::vector<double> r_;
};

inline SolveResult SolveStacked(
	const std::vector<std::vector<double> >& blocks,
	int species_count,
	std::size_t global_original_energy_cfg_count)
{
	if (species_count <= 0)
		throw std::invalid_argument("species_count should be positive");
	const int cols = species_count + 1;
	if (blocks.empty())
		throw std::invalid_argument("at least one QR block is required");
	for (const std::vector<double>& block : blocks)
		if (block.size() != static_cast<std::size_t>(cols) * cols)
			throw std::invalid_argument("QR block has wrong shape");

#ifndef MLIP_INTEL_MKL
	(void)global_original_energy_cfg_count;
	throw std::runtime_error(
		"adaptive PyTorch composition initialization requires MLIP_INTEL_MKL/LAPACKE_dgelsd");
#else
	const int m = static_cast<int>(blocks.size()) * cols;
	const int n = species_count;
	std::vector<double> a(static_cast<std::size_t>(m) * n, 0.0);
	std::vector<double> b(static_cast<std::size_t>(m), 0.0);
	for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
		const std::vector<double>& block = blocks[block_index];
		for (int row = 0; row < cols; ++row) {
			const int dst_row = static_cast<int>(block_index) * cols + row;
			for (int col = 0; col < n; ++col)
				a[static_cast<std::size_t>(dst_row) * n + col] =
					block[static_cast<std::size_t>(row) * cols + col];
			b[static_cast<std::size_t>(dst_row)] =
				block[static_cast<std::size_t>(row) * cols + species_count];
		}
	}
	SolveResult result;
	result.e0.assign(static_cast<std::size_t>(species_count), 0.0);
	result.singular_values.assign(static_cast<std::size_t>(species_count), 0.0);
	const double rcond = std::numeric_limits<double>::epsilon()
		* static_cast<double>(
			std::max<std::size_t>(
				global_original_energy_cfg_count,
				static_cast<std::size_t>(species_count)));
	result.info = LAPACKE_dgelsd(
		LAPACK_ROW_MAJOR,
		m,
		n,
		1,
		a.data(),
		n,
		b.data(),
		1,
		result.singular_values.data(),
		rcond,
		&result.rank);
	if (result.info != 0)
		throw std::runtime_error("LAPACKE_dgelsd failed in composition least squares");
	for (int i = 0; i < species_count; ++i)
		result.e0[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)];
	return result;
#endif
}

} // namespace adaptive_pytorch_composition
} // namespace sus2

#endif
