#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 7;
constexpr int kWorkWidth = 2521;
constexpr int kOuterWidth = 645;

void Require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

std::string ReadFile(const char *path)
{
  std::ifstream input(path);
  Require(input.good(), "unable to read source file");
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

double &At(std::vector<double> &buffer, int row, int column, int rows = kRows)
{
  return buffer[static_cast<std::size_t>(row) +
                static_cast<std::size_t>(rows) * column];
}

bool SameBits(double lhs, double rhs)
{
  return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

double SeedValue(int row, int column)
{
  return 0.125 + 0.003 * static_cast<double>(row + 1) +
         0.000017 * static_cast<double>(column + 1);
}

void CheckPhaseArithmetic()
{
  Require(2 * kOuterWidth <= kWorkWidth,
          "current R2 outer phases do not fit in one work matrix");

  std::vector<double> separate_values(
      static_cast<std::size_t>(kRows) * kWorkWidth, 0.0);
  std::vector<double> separate_adjoints(
      static_cast<std::size_t>(kRows) * kWorkWidth, 0.0);
  std::vector<double> overlay(
      static_cast<std::size_t>(kRows) * kWorkWidth, 0.0);

  // Gate forward. The two implementations execute identical assignments.
  for (int column = 0; column < kWorkWidth; ++column) {
    for (int row = 0; row < kRows; ++row) {
      const double value = SeedValue(row, column);
      At(separate_values, row, column) = value;
      At(overlay, row, column) = value;
    }
  }

  // Stand in for the compact gate cache: reverse mode retains only its
  // validated operands, independently of either work matrix.
  std::vector<int> cache_moments;
  for (int column = 0; column < kWorkWidth; column += 11)
    cache_moments.push_back(column);
  std::vector<double> gate_cache(
      static_cast<std::size_t>(kRows) * cache_moments.size());
  for (std::size_t slot = 0; slot < cache_moments.size(); ++slot)
    for (int row = 0; row < kRows; ++row)
      gate_cache[static_cast<std::size_t>(row) + kRows * slot] =
          At(overlay, row, cache_moments[slot]);

  // Outer forward and adjoint: values occupy columns [0,645), adjoints occupy
  // [645,1290). Arithmetic and iteration order match the separate baseline.
  for (int column = 0; column < kOuterWidth; ++column) {
    for (int row = 0; row < kRows; ++row) {
      At(separate_values, row, column) = 0.0;
      At(separate_adjoints, row, column) = 0.0;
      At(overlay, row, column) = 0.0;
      At(overlay, row, kOuterWidth + column) = 0.0;
    }
  }
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kOuterWidth; ++column) {
      const double value = SeedValue(row + 3, column + 17) * 1.03125;
      At(separate_values, row, column) = value;
      At(overlay, row, column) = value;
      const double seed = value * (0.75 + 0.0001 * column);
      At(separate_adjoints, row, column) += seed;
      At(overlay, row, kOuterWidth + column) += seed;
    }
    for (int column = kOuterWidth - 1; column >= 2; --column) {
      const double contribution =
          At(separate_adjoints, row, column) *
          At(separate_values, row, column - 1);
      At(separate_adjoints, row, column - 2) += contribution;
      const double overlay_contribution =
          At(overlay, row, kOuterWidth + column) *
          At(overlay, row, column - 1);
      At(overlay, row, kOuterWidth + column - 2) +=
          overlay_contribution;
    }
  }
  for (int column = 0; column < kOuterWidth; ++column)
    for (int row = 0; row < kRows; ++row) {
      Require(SameBits(At(separate_values, row, column),
                       At(overlay, row, column)),
              "outer value changed under phase overlay");
      Require(SameBits(At(separate_adjoints, row, column),
                       At(overlay, row, kOuterWidth + column)),
              "outer adjoint changed under phase overlay");
    }

  // Gate reverse. The outer values are dead, so the whole allocation becomes
  // the gate adjoint matrix; forward operands come from the compact cache.
  std::fill(separate_adjoints.begin(), separate_adjoints.end(), 0.0);
  std::fill(overlay.begin(), overlay.end(), 0.0);
  for (int row = 0; row < kRows; ++row) {
    for (std::size_t slot = 0; slot < cache_moments.size(); ++slot) {
      const int column = cache_moments[slot];
      const double value = gate_cache[static_cast<std::size_t>(row) +
                                      kRows * slot];
      const double seed = value * (0.5 + 0.00001 * column);
      At(separate_adjoints, row, column) += seed;
      At(overlay, row, column) += seed;
    }
    for (std::size_t slot = cache_moments.size(); slot-- > 1;) {
      const int output = cache_moments[slot];
      const int input = cache_moments[slot - 1];
      const double coeff = gate_cache[static_cast<std::size_t>(row) +
                                      kRows * (slot - 1)];
      At(separate_adjoints, row, input) +=
          At(separate_adjoints, row, output) * coeff;
      At(overlay, row, input) += At(overlay, row, output) * coeff;
    }
  }
  for (int column = 0; column < kWorkWidth; ++column)
    for (int row = 0; row < kRows; ++row)
      Require(SameBits(At(separate_adjoints, row, column),
                       At(overlay, row, column)),
              "gate reverse adjoint changed under phase overlay");
}

void CheckSourceGuards(const char *cpp_path, const char *header_path)
{
  const std::string source = ReadFile(cpp_path);
  const std::string header = ReadFile(header_path);

  Require(header.find("d_nbh_energy_ders_storage") != std::string::npos &&
              header.find("Kokkos::MemoryTraits<Kokkos::Unmanaged>") !=
                  std::string::npos,
          "adjoint owner/facade split is missing");
  const std::size_t guard =
      source.find("bool use_rank_work_phase_overlay");
  Require(guard != std::string::npos &&
              source.find("execution_space == Device", guard) !=
                  std::string::npos &&
              source.find("outer_k_rank_enabled", guard) !=
                  std::string::npos &&
              source.find("two_layer_gate_enabled", guard) !=
                  std::string::npos &&
              source.find("use_gate_moment_cache", guard) !=
                  std::string::npos &&
              source.find("two_layer_gate_kk_compact_cache_enabled", guard) !=
                  std::string::npos &&
              source.find("rank_phase_kk_work_width / 2", guard) !=
                  std::string::npos,
          "rank work overlay guards are incomplete");
  const std::size_t stride_guard =
      source.find("span_is_contiguous()", guard);
  Require(stride_guard != std::string::npos &&
              source.find("stride_1()", stride_guard) != std::string::npos &&
              source.find("use_rank_work_phase_overlay = false", stride_guard) !=
                  std::string::npos,
          "rank work overlay lacks a padded-stride fallback");

  const std::size_t cache = source.find("StoreCompactRankGateMomentCache");
  const std::size_t outer_bind =
      source.find("Reinterpret the unused tail", cache);
  const std::size_t outer_clear =
      source.find("ClearRankOuterMainBuffers", outer_bind);
  const std::size_t reverse_comm = source.find("comm->reverse_comm(this);");
  const std::size_t reverse_bind =
      source.find("complete allocation can now become", reverse_comm);
  const std::size_t reverse_clear =
      source.find("ClearRankGateChainAdjoints", reverse_bind);
  Require(cache != std::string::npos && outer_bind != std::string::npos &&
              outer_clear != std::string::npos &&
              reverse_comm != std::string::npos &&
              reverse_bind != std::string::npos &&
              reverse_clear != std::string::npos && cache < outer_bind &&
              outer_bind < outer_clear && outer_clear < reverse_comm &&
              reverse_comm < reverse_bind && reverse_bind < reverse_clear,
          "rank work phase bindings are out of lifetime order");

  Require(source.find("d_nbh_energy_ders_storage.data()", guard) !=
                  std::string::npos &&
              source.find("d_moment_tensor_vals.data() + adjoint_offset",
                          outer_bind) != std::string::npos &&
              source.find("d_moment_tensor_vals.data()", reverse_bind) !=
                  std::string::npos,
          "rank work overlay or fallback binding is missing");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 3, "expected pair cpp and header paths");
  CheckPhaseArithmetic();
  CheckSourceGuards(argv[1], argv[2]);
  std::cout << "LAMMPS_KK_RANK_WORK_PHASE_OVERLAY_CHECK: ok\n";
  return 0;
}
