#include "sus2_mtp_rank_phase_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using LAMMPS_NS::SUS2RankPhase::CompactLayout;
using LAMMPS_NS::SUS2RankPhase::build_compact_layout;
using LAMMPS_NS::SUS2RankPhase::checked_matrix_bytes;
using LAMMPS_NS::SUS2RankPhase::outer_physical_index;

void Require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

bool SameBits(double a, double b)
{
  std::uint64_t aa = 0;
  std::uint64_t bb = 0;
  std::memcpy(&aa, &a, sizeof(aa));
  std::memcpy(&bb, &b, sizeof(bb));
  return aa == bb;
}

std::string ReadFile(const char *path)
{
  std::ifstream input(path);
  Require(static_cast<bool>(input), "failed to open source-guard file");
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

void CheckCurrentR2Counts()
{
  constexpr int moment_count = 3858;
  constexpr int basic_count = 100;
  constexpr int gate_width = 2521;
  constexpr int outer_begin = 3313;
  constexpr int outer_basic_count = 50;
  const int a0[] = {0, 100, 2200, 2600, 3313, 3315};
  const int a1[] = {1, 1, 1, 1, 3314, 3363};
  const int out[] = {100, 200, 2600, 3000, 3363, 3364};
  const int scalars[] = {3363, 3857};
  std::vector<int> raw(basic_count);
  for (int i = 0; i < basic_count; ++i) raw[i] = i;
  CompactLayout layout;
  std::string reason;
  Require(build_compact_layout(
              moment_count, basic_count, gate_width, 2, outer_begin,
              outer_basic_count, 4, a0, a1, out, 6, scalars, 2,
              raw.data(), static_cast<int>(raw.size()), layout, reason),
          "current R2-sized phase layout was rejected");
  Require(reason.empty(), "valid R2-sized phase layout retained an error");
  Require(layout.gate_width == 2521 && layout.outer_width == 645 &&
              layout.work_width == 2521 &&
              layout.outer_contracted_begin == 100,
          "current R2 phase counts are not 3858/2521/645/2521");
  Require(layout.outer_product_count == 2 &&
              layout.outer_product_a0[0] == 100 &&
              layout.outer_product_a1[0] == 101 &&
              layout.outer_product_out[0] == 150 &&
              layout.outer_scalar_moments[1] == 644,
          "R2 logical-to-physical mapping is incorrect");
}

void CheckRandomForwardReverseEquivalence()
{
  constexpr int moment_count = 180;
  constexpr int basic_count = 12;
  constexpr int gate_width = 72;
  constexpr int outer_begin = 120;
  constexpr int outer_basic_count = 12;
  constexpr int prefix_products = 4;
  constexpr int outer_products = 32;
  constexpr int product_count = prefix_products + outer_products;
  constexpr int scalar_count = 9;

  std::mt19937_64 rng(0x5355533250484153ULL);
  std::uniform_real_distribution<double> value_dist(-0.75, 0.75);
  for (int trial = 0; trial < 128; ++trial) {
    std::vector<int> a0(product_count);
    std::vector<int> a1(product_count);
    std::vector<int> out(product_count);
    std::vector<double> coeff(product_count);
    a0[0] = 0;
    a1[0] = 1;
    out[0] = 12;
    a0[1] = 12;
    a1[1] = 2;
    out[1] = 20;
    a0[2] = 50;
    a1[2] = 3;
    out[2] = 60;
    a0[3] = 60;
    a1[3] = 4;
    out[3] = 90;
    for (int k = 0; k < prefix_products; ++k)
      coeff[k] = value_dist(rng);

    for (int local = 0; local < outer_products; ++local) {
      const int k = prefix_products + local;
      // Two adjacent terms deliberately accumulate into the same target.
      const int target = outer_begin + outer_basic_count + local / 2;
      std::uniform_int_distribution<int> operand_dist(outer_begin,
                                                       target - 1);
      a0[k] = operand_dist(rng);
      a1[k] = operand_dist(rng);
      out[k] = target;
      coeff[k] = value_dist(rng);
    }
    std::vector<int> scalars(scalar_count);
    for (int q = 0; q < scalar_count; ++q)
      scalars[q] = outer_begin + outer_basic_count +
                   (q * 7) % (outer_products / 2);
    std::vector<int> raw(basic_count);
    for (int i = 0; i < basic_count; ++i) raw[i] = i;

    CompactLayout layout;
    std::string reason;
    Require(build_compact_layout(
                moment_count, basic_count, gate_width, 2, outer_begin,
                outer_basic_count, prefix_products, a0.data(), a1.data(),
                out.data(), product_count, scalars.data(), scalar_count,
                raw.data(), static_cast<int>(raw.size()), layout, reason),
            "valid randomized phase layout was rejected");

    std::vector<double> logical(moment_count, 0.0);
    std::vector<double> physical(layout.work_width, 0.0);
    for (int i = 0; i < basic_count; ++i) {
      logical[i] = value_dist(rng);
      physical[i] = logical[i];
    }
    for (int i = 0; i < outer_basic_count; ++i) {
      logical[outer_begin + i] = value_dist(rng);
      physical[layout.outer_contracted_begin + i] =
          logical[outer_begin + i];
    }

    for (int k = prefix_products; k < product_count; ++k)
      logical[out[k]] += coeff[k] * logical[a0[k]] * logical[a1[k]];
    for (int k = 0; k < layout.outer_product_count; ++k)
      physical[layout.outer_product_out[k]] +=
          coeff[prefix_products + k] *
          physical[layout.outer_product_a0[k]] *
          physical[layout.outer_product_a1[k]];

    for (int i = 0; i < basic_count; ++i)
      Require(SameBits(logical[i], physical[i]),
              "randomized raw forward value changed bits");
    for (int logical_index = outer_begin; logical_index < moment_count;
         ++logical_index) {
      const int physical_index = outer_physical_index(
          logical_index, basic_count, outer_begin, moment_count);
      Require(physical_index >= 0 &&
                  SameBits(logical[logical_index], physical[physical_index]),
              "randomized outer forward value changed bits");
    }

    std::vector<double> logical_adj(moment_count, 0.0);
    std::vector<double> physical_adj(layout.work_width, 0.0);
    for (int q = 0; q < scalar_count; ++q) {
      const double seed = value_dist(rng);
      logical_adj[scalars[q]] += seed;
      physical_adj[layout.outer_scalar_moments[q]] += seed;
    }
    for (int k = product_count - 1; k >= prefix_products; --k) {
      const double adj_coeff = logical_adj[out[k]] * coeff[k];
      logical_adj[a1[k]] += adj_coeff * logical[a0[k]];
      logical_adj[a0[k]] += adj_coeff * logical[a1[k]];
    }
    for (int k = layout.outer_product_count - 1; k >= 0; --k) {
      const double adj_coeff =
          physical_adj[layout.outer_product_out[k]] *
          coeff[prefix_products + k];
      physical_adj[layout.outer_product_a1[k]] +=
          adj_coeff * physical[layout.outer_product_a0[k]];
      physical_adj[layout.outer_product_a0[k]] +=
          adj_coeff * physical[layout.outer_product_a1[k]];
    }
    for (int i = 0; i < basic_count; ++i)
      Require(SameBits(logical_adj[i], physical_adj[i]),
              "randomized raw reverse adjoint changed bits");
    for (int logical_index = outer_begin; logical_index < moment_count;
         ++logical_index) {
      const int physical_index = outer_physical_index(
          logical_index, basic_count, outer_begin, moment_count);
      Require(SameBits(logical_adj[logical_index],
                       physical_adj[physical_index]),
              "randomized outer reverse adjoint changed bits");
    }
  }
}

void CheckDuplicateOutputAccumulation()
{
  const int a0[] = {0, 20, 21};
  const int a1[] = {1, 21, 20};
  const int out[] = {2, 25, 25};
  const int scalar[] = {25};
  const int raw[] = {0, 1, 2, 3};
  CompactLayout layout;
  std::string reason;
  Require(build_compact_layout(32, 4, 8, 1, 20, 2, 1, a0, a1,
                              out, 3, scalar, 1, raw, 4, layout, reason),
          "duplicate-output layout was rejected");
  std::vector<double> logical(32, 0.0);
  std::vector<double> physical(layout.work_width, 0.0);
  logical[20] = physical[4] = 2.0;
  logical[21] = physical[5] = 3.0;
  const double coeff[] = {0.0, 5.0, 7.0};
  for (int k = 1; k < 3; ++k)
    logical[out[k]] += coeff[k] * logical[a0[k]] * logical[a1[k]];
  for (int k = 0; k < 2; ++k)
    physical[layout.outer_product_out[k]] += coeff[k + 1] *
        physical[layout.outer_product_a0[k]] *
        physical[layout.outer_product_a1[k]];
  Require(logical[25] == 72.0 && SameBits(logical[25], physical[9]),
          "duplicate product targets were assigned instead of accumulated");
}

void CheckZeroOuterProductLayout()
{
  const int scalar[] = {20, 21};
  const int raw[] = {0, 1, 2, 3};
  CompactLayout layout;
  std::string reason;
  Require(build_compact_layout(
              22, 4, 8, 0, 20, 2, 0,
              nullptr, nullptr, nullptr, 0,
              scalar, 2, raw, 4, layout, reason),
          "zero-product outer layout was rejected");
  Require(layout.outer_width == 6 && layout.work_width == 8 &&
              layout.outer_product_count == 0 &&
              layout.outer_scalar_moments.size() == 2 &&
              layout.outer_scalar_moments[0] == 4 &&
              layout.outer_scalar_moments[1] == 5,
          "zero-product outer layout was mapped incorrectly");
}

void CheckMalformedLayouts()
{
  const int valid_a0[] = {0, 20};
  const int valid_a1[] = {1, 21};
  const int valid_out[] = {2, 25};
  const int scalar[] = {25};
  const int raw[] = {0, 1, 2, 3};
  CompactLayout layout;
  std::string reason;
  auto reject = [&](const int *a0, const int *a1, const int *out,
                    int gate_end, int outer_begin_product,
                    const int *scalars, const int *raws) {
    reason.clear();
    const bool ok = build_compact_layout(
        32, 4, 8, gate_end, 20, 2, outer_begin_product,
        a0, a1, out, 2, scalars, 1, raws, 4, layout, reason);
    Require(!ok && !reason.empty(), "malformed phase layout was accepted");
  };

  int a0[2] = {0, 20};
  int a1[2] = {1, 21};
  int out[2] = {8, 25};
  reject(a0, a1, out, 1, 1, scalar, raw);
  out[0] = 20;
  reject(a0, a1, out, 0, 1, scalar, raw);
  out[0] = 2;
  a0[1] = 4;
  reject(a0, a1, out, 1, 1, scalar, raw);
  a0[1] = 20;
  out[1] = 21;
  reject(a0, a1, out, 1, 1, scalar, raw);
  out[1] = 25;
  a0[1] = 25;
  reject(a0, a1, out, 1, 1, scalar, raw);

  const int bad_scalar[] = {7};
  reject(valid_a0, valid_a1, valid_out, 1, 1, bad_scalar, raw);
  const int bad_raw[] = {0, 1, 2, 4};
  reject(valid_a0, valid_a1, valid_out, 1, 1, scalar, bad_raw);
  Require(!build_compact_layout(32, 4, 8, 2, 20, 2, 1,
                               valid_a0, valid_a1, valid_out, 2,
                               scalar, 1, raw, 4, layout, reason),
          "reversed gate/outer product boundary was accepted");
  Require(!build_compact_layout(32, 4, 21, 1, 20, 2, 1,
                               valid_a0, valid_a1, valid_out, 2,
                               scalar, 1, raw, 4, layout, reason),
          "overlapping gate/outer moment boundary was accepted");
  Require(!build_compact_layout(32, 4, 8, 1, 20, 13, 1,
                               valid_a0, valid_a1, valid_out, 2,
                               scalar, 1, raw, 4, layout, reason),
          "overflowing contracted range was accepted");
  Require(!build_compact_layout(32, 4, 8, 1, 20, 2, 1,
                               nullptr, valid_a1, valid_out, 2,
                               scalar, 1, raw, 4, layout, reason),
          "missing product metadata was accepted");

  std::size_t bytes = 0;
  Require(checked_matrix_bytes(1024, 2521, sizeof(double), bytes) &&
              bytes == static_cast<std::size_t>(1024) * 2521 * sizeof(double),
          "valid matrix byte count failed");
  Require(!checked_matrix_bytes(std::numeric_limits<std::size_t>::max(),
                               2, sizeof(double), bytes),
          "size_t row/column overflow was accepted");
  Require(!checked_matrix_bytes(
              std::numeric_limits<std::size_t>::max() / 4, 1, 8, bytes),
          "size_t element-byte overflow was accepted");

  const int int_max = std::numeric_limits<int>::max();
  Require(outer_physical_index(int_max - 1, 4, 10, int_max) ==
              int_max - 7,
          "logical-to-physical mapping overflowed an intermediate int sum");
}

void CheckSourceGuards(const char *cpp_path, const char *header_path)
{
  const std::string cpp = ReadFile(cpp_path);
  const std::string header = ReadFile(header_path);
  Require(cpp.find("rank_phase_kk_work_width = alpha_moment_count") !=
              std::string::npos,
          "rank phase compaction is not default-off");
  Require(cpp.find("rank_phase_kk_outer_width") != std::string::npos &&
              cpp.find("ClearRankOuterMainBuffers") != std::string::npos,
          "continuous physical outer clear is missing");
  Require(cpp.find("d_outer_alpha_times_a0") != std::string::npos &&
              cpp.find("d_outer_alpha_times_a1") != std::string::npos &&
              cpp.find("d_outer_alpha_times_out") != std::string::npos &&
              cpp.find("d_outer_alpha_moment_mapping") != std::string::npos,
          "pre-mapped outer device metadata is missing");
	  Require(cpp.find(
	              "d_moment_tensor_vals(ii, a3) += mult * val0 * val1;") !=
	              std::string::npos,
	          "outer products no longer accumulate duplicate outputs");
  Require(cpp.find("outer_physical_index(") == std::string::npos,
          "hot Kokkos source performs a runtime logical mapping");
  Require(cpp.find("alpha_moment_count <= 4000") != std::string::npos,
          "gate moment-cache policy no longer uses logical moment count");
  Require(cpp.find("trace_width = outer_k_rank_enabled") !=
              std::string::npos &&
              cpp.find("rank_phase_kk_work_width : alpha_moment_count") !=
                  std::string::npos,
          "trace path does not use the physical work width");
  Require(cpp.find("d_alpha_index_times") == std::string::npos &&
              header.find("d_alpha_index_times") == std::string::npos,
          "unused unsynchronized d_alpha_index_times view remains");
  Require(header.find("d_moment_tensor_vals") != std::string::npos &&
              header.find("d_nbh_energy_ders_wrt_moments") !=
                  std::string::npos,
          "the two independent moment buffers were replaced by an overlay");
  Require(cpp.find("d_moment_tensor_vals, chunk_size,\n\t                    alpha_moment_count") !=
              std::string::npos,
          "rank-off moment allocation path changed");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 3, "expected pair cpp and header paths");
  CheckCurrentR2Counts();
  CheckRandomForwardReverseEquivalence();
  CheckDuplicateOutputAccumulation();
  CheckZeroOuterProductLayout();
  CheckMalformedLayouts();
  CheckSourceGuards(argv[1], argv[2]);
  std::cout << "LAMMPS_KK_RANK_PHASE_COMPACT_CHECK: ok\n";
  return 0;
}
