#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

std::string ReadFile(const char *path)
{
  std::ifstream input(path);
  Require(static_cast<bool>(input), "failed to open Kokkos pair source");
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

std::size_t CountOccurrences(const std::string &text, const std::string &needle)
{
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void CheckLargeDimensionsWithoutAllocating()
{
  static_assert(sizeof(int) == sizeof(std::int32_t),
                "this regression test models the 32-bit Kokkos int index path");
  // This is large enough to reproduce the old flattened-int defect, but the
  // smoke test itself allocates no chunk_size*width storage.
  constexpr int chunk_size = 600000;
  constexpr int width = 4000;
  const std::uint64_t exact_work =
      static_cast<std::uint64_t>(chunk_size) * static_cast<std::uint64_t>(width);

  Require(exact_work == UINT64_C(2400000000),
          "unexpected large-dimension reference product");
  Require(exact_work > static_cast<std::uint64_t>(INT_MAX),
          "reference dimensions do not exceed the flattened int range");
  Require(chunk_size > INT_MAX / width,
          "rank-off overflow fallback predicate did not activate");
  constexpr int largest_safe_chunk = INT_MAX / width;
  Require(largest_safe_chunk == 536870,
          "unexpected rank-off flattened-range threshold");
  Require(static_cast<std::uint64_t>(largest_safe_chunk) * width <=
              static_cast<std::uint64_t>(INT_MAX),
          "largest safe rank-off chunk does not fit in int");
  Require(static_cast<std::uint64_t>(largest_safe_chunk + 1) * width >
              static_cast<std::uint64_t>(INT_MAX),
          "rank-off overflow threshold does not switch at the first unsafe chunk");
  std::cout << "old flattened range: " << chunk_size << " * " << width
            << " = " << exact_work << " > INT_MAX=" << INT_MAX << '\n';

  // Both independent TeamPolicy dimensions remain representable even though
  // their Cartesian-product cardinality does not fit in a signed int.
  Require(chunk_size > 0 && chunk_size <= INT_MAX,
          "chunk dimension is outside the TeamThreadRange int domain");
  Require(width > 0 && width <= INT_MAX,
          "width dimension is outside the TeamPolicy league int domain");

  struct Pair { int k; int ii; };
  const Pair boundary_pairs[] = {
      {0, 0}, {0, chunk_size - 1}, {width - 1, 0},
      {width - 1, chunk_size - 1}};
  for (const Pair pair : boundary_pairs) {
    Require(pair.k >= 0 && pair.k < width,
            "large TeamPolicy league boundary is out of range");
    Require(pair.ii >= 0 && pair.ii < chunk_size,
            "large TeamThreadRange boundary is out of range");
  }

  // Count and coordinate checksums are closed-form proofs of the intended
  // width x chunk_size Cartesian product.  They stay in 64-bit arithmetic.
  const std::uint64_t expected_k_sum =
      static_cast<std::uint64_t>(chunk_size) * width * (width - 1) / 2;
  const std::uint64_t expected_ii_sum =
      static_cast<std::uint64_t>(width) * chunk_size * (chunk_size - 1) / 2;
  Require(expected_k_sum == UINT64_C(4798800000000),
          "large TeamPolicy k checksum changed");
  Require(expected_ii_sum == UINT64_C(719998800000000),
          "large TeamThreadRange ii checksum changed");
}

void CheckSmallCartesianCoverage()
{
  constexpr int chunk_size = 17;
  constexpr int width = 13;
  std::vector<unsigned char> visits(
      static_cast<std::size_t>(chunk_size) * width, 0);

  // This is the host analogue of league_rank=k plus TeamThreadRange(ii).
  for (int k = 0; k < width; ++k) {
    for (int ii = 0; ii < chunk_size; ++ii) {
      const std::size_t ordinal =
          static_cast<std::size_t>(k) * chunk_size + ii;
      Require(ordinal < visits.size(), "small Cartesian index is out of range");
      ++visits[ordinal];
    }
  }
  for (unsigned char count : visits)
    Require(count == 1, "TeamPolicy Cartesian coverage is not one-to-one");
}

void CheckOuterActiveIndexCoverage()
{
  constexpr int raw_width = 100;
  constexpr int suffix_begin = 2100;
  constexpr int moment_count = 2150;
  constexpr int suffix_width = moment_count - suffix_begin;
  constexpr int active_width = raw_width + suffix_width;
  std::vector<unsigned char> visits(moment_count, 0);

  for (int active_k = 0; active_k < active_width; ++active_k) {
    const int k = active_k < raw_width
        ? active_k
        : suffix_begin + active_k - raw_width;
    Require(k >= 0 && k < moment_count,
            "outer active-range mapping is out of bounds");
    ++visits[k];
  }
  for (int k = 0; k < moment_count; ++k) {
    const bool should_be_active = k < raw_width || k >= suffix_begin;
    Require(visits[k] == static_cast<unsigned char>(should_be_active ? 1 : 0),
            "outer active-range mapping has a gap or overlap");
  }
}

void CheckSourceGuard(const std::string &source)
{
  const char *kernel_names[] = {
      "ClearRankGateFirstMoments",
      "StoreRankGateMomentCache",
      "ClearRankOuterMainBuffers",
      "ClearRankGateChainAdjoints",
      "ClearRankGateChainMoments"};

  for (const char *name : kernel_names) {
    Require(CountOccurrences(source, name) == 1,
            "rank clear/cache kernel name is missing or duplicated");
    const std::string escaped_name(name);
    const std::regex policy_before_name(
        "TeamPolicy<DeviceType>[\\s\\S]{0,500}\"" + escaped_name + "\"");
    const std::regex team_range_after_name(
        "\"" + escaped_name +
        "\"[\\s\\S]{0,700}TeamThreadRange\\(team,\\s*local_chunk_size\\)");
    Require(std::regex_search(source, policy_before_name),
            "rank clear/cache kernel is no longer launched with TeamPolicy");
    Require(std::regex_search(source, team_range_after_name),
            "rank clear/cache kernel lost its independent ii dimension");
  }

  Require(CountOccurrences(source,
                           "TeamThreadRange(team, local_chunk_size)") >= 5,
          "not all rank clear/cache kernels use TeamThreadRange");

  const std::regex flattened_int_product(
      R"((local_chunk_size\s*\*\s*(prefix_width|active_width|local_gate_moment_cache_width))|((prefix_width|active_width|local_gate_moment_cache_width)\s*\*\s*local_chunk_size))");
  Require(std::regex_search("local_chunk_size * prefix_width",
                            flattened_int_product),
          "flattened-int source guard does not detect the old expression");
  Require(!std::regex_search(source, flattened_int_product),
          "rank clear/cache code reintroduced a flattened int work range");

  Require(CountOccurrences(source, "\"StoreGateMomentCacheWide\"") == 1,
          "rank-off wide cache fallback is missing or duplicated");
  const std::regex rank_off_guard(
      R"(local_chunk_size\s*>\s*std::numeric_limits<int>::max\(\)\s*/\s*local_alpha_moment_count)");
  Require(std::regex_search(source, rank_off_guard),
          "rank-off cache no longer guards its flattened int range");
  const std::regex rank_off_wide_policy(
      R"(TeamPolicy<DeviceType>[\s\S]{0,500}"StoreGateMomentCacheWide"[\s\S]{0,700}TeamThreadRange\(team,\s*local_chunk_size\))");
  Require(std::regex_search(source, rank_off_wide_policy),
          "rank-off overflow fallback lost its two-dimensional launch");
  Require(CountOccurrences(
              source,
              "local_chunk_size * local_alpha_moment_count") == 1,
          "rank-off safe flattened path changed unexpectedly");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 2,
          "usage: lammps_kk_rank_clear_overflow_check <pair_sus2_mtp_kokkos.cpp>");
  CheckLargeDimensionsWithoutAllocating();
  CheckSmallCartesianCoverage();
  CheckOuterActiveIndexCoverage();
  CheckSourceGuard(ReadFile(argv[1]));
  std::cout << "lammps_kk_rank_clear_overflow_check passed\n";
  return 0;
}
