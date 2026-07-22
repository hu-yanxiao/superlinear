#include "sus2_mtp_gate_cache_layout.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using LAMMPS_NS::SUS2GateCache::CompactLayout;
using LAMMPS_NS::SUS2GateCache::build_compact_layout;
using LAMMPS_NS::SUS2GateCache::kIndexMask;
using LAMMPS_NS::SUS2GateCache::unpack_moment;
using LAMMPS_NS::SUS2GateCache::unpack_slot;

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

std::size_t CountOccurrences(const std::string &text,
                             const std::string &needle)
{
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::string ParallelForContaining(const std::string &source,
                                  const std::string &quoted_kernel_name)
{
  const std::size_t marker = source.find(quoted_kernel_name);
  Require(marker != std::string::npos, "source guard kernel name is missing");
  Require(source.find(quoted_kernel_name, marker + quoted_kernel_name.size()) ==
              std::string::npos,
          "source guard kernel name is duplicated");

  const std::string call_prefix = "Kokkos::parallel_for(";
  const std::size_t call = source.rfind(call_prefix, marker);
  Require(call != std::string::npos,
          "failed to find the parallel_for containing a guarded kernel");
  const std::size_t open = source.find('(', call);
  Require(open != std::string::npos && open < marker,
          "guarded kernel is not inside the expected parallel_for call");

  enum class LexState { normal, string_literal, character_literal,
                        line_comment, block_comment };
  LexState state = LexState::normal;
  int depth = 0;
  bool escaped = false;
  for (std::size_t pos = open; pos < source.size(); ++pos) {
    const char ch = source[pos];
    const char next = pos + 1 < source.size() ? source[pos + 1] : '\0';
    if (state == LexState::line_comment) {
      if (ch == '\n') state = LexState::normal;
      continue;
    }
    if (state == LexState::block_comment) {
      if (ch == '*' && next == '/') {
        state = LexState::normal;
        ++pos;
      }
      continue;
    }
    if (state == LexState::string_literal ||
        state == LexState::character_literal) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch == '\\') {
        escaped = true;
        continue;
      }
      if ((state == LexState::string_literal && ch == '"') ||
          (state == LexState::character_literal && ch == '\''))
        state = LexState::normal;
      continue;
    }
    if (ch == '/' && next == '/') {
      state = LexState::line_comment;
      ++pos;
      continue;
    }
    if (ch == '/' && next == '*') {
      state = LexState::block_comment;
      ++pos;
      continue;
    }
    if (ch == '"') {
      state = LexState::string_literal;
      continue;
    }
    if (ch == '\'') {
      state = LexState::character_literal;
      continue;
    }
    if (ch == '(') {
      ++depth;
    } else if (ch == ')') {
      --depth;
      Require(depth >= 0, "parallel_for parser encountered unbalanced input");
      if (depth == 0) return source.substr(call, pos - call + 1);
    }
  }
  Require(false, "parallel_for parser did not find the closing parenthesis");
  return std::string();
}

void CheckUniqueOperandsAndStableSlots()
{
  const int a0[] = {5, 2, 5, 9, 2, 14};
  const int a1[] = {2, 7, 7, 2, 11, 5};
  const std::vector<int> expected_slots = {2, 5, 7, 9, 11, 14};
  CompactLayout first;
  CompactLayout second;
  std::string reason;

  Require(build_compact_layout(a0, a1, 6, 16, first, reason),
          "valid compact layout was rejected");
  Require(reason.empty(), "successful compact layout retained an error reason");
  Require(first.slot_to_moment == expected_slots,
          "compact slots are not the sorted unique operand set");
  Require(std::is_sorted(first.slot_to_moment.begin(),
                         first.slot_to_moment.end()),
          "compact slots are not in ascending moment order");
  Require(std::adjacent_find(first.slot_to_moment.begin(),
                             first.slot_to_moment.end()) ==
              first.slot_to_moment.end(),
          "compact slots contain a duplicate operand");

  Require(build_compact_layout(a0, a1, 6, 16, second, reason),
          "repeated compact-layout construction failed");
  Require(first.slot_to_moment == second.slot_to_moment &&
              first.packed_a0 == second.packed_a0 &&
              first.packed_a1 == second.packed_a1,
          "compact-layout construction is not stable");

  for (int k = 0; k < 6; ++k) {
    const int moment0 = unpack_moment(first.packed_a0[k]);
    const int moment1 = unpack_moment(first.packed_a1[k]);
    const int slot0 = unpack_slot(first.packed_a0[k]);
    const int slot1 = unpack_slot(first.packed_a1[k]);
    Require(moment0 == a0[k] && moment1 == a1[k],
            "packed operand moment failed its round trip");
    Require(slot0 >= 0 &&
                slot0 < static_cast<int>(first.slot_to_moment.size()) &&
                slot1 >= 0 &&
                slot1 < static_cast<int>(first.slot_to_moment.size()),
            "packed operand slot is out of range");
    Require(first.slot_to_moment[slot0] == a0[k] &&
                first.slot_to_moment[slot1] == a1[k],
            "packed slot does not map back to its operand moment");
  }
}

void CheckCurrentOuterRank2ScaleWithoutModelData()
{
  // The current million-atom outer-rank2 model's gate dependency closure has
  // a dense product prefix of 10097 entries and 1065 unique operands.  This
  // synthetic graph exercises the same packed-layout scale without depending
  // on an external model file.
  constexpr int product_count = 10097;
  constexpr int moment_width = 3313;
  constexpr int unique_operand_count = 1065;
  std::vector<int> a0(product_count);
  std::vector<int> a1(product_count);
  for (int k = 0; k < product_count; ++k) {
    a0[k] = 2 * (k % unique_operand_count);
    a1[k] = 2 * ((k + 137) % unique_operand_count);
  }

  CompactLayout layout;
  std::string reason;
  Require(build_compact_layout(a0.data(), a1.data(), product_count,
                               moment_width, layout, reason),
          "current-scale synthetic compact layout was rejected");
  Require(layout.slot_to_moment.size() == unique_operand_count,
          "current-scale synthetic operand count changed");
  Require(layout.slot_to_moment.front() == 0 &&
              layout.slot_to_moment.back() ==
                  2 * (unique_operand_count - 1),
          "current-scale synthetic slot boundaries are incorrect");
  for (int slot = 0; slot < unique_operand_count; ++slot)
    Require(layout.slot_to_moment[slot] == 2 * slot,
            "current-scale synthetic slots lost their ascending order");
}

void CheckInvalidInputsAndPackedDomainFallback()
{
  CompactLayout layout;
  std::string reason;
  const int valid0[] = {0};
  const int valid1[] = {1};

  Require(!build_compact_layout(valid0, valid1, -1, 4, layout, reason) &&
              !reason.empty(),
          "negative product count did not select the fallback");
  Require(!build_compact_layout(valid0, valid1, 1, 0, layout, reason) &&
              !reason.empty(),
          "zero moment width did not select the fallback");
  Require(!build_compact_layout(valid0, valid1, 1, kIndexMask + 1,
                                layout, reason) &&
              !reason.empty(),
          "12-bit moment-width overflow did not select the fallback");
  Require(!build_compact_layout(nullptr, valid1, 1, 4, layout, reason) &&
              !reason.empty(),
          "missing a0 array did not select the fallback");

  const int negative[] = {-1};
  Require(!build_compact_layout(negative, valid1, 1, 4, layout, reason) &&
              !reason.empty(),
          "negative operand did not select the fallback");
  const int past_end[] = {4};
  Require(!build_compact_layout(past_end, valid1, 1, 4, layout, reason) &&
              !reason.empty(),
          "out-of-prefix operand did not select the fallback");

  const int largest_legal0[] = {0};
  const int largest_legal1[] = {kIndexMask - 1};
  Require(build_compact_layout(largest_legal0, largest_legal1, 1,
                               kIndexMask, layout, reason),
          "largest supported packed-index domain was rejected");
  Require(unpack_moment(layout.packed_a1[0]) == kIndexMask - 1,
          "largest supported moment failed its packed round trip");

  Require(build_compact_layout(nullptr, nullptr, 0, 16, layout, reason),
          "empty product graph should produce a valid empty layout");
  Require(layout.slot_to_moment.empty() && layout.packed_a0.empty() &&
              layout.packed_a1.empty(),
          "empty product graph produced nonempty compact storage");
}

void CheckListOrdinalChunkRows()
{
  struct ChunkCase {
    int inum;
    int forward_chunk;
    int reverse_chunk;
  };
  const ChunkCase cases[] = {
      {0, 3, 5}, {1, 8, 1}, {2, 1, 8}, {8, 3, 5}, {17, 5, 3}};
  for (const ChunkCase test : cases) {
    std::vector<unsigned char> forward_visits(test.inum, 0);
    std::vector<unsigned char> reverse_visits(test.inum, 0);
    std::vector<int> cached_values(test.inum, -1);
    for (int local_chunk_offset = 0; local_chunk_offset < test.inum;
         local_chunk_offset += test.forward_chunk) {
      const int local_chunk_size =
          std::min(test.forward_chunk, test.inum - local_chunk_offset);
      for (int ii = 0; ii < local_chunk_size; ++ii) {
        const int row = local_chunk_offset + ii;
        Require(row >= 0 && row < test.inum,
                "forward list-ordinal cache row is out of range");
        ++forward_visits[row];
        cached_values[row] = 1000 + row;
      }
    }
    for (int local_chunk_offset = 0; local_chunk_offset < test.inum;
         local_chunk_offset += test.reverse_chunk) {
      const int local_chunk_size =
          std::min(test.reverse_chunk, test.inum - local_chunk_offset);
      for (int ii = 0; ii < local_chunk_size; ++ii) {
        const int row = local_chunk_offset + ii;
        Require(row >= 0 && row < test.inum,
                "reverse list-ordinal cache row is out of range");
        Require(cached_values[row] == 1000 + row,
                "reverse chunking did not hit the forward cache row");
        ++reverse_visits[row];
      }
    }
    for (int row = 0; row < test.inum; ++row) {
      Require(forward_visits[row] == 1,
              "forward chunking has a cache-row gap or overlap");
      Require(reverse_visits[row] == 1,
              "reverse chunking has a cache-row gap or overlap");
    }
  }
}

void CheckSourceGuard(const std::string &source)
{
  Require(CountOccurrences(
              source, "const int cache_row = ii + chunk_offset;") == 1,
          "compact gate-cache backprop lost its list-ordinal cache row");
  Require(source.find(
              "d_two_layer_gate_cached_moment_vals(cache_row, slot0)") !=
              std::string::npos &&
              source.find(
                  "d_two_layer_gate_cached_moment_vals(cache_row, slot1)") !=
                  std::string::npos,
          "compact gate-cache backprop no longer reads the ordinal row");

  const std::string compact_name =
      "\"StoreCompactRankGateMomentCache\"";
  const std::string compact = ParallelForContaining(
      source, compact_name);
  const std::size_t compact_marker = source.find(compact_name);
  const std::size_t context_begin = compact_marker > 700
      ? compact_marker - 700 : 0;
  const std::string compact_context = source.substr(
      context_begin, compact_marker - context_begin + compact.size());
  Require(compact_context.find(
              "using cache_policy = Kokkos::TeamPolicy<DeviceType>") !=
              std::string::npos &&
              compact.find(
                  "cache_policy(local_gate_cache_slot_count, Kokkos::AUTO)") !=
                  std::string::npos,
          "compact gate-cache store lost its TeamPolicy alias");
  Require(compact.find("TeamThreadRange(team, local_chunk_size)") !=
              std::string::npos,
          "compact gate-cache store lost its list-ordinal thread range");
  const std::regex ordinal_row(
      R"(const\s+int\s+row\s*=\s*local_chunk_offset\s*\+\s*ii\s*;)");
  Require(std::regex_search(compact, ordinal_row),
          "compact gate-cache store no longer uses row=chunk_offset+ii");
  const std::regex compact_write(
      R"(\(\s*row\s*,\s*slot\s*\)\s*=)");
  Require(std::regex_search(compact, compact_write),
          "compact gate-cache store does not write its compact row and slot");
  Require(compact.find("d_ilist") == std::string::npos &&
              compact.find("d_local_ilist") == std::string::npos,
          "compact gate-cache store incorrectly indexes by atom id");

  Require(CountOccurrences(source, "\"StoreGateMomentCacheWide\"") == 1,
          "rank-off wide cache fallback is missing or duplicated");
  Require(CountOccurrences(source, "\"StoreGateMomentCache\"") == 1,
          "rank-off ordinary cache path is missing or duplicated");
  const std::string rank_off_wide = ParallelForContaining(
      source, "\"StoreGateMomentCacheWide\"");
  const std::string rank_off_flat = ParallelForContaining(
      source, "\"StoreGateMomentCache\"");
  Require(rank_off_wide.find("d_local_ilist") != std::string::npos &&
              rank_off_flat.find("d_local_ilist") != std::string::npos,
          "rank-off cache paths no longer retain atom-index storage");
  Require(source.find(
              "local_chunk_size > std::numeric_limits<int>::max() /") !=
              std::string::npos,
          "rank-off cache overflow predicate is missing");
  Require(CountOccurrences(
              source, "local_chunk_size * local_alpha_moment_count") == 1,
          "rank-off safe flattened cache path changed unexpectedly");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 2,
          "usage: lammps_kk_gate_cache_layout_check <pair_sus2_mtp_kokkos.cpp>");
  CheckUniqueOperandsAndStableSlots();
  CheckCurrentOuterRank2ScaleWithoutModelData();
  CheckInvalidInputsAndPackedDomainFallback();
  CheckListOrdinalChunkRows();
  CheckSourceGuard(ReadFile(argv[1]));
  std::cout << "lammps_kk_gate_cache_layout_check passed\n";
  return 0;
}
