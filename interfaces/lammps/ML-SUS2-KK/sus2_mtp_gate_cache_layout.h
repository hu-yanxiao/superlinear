/* -*- c++ -*- ----------------------------------------------------------
   Exact compact-cache layout for the SUS2-SH Kokkos two-layer gate.
------------------------------------------------------------------------- */

#ifndef LMP_SUS2_MTP_GATE_CACHE_LAYOUT_H
#define LMP_SUS2_MTP_GATE_CACHE_LAYOUT_H

#include <string>
#include <vector>

namespace LAMMPS_NS {
namespace SUS2GateCache {

constexpr int kIndexBits = 12;
constexpr int kIndexMask = (1 << kIndexBits) - 1;

struct CompactLayout {
  std::vector<int> slot_to_moment;
  std::vector<int> packed_a0;
  std::vector<int> packed_a1;
};

inline bool build_active_moment_width(
    const int *product_a0, const int *product_a1, const int *product_out,
    int product_count, int basic_count, int moment_boundary,
    const int *scalar_moments, int scalar_count,
    const int *edge_l1_bases, int edge_l1_count,
    int rank_basic_begin, int rank_basic_count,
    const int *rank_raw_basics, int rank_raw_count,
    int &active_width, std::string &reason)
{
  active_width = moment_boundary;
  reason.clear();
  if (moment_boundary <= 0 || basic_count <= 0 ||
      basic_count > moment_boundary) {
    reason = "invalid basic or boundary width";
    return false;
  }
  if (product_count < 0 || scalar_count < 0 || edge_l1_count < 0 ||
      rank_basic_count < 0 || rank_raw_count < 0) {
    reason = "negative active-width count";
    return false;
  }
  if (product_count > 0 &&
      (product_a0 == nullptr || product_a1 == nullptr ||
       product_out == nullptr)) {
    reason = "missing active product array";
    return false;
  }
  if (scalar_count > 0 && scalar_moments == nullptr) {
    reason = "missing active scalar array";
    return false;
  }
  if (edge_l1_count > 0 && edge_l1_bases == nullptr) {
    reason = "missing active edge-L1 array";
    return false;
  }
  if (rank_raw_count > 0 && rank_raw_basics == nullptr) {
    reason = "missing active rank raw-basic array";
    return false;
  }
  if ((rank_basic_count == 0) != (rank_raw_count == 0)) {
    reason = "rank active-width metadata is incomplete";
    return false;
  }

  int width = basic_count;
  auto include_index = [&](int index, const char *source) {
    if (index < 0 || index >= moment_boundary) {
      reason = std::string(source) + " lies outside the gate boundary";
      return false;
    }
    if (index + 1 > width) width = index + 1;
    return true;
  };

  for (int k = 0; k < product_count; ++k) {
    if (!include_index(product_a0[k], "gate product a0") ||
        !include_index(product_a1[k], "gate product a1") ||
        !include_index(product_out[k], "gate product out"))
      return false;
  }
  for (int q = 0; q < scalar_count; ++q)
    if (!include_index(scalar_moments[q], "gate scalar moment"))
      return false;

  for (int source = 0; source < edge_l1_count; ++source) {
    const int base = edge_l1_bases[source];
    if (base < 0 || moment_boundary < 3 || base > moment_boundary - 3) {
      reason = "edge-L1 source lies outside the gate boundary";
      return false;
    }
    if (base + 3 > width) width = base + 3;
  }

  if (rank_basic_count > 0) {
    if (rank_basic_begin < basic_count ||
        rank_basic_begin > moment_boundary ||
        rank_basic_count > moment_boundary - rank_basic_begin) {
      reason = "rank contracted range lies outside the gate boundary";
      return false;
    }
    if (rank_basic_begin + rank_basic_count > width)
      width = rank_basic_begin + rank_basic_count;
    for (int k = 0; k < rank_raw_count; ++k) {
      const int raw = rank_raw_basics[k];
      if (raw < 0 || raw >= basic_count) {
        reason = "rank raw basic lies outside the raw-basic range";
        return false;
      }
    }
  }

  if (width <= 0 || width > moment_boundary) {
    reason = "active moment width is outside the gate boundary";
    return false;
  }
  active_width = width;
  return true;
}

inline int unpack_moment(int packed)
{
  return packed & kIndexMask;
}

inline int unpack_slot(int packed)
{
  return packed >> kIndexBits;
}

inline bool build_compact_layout(const int *product_a0,
                                 const int *product_a1,
                                 int product_count,
                                 int moment_width,
                                 CompactLayout &layout,
                                 std::string &reason)
{
  layout = CompactLayout();
  reason.clear();
  if (product_count < 0) {
    reason = "negative product count";
    return false;
  }
  if (moment_width <= 0 || moment_width > kIndexMask) {
    reason = "moment width exceeds the packed-index domain";
    return false;
  }
  if (product_count > 0 && (product_a0 == nullptr || product_a1 == nullptr)) {
    reason = "missing product operand array";
    return false;
  }

  std::vector<unsigned char> is_operand(
      static_cast<std::size_t>(moment_width), 0);
  for (int k = 0; k < product_count; ++k) {
    const int a0 = product_a0[k];
    const int a1 = product_a1[k];
    if (a0 < 0 || a0 >= moment_width || a1 < 0 || a1 >= moment_width) {
      reason = "product operand lies outside the cached gate prefix";
      return false;
    }
    is_operand[a0] = 1;
    is_operand[a1] = 1;
  }

  std::vector<int> moment_to_slot(
      static_cast<std::size_t>(moment_width), -1);
  for (int moment = 0; moment < moment_width; ++moment) {
    if (!is_operand[moment]) continue;
    moment_to_slot[moment] =
        static_cast<int>(layout.slot_to_moment.size());
    layout.slot_to_moment.push_back(moment);
  }
  if (layout.slot_to_moment.size() >
      static_cast<std::size_t>(kIndexMask)) {
    reason = "cache slot count exceeds the packed-index domain";
    layout = CompactLayout();
    return false;
  }

  layout.packed_a0.resize(static_cast<std::size_t>(product_count));
  layout.packed_a1.resize(static_cast<std::size_t>(product_count));
  for (int k = 0; k < product_count; ++k) {
    const int a0 = product_a0[k];
    const int a1 = product_a1[k];
    const int slot0 = moment_to_slot[a0];
    const int slot1 = moment_to_slot[a1];
    if (slot0 < 0 || slot1 < 0) {
      reason = "product operand has no compact cache slot";
      layout = CompactLayout();
      return false;
    }
    const int packed0 = a0 | (slot0 << kIndexBits);
    const int packed1 = a1 | (slot1 << kIndexBits);
    if (unpack_moment(packed0) != a0 || unpack_slot(packed0) != slot0 ||
        unpack_moment(packed1) != a1 || unpack_slot(packed1) != slot1 ||
        layout.slot_to_moment[slot0] != a0 ||
        layout.slot_to_moment[slot1] != a1) {
      reason = "packed compact-cache index failed its round trip";
      layout = CompactLayout();
      return false;
    }
    layout.packed_a0[k] = packed0;
    layout.packed_a1[k] = packed1;
  }
  return true;
}

}  // namespace SUS2GateCache
}  // namespace LAMMPS_NS

#endif
