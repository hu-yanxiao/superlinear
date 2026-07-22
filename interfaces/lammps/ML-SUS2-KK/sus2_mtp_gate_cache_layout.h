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
