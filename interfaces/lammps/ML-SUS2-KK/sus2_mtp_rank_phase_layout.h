/* -*- c++ -*- ----------------------------------------------------------
   Exact phase-local moment layout for ranked SUS2-SH Kokkos models.
------------------------------------------------------------------------- */

#ifndef LMP_SUS2_MTP_RANK_PHASE_LAYOUT_H
#define LMP_SUS2_MTP_RANK_PHASE_LAYOUT_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace LAMMPS_NS {
namespace SUS2RankPhase {

struct CompactLayout {
  int gate_width = 0;
  int outer_width = 0;
  int work_width = 0;
  int outer_contracted_begin = 0;
  int outer_product_count = 0;
  std::vector<int> outer_product_a0;
  std::vector<int> outer_product_a1;
  std::vector<int> outer_product_out;
  std::vector<int> outer_scalar_moments;
};

inline bool checked_matrix_bytes(std::size_t rows, std::size_t columns,
                                 std::size_t element_bytes,
                                 std::size_t &bytes)
{
  bytes = 0;
  if (element_bytes == 0) return false;
  if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
    return false;
  const std::size_t elements = rows * columns;
  if (elements > std::numeric_limits<std::size_t>::max() / element_bytes)
    return false;
  bytes = elements * element_bytes;
  return true;
}

inline int outer_physical_index(int logical, int basic_count,
                                int outer_basic_begin,
                                int logical_moment_count)
{
  if (logical >= 0 && logical < basic_count) return logical;
  if (logical >= outer_basic_begin && logical < logical_moment_count)
    return basic_count + (logical - outer_basic_begin);
  return -1;
}

inline bool build_compact_layout(
    int logical_moment_count, int basic_count, int gate_active_width,
    int gate_product_end, int outer_basic_begin, int outer_basic_count,
    int outer_product_begin, const int *product_a0, const int *product_a1,
    const int *product_out, int product_count, const int *scalar_moments,
    int scalar_count, const int *rank_raw_basics, int rank_raw_count,
    CompactLayout &layout, std::string &reason)
{
  layout = CompactLayout();
  reason.clear();
  if (logical_moment_count <= 0 || basic_count <= 0 ||
      basic_count > logical_moment_count) {
    reason = "invalid logical moment or raw-basic count";
    return false;
  }
  if (gate_active_width < basic_count ||
      gate_active_width > outer_basic_begin ||
      outer_basic_begin < basic_count ||
      outer_basic_begin > logical_moment_count) {
    reason = "invalid gate/unused/outer moment boundary";
    return false;
  }
  if (outer_basic_count <= 0 ||
      outer_basic_count > logical_moment_count - outer_basic_begin) {
    reason = "outer contracted range lies outside the logical suffix";
    return false;
  }
  if (product_count < 0 || scalar_count < 0 || rank_raw_count < 0 ||
      gate_product_end < 0 || gate_product_end > outer_product_begin ||
      outer_product_begin < 0 || outer_product_begin > product_count) {
    reason = "invalid gate/unused/outer product boundary";
    return false;
  }
  if (product_count > 0 &&
      (product_a0 == nullptr || product_a1 == nullptr ||
       product_out == nullptr)) {
    reason = "missing product metadata";
    return false;
  }
  if (scalar_count > 0 && scalar_moments == nullptr) {
    reason = "missing outer scalar metadata";
    return false;
  }
  if (rank_raw_count > 0 && rank_raw_basics == nullptr) {
    reason = "missing outer raw-basic metadata";
    return false;
  }

  const std::size_t outer_width_size =
      static_cast<std::size_t>(basic_count) +
      static_cast<std::size_t>(logical_moment_count - outer_basic_begin);
  if (outer_width_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    reason = "outer physical width exceeds the int index domain";
    return false;
  }
  const int outer_width = static_cast<int>(outer_width_size);
  const int outer_contracted_begin = basic_count;
  if (outer_basic_count > outer_width - outer_contracted_begin) {
    reason = "outer physical contracted range overflows its suffix";
    return false;
  }
  std::size_t checked_bytes = 0;
  const int work_width =
      gate_active_width > outer_width ? gate_active_width : outer_width;
  if (!checked_matrix_bytes(1, static_cast<std::size_t>(work_width),
                            sizeof(double), checked_bytes)) {
    reason = "rank phase work width overflows size_t";
    return false;
  }

  for (int k = 0; k < gate_product_end; ++k) {
    if (product_a0[k] < 0 || product_a0[k] >= gate_active_width ||
        product_a1[k] < 0 || product_a1[k] >= gate_active_width ||
        product_out[k] < 0 || product_out[k] >= gate_active_width) {
      reason = "executed gate product lies outside the active gate prefix";
      return false;
    }
  }
  for (int k = gate_product_end; k < outer_product_begin; ++k) {
    if (product_a0[k] < 0 || product_a0[k] >= outer_basic_begin ||
        product_a1[k] < 0 || product_a1[k] >= outer_basic_begin ||
        product_out[k] < 0 || product_out[k] >= outer_basic_begin) {
      reason = "unused pre-outer product crosses into the outer suffix";
      return false;
    }
  }

  const int outer_basic_end = outer_basic_begin + outer_basic_count;
  const int outer_count = product_count - outer_product_begin;
  if (!checked_matrix_bytes(static_cast<std::size_t>(outer_count), 3,
                            sizeof(int), checked_bytes)) {
    reason = "outer product metadata size overflows size_t";
    return false;
  }
  layout.outer_product_a0.reserve(static_cast<std::size_t>(outer_count));
  layout.outer_product_a1.reserve(static_cast<std::size_t>(outer_count));
  layout.outer_product_out.reserve(static_cast<std::size_t>(outer_count));
  for (int k = outer_product_begin; k < product_count; ++k) {
    const int a0 = product_a0[k];
    const int a1 = product_a1[k];
    const int out = product_out[k];
    if (a0 < outer_basic_begin || a0 >= logical_moment_count ||
        a1 < outer_basic_begin || a1 >= logical_moment_count ||
        out < outer_basic_end || out >= logical_moment_count) {
      reason = "outer product is not a closed suffix or overwrites a contracted primitive";
      layout = CompactLayout();
      return false;
    }
    if (a0 >= out || a1 >= out) {
      reason = "outer product graph is not in forward topological order";
      layout = CompactLayout();
      return false;
    }
    const int physical_a0 =
        outer_physical_index(a0, basic_count, outer_basic_begin,
                             logical_moment_count);
    const int physical_a1 =
        outer_physical_index(a1, basic_count, outer_basic_begin,
                             logical_moment_count);
    const int physical_out =
        outer_physical_index(out, basic_count, outer_basic_begin,
                             logical_moment_count);
    if (physical_a0 < 0 || physical_a0 >= outer_width ||
        physical_a1 < 0 || physical_a1 >= outer_width ||
        physical_out < 0 || physical_out >= outer_width) {
      reason = "outer product failed its logical-to-physical mapping";
      layout = CompactLayout();
      return false;
    }
    layout.outer_product_a0.push_back(physical_a0);
    layout.outer_product_a1.push_back(physical_a1);
    layout.outer_product_out.push_back(physical_out);
  }

  if (!checked_matrix_bytes(static_cast<std::size_t>(scalar_count), 1,
                            sizeof(int), checked_bytes)) {
    reason = "outer scalar metadata size overflows size_t";
    layout = CompactLayout();
    return false;
  }
  layout.outer_scalar_moments.reserve(static_cast<std::size_t>(scalar_count));
  for (int q = 0; q < scalar_count; ++q) {
    const int logical = scalar_moments[q];
    if (logical < outer_basic_begin || logical >= logical_moment_count) {
      reason = "outer scalar lies outside the logical outer suffix";
      layout = CompactLayout();
      return false;
    }
    const int physical = outer_physical_index(
        logical, basic_count, outer_basic_begin, logical_moment_count);
    if (physical < 0 || physical >= outer_width) {
      reason = "outer scalar failed its logical-to-physical mapping";
      layout = CompactLayout();
      return false;
    }
    layout.outer_scalar_moments.push_back(physical);
  }

  for (int k = 0; k < rank_raw_count; ++k) {
    const int raw = rank_raw_basics[k];
    if (raw < 0 || raw >= basic_count) {
      reason = "outer rank raw basic lies outside the identity raw prefix";
      layout = CompactLayout();
      return false;
    }
  }

  layout.gate_width = gate_active_width;
  layout.outer_width = outer_width;
  layout.work_width = work_width;
  layout.outer_contracted_begin = outer_contracted_begin;
  layout.outer_product_count = outer_count;
  return true;
}

}  // namespace SUS2RankPhase
}  // namespace LAMMPS_NS

#endif
