/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#include "pair_sus2_mtp.h"

#include "sus2_mtp_radial_basis.h"
#include "sus2_mtp_rb_chebyshev_basis.h"
#include "sus2_mtp_rb_chebyshev_ss_basis.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
//#define LAMMPS_VERSION_NUMBER 20220324 // use the new neighbor list starting from this version

using namespace LAMMPS_NS;

namespace {

constexpr int kEnvGateChannels = 6;
constexpr double kEnvGateMaxLogDensityCoeff = 6.0;
constexpr double kEnvGateDefaultActivationOnRatio = 0.5;
constexpr int kMaxBodyGateStrideFast = 5;
constexpr int kMaxGateSignalGroupFast = 32;

bool env_flag_enabled(const char *name)
{
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int env_flag_override(const char *primary, const char *alias = nullptr)
{
  const char *value = std::getenv(primary);
  if ((value == nullptr || value[0] == '\0') && alias != nullptr)
    value = std::getenv(alias);
  if (value == nullptr || value[0] == '\0') return -1;
  if (value[0] == '0' ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0 ||
      std::strcmp(value, "no") == 0 || std::strcmp(value, "NO") == 0)
    return 0;
  return 1;
}

bool gate_required_subset_available(int required_basic_count,
                                    int alpha_index_basic_count)
{
  return required_basic_count > 0 &&
         required_basic_count < alpha_index_basic_count;
}

bool should_use_gate_required_basic_subset(int required_basic_count,
                                           int alpha_index_basic_count,
                                           bool required_basics_are_y00,
                                           bool body_linear_combo,
                                           bool center_gate_enabled,
                                           bool cache_body_signal_derivs,
                                           bool sh_basic_mu_fast,
                                           int gate_body_stride,
                                           int gate_signal_stride)
{
  if (!gate_required_subset_available(required_basic_count,
                                      alpha_index_basic_count))
    return false;
  const int override = env_flag_override("SUS2_LAMMPS_GATE_REQUIRED_BASIC",
                                        "SUS2_SH_GATE_REQUIRED_BASIC");
  if (override >= 0) return override != 0;

  if (required_basics_are_y00 && !center_gate_enabled &&
      !cache_body_signal_derivs && sh_basic_mu_fast)
    return true;

  return body_linear_combo && !center_gate_enabled && !cache_body_signal_derivs &&
         sh_basic_mu_fast && gate_body_stride > 0 && gate_body_stride <= 4 &&
         gate_signal_stride == gate_body_stride &&
         alpha_index_basic_count - required_basic_count >=
             std::max(16, alpha_index_basic_count / 4);
}

bool should_use_gate_required_dot(int required_basic_count,
                                  int alpha_index_basic_count,
                                  bool required_basics_are_y00,
                                  bool body_linear_combo,
                                  bool center_gate_enabled,
                                  bool cache_body_signal_derivs,
                                  bool sh_basic_mu_fast,
                                  int gate_body_stride,
                                  int gate_signal_stride)
{
  if (!gate_required_subset_available(required_basic_count,
                                      alpha_index_basic_count))
    return false;
  const int override = env_flag_override("SUS2_LAMMPS_GATE_REQUIRED_DOT",
                                        "SUS2_SH_GATE_REQUIRED_DOT");
  if (override >= 0) return override != 0;

  if (required_basics_are_y00 && !center_gate_enabled &&
      !cache_body_signal_derivs && sh_basic_mu_fast)
    return true;

  return body_linear_combo && !center_gate_enabled && !cache_body_signal_derivs &&
         sh_basic_mu_fast && gate_body_stride > 0 && gate_body_stride <= 4 &&
         gate_signal_stride == gate_body_stride &&
         required_basic_count * 2 <= alpha_index_basic_count;
}

bool disable_two_layer_gate_tanh_cache()
{
  static const bool disabled =
      env_flag_enabled("SUS2_LAMMPS_DISABLE_TWO_LAYER_GATE_TANH_CACHE") ||
      env_flag_enabled("SUS2_SH_DISABLE_TWO_LAYER_GATE_TANH_CACHE");
  return disabled;
}

int full_gate_force_jacobian_cache_override()
{
  const char *value = std::getenv("SUS2_LAMMPS_FULL_GATE_FORCE_JAC_CACHE");
  if (value == nullptr || value[0] == '\0')
    value = std::getenv("SUS2_SH_FULL_GATE_FORCE_JAC_CACHE");
  if (value == nullptr || value[0] == '\0') return -1;
  if (value[0] == '0') return 0;
  if (std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0 ||
      std::strcmp(value, "no") == 0 || std::strcmp(value, "NO") == 0)
    return 0;
  return 1;
}

bool should_cache_full_gate_force_jacobians(size_t edge_capacity,
                                           int alpha_index_basic_count)
{
  const int override = full_gate_force_jacobian_cache_override();
  if (override >= 0) return override != 0;
  (void) edge_capacity;
  (void) alpha_index_basic_count;
  return false;
}

enum {
  kGateProfileV2FirstRadial = 0,
  kGateProfileV2FirstSHBasic,
  kGateProfileV2FirstProductsSignal,
  kGateProfileV2FirstSignalDerivCache,
  kGateProfileV2ForwardComm,
  kGateProfileV2MainGateRadial,
  kGateProfileV2MainSHBasic,
  kGateProfileV2MainProductsBackprop,
  kGateProfileV2MainForceAdjoint,
  kGateProfileV2ReverseComm,
  kGateProfileV2GateForceSeedBackprop,
  kGateProfileV2GateForceJacobianDot,
  kGateProfileV2TimerCount
};

const char *kGateProfileV2TimerNames[kGateProfileV2TimerCount] = {
    "first_radial",
    "first_sh_basic",
    "first_products_signal",
    "first_signal_deriv_cache",
    "forward_comm",
    "main_gate_radial",
    "main_sh_basic",
    "main_products_backprop",
    "main_force_adjoint",
    "reverse_comm",
    "gate_force_seed_backprop",
    "gate_force_jacobian_dot"};

enum {
  kGateProfileV2CounterNLocal = 0,
  kGateProfileV2CounterNGhost,
  kGateProfileV2CounterNAll,
  kGateProfileV2CounterInum,
  kGateProfileV2CounterEdgeCapacity,
  kGateProfileV2CounterActiveEdges,
  kGateProfileV2CounterRadialFunc,
  kGateProfileV2CounterAlphaBasic,
  kGateProfileV2CounterAlphaTimes,
  kGateProfileV2CounterAlphaScalar,
  kGateProfileV2CounterGateProductLimit,
  kGateProfileV2CounterGateMomentWidth,
  kGateProfileV2CounterGateSignalStride,
  kGateProfileV2CounterGateScalarCount,
  kGateProfileV2CounterGateWeightCount,
  kGateProfileV2CounterGateBodyStride,
  kGateProfileV2CounterSHComponents,
  kGateProfileV2CounterBodyLinearCombo,
  kGateProfileV2CounterFullScalarWeights,
  kGateProfileV2CounterCenterGate,
  kGateProfileV2CounterSharedRadial,
  kGateProfileV2CounterRecomputeGateForceJac,
  kGateProfileV2CounterCacheFullGateForceJac,
  kGateProfileV2CounterSignalDerivCache,
  kGateProfileV2CounterCompactRaw,
  kGateProfileV2CounterFastMuBasic,
  kGateProfileV2CounterRequiredBasicSubset,
  kGateProfileV2CounterRequiredDot,
  kGateProfileV2CounterRequiredY00,
  kGateProfileV2CounterEdgeMetadataBytes,
  kGateProfileV2CounterRadialCacheBytes,
  kGateProfileV2CounterSHCacheBytes,
  kGateProfileV2CounterCompactRawBytes,
  kGateProfileV2CounterJacobianCacheBytes,
  kGateProfileV2CounterSignalDerivBytes,
  kGateProfileV2CounterMomentCacheBytes,
  kGateProfileV2CounterAtomGateBytes,
  kGateProfileV2CounterCount
};

const char *kGateProfileV2CounterNames[kGateProfileV2CounterCount] = {
    "nlocal",
    "nghost",
    "nall",
    "inum",
    "edge_capacity",
    "active_edges",
    "radial_func_count",
    "alpha_index_basic_count",
    "alpha_index_times_count",
    "alpha_scalar_count",
    "gate_product_limit",
    "gate_moment_cache_width",
    "gate_signal_stride",
    "gate_scalar_count",
    "gate_weight_count",
    "gate_body_stride",
    "sh_component_count",
    "body_linear_combo",
    "full_scalar_weights",
    "center_gate",
    "shared_radial",
    "recompute_gate_force_jac",
    "cache_full_gate_force_jac",
    "signal_deriv_cache",
    "compact_raw",
    "sh_basic_mu_fast",
    "required_basic_subset",
    "required_dot",
    "required_y00",
    "edge_metadata_bytes",
    "radial_cache_bytes",
    "sh_cache_bytes",
    "compact_raw_bytes",
    "jacobian_cache_bytes",
    "signal_deriv_bytes",
    "moment_cache_bytes",
    "atom_gate_bytes"};

inline void add_profile_time(bool enabled, double &dst, double start)
{
  if (enabled) dst += MPI_Wtime() - start;
}

inline double dot_small_stride(const double *weights,
                               const double *values,
                               int stride)
{
  switch (stride) {
  case 1:
    return weights[0] * values[0];
  case 2:
    return weights[0] * values[0] + weights[1] * values[1];
  case 3:
    return weights[0] * values[0] + weights[1] * values[1] +
           weights[2] * values[2];
  case 4:
    return weights[0] * values[0] + weights[1] * values[1] +
           weights[2] * values[2] + weights[3] * values[3];
  case 5:
    return weights[0] * values[0] + weights[1] * values[1] +
           weights[2] * values[2] + weights[3] * values[3] +
           weights[4] * values[4];
  default: {
    double result = 0.0;
    for (int i = 0; i < stride; i++) result += weights[i] * values[i];
    return result;
  }
  }
}

inline void accumulate_small_stride(double adjoint,
                                    const double *weights,
                                    int stride,
                                    double *accum)
{
  switch (stride) {
  case 1:
    accum[0] += adjoint * weights[0];
    return;
  case 2:
    accum[0] += adjoint * weights[0];
    accum[1] += adjoint * weights[1];
    return;
  case 3:
    accum[0] += adjoint * weights[0];
    accum[1] += adjoint * weights[1];
    accum[2] += adjoint * weights[2];
    return;
  case 4:
    accum[0] += adjoint * weights[0];
    accum[1] += adjoint * weights[1];
    accum[2] += adjoint * weights[2];
    accum[3] += adjoint * weights[3];
    return;
  case 5:
    accum[0] += adjoint * weights[0];
    accum[1] += adjoint * weights[1];
    accum[2] += adjoint * weights[2];
    accum[3] += adjoint * weights[3];
    accum[4] += adjoint * weights[4];
    return;
  default:
    for (int i = 0; i < stride; i++) accum[i] += adjoint * weights[i];
    return;
  }
}

template <int Stride>
inline double dot_fixed_stride(const double *a, const double *b)
{
  double result = 0.0;
  for (int i = 0; i < Stride; i++) result += a[i] * b[i];
  return result;
}

template <int Stride>
int pack_forward_gate_values_fixed(const int n,
                                   const int *list,
                                   const double *values,
                                   double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const double *__restrict src =
        values + static_cast<size_t>(list[i]) * Stride;
    double *__restrict dst = buf + m;
    for (int g = 0; g < Stride; g++) dst[g] = src[g];
    m += Stride;
  }
  return m;
}

inline bool pack_forward_gate_values_specialized(const int stride,
                                                 const int n,
                                                 const int *list,
                                                 const double *values,
                                                 double *buf,
                                                 int &packed_count)
{
  switch (stride) {
  case 1:
    packed_count = pack_forward_gate_values_fixed<1>(n, list, values, buf);
    return true;
  case 2:
    packed_count = pack_forward_gate_values_fixed<2>(n, list, values, buf);
    return true;
  case 3:
    packed_count = pack_forward_gate_values_fixed<3>(n, list, values, buf);
    return true;
  case 4:
    packed_count = pack_forward_gate_values_fixed<4>(n, list, values, buf);
    return true;
  case 5:
    packed_count = pack_forward_gate_values_fixed<5>(n, list, values, buf);
    return true;
  case 6:
    packed_count = pack_forward_gate_values_fixed<6>(n, list, values, buf);
    return true;
  case 8:
    packed_count = pack_forward_gate_values_fixed<8>(n, list, values, buf);
    return true;
  case 9:
    packed_count = pack_forward_gate_values_fixed<9>(n, list, values, buf);
    return true;
  case 10:
    packed_count = pack_forward_gate_values_fixed<10>(n, list, values, buf);
    return true;
  case 12:
    packed_count = pack_forward_gate_values_fixed<12>(n, list, values, buf);
    return true;
  case 15:
    packed_count = pack_forward_gate_values_fixed<15>(n, list, values, buf);
    return true;
  case 16:
    packed_count = pack_forward_gate_values_fixed<16>(n, list, values, buf);
    return true;
  case 20:
    packed_count = pack_forward_gate_values_fixed<20>(n, list, values, buf);
    return true;
  default:
    return false;
  }
}

template <int Stride>
void unpack_reverse_gate_adjoints_fixed(const int n,
                                        const int *list,
                                        const double *buf,
                                        double *adjoints)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    double *__restrict dst =
        adjoints + static_cast<size_t>(list[i]) * Stride;
    const double *__restrict src = buf + m;
    for (int g = 0; g < Stride; g++) dst[g] += src[g];
    m += Stride;
  }
}

inline bool unpack_reverse_gate_adjoints_specialized(const int stride,
                                                     const int n,
                                                     const int *list,
                                                     const double *buf,
                                                     double *adjoints)
{
  switch (stride) {
  case 1:
    unpack_reverse_gate_adjoints_fixed<1>(n, list, buf, adjoints);
    return true;
  case 2:
    unpack_reverse_gate_adjoints_fixed<2>(n, list, buf, adjoints);
    return true;
  case 3:
    unpack_reverse_gate_adjoints_fixed<3>(n, list, buf, adjoints);
    return true;
  case 4:
    unpack_reverse_gate_adjoints_fixed<4>(n, list, buf, adjoints);
    return true;
  case 5:
    unpack_reverse_gate_adjoints_fixed<5>(n, list, buf, adjoints);
    return true;
  case 6:
    unpack_reverse_gate_adjoints_fixed<6>(n, list, buf, adjoints);
    return true;
  case 8:
    unpack_reverse_gate_adjoints_fixed<8>(n, list, buf, adjoints);
    return true;
  case 9:
    unpack_reverse_gate_adjoints_fixed<9>(n, list, buf, adjoints);
    return true;
  case 10:
    unpack_reverse_gate_adjoints_fixed<10>(n, list, buf, adjoints);
    return true;
  case 12:
    unpack_reverse_gate_adjoints_fixed<12>(n, list, buf, adjoints);
    return true;
  case 15:
    unpack_reverse_gate_adjoints_fixed<15>(n, list, buf, adjoints);
    return true;
  case 16:
    unpack_reverse_gate_adjoints_fixed<16>(n, list, buf, adjoints);
    return true;
  case 20:
    unpack_reverse_gate_adjoints_fixed<20>(n, list, buf, adjoints);
    return true;
  default:
    return false;
  }
}

template <int BodyStride, bool FillCache>
void fill_body_combo_gate_table_radials_fixed(
    const int radial_func_count,
    const double *base_row,
    const double *base_next_row,
    const double *base_der_row,
    const double *base_der_next_row,
    const double ddr,
    const double additive_coeff,
    const double tanh_amplitude,
    const double *body_mix_weights,
    const double *gate_signal_by_mu,
    double *residual_radial_vals,
    double *radial_values,
    double *radial_derivatives,
    double *gate_multiplier_cache,
    double *gate_deriv_cache)
{
  #pragma omp simd
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double base_val =
        base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
    const double base_der =
        base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
    const double gate_signal = dot_fixed_stride<BodyStride>(
        body_mix_weights + static_cast<size_t>(mu) * BodyStride,
        gate_signal_by_mu);
    const double arg = additive_coeff * gate_signal;
    const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
    const double sech2 = 1.0 - tanh_arg * tanh_arg;
    const double gate_multiplier = 1.0 + tanh_amplitude * tanh_arg;
    const double gate_deriv = tanh_amplitude * additive_coeff * sech2;
    if (FillCache) {
      gate_multiplier_cache[mu] = gate_multiplier;
      gate_deriv_cache[mu] = gate_deriv;
    }
    residual_radial_vals[mu] = base_val * gate_deriv;
    radial_values[mu] = base_val * gate_multiplier;
    radial_derivatives[mu] = base_der * gate_multiplier;
  }
}

inline bool fill_body_combo_gate_table_radials_specialized(
    const int body_stride,
    const bool fill_cache,
    const int radial_func_count,
    const double *base_row,
    const double *base_next_row,
    const double *base_der_row,
    const double *base_der_next_row,
    const double ddr,
    const double additive_coeff,
    const double tanh_amplitude,
    const double *body_mix_weights,
    const double *gate_signal_by_mu,
    double *residual_radial_vals,
    double *radial_values,
    double *radial_derivatives,
    double *gate_multiplier_cache,
    double *gate_deriv_cache)
{
  switch (body_stride) {
  case 1:
    if (fill_cache)
      fill_body_combo_gate_table_radials_fixed<1, true>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    else
      fill_body_combo_gate_table_radials_fixed<1, false>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    return true;
  case 2:
    if (fill_cache)
      fill_body_combo_gate_table_radials_fixed<2, true>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    else
      fill_body_combo_gate_table_radials_fixed<2, false>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    return true;
  case 3:
    if (fill_cache)
      fill_body_combo_gate_table_radials_fixed<3, true>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    else
      fill_body_combo_gate_table_radials_fixed<3, false>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    return true;
  case 4:
    if (fill_cache)
      fill_body_combo_gate_table_radials_fixed<4, true>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    else
      fill_body_combo_gate_table_radials_fixed<4, false>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    return true;
  case 5:
    if (fill_cache)
      fill_body_combo_gate_table_radials_fixed<5, true>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    else
      fill_body_combo_gate_table_radials_fixed<5, false>(
          radial_func_count, base_row, base_next_row, base_der_row,
          base_der_next_row, ddr, additive_coeff, tanh_amplitude,
          body_mix_weights, gate_signal_by_mu, residual_radial_vals,
          radial_values, radial_derivatives, gate_multiplier_cache,
          gate_deriv_cache);
    return true;
  default:
    return false;
  }
}

template <bool FillCache>
void fill_full_grouped_gate_table_radials_fast(
    const int radial_func_count,
    const int gate_signal_count,
    const int *full_signal_group_for_mu,
    const double *base_row,
    const double *base_next_row,
    const double *base_der_row,
    const double *base_der_next_row,
    const double ddr,
    const double additive_coeff,
    const double tanh_amplitude,
    const double *gate_signal_by_group,
    double *residual_radial_vals,
    double *radial_values,
    double *radial_derivatives,
    double *gate_multiplier_cache,
    double *gate_deriv_cache)
{
  double group_multiplier[kMaxGateSignalGroupFast];
  double group_deriv[kMaxGateSignalGroupFast];
  for (int g = 0; g < gate_signal_count; g++) {
    const double arg = additive_coeff * gate_signal_by_group[g];
    const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
    const double sech2 = 1.0 - tanh_arg * tanh_arg;
    group_multiplier[g] = 1.0 + tanh_amplitude * tanh_arg;
    group_deriv[g] = tanh_amplitude * additive_coeff * sech2;
  }

  #pragma omp simd
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double base_val =
        base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
    const double base_der =
        base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
    const int group = full_signal_group_for_mu[mu];
    const double gate_multiplier = group_multiplier[group];
    const double gate_deriv = group_deriv[group];
    if (FillCache) {
      gate_multiplier_cache[mu] = gate_multiplier;
      gate_deriv_cache[mu] = gate_deriv;
    }
    residual_radial_vals[mu] = base_val * gate_deriv;
    radial_values[mu] = base_val * gate_multiplier;
    radial_derivatives[mu] = base_der * gate_multiplier;
  }
}

template <int Stride>
void accumulate_full_gate_values_fixed(const int scalar_count,
                                       const double *full_weights_by_scalar,
                                       const int *scalar_moment_indices,
                                       const double *moments,
                                       double *gate_values)
{
  double accum[Stride] = {};
  for (int q = 0; q < scalar_count; q++) {
    const double scalar_value = moments[scalar_moment_indices[q]];
    const double *__restrict weights =
        full_weights_by_scalar + static_cast<size_t>(q) * Stride;
    for (int g = 0; g < Stride; g++)
      accum[g] += scalar_value * weights[g];
  }
  for (int g = 0; g < Stride; g++) gate_values[g] = accum[g];
}

inline bool accumulate_full_gate_values_specialized(
    const int stride,
    const int scalar_count,
    const double *full_weights_by_scalar,
    const int *scalar_moment_indices,
    const double *moments,
    double *gate_values)
{
  switch (stride) {
  case 1:
    accumulate_full_gate_values_fixed<1>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 2:
    accumulate_full_gate_values_fixed<2>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 3:
    accumulate_full_gate_values_fixed<3>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 4:
    accumulate_full_gate_values_fixed<4>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 5:
    accumulate_full_gate_values_fixed<5>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 6:
    accumulate_full_gate_values_fixed<6>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 8:
    accumulate_full_gate_values_fixed<8>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 9:
    accumulate_full_gate_values_fixed<9>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 10:
    accumulate_full_gate_values_fixed<10>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 12:
    accumulate_full_gate_values_fixed<12>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 15:
    accumulate_full_gate_values_fixed<15>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 16:
    accumulate_full_gate_values_fixed<16>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  case 20:
    accumulate_full_gate_values_fixed<20>(
        scalar_count, full_weights_by_scalar, scalar_moment_indices, moments,
        gate_values);
    return true;
  default:
    return false;
  }
}

template <int Stride>
void accumulate_full_scalar_adjoints_fixed(const int scalar_count,
                                           const double *gate_adjoint,
                                           const double *full_weights_by_scalar,
                                           const int *scalar_moment_indices,
                                           double *moment_adjoints)
{
  for (int q = 0; q < scalar_count; q++) {
    const double *__restrict weights =
        full_weights_by_scalar + static_cast<size_t>(q) * Stride;
    const double adjoint = dot_fixed_stride<Stride>(gate_adjoint, weights);
    if (adjoint == 0.0) continue;
    moment_adjoints[scalar_moment_indices[q]] += adjoint;
  }
}

inline bool accumulate_full_scalar_adjoints_specialized(
    const int stride,
    const int scalar_count,
    const double *gate_adjoint,
    const double *full_weights_by_scalar,
    const int *scalar_moment_indices,
    double *moment_adjoints)
{
  switch (stride) {
  case 1:
    accumulate_full_scalar_adjoints_fixed<1>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 2:
    accumulate_full_scalar_adjoints_fixed<2>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 3:
    accumulate_full_scalar_adjoints_fixed<3>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 4:
    accumulate_full_scalar_adjoints_fixed<4>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 5:
    accumulate_full_scalar_adjoints_fixed<5>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 6:
    accumulate_full_scalar_adjoints_fixed<6>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 8:
    accumulate_full_scalar_adjoints_fixed<8>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 9:
    accumulate_full_scalar_adjoints_fixed<9>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 10:
    accumulate_full_scalar_adjoints_fixed<10>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 12:
    accumulate_full_scalar_adjoints_fixed<12>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 15:
    accumulate_full_scalar_adjoints_fixed<15>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 16:
    accumulate_full_scalar_adjoints_fixed<16>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  case 20:
    accumulate_full_scalar_adjoints_fixed<20>(
        scalar_count, gate_adjoint, full_weights_by_scalar,
        scalar_moment_indices, moment_adjoints);
    return true;
  default:
    return false;
  }
}

void accumulate_full_gate_values_batched(
    const int inum,
    const int *ilist,
    const int scalar_count,
    const int gate_signal_stride,
    const double *full_weights_by_scalar,
    const double *scalar_cache_by_q,
    double *gate_values)
{
  for (int ii = 0; ii < inum; ii++) {
    double *__restrict gate_values_i =
        gate_values + static_cast<size_t>(ilist[ii]) * gate_signal_stride;
    std::fill(gate_values_i, gate_values_i + gate_signal_stride, 0.0);
  }
  for (int q = 0; q < scalar_count; q++) {
    const double *__restrict weights =
        full_weights_by_scalar + static_cast<size_t>(q) * gate_signal_stride;
    const double *__restrict scalars =
        scalar_cache_by_q + static_cast<size_t>(q) * inum;
    for (int ii = 0; ii < inum; ii++) {
      const double scalar_value = scalars[ii];
      if (scalar_value == 0.0) continue;
      double *__restrict gate_values_i =
          gate_values + static_cast<size_t>(ilist[ii]) * gate_signal_stride;
      #pragma omp simd
      for (int g = 0; g < gate_signal_stride; g++)
        gate_values_i[g] += scalar_value * weights[g];
    }
  }
}

void accumulate_full_scalar_adjoints_batched(
    const int inum,
    const int *ilist,
    const int scalar_count,
    const int gate_signal_stride,
    const double *gate_adjoints,
    const double *full_weights_by_scalar,
    double *scalar_adjoint_cache_by_q)
{
  for (int q = 0; q < scalar_count; q++) {
    const double *__restrict weights =
        full_weights_by_scalar + static_cast<size_t>(q) * gate_signal_stride;
    double *__restrict scalar_adjoints =
        scalar_adjoint_cache_by_q + static_cast<size_t>(q) * inum;
    for (int ii = 0; ii < inum; ii++) {
      const double *__restrict gate_adjoint =
          gate_adjoints + static_cast<size_t>(ilist[ii]) * gate_signal_stride;
      double adjoint = 0.0;
      #pragma omp simd reduction(+:adjoint)
      for (int g = 0; g < gate_signal_stride; g++)
        adjoint += gate_adjoint[g] * weights[g];
      scalar_adjoints[ii] = adjoint;
    }
  }
}

double stable_sigmoid(double value)
{
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

double env_gate_density_coeff(double raw_log_density)
{
  return std::exp(std::min(raw_log_density, kEnvGateMaxLogDensityCoeff));
}

double clamp01(double value)
{
  return std::max(0.0, std::min(1.0, value));
}

void bernstein_degree5(double y, double *basis, double *basis_der)
{
  y = clamp01(y);
  const double u = 1.0 - y;
  const double y2 = y * y;
  const double y3 = y2 * y;
  const double y4 = y3 * y;
  const double y5 = y4 * y;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;

  basis[0] = u5;
  basis[1] = 5.0 * y * u4;
  basis[2] = 10.0 * y2 * u3;
  basis[3] = 10.0 * y3 * u2;
  basis[4] = 5.0 * y4 * u;
  basis[5] = y5;

  const double degree4[5] = {u4, 4.0 * y * u3, 6.0 * y2 * u2, 4.0 * y3 * u, y4};
  basis_der[0] = -5.0 * degree4[0];
  for (int q = 1; q < 5; ++q) basis_der[q] = 5.0 * (degree4[q - 1] - degree4[q]);
  basis_der[5] = 5.0 * degree4[4];
}

void env_gate_activation(double r,
                         double r_env,
                         double activation_on_ratio,
                         double *activation,
                         double *activation_der)
{
  const double r_on = activation_on_ratio * r_env;
  if (r <= r_on) {
    *activation = 0.0;
    *activation_der = 0.0;
    return;
  }
  if (r >= r_env) {
    *activation = 1.0;
    *activation_der = 0.0;
    return;
  }

  const double width = r_env - r_on;
  const double t = (r - r_on) / width;
  const double t2 = t * t;
  const double t3 = t2 * t;
  *activation = 10.0 * t3 - 15.0 * t3 * t + 6.0 * t3 * t2;
  *activation_der = 30.0 * t2 * (1.0 - t) * (1.0 - t) / width;
}

bool uses_preinterpolation_table(int basis_type)
{
  return basis_type == SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP ||
         basis_type == SUS2RadialMTPBasis::CHEBYSHEV_SSS_RATIONAL_LMP ||
         basis_type == SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP ||
         basis_type == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP ||
         basis_type == SUS2RadialMTPBasis::JACOBI_SSS_LMP;
}

bool basis_requires_per_mu_sigma(int basis_type)
{
  return basis_type == SUS2RadialMTPBasis::JACOBI_SSS_LMP;
}

std::string first_keyword(const std::string &line)
{
  std::istringstream stream(line);
  std::string keyword;
  stream >> keyword;
  return keyword;
}

std::string assignment_tail(std::string line)
{
  const std::size_t equals = line.find('=');
  if (equals != std::string::npos) line = line.substr(equals + 1);
  return line;
}

std::vector<double> parse_double_list_line(std::string line)
{
  line = assignment_tail(line);
  for (char &ch : line)
    if (ch == '{' || ch == '}' || ch == ',') ch = ' ';
  std::istringstream stream(line);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) values.push_back(value);
  return values;
}

std::vector<int> parse_int_list_line(std::string line)
{
  line = assignment_tail(line);
  for (char &ch : line)
    if (ch == '{' || ch == '}' || ch == ',') ch = ' ';
  std::istringstream stream(line);
  std::vector<int> values;
  int value = 0;
  while (stream >> value) values.push_back(value);
  return values;
}

std::string parse_string_assignment(std::string line)
{
  line = assignment_tail(line);
  std::istringstream stream(line);
  std::string value;
  stream >> value;
  return value;
}

int parse_int_assignment(const std::string &line)
{
  return std::atoi(parse_string_assignment(line).c_str());
}

double parse_double_assignment(const std::string &line)
{
  return std::atof(parse_string_assignment(line).c_str());
}

bool parse_bool_assignment(const std::string &line)
{
  const std::string value = parse_string_assignment(line);
  return value == "true" || value == "1" || value == "yes" || value == "on";
}

void reset_zbl_storage(LAMMPS_NS::Memory *memory,
                       int *&atomic_numbers,
                       double *&pair_inner_cutoffs,
                       double *&pair_outer_cutoffs,
                       double *&pair_outer_sq,
                       SUS2MTPZBLPairConstants *&pair_constants)
{
  memory->destroy(atomic_numbers);
  memory->destroy(pair_inner_cutoffs);
  memory->destroy(pair_outer_cutoffs);
  memory->destroy(pair_outer_sq);
  memory->destroy(pair_constants);
  atomic_numbers = nullptr;
  pair_inner_cutoffs = nullptr;
  pair_outer_cutoffs = nullptr;
  pair_outer_sq = nullptr;
  pair_constants = nullptr;
}

void interpolate_table_at_bin(double ***table,
                              double ***der_table,
                              int table_index,
                              int r_list,
                              double ddr,
                              int count,
                              double *values,
                              double *ders)
{
  const int r_next = r_list + 1;
  const double *__restrict row = table[table_index][r_list];
  const double *__restrict next_row = table[table_index][r_next];
  const double *__restrict der_row = der_table ? der_table[table_index][r_list] : nullptr;
  const double *__restrict der_next_row = der_table ? der_table[table_index][r_next] : nullptr;
  double *__restrict out_values = values;
  double *__restrict out_ders = ders;
  #pragma omp simd
  for (int m = 0; m < count; ++m) {
    out_values[m] = row[m] + ddr * (next_row[m] - row[m]);
    if (out_ders) out_ders[m] = der_row[m] + ddr * (der_next_row[m] - der_row[m]);
  }
}

void interpolate_table(double ***table,
                       double ***der_table,
                       int table_index,
                       int list_grid_size,
                       double inv_dr,
                       double dist,
                       int count,
                       double *values,
                       double *ders)
{
  const double scaled_dist = dist * inv_dr;
  int r_list = static_cast<int>(scaled_dist);
  const int last_interval = list_grid_size - 2;
  if (r_list < 0) r_list = 0;
  if (r_list > last_interval) r_list = last_interval;
  double ddr = scaled_dist - r_list;
  if (ddr < 0.0) ddr = 0.0;
  if (ddr > 1.0) ddr = 1.0;
  interpolate_table_at_bin(table, der_table, table_index, r_list, ddr, count,
                           values, ders);
}

constexpr int kMaxSHComponents = 25;
const double kPi = std::acos(-1.0);
const double kRealY00 = 0.5 / std::sqrt(kPi);
const double kRealY1 = 0.5 * std::sqrt(3.0 / kPi);
const double kRealY2A = 0.5 * std::sqrt(15.0 / kPi);
const double kRealY20 = 0.25 * std::sqrt(5.0 / kPi);
const double kRealY22 = 0.25 * std::sqrt(15.0 / kPi);
const double kRealY33 = 0.125 * std::sqrt(70.0 / kPi);
const double kRealY32 = 0.5 * std::sqrt(105.0 / kPi);
const double kRealY31 = 0.125 * std::sqrt(42.0 / kPi);
const double kRealY30 = 0.25 * std::sqrt(7.0 / kPi);
const double kRealY3p2 = 0.25 * std::sqrt(105.0 / kPi);
const double kRealY44m = 0.75 * std::sqrt(35.0 / kPi);
const double kRealY43 = 0.375 * std::sqrt(70.0 / kPi);
const double kRealY42m = 0.75 * std::sqrt(5.0 / kPi);
const double kRealY41 = 0.375 * std::sqrt(10.0 / kPi);
const double kRealY40 = 0.1875 / std::sqrt(kPi);
const double kRealY42 = 0.375 * std::sqrt(5.0 / kPi);
const double kRealY44 = 0.1875 * std::sqrt(35.0 / kPi);

int sh_flat_index(int l, int m)
{
  return l * l + (m + l);
}

double sh_inv_power(int l, double r)
{
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  if (l == 0) return 1.0;
  if (l == 1) return inv_r;
  if (l == 2) return inv_r2;
  if (l == 3) return inv_r2 * inv_r;
  if (l == 4) return inv_r2 * inv_r2;
  return std::pow(r, -l);
}

void add_real_sh(int l, int m, double coeff, double poly,
                 double dpx, double dpy, double dpz,
                 const double *rvec, double inv_pow, double inv_pow_der,
                 double *values, double *ders)
{
  const int idx = sh_flat_index(l, m);
  values[idx] = coeff * poly * inv_pow;
  ders[3 * idx + 0] = coeff * (dpx * inv_pow + poly * inv_pow_der * rvec[0]);
  ders[3 * idx + 1] = coeff * (dpy * inv_pow + poly * inv_pow_der * rvec[1]);
  ders[3 * idx + 2] = coeff * (dpz * inv_pow + poly * inv_pow_der * rvec[2]);
}

void add_real_sh_value(int l, int m, double coeff, double poly,
                       double inv_pow, double *values)
{
  values[sh_flat_index(l, m)] = coeff * poly * inv_pow;
}

void eval_real_sh_l1_compact(const double *rvec, double inv_r,
                             double *values, double *ders)
{
  const double x = rvec[0];
  const double y = rvec[1];
  const double z = rvec[2];
  const double inv_r3 = inv_r * inv_r * inv_r;
  values[0] = kRealY1 * y * inv_r;
  values[1] = kRealY1 * z * inv_r;
  values[2] = kRealY1 * x * inv_r;
  ders[0] = -kRealY1 * y * x * inv_r3;
  ders[1] = kRealY1 * (inv_r - y * y * inv_r3);
  ders[2] = -kRealY1 * y * z * inv_r3;
  ders[3] = -kRealY1 * z * x * inv_r3;
  ders[4] = -kRealY1 * z * y * inv_r3;
  ders[5] = kRealY1 * (inv_r - z * z * inv_r3);
  ders[6] = kRealY1 * (inv_r - x * x * inv_r3);
  ders[7] = -kRealY1 * x * y * inv_r3;
  ders[8] = -kRealY1 * x * z * inv_r3;
}

void eval_real_sh_l1_values_compact(const double *rvec, double inv_r,
                                    double *values)
{
  values[0] = kRealY1 * rvec[1] * inv_r;
  values[1] = kRealY1 * rvec[2] * inv_r;
  values[2] = kRealY1 * rvec[0] * inv_r;
}

void eval_real_sh_values(const double *rvec, double inv_r, int lmax,
                         double *values)
{
  const double x = rvec[0];
  const double y = rvec[1];
  const double z = rvec[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double inv_r2 = inv_r * inv_r;
  const double inv_pow0 = 1.0;
  const double inv_pow1 = inv_r;
  const double inv_pow2 = inv_r2;
  const double inv_pow3 = inv_r2 * inv_r;
  const double inv_pow4 = inv_r2 * inv_r2;

  add_real_sh_value(0, 0, kRealY00, 1.0, inv_pow0, values);
  if (lmax == 0) return;

  add_real_sh_value(1, -1, kRealY1, y, inv_pow1, values);
  add_real_sh_value(1, 0, kRealY1, z, inv_pow1, values);
  add_real_sh_value(1, 1, kRealY1, x, inv_pow1, values);
  if (lmax == 1) return;

  add_real_sh_value(2, -2, kRealY2A, x * y, inv_pow2, values);
  add_real_sh_value(2, -1, kRealY2A, y * z, inv_pow2, values);
  const double p20 = 2.0 * z2 - x2 - y2;
  add_real_sh_value(2, 0, kRealY20, p20, inv_pow2, values);
  add_real_sh_value(2, 1, kRealY2A, x * z, inv_pow2, values);
  add_real_sh_value(2, 2, kRealY22, x2 - y2, inv_pow2, values);
  if (lmax == 2) return;

  const double a31 = 4.0 * z2 - x2 - y2;
  const double p3m3 = 3.0 * x2 * y - y * y2;
  add_real_sh_value(3, -3, kRealY33, p3m3, inv_pow3, values);
  add_real_sh_value(3, -2, kRealY32, x * y * z, inv_pow3, values);
  add_real_sh_value(3, -1, kRealY31, y * a31, inv_pow3, values);
  const double p30 = z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
  add_real_sh_value(3, 0, kRealY30, p30, inv_pow3, values);
  add_real_sh_value(3, 1, kRealY31, x * a31, inv_pow3, values);
  const double p32 = z * (x2 - y2);
  add_real_sh_value(3, 2, kRealY3p2, p32, inv_pow3, values);
  const double p33 = x * x2 - 3.0 * x * y2;
  add_real_sh_value(3, 3, kRealY33, p33, inv_pow3, values);
  if (lmax == 3) return;

  const double rho2 = x2 + y2;
  const double a42 = 6.0 * z2 - rho2;
  const double a41 = 4.0 * z2 - 3.0 * rho2;
  const double p44base = x2 - y2;
  const double p4m4 = x * y * p44base;
  add_real_sh_value(4, -4, kRealY44m, p4m4, inv_pow4, values);
  add_real_sh_value(4, -3, kRealY43, z * p3m3, inv_pow4, values);
  add_real_sh_value(4, -2, kRealY42m, x * y * a42, inv_pow4, values);
  add_real_sh_value(4, -1, kRealY41, y * z * a41, inv_pow4, values);
  const double p40 = 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2;
  add_real_sh_value(4, 0, kRealY40, p40, inv_pow4, values);
  add_real_sh_value(4, 1, kRealY41, x * z * a41, inv_pow4, values);
  add_real_sh_value(4, 2, kRealY42, p44base * a42, inv_pow4, values);
  add_real_sh_value(4, 3, kRealY43, z * p33, inv_pow4, values);
  const double p44 = x2 * x2 - 6.0 * x2 * y2 + y2 * y2;
  add_real_sh_value(4, 4, kRealY44, p44, inv_pow4, values);
}

void eval_real_sh(const double *rvec, double inv_r, int lmax, double *values, double *ders)
{
  const double x = rvec[0];
  const double y = rvec[1];
  const double z = rvec[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double inv_r2 = inv_r * inv_r;
  const double inv_pow0 = 1.0;
  const double inv_pow1 = inv_r;
  const double inv_pow2 = inv_r2;
  const double inv_pow3 = inv_r2 * inv_r;
  const double inv_pow4 = inv_r2 * inv_r2;
  const double inv_der0 = 0.0;
  const double inv_der1 = -1.0 * inv_pow1 * inv_r * inv_r;
  const double inv_der2 = -2.0 * inv_pow2 * inv_r * inv_r;
  const double inv_der3 = -3.0 * inv_pow3 * inv_r * inv_r;
  const double inv_der4 = -4.0 * inv_pow4 * inv_r * inv_r;

  add_real_sh(0, 0, kRealY00, 1.0, 0.0, 0.0, 0.0, rvec,
              inv_pow0, inv_der0, values, ders);
  if (lmax == 0) return;

  add_real_sh(1, -1, kRealY1, y, 0.0, 1.0, 0.0, rvec,
              inv_pow1, inv_der1, values, ders);
  add_real_sh(1, 0, kRealY1, z, 0.0, 0.0, 1.0, rvec,
              inv_pow1, inv_der1, values, ders);
  add_real_sh(1, 1, kRealY1, x, 1.0, 0.0, 0.0, rvec,
              inv_pow1, inv_der1, values, ders);
  if (lmax == 1) return;

  add_real_sh(2, -2, kRealY2A, x * y, y, x, 0.0, rvec,
              inv_pow2, inv_der2, values, ders);
  add_real_sh(2, -1, kRealY2A, y * z, 0.0, z, y, rvec,
              inv_pow2, inv_der2, values, ders);
  const double p20 = 2.0 * z2 - x2 - y2;
  add_real_sh(2, 0, kRealY20, p20, -2.0 * x, -2.0 * y, 4.0 * z, rvec,
              inv_pow2, inv_der2, values, ders);
  add_real_sh(2, 1, kRealY2A, x * z, z, 0.0, x, rvec,
              inv_pow2, inv_der2, values, ders);
  add_real_sh(2, 2, kRealY22, x2 - y2, 2.0 * x, -2.0 * y, 0.0, rvec,
              inv_pow2, inv_der2, values, ders);
  if (lmax == 2) return;

  const double a31 = 4.0 * z2 - x2 - y2;
  const double p3m3 = 3.0 * x2 * y - y * y2;
  add_real_sh(3, -3, kRealY33, p3m3, 6.0 * x * y, 3.0 * x2 - 3.0 * y2, 0.0,
              rvec, inv_pow3, inv_der3, values, ders);
  add_real_sh(3, -2, kRealY32, x * y * z, y * z, x * z, x * y,
              rvec, inv_pow3, inv_der3, values, ders);
  add_real_sh(3, -1, kRealY31, y * a31, -2.0 * x * y, a31 - 2.0 * y2,
              8.0 * y * z, rvec, inv_pow3, inv_der3, values, ders);
  const double p30 = z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
  add_real_sh(3, 0, kRealY30, p30, -6.0 * x * z, -6.0 * y * z,
              6.0 * z2 - 3.0 * x2 - 3.0 * y2, rvec,
              inv_pow3, inv_der3, values, ders);
  add_real_sh(3, 1, kRealY31, x * a31, a31 - 2.0 * x2, -2.0 * x * y,
              8.0 * x * z, rvec, inv_pow3, inv_der3, values, ders);
  const double p32 = z * (x2 - y2);
  add_real_sh(3, 2, kRealY3p2, p32, 2.0 * x * z, -2.0 * y * z, x2 - y2,
              rvec, inv_pow3, inv_der3, values, ders);
  const double p33 = x * x2 - 3.0 * x * y2;
  add_real_sh(3, 3, kRealY33, p33, 3.0 * x2 - 3.0 * y2, -6.0 * x * y, 0.0,
              rvec, inv_pow3, inv_der3, values, ders);
  if (lmax == 3) return;

  const double rho2 = x2 + y2;
  const double a42 = 6.0 * z2 - rho2;
  const double a41 = 4.0 * z2 - 3.0 * rho2;
  const double p44base = x2 - y2;
  const double p4m4 = x * y * p44base;
  add_real_sh(4, -4, kRealY44m, p4m4, y * (3.0 * x2 - y2),
              x * (x2 - 3.0 * y2), 0.0, rvec, inv_pow4, inv_der4,
              values, ders);
  add_real_sh(4, -3, kRealY43, z * p3m3, 6.0 * x * y * z,
              z * (3.0 * x2 - 3.0 * y2), p3m3, rvec,
              inv_pow4, inv_der4, values, ders);
  add_real_sh(4, -2, kRealY42m, x * y * a42,
              y * a42 - 2.0 * x2 * y, x * a42 - 2.0 * x * y2,
              12.0 * x * y * z, rvec, inv_pow4, inv_der4, values, ders);
  add_real_sh(4, -1, kRealY41, y * z * a41, -6.0 * x * y * z,
              z * (a41 - 6.0 * y2), y * (12.0 * z2 - 3.0 * rho2),
              rvec, inv_pow4, inv_der4, values, ders);
  const double p40 = 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2;
  add_real_sh(4, 0, kRealY40, p40, 12.0 * x * (rho2 - 4.0 * z2),
              12.0 * y * (rho2 - 4.0 * z2),
              16.0 * z * (2.0 * z2 - 3.0 * rho2), rvec,
              inv_pow4, inv_der4, values, ders);
  add_real_sh(4, 1, kRealY41, x * z * a41, z * (a41 - 6.0 * x2),
              -6.0 * x * y * z, x * (12.0 * z2 - 3.0 * rho2),
              rvec, inv_pow4, inv_der4, values, ders);
  add_real_sh(4, 2, kRealY42, p44base * a42,
              2.0 * x * a42 - 2.0 * x * p44base,
              -2.0 * y * a42 - 2.0 * y * p44base,
              12.0 * z * p44base, rvec, inv_pow4, inv_der4, values, ders);
  add_real_sh(4, 3, kRealY43, z * p33, z * (3.0 * x2 - 3.0 * y2),
              -6.0 * x * y * z, p33, rvec, inv_pow4, inv_der4, values, ders);
  const double p44 = x2 * x2 - 6.0 * x2 * y2 + y2 * y2;
  add_real_sh(4, 4, kRealY44, p44, 4.0 * x * x2 - 12.0 * x * y2,
              -12.0 * x2 * y + 4.0 * y * y2, 0.0, rvec,
              inv_pow4, inv_der4, values, ders);
}

}    // namespace

PairSUS2MTP::PairSUS2MTP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
//#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag = CENTROID_AVAIL;
//#else
//  centroidstressflag = 2;
//#endif
  //centroidstressflag = CENTROID_AVAIL;
}

/* ---------------------------------------------------------------------- */

PairSUS2MTP::~PairSUS2MTP()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(radial_list);
	    memory->destroy(radial_der_list);
	    memory->destroy(two_layer_gate_radial_list);
	    memory->destroy(two_layer_gate_radial_der_list);
	    memory->destroy(env_gate_radial_list);
	    memory->destroy(env_gate_radial_der_list);
	    memory->destroy(env_gate_rho_list);
	    memory->destroy(env_gate_rho_der_list);
	    memory->destroy(pair_to_table_index);
	    memory->destroy(zbl_atomic_numbers);
	    memory->destroy(zbl_pair_inner_cutoffs);
	    memory->destroy(zbl_pair_outer_cutoffs);
	    memory->destroy(zbl_pair_outer_sq);
	    memory->destroy(zbl_pair_constants);
	    memory->destroy(moment_tensor_vals);
    memory->destroy(regression_coeffs);
    memory->destroy(linear_coeffs);
    memory->destroy(species_coeffs);
    memory->destroy(alpha_index_basic);
    memory->destroy(alpha_index_times);
    memory->destroy(alpha_moment_mapping);
    memory->destroy(alpha_basic_mu);
    memory->destroy(alpha_basic_a0);
    memory->destroy(alpha_basic_a1);
    memory->destroy(alpha_basic_a2);
    memory->destroy(alpha_basic_norm_rank);
    memory->destroy(alpha_basic_sh_index);
    memory->destroy(alpha_times_a0);
    memory->destroy(alpha_times_a1);
    memory->destroy(alpha_times_multiplier);
    memory->destroy(alpha_times_coeff);
    memory->destroy(alpha_times_out);
    memory->destroy(moment_jacobian_x);
    memory->destroy(moment_jacobian_y);
    memory->destroy(moment_jacobian_z);
    memory->destroy(nbh_energy_ders_wrt_moments);
    memory->destroy(within_cutoff);
    memory->destroy(inv_dist_powers);
    memory->destroy(coord_powers_x);
    memory->destroy(coord_powers_y);
    memory->destroy(coord_powers_z);
    memory->destroy(radial_vals);
    memory->destroy(radial_ders);
    memory->destroy(two_layer_raw_basic_vals);
    memory->destroy(two_layer_center_raw_basic_vals);
    memory->destroy(two_layer_gate_residual_radial_vals);
    memory->destroy(two_layer_gate_center_residual_radial_vals);
    memory->destroy(two_layer_gate_adjoint_scratch);
    memory->destroy(two_layer_gate_center_adjoint_scratch);
    memory->destroy(two_layer_gate_values);
    memory->destroy(two_layer_gate_adjoints);
    memory->destroy(two_layer_gate_multiplier_mu_cache);
    memory->destroy(two_layer_gate_deriv_mu_cache);
    memory->destroy(two_layer_gate_mu_cache_valid);
    memory->destroy(two_layer_radial_cache_vals);
    memory->destroy(two_layer_radial_cache_ders);
    memory->destroy(two_layer_gate_edge_neighbors_raw);
    memory->destroy(two_layer_gate_edge_types_raw);
    memory->destroy(two_layer_gate_edge_table_indices_raw);
    memory->destroy(two_layer_gate_edge_table_bins_raw);
    memory->destroy(two_layer_gate_edge_dx_raw);
    memory->destroy(two_layer_gate_edge_dy_raw);
    memory->destroy(two_layer_gate_edge_dz_raw);
    memory->destroy(two_layer_gate_edge_dist_raw);
    memory->destroy(two_layer_gate_edge_table_fracs_raw);
    memory->destroy(two_layer_gate_edge_radial_vals_raw);
    memory->destroy(two_layer_gate_edge_radial_ders_raw);
    memory->destroy(two_layer_gate_edge_sh_values_raw);
    memory->destroy(two_layer_gate_edge_sh_ders_raw);
    memory->destroy(two_layer_gate_edge_main_sh_values_raw);
    memory->destroy(two_layer_gate_edge_residual_radial_vals_raw);
    memory->destroy(two_layer_gate_edge_deriv_x_raw);
    memory->destroy(two_layer_gate_edge_deriv_y_raw);
    memory->destroy(two_layer_gate_edge_deriv_z_raw);
    memory->destroy(two_layer_gate_edge_signal_deriv_x_raw);
    memory->destroy(two_layer_gate_edge_signal_deriv_y_raw);
    memory->destroy(two_layer_gate_edge_signal_deriv_z_raw);
    memory->destroy(two_layer_gate_edge_l1_weighted_values);
    memory->destroy(two_layer_gate_edge_l1_weighted_adjoints);
    memory->destroy(weighted_basic_moment_ders);
    memory->destroy(static_fixed_basic_cache);
    memory->destroy(static_fixed_basic_cache_valid);
    memory->destroy(static_fixed_basic_cache_tags);
    memory->destroy(static_fixed_gate_basic_cache);
    memory->destroy(static_fixed_gate_basic_cache_valid);
    memory->destroy(static_fixed_gate_value_cache);
    memory->destroy(static_fixed_gate_value_cache_valid);
    memory->destroy(static_fixed_gate_cache_tags);
    memory->destroy(env_rho_dr);
    memory->destroy(env_activation_basic_vals);
    memory->destroy(mu_to_K);
    memory->destroy(mu_to_sigma);

    delete radial_basis;
    radial_basis = nullptr;
  }
}

/* ----------------------------------------------------------------------
   Two-layer SUS2-SH helpers
------------------------------------------------------------------------- */

bool PairSUS2MTP::has_nonzero_two_layer_gate_weights() const
{
  for (double weight : two_layer_gate_weights)
    if (weight != 0.0) return true;
  return false;
}

bool PairSUS2MTP::has_nonzero_two_layer_gate_edge_l1_weights() const
{
  if (!two_layer_gate_edge_l1_enabled) return false;
  for (double weight : two_layer_gate_edge_l1_weights)
    if (weight != 0.0) return true;
  return false;
}

bool PairSUS2MTP::two_layer_gate_edge_l1_active() const
{
  return two_layer_gate_enabled && two_layer_gate_edge_l1_enabled &&
         has_nonzero_two_layer_gate_edge_l1_weights();
}

bool PairSUS2MTP::requires_two_layer_gate_sh() const
{
  return is_sh_model && two_layer_gate_enabled &&
         (has_nonzero_two_layer_gate_weights() ||
          has_nonzero_two_layer_gate_edge_l1_weights());
}

int PairSUS2MTP::two_layer_gate_signal_stride() const
{
  if (!two_layer_gate_enabled) return 0;
  if (two_layer_gate_body_linear_combo) {
    if (!two_layer_gate_body_signal_buckets.empty())
      return static_cast<int>(two_layer_gate_body_signal_buckets.size());
    return two_layer_gate_body_order_max - 1;
  }
  if (two_layer_gate_full_signal_count > 0)
    return two_layer_gate_full_signal_count;
  return radial_func_count;
}

int PairSUS2MTP::two_layer_gate_edge_l1_weighted_stride() const
{
  return two_layer_gate_edge_l1_active() ? 3 * radial_func_count : 0;
}

int PairSUS2MTP::two_layer_gate_comm_stride() const
{
  return two_layer_gate_signal_stride() + two_layer_gate_edge_l1_weighted_stride();
}

double PairSUS2MTP::two_layer_gate_mu_signal(const double *gate_signal,
                                             int mu) const
{
  if (gate_signal == nullptr) return 0.0;
  if (mu < 0 || mu >= radial_func_count)
    error->one(FLERR, "SUS2-SH two-layer gate mu index is out of range.");
  if (!two_layer_gate_body_linear_combo) {
    if (two_layer_gate_full_identity_signal) return gate_signal[mu];
    if (static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
        radial_func_count) {
      const int group = two_layer_gate_full_signal_group_for_mu[mu];
      if (group < 0 || group >= two_layer_gate_signal_stride())
        error->one(FLERR, "SUS2-SH full gate signal group is out of range.");
      return gate_signal[group];
    }
    return gate_signal[mu];
  }

  const int gate_body_stride = two_layer_gate_body_order_max - 1;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  if (gate_body_stride <= 0 || gate_signal_stride <= 0)
    error->one(FLERR, "SUS2-SH two-layer gate body stride is invalid.");
  const double *weights = !two_layer_gate_body_signal_mix_weights.empty()
      ? two_layer_gate_body_signal_mix_weights.data() +
            static_cast<size_t>(mu) * gate_signal_stride
      : two_layer_gate_body_mix_weights.data() +
            static_cast<size_t>(mu) * gate_body_stride;
  return dot_small_stride(weights, gate_signal, gate_signal_stride);
}

void PairSUS2MTP::accumulate_two_layer_gate_signal_adjoints(
    double *gate_signal_adjoints,
    const double *gate_adjoint_by_mu) const
{
  if (gate_signal_adjoints == nullptr || gate_adjoint_by_mu == nullptr) return;
  if (!two_layer_gate_body_linear_combo) {
    if (two_layer_gate_full_identity_signal) {
      #pragma omp simd
      for (int mu = 0; mu < radial_func_count; mu++)
        gate_signal_adjoints[mu] += gate_adjoint_by_mu[mu];
    } else if (static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
        radial_func_count) {
      const int group_count = two_layer_gate_signal_stride();
      if (group_count == radial_func_count) {
        #pragma omp simd
        for (int mu = 0; mu < radial_func_count; mu++)
          gate_signal_adjoints[mu] += gate_adjoint_by_mu[mu];
      } else if (group_count == 1) {
        double adjoint = 0.0;
        #pragma omp simd reduction(+:adjoint)
        for (int mu = 0; mu < radial_func_count; mu++)
          adjoint += gate_adjoint_by_mu[mu];
        gate_signal_adjoints[0] += adjoint;
      } else {
        for (int mu = 0; mu < radial_func_count; mu++)
          gate_signal_adjoints[two_layer_gate_full_signal_group_for_mu[mu]] +=
              gate_adjoint_by_mu[mu];
      }
    } else {
      #pragma omp simd
      for (int mu = 0; mu < radial_func_count; mu++)
        gate_signal_adjoints[mu] += gate_adjoint_by_mu[mu];
    }
    return;
  }

  const int gate_body_stride = two_layer_gate_body_order_max - 1;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  if (gate_signal_stride > 0 && gate_signal_stride <= kMaxBodyGateStrideFast) {
    double body_adjoints[kMaxBodyGateStrideFast] = {};
    for (int mu = 0; mu < radial_func_count; mu++) {
      const double adjoint = gate_adjoint_by_mu[mu];
      if (adjoint == 0.0) continue;
      const double *weights =
          !two_layer_gate_body_signal_mix_weights.empty()
          ? two_layer_gate_body_signal_mix_weights.data() +
                static_cast<size_t>(mu) * gate_signal_stride
          : two_layer_gate_body_mix_weights.data() +
                static_cast<size_t>(mu) * gate_body_stride;
      accumulate_small_stride(adjoint, weights, gate_signal_stride, body_adjoints);
    }
    for (int b = 0; b < gate_signal_stride; b++)
      gate_signal_adjoints[b] += body_adjoints[b];
    return;
  }
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double adjoint = gate_adjoint_by_mu[mu];
    if (adjoint == 0.0) continue;
    const double *weights =
        !two_layer_gate_body_signal_mix_weights.empty()
        ? two_layer_gate_body_signal_mix_weights.data() +
              static_cast<size_t>(mu) * gate_signal_stride
        : two_layer_gate_body_mix_weights.data() +
              static_cast<size_t>(mu) * gate_body_stride;
    for (int b = 0; b < gate_signal_stride; b++)
      gate_signal_adjoints[b] += adjoint * weights[b];
  }
}

int PairSUS2MTP::find_two_layer_gate_edge_l1_raw_source_moment_index(int mu) const
{
  if (mu < 0 || mu >= radial_func_count)
    error->all(FLERR, "SUS2-SH edge L1 gate source mu index is out of range.");
  for (int k = 0; k + 2 < alpha_index_basic_count; k++) {
    if (alpha_basic_mu[k] == mu &&
        alpha_basic_a0[k] == 1 && alpha_basic_a1[k] == -1 &&
        alpha_basic_mu[k + 1] == mu &&
        alpha_basic_a0[k + 1] == 1 && alpha_basic_a1[k + 1] == 0 &&
        alpha_basic_mu[k + 2] == mu &&
        alpha_basic_a0[k + 2] == 1 && alpha_basic_a1[k + 2] == 1)
      return k;
  }
  error->all(FLERR, "SUS2-SH edge L1 gate source mu does not map to a basic l=1 moment.");
  return -1;
}

void PairSUS2MTP::prepare_two_layer_gate_edge_l1_layout()
{
  two_layer_gate_edge_l1_weights_by_source.clear();
  if (!two_layer_gate_edge_l1_enabled) return;
  if (!two_layer_gate_enabled)
    error->all(FLERR, "SUS2-SH edge L1 gate requires two-layer gate.");
  if (two_layer_gate_center_enabled)
    error->all(FLERR, "SUS2-SH edge L1 gate supports neighbor site mode only in LAMMPS.");
  if (sh_l_max < 1)
    error->all(FLERR, "SUS2-SH edge L1 gate requires sh_l_max >= 1.");
  if (two_layer_gate_edge_l1_source_count <= 0 ||
      static_cast<int>(two_layer_gate_edge_l1_source_moment_indices.size()) !=
          two_layer_gate_edge_l1_source_count)
    error->all(FLERR, "SUS2-SH edge L1 gate source metadata is inconsistent.");
  const int expected_weight_count =
      radial_func_count * two_layer_gate_edge_l1_source_count;
  if (two_layer_gate_edge_l1_weight_count != expected_weight_count ||
      static_cast<int>(two_layer_gate_edge_l1_weights.size()) !=
          expected_weight_count)
    error->all(FLERR, "SUS2-SH edge L1 gate weight metadata is inconsistent.");
  for (int source = 0; source < two_layer_gate_edge_l1_source_count; source++) {
    const int base = two_layer_gate_edge_l1_source_moment_indices[source];
    if (base < 0 || base + 2 >= alpha_moment_count)
      error->all(FLERR, "SUS2-SH edge L1 gate source moment index is out of range.");
  }
  two_layer_gate_edge_l1_weights_by_source.assign(
      static_cast<size_t>(two_layer_gate_edge_l1_source_count) *
          radial_func_count,
      0.0);
  for (int source = 0; source < two_layer_gate_edge_l1_source_count; source++) {
    double *dst = two_layer_gate_edge_l1_weights_by_source.data() +
        static_cast<size_t>(source) * radial_func_count;
    for (int mu = 0; mu < radial_func_count; mu++)
      dst[mu] = two_layer_gate_edge_l1_weights[
          static_cast<size_t>(mu) * two_layer_gate_edge_l1_source_count +
          source];
  }
}

void PairSUS2MTP::fill_two_layer_gate_edge_l1_weighted_values_for_atom(
    int atom_index,
    const double *moments)
{
  if (!two_layer_gate_edge_l1_active()) return;
  if (atom_index < 0 || atom_index >= atom->nmax)
    error->one(FLERR, "SUS2-SH edge L1 gate atom index is out of range.");
  const int source_count = two_layer_gate_edge_l1_source_count;
  double *u_atom = two_layer_gate_edge_l1_weighted_values +
      static_cast<size_t>(atom_index) * radial_func_count * 3;
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double *weights =
        two_layer_gate_edge_l1_weights.data() +
        static_cast<size_t>(mu) * source_count;
    double u0 = 0.0;
    double u1 = 0.0;
    double u2 = 0.0;
    for (int source = 0; source < source_count; source++) {
      const int base = two_layer_gate_edge_l1_source_moment_indices[source];
      const double w = weights[source];
      u0 += w * moments[base + 0];
      u1 += w * moments[base + 1];
      u2 += w * moments[base + 2];
    }
    double *u = u_atom + 3 * mu;
    u[0] = u0;
    u[1] = u1;
    u[2] = u2;
  }
}

void PairSUS2MTP::prepare_two_layer_gate_edge_l1_signal(
    int atom_index,
    const double *base_gate_signal,
    double dist,
    const double *r,
    double *out_signal)
{
  const double inv_dist = 1.0 / dist;
  double sh_values[4] = {};
  eval_real_sh_values(r, inv_dist, 1, sh_values);
  const double y1[3] = {
      sh_values[sh_flat_index(1, -1)],
      sh_values[sh_flat_index(1, 0)],
      sh_values[sh_flat_index(1, 1)]};
  const double *u_atom = two_layer_gate_edge_l1_weighted_values +
      static_cast<size_t>(atom_index) * radial_func_count * 3;
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double *u = u_atom + 3 * mu;
    out_signal[mu] =
        two_layer_gate_mu_signal(base_gate_signal, mu) +
        u[0] * y1[0] + u[1] * y1[1] + u[2] * y1[2];
  }
}

void PairSUS2MTP::accumulate_two_layer_gate_edge_l1_u_adjoints(
    int atom_index,
    const double *r,
    const double *gate_adjoint_by_mu)
{
  if (!two_layer_gate_edge_l1_active()) return;
  if (atom_index < 0 || atom_index >= atom->nmax)
    error->one(FLERR, "SUS2-SH edge L1 gate adjoint atom index is out of range.");
  const double dist = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  const double inv_dist = 1.0 / dist;
  double sh_values[4] = {};
  eval_real_sh_values(r, inv_dist, 1, sh_values);
  const double y1[3] = {
      sh_values[sh_flat_index(1, -1)],
      sh_values[sh_flat_index(1, 0)],
      sh_values[sh_flat_index(1, 1)]};
  double *u_adj_atom = two_layer_gate_edge_l1_weighted_adjoints +
      static_cast<size_t>(atom_index) * radial_func_count * 3;
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double adj = gate_adjoint_by_mu[mu];
    if (adj == 0.0) continue;
    double *u_adj = u_adj_atom + 3 * mu;
    u_adj[0] += adj * y1[0];
    u_adj[1] += adj * y1[1];
    u_adj[2] += adj * y1[2];
  }
}

void PairSUS2MTP::seed_two_layer_gate_edge_l1_moment_adjoints(
    int atom_index,
    double *moment_adjoints) const
{
  if (!two_layer_gate_edge_l1_active()) return;
  if (atom_index < 0 || atom_index >= atom->nmax)
    error->one(FLERR, "SUS2-SH edge L1 gate seed atom index is out of range.");
  const int source_count = two_layer_gate_edge_l1_source_count;
  const double *u_adj_atom = two_layer_gate_edge_l1_weighted_adjoints +
      static_cast<size_t>(atom_index) * radial_func_count * 3;
  for (int source = 0; source < source_count; source++) {
    const double *weight_col =
        two_layer_gate_edge_l1_weights_by_source.data() +
        static_cast<size_t>(source) * radial_func_count;
    double h_adj[3] = {0.0, 0.0, 0.0};
    for (int mu = 0; mu < radial_func_count; mu++) {
      const double w = weight_col[mu];
      if (w == 0.0) continue;
      const double *u_adj = u_adj_atom + 3 * mu;
      h_adj[0] += w * u_adj[0];
      h_adj[1] += w * u_adj[1];
      h_adj[2] += w * u_adj[2];
    }
    const int base = two_layer_gate_edge_l1_source_moment_indices[source];
    moment_adjoints[base + 0] += h_adj[0];
    moment_adjoints[base + 1] += h_adj[1];
    moment_adjoints[base + 2] += h_adj[2];
  }
}

void PairSUS2MTP::prepare_two_layer_gate_weight_layouts()
{
  two_layer_gate_full_weights_by_scalar.clear();
  two_layer_gate_full_scalar_cache.clear();
  two_layer_gate_full_scalar_adjoint_cache.clear();
  two_layer_gate_scalar_body_bucket.clear();
  two_layer_gate_scalar_moment_indices.clear();
  two_layer_gate_body_bucket_offsets.clear();
  two_layer_gate_body_bucket_scalar_indices.clear();
  two_layer_gate_body_signal_buckets.clear();
  two_layer_gate_body_signal_mix_weights.clear();
  two_layer_gate_full_signal_group_for_mu.clear();
  two_layer_gate_full_signal_count = 0;
  two_layer_gate_full_identity_signal = false;
  if (!two_layer_gate_enabled) return;
  if (two_layer_gate_scalar_count > 0) {
    if (static_cast<int>(two_layer_gate_scalar_indices.size()) !=
        two_layer_gate_scalar_count ||
        static_cast<int>(two_layer_gate_scalar_body_order.size()) !=
            two_layer_gate_scalar_count)
      error->all(FLERR, "SUS2-SH two-layer gate scalar metadata is inconsistent.");
    two_layer_gate_scalar_body_bucket.resize(two_layer_gate_scalar_count);
    two_layer_gate_scalar_moment_indices.resize(two_layer_gate_scalar_count);
    for (int q = 0; q < two_layer_gate_scalar_count; q++) {
      const int scalar_index = two_layer_gate_scalar_indices[q];
      if (scalar_index < 0 || scalar_index >= alpha_scalar_count)
        error->all(FLERR, "SUS2-SH two-layer gate scalar index is out of range.");
      const int moment_index = alpha_moment_mapping[scalar_index];
      if (moment_index < 0 || moment_index >= alpha_moment_count)
        error->all(FLERR, "SUS2-SH two-layer gate moment index is out of range.");
      const int body_bucket = two_layer_gate_scalar_body_order[q] - 2;
      if (body_bucket < 0 ||
          body_bucket >= std::max(0, two_layer_gate_body_order_max - 1))
        error->all(FLERR, "SUS2-SH two-layer gate scalar body bucket is out of range.");
      two_layer_gate_scalar_moment_indices[q] = moment_index;
      two_layer_gate_scalar_body_bucket[q] = body_bucket;
    }
    const int body_stride = std::max(0, two_layer_gate_body_order_max - 1);
    if (body_stride > 0) {
      two_layer_gate_body_bucket_offsets.assign(body_stride + 1, 0);
      for (int q = 0; q < two_layer_gate_scalar_count; q++)
        two_layer_gate_body_bucket_offsets[two_layer_gate_scalar_body_bucket[q] + 1]++;
      for (int b = 0; b < body_stride; b++)
        two_layer_gate_body_bucket_offsets[b + 1] +=
            two_layer_gate_body_bucket_offsets[b];
      two_layer_gate_body_bucket_scalar_indices.assign(two_layer_gate_scalar_count, 0);
      std::vector<int> bucket_cursor = two_layer_gate_body_bucket_offsets;
      for (int q = 0; q < two_layer_gate_scalar_count; q++) {
        const int bucket = two_layer_gate_scalar_body_bucket[q];
        two_layer_gate_body_bucket_scalar_indices[bucket_cursor[bucket]++] = q;
      }
    }
  }

  if (!two_layer_gate_full_scalar_weights) return;
  if (radial_func_count <= 0 || two_layer_gate_scalar_count <= 0) return;
  if (static_cast<int>(two_layer_gate_weights.size()) !=
      radial_func_count * two_layer_gate_scalar_count)
    return;

  std::vector<int> representative_mu;
  two_layer_gate_full_signal_group_for_mu.assign(radial_func_count, -1);
  representative_mu.reserve(radial_func_count);
  for (int mu = 0; mu < radial_func_count; mu++) {
    int group = -1;
    const size_t row_offset =
        static_cast<size_t>(mu) * two_layer_gate_scalar_count;
    for (int g = 0; g < static_cast<int>(representative_mu.size()); g++) {
      const size_t rep_offset =
          static_cast<size_t>(representative_mu[g]) * two_layer_gate_scalar_count;
      bool same = true;
      for (int q = 0; q < two_layer_gate_scalar_count; q++) {
        if (two_layer_gate_weights[row_offset + q] !=
            two_layer_gate_weights[rep_offset + q]) {
          same = false;
          break;
        }
      }
      if (same) {
        group = g;
        break;
      }
    }
    if (group < 0) {
      group = static_cast<int>(representative_mu.size());
      representative_mu.push_back(mu);
    }
    two_layer_gate_full_signal_group_for_mu[mu] = group;
  }
  two_layer_gate_full_signal_count =
      static_cast<int>(representative_mu.size());
  two_layer_gate_full_identity_signal =
      two_layer_gate_full_signal_count == radial_func_count;
  if (two_layer_gate_full_identity_signal) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      if (two_layer_gate_full_signal_group_for_mu[mu] != mu) {
        two_layer_gate_full_identity_signal = false;
        break;
      }
    }
  }

  two_layer_gate_full_weights_by_scalar.assign(
      static_cast<size_t>(two_layer_gate_scalar_count) *
          two_layer_gate_full_signal_count,
      0.0);
  for (int group = 0; group < two_layer_gate_full_signal_count; group++) {
    const int mu = representative_mu[group];
    const size_t src_offset =
        static_cast<size_t>(mu) * two_layer_gate_scalar_count;
    for (int q = 0; q < two_layer_gate_scalar_count; q++) {
      two_layer_gate_full_weights_by_scalar[
          static_cast<size_t>(q) * two_layer_gate_full_signal_count + group] =
          two_layer_gate_weights[src_offset + q];
    }
  }
}

void PairSUS2MTP::prepare_two_layer_gate_product_layout()
{
  two_layer_gate_product_limit = alpha_index_times_count;
  two_layer_gate_moment_cache_width = alpha_moment_count;
  two_layer_gate_product_indices.clear();
  two_layer_gate_product_a0_indices.clear();
  two_layer_gate_product_a1_indices.clear();
  two_layer_gate_product_out_indices.clear();
  two_layer_gate_product_compact_a0_indices.clear();
  two_layer_gate_product_compact_a1_indices.clear();
  two_layer_gate_product_coeffs.clear();
  two_layer_gate_required_moments.clear();
  two_layer_gate_required_moment_indices.clear();
  two_layer_gate_required_moment_compact_index.clear();
  two_layer_gate_required_basic_indices.clear();
  two_layer_gate_required_basic_mu_offsets.clear();
  two_layer_gate_required_basic_mu_indices.clear();
  two_layer_gate_required_basics_are_y00 = false;
  if (!two_layer_gate_enabled || two_layer_gate_scalar_count <= 0 ||
      alpha_moment_count <= 0)
    return;
  if (static_cast<int>(two_layer_gate_scalar_moment_indices.size()) !=
      two_layer_gate_scalar_count)
    return;
  if (two_layer_gate_body_linear_combo && alpha_index_times_count == 0) {
    const int body_stride = std::max(0, two_layer_gate_body_order_max - 1);
    if (body_stride > 0 && body_stride <= kMaxBodyGateStrideFast &&
        static_cast<int>(two_layer_gate_body_mix_weights.size()) ==
            radial_func_count * body_stride) {
      std::vector<unsigned char> active_body(body_stride, 0);
      for (int q = 0; q < two_layer_gate_scalar_count; q++) {
        const int moment_index = two_layer_gate_scalar_moment_indices[q];
        if (moment_index < 0 || moment_index >= alpha_index_basic_count)
          continue;
        const int bucket = two_layer_gate_scalar_body_bucket[q];
        if (bucket >= 0 && bucket < body_stride) active_body[bucket] = 1;
      }
      int active_count = 0;
      for (unsigned char active : active_body)
        if (active) active_count++;
      if (active_count > 0 && active_count < body_stride) {
        two_layer_gate_body_signal_buckets.clear();
        two_layer_gate_body_signal_buckets.reserve(active_count);
        for (int b = 0; b < body_stride; b++)
          if (active_body[b]) two_layer_gate_body_signal_buckets.push_back(b);
        two_layer_gate_body_signal_mix_weights.assign(
            static_cast<size_t>(radial_func_count) * active_count, 0.0);
        for (int mu = 0; mu < radial_func_count; mu++) {
          const size_t src_offset = static_cast<size_t>(mu) * body_stride;
          const size_t dst_offset = static_cast<size_t>(mu) * active_count;
          for (int s = 0; s < active_count; s++)
            two_layer_gate_body_signal_mix_weights[dst_offset + s] =
                two_layer_gate_body_mix_weights[
                    src_offset + two_layer_gate_body_signal_buckets[s]];
        }
      }
    }
  }
  if (two_layer_gate_scalar_count >= alpha_scalar_count) return;

  std::vector<unsigned char> needed(alpha_moment_count, 0);
  for (int q = 0; q < two_layer_gate_scalar_count; q++) {
    const int moment_index = two_layer_gate_scalar_moment_indices[q];
    if (moment_index < 0 || moment_index >= alpha_moment_count)
      error->all(FLERR, "SUS2-SH two-layer gate moment index is out of range.");
    needed[moment_index] = 1;
  }
  if (two_layer_gate_edge_l1_enabled) {
    for (int source = 0; source < two_layer_gate_edge_l1_source_count;
         source++) {
      const int base = two_layer_gate_edge_l1_source_moment_indices[source];
      if (base < 0 || base + 2 >= alpha_moment_count)
        error->all(FLERR, "SUS2-SH edge L1 gate source moment index is out of range.");
      needed[base + 0] = 1;
      needed[base + 1] = 1;
      needed[base + 2] = 1;
    }
  }

  int product_limit = 0;
  for (int k = alpha_index_times_count - 1; k >= 0; k--) {
    const int out = alpha_times_out[k];
    if (!needed[out]) continue;
    needed[alpha_times_a0[k]] = 1;
    needed[alpha_times_a1[k]] = 1;
    two_layer_gate_product_indices.push_back(k);
    if (product_limit == 0) product_limit = k + 1;
  }
  std::reverse(two_layer_gate_product_indices.begin(),
               two_layer_gate_product_indices.end());
  two_layer_gate_product_a0_indices.reserve(
      two_layer_gate_product_indices.size());
  two_layer_gate_product_a1_indices.reserve(
      two_layer_gate_product_indices.size());
  two_layer_gate_product_out_indices.reserve(
      two_layer_gate_product_indices.size());
  two_layer_gate_product_coeffs.reserve(two_layer_gate_product_indices.size());
  for (int k : two_layer_gate_product_indices) {
    two_layer_gate_product_a0_indices.push_back(alpha_times_a0[k]);
    two_layer_gate_product_a1_indices.push_back(alpha_times_a1[k]);
    two_layer_gate_product_out_indices.push_back(alpha_times_out[k]);
    two_layer_gate_product_coeffs.push_back(alpha_times_coeff[k]);
  }
  two_layer_gate_required_moments = needed;
  two_layer_gate_required_moment_compact_index.assign(alpha_moment_count, -1);
  for (int k = 0; k < alpha_moment_count; k++) {
    if (!needed[k]) continue;
    two_layer_gate_required_moment_compact_index[k] =
        static_cast<int>(two_layer_gate_required_moment_indices.size());
    two_layer_gate_required_moment_indices.push_back(k);
  }
  two_layer_gate_product_compact_a0_indices.reserve(
      two_layer_gate_product_indices.size());
  two_layer_gate_product_compact_a1_indices.reserve(
      two_layer_gate_product_indices.size());
  for (int idx = 0;
       idx < static_cast<int>(two_layer_gate_product_indices.size()); idx++) {
    const int a0 = two_layer_gate_product_a0_indices[idx];
    const int a1 = two_layer_gate_product_a1_indices[idx];
    two_layer_gate_product_compact_a0_indices.push_back(
        two_layer_gate_required_moment_compact_index[a0]);
    two_layer_gate_product_compact_a1_indices.push_back(
        two_layer_gate_required_moment_compact_index[a1]);
  }
  for (int k = 0; k < alpha_index_basic_count; k++)
    if (needed[k]) two_layer_gate_required_basic_indices.push_back(k);
  if (!two_layer_gate_required_basic_indices.empty()) {
    two_layer_gate_required_basics_are_y00 = true;
    const int y00_index = sh_flat_index(0, 0);
    for (int k : two_layer_gate_required_basic_indices) {
      if (alpha_basic_sh_index[k] != y00_index) {
        two_layer_gate_required_basics_are_y00 = false;
        break;
      }
    }
  }
  if (!two_layer_gate_required_basic_indices.empty() &&
      static_cast<int>(two_layer_gate_required_basic_indices.size()) <
          alpha_index_basic_count &&
      radial_func_count > 0) {
    two_layer_gate_required_basic_mu_offsets.assign(radial_func_count + 1, 0);
    for (int k : two_layer_gate_required_basic_indices) {
      const int mu = alpha_basic_mu[k];
      if (mu < 0 || mu >= radial_func_count)
        error->all(FLERR, "SUS2-SH gate-required basic mu index is out of range.");
      two_layer_gate_required_basic_mu_offsets[mu + 1]++;
    }
    for (int mu = 0; mu < radial_func_count; mu++)
      two_layer_gate_required_basic_mu_offsets[mu + 1] +=
          two_layer_gate_required_basic_mu_offsets[mu];
    two_layer_gate_required_basic_mu_indices.assign(
        two_layer_gate_required_basic_indices.size(), 0);
    std::vector<int> cursor = two_layer_gate_required_basic_mu_offsets;
    for (int k : two_layer_gate_required_basic_indices) {
      const int mu = alpha_basic_mu[k];
      two_layer_gate_required_basic_mu_indices[cursor[mu]++] = k;
    }
  }

  int cache_width = alpha_index_basic_count;
  for (int q = 0; q < two_layer_gate_scalar_count; q++)
    cache_width =
        std::max(cache_width, two_layer_gate_scalar_moment_indices[q] + 1);
  for (int k : two_layer_gate_product_indices) {
    cache_width = std::max(cache_width, alpha_times_a0[k] + 1);
    cache_width = std::max(cache_width, alpha_times_a1[k] + 1);
    cache_width = std::max(cache_width, alpha_times_out[k] + 1);
  }
  if (cache_width < 0 || cache_width > alpha_moment_count)
    error->all(FLERR, "SUS2-SH two-layer gate product cache width is invalid.");

  two_layer_gate_product_limit = product_limit;
  two_layer_gate_moment_cache_width = cache_width;
}

int PairSUS2MTP::gate_body_order_for_mu(int mu) const
{
  if (mu < 0 || mu >= radial_func_count)
    error->one(FLERR, "SUS2-SH two-layer gate mu index is out of range.");
  if (mu < static_cast<int>(two_layer_gate_mu_body_order.size()))
    return two_layer_gate_mu_body_order[mu];
  if (L_max <= 0)
    error->one(FLERR, "SUS2-SH two-layer gate has invalid L_max.");
  return mu / L_max + 2;
}

bool PairSUS2MTP::is_static_fixed_type(int zero_based_type) const
{
  return static_fixed_basic_cache_enabled &&
         zero_based_type >= 0 &&
         zero_based_type < static_cast<int>(static_fixed_type_mask.size()) &&
         static_fixed_type_mask[zero_based_type] != 0;
}

bool PairSUS2MTP::static_fixed_cache_tag_matches(const tagint *cache_tags,
                                                 int atom_index) const
{
  return cache_tags != nullptr &&
         (atom->tag == nullptr || cache_tags[atom_index] == atom->tag[atom_index]);
}

void PairSUS2MTP::configure_static_fixed_types()
{
  static_fixed_type_mask.assign(species_count, 0);
  static_fixed_basic_cache_enabled = false;
  invalidate_static_fixed_basic_cache();
  invalidate_static_fixed_gate_cache();
  invalidate_static_fixed_gate_main_cache();
  if (static_fixed_types_arg.empty()) return;

  std::stringstream ss(static_fixed_types_arg);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    const int lammps_type = utils::inumeric(FLERR, token, true, lmp);
    if (lammps_type < 1 || lammps_type > species_count)
      error->all(FLERR, "Pair sus2mtp static_fixed_types entry is outside the model species range.");
    static_fixed_type_mask[lammps_type - 1] = 1;
    static_fixed_basic_cache_enabled = true;
  }

  if (static_fixed_basic_cache_enabled)
    utils::logmesg(lmp,
                   "SUS2-SH static fixed basic cache enabled for types: {}\n",
                   static_fixed_types_arg);
}

void PairSUS2MTP::ensure_static_fixed_basic_cache()
{
  if (!static_fixed_basic_cache_enabled) return;
  const int nmax = atom->nmax;
  if (nmax <= 0 || alpha_index_basic_count <= 0) return;
  if (static_fixed_basic_cache_atom_size == nmax &&
      static_fixed_basic_cache_alpha_size == alpha_index_basic_count)
    return;

  memory->destroy(static_fixed_basic_cache);
  memory->destroy(static_fixed_basic_cache_valid);
  memory->destroy(static_fixed_basic_cache_tags);
  memory->create(static_fixed_basic_cache,
                 static_cast<size_t>(nmax) * alpha_index_basic_count,
                 "static_fixed_basic_cache");
  memory->create(static_fixed_basic_cache_valid, nmax,
                 "static_fixed_basic_cache_valid");
  memory->create(static_fixed_basic_cache_tags, nmax,
                 "static_fixed_basic_cache_tags");
  static_fixed_basic_cache_atom_size = nmax;
  static_fixed_basic_cache_alpha_size = alpha_index_basic_count;
  std::fill(static_fixed_basic_cache,
            static_fixed_basic_cache +
                static_cast<size_t>(nmax) * alpha_index_basic_count,
            0.0);
  std::fill(static_fixed_basic_cache_valid,
            static_fixed_basic_cache_valid + nmax, 0);
  std::fill(static_fixed_basic_cache_tags,
            static_fixed_basic_cache_tags + nmax, 0);
}

void PairSUS2MTP::invalidate_static_fixed_basic_cache()
{
  if (static_fixed_basic_cache_valid != nullptr &&
      static_fixed_basic_cache_atom_size > 0)
    std::fill(static_fixed_basic_cache_valid,
              static_fixed_basic_cache_valid + static_fixed_basic_cache_atom_size,
              0);
  if (static_fixed_basic_cache_tags != nullptr &&
      static_fixed_basic_cache_atom_size > 0)
    std::fill(static_fixed_basic_cache_tags,
              static_fixed_basic_cache_tags + static_fixed_basic_cache_atom_size,
              0);
}

void PairSUS2MTP::ensure_static_fixed_gate_cache()
{
  if (!static_fixed_basic_cache_enabled) return;
  const int nmax = atom->nmax;
  if (nmax <= 0 || alpha_index_basic_count <= 0) return;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  if (gate_signal_stride <= 0) return;
  if (static_fixed_gate_cache_atom_size == nmax &&
      static_fixed_gate_cache_alpha_size == alpha_index_basic_count &&
      static_fixed_gate_cache_signal_size == gate_signal_stride)
    return;

  memory->destroy(static_fixed_gate_basic_cache);
  memory->destroy(static_fixed_gate_basic_cache_valid);
  memory->destroy(static_fixed_gate_value_cache);
  memory->destroy(static_fixed_gate_value_cache_valid);
  memory->destroy(static_fixed_gate_cache_tags);
  memory->create(static_fixed_gate_basic_cache,
                 static_cast<size_t>(nmax) * alpha_index_basic_count,
                 "static_fixed_gate_basic_cache");
  memory->create(static_fixed_gate_basic_cache_valid, nmax,
                 "static_fixed_gate_basic_cache_valid");
  memory->create(static_fixed_gate_value_cache,
                 static_cast<size_t>(nmax) * gate_signal_stride,
                 "static_fixed_gate_value_cache");
  memory->create(static_fixed_gate_value_cache_valid, nmax,
                 "static_fixed_gate_value_cache_valid");
  memory->create(static_fixed_gate_cache_tags, nmax,
                 "static_fixed_gate_cache_tags");
  static_fixed_gate_cache_atom_size = nmax;
  static_fixed_gate_cache_alpha_size = alpha_index_basic_count;
  static_fixed_gate_cache_signal_size = gate_signal_stride;
  std::fill(static_fixed_gate_basic_cache,
            static_fixed_gate_basic_cache +
                static_cast<size_t>(nmax) * alpha_index_basic_count,
            0.0);
  std::fill(static_fixed_gate_basic_cache_valid,
            static_fixed_gate_basic_cache_valid + nmax, 0);
  std::fill(static_fixed_gate_value_cache,
            static_fixed_gate_value_cache +
                static_cast<size_t>(nmax) * gate_signal_stride,
            0.0);
  std::fill(static_fixed_gate_value_cache_valid,
            static_fixed_gate_value_cache_valid + nmax, 0);
  std::fill(static_fixed_gate_cache_tags,
            static_fixed_gate_cache_tags + nmax, 0);
}

void PairSUS2MTP::invalidate_static_fixed_gate_cache()
{
  if (static_fixed_gate_basic_cache_valid != nullptr &&
      static_fixed_gate_cache_atom_size > 0)
    std::fill(static_fixed_gate_basic_cache_valid,
              static_fixed_gate_basic_cache_valid + static_fixed_gate_cache_atom_size,
              0);
  if (static_fixed_gate_value_cache_valid != nullptr &&
      static_fixed_gate_cache_atom_size > 0)
    std::fill(static_fixed_gate_value_cache_valid,
              static_fixed_gate_value_cache_valid + static_fixed_gate_cache_atom_size,
              0);
  if (static_fixed_gate_cache_tags != nullptr &&
      static_fixed_gate_cache_atom_size > 0)
    std::fill(static_fixed_gate_cache_tags,
              static_fixed_gate_cache_tags + static_fixed_gate_cache_atom_size,
              0);
}

void PairSUS2MTP::ensure_static_fixed_gate_main_cache()
{
  if (!static_fixed_basic_cache_enabled) return;
  const int nmax = atom->nmax;
  if (nmax <= 0 || alpha_index_basic_count <= 0) return;
  if (static_fixed_gate_main_cache_atom_size == nmax &&
      static_fixed_gate_main_cache_alpha_size == alpha_index_basic_count)
    return;

  static_fixed_gate_main_cache_valid.assign(nmax, 0);
  static_fixed_gate_main_cache_tags.assign(nmax, 0);
  static_fixed_gate_main_cache_offsets.assign(nmax, 0);
  static_fixed_gate_main_cache_counts.assign(nmax, 0);
  static_fixed_gate_main_neighbors.clear();
  static_fixed_gate_main_neighbor_tags.clear();
  static_fixed_gate_main_base_basic.clear();
  static_fixed_gate_main_cache_atom_size = nmax;
  static_fixed_gate_main_cache_alpha_size = alpha_index_basic_count;
}

void PairSUS2MTP::invalidate_static_fixed_gate_main_cache()
{
  std::fill(static_fixed_gate_main_cache_valid.begin(),
            static_fixed_gate_main_cache_valid.end(), 0);
  std::fill(static_fixed_gate_main_cache_tags.begin(),
            static_fixed_gate_main_cache_tags.end(), 0);
  std::fill(static_fixed_gate_main_cache_offsets.begin(),
            static_fixed_gate_main_cache_offsets.end(), 0);
  std::fill(static_fixed_gate_main_cache_counts.begin(),
            static_fixed_gate_main_cache_counts.end(), 0);
  static_fixed_gate_main_neighbors.clear();
  static_fixed_gate_main_neighbor_tags.clear();
  static_fixed_gate_main_base_basic.clear();
}

bool PairSUS2MTP::static_fixed_gate_main_cache_valid_for_center(int i) const
{
  if (i < 0 || i >= static_fixed_gate_main_cache_atom_size ||
      i >= static_cast<int>(static_fixed_gate_main_cache_valid.size()))
    return false;
  if (!static_fixed_gate_main_cache_valid[i]) return false;
  if (atom->tag != nullptr &&
      static_fixed_gate_main_cache_tags[i] != atom->tag[i])
    return false;
  const size_t offset = static_fixed_gate_main_cache_offsets[i];
  const int count = static_fixed_gate_main_cache_counts[i];
  if (offset + count > static_fixed_gate_main_neighbors.size())
    return false;
  for (int e = 0; e < count; e++) {
    const int j = static_fixed_gate_main_neighbors[offset + e];
    if (j < 0 || j >= atom->nmax) return false;
    if (atom->tag != nullptr &&
        static_fixed_gate_main_neighbor_tags[offset + e] != atom->tag[j])
      return false;
  }
  return true;
}

void PairSUS2MTP::build_static_fixed_gate_basic_cache_for_center(int i,
                                                                 int itype,
                                                                 int jnum,
                                                                 int *firstneigh_i,
                                                                 const double *xi)
{
  ensure_static_fixed_gate_cache();
  if (static_fixed_gate_basic_cache == nullptr ||
      static_fixed_gate_basic_cache_valid == nullptr ||
      i >= static_fixed_gate_cache_atom_size)
    error->one(FLERR, "SUS2-SH static fixed gate cache atom index is out of range.");

  std::fill(moment_tensor_vals,
            moment_tensor_vals + alpha_moment_count, 0.0);
  for (int jj = 0; jj < jnum; jj++) {
    int j = firstneigh_i[jj];
    j &= NEIGHMASK;
    const int jtype = atom->type[j] - 1;
    if (!is_static_fixed_type(jtype)) continue;
    const double r[3] = {atom->x[j][0] - xi[0],
                         atom->x[j][1] - xi[1],
                         atom->x[j][2] - xi[2]};
    const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq <= 0.0 || rsq > max_cutoff_sq) continue;
    const double dist = std::sqrt(rsq);
    int table_index = -1;
    int table_bin = 0;
    double table_frac = 0.0;
    get_radial_table_info(itype, jtype, dist, table_index, table_bin,
                          table_frac);
    calc_pair_radial_values(itype, jtype, dist, two_layer_gate_shared_radial,
                            nullptr, false, -1, table_index, table_bin,
                            table_frac);
    accumulate_sh_basic_edge(-1, r, dist, 1.0, false, 0, false);
  }

  double *cache = static_fixed_gate_basic_cache +
      static_cast<size_t>(i) * alpha_index_basic_count;
  std::copy(moment_tensor_vals,
            moment_tensor_vals + alpha_index_basic_count, cache);
  static_fixed_gate_basic_cache_valid[i] = 1;
  static_fixed_gate_cache_tags[i] = atom->tag != nullptr ? atom->tag[i] : 0;
  if (static_fixed_gate_value_cache_valid != nullptr)
    static_fixed_gate_value_cache_valid[i] = 0;
}

void PairSUS2MTP::build_static_fixed_basic_cache_for_center(int i,
                                                            int itype,
                                                            int jnum,
                                                            int *firstneigh_i,
                                                            const double *xi)
{
  ensure_static_fixed_basic_cache();
  if (static_fixed_basic_cache == nullptr ||
      static_fixed_basic_cache_valid == nullptr ||
      i >= static_fixed_basic_cache_atom_size)
    error->one(FLERR, "SUS2-SH static fixed basic cache atom index is out of range.");

  std::fill(moment_tensor_vals,
            moment_tensor_vals + alpha_moment_count, 0.0);
  for (int jj = 0; jj < jnum; jj++) {
    int j = firstneigh_i[jj];
    j &= NEIGHMASK;
    const int jtype = atom->type[j] - 1;
    if (!is_static_fixed_type(jtype)) continue;
    const double r[3] = {atom->x[j][0] - xi[0],
                         atom->x[j][1] - xi[1],
                         atom->x[j][2] - xi[2]};
    const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq <= 0.0 || rsq > max_cutoff_sq) continue;
    const double dist = std::sqrt(rsq);
    calc_pair_radial_values(itype, jtype, dist, false);
    accumulate_sh_basic_edge(-1, r, dist, 1.0, false, 0, false);
  }

  double *cache = static_fixed_basic_cache +
      static_cast<size_t>(i) * alpha_index_basic_count;
  std::copy(moment_tensor_vals,
            moment_tensor_vals + alpha_index_basic_count, cache);
  static_fixed_basic_cache_valid[i] = 1;
  static_fixed_basic_cache_tags[i] = atom->tag != nullptr ? atom->tag[i] : 0;
}

void PairSUS2MTP::build_static_fixed_gate_main_cache_for_center(int i,
                                                                int itype,
                                                                int jnum,
                                                                int *firstneigh_i,
                                                                const double *xi)
{
  ensure_static_fixed_gate_main_cache();
  if (i < 0 || i >= static_fixed_gate_main_cache_atom_size)
    error->one(FLERR, "SUS2-SH static fixed gate main cache atom index is out of range.");

  const size_t edge_offset = static_fixed_gate_main_neighbors.size();
  int edge_count = 0;
  std::vector<double> edge_basic(alpha_index_basic_count);
  for (int jj = 0; jj < jnum; jj++) {
    int j = firstneigh_i[jj];
    j &= NEIGHMASK;
    const int jtype = atom->type[j] - 1;
    if (!is_static_fixed_type(jtype)) continue;
    const double r[3] = {atom->x[j][0] - xi[0],
                         atom->x[j][1] - xi[1],
                         atom->x[j][2] - xi[2]};
    const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rsq <= 0.0 || rsq > max_cutoff_sq) continue;
    const double dist = std::sqrt(rsq);
    int table_index = -1;
    int table_bin = 0;
    double table_frac = 0.0;
    get_radial_table_info(itype, jtype, dist, table_index, table_bin,
                          table_frac);
    calc_pair_radial_values(itype, jtype, dist, false, nullptr, false, -1,
                            table_index, table_bin, table_frac);

    double sh_values[kMaxSHComponents];
    double sh_ders[3 * kMaxSHComponents];
    eval_real_sh(r, 1.0 / dist, sh_l_max, sh_values, sh_ders);
    if (sh_basic_mu_grouped || sh_basic_mu_indexed) {
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int begin = sh_basic_mu_offsets[mu];
        const int end = sh_basic_mu_offsets[mu + 1];
        const double radial_val = radial_vals[mu];
        for (int pos = begin; pos < end; pos++) {
          const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
          edge_basic[k] = radial_val * sh_values[alpha_basic_sh_index[k]];
        }
      }
    } else {
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const int mu = alpha_basic_mu[k];
        const int sh_idx = alpha_basic_sh_index[k];
        edge_basic[k] = radial_vals[mu] * sh_values[sh_idx];
      }
    }

    static_fixed_gate_main_neighbors.push_back(j);
    static_fixed_gate_main_neighbor_tags.push_back(
        atom->tag != nullptr ? atom->tag[j] : 0);
    static_fixed_gate_main_base_basic.insert(
        static_fixed_gate_main_base_basic.end(),
        edge_basic.begin(), edge_basic.end());
    edge_count++;
  }

  static_fixed_gate_main_cache_offsets[i] = edge_offset;
  static_fixed_gate_main_cache_counts[i] = edge_count;
  static_fixed_gate_main_cache_tags[i] = atom->tag != nullptr ? atom->tag[i] : 0;
  static_fixed_gate_main_cache_valid[i] = 1;
}

void PairSUS2MTP::ensure_two_layer_gate_mu_cache_for_atom(int jtype,
                                                          int atom_index,
                                                          const double *gate_signal_by_mu)
{
  if (disable_two_layer_gate_tanh_cache() ||
      two_layer_gate_mu_cache_valid == nullptr ||
      two_layer_gate_multiplier_mu_cache == nullptr ||
      two_layer_gate_deriv_mu_cache == nullptr)
    error->one(FLERR, "SUS2-SH static fixed gate main cache requires the gate mu cache.");
  if (atom_index < 0 || atom_index >= atom->nmax)
    error->one(FLERR, "SUS2-SH gate mu cache atom index is out of range.");
  if (two_layer_gate_mu_cache_valid[atom_index]) return;
  if (jtype < 0 ||
      jtype >= static_cast<int>(two_layer_gate_additive_ratio_valid.size()) ||
      !two_layer_gate_additive_ratio_valid[jtype])
    error->one(FLERR, "SUS2-SH static fixed gate main cache requires additive gate ratios.");

  const double additive_coeff = two_layer_gate_additive_coeffs[jtype];
  double *__restrict mult = two_layer_gate_multiplier_mu_cache +
      static_cast<size_t>(atom_index) * radial_func_count;
  double *__restrict deriv = two_layer_gate_deriv_mu_cache +
      static_cast<size_t>(atom_index) * radial_func_count;
  #pragma omp simd
  for (int mu = 0; mu < radial_func_count; mu++) {
    const double gate_signal = two_layer_gate_mu_signal(gate_signal_by_mu, mu);
    const double arg = additive_coeff * gate_signal;
    const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
    const double sech2 = 1.0 - tanh_arg * tanh_arg;
    mult[mu] = 1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
    deriv[mu] = two_layer_gate_tanh_amplitude * additive_coeff * sech2;
  }
  two_layer_gate_mu_cache_valid[atom_index] = 1;
}

bool PairSUS2MTP::apply_static_fixed_gate_main_cache_moments(int i)
{
  if (!static_fixed_gate_main_cache_valid_for_center(i)) return false;
  const size_t edge_offset = static_fixed_gate_main_cache_offsets[i];
  const int edge_count = static_fixed_gate_main_cache_counts[i];
  for (int e = 0; e < edge_count; e++) {
    const size_t edge_index = edge_offset + e;
    const int j = static_fixed_gate_main_neighbors[edge_index];
    const int jtype = atom->type[j] - 1;
    const double *gate_signal_by_mu =
        two_layer_gate_values +
        static_cast<size_t>(j) * two_layer_gate_signal_stride();
    ensure_two_layer_gate_mu_cache_for_atom(jtype, j, gate_signal_by_mu);
    const double *__restrict mult = two_layer_gate_multiplier_mu_cache +
        static_cast<size_t>(j) * radial_func_count;
    const double *__restrict base =
        static_fixed_gate_main_base_basic.data() +
        edge_index * alpha_index_basic_count;
    if (sh_basic_mu_grouped || sh_basic_mu_indexed) {
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int begin = sh_basic_mu_offsets[mu];
        const int end = sh_basic_mu_offsets[mu + 1];
        const double multiplier = mult[mu];
        for (int pos = begin; pos < end; pos++) {
          const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
          moment_tensor_vals[k] += base[k] * multiplier;
        }
      }
    } else {
      #pragma omp simd
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const int mu = alpha_basic_mu[k];
        moment_tensor_vals[k] += base[k] * mult[mu];
      }
    }
  }
  return true;
}

void PairSUS2MTP::apply_static_fixed_gate_main_cache_adjoints(int i)
{
  const size_t edge_offset = static_fixed_gate_main_cache_offsets[i];
  const int edge_count = static_fixed_gate_main_cache_counts[i];
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const int gate_body_stride = two_layer_gate_body_order_max - 1;
  const double *body_signal_mix_weights =
      !two_layer_gate_body_signal_mix_weights.empty()
      ? two_layer_gate_body_signal_mix_weights.data()
      : two_layer_gate_body_mix_weights.data();
  const bool sh_basic_mu_fast = sh_basic_mu_grouped || sh_basic_mu_indexed;
  const bool direct_body_combo_gate_adjoints =
      two_layer_gate_body_linear_combo && gate_body_stride > 0 &&
      gate_signal_stride > 0 && gate_signal_stride <= kMaxBodyGateStrideFast &&
      !two_layer_gate_center_enabled && sh_basic_mu_fast;
  const bool direct_full_gate_adjoints =
      !two_layer_gate_body_linear_combo &&
      !two_layer_gate_center_enabled && sh_basic_mu_fast &&
      gate_signal_stride == radial_func_count;
  const bool direct_grouped_full_gate_adjoints =
      !two_layer_gate_body_linear_combo &&
      !two_layer_gate_center_enabled && sh_basic_mu_fast &&
      gate_signal_stride > 0 && gate_signal_stride < radial_func_count &&
      static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
          radial_func_count;
  for (int e = 0; e < edge_count; e++) {
    const size_t edge_index = edge_offset + e;
    const int j = static_fixed_gate_main_neighbors[edge_index];
    const double *__restrict deriv = two_layer_gate_deriv_mu_cache +
        static_cast<size_t>(j) * radial_func_count;
    const double *__restrict base =
        static_fixed_gate_main_base_basic.data() +
        edge_index * alpha_index_basic_count;
    double *adj =
        two_layer_gate_adjoints + static_cast<size_t>(j) * gate_signal_stride;
    const bool direct_gate_adjoint_accumulation =
        direct_body_combo_gate_adjoints || direct_full_gate_adjoints ||
        direct_grouped_full_gate_adjoints;
    double *__restrict gate_adjoint_by_mu =
        direct_gate_adjoint_accumulation ? nullptr : two_layer_gate_adjoint_scratch;
    if (!direct_gate_adjoint_accumulation)
      std::fill(gate_adjoint_by_mu,
                gate_adjoint_by_mu + radial_func_count, 0.0);
    double gate_body_adjoints[kMaxBodyGateStrideFast] = {};
    if (sh_basic_mu_fast) {
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int begin = sh_basic_mu_offsets[mu];
        const int end = sh_basic_mu_offsets[mu + 1];
        const double deriv_mu = deriv[mu];
        double gate_adjoint_mu = 0.0;
        for (int pos = begin; pos < end; pos++) {
          const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
          gate_adjoint_mu += weighted_basic_moment_ders[k] * base[k] * deriv_mu;
        }
        if (direct_body_combo_gate_adjoints) {
          if (gate_adjoint_mu != 0.0) {
            const double *weights =
                body_signal_mix_weights +
                static_cast<size_t>(mu) * gate_signal_stride;
            accumulate_small_stride(gate_adjoint_mu, weights,
                                    gate_signal_stride, gate_body_adjoints);
          }
        } else if (direct_full_gate_adjoints) {
          adj[mu] += gate_adjoint_mu;
        } else if (direct_grouped_full_gate_adjoints) {
          adj[two_layer_gate_full_signal_group_for_mu[mu]] += gate_adjoint_mu;
        } else {
          gate_adjoint_by_mu[mu] += gate_adjoint_mu;
        }
      }
    } else {
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const int mu = alpha_basic_mu[k];
        gate_adjoint_by_mu[mu] += weighted_basic_moment_ders[k] * base[k] * deriv[mu];
      }
    }
    if (direct_body_combo_gate_adjoints) {
      for (int b = 0; b < gate_signal_stride; b++)
        adj[b] += gate_body_adjoints[b];
    } else if (!direct_full_gate_adjoints &&
               !direct_grouped_full_gate_adjoints) {
      accumulate_two_layer_gate_signal_adjoints(adj, gate_adjoint_by_mu);
    }
  }
}

int PairSUS2MTP::two_layer_gate_additive_coeff_index(int type_outer, int mu) const
{
  if (mu < 0 || mu >= radial_func_count)
    error->one(FLERR, "SUS2-SH two-layer gate additive mu index is out of range.");
  return type_outer;
}

double PairSUS2MTP::two_layer_gate_additive_coeff(int type_outer, int mu) const
{
  const int idx = two_layer_gate_additive_coeff_index(type_outer, mu);
  if (idx < 0 || idx >= static_cast<int>(two_layer_gate_additive_coeffs.size()))
    error->one(FLERR, "SUS2-SH two-layer gate additive coefficient index is out of range.");
  return two_layer_gate_additive_coeffs[idx];
}

void PairSUS2MTP::prepare_two_layer_gate_additive_ratios()
{
  two_layer_gate_additive_ratios.clear();
  two_layer_gate_additive_ratio_valid.clear();
  if (!two_layer_gate_enabled) return;
  if (static_cast<int>(two_layer_gate_additive_coeffs.size()) !=
      species_count)
    return;

  two_layer_gate_additive_ratios.assign(
      static_cast<size_t>(species_count) * radial_func_count, 0.0);
  two_layer_gate_additive_ratio_valid.assign(species_count, 0);
  for (int jtype = 0; jtype < species_count; jtype++) {
    two_layer_gate_additive_ratio_valid[jtype] = 1;
    const size_t ratio_offset = static_cast<size_t>(jtype) * radial_func_count;
    for (int mu = 0; mu < radial_func_count; mu++)
      two_layer_gate_additive_ratios[ratio_offset + mu] =
          two_layer_gate_additive_coeffs[jtype];
  }
}

void PairSUS2MTP::ensure_two_layer_atom_buffers()
{
  const int nmax = atom->nmax;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  if (gate_signal_stride <= 0)
    error->one(FLERR, "SUS2-SH two-layer gate signal stride is invalid.");
  if (two_layer_atom_buffer_size < nmax) {
    memory->grow(two_layer_gate_values,
                 static_cast<size_t>(nmax) * gate_signal_stride,
                 "two_layer_gate_values");
    memory->grow(two_layer_gate_adjoints,
                 static_cast<size_t>(nmax) * gate_signal_stride,
                 "two_layer_gate_adjoints");
    two_layer_atom_buffer_size = nmax;
  }
  if (two_layer_gate_edge_l1_active()) {
    const int weighted_stride = two_layer_gate_edge_l1_weighted_stride();
    if (weighted_stride <= 0)
      error->one(FLERR, "SUS2-SH edge L1 gate weighted stride is invalid.");
    if (two_layer_gate_edge_l1_atom_buffer_size < nmax) {
      memory->grow(two_layer_gate_edge_l1_weighted_values,
                   static_cast<size_t>(nmax) * weighted_stride,
                   "two_layer_gate_edge_l1_weighted_values");
      memory->grow(two_layer_gate_edge_l1_weighted_adjoints,
                   static_cast<size_t>(nmax) * weighted_stride,
                   "two_layer_gate_edge_l1_weighted_adjoints");
      two_layer_gate_edge_l1_atom_buffer_size = nmax;
    }
  }
  if (!disable_two_layer_gate_tanh_cache() && radial_func_count > 0) {
    const int cache_size = nmax * radial_func_count;
    if (two_layer_gate_mu_cache_atom_size < nmax) {
      memory->grow(two_layer_gate_mu_cache_valid, nmax,
                   "two_layer_gate_mu_cache_valid");
      std::fill(two_layer_gate_mu_cache_valid,
                two_layer_gate_mu_cache_valid + nmax, 0);
      two_layer_gate_mu_cache_atom_size = nmax;
    }
    if (two_layer_gate_mu_cache_size < cache_size) {
      memory->grow(two_layer_gate_multiplier_mu_cache, cache_size,
                   "two_layer_gate_multiplier_mu_cache");
      memory->grow(two_layer_gate_deriv_mu_cache, cache_size,
                   "two_layer_gate_deriv_mu_cache");
      two_layer_gate_mu_cache_size = cache_size;
    }
  }
}

void PairSUS2MTP::ensure_two_layer_edge_buffer(int jnum)
{
  if (jac_size < jnum) {
    memory->grow(moment_jacobian_x, jnum * alpha_index_basic_count,
                 "moment_jacobian_x");
    memory->grow(moment_jacobian_y, jnum * alpha_index_basic_count,
                 "moment_jacobian_y");
    memory->grow(moment_jacobian_z, jnum * alpha_index_basic_count,
                 "moment_jacobian_z");
    memory->grow(within_cutoff, jnum, "within_cutoff");
    memory->grow(env_rho_dr, jnum, "env_rho_dr");
    jac_size = jnum;
  }
  if (two_layer_raw_jac_size < jnum) {
	    memory->grow(two_layer_raw_basic_vals, jnum * alpha_index_basic_count,
	                 "two_layer_raw_basic_vals");
	    memory->grow(two_layer_center_raw_basic_vals,
	                 jnum * alpha_index_basic_count,
	                 "two_layer_center_raw_basic_vals");
	    two_layer_raw_jac_size = jnum;
	  }
}

bool PairSUS2MTP::get_radial_table_info(int itype,
                                         int jtype,
                                         double dist,
                                         int &table_index,
                                         int &r_list,
                                         double &ddr) const
{
  table_index = -1;
  r_list = 0;
  ddr = 0.0;
  if (!do_list) return false;
  const int shift = species_count * itype + jtype;
  table_index = pair_to_table_index[shift];
  if (table_index < 0) return false;

  const double scaled_dist = dist * inv_dr;
  r_list = static_cast<int>(scaled_dist);
  const int last_interval = list_grid_size - 2;
  if (r_list < 0) r_list = 0;
  if (r_list > last_interval) r_list = last_interval;
  ddr = scaled_dist - r_list;
  if (ddr < 0.0) ddr = 0.0;
  if (ddr > 1.0) ddr = 1.0;
  return true;
}

bool PairSUS2MTP::calc_gate_additive_table_radial_values(int jtype,
                                                         const double *gate_signal_by_mu,
                                                         int gate_atom_index,
                                                         int table_index,
                                                         int r_list,
                                                         double ddr)
{
  if (!do_list || table_index < 0) return false;
  if (jtype < 0 ||
      jtype >= static_cast<int>(two_layer_gate_additive_ratio_valid.size()) ||
      !two_layer_gate_additive_ratio_valid[jtype])
    return false;
  if (static_cast<int>(two_layer_gate_additive_coeffs.size()) != species_count)
    error->one(FLERR, "SUS2-SH two-layer gate additive coefficient storage is inconsistent.");
  if (static_cast<int>(two_layer_gate_additive_ratios.size()) !=
      species_count * radial_func_count)
    error->one(FLERR, "SUS2-SH two-layer gate additive ratio storage is inconsistent.");

  const bool use_gate_mu_cache =
      gate_atom_index >= 0 &&
      !disable_two_layer_gate_tanh_cache() &&
      two_layer_gate_multiplier_mu_cache != nullptr &&
      two_layer_gate_deriv_mu_cache != nullptr &&
      two_layer_gate_mu_cache_valid != nullptr;
  double *__restrict gate_multiplier_cache = nullptr;
  double *__restrict gate_deriv_cache = nullptr;
  bool fill_gate_mu_cache = false;
  if (use_gate_mu_cache) {
    if (gate_atom_index >= atom->nmax)
      error->one(FLERR, "SUS2-SH LAMMPS two-layer gate mu cache atom index is out of range.");
    gate_multiplier_cache = two_layer_gate_multiplier_mu_cache +
        static_cast<size_t>(gate_atom_index) * radial_func_count;
    gate_deriv_cache = two_layer_gate_deriv_mu_cache +
        static_cast<size_t>(gate_atom_index) * radial_func_count;
    fill_gate_mu_cache = two_layer_gate_mu_cache_valid[gate_atom_index] == 0;
  }

  const int r_next = r_list + 1;
  const double *__restrict base_row = radial_list[table_index][r_list];
  const double *__restrict base_next_row = radial_list[table_index][r_next];
  const double *__restrict base_der_row = radial_der_list[table_index][r_list];
  const double *__restrict base_der_next_row = radial_der_list[table_index][r_next];
  const double additive_coeff = two_layer_gate_additive_coeffs[jtype];
  const bool full_identity_gate_signal =
      !two_layer_gate_body_linear_combo && two_layer_gate_full_identity_signal;
  const int body_gate_signal_stride =
      two_layer_gate_body_linear_combo ? two_layer_gate_signal_stride() : 0;
  const bool body_small_gate_signal =
      body_gate_signal_stride > 0 &&
      body_gate_signal_stride <= kMaxBodyGateStrideFast;
  const double *__restrict body_mix_weights =
      body_small_gate_signal
      ? (!two_layer_gate_body_signal_mix_weights.empty()
         ? two_layer_gate_body_signal_mix_weights.data()
         : two_layer_gate_body_mix_weights.data())
      : nullptr;
  const int full_group_signal_count =
      (!two_layer_gate_body_linear_combo && !full_identity_gate_signal)
      ? two_layer_gate_signal_stride()
      : 0;
  const bool full_grouped_gate_signal =
      full_group_signal_count > 0 &&
      full_group_signal_count < radial_func_count &&
      full_group_signal_count <= kMaxGateSignalGroupFast &&
      static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
          radial_func_count;

  if (full_grouped_gate_signal &&
      (gate_multiplier_cache == nullptr || fill_gate_mu_cache)) {
    if (fill_gate_mu_cache) {
      fill_full_grouped_gate_table_radials_fast<true>(
          radial_func_count, full_group_signal_count,
          two_layer_gate_full_signal_group_for_mu.data(), base_row,
          base_next_row, base_der_row, base_der_next_row, ddr,
          additive_coeff, two_layer_gate_tanh_amplitude, gate_signal_by_mu,
          two_layer_gate_residual_radial_vals, radial_vals, radial_ders,
          gate_multiplier_cache, gate_deriv_cache);
      two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
    } else {
      fill_full_grouped_gate_table_radials_fast<false>(
          radial_func_count, full_group_signal_count,
          two_layer_gate_full_signal_group_for_mu.data(), base_row,
          base_next_row, base_der_row, base_der_next_row, ddr,
          additive_coeff, two_layer_gate_tanh_amplitude, gate_signal_by_mu,
          two_layer_gate_residual_radial_vals, radial_vals, radial_ders,
          gate_multiplier_cache, gate_deriv_cache);
    }
    return true;
  }

  if (body_small_gate_signal &&
      (gate_multiplier_cache == nullptr || fill_gate_mu_cache) &&
      fill_body_combo_gate_table_radials_specialized(
          body_gate_signal_stride, fill_gate_mu_cache, radial_func_count,
          base_row, base_next_row, base_der_row, base_der_next_row, ddr,
          additive_coeff, two_layer_gate_tanh_amplitude, body_mix_weights,
          gate_signal_by_mu, two_layer_gate_residual_radial_vals,
          radial_vals, radial_ders, gate_multiplier_cache, gate_deriv_cache)) {
    if (fill_gate_mu_cache)
      two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
    return true;
  }

  if (gate_multiplier_cache == nullptr) {
    #pragma omp simd
    for (int mu = 0; mu < radial_func_count; mu++) {
      const double base_val =
          base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
      const double base_der =
          base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
      const double gate_signal = full_identity_gate_signal
          ? gate_signal_by_mu[mu]
          : (body_small_gate_signal
             ? dot_small_stride(
                   body_mix_weights +
                       static_cast<size_t>(mu) * body_gate_signal_stride,
                   gate_signal_by_mu, body_gate_signal_stride)
             : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
      const double arg = additive_coeff * gate_signal;
      const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
      const double sech2 = 1.0 - tanh_arg * tanh_arg;
      const double gate_multiplier =
          1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
      const double gate_deriv =
          two_layer_gate_tanh_amplitude * additive_coeff * sech2;
      two_layer_gate_residual_radial_vals[mu] = base_val * gate_deriv;
      radial_vals[mu] = base_val * gate_multiplier;
      radial_ders[mu] = base_der * gate_multiplier;
    }
  } else if (fill_gate_mu_cache) {
    #pragma omp simd
    for (int mu = 0; mu < radial_func_count; mu++) {
      const double base_val =
          base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
      const double base_der =
          base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
      const double gate_signal = full_identity_gate_signal
          ? gate_signal_by_mu[mu]
          : (body_small_gate_signal
             ? dot_small_stride(
                   body_mix_weights +
                       static_cast<size_t>(mu) * body_gate_signal_stride,
                   gate_signal_by_mu, body_gate_signal_stride)
             : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
      const double arg = additive_coeff * gate_signal;
      const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
      const double sech2 = 1.0 - tanh_arg * tanh_arg;
      const double gate_multiplier =
          1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
      const double gate_deriv =
          two_layer_gate_tanh_amplitude * additive_coeff * sech2;
      gate_multiplier_cache[mu] = gate_multiplier;
      gate_deriv_cache[mu] = gate_deriv;
      two_layer_gate_residual_radial_vals[mu] = base_val * gate_deriv;
      radial_vals[mu] = base_val * gate_multiplier;
      radial_ders[mu] = base_der * gate_multiplier;
    }
    two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
  } else {
    #pragma omp simd
    for (int mu = 0; mu < radial_func_count; mu++) {
      const double base_val =
          base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
      const double base_der =
          base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
      const double gate_multiplier = gate_multiplier_cache[mu];
      const double gate_deriv = gate_deriv_cache[mu];
      two_layer_gate_residual_radial_vals[mu] = base_val * gate_deriv;
      radial_vals[mu] = base_val * gate_multiplier;
      radial_ders[mu] = base_der * gate_multiplier;
    }
  }

  return true;
}

void PairSUS2MTP::calc_pair_radial_values(int itype,
                                          int jtype,
                                          double dist,
                                          bool use_gate_radial,
                                          const double *gate_signal_by_mu,
                                          bool use_gate_additive,
                                          int gate_atom_index,
                                          int cached_table_index,
                                          int cached_table_bin,
                                          double cached_table_frac,
                                          bool gate_signal_is_full_mu)
{
  const bool use_gate_mu_cache =
      use_gate_additive &&
      gate_atom_index >= 0 &&
      !disable_two_layer_gate_tanh_cache() &&
      two_layer_gate_multiplier_mu_cache != nullptr &&
      two_layer_gate_deriv_mu_cache != nullptr &&
      two_layer_gate_mu_cache_valid != nullptr;
  double *gate_multiplier_cache = nullptr;
  double *gate_deriv_cache = nullptr;
  bool fill_gate_mu_cache = false;
  if (use_gate_mu_cache) {
    if (gate_atom_index >= atom->nmax)
      error->one(FLERR, "SUS2-SH LAMMPS two-layer gate mu cache atom index is out of range.");
    gate_multiplier_cache = two_layer_gate_multiplier_mu_cache +
        static_cast<size_t>(gate_atom_index) * radial_func_count;
    gate_deriv_cache = two_layer_gate_deriv_mu_cache +
        static_cast<size_t>(gate_atom_index) * radial_func_count;
    fill_gate_mu_cache = two_layer_gate_mu_cache_valid[gate_atom_index] == 0;
  }

  bool used_precomputed_table = false;
  if (do_list) {
    int table_index = cached_table_index;
    int r_list = cached_table_bin;
    double ddr = cached_table_frac;
    if (table_index == -2)
      get_radial_table_info(itype, jtype, dist, table_index, r_list, ddr);
    if (table_index >= 0) {
      if (use_gate_additive) {
        if (jtype >= 0 &&
            jtype < static_cast<int>(two_layer_gate_additive_ratio_valid.size()) &&
            two_layer_gate_additive_ratio_valid[jtype]) {
          if (static_cast<int>(two_layer_gate_additive_coeffs.size()) != species_count)
            error->one(FLERR, "SUS2-SH two-layer gate additive coefficient storage is inconsistent.");
          if (static_cast<int>(two_layer_gate_additive_ratios.size()) !=
              species_count * radial_func_count)
            error->one(FLERR, "SUS2-SH two-layer gate additive ratio storage is inconsistent.");
          const int r_next = r_list + 1;
          double *base_row = radial_list[table_index][r_list];
          double *base_next_row = radial_list[table_index][r_next];
          double *base_der_row = radial_der_list[table_index][r_list];
          double *base_der_next_row = radial_der_list[table_index][r_next];
          const double additive_coeff = two_layer_gate_additive_coeffs[jtype];
          const bool table_full_identity_gate_signal =
              gate_signal_is_full_mu ||
              (!two_layer_gate_body_linear_combo && two_layer_gate_full_identity_signal);
          const int table_body_gate_signal_stride =
              (!gate_signal_is_full_mu && two_layer_gate_body_linear_combo)
              ? two_layer_gate_signal_stride() : 0;
          const bool table_body_small_gate_signal =
              table_body_gate_signal_stride > 0 &&
              table_body_gate_signal_stride <= kMaxBodyGateStrideFast;
          const double *__restrict table_body_mix_weights =
              table_body_small_gate_signal
              ? (!two_layer_gate_body_signal_mix_weights.empty()
                 ? two_layer_gate_body_signal_mix_weights.data()
                 : two_layer_gate_body_mix_weights.data())
              : nullptr;
          const int table_full_group_signal_count =
              (!two_layer_gate_body_linear_combo &&
               !table_full_identity_gate_signal)
              ? two_layer_gate_signal_stride()
              : 0;
          const bool table_full_grouped_gate_signal =
              table_full_group_signal_count > 0 &&
              table_full_group_signal_count < radial_func_count &&
              table_full_group_signal_count <= kMaxGateSignalGroupFast &&
              static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
                  radial_func_count;
          if (table_full_grouped_gate_signal &&
              (gate_multiplier_cache == nullptr || fill_gate_mu_cache)) {
            if (fill_gate_mu_cache) {
              fill_full_grouped_gate_table_radials_fast<true>(
                  radial_func_count, table_full_group_signal_count,
                  two_layer_gate_full_signal_group_for_mu.data(), base_row,
                  base_next_row, base_der_row, base_der_next_row, ddr,
                  additive_coeff, two_layer_gate_tanh_amplitude,
                  gate_signal_by_mu, two_layer_gate_residual_radial_vals,
                  radial_vals, radial_ders, gate_multiplier_cache,
                  gate_deriv_cache);
              two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
            } else {
              fill_full_grouped_gate_table_radials_fast<false>(
                  radial_func_count, table_full_group_signal_count,
                  two_layer_gate_full_signal_group_for_mu.data(), base_row,
                  base_next_row, base_der_row, base_der_next_row, ddr,
                  additive_coeff, two_layer_gate_tanh_amplitude,
                  gate_signal_by_mu, two_layer_gate_residual_radial_vals,
                  radial_vals, radial_ders, gate_multiplier_cache,
                  gate_deriv_cache);
            }
            used_precomputed_table = true;
          } else if (table_body_small_gate_signal &&
                     (gate_multiplier_cache == nullptr ||
                      fill_gate_mu_cache) &&
                     fill_body_combo_gate_table_radials_specialized(
                         table_body_gate_signal_stride, fill_gate_mu_cache,
                         radial_func_count, base_row, base_next_row,
                         base_der_row, base_der_next_row, ddr,
                         additive_coeff, two_layer_gate_tanh_amplitude,
                         table_body_mix_weights, gate_signal_by_mu,
                         two_layer_gate_residual_radial_vals, radial_vals,
	                         radial_ders, gate_multiplier_cache,
	                         gate_deriv_cache)) {
	            if (fill_gate_mu_cache)
	              two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
	            used_precomputed_table = true;
	          } else if (table_full_identity_gate_signal &&
	                     gate_multiplier_cache == nullptr) {
	            #pragma omp simd
	            for (int mu = 0; mu < radial_func_count; mu++) {
	              const double base_val =
	                  base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
	              const double base_der =
	                  base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
	              const double arg = additive_coeff * gate_signal_by_mu[mu];
	              const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
	              const double sech2 = 1.0 - tanh_arg * tanh_arg;
	              const double gate_multiplier =
	                  1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
	              const double gate_deriv =
	                  two_layer_gate_tanh_amplitude * additive_coeff * sech2;
	              two_layer_gate_residual_radial_vals[mu] = base_val * gate_deriv;
	              radial_vals[mu] = base_val * gate_multiplier;
	              radial_ders[mu] = base_der * gate_multiplier;
	            }
	            used_precomputed_table = true;
	          } else {
		          for (int mu = 0; mu < radial_func_count; mu++) {
	            const double base_val =
	                base_row[mu] + ddr * (base_next_row[mu] - base_row[mu]);
            const double base_der =
                base_der_row[mu] + ddr * (base_der_next_row[mu] - base_der_row[mu]);
            double gate_multiplier;
            double gate_deriv;
            if (gate_multiplier_cache == nullptr) {
              const double gate_signal = table_full_identity_gate_signal
                  ? gate_signal_by_mu[mu]
                  : (table_body_small_gate_signal
                     ? dot_small_stride(
                           table_body_mix_weights +
                               static_cast<size_t>(mu) *
                                   table_body_gate_signal_stride,
                           gate_signal_by_mu, table_body_gate_signal_stride)
                     : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
              const double arg = additive_coeff * gate_signal;
              const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
              const double sech2 = 1.0 - tanh_arg * tanh_arg;
              gate_multiplier = 1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
              gate_deriv = two_layer_gate_tanh_amplitude * additive_coeff * sech2;
            } else if (fill_gate_mu_cache) {
              const double gate_signal = table_full_identity_gate_signal
                  ? gate_signal_by_mu[mu]
                  : (table_body_small_gate_signal
                     ? dot_small_stride(
                           table_body_mix_weights +
                               static_cast<size_t>(mu) *
                                   table_body_gate_signal_stride,
                           gate_signal_by_mu, table_body_gate_signal_stride)
                     : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
              const double arg = additive_coeff * gate_signal;
              const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
              const double sech2 = 1.0 - tanh_arg * tanh_arg;
              gate_multiplier =
                  1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
              gate_deriv =
                  two_layer_gate_tanh_amplitude * additive_coeff * sech2;
              gate_multiplier_cache[mu] = gate_multiplier;
              gate_deriv_cache[mu] = gate_deriv;
            } else {
              gate_multiplier = gate_multiplier_cache[mu];
              gate_deriv = gate_deriv_cache[mu];
            }
	            two_layer_gate_residual_radial_vals[mu] = base_val * gate_deriv;
	            radial_vals[mu] = base_val * gate_multiplier;
	            radial_ders[mu] = base_der * gate_multiplier;
	          }
	          if (fill_gate_mu_cache)
	            two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
	          used_precomputed_table = true;
	          }
	        }
        // two_layer_gate_additive_table_fallback: if the exact main-table
        // ratio is unavailable, leave the table path and use the analytic
        // additive calculation below.
      } else if (!use_gate_additive) {
        double ***value_table =
            (use_gate_radial && two_layer_gate_radial_list) ? two_layer_gate_radial_list : radial_list;
        double ***der_table =
            (use_gate_radial && two_layer_gate_radial_der_list) ? two_layer_gate_radial_der_list : radial_der_list;
        interpolate_table_at_bin(value_table, der_table, table_index, r_list,
                                 ddr, radial_func_count, radial_vals,
                                 radial_ders);
        used_precomputed_table = true;
      }
    }
  }
  if (used_precomputed_table) return;

  const int C = species_count;
  const int R = radial_basis_size;
  const int pairs_count = C * C;
  const bool per_mu_sigma = basis_requires_per_mu_sigma(radial_basis_type_index);
  const int radial_cache_count = per_mu_sigma ? radial_func_count : K_scaling;
  const int temp_size = radial_cache_count * R;
  if (two_layer_radial_cache_size < temp_size) {
    memory->grow(two_layer_radial_cache_vals, temp_size, "two_layer_radial_cache_vals");
    memory->grow(two_layer_radial_cache_ders, temp_size, "two_layer_radial_cache_ders");
    two_layer_radial_cache_size = temp_size;
  }

  if (per_mu_sigma) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int k_ = mu_to_K[mu];
      const int sigma = mu_to_sigma[mu];
      const double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
      const double shift = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];
      radial_basis->calc_radial_basis_ders(dist, scal, shift, sigma);
      for (int ri = 0; ri < R; ri++) {
        two_layer_radial_cache_vals[mu * R + ri] = radial_basis->radial_basis_vals[ri];
        two_layer_radial_cache_ders[mu * R + ri] = radial_basis->radial_basis_ders[ri];
      }
    }
  } else {
    for (int k_ = 0; k_ < K_scaling; k_++) {
      const int sigma = mu_to_sigma[k_];
      const double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
      const double shift = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];
      radial_basis->calc_radial_basis_ders(dist, scal, shift, sigma);
      for (int ri = 0; ri < R; ri++) {
        two_layer_radial_cache_vals[k_ * R + ri] = radial_basis->radial_basis_vals[ri];
        two_layer_radial_cache_ders[k_ * R + ri] = radial_basis->radial_basis_ders[ri];
      }
    }
  }

  const double center_type_coeff = regression_coeffs[radial_coeffs_offset + R + itype];
  const double outer_type_coeff = regression_coeffs[radial_coeffs_offset + R + jtype];
  const double species_factor = center_type_coeff * outer_type_coeff;
  if (use_gate_radial && two_layer_gate_shared_radial &&
      static_cast<int>(two_layer_gate_radial_coeffs.size()) < radial_func_count * R)
    error->one(FLERR, "SUS2-SH two-layer gate radial coefficient storage is inconsistent.");
  if (use_gate_additive &&
      static_cast<int>(two_layer_gate_additive_coeffs.size()) != species_count)
    error->one(FLERR, "SUS2-SH two-layer gate additive coefficient storage is inconsistent.");
  const double gate_additive_coeff =
      use_gate_additive ? two_layer_gate_additive_coeffs[jtype] : 0.0;
  const bool full_identity_gate_signal =
      use_gate_additive &&
      (gate_signal_is_full_mu ||
       (!two_layer_gate_body_linear_combo && two_layer_gate_full_identity_signal));
  const int body_gate_signal_stride =
      (use_gate_additive && !gate_signal_is_full_mu &&
       two_layer_gate_body_linear_combo)
      ? two_layer_gate_signal_stride()
      : 0;
  const bool body_small_gate_signal =
      body_gate_signal_stride > 0 &&
      body_gate_signal_stride <= kMaxBodyGateStrideFast;
  const double *__restrict body_mix_weights =
      body_small_gate_signal
      ? (!two_layer_gate_body_signal_mix_weights.empty()
         ? two_layer_gate_body_signal_mix_weights.data()
         : two_layer_gate_body_mix_weights.data())
      : nullptr;
  for (int mu = 0; mu < radial_func_count; mu++) {
    const int k_ = mu_to_K[mu];
    const int radial_cache_index = per_mu_sigma ? mu : k_;
    double val = 0.0;
    double der = 0.0;
    if (use_gate_additive)
      two_layer_gate_residual_radial_vals[mu] = 0.0;
    double final_type_factor = species_factor;
    double residual_type_factor = 0.0;
    if (use_gate_additive) {
      double gate_multiplier;
      double gate_deriv;
      if (gate_multiplier_cache == nullptr) {
        const double additive_coeff = gate_additive_coeff;
        const double gate_signal = full_identity_gate_signal
            ? gate_signal_by_mu[mu]
            : (body_small_gate_signal
               ? dot_small_stride(
                     body_mix_weights +
                         static_cast<size_t>(mu) * body_gate_signal_stride,
                     gate_signal_by_mu, body_gate_signal_stride)
               : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
        const double arg = additive_coeff * gate_signal;
        const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
        const double sech2 = 1.0 - tanh_arg * tanh_arg;
        gate_multiplier = 1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
        gate_deriv = two_layer_gate_tanh_amplitude * additive_coeff * sech2;
      } else if (fill_gate_mu_cache) {
        const double additive_coeff = gate_additive_coeff;
        const double gate_signal = full_identity_gate_signal
            ? gate_signal_by_mu[mu]
            : (body_small_gate_signal
               ? dot_small_stride(
                     body_mix_weights +
                         static_cast<size_t>(mu) * body_gate_signal_stride,
                     gate_signal_by_mu, body_gate_signal_stride)
               : two_layer_gate_mu_signal(gate_signal_by_mu, mu));
        const double arg = additive_coeff * gate_signal;
        const double tanh_arg = (arg == 0.0 ? 0.0 : std::tanh(arg));
        const double sech2 = 1.0 - tanh_arg * tanh_arg;
        gate_multiplier = 1.0 + two_layer_gate_tanh_amplitude * tanh_arg;
        gate_deriv = two_layer_gate_tanh_amplitude * additive_coeff * sech2;
        gate_multiplier_cache[mu] = gate_multiplier;
        gate_deriv_cache[mu] = gate_deriv;
      } else {
        gate_multiplier = gate_multiplier_cache[mu];
        gate_deriv = gate_deriv_cache[mu];
      }
      final_type_factor = species_factor * gate_multiplier;
      residual_type_factor = species_factor * gate_deriv;
    }
    if (use_gate_radial && two_layer_gate_shared_radial) {
      const int offset = mu * R;
      for (int ri = 0; ri < R; ri++) {
        const double coeff = two_layer_gate_radial_coeffs[offset + ri] * species_factor;
        val += coeff * two_layer_radial_cache_vals[radial_cache_index * R + ri];
        der += coeff * two_layer_radial_cache_ders[radial_cache_index * R + ri];
      }
    } else {
      const int offset_mu = mu * (R + C);
      for (int ri = 0; ri < R; ri++) {
        const double coeff =
            regression_coeffs[radial_coeffs_offset + offset_mu + ri] * final_type_factor;
        val += coeff * two_layer_radial_cache_vals[radial_cache_index * R + ri];
        der += coeff * two_layer_radial_cache_ders[radial_cache_index * R + ri];
        if (use_gate_additive)
          two_layer_gate_residual_radial_vals[mu] +=
              regression_coeffs[radial_coeffs_offset + offset_mu + ri] *
              residual_type_factor *
              two_layer_radial_cache_vals[radial_cache_index * R + ri];
      }
    }
    radial_vals[mu] = val;
    radial_ders[mu] = der;
  }
  if (fill_gate_mu_cache)
    two_layer_gate_mu_cache_valid[gate_atom_index] = 1;
}

void PairSUS2MTP::accumulate_sh_basic_edge(int jj,
                                           const double *r,
                                           double dist,
                                           double gate_scale,
                                           bool store_raw,
                                           int raw_offset,
                                           bool store_gate_residual_raw,
                                           bool store_center_gate_residual_raw,
                                           double *direct_jac_x,
                                           double *direct_jac_y,
                                           double *direct_jac_z,
                                           double *cached_sh_values,
                                           double *cached_sh_ders)
{
  const double inv_dist = 1.0 / dist;
  if (jj < 0) {
    double sh_values[kMaxSHComponents];
    if (cached_sh_values != nullptr || cached_sh_ders != nullptr) {
      double sh_ders[3 * kMaxSHComponents];
      const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
      eval_real_sh(r, inv_dist, sh_l_max, sh_values, sh_ders);
      if (cached_sh_values != nullptr)
        std::memcpy(cached_sh_values, sh_values,
                    static_cast<size_t>(sh_component_count) * sizeof(double));
      if (cached_sh_ders != nullptr)
        std::memcpy(cached_sh_ders, sh_ders,
                    static_cast<size_t>(3 * sh_component_count) * sizeof(double));
    } else {
      eval_real_sh_values(r, inv_dist, sh_l_max, sh_values);
    }
    if (sh_basic_mu_grouped || sh_basic_mu_indexed) {
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int begin = sh_basic_mu_offsets[mu];
        const int end = sh_basic_mu_offsets[mu + 1];
        if (begin == end) continue;
        const double radial_val = radial_vals[mu];
        for (int pos = begin; pos < end; pos++) {
          const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
          moment_tensor_vals[k] +=
              gate_scale * radial_val * sh_values[alpha_basic_sh_index[k]];
        }
      }
    } else {
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const int mu = alpha_basic_mu[k];
        const int sh_idx = alpha_basic_sh_index[k];
        moment_tensor_vals[k] += gate_scale * radial_vals[mu] * sh_values[sh_idx];
      }
    }
    return;
  }

  double sh_values[kMaxSHComponents];
  double sh_ders[3 * kMaxSHComponents];
  eval_real_sh(r, inv_dist, sh_l_max, sh_values, sh_ders);
  if (cached_sh_values != nullptr || cached_sh_ders != nullptr) {
    const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
    if (cached_sh_values != nullptr)
      std::memcpy(cached_sh_values, sh_values,
                  static_cast<size_t>(sh_component_count) * sizeof(double));
    if (cached_sh_ders != nullptr)
      std::memcpy(cached_sh_ders, sh_ders,
                  static_cast<size_t>(3 * sh_component_count) * sizeof(double));
  }

  if (sh_basic_mu_grouped || sh_basic_mu_indexed) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int begin = sh_basic_mu_offsets[mu];
      const int end = sh_basic_mu_offsets[mu + 1];
      if (begin == end) continue;
      const double radial_val = radial_vals[mu];
      for (int pos = begin; pos < end; pos++) {
        const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
        const int sh_idx = alpha_basic_sh_index[k];
        const double ylm = sh_values[sh_idx];
        const double raw_contrib = radial_val * ylm;
        moment_tensor_vals[k] += gate_scale * raw_contrib;
        if (jj >= 0) {
          const double radial_der = radial_ders[mu];
          const double radial_der_pref = radial_der * inv_dist * ylm;
          const double raw_jac_x =
              radial_der_pref * r[0] + radial_val * sh_ders[3 * sh_idx + 0];
          const double raw_jac_y =
              radial_der_pref * r[1] + radial_val * sh_ders[3 * sh_idx + 1];
          const double raw_jac_z =
              radial_der_pref * r[2] + radial_val * sh_ders[3 * sh_idx + 2];
          if (direct_jac_x != nullptr) {
            direct_jac_x[k] = gate_scale * raw_jac_x;
            direct_jac_y[k] = gate_scale * raw_jac_y;
            direct_jac_z[k] = gate_scale * raw_jac_z;
          } else {
            const size_t jac_idx =
                static_cast<size_t>(jj) * alpha_index_basic_count + k;
            moment_jacobian_x[jac_idx] = gate_scale * raw_jac_x;
            moment_jacobian_y[jac_idx] = gate_scale * raw_jac_y;
            moment_jacobian_z[jac_idx] = gate_scale * raw_jac_z;
          }
          if (store_raw) {
            const double gate_raw = store_gate_residual_raw
                ? two_layer_gate_residual_radial_vals[mu] * ylm
                : raw_contrib;
            two_layer_raw_basic_vals[raw_offset + k] = gate_raw;
            if (store_center_gate_residual_raw)
              two_layer_center_raw_basic_vals[raw_offset + k] =
                  two_layer_gate_center_residual_radial_vals[mu] * ylm;
          }
        }
      }
    }
    return;
  }

  for (int k = 0; k < alpha_index_basic_count; k++) {
    const int mu = alpha_basic_mu[k];
    const int sh_idx = alpha_basic_sh_index[k];
    const double radial_val = radial_vals[mu];
    const double ylm = sh_values[sh_idx];
    const double raw_contrib = radial_val * ylm;
    moment_tensor_vals[k] += gate_scale * raw_contrib;
    if (jj >= 0) {
      const double radial_der = radial_ders[mu];
      const double radial_der_pref = radial_der * inv_dist * ylm;
      const double raw_jac_x =
          radial_der_pref * r[0] + radial_val * sh_ders[3 * sh_idx + 0];
      const double raw_jac_y =
          radial_der_pref * r[1] + radial_val * sh_ders[3 * sh_idx + 1];
      const double raw_jac_z =
          radial_der_pref * r[2] + radial_val * sh_ders[3 * sh_idx + 2];
      if (direct_jac_x != nullptr) {
        direct_jac_x[k] = gate_scale * raw_jac_x;
        direct_jac_y[k] = gate_scale * raw_jac_y;
        direct_jac_z[k] = gate_scale * raw_jac_z;
      } else {
        const size_t jac_idx = static_cast<size_t>(jj) * alpha_index_basic_count + k;
        moment_jacobian_x[jac_idx] = gate_scale * raw_jac_x;
        moment_jacobian_y[jac_idx] = gate_scale * raw_jac_y;
        moment_jacobian_z[jac_idx] = gate_scale * raw_jac_z;
      }
      if (store_raw) {
        const double gate_raw = store_gate_residual_raw
            ? two_layer_gate_residual_radial_vals[mu] * ylm
            : raw_contrib;
        two_layer_raw_basic_vals[raw_offset + k] = gate_raw;
        if (store_center_gate_residual_raw)
          two_layer_center_raw_basic_vals[raw_offset + k] =
              two_layer_gate_center_residual_radial_vals[mu] * ylm;
      }
    }
  }
}

void PairSUS2MTP::accumulate_sh_basic_edge_gate_required(
    const double *r,
    double dist,
    double *cached_sh_values,
    double *cached_sh_ders)
{
  if (two_layer_gate_required_basic_indices.empty() ||
      static_cast<int>(two_layer_gate_required_basic_indices.size()) >=
          alpha_index_basic_count) {
    accumulate_sh_basic_edge(-1, r, dist, 1.0, false, 0, false, false,
                             nullptr, nullptr, nullptr, cached_sh_values,
                             cached_sh_ders);
    return;
  }

  if (two_layer_gate_required_basics_are_y00) {
    if (cached_sh_values != nullptr) {
      const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
      std::memset(cached_sh_values, 0,
                  static_cast<size_t>(sh_component_count) * sizeof(double));
      cached_sh_values[sh_flat_index(0, 0)] = kRealY00;
    }
    if (cached_sh_ders != nullptr) {
      const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
      std::memset(cached_sh_ders, 0,
                  static_cast<size_t>(3 * sh_component_count) * sizeof(double));
    }
    if (static_cast<int>(two_layer_gate_required_basic_mu_offsets.size()) ==
            radial_func_count + 1 &&
        static_cast<int>(two_layer_gate_required_basic_mu_indices.size()) ==
            static_cast<int>(two_layer_gate_required_basic_indices.size())) {
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int begin = two_layer_gate_required_basic_mu_offsets[mu];
        const int end = two_layer_gate_required_basic_mu_offsets[mu + 1];
        if (begin == end) continue;
        const double raw = radial_vals[mu] * kRealY00;
        for (int pos = begin; pos < end; pos++)
          moment_tensor_vals[two_layer_gate_required_basic_mu_indices[pos]] += raw;
      }
      return;
    }
    for (int k : two_layer_gate_required_basic_indices)
      moment_tensor_vals[k] += radial_vals[alpha_basic_mu[k]] * kRealY00;
    return;
  }

  const double inv_dist = 1.0 / dist;
  double sh_values[kMaxSHComponents];
  if (cached_sh_values != nullptr || cached_sh_ders != nullptr) {
    double sh_ders[3 * kMaxSHComponents];
    const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
    eval_real_sh(r, inv_dist, sh_l_max, sh_values, sh_ders);
    if (cached_sh_values != nullptr)
      std::memcpy(cached_sh_values, sh_values,
                  static_cast<size_t>(sh_component_count) * sizeof(double));
    if (cached_sh_ders != nullptr)
      std::memcpy(cached_sh_ders, sh_ders,
                  static_cast<size_t>(3 * sh_component_count) * sizeof(double));
  } else {
    eval_real_sh_values(r, inv_dist, sh_l_max, sh_values);
  }

  if (static_cast<int>(two_layer_gate_required_basic_mu_offsets.size()) ==
          radial_func_count + 1 &&
      static_cast<int>(two_layer_gate_required_basic_mu_indices.size()) ==
          static_cast<int>(two_layer_gate_required_basic_indices.size())) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int begin = two_layer_gate_required_basic_mu_offsets[mu];
      const int end = two_layer_gate_required_basic_mu_offsets[mu + 1];
      if (begin == end) continue;
      const double radial_val = radial_vals[mu];
      for (int pos = begin; pos < end; pos++) {
        const int k = two_layer_gate_required_basic_mu_indices[pos];
        moment_tensor_vals[k] +=
            radial_val * sh_values[alpha_basic_sh_index[k]];
      }
    }
    return;
  }

  for (int k : two_layer_gate_required_basic_indices) {
    const int mu = alpha_basic_mu[k];
    const int sh_idx = alpha_basic_sh_index[k];
    moment_tensor_vals[k] += radial_vals[mu] * sh_values[sh_idx];
  }
}

void PairSUS2MTP::dot_sh_basic_edge_jacobian(int itype,
                                             int jtype,
                                             const double *r,
                                             double dist,
                                             int table_index,
                                             int table_bin,
                                             double table_frac,
                                             const double *basic_adjoints,
                                             bool use_required_gate_basics,
                                             double &fx,
                                             double &fy,
                                             double &fz,
                                             const double *cached_radial_vals,
                                             const double *cached_radial_ders,
                                             const double *cached_sh_values,
                                             const double *cached_sh_ders)
{
  if (cached_radial_vals == nullptr || cached_radial_ders == nullptr)
    calc_pair_radial_values(itype, jtype, dist, two_layer_gate_shared_radial,
                            nullptr, false, -1, table_index, table_bin,
                            table_frac);
  const double *__restrict edge_radial_vals =
      cached_radial_vals != nullptr ? cached_radial_vals : radial_vals;
  const double *__restrict edge_radial_ders =
      cached_radial_ders != nullptr ? cached_radial_ders : radial_ders;

  const double inv_dist = 1.0 / dist;
  if (use_required_gate_basics && two_layer_gate_required_basics_are_y00 &&
      static_cast<int>(two_layer_gate_required_basic_mu_offsets.size()) ==
          radial_func_count + 1 &&
      static_cast<int>(two_layer_gate_required_basic_mu_indices.size()) ==
          static_cast<int>(two_layer_gate_required_basic_indices.size())) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int begin = two_layer_gate_required_basic_mu_offsets[mu];
      const int end = two_layer_gate_required_basic_mu_offsets[mu + 1];
      if (begin == end) continue;
      double adjoint_sum = 0.0;
      for (int pos = begin; pos < end; pos++) {
        const int k = two_layer_gate_required_basic_mu_indices[pos];
        adjoint_sum += basic_adjoints[k];
      }
      if (adjoint_sum == 0.0) continue;
      const double radial_der_pref =
          edge_radial_ders[mu] * inv_dist * kRealY00 * adjoint_sum;
      fx += radial_der_pref * r[0];
      fy += radial_der_pref * r[1];
      fz += radial_der_pref * r[2];
    }
    return;
  }

  double sh_values[kMaxSHComponents];
  double sh_ders[3 * kMaxSHComponents];
  const double *__restrict edge_sh_values = cached_sh_values;
  const double *__restrict edge_sh_ders = cached_sh_ders;
  if (edge_sh_values == nullptr || edge_sh_ders == nullptr) {
    eval_real_sh(r, inv_dist, sh_l_max, sh_values, sh_ders);
    edge_sh_values = sh_values;
    edge_sh_ders = sh_ders;
  }

  if (use_required_gate_basics) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int begin = two_layer_gate_required_basic_mu_offsets[mu];
      const int end = two_layer_gate_required_basic_mu_offsets[mu + 1];
      if (begin == end) continue;
      const double radial_val = edge_radial_vals[mu];
      const double radial_der = edge_radial_ders[mu];
      double ylm_adjoint = 0.0;
      double dylm_adjoint_x = 0.0;
      double dylm_adjoint_y = 0.0;
      double dylm_adjoint_z = 0.0;
      for (int pos = begin; pos < end; pos++) {
        const int k = two_layer_gate_required_basic_mu_indices[pos];
        const double adjoint = basic_adjoints[k];
        if (adjoint == 0.0) continue;
        const int sh_idx = alpha_basic_sh_index[k];
        ylm_adjoint += adjoint * edge_sh_values[sh_idx];
        dylm_adjoint_x += adjoint * edge_sh_ders[3 * sh_idx + 0];
        dylm_adjoint_y += adjoint * edge_sh_ders[3 * sh_idx + 1];
        dylm_adjoint_z += adjoint * edge_sh_ders[3 * sh_idx + 2];
      }
      const double radial_der_over_r = radial_der * inv_dist;
      fx += radial_der_over_r * r[0] * ylm_adjoint +
            radial_val * dylm_adjoint_x;
      fy += radial_der_over_r * r[1] * ylm_adjoint +
            radial_val * dylm_adjoint_y;
      fz += radial_der_over_r * r[2] * ylm_adjoint +
            radial_val * dylm_adjoint_z;
    }
    return;
  }

  const bool sh_basic_mu_adjoint_fast =
      sh_basic_mu_grouped || sh_basic_mu_adjoint_indexed;
  if (sh_basic_mu_adjoint_fast) {
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int begin = sh_basic_mu_grouped
          ? sh_basic_mu_offsets[mu]
          : sh_basic_mu_adjoint_offsets[mu];
      const int end = sh_basic_mu_grouped
          ? sh_basic_mu_offsets[mu + 1]
          : sh_basic_mu_adjoint_offsets[mu + 1];
      if (begin == end) continue;
      const double radial_val = edge_radial_vals[mu];
      const double radial_der = edge_radial_ders[mu];
      double ylm_adjoint = 0.0;
      double dylm_adjoint_x = 0.0;
      double dylm_adjoint_y = 0.0;
      double dylm_adjoint_z = 0.0;
      for (int pos = begin; pos < end; pos++) {
        const int k = sh_basic_mu_grouped
            ? pos
            : sh_basic_mu_adjoint_indices[pos];
        const double adjoint = basic_adjoints[k];
        const int sh_idx = alpha_basic_sh_index[k];
        ylm_adjoint += adjoint * edge_sh_values[sh_idx];
        dylm_adjoint_x += adjoint * edge_sh_ders[3 * sh_idx + 0];
        dylm_adjoint_y += adjoint * edge_sh_ders[3 * sh_idx + 1];
        dylm_adjoint_z += adjoint * edge_sh_ders[3 * sh_idx + 2];
      }
      const double radial_der_over_r = radial_der * inv_dist;
      fx += radial_der_over_r * r[0] * ylm_adjoint +
            radial_val * dylm_adjoint_x;
      fy += radial_der_over_r * r[1] * ylm_adjoint +
            radial_val * dylm_adjoint_y;
      fz += radial_der_over_r * r[2] * ylm_adjoint +
            radial_val * dylm_adjoint_z;
    }
    return;
  }

  for (int k = 0; k < alpha_index_basic_count; k++) {
    const double adjoint = basic_adjoints[k];
    if (adjoint == 0.0) continue;
    const int mu = alpha_basic_mu[k];
    const int sh_idx = alpha_basic_sh_index[k];
    const double radial_val = edge_radial_vals[mu];
    const double radial_der = edge_radial_ders[mu];
    const double ylm = edge_sh_values[sh_idx];
    const double radial_der_pref = radial_der * inv_dist * ylm;
    fx += adjoint *
          (radial_der_pref * r[0] + radial_val * edge_sh_ders[3 * sh_idx + 0]);
    fy += adjoint *
          (radial_der_pref * r[1] + radial_val * edge_sh_ders[3 * sh_idx + 1]);
    fz += adjoint *
          (radial_der_pref * r[2] + radial_val * edge_sh_ders[3 * sh_idx + 2]);
  }
}

void PairSUS2MTP::forward_sh_products()
{
  for (int k = 0; k < alpha_index_times_count; k++) {
    moment_tensor_vals[alpha_times_out[k]] +=
        alpha_times_coeff[k] * moment_tensor_vals[alpha_times_a0[k]] *
        moment_tensor_vals[alpha_times_a1[k]];
  }
}

void PairSUS2MTP::backprop_sh_products()
{
  backprop_sh_products_from(moment_tensor_vals, alpha_index_times_count);
}

void PairSUS2MTP::backprop_sh_products_from(const double *moments, int product_limit)
{
  for (int k = product_limit - 1; k >= 0; k--) {
    const int a0 = alpha_times_a0[k];
    const int a1 = alpha_times_a1[k];
    const int out = alpha_times_out[k];
    const double coeff = alpha_times_coeff[k];
    const double adj = nbh_energy_ders_wrt_moments[out];
    nbh_energy_ders_wrt_moments[a1] += adj * coeff * moments[a0];
    nbh_energy_ders_wrt_moments[a0] += adj * coeff * moments[a1];
  }
}

void PairSUS2MTP::forward_two_layer_gate_products()
{
  if (!two_layer_gate_product_out_indices.empty()) {
    const int count =
        static_cast<int>(two_layer_gate_product_out_indices.size());
    const int *__restrict product_a0 =
        two_layer_gate_product_a0_indices.data();
    const int *__restrict product_a1 =
        two_layer_gate_product_a1_indices.data();
    const int *__restrict product_out =
        two_layer_gate_product_out_indices.data();
    const double *__restrict product_coeff =
        two_layer_gate_product_coeffs.data();
    for (int idx = 0; idx < count; idx++) {
      moment_tensor_vals[product_out[idx]] +=
          product_coeff[idx] * moment_tensor_vals[product_a0[idx]] *
          moment_tensor_vals[product_a1[idx]];
    }
    return;
  }
  for (int k = 0; k < two_layer_gate_product_limit; k++) {
    moment_tensor_vals[alpha_times_out[k]] +=
        alpha_times_coeff[k] * moment_tensor_vals[alpha_times_a0[k]] *
        moment_tensor_vals[alpha_times_a1[k]];
  }
}

void PairSUS2MTP::backprop_two_layer_gate_products(const double *moments)
{
  if (!two_layer_gate_product_out_indices.empty()) {
    const int *__restrict product_a0 =
        two_layer_gate_product_a0_indices.data();
    const int *__restrict product_a1 =
        two_layer_gate_product_a1_indices.data();
    const int *__restrict product_out =
        two_layer_gate_product_out_indices.data();
    const double *__restrict product_coeff =
        two_layer_gate_product_coeffs.data();
    for (int idx = static_cast<int>(two_layer_gate_product_out_indices.size()) - 1;
         idx >= 0; idx--) {
      const int a0 = product_a0[idx];
      const int a1 = product_a1[idx];
      const int out = product_out[idx];
      const double coeff = product_coeff[idx];
      const double adj = nbh_energy_ders_wrt_moments[out];
      nbh_energy_ders_wrt_moments[a1] += adj * coeff * moments[a0];
      nbh_energy_ders_wrt_moments[a0] += adj * coeff * moments[a1];
    }
    return;
  }
  backprop_sh_products_from(moments, two_layer_gate_product_limit);
}

void PairSUS2MTP::backprop_two_layer_gate_products_compact(const double *moments)
{
  if (moments == nullptr ||
      two_layer_gate_required_moment_compact_index.empty() ||
      two_layer_gate_product_out_indices.empty() ||
      two_layer_gate_product_compact_a0_indices.empty() ||
      two_layer_gate_product_compact_a1_indices.empty()) {
    backprop_two_layer_gate_products(moments);
    return;
  }

  const int *__restrict product_a0 =
      two_layer_gate_product_a0_indices.data();
  const int *__restrict product_a1 =
      two_layer_gate_product_a1_indices.data();
  const int *__restrict product_out =
      two_layer_gate_product_out_indices.data();
  const int *__restrict product_compact_a0 =
      two_layer_gate_product_compact_a0_indices.data();
  const int *__restrict product_compact_a1 =
      two_layer_gate_product_compact_a1_indices.data();
  const double *__restrict product_coeff =
      two_layer_gate_product_coeffs.data();
  for (int idx = static_cast<int>(two_layer_gate_product_out_indices.size()) - 1;
       idx >= 0; idx--) {
    const int a0 = product_a0[idx];
    const int a1 = product_a1[idx];
    const int out = product_out[idx];
    const int a0_compact = product_compact_a0[idx];
    const int a1_compact = product_compact_a1[idx];
    if (a0_compact < 0 || a1_compact < 0)
      error->one(FLERR, "SUS2-SH compact gate moment cache is inconsistent.");
    const double coeff = product_coeff[idx];
    const double adj = nbh_energy_ders_wrt_moments[out];
    nbh_energy_ders_wrt_moments[a1] += adj * coeff * moments[a0_compact];
    nbh_energy_ders_wrt_moments[a0] += adj * coeff * moments[a1_compact];
  }
}

int PairSUS2MTP::pack_forward_comm(int n, int *list, double *buf, int, int *)
{
  int m = 0;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const bool edge_l1_active = two_layer_gate_edge_l1_active();
  const int edge_l1_stride = two_layer_gate_edge_l1_weighted_stride();
  if (edge_l1_active) {
    double *gate_buf = buf;
    double *edge_buf =
        buf + static_cast<size_t>(n) * gate_signal_stride;
    const size_t gate_bytes =
        static_cast<size_t>(gate_signal_stride) * sizeof(double);
    const size_t edge_bytes =
        static_cast<size_t>(edge_l1_stride) * sizeof(double);
    for (int i = 0; i < n; i++) {
      const int atom_index = list[i];
      const double *gate_src =
          two_layer_gate_values + static_cast<size_t>(atom_index) * gate_signal_stride;
      const double *edge_src =
          two_layer_gate_edge_l1_weighted_values +
          static_cast<size_t>(atom_index) * edge_l1_stride;
      std::memcpy(gate_buf + static_cast<size_t>(i) * gate_signal_stride,
                  gate_src, gate_bytes);
      std::memcpy(edge_buf + static_cast<size_t>(i) * edge_l1_stride,
                  edge_src, edge_bytes);
    }
    return n * (gate_signal_stride + edge_l1_stride);
  }
  if (pack_forward_gate_values_specialized(
          gate_signal_stride, n, list, two_layer_gate_values, buf, m))
    return m;
  const size_t bytes = static_cast<size_t>(gate_signal_stride) * sizeof(double);
  for (int i = 0; i < n; i++) {
    const double *src =
        two_layer_gate_values + static_cast<size_t>(list[i]) * gate_signal_stride;
    std::memcpy(buf + m, src, bytes);
    m += gate_signal_stride;
  }
  return m;
}

void PairSUS2MTP::unpack_forward_comm(int n, int first, double *buf)
{
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const bool edge_l1_active = two_layer_gate_edge_l1_active();
  const int edge_l1_stride = two_layer_gate_edge_l1_weighted_stride();
  if (edge_l1_active) {
    const size_t gate_count = static_cast<size_t>(n) * gate_signal_stride;
    const size_t edge_count = static_cast<size_t>(n) * edge_l1_stride;
    std::memcpy(two_layer_gate_values +
                    static_cast<size_t>(first) * gate_signal_stride,
                buf, gate_count * sizeof(double));
    std::memcpy(two_layer_gate_edge_l1_weighted_values +
                    static_cast<size_t>(first) * edge_l1_stride,
                buf + gate_count, edge_count * sizeof(double));
    return;
  }
  if (gate_signal_stride == 1) {
    std::memcpy(two_layer_gate_values + first, buf,
                static_cast<size_t>(n) * sizeof(double));
    return;
  }
  const size_t count = static_cast<size_t>(n) * gate_signal_stride;
  double *dst =
      two_layer_gate_values + static_cast<size_t>(first) * gate_signal_stride;
  std::memcpy(dst, buf, count * sizeof(double));
}

int PairSUS2MTP::pack_reverse_comm(int n, int first, double *buf)
{
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const bool edge_l1_active = two_layer_gate_edge_l1_active();
  const int edge_l1_stride = two_layer_gate_edge_l1_weighted_stride();
  if (edge_l1_active) {
    const size_t gate_count = static_cast<size_t>(n) * gate_signal_stride;
    const size_t edge_count = static_cast<size_t>(n) * edge_l1_stride;
    std::memcpy(buf,
                two_layer_gate_adjoints +
                    static_cast<size_t>(first) * gate_signal_stride,
                gate_count * sizeof(double));
    std::memcpy(buf + gate_count,
                two_layer_gate_edge_l1_weighted_adjoints +
                    static_cast<size_t>(first) * edge_l1_stride,
                edge_count * sizeof(double));
    return static_cast<int>(gate_count + edge_count);
  }
  if (gate_signal_stride == 1) {
    std::memcpy(buf, two_layer_gate_adjoints + first,
                static_cast<size_t>(n) * sizeof(double));
    return n;
  }
  const size_t count = static_cast<size_t>(n) * gate_signal_stride;
  const double *src =
      two_layer_gate_adjoints + static_cast<size_t>(first) * gate_signal_stride;
  std::memcpy(buf, src, count * sizeof(double));
  return static_cast<int>(count);
}

void PairSUS2MTP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const bool edge_l1_active = two_layer_gate_edge_l1_active();
  const int edge_l1_stride = two_layer_gate_edge_l1_weighted_stride();
  if (edge_l1_active) {
    const double *gate_buf = buf;
    const double *edge_buf =
        buf + static_cast<size_t>(n) * gate_signal_stride;
    for (int i = 0; i < n; i++) {
      const int atom_index = list[i];
      double *gate_dst =
          two_layer_gate_adjoints + static_cast<size_t>(atom_index) * gate_signal_stride;
      const double *gate_src =
          gate_buf + static_cast<size_t>(i) * gate_signal_stride;
      for (int k = 0; k < gate_signal_stride; k++) gate_dst[k] += gate_src[k];
      double *edge_dst =
          two_layer_gate_edge_l1_weighted_adjoints +
          static_cast<size_t>(atom_index) * edge_l1_stride;
      const double *edge_src =
          edge_buf + static_cast<size_t>(i) * edge_l1_stride;
      #pragma omp simd
      for (int k = 0; k < edge_l1_stride; k++) edge_dst[k] += edge_src[k];
    }
    return;
  }
  if (unpack_reverse_gate_adjoints_specialized(
          gate_signal_stride, n, list, buf, two_layer_gate_adjoints))
    return;
  for (int i = 0; i < n; i++) {
    double *dst =
        two_layer_gate_adjoints + static_cast<size_t>(list[i]) * gate_signal_stride;
    for (int k = 0; k < gate_signal_stride; k++) dst[k] += buf[m++];
  }
}

void PairSUS2MTP::compute_two_layer_gate_sh(int eflag, int vflag)
{
  const bool profile_gate = env_flag_enabled("SUS2_LAMMPS_GATE_PROFILE");
  const bool profile_gate_v2 = env_flag_enabled("SUS2_LAMMPS_GATE_PROFILE_V2");
  const bool profile_any_gate = profile_gate || profile_gate_v2;
  const double profile_t0 = profile_any_gate ? MPI_Wtime() : 0.0;
  double profile_first_layer = 0.0;
  double profile_forward_comm = 0.0;
  double profile_main_layer = 0.0;
  double profile_reverse_comm = 0.0;
  double profile_gate_force = 0.0;
  double profile_v2_times[kGateProfileV2TimerCount] = {};

  if (env_gate_enabled)
    error->all(FLERR, "LAMMPS SUS2-SH two-layer gate cannot be combined with env_gate.");
  if (two_layer_residual_enabled)
    error->all(FLERR, "LAMMPS SUS2-SH interface does not support residual two-layer models.");
  const int expected_gate_weight_count = two_layer_gate_full_scalar_weights
      ? radial_func_count * two_layer_gate_scalar_count
      : two_layer_gate_scalar_count;
  const int expected_gate_body_mix_count = two_layer_gate_body_linear_combo
      ? radial_func_count * (two_layer_gate_body_order_max - 1)
      : 0;
  const int gate_signal_stride = two_layer_gate_signal_stride();
  const bool edge_l1_active = two_layer_gate_edge_l1_active();
  const int edge_l1_weighted_stride = two_layer_gate_edge_l1_weighted_stride();
  if (two_layer_gate_scalar_count != static_cast<int>(two_layer_gate_scalar_indices.size()) ||
      two_layer_gate_weight_count != expected_gate_weight_count ||
      two_layer_gate_weight_count != static_cast<int>(two_layer_gate_weights.size()) ||
      two_layer_gate_body_mix_weight_count != expected_gate_body_mix_count ||
      two_layer_gate_body_mix_weight_count !=
          static_cast<int>(two_layer_gate_body_mix_weights.size()) ||
	      two_layer_gate_scalar_count !=
	          static_cast<int>(two_layer_gate_scalar_moment_indices.size()) ||
	      two_layer_gate_scalar_count !=
	          static_cast<int>(two_layer_gate_scalar_body_bucket.size()) ||
	      (two_layer_gate_body_linear_combo &&
	       (static_cast<int>(two_layer_gate_body_bucket_offsets.size()) !=
	            std::max(0, two_layer_gate_body_order_max - 1) + 1 ||
	        static_cast<int>(two_layer_gate_body_bucket_scalar_indices.size()) !=
	            two_layer_gate_scalar_count)) ||
	      gate_signal_stride <= 0 ||
        (edge_l1_active &&
         (edge_l1_weighted_stride != 3 * radial_func_count ||
          static_cast<int>(two_layer_gate_edge_l1_weights_by_source.size()) !=
              two_layer_gate_edge_l1_source_count * radial_func_count)))
	    error->all(FLERR, "SUS2-SH two-layer gate metadata has inconsistent sizes.");

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  const int gate_product_limit = two_layer_gate_product_limit;
  const int gate_moment_cache_width = two_layer_gate_moment_cache_width;
  if (gate_product_limit < 0 || gate_product_limit > alpha_index_times_count ||
      gate_moment_cache_width <= 0 ||
      gate_moment_cache_width > alpha_moment_count)
    error->all(FLERR, "SUS2-SH two-layer gate product layout is inconsistent.");

  ensure_two_layer_atom_buffers();
  const int nall = atom->nlocal + atom->nghost;
  const int gate_body_stride = two_layer_gate_body_order_max - 1;
  const bool compact_body_signal =
      two_layer_gate_body_linear_combo &&
      !two_layer_gate_body_signal_buckets.empty();
  const double *body_signal_mix_weights =
      compact_body_signal
      ? two_layer_gate_body_signal_mix_weights.data()
      : two_layer_gate_body_mix_weights.data();
  const bool sh_basic_mu_fast = sh_basic_mu_grouped || sh_basic_mu_indexed;
  const bool direct_body_combo_gate_adjoints =
      two_layer_gate_body_linear_combo && gate_body_stride > 0 &&
      gate_signal_stride > 0 && gate_signal_stride <= kMaxBodyGateStrideFast &&
      !two_layer_gate_center_enabled && sh_basic_mu_fast && !edge_l1_active;
  const bool direct_full_gate_adjoints =
      !two_layer_gate_body_linear_combo &&
      !two_layer_gate_center_enabled && sh_basic_mu_fast &&
      gate_signal_stride == radial_func_count && !edge_l1_active;
	  const bool direct_grouped_full_gate_adjoints =
	      !two_layer_gate_body_linear_combo &&
	      !two_layer_gate_center_enabled && sh_basic_mu_fast &&
	      gate_signal_stride > 0 && gate_signal_stride < radial_func_count &&
	      static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
	          radial_func_count && !edge_l1_active;
	  const bool edge_l1_direct_scalar_gate_adjoint =
	      edge_l1_active && !two_layer_gate_body_linear_combo &&
	      !two_layer_gate_center_enabled && gate_signal_stride == 1 &&
	      static_cast<int>(two_layer_gate_full_signal_group_for_mu.size()) ==
	          radial_func_count;
  const bool cache_body_signal_derivs =
      two_layer_gate_body_linear_combo && !two_layer_gate_center_enabled &&
      !compact_body_signal && gate_body_stride > 0 &&
      gate_body_stride <= kMaxBodyGateStrideFast &&
      !edge_l1_active &&
      env_flag_enabled("SUS2_LAMMPS_GATE_SIGNAL_DERIV_CACHE");
  const bool gate_needs_product_backprop =
      gate_product_limit > 0 || !two_layer_gate_product_indices.empty();
  const bool store_gate_moment_cache =
      !cache_body_signal_derivs && gate_needs_product_backprop;
  const bool compact_gate_moment_cache =
      store_gate_moment_cache &&
      env_flag_enabled("SUS2_LAMMPS_GATE_COMPACT_MOMENT_CACHE") &&
      !two_layer_gate_product_indices.empty() &&
      !two_layer_gate_required_moment_indices.empty() &&
      static_cast<int>(two_layer_gate_required_moment_compact_index.size()) ==
          alpha_moment_count &&
      static_cast<int>(two_layer_gate_required_moment_indices.size()) <
          gate_moment_cache_width;
  const int gate_moment_cache_storage_width =
      compact_gate_moment_cache
      ? static_cast<int>(two_layer_gate_required_moment_indices.size())
      : gate_moment_cache_width;
  const bool gate_scalars_basic_only = !gate_needs_product_backprop;
  const bool use_gate_required_basic_subset =
      should_use_gate_required_basic_subset(
          static_cast<int>(two_layer_gate_required_basic_indices.size()),
          alpha_index_basic_count, two_layer_gate_required_basics_are_y00,
          two_layer_gate_body_linear_combo, two_layer_gate_center_enabled,
          cache_body_signal_derivs, sh_basic_mu_fast, gate_body_stride,
          gate_signal_stride);
  const bool gate_required_dot_available =
      static_cast<int>(two_layer_gate_required_basic_mu_offsets.size()) ==
          radial_func_count + 1 &&
      static_cast<int>(two_layer_gate_required_basic_mu_indices.size()) ==
          static_cast<int>(two_layer_gate_required_basic_indices.size());
  std::vector<int> basic_gate_scalar_indices;
  std::vector<int> basic_gate_body_bucket_offsets;
  std::vector<int> basic_gate_body_bucket_scalar_indices;
  if (gate_scalars_basic_only) {
    // With no product graph, only basic moments are ever populated; mapped
    // non-basic gate scalars are exact zero contributions in this LAMMPS path.
    basic_gate_scalar_indices.reserve(two_layer_gate_scalar_count);
    if (two_layer_gate_body_linear_combo)
      basic_gate_body_bucket_offsets.assign(gate_body_stride + 1, 0);
    for (int q = 0; q < two_layer_gate_scalar_count; q++) {
      const int moment_index = two_layer_gate_scalar_moment_indices[q];
      if (moment_index < 0 || moment_index >= alpha_index_basic_count)
        continue;
      basic_gate_scalar_indices.push_back(q);
      if (two_layer_gate_body_linear_combo)
        basic_gate_body_bucket_offsets[two_layer_gate_scalar_body_bucket[q] + 1]++;
    }
    if (two_layer_gate_body_linear_combo) {
      for (int b = 0; b < gate_body_stride; b++)
        basic_gate_body_bucket_offsets[b + 1] +=
            basic_gate_body_bucket_offsets[b];
      basic_gate_body_bucket_scalar_indices.assign(
          basic_gate_scalar_indices.size(), 0);
      std::vector<int> cursor = basic_gate_body_bucket_offsets;
      for (int q : basic_gate_scalar_indices) {
        const int bucket = two_layer_gate_scalar_body_bucket[q];
        basic_gate_body_bucket_scalar_indices[cursor[bucket]++] = q;
      }
    }
  }
  const int first_gate_moment_fill_width =
      gate_scalars_basic_only ? alpha_index_basic_count : gate_moment_cache_width;
  const int gate_force_adjoint_fill_width =
      gate_scalars_basic_only ? alpha_index_basic_count : gate_moment_cache_width;
  const bool use_full_gate_batched_contraction =
      !two_layer_gate_body_linear_combo && !gate_scalars_basic_only &&
      !static_fixed_basic_cache_enabled &&
      env_flag_enabled("SUS2_LAMMPS_FULL_GATE_BATCH_CONTRACT") &&
      static_cast<int>(two_layer_gate_full_weights_by_scalar.size()) ==
          two_layer_gate_scalar_count * gate_signal_stride;
  std::vector<double> gate_body_values(gate_body_stride, 0.0);
  std::vector<double> gate_signal_moment_ders;
  if (cache_body_signal_derivs)
    gate_signal_moment_ders.resize(
        static_cast<size_t>(gate_body_stride) * gate_moment_cache_width);
  if (store_gate_moment_cache)
    two_layer_gate_moment_cache.resize(
        static_cast<size_t>(inum) * gate_moment_cache_storage_width);
  if (use_full_gate_batched_contraction) {
    const size_t full_scalar_cache_size =
        static_cast<size_t>(two_layer_gate_scalar_count) * inum;
    two_layer_gate_full_scalar_cache.resize(full_scalar_cache_size);
    two_layer_gate_full_scalar_adjoint_cache.resize(full_scalar_cache_size);
  } else {
    two_layer_gate_full_scalar_cache.clear();
    two_layer_gate_full_scalar_adjoint_cache.clear();
  }
  std::fill(two_layer_gate_adjoints,
            two_layer_gate_adjoints + static_cast<size_t>(nall) * gate_signal_stride,
            0.0);
  if (edge_l1_active) {
    std::fill(two_layer_gate_edge_l1_weighted_values,
              two_layer_gate_edge_l1_weighted_values +
                  static_cast<size_t>(nall) * edge_l1_weighted_stride,
              0.0);
    std::fill(two_layer_gate_edge_l1_weighted_adjoints,
              two_layer_gate_edge_l1_weighted_adjoints +
                  static_cast<size_t>(nall) * edge_l1_weighted_stride,
              0.0);
  }
  std::vector<double> edge_l1_signal_scratch;
  if (edge_l1_active) edge_l1_signal_scratch.resize(radial_func_count);

  two_layer_gate_edge_offsets.resize(static_cast<size_t>(inum) + 1);
  size_t edge_capacity = 0;
  for (int ii = 0; ii < inum; ii++)
    edge_capacity += static_cast<size_t>(numneigh[ilist[ii]]);
  const int sh_component_count = (sh_l_max + 1) * (sh_l_max + 1);
  if (sh_component_count <= 0 || sh_component_count > kMaxSHComponents)
    error->all(FLERR, "SUS2-SH two-layer gate SH cache width is invalid.");
  const bool use_compact_gate_raw =
      !two_layer_gate_center_enabled && sh_basic_mu_fast &&
      env_flag_enabled("SUS2_LAMMPS_GATE_COMPACT_RAW");
  const bool cache_full_gate_force_jacobians =
      !two_layer_gate_body_linear_combo &&
      should_cache_full_gate_force_jacobians(edge_capacity,
                                             alpha_index_basic_count);
  const bool recompute_body_combo_gate_force_jacobians =
      two_layer_gate_body_linear_combo && !cache_body_signal_derivs;
  const bool recompute_gate_force_jacobians =
      recompute_body_combo_gate_force_jacobians ||
      (!two_layer_gate_body_linear_combo && !cache_full_gate_force_jacobians);
  const bool use_gate_required_dot =
      gate_required_dot_available && recompute_gate_force_jacobians &&
      should_use_gate_required_dot(
          static_cast<int>(two_layer_gate_required_basic_indices.size()),
          alpha_index_basic_count, two_layer_gate_required_basics_are_y00,
          two_layer_gate_body_linear_combo, two_layer_gate_center_enabled,
          cache_body_signal_derivs, sh_basic_mu_fast, gate_body_stride,
          gate_signal_stride);
  const bool cache_gate_force_radials = recompute_gate_force_jacobians;
  const bool cache_gate_force_sh_derivs =
      recompute_gate_force_jacobians &&
      env_flag_enabled("SUS2_LAMMPS_GATE_SH_DERIV_CACHE");
  const size_t radial_cache_capacity =
      cache_gate_force_radials
      ? edge_capacity * static_cast<size_t>(radial_func_count)
      : 0;
  const size_t sh_cache_capacity =
      cache_gate_force_sh_derivs
      ? edge_capacity * static_cast<size_t>(sh_component_count)
      : 0;
  const size_t main_sh_cache_capacity =
      use_compact_gate_raw
      ? edge_capacity * static_cast<size_t>(sh_component_count)
      : 0;
  const size_t residual_radial_cache_capacity =
      use_compact_gate_raw
      ? edge_capacity * static_cast<size_t>(radial_func_count)
      : 0;
  const size_t deriv_capacity =
      (recompute_gate_force_jacobians || cache_body_signal_derivs)
      ? 0
      : edge_capacity * static_cast<size_t>(alpha_index_basic_count);
  const size_t signal_deriv_capacity =
      cache_body_signal_derivs
      ? edge_capacity * static_cast<size_t>(gate_signal_stride)
      : 0;
  if (edge_capacity > two_layer_gate_edge_capacity) {
    memory->grow(two_layer_gate_edge_neighbors_raw, edge_capacity,
                 "two_layer_gate_edge_neighbors_raw");
    memory->grow(two_layer_gate_edge_types_raw, edge_capacity,
                 "two_layer_gate_edge_types_raw");
    memory->grow(two_layer_gate_edge_table_indices_raw, edge_capacity,
                 "two_layer_gate_edge_table_indices_raw");
    memory->grow(two_layer_gate_edge_table_bins_raw, edge_capacity,
                 "two_layer_gate_edge_table_bins_raw");
    memory->grow(two_layer_gate_edge_dx_raw, edge_capacity,
                 "two_layer_gate_edge_dx_raw");
    memory->grow(two_layer_gate_edge_dy_raw, edge_capacity,
                 "two_layer_gate_edge_dy_raw");
    memory->grow(two_layer_gate_edge_dz_raw, edge_capacity,
                 "two_layer_gate_edge_dz_raw");
    memory->grow(two_layer_gate_edge_dist_raw, edge_capacity,
                 "two_layer_gate_edge_dist_raw");
    memory->grow(two_layer_gate_edge_table_fracs_raw, edge_capacity,
                 "two_layer_gate_edge_table_fracs_raw");
    two_layer_gate_edge_capacity = edge_capacity;
  }
  if (radial_cache_capacity > two_layer_gate_edge_radial_capacity) {
    memory->grow(two_layer_gate_edge_radial_vals_raw, radial_cache_capacity,
                 "two_layer_gate_edge_radial_vals_raw");
    memory->grow(two_layer_gate_edge_radial_ders_raw, radial_cache_capacity,
                 "two_layer_gate_edge_radial_ders_raw");
    two_layer_gate_edge_radial_capacity = radial_cache_capacity;
  }
  if (sh_cache_capacity > two_layer_gate_edge_sh_capacity) {
    memory->grow(two_layer_gate_edge_sh_values_raw, sh_cache_capacity,
                 "two_layer_gate_edge_sh_values_raw");
    memory->grow(two_layer_gate_edge_sh_ders_raw, 3 * sh_cache_capacity,
                 "two_layer_gate_edge_sh_ders_raw");
    two_layer_gate_edge_sh_capacity = sh_cache_capacity;
  }
  if (main_sh_cache_capacity > two_layer_gate_edge_main_sh_capacity) {
    memory->grow(two_layer_gate_edge_main_sh_values_raw, main_sh_cache_capacity,
                 "two_layer_gate_edge_main_sh_values_raw");
    two_layer_gate_edge_main_sh_capacity = main_sh_cache_capacity;
  }
  if (residual_radial_cache_capacity >
      two_layer_gate_edge_residual_radial_capacity) {
    memory->grow(two_layer_gate_edge_residual_radial_vals_raw,
                 residual_radial_cache_capacity,
                 "two_layer_gate_edge_residual_radial_vals_raw");
    two_layer_gate_edge_residual_radial_capacity =
        residual_radial_cache_capacity;
  }
  if (deriv_capacity > two_layer_gate_edge_deriv_capacity) {
    memory->grow(two_layer_gate_edge_deriv_x_raw, deriv_capacity,
                 "two_layer_gate_edge_deriv_x_raw");
    memory->grow(two_layer_gate_edge_deriv_y_raw, deriv_capacity,
                 "two_layer_gate_edge_deriv_y_raw");
    memory->grow(two_layer_gate_edge_deriv_z_raw, deriv_capacity,
                 "two_layer_gate_edge_deriv_z_raw");
    two_layer_gate_edge_deriv_capacity = deriv_capacity;
  }
  if (signal_deriv_capacity > two_layer_gate_edge_signal_deriv_capacity) {
    memory->grow(two_layer_gate_edge_signal_deriv_x_raw, signal_deriv_capacity,
                 "two_layer_gate_edge_signal_deriv_x_raw");
    memory->grow(two_layer_gate_edge_signal_deriv_y_raw, signal_deriv_capacity,
                 "two_layer_gate_edge_signal_deriv_y_raw");
    memory->grow(two_layer_gate_edge_signal_deriv_z_raw, signal_deriv_capacity,
                 "two_layer_gate_edge_signal_deriv_z_raw");
    two_layer_gate_edge_signal_deriv_capacity = signal_deriv_capacity;
  }
  two_layer_gate_edge_offsets[0] = 0;

  double profile_stage_start = profile_any_gate ? MPI_Wtime() : 0.0;
  size_t edge_count = 0;
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i] - 1;
    if (itype >= species_count)
      error->one(FLERR, "Too few species count in the MTP potential!");
    const int jnum = numneigh[i];
    const double xi[3] = {x[i][0], x[i][1], x[i][2]};
    ensure_two_layer_edge_buffer(jnum);
    std::fill(moment_tensor_vals,
              moment_tensor_vals + first_gate_moment_fill_width,
              0.0);
    const bool use_static_fixed_gate_cache =
        static_fixed_basic_cache_enabled && is_sh_model &&
        !eflag && !vflag && is_static_fixed_type(itype) &&
        !edge_l1_active;
    if (use_static_fixed_gate_cache) {
      ensure_static_fixed_gate_cache();
      const bool gate_cache_tag_match =
          static_fixed_cache_tag_matches(static_fixed_gate_cache_tags, i);
      if (static_fixed_gate_basic_cache_valid[i] && gate_cache_tag_match) {
        const double *cache = static_fixed_gate_basic_cache +
            static_cast<size_t>(i) * alpha_index_basic_count;
        std::copy(cache, cache + alpha_index_basic_count, moment_tensor_vals);
      } else {
        build_static_fixed_gate_basic_cache_for_center(i, itype, jnum,
                                                       firstneigh[i], xi);
      }
    }
    const size_t active_begin = edge_count;
    int first_layer_active_local_count = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj] & NEIGHMASK;
      const int jtype = type[j] - 1;
      if (jtype >= species_count)
        error->one(FLERR, "Too few species count in the MTP potential!");
      const bool static_first_layer_edge =
          use_static_fixed_gate_cache && is_static_fixed_type(jtype);
      const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      if (!static_first_layer_edge && zbl_enabled && rsq < zbl_max_outer_sq)
        accumulate_zbl_pair(i, j, itype, jtype, r, rsq, eflag, vflag);
      if (rsq > max_cutoff_sq) continue;
      const double dist = std::sqrt(rsq);
      if (static_first_layer_edge) continue;
      int table_index = -1;
      int table_bin = 0;
      double table_frac = 0.0;
      get_radial_table_info(itype, jtype, dist, table_index, table_bin,
                            table_frac);
      const size_t edge_index = edge_count++;
      two_layer_gate_edge_neighbors_raw[edge_index] = j;
      two_layer_gate_edge_types_raw[edge_index] = jtype;
      two_layer_gate_edge_table_indices_raw[edge_index] = table_index;
      two_layer_gate_edge_table_bins_raw[edge_index] = table_bin;
      two_layer_gate_edge_dx_raw[edge_index] = r[0];
      two_layer_gate_edge_dy_raw[edge_index] = r[1];
      two_layer_gate_edge_dz_raw[edge_index] = r[2];
      two_layer_gate_edge_dist_raw[edge_index] = dist;
      two_layer_gate_edge_table_fracs_raw[edge_index] = table_frac;
      double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
      calc_pair_radial_values(itype, jtype, dist, two_layer_gate_shared_radial,
                              nullptr, false, -1, table_index, table_bin,
                              table_frac);
      add_profile_time(profile_gate_v2,
                       profile_v2_times[kGateProfileV2FirstRadial],
                       profile_v2_sub_start);
      if (cache_gate_force_radials) {
        const size_t radial_base =
            edge_index * static_cast<size_t>(radial_func_count);
        std::memcpy(two_layer_gate_edge_radial_vals_raw + radial_base,
                    radial_vals,
                    static_cast<size_t>(radial_func_count) * sizeof(double));
        std::memcpy(two_layer_gate_edge_radial_ders_raw + radial_base,
                    radial_ders,
                    static_cast<size_t>(radial_func_count) * sizeof(double));
      }
      const bool store_global_gate_derivs =
          !recompute_gate_force_jacobians && !cache_body_signal_derivs;
      const size_t deriv_base = store_global_gate_derivs
          ? edge_index * static_cast<size_t>(alpha_index_basic_count)
          : 0;
      const int first_layer_jac_slot =
          (recompute_gate_force_jacobians && !cache_body_signal_derivs)
          ? 0
          : first_layer_active_local_count;
      const size_t sh_cache_base =
          edge_index * static_cast<size_t>(sh_component_count);
      profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
      if (use_gate_required_basic_subset &&
          recompute_gate_force_jacobians && !cache_body_signal_derivs) {
        accumulate_sh_basic_edge_gate_required(
            r, dist,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_values_raw + sh_cache_base
            : nullptr,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_ders_raw + 3 * sh_cache_base
            : nullptr);
      } else {
        accumulate_sh_basic_edge(
            (recompute_gate_force_jacobians && !cache_body_signal_derivs)
            ? -1
            : first_layer_jac_slot,
            r, dist, 1.0, false, 0, false, false,
            store_global_gate_derivs
            ? two_layer_gate_edge_deriv_x_raw + deriv_base
            : nullptr,
            store_global_gate_derivs
            ? two_layer_gate_edge_deriv_y_raw + deriv_base
            : nullptr,
            store_global_gate_derivs
            ? two_layer_gate_edge_deriv_z_raw + deriv_base
            : nullptr,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_values_raw + sh_cache_base
            : nullptr,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_ders_raw + 3 * sh_cache_base
            : nullptr);
      }
      add_profile_time(profile_gate_v2,
                       profile_v2_times[kGateProfileV2FirstSHBasic],
                       profile_v2_sub_start);
      first_layer_active_local_count++;
    }
    const size_t active_end = edge_count;
    two_layer_gate_edge_offsets[ii + 1] = active_end;

    if (use_static_fixed_gate_cache &&
        first_layer_active_local_count == 0 &&
        static_fixed_gate_value_cache_valid[i] &&
        static_fixed_cache_tag_matches(static_fixed_gate_cache_tags, i)) {
      const size_t gate_offset = static_cast<size_t>(i) * gate_signal_stride;
      std::copy(static_fixed_gate_value_cache + gate_offset,
                static_fixed_gate_value_cache + gate_offset + gate_signal_stride,
                two_layer_gate_values + gate_offset);
      continue;
    }

	    double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	    forward_two_layer_gate_products();
    if (use_full_gate_batched_contraction) {
      double *__restrict scalar_cache =
          two_layer_gate_full_scalar_cache.data() + static_cast<size_t>(ii);
      for (int q = 0; q < two_layer_gate_scalar_count; q++)
        scalar_cache[static_cast<size_t>(q) * inum] =
            moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
    }
	    if (store_gate_moment_cache) {
	      double *__restrict cache_row =
	          two_layer_gate_moment_cache.data() +
          static_cast<size_t>(ii) * gate_moment_cache_storage_width;
      if (compact_gate_moment_cache) {
        for (int c = 0; c < gate_moment_cache_storage_width; c++)
          cache_row[c] =
              moment_tensor_vals[two_layer_gate_required_moment_indices[c]];
      } else {
        std::copy(moment_tensor_vals,
                  moment_tensor_vals + gate_moment_cache_width,
                  cache_row);
      }
    }
	    double *gate_values_i =
	        two_layer_gate_values + static_cast<size_t>(i) * gate_signal_stride;
	    if (two_layer_gate_body_linear_combo) {
	      if (gate_body_stride > 0 && gate_body_stride <= kMaxBodyGateStrideFast) {
	        const std::vector<int> &bucket_offsets =
	            gate_scalars_basic_only ? basic_gate_body_bucket_offsets
	                                    : two_layer_gate_body_bucket_offsets;
	        const std::vector<int> &bucket_scalar_indices =
	            gate_scalars_basic_only ? basic_gate_body_bucket_scalar_indices
	                                    : two_layer_gate_body_bucket_scalar_indices;
	        const int output_count = compact_body_signal
	            ? gate_signal_stride : gate_body_stride;
	        for (int s = 0; s < output_count; s++) {
	          const int b = compact_body_signal
	              ? two_layer_gate_body_signal_buckets[s] : s;
	          double value = 0.0;
	          const int begin = bucket_offsets[b];
	          const int end = bucket_offsets[b + 1];
	          for (int pos = begin; pos < end; pos++) {
	            const int q = bucket_scalar_indices[pos];
	            value += two_layer_gate_weights[q] *
	                     moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
	          }
	          gate_values_i[s] = value;
	        }
	      } else {
        std::fill(gate_body_values.begin(), gate_body_values.end(), 0.0);
        if (gate_scalars_basic_only) {
          for (int q : basic_gate_scalar_indices) {
            const double scalar_value =
                moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
            gate_body_values[two_layer_gate_scalar_body_bucket[q]] +=
                two_layer_gate_weights[q] * scalar_value;
          }
        } else {
          for (int q = 0; q < two_layer_gate_scalar_count; q++) {
          const double scalar_value =
              moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
          gate_body_values[two_layer_gate_scalar_body_bucket[q]] +=
              two_layer_gate_weights[q] * scalar_value;
          }
        }
        std::copy(gate_body_values.begin(), gate_body_values.end(), gate_values_i);
      }
	    } else {
	      if (static_cast<int>(two_layer_gate_full_weights_by_scalar.size()) !=
	          two_layer_gate_scalar_count * gate_signal_stride)
	        error->all(FLERR, "SUS2-SH full mu/scalar gate weight layout is inconsistent.");
      if (use_full_gate_batched_contraction) {
        // Gate values are formed after all local scalar rows are available.
      } else if (gate_scalars_basic_only) {
	        std::fill(gate_values_i, gate_values_i + gate_signal_stride, 0.0);
	        for (int q : basic_gate_scalar_indices) {
	          const double scalar_value =
              moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
          const double *__restrict weights_by_mu =
              two_layer_gate_full_weights_by_scalar.data() +
              static_cast<size_t>(q) * gate_signal_stride;
          #pragma omp simd
          for (int g = 0; g < gate_signal_stride; g++)
            gate_values_i[g] += scalar_value * weights_by_mu[g];
        }
      } else if (!accumulate_full_gate_values_specialized(
              gate_signal_stride, two_layer_gate_scalar_count,
              two_layer_gate_full_weights_by_scalar.data(),
              two_layer_gate_scalar_moment_indices.data(),
              moment_tensor_vals, gate_values_i)) {
        std::fill(gate_values_i, gate_values_i + gate_signal_stride, 0.0);
        for (int q = 0; q < two_layer_gate_scalar_count; q++) {
          const double scalar_value =
              moment_tensor_vals[two_layer_gate_scalar_moment_indices[q]];
          const double *__restrict weights_by_mu =
              two_layer_gate_full_weights_by_scalar.data() +
              static_cast<size_t>(q) * gate_signal_stride;
          #pragma omp simd
          for (int g = 0; g < gate_signal_stride; g++)
            gate_values_i[g] += scalar_value * weights_by_mu[g];
	        }
	      }
    }
    if (edge_l1_active)
      fill_two_layer_gate_edge_l1_weighted_values_for_atom(i,
                                                           moment_tensor_vals);
    add_profile_time(profile_gate_v2,
                     profile_v2_times[kGateProfileV2FirstProductsSignal],
                     profile_v2_sub_start);
    if (cache_body_signal_derivs) {
      profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
      std::fill(gate_signal_moment_ders.begin(),
                gate_signal_moment_ders.end(), 0.0);
      const std::vector<int> &bucket_offsets =
          gate_scalars_basic_only ? basic_gate_body_bucket_offsets
                                  : two_layer_gate_body_bucket_offsets;
      const std::vector<int> &bucket_scalar_indices =
          gate_scalars_basic_only ? basic_gate_body_bucket_scalar_indices
                                  : two_layer_gate_body_bucket_scalar_indices;
      for (int b = 0; b < gate_body_stride; b++) {
        double *body_moment_ders =
            gate_signal_moment_ders.data() +
            static_cast<size_t>(b) * gate_moment_cache_width;
        const int begin = bucket_offsets[b];
        const int end = bucket_offsets[b + 1];
        for (int pos = begin; pos < end; pos++) {
          const int q = bucket_scalar_indices[pos];
          const double weight = two_layer_gate_weights[q];
          if (weight == 0.0) continue;
          body_moment_ders[two_layer_gate_scalar_moment_indices[q]] += weight;
        }
      }
      if (!two_layer_gate_product_out_indices.empty()) {
        const int *__restrict product_a0 =
            two_layer_gate_product_a0_indices.data();
        const int *__restrict product_a1 =
            two_layer_gate_product_a1_indices.data();
        const int *__restrict product_out =
            two_layer_gate_product_out_indices.data();
        const double *__restrict product_coeff =
            two_layer_gate_product_coeffs.data();
        for (int idx = static_cast<int>(two_layer_gate_product_out_indices.size()) - 1;
             idx >= 0; idx--) {
          const int a0 = product_a0[idx];
          const int a1 = product_a1[idx];
          const int out = product_out[idx];
          const double coeff = product_coeff[idx];
          const double moments_a0 = moment_tensor_vals[a0];
          const double moments_a1 = moment_tensor_vals[a1];
          for (int b = 0; b < gate_body_stride; b++) {
            double *body_moment_ders =
                gate_signal_moment_ders.data() +
                static_cast<size_t>(b) * gate_moment_cache_width;
            const double adjoint = body_moment_ders[out];
            body_moment_ders[a1] += adjoint * coeff * moments_a0;
            body_moment_ders[a0] += adjoint * coeff * moments_a1;
          }
        }
      } else {
        for (int k = gate_product_limit - 1; k >= 0; k--) {
          const int a0 = alpha_times_a0[k];
          const int a1 = alpha_times_a1[k];
          const int out = alpha_times_out[k];
          const double coeff = alpha_times_coeff[k];
          const double moments_a0 = moment_tensor_vals[a0];
          const double moments_a1 = moment_tensor_vals[a1];
          for (int b = 0; b < gate_body_stride; b++) {
            double *body_moment_ders =
                gate_signal_moment_ders.data() +
                static_cast<size_t>(b) * gate_moment_cache_width;
            const double adjoint = body_moment_ders[out];
            body_moment_ders[a1] += adjoint * coeff * moments_a0;
            body_moment_ders[a0] += adjoint * coeff * moments_a1;
          }
        }
      }
      for (size_t active_idx = active_begin; active_idx < active_end;
           active_idx++) {
        const size_t active_local = active_idx - active_begin;
        const size_t jac_offset =
            active_local * static_cast<size_t>(alpha_index_basic_count);
        const double *__restrict jac_x = moment_jacobian_x + jac_offset;
        const double *__restrict jac_y = moment_jacobian_y + jac_offset;
        const double *__restrict jac_z = moment_jacobian_z + jac_offset;
        double gx[kMaxBodyGateStrideFast] = {};
        double gy[kMaxBodyGateStrideFast] = {};
        double gz[kMaxBodyGateStrideFast] = {};
        for (int k = 0; k < alpha_index_basic_count; k++) {
          const double jx = jac_x[k];
          const double jy = jac_y[k];
          const double jz = jac_z[k];
          for (int b = 0; b < gate_body_stride; b++) {
            const double adjoint =
                gate_signal_moment_ders[
                    static_cast<size_t>(b) * gate_moment_cache_width + k];
            gx[b] += adjoint * jx;
            gy[b] += adjoint * jy;
            gz[b] += adjoint * jz;
          }
        }
        const size_t signal_offset =
            active_idx * static_cast<size_t>(gate_signal_stride);
        for (int b = 0; b < gate_body_stride; b++) {
          two_layer_gate_edge_signal_deriv_x_raw[signal_offset + b] = gx[b];
          two_layer_gate_edge_signal_deriv_y_raw[signal_offset + b] = gy[b];
          two_layer_gate_edge_signal_deriv_z_raw[signal_offset + b] = gz[b];
        }
      }
      add_profile_time(profile_gate_v2,
                       profile_v2_times[kGateProfileV2FirstSignalDerivCache],
                       profile_v2_sub_start);
    }
    if (use_static_fixed_gate_cache &&
        first_layer_active_local_count == 0) {
      const size_t gate_offset = static_cast<size_t>(i) * gate_signal_stride;
      std::copy(gate_values_i, gate_values_i + gate_signal_stride,
                static_fixed_gate_value_cache + gate_offset);
      static_fixed_gate_value_cache_valid[i] = 1;
      continue;
    }

	  }
  if (use_full_gate_batched_contraction) {
    double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
    accumulate_full_gate_values_batched(
        inum, ilist, two_layer_gate_scalar_count, gate_signal_stride,
        two_layer_gate_full_weights_by_scalar.data(),
        two_layer_gate_full_scalar_cache.data(), two_layer_gate_values);
    add_profile_time(profile_gate_v2,
                     profile_v2_times[kGateProfileV2FirstProductsSignal],
                     profile_v2_sub_start);
  }
	  if (profile_any_gate)
	    profile_first_layer = MPI_Wtime() - profile_stage_start;

  profile_stage_start = profile_any_gate ? MPI_Wtime() : 0.0;
  comm->forward_comm(this);
  if (!disable_two_layer_gate_tanh_cache() &&
      two_layer_gate_mu_cache_valid != nullptr) {
    std::fill(two_layer_gate_mu_cache_valid,
              two_layer_gate_mu_cache_valid + nall, 0);
  }
  if (profile_any_gate)
    profile_forward_comm = MPI_Wtime() - profile_stage_start;
  if (profile_gate_v2)
    profile_v2_times[kGateProfileV2ForwardComm] = profile_forward_comm;

  profile_stage_start = profile_any_gate ? MPI_Wtime() : 0.0;
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i] - 1;
    if (itype >= species_count)
      error->one(FLERR, "Too few species count in the MTP potential!");
    const size_t active_begin = two_layer_gate_edge_offsets[ii];
    const size_t active_end = two_layer_gate_edge_offsets[ii + 1];
    const int active_count = static_cast<int>(active_end - active_begin);
    ensure_two_layer_edge_buffer(active_count);
    std::fill(moment_tensor_vals, moment_tensor_vals + alpha_moment_count, 0.0);
    std::fill(nbh_energy_ders_wrt_moments,
              nbh_energy_ders_wrt_moments + alpha_moment_count, 0.0);
    const bool use_static_fixed_gate_main_cache =
        static_fixed_basic_cache_enabled && is_sh_model &&
        !eflag && !vflag && is_static_fixed_type(itype) &&
        !two_layer_gate_center_enabled &&
        !disable_two_layer_gate_tanh_cache() && !edge_l1_active;
    if (use_static_fixed_gate_main_cache) {
      ensure_static_fixed_gate_main_cache();
      if (!static_fixed_gate_main_cache_valid_for_center(i))
        build_static_fixed_gate_main_cache_for_center(i, itype, numneigh[i],
                                                      firstneigh[i], x[i]);
    }
    const double *center_gate_multiplier_by_mu = nullptr;
    const double *center_gate_deriv_by_mu = nullptr;
    if (two_layer_gate_center_enabled) {
      const double *center_gate_signal_by_mu =
          two_layer_gate_values + static_cast<size_t>(i) * gate_signal_stride;
      ensure_two_layer_gate_mu_cache_for_atom(itype, i, center_gate_signal_by_mu);
      center_gate_multiplier_by_mu =
          two_layer_gate_multiplier_mu_cache +
          static_cast<size_t>(i) * radial_func_count;
      center_gate_deriv_by_mu =
          two_layer_gate_deriv_mu_cache +
          static_cast<size_t>(i) * radial_func_count;
    }

	    for (size_t active_idx = active_begin; active_idx < active_end; active_idx++) {
	      const int active_local = static_cast<int>(active_idx - active_begin);
      const int j = two_layer_gate_edge_neighbors_raw[active_idx];
      const int jtype = two_layer_gate_edge_types_raw[active_idx];
      if (use_static_fixed_gate_main_cache && is_static_fixed_type(jtype))
        continue;
      const double r[3] = {two_layer_gate_edge_dx_raw[active_idx],
                           two_layer_gate_edge_dy_raw[active_idx],
                           two_layer_gate_edge_dz_raw[active_idx]};
      const double dist = two_layer_gate_edge_dist_raw[active_idx];
      const double *gate_signal_by_mu =
          two_layer_gate_values + static_cast<size_t>(j) * gate_signal_stride;
      const double *radial_gate_signal_by_mu = gate_signal_by_mu;
	      bool radial_gate_signal_is_full_mu = false;
	      if (edge_l1_active) {
	        double y1_values[3];
	        eval_real_sh_l1_values_compact(r, 1.0 / dist, y1_values);
	        const double *u_atom =
	            two_layer_gate_edge_l1_weighted_values +
	            static_cast<size_t>(j) * edge_l1_weighted_stride;
	        for (int mu = 0; mu < radial_func_count; mu++) {
	          const double *u = u_atom + 3 * mu;
	          edge_l1_signal_scratch[mu] =
	              two_layer_gate_mu_signal(gate_signal_by_mu, mu) +
	              u[0] * y1_values[0] + u[1] * y1_values[1] +
	              u[2] * y1_values[2];
	        }
	        radial_gate_signal_by_mu = edge_l1_signal_scratch.data();
	        radial_gate_signal_is_full_mu = true;
	      }
	      const int table_index = two_layer_gate_edge_table_indices_raw[active_idx];
	      const int table_bin = two_layer_gate_edge_table_bins_raw[active_idx];
	      const double table_frac = two_layer_gate_edge_table_fracs_raw[active_idx];
	      double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	      if (edge_l1_active ||
            !calc_gate_additive_table_radial_values(
	              jtype, gate_signal_by_mu, j, table_index, table_bin, table_frac)) {
	        calc_pair_radial_values(itype, jtype, dist, false,
                                  radial_gate_signal_by_mu, true,
                                  edge_l1_active ? -1 : j, table_index,
                                  table_bin, table_frac,
                                  radial_gate_signal_is_full_mu);
	      }
      if (two_layer_gate_center_enabled) {
        for (int mu = 0; mu < radial_func_count; mu++) {
          const double neighbor_scaled_val = radial_vals[mu];
          two_layer_gate_center_residual_radial_vals[mu] =
              neighbor_scaled_val * center_gate_deriv_by_mu[mu];
          const double center_multiplier = center_gate_multiplier_by_mu[mu];
	          radial_vals[mu] *= center_multiplier;
	          radial_ders[mu] *= center_multiplier;
	          two_layer_gate_residual_radial_vals[mu] *= center_multiplier;
	        }
	      }
	      add_profile_time(profile_gate_v2,
	                       profile_v2_times[kGateProfileV2MainGateRadial],
	                       profile_v2_sub_start);
	      if (use_compact_gate_raw) {
	        std::memcpy(two_layer_gate_edge_residual_radial_vals_raw +
	                        active_idx * static_cast<size_t>(radial_func_count),
                    two_layer_gate_residual_radial_vals,
                    static_cast<size_t>(radial_func_count) * sizeof(double));
      }
	      const int raw_offset = active_local * alpha_index_basic_count;
	      const size_t main_sh_base =
	          active_idx * static_cast<size_t>(sh_component_count);
	      profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	      accumulate_sh_basic_edge(
	          active_local, r, dist, 1.0, !use_compact_gate_raw, raw_offset,
	          true, two_layer_gate_center_enabled,
          nullptr, nullptr, nullptr,
          use_compact_gate_raw
	          ? two_layer_gate_edge_main_sh_values_raw + main_sh_base
	          : nullptr,
	          nullptr);
	      add_profile_time(profile_gate_v2,
	                       profile_v2_times[kGateProfileV2MainSHBasic],
	                       profile_v2_sub_start);
	    }
    if (use_static_fixed_gate_main_cache &&
        !apply_static_fixed_gate_main_cache_moments(i)) {
      build_static_fixed_gate_main_cache_for_center(i, itype, numneigh[i],
                                                    firstneigh[i], x[i]);
      if (!apply_static_fixed_gate_main_cache_moments(i))
        error->one(FLERR, "SUS2-SH static fixed gate main cache rebuild failed.");
    }

	    double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	    forward_sh_products();

    double nbh_energy = 0.0;
    if (eflag_atom || eflag_global) {
      nbh_energy = shift_coeffs[itype] + species_coeffs[itype];
      for (int k = 0; k < alpha_scalar_count; k++)
        nbh_energy += linear_coeffs[k] *
                      moment_tensor_vals[alpha_moment_mapping[k]] *
                      species_coeffs[itype];
      if (eflag_atom) eatom[i] += nbh_energy;
      if (eflag_global) eng_vdwl += nbh_energy;
    }

    for (int k = 0; k < alpha_scalar_count; k++)
      nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];
    backprop_sh_products();
    const double species_weight = species_coeffs[itype];
	    for (int k = 0; k < alpha_index_basic_count; k++)
	      weighted_basic_moment_ders[k] =
	          nbh_energy_ders_wrt_moments[k] * species_weight;
	    add_profile_time(profile_gate_v2,
	                     profile_v2_times[kGateProfileV2MainProductsBackprop],
	                     profile_v2_sub_start);

	    profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	    for (size_t active_idx = active_begin; active_idx < active_end; active_idx++) {
      const int active_local = static_cast<int>(active_idx - active_begin);
      const int j = two_layer_gate_edge_neighbors_raw[active_idx];
      if (use_static_fixed_gate_main_cache &&
          is_static_fixed_type(two_layer_gate_edge_types_raw[active_idx]))
        continue;
      if (!two_layer_gate_center_enabled) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double *gate_adjoints_by_mu =
            two_layer_gate_adjoints + static_cast<size_t>(j) * gate_signal_stride;
        const bool direct_gate_adjoint_accumulation =
            direct_body_combo_gate_adjoints || direct_full_gate_adjoints ||
            direct_grouped_full_gate_adjoints;
        double *__restrict gate_adjoint_by_mu =
            direct_gate_adjoint_accumulation ? nullptr : two_layer_gate_adjoint_scratch;
	        if (!direct_gate_adjoint_accumulation)
	          std::fill(gate_adjoint_by_mu,
	                    gate_adjoint_by_mu + radial_func_count, 0.0);
	        double edge_l1_scalar_gate_adjoint = 0.0;
	        const size_t jac_offset =
	            static_cast<size_t>(active_local) * alpha_index_basic_count;
        const double *__restrict jac_x = moment_jacobian_x + jac_offset;
        const double *__restrict jac_y = moment_jacobian_y + jac_offset;
        const double *__restrict jac_z = moment_jacobian_z + jac_offset;
        const double *__restrict raw =
            use_compact_gate_raw ? nullptr : two_layer_raw_basic_vals + jac_offset;
        const double *__restrict compact_residual_radial =
            use_compact_gate_raw
            ? two_layer_gate_edge_residual_radial_vals_raw +
                active_idx * static_cast<size_t>(radial_func_count)
            : nullptr;
        const double *__restrict compact_sh_values =
            use_compact_gate_raw
            ? two_layer_gate_edge_main_sh_values_raw +
                active_idx * static_cast<size_t>(sh_component_count)
            : nullptr;
        double gate_body_adjoints[kMaxBodyGateStrideFast] = {};
        if (sh_basic_mu_fast) {
          for (int mu = 0; mu < radial_func_count; mu++) {
            const int begin = sh_basic_mu_offsets[mu];
            const int end = sh_basic_mu_offsets[mu + 1];
            double gate_adjoint_mu = 0.0;
            const double compact_residual =
                use_compact_gate_raw ? compact_residual_radial[mu] : 0.0;
            for (int pos = begin; pos < end; pos++) {
              const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
              const double pref = weighted_basic_moment_ders[k];
              fx += pref * jac_x[k];
              fy += pref * jac_y[k];
              fz += pref * jac_z[k];
              const double raw_value = use_compact_gate_raw
                  ? compact_residual * compact_sh_values[alpha_basic_sh_index[k]]
                  : raw[k];
              gate_adjoint_mu += pref * raw_value;
            }
	            if (edge_l1_direct_scalar_gate_adjoint) {
	              gate_adjoint_by_mu[mu] += gate_adjoint_mu;
	              edge_l1_scalar_gate_adjoint += gate_adjoint_mu;
	            } else if (direct_body_combo_gate_adjoints) {
	              if (gate_adjoint_mu != 0.0) {
	                const double *weights =
	                    body_signal_mix_weights +
                    static_cast<size_t>(mu) * gate_signal_stride;
                accumulate_small_stride(gate_adjoint_mu, weights,
                                        gate_signal_stride, gate_body_adjoints);
              }
            } else if (direct_full_gate_adjoints) {
              gate_adjoints_by_mu[mu] += gate_adjoint_mu;
            } else if (direct_grouped_full_gate_adjoints) {
              gate_adjoints_by_mu[two_layer_gate_full_signal_group_for_mu[mu]] +=
                  gate_adjoint_mu;
            } else {
              gate_adjoint_by_mu[mu] += gate_adjoint_mu;
            }
          }
        } else {
          for (int k = 0; k < alpha_index_basic_count; k++) {
            const double pref = weighted_basic_moment_ders[k];
            fx += pref * jac_x[k];
            fy += pref * jac_y[k];
            fz += pref * jac_z[k];
	            const int mu = alpha_basic_mu[k];
	            if (edge_l1_direct_scalar_gate_adjoint) {
	              const double adjoint = pref * raw[k];
	              gate_adjoint_by_mu[mu] += adjoint;
	              edge_l1_scalar_gate_adjoint += adjoint;
	            } else if (direct_body_combo_gate_adjoints) {
	              const double adjoint = pref * raw[k];
	              const double *weights =
	                  body_signal_mix_weights +
                  static_cast<size_t>(mu) * gate_signal_stride;
              accumulate_small_stride(adjoint, weights, gate_signal_stride,
                                      gate_body_adjoints);
            } else {
              gate_adjoint_by_mu[mu] += pref * raw[k];
            }
          }
	        }
	        if (edge_l1_direct_scalar_gate_adjoint) {
	          gate_adjoints_by_mu[0] += edge_l1_scalar_gate_adjoint;
	        } else if (direct_body_combo_gate_adjoints) {
	          for (int b = 0; b < gate_signal_stride; b++)
	            gate_adjoints_by_mu[b] += gate_body_adjoints[b];
        } else if (!direct_full_gate_adjoints &&
                   !direct_grouped_full_gate_adjoints) {
          accumulate_two_layer_gate_signal_adjoints(gate_adjoints_by_mu,
                                                    gate_adjoint_by_mu);
	        }
	        if (edge_l1_active) {
	          const double r_edge[3] = {
	              two_layer_gate_edge_dx_raw[active_idx],
	              two_layer_gate_edge_dy_raw[active_idx],
	              two_layer_gate_edge_dz_raw[active_idx]};
	          const double dist_edge = two_layer_gate_edge_dist_raw[active_idx];
	          double y1_values[3];
	          double y1_ders[9];
	          eval_real_sh_l1_compact(r_edge, 1.0 / dist_edge, y1_values,
	                                  y1_ders);
	          const double *u_atom =
	              two_layer_gate_edge_l1_weighted_values +
	              static_cast<size_t>(j) * edge_l1_weighted_stride;
          double *u_adj_atom =
              two_layer_gate_edge_l1_weighted_adjoints +
              static_cast<size_t>(j) * edge_l1_weighted_stride;
          double y_adj[3] = {0.0, 0.0, 0.0};
          for (int mu = 0; mu < radial_func_count; mu++) {
            const double adj = gate_adjoint_by_mu[mu];
            if (adj == 0.0) continue;
            const double *u = u_atom + 3 * mu;
	            double *u_adj = u_adj_atom + 3 * mu;
	            for (int m = 0; m < 3; m++) {
	              u_adj[m] += adj * y1_values[m];
	              y_adj[m] += adj * u[m];
	            }
	          }
	          for (int m = 0; m < 3; m++) {
	            const int der_base = 3 * m;
	            fx += y_adj[m] * y1_ders[der_base + 0];
	            fy += y_adj[m] * y1_ders[der_base + 1];
	            fz += y_adj[m] * y1_ders[der_base + 2];
	          }
	        }

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;

        if (vflag) {
          const double r[3] = {two_layer_gate_edge_dx_raw[active_idx],
                               two_layer_gate_edge_dy_raw[active_idx],
                               two_layer_gate_edge_dz_raw[active_idx]};
          virial[0] -= fx * r[0];
          virial[1] -= fy * r[1];
          virial[2] -= fz * r[2];
          virial[3] -= fx * r[1];
          virial[4] -= fx * r[2];
          virial[5] -= fy * r[2];
          if (cvflag_atom) {
            cvatom[j][0] -= fx * r[0];
            cvatom[j][1] -= fy * r[1];
            cvatom[j][2] -= fz * r[2];
            cvatom[j][3] -= fx * r[1];
            cvatom[j][4] -= fx * r[2];
            cvatom[j][5] -= fy * r[2];
            cvatom[j][6] -= fy * r[0];
            cvatom[j][7] -= fz * r[0];
            cvatom[j][8] -= fz * r[1];
          }
        }
        continue;
      }
      double fx = 0.0, fy = 0.0, fz = 0.0;
      double *gate_adjoints_by_mu =
          two_layer_gate_adjoints + static_cast<size_t>(j) * gate_signal_stride;
      double *center_gate_adjoints_by_mu =
          two_layer_gate_center_enabled
          ? two_layer_gate_adjoints + static_cast<size_t>(i) * gate_signal_stride
          : nullptr;
      const bool direct_gate_adjoint_accumulation =
          direct_body_combo_gate_adjoints || direct_full_gate_adjoints ||
          direct_grouped_full_gate_adjoints;
      double *__restrict gate_adjoint_by_mu =
          direct_gate_adjoint_accumulation ? nullptr : two_layer_gate_adjoint_scratch;
      double *__restrict center_gate_adjoint_by_mu =
          two_layer_gate_center_adjoint_scratch;
      if (!direct_gate_adjoint_accumulation)
        std::fill(gate_adjoint_by_mu,
                  gate_adjoint_by_mu + radial_func_count, 0.0);
      if (two_layer_gate_center_enabled)
        std::fill(center_gate_adjoint_by_mu,
                  center_gate_adjoint_by_mu + radial_func_count, 0.0);
      const size_t jac_offset =
          static_cast<size_t>(active_local) * alpha_index_basic_count;
      const double *__restrict jac_x = moment_jacobian_x + jac_offset;
      const double *__restrict jac_y = moment_jacobian_y + jac_offset;
      const double *__restrict jac_z = moment_jacobian_z + jac_offset;
      const double *__restrict raw = two_layer_raw_basic_vals + jac_offset;
      const double *__restrict center_raw =
          two_layer_gate_center_enabled
          ? two_layer_center_raw_basic_vals + jac_offset
          : nullptr;
      double gate_body_adjoints[kMaxBodyGateStrideFast] = {};
      if (sh_basic_mu_fast) {
        for (int mu = 0; mu < radial_func_count; mu++) {
          const int begin = sh_basic_mu_offsets[mu];
          const int end = sh_basic_mu_offsets[mu + 1];
          double gate_adjoint_mu = 0.0;
          double center_gate_adjoint_mu = 0.0;
          for (int pos = begin; pos < end; pos++) {
            const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
            const double pref = weighted_basic_moment_ders[k];
            fx += pref * jac_x[k];
            fy += pref * jac_y[k];
            fz += pref * jac_z[k];
            gate_adjoint_mu += pref * raw[k];
            if (two_layer_gate_center_enabled)
              center_gate_adjoint_mu += pref * center_raw[k];
          }
          if (direct_body_combo_gate_adjoints) {
            if (gate_adjoint_mu != 0.0) {
              const double *weights =
                  body_signal_mix_weights +
                  static_cast<size_t>(mu) * gate_signal_stride;
              accumulate_small_stride(gate_adjoint_mu, weights,
                                      gate_signal_stride, gate_body_adjoints);
            }
          } else if (direct_full_gate_adjoints) {
            gate_adjoints_by_mu[mu] += gate_adjoint_mu;
          } else if (direct_grouped_full_gate_adjoints) {
            gate_adjoints_by_mu[two_layer_gate_full_signal_group_for_mu[mu]] +=
                gate_adjoint_mu;
          } else {
            gate_adjoint_by_mu[mu] += gate_adjoint_mu;
          }
          if (two_layer_gate_center_enabled)
            center_gate_adjoint_by_mu[mu] += center_gate_adjoint_mu;
        }
      } else {
        for (int k = 0; k < alpha_index_basic_count; k++) {
          const double pref = weighted_basic_moment_ders[k];
          fx += pref * jac_x[k];
          fy += pref * jac_y[k];
          fz += pref * jac_z[k];
          const int mu = alpha_basic_mu[k];
          if (direct_body_combo_gate_adjoints) {
            const double adjoint = pref * raw[k];
            const double *weights =
                body_signal_mix_weights +
                static_cast<size_t>(mu) * gate_signal_stride;
            accumulate_small_stride(adjoint, weights, gate_signal_stride,
                                    gate_body_adjoints);
          } else {
            gate_adjoint_by_mu[mu] += pref * raw[k];
          }
          if (two_layer_gate_center_enabled)
            center_gate_adjoint_by_mu[mu] += pref * center_raw[k];
        }
      }
      if (direct_body_combo_gate_adjoints) {
        for (int b = 0; b < gate_signal_stride; b++)
          gate_adjoints_by_mu[b] += gate_body_adjoints[b];
      } else if (!direct_full_gate_adjoints &&
                 !direct_grouped_full_gate_adjoints) {
        accumulate_two_layer_gate_signal_adjoints(gate_adjoints_by_mu,
                                                  gate_adjoint_by_mu);
      }
      if (two_layer_gate_center_enabled)
        accumulate_two_layer_gate_signal_adjoints(center_gate_adjoints_by_mu,
                                                  center_gate_adjoint_by_mu);

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      f[j][0] -= fx;
      f[j][1] -= fy;
      f[j][2] -= fz;

      if (vflag) {
        const double r[3] = {two_layer_gate_edge_dx_raw[active_idx],
                             two_layer_gate_edge_dy_raw[active_idx],
                             two_layer_gate_edge_dz_raw[active_idx]};
        virial[0] -= fx * r[0];
        virial[1] -= fy * r[1];
        virial[2] -= fz * r[2];
        virial[3] -= fx * r[1];
        virial[4] -= fx * r[2];
        virial[5] -= fy * r[2];
        if (cvflag_atom) {
          cvatom[j][0] -= fx * r[0];
          cvatom[j][1] -= fy * r[1];
          cvatom[j][2] -= fz * r[2];
          cvatom[j][3] -= fx * r[1];
          cvatom[j][4] -= fx * r[2];
          cvatom[j][5] -= fy * r[2];
          cvatom[j][6] -= fy * r[0];
          cvatom[j][7] -= fz * r[0];
          cvatom[j][8] -= fz * r[1];
        }
      }
	    }
	    add_profile_time(profile_gate_v2,
	                     profile_v2_times[kGateProfileV2MainForceAdjoint],
	                     profile_v2_sub_start);
	    if (use_static_fixed_gate_main_cache)
	      apply_static_fixed_gate_main_cache_adjoints(i);
	  }
	  if (profile_any_gate)
	    profile_main_layer = MPI_Wtime() - profile_stage_start;

	  profile_stage_start = profile_any_gate ? MPI_Wtime() : 0.0;
	  comm->reverse_comm(this);
	  if (profile_any_gate)
	    profile_reverse_comm = MPI_Wtime() - profile_stage_start;
		  if (profile_gate_v2)
		    profile_v2_times[kGateProfileV2ReverseComm] = profile_reverse_comm;

		  profile_stage_start = profile_any_gate ? MPI_Wtime() : 0.0;
  if (use_full_gate_batched_contraction) {
    double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
    accumulate_full_scalar_adjoints_batched(
        inum, ilist, two_layer_gate_scalar_count, gate_signal_stride,
        two_layer_gate_adjoints, two_layer_gate_full_weights_by_scalar.data(),
        two_layer_gate_full_scalar_adjoint_cache.data());
    add_profile_time(profile_gate_v2,
                     profile_v2_times[kGateProfileV2GateForceSeedBackprop],
                     profile_v2_sub_start);
  }
	  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i] - 1;
    if (itype >= species_count)
      error->one(FLERR, "Too few species count in the MTP potential!");
    const double *gate_adjoint_center_by_mu =
        two_layer_gate_adjoints + static_cast<size_t>(i) * gate_signal_stride;
    const double *edge_l1_adjoint_center =
        edge_l1_active
        ? two_layer_gate_edge_l1_weighted_adjoints +
              static_cast<size_t>(i) * edge_l1_weighted_stride
        : nullptr;
    bool has_gate_adjoint_center = false;
    const int adjoint_check_count = gate_signal_stride;
    for (int g = 0; g < adjoint_check_count; g++) {
      if (gate_adjoint_center_by_mu[g] != 0.0) {
        has_gate_adjoint_center = true;
        break;
      }
    }
    if (!has_gate_adjoint_center && edge_l1_active) {
      for (int g = 0; g < edge_l1_weighted_stride; g++) {
        if (edge_l1_adjoint_center[g] != 0.0) {
          has_gate_adjoint_center = true;
          break;
        }
      }
    }
    if (!has_gate_adjoint_center) continue;
	    const size_t active_begin = two_layer_gate_edge_offsets[ii];
	    const size_t active_end = two_layer_gate_edge_offsets[ii + 1];
	    if (cache_body_signal_derivs) {
	      double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	      for (size_t active_idx = active_begin; active_idx < active_end;
	           active_idx++) {
        const int j = two_layer_gate_edge_neighbors_raw[active_idx];
        const size_t signal_base =
            active_idx * static_cast<size_t>(gate_signal_stride);
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        #pragma omp simd reduction(+:fx,fy,fz)
        for (int b = 0; b < gate_body_stride; b++) {
          const double adjoint = gate_adjoint_center_by_mu[b];
          fx += adjoint * two_layer_gate_edge_signal_deriv_x_raw[signal_base + b];
          fy += adjoint * two_layer_gate_edge_signal_deriv_y_raw[signal_base + b];
          fz += adjoint * two_layer_gate_edge_signal_deriv_z_raw[signal_base + b];
        }

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;

        if (vflag) {
          const double r[3] = {two_layer_gate_edge_dx_raw[active_idx],
                               two_layer_gate_edge_dy_raw[active_idx],
                               two_layer_gate_edge_dz_raw[active_idx]};
          virial[0] -= fx * r[0];
          virial[1] -= fy * r[1];
          virial[2] -= fz * r[2];
          virial[3] -= fx * r[1];
          virial[4] -= fx * r[2];
          virial[5] -= fy * r[2];
          if (cvflag_atom) {
            cvatom[j][0] -= fx * r[0];
            cvatom[j][1] -= fy * r[1];
            cvatom[j][2] -= fz * r[2];
            cvatom[j][3] -= fx * r[1];
            cvatom[j][4] -= fx * r[2];
            cvatom[j][5] -= fy * r[2];
            cvatom[j][6] -= fy * r[0];
            cvatom[j][7] -= fz * r[0];
            cvatom[j][8] -= fz * r[1];
	          }
	        }
	      }
	      add_profile_time(profile_gate_v2,
	                       profile_v2_times[kGateProfileV2GateForceJacobianDot],
	                       profile_v2_sub_start);
	      continue;
	    }
	    double profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	    const double *gate_moments = store_gate_moment_cache
	        ? two_layer_gate_moment_cache.data() +
	              static_cast<size_t>(ii) * gate_moment_cache_storage_width
	        : nullptr;
	    if (compact_gate_moment_cache) {
	      std::fill(nbh_energy_ders_wrt_moments,
	                nbh_energy_ders_wrt_moments + alpha_index_basic_count, 0.0);
	      for (int moment_index : two_layer_gate_required_moment_indices)
	        if (moment_index >= alpha_index_basic_count)
	          nbh_energy_ders_wrt_moments[moment_index] = 0.0;
	    } else {
	      std::fill(nbh_energy_ders_wrt_moments,
	                nbh_energy_ders_wrt_moments + gate_force_adjoint_fill_width, 0.0);
	    }
	    if (two_layer_gate_body_linear_combo) {
	      const std::vector<int> &bucket_offsets =
	          gate_scalars_basic_only ? basic_gate_body_bucket_offsets
	                                  : two_layer_gate_body_bucket_offsets;
	      const std::vector<int> &bucket_scalar_indices =
	          gate_scalars_basic_only ? basic_gate_body_bucket_scalar_indices
	                                  : two_layer_gate_body_bucket_scalar_indices;
	      const int signal_count = compact_body_signal
	          ? gate_signal_stride : gate_body_stride;
	      for (int s = 0; s < signal_count; s++) {
	        const int b = compact_body_signal
	            ? two_layer_gate_body_signal_buckets[s] : s;
	        const double body_adjoint = gate_adjoint_center_by_mu[s];
	        if (body_adjoint == 0.0) continue;
	        const int begin = bucket_offsets[b];
	        const int end = bucket_offsets[b + 1];
	        for (int pos = begin; pos < end; pos++) {
	          const int q = bucket_scalar_indices[pos];
	          const double adjoint = body_adjoint * two_layer_gate_weights[q];
	          if (adjoint == 0.0) continue;
	          nbh_energy_ders_wrt_moments[two_layer_gate_scalar_moment_indices[q]] +=
	              adjoint;
	        }
	      }
	    } else {
	      if (static_cast<int>(two_layer_gate_full_weights_by_scalar.size()) !=
	          two_layer_gate_scalar_count * gate_signal_stride)
	        error->all(FLERR, "SUS2-SH full mu/scalar gate weight layout is inconsistent.");
      if (use_full_gate_batched_contraction) {
        const double *__restrict scalar_adjoints =
            two_layer_gate_full_scalar_adjoint_cache.data() +
            static_cast<size_t>(ii);
        for (int q = 0; q < two_layer_gate_scalar_count; q++) {
          const double adjoint = scalar_adjoints[static_cast<size_t>(q) * inum];
          if (adjoint == 0.0) continue;
          nbh_energy_ders_wrt_moments[two_layer_gate_scalar_moment_indices[q]] +=
              adjoint;
        }
      } else if (gate_scalars_basic_only) {
	        for (int q : basic_gate_scalar_indices) {
	          const double *__restrict weights_by_mu =
	              two_layer_gate_full_weights_by_scalar.data() +
              static_cast<size_t>(q) * gate_signal_stride;
          double adjoint = 0.0;
          #pragma omp simd reduction(+:adjoint)
          for (int g = 0; g < gate_signal_stride; g++)
            adjoint += gate_adjoint_center_by_mu[g] * weights_by_mu[g];
          if (adjoint == 0.0) continue;
          nbh_energy_ders_wrt_moments[two_layer_gate_scalar_moment_indices[q]] +=
              adjoint;
        }
      } else if (!accumulate_full_scalar_adjoints_specialized(
              gate_signal_stride, two_layer_gate_scalar_count,
              gate_adjoint_center_by_mu,
              two_layer_gate_full_weights_by_scalar.data(),
              two_layer_gate_scalar_moment_indices.data(),
              nbh_energy_ders_wrt_moments)) {
        for (int q = 0; q < two_layer_gate_scalar_count; q++) {
          const double *__restrict weights_by_mu =
              two_layer_gate_full_weights_by_scalar.data() +
              static_cast<size_t>(q) * gate_signal_stride;
          double adjoint = 0.0;
          #pragma omp simd reduction(+:adjoint)
          for (int g = 0; g < gate_signal_stride; g++)
            adjoint += gate_adjoint_center_by_mu[g] * weights_by_mu[g];
          if (adjoint == 0.0) continue;
          nbh_energy_ders_wrt_moments[two_layer_gate_scalar_moment_indices[q]] +=
              adjoint;
	        }
	      }
	    }
      if (edge_l1_active)
        seed_two_layer_gate_edge_l1_moment_adjoints(i,
                                                     nbh_energy_ders_wrt_moments);
	    if (gate_needs_product_backprop) {
	      if (compact_gate_moment_cache)
	        backprop_two_layer_gate_products_compact(gate_moments);
	      else
	        backprop_two_layer_gate_products(gate_moments);
	    }
	    add_profile_time(profile_gate_v2,
	                     profile_v2_times[kGateProfileV2GateForceSeedBackprop],
	                     profile_v2_sub_start);

	    profile_v2_sub_start = profile_gate_v2 ? MPI_Wtime() : 0.0;
	    for (size_t active_idx = active_begin; active_idx < active_end; active_idx++) {
      const int j = two_layer_gate_edge_neighbors_raw[active_idx];
      const int jtype = two_layer_gate_edge_types_raw[active_idx];
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      const double r[3] = {two_layer_gate_edge_dx_raw[active_idx],
                           two_layer_gate_edge_dy_raw[active_idx],
                           two_layer_gate_edge_dz_raw[active_idx]};
      if (recompute_gate_force_jacobians) {
        const size_t sh_cache_base =
            active_idx * static_cast<size_t>(sh_component_count);
        dot_sh_basic_edge_jacobian(
            itype, jtype, r, two_layer_gate_edge_dist_raw[active_idx],
            two_layer_gate_edge_table_indices_raw[active_idx],
            two_layer_gate_edge_table_bins_raw[active_idx],
            two_layer_gate_edge_table_fracs_raw[active_idx],
            nbh_energy_ders_wrt_moments, use_gate_required_dot, fx, fy, fz,
            cache_gate_force_radials
            ? two_layer_gate_edge_radial_vals_raw +
                active_idx * static_cast<size_t>(radial_func_count)
            : nullptr,
            cache_gate_force_radials
            ? two_layer_gate_edge_radial_ders_raw +
                active_idx * static_cast<size_t>(radial_func_count)
            : nullptr,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_values_raw + sh_cache_base
            : nullptr,
            cache_gate_force_sh_derivs
            ? two_layer_gate_edge_sh_ders_raw + 3 * sh_cache_base
            : nullptr);
      } else {
        const size_t deriv_base =
            active_idx * static_cast<size_t>(alpha_index_basic_count);
        const double *__restrict jac_x =
            two_layer_gate_edge_deriv_x_raw + deriv_base;
        const double *__restrict jac_y =
            two_layer_gate_edge_deriv_y_raw + deriv_base;
        const double *__restrict jac_z =
            two_layer_gate_edge_deriv_z_raw + deriv_base;
        #pragma omp simd reduction(+:fx,fy,fz)
        for (int k = 0; k < alpha_index_basic_count; k++) {
          const double adjoint = nbh_energy_ders_wrt_moments[k];
          fx += adjoint * jac_x[k];
          fy += adjoint * jac_y[k];
          fz += adjoint * jac_z[k];
        }
      }

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      f[j][0] -= fx;
      f[j][1] -= fy;
      f[j][2] -= fz;

      if (vflag) {
        virial[0] -= fx * r[0];
        virial[1] -= fy * r[1];
        virial[2] -= fz * r[2];
        virial[3] -= fx * r[1];
        virial[4] -= fx * r[2];
        virial[5] -= fy * r[2];
        if (cvflag_atom) {
          cvatom[j][0] -= fx * r[0];
          cvatom[j][1] -= fy * r[1];
          cvatom[j][2] -= fz * r[2];
          cvatom[j][3] -= fx * r[1];
          cvatom[j][4] -= fx * r[2];
          cvatom[j][5] -= fy * r[2];
          cvatom[j][6] -= fy * r[0];
          cvatom[j][7] -= fz * r[0];
          cvatom[j][8] -= fz * r[1];
	        }
	      }
	    }
	    add_profile_time(profile_gate_v2,
	                     profile_v2_times[kGateProfileV2GateForceJacobianDot],
	                     profile_v2_sub_start);
	  }
  if (profile_any_gate)
    profile_gate_force = MPI_Wtime() - profile_stage_start;

  if (profile_gate) {
    const double profile_total = MPI_Wtime() - profile_t0;
    const double local_times[6] = {profile_total, profile_first_layer,
                                   profile_forward_comm, profile_main_layer,
                                   profile_reverse_comm, profile_gate_force};
    double max_times[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, world);
    if (comm->me == 0 && screen) {
      fprintf(screen,
              "SUS2_SH_GATE_PROFILE total=%g first=%g fcomm=%g main=%g rcomm=%g gate_force=%g\n",
              max_times[0], max_times[1], max_times[2], max_times[3],
              max_times[4], max_times[5]);
    }
  }

  if (profile_gate_v2) {
    const double profile_total = MPI_Wtime() - profile_t0;
    const double local_summary[6] = {profile_total, profile_first_layer,
                                     profile_forward_comm, profile_main_layer,
                                     profile_reverse_comm, profile_gate_force};
    double max_summary[6] = {};
    double sum_summary[6] = {};
    double max_times[kGateProfileV2TimerCount] = {};
    double sum_times[kGateProfileV2TimerCount] = {};
    MPI_Reduce(local_summary, max_summary, 6, MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(local_summary, sum_summary, 6, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(profile_v2_times, max_times, kGateProfileV2TimerCount,
               MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(profile_v2_times, sum_times, kGateProfileV2TimerCount,
               MPI_DOUBLE, MPI_SUM, 0, world);

    const double edge_metadata_bytes =
        static_cast<double>(edge_capacity) *
        (4.0 * sizeof(int) + 5.0 * sizeof(double));
    const double radial_cache_bytes =
        static_cast<double>(radial_cache_capacity) * 2.0 * sizeof(double);
    const double sh_cache_bytes =
        static_cast<double>(sh_cache_capacity) * 4.0 * sizeof(double);
    const double compact_raw_bytes =
        static_cast<double>(main_sh_cache_capacity + residual_radial_cache_capacity) *
        sizeof(double);
    const double jacobian_cache_bytes =
        static_cast<double>(deriv_capacity) * 3.0 * sizeof(double);
    const double signal_deriv_bytes =
        static_cast<double>(signal_deriv_capacity) * 3.0 * sizeof(double);
    const double moment_cache_bytes =
        store_gate_moment_cache
        ? static_cast<double>(inum) * gate_moment_cache_storage_width *
              sizeof(double)
        : 0.0;
    const double atom_gate_bytes =
        static_cast<double>(nall) *
        (2.0 * gate_signal_stride + 2.0 * radial_func_count) * sizeof(double);
    const double local_counters[kGateProfileV2CounterCount] = {
        static_cast<double>(atom->nlocal),
        static_cast<double>(atom->nghost),
        static_cast<double>(nall),
        static_cast<double>(inum),
        static_cast<double>(edge_capacity),
        static_cast<double>(edge_count),
        static_cast<double>(radial_func_count),
        static_cast<double>(alpha_index_basic_count),
        static_cast<double>(alpha_index_times_count),
        static_cast<double>(alpha_scalar_count),
        static_cast<double>(gate_product_limit),
        static_cast<double>(gate_moment_cache_width),
        static_cast<double>(gate_signal_stride),
        static_cast<double>(two_layer_gate_scalar_count),
        static_cast<double>(two_layer_gate_weight_count),
        static_cast<double>(gate_body_stride),
        static_cast<double>(sh_component_count),
        two_layer_gate_body_linear_combo ? 1.0 : 0.0,
        two_layer_gate_full_scalar_weights ? 1.0 : 0.0,
        two_layer_gate_center_enabled ? 1.0 : 0.0,
        two_layer_gate_shared_radial ? 1.0 : 0.0,
        recompute_gate_force_jacobians ? 1.0 : 0.0,
        cache_full_gate_force_jacobians ? 1.0 : 0.0,
        cache_body_signal_derivs ? 1.0 : 0.0,
        use_compact_gate_raw ? 1.0 : 0.0,
        sh_basic_mu_fast ? 1.0 : 0.0,
        use_gate_required_basic_subset ? 1.0 : 0.0,
        use_gate_required_dot ? 1.0 : 0.0,
        two_layer_gate_required_basics_are_y00 ? 1.0 : 0.0,
        edge_metadata_bytes,
        radial_cache_bytes,
        sh_cache_bytes,
        compact_raw_bytes,
        jacobian_cache_bytes,
        signal_deriv_bytes,
        moment_cache_bytes,
        atom_gate_bytes};
    double max_counters[kGateProfileV2CounterCount] = {};
    double sum_counters[kGateProfileV2CounterCount] = {};
    MPI_Reduce(local_counters, max_counters, kGateProfileV2CounterCount,
               MPI_DOUBLE, MPI_MAX, 0, world);
    MPI_Reduce(local_counters, sum_counters, kGateProfileV2CounterCount,
               MPI_DOUBLE, MPI_SUM, 0, world);

    int world_size = 1;
    MPI_Comm_size(world, &world_size);
    const double inv_world_size = world_size > 0 ? 1.0 / world_size : 1.0;
    if (comm->me == 0 && screen) {
      fprintf(screen,
              "SUS2_SH_GATE_PROFILE_V2_SUMMARY total_max=%g total_avg=%g first_max=%g first_avg=%g fcomm_max=%g fcomm_avg=%g main_max=%g main_avg=%g rcomm_max=%g rcomm_avg=%g gate_force_max=%g gate_force_avg=%g\n",
              max_summary[0], sum_summary[0] * inv_world_size,
              max_summary[1], sum_summary[1] * inv_world_size,
              max_summary[2], sum_summary[2] * inv_world_size,
              max_summary[3], sum_summary[3] * inv_world_size,
              max_summary[4], sum_summary[4] * inv_world_size,
              max_summary[5], sum_summary[5] * inv_world_size);
      fprintf(screen, "SUS2_SH_GATE_PROFILE_V2_TIMES");
      for (int i = 0; i < kGateProfileV2TimerCount; i++) {
        fprintf(screen, " %s_max=%g %s_avg=%g",
                kGateProfileV2TimerNames[i], max_times[i],
                kGateProfileV2TimerNames[i], sum_times[i] * inv_world_size);
      }
      fprintf(screen, "\n");
      fprintf(screen, "SUS2_SH_GATE_PROFILE_V2_COUNTERS");
      for (int i = 0; i < kGateProfileV2CounterCount; i++) {
        fprintf(screen, " %s_sum=%g %s_max=%g",
                kGateProfileV2CounterNames[i], sum_counters[i],
                kGateProfileV2CounterNames[i], max_counters[i]);
      }
      fprintf(screen, "\n");
    }
  }
}

void PairSUS2MTP::accumulate_zbl_pair(int i,
                                      int j,
                                      int itype,
                                      int jtype,
                                      const double *r,
                                      double rsq,
                                      int eflag,
                                      int vflag)
{
  if (!zbl_enabled || rsq <= 0.0) return;
  const int C = species_count;
  if (itype < 0 || itype >= C)
    error->one(FLERR, "ZBL atom type is outside the SUS2 model species map.");
  if (jtype < 0 || jtype >= C)
    error->one(FLERR, "ZBL neighbor type is outside the SUS2 model species map.");
  const int pair_index = itype * C + jtype;
  if (rsq >= zbl_pair_outer_sq[pair_index]) return;

  const double dist = std::sqrt(rsq);
  const SUS2MTPZBLPairValue pair =
      sus2_mtp_zbl::ComputePairHostCached(
          zbl_pair_constants[pair_index], dist,
          zbl_pair_inner_cutoffs[pair_index],
          zbl_pair_outer_cutoffs[pair_index]);
  if (pair.energy == 0.0 && pair.dEdr == 0.0) return;

  double **f = atom->f;
  const double pref = 0.5 * pair.dEdr / dist;
  const double fx = pref * r[0];
  const double fy = pref * r[1];
  const double fz = pref * r[2];
  f[i][0] += fx;
  f[i][1] += fy;
  f[i][2] += fz;
  f[j][0] -= fx;
  f[j][1] -= fy;
  f[j][2] -= fz;

  if (eflag_global) eng_vdwl += 0.5 * pair.energy;
  if (eflag_atom) eatom[i] += 0.5 * pair.energy;
  if (vflag) {
    virial[0] -= fx * r[0];
    virial[1] -= fy * r[1];
    virial[2] -= fz * r[2];
    virial[3] -= fx * r[1];
    virial[4] -= fx * r[2];
    virial[5] -= fy * r[2];
    if (cvflag_atom) {
      cvatom[j][0] -= fx * r[0];
      cvatom[j][1] -= fy * r[1];
      cvatom[j][2] -= fz * r[2];
      cvatom[j][3] -= fx * r[1];
      cvatom[j][4] -= fx * r[2];
      cvatom[j][5] -= fy * r[2];
      cvatom[j][6] -= fy * r[0];
      cvatom[j][7] -= fz * r[0];
      cvatom[j][8] -= fz * r[1];
    }
  }
}

void PairSUS2MTP::compute_zbl(int eflag, int vflag)
{
  if (!zbl_enabled) return;
  double **x = atom->x;
  int *type = atom->type;
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i] - 1;
    const double xi[3] = {x[i][0], x[i][1], x[i][2]};
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];
      j &= NEIGHMASK;
      const int jtype = type[j] - 1;
      const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      if (rsq < zbl_max_outer_sq)
        accumulate_zbl_pair(i, j, itype, jtype, r, rsq, eflag, vflag);
    }
  }
}

/* ----------------------------------------------------------------------
   Straightfoward SUS2-MTP implementation based on SUS2-MLIP-1.1
   ---------------------------------------------------------------------- */

void PairSUS2MTP::compute(int eflag, int vflag)
{
  ev_setup(eflag, vflag);

  // CRITICAL: `alpha_index_basic_count` must be valid. Working buffers
  // (`moment_jacobian`, `within_cutoff`, etc.) are lazily allocated below
  // when a neighbourhood is encountered (via memory->grow). Do not treat
  // unallocated buffers as an immediate fatal error here because in MPI
  // runs some ranks may not have allocated them yet.
  if (alpha_index_basic_count <= 0) {
    error->one(FLERR, "alpha_index_basic_count is invalid: {}. This indicates the MTP file was not read properly.", alpha_index_basic_count);
  }

  if (requires_two_layer_gate_sh()) {
    compute_two_layer_gate_sh(eflag, vflag);
    return;
  }

  double **x = atom->x;      // atomic positons
  double **f = atom->f;      // atomic forces
  int *type = atom->type;    //atomic types

  int inum = list->inum;             // The number of central atoms (neigbhourhoods)
  int *ilist = list->ilist;          // List of central atom ids
  int *numneigh = list->numneigh;    // List of the number of neighbours for each central atom
  int **firstneigh =
      list->firstneigh;    //List  (head of array) of neighbours for a given central atom

  // Local constants for coefficient indexing
  const int C = species_count;
  const int R = radial_basis_size;
  const int pairs_count = C * C;
  const int K_ = K_scaling;
  const bool trace_first_site =
      is_sh_model && env_flag_enabled("SUS2_SH_TRACE_FIRST_SITE") &&
      comm->me == 0;


  // Loop over all provided neighbourhoods
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];          // Set central atom index
    const int itype = type[i] - 1;    // Set central atom type. Convert back to zero indexing.
    if (itype >= species_count)
      error->one(FLERR,
                 "Too few species count in the MTP potential!");
    
    int jnum = numneigh[i];                                         // Set number of neighbours
    double nbh_energy = 0;
    const double xi[3] = {x[i][0], x[i][1],
                          x[i][2]};    // Cache the position of the central atom for efficiency

    // Grow jacobian working arrays if needed
    if (jac_size < jnum) {
      memory->grow(moment_jacobian_x, jnum * alpha_index_basic_count,
                   "moment_jacobian_x");
      memory->grow(moment_jacobian_y, jnum * alpha_index_basic_count,
                   "moment_jacobian_y");
      memory->grow(moment_jacobian_z, jnum * alpha_index_basic_count,
                   "moment_jacobian_z");
      memory->grow(within_cutoff, jnum, "within_cutoff");    // Resize within cuf
      memory->grow(env_rho_dr, jnum, "env_rho_dr");
      jac_size = jnum;
    }
	    std::fill(&moment_tensor_vals[0], &moment_tensor_vals[0] + alpha_moment_count,
	              0.0);    //Fill moments with 0
	    std::fill(&nbh_energy_ders_wrt_moments[0], &nbh_energy_ders_wrt_moments[0] + alpha_moment_count,
	              0.0);    //Fill moment derivatives with 0
	    const bool use_static_fixed_basic_cache =
	        static_fixed_basic_cache_enabled && is_sh_model && !env_gate_enabled &&
	        !eflag && !vflag && is_static_fixed_type(itype);
	    if (use_static_fixed_basic_cache) {
	      ensure_static_fixed_basic_cache();
	      if (static_fixed_basic_cache_valid[i] &&
	          static_fixed_cache_tag_matches(static_fixed_basic_cache_tags, i)) {
	        const double *cache = static_fixed_basic_cache +
	            static_cast<size_t>(i) * alpha_index_basic_count;
	        std::copy(cache, cache + alpha_index_basic_count, moment_tensor_vals);
	      } else {
	        build_static_fixed_basic_cache_for_center(i, itype, jnum,
	                                                  firstneigh[i], xi);
	      }
	    }

    double env_screen_strength = 0.0;
    double env_rho_factor = 0.0;
    const double r_env = env_gate_enabled ? env_gate_cutoff_ratio * max_cutoff : 0.0;
    const double r_env_sq = r_env * r_env;
    double env_density_coeffs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (env_gate_enabled) {
      if (r_env <= 0.0) error->one(FLERR, "Invalid env_gate_cutoff_ratio in SUS2 model.");
      if (env_activation_basic_size < alpha_index_basic_count) {
        memory->grow(env_activation_basic_vals, alpha_index_basic_count,
                     "env_activation_basic_vals");
        env_activation_basic_size = alpha_index_basic_count;
      }
      if (jnum > 0) std::fill(env_rho_dr, env_rho_dr + jnum, 0.0);
      std::fill(env_activation_basic_vals,
                env_activation_basic_vals + alpha_index_basic_count, 0.0);
      for (int q = 0; q < env_gate_channel_count; ++q)
        env_density_coeffs[q] =
            env_gate_density_coeff(regression_coeffs[env_gate_log_density_coeffs_offset + itype * env_gate_channel_count + q]);

      double rho = 0.0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = firstneigh[i][jj];
        j &= NEIGHMASK;
        const int jtype = type[j] - 1;
        if (jtype >= species_count)
          error->one(FLERR, "Too few species count in the MTP potential!");
        const double rvec[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
        const double rsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
        if (rsq <= 0.0 || rsq >= r_env_sq) continue;
        const double dist = std::sqrt(rsq);
        double local_rho = 0.0;
        double local_rho_dr = 0.0;
        if (do_list && env_gate_rho_list) {
          const double scaled_dist = dist * inv_dr;
          int r_list = static_cast<int>(scaled_dist);
          const int last_interval = list_grid_size - 2;
          if (r_list < 0) r_list = 0;
          if (r_list > last_interval) r_list = last_interval;
          const int r_next = r_list + 1;
          double ddr = scaled_dist - r_list;
          if (ddr < 0.0) ddr = 0.0;
          if (ddr > 1.0) ddr = 1.0;
          const double v1 = env_gate_rho_list[itype][r_list];
          const double v2 = env_gate_rho_list[itype][r_next];
          const double d1 = env_gate_rho_der_list[itype][r_list];
          const double d2 = env_gate_rho_der_list[itype][r_next];
          local_rho = v1 + ddr * (v2 - v1);
          local_rho_dr = d1 + ddr * (d2 - d1);
        } else {
          const double y = dist / r_env;
          const double dy_dr = 1.0 / r_env;
          const double cutoff = 1.0 - dist / r_env;
          const double f_env = cutoff * cutoff;
          const double df_dr = -2.0 * cutoff / r_env;
          double basis[6];
          double basis_der[6];
          bernstein_degree5(y, basis, basis_der);
          for (int q = 0; q < env_gate_channel_count; ++q) {
            const double weighted_basis = f_env * basis[q];
            const double weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
            local_rho += env_density_coeffs[q] * weighted_basis;
            local_rho_dr += env_density_coeffs[q] * weighted_basis_der;
          }
        }
        rho += local_rho;
        env_rho_dr[jj] = local_rho_dr;
      }
      const double lambda = stable_sigmoid(regression_coeffs[env_gate_lambda_raw_offset + itype]);
      const double tanh_rho = std::tanh(rho);
      const double sech2_rho = 1.0 - tanh_rho * tanh_rho;
      env_screen_strength = lambda * tanh_rho;
      env_rho_factor = -lambda * sech2_rho;
    }

    // ------------ Begin Alpha Basic Calc ------------
    // Loop over all neighbours
    int active_dynamic_edge_count = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];    //List of neighbours
      j &= NEIGHMASK;
	      const int jtype = type[j] - 1;    // Convert back to zero indexing
	      if (jtype >= species_count)
	        error->one(FLERR,
	                   "Too few species count in the MTP potential!");
	      if (use_static_fixed_basic_cache && is_static_fixed_type(jtype)) {
	        within_cutoff[jj] = false;
	        continue;
	      }

	      const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      if (zbl_enabled && rsq < zbl_max_outer_sq)
        accumulate_zbl_pair(i, j, itype, jtype, r, rsq, eflag, vflag);

      if (rsq > max_cutoff_sq) {
        within_cutoff[jj] = false;
        continue;
      }
      within_cutoff[jj] = true;
      active_dynamic_edge_count++;

      const double dist = std::sqrt(rsq);
      const double inv_dist = 1.0 / dist;
      double env_activation = 0.0;
      double env_activation_der = 0.0;
      double pair_gate = 1.0;
      if (env_gate_enabled) {
        env_gate_activation(dist, r_env, env_gate_activation_on_ratio,
                            &env_activation, &env_activation_der);
        pair_gate = 1.0 - env_screen_strength * env_activation;
      }
      
      // SUS2-MLIP: Precompute radial basis function values for all k_ values
      // Use local temporary array as in reference implementation
      // This prevents issues with array indexing and memory overlap

      bool used_precomputed_table = false;
      if (do_list) {
        const int shift = C * itype + jtype;
        const int table_index = pair_to_table_index[shift];
        if (table_index >= 0) {
          const double scaled_dist = dist * inv_dr;
          int r_list = static_cast<int>(scaled_dist);
          const int last_interval = list_grid_size - 2;
          if (r_list < 0) r_list = 0;
          if (r_list > last_interval) r_list = last_interval;
          int r_next = r_list + 1;
          double ddr = scaled_dist - r_list;
          if (ddr < 0.0) ddr = 0.0;
          if (ddr > 1.0) ddr = 1.0;
          const double *__restrict radial_row = radial_list[table_index][r_list];
          const double *__restrict radial_next_row = radial_list[table_index][r_next];
          const double *__restrict deriv_row = radial_der_list[table_index][r_list];
          const double *__restrict deriv_next_row = radial_der_list[table_index][r_next];
          double *__restrict vals = radial_vals;
          double *__restrict ders = radial_ders;
          #pragma omp simd
          for (int m = 0; m < radial_func_count; m++) {
            const double v1 = radial_row[m];
            const double v2 = radial_next_row[m];
            const double d1 = deriv_row[m];
            const double d2 = deriv_next_row[m];
            vals[m] = v1 + ddr * (v2 - v1);
            ders[m] = d1 + ddr * (d2 - d1);
          }
          used_precomputed_table = true;
        }
      }

      if (!used_precomputed_table) {
      const bool per_mu_sigma = basis_requires_per_mu_sigma(radial_basis_type_index);
      const int radial_cache_count = per_mu_sigma ? radial_func_count : K_;
      int temp_size = radial_cache_count * R;
      std::vector<double> val_temp(temp_size, 0.0);
      std::vector<double> der_temp(temp_size, 0.0);

      if (per_mu_sigma) {
        for (int mu = 0; mu < radial_func_count; mu++) {
          const int k_ = mu_to_K[mu];
          const int sigma = mu_to_sigma[mu];
          double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
          double s = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];

          radial_basis->calc_radial_basis_ders(dist, scal, s, sigma);

          for (int ri = 0; ri < R; ri++) {
            val_temp[mu * R + ri] = radial_basis->radial_basis_vals[ri];
            der_temp[mu * R + ri] = radial_basis->radial_basis_ders[ri];
          }
        }
      } else {
        for (int k_ = 0; k_ < K_; k_++) {
          int sigma = mu_to_sigma[k_];

          // Read s1 and s2 from unified regression_coeffs array
          // SUS2-MLIP formula: regression_coeffs[C + 2*k_*C*C + C*type_central + type_outer]
          double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
          double s = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];

          // Call RB_Calc with s1 and s2 parameters
          radial_basis->calc_radial_basis_ders(dist, scal, s, sigma);

          // Store scaled values in temporary arrays
          for (int ri = 0; ri < R; ri++) {
            val_temp[k_ * R + ri] = radial_basis->radial_basis_vals[ri];
            der_temp[k_ * R + ri] = radial_basis->radial_basis_ders[ri];
          }
        }
      }

      // Now compute radial_vals for each mu

      for (int mu = 0; mu < radial_func_count; mu++) {

        double val = 0;
        double der = 0;
        // SUS2-MLIP: Map mu to K using mu_to_K
        int k_ = mu_to_K[mu];
        int radial_cache_index = per_mu_sigma ? mu : k_;
        int offset_mu = mu * (R + C);

        // SUS2-MLIP: Include species-dependent factors in radial basis calculation
        // Original: regression_coeffs[C+2*C*C*K_ + mu*(R+C) + R + type_central/outer]
        double species_factor = regression_coeffs[radial_coeffs_offset + R + itype] *
                              regression_coeffs[radial_coeffs_offset + R + jtype];

        #pragma omp simd reduction(+:val,der)
        for (int ri = 0; ri < R; ri++) {
          val += regression_coeffs[radial_coeffs_offset + offset_mu + ri] * species_factor *
                 val_temp[radial_cache_index * R + ri];
          der += regression_coeffs[radial_coeffs_offset + offset_mu + ri] * species_factor *
                 der_temp[radial_cache_index * R + ri];
        }

        radial_vals[mu] = val;
        radial_ders[mu] = der;
      }
      }


      //Calculate the alpha basics
      if (is_sh_model) {
        double sh_values[kMaxSHComponents];
        double sh_ders[3 * kMaxSHComponents];
        eval_real_sh(r, inv_dist, sh_l_max, sh_values, sh_ders);

        if (env_gate_enabled) {
          for (int k = 0; k < alpha_index_basic_count; k++) {
            const size_t jac_idx = static_cast<size_t>(jj) * alpha_index_basic_count + k;
            const int mu = alpha_basic_mu[k];
            const int sh_idx = alpha_basic_sh_index[k];
            const double radial_val = radial_vals[mu];
            const double radial_der = radial_ders[mu];
            const double ylm = sh_values[sh_idx];
            const double raw_contrib = radial_val * ylm;
            const double radial_der_pref = radial_der * inv_dist * ylm;
            const double raw_jac_x =
                radial_der_pref * r[0] + radial_val * sh_ders[3 * sh_idx + 0];
            const double raw_jac_y =
                radial_der_pref * r[1] + radial_val * sh_ders[3 * sh_idx + 1];
            const double raw_jac_z =
                radial_der_pref * r[2] + radial_val * sh_ders[3 * sh_idx + 2];
            const double activation_der_factor =
                -env_screen_strength * env_activation_der * raw_contrib * inv_dist;
            env_activation_basic_vals[k] += env_activation * raw_contrib;
            moment_tensor_vals[k] += pair_gate * raw_contrib;
            moment_jacobian_x[jac_idx] = pair_gate * raw_jac_x + activation_der_factor * r[0];
            moment_jacobian_y[jac_idx] = pair_gate * raw_jac_y + activation_der_factor * r[1];
            moment_jacobian_z[jac_idx] = pair_gate * raw_jac_z + activation_der_factor * r[2];
          }
        } else if (sh_basic_mu_grouped || sh_basic_mu_indexed) {
          for (int mu = 0; mu < radial_func_count; mu++) {
            const int begin = sh_basic_mu_offsets[mu];
            const int end = sh_basic_mu_offsets[mu + 1];
            if (begin == end) continue;
            const double radial_val = radial_vals[mu];
            const double radial_der = radial_ders[mu];
            for (int pos = begin; pos < end; pos++) {
              const int k = sh_basic_mu_grouped ? pos : sh_basic_mu_indices[pos];
              const size_t jac_idx =
                  static_cast<size_t>(jj) * alpha_index_basic_count + k;
              const int sh_idx = alpha_basic_sh_index[k];
              const double ylm = sh_values[sh_idx];
              const double raw_contrib = radial_val * ylm;
              const double radial_der_pref = radial_der * inv_dist * ylm;
              moment_tensor_vals[k] += raw_contrib;
              moment_jacobian_x[jac_idx] =
                  radial_der_pref * r[0] + radial_val * sh_ders[3 * sh_idx + 0];
              moment_jacobian_y[jac_idx] =
                  radial_der_pref * r[1] + radial_val * sh_ders[3 * sh_idx + 1];
              moment_jacobian_z[jac_idx] =
                  radial_der_pref * r[2] + radial_val * sh_ders[3 * sh_idx + 2];
            }
          }
        } else {
          for (int k = 0; k < alpha_index_basic_count; k++) {
            const size_t jac_idx = static_cast<size_t>(jj) * alpha_index_basic_count + k;
            const int mu = alpha_basic_mu[k];
            const int sh_idx = alpha_basic_sh_index[k];
            const double radial_val = radial_vals[mu];
            const double radial_der = radial_ders[mu];
            const double ylm = sh_values[sh_idx];
            const double raw_contrib = radial_val * ylm;
            const double radial_der_pref = radial_der * inv_dist * ylm;
            moment_tensor_vals[k] += raw_contrib;
            moment_jacobian_x[jac_idx] =
                radial_der_pref * r[0] + radial_val * sh_ders[3 * sh_idx + 0];
            moment_jacobian_y[jac_idx] =
                radial_der_pref * r[1] + radial_val * sh_ders[3 * sh_idx + 1];
            moment_jacobian_z[jac_idx] =
                radial_der_pref * r[2] + radial_val * sh_ders[3 * sh_idx + 2];
          }
        }
      } else {
        inv_dist_powers[0] = 1.0;
        coord_powers_x[0] = 1.0;
        coord_powers_y[0] = 1.0;
        coord_powers_z[0] = 1.0;
        for (int k = 1; k < max_alpha_index_basic; k++) {
          inv_dist_powers[k] = inv_dist_powers[k - 1] * inv_dist;
          coord_powers_x[k] = coord_powers_x[k - 1] * r[0];
          coord_powers_y[k] = coord_powers_y[k - 1] * r[1];
          coord_powers_z[k] = coord_powers_z[k - 1] * r[2];
        }

        for (int k = 0; k < alpha_index_basic_count; k++) {
          const size_t jac_idx = static_cast<size_t>(jj) * alpha_index_basic_count + k;

          double val = 0;
          double der = 0;
          const int mu = alpha_basic_mu[k];

          val = radial_vals[mu];
          der = radial_ders[mu];
          // Normalize by the rank of alpha's corresponding tensor
          const int norm_rank = alpha_basic_norm_rank[k];
          double norm_fac = inv_dist_powers[norm_rank];
          val *= norm_fac;
          der = der * norm_fac - norm_rank * val * inv_dist;

          const int a0 = alpha_basic_a0[k];
          const int a1 = alpha_basic_a1[k];
          const int a2 = alpha_basic_a2[k];
          double pow0 = coord_powers_x[a0];
          double pow1 = coord_powers_y[a1];
          double pow2 = coord_powers_z[a2];
          const double pow = pow0 * pow1 * pow2;
          const double raw_contrib = val * pow;

          const double radial_der_pref = pow * der / dist;
          double raw_jac_x = radial_der_pref * r[0];
          double raw_jac_y = radial_der_pref * r[1];
          double raw_jac_z = radial_der_pref * r[2];

          if (a0 != 0) {
            raw_jac_x += val * a0 *
                coord_powers_x[a0 - 1] * pow1 * pow2;
          }
          if (a1 != 0) {
            raw_jac_y += val * a1 * pow0 *
                coord_powers_y[a1 - 1] * pow2;
          }
          if (a2 != 0) {
            raw_jac_z += val * a2 * pow0 * pow1 *
                coord_powers_z[a2 - 1];
          }

          if (env_gate_enabled) {
            const double activation_der_factor =
                -env_screen_strength * env_activation_der * raw_contrib * inv_dist;
            env_activation_basic_vals[k] += env_activation * raw_contrib;
            moment_tensor_vals[k] += pair_gate * raw_contrib;
            moment_jacobian_x[jac_idx] = pair_gate * raw_jac_x + activation_der_factor * r[0];
            moment_jacobian_y[jac_idx] = pair_gate * raw_jac_y + activation_der_factor * r[1];
            moment_jacobian_z[jac_idx] = pair_gate * raw_jac_z + activation_der_factor * r[2];
	          } else {
	            moment_tensor_vals[k] += raw_contrib;
	            moment_jacobian_x[jac_idx] = raw_jac_x;
	            moment_jacobian_y[jac_idx] = raw_jac_y;
	            moment_jacobian_z[jac_idx] = raw_jac_z;
	          }
	        }
	      }
	    }

    if (use_static_fixed_basic_cache && active_dynamic_edge_count == 0 &&
        !eflag && !vflag) {
      continue;
    }

    auto trace_cpu_moments = [&](const char *stage) {
      if (!trace_first_site || ii != 0) return;
      std::ostringstream oss;
      oss.setf(std::ios::scientific);
      oss.precision(16);
      oss << "SUS2_SH_TRACE CPU " << stage << " first_moments";
      const int moment_limit = std::min(alpha_moment_count, 12);
      for (int k = 0; k < moment_limit; k++)
        oss << " m[" << k << "]=" << moment_tensor_vals[k];
      oss << " mapped_scalars";
      const int scalar_limit = std::min(alpha_scalar_count, 8);
      for (int k = 0; k < scalar_limit; k++) {
        const int idx = alpha_moment_mapping[k];
        oss << " B" << k << "(m[" << idx << "])=" << moment_tensor_vals[idx];
      }
      utils::logmesg(lmp, "{}\n", oss.str());
    };
    trace_cpu_moments("after_basic");

    // ------------ Contruct Other Alphas  ------------
    const int * __restrict times_a0 = alpha_times_a0;
    const int * __restrict times_a1 = alpha_times_a1;
    const double * __restrict times_coeff = alpha_times_coeff;
    const int * __restrict times_out = alpha_times_out;
    double * __restrict moments = moment_tensor_vals;

    for (int k = 0; k < alpha_index_times_count; k++) {
      double val0 = moments[times_a0[k]];
      double val1 = moments[times_a1[k]];
      double val2 = times_coeff[k];

      moments[times_out[k]] += val2 * val0 * val1;
    }
    trace_cpu_moments("after_products");
    // ------------ Compute Basis Set From Alpha Map ------------
    if (eflag_atom || eflag_global) {
      // SUS2-MLIP: Include shift_coeffs in energy calculation
      nbh_energy = shift_coeffs[itype] + species_coeffs[itype];
      for (int k = 0; k < alpha_scalar_count; k++)
        nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]] * species_coeffs[itype];

      // Tally energies per flags
      if (eflag_atom) eatom[i] += nbh_energy;
      if (eflag_global) eng_vdwl += nbh_energy;
    }

    // =========== Begin Backpropogation ===========

    //------------ Step 1: NBH energy derivative is the corresponding linear combination------------
    for (int k = 0; k < alpha_scalar_count; k++)
      nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

    //------------ Step 2: Propogate chain rule through the alpha times to the alpha basics ------------
    const int * __restrict times_a0_back = alpha_times_a0;
    const int * __restrict times_a1_back = alpha_times_a1;
    const double * __restrict times_coeff_back = alpha_times_coeff;
    const int * __restrict times_out_back = alpha_times_out;
    double * __restrict moment_ders = nbh_energy_ders_wrt_moments;

    for (int k = alpha_index_times_count - 1; k >= 0; k--) {
      int a0 = times_a0_back[k];
      int a1 = times_a1_back[k];
      double multiplier = times_coeff_back[k];
      int a3 = times_out_back[k];

      double val0 = moment_tensor_vals[a0];
      double val1 = moment_tensor_vals[a1];
      double val3 = moment_ders[a3];

      moment_ders[a1] += val3 * multiplier * val0;
      moment_ders[a0] += val3 * multiplier * val1;
    }
    const double species_weight = species_coeffs[itype];
    for (int k = 0; k < alpha_index_basic_count; k++)
      weighted_basic_moment_ders[k] = nbh_energy_ders_wrt_moments[k] * species_weight;

    double env_rho_force_prefactor = 0.0;
    if (env_gate_enabled) {
      double env_activation_derivative_weight = 0.0;
      for (int k = 0; k < alpha_index_basic_count; k++)
        env_activation_derivative_weight += weighted_basic_moment_ders[k] * env_activation_basic_vals[k];
      env_rho_force_prefactor = env_rho_factor * env_activation_derivative_weight;
    }

    //------------ Step 3: Multiply energy ders wrt moment by the Jacobian to get forces ------------
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];
      j &= NEIGHMASK;
      if (!within_cutoff[jj] &&
          (!env_gate_enabled || env_rho_dr[jj] == 0.0))
        continue;

      double temp_force[3] = {0, 0, 0};
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      if (within_cutoff[jj]) {
        const size_t jac_offset = static_cast<size_t>(jj) * alpha_index_basic_count;
        const double * __restrict jac_x = moment_jacobian_x + jac_offset;
        const double * __restrict jac_y = moment_jacobian_y + jac_offset;
        const double * __restrict jac_z = moment_jacobian_z + jac_offset;
        const double * __restrict weighted = weighted_basic_moment_ders;

        #pragma omp simd reduction(+:fx,fy,fz)
	        for (int k = 0; k < alpha_index_basic_count; k++) {
	          const double pref = weighted[k];
	          fx += pref * jac_x[k];
	          fy += pref * jac_y[k];
	          fz += pref * jac_z[k];
	        }
	      }
      double r[3] = {0.0, 0.0, 0.0};
      bool have_r = false;
      if (env_gate_enabled && env_rho_dr[jj] != 0.0) {
        r[0] = x[j][0] - xi[0];
        r[1] = x[j][1] - xi[1];
        r[2] = x[j][2] - xi[2];
        have_r = true;
        const double dist = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
	        if (dist > 0.0) {
	          const double rho_der_factor = env_rho_force_prefactor * env_rho_dr[jj] / dist;
	          fx += rho_der_factor * r[0];
	          fy += rho_der_factor * r[1];
	          fz += rho_der_factor * r[2];
	        }
	      }
	      temp_force[0] = fx;
      temp_force[1] = fy;
      temp_force[2] = fz;

      f[i][0] += temp_force[0];
      f[i][1] += temp_force[1];
      f[i][2] += temp_force[2];

      f[j][0] -= temp_force[0];
      f[j][1] -= temp_force[1];
      f[j][2] -= temp_force[2];

      //Calculate virial stress
      if (vflag) {
        // We only need to calculate rel pos again if stress are needed
        if (!have_r) {
          r[0] = x[j][0] - xi[0];
          r[1] = x[j][1] - xi[1];
          r[2] = x[j][2] - xi[2];
        }
        virial[0] -= temp_force[0] * r[0];    //xx
        virial[1] -= temp_force[1] * r[1];    //yy
        virial[2] -= temp_force[2] * r[2];    //zz

        virial[3] -= temp_force[0] * r[1] ;    //xy
        virial[4] -= temp_force[0] * r[2] ;    //xz
        virial[5] -= temp_force[1] * r[2] ;    //yz

        if (cvflag_atom) {
          cvatom[j][0] -= temp_force[0] * r[0];    //xx
          cvatom[j][1] -= temp_force[1] * r[1];    //yy
          cvatom[j][2] -= temp_force[2] * r[2];    //zz

          cvatom[j][3] -= temp_force[0] * r[1] ;    //xy
          cvatom[j][4] -= temp_force[0] * r[2] ;    //xz
          cvatom[j][5] -= temp_force[1] * r[2] ;    //yz

          cvatom[j][6] -= temp_force[1] * r[0] ;    //yx
          cvatom[j][7] -= temp_force[2] * r[0] ;    //zx
          cvatom[j][8] -= temp_force[2] * r[1] ;    //zy
        }
      }
	    }
	  }
}
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSUS2MTP::settings(int narg, char **arg)
{
  requested_tabstep = 1.0e-4;
  tabstep_set_by_user = false;
  static_fixed_types_arg.clear();

  if (narg < 1) error->all(FLERR, "Pair sus2mtp requires a potential file.");
  if ((narg - 1) % 2 != 0)
    error->all(FLERR,
               "Pair sus2mtp optional arguments must be specified as keyword/value pairs.");

  for (int i = 1; i < narg; i += 2) {
    const std::string keyword = LAMMPS_NS::utils::lowercase(arg[i]);
    if (keyword == "tabstep") {
      requested_tabstep = utils::numeric(FLERR, arg[i + 1], true, lmp);
      if (requested_tabstep <= 0.0)
        error->all(FLERR, "Pair sus2mtp tabstep must be greater than 0.");
      tabstep_set_by_user = true;
      continue;
    }
    if (keyword == "static_fixed_types") {
      static_fixed_types_arg = arg[i + 1];
      continue;
    }

    error->all(FLERR,
               "Pair sus2mtp only supports optional keywords \"tabstep\" and \"static_fixed_types\".");
  }

  FILE *mtp_file = utils::open_potential(arg[0], lmp, nullptr);
  read_file(mtp_file, arg[0]);
  fclose(mtp_file);
  configure_static_fixed_types();
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSUS2MTP::coeff(int narg, char **arg)
{
  // The potential file is specified in the setting function instead.
  if (narg != 2) error->all(FLERR, "Only \"pair_coeff * *\" is permitted");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSUS2MTP::init_style()
{
  if (force->newton_pair != 1) error->all(FLERR, "Pair style SUS2MTP requires Newton Pair on");
  if (two_layer_gate_enabled) {
    const int gate_comm_stride = two_layer_gate_comm_stride();
    comm_forward = gate_comm_stride;
    comm_reverse = gate_comm_stride;
  }

  // Request a full neighbourhood list which is needed for MTP
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSUS2MTP::init_one(int i, int j)
{
  if (i > species_count || j > species_count)
    error->all(FLERR,
               "SUS2MTP model has {} species, but LAMMPS data contains atom type pair {}-{}.",
               species_count, i, j);

  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set. See types {}-{}.", i, j);

  return interaction_cutoff > 0.0 ? interaction_cutoff : radial_basis->max_cutoff;
}

/* ----------------------------------------------------------------------
   SUS2-MTP file parsing helper function. 
   Based on SUS2-MLIP-1.1/dev_src/mtpr.cpp Load() function.
   Supports L parameter, scaling_map, shift_coeffs, scal_coeffs, and multiple radial basis types.
------------------------------------------------------------------------- */
void PairSUS2MTP::read_file(FILE *mtp_file, const char *)
{
  /*NOTE: TextFileReader is used in lieu of PotentialFileReader to ensure compatability 
with the MLIP-3 package. The alpha indicies in this format are all in one line, requiring
access to the buffer size that is not provided in PFR.
*/
  std::vector<double> alpha_times_coeff_buffer;
  sh_body_order = 0;
  sh_scalar_body_order.clear();
  two_layer_gate_enabled = false;
  two_layer_gate_shared_radial = false;
  two_layer_gate_body_linear_combo = true;
  two_layer_gate_full_scalar_weights = false;
  two_layer_gate_center_enabled = false;
  two_layer_residual_enabled = false;
  two_layer_gate_tanh_amplitude = 0.8;
  two_layer_gate_body_order_max = 0;
  two_layer_gate_scalar_count = 0;
  two_layer_gate_weight_count = 0;
  two_layer_gate_body_mix_weight_count = 0;
  two_layer_gate_product_limit = 0;
  two_layer_gate_moment_cache_width = 0;
  two_layer_gate_product_indices.clear();
  two_layer_gate_product_a0_indices.clear();
  two_layer_gate_product_a1_indices.clear();
  two_layer_gate_product_out_indices.clear();
  two_layer_gate_product_compact_a0_indices.clear();
  two_layer_gate_product_compact_a1_indices.clear();
  two_layer_gate_product_coeffs.clear();
  two_layer_gate_required_moments.clear();
  two_layer_gate_required_moment_indices.clear();
  two_layer_gate_required_moment_compact_index.clear();
  two_layer_gate_required_basic_indices.clear();
  two_layer_gate_required_basic_mu_offsets.clear();
  two_layer_gate_required_basic_mu_indices.clear();
  two_layer_gate_required_basics_are_y00 = false;
  two_layer_gate_scalar_body_order.clear();
  two_layer_gate_scalar_body_bucket.clear();
  two_layer_gate_scalar_moment_indices.clear();
  two_layer_gate_body_bucket_offsets.clear();
  two_layer_gate_body_bucket_scalar_indices.clear();
  two_layer_gate_mu_body_order.clear();
  two_layer_gate_scalar_indices.clear();
  two_layer_gate_weights.clear();
  two_layer_gate_full_signal_count = 0;
  two_layer_gate_full_identity_signal = false;
  two_layer_gate_full_signal_group_for_mu.clear();
  two_layer_gate_edge_l1_enabled = false;
  two_layer_gate_edge_l1_source_count = 0;
  two_layer_gate_edge_l1_weight_count = 0;
  two_layer_gate_edge_l1_source_moment_indices.clear();
  two_layer_gate_edge_l1_weights.clear();
  two_layer_gate_edge_l1_weights_by_source.clear();
  two_layer_gate_edge_sh_capacity = 0;
  two_layer_gate_edge_main_sh_capacity = 0;
  two_layer_gate_edge_residual_radial_capacity = 0;
  two_layer_gate_edge_signal_deriv_capacity = 0;
	  two_layer_gate_full_weights_by_scalar.clear();
  two_layer_gate_full_scalar_cache.clear();
  two_layer_gate_full_scalar_adjoint_cache.clear();
	  two_layer_gate_body_mix_weights.clear();
  two_layer_gate_radial_coeffs.clear();
  two_layer_gate_additive_coeffs.clear();
  zbl_enabled = false;
  zbl_inner = sus2_mtp_zbl::DefaultInnerCutoff();
  zbl_outer = sus2_mtp_zbl::DefaultOuterCutoff();
  zbl_max_outer = zbl_outer;
  zbl_max_outer_sq = zbl_max_outer * zbl_max_outer;
  zbl_typewise_cutoff_enabled = false;
  zbl_typewise_cutoff_factor = sus2_mtp_zbl::DefaultTypewiseCutoffFactor();
  interaction_cutoff = 0.0;
  interaction_cutoff_sq = 0.0;
  reset_zbl_storage(memory, zbl_atomic_numbers, zbl_pair_inner_cutoffs,
                    zbl_pair_outer_cutoffs, zbl_pair_outer_sq,
                    zbl_pair_constants);

  //Open the MTP file on proc 0
  if (comm->me == 0) {
    TextFileReader tfr(mtp_file, "ml-mtp");
    tfr.ignore_comments = true;
    std::string new_separators = "=, ";
    std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

    ValueTokenizer line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    std::string keyword = line_tokens.next_string();

    if (keyword != "MTP")    // Files checking
      error->one(FLERR, "Only MTP potential files are accepted.");
    std::string version_line = std::string(tfr.next_line());
    if (version_line.find("version = 1.1.0") != std::string::npos) {
      env_gate_enabled = false;
    } else if (version_line.find("version = 2.0.0") != std::string::npos) {
      env_gate_enabled = true;
      env_gate_cutoff_ratio = 0.5;
      env_gate_activation_on_ratio = kEnvGateDefaultActivationOnRatio;
      env_gate_channel_count = kEnvGateChannels;
    } else {
      error->one(FLERR, "MTP file must have version \"1.1.0\" or \"2.0.0\"");
    }

    // Read the potential name (optional)
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();

    if (keyword == "potential_name") {
      try {
        potential_name = line_tokens.next_string();
      } catch (TokenizerException e) {
        potential_name = "";
      }
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }

    //Check scaling (optional)
    if (keyword == "scaling") {
      scaling = line_tokens.next_double();
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      scaling = 1;
    }

    utils::logmesg(lmp, "The scaling is : {:.2e}.\n", scaling);

    // Read L parameter (SUS2-MLIP specific, optional)
    if (keyword == "L") {
      L_max = line_tokens.next_int();
      L_max += 1;  // Convert from 0-indexed to count (L=0,1,...,L_max)
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      L_max = 1;  // Default value
    }

    // Read scaling_map (SUS2-MLIP specific)
    if (keyword == "scaling_map") {
      scaling_map = line_tokens.next_string();
      if (scaling_map != "K" && scaling_map != "L" && scaling_map != "LK")
        error->one(FLERR, "scaling_map must be 'K', 'L', or 'LK'");
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      scaling_map = "LK";  // Default
    }

    utils::logmesg(lmp, "L_max = {}, scaling_map = {}\n", L_max, scaling_map);

    // Read the species count
    if (keyword != "species_count")
      error->one(FLERR, "Error reading MTP file. Species count not found.");
    species_count = line_tokens.next_int();
    utils::logmesg(lmp, "There are {} species.\n", species_count);

    int np1 = ((atom && atom->ntypes > species_count) ? atom->ntypes : species_count) + 1;
    memory->create(setflag, np1, np1, "pair:setflag");
    memory->create(cutsq, np1, np1, "pair:cutsq");
    for (int i = 0; i < np1; ++i) {
      for (int j = 0; j < np1; ++j) {
        setflag[i][j] = 0;
        cutsq[i][j] = 0.0;
      }
    }

    std::string header_line = std::string(tfr.next_line());
    keyword = first_keyword(header_line);
    if (keyword == "zbl_enabled") {
      zbl_enabled = parse_bool_assignment(header_line);
      if (zbl_enabled) {
        header_line = std::string(tfr.next_line());
        if (first_keyword(header_line) != "zbl_inner")
          error->one(FLERR, "SUS2 ZBL section is missing zbl_inner.");
        zbl_inner = parse_double_assignment(header_line);

        header_line = std::string(tfr.next_line());
        if (first_keyword(header_line) != "zbl_outer")
          error->one(FLERR, "SUS2 ZBL section is missing zbl_outer.");
        zbl_outer = parse_double_assignment(header_line);

        header_line = std::string(tfr.next_line());
        if (first_keyword(header_line) != "zbl_typewise_cutoff_enabled")
          error->one(FLERR, "SUS2 ZBL section is missing zbl_typewise_cutoff_enabled.");
        zbl_typewise_cutoff_enabled = parse_bool_assignment(header_line);

        header_line = std::string(tfr.next_line());
        if (first_keyword(header_line) != "zbl_typewise_cutoff_factor")
          error->one(FLERR, "SUS2 ZBL section is missing zbl_typewise_cutoff_factor.");
        zbl_typewise_cutoff_factor = parse_double_assignment(header_line);

        header_line = std::string(tfr.next_line());
        if (first_keyword(header_line) != "zbl_atomic_numbers")
          error->one(FLERR, "SUS2 ZBL section is missing zbl_atomic_numbers.");
        const std::vector<int> zbl_numbers = parse_int_list_line(header_line);
        if (static_cast<int>(zbl_numbers.size()) != species_count)
          error->one(FLERR, "SUS2 ZBL atomic-number count should match species_count.");
        if (!std::isfinite(zbl_inner) || zbl_inner < 0.0)
          error->one(FLERR, "SUS2 ZBL inner cutoff should be finite and non-negative.");
        if (!std::isfinite(zbl_outer) || zbl_outer <= 0.0)
          error->one(FLERR, "SUS2 ZBL outer cutoff should be finite and positive.");
        if (!zbl_typewise_cutoff_enabled && zbl_outer <= zbl_inner)
          error->one(FLERR, "SUS2 ZBL fixed cutoffs should satisfy inner < outer.");
        if (!std::isfinite(zbl_typewise_cutoff_factor) ||
            zbl_typewise_cutoff_factor < 0.5)
          error->one(FLERR, "SUS2 ZBL typewise cutoff factor should be finite and >= 0.5.");
        memory->create(zbl_atomic_numbers, species_count, "zbl_atomic_numbers");
        for (int i = 0; i < species_count; ++i) {
          if (zbl_numbers[i] < 1 || zbl_numbers[i] > 94)
            error->one(FLERR, "SUS2 ZBL atomic number should be in [1, 94].");
          zbl_atomic_numbers[i] = zbl_numbers[i];
        }
        zbl_max_outer =
            sus2_mtp_zbl::MaxPairOuterCutoff(species_count, zbl_atomic_numbers,
                                             zbl_outer,
                                             zbl_typewise_cutoff_enabled,
                                             zbl_typewise_cutoff_factor);
        zbl_max_outer_sq = zbl_max_outer * zbl_max_outer;
      }
      do {
        header_line = std::string(tfr.next_line());
        keyword = first_keyword(header_line);
      } while (!zbl_enabled &&
               (keyword == "zbl_inner" ||
                keyword == "zbl_outer" ||
                keyword == "zbl_typewise_cutoff_enabled" ||
                keyword == "zbl_typewise_cutoff_factor" ||
                keyword == "zbl_atomic_numbers"));
    }

    // Read the potential tag (also optional)
    line_tokens = ValueTokenizer(header_line, separators);
    keyword = line_tokens.next_string();
    if (keyword == "potential_tag") {
      try {
        potential_tag = line_tokens.next_string();
      } catch (TokenizerException e) {
        potential_tag = "";
      }
      is_sh_model = (potential_tag == "SUS2-SH");
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }
    if (is_sh_model) {
      if (keyword != "sh_l_max")
        error->one(FLERR, "SUS2-SH model is missing sh_l_max.");
      sh_l_max = line_tokens.next_int();
      if (sh_l_max < 0 || sh_l_max > 4)
        error->one(FLERR, "LAMMPS SUS2-SH interface currently supports 0 <= sh_l_max <= 4, got {}.", sh_l_max);

      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
      if (keyword != "sh_k_max")
        error->one(FLERR, "SUS2-SH model is missing sh_k_max.");
      sh_k_max = line_tokens.next_int();
      if (sh_k_max <= 0)
        error->one(FLERR, "Invalid SUS2-SH sh_k_max: {}.", sh_k_max);
      if (L_max != sh_l_max + 1)
        error->one(FLERR, "SUS2-SH L header is inconsistent with sh_l_max.");
      if (scaling_map != "LK")
        error->one(FLERR, "SUS2-SH requires scaling_map = LK.");

      do {
        line_tokens = ValueTokenizer(tfr.next_line(), separators);
        keyword = line_tokens.next_string();
        if (keyword == "sh_body_order") sh_body_order = line_tokens.next_int();
      } while (keyword.rfind("sh_", 0) == 0 && keyword != "radial_basis_type");
    }

    // Read the radial basis type
    if (keyword != "radial_basis_type")
      error->one(FLERR, "Error reading MTP file. No radial basis set type is specified.");
    std::string radial_basis_type = line_tokens.next_string();
    radial_basis_type_str = radial_basis_type;

    // Set the type of radial basis based on SUS2-MLIP supported types
    if (radial_basis_type == "RBChebyshev") {
      radial_basis = new RBChebyshev(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_s") {
      radial_basis = new RBChebyshev_s(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_S;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_ss") {
      radial_basis = new RBChebyshev_ss(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SS;
      radial_basis->scaling = scaling;
	    } else if (radial_basis_type == "RBChebyshev_sss") {
	      radial_basis = new RBChebyshev_sss(tfr, lmp);
	      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS;
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type == "RBChebyshev_sss_rational") {
	      radial_basis = new RBChebyshev_sss_rational(tfr, lmp);
	      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS_RATIONAL;
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type == "RBChebyshev_sss_lmp") {
	      radial_basis = new RBChebyshev_sss_lmp(tfr, lmp);
	      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP;
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type == "RBChebyshev_sss_rational_lmp") {
	      radial_basis = new RBChebyshev_sss_rational_lmp(tfr, lmp);
	      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS_RATIONAL_LMP;
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type == "RBLaguerre_log1p") {
      radial_basis = new RBLaguerre_log1p(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_lmp") {
      radial_basis = new RBLaguerre_log1p_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_pos") {
      radial_basis = new RBLaguerre_log1p_pos(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_pos_lmp") {
      radial_basis = new RBLaguerre_log1p_pos_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBJacobi_sss_lmp") {
      radial_basis = new RBJacobi_sss_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::JACOBI_SSS_LMP;
      radial_basis->scaling = scaling;
    } else


      error->one(FLERR,
                 "Error reading MTP file. The specified radial basis set type, {}, was not found..",
                 radial_basis_type);

    radial_basis_size = radial_basis->size;
    
    utils::logmesg(lmp, "Radial basis type: {}, size: {}, index: {} \n", radial_basis_type, radial_basis_size,radial_basis_type_index);

    // Read the basis function count
    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "radial_funcs_count")
      lmp->error->one(FLERR, "Error in reading MTP file. Cannot read radial function count.");
    radial_func_count = line_tokens.next_int();    // Assuming count is an int
    if (is_sh_model) {
      const int expected_radial_count = sh_k_max * (sh_l_max + 1);
      if (radial_func_count != expected_radial_count)
        error->one(FLERR, "SUS2-SH radial_funcs_count mismatch: got {}, expected sh_k_max*(sh_l_max+1) = {}.",
                   radial_func_count, expected_radial_count);
    }

    // Compute K_scaling based on scaling_map
    if (scaling_map == "K") {
      K_scaling = radial_func_count / L_max;
    } else if (scaling_map == "L") {
      K_scaling = L_max;
    } else {  // LK
      K_scaling = radial_func_count;
    }

    utils::logmesg(lmp, "radial_funcs_count = {}, K_scaling = {}\n", radial_func_count, K_scaling);

    // Allocate mu_to_K and mu_to_sigma arrays
    memory->create(mu_to_K, radial_func_count, "mu_to_K");
    memory->create(mu_to_sigma, radial_func_count, "mu_to_sigma");
    
    // Initialize mu_to_sigma (always n/L)
    for (int n = 0; n < radial_func_count; n++) {
      mu_to_sigma[n] = n / L_max;
    }
    
    // Initialize mu_to_K based on scaling_map
    if (scaling_map == "K") {
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n / L_max;
      }
    } else if (scaling_map == "L") {
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n % L_max;
      }
    } else {  // LK
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n;
      }
    }

    // Check for magnetic basis which is currently unsupported.
    // CRITICAL FIX: Determine expected scal_coeffs count BEFORE reading the line to set proper buffer size
    int pairs_count_temp = species_count * species_count;
    int scal_coeffs_count_temp = 2 * pairs_count_temp * K_scaling;
    int scal_coeffs_buffer_size = scal_coeffs_count_temp * 30 + 500;
    
    // Set large buffer size BEFORE reading to ensure scal_coeffs line is not truncated
    tfr.set_bufsize(scal_coeffs_buffer_size);
    
    // Save the raw line content before parsing for potential re-parsing with different separators
    std::string current_raw_line = std::string(tfr.next_line());
    utils::logmesg(lmp, "Raw line content: '{}'\n", current_raw_line);
    
    line_tokens = ValueTokenizer(current_raw_line, separators);
    keyword = line_tokens.next_string();
    utils::logmesg(lmp, "Next keyword after radial_funcs_count: '{}'\n", keyword);

    // Initialize saved_line and saved_line_valid for use across multiple sections
    std::string saved_line;
    bool saved_line_valid = false;
    if (keyword != "radial_coeffs" && keyword != "shift_coeffs" && keyword != "scal_coeffs") {
      if (keyword == "alpha_moments_count" || keyword == "env_gate_type") {
        saved_line = current_raw_line;
        saved_line_valid = true;
      } else {
        lmp->error->one(FLERR, "Error in reading MTP file. Expected 'radial_coeffs', 'shift_coeffs', 'scal_coeffs', 'alpha_moments_count', or 'env_gate_type', but found '{}'.", keyword);
      }
    }

    // Apply the robust keyword detection strategy from the reference code
    // but without additional memory allocation - use existing unified array
    utils::logmesg(lmp, "Applying robust keyword detection for MTP file reading\n");

    // Allocate unified regression_coeffs array
    // SUS2-MLIP-1.1: regression_coeffs.resize(C + 2*C*C*K_ + radial_func_count*(rb_size + species_count));
    shift_coeffs_offset = 0;
    scal_coeffs_offset = species_count;
    int pairs_count = species_count * species_count;
    int scal_coeffs_count_local = 2 * pairs_count * K_scaling;
    int radial_coeff_count_local = radial_func_count * (radial_basis_size + species_count);
    int env_gate_coeff_count_local =
        env_gate_enabled ? species_count * (1 + env_gate_channel_count) : 0;
    env_gate_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local + radial_coeff_count_local;
    env_gate_lambda_raw_offset = env_gate_coeffs_offset;
    env_gate_log_density_coeffs_offset = env_gate_coeffs_offset + species_count;
    regression_coeffs_count = env_gate_coeffs_offset + env_gate_coeff_count_local;
    
    memory->create(regression_coeffs, regression_coeffs_count, "regression_coeffs");

    // Set offset pointers
    radial_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local;
    
    shift_coeffs = &regression_coeffs[shift_coeffs_offset];
    scal_coeffs = &regression_coeffs[scal_coeffs_offset];
    radial_basis_coeffs = &regression_coeffs[radial_coeffs_offset];

    // Read shift_coeffs (SUS2-MLIP specific, optional) - robust detection without extra memory allocation
    // FIXED: Parse values from the same line (keyword = {values} format) and remove trailing braces
    // CRITICAL FIX: Use custom separators that don't split scientific notation (e.g., e+00)
    if (keyword == "shift_coeffs") {
      utils::logmesg(lmp, "Found shift_coeffs keyword, reading coefficients...\n");
      
      // Re-parse the line with custom separators that preserve scientific notation
      // Don't include +, -, * as separators since they appear in scientific notation
      // Use the previously saved raw line content
      std::string shift_separators = " ={},\t\r\n";
      ValueTokenizer shift_tokens(current_raw_line, shift_separators);
      
      // Skip "shift_coeffs" keyword
      shift_tokens.next_string();
      
      // The equal sign is skipped automatically by tokenizer
      // Now parse the values inside braces
      for (int i = 0; i < species_count; i++) {
        try {
          // Get the next token - should be the full scientific notation number
          std::string token = shift_tokens.next_string();
          
          // Remove possible braces and trim whitespace
          while (!token.empty() && (token.front() == '{' || token.front() == ' ' || token.front() == '\t')) {
            token.erase(0, 1);
          }
          while (!token.empty() && (token.back() == '}' || token.back() == ',' || token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
          }
          
          // Now parse the clean token
          shift_coeffs[i] = std::stod(token);
        } catch (const std::exception& e) {
          error->one(FLERR, "Error parsing shift_coeffs[{}]: error={}", i, e.what());
        }
        if (i < 5) {  // Log first few coefficients for debugging
          utils::logmesg(lmp, "shift_coeffs[{}] = {}\n", i, shift_coeffs[i]);
        }
      }
      // Move to next line after successful read
      // CRITICAL FIX: Update current_raw_line to the newly read line for proper scal_coeffs parsing
      current_raw_line = std::string(tfr.next_line());
      line_tokens = ValueTokenizer(current_raw_line, separators);
      keyword = line_tokens.next_string();
      utils::logmesg(lmp, "shift_coeffs read for {} species, next keyword: '{}'\n", species_count, keyword);
    } else {
      // Default values if shift_coeffs not present
      utils::logmesg(lmp, "shift_coeffs keyword not found (current keyword: '{}'), using defaults\n", keyword);
      for (int i = 0; i < species_count; i++) {
        shift_coeffs[i] = -1.0;
      }
    }

    // Read scal_coeffs (SUS2-MLIP specific, optional) - robust detection with error handling
    // FIXED: Parse values from the same line (keyword = {values} format)
    // CRITICAL FIX: Use custom separators that don't split scientific notation (e.g., e+00)
    if (keyword == "scal_coeffs") {
      utils::logmesg(lmp, "Found scal_coeffs keyword, reading coefficients...\n");
      
      // CRITICAL FIX: The scal_coeffs values are on the SAME line as the keyword, not on the next line!
      // Use the previously saved current_raw_line which contains the full scal_coeffs line
      std::string scal_raw_line = current_raw_line;
      
      try {
      // Re-parse the line with custom separators that preserve scientific notation
      // Don't include +, -, * as separators since they appear in scientific notation
      // CRITICAL FIX: Don't include "=" as a separator to avoid skipping the first value
      std::string scal_separators = " {},\t\r\n";
      ValueTokenizer scal_tokens(scal_raw_line, scal_separators);
      
      // Skip "scal_coeffs" keyword (the "=" will be handled by next_string automatically)
      scal_tokens.next_string();  // Skip "scal_coeffs"
      scal_tokens.next_string();  // This will skip the "=" token
        
        // Read all scal_coeffs directly into the existing unified regression_coeffs array
        // with error handling for each coefficient
        for (int i = 0; i < scal_coeffs_count_local; i++) {
          std::string token = scal_tokens.next_string();
          
          // Remove possible braces and trim whitespace
          while (!token.empty() && (token.front() == '{' || token.front() == ' ' || token.front() == '\t')) {
            token.erase(0, 1);
          }
          while (!token.empty() && (token.back() == '}' || token.back() == ',' || token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
          }
          
          // Parse the clean token
          scal_coeffs[i] = std::stod(token);
          if (i < 5) {  // Log first few coefficients for debugging
            utils::logmesg(lmp, "scal_coeffs[{}] = {}\n", i, scal_coeffs[i]);
          }
        }
        
        utils::logmesg(lmp, "scal_coeffs read successfully, count = {}\n", scal_coeffs_count_local);
      } catch (const std::exception& e) {
        error->one(FLERR, "MTP error: Exception while reading scal_coeffs: {}", e.what());
      } catch (...) {
        error->one(FLERR, "MTP error: Unknown exception while reading scal_coeffs");
      }
      
      // Move to next line after successful read
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      // Default values if scal_coeffs not present - use existing unified array
      utils::logmesg(lmp, "WARNING: scal_coeffs keyword not found (current keyword: '{}'), using defaults\n", keyword);
      
      // CRITICAL: Validate radial_basis before accessing its members
      if (!radial_basis) {
        error->one(FLERR, "MTP error: radial_basis is null when calculating default scaling coefficients");
      }
      
      double cutoff_range = radial_basis->max_cutoff - radial_basis->min_cutoff;
      if (cutoff_range <= 0.0) {
        error->one(FLERR, "MTP error: Invalid cutoff range (max={}, min={}) for scaling coefficients",
                   radial_basis->max_cutoff, radial_basis->min_cutoff);
      }
      
      double s1_default = 3.3 * 2 / cutoff_range;
      double s2_default = radial_basis->min_cutoff;
      utils::logmesg(lmp, "Default values: s1 = {}, s2 = {}, K_scaling = {}\n", s1_default, s2_default, K_scaling);
      
      // Use existing unified array structure - no additional memory allocation needed
      for (int j = 0; j < K_scaling; j++) {
        for (int i = 0; i < pairs_count; i++) {
          scal_coeffs[j * 2 * pairs_count + i] = s1_default;                    // s1 coefficients
          scal_coeffs[j * 2 * pairs_count + pairs_count + i] = s2_default;      // s2 coefficients
        }
      }
      utils::logmesg(lmp, "scal_coeffs initialized to defaults using existing unified array\n");
    }

    // Read radial basis coeffs (SUS2-MLIP format) - robust detection using existing unified array
    // CRITICAL: Only read 0-0 pair coefficients (all pairs share these coefficients)
    if (keyword == "radial_coeffs") {
      utils::logmesg(lmp, "Found radial_coeffs keyword, reading coefficients...\n");
      
      bool found_zero_zero = false;
      bool reached_next_section = false;
      while (!reached_next_section) {
        std::string line = std::string(tfr.next_line());
        
        // Check if we've reached the next section (alpha_moments_count)
        if (line.find("alpha_moments_count") != std::string::npos ||
            line.find("env_gate_type") != std::string::npos) {
          saved_line = line;  // Save this line for later processing
          saved_line_valid = true;
          reached_next_section = true;
          break;
        }
        
        // Skip empty lines
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        
        // Check if this is a pair header (starts with digit or sign)
        char c = line[pos];
        if (!((c >= '0' && c <= '9') || c == '+' || c == '-')) continue;
        
        // Parse pair type
        line_tokens = ValueTokenizer(line, separators + "-");
        int type1, type2;
        try {
          type1 = line_tokens.next_int();
          type2 = line_tokens.next_int();
        } catch (...) { continue; }
        
        utils::logmesg(lmp, "Found pair header: {}-{}\n", type1, type2);
        
        // Only process 0-0 pair (all pairs share these coefficients)
        if (type1 == 0 && type2 == 0) {
          found_zero_zero = true;
          
          for (int mu = 0; mu < radial_func_count; mu++) {
            std::string coeff_line = std::string(tfr.next_line());
            line_tokens = ValueTokenizer(coeff_line, separators + "{,}");
            for (int ri = 0; ri < radial_basis_size + species_count; ri++) {
              std::string token = line_tokens.next_string();
              
              // Remove possible braces
              if (!token.empty() && token.front() == '{') token.erase(0, 1);
              if (!token.empty() && token.back() == '}') token.pop_back();
              
              try {
                radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = std::stod(token);
                if (mu < 2 && ri < 3) {  // Log first few coefficients for debugging
                  utils::logmesg(lmp, "radial_basis_coeffs[mu={}, ri={}] = {}\n", mu, ri,
                               radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri]);
                }
              } catch (const std::exception& e) {
                error->one(FLERR, "Error parsing radial coefficient at mu={}, ri={}: {}", mu, ri, e.what());
              }
            }
          }
          utils::logmesg(lmp, "Successfully read 0-0 pair radial coefficients\n");
        } else {
          // Skip other pairs - skip all their coefficient lines
          utils::logmesg(lmp, "Skipping pair {}-{} (only 0-0 needed)\n", type1, type2);
          for (int j = 0; j < radial_func_count; j++) {
            std::string(tfr.next_line());  // Skip coefficient lines
          }
        }
      }
      
      if (!found_zero_zero) {
        error->one(FLERR, "MTP error: No 0-0 radial coefficients found in file");
      }
      
      // Update keyword for next section
      if (reached_next_section && saved_line_valid) {
        line_tokens = ValueTokenizer(saved_line, separators);
        keyword = line_tokens.next_string();
      }
      
      utils::logmesg(lmp, "radial_coeffs read successfully (0-0 pair only)\n");
    } else {
      // Initialize default values if not present - use existing unified array
      utils::logmesg(lmp, "radial_coeffs keyword not found, using defaults\n");
      for (int mu = 0; mu < radial_func_count; mu++) {
        for (int ri = 0; ri < radial_basis_size + species_count; ri++) {
          if (ri < radial_basis_size) {
            radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = 1e-2;
          } else {
            radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = 1.1;
          }
        }
      }
      utils::logmesg(lmp, "radial_basis_coeffs initialized to defaults using existing unified array\n");
    }

    if (env_gate_enabled) {
      if (!saved_line_valid || saved_line.find("env_gate_type") == std::string::npos)
        error->one(FLERR, "version = 2.0.0 model must contain env_gate_type section.");

      line_tokens = ValueTokenizer(saved_line, separators);
      keyword = line_tokens.next_string();
      std::string env_gate_type = line_tokens.next_string();
      if (env_gate_type != "centered_tanh_screen" && env_gate_type != "centered_exp_screen")
        error->one(FLERR, "Unsupported env_gate_type: {}", env_gate_type);

      line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
      keyword = line_tokens.next_string();
      if (keyword != "env_gate_cutoff_ratio")
        error->one(FLERR, "Error reading env_gate_cutoff_ratio.");
      env_gate_cutoff_ratio = line_tokens.next_double();

      line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
      keyword = line_tokens.next_string();
      if (keyword == "env_gate_activation_on_ratio") {
        env_gate_activation_on_ratio = line_tokens.next_double();
        if (!std::isfinite(env_gate_activation_on_ratio) ||
            env_gate_activation_on_ratio < 0.0 ||
            env_gate_activation_on_ratio >= 1.0)
          error->one(FLERR, "env_gate_activation_on_ratio should be in [0, 1).");
        line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
        keyword = line_tokens.next_string();
      } else {
        env_gate_activation_on_ratio = kEnvGateDefaultActivationOnRatio;
      }
      if (keyword != "env_gate_channel_count")
        error->one(FLERR, "Error reading env_gate_channel_count.");
      env_gate_channel_count = line_tokens.next_int();
      if (env_gate_channel_count != kEnvGateChannels)
        error->one(FLERR, "centered_tanh_screen env-gate requires 6 channels.");

      std::string gate_line = std::string(tfr.next_line());
      std::string gate_separators = " ={},\t\r\n";
      line_tokens = ValueTokenizer(gate_line, gate_separators);
      keyword = line_tokens.next_string();
      if (keyword == "env_gate_scal_coeffs") {
        for (int i = 0; i < 2 * pairs_count; ++i) {
          (void)std::stod(line_tokens.next_string());
        }
        gate_line = std::string(tfr.next_line());
        line_tokens = ValueTokenizer(gate_line, separators);
        keyword = line_tokens.next_string();
      }
      if (keyword != "env_gate_lambda_raw")
        error->one(FLERR, "Error reading env_gate_lambda_raw.");
      {
        std::string lambda_values = gate_line;
        const std::size_t equals_pos = lambda_values.find('=');
        if (equals_pos != std::string::npos)
          lambda_values = lambda_values.substr(equals_pos + 1);
        for (char &ch : lambda_values) {
          if (ch == '{' || ch == '}' || ch == ',') ch = ' ';
        }
        std::istringstream lambda_stream(lambda_values);
        std::vector<double> lambda_raw_values;
        double lambda_value = 0.0;
        while (lambda_stream >> lambda_value)
          lambda_raw_values.push_back(lambda_value);
        if (lambda_raw_values.empty())
          error->one(FLERR, "Error reading env_gate_lambda_raw.");
        if (lambda_raw_values.size() == 1) {
          for (int type_i = 0; type_i < species_count; ++type_i)
            regression_coeffs[env_gate_lambda_raw_offset + type_i] = lambda_raw_values[0];
        } else {
          if (static_cast<int>(lambda_raw_values.size()) != species_count)
            error->one(FLERR, "env_gate_lambda_raw count should be 1 or species_count.");
          for (int type_i = 0; type_i < species_count; ++type_i)
            regression_coeffs[env_gate_lambda_raw_offset + type_i] = lambda_raw_values[type_i];
        }
      }

      std::string density_line = std::string(tfr.next_line());
      std::string density_separators = " ={},\t\r\n";
      ValueTokenizer density_tokens(density_line, density_separators);
      keyword = density_tokens.next_string();
      if (keyword != "env_gate_log_density_coeffs")
        error->one(FLERR, "Error reading env_gate_log_density_coeffs.");
      for (int type_i = 0; type_i < species_count; ++type_i)
        for (int q = 0; q < env_gate_channel_count; ++q)
          regression_coeffs[env_gate_log_density_coeffs_offset + type_i * env_gate_channel_count + q] =
              std::stod(density_tokens.next_string());

      saved_line = std::string(tfr.next_line());
      saved_line_valid = true;
    }

    // Setup cutoff values
    interaction_cutoff = radial_basis->max_cutoff;
    if (zbl_enabled) interaction_cutoff = std::max(interaction_cutoff, zbl_max_outer);
    interaction_cutoff_sq = interaction_cutoff * interaction_cutoff;
    double rcutmaxsq = interaction_cutoff_sq;
    for (int i = 0; i < species_count; i++) {
      for (int j = 0; j < species_count; j++) {
        setflag[i + 1][j + 1] = 1;
        cutsq[i + 1][j + 1] = rcutmaxsq;
      }
    }

    // CRITICAL FIX: Robust search for alpha_moments_count with saved_line and attempt limits
    // This handles cases where the keyword might be embedded in other content or on unexpected lines
    bool found_alpha_moments = false;
    int max_attempts = 1000;  // Prevent infinite loops
    int attempt_count = 0;
    
    utils::logmesg(lmp, "Searching for alpha_moments_count in MTP file...\n");
    if (saved_line_valid && saved_line.find("alpha_moments_count") != std::string::npos) {
      found_alpha_moments = true;
      utils::logmesg(lmp, "Using saved alpha_moments_count line: '{}'\n", saved_line);
    }
    
    // First, try to find alpha_moments_count using saved_line approach
    // This handles cases where we might have already read past the line
    while (!found_alpha_moments && attempt_count < max_attempts) {
      attempt_count++;
      try {
        std::string next_line = std::string(tfr.next_line());
        
        // Check for empty lines
        if (next_line.empty()) {
          utils::logmesg(lmp, "WARNING: Empty line encountered while searching for alpha_moments_count (attempt {})\n", attempt_count);
          continue;
        }
        
        // Check if this line contains alpha_moments_count
        if (next_line.find("alpha_moments_count") != std::string::npos) {
          saved_line = next_line;
          saved_line_valid = true;
          found_alpha_moments = true;
          utils::logmesg(lmp, "Found alpha_moments_count in line (attempt {}): '{}'\n", attempt_count, saved_line);
          break;
        }
        
        // Trim leading whitespace and check if line starts with a digit (might be pair header)
        size_t pos = next_line.find_first_not_of(" \t\r\n");
        if (pos != std::string::npos) {
          char c = next_line[pos];
          // Skip lines that look like pair headers (start with digit or sign)
          if ((c >= '0' && c <= '9') || c == '+' || c == '-') {
            // This might be from radial_coeffs section, skip it
            continue;
          }
        }
        
      } catch (const std::exception& e) {
        error->one(FLERR, std::string("MTP error: Exception while searching for alpha_moments_count: ") + e.what());
        break;
      } catch (...) {
        error->one(FLERR, "MTP error: Unknown exception while searching for alpha_moments_count (attempt {})", attempt_count);
        break;
      }
    }
    
    // Check if we found alpha_moments_count
    if (!found_alpha_moments) {
      if (attempt_count >= max_attempts) {
        error->one(FLERR, "MTP error: Could not find alpha_moments_count after {} attempts. File may be corrupted or missing required field.", max_attempts);
      } else {
        error->one(FLERR, "Error reading MTP file. Alpha moment count not found.");
      }
    }
    
    // Parse alpha_moments_count from saved_line with validation
    if (saved_line_valid) {
      std::istringstream iss(saved_line);
      std::string keyword_str, equals_str;
      int value;
      iss >> keyword_str >> equals_str >> value;
      
      // Validate parsed values
      if (keyword_str != "alpha_moments_count" || equals_str != "=") {
        error->one(FLERR, "MTP error: Invalid alpha_moments_count format. Expected 'alpha_moments_count = value', got: '{}'", saved_line);
      }
      
      // CRITICAL: Validate the value - alpha_moment_count must be positive
      if (value <= 0) {
        error->one(FLERR, "MTP error: Invalid alpha_moments_count value: {}. Must be positive integer.", value);
      }
      
      // Check for unreasonably large values
      if (value > 1000000) {
        utils::logmesg(lmp, "WARNING: alpha_moments_count is unusually large: {}. File may have issues.\n", value);
      }
      
      alpha_moment_count = value;
      utils::logmesg(lmp, "Successfully parsed alpha_moments_count = {}\n", alpha_moment_count);
    } else {
      error->one(FLERR, "MTP error: alpha_moments_count found but could not be parsed from: '{}'", saved_line);
    }
    
    memory->create(moment_tensor_vals, alpha_moment_count, "moment_tensor_vals");
    memory->create(nbh_energy_ders_wrt_moments, alpha_moment_count, "nbh_energy_ders_wrt_moments");

    // Get the basic alpha count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic_count")
      error->one(FLERR, "Error reading MTP file. Alpha moment count not found.");
    alpha_index_basic_count = line_tokens.next_int();

    // Read the basic alphas
    int radial_func_max = 0;
    tfr.set_bufsize(
        (alpha_index_basic_count * 20 + 20) *
        sizeof(
            char));    // Adjust the buffer size. This needed to ensure cross-compatability since the MLIP files stores all the alpha indicies on the same line.
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic")
      error->one(FLERR, "Error reading MTP file. Alpha index basic not found.");
    memory->create(alpha_index_basic, alpha_index_basic_count, 4, "alpha_index_basic");
    for (int i = 0; i < alpha_index_basic_count; i++) {
      if (is_sh_model) {
        const int k_index = line_tokens.next_int();
        const int l_index = line_tokens.next_int();
        const int m_index = line_tokens.next_int();
        if (k_index < 0 || k_index >= sh_k_max || l_index < 0 || l_index > sh_l_max ||
            m_index < -l_index || m_index > l_index)
          error->one(FLERR, "Invalid SUS2-SH alpha_index_basic entry ({}, {}, {}).",
                     k_index, l_index, m_index);
        alpha_index_basic[i][0] = k_index * (sh_l_max + 1) + l_index;
        alpha_index_basic[i][1] = l_index;
        alpha_index_basic[i][2] = m_index;
        alpha_index_basic[i][3] = 0;
      } else {
        for (int j = 0; j < 4; j++) {
          int index = line_tokens.next_int();
          alpha_index_basic[i][j] = index;
        }
      }
      if (alpha_index_basic[i][0] > radial_func_max) radial_func_max = alpha_index_basic[i][0];
    }
    if (radial_func_max != radial_func_count - 1)    //Index validity check
      error->one(FLERR, "Wrong number of radial functions specified!");

    //Precompute the maximum alpha basic index
    if (is_sh_model) {
      max_alpha_index_basic = 1;
    } else {
      max_alpha_index_basic = 0;
      for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic =
            std::max(max_alpha_index_basic,
                     alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
      max_alpha_index_basic++;    // Add 1 to account for zeroth order indicies
    }

    // Get the alpha times count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times_count")
      error->one(FLERR, "Error reading MTP file. Alpha index times count not found.");
    const int legacy_alpha_index_times_count = line_tokens.next_int();

    // Read the legacy alpha-times line. SUS2-SH keeps this empty and stores the
    // canonical real-SH tensor product graph in sh_products.
    tfr.set_bufsize(
        std::max(std::max(legacy_alpha_index_times_count, 1) * 32 + 20, 128) *
        sizeof(
            char));    // Adjust the buffer size. This needed to ensure cross-compatability since the MLIP files stores all the alpha indicies on the same line.
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times")
      error->one(FLERR, "Error reading MTP file. Alpha index times not found.");
    if (is_sh_model) {
      if (legacy_alpha_index_times_count != 0)
        error->one(FLERR, "SUS2-SH model should use sh_products instead of alpha_index_times.");

      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
      if (keyword != "sh_product_count")
        error->one(FLERR, "SUS2-SH model is missing sh_product_count.");
      alpha_index_times_count = line_tokens.next_int();

      tfr.set_bufsize((std::max(alpha_index_times_count, 1) * 64 + 20) * sizeof(char));
      line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");
      keyword = line_tokens.next_string();
      if (keyword != "sh_products")
        error->one(FLERR, "SUS2-SH model is missing sh_products.");

      memory->create(alpha_index_times, alpha_index_times_count, 4, "alpha_index_times");
      alpha_times_coeff_buffer.assign(alpha_index_times_count, 0.0);
      for (int i = 0; i < alpha_index_times_count; i++) {
        alpha_index_times[i][0] = line_tokens.next_int();
        alpha_index_times[i][1] = line_tokens.next_int();
        alpha_index_times[i][3] = line_tokens.next_int();
        alpha_index_times[i][2] = 0;
        alpha_times_coeff_buffer[i] = line_tokens.next_double();
      }
    } else {
      alpha_index_times_count = legacy_alpha_index_times_count;
      memory->create(alpha_index_times, alpha_index_times_count, 4, "alpha_index_times");
      alpha_times_coeff_buffer.assign(alpha_index_times_count, 0.0);
      for (int i = 0; i < alpha_index_times_count; i++) {
        for (int j = 0; j < 4; j++) { alpha_index_times[i][j] = line_tokens.next_int(); }
        alpha_times_coeff_buffer[i] = static_cast<double>(alpha_index_times[i][2]);
      }
    }

    // Get the alpha scalar count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_scalar_moments")
      error->one(FLERR, "Error reading MTP file. Alpha scalar moment count not found.");
    alpha_scalar_count = line_tokens.next_int();

    //Read the alpha moment mappings
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_moment_mapping")
      error->one(FLERR, "Error reading MTP file. Alpha moment mappings not found.");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");
    for (int i = 0; i < alpha_scalar_count; i++) {
      alpha_moment_mapping[i] = line_tokens.next_int();
    }

    std::string post_scalar_line = std::string(tfr.next_line());
    keyword = first_keyword(post_scalar_line);
    if (is_sh_model && keyword == "sh_scalar_info_count") {
      const int info_count = parse_int_assignment(post_scalar_line);
      if (info_count != alpha_scalar_count)
        error->one(FLERR, "SUS2-SH sh_scalar_info_count should match alpha_scalar_moments.");
      std::string info_line = std::string(tfr.next_line());
      if (first_keyword(info_line) != "sh_scalar_info")
        error->one(FLERR, "Cannot read SUS2-SH scalar metadata.");
      std::vector<int> info_values = parse_int_list_line(info_line);
      if (static_cast<int>(info_values.size()) != info_count * 7)
        error->one(FLERR, "SUS2-SH sh_scalar_info should contain 7 integers per scalar.");
      sh_scalar_body_order.assign(info_count, 0);
      for (int i = 0; i < info_count; i++)
        sh_scalar_body_order[i] = info_values[7 * i];
      post_scalar_line = std::string(tfr.next_line());
      keyword = first_keyword(post_scalar_line);
    }

    if (is_sh_model && keyword == "two_layer_gate_enabled") {
      two_layer_gate_enabled = parse_bool_assignment(post_scalar_line);

      post_scalar_line = std::string(tfr.next_line());
      if (first_keyword(post_scalar_line) != "two_layer_gate_mode")
        error->one(FLERR, "SUS2-SH legacy two-layer gate models are not compatible with mu-body-order gate mode.");
      {
        std::string gate_mode = parse_string_assignment(post_scalar_line);
        if (gate_mode == "mu-body-order") gate_mode = "mu-scalar-full";
        if (gate_mode == "mu-body-linear-combo") {
          two_layer_gate_body_linear_combo = true;
          two_layer_gate_full_scalar_weights = false;
        } else if (gate_mode == "mu-scalar-full") {
          two_layer_gate_body_linear_combo = false;
          two_layer_gate_full_scalar_weights = true;
        } else {
          error->one(FLERR, "LAMMPS SUS2-SH two-layer gate has unknown mode: {}", gate_mode);
        }
      }

      post_scalar_line = std::string(tfr.next_line());
      if (first_keyword(post_scalar_line) != "two_layer_gate_body_order_max")
        error->one(FLERR, "SUS2-SH two-layer gate is missing two_layer_gate_body_order_max.");
      two_layer_gate_body_order_max = parse_int_assignment(post_scalar_line);
      if (two_layer_gate_body_order_max != sh_k_max + 1)
        error->one(FLERR, "SUS2-SH mu-body-order gate body order should be sh_k_max + 1.");
      if (two_layer_gate_body_order_max < 2 ||
          two_layer_gate_body_order_max > sh_body_order)
        error->one(FLERR, "SUS2-SH mu-body-order gate body order is out of range.");

      post_scalar_line = std::string(tfr.next_line());
      if (first_keyword(post_scalar_line) != "two_layer_gate_include_one_body")
        error->one(FLERR, "SUS2-SH two-layer gate is missing two_layer_gate_include_one_body.");
      if (parse_bool_assignment(post_scalar_line))
        error->one(FLERR, "LAMMPS SUS2-SH two-layer gate requires include_one_body = false.");

      post_scalar_line = std::string(tfr.next_line());
      keyword = first_keyword(post_scalar_line);
      if (keyword == "two_layer_gate_site_mode") {
        const std::string site_mode = parse_string_assignment(post_scalar_line);
        if (site_mode == "neighbor") {
          two_layer_gate_center_enabled = false;
        } else if (site_mode == "double") {
          two_layer_gate_center_enabled = true;
        } else {
          error->one(FLERR, "Unknown SUS2-SH two-layer gate site mode: {}", site_mode);
        }
        post_scalar_line = std::string(tfr.next_line());
        keyword = first_keyword(post_scalar_line);
      } else {
        two_layer_gate_center_enabled = false;
      }
      if (keyword == "two_layer_residual_enabled") {
        two_layer_residual_enabled = parse_bool_assignment(post_scalar_line);
        if (two_layer_residual_enabled)
          error->one(FLERR, "LAMMPS SUS2-SH interface does not support residual two-layer models.");
        post_scalar_line = std::string(tfr.next_line());
        keyword = first_keyword(post_scalar_line);
      }
      if (keyword == "two_layer_gate_scale_mode") {
        error->one(FLERR, "SUS2-SH mu-body-order gate models do not use two_layer_gate_scale_mode.");
      }
      if (keyword == "two_layer_gate_bias") {
        error->one(FLERR, "SUS2-SH mu-body-order gate models do not use two_layer_gate_bias.");
      }
      if (keyword == "two_layer_gate_tanh_amplitude") {
        two_layer_gate_tanh_amplitude = parse_double_assignment(post_scalar_line);
        if (!std::isfinite(two_layer_gate_tanh_amplitude) ||
            two_layer_gate_tanh_amplitude < 0.0 ||
            two_layer_gate_tanh_amplitude > 1.0)
          error->one(FLERR, "SUS2-SH two_layer_gate_tanh_amplitude should be finite and in [0, 1].");
        post_scalar_line = std::string(tfr.next_line());
        keyword = first_keyword(post_scalar_line);
      }
      if (keyword == "two_layer_gate_radial_mode") {
        const std::string radial_mode = parse_string_assignment(post_scalar_line);
        if (radial_mode == "shared-radial") {
          two_layer_gate_shared_radial = true;
          post_scalar_line = std::string(tfr.next_line());
          if (first_keyword(post_scalar_line) != "two_layer_gate_radial_coeff_count")
            error->one(FLERR, "SUS2-SH two-layer gate is missing radial coeff count.");
          const int gate_radial_count = parse_int_assignment(post_scalar_line);
          const int expected_gate_radial_count = radial_func_count * radial_basis_size;
          if (gate_radial_count != expected_gate_radial_count)
            error->one(FLERR, "SUS2-SH two-layer gate radial coefficient count is inconsistent.");
          post_scalar_line = std::string(tfr.next_line());
          if (first_keyword(post_scalar_line) != "two_layer_gate_radial_coeffs")
            error->one(FLERR, "SUS2-SH two-layer gate is missing radial coeffs.");
          two_layer_gate_radial_coeffs = parse_double_list_line(post_scalar_line);
          if (static_cast<int>(two_layer_gate_radial_coeffs.size()) != gate_radial_count)
            error->one(FLERR, "SUS2-SH two-layer gate radial coeff list has wrong size.");
          post_scalar_line = std::string(tfr.next_line());
          keyword = first_keyword(post_scalar_line);
        } else if (radial_mode == "base-radial" || radial_mode == "legacy") {
          two_layer_gate_shared_radial = false;
          post_scalar_line = std::string(tfr.next_line());
          keyword = first_keyword(post_scalar_line);
        } else {
          error->one(FLERR, "Unknown SUS2-SH two-layer gate radial mode: {}", radial_mode);
        }
      }
      if (keyword == "two_layer_gate_additive_coeff_count") {
        const int additive_count = parse_int_assignment(post_scalar_line);
        const int expected_additive_count = species_count;
        if (additive_count != expected_additive_count)
          error->one(FLERR, "SUS2-SH two-layer gate additive coefficient count is inconsistent.");
        post_scalar_line = std::string(tfr.next_line());
        if (first_keyword(post_scalar_line) != "two_layer_gate_additive_coeffs")
          error->one(FLERR, "SUS2-SH two-layer gate is missing additive coeffs.");
        two_layer_gate_additive_coeffs = parse_double_list_line(post_scalar_line);
        if (static_cast<int>(two_layer_gate_additive_coeffs.size()) != additive_count)
          error->one(FLERR, "SUS2-SH two-layer gate additive coeff list has wrong size.");
        post_scalar_line = std::string(tfr.next_line());
        keyword = first_keyword(post_scalar_line);
      } else {
        two_layer_gate_additive_coeffs.assign(species_count, 1.0 / 12.0);
      }
      if (keyword != "two_layer_gate_weight_count")
        error->one(FLERR, "SUS2-SH two-layer gate is missing two_layer_gate_weight_count.");
      two_layer_gate_weight_count = parse_int_assignment(post_scalar_line);
      if (two_layer_gate_weight_count <= 0)
        error->one(FLERR, "SUS2-SH two-layer gate should contain at least one scalar weight.");
      two_layer_gate_scalar_count = two_layer_gate_weight_count;
      if (two_layer_gate_full_scalar_weights) {
        if (radial_func_count <= 0 ||
            two_layer_gate_weight_count % radial_func_count != 0)
          error->one(FLERR, "SUS2-SH full mu/scalar gate weight count is inconsistent.");
        two_layer_gate_scalar_count =
            two_layer_gate_weight_count / radial_func_count;
      }
      {
        const long sh_product_line_buffer =
            (std::max(alpha_index_times_count, 1) * 64L + 20L) *
            static_cast<long>(sizeof(char));
        const long moment_coeff_line_buffer =
            std::max(1, alpha_scalar_count) * 64L + 1024L;
        const long gate_line_buffer =
            std::max(4096L,
                     std::max(static_cast<long>(two_layer_gate_weight_count),
                              static_cast<long>(two_layer_gate_scalar_count)) *
                             32L +
                         1024L);
        const long post_gate_line_buffer =
            std::max(gate_line_buffer,
                     std::max(sh_product_line_buffer, moment_coeff_line_buffer));
        if (post_gate_line_buffer > 100000000L)
          error->one(FLERR, "SUS2-SH two-layer gate metadata line is too large.");
        tfr.set_bufsize(static_cast<int>(post_gate_line_buffer));
      }

      post_scalar_line = std::string(tfr.next_line());
      if (first_keyword(post_scalar_line) != "two_layer_gate_scalar_indices")
        error->one(FLERR, "SUS2-SH two-layer gate is missing scalar indices.");
      two_layer_gate_scalar_indices = parse_int_list_line(post_scalar_line);
      if (static_cast<int>(two_layer_gate_scalar_indices.size()) != two_layer_gate_scalar_count)
        error->one(FLERR, "SUS2-SH two-layer gate scalar index list has wrong size.");

      post_scalar_line = std::string(tfr.next_line());
      if (first_keyword(post_scalar_line) != "two_layer_gate_weights")
        error->one(FLERR, "SUS2-SH two-layer gate is missing weights.");
      two_layer_gate_weights = parse_double_list_line(post_scalar_line);
      if (static_cast<int>(two_layer_gate_weights.size()) != two_layer_gate_weight_count)
        error->one(FLERR, "SUS2-SH two-layer gate weight list has wrong size.");

      if (const char *next_line = tfr.next_line()) {
        post_scalar_line = std::string(next_line);
        keyword = first_keyword(post_scalar_line);
      } else {
        post_scalar_line.clear();
        keyword.clear();
      }
      if (two_layer_gate_body_linear_combo) {
        if (keyword != "two_layer_gate_body_mix_weight_count")
          error->one(FLERR, "SUS2-SH two-layer gate is missing body mix weight count.");
        two_layer_gate_body_mix_weight_count = parse_int_assignment(post_scalar_line);
        const int expected_mix_count =
            radial_func_count * (two_layer_gate_body_order_max - 1);
        if (two_layer_gate_body_mix_weight_count != expected_mix_count)
          error->one(FLERR, "SUS2-SH two-layer gate body mix weight count is inconsistent.");
        post_scalar_line = std::string(tfr.next_line());
        if (first_keyword(post_scalar_line) != "two_layer_gate_body_mix_weights")
          error->one(FLERR, "SUS2-SH two-layer gate is missing body mix weights.");
        two_layer_gate_body_mix_weights = parse_double_list_line(post_scalar_line);
        if (static_cast<int>(two_layer_gate_body_mix_weights.size()) !=
            two_layer_gate_body_mix_weight_count)
          error->one(FLERR, "SUS2-SH two-layer gate body mix weight list has wrong size.");
        post_scalar_line = std::string(tfr.next_line());
        keyword = first_keyword(post_scalar_line);
      } else {
        two_layer_gate_body_mix_weight_count = 0;
        two_layer_gate_body_mix_weights.clear();
        if (keyword == "two_layer_gate_body_mix_weight_count")
          error->one(FLERR, "SUS2-SH mu-scalar-full gate models should not contain body mix weights.");
      }
      if (keyword == "two_layer_gate_edge_l1_enabled") {
        two_layer_gate_edge_l1_enabled = parse_bool_assignment(post_scalar_line);
        if (const char *next_line = tfr.next_line()) {
          post_scalar_line = std::string(next_line);
          keyword = first_keyword(post_scalar_line);
        } else {
          post_scalar_line.clear();
          keyword.clear();
        }
        if (two_layer_gate_edge_l1_enabled) {
          if (keyword.empty())
            error->one(FLERR, "SUS2-SH edge L1 gate is missing source count metadata.");
          if (two_layer_gate_center_enabled)
            error->one(FLERR, "SUS2-SH edge L1 gate supports neighbor site mode only in LAMMPS.");
          if (sh_l_max < 1)
            error->one(FLERR, "SUS2-SH edge L1 gate requires sh_l_max >= 1.");
          const bool has_source_moments =
              keyword == "two_layer_gate_edge_l1_source_moment_count" ||
              keyword == "source_moment_count";
          const bool has_source_mus =
              keyword == "two_layer_gate_edge_l1_source_mu_count" ||
              keyword == "source_mu_count";
          if (!has_source_moments && !has_source_mus)
            error->one(FLERR, "SUS2-SH edge L1 gate is missing source count metadata.");
          two_layer_gate_edge_l1_source_count =
              parse_int_assignment(post_scalar_line);
          if (two_layer_gate_edge_l1_source_count <= 0)
            error->one(FLERR, "SUS2-SH edge L1 gate source count should be positive.");

          post_scalar_line = std::string(tfr.next_line());
          keyword = first_keyword(post_scalar_line);
          if (has_source_moments) {
            if (keyword != "two_layer_gate_edge_l1_source_moment_indices" &&
                keyword != "source_moment_indices")
              error->one(FLERR, "SUS2-SH edge L1 gate is missing source moment indices.");
          } else if (keyword != "two_layer_gate_edge_l1_source_mu_indices" &&
                     keyword != "source_mu_indices") {
            error->one(FLERR, "SUS2-SH edge L1 gate is missing source mu indices.");
          }
          std::vector<int> source_indices = parse_int_list_line(post_scalar_line);
          if (static_cast<int>(source_indices.size()) !=
              two_layer_gate_edge_l1_source_count)
            error->one(FLERR, "SUS2-SH edge L1 gate source index list has wrong size.");
          two_layer_gate_edge_l1_source_moment_indices.clear();
          if (has_source_moments) {
            for (int base : source_indices) {
              if (base < 0 || base + 2 >= alpha_moment_count)
                error->one(FLERR, "SUS2-SH edge L1 gate source moment index is out of range.");
              two_layer_gate_edge_l1_source_moment_indices.push_back(base);
            }
          } else {
            for (int mu : source_indices) {
              int base_index = -1;
              for (int k = 0; k + 2 < alpha_index_basic_count; k++) {
                if (alpha_index_basic[k][0] == mu &&
                    alpha_index_basic[k][1] == 1 &&
                    alpha_index_basic[k][2] == -1 &&
                    alpha_index_basic[k + 1][0] == mu &&
                    alpha_index_basic[k + 1][1] == 1 &&
                    alpha_index_basic[k + 1][2] == 0 &&
                    alpha_index_basic[k + 2][0] == mu &&
                    alpha_index_basic[k + 2][1] == 1 &&
                    alpha_index_basic[k + 2][2] == 1) {
                  base_index = k;
                  break;
                }
              }
              if (base_index < 0)
                error->one(FLERR, "SUS2-SH edge L1 gate source mu does not map to a basic l=1 moment.");
              two_layer_gate_edge_l1_source_moment_indices.push_back(base_index);
            }
          }

          post_scalar_line = std::string(tfr.next_line());
          keyword = first_keyword(post_scalar_line);
          if (keyword != "two_layer_gate_edge_l1_weight_count" &&
              keyword != "weight_count")
            error->one(FLERR, "SUS2-SH edge L1 gate is missing weight count.");
          two_layer_gate_edge_l1_weight_count =
              parse_int_assignment(post_scalar_line);
          const int expected_edge_l1_weight_count =
              radial_func_count * two_layer_gate_edge_l1_source_count;
          if (two_layer_gate_edge_l1_weight_count !=
              expected_edge_l1_weight_count)
            error->one(FLERR, "SUS2-SH edge L1 gate weight count is inconsistent.");

          post_scalar_line = std::string(tfr.next_line());
          keyword = first_keyword(post_scalar_line);
          if (keyword != "two_layer_gate_edge_l1_weights" &&
              keyword != "weights")
            error->one(FLERR, "SUS2-SH edge L1 gate is missing weights.");
          two_layer_gate_edge_l1_weights =
              parse_double_list_line(post_scalar_line);
          if (static_cast<int>(two_layer_gate_edge_l1_weights.size()) !=
              two_layer_gate_edge_l1_weight_count)
            error->one(FLERR, "SUS2-SH edge L1 gate weight list has wrong size.");

          if (const char *next_line = tfr.next_line()) {
            post_scalar_line = std::string(next_line);
            keyword = first_keyword(post_scalar_line);
          } else {
            post_scalar_line.clear();
            keyword.clear();
          }
        } else {
          two_layer_gate_edge_l1_source_count = 0;
          two_layer_gate_edge_l1_weight_count = 0;
          two_layer_gate_edge_l1_source_moment_indices.clear();
          two_layer_gate_edge_l1_weights.clear();
        }
      }
      if (keyword == "two_layer_residual_e0_coeff_count")
        error->one(FLERR, "LAMMPS SUS2-SH interface does not support residual two-layer models.");

      if (sh_scalar_body_order.empty())
        error->one(FLERR, "SUS2-SH mu-body-order gate requires sh_scalar_info metadata.");
      two_layer_gate_scalar_body_order.assign(two_layer_gate_scalar_count, 0);
      std::vector<int> body_order_counts(two_layer_gate_body_order_max + 1, 0);
      for (int q = 0; q < two_layer_gate_scalar_count; q++) {
        const int scalar_index = two_layer_gate_scalar_indices[q];
        if (scalar_index < 0 || scalar_index >= alpha_scalar_count)
          error->one(FLERR, "SUS2-SH two-layer gate scalar index is out of range.");
        const int body_order = sh_scalar_body_order[scalar_index];
        if (body_order < 2 || body_order > two_layer_gate_body_order_max)
          error->one(FLERR, "SUS2-SH mu-body-order gate scalar index has wrong body order.");
        two_layer_gate_scalar_body_order[q] = body_order;
        body_order_counts[body_order]++;
      }
      for (int body_order = 2; body_order <= two_layer_gate_body_order_max;
           body_order++)
        if (body_order_counts[body_order] == 0)
          error->one(FLERR, "SUS2-SH mu-body-order gate is missing a required scalar body-order bucket.");

      two_layer_gate_mu_body_order.assign(radial_func_count, 0);
      for (int mu = 0; mu < radial_func_count; mu++) {
        const int body_order = mu / L_max + 2;
        if (body_order < 2 || body_order > two_layer_gate_body_order_max)
          error->one(FLERR, "SUS2-SH mu-body-order gate has an invalid mu body order.");
        two_layer_gate_mu_body_order[mu] = body_order;
      }
    }

    //Read the species coefficients
    keyword = first_keyword(post_scalar_line);
    memory->create(species_coeffs, species_count, "species_coeffs");
    memory->create(linear_coeffs, alpha_scalar_count, "moment_coeffs");
    if (keyword.empty()) {
      std::fill(species_coeffs, species_coeffs + species_count, 1.0e-3);
      std::fill(linear_coeffs, linear_coeffs + alpha_scalar_count, 1.0e-3);
    } else {
      line_tokens = ValueTokenizer(post_scalar_line, separators + "{},");

      keyword = line_tokens.next_string();
      if (keyword != "species_coeffs")
        error->one(FLERR, "Error reading MTP file. Species coefficients not found.");
      for (int i = 0; i < species_count; i++) { species_coeffs[i] = line_tokens.next_double(); }

      //Read the linear MTP basis coefficients
      if (const char *moment_line = tfr.next_line()) {
        line_tokens = ValueTokenizer(moment_line, separators + "{},");
      } else {
        error->one(FLERR, "Error reading MTP file. Moment coefficients not found.");
      }

      keyword = line_tokens.next_string();
      if (keyword != "moment_coeffs")
        error->one(FLERR, "Error reading MTP file. Moment coefficients not found.");
      for (int i = 0; i < alpha_scalar_count; i++) { linear_coeffs[i] = line_tokens.next_double(); }
    }
  }    // Proc 0

  // ---------- CRITICAL FIX: Broadcast ALL single values BEFORE allocating arrays ----------
  // This ensures all procs have correct parameter values before they are used
  //Radial Basis Set Type First
  MPI_Bcast(&radial_basis_type_index, 1, MPI_INT, 0, world);

  //Then Single Values
  MPI_Bcast(&scaling, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&species_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&L_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&K_scaling, 1, MPI_INT, 0, world);
  int is_sh_model_int = is_sh_model ? 1 : 0;
  MPI_Bcast(&is_sh_model_int, 1, MPI_INT, 0, world);
  is_sh_model = (is_sh_model_int != 0);
  MPI_Bcast(&sh_l_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&sh_k_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&sh_body_order, 1, MPI_INT, 0, world);
  MPI_Bcast(&radial_basis_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&radial_func_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_moment_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_index_basic_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&max_alpha_index_basic, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_index_times_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_scalar_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&regression_coeffs_count, 1, MPI_INT, 0, world);
  int env_gate_enabled_int = env_gate_enabled ? 1 : 0;
  MPI_Bcast(&env_gate_enabled_int, 1, MPI_INT, 0, world);
  env_gate_enabled = (env_gate_enabled_int != 0);
  MPI_Bcast(&env_gate_cutoff_ratio, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&env_gate_activation_on_ratio, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&env_gate_channel_count, 1, MPI_INT, 0, world);
  int zbl_enabled_int = zbl_enabled ? 1 : 0;
  int zbl_typewise_cutoff_enabled_int = zbl_typewise_cutoff_enabled ? 1 : 0;
  MPI_Bcast(&zbl_enabled_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&zbl_inner, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&zbl_outer, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&zbl_typewise_cutoff_enabled_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&zbl_typewise_cutoff_factor, 1, MPI_DOUBLE, 0, world);
  zbl_enabled = (zbl_enabled_int != 0);
  zbl_typewise_cutoff_enabled = (zbl_typewise_cutoff_enabled_int != 0);
  zbl_max_outer = zbl_outer;
  zbl_max_outer_sq = zbl_max_outer * zbl_max_outer;
  if (zbl_enabled) {
    if (comm->me != 0)
      memory->create(zbl_atomic_numbers, species_count, "zbl_atomic_numbers");
    MPI_Bcast(zbl_atomic_numbers, species_count, MPI_INT, 0, world);
    const int zbl_pair_count = species_count * species_count;
    memory->create(zbl_pair_inner_cutoffs, zbl_pair_count, "zbl_pair_inner_cutoffs");
    memory->create(zbl_pair_outer_cutoffs, zbl_pair_count, "zbl_pair_outer_cutoffs");
    memory->create(zbl_pair_outer_sq, zbl_pair_count, "zbl_pair_outer_sq");
    memory->create(zbl_pair_constants, zbl_pair_count, "zbl_pair_constants");
    sus2_mtp_zbl::FillPairCutoffTables(species_count, zbl_atomic_numbers,
                                       zbl_inner, zbl_outer,
                                       zbl_typewise_cutoff_enabled,
                                       zbl_typewise_cutoff_factor,
                                       zbl_pair_inner_cutoffs,
                                       zbl_pair_outer_cutoffs,
                                       zbl_pair_outer_sq);
    zbl_max_outer =
        sus2_mtp_zbl::MaxPairOuterCutoff(species_count, zbl_atomic_numbers,
                                         zbl_outer,
                                         zbl_typewise_cutoff_enabled,
                                         zbl_typewise_cutoff_factor);
    zbl_max_outer_sq = zbl_max_outer * zbl_max_outer;
    sus2_mtp_zbl::FillPairConstants(species_count, zbl_atomic_numbers,
                                    zbl_pair_constants);
  }
  int two_layer_gate_enabled_int = two_layer_gate_enabled ? 1 : 0;
  int two_layer_gate_shared_radial_int = two_layer_gate_shared_radial ? 1 : 0;
  int two_layer_gate_body_linear_combo_int =
      two_layer_gate_body_linear_combo ? 1 : 0;
  int two_layer_gate_full_scalar_weights_int =
      two_layer_gate_full_scalar_weights ? 1 : 0;
  int two_layer_gate_center_enabled_int = two_layer_gate_center_enabled ? 1 : 0;
  int two_layer_residual_enabled_int = two_layer_residual_enabled ? 1 : 0;
  int two_layer_gate_edge_l1_enabled_int =
      two_layer_gate_edge_l1_enabled ? 1 : 0;
  MPI_Bcast(&two_layer_gate_enabled_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_shared_radial_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_body_linear_combo_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_full_scalar_weights_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_center_enabled_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_residual_enabled_int, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_edge_l1_enabled_int, 1, MPI_INT, 0, world);
  two_layer_gate_enabled = (two_layer_gate_enabled_int != 0);
  two_layer_gate_shared_radial = (two_layer_gate_shared_radial_int != 0);
  two_layer_gate_body_linear_combo =
      (two_layer_gate_body_linear_combo_int != 0);
  two_layer_gate_full_scalar_weights =
      (two_layer_gate_full_scalar_weights_int != 0);
  two_layer_gate_center_enabled = (two_layer_gate_center_enabled_int != 0);
  two_layer_residual_enabled = (two_layer_residual_enabled_int != 0);
  two_layer_gate_edge_l1_enabled =
      (two_layer_gate_edge_l1_enabled_int != 0);
  MPI_Bcast(&two_layer_gate_tanh_amplitude, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&two_layer_gate_body_order_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_scalar_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_weight_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_body_mix_weight_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_edge_l1_source_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_edge_l1_weight_count, 1, MPI_INT, 0, world);
  int sh_scalar_info_count = static_cast<int>(sh_scalar_body_order.size());
  MPI_Bcast(&sh_scalar_info_count, 1, MPI_INT, 0, world);
  if (comm->me != 0) sh_scalar_body_order.resize(sh_scalar_info_count);
  if (sh_scalar_info_count > 0)
    MPI_Bcast(sh_scalar_body_order.data(), sh_scalar_info_count, MPI_INT, 0, world);
  int two_layer_gate_radial_count =
      static_cast<int>(two_layer_gate_radial_coeffs.size());
  int two_layer_gate_additive_count =
      static_cast<int>(two_layer_gate_additive_coeffs.size());
  MPI_Bcast(&two_layer_gate_radial_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&two_layer_gate_additive_count, 1, MPI_INT, 0, world);
  if (comm->me != 0) {
    two_layer_gate_scalar_indices.resize(two_layer_gate_scalar_count);
    two_layer_gate_weights.resize(two_layer_gate_weight_count);
    two_layer_gate_body_mix_weights.resize(two_layer_gate_body_mix_weight_count);
    two_layer_gate_radial_coeffs.resize(two_layer_gate_radial_count);
    two_layer_gate_additive_coeffs.resize(two_layer_gate_additive_count);
    two_layer_gate_edge_l1_source_moment_indices.resize(
        two_layer_gate_edge_l1_source_count);
    two_layer_gate_edge_l1_weights.resize(
        two_layer_gate_edge_l1_weight_count);
  }
  if (two_layer_gate_scalar_count > 0)
    MPI_Bcast(two_layer_gate_scalar_indices.data(), two_layer_gate_scalar_count,
              MPI_INT, 0, world);
  if (two_layer_gate_weight_count > 0) {
    MPI_Bcast(two_layer_gate_weights.data(), two_layer_gate_weight_count,
              MPI_DOUBLE, 0, world);
  }
  if (two_layer_gate_body_mix_weight_count > 0) {
    MPI_Bcast(two_layer_gate_body_mix_weights.data(),
              two_layer_gate_body_mix_weight_count, MPI_DOUBLE, 0, world);
  }
  if (two_layer_gate_edge_l1_source_count > 0) {
    MPI_Bcast(two_layer_gate_edge_l1_source_moment_indices.data(),
              two_layer_gate_edge_l1_source_count, MPI_INT, 0, world);
  }
  if (two_layer_gate_edge_l1_weight_count > 0) {
    MPI_Bcast(two_layer_gate_edge_l1_weights.data(),
              two_layer_gate_edge_l1_weight_count, MPI_DOUBLE, 0, world);
  }
  if (two_layer_gate_enabled) {
    if (sh_scalar_body_order.empty())
      error->all(FLERR, "SUS2-SH mu-body-order gate requires sh_scalar_info metadata.");
    if (two_layer_gate_body_order_max != sh_k_max + 1)
      error->all(FLERR, "SUS2-SH mu-body-order gate body order should be sh_k_max + 1.");
    const int expected_gate_weight_count =
        two_layer_gate_full_scalar_weights
            ? radial_func_count * two_layer_gate_scalar_count
            : two_layer_gate_scalar_count;
    const int expected_gate_body_mix_count =
        two_layer_gate_body_linear_combo
            ? radial_func_count * (two_layer_gate_body_order_max - 1)
            : 0;
    if (radial_func_count <= 0 ||
        two_layer_gate_weight_count != expected_gate_weight_count ||
        two_layer_gate_body_mix_weight_count != expected_gate_body_mix_count)
      error->all(FLERR, "SUS2-SH two-layer gate metadata has inconsistent sizes.");
    two_layer_gate_scalar_body_order.assign(two_layer_gate_scalar_count, 0);
    std::vector<int> body_order_counts(two_layer_gate_body_order_max + 1, 0);
    for (int q = 0; q < two_layer_gate_scalar_count; q++) {
      const int scalar_index = two_layer_gate_scalar_indices[q];
      if (scalar_index < 0 || scalar_index >= alpha_scalar_count)
        error->all(FLERR, "SUS2-SH two-layer gate scalar index is out of range.");
      const int body_order = sh_scalar_body_order[scalar_index];
      if (body_order < 2 || body_order > two_layer_gate_body_order_max)
        error->all(FLERR, "SUS2-SH mu-body-order gate scalar index has wrong body order.");
      two_layer_gate_scalar_body_order[q] = body_order;
      body_order_counts[body_order]++;
    }
    for (int body_order = 2; body_order <= two_layer_gate_body_order_max;
         body_order++)
      if (body_order_counts[body_order] == 0)
        error->all(FLERR, "SUS2-SH mu-body-order gate is missing a required scalar body-order bucket.");
    two_layer_gate_mu_body_order.assign(radial_func_count, 0);
    for (int mu = 0; mu < radial_func_count; mu++) {
      const int body_order = mu / L_max + 2;
      if (body_order < 2 || body_order > two_layer_gate_body_order_max)
        error->all(FLERR, "SUS2-SH mu-body-order gate has an invalid mu body order.");
      two_layer_gate_mu_body_order[mu] = body_order;
    }
    if (two_layer_gate_edge_l1_enabled) {
      if (two_layer_gate_center_enabled)
        error->all(FLERR, "SUS2-SH edge L1 gate supports neighbor site mode only in LAMMPS.");
      if (sh_l_max < 1)
        error->all(FLERR, "SUS2-SH edge L1 gate requires sh_l_max >= 1.");
      const int expected_edge_l1_weight_count =
          radial_func_count * two_layer_gate_edge_l1_source_count;
      if (two_layer_gate_edge_l1_source_count <= 0 ||
          two_layer_gate_edge_l1_weight_count != expected_edge_l1_weight_count ||
          static_cast<int>(two_layer_gate_edge_l1_source_moment_indices.size()) !=
              two_layer_gate_edge_l1_source_count ||
          static_cast<int>(two_layer_gate_edge_l1_weights.size()) !=
              two_layer_gate_edge_l1_weight_count)
        error->all(FLERR, "SUS2-SH edge L1 gate metadata has inconsistent sizes.");
    }
  }
  if (two_layer_gate_radial_count > 0)
    MPI_Bcast(two_layer_gate_radial_coeffs.data(), two_layer_gate_radial_count,
              MPI_DOUBLE, 0, world);
  if (two_layer_gate_additive_count > 0)
    MPI_Bcast(two_layer_gate_additive_coeffs.data(), two_layer_gate_additive_count,
              MPI_DOUBLE, 0, world);

  // Broadcast scaling_map string
  int scaling_map_len = scaling_map.length();
  MPI_Bcast(&scaling_map_len, 1, MPI_INT, 0, world);
  if (comm->me != 0) scaling_map.resize(scaling_map_len);
  MPI_Bcast(&scaling_map[0], scaling_map_len, MPI_CHAR, 0, world);

  // ---------- Now all procs have correct parameter values ----------
  // Pre-allocate arrays on ALL procs (including proc 0)
  int np1 = ((atom && atom->ntypes > species_count) ? atom->ntypes : species_count) + 1;

  //Working buffers
  memory->create(inv_dist_powers, max_alpha_index_basic, "inv_dist_powers");
  memory->create(coord_powers_x, max_alpha_index_basic, "coord_powers_x");
  memory->create(coord_powers_y, max_alpha_index_basic, "coord_powers_y");
  memory->create(coord_powers_z, max_alpha_index_basic, "coord_powers_z");
  memory->create(alpha_basic_mu, alpha_index_basic_count, "alpha_basic_mu");
  memory->create(alpha_basic_a0, alpha_index_basic_count, "alpha_basic_a0");
  memory->create(alpha_basic_a1, alpha_index_basic_count, "alpha_basic_a1");
  memory->create(alpha_basic_a2, alpha_index_basic_count, "alpha_basic_a2");
  memory->create(alpha_basic_norm_rank, alpha_index_basic_count, "alpha_basic_norm_rank");
  memory->create(alpha_basic_sh_index, alpha_index_basic_count, "alpha_basic_sh_index");
  memory->create(alpha_times_a0, alpha_index_times_count, "alpha_times_a0");
  memory->create(alpha_times_a1, alpha_index_times_count, "alpha_times_a1");
  memory->create(alpha_times_multiplier, alpha_index_times_count, "alpha_times_multiplier");
  memory->create(alpha_times_coeff, alpha_index_times_count, "alpha_times_coeff");
  if (comm->me == 0) {
    for (int i = 0; i < alpha_index_times_count; i++)
      alpha_times_coeff[i] = alpha_times_coeff_buffer[i];
  }
  memory->create(alpha_times_out, alpha_index_times_count, "alpha_times_out");
  memory->create(weighted_basic_moment_ders, alpha_index_basic_count, "weighted_basic_moment_ders");
  
  // Allocate radial_vals/ders for final storage only (one value per mu)
  // Temporary arrays are now created locally in compute() function
  memory->create(radial_vals, radial_func_count, "radial_vals");
  memory->create(radial_ders, radial_func_count, "radial_ders");
  memory->create(two_layer_gate_residual_radial_vals, radial_func_count,
                 "two_layer_gate_residual_radial_vals");
  memory->create(two_layer_gate_center_residual_radial_vals, radial_func_count,
                 "two_layer_gate_center_residual_radial_vals");
  memory->create(two_layer_gate_adjoint_scratch, radial_func_count,
                 "two_layer_gate_adjoint_scratch");
  memory->create(two_layer_gate_center_adjoint_scratch, radial_func_count,
                 "two_layer_gate_center_adjoint_scratch");


  // Now allocate arrays that depend on broadcasted values
  if (comm->me != 0) {    // Non-zero proc
    if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV) {
      radial_basis = new RBChebyshev(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_S) {
      radial_basis = new RBChebyshev_s(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SS) {
      radial_basis = new RBChebyshev_ss(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
	    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS) {
	      radial_basis = new RBChebyshev_sss(radial_basis_size, lmp);
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS_RATIONAL) {
	      radial_basis = new RBChebyshev_sss_rational(radial_basis_size, lmp);
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP) {
	      radial_basis = new RBChebyshev_sss_lmp(radial_basis_size, lmp);
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS_RATIONAL_LMP) {
	      radial_basis = new RBChebyshev_sss_rational_lmp(radial_basis_size, lmp);
	      radial_basis->scaling = scaling;
	    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P) {
      radial_basis = new RBLaguerre_log1p(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP) {
      radial_basis = new RBLaguerre_log1p_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS) {
      radial_basis = new RBLaguerre_log1p_pos(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP) {
      radial_basis = new RBLaguerre_log1p_pos_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::JACOBI_SSS_LMP) {
      radial_basis = new RBJacobi_sss_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    }
    //Flags
    memory->create(cutsq, np1, np1, "pair:cutsq");
    memory->create(setflag, np1, np1, "pair:setflag");
    for (int i = 0; i < np1; ++i) {
      for (int j = 0; j < np1; ++j) {
        cutsq[i][j] = 0.0;
        setflag[i][j] = 0;
      }
    }

    //Alpha indicies
    memory->create(alpha_index_basic, alpha_index_basic_count, 4, "alpha_index_basic");
    memory->create(alpha_index_times, alpha_index_times_count, 4, "alpha_index_times");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");

    //Working buffers
    memory->create(moment_tensor_vals, alpha_moment_count, "moment_tensor_vals");
    memory->create(nbh_energy_ders_wrt_moments, alpha_moment_count, "nbh_energy_ders_wrt_moments");
    // CRITICAL FIX: Initialize Jacobian pointers to nullptr for all MPI procs
    // These arrays will be allocated with memory->grow during calculation
    moment_jacobian_x = nullptr;
    moment_jacobian_y = nullptr;
    moment_jacobian_z = nullptr;
    within_cutoff = nullptr;
    radial_list = nullptr;
    radial_der_list = nullptr;
    two_layer_gate_radial_list = nullptr;
    two_layer_gate_radial_der_list = nullptr;
    jac_size = 0;

    //Coefficients - allocate unified array
    memory->create(regression_coeffs, regression_coeffs_count, "regression_coeffs");
    
    // Set offset pointers
    shift_coeffs_offset = 0;
    scal_coeffs_offset = species_count;
    int pairs_count = species_count * species_count;
    int scal_coeffs_count_local = 2 * pairs_count * K_scaling;
    radial_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local;
    const int radial_coeff_count_local = radial_func_count * (radial_basis_size + species_count);
    env_gate_coeffs_offset = radial_coeffs_offset + radial_coeff_count_local;
    env_gate_lambda_raw_offset = env_gate_coeffs_offset;
    env_gate_log_density_coeffs_offset = env_gate_coeffs_offset + species_count;
    
    shift_coeffs = &regression_coeffs[shift_coeffs_offset];
    scal_coeffs = &regression_coeffs[scal_coeffs_offset];
    radial_basis_coeffs = &regression_coeffs[radial_coeffs_offset];
    
    memory->create(linear_coeffs, alpha_scalar_count, "linear_coeffs");
    memory->create(species_coeffs, species_count, "species_coeffs");
    
    // SUS2-MLIP mapping arrays
    memory->create(mu_to_K, radial_func_count, "mu_to_K");
    memory->create(mu_to_sigma, radial_func_count, "mu_to_sigma");
    
  }

  //We can then populate the cutoffs
  MPI_Bcast(&radial_basis->min_cutoff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&radial_basis->max_cutoff, 1, MPI_DOUBLE, 0, world);
  min_cutoff = radial_basis->min_cutoff;
  max_cutoff = radial_basis->max_cutoff;
  max_cutoff_sq = max_cutoff * max_cutoff;
  interaction_cutoff = max_cutoff;
  if (zbl_enabled) interaction_cutoff = std::max(interaction_cutoff, zbl_max_outer);
  interaction_cutoff_sq = interaction_cutoff * interaction_cutoff;

  //Now we B Cast into arrays
  //Flags
  MPI_Bcast(&cutsq[0][0], np1 * np1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&setflag[0][0], np1 * np1, MPI_INT, 0, world);

  // Alphas
  MPI_Bcast(&alpha_index_basic[0][0], alpha_index_basic_count * 4, MPI_INT, 0, world);
  if (alpha_index_times_count > 0) {
    MPI_Bcast(&alpha_index_times[0][0], alpha_index_times_count * 4, MPI_INT, 0, world);
    MPI_Bcast(alpha_times_coeff, alpha_index_times_count, MPI_DOUBLE, 0, world);
  }
  MPI_Bcast(alpha_moment_mapping, alpha_scalar_count, MPI_INT, 0, world);
  if (two_layer_gate_enabled)
    prepare_two_layer_gate_weight_layouts();
  for (int i = 0; i < alpha_index_basic_count; i++) {
    alpha_basic_mu[i] = alpha_index_basic[i][0];
    alpha_basic_a0[i] = alpha_index_basic[i][1];
    alpha_basic_a1[i] = alpha_index_basic[i][2];
    alpha_basic_a2[i] = alpha_index_basic[i][3];
    alpha_basic_norm_rank[i] = alpha_basic_a0[i] + alpha_basic_a1[i] + alpha_basic_a2[i];
    alpha_basic_sh_index[i] =
        is_sh_model ? sh_flat_index(alpha_basic_a0[i], alpha_basic_a1[i]) : 0;
  }
  sh_basic_mu_grouped = false;
  sh_basic_mu_indexed = false;
  sh_basic_mu_adjoint_indexed = false;
  sh_basic_mu_offsets.clear();
  sh_basic_mu_indices.clear();
  sh_basic_mu_adjoint_offsets.clear();
  sh_basic_mu_adjoint_indices.clear();
  if (is_sh_model && radial_func_count > 0) {
    sh_basic_mu_offsets.assign(radial_func_count + 1, 0);
    int pos = 0;
    bool grouped = true;
    for (int mu = 0; mu < radial_func_count; mu++) {
      sh_basic_mu_offsets[mu] = pos;
      while (pos < alpha_index_basic_count && alpha_basic_mu[pos] == mu) pos++;
      sh_basic_mu_offsets[mu + 1] = pos;
      if (pos < alpha_index_basic_count && alpha_basic_mu[pos] < mu)
        grouped = false;
    }
    if (pos != alpha_index_basic_count) grouped = false;
    if (grouped) sh_basic_mu_grouped = true;
	    if (!grouped && alpha_index_basic_count > 0) {
	      sh_basic_mu_adjoint_offsets.assign(radial_func_count + 1, 0);
	      for (int k = 0; k < alpha_index_basic_count; k++) {
	        const int mu = alpha_basic_mu[k];
	        if (mu >= 0 && mu < radial_func_count)
          sh_basic_mu_adjoint_offsets[mu + 1]++;
      }
      for (int mu = 0; mu < radial_func_count; mu++)
        sh_basic_mu_adjoint_offsets[mu + 1] +=
            sh_basic_mu_adjoint_offsets[mu];
      sh_basic_mu_adjoint_indices.assign(alpha_index_basic_count, 0);
      std::vector<int> cursor = sh_basic_mu_adjoint_offsets;
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const int mu = alpha_basic_mu[k];
	        if (mu >= 0 && mu < radial_func_count)
	          sh_basic_mu_adjoint_indices[cursor[mu]++] = k;
	      }
	      sh_basic_mu_adjoint_indexed = true;
	      if (env_flag_enabled("SUS2_LAMMPS_GATE_MU_INDEXED_BASIC") ||
	          env_flag_enabled("SUS2_SH_GATE_MU_INDEXED_BASIC")) {
	        sh_basic_mu_offsets = sh_basic_mu_adjoint_offsets;
	        sh_basic_mu_indices = sh_basic_mu_adjoint_indices;
	        sh_basic_mu_indexed = true;
	      }
	    }
	  }
  for (int i = 0; i < alpha_index_times_count; i++) {
    alpha_times_a0[i] = alpha_index_times[i][0];
    alpha_times_a1[i] = alpha_index_times[i][1];
    alpha_times_multiplier[i] = alpha_index_times[i][2];
    if (!is_sh_model) alpha_times_coeff[i] = static_cast<double>(alpha_times_multiplier[i]);
    alpha_times_out[i] = alpha_index_times[i][3];
  }
  if (two_layer_gate_enabled) {
    prepare_two_layer_gate_edge_l1_layout();
    prepare_two_layer_gate_product_layout();
  }

  //Working buffers
  //Preassign constant values for dist powers and coord powers. Other buffers can be uninited.
  inv_dist_powers[0] = 1.0;
  if (max_alpha_index_basic > 0) {
    coord_powers_x[0] = 1.0;
    coord_powers_y[0] = 1.0;
    coord_powers_z[0] = 1.0;
  }

  // Coefficients - broadcast the entire unified regression_coeffs array
  MPI_Bcast(regression_coeffs, regression_coeffs_count, MPI_DOUBLE, 0, world);
  MPI_Bcast(linear_coeffs, alpha_scalar_count, MPI_DOUBLE, 0, world);
  MPI_Bcast(species_coeffs, species_count, MPI_DOUBLE, 0, world);
  
  // SUS2-MLIP mapping arrays
  MPI_Bcast(mu_to_K, radial_func_count, MPI_INT, 0, world);
  MPI_Bcast(mu_to_sigma, radial_func_count, MPI_INT, 0, world);

  prepare_two_layer_gate_additive_ratios();

  do_list = false;
  list_grid_size = 0;
  actual_dr = 0.0;
  inv_dr = 0.0;
  total_list_elements = 0;
  used_species_count = species_count;
  used_pair_count = species_count * species_count;
  memory->create(pair_to_table_index, species_count * species_count, "pair_to_table_index");

  std::vector<int> used_species_flags(species_count, 1);
  if (atom && atom->type && atom->natoms > 0) {
    std::fill(used_species_flags.begin(), used_species_flags.end(), 0);
    for (int i = 0; i < atom->nlocal; i++) {
      const int atype = atom->type[i] - 1;
      if (atype >= 0 && atype < species_count) {
        used_species_flags[atype] = 1;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, used_species_flags.data(), species_count, MPI_INT, MPI_MAX, world);

    used_species_count = 0;
    for (int i = 0; i < species_count; i++) {
      used_species_count += used_species_flags[i];
    }
    if (used_species_count == 0) {
      std::fill(used_species_flags.begin(), used_species_flags.end(), 1);
      used_species_count = species_count;
    }
  }

  used_pair_count = 0;
  for (int i = 0; i < species_count; i++) {
    for (int j = 0; j < species_count; j++) {
      const int shift = i * species_count + j;
      if (used_species_flags[i] && used_species_flags[j]) {
        pair_to_table_index[shift] = used_pair_count++;
      } else {
        pair_to_table_index[shift] = -1;
      }
    }
  }

  if (uses_preinterpolation_table(radial_basis_type_index)) {
    do_list = true;
    list_grid_size = static_cast<int>(std::ceil(max_cutoff / requested_tabstep)) + 1;
    if (list_grid_size < 2) list_grid_size = 2;
    actual_dr = max_cutoff / static_cast<double>(list_grid_size - 1);
    inv_dr = 1.0 / actual_dr;

    if (comm->me == 0) {
      std::ostringstream used_species_stream;
      bool first_used_species = true;
      for (int i = 0; i < species_count; i++) {
        if (!used_species_flags[i]) continue;
        if (!first_used_species) used_species_stream << ", ";
        used_species_stream << (i + 1);
        first_used_species = false;
      }
      utils::logmesg(
          lmp,
          "Building Radial List, requested_tabstep={} A, actual_dr={} A, grid_points={}, max_cutoff={} A\n",
          requested_tabstep, actual_dr, list_grid_size, max_cutoff);
      utils::logmesg(
          lmp,
          "  - Precomputing {} of {} species pairs from used atom types [{}]\n",
          used_pair_count, species_count * species_count, used_species_stream.str());
    }

    memory->create(radial_list, used_pair_count, list_grid_size, radial_func_count,
                   "radial_list");
    memory->create(radial_der_list, used_pair_count, list_grid_size, radial_func_count,
                   "radial_der_list");
    total_list_elements =
        static_cast<size_t>(used_pair_count) * list_grid_size * radial_func_count;
    std::fill_n(&radial_list[0][0][0], total_list_elements, 0.0);
    std::fill_n(&radial_der_list[0][0][0], total_list_elements, 0.0);

    const int C = species_count;
    const int R = radial_basis_size;
    const double dr = actual_dr;
    double factor;
    for (int i = 0; i < C; i++) {
      for (int j = 0; j < C; j++) {
        const int table_index = pair_to_table_index[i * C + j];
        if (table_index < 0) continue;
        const double center_type_coeff =
            regression_coeffs[C + 2 * C * C * K_scaling + R + i];
        const double outer_type_coeff =
            regression_coeffs[C + 2 * C * C * K_scaling + R + j];
        for (int n = 0; n < list_grid_size; n++) {
          const double dist = dr * n;
          for (int mu = 0; mu < radial_func_count; mu++) {
            const int k_ = mu_to_K[mu];
            const int sigma = mu_to_sigma[mu];

            radial_basis->calc_radial_basis_ders(
                dist, regression_coeffs[C + 2 * k_ * C * C + C * i + j],
                regression_coeffs[C + 2 * k_ * C * C + C * C + C * i + j], sigma);

            for (int xi = 0; xi < R; xi++) {
              factor = regression_coeffs[C + 2 * C * C * K_scaling + mu * (R + C) + xi] *
                       center_type_coeff * outer_type_coeff;
              radial_list[table_index][n][mu] += radial_basis->radial_basis_vals[xi] * factor;
              radial_der_list[table_index][n][mu] += radial_basis->radial_basis_ders[xi] * factor;
            }
          }
        }
      }
    }

    if (two_layer_gate_enabled && two_layer_gate_shared_radial) {
      const int expected_gate_radial_count = radial_func_count * R;
      if (static_cast<int>(two_layer_gate_radial_coeffs.size()) != expected_gate_radial_count)
        error->all(FLERR, "SUS2-SH two-layer gate radial coefficient storage is inconsistent.");
      memory->create(two_layer_gate_radial_list, used_pair_count, list_grid_size,
                     radial_func_count, "two_layer_gate_radial_list");
      memory->create(two_layer_gate_radial_der_list, used_pair_count, list_grid_size,
                     radial_func_count, "two_layer_gate_radial_der_list");
      std::fill_n(&two_layer_gate_radial_list[0][0][0], total_list_elements, 0.0);
      std::fill_n(&two_layer_gate_radial_der_list[0][0][0], total_list_elements, 0.0);

      for (int i = 0; i < C; i++) {
        for (int j = 0; j < C; j++) {
          const int table_index = pair_to_table_index[i * C + j];
          if (table_index < 0) continue;
          const double species_factor =
              regression_coeffs[C + 2 * C * C * K_scaling + R + i] *
              regression_coeffs[C + 2 * C * C * K_scaling + R + j];
          for (int n = 0; n < list_grid_size; n++) {
            const double dist = dr * n;
            for (int mu = 0; mu < radial_func_count; mu++) {
              const int k_ = mu_to_K[mu];
              const int sigma = mu_to_sigma[mu];
              radial_basis->calc_radial_basis_ders(
                  dist, regression_coeffs[C + 2 * k_ * C * C + C * i + j],
                  regression_coeffs[C + 2 * k_ * C * C + C * C + C * i + j], sigma);
              for (int xi = 0; xi < R; xi++) {
                const double factor =
                    two_layer_gate_radial_coeffs[mu * R + xi] * species_factor;
                two_layer_gate_radial_list[table_index][n][mu] +=
                    radial_basis->radial_basis_vals[xi] * factor;
                two_layer_gate_radial_der_list[table_index][n][mu] +=
                    radial_basis->radial_basis_ders[xi] * factor;
              }
            }
          }
        }
      }
    }

    if (env_gate_enabled) {
      memory->create(env_gate_rho_list, C, list_grid_size, "env_gate_rho_list");
      memory->create(env_gate_rho_der_list, C, list_grid_size, "env_gate_rho_der_list");
      for (int i = 0; i < C; i++) {
        for (int n = 0; n < list_grid_size; n++) {
          env_gate_rho_list[i][n] = 0.0;
          env_gate_rho_der_list[i][n] = 0.0;
        }
      }

      const double r_env = env_gate_cutoff_ratio * max_cutoff;
      if (r_env <= 0.0) error->all(FLERR, "Invalid env_gate_cutoff_ratio in SUS2 model.");
      for (int i = 0; i < C; i++) {
        double density_coeffs[6];
        for (int q = 0; q < env_gate_channel_count; ++q)
          density_coeffs[q] =
              env_gate_density_coeff(regression_coeffs[env_gate_log_density_coeffs_offset + i * env_gate_channel_count + q]);
        for (int n = 0; n < list_grid_size; n++) {
          const double dist = dr * n;
          if (dist <= 0.0 || dist >= r_env) continue;
          const double y = dist / r_env;
          const double dy_dr = 1.0 / r_env;
          const double cutoff = 1.0 - dist / r_env;
          const double f_env = cutoff * cutoff;
          const double df_dr = -2.0 * cutoff / r_env;
          double basis[6];
          double basis_der[6];
          bernstein_degree5(y, basis, basis_der);
          for (int q = 0; q < env_gate_channel_count; ++q) {
            const double weighted_basis = f_env * basis[q];
            const double weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
            env_gate_rho_list[i][n] += density_coeffs[q] * weighted_basis;
            env_gate_rho_der_list[i][n] += density_coeffs[q] * weighted_basis_der;
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (comm->me == 0) {
      utils::logmesg(lmp, "Radial List Completed\n");
    }
  } else if (tabstep_set_by_user && comm->me == 0) {
    utils::logmesg(
        lmp,
        "Ignoring tabstep={} A because radial_basis_type={} does not use the preinterpolation table.\n",
        requested_tabstep, radial_basis_type_str);
  }

  allocated = 1;
}
