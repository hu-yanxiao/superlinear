#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
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
  Require(static_cast<bool>(input), "failed to open source-guard file");
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
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

bool SameBits(double left, double right)
{
  std::uint64_t left_bits = 0;
  std::uint64_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

enum class SignalMode { kBodyCombo, kFullIdentity, kGrouped };

double GateSignal(const std::vector<double> &values, int atom, int mu,
                  int stride, int mu_count, SignalMode mode,
                  const std::vector<double> &mix,
                  const std::vector<int> &group)
{
  const std::size_t offset = static_cast<std::size_t>(atom) * stride;
  if (mode == SignalMode::kBodyCombo) {
    double signal = 0.0;
    const std::size_t weight_offset = static_cast<std::size_t>(mu) * stride;
    for (int s = 0; s < stride; ++s)
      signal += mix[weight_offset + s] * values[offset + s];
    return signal;
  }
  if (mode == SignalMode::kFullIdentity) {
    Require(stride >= mu_count, "identity signal stride is too small");
    return values[offset + mu];
  }
  return values[offset + group[mu]];
}

void AccumulateAdjoint(std::vector<double> &adjoints, int atom, int mu,
                       int stride, SignalMode mode, double seed,
                       const std::vector<double> &mix,
                       const std::vector<int> &group)
{
  const std::size_t offset = static_cast<std::size_t>(atom) * stride;
  if (mode == SignalMode::kBodyCombo) {
    const std::size_t weight_offset = static_cast<std::size_t>(mu) * stride;
    for (int s = 0; s < stride; ++s)
      adjoints[offset + s] += seed * mix[weight_offset + s];
  } else if (mode == SignalMode::kFullIdentity) {
    adjoints[offset + mu] += seed;
  } else {
    adjoints[offset + group[mu]] += seed;
  }
}

void CheckRandomOverlayEquivalence(SignalMode mode)
{
  constexpr int nlocal = 19;
  constexpr int nghost = 11;
  constexpr int nall = nlocal + nghost;
  constexpr int stride = 9;
  constexpr int mu_count = 7;
  std::mt19937_64 rng(0x535553324f564552ULL +
                      static_cast<unsigned>(mode));
  std::uniform_real_distribution<double> value_dist(-0.75, 0.75);
  std::uniform_int_distribution<int> owner_dist(0, nlocal - 1);

  std::vector<double> mix(mu_count * stride);
  for (double &value : mix) value = value_dist(rng);
  std::vector<int> group(mu_count);
  for (int mu = 0; mu < mu_count; ++mu) group[mu] = (3 * mu + 1) % stride;
  std::vector<double> additive(3 * mu_count);
  for (double &value : additive) value = value_dist(rng);
  std::vector<int> type(nall);
  for (int &value : type) value = 1 + static_cast<int>(rng() % 3);

  std::vector<double> separate_values(nall * stride, 0.0);
  for (int atom = 0; atom < nlocal; ++atom)
    for (int s = 0; s < stride; ++s)
      separate_values[static_cast<std::size_t>(atom) * stride + s] =
          value_dist(rng);
  std::vector<int> ghost_owner(nghost);
  for (int ghost = 0; ghost < nghost; ++ghost) {
    const int owner = owner_dist(rng);
    ghost_owner[ghost] = owner;
    const int atom = nlocal + ghost;
    for (int s = 0; s < stride; ++s)
      separate_values[static_cast<std::size_t>(atom) * stride + s] =
          separate_values[static_cast<std::size_t>(owner) * stride + s];
  }
  std::vector<double> overlay = separate_values;
  std::vector<double> separate_multipliers(nall * mu_count);
  std::vector<double> overlay_multipliers(nall * mu_count);
  for (int atom = 0; atom < nall; ++atom) {
    for (int mu = 0; mu < mu_count; ++mu) {
      const int coeff_index = (type[atom] - 1) * mu_count + mu;
      separate_multipliers[atom * mu_count + mu] =
          1.0 + additive[coeff_index] *
                    GateSignal(separate_values, atom, mu, stride, mu_count,
                               mode, mix, group);
      overlay_multipliers[atom * mu_count + mu] =
          1.0 + additive[coeff_index] *
                    GateSignal(overlay, atom, mu, stride, mu_count, mode,
                               mix, group);
    }
  }
  for (std::size_t i = 0; i < separate_multipliers.size(); ++i)
    Require(SameBits(separate_multipliers[i], overlay_multipliers[i]),
            "overlay changed a cached gate multiplier bit");

  std::vector<double> separate_adjoints(nall * stride, 0.0);
  std::fill(overlay.begin(), overlay.end(), 0.0);
  for (int atom = 0; atom < nall; ++atom) {
    for (int mu = 0; mu < mu_count; ++mu) {
      const double seed = value_dist(rng);
      AccumulateAdjoint(separate_adjoints, atom, mu, stride, mode, seed,
                        mix, group);
      AccumulateAdjoint(overlay, atom, mu, stride, mode, seed, mix, group);
    }
  }
  for (int ghost = 0; ghost < nghost; ++ghost) {
    const int atom = nlocal + ghost;
    const int owner = ghost_owner[ghost];
    for (int s = 0; s < stride; ++s) {
      separate_adjoints[static_cast<std::size_t>(owner) * stride + s] +=
          separate_adjoints[static_cast<std::size_t>(atom) * stride + s];
      overlay[static_cast<std::size_t>(owner) * stride + s] +=
          overlay[static_cast<std::size_t>(atom) * stride + s];
    }
  }
  for (std::size_t i = 0; i < overlay.size(); ++i)
    Require(SameBits(separate_adjoints[i], overlay[i]),
            "overlay changed a gate adjoint bit");
}

void CheckSourceGuards(const std::string &source, const std::string &header)
{
  Require(CountOccurrences(header,
                           "d_two_layer_gate_signal_storage;") == 1,
          "gate signal storage must have exactly one device View");
  Require(CountOccurrences(header,
                           "d_two_layer_gate_adjoint_storage;") == 1 &&
              CountOccurrences(header,
                               "two_layer_gate_kk_signal_overlay") == 1,
          "conditional gate adjoint storage metadata is invalid");
  Require(source.find("d_two_layer_gate_values") == std::string::npos &&
              header.find("d_two_layer_gate_values") == std::string::npos &&
              source.find("d_two_layer_gate_adjoints") == std::string::npos &&
              header.find("d_two_layer_gate_adjoints") == std::string::npos,
          "a separate gate value or adjoint View was reintroduced");
  Require(CountOccurrences(
              source,
              "MemKK::realloc_kokkos(d_two_layer_gate_signal_storage,") == 1,
          "gate signal storage must be allocated exactly once at setup");
  Require(CountOccurrences(
              source,
              "MemKK::realloc_kokkos(d_two_layer_gate_adjoint_storage,") == 2,
          "gate adjoint setup/reinit allocation paths are invalid");
  Require(CountOccurrences(
              source,
              "d_two_layer_gate_adjoint_storage = d_two_layer_gate_signal_storage;") == 2,
          "device pair gate signal overlay is not bound at init and resize");
  Require(CountOccurrences(
              source,
              "WithoutInitializing, d_two_layer_gate_signal_storage") == 1,
          "gate signal storage must have exactly one resize path");
  Require(CountOccurrences(
              source,
              "WithoutInitializing,\n\t\t                      d_two_layer_gate_adjoint_storage") == 1,
          "host-staged gate adjoint storage must have one resize path");
  Require(CountOccurrences(
              source,
              "Kokkos::deep_copy(d_two_layer_gate_adjoint_storage, 0.0);") == 2,
          "gate adjoint clear must have host-staged and overlay branches");

  const std::size_t init_style = source.find("::init_style()");
  const std::size_t overlay_select =
      source.find("const int use_gate_signal_overlay", init_style);
  const std::size_t settings = source.find("::settings(");
  Require(init_style != std::string::npos &&
              overlay_select != std::string::npos &&
              settings != std::string::npos && overlay_select < settings,
          "gate signal overlay must be selected in init_style");
  Require(source.find("execution_space == Device", overlay_select) !=
              std::string::npos &&
              source.find("forward_pair_comm_legacy", overlay_select) !=
                  std::string::npos &&
              source.find("reverse_pair_comm_legacy", overlay_select) !=
                  std::string::npos &&
              source.find("d_two_layer_gate_adjoint_storage.data() ==",
                          overlay_select) != std::string::npos,
          "gate signal overlay communication guards are incomplete");

  const std::size_t forward = source.find("comm->forward_comm(this);");
  const std::size_t cache = source.find("\"ComputeGateMuCache\"");
  const std::size_t early_clear = source.find(
      "Kokkos::deep_copy(d_two_layer_gate_adjoint_storage, 0.0);");
  const std::size_t late_clear = source.find(
      "Kokkos::deep_copy(d_two_layer_gate_adjoint_storage, 0.0);",
      early_clear + 1);
  const std::size_t reverse = source.find("comm->reverse_comm(this);");
  Require(forward != std::string::npos && cache != std::string::npos &&
              early_clear != std::string::npos &&
              late_clear != std::string::npos &&
              reverse != std::string::npos && early_clear < forward &&
              forward < cache && cache < late_clear && late_clear < reverse,
          "gate signal value-to-adjoint phase order is invalid");
  Require(source.find("if (!two_layer_gate_kk_signal_overlay)") <
              early_clear &&
              source.find("if (two_layer_gate_kk_signal_overlay)", cache) <
                  late_clear,
          "gate signal clears are not guarded by communication mode");

  Require(source.find("two_layer_gate_mu_signal_kk") == std::string::npos &&
              header.find("two_layer_gate_mu_signal_kk") == std::string::npos &&
              source.find("two_layer_gate_edge_l1_signal_kk") ==
                  std::string::npos &&
              header.find("two_layer_gate_edge_l1_signal_kk") ==
                  std::string::npos,
          "dead post-cache scalar-value helper was reintroduced");
  Require(source.find("d_two_layer_gate_edge_l1_weighted_values") !=
              std::string::npos &&
              source.find("d_two_layer_gate_edge_l1_weighted_adjoints") !=
                  std::string::npos,
          "edge-L1 weighted value/adjoint storage was removed");
  Require(source.find(
              "d_two_layer_gate_edge_l1_weighted_adjoints =") ==
              std::string::npos,
          "overlapping edge-L1 weighted arrays were incorrectly aliased");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 3, "expected pair cpp and header paths");
  CheckRandomOverlayEquivalence(SignalMode::kBodyCombo);
  CheckRandomOverlayEquivalence(SignalMode::kFullIdentity);
  CheckRandomOverlayEquivalence(SignalMode::kGrouped);
  CheckSourceGuards(ReadFile(argv[1]), ReadFile(argv[2]));
  std::cout << "LAMMPS_KK_GATE_SIGNAL_OVERLAY_CHECK: ok\n";
  return 0;
}
