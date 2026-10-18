#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

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

template <class ForceFloat>
double OldCachedDerivative(double additive_coeff)
{
  const ForceFloat additive_ratio = additive_coeff;
  const double stored_derivative = additive_ratio;
  return stored_derivative;
}

template <class ForceFloat>
double NewInlineDerivative(double additive_coeff)
{
  return static_cast<double>(static_cast<ForceFloat>(additive_coeff));
}

template <class Value>
bool SameBits(const Value &left, const Value &right)
{
  return std::memcmp(&left, &right, sizeof(Value)) == 0;
}

template <class ForceFloat>
void CheckRoundingEquivalence()
{
  const double values[] = {
      0.0,
      -0.0,
      1.0 + std::ldexp(1.0, -30),
      -1.0 - std::ldexp(1.0, -29),
      std::ldexp(1.0, -80),
      12345.6789012345};
  for (const double value : values) {
    const double old_cached = OldCachedDerivative<ForceFloat>(value);
    const double new_inline = NewInlineDerivative<ForceFloat>(value);
    Require(SameBits(old_cached, new_inline),
            "inline derivative changed cached derivative bits");

    const ForceFloat old_force_read = old_cached;
    const ForceFloat new_force_read = new_inline;
    Require(SameBits(old_force_read, new_force_read),
            "inline derivative changed F_FLOAT read bits");

    const ForceFloat edge_projection = static_cast<ForceFloat>(0.6171875);
    const double old_edge_product = old_cached * edge_projection;
    const double new_edge_product = new_inline * edge_projection;
    Require(SameBits(old_edge_product, new_edge_product),
            "inline derivative changed mixed-precision edge arithmetic bits");
  }
}

void CheckSourceGuard(const std::string &source, const std::string &header)
{
  Require(source.find("d_two_layer_gate_mu_derivs") == std::string::npos &&
              header.find("d_two_layer_gate_mu_derivs") == std::string::npos,
          "removed per-atom derivative View was reintroduced");
  Require(source.find("d_mu_derivs") == std::string::npos,
          "removed derivative cache alias or store was reintroduced");

  Require(CountOccurrences(header, "two_layer_gate_mu_deriv_kk") == 1,
          "inline derivative helper declaration is missing or duplicated");
  Require(CountOccurrences(source, "two_layer_gate_mu_deriv_kk") == 5,
          "inline derivative helper must have one definition and four reads");
  Require(CountOccurrences(source,
                           "two_layer_gate_mu_deriv_kk(jtype, mu)") == 4,
          "not all four derivative reads use the neighbor species");
  Require(header.find(
              "KOKKOS_INLINE_FUNCTION double two_layer_gate_mu_deriv_kk(") !=
              std::string::npos,
          "inline derivative helper no longer returns double");

  const std::regex edge_l1_read(
      R"(gate_multiplier\s*\+=\s*two_layer_gate_mu_deriv_kk\s*\(\s*jtype\s*,\s*mu\s*\)\s*\*\s*edge_projection\s*;)");
  Require(std::regex_search(source, edge_l1_read),
          "edge-L1 derivative read no longer uses the inline helper");

  const std::regex exact_rounding(
      R"(return\s+static_cast<double>\s*\(\s*static_cast<F_FLOAT>\s*\(\s*d_two_layer_gate_additive_coeffs\s*\(\s*jtype\s*\*\s*radial_func_count\s*\+\s*mu\s*\)\s*\)\s*\)\s*;)");
  Require(std::regex_search(source, exact_rounding),
          "inline derivative helper lost double-to-F_FLOAT-to-double rounding");

  Require(CountOccurrences(source,
                           "d_mu_multipliers(atom_index, mu) =") == 1,
          "gate multiplier cache store changed unexpectedly");
}

}  // namespace

int main(int argc, char **argv)
{
  Require(argc == 3,
          "usage: lammps_kk_gate_mu_deriv_check <pair.cpp> <pair.h>");
  CheckRoundingEquivalence<float>();
  CheckRoundingEquivalence<double>();
  CheckSourceGuard(ReadFile(argv[1]), ReadFile(argv[2]));
  std::cout << "lammps_kk_gate_mu_deriv_check passed\n";
  return 0;
}
