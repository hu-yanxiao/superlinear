// Package-local inference radial evaluator. No training-library dependency.
#ifndef SUS2_SL_RADIAL_H
#define SUS2_SL_RADIAL_H
#include "sus2_sl_radial_math.h"
#include <memory>
#include <map>
#include <array>
namespace LAMMPS_NS { namespace SUS2SL {
struct Model;
using RadialTableMode = radial_math::RadialTableMode;
using RadialTableStats = radial_math::RadialTableStats;
// Caller invalidates this when geometry or radial configuration changes.
struct RadialInterpolation {
  std::size_t interval=0;
  std::array<double,8> weights{};
  bool ready=false;
};
class Radial {
 public:
  Radial(const std::string& type, double min_dist, double max_dist, int size,
         double scaling = 1.0, const std::string& cutoff = "legacy", int p = 5);
  void configure(RadialTableMode mode, double spacing = 0.001,
                 std::size_t max_bytes = std::numeric_limits<std::size_t>::max());
  // With derivatives=false only vals is refreshed; ders remains unchanged.
  void eval(double r, double slope, double shift, int k,
            std::vector<double>& vals, std::vector<double>& ders,
            bool derivatives = true);
  // Frozen hits/fallbacks count scalar contraction requests; generic counts
  // count basis-vector requests. tables/bytes describe resident knot storage.
  RadialTableStats stats() const;
  // Frozen Legacy coefficient contractions. Dynamic Adaptive coefficients must
  // continue to use eval(). Calling configure invalidates knots, not requests.
  void freeze(const Model& model);
  // Only ordered pairs of these original model species receive requests.
  // An explicitly empty set creates no frozen requests; freeze(model) uses all.
  void freeze(const Model& model, const std::vector<int>& used_species);
  bool eval_frozen(double r, int pair, bool gate,
                   std::vector<double>& vals, std::vector<double>& ders,
                   RadialInterpolation* interpolation=nullptr, bool derivatives=true);
  bool frozen_available() const { return mode_!=RadialTableMode::Direct&&!frozen_ids_.empty(); }
  int size() const { return basis_->rb_size; }
 private:
  std::unique_ptr<radial_math::AnyRadialBasis> basis_;
  struct Frozen {
    double slope=1, shift=0, step=0;
    int channel=0, status=0; // 0 unbuilt, 1 table, 2 invalid, 3 capacity
    std::size_t intervals=0, tile=0, lane=0;
    bool packed=false;
    std::vector<double> coeff, knots;
  };
  struct Tile {
    std::size_t lanes=0;
    // [grid point][occupied lane][value, derivative], no padding lanes.
    std::vector<double> knots;
  };
  std::vector<Frozen> frozen_;
  std::vector<Tile> frozen_tiles_;
  std::vector<std::size_t> frozen_ids_;
  std::vector<int> frozen_species_slots_;
  std::size_t frozen_pairs_=0, frozen_model_pairs_=0;
  std::size_t frozen_used_species_=0;
  int frozen_model_species_=0, frozen_funcs_=0;
  RadialTableMode mode_=RadialTableMode::Direct;
  double spacing_=0.001;
  std::size_t budget_=std::numeric_limits<std::size_t>::max();
  RadialTableStats frozen_stats_, retired_stats_;
  bool tiled_=true;
  bool bulk_=true;
  std::size_t tile_width_=4;
  void build_frozen(Frozen& entry);
  void pack_frozen(Frozen& entry);
  void reset_generic_budget();
};
}}
#endif
