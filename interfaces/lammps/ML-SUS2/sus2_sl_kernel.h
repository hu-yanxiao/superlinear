#ifndef LMP_SUS2_SL_KERNEL_H
#define LMP_SUS2_SL_KERNEL_H
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
#include "sus2_sl_joint.h"
#include "sus2_sl_adaptive.h"
#include "sus2_sl_zbl.h"
#include <memory>
#include <limits>
namespace LAMMPS_NS { namespace SUS2SL {
// Edges are directed, owned-center full neighbors. d = x[j]-x[i].
struct Edge { int i,j; std::array<double,3> d; };
struct Result { double energy=0; std::vector<double> site_energy;
  std::vector<std::array<double,3> > force; std::array<double,6> virial{{0,0,0,0,0,0}}; };
class Kernel {
 public:
  explicit Kernel(const Model&);
  void configure_radial(RadialTableMode,double spacing=0.001,
                        std::size_t max_bytes=std::numeric_limits<std::size_t>::max());
  // Model species IDs, not compact LAMMPS type numbers. An explicitly empty
  // selection admits no frozen tables; an unset selection preserves standalone
  // callers' historical all-model-species behavior.
  void configure_species(std::vector<int>);
  int species_count() const { return model_.species_count; }
  int communication_width() const;
  double cutoff() const { return model_.cutoff(); }
  RadialTableStats radial_stats() const { return radial_.stats(); }
  double memory_usage() const;
  void configure_profile(bool enabled) { profile_enabled_=enabled; outer_profile_.fill(0); }
  // Per-call seconds: forward+contract, mixer/Adaptive forward, readout+graph
  // reverse, Legacy coordinate reverse, mixer/Adaptive coordinate reverse.
  const std::array<double,5>& outer_profile() const { return outer_profile_; }
  void begin(int nowned,int nall,const std::vector<int>& types,const std::vector<Edge>&);
  void pack_forward(int atom,double*) const;
  void unpack_forward(int atom,const double*);
  void outer();
  void pack_reverse(int atom,double*) const;
  void add_reverse(int atom,const double*);
  // The result remains valid until the next begin() or finish().
  const Result& finish();
 private:
  Model model_; Radial radial_; int owned_=0,all_=0;
  std::vector<int> types_; std::vector<Edge> edges_;
  std::vector<double> gate_moments_,gate_,gate_adj_,outer_moments_,outer_adj_;
  std::vector<double> weights_,linear_,mult_,gate_type_,gate_add_,gate_radial_;
  std::vector<double> scal_,gate_scal_,gate_rank_weights_,outer_rank_weights_,species_,bias_,body_mix_;
  std::vector<double> mixer_;
  std::vector<double> gate_readout_;
  bool zbl_=false;
  std::vector<SUS2SLZBLPairConstants> zbl_constants_;
  std::vector<double> zbl_inner_,zbl_outer_;
  std::unique_ptr<JointReadout> joint_;
  std::unique_ptr<Adaptive> adaptive_;
  std::vector<double> local_gate_adj_,extra_energy_;
  Result partial_;
  std::vector<int> gate_nodes_,body_;
  int gate_rank_=0,outer_rank_=0,gate_begin_=0,outer_begin_=0,outer_product_=0;
  bool gated_=false;
  bool pure_=false,mixing_=false;
  bool adaptive_mode_=false;
  int feature_width_=0;
  std::vector<int> feature_nodes_;
  int scaling_mode_=3,gate_product_begin_=0,gate_product_end_=0,body_count_=0;
  bool species_readout_=false,body_combo_=false,gate_readout_enabled_=false;
  std::vector<double> y_scratch_,dy_scratch_,rb_scratch_,dr_scratch_,rv_scratch_,rd_scratch_;
  std::vector<double> v_scratch_,d_scratch_,s_scratch_,gate_moment_adj_;
  std::vector<std::array<double,4>> edge_coeff_scratch_;
  std::vector<int> basic_mu_,basic_sh_;
  struct Contraction { int src,dst; double coefficient; };
  std::vector<Contraction> gate_contraction_,outer_contraction_;
  // Geometry/interpolation metadata only. Rebuilt for the current directed
  // edge list on every begin(); no atom identity or periodic image survives.
  struct EdgeMetadata { double distance=0; int pair=0; RadialInterpolation interpolation; };
  std::vector<EdgeMetadata> edge_metadata_;
  bool metadata_cache_enabled_=false;
  bool frozen_enabled_=false;
  bool species_configured_=false,edge_metadata_requested_=false;
  std::vector<int> active_species_;
  RadialTableMode radial_mode_=RadialTableMode::Direct;
  bool reverse_stream_=true;
  bool profile_enabled_=false;
  std::array<double,5> outer_profile_{};
  EdgeMetadata* edge_metadata(const Edge&);
  template<bool Frozen,bool Derivatives,bool Materialize=true>
  double edge_basis(const Edge&,bool,std::vector<double>&,std::vector<double>&);
  template<bool Frozen,bool Derivatives>
  void edge_mixer(const Edge&,std::vector<double>&,std::vector<double>&,std::vector<double>&);
  template<bool Frozen> void begin_impl(int,int,const std::vector<int>&,const std::vector<Edge>&);
  template<bool Frozen> void outer_impl();
  template<bool Frozen> const Result& finish_impl();
  template<bool Gate> void prepare_edge_coefficients(const Edge&);
  template<bool Frozen,bool Stream,bool Gate>
  void coordinate_reverse(const std::vector<double>&,Result&);
  void contract(double*,bool,bool reverse=false) const;
  void graph(double*,bool,bool reverse=false,const double* values=nullptr) const;
  double gate_weight(int,int) const;
  void add_force(Result&,const Edge&,const std::array<double,3>&) const;
};
} }
#endif
