#ifndef LMP_SUS2_SL_ADAPTIVE_H
#define LMP_SUS2_SL_ADAPTIVE_H
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
namespace LAMMPS_NS { namespace SUS2SL {
class Adaptive {
 public:
  explicit Adaptive(const Model&);
  int signal_width() const;
  int cached_radial_width() const;
  int projection_width() const { return (!shared_n_ && !tanh_ && tl_==0)?rows_:0; }
  void project(const double*, double*) const;
  void prepare_coefficients(double*, int source_type) const;
  void project_reverse(const double*, double*) const;
  int feature_moment_index(int feature) const;
  void edge(Radial&,const std::array<double,3>& d,int type_i,int type_j,
            const double* signal,std::vector<double>& value,
            std::vector<double>& coord_der,std::vector<double>& signal_der, bool derivatives=true, const double* adj=nullptr, double* signal_adj=nullptr, const double* projected=nullptr, double* projection_adj=nullptr, double* cached_radial=nullptr, bool cached_ready=false) const;
 private:
  Model model_;
  int q_,o_,g_,tl_,latent_,rows_,width_,scaling_map_;
  bool shared_n_,species_,tanh_,coefficient_cache_=false;
  std::vector<double> amplitude_,weight_,tensor_,scale_;
  std::array<double,25> harmonic_norm_;
  mutable std::vector<double> y_,dy_,rb_,dr_,phi_,first_,rq_;
  mutable std::vector<std::vector<double>> basis_cache_, derivative_cache_;
  std::vector<int> gate_nodes_;
};
} }
#endif
