#ifndef LMP_SUS2_SL_ADAPTIVE_H
#define LMP_SUS2_SL_ADAPTIVE_H
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
namespace LAMMPS_NS { namespace SUS2SL {
class Adaptive {
 public:
  explicit Adaptive(const Model&);
  int signal_width() const;
  int feature_moment_index(int feature) const;
  void edge(Radial&,const std::array<double,3>& d,int type_i,int type_j,
            const double* signal,std::vector<double>& value,
            std::vector<double>& coord_der,std::vector<double>& signal_der) const;
 private:
  Model model_;
  int q_,o_,g_,tl_,latent_,rows_,width_,scaling_map_;
  bool shared_n_,species_,tanh_;
  std::vector<double> amplitude_,weight_,tensor_,scale_;
  std::array<double,25> harmonic_norm_;
  mutable std::vector<double> y_,dy_,rb_,dr_,phi_,first_,rq_;
  std::vector<int> gate_nodes_;
};
} }
#endif
