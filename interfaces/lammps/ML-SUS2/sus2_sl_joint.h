#ifndef LMP_SUS2_SL_JOINT_H
#define LMP_SUS2_SL_JOINT_H
#include "sus2_sl_model.h"
namespace LAMMPS_NS { namespace SUS2SL {
// Joint invariant head on the means of post-rank outer and gate harmonics.
// Both sources and both accumulating adjoint arrays use full Model::moments
// storage for one owned atom. Gate adjoints are local moment adjoints, not
// the communicated neighbor-gate scalar adjoints.
class JointReadout {
 public:
  explicit JointReadout(const Model&);
  bool enabled() const { return enabled_; }
  std::size_t memory_usage() const;
  double evaluate(int type,double center,const double* outer,const double* gate,
                  double* outer_adj,double* gate_adj) const;
 private:
  bool enabled_=false,species_=false;
  int lmax_=0,outer_rank_=0,gate_rank_=0,outer_begin_=0,gate_begin_=0;
  int moments_=0,scalar_count_=0;
  std::vector<std::array<int,3>> basic_;
  std::vector<Product> products_;
  std::vector<int> mapping_;
  std::vector<double> coeffs_;
  // One workspace per readout instance; evaluate() is not concurrently reentrant.
  mutable std::vector<double> values_,adj_;
  int source_node(int source,int channel,int l,int m) const;
};
}}
#endif
