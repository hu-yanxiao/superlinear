/* Native SUS2-SH superlinear CPU adapter. See ../README.superlinear-cpu.md. */
#ifdef PAIR_CLASS
PairStyle(sus2/superlinear,PairSUS2Superlinear);
#else
#ifndef LMP_PAIR_SUS2_SUPERLINEAR_H
#define LMP_PAIR_SUS2_SUPERLINEAR_H

#include "pair.h"
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {
class PairSUS2Superlinear : public Pair {
 public:
  explicit PairSUS2Superlinear(class LAMMPS *);
  ~PairSUS2Superlinear() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

 private:
  struct Native;
  std::unique_ptr<Native> native_;
  std::vector<int> type_map_;
  std::vector<std::string> model_elements_;
  std::string radial_mode_ = "direct";
  double tabstep_ = 1.0e-3;
  std::size_t table_max_bytes_ = 64 * 1024 * 1024;
  bool radial_reported_ = false;
  double cutoff_ = 0.0;
  void allocate();
};
}
#endif
#endif
