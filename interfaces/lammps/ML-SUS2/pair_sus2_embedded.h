/* Self-contained CPU SUS2 superlinear package. No native/training library. */
#ifdef PAIR_CLASS
PairStyle(sus2/embedded,PairSUS2Embedded);
#else
#ifndef LMP_PAIR_SUS2_EMBEDDED_H
#define LMP_PAIR_SUS2_EMBEDDED_H
#include "pair.h"
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace LAMMPS_NS {
class PairSUS2Embedded : public Pair {
 public:
  explicit PairSUS2Embedded(class LAMMPS *);
  ~PairSUS2Embedded() override;
  void compute(int,int) override;
  void settings(int,char**) override;
  void coeff(int,char**) override;
  void init_style() override;
  double init_one(int,int) override;
  int pack_forward_comm(int,int*,double*,int,int*) override;
  void unpack_forward_comm(int,int,double*) override;
  int pack_reverse_comm(int,int,double*) override;
  void unpack_reverse_comm(int,int*,double*) override;
  double memory_usage() override;
 private:
  struct Embedded;
  std::unique_ptr<Embedded> embedded_;
  std::vector<int> type_map_;
  std::vector<std::string> model_elements_;
  std::string model_file_;
  std::string radial_mode_="direct";
  double tabstep_=1e-3,cutoff_=0;
  std::size_t table_max_bytes_=64*1024*1024;
  bool radial_reported_=false;
  int profile_every_=0;
  unsigned long long profile_calls_=0;
  std::array<double,7> profile_seconds_{};
  std::array<double,5> profile_outer_seconds_{};
  std::array<unsigned long long,5> profile_radial_previous_{};
  void report_profile();
  void allocate();
};
}
#endif
#endif
