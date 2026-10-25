/* Device inference and halo exchange for the current superlinear format. */
#ifdef PAIR_CLASS
PairStyle(sus2/embedded/kk,PairSUS2EmbeddedKokkos<LMPDeviceType>);
PairStyle(sus2/embedded/kk/device,PairSUS2EmbeddedKokkos<LMPDeviceType>);
PairStyle(sus2/embedded/kk/host,PairSUS2EmbeddedKokkos<LMPHostType>);
#else
#ifndef LMP_PAIR_SUS2_EMBEDDED_KOKKOS_H
#define LMP_PAIR_SUS2_EMBEDDED_KOKKOS_H

#include "pair_sus2_embedded.h"
#include "kokkos_base.h"
#include <memory>

namespace LAMMPS_NS {
template<class DeviceType>
class PairSUS2EmbeddedKokkos : public PairSUS2Embedded, public KokkosBase {
 public:
  explicit PairSUS2EmbeddedKokkos(class LAMMPS*);
  ~PairSUS2EmbeddedKokkos() override;
  void coeff(int,char**) override;
  void init_style() override;
  void compute(int,int) override;
  double memory_usage() override;
  int pack_forward_comm(int,int*,double*,int,int*) override;
  void unpack_forward_comm(int,int,double*) override;
  int pack_reverse_comm(int,int,double*) override;
  void unpack_reverse_comm(int,int*,double*) override;
  int pack_forward_comm_kokkos(int,DAT::tdual_int_1d,
                             DAT::tdual_double_1d&,int,int*) override;
  void unpack_forward_comm_kokkos(int,int,DAT::tdual_double_1d&) override;
  int pack_reverse_comm_kokkos(int,int,DAT::tdual_double_1d&) override;
  void unpack_reverse_comm_kokkos(int,DAT::tdual_int_1d,
                               DAT::tdual_double_1d&) override;
  // NVCC requires the enclosing function of an extended device lambda to
  // be public. These host launch helpers do not expose device storage.
  void rebuild_neighbors();
  void update_device_species();
 private:
  struct DeviceState;
  std::unique_ptr<DeviceState> device_;
  void report_device(bool=false);
};
}
#endif
#endif
