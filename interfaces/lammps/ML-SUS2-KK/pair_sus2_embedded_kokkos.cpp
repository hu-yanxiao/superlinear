/* Current superlinear inference. Host work is setup and scalar reductions;
   all geometry, graph evaluation and coordinate/feature derivatives use the
   selected Kokkos execution space. No native training runtime is linked. */
#include "pair_sus2_embedded_kokkos.h"
#include "sus2_sl_kokkos_kernel.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace LAMMPS_NS;
template<class DeviceType>
struct PairSUS2EmbeddedKokkos<DeviceType>::DeviceState {
  using Engine=SUS2SL::KokkosKernel<DeviceType>;
  using I=Kokkos::View<int*,DeviceType>;
  std::unique_ptr<Engine> kernel;
  I mapping,status;
  int owned=-1,all=-1,edges=0;
  NeighList* last_list=nullptr;
  std::vector<int> active_flags;
  std::size_t table_budget=0;
  unsigned long long calls=0,counted_calls=0,signal_reverse=0,force_exports=0;
  bool reported=false,comm_reported=false;
};

template<class DeviceType>
PairSUS2EmbeddedKokkos<DeviceType>::PairSUS2EmbeddedKokkos(LAMMPS* lmp)
  : PairSUS2Embedded(lmp),device_(new DeviceState)
{
  kokkosable=1;
  execution_space=ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read=X_MASK|TYPE_MASK|F_MASK;
  datamask_modify=F_MASK;
  reverse_comm_device=1;
}
template<class DeviceType>
PairSUS2EmbeddedKokkos<DeviceType>::~PairSUS2EmbeddedKokkos() = default;

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::coeff(int narg,char** arg)
{
  PairSUS2Embedded::coeff(narg,arg);
  try {
    auto next=std::make_unique<DeviceState>();
    next->kernel=std::make_unique<typename DeviceState::Engine>(model_data());
    next->mapping=typename DeviceState::I("sus2_type_mapping",type_map_.size());
    auto h=Kokkos::create_mirror_view(next->mapping);
    for(std::size_t i=0;i<type_map_.size();++i)h(i)=type_map_[i];
    Kokkos::deep_copy(next->mapping,h);
    next->status=typename DeviceState::I("sus2_pair_status",1);
    comm_forward=comm_reverse=next->kernel->communication_width();
    // Finite automatic budget per MPI rank. The current free device memory
    // further bounds this on CUDA; a user-supplied cap is never enlarged.
    next->table_budget=table_max_bytes_;
    if(next->table_budget==std::numeric_limits<std::size_t>::max()) {
      next->table_budget=512ULL*1024*1024;
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr(std::is_same_v<typename DeviceType::execution_space,Kokkos::Cuda>) {
        std::size_t free_bytes=0,total_bytes=0;
        if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)
          throw std::runtime_error("Cannot query free CUDA memory for automatic radial-table budget");
        next->table_budget=std::min(next->table_budget,free_bytes/8);
      }
#endif
    }
    device_=std::move(next);
  } catch(const std::exception& e) {error->one(FLERR,"SUS2 Kokkos model: {}",e.what());}
}

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::init_style()
{
  if(sizeof(KK_FLOAT)!=sizeof(double)||sizeof(KK_ACC_FLOAT)!=sizeof(double))
    error->all(FLERR,"sus2/embedded/kk requires KOKKOS_PREC=double for FP64 coordinates, forces and accumulation");
  validate_configuration();
  if(!device_->kernel)error->all(FLERR,"sus2/embedded/kk requires pair_coeff");
  auto request=neighbor->add_request(this,NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                          !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  // Rebuild after run boundaries, atom sorting, migration or model reload.
  device_->last_list=nullptr;
  using Exec=typename DeviceType::execution_space;
  const char* backend=Exec::name();
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr(std::is_same_v<Exec,Kokkos::Cuda>)backend="CUDA";
#endif
  if(comm->me==0)utils::logmesg(lmp,"SUS2_KOKKOS_BACKEND {}\n",
    backend);
  if(comm->me==0)utils::logmesg(lmp,
    "SUS2_KOKKOS_GENERIC_REFERENCE {}\n"
    "SUS2_KOKKOS_FORWARD_RADIAL_TILE {}\n"
    "SUS2_KOKKOS_REVERSE_CENTER_REDUCE {}\n"
    "SUS2_KOKKOS_FORWARD_ROLE_SPECIALIZE {}\n",
    int(device_->kernel->uses_generic_reference()),
    int(device_->kernel->uses_forward_radial_tile()),
    int(device_->kernel->uses_reverse_center_reduce()),
    int(device_->kernel->uses_forward_role_specialize()));
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr(std::is_same_v<Exec,Kokkos::Cuda>) {
    int selected=-1,visible=0;
    cudaDeviceProp properties{};
    if(cudaGetDevice(&selected)!=cudaSuccess ||
       cudaGetDeviceCount(&visible)!=cudaSuccess ||
       cudaGetDeviceProperties(&properties,selected)!=cudaSuccess)
      error->one(FLERR,"Cannot inspect the selected SUS2 CUDA device");
    // Runtime UUID also identifies scheduler-isolated devices when LSF uses
    // device namespaces rather than exporting CUDA_VISIBLE_DEVICES.
    static constexpr char hex[]="0123456789abcdef";
    std::string uuid="GPU-";
    for(int b=0;b<16;++b) {
      if(b==4||b==6||b==8||b==10)uuid+='-';
      const auto byte=static_cast<unsigned char>(properties.uuid.bytes[b]);
      uuid+=hex[byte>>4];uuid+=hex[byte&15];
    }
    if(comm->me==0)utils::logmesg(lmp,
      "SUS2_KOKKOS_DEVICE device={} visible_devices={} uuid={} name={} memory_bytes={}\n",
      selected,visible,uuid,properties.name,
      static_cast<unsigned long long>(properties.totalGlobalMem));
  }
#endif
}

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::rebuild_neighbors()
{
  using Range=Kokkos::RangePolicy<DeviceType>;
  using I=typename DeviceState::I;
  auto& d=*device_;
  const int owned=atom->nlocal,all=owned+atom->nghost;
  // CommKokkos grows int-sized pair buffers by 1.5. Keep headroom for that
  // framework allocation as well as the actual MPI count, without restricting
  // otherwise native-legal model channel shapes.
  if(comm_forward && all>(std::numeric_limits<int>::max()/2)/comm_forward)
    error->one(FLERR,"SUS2 Kokkos feature halo exceeds LAMMPS' integer communication count limit");
  if(list->inum!=owned)
    error->one(FLERR,"SUS2 Kokkos requires a full neighbor list for every owned atom");
  auto* kl=static_cast<NeighListKokkos<DeviceType>*>(list);
  const auto ilist=kl->d_ilist,numneigh=kl->d_numneigh;
  const auto neighbors=kl->d_neighbors;
  I counts("sus2_CSR_counts",owned+1),seen("sus2_CSR_seen",owned);
  I offsets("sus2_CSR_prefix",owned+1);
  const auto status=d.status;
  Kokkos::deep_copy(status,0);
  Kokkos::deep_copy(counts,0);
  Kokkos::deep_copy(seen,0);
  Kokkos::parallel_for("sus2_CSR_counts",Range(0,owned),KOKKOS_LAMBDA(int ii){
    const int i=ilist(ii);
    if(i<0||i>=owned){Kokkos::atomic_exchange(&status(0),1);return;}
    if(Kokkos::atomic_fetch_add(&seen(i),1)!=0||numneigh(i)<0) {
      Kokkos::atomic_exchange(&status(0),1);return;
    }
    counts(i)=numneigh(i);
  });
  long long total=0;
  const int index_limit=std::numeric_limits<int>::max();
  Kokkos::parallel_scan("sus2_CSR_prefix",Range(0,owned+1),
    KOKKOS_LAMBDA(int i,long long& sum,bool final){
      if(final && sum<=index_limit)offsets(i)=static_cast<int>(sum);
      sum+=counts(i);
    },total);
  auto hs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),status);
  if(hs(0)||total>std::numeric_limits<int>::max())
    error->one(FLERR,"Invalid or oversized SUS2 Kokkos neighbor CSR");
  d.edges=static_cast<int>(total);
  d.kernel->prepare(owned,all,d.edges);
  Kokkos::deep_copy(d.kernel->offsets,offsets);
  const auto csr=d.kernel->neighbors;
  // Preserve skin neighbors. Filtering only by force cutoff here would lose
  // pairs that enter the cutoff between two neighbor rebuilds.
  Kokkos::parallel_for("sus2_CSR_fill",Range(0,owned),KOKKOS_LAMBDA(int i){
    for(int n=0;n<numneigh(i);++n) {
      const int j=neighbors(i,n)&NEIGHMASK;
      if(j<0||j>=all)Kokkos::atomic_exchange(&status(0),1);
      csr(offsets(i)+n)=j;
    }
  });
  hs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),status);
  if(hs(0))error->one(FLERR,"Invalid SUS2 Kokkos neighbor atom index");
  d.owned=owned;d.all=all;d.last_list=list;
}

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::update_device_species()
{
  using Range=Kokkos::RangePolicy<DeviceType>;
  auto& d=*device_;
  const int count=model_data().species_count;
  typename DeviceState::I flags("sus2_active_species",count);
  Kokkos::deep_copy(flags,0);
  const auto types=d.kernel->types;
  Kokkos::parallel_for("sus2_active_species",Range(0,d.owned),KOKKOS_LAMBDA(int i){
    Kokkos::atomic_exchange(&flags(types(i)),1);
  });
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),flags);
  std::vector<int> used(count),species;
  for(int i=0;i<count;++i)used[i]=h(i);
  MPI_Allreduce(MPI_IN_PLACE,used.data(),count,MPI_INT,MPI_MAX,world);
  if(used==d.active_flags)return;
  for(int i=0;i<count;++i)if(used[i])species.push_back(i);
  const auto mode=radial_mode_=="direct"?SUS2SL::RadialTableMode::Direct:
    radial_mode_=="table"?SUS2SL::RadialTableMode::Table:SUS2SL::RadialTableMode::Auto;
  d.kernel->configure_radial(mode,tabstep_,d.table_budget,species);
  d.active_flags=std::move(used);d.reported=false;d.counted_calls=0;
  if(comm->me==0)utils::logmesg(lmp,
    "SUS2 Kokkos radial mode={} active_species={}/{} budget_bytes={}\n",
    radial_mode_,species.size(),count,d.table_budget);
}

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::compute(int eflag,int vflag)
{
  using Range=Kokkos::RangePolicy<DeviceType>;
  ev_init(eflag,vflag);
  if(vflag_atom||cvflag_atom)
    error->all(FLERR,"sus2/embedded/kk supports global virial only; per-atom and centroid virials are unavailable");
  try {
    auto& d=*device_;
    auto* ak=static_cast<AtomKokkos*>(atom);
    ak->sync(execution_space,X_MASK|TYPE_MASK|F_MASK);
    const bool rebuild=neighbor->ago==0||d.last_list!=list||
      d.owned!=atom->nlocal||d.all!=atom->nlocal+atom->nghost;
    if(rebuild)rebuild_neighbors();
    const auto x=ak->k_x.template view<DeviceType>();
    const auto type=ak->k_type.template view<DeviceType>();
    const auto f=ak->k_f.template view<DeviceType>();
    const auto pos=d.kernel->positions;
    const auto mapped=d.kernel->types;
    const auto mapping=d.mapping,status=d.status;
    const int map_size=static_cast<int>(mapping.size());
    Kokkos::deep_copy(status,0);
    Kokkos::parallel_for("sus2_positions_types",Range(0,d.all),KOKKOS_LAMBDA(int i){
      for(int q=0;q<3;++q)pos(i,q)=x(i,q);
      const int t=type(i);
      if(t<1||t>=map_size){mapped(i)=0;Kokkos::atomic_exchange(&status(0),1);}
      else mapped(i)=mapping(t);
    });
    auto hs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),status);
    if(hs(0))error->one(FLERR,"Atom type is outside the configured SUS2 mapping");
    if(rebuild)update_device_species();
    const bool count_radial=!d.reported||(profile_every_&&d.calls%profile_every_==0);
    d.kernel->configure_radial_counting(count_radial);
    if(count_radial)++d.counted_calls;
    d.kernel->begin();
    if(comm_forward)comm->forward_comm(this);
    d.kernel->outer();
    if(comm_reverse)comm->reverse_comm(this);
    d.kernel->finish();
    if(d.kernel->error_code())error->one(FLERR,"SUS2 Kokkos encountered invalid neighbor geometry or model indices");
    const auto df=d.kernel->force;
    Kokkos::parallel_for("sus2_add_atom_forces",Range(0,d.all),KOKKOS_LAMBDA(int i){
      for(int q=0;q<3;++q)f(i,q)+=df(i,q);
    });
    ak->modified(execution_space,F_MASK);
    const auto energy=d.kernel->site_energy;
    const auto vir=d.kernel->virial;
    if(eflag_global){double sum=0.;Kokkos::parallel_reduce("sus2_energy",Range(0,d.owned),
      KOKKOS_LAMBDA(int i,double& value){value+=energy(i);},sum);eng_vdwl+=sum;}
    if(vflag_global)for(int q=0;q<6;++q){double sum=0.;Kokkos::parallel_reduce("sus2_virial",Range(0,d.owned),
      KOKKOS_LAMBDA(int i,double& value){value+=vir(i,q);},sum);virial[q]+=sum;}
    if(eflag_atom){auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),energy);
      for(int i=0;i<d.owned;++i)eatom[i]+=h(i);}
    // Coordinates/forces never stage through the host. This is only the
    // number of ghost force slots handed to LAMMPS' independent atom reverse
    // communication, not a fabricated count of MPI messages executed there.
    d.force_exports+=atom->nghost;
    ++d.calls;
    report_device();
  }catch(const std::exception& e){error->one(FLERR,"SUS2 Kokkos compute: {}",e.what());}
}

template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::report_device(bool force_report)
{
  auto& d=*device_;
  if(!d.reported||force_report||(profile_every_&&d.calls%profile_every_==0)) {
    const auto stats=d.kernel->radial_stats();
    if(comm->me==0)utils::logmesg(lmp,
      "SUS2_KOKKOS_RADIAL tables={} bytes={} hits={} direct={} domain={} capacity={} invalid={} memory_bytes={} counted_force_calls={} counting=sampled\n",
      stats.tables,stats.bytes,stats.hits,stats.direct_fallbacks,stats.domain_fallbacks,
      stats.memory_fallbacks,stats.invalid_fallbacks,static_cast<unsigned long long>(d.kernel->memory_usage()),d.counted_calls);
    d.reported=true;
  }
  if(!d.comm_reported&&d.signal_reverse&&d.force_exports) {
    unsigned long long values[2]={d.signal_reverse,d.force_exports};
    // No collective here: rank zero's counters describe its observed callback
    // traffic only. Full ghost-force correctness is checked against native MPI.
    if(comm->me==0)utils::logmesg(lmp,"SUS2_KOKKOS_COMM signal_reverse={} force_exports={}\n",values[0],values[1]);
    d.comm_reported=true;
  }
}

template<class DeviceType>
int PairSUS2EmbeddedKokkos<DeviceType>::pack_forward_comm_kokkos(
    int n,DAT::tdual_int_1d list,DAT::tdual_double_1d& buffer,int,int*)
{
  const auto indices=list.template view<DeviceType>();
  const auto buf=buffer.template view<DeviceType>();
  const auto values=device_->kernel->signal;const int w=comm_forward;
  if(w&&n>(std::numeric_limits<int>::max()/2)/w)
    error->one(FLERR,"SUS2 Kokkos forward halo exceeds MPI integer count limit");
  Kokkos::parallel_for("sus2_pack_signal",Kokkos::RangePolicy<DeviceType>(0,n),KOKKOS_LAMBDA(int i){
    for(int q=0;q<w;++q)buf(i*w+q)=values(indices(i),q);
  });return n*w;
}
template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int n,int first,DAT::tdual_double_1d& buffer)
{
  const auto buf=buffer.template view<DeviceType>();const auto values=device_->kernel->signal;
  const int w=comm_forward;
  if(w&&n>(std::numeric_limits<int>::max()/2)/w)
    error->one(FLERR,"SUS2 Kokkos forward halo exceeds MPI integer count limit");
  Kokkos::parallel_for("sus2_unpack_signal",Kokkos::RangePolicy<DeviceType>(0,n),KOKKOS_LAMBDA(int i){
    for(int q=0;q<w;++q)values(first+i,q)=buf(i*w+q);
  });
  // CommKokkos can post the next MPI_Irecv into this same buffer before its
  // next send-side fence. Return buffer ownership only after this consumer
  // finishes; stream ordering alone cannot order an incoming MPI write.
  // With one rank every swap is a self-copy on the same execution stream.
  if(comm->nprocs>1)DeviceType().fence("sus2 forward receive buffer consumed");
}
template<class DeviceType>
int PairSUS2EmbeddedKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n,int first,DAT::tdual_double_1d& buffer)
{
  const auto buf=buffer.template view<DeviceType>();const auto values=device_->kernel->signal_adjoint;
  const int w=comm_reverse;
  if(w&&n>(std::numeric_limits<int>::max()/2)/w)
    error->one(FLERR,"SUS2 Kokkos reverse halo exceeds MPI integer count limit");
  Kokkos::parallel_for("sus2_pack_adjoint",Kokkos::RangePolicy<DeviceType>(0,n),KOKKOS_LAMBDA(int i){
    for(int q=0;q<w;++q)buf(i*w+q)=values(first+i,q);
  });device_->signal_reverse+=n;return n*w;
}
template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n,DAT::tdual_int_1d list,DAT::tdual_double_1d& buffer)
{
  const auto indices=list.template view<DeviceType>();
  const auto buf=buffer.template view<DeviceType>();
  const auto values=device_->kernel->signal_adjoint;const int w=comm_reverse;
  if(w&&n>(std::numeric_limits<int>::max()/2)/w)
    error->one(FLERR,"SUS2 Kokkos reverse halo exceeds MPI integer count limit");
  Kokkos::parallel_for("sus2_unpack_adjoint",Kokkos::RangePolicy<DeviceType>(0,n),KOKKOS_LAMBDA(int i){
    for(int q=0;q<w;++q)Kokkos::atomic_add(&values(indices(i),q),buf(i*w+q));
  });
  if(comm->nprocs>1)DeviceType().fence("sus2 reverse receive buffer consumed");
  device_->signal_reverse+=n;
}

// The framework may deliberately choose legacy/staged pair communication
// (e.g. non-CUDA-aware MPI). Only the feature payload is mirrored here; forces
// and geometry are still evaluated on the selected device execution space.
template<class DeviceType>
int PairSUS2EmbeddedKokkos<DeviceType>::pack_forward_comm(int n,int* list,double* buf,int,int*)
{
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device_->kernel->signal);
  const int w=comm_forward;
  if(w&&n>(std::numeric_limits<int>::max()/2)/w)
    error->one(FLERR,"SUS2 Kokkos forward halo exceeds MPI integer count limit");
  for(int i=0;i<n;++i)for(int q=0;q<w;++q)buf[i*w+q]=h(list[i],q);
  return n*w;
}
template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::unpack_forward_comm(int n,int first,double* buf)
{
  if(comm_forward&&n>(std::numeric_limits<int>::max()/2)/comm_forward)
    error->one(FLERR,"SUS2 Kokkos forward halo exceeds MPI integer count limit");
  auto slice=Kokkos::subview(device_->kernel->signal,std::make_pair(first,first+n),Kokkos::ALL());
  auto h=Kokkos::create_mirror_view(slice);
  for(int i=0;i<n;++i)for(int q=0;q<comm_forward;++q)h(i,q)=buf[i*comm_forward+q];
  Kokkos::deep_copy(slice,h);
}
template<class DeviceType>
int PairSUS2EmbeddedKokkos<DeviceType>::pack_reverse_comm(int n,int first,double* buf)
{
  if(comm_reverse&&n>(std::numeric_limits<int>::max()/2)/comm_reverse)
    error->one(FLERR,"SUS2 Kokkos reverse halo exceeds MPI integer count limit");
  auto slice=Kokkos::subview(device_->kernel->signal_adjoint,std::make_pair(first,first+n),Kokkos::ALL());
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),slice);
  for(int i=0;i<n;++i)for(int q=0;q<comm_reverse;++q)buf[i*comm_reverse+q]=h(i,q);
  device_->signal_reverse+=n;return n*comm_reverse;
}
template<class DeviceType>
void PairSUS2EmbeddedKokkos<DeviceType>::unpack_reverse_comm(int n,int* list,double* buf)
{
  if(comm_reverse&&n>(std::numeric_limits<int>::max()/2)/comm_reverse)
    error->one(FLERR,"SUS2 Kokkos reverse halo exceeds MPI integer count limit");
  const auto values=device_->kernel->signal_adjoint;
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);
  for(int i=0;i<n;++i)for(int q=0;q<comm_reverse;++q)h(list[i],q)+=buf[i*comm_reverse+q];
  Kokkos::deep_copy(values,h);device_->signal_reverse+=n;
}
template<class DeviceType>
double PairSUS2EmbeddedKokkos<DeviceType>::memory_usage()
{
  return PairSUS2Embedded::memory_usage()+(device_->kernel?device_->kernel->memory_usage():0.)+
    sizeof(int)*(device_->mapping.size()+device_->status.size());
}
namespace LAMMPS_NS {
template class PairSUS2EmbeddedKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSUS2EmbeddedKokkos<LMPHostType>;
#endif
}
