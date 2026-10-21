/* LAMMPS ownership and communication for the package-local MD engine.
   Model parsing, SH graphs, radial tables and differentiation all live in
   ML-SUS2 source units. There is no MLMTPR or training-library dependency. */
#include "pair_sus2_embedded.h"
#include "sus2_sl_kernel.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;
namespace {
const std::vector<std::string>& symbols()
{
  static const std::vector<std::string> names={
    "X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db",
    "Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"};
  return names;
}
std::vector<std::string> parse_elements(const char* input)
{
  std::vector<std::string> result;
  std::istringstream stream(input);
  std::string item;
  while (std::getline(stream,item,',')) {
    if (item.empty() || std::find(symbols().begin()+1,symbols().end(),item)==symbols().end())
      throw std::runtime_error("model_elements must be comma-separated chemical symbols in model species order");
    if (std::find(result.begin(),result.end(),item)!=result.end())
      throw std::runtime_error("model_elements contains a duplicate symbol");
    result.push_back(item);
  }
  if (result.empty() || std::string(input).back()==',')
    throw std::runtime_error("model_elements must not be empty or end with a comma");
  return result;
}
double positive_number(const char* input,const std::string& key,bool zero=false)
{
  const std::string text(input); size_t used=0;
  const double value=std::stod(text,&used);
  if (used!=text.size() || !std::isfinite(value) || (zero ? value<0 : value<=0))
    throw std::runtime_error(key+" must be finite and "+(zero?"nonnegative":"positive"));
  return value;
}
}

struct PairSUS2Embedded::Embedded {
  std::unique_ptr<SUS2SL::Kernel> kernel;
  std::vector<int> types;
  std::vector<SUS2SL::Edge> edges;
  std::vector<int> active_species_flags;
};
PairSUS2Embedded::PairSUS2Embedded(LAMMPS* lmp) : Pair(lmp),embedded_(new Embedded)
{
  single_enable=0; restartinfo=0; one_coeff=1; manybody_flag=1;
  centroidstressflag=CENTROID_NOTAVAIL;
  no_virial_fdotr_compute=1;
}
PairSUS2Embedded::~PairSUS2Embedded()
{
  if (copymode) return;
  if (allocated) { memory->destroy(setflag); memory->destroy(cutsq); }
}
void PairSUS2Embedded::allocate()
{
  allocated=1; const int n=atom->ntypes;
  memory->create(setflag,n+1,n+1,"sus2/embedded:setflag");
  memory->create(cutsq,n+1,n+1,"sus2/embedded:cutsq");
  for(int i=1;i<=n;++i) for(int j=1;j<=n;++j) setflag[i][j]=0;
}
void PairSUS2Embedded::settings(int narg,char** arg)
{
  if (embedded_->kernel) error->all(FLERR,"Set sus2/embedded options before pair_coeff");
  try {
    // Historical input convention: the optional first argument is the model
    // filename. Without it, retain the model-in-pair_coeff form.
    int first=0;
    if(narg%2!=0) {
      const std::string token(arg[0]);
      if(token=="model_elements" || token=="radial_mode" || token=="tabstep" ||
         token=="table_max_mb" || token=="profile_every")
        throw std::runtime_error("sus2/embedded options require key/value pairs");
      model_file_=token;
      first=1;
    }
    for(int i=first;i<narg;i+=2) {
      if(i+1>=narg) throw std::runtime_error("sus2/embedded options require key/value pairs");
      const std::string key(arg[i]);
      if(key=="model_elements") model_elements_=parse_elements(arg[i+1]);
      else if(key=="radial_mode") {
        radial_mode_=arg[i+1];
        if(radial_mode_!="direct" && radial_mode_!="table" && radial_mode_!="auto")
          throw std::runtime_error("radial_mode must be auto, direct or table");
      } else if(key=="tabstep") tabstep_=positive_number(arg[i+1],key);
      else if(key=="table_max_mb") {
        if(std::strcmp(arg[i+1],"auto")==0) {
          table_max_bytes_=std::numeric_limits<std::size_t>::max();
          continue;
        }
        const double mb=positive_number(arg[i+1],key,true);
        if(mb>=static_cast<double>(std::numeric_limits<std::size_t>::max())/(1024*1024))
          throw std::runtime_error("table_max_mb exceeds addressable memory");
        table_max_bytes_=static_cast<std::size_t>(mb*1024*1024);
      } else if(key=="profile_every") {
        const double count=positive_number(arg[i+1],key,true);
        if(count>std::numeric_limits<int>::max() || std::floor(count)!=count)
          throw std::runtime_error("profile_every must be a nonnegative integer");
        profile_every_=static_cast<int>(count);
      } else throw std::runtime_error("Unknown sus2/embedded option: "+key);
    }
  } catch(const std::exception& e) { error->all(FLERR,"{}",e.what()); }
}
void PairSUS2Embedded::coeff(int narg,char** arg)
{
  if(narg<2 || std::strcmp(arg[0],"*") || std::strcmp(arg[1],"*") ||
     (narg!=2 && narg!=3 && narg!=atom->ntypes+3))
    error->all(FLERR,"Expected pair_coeff * * [model.sus2 [symbol1 ... symbolN]] for sus2/embedded");
  try {
    if(narg==2 && model_file_.empty())
      throw std::runtime_error("Specify the model filename in pair_style or pair_coeff");
    if(narg>=3 && !model_file_.empty())
      throw std::runtime_error("Specify the model filename only once, in pair_style or pair_coeff");
    const bool numbered_types=narg<=3;
    SUS2SL::Model model; model.load(narg==2?model_file_:arg[2]);
    std::vector<std::string> metadata;
    const auto atomic_numbers=model.integers("zbl_atomic_numbers");
    if(!atomic_numbers.empty()) {
      if(static_cast<int>(atomic_numbers.size())!=model.species_count)
        throw std::runtime_error("ZBL element metadata does not match species count");
      for(int z:atomic_numbers) {
        if(z<1 || z>=static_cast<int>(symbols().size()))
          throw std::runtime_error("Invalid atomic number in ZBL metadata");
        metadata.push_back(symbols()[z]);
      }
    }
    if(!model_elements_.empty() && static_cast<int>(model_elements_.size())!=model.species_count)
      throw std::runtime_error("model_elements length does not match model species count");
    if(!metadata.empty() && !model_elements_.empty() && metadata!=model_elements_)
      throw std::runtime_error("model_elements disagrees with ZBL atomic-number metadata");
    // Do not persist inferred metadata: a later pair_coeff may load a different
    // model. Explicit model_elements remains an input option, not mutable state.
    const auto& elements=model_elements_.empty()?metadata:model_elements_;
    if(!numbered_types && static_cast<int>(elements.size())!=model.species_count)
      throw std::runtime_error("Specify pair_style sus2/embedded model_elements H,O,... in model species order");
    std::vector<int> mapping(atom->ntypes+1,-1);
    if(numbered_types) {
      if(atom->ntypes>model.species_count)
        throw std::runtime_error("LAMMPS atom type count exceeds model species count for numbered mapping");
      // Match PairSUS2MTP: data type 1 -> native species 0, etc. Masses and
      // comments are not element metadata and must never be used to guess.
      for(int i=1;i<=atom->ntypes;++i) mapping[i]=i-1;
    } else {
      for(int i=1;i<=atom->ntypes;++i) {
        const auto found=std::find(elements.begin(),elements.end(),arg[i+2]);
        if(found==elements.end())
          throw std::runtime_error("Unmapped pair_coeff symbol: "+std::string(arg[i+2])+"; NULL/hybrid mappings are not supported");
        mapping[i]=static_cast<int>(found-elements.begin());
      }
    }
    std::unique_ptr<SUS2SL::Kernel> kernel(new SUS2SL::Kernel(model));
    // Actual atom types are not known until run initialization. Avoid building
    // all-model table metadata merely because the file supports many elements.
    kernel->configure_species({});
    const auto mode=radial_mode_=="direct"?SUS2SL::RadialTableMode::Direct:
      (radial_mode_=="table"?SUS2SL::RadialTableMode::Table:SUS2SL::RadialTableMode::Auto);
    kernel->configure_radial(mode,tabstep_,table_max_bytes_);
    kernel->configure_profile(profile_every_!=0);
    cutoff_=kernel->cutoff();
    comm_forward=comm_reverse=kernel->communication_width();
    type_map_=std::move(mapping); embedded_->kernel=std::move(kernel);
    embedded_->active_species_flags.clear();radial_reported_=false;
    profile_calls_=0; profile_seconds_.fill(0); profile_outer_seconds_.fill(0); profile_radial_previous_.fill(0);
    if(!allocated) allocate();
    for(int i=1;i<=atom->ntypes;++i) for(int j=1;j<=atom->ntypes;++j) setflag[i][j]=1;
  } catch(const std::exception& e) { error->one(FLERR,"Embedded SUS2 model: {}",e.what()); }
}
void PairSUS2Embedded::init_style()
{
  if(std::strcmp(update->unit_style,"metal")) error->all(FLERR,"sus2/embedded requires units metal");
  if(force->newton_pair!=1) error->all(FLERR,"sus2/embedded requires newton pair on");
  if(!embedded_->kernel) error->all(FLERR,"sus2/embedded requires pair_coeff");
  if(force->pair_style && std::strncmp(force->pair_style,"hybrid",6)==0)
    error->all(FLERR,"sus2/embedded does not support hybrid pair styles");
  if(neighbor->includegroup || neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol)
    error->all(FLERR,"sus2/embedded requires complete neighborhoods; neigh_modify include/exclude is not supported");
  if(atom->molecular) for(int k=1;k<=3;++k) if(force->special_lj[k]!=1.0)
    error->all(FLERR,"sus2/embedded requires special_bonds lj 1 1 1 for molecular atom styles");
  neighbor->add_request(this,NeighConst::REQ_FULL);
  try {update_table_species();}
  catch(const std::exception& e) {error->one(FLERR,"Embedded SUS2 species selection: {}",e.what());}
}
void PairSUS2Embedded::update_table_species()
{
  if(radial_mode_=="direct")return;
  const int count=embedded_->kernel->species_count();
  std::vector<int> flags(count,0);
  for(int i=0;i<atom->nlocal;++i) {
    const int type=atom->type[i];
    if(type<=0||type>=static_cast<int>(type_map_.size()))
      throw std::runtime_error("Atom type is outside the configured SUS2 mapping");
    flags[type_map_[type]]=1;
  }
  // Owned atoms only; OR across ranks preserves species absent on an empty
  // rank or temporarily represented there only by ghosts. Repeated run/run 0
  // initialization and neighbor rebuilds also cover inserted/deleted types.
  MPI_Allreduce(MPI_IN_PLACE,flags.data(),count,MPI_INT,MPI_MAX,world);
  if(flags==embedded_->active_species_flags)return;
  std::vector<int> species;
  for(int t=0;t<count;++t)if(flags[t])species.push_back(t);
  embedded_->kernel->configure_species(species);
  embedded_->active_species_flags=std::move(flags);
  radial_reported_=false;
  profile_calls_=0;profile_seconds_.fill(0);profile_outer_seconds_.fill(0);
  profile_radial_previous_.fill(0);
  if(comm->me==0) {
    const std::size_t active=species.size(),total=static_cast<std::size_t>(count);
    utils::logmesg(lmp,"SUS2 radial table species: active={}/{} pairs={}/{} allocation={}\n",
      active,total,active*active,total*total,
      table_max_bytes_==std::numeric_limits<std::size_t>::max()?"adaptive":"explicit-cap");
  }
}
double PairSUS2Embedded::init_one(int i,int j)
{
  if(!setflag[i][j]) error->all(FLERR,"All sus2/embedded pair coefficients must be set");
  return cutoff_;
}
void PairSUS2Embedded::compute(int eflag,int vflag)
{
  ev_init(eflag,vflag);
  if(vflag_atom || cvflag_atom)
    error->all(FLERR,"sus2/embedded supports global virial only; per-atom and centroid virials are not implemented");
  try {
    // Neighbor rebuilds are collective. No new collective is inserted into
    // ordinary force calls; an unexpectedly new pair before the next rebuild
    // can still use the existing generic radial evaluation path safely.
    if(neighbor->ago==0)update_table_species();
    // No timer or extra collective is used unless profiling is explicitly enabled.
    double stamp=profile_every_?MPI_Wtime():0.;
    const auto stage=[&](int index) {
      if(profile_every_) { const double now=MPI_Wtime(); profile_seconds_[index]+=now-stamp; stamp=now; }
    };
    const int nlocal=atom->nlocal,nall=nlocal+atom->nghost;
    auto& types=embedded_->types; types.resize(nall);
    for(int i=0;i<nall;++i) types[i]=type_map_[atom->type[i]];
    auto& edges=embedded_->edges; edges.clear();
    const double cutsq=cutoff_*cutoff_;
    if(list->inum!=nlocal)
      throw std::runtime_error("sus2/embedded requires a full neighbor list for every owned atom");
    for(int ii=0;ii<list->inum;++ii) {
      const int i=list->ilist[ii];
      for(int jj=0;jj<list->numneigh[i];++jj) {
        const int j=list->firstneigh[i][jj]&NEIGHMASK;
        SUS2SL::Edge edge; edge.i=i; edge.j=j; double rsq=0;
        for(int a=0;a<3;++a) { edge.d[a]=atom->x[j][a]-atom->x[i][a]; rsq+=edge.d[a]*edge.d[a]; }
        if(rsq<cutsq || !std::isfinite(rsq)) edges.push_back(edge);
      }
    }
    auto& kernel=*embedded_->kernel;
    stage(0);
    kernel.begin(nlocal,nall,types,edges);
    stage(1);
    if(comm_forward) comm->forward_comm(this);
    stage(2);
    kernel.outer();
    stage(3);
    if(profile_every_) {
      const auto& detail=kernel.outer_profile();
      for(std::size_t i=0;i<detail.size();++i) profile_outer_seconds_[i]+=detail[i];
    }
    if(comm_reverse) comm->reverse_comm(this);
    stage(4);
    const auto& result=kernel.finish();
    stage(5);
    for(int i=0;i<nall;++i) for(int a=0;a<3;++a) atom->f[i][a]+=result.force[i][a];
    if(eflag_global) eng_vdwl+=result.energy;
    if(eflag_atom) for(int i=0;i<nlocal;++i) eatom[i]+=result.site_energy[i];
    if(vflag_global) for(int a=0;a<6;++a) virial[a]+=result.virial[a];
    stage(6);
    if(!radial_reported_) {
      const auto stats=kernel.radial_stats();
      if(comm->me==0) utils::logmesg(lmp,
        "SUS2 embedded CPU: radial_mode={} tabstep={} forward={} reverse={} rank0_tables={} bytes={} hits={} direct_fallbacks={} domain_fallbacks={} memory_fallbacks={} invalid_fallbacks={}\n",
        radial_mode_,tabstep_,comm_forward,comm_reverse,stats.tables,stats.bytes,stats.hits,
        stats.direct_fallbacks,stats.domain_fallbacks,stats.memory_fallbacks,stats.invalid_fallbacks);
      radial_reported_=true;
    }
    if(profile_every_ && ++profile_calls_%profile_every_==0) report_profile();
  // Rank-local failures must abort peers waiting in MPI rather than enter a
  // collective Error::all barrier and deadlock.
  } catch(const std::exception& e) { error->one(FLERR,"Embedded SUS2 evaluation: {}",e.what()); }
}
int PairSUS2Embedded::pack_forward_comm(int n,int* indices,double* buf,int,int*)
{
  for(int k=0;k<n;++k) embedded_->kernel->pack_forward(indices[k],buf+static_cast<size_t>(k)*comm_forward);
  return n*comm_forward;
}
void PairSUS2Embedded::report_profile()
{
  // Rank-local elapsed times include MPI wait. Maxima cannot be summed to
  // infer the critical path, and this is not a hardware sampling profile.
  std::array<double,7> sums{},maxima{};
  MPI_Reduce(profile_seconds_.data(),sums.data(),7,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(profile_seconds_.data(),maxima.data(),7,MPI_DOUBLE,MPI_MAX,0,world);
  std::array<double,5> outer_sums{},outer_maxima{};
  MPI_Reduce(profile_outer_seconds_.data(),outer_sums.data(),5,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(profile_outer_seconds_.data(),outer_maxima.data(),5,MPI_DOUBLE,MPI_MAX,0,world);
  const auto stats=embedded_->kernel->radial_stats();
  const std::array<unsigned long long,5> counts{{stats.hits,stats.direct_fallbacks,
    stats.domain_fallbacks,stats.memory_fallbacks,stats.invalid_fallbacks}};
  std::array<unsigned long long,5> delta{},global{};
  for(int i=0;i<5;++i) delta[i]=counts[i]-profile_radial_previous_[i];
  MPI_Reduce(delta.data(),global.data(),5,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,world);
  const unsigned long long bytes=stats.bytes;
  unsigned long long max_bytes=0;
  MPI_Reduce(&bytes,&max_bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,0,world);
  if(comm->me==0) {
    const char* names[]={"neighbors","inner_forward","feature_forward","outer",
      "feature_reverse","inner_reverse_finish","scatter"};
    utils::logmesg(lmp,"SUS2 embedded profile: calls={} interval={} ranks={} allrank_table_hits={} allrank_direct_fallbacks={} domain={} memory={} invalid={} max_rank_table_bytes={}\n",
      profile_calls_,profile_every_,comm->nprocs,global[0],global[1],global[2],global[3],global[4],max_bytes);
    for(int i=0;i<7;++i) utils::logmesg(lmp,
      "SUS2 embedded profile stage={} mean_ms_per_call={} max_rank_ms_per_call={}\n",
      names[i],1000*sums[i]/comm->nprocs/profile_every_,1000*maxima[i]/profile_every_);
    const char* outer_names[]={"legacy_forward_contract","mixer_adaptive_forward",
      "readout_graph_reverse","legacy_coordinate_reverse","mixer_adaptive_reverse"};
    for(int i=0;i<5;++i) utils::logmesg(lmp,
      "SUS2 embedded profile detail={} mean_ms_per_call={} max_rank_ms_per_call={}\n",
      outer_names[i],1000*outer_sums[i]/comm->nprocs/profile_every_,1000*outer_maxima[i]/profile_every_);
  }
  profile_seconds_.fill(0); profile_outer_seconds_.fill(0); profile_radial_previous_=counts;
}
void PairSUS2Embedded::unpack_forward_comm(int n,int first,double* buf)
{
  for(int k=0;k<n;++k) embedded_->kernel->unpack_forward(first+k,buf+static_cast<size_t>(k)*comm_forward);
}
int PairSUS2Embedded::pack_reverse_comm(int n,int first,double* buf)
{
  for(int k=0;k<n;++k) embedded_->kernel->pack_reverse(first+k,buf+static_cast<size_t>(k)*comm_reverse);
  return n*comm_reverse;
}
void PairSUS2Embedded::unpack_reverse_comm(int n,int* indices,double* buf)
{
  for(int k=0;k<n;++k) embedded_->kernel->add_reverse(indices[k],buf+static_cast<size_t>(k)*comm_reverse);
}
double PairSUS2Embedded::memory_usage()
{
  return type_map_.capacity()*sizeof(int)+embedded_->types.capacity()*sizeof(int)+
    embedded_->edges.capacity()*sizeof(SUS2SL::Edge)+(embedded_->kernel?embedded_->kernel->memory_usage():0);
}
