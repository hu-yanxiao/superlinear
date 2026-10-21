#include "sus2_sl_radial.h"
#include "sus2_sl_model.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdlib>
namespace LAMMPS_NS { namespace SUS2SL {
Radial::Radial(const std::string& type, double lo, double hi, int n,
               double scaling, const std::string& cutoff, int p) {
  if (n <= 0 || !std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo) ||
      !std::isfinite(scaling)) throw std::invalid_argument("Invalid radial basis metadata");
  if (type == "RBShapeev") basis_.reset(new radial_math::RadialBasis_Shapeev(lo,hi,n));
  else if (type == "RBChebyshev") basis_.reset(new radial_math::RadialBasis_Chebyshev(lo,hi,n));
  else if (type == "RBChebyshev_s") basis_.reset(new radial_math::RadialBasis_Chebyshev_s(lo,hi,n));
  else if (type == "RBChebyshev_ss") basis_.reset(new radial_math::RadialBasis_Chebyshev_ss(lo,hi,n));
  else if (type == "RBChebyshev_ssw") basis_.reset(new radial_math::RadialBasis_Chebyshev_ssw(lo,hi,n));
  else if (type == "RBChebyshev_sss") basis_.reset(new radial_math::RadialBasis_Chebyshev_sss(lo,hi,n));
  else if (type == "RBChebyshev_sss_rational") basis_.reset(new radial_math::RadialBasis_Chebyshev_sss_rational(lo,hi,n));
  else if (type == "RBChebyshev_sssw") basis_.reset(new radial_math::RadialBasis_Chebyshev_sssw(lo,hi,n));
  else if (type == "RBChebyshev_sss_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_sss_lmp(lo,hi,n));
  else if (type == "RBChebyshev_sss_rational_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_sss_rational_lmp(lo,hi,n));
  else if (type == "RBChebyshev_sssw_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_sssw_lmp(lo,hi,n));
  else if (type == "RBChebyshev_ss_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_ss_lmp(lo,hi,n));
  else if (type == "RBChebyshev_ssw_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_ssw_lmp(lo,hi,n));
  else if (type == "RBChebyshev_s_lmp") basis_.reset(new radial_math::RadialBasis_Chebyshev_s_lmp(lo,hi,n));
  else if (type == "RBChebyshev_ssss") basis_.reset(new radial_math::RadialBasis_Chebyshev_ssss(lo,hi,n));
  else if (type == "RBChebyshev_sssss") basis_.reset(new radial_math::RadialBasis_Chebyshev_sssss(lo,hi,n));
  else if (type == "RBChebyshev_sigma") basis_.reset(new radial_math::RadialBasis_Chebyshev_sigma(lo,hi,n));
  else if (type == "RBLaguerre_log1p") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p(lo,hi,n));
  else if (type == "RBLaguerre_log1p_lmp") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p_lmp(lo,hi,n));
  else if (type == "RBLaguerre_log1p_pos") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p_pos(lo,hi,n));
  else if (type == "RBLaguerre_log1p_pos_lmp") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p_pos_lmp(lo,hi,n));
  else if (type == "RBLaguerre_log1p_noenv") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p_noenv(lo,hi,n));
  else if (type == "RBLaguerre_log1p_noenv_lmp") basis_.reset(new radial_math::RadialBasis_Laguerre_log1p_noenv_lmp(lo,hi,n));
  else if (type == "RBJacobi_sss") basis_.reset(new radial_math::RadialBasis_Jacobi_sss(lo,hi,n));
  else if (type == "RBJacobi_sss_lmp") basis_.reset(new radial_math::RadialBasis_Jacobi_sss_lmp(lo,hi,n));
  else if (type == "RBJacobi_sss_noweight") basis_.reset(new radial_math::RadialBasis_Jacobi_sss_noweight(lo,hi,n));
  else if (type == "RBJacobi_sss_noweight_lmp") basis_.reset(new radial_math::RadialBasis_Jacobi_sss_noweight_lmp(lo,hi,n));
  else if (type == "RBBessel") basis_.reset(new radial_math::RadialBasis_Bessel(lo,hi,n));
  else if (type == "RBBesselw") basis_.reset(new radial_math::RadialBasis_Besselw(lo,hi,n));
  else if (type == "RBChebyshev_Tri") basis_.reset(new radial_math::RadialBasis_Chebyshev_Tri(lo,hi,n));
  else if (type == "RBBessel_sss") basis_.reset(new radial_math::RadialBasis_Bessel_sss(lo,hi,n));
  else if (type == "RBBessel_sssw") basis_.reset(new radial_math::RadialBasis_Bessel_sssw(lo,hi,n));
  else if (type == "RBChebyshev_tanhexp") basis_.reset(new radial_math::RadialBasis_Chebyshev_tanhexp(lo,hi,n));
  else if (type == "RBChebyshev_tanhexp_w") basis_.reset(new radial_math::RadialBasis_Chebyshev_tanhexp_w(lo,hi,n));
  else if (type == "RBChebyshev_repuls") basis_.reset(new radial_math::RadialBasis_Chebyshev_repuls(lo,hi,n));
  else if (type == "RBTaylor") basis_.reset(new radial_math::RadialBasis_Taylor(lo,hi,n));
  else throw std::invalid_argument("Unsupported radial basis: " + type);
  basis_->scaling = scaling;
  basis_->SetRadialCutoff(cutoff,p);
  basis_->SetInferenceRadialTableActive(true);
}
void Radial::configure(RadialTableMode mode, double spacing, std::size_t max_bytes) {
  basis_->ConfigureInferenceRadialTable(mode,spacing,max_bytes);
  mode_=mode;spacing_=spacing;budget_=max_bytes;
  frozen_stats_=RadialTableStats();retired_stats_=RadialTableStats();
  const char* layout=std::getenv("SUS2_SL_FROZEN_LAYOUT");
  if(layout&&std::strcmp(layout,"scalar")&&std::strcmp(layout,"tile4")&&std::strcmp(layout,"tile8"))
    throw std::invalid_argument("SUS2_SL_FROZEN_LAYOUT must be scalar, tile4 or tile8");
  tiled_=!layout||std::strcmp(layout,"scalar")!=0;
  tile_width_=layout&&std::strcmp(layout,"tile8")==0?8:4;
  const char* bulk=std::getenv("SUS2_SL_FROZEN_BULK");
  if(bulk&&std::strcmp(bulk,"0")&&std::strcmp(bulk,"1"))
    throw std::invalid_argument("SUS2_SL_FROZEN_BULK must be 0 or 1");
  bulk_=!bulk||std::strcmp(bulk,"1")==0;
  for(auto& f:frozen_){f.status=0;f.packed=false;f.intervals=0;std::vector<double>().swap(f.knots);}
  for(auto& t:frozen_tiles_){t.lanes=0;std::vector<double>().swap(t.knots);}
  try {frozen_tiles_.resize(frozen_.size()/tile_width_+(frozen_.size()%tile_width_!=0));}
  catch(const std::bad_alloc&){frozen_tiles_.clear();tiled_=false;}
  catch(const std::length_error&){frozen_tiles_.clear();tiled_=false;}
}
void Radial::eval(double r,double slope,double shift,int k,
                  std::vector<double>& vals,std::vector<double>& ders,bool derivatives) {
  if(derivatives)basis_->RB_CalcRDerOnly(r,slope,shift,k);
  else basis_->RB_CalcValsOnly(r,slope,shift,k);
  vals.assign(basis_->rb_vals.begin(),basis_->rb_vals.begin()+basis_->rb_size);
  if(derivatives)ders.assign(basis_->rb_ders.begin(),basis_->rb_ders.begin()+basis_->rb_size);
}
namespace {
void add_stats(RadialTableStats& a,const RadialTableStats& b) {
 a.tables+=b.tables;a.bytes+=b.bytes;a.hits+=b.hits;
 a.direct_fallbacks+=b.direct_fallbacks;a.domain_fallbacks+=b.domain_fallbacks;
 a.memory_fallbacks+=b.memory_fallbacks;a.invalid_fallbacks+=b.invalid_fallbacks;
}
double number(const std::vector<double>& v,std::size_t i,double fallback=0) {
 return i<v.size()?v[i]:fallback;
}
template<bool Derivatives>
void interpolate_four(const double* a,const double* b,const double* w,double* values,double* ders) {
 // Independent lanes preserve the original four-term arithmetic. Small local
 // output arrays let the compiler load contiguous packed inputs without alias
 // checks against the caller's reusable output vectors.
 double v[4],d[4];
 for(int lane=0;lane<4;++lane){
  v[lane]=w[0]*a[2*lane]+w[1]*a[2*lane+1]+w[2]*b[2*lane]+w[3]*b[2*lane+1];
  if(Derivatives)d[lane]=w[4]*a[2*lane]+w[5]*a[2*lane+1]+w[6]*b[2*lane]+w[7]*b[2*lane+1];
 }
 for(int lane=0;lane<4;++lane){values[lane]=v[lane];if(Derivatives)ders[lane]=d[lane];}
}
}
RadialTableStats Radial::stats() const {
 auto s=basis_->GetInferenceRadialTableStats();add_stats(s,retired_stats_);add_stats(s,frozen_stats_);return s;
}
void Radial::reset_generic_budget() {
 auto s=basis_->GetInferenceRadialTableStats();s.tables=0;s.bytes=0;add_stats(retired_stats_,s);
 basis_->ConfigureInferenceRadialTable(mode_,spacing_,budget_-frozen_stats_.bytes);
}
void Radial::freeze(const Model& m) {
 std::vector<int> species;
 try {if(m.species_count>0){species.resize(m.species_count);for(int i=0;i<m.species_count;++i)species[i]=i;}}
 catch(const std::bad_alloc&){species.clear();}
 catch(const std::length_error&){species.clear();}
 freeze(m,species);
}
void Radial::freeze(const Model& m,const std::vector<int>& used_species) {
 configure(mode_,spacing_,budget_);
 frozen_.clear();frozen_ids_.clear();frozen_tiles_.clear();frozen_species_slots_.clear();frozen_pairs_=0;frozen_model_pairs_=0;frozen_used_species_=0;frozen_model_species_=0;frozen_funcs_=0;
 if(used_species.empty()||m.species_count<=0||m.radial_funcs<=0||m.radial_size<=0||m.lmax<0)return;
 try {
 std::vector<int> species=used_species;
 for(int t:species)if(t<0||t>=m.species_count)throw std::out_of_range("Frozen radial species index");
 std::sort(species.begin(),species.end());species.erase(std::unique(species.begin(),species.end()),species.end());
 const std::size_t Csize=std::size_t(m.species_count),Fsize=std::size_t(m.radial_funcs),used=species.size();
 const std::size_t limit=std::numeric_limits<std::size_t>::max();
 if(Csize>limit/Csize||used>limit/used)return;
 const std::size_t model_pairs=Csize*Csize,pairs=used*used;
 if(model_pairs>std::size_t(std::numeric_limits<int>::max())||pairs>limit/2/Fsize)return;
 const std::size_t requests=2*pairs*Fsize;
 if(requests>frozen_ids_.max_size())return;
 frozen_pairs_=pairs;frozen_model_pairs_=model_pairs;frozen_funcs_=m.radial_funcs;frozen_ids_.reserve(requests);
 frozen_model_species_=m.species_count;frozen_used_species_=used;
 frozen_species_slots_.assign(Csize,-1);
 for(std::size_t slot=0;slot<used;++slot)frozen_species_slots_[species[slot]]=int(slot);
 const auto scale=m.numbers("scal_coeffs"),gscale=m.numbers("two_layer_gate_scaling_coeffs");
 const auto gc=m.numbers("two_layer_gate_radial_coeffs");
 const std::string sm=m.text("scaling_map","LK");const int L=m.lmax+1,R=m.radial_size,C=m.species_count;
 // The immutable request index is the hot-path handle. Identical requests share
 // one table, including gate/outer coefficients and exact signed-zero warps.
 std::map<std::vector<std::uint64_t>,std::size_t> unique;
 for(int gate=0;gate<2;++gate)for(int ti:species)for(int tj:species)for(int mu=0;mu<frozen_funcs_;++mu){
  const std::size_t pair=std::size_t(ti)*Csize+std::size_t(tj);
  Frozen f;f.channel=mu/L;const int l=mu%L;
  const int sb=sm=="K"?f.channel:(sm=="L"?l:(sm=="pair"?0:mu));
  const auto& sc=gate&&!gscale.empty()?gscale:scale;const std::size_t off=2*std::size_t(sb)*model_pairs+pair;
  f.slope=number(sc,off,1);f.shift=number(sc,off+model_pairs);
  const auto& co=gate&&!gc.empty()?gc:m.radial_coeffs;const int stride=gate&&!gc.empty()?R:R+C;
  f.coeff.resize(R);bool finite=std::isfinite(f.slope)&&std::isfinite(f.shift);
  for(int b=0;b<R;++b){f.coeff[b]=number(co,std::size_t(mu)*stride+b);finite=finite&&std::isfinite(f.coeff[b]);}
  // Avoid floating-point map comparison for NaNs. Such requests remain direct.
  std::vector<std::uint64_t> key;auto bits=[&](double x){std::uint64_t u;std::memcpy(&u,&x,sizeof(u));key.push_back(u);};
  bits(f.slope);bits(f.shift);key.push_back(std::uint64_t(f.channel));for(double x:f.coeff)bits(x);
  auto it=finite?unique.find(key):unique.end();
  if(it!=unique.end())frozen_ids_.push_back(it->second);
  else {std::size_t id=frozen_.size();frozen_ids_.push_back(id);frozen_.push_back(std::move(f));if(finite)unique.emplace(std::move(key),id);}
 }
 frozen_tiles_.resize(frozen_.size()/tile_width_+(frozen_.size()%tile_width_!=0));
 }catch(const std::bad_alloc&){std::vector<Frozen>().swap(frozen_);std::vector<std::size_t>().swap(frozen_ids_);std::vector<Tile>().swap(frozen_tiles_);frozen_species_slots_.clear();frozen_pairs_=0;frozen_model_pairs_=0;frozen_used_species_=0;frozen_model_species_=0;frozen_funcs_=0;}
 catch(const std::length_error&){std::vector<Frozen>().swap(frozen_);std::vector<std::size_t>().swap(frozen_ids_);std::vector<Tile>().swap(frozen_tiles_);frozen_species_slots_.clear();frozen_pairs_=0;frozen_model_pairs_=0;frozen_used_species_=0;frozen_model_species_=0;frozen_funcs_=0;}
}
void Radial::pack_frozen(Frozen& f) {
 if(!tiled_)return;
 const std::size_t id=std::size_t(&f-frozen_.data()),group=id/tile_width_;
 Tile& tile=frozen_tiles_[group];const std::size_t old=tile.lanes,lanes=old+1,nodes=f.intervals+1;
 // Bound transient repacking to one small tile. Large grids retain scalar
 // ownership. No knot is duplicated persistently or reserved before admission.
 if(nodes>1024*1024/sizeof(double)/2/lanes)return;
 try {
  std::vector<double> packed(nodes*lanes*2);
  for(std::size_t g=0;g<nodes;++g){
   for(std::size_t lane=0;lane<old;++lane){
    packed[(g*lanes+lane)*2]=tile.knots[(g*old+lane)*2];
    packed[(g*lanes+lane)*2+1]=tile.knots[(g*old+lane)*2+1];
   }
   packed[(g*lanes+old)*2]=f.knots[g*2];
   packed[(g*lanes+old)*2+1]=f.knots[g*2+1];
  }
  tile.knots.swap(packed);tile.lanes=lanes;
  f.tile=group;f.lane=old;f.packed=true;std::vector<double>().swap(f.knots);
 }catch(const std::bad_alloc&){ /* Keep the already-qualified scalar table. */ }
}
void Radial::build_frozen(Frozen& f) {
 const double lo=std::max(0.0,basis_->min_dist),hi=basis_->max_dist;
 const double count=std::ceil((hi-lo)/spacing_);const std::size_t available=budget_-frozen_stats_.bytes;
 if(!std::isfinite(count)||count<1||count>=double(std::numeric_limits<std::size_t>::max())){f.status=3;return;}
 const std::size_t n=std::size_t(count);
 if(n>=available/sizeof(double)/2||n>=f.knots.max_size()/2){f.status=3;return;}
 reset_generic_budget(); // Free generic knots before allocating contracted knots.
 f.step=(hi-lo)/double(n);const int R=basis_->rb_size;
 const bool active=basis_->SetInferenceRadialTableActive(false);
 struct Restore {radial_math::AnyRadialBasis* b;bool a;~Restore(){b->SetInferenceRadialTableActive(a);}}restore{basis_.get(),active};
 try {
  std::vector<double> knots((n+1)*2),leftv(R),leftd(R),rightv(R),rightd(R);
  auto sample=[&](double r,double* dest,std::vector<double>&v,std::vector<double>&d){
   basis_->RB_CalcRDerOnly(r,f.slope,f.shift,f.channel);double value=0,der=0;
   for(int b=0;b<R;++b){v[b]=basis_->rb_vals[b];d[b]=basis_->rb_ders[b];
    if(!std::isfinite(v[b])||!std::isfinite(d[b]))return false;
    value+=f.coeff[b]*v[b];der+=f.coeff[b]*d[b];}
   dest[0]=value;dest[1]=der;return std::isfinite(value)&&std::isfinite(der);
  };
  if(!sample(lo,knots.data(),leftv,leftd)){f.status=2;return;}
  for(std::size_t i=0;i<n;++i){
   if(!sample(i+1==n?hi:lo+(i+1)*f.step,knots.data()+2*(i+1),rightv,rightd)){f.status=2;return;}
   basis_->RB_CalcRDerOnly(lo+(i+0.5)*f.step,f.slope,f.shift,f.channel);
   double midv=0,midd=0;
   for(int b=0;b<R;++b){const double v=basis_->rb_vals[b],d=basis_->rb_ders[b];
    const double hv=.5*(leftv[b]+rightv[b])+.125*f.step*(leftd[b]-rightd[b]);
    const double hd=1.5*(rightv[b]-leftv[b])/f.step-.25*(leftd[b]+rightd[b]);
    if(!std::isfinite(v)||!std::isfinite(d)||std::abs(hv-v)>1e-7*(1+std::abs(v))||std::abs(hd-d)>1e-4*(1+std::abs(d))){f.status=2;return;}
    midv+=f.coeff[b]*v;midd+=f.coeff[b]*d;
   }
   const double* a=knots.data()+2*i;const double* b=a+2;
   const double hv=.5*(a[0]+b[0])+.125*f.step*(a[1]-b[1]);
   const double hd=1.5*(b[0]-a[0])/f.step-.25*(a[1]+b[1]);
   if(!std::isfinite(midv)||!std::isfinite(midd)||std::abs(hv-midv)>1e-7*(1+std::abs(midv))||std::abs(hd-midd)>1e-4*(1+std::abs(midd))){f.status=2;return;}
   leftv.swap(rightv);leftd.swap(rightd);
  }
  f.knots.swap(knots);f.intervals=n;f.status=1;
  frozen_stats_.bytes+=f.knots.size()*sizeof(double);++frozen_stats_.tables;
  pack_frozen(f);
 }catch(const std::bad_alloc&){f.status=3;}
 catch(const std::length_error&){f.status=3;}
 // Generic tables and frozen tables share one knot-storage budget. Generic
 // counters survive eviction, and accepted frozen knots remain stable.
 if(f.status==1)reset_generic_budget();
}
bool Radial::eval_frozen(double r,int pair,bool gate,std::vector<double>& vals,std::vector<double>& ders,
                         RadialInterpolation* interpolation,bool derivatives) {
 if(mode_==RadialTableMode::Direct||frozen_ids_.empty())return false;
 if(pair<0||std::size_t(pair)>=frozen_model_pairs_)throw std::out_of_range("Frozen radial pair index");
 const int si=frozen_species_slots_[pair/frozen_model_species_],sj=frozen_species_slots_[pair%frozen_model_species_];
 if(si<0||sj<0)return false;
 const std::size_t pair_slot=std::size_t(si)*frozen_used_species_+std::size_t(sj);
 vals.resize(frozen_funcs_);if(derivatives)ders.resize(frozen_funcs_);
 const double lo=std::max(0.0,basis_->min_dist),hi=basis_->max_dist;
 const bool domain=std::isfinite(r)&&r>=lo&&r<=hi;
 bool have_direct=false;int direct_channel=0;
 RadialInterpolation local;RadialInterpolation& context=interpolation?*interpolation:local;
 double direct_slope=0,direct_shift=0;double* w=context.weights.data();
 const std::size_t* handles=frozen_ids_.data()+(std::size_t(gate)*frozen_pairs_+pair_slot)*std::size_t(frozen_funcs_);
 for(int mu=0;mu<frozen_funcs_;++mu){Frozen& f=frozen_[handles[mu]];
  if(domain&&f.status==0){build_frozen(f);have_direct=false;}
  if(domain&&f.status==1){const std::size_t n=f.intervals;
   if(!context.ready){context.interval=std::min(std::size_t((r-lo)/f.step),n-1);
    const double t=r==hi?1:(r-(lo+context.interval*f.step))/f.step,t2=t*t,t3=t2*t;
    w[0]=2*t3-3*t2+1;w[1]=(t3-2*t2+t)*f.step;w[2]=-2*t3+3*t2;w[3]=(t3-t2)*f.step;
    w[4]=(6*t2-6*t)/f.step;w[5]=3*t2-4*t+1;w[6]=(-6*t2+6*t)/f.step;w[7]=3*t2-2*t;context.ready=true;
   }
   const std::size_t width=f.packed?2*frozen_tiles_[f.tile].lanes:2;
   const double* data=f.packed?frozen_tiles_[f.tile].knots.data()+2*f.lane:f.knots.data();
   const double* a=data+width*context.interval;const double* b=a+width;
   // Only inspect already-built neighbors. Building a later mu here would
   // change lazy admission order and could invalidate a/b by repacking a tile.
   if(bulk_&&f.packed&&frozen_funcs_-mu>=4&&f.lane+4<=width/2){
    bool consecutive=true;
    for(int lane=1;lane<4;++lane){const Frozen& next=frozen_[handles[mu+lane]];
     if(next.status!=1||!next.packed||next.tile!=f.tile||next.lane!=f.lane+std::size_t(lane)){consecutive=false;break;}
    }
    if(consecutive){
     if(derivatives)interpolate_four<true>(a,b,w,vals.data()+mu,ders.data()+mu);
     else interpolate_four<false>(a,b,w,vals.data()+mu,nullptr);
     frozen_stats_.hits+=4;mu+=3;continue;
    }
   }
   vals[mu]=w[0]*a[0]+w[1]*a[1]+w[2]*b[0]+w[3]*b[1];
   if(derivatives)ders[mu]=w[4]*a[0]+w[5]*a[1]+w[6]*b[0]+w[7]*b[1];
   ++frozen_stats_.hits;
  }else{
   ++frozen_stats_.direct_fallbacks;if(!domain)++frozen_stats_.domain_fallbacks;
   else if(f.status==3)++frozen_stats_.memory_fallbacks;else ++frozen_stats_.invalid_fallbacks;
   if(!have_direct||direct_channel!=f.channel||
      std::memcmp(&direct_slope,&f.slope,sizeof(double))||std::memcmp(&direct_shift,&f.shift,sizeof(double))){
    const bool active=basis_->SetInferenceRadialTableActive(false);
    struct Restore {radial_math::AnyRadialBasis* b;bool a;~Restore(){b->SetInferenceRadialTableActive(a);}}restore{basis_.get(),active};
    if(derivatives)basis_->RB_CalcRDerOnly(r,f.slope,f.shift,f.channel);
    else basis_->RB_CalcValsOnly(r,f.slope,f.shift,f.channel);
    have_direct=true;direct_channel=f.channel;direct_slope=f.slope;direct_shift=f.shift;
   }
   double v=0,d=0;for(int b=0;b<basis_->rb_size;++b){v+=f.coeff[b]*basis_->rb_vals[b];if(derivatives)d+=f.coeff[b]*basis_->rb_ders[b];}
   vals[mu]=v;if(derivatives)ders[mu]=d;
  }
 }
 return true;
}
}}
