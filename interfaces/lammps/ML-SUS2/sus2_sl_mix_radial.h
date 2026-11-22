// Frozen radial rows for Effect mix-basis. Only the outer environment signal
// varies per neighbor; keep its k index until the atom's gate is available.
#ifndef LMP_SUS2_SL_MIX_RADIAL_H
#define LMP_SUS2_SL_MIX_RADIAL_H
#include "sus2_sl_radial.h"
#include "sus2_sl_model.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace LAMMPS_NS { namespace SUS2SL {
class MixRadial {
 public:
  explicit MixRadial(const Model &m)
      : model_(m), basis_(m.text("radial_basis_type"),m.min_dist,m.max_dist,m.radial_size,
                          m.scaling,m.text("radial_cutoff_type","legacy"),
                          m.integer("radial_cutoff_p",5),m.angular_max_dist),
        L_(m.lmax+1), K_(m.kmax), G_(m.integer("two_layer_gate_k_rank")), O_(m.integer("outer_k_rank")),
        inner_(m.numbers("effect_inner_mix_coeffs")), outer_(m.numbers("two_layer_gate_radial_mixing_coeffs")),
        scales_(m.numbers("scal_coeffs")), separate_(m.numbers("two_layer_gate_scaling_coeffs")),
        type_(m.numbers("two_layer_gate_type_coeffs")), by_k_(m.text("scaling_map")=="K") {
    const int C=m.species_count, F=m.radial_funcs;
    if(type_.size()!=static_cast<std::size_t>(C)*F) {
      const auto old=type_;
      type_.resize(static_cast<std::size_t>(C)*F);
      for(int t=0;t<C;++t) for(int mu=0;mu<F;++mu)
        type_[t*F+mu]=old.empty()?m.radial_coeffs[mu*(m.radial_size+C)+m.radial_size+t]:old.at(t);
    }
    basis_.configure(RadialTableMode::Direct);
  }

  void configure(RadialTableMode mode,double spacing,std::size_t budget) {
    if(!std::isfinite(spacing)||spacing<=0) throw std::invalid_argument("Invalid radial table spacing");
    mode_=mode; spacing_=spacing; budget_=budget; tables_.clear(); stats_=RadialTableStats();
  }

  // Each shell is independent so interpolation never crosses a cutoff.
  // Layout within a shell: inner [l,g], outer [l,k,o].
  int rows(bool gate,int shell) const {return (shell?L_-1:1)*(gate?G_:K_*O_);}
  double cutoff(int shell) const {return shell?model_.angular_max_dist:model_.max_dist;}
  const RadialTableStats& stats() const {return stats_;}

  struct Table {int state=0; std::size_t intervals=0; double step=0; std::vector<double> knots;};

  Table build_table(int ti,int tj,bool gate,int shell) {
    Table table;
    if(rows(gate,shell)) build(table,ti,tj,gate,shell);
    return table;
  }

  void evaluate(double r,int ti,int tj,bool gate,int shell,bool deriv,double* v,double* d) {
    const int width=rows(gate,shell);
    if(!width)return;
    if(r>=cutoff(shell)) {
      std::fill(v,v+width,0.0); if(deriv) std::fill(d,d+width,0.0); return;
    }
    const auto pair=static_cast<std::size_t>(ti)*model_.species_count+tj;
    if(mode_!=RadialTableMode::Direct && r>=model_.min_dist) {
      auto &t=tables_[4*pair+2*static_cast<int>(gate)+shell];
      if(!t.state) build(t,ti,tj,gate,shell);
      if(t.state==1) {
        const double x=(r-model_.min_dist)/t.step;
        const auto i=std::min(t.intervals-1,static_cast<std::size_t>(x));
        const double u=x-static_cast<double>(i), w=1-u;
        const auto stride=2*static_cast<std::size_t>(width);
        const double *a=t.knots.data()+i*stride, *b=a+stride;
        const double av=(1+2*u)*w*w, ad=t.step*u*w*w, bv=u*u*(3-2*u), bd=-t.step*u*u*w;
        const double dv=6*u*w/t.step, da=w*(1-3*u), db=u*(3*u-2);
        for(int j=0;j<width;++j) {
          v[j]=av*a[j]+ad*a[static_cast<std::size_t>(width)+j]+bv*b[j]+bd*b[static_cast<std::size_t>(width)+j];
          if(deriv) d[j]=dv*(b[j]-a[j])+da*a[static_cast<std::size_t>(width)+j]+db*b[static_cast<std::size_t>(width)+j];
        }
        ++stats_.hits; return;
      }
      if(t.state==2) ++stats_.memory_fallbacks; else ++stats_.invalid_fallbacks;
    } else if(mode_!=RadialTableMode::Direct) ++stats_.domain_fallbacks;
    if(mode_!=RadialTableMode::Direct) ++stats_.direct_fallbacks;
    direct(r,ti,tj,gate,shell,deriv,v,d);
  }

 private:
  const Model &model_;
  Radial basis_;
  const int L_,K_,G_,O_;
  std::vector<double> inner_,outer_,scales_,separate_,type_,rb_,dr_;
  bool by_k_;
  RadialTableMode mode_=RadialTableMode::Direct;
  double spacing_=0.001;
  std::size_t budget_=0;
  std::map<std::size_t,Table> tables_;
  RadialTableStats stats_;

  void direct(double r,int ti,int tj,bool gate,int shell,bool deriv,double* v,double* d) {
    const auto &m=model_;
    const int C=m.species_count, F=m.radial_funcs, B=m.radial_size, channels=gate?G_:O_;
    const auto &scale=gate && !separate_.empty()?separate_:scales_;
    const auto &weights=gate?inner_:outer_;
    const int first=shell?1:0, end=shell?L_:1;
    if(gate){std::fill(v,v+rows(true,shell),0.);if(deriv)std::fill(d,d+rows(true,shell),0.);}
    for(int k=0;k<K_;++k) for(int l=first;l<end;++l) {
      const int mu=k*L_+l;
      if(!by_k_ || l==first) {
        const int ix=2*(by_k_?k:mu)*C*C+ti*C+tj;
        basis_.eval(r,scale[ix],scale[ix+C*C],k,rb_,dr_,deriv,shell!=0);
      }
      const double factor=gate?type_[ti*F+mu]*type_[tj*F+mu]:1.;
      for(int c=0;c<channels;++c){
        const double *w=weights.data()+((l*channels+c)*K_+k)*B;
        double a=0,b=0;
        for(int n=0;n<B;++n){a+=w[n]*rb_[n];if(deriv)b+=w[n]*dr_[n];}
        if(gate){const int idx=(l-first)*G_+c;v[idx]+=factor*a;if(deriv)d[idx]+=factor*b;}
        else {const int idx=((l-first)*K_+k)*O_+c;v[idx]=a;if(deriv)d[idx]=b;}
      }
    }
  }

  void build(Table& t,int ti,int tj,bool gate,int shell) {
    const int width=rows(gate,shell);
    const double lo=model_.min_dist, hi=cutoff(shell), span=hi-lo;
    if(!(span>0.)){t.state=3;return;}
    const double intervals=std::ceil(span/spacing_);
    const auto rowbytes=2*sizeof(double)*static_cast<std::size_t>(width);
    if(!std::isfinite(intervals)||intervals<1||intervals>=static_cast<double>((budget_-stats_.bytes)/rowbytes)){
      t.state=2;return;
    }
    t.intervals=static_cast<std::size_t>(intervals); t.step=span/t.intervals;
    const auto stride=2*static_cast<std::size_t>(width);
    std::vector<double> knots, mid;
    if(t.intervals+1>knots.max_size()/stride){t.state=2;return;}
    try {
      knots.resize((t.intervals+1)*stride);
      mid.resize(stride);
    } catch(const std::bad_alloc&) { t.state=2; return; }
    for(std::size_t i=0;i<=t.intervals;++i){
      auto *b=knots.data()+i*stride;
      direct(i==t.intervals?hi:lo+i*t.step,ti,tj,gate,shell,true,b,b+width);
      if(!i)continue;
      const auto *a=b-stride;
      direct(lo+(i-0.5)*t.step,ti,tj,gate,shell,true,mid.data(),mid.data()+width);
      for(int j=0;j<width;++j){
        const double v=.5*(a[j]+b[j])+t.step*.125*(a[static_cast<std::size_t>(width)+j]-b[static_cast<std::size_t>(width)+j]);
        const double d=1.5*(b[j]-a[j])/t.step-.25*(a[static_cast<std::size_t>(width)+j]+b[static_cast<std::size_t>(width)+j]);
        if(!(std::abs(v-mid[j])<=1e-7*(1+std::abs(mid[j])) &&
             std::abs(d-mid[static_cast<std::size_t>(width)+j])<=1e-4*(1+std::abs(mid[static_cast<std::size_t>(width)+j])))){
          t.state=3;return;
        }
      }
    }
    t.knots.swap(knots);t.state=1;++stats_.tables;stats_.bytes+=t.knots.size()*sizeof(double);
  }
};
}}
#endif
