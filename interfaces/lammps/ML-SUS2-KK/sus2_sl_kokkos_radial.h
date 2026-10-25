// Native superlinear device radial evaluator. Host setup is the only place
// using containers, model parsing or the reference virtual radial evaluator.
#ifndef SUS2_SL_KOKKOS_RADIAL_H
#define SUS2_SL_KOKKOS_RADIAL_H
#include <Kokkos_Core.hpp>
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace LAMMPS_NS { namespace SUS2SL {
template<class DeviceType> class DeviceRadial {
 public:
  enum Kind { Shapeev,Cheb,ChebS,ChebSS,ChebSSW,ChebSSS,ChebRational,
    ChebSSSW,ChebSSS_LMP,ChebRational_LMP,ChebSSSW_LMP,ChebSS_LMP,
    ChebSSW_LMP,ChebS_LMP,ChebSSSS,ChebSSSSS,ChebSigma,Laguerre,
    Laguerre_LMP,LaguerrePos,LaguerrePos_LMP,LaguerreNoenv,
    LaguerreNoenv_LMP,Jacobi,Jacobi_LMP,JacobiNoWeight,
    JacobiNoWeight_LMP,Bessel,BesselW,ChebTri,BesselSSS,BesselSSSW,
    ChebTanhExp,ChebTanhExpW,ChebRepuls,Taylor };
  struct Table {
    std::uint64_t slope_bits=0,shift_bits=0;
    std::size_t offset=0,intervals=0;
    double slope=1,shift=0,step=0;
    int channel=0,width=0,status=0; // 1 resident, 2 invalid, 3 capacity
  };
  using RealView=Kokkos::View<double*,DeviceType>;
  using IntView=Kokkos::View<int*,DeviceType>;
  using TableView=Kokkos::View<Table*,DeviceType>;
  using CountView=Kokkos::View<std::uint64_t*,DeviceType>;
  DeviceRadial()=default;
  explicit DeviceRadial(const Model& m);
  void configure(RadialTableMode mode,double spacing,std::size_t max_bytes,
                 const std::vector<int>& active_species);
  // Host launch control: copies captured by already-launched kernels retain
  // their own flag. Sampled counters describe enabled force calls only.
  void set_counting(bool enabled) {count_enabled_=enabled;}
  RadialTableStats stats() const;
  std::size_t memory_usage() const;
  KOKKOS_INLINE_FUNCTION void basis(double r,double slope,double shift,int k,
                                    int n,double& value,double& deriv) const;
  KOKKOS_INLINE_FUNCTION void contract(double r,int ti,int tj,int k,int l,
                                     bool gate,double& value,double& deriv) const;
  // Exposed for device qualification without table interception.
  KOKKOS_INLINE_FUNCTION void direct_basis(double r,double slope,double shift,
                                          int k,int n,double& v,double& d) const;
 private:
  int kind_=ChebSSS,B_=0,C_=0,F_=0,L_=0,K_=0,O_=0,sm_=3,p_=5,basis_channels_=0;
  bool polynomial_=false,frozen_outer_=true,needs_basis_=false,count_enabled_=true;
  double lo_=0,hi_=0,scaling_=1,lower_=0;
  int table_mode_=0,active_count_=0;
  RealView scale_,gate_scale_,coeff_,gate_coeff_,shape_,knots_;
  IntView species_slots_,frozen_handles_;
  TableView basis_tables_,frozen_tables_;
  CountView counters_;
  std::size_t table_bytes_=0,table_count_=0;
  static const char* name(int kind);
  static std::vector<double> shape_coefficients(double min_dist,double max_dist);
  template<class T> static Kokkos::View<T*,DeviceType> upload(const char* label,const std::vector<T>& v) {
    Kokkos::View<T*,DeviceType> out(std::string(label),v.size());
    auto host=Kokkos::create_mirror_view(out);
    for(std::size_t i=0;i<v.size();++i)host(i)=v[i];
    Kokkos::deep_copy(out,host);return out;
  }
  KOKKOS_INLINE_FUNCTION static bool finite(double x) {
    return x==x && x<=1.7976931348623157e308 && x>=-1.7976931348623157e308;
  }
  KOKKOS_INLINE_FUNCTION static std::uint64_t bits(double x) {
    union {double d;std::uint64_t u;} v;v.d=x;return v.u;
  }
  KOKKOS_INLINE_FUNCTION static double softplus(double x) {
    return x>40?x:(x < -40?::exp(x) : ::log1p(::exp(x)));
  }
  KOKKOS_INLINE_FUNCTION void cutoff(double r,double& f,double& d) const {
    const double delta=r-hi_;f=delta*delta;d=2*delta;
    if(!polynomial_)return;
    if(r>=hi_){f=d=0;return;}
    const double x=r/hi_,a=1-x;double poly=0;
    for(int k=p_-1;k>=0;--k)poly=poly*x+0.5*double((k+1)*(k+2));
    f=a*a*a*poly;const double p=double(p_);
    d=-0.5*p*(p+1)*(p+2)*(p_==1?1.0 : ::pow(x,p_-1))*a*a/hi_;
  }
  KOKKOS_INLINE_FUNCTION void count(bool hit,int reason,bool enabled=true) const {
    if(!count_enabled_||!enabled||!table_mode_||!counters_.extent(0))return;
    Kokkos::atomic_add(&counters_(hit?0:1),std::uint64_t(1));
    if(!hit)Kokkos::atomic_add(&counters_(reason),std::uint64_t(1));
  }
  KOKKOS_INLINE_FUNCTION bool interpolate(const Table& t,double r,int n,
                                         double& v,double& d) const {
    if(t.status!=1||!finite(r)||r<lower_||r>hi_)return false;
    std::size_t cell=std::size_t((r-lower_)/t.step);
    if(cell>=t.intervals)cell=t.intervals-1;
    const double x=r==hi_?1.0:(r-(lower_+cell*t.step))/t.step,x2=x*x,x3=x2*x;
    const std::size_t a=t.offset+cell*std::size_t(t.width)*2+std::size_t(n)*2;
    const std::size_t b=a+std::size_t(t.width)*2;
    v=(2*x3-3*x2+1)*knots_(a)+(x3-2*x2+x)*t.step*knots_(a+1)
      +(-2*x3+3*x2)*knots_(b)+(x3-x2)*t.step*knots_(b+1);
    d=(6*x2-6*x)/t.step*knots_(a)+(3*x2-4*x+1)*knots_(a+1)
      +(-6*x2+6*x)/t.step*knots_(b)+(3*x2-2*x)*knots_(b+1);
    return true;
  }
  KOKKOS_INLINE_FUNCTION void direct_sum(double r,double slope,double shift,int k,
      const RealView& co,std::size_t base,double& value,double& deriv) const {
    value=deriv=0;
    const bool cheb=kind_==ChebSSS||kind_==ChebSSS_LMP||kind_==ChebRational||kind_==ChebRational_LMP;
    const bool lag=kind_==Laguerre||kind_==Laguerre_LMP||kind_==LaguerrePos||kind_==LaguerrePos_LMP||kind_==LaguerreNoenv||kind_==LaguerreNoenv_LMP;
    if(!cheb&&!lag){for(int n=0;n<B_;++n){double v,d;direct_basis(r,slope,shift,k,n,v,d);
      const double c=base+n<co.extent(0)?co(base+n):0;value+=c*v;deriv+=c*d;}return;}
    // Common production families contract a single streaming recurrence. No
    // B-sized thread-local scratch and no repeated O(B^2) polynomial work.
    double f,df;cutoff(r,f,df);double q=0,dq=0,u=0,du=0,v,d;
    if(cheb){const double x=slope*(r-shift)/2;
      if(kind_==ChebRational||kind_==ChebRational_LMP){const double inv=1/::sqrt(1+x*x);q=x*inv;dq=inv*inv*inv*slope/2;}
      else{q=::tanh(x);dq=(1-q*q)*slope/2;}
      v=scaling_*f;d=df*scaling_;
    }else{
      if(kind_==LaguerrePos||kind_==LaguerrePos_LMP){slope=1e-6+softplus(slope);shift=1e-6+softplus(shift);}
      const double rho=shift>1e-8?shift:1e-8;u=slope*::log1p(r/rho);du=slope/(rho+r);
      const bool env=kind_!=LaguerreNoenv&&kind_!=LaguerreNoenv_LMP;
      const double e=env?::exp(-0.5*u):1;v=scaling_*f*e;d=scaling_*df*e;if(env)d-=0.5*du*v;
    }
    double pv=0,pd=0;
    for(int n=0;n<B_;++n){const double c=base+n<co.extent(0)?co(base+n):0;value+=c*v;deriv+=c*d;
      if(n+1==B_)break;double nv,nd;
      if(cheb){if(n==0){nv=scaling_*(q*f);nd=scaling_*(dq*f+q*df);}
        else{nv=2*q*v-pv;nd=2*(dq*v+q*d)-pd;}}
      else{const double inv=1/(n+1.0),a=(2*n+1.0-u)*inv,b=n*inv;nv=a*v-b*pv;nd=-du*inv*v+a*d-b*pd;}
      pv=v;pd=d;v=nv;d=nd;
    }
  }
};

template<class D> KOKKOS_INLINE_FUNCTION void DeviceRadial<D>::direct_basis(
    double r,double scal,double s,int k,int n,double& v,double& d) const {
  v=d=0;if(n<0||n>=B_)return;
  if(kind_==Shapeev){
    for(int i=-2;i<=n;++i){const double c=shape_(std::size_t(n)*14+i+2);
#ifdef __CUDA_ARCH__
      // Native CPU monomial accumulation rounds each product and then the
      // addition. FMA changes cancellation-sensitive high Shapeev rows by
      // many output ulps; preserve that sequence only on this CUDA path.
      const double a=__dmul_rn(scaling_,c);
      v=__dadd_rn(v,__dmul_rn(a,::pow(r,i)));
      d=__dadd_rn(d,__dmul_rn(__dmul_rn(a,static_cast<double>(i)),::pow(r,i-1)));
#else
      v+=scaling_*c*::pow(r,i);d+=scaling_*c*i*::pow(r,i-1);
#endif
    }return;
  }
  if(kind_==Taylor){
    const double z=r/hi_,f=::pow(1-z,2.0),df=-2*(1-z)/hi_;
    v=scaling_*f;if(n==0)return;
    double prev=v;for(int i=1;i<=n;++i){v=scaling_*z*prev;
      d=scaling_*(i*prev/hi_+df*v/f);prev=v;}return;
  }
  if(kind_==ChebRepuls){
    if(r<lo_)r=lo_;const double x=r/hi_,p=8,step=(hi_-lo_)/B_;
    const double f=1-(p+1)*(p+2)*::pow(x,p)*0.5+p*(p+2)*::pow(x,p+1)-(p+1)*p*::pow(x,p+2)*0.5;
    const double df=p*(p+1)*(p+2)*(::pow(x,p)-0.5*(::pow(x,p+1)+::pow(x,p-1)))/hi_;
    const double a=1-n*(1-lo_/hi_)/B_,b=1/(1-lo_/(hi_-n*step));
    const double ri=(x/a)*b,rn=(lo_/hi_/a)*b,g=::log(ri+1)-ri;
    const double gd=(1/(ri+1)-1)/a/hi_*b,norm=::log(rn+1)-rn;
    v=scaling_*g*f/norm;d=r==lo_?0:scaling_*(df*g+f*gd)/norm;return;
  }
  double f=(r-hi_)*(r-hi_),df=2*(r-hi_);
  if(kind_==Laguerre||kind_==Laguerre_LMP||kind_==LaguerrePos||kind_==LaguerrePos_LMP||kind_==LaguerreNoenv||kind_==LaguerreNoenv_LMP){
    if(kind_==LaguerrePos||kind_==LaguerrePos_LMP){scal=1e-6+softplus(scal);s=1e-6+softplus(s);}
    const double rho=s>1e-8?s:1e-8,u=scal*::log1p(r/rho),du=scal/(rho+r);
    cutoff(r,f,df);const bool env=kind_!=LaguerreNoenv&&kind_!=LaguerreNoenv_LMP;
    const double e=env?::exp(-0.5*u):1;v=scaling_*f*e;d=scaling_*df*e;
    if(env)d-=0.5*du*v;double pv=0,pd=0;
    for(int i=0;i<n;++i){const double inv=1/(i+1.0),a=(2*i+1.0-u)*inv,b=i*inv;
      const double nv=a*v-b*pv,nd=-du*inv*v+a*d-b*pd;pv=v;pd=d;v=nv;d=nd;}return;
  }
  const double x=scal*(r-s)/2;
  double q=0,dq=0;
  if(kind_==Jacobi||kind_==Jacobi_LMP||kind_==JacobiNoWeight||kind_==JacobiNoWeight_LMP){
    if(k<0||k>5){v=d=0.0/0.0;return;}
    q=::tanh(x);q=q<(-1+1e-12)?(-1+1e-12):(q>1-1e-12?1-1e-12:q);
    dq=0.5*scal*(1-q*q);const int alpha=k==0?0:(k<3?1:2),beta=k==2?1:(k==4?1:(k==5?2:0));
    double weight=1,logw=0;const bool weighted=kind_==Jacobi||kind_==Jacobi_LMP;
    const double am=(1-q)>1e-12?1-q:1e-12,ap=(1+q)>1e-12?1+q:1e-12;
    if(weighted){if(alpha==1)weight*=::sqrt(am);else if(alpha==2)weight*=am;
      if(beta==1)weight*=::sqrt(ap);else if(beta==2)weight*=ap;
      if(alpha)logw-=0.5*alpha/am;if(beta)logw+=0.5*beta/ap;}
    double y=weight,dy=weight*logw,py=0,pdy=0;
    const double lc=0.5*(alpha-beta),lx=0.5*(alpha+beta+2);
    for(int i=1;i<=n;++i){double a,b,c;
      if(i==1){a=lc+lx*q;b=lx;c=0;}else{const double z=i,A=alpha,B=beta;
        const double den=2*z*(z+A+B)*(2*z+A+B-2),t=2*z+A+B-1;
        const double ax=t*(2*z+A+B)*(2*z+A+B-2)/den;
        a=t*(A*A-B*B)/den+ax*q;b=ax;c=2*(z+A-1)*(z+B-1)*(2*z+A+B)/den;}
      const double ny=a*y-c*py,nd=b*y+a*dy-c*pdy;py=y;pdy=dy;y=ny;dy=nd;}
    cutoff(r,f,df);v=scaling_*f*y;d=scaling_*df*y+scaling_*f*dy*dq;return;
  }
  if(kind_==Cheb){q=(2*r-(lo_+hi_))/(hi_-lo_);dq=2/(hi_-lo_);}
  else if(kind_==ChebS||kind_==ChebS_LMP||kind_==ChebSSSS||(kind_==ChebSigma&&k%3==1)){
    const double em=::exp(-x),xp=1+x;q=-2*::exp(xp*em-1)+1;dq=2*::exp((-1+em)*xp)*x*scal/2;
  }else if(kind_==ChebTanhExp||kind_==ChebTanhExpW||(kind_==ChebSigma&&k%3==2)){
    const double e=::exp(-0.5559-x);double offset=(k+1)*0.04;
    if(kind_==ChebTanhExpW)offset=k%3==0?0.02:(k%3==1?0.15:0.5);
    if(kind_==ChebSigma)offset=0.02;const double t=::tanh(e+offset),z=2*t-1;
    q=2*z*z-1;dq=-8*z*e*(1-t*t)*scal/2;
  }else if(kind_==ChebTri){const double pi=3.141592654,z=scal*(r-s)*0.05*pi;q=::sin(z);dq=::cos(z)*scal*0.05*pi;}
  else if(kind_==ChebSSS||kind_==ChebSSS_LMP||kind_==ChebSSSW||kind_==ChebSSSW_LMP||kind_==BesselSSS||kind_==BesselSSSW){
    const double w=kind_==BesselSSSW?(1+k)*0.1:1.0;q=::tanh(w*x);dq=(1-q*q)*w*scal/2;
    if(kind_==ChebSSSW||kind_==ChebSSSW_LMP){double power=1+2*k;
      if(kind_==ChebSSSW){if(k<0||k>5){v=d=0.0/0.0;return;}power=k==0?1.0:(k==1?5.0/7:(k==2?9.0/7:(k==3?3.0/7:(k==4?11.0/7:13.0/7))));}
      if(power!=1){dq=power*::pow(q,power-1)*dq;q=::pow(q,power);}}
  }else{
    const double a=(kind_==ChebSSW||kind_==ChebSSW_LMP)?::exp(double(k-1)):(kind_==BesselW?1.0+k:1.0);
    const double den=x*x+a,root=::sqrt(den);q=x/root;dq=a/(den*root)*scal/2;
    if(kind_==ChebRational||kind_==ChebRational_LMP){const double inv=1/::sqrt(1+x*x);q=x*inv;dq=inv*inv*inv*scal/2;}
  }
  if(kind_==Bessel||kind_==BesselW||kind_==BesselSSS||kind_==BesselSSSW){
    const double z=r/hi_-1,p=6;
    f=1-(p+1)*(p+2)*::pow(z,p)*0.5+p*(p+2)*::pow(z,p+1)-(p+1)*p*::pow(z,p+2)*0.5;
    df=p*(p+1)*(p+2)*(::pow(z,p)-0.5*(::pow(z,p+1)+::pow(z,p-1)))/hi_;
    const double t=q+1.00001,w=3.141592654*(n+1)/2,si=::sin(w*t),co=::cos(w*t),b=si/t;
    v=scaling_*b*f;d=scaling_*((w*co/t-si/t/t)*dq*f+b*df);return;
  }
  // Preserve historical split-family initialization, including its zero
  // radial derivative at the constant row of secondary blocks.
  bool secondary=false;
  if(kind_==ChebSSSS||kind_==ChebSSSSS){
    const int width=B_/(kind_==ChebSSSS?2:3),block=width?n/width:0;
    if(width<2||block>=(kind_==ChebSSSS?2:3))return;
    secondary=block>0;n%=width;
    if(kind_==ChebSSSS&&secondary){const double den=x*x+1,root=::sqrt(den);q=x/root;dq=scal/(den*root)/2;}
    if(kind_==ChebSSSSS){const double den=x*x+1,root=::sqrt(den),a=x/root,da=scal/(den*root)/2;
      q=block==0?a:(block==1?2*a*a-1:a*a*a);dq=block==0?da:(block==1?4*a*da:3*a*a*da);}
  }
  if(kind_==ChebSSS||kind_==ChebSSS_LMP||kind_==ChebRational||kind_==ChebRational_LMP)cutoff(r,f,df);
  v=scaling_*f;d=secondary?0:df*scaling_;
  if(n==0){if(kind_==ChebTanhExpW){v*=1-q;d=(df*(1-q)-dq*f)*scaling_;}return;}
  double pv=v,pd=d;v=scaling_*(q*f);d=scaling_*(dq*f+q*df);
  if(kind_==ChebTanhExpW){const double weight=1-q;
    pv=scaling_*f*weight;pd=(df*weight-dq*f)*scaling_;
    v=scaling_*(q*f)*weight;d=scaling_*((dq*f+q*df)*weight-dq*(q*f));}
  for(int i=2;i<=n;++i){const double nv=2*q*v-pv,nd=2*(dq*v+q*d)-pd;pv=v;pd=d;v=nv;d=nd;}
}

template<class D> const char* DeviceRadial<D>::name(int kind) {
  static const char* names[]={"RBShapeev","RBChebyshev","RBChebyshev_s","RBChebyshev_ss","RBChebyshev_ssw","RBChebyshev_sss","RBChebyshev_sss_rational","RBChebyshev_sssw","RBChebyshev_sss_lmp","RBChebyshev_sss_rational_lmp","RBChebyshev_sssw_lmp","RBChebyshev_ss_lmp","RBChebyshev_ssw_lmp","RBChebyshev_s_lmp","RBChebyshev_ssss","RBChebyshev_sssss","RBChebyshev_sigma","RBLaguerre_log1p","RBLaguerre_log1p_lmp","RBLaguerre_log1p_pos","RBLaguerre_log1p_pos_lmp","RBLaguerre_log1p_noenv","RBLaguerre_log1p_noenv_lmp","RBJacobi_sss","RBJacobi_sss_lmp","RBJacobi_sss_noweight","RBJacobi_sss_noweight_lmp","RBBessel","RBBesselw","RBChebyshev_Tri","RBBessel_sss","RBBessel_sssw","RBChebyshev_tanhexp","RBChebyshev_tanhexp_w","RBChebyshev_repuls","RBTaylor"};
  return names[kind];
}

template<class D> DeviceRadial<D>::DeviceRadial(const Model& m) {
  const std::string type=m.text("radial_basis_type");kind_=-1;
  for(int i=Shapeev;i<=Taylor;++i)if(type==name(i))kind_=i;
  if(kind_<0)throw std::invalid_argument("Unsupported native device radial basis: "+type);
  B_=m.radial_size;C_=m.species_count;F_=m.radial_funcs;L_=m.lmax+1;K_=m.kmax;
  O_=m.integer("outer_k_rank",1);lo_=m.min_dist;hi_=m.max_dist;scaling_=m.scaling;
  lower_=std::max(0.0,lo_);p_=m.integer("radial_cutoff_p",5);
  const auto cutoff_name=m.text("radial_cutoff_type","legacy");
  if(cutoff_name!="legacy"&&cutoff_name!="polynomial")throw std::invalid_argument("Unknown radial cutoff");
  polynomial_=cutoff_name=="polynomial";
  if(B_<=0||C_<=0||F_<=0||L_<=0||!std::isfinite(lo_)||!std::isfinite(hi_)||hi_<=lo_||!std::isfinite(scaling_)||p_<1||p_>64||(polynomial_&&hi_<=0))throw std::invalid_argument("Invalid device radial metadata");
  if(kind_==Shapeev&&B_>12)throw std::invalid_argument("RBShapeev supports at most 12 basis functions");
  const std::string sm=m.text("scaling_map","LK");sm_=sm=="pair"?0:(sm=="L"?1:(sm=="K"?2:3));
  frozen_outer_=m.text("outer_radial_mode","legacy-plus-mixer")=="legacy-plus-mixer";
  needs_basis_=!frozen_outer_||m.text("two_layer_gate_radial_mixing","none")=="full";
  basis_channels_=m.text("outer_radial_mode")=="adaptive-coefficients"?O_:K_;
  scale_=upload("SUS2 radial warp",m.numbers("scal_coeffs"));
  gate_scale_=upload("SUS2 gate radial warp",m.numbers("two_layer_gate_scaling_coeffs"));
  coeff_=upload("SUS2 radial coefficients",m.radial_coeffs);
  gate_coeff_=upload("SUS2 gate radial coefficients",m.numbers("two_layer_gate_radial_coeffs"));
  if(kind_==Shapeev)shape_=upload("SUS2 Shapeev coefficients",shape_coefficients(lo_,hi_));
  counters_=CountView("SUS2 radial counters",5);Kokkos::deep_copy(counters_,std::uint64_t(0));
}

template<class D> KOKKOS_INLINE_FUNCTION void DeviceRadial<D>::basis(
    double r,double slope,double shift,int k,int n,double& v,double& d) const {
  if(n<0||n>=B_){v=d=0;return;}
  if(!table_mode_){direct_basis(r,slope,shift,k,n,v,d);return;}
  if(!finite(r)||r<lower_||r>hi_){count(false,2,n==0);direct_basis(r,slope,shift,k,n,v,d);return;}
  const std::uint64_t a=bits(slope),b=bits(shift);
  std::size_t lo=0,hi=basis_tables_.extent(0);
  while(lo<hi){const std::size_t mid=lo+(hi-lo)/2;const Table t=basis_tables_(mid);
    const bool less=t.slope_bits<a||(t.slope_bits==a&&(t.shift_bits<b||(t.shift_bits==b&&t.channel<k)));
    if(less)lo=mid+1;else hi=mid;}
  if(lo<basis_tables_.extent(0)){const Table t=basis_tables_(lo);
    if(t.slope_bits==a&&t.shift_bits==b&&t.channel==k){
      if(interpolate(t,r,n,v,d)){count(true,0,n==0);return;}
      count(false,t.status==3?3:4,n==0);direct_basis(r,slope,shift,k,n,v,d);return;}}
  count(false,4,n==0);direct_basis(r,slope,shift,k,n,v,d);
}

template<class D> KOKKOS_INLINE_FUNCTION void DeviceRadial<D>::contract(
    double r,int ti,int tj,int k,int l,bool gate,double& v,double& d) const {
  v=d=0;const int mu=k*L_+l;if(ti<0||tj<0||ti>=C_||tj>=C_||mu<0||mu>=F_)return;
  const std::size_t pairs=std::size_t(C_)*C_,pair=std::size_t(ti)*C_+tj;
  const int sb=sm_==0?0:(sm_==1?l:(sm_==2?k:mu));
  const std::size_t off=2*std::size_t(sb)*pairs+pair;
  const RealView sc=gate&&gate_scale_.extent(0)?gate_scale_:scale_;
  const double slope=off<sc.extent(0)?sc(off):1,shift=off+pairs<sc.extent(0)?sc(off+pairs):0;
  const RealView co=gate&&gate_coeff_.extent(0)?gate_coeff_:coeff_;
  const std::size_t base=std::size_t(mu)*(gate&&gate_coeff_.extent(0)?B_:B_+C_);
  if(!table_mode_){direct_sum(r,slope,shift,k,co,base,v,d);return;}
  if(table_mode_){
    const int a=species_slots_.extent(0)?species_slots_(ti):-1,b=species_slots_.extent(0)?species_slots_(tj):-1;
    if(a>=0&&b>=0&&frozen_handles_.extent(0)){
      const std::size_t at=((std::size_t(gate)*active_count_+a)*active_count_+b)*F_+mu;
      const int handle=frozen_handles_(at);
      if(handle>=0){const Table t=frozen_tables_(handle);
        if(interpolate(t,r,0,v,d)){count(true,0);return;}
        count(false,!finite(r)||r<lower_||r>hi_?2:(t.status==3?3:4));
        direct_sum(r,slope,shift,k,co,base,v,d);return;
      }
    }
  }
  for(int n=0;n<B_;++n){double bv,bd;basis(r,slope,shift,k,n,bv,bd);const double c=base+n<co.extent(0)?co(base+n):0;v+=c*bv;d+=c*bd;}
}

template<class D> void DeviceRadial<D>::configure(RadialTableMode mode,double spacing,
    std::size_t max_bytes,const std::vector<int>& active_species) {
  if(mode!=RadialTableMode::Direct&&mode!=RadialTableMode::Table&&mode!=RadialTableMode::Auto)
    throw std::invalid_argument("Unknown device radial table mode");
  if(!(spacing>0)||!std::isfinite(spacing))throw std::invalid_argument("Radial table spacing must be finite and positive");
  // Configuration runs only after outstanding kernels have completed. Clearing
  // the handles first releases old knot storage before building the new cache.
  Kokkos::fence();knots_=RealView();basis_tables_=TableView();frozen_tables_=TableView();
  frozen_handles_=IntView();species_slots_=IntView();table_bytes_=table_count_=0;
  table_mode_=mode==RadialTableMode::Direct?0:1;active_count_=0;
  if(counters_.extent(0))Kokkos::deep_copy(counters_,std::uint64_t(0));
  if(!table_mode_)return;
  std::vector<int> used=active_species;
  for(int t:used)if(t<0||t>=C_)throw std::out_of_range("Device radial active species index");
  std::sort(used.begin(),used.end());used.erase(std::unique(used.begin(),used.end()),used.end());
  if(used.empty())return;
  active_count_=int(used.size());std::vector<int> slots(C_,-1);
  for(int i=0;i<active_count_;++i)slots[used[i]]=i;
  species_slots_=upload("SUS2 active radial species",slots);
  auto hs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),scale_);
  auto hgs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gate_scale_);
  auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),coeff_);
  auto hgc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gate_coeff_);
  Radial reference(name(kind_),lo_,hi_,B_,scaling_,polynomial_?"polynomial":"legacy",p_);
  reference.configure(RadialTableMode::Direct);
  std::vector<double> storage;
  std::vector<Table> frozen,generic;
  std::vector<double> bv,bd;
  const double count_intervals=std::ceil((hi_-lower_)/spacing);
  auto build=[&](Table& t,const std::vector<double>* coefficients){
    t.width=coefficients?1:B_;t.status=2;
    if(!std::isfinite(t.slope)||!std::isfinite(t.shift)||!(hi_>lower_))return;
    if(coefficients)for(double c:*coefficients)if(!std::isfinite(c))return;
    const std::size_t width=std::size_t(t.width)*2,available=max_bytes-storage.size()*sizeof(double);
    if(!std::isfinite(count_intervals)||count_intervals<1||count_intervals>=double(std::numeric_limits<std::size_t>::max())){t.status=3;return;}
    const std::size_t intervals=std::size_t(count_intervals);
    if(intervals>=available/sizeof(double)/width||intervals>=(storage.max_size()-storage.size())/width){t.status=3;return;}
    t.intervals=intervals;t.step=(hi_-lower_)/double(intervals);
    const std::size_t old=storage.size(),length=(intervals+1)*width;
    try {
      storage.resize(old+length);t.offset=old;
      for(std::size_t i=0;i<=intervals;++i){
        const double r=i==intervals?hi_:lower_+i*t.step;reference.eval(r,t.slope,t.shift,t.channel,bv,bd);
        double value=0,der=0;
        for(int n=0;n<B_;++n){if(!std::isfinite(bv[n])||!std::isfinite(bd[n])){storage.resize(old);return;}
          if(coefficients){value+=(*coefficients)[n]*bv[n];der+=(*coefficients)[n]*bd[n];}
          else{storage[old+i*width+2*n]=bv[n];storage[old+i*width+2*n+1]=bd[n];}}
        if(coefficients){if(!std::isfinite(value)||!std::isfinite(der)){storage.resize(old);return;}
          storage[old+i*width]=value;storage[old+i*width+1]=der;}
      }
      std::vector<double> lv,ld,rv,rd;
      for(std::size_t i=0;i<intervals;++i){
        if(coefficients){
          reference.eval(lower_+i*t.step,t.slope,t.shift,t.channel,lv,ld);
          reference.eval(i+1==intervals?hi_:lower_+(i+1)*t.step,t.slope,t.shift,t.channel,rv,rd);
        }
        reference.eval(lower_+(i+0.5)*t.step,t.slope,t.shift,t.channel,bv,bd);
        double value=0,der=0;
        for(int n=0;n<B_;++n){if(!std::isfinite(bv[n])||!std::isfinite(bd[n])){storage.resize(old);return;}
          if(coefficients){
            const double hv=.5*(lv[n]+rv[n])+.125*t.step*(ld[n]-rd[n]);
            const double hd=1.5*(rv[n]-lv[n])/t.step-.25*(ld[n]+rd[n]);
            if(std::abs(hv-bv[n])>1e-7*(1+std::abs(bv[n]))||std::abs(hd-bd[n])>1e-4*(1+std::abs(bd[n]))){storage.resize(old);return;}
            value+=(*coefficients)[n]*bv[n];der+=(*coefficients)[n]*bd[n];}
          else{const std::size_t a=old+i*width+2*n,b=a+width;
            const double hv=.5*(storage[a]+storage[b])+.125*t.step*(storage[a+1]-storage[b+1]);
            const double hd=1.5*(storage[b]-storage[a])/t.step-.25*(storage[a+1]+storage[b+1]);
            if(std::abs(hv-bv[n])>1e-7*(1+std::abs(bv[n]))||std::abs(hd-bd[n])>1e-4*(1+std::abs(bd[n]))){storage.resize(old);return;}}}
        if(coefficients){const std::size_t a=old+i*width,b=a+width;
          const double hv=.5*(storage[a]+storage[b])+.125*t.step*(storage[a+1]-storage[b+1]);
          const double hd=1.5*(storage[b]-storage[a])/t.step-.25*(storage[a+1]+storage[b+1]);
          if(!std::isfinite(value)||!std::isfinite(der)||std::abs(hv-value)>1e-7*(1+std::abs(value))||std::abs(hd-der)>1e-4*(1+std::abs(der))){storage.resize(old);return;}}
      }
      t.status=1;++table_count_;
    }catch(const std::bad_alloc&){storage.resize(old);t.status=3;}
    catch(const std::length_error&){storage.resize(old);t.status=3;}
  };
  using Key=std::vector<std::uint64_t>;
  std::map<Key,int> frozen_ids;
  std::map<Key,Table> basis_requests;
  const std::size_t pairs=std::size_t(C_)*C_,used_pairs=used.size()*used.size();
  if(used_pairs>std::numeric_limits<std::size_t>::max()/2/std::size_t(F_))throw std::length_error("Device radial handle count exceeds size_t");
  std::vector<int> handles(2*used_pairs*std::size_t(F_),-1);
  auto request=[&](bool gate,int ti,int tj,int k,int l,bool contracted){
    const int mu=k*L_+l,sb=sm_==0?0:(sm_==1?l:(sm_==2?k:mu));
    const std::size_t pair=std::size_t(ti)*C_+tj,off=2*std::size_t(sb)*pairs+pair;
    const auto& sc=gate&&hgs.extent(0)?hgs:hs;
    Table t;t.channel=k;t.slope=off<sc.extent(0)?sc(off):1;t.shift=off+pairs<sc.extent(0)?sc(off+pairs):0;
    t.slope_bits=bits(t.slope);t.shift_bits=bits(t.shift);
    Key key{t.slope_bits,t.shift_bits,std::uint64_t(k)};
    if(!contracted){basis_requests.emplace(key,t);return;}
    const auto& co=gate&&hgc.extent(0)?hgc:hc;const std::size_t base=std::size_t(mu)*(gate&&hgc.extent(0)?B_:B_+C_);
    std::vector<double> cs(B_);for(int n=0;n<B_;++n){cs[n]=base+n<co.extent(0)?co(base+n):0;key.push_back(bits(cs[n]));}
    auto found=frozen_ids.find(key);int id;
    if(found!=frozen_ids.end())id=found->second;
    else{if(frozen.size()>=std::size_t(std::numeric_limits<int>::max()))throw std::length_error("Device radial table index exceeds int");
      id=int(frozen.size());build(t,&cs);frozen.push_back(t);frozen_ids.emplace(std::move(key),id);}
    handles[((std::size_t(gate)*active_count_+slots[ti])*active_count_+slots[tj])*F_+mu]=id;
  };
  // Fixed contractions get first admission: their scalar knots are much
  // smaller than a basis-vector table. Adaptive coefficients are never frozen.
  for(int gate=0;gate<2;++gate)if(gate||frozen_outer_)
    for(int ti:used)for(int tj:used)for(int mu=0;mu<F_;++mu)request(gate,ti,tj,mu/L_,mu%L_,true);
  if(needs_basis_)for(int ti:used)for(int tj:used)for(int k=0;k<basis_channels_;++k)for(int l=0;l<L_;++l)
    request(false,ti,tj,k,l,false);
  for(auto& item:basis_requests){build(item.second,nullptr);generic.push_back(item.second);}
  table_bytes_=storage.size()*sizeof(double);
  frozen_tables_=upload("SUS2 frozen radial handles",frozen);
  basis_tables_=upload("SUS2 radial basis handles",generic);
  frozen_handles_=upload("SUS2 radial request handles",handles);
  knots_=upload("SUS2 immutable radial knots",storage);
}

template<class D> RadialTableStats DeviceRadial<D>::stats() const {
  RadialTableStats s;s.tables=table_count_;s.bytes=table_bytes_;
  if(counters_.extent(0)){auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counters_);
    s.hits=h(0);s.direct_fallbacks=h(1);s.domain_fallbacks=h(2);s.memory_fallbacks=h(3);s.invalid_fallbacks=h(4);}return s;
}
template<class D> std::size_t DeviceRadial<D>::memory_usage() const {
  return sizeof(double)*(scale_.extent(0)+gate_scale_.extent(0)+coeff_.extent(0)+gate_coeff_.extent(0)+shape_.extent(0)+knots_.extent(0))
    +sizeof(int)*(species_slots_.extent(0)+frozen_handles_.extent(0))
    +sizeof(Table)*(basis_tables_.extent(0)+frozen_tables_.extent(0))+sizeof(std::uint64_t)*counters_.extent(0);
}

template<class D> std::vector<double> DeviceRadial<D>::shape_coefficients(double min_dist,double max_dist) {
  using std::sqrt;using std::pow;double rb_coeffs[12][14]{};
	rb_coeffs[0][0] = (sqrt(21)*pow(max_dist, 2)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][1] = -((sqrt(21)*max_dist*pow(min_dist, 2)) / pow(max_dist - min_dist, 2));
	rb_coeffs[0][2] = (sqrt(21)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][3] = 0;
	rb_coeffs[0][4] = 0;
	rb_coeffs[0][5] = 0;
	rb_coeffs[0][6] = 0;
	rb_coeffs[0][7] = 0;
	rb_coeffs[0][8] = 0;
	rb_coeffs[0][9] = 0;
	rb_coeffs[0][10] = 0;
	rb_coeffs[0][11] = 0;
	rb_coeffs[0][12] = 0;
	rb_coeffs[0][13] = 0;
	rb_coeffs[1][0] = (-3 * sqrt(7)*pow(max_dist, 2)*pow(min_dist, 2)*(max_dist + 3 * min_dist)) / (4.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][1] = (9 * sqrt(7)*max_dist*pow(min_dist, 2)*(max_dist + min_dist)) / (2.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][2] = (9 * sqrt(7)*pow(min_dist, 2)*(3 * max_dist + min_dist)) / (4.*pow(-max_dist + min_dist, 3));
	rb_coeffs[1][3] = (3 * sqrt(7)*pow(min_dist, 2)) / pow(max_dist - min_dist, 3);
	rb_coeffs[1][4] = 0;
	rb_coeffs[1][5] = 0;
	rb_coeffs[1][6] = 0;
	rb_coeffs[1][7] = 0;
	rb_coeffs[1][8] = 0;
	rb_coeffs[1][9] = 0;
	rb_coeffs[1][10] = 0;
	rb_coeffs[1][11] = 0;
	rb_coeffs[1][12] = 0;
	rb_coeffs[1][13] = 0;
	rb_coeffs[2][0] = (sqrt(3.6666666666666665)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 2) + 7 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][1] = -((sqrt(3.6666666666666665)*max_dist*pow(min_dist, 2)*(11 * pow(max_dist, 2) + 35 * max_dist*min_dist + 14 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][2] = (sqrt(3.6666666666666665)*pow(min_dist, 2)*(34 * pow(max_dist, 2) + 49 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][3] = -((sqrt(33)*pow(min_dist, 2)*(13 * max_dist + 7 * min_dist)) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][4] = (5 * sqrt(33)*pow(min_dist, 2)) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][5] = 0;
	rb_coeffs[2][6] = 0;
	rb_coeffs[2][7] = 0;
	rb_coeffs[2][8] = 0;
	rb_coeffs[2][9] = 0;
	rb_coeffs[2][10] = 0;
	rb_coeffs[2][11] = 0;
	rb_coeffs[2][12] = 0;
	rb_coeffs[2][13] = 0;
	rb_coeffs[3][0] = (-3 * sqrt(6.5)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 3) + 12 * pow(max_dist, 2)*min_dist + 28 * max_dist*pow(min_dist, 2) + 14 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][1] = (3 * sqrt(6.5)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 3) + 104 * pow(max_dist, 2)*min_dist + 126 * max_dist*pow(min_dist, 2) + 28 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][2] = (3 * sqrt(6.5)*pow(min_dist, 2)*(43 * pow(max_dist, 3) + 141 * pow(max_dist, 2)*min_dist + 84 * max_dist*pow(min_dist, 2) + 7 * pow(min_dist, 3))) / (2.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][3] = (15 * sqrt(6.5)*pow(min_dist, 2)*(18 * pow(max_dist, 2) + 30 * max_dist*min_dist + 7 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][4] = (165 * sqrt(6.5)*pow(min_dist, 2)*(3 * max_dist + 2 * min_dist)) / (4.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][5] = (165 * sqrt(6.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][6] = 0;
	rb_coeffs[3][7] = 0;
	rb_coeffs[3][8] = 0;
	rb_coeffs[3][9] = 0;
	rb_coeffs[3][10] = 0;
	rb_coeffs[3][11] = 0;
	rb_coeffs[3][12] = 0;
	rb_coeffs[3][13] = 0;
	rb_coeffs[4][0] = (sqrt(0.6)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 4) + 90 * pow(max_dist, 3)*min_dist + 360 * pow(max_dist, 2)*pow(min_dist, 2) + 420 * max_dist*pow(min_dist, 3) + 126 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][1] = (-3 * sqrt(0.6)*max_dist*pow(min_dist, 2)*(20 * pow(max_dist, 4) + 195 * pow(max_dist, 3)*min_dist + 450 * pow(max_dist, 2)*pow(min_dist, 2) + 294 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][2] = (3 * sqrt(0.6)*pow(min_dist, 2)*(295 * pow(max_dist, 4) + 1680 * pow(max_dist, 3)*min_dist + 2232 * pow(max_dist, 2)*pow(min_dist, 2) + 756 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][3] = (-22 * sqrt(0.6)*pow(min_dist, 2)*(65 * pow(max_dist, 3) + 216 * pow(max_dist, 2)*min_dist + 153 * max_dist*pow(min_dist, 2) + 21 * pow(min_dist, 3))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][4] = (33 * sqrt(0.6)*pow(min_dist, 2)*(137 * pow(max_dist, 2) + 246 * max_dist*min_dist + 72 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][5] = (-429 * sqrt(0.6)*pow(min_dist, 2)*(4 * max_dist + 3 * min_dist)) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][6] = (1001 * sqrt(0.6)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][7] = 0;
	rb_coeffs[4][8] = 0;
	rb_coeffs[4][9] = 0;
	rb_coeffs[4][10] = 0;
	rb_coeffs[4][11] = 0;
	rb_coeffs[4][12] = 0;
	rb_coeffs[4][13] = 0;
	rb_coeffs[5][0] = -(sqrt(62.333333333333336)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 5) + 25 * pow(max_dist, 4)*min_dist + 150 * pow(max_dist, 3)*pow(min_dist, 2) + 300 * pow(max_dist, 2)*pow(min_dist, 3) + 210 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][1] = (sqrt(62.333333333333336)*max_dist*pow(min_dist, 2)*(16 * pow(max_dist, 5) + 225 * pow(max_dist, 4)*min_dist + 825 * pow(max_dist, 3)*pow(min_dist, 2) + 1020 * pow(max_dist, 2)*pow(min_dist, 3) + 420 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][2] = -(sqrt(561)*pow(min_dist, 2)*(107 * pow(max_dist, 5) + 925 * pow(max_dist, 4)*min_dist + 2120 * pow(max_dist, 3)*pow(min_dist, 2) + 1580 * pow(max_dist, 2)*pow(min_dist, 3) + 350 * max_dist*pow(min_dist, 4) + 14 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][3] = (5 * sqrt(62.333333333333336)*pow(min_dist, 2)*(73 * pow(max_dist, 4) + 397 * pow(max_dist, 3)*min_dist + 555 * pow(max_dist, 2)*pow(min_dist, 2) + 228 * max_dist*pow(min_dist, 3) + 21 * pow(min_dist, 4))) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][4] = (-65 * sqrt(62.333333333333336)*pow(min_dist, 2)*(53 * pow(max_dist, 3) + 177 * pow(max_dist, 2)*min_dist + 138 * max_dist*pow(min_dist, 2) + 24 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][5] = (91 * sqrt(561)*pow(min_dist, 2)*(8 * pow(max_dist, 2) + 15 * max_dist*min_dist + 5 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][6] = (-91 * sqrt(62.333333333333336)*pow(min_dist, 2)*(31 * max_dist + 25 * min_dist)) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][7] = (182 * sqrt(62.333333333333336)*pow(min_dist, 2)) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][8] = 0;
	rb_coeffs[5][9] = 0;
	rb_coeffs[5][10] = 0;
	rb_coeffs[5][11] = 0;
	rb_coeffs[5][12] = 0;
	rb_coeffs[5][13] = 0;
	rb_coeffs[6][0] = (sqrt(4.071428571428571)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 6) + 33 * pow(max_dist, 5)*min_dist + 275 * pow(max_dist, 4)*pow(min_dist, 2) + 825 * pow(max_dist, 3)*pow(min_dist, 3) + 990 * pow(max_dist, 2)*pow(min_dist, 4) + 462 * max_dist*pow(min_dist, 5) + 66 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][1] = -((sqrt(4.071428571428571)*max_dist*pow(min_dist, 2)*(41 * pow(max_dist, 6) + 781 * pow(max_dist, 5)*min_dist + 4125 * pow(max_dist, 4)*pow(min_dist, 2) + 8085 * pow(max_dist, 3)*pow(min_dist, 3) + 6270 * pow(max_dist, 2)*pow(min_dist, 4) + 1782 * max_dist*pow(min_dist, 5) + 132 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8));
	rb_coeffs[6][2] = (3 * sqrt(16.285714285714285)*pow(min_dist, 2)*(89 * pow(max_dist, 6) + 1078 * pow(max_dist, 5)*min_dist + 3740 * pow(max_dist, 4)*pow(min_dist, 2) + 4785 * pow(max_dist, 3)*pow(min_dist, 3) + 2310 * pow(max_dist, 2)*pow(min_dist, 4) + 363 * max_dist*pow(min_dist, 5) + 11 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][3] = (-13 * sqrt(16.285714285714285)*pow(min_dist, 2)*(124 * pow(max_dist, 5) + 990 * pow(max_dist, 4)*min_dist + 2255 * pow(max_dist, 3)*pow(min_dist, 2) + 1815 * pow(max_dist, 2)*pow(min_dist, 3) + 495 * max_dist*pow(min_dist, 4) + 33 * pow(min_dist, 5))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][4] = (65 * sqrt(4.071428571428571)*pow(min_dist, 2)*(161 * pow(max_dist, 4) + 847 * pow(max_dist, 3)*min_dist + 1221 * pow(max_dist, 2)*pow(min_dist, 2) + 561 * max_dist*pow(min_dist, 3) + 66 * pow(min_dist, 4))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][5] = (-39 * sqrt(4.071428571428571)*pow(min_dist, 2)*(497 * pow(max_dist, 3) + 1661 * pow(max_dist, 2)*min_dist + 1375 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][6] = (26 * sqrt(16.285714285714285)*pow(min_dist, 2)*(394 * pow(max_dist, 2) + 759 * max_dist*min_dist + 275 * pow(min_dist, 2))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][7] = (-442 * sqrt(16.285714285714285)*pow(min_dist, 2)*(13 * max_dist + 11 * min_dist)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][8] = (1326 * sqrt(16.285714285714285)*pow(min_dist, 2)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][9] = 0;
	rb_coeffs[6][10] = 0;
	rb_coeffs[6][11] = 0;
	rb_coeffs[6][12] = 0;
	rb_coeffs[6][13] = 0;
	rb_coeffs[7][0] = -(sqrt(273)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 7) + 42 * pow(max_dist, 6)*min_dist + 462 * pow(max_dist, 5)*pow(min_dist, 2) + 1925 * pow(max_dist, 4)*pow(min_dist, 3) + 3465 * pow(max_dist, 3)*pow(min_dist, 4) + 2772 * pow(max_dist, 2)*pow(min_dist, 5) + 924 * max_dist*pow(min_dist, 6) + 99 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][1] = (3 * sqrt(273)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 7) + 420 * pow(max_dist, 6)*min_dist + 3003 * pow(max_dist, 5)*pow(min_dist, 2) + 8470 * pow(max_dist, 4)*pow(min_dist, 3) + 10395 * pow(max_dist, 3)*pow(min_dist, 4) + 5544 * pow(max_dist, 2)*pow(min_dist, 5) + 1155 * max_dist*pow(min_dist, 6) + 66 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][2] = (-3 * sqrt(273)*pow(min_dist, 2)*(278 * pow(max_dist, 7) + 4473 * pow(max_dist, 6)*min_dist + 21714 * pow(max_dist, 5)*pow(min_dist, 2) + 41965 * pow(max_dist, 4)*pow(min_dist, 3) + 34650 * pow(max_dist, 3)*pow(min_dist, 4) + 11781 * pow(max_dist, 2)*pow(min_dist, 5) + 1386 * max_dist*pow(min_dist, 6) + 33 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][3] = (7 * sqrt(273)*pow(min_dist, 2)*(917 * pow(max_dist, 6) + 10038 * pow(max_dist, 5)*min_dist + 33495 * pow(max_dist, 4)*pow(min_dist, 2) + 43780 * pow(max_dist, 3)*pow(min_dist, 3) + 23265 * pow(max_dist, 2)*pow(min_dist, 4) + 4554 * max_dist*pow(min_dist, 5) + 231 * pow(min_dist, 6))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][4] = (-105 * sqrt(273)*pow(min_dist, 2)*(259 * pow(max_dist, 5) + 1953 * pow(max_dist, 4)*min_dist + 4422 * pow(max_dist, 3)*pow(min_dist, 2) + 3740 * pow(max_dist, 2)*pow(min_dist, 3) + 1155 * max_dist*pow(min_dist, 4) + 99 * pow(min_dist, 5))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][5] = (21 * sqrt(273)*pow(min_dist, 2)*(1624 * pow(max_dist, 4) + 8328 * pow(max_dist, 3)*min_dist + 12243 * pow(max_dist, 2)*pow(min_dist, 2) + 6050 * max_dist*pow(min_dist, 3) + 825 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][6] = (-119 * sqrt(273)*pow(min_dist, 2)*(436 * pow(max_dist, 3) + 1455 * pow(max_dist, 2)*min_dist + 1254 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][7] = (153 * sqrt(273)*pow(min_dist, 2)*(307 * pow(max_dist, 2) + 602 * max_dist*min_dist + 231 * pow(min_dist, 2))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][8] = (-2907 * sqrt(273)*pow(min_dist, 2)*(8 * max_dist + 7 * min_dist)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][9] = (4845 * sqrt(273)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][10] = 0;
	rb_coeffs[7][11] = 0;
	rb_coeffs[7][12] = 0;
	rb_coeffs[7][13] = 0;
	rb_coeffs[8][0] = (sqrt(161)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 8) + 52 * pow(max_dist, 7)*min_dist + 728 * pow(max_dist, 6)*pow(min_dist, 2) + 4004 * pow(max_dist, 5)*pow(min_dist, 3) + 10010 * pow(max_dist, 4)*pow(min_dist, 4) + 12012 * pow(max_dist, 3)*pow(min_dist, 5) + 6864 * pow(max_dist, 2)*pow(min_dist, 6) + 1716 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][1] = -(sqrt(161)*max_dist*pow(min_dist, 2)*(31 * pow(max_dist, 8) + 962 * pow(max_dist, 7)*min_dist + 8918 * pow(max_dist, 6)*pow(min_dist, 2) + 34034 * pow(max_dist, 5)*pow(min_dist, 3) + 60060 * pow(max_dist, 4)*pow(min_dist, 4) + 50622 * pow(max_dist, 3)*pow(min_dist, 5) + 19734 * pow(max_dist, 2)*pow(min_dist, 6) + 3146 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][2] = (sqrt(161)*pow(min_dist, 2)*(1241 * pow(max_dist, 8) + 25532 * pow(max_dist, 7)*min_dist + 164528 * pow(max_dist, 6)*pow(min_dist, 2) + 444444 * pow(max_dist, 5)*pow(min_dist, 3) + 553410 * pow(max_dist, 4)*pow(min_dist, 4) + 320892 * pow(max_dist, 3)*pow(min_dist, 5) + 81224 * pow(max_dist, 2)*pow(min_dist, 6) + 7436 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][3] = (-10 * sqrt(161)*pow(min_dist, 2)*(591 * pow(max_dist, 7) + 8463 * pow(max_dist, 6)*min_dist + 38675 * pow(max_dist, 5)*pow(min_dist, 2) + 73931 * pow(max_dist, 4)*pow(min_dist, 3) + 63635 * pow(max_dist, 3)*pow(min_dist, 4) + 24167 * pow(max_dist, 2)*pow(min_dist, 5) + 3575 * max_dist*pow(min_dist, 6) + 143 * pow(min_dist, 7))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][4] = (10 * sqrt(161)*pow(min_dist, 2)*(3150 * pow(max_dist, 6) + 32032 * pow(max_dist, 5)*min_dist + 103792 * pow(max_dist, 4)*pow(min_dist, 2) + 137566 * pow(max_dist, 3)*pow(min_dist, 3) + 77935 * pow(max_dist, 2)*pow(min_dist, 4) + 17446 * max_dist*pow(min_dist, 5) + 1144 * pow(min_dist, 6))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][5] = (-68 * sqrt(161)*pow(min_dist, 2)*(1498 * pow(max_dist, 5) + 10816 * pow(max_dist, 4)*min_dist + 24349 * pow(max_dist, 3)*pow(min_dist, 2) + 21307 * pow(max_dist, 2)*pow(min_dist, 3) + 7150 * max_dist*pow(min_dist, 4) + 715 * pow(min_dist, 5))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][6] = (34 * sqrt(161)*pow(min_dist, 2)*(6102 * pow(max_dist, 4) + 30654 * pow(max_dist, 3)*min_dist + 45656 * pow(max_dist, 2)*pow(min_dist, 2) + 23738 * max_dist*pow(min_dist, 3) + 3575 * pow(min_dist, 4))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][7] = (-1292 * sqrt(161)*pow(min_dist, 2)*(207 * pow(max_dist, 3) + 689 * pow(max_dist, 2)*min_dist + 611 * max_dist*pow(min_dist, 2) + 143 * pow(min_dist, 3))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][8] = (1615 * sqrt(161)*pow(min_dist, 2)*(131 * pow(max_dist, 2) + 260 * max_dist*min_dist + 104 * pow(min_dist, 2))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][9] = (-3230 * sqrt(161)*pow(min_dist, 2)*(29 * max_dist + 26 * min_dist)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][10] = (17765 * sqrt(161)*pow(min_dist, 2)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][11] = 0;
	rb_coeffs[8][12] = 0;
	rb_coeffs[8][13] = 0;
	rb_coeffs[9][0] = -(sqrt(3)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 9) + 315 * pow(max_dist, 8)*min_dist + 5460 * pow(max_dist, 7)*pow(min_dist, 2) + 38220 * pow(max_dist, 6)*pow(min_dist, 3) + 126126 * pow(max_dist, 5)*pow(min_dist, 4) + 210210 * pow(max_dist, 4)*pow(min_dist, 5) + 180180 * pow(max_dist, 3)*pow(min_dist, 6) + 77220 * pow(max_dist, 2)*pow(min_dist, 7) + 15015 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][1] = (sqrt(3)*max_dist*pow(min_dist, 2)*(185 * pow(max_dist, 9) + 7035 * pow(max_dist, 8)*min_dist + 81900 * pow(max_dist, 7)*pow(min_dist, 2) + 405132 * pow(max_dist, 6)*pow(min_dist, 3) + 966966 * pow(max_dist, 5)*pow(min_dist, 4) + 1171170 * pow(max_dist, 4)*pow(min_dist, 5) + 720720 * pow(max_dist, 3)*pow(min_dist, 6) + 214500 * pow(max_dist, 2)*pow(min_dist, 7) + 27027 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][2] = -(sqrt(3)*pow(min_dist, 2)*(8885 * pow(max_dist, 9) + 227115 * pow(max_dist, 8)*min_dist + 1870596 * pow(max_dist, 7)*pow(min_dist, 2) + 6703788 * pow(max_dist, 6)*pow(min_dist, 3) + 11657646 * pow(max_dist, 5)*pow(min_dist, 4) + 10180170 * pow(max_dist, 4)*pow(min_dist, 5) + 4384380 * pow(max_dist, 3)*pow(min_dist, 6) + 859716 * pow(max_dist, 2)*pow(min_dist, 7) + 63063 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][3] = (6 * sqrt(3)*pow(min_dist, 2)*(4265 * pow(max_dist, 8) + 77196 * pow(max_dist, 7)*min_dist + 461188 * pow(max_dist, 6)*pow(min_dist, 2) + 1206296 * pow(max_dist, 5)*pow(min_dist, 3) + 1516515 * pow(max_dist, 4)*pow(min_dist, 4) + 930930 * pow(max_dist, 3)*pow(min_dist, 5) + 266266 * pow(max_dist, 2)*pow(min_dist, 6) + 30888 * max_dist*pow(min_dist, 7) + 1001 * pow(min_dist, 8))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][4] = (-51 * sqrt(3)*pow(min_dist, 2)*(3274 * pow(max_dist, 7) + 43022 * pow(max_dist, 6)*min_dist + 187824 * pow(max_dist, 5)*pow(min_dist, 2) + 355810 * pow(max_dist, 4)*pow(min_dist, 3) + 315315 * pow(max_dist, 3)*pow(min_dist, 4) + 129129 * pow(max_dist, 2)*pow(min_dist, 5) + 22022 * max_dist*pow(min_dist, 6) + 1144 * pow(min_dist, 7))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][5] = (714 * sqrt(3)*pow(min_dist, 2)*(942 * pow(max_dist, 6) + 9054 * pow(max_dist, 5)*min_dist + 28665 * pow(max_dist, 4)*pow(min_dist, 2) + 38350 * pow(max_dist, 3)*pow(min_dist, 3) + 22737 * pow(max_dist, 2)*pow(min_dist, 4) + 5577 * max_dist*pow(min_dist, 5) + 429 * pow(min_dist, 6))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][6] = (-6783 * sqrt(3)*pow(min_dist, 2)*(258 * pow(max_dist, 5) + 1800 * pow(max_dist, 4)*min_dist + 4030 * pow(max_dist, 3)*pow(min_dist, 2) + 3614 * pow(max_dist, 2)*pow(min_dist, 3) + 1287 * max_dist*pow(min_dist, 4) + 143 * pow(min_dist, 5))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][7] = (1938 * sqrt(3)*pow(min_dist, 2)*(1545 * pow(max_dist, 4) + 7630 * pow(max_dist, 3)*min_dist + 11466 * pow(max_dist, 2)*pow(min_dist, 2) + 6188 * max_dist*pow(min_dist, 3) + 1001 * pow(min_dist, 4))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][8] = (-969 * sqrt(3)*pow(min_dist, 2)*(6905 * pow(max_dist, 3) + 22911 * pow(max_dist, 2)*min_dist + 20748 * max_dist*pow(min_dist, 2) + 5096 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][9] = (3553 * sqrt(3)*pow(min_dist, 2)*(661 * pow(max_dist, 2) + 1323 * max_dist*min_dist + 546 * pow(min_dist, 2))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][10] = (-81719 * sqrt(3)*pow(min_dist, 2)*(23 * max_dist + 21 * min_dist)) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][11] = (163438 * sqrt(3)*pow(min_dist, 2)) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][12] = 0;
	rb_coeffs[9][13] = 0;
	rb_coeffs[10][0] = (3 * sqrt(0.5454545454545454)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 10) + 75 * pow(max_dist, 9)*min_dist + 1575 * pow(max_dist, 8)*pow(min_dist, 2) + 13650 * pow(max_dist, 7)*pow(min_dist, 3) + 57330 * pow(max_dist, 6)*pow(min_dist, 4) + 126126 * pow(max_dist, 5)*pow(min_dist, 5) + 150150 * pow(max_dist, 4)*pow(min_dist, 6) + 96525 * pow(max_dist, 3)*pow(min_dist, 7) + 32175 * pow(max_dist, 2)*pow(min_dist, 8) + 5005 * max_dist*pow(min_dist, 9) + 273 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][1] = (-9 * sqrt(0.5454545454545454)*max_dist*pow(min_dist, 2)*(29 * pow(max_dist, 10) + 1325 * pow(max_dist, 9)*min_dist + 18900 * pow(max_dist, 8)*pow(min_dist, 2) + 117390 * pow(max_dist, 7)*pow(min_dist, 3) + 363090 * pow(max_dist, 6)*pow(min_dist, 4) + 594594 * pow(max_dist, 5)*pow(min_dist, 5) + 525525 * pow(max_dist, 4)*pow(min_dist, 6) + 246675 * pow(max_dist, 3)*pow(min_dist, 7) + 57915 * pow(max_dist, 2)*pow(min_dist, 8) + 5915 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][2] = (9 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(822 * pow(max_dist, 10) + 25525 * pow(max_dist, 9)*min_dist + 261135 * pow(max_dist, 8)*pow(min_dist, 2) + 1195740 * pow(max_dist, 7)*pow(min_dist, 3) + 2757300 * pow(max_dist, 6)*pow(min_dist, 4) + 3360357 * pow(max_dist, 5)*pow(min_dist, 5) + 2177175 * pow(max_dist, 4)*pow(min_dist, 6) + 725010 * pow(max_dist, 3)*pow(min_dist, 7) + 113490 * pow(max_dist, 2)*pow(min_dist, 8) + 6825 * max_dist*pow(min_dist, 9) + 91 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][3] = (-255 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(397 * pow(max_dist, 9) + 8847 * pow(max_dist, 8)*min_dist + 66780 * pow(max_dist, 7)*pow(min_dist, 2) + 228228 * pow(max_dist, 6)*pow(min_dist, 3) + 392301 * pow(max_dist, 5)*pow(min_dist, 4) + 351351 * pow(max_dist, 4)*pow(min_dist, 5) + 162162 * pow(max_dist, 3)*pow(min_dist, 6) + 36270 * pow(max_dist, 2)*pow(min_dist, 7) + 3393 * max_dist*pow(min_dist, 8) + 91 * pow(min_dist, 9))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][4] = (2295 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(345 * pow(max_dist, 8) + 5676 * pow(max_dist, 7)*min_dist + 32004 * pow(max_dist, 6)*pow(min_dist, 2) + 81627 * pow(max_dist, 5)*pow(min_dist, 3) + 103285 * pow(max_dist, 4)*pow(min_dist, 4) + 66066 * pow(max_dist, 3)*pow(min_dist, 5) + 20566 * pow(max_dist, 2)*pow(min_dist, 6) + 2769 * max_dist*pow(min_dist, 7) + 117 * pow(min_dist, 8))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][5] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(444 * pow(max_dist, 7) + 5460 * pow(max_dist, 6)*min_dist + 22995 * pow(max_dist, 5)*pow(min_dist, 2) + 43225 * pow(max_dist, 4)*pow(min_dist, 3) + 39130 * pow(max_dist, 3)*pow(min_dist, 4) + 16926 * pow(max_dist, 2)*pow(min_dist, 5) + 3185 * max_dist*pow(min_dist, 6) + 195 * pow(min_dist, 7))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][6] = (20349 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(612 * pow(max_dist, 6) + 5625 * pow(max_dist, 5)*min_dist + 17475 * pow(max_dist, 4)*pow(min_dist, 2) + 23530 * pow(max_dist, 3)*pow(min_dist, 3) + 14430 * pow(max_dist, 2)*pow(min_dist, 4) + 3783 * max_dist*pow(min_dist, 5) + 325 * pow(min_dist, 6))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][7] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(3099 * pow(max_dist, 5) + 21025 * pow(max_dist, 4)*min_dist + 46830 * pow(max_dist, 3)*pow(min_dist, 2) + 42770 * pow(max_dist, 2)*pow(min_dist, 3) + 15925 * max_dist*pow(min_dist, 4) + 1911 * pow(min_dist, 5))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][8] = (43605 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(913 * pow(max_dist, 4) + 4444 * pow(max_dist, 3)*min_dist + 6720 * pow(max_dist, 2)*pow(min_dist, 2) + 3731 * max_dist*pow(min_dist, 3) + 637 * pow(min_dist, 4))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][9] = (-111435 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(352 * pow(max_dist, 3) + 1164 * pow(max_dist, 2)*min_dist + 1071 * max_dist*pow(min_dist, 2) + 273 * pow(min_dist, 3))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][10] = (334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(74 * pow(max_dist, 2) + 149 * max_dist*min_dist + 63 * pow(min_dist, 2))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][11] = (-334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(27 * max_dist + 25 * min_dist)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][12] = (1448655 * sqrt(0.5454545454545454)*pow(min_dist, 2)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][13] = 0;
	rb_coeffs[11][0] = -(sqrt(82.16666666666667)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 11) + 88 * pow(max_dist, 10)*min_dist + 2200 * pow(max_dist, 9)*pow(min_dist, 2) + 23100 * pow(max_dist, 8)*pow(min_dist, 3) + 120120 * pow(max_dist, 7)*pow(min_dist, 4) + 336336 * pow(max_dist, 6)*pow(min_dist, 5) + 528528 * pow(max_dist, 5)*pow(min_dist, 6) + 471900 * pow(max_dist, 4)*pow(min_dist, 7) + 235950 * pow(max_dist, 3)*pow(min_dist, 8) + 62920 * pow(max_dist, 2)*pow(min_dist, 9) + 8008 * max_dist*pow(min_dist, 10) + 364 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][1] = (sqrt(82.16666666666667)*max_dist*pow(min_dist, 2)*(101 * pow(max_dist, 11) + 5456 * pow(max_dist, 10)*min_dist + 93500 * pow(max_dist, 9)*pow(min_dist, 2) + 711480 * pow(max_dist, 8)*pow(min_dist, 3) + 2762760 * pow(max_dist, 7)*pow(min_dist, 4) + 5861856 * pow(max_dist, 6)*pow(min_dist, 5) + 7002996 * pow(max_dist, 5)*pow(min_dist, 6) + 4719000 * pow(max_dist, 4)*pow(min_dist, 7) + 1746030 * pow(max_dist, 3)*pow(min_dist, 8) + 331760 * pow(max_dist, 2)*pow(min_dist, 9) + 28028 * max_dist*pow(min_dist, 10) + 728 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][2] = -(sqrt(82.16666666666667)*pow(min_dist, 2)*(1667 * pow(max_dist, 11) + 61754 * pow(max_dist, 10)*min_dist + 767360 * pow(max_dist, 9)*pow(min_dist, 2) + 4363590 * pow(max_dist, 8)*pow(min_dist, 3) + 12852840 * pow(max_dist, 7)*pow(min_dist, 4) + 20762742 * pow(max_dist, 6)*pow(min_dist, 5) + 18762744 * pow(max_dist, 5)*pow(min_dist, 6) + 9390810 * pow(max_dist, 4)*pow(min_dist, 7) + 2492490 * pow(max_dist, 3)*pow(min_dist, 8) + 318890 * pow(max_dist, 2)*pow(min_dist, 9) + 16016 * max_dist*pow(min_dist, 10) + 182 * pow(min_dist, 11))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][3] = (11 * sqrt(739.5)*pow(min_dist, 2)*(809 * pow(max_dist, 10) + 21740 * pow(max_dist, 9)*min_dist + 201990 * pow(max_dist, 8)*pow(min_dist, 2) + 871920 * pow(max_dist, 7)*pow(min_dist, 3) + 1957410 * pow(max_dist, 6)*pow(min_dist, 4) + 2395484 * pow(max_dist, 5)*pow(min_dist, 5) + 1611610 * pow(max_dist, 4)*pow(min_dist, 6) + 580840 * pow(max_dist, 3)*pow(min_dist, 7) + 104520 * pow(max_dist, 2)*pow(min_dist, 8) + 8060 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][4] = (-1045 * sqrt(739.5)*pow(min_dist, 2)*(157 * pow(max_dist, 9) + 3156 * pow(max_dist, 8)*min_dist + 22272 * pow(max_dist, 7)*pow(min_dist, 2) + 73332 * pow(max_dist, 6)*pow(min_dist, 3) + 124852 * pow(max_dist, 5)*pow(min_dist, 4) + 113932 * pow(max_dist, 4)*pow(min_dist, 5) + 55328 * pow(max_dist, 3)*pow(min_dist, 6) + 13572 * pow(max_dist, 2)*pow(min_dist, 7) + 1482 * max_dist*pow(min_dist, 8) + 52 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][5] = (209 * sqrt(739.5)*pow(min_dist, 2)*(4569 * pow(max_dist, 8) + 69792 * pow(max_dist, 7)*min_dist + 375900 * pow(max_dist, 6)*pow(min_dist, 2) + 939400 * pow(max_dist, 5)*pow(min_dist, 3) + 1193920 * pow(max_dist, 4)*pow(min_dist, 4) + 787696 * pow(max_dist, 3)*pow(min_dist, 5) + 260988 * pow(max_dist, 2)*pow(min_dist, 6) + 39000 * max_dist*pow(min_dist, 7) + 1950 * pow(min_dist, 8))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][6] = (-1463 * sqrt(739.5)*pow(min_dist, 2)*(633 * pow(max_dist, 7) + 7383 * pow(max_dist, 6)*min_dist + 30200 * pow(max_dist, 5)*pow(min_dist, 2) + 56385 * pow(max_dist, 4)*pow(min_dist, 3) + 51870 * pow(max_dist, 3)*pow(min_dist, 4) + 23387 * pow(max_dist, 2)*pow(min_dist, 5) + 4732 * max_dist*pow(min_dist, 6) + 325 * pow(min_dist, 7))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][7] = (209 * sqrt(739.5)*pow(min_dist, 2)*(11814 * pow(max_dist, 6) + 104698 * pow(max_dist, 5)*min_dist + 320155 * pow(max_dist, 4)*pow(min_dist, 2) + 433020 * pow(max_dist, 3)*pow(min_dist, 3) + 272545 * pow(max_dist, 2)*pow(min_dist, 4) + 75166 * max_dist*pow(min_dist, 5) + 7007 * pow(min_dist, 6))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][8] = (-4807 * sqrt(739.5)*pow(min_dist, 2)*(3817 * pow(max_dist, 5) + 25300 * pow(max_dist, 4)*min_dist + 56080 * pow(max_dist, 3)*pow(min_dist, 2) + 51940 * pow(max_dist, 2)*pow(min_dist, 3) + 20020 * max_dist*pow(min_dist, 4) + 2548 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][9] = (24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(2959 * pow(max_dist, 4) + 14224 * pow(max_dist, 3)*min_dist + 21604 * pow(max_dist, 2)*pow(min_dist, 2) + 12264 * max_dist*pow(min_dist, 3) + 2184 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][10] = (-24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(1303 * pow(max_dist, 3) + 4294 * pow(max_dist, 2)*min_dist + 4000 * max_dist*pow(min_dist, 2) + 1050 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][11] = (28405 * sqrt(82.16666666666667)*pow(min_dist, 2)*(631 * pow(max_dist, 2) + 1276 * max_dist*min_dist + 550 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][12] = (-85215 * sqrt(739.5)*pow(min_dist, 2)*(47 * max_dist + 44 * min_dist)) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][13] = (596505 * sqrt(739.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 13));
  std::vector<double> out(12*14);for(int i=0;i<12;++i)for(int j=0;j<14;++j)out[i*14+j]=rb_coeffs[i][j];return out;
}

}} // LAMMPS_NS::SUS2SL
#endif
