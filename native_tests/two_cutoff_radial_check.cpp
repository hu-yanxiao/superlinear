#include "../src/radial_basis.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <memory>
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(double x,double y,double tol,const char* m){check(std::abs(x-y)<=tol*(1+std::abs(x)+std::abs(y)),m);}
void qualify(AnyRadialBasis& b) {
 const int R=b.rb_size;const double r=2.1,sigma=.8,shift=.4,rc=3.5,h=1e-5;
 b.angular_max_dist=rc;
 for(const std::string type:{std::string("legacy"),std::string("polynomial")}) {
  b.SetRadialCutoff(type,5);
  b.RB_CalcForCutoff(r,sigma,shift,0,rc,2);auto v=b.rb_vals,d=b.rb_ders;
  std::unique_ptr<AnyRadialBasis> ref;
  if(b.GetRBTypeString()=="RBChebyshev_sss")ref.reset(new RadialBasis_Chebyshev_sss(0,rc,R));
  else ref.reset(new RadialBasis_Laguerre_log1p(0,rc,R));
  ref->SetRadialCutoff(type,5);ref->RB_Calc(r,sigma,shift,0);
  for(int n=0;n<R;++n)close(v[n],ref->rb_vals[n],1e-13,"short envelope reference");
  for(int n=0;n<5*R;++n)close(d[n],ref->rb_ders[n],1e-13,"short derivative reference");

  b.RB_CalcForCutoff(r+h,sigma,shift,0,rc,2);auto vp=b.rb_vals;
  b.RB_CalcForCutoff(r-h,sigma,shift,0,rc,2);auto vm=b.rb_vals;
  for(int n=0;n<R;++n) close(d[n],(vp[n]-vm[n])/(2*h),1e-7,"coordinate derivative");
  for(int parameter=0;parameter<2;++parameter) {
   b.RB_CalcForCutoff(r,sigma+(parameter==0?h:0),shift+(parameter==1?h:0),0,rc,2);auto p=b.rb_vals,pd=b.rb_ders;
   b.RB_CalcForCutoff(r,sigma-(parameter==0?h:0),shift-(parameter==1?h:0),0,rc,2);auto m=b.rb_vals,md=b.rb_ders;
   for(int n=0;n<R;++n) {
    close(d[(1+2*parameter)*R+n],(p[n]-m[n])/(2*h),1e-7,"parameter derivative");
    close(d[(2+2*parameter)*R+n],(pd[n]-md[n])/(2*h),1e-7,"mixed derivative");
   }
  }
  // Alternate envelope groups over identical raw jets; no stale-enveloped reuse.
  b.RB_CalcForCutoff(r,sigma,shift,0,b.max_dist,2);auto longv=b.rb_vals;
  b.RB_CalcForCutoff(r,sigma,shift,0,rc,2);
  for(int n=0;n<R;++n) close(v[n],b.rb_vals[n],1e-14,"group reuse");
  for(double x:{rc,rc+h,b.max_dist}) {
   b.RB_CalcForCutoff(x,sigma,shift,0,rc,2);
   for(double y:b.rb_vals)check(y==0&&!std::signbit(y),"cutoff value not positive zero");
   for(double y:b.rb_ders)check(y==0&&!std::signbit(y),"cutoff derivative not positive zero");
  }
  b.RB_CalcForCutoff(r,sigma,shift,0,rc,0);for(int n=0;n<R;++n)close(v[n],b.rb_vals[n],1e-14,"values mode");
  b.RB_CalcForCutoff(r,sigma,shift,0,rc,1);for(int n=0;n<R;++n)close(d[n],b.rb_ders[n],1e-14,"coordinate mode");
  b.angular_max_dist=b.max_dist;b.RB_Calc(r,sigma,shift,0);v=b.rb_vals;d=b.rb_ders;
  b.RB_CalcForCutoff(r,sigma,shift,0,b.max_dist,2);
  check(v==b.rb_vals&&d==b.rb_ders,"equal cutoff changed baseline");b.angular_max_dist=rc;
 }
}
int main(){try{RadialBasis_Chebyshev_sss a(0,6.5,8);RadialBasis_Laguerre_log1p b(0,6.5,8);qualify(a);qualify(b);std::cout<<"two cutoff radial jets PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
