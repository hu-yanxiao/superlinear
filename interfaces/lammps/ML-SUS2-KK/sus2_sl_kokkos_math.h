#ifndef LMP_SUS2_SL_KOKKOS_MATH_H
#define LMP_SUS2_SL_KOKKOS_MATH_H
#include <Kokkos_Core.hpp>
#include <cmath>

namespace LAMMPS_NS { namespace SUS2SL { namespace kokkos_detail {
KOKKOS_INLINE_FUNCTION int harmonic(int l,int m) { return l*l+l+m; }
KOKKOS_INLINE_FUNCTION int ranked(int rank,int H,int l,int a,int m) {
  return rank*(H-(l+1)*(l+1))+(rank-1-a)*(2*l+1)+m+l;
}

// Regular solid-harmonic recurrence, in the native real SH convention. The
// only fixed bound is the native model's mathematical Lmax <= 4 contract.
template<bool Derivatives,class Norm>
KOKKOS_INLINE_FUNCTION void harmonics(double x,double y,double z,int L,
                                     const Norm& norm,double* value,double* deriv) {
  double re[25]={},im[25]={},dre[75]={},dim[75]={}; re[0]=1.;
  const double xyz[3]={x,y,z},r2=x*x+y*y+z*z,r=sqrt(r2);
  for(int m=1;m<=L;++m) {
    const int i=harmonic(m,m),p=harmonic(m-1,m-1);const double c=-(2*m-1);
    re[i]=c*(x*re[p]-y*im[p]);im[i]=c*(x*im[p]+y*re[p]);
    if(Derivatives) for(int a=0;a<3;++a) {
      dre[3*i+a]=c*((a==0?re[p]:0.)-(a==1?im[p]:0.)+x*dre[3*p+a]-y*dim[3*p+a]);
      dim[3*i+a]=c*((a==0?im[p]:0.)+(a==1?re[p]:0.)+x*dim[3*p+a]+y*dre[3*p+a]);
    }
  }
  for(int m=0;m<=L;++m) {
    if(m<L) {
      const int i=harmonic(m+1,m),p=harmonic(m,m);const double c=2*m+1;
      re[i]=c*z*re[p];im[i]=c*z*im[p];
      if(Derivatives)for(int a=0;a<3;++a) {
        dre[3*i+a]=c*((a==2?re[p]:0.)+z*dre[3*p+a]);
        dim[3*i+a]=c*((a==2?im[p]:0.)+z*dim[3*p+a]);
      }
    }
    for(int l=m+2;l<=L;++l) {
      const int i=harmonic(l,m),p=harmonic(l-1,m),q=harmonic(l-2,m);
      const double A=2*l-1,B=l+m-1,D=l-m;
      re[i]=(A*z*re[p]-B*r2*re[q])/D;im[i]=(A*z*im[p]-B*r2*im[q])/D;
      if(Derivatives)for(int a=0;a<3;++a) {
        dre[3*i+a]=(A*((a==2?re[p]:0.)+z*dre[3*p+a])-B*(2*xyz[a]*re[q]+r2*dre[3*q+a]))/D;
        dim[3*i+a]=(A*((a==2?im[p]:0.)+z*dim[3*p+a])-B*(2*xyz[a]*im[q]+r2*dim[3*q+a]))/D;
      }
    }
  }
  double invpow=1.;
  for(int l=0;l<=L;++l,invpow/=r)for(int m=0;m<=l;++m) {
    const int i=harmonic(l,m),j=harmonic(l,-m);const double c=norm(i)*invpow;
    value[i]=c*re[i];if(m)value[j]=c*im[i];
    if(Derivatives)for(int a=0;a<3;++a) {
      deriv[3*i+a]=c*(dre[3*i+a]-l*re[i]*xyz[a]/r2);
      if(m)deriv[3*j+a]=c*(dim[3*i+a]-l*im[i]*xyz[a]/r2);
    }
  }
}
}}}
#endif
