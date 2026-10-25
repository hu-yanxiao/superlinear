// Read-only numerical diagnostic, not a production evaluator.
#include "sus2_sl_kokkos_radial.h"
#include "sus2_sl_radial_math.h"
#include <cmath>
#include <iomanip>
#include <iostream>
using DR=LAMMPS_NS::SUS2SL::DeviceRadial<Kokkos::DefaultExecutionSpace::device_type>;
using CPU=LAMMPS_NS::SUS2SL::radial_math::RadialBasis_Shapeev;
// Explicit instantiation permits private-member access for this standalone
// diagnostic, without changing either production class's interface/storage.
template<class Tag,typename Tag::type Member>struct Access {
  friend typename Tag::type get(Tag){return Member;}
};
struct DeviceCoeffs {using type=std::vector<double>(*)(double,double);friend type get(DeviceCoeffs);};
struct CpuCoeffs {using type=double(CPU::*)[12][14];friend type get(CpuCoeffs);};
template struct Access<DeviceCoeffs,&DR::shape_coefficients>;
template struct Access<CpuCoeffs,&CPU::rb_coeffs>;
int main(){
  const auto coeff=get(DeviceCoeffs{})(.5,5.);CPU cpu(.5,5.,12);cpu.scaling=.73;
  const auto member=get(CpuCoeffs{});const double r=2.5;
  cpu.RB_Calc(r,2.4,.7,5);
  std::cout<<std::setprecision(17);
  for(int n=0;n<12;++n){double plain=0,fused=0,integer=0,plain_d=0,fused_d=0;
    for(int i=-2;i<=n;++i){const double c=coeff[n*14+i+2],a=.73*c;
      const double power=std::pow(r,i);volatile double term=a*power;
      plain+=term;fused=std::fma(a,power,fused);
      const double b=a*i,dpower=std::pow(r,i-1);volatile double dterm=b*dpower;
      plain_d+=dterm;fused_d=std::fma(b,dpower,fused_d);
      double ip=1;for(int j=0;j<std::abs(i);++j)ip*=r;if(i<0)ip=1/ip;
      integer=std::fma(a,ip,integer);
      std::cout<<"COEFF "<<n<<" "<<i<<" "<<std::hexfloat<<c<<" "<<(cpu.*member)[n][i+2]<<std::defaultfloat<<"\n";
    }
    std::cout<<"VALUE "<<n<<" "<<cpu.rb_vals[n]<<" "<<plain<<" "<<fused<<" "<<integer<<"\n";
    std::cout<<"DERIVATIVE "<<n<<" "<<cpu.rb_ders[n]<<" "<<plain_d<<" "<<fused_d<<"\n";
    if(plain!=cpu.rb_vals[n]||plain_d!=cpu.rb_ders[n])return 1;
  }
}
