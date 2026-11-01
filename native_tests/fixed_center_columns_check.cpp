// Explicit fixed-center columns must reconstruct physical EFS for every shared
// head; default raw TrainLinear columns must remain unchanged by this request.
#include "dev_src/mtpr.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
double LinearEquationCurrentCoeff(const MLMTPR& mtpr, int column)
{
	if (column >= mtpr.JointReadoutLinearColumnOffset()
	    && column < mtpr.JointReadoutLinearColumnOffset() + mtpr.JointReadoutCoeffCount())
		return mtpr.regression_coeffs[mtpr.JointReadoutCoeffOffset() + column - mtpr.JointReadoutLinearColumnOffset()];
	const int linear_count = mtpr.LinearCoeffCount();
	const int linear_begin = mtpr.LinearCoeffOffset();
	if (column < linear_count) {
		if (column < mtpr.species_count)
			return mtpr.regression_coeffs[column]
			     + mtpr.regression_coeffs[linear_begin + column];
		const int scalar_index = column - mtpr.species_count;
		double value = mtpr.regression_coeffs[linear_begin + column];
		const int q = mtpr.UsesSpeciesReadout()
			? scalar_index % mtpr.alpha_scalar_moments : scalar_index;
		if (q >= 0 && q < static_cast<int>(mtpr.linear_mults.size()))
			value *= mtpr.linear_mults[q];
		return value;
	}
	const int e0_index = column - linear_count;
	const int e0_begin = mtpr.TwoLayerResidualE0CoeffOffset();
	if (e0_index >= 0 && e0_index < mtpr.TwoLayerResidualE0CoeffCount()) {
		double value = mtpr.regression_coeffs[e0_begin + e0_index];
		if (e0_index < static_cast<int>(mtpr.linear_mults.size()))
			value *= mtpr.linear_mults[e0_index];
		return value;
	}
	const int readout_index =
		column - mtpr.TwoLayerGateReadoutLinearColumnOffset();
	if (readout_index >= 0
	    && readout_index < mtpr.TwoLayerGateReadoutCoeffCount())
		return mtpr.TwoLayerGateReadoutCoeff(readout_index);
	ERROR("MTPR linear equation column is out of range");
	return 0.0;
}

int main(int argc,char** argv) {
 if(argc!=3)return 2;MLMTPR model(argv[1]);Configuration cfg;
 std::ifstream f(argv[2]);if(!cfg.Load(f))return 3;
 const int count=model.LinearEquationCount();double worst=0;
 for(double center:{0.001,0.7,0.,-0.4}) {
  for(int t=0;t<model.species_count;++t) {
   model.Coeff()[t]=-.2+t*.13;
   model.Coeff()[model.LinearCoeffOffset()+t]=(t+1)*center;
  }
  model.scaling=.4;
  for(int q=0;q<model.alpha_scalar_moments;++q)model.linear_mults[q]=(q%2?-1.:1.)*(.6+.01*q);
  for(int q=model.species_count;q<model.LinearCoeffCount();++q)model.Coeff()[model.LinearCoeffOffset()+q]=.03*std::sin(q+.4);
  model.LinCoeff();Configuration direct=cfg;model.CalcEFS(direct);
  Neighborhoods nbh(cfg,model.CutOff());model.CalcEFSComponents(cfg,nbh,true,true,false);
  auto snapshot=[&]() {std::vector<double> v(model.energy_cmpnts,model.energy_cmpnts+count);
   for(int i=0;i<cfg.size();++i)for(int c=0;c<count;++c)for(int a=0;a<3;++a)v.push_back(model.forces_cmpnts(i,c,a));
   for(int c=0;c<count;++c)for(int a=0;a<3;++a)for(int b=0;b<3;++b)v.push_back(model.stress_cmpnts[c][a][b]);return v;};
  const auto raw=snapshot();
  model.CalcEFSComponents(cfg,nbh,true,true,!model.UsesSpeciesReadout());
  double energy=0;std::vector<double> forces(3*cfg.size(),0),virial(9,0);
  for(int c=0;c<count;++c) {
   const double coeff=LinearEquationCurrentCoeff(model,c);energy+=coeff*model.energy_cmpnts[c];
   for(int i=0;i<cfg.size();++i)for(int a=0;a<3;++a)forces[3*i+a]+=coeff*model.forces_cmpnts(i,c,a);
   for(int a=0;a<3;++a)for(int b=0;b<3;++b)virial[3*a+b]+=coeff*model.stress_cmpnts[c][a][b];
  }
  worst=std::max(worst,std::abs(energy-direct.energy));
  for(int i=0;i<cfg.size();++i)for(int a=0;a<3;++a)worst=std::max(worst,std::abs(forces[3*i+a]-direct.force(i,a)));
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)worst=std::max(worst,std::abs(virial[3*a+b]-direct.stresses[a][b]));
  model.CalcEFSComponents(cfg,nbh,true,true,false);
  if(raw!=snapshot())return 5;
 }
 std::cout<<"fixed_center_columns_check max_abs="<<worst<<" species="<<model.UsesSpeciesReadout()<<std::endl;
 return worst<1e-9?0:1;
}
