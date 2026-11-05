#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "dev_src/mtpr.h"
static std::vector<double> efs(MLMTPR& m, const Configuration& in) {
 Configuration c=in;m.CalcEFS(c);std::vector<double> v(1+3*c.size()+9);v[0]=c.energy;
 for(int i=0;i<c.size();++i)for(int a=0;a<3;++a)v[1+3*i+a]=c.force(i,a);
 for(int a=0;a<3;++a)for(int b=0;b<3;++b)v[1+3*c.size()+3*a+b]=c.stresses[a][b];return v;
}
int main(int argc,char**argv) {
 if(argc!=3)return 2;MLMTPR m(argv[1]);Configuration cfg;std::ifstream f(argv[2]);if(!cfg.Load(f))return 3;
 if(!m.AdaptiveRadialSharedO() || !m.JointReadoutEnabled())return 4;
 std::vector<int> idx;const int outer=m.ScalarReadoutCoeffCount();
 for(int q=0;q<outer;++q)idx.push_back(m.LinearCoeffOffset()+m.species_count+q);
 for(int q=0;q<m.JointReadoutCoeffCount();++q)idx.push_back(m.JointReadoutCoeffOffset()+q);
 const int n=idx.size(),rows=1+3*cfg.size()+9;std::vector<double> base(n);
 for(int q=0;q<n;++q){base[q]=m.Coeff()[idx[q]];m.Coeff()[idx[q]]=0;}
 const auto constant=efs(m,cfg);std::vector<double>A(n*rows);
 for(int q=0;q<n;++q){m.Coeff()[idx[q]]=1;auto v=efs(m,cfg);for(int r=0;r<rows;++r)A[q*rows+r]=v[r]-constant[r];m.Coeff()[idx[q]]=0;}
 // Independently validate native backward against all physical linear columns.
 std::vector<double> ew(cfg.size(),.37);std::vector<Vector3>fw(cfg.size(),Vector3(0,0,0));Matrix3 sw(0,0,0,0,0,0,0,0,0);std::vector<double>w(rows,.37);
 for(int i=0;i<cfg.size();++i)for(int a=0;a<3;++a)w[1+3*i+a]=fw[i][a]=std::sin(3*i+a+.2);
 for(int a=0;a<3;++a)for(int b=0;b<3;++b)w[1+3*cfg.size()+3*a+b]=-(sw[a][b]=.2*std::cos(3*a+b+.3));
 Array1D grad(m.CoeffCount());std::fill(&grad[0],&grad[0]+m.CoeffCount(),0.0);m.AccumulateEFSCombinationGrad(cfg,ew,fw,sw,grad);double backward=0;
 for(int q=0;q<n;++q){double z=0;for(int r=0;r<rows;++r)z+=A[q*rows+r]*w[r];backward=std::max(backward,std::abs(z-grad[idx[q]]));}
 std::vector<double>target(rows),G(n*n),rhs(n);double cross=0,largest=0;
 for(int r=0;r<rows;++r)for(int q=0;q<n;++q)target[r]+=A[q*rows+r]*std::sin(.3+q);
 for(int q=0;q<n;++q){for(int r=0;r<rows;++r)rhs[q]+=A[q*rows+r]*target[r];for(int k=0;k<n;++k){for(int r=0;r<rows;++r)G[q*n+k]+=A[q*rows+r]*A[k*rows+r];if(q<outer&&k>=outer)cross=std::max(cross,std::abs(G[q*n+k]));}largest=std::max(largest,G[q*n+q]);}
 for(int q=0;q<n;++q)G[q*n+q]+=std::max(1e-20,largest*1e-12);
 for(int k=0;k<n;++k){int p=k;for(int i=k+1;i<n;++i)if(std::abs(G[i*n+k])>std::abs(G[p*n+k]))p=i;for(int j=k;j<n;++j)std::swap(G[k*n+j],G[p*n+j]);std::swap(rhs[k],rhs[p]);if(G[k*n+k]==0)return 5;for(int i=k+1;i<n;++i){double z=G[i*n+k]/G[k*n+k];for(int j=k;j<n;++j)G[i*n+j]-=z*G[k*n+j];rhs[i]-=z*rhs[k];}}
 std::vector<double>x(n);for(int i=n-1;i>=0;--i){double z=rhs[i];for(int j=i+1;j<n;++j)z-=G[i*n+j]*x[j];x[i]=z/G[i*n+i];m.Coeff()[idx[i]]=x[i];}
 auto fit=efs(m,cfg);double err=0,cnorm=0;for(int r=0;r<rows;++r){err=std::max(err,std::abs(fit[r]-constant[r]-target[r]));cnorm=std::max(cnorm,std::abs(constant[r]));}
 std::cout<<"outer_columns="<<outer<<" joint_columns="<<n-outer<<" cross_head_gram="<<cross<<" constant="<<cnorm<<" backward_error="<<backward<<" full_joint_fit_error="<<err<<std::endl;
 return cross>1e-14 && cnorm>1e-10 && backward<1e-8 && err<2e-7 ? 0:1;
}
