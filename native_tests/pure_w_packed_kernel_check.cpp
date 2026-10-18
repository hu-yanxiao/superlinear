#include "dev_src/pure_w_execution.h"
#include <cmath>
#include <iostream>

int main()
{
    if (PureWExecution::PreferBasis(4,10,6400,96)
        || PureWExecution::PreferBasis(8,11,6400,96)
        || PureWExecution::PreferBasis(8,10,15,1)
        || PureWExecution::PreferBasis(8,10,0,0)
        || !PureWExecution::PreferBasis(8,10,16,1)) return 2;
    double worst=0.0;
    for (int O : {1,2,3,4,8,16}) for (int K : {1,2,8}) for (int R : {1,3,10}) {
        const int L=4, E=3;
        std::vector<int> map(K*L);
        for (int m=0;m<K*L;++m) map[m]=m%E; // deliberately shared scaling blocks
        std::vector<double> c(L*O*K*R), rb(E*R), dr(E*R*5), w(L*O*K), dw(w.size());
        for (size_t i=0;i<c.size();++i) c[i]=std::sin(0.31*i);
        for (size_t i=0;i<rb.size();++i) rb[i]=std::cos(0.73*i);
        for (size_t i=0;i<dr.size();++i) dr[i]=std::sin(0.13*i);
        PureWExecution ex;
        for (int update=0;update<3;++update) {
            for (size_t i=0;i<c.size();++i) c[i]+=0.01*std::cos(0.2*i+update);
            ex.Pack(L,O,K,R,c.data());
            for (bool derivatives : {false,true}) {
                ex.Kernels(map.data(),rb.data(),derivatives?dr.data():nullptr,
                           w.data(),derivatives?dw.data():nullptr);
                for (int l=0;l<L;++l) for (int o=0;o<O;++o) for (int k=0;k<K;++k) {
                    const int row=(l*O+o)*K+k, base=map[k*L+l]*R;
                    double v=0,d=0;
                    for (int b=0;b<R;++b) {v+=c[row*R+b]*rb[base+b]; d+=c[row*R+b]*dr[5*base+b];}
                    worst=std::max(worst,std::abs(v-w[row]));
                    if (derivatives) worst=std::max(worst,std::abs(d-dw[row]));
                }
            }
        }
    }
    std::cout << "packed W/Wprime and in-place repacking max_abs=" << worst << std::endl;
    return worst<1e-12 ? 0 : 1;
}
