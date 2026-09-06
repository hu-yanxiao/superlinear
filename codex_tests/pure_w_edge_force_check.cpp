#include "dev_src/pure_w_basis_kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
static void fill(std::vector<double>& a,double s) {
    for (size_t i=0;i<a.size();++i) a[i]=std::sin(.23*i+s);
}
int main() {
    double worst=0;
    for (int L : {1,4,7}) for (int K : {1,2,8}) for (int O : {1,2,3,4,8,16})
    for (int J : {0,1,17}) for (bool gated : {false,true}) {
        const int H=L*L, channels=L*K, stride=L*O*K;
        std::vector<double> w(J*stride),dw(w.size()),adj(O*H),y(J*H),dy(3*y.size());
        std::vector<double> x(J*channels),factor(x.size()),gate(3*channels),f(4*J);
        fill(w,.1);fill(dw,.2);fill(adj,.3);fill(y,.4);fill(dy,.5);
        fill(x,.6);fill(factor,.7);fill(gate,.8);
        const auto initial=gate;
        std::vector<int> neighbors(J);
        for (int j=0;j<J;++j) neighbors[j]=j%3;
        auto gr=gate, fr=f;
        // Direct derivative of sum_{edge,l,o,k,m} adj * W * x * Y.
        // Deliberately do not factor k/m sums as the optimized evaluator does.
        for (int j=0;j<J;++j) for (int l=0;l<L;++l) for (int o=0;o<O;++o)
        for (int k=0;k<K;++k) for (int m=0;m<2*l+1;++m) {
            const int mu=k*L+l,h=l*l+m,at=j*stride+(l*O+o)*K+k;
            const double a=adj[o*H+h],v=x[j*channels+mu];
            fr[4*j]+=a*dw[at]*v*y[j*H+h];
            for (int d=0;d<3;++d) fr[4*j+1+d]+=a*w[at]*v*dy[3*(j*H+h)+d];
            if (gated) gr[neighbors[j]*channels+mu]+=a*y[j*H+h]*w[at]*factor[j*channels+mu];
        }
        PureWBasisKernels::Block b;
        b.count=J;b.L=L;b.K=K;b.neighbors=neighbors.data();b.sh=y.data();b.sh_dr=dy.data();
        b.input=x.data();b.gate_factor=factor.data();b.gate_adjoint=gated?gate.data():nullptr;
        PureWBasisKernels::EdgeForce(O,b,adj.data(),w.data(),dw.data(),f.data());
        auto compare=[&](const std::vector<double>& a,const std::vector<double>& ref) {
            for (size_t i=0;i<a.size();++i) {
                const double e=std::abs(a[i]-ref[i])/std::max(1.,std::max(std::abs(a[i]),std::abs(ref[i])));
                if (!std::isfinite(e)) return false;
                worst=std::max(worst,e);
                if (e>3e-12) return false;
            }
            return true;
        };
        if (!compare(f,fr)||!compare(gate,gr)) {
            std::cerr<<"FAIL L="<<L<<" K="<<K<<" O="<<O<<" J="<<J<<" error="<<worst<<'\n';
            return 1;
        }
    }
    std::cout<<"edge force and global gate adjoints PASS max_scaled="<<worst<<'\n';
}
