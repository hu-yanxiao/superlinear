#include "dev_src/pure_w_basis_kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static void fill(std::vector<double>& a,double seed) {
    for (size_t i=0;i<a.size();++i) a[i]=std::sin(seed+0.137*i);
}
int main() {
    double worst=0.0;
    for (int L=1;L<=7;++L) for (int K : {1,2,8}) for (int R : {2,10,17})
    for (int J : {0,1,5,31}) for (bool gated : {false,true}) {
        const int H=L*L, mu_count=K*L, evals=K+1, stride=evals*R;
        std::vector<int> map(mu_count),neighbors(J);
        for (int mu=0;mu<mu_count;++mu) map[mu]=(mu*3+1)%evals;
        for (int j=0;j<J;++j) neighbors[j]=j%3; // repeated periodic images
        std::vector<double> rb(J*stride),dr(rb.size()),sh(J*H),dsh(3*sh.size());
        std::vector<double> input(J*mu_count),factor(input.size()),adj(K*R*H);
        std::vector<double> q(adj.size()),force(4*J),gate(3*mu_count);
        fill(rb,.1);fill(dr,.2);fill(sh,.3);fill(dsh,.4);fill(input,.5);
        fill(factor,.6);fill(adj,.7);fill(q,.8);fill(force,.9);fill(gate,1.1);
        auto qr=q, fr=force, gr=gate;
        // Original edge/l/k/b/m ordering. Nonzero outputs test += semantics.
        for (int j=0;j<J;++j) for (int l=0;l<L;++l) {
            const int M=2*l+1;
            for (int k=0;k<K;++k) {
                const int mu=k*L+l, base=j*stride+map[mu]*R;
                double pa[13]={},ra[13]={};
                for (int r=0;r<R;++r) {
                    const double v=input[j*mu_count+mu]*rb[base+r];
                    for (int m=0;m<M;++m) {
                        const int index=K*R*l*l+(k*R+r)*M+m;
                        qr[index]+=v*sh[j*H+l*l+m];
                        pa[m]+=rb[base+r]*adj[index];
                        ra[m]+=dr[base+r]*adj[index];
                    }
                }
                double ia=0,pd=0,ang[3]={};
                for (int m=0;m<M;++m) {
                    ia+=pa[m]*sh[j*H+l*l+m];
                    pd+=ra[m]*sh[j*H+l*l+m];
                    for (int a=0;a<3;++a) ang[a]+=pa[m]*dsh[3*(j*H+l*l+m)+a];
                }
                fr[4*j]+=input[j*mu_count+mu]*pd;
                for (int a=0;a<3;++a) fr[4*j+1+a]+=input[j*mu_count+mu]*ang[a];
                if (gated) gr[neighbors[j]*mu_count+mu]+=ia*factor[j*mu_count+mu];
            }
        }
        PureWBasisKernels::Block b;
        b.count=J;b.L=L;b.K=K;b.R=R;b.radial_stride=stride;
        b.mu_to_eval=map.data();b.neighbors=neighbors.data();
        b.radial=rb.data();b.radial_dr=dr.data();b.sh=sh.data();b.sh_dr=dsh.data();
        b.input=input.data();b.gate_factor=factor.data();b.gate_adjoint=gated?gate.data():nullptr;
        for (int l=0;l<L;++l) {
            PureWBasisKernels::Aggregate(l,b,q.data()+K*R*l*l);
            PureWBasisKernels::Force(l,b,adj.data()+K*R*l*l,force.data());
        }
        auto compare=[&](const std::vector<double>& x,const std::vector<double>& y) {
            for (size_t i=0;i<x.size();++i) {
                const double e=std::abs(x[i]-y[i])/std::max(1.0,std::max(std::abs(x[i]),std::abs(y[i])));
                if (!std::isfinite(e)) return false;
                worst=std::max(worst,e);
                if (e>2e-12) return false;
            }
            return true;
        };
        if (!compare(q,qr)||!compare(force,fr)||!compare(gate,gr)) {
            std::cerr<<"FAIL L="<<L<<" K="<<K<<" R="<<R<<" J="<<J<<" error="<<worst<<'\n';
            return 1;
        }
    }
    std::cout<<"basis fixed-width aggregation/force/gate PASS max_scaled="<<worst<<'\n';
    return 0;
}
