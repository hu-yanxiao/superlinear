#include "pure_w_basis_kernels.h"
#include "../src/mlip.h"

namespace PureWBasisKernels {
namespace {
template<int M>
void AggregateFixed(const Block& b, double* q) {
    const int l=(M-1)/2, H=b.L*b.L, channels=b.K*b.L;
    for (int j=0;j<b.count;++j) {
        const double* rb=b.radial+static_cast<size_t>(j)*b.radial_stride;
        const double* y=b.sh+static_cast<size_t>(j)*H+l*l;
        const double* input=b.input+static_cast<size_t>(j)*channels;
        for (int k=0;k<b.K;++k) {
            const int mu=k*b.L+l;
            const double* phi=rb+b.mu_to_eval[mu]*b.R;
            for (int r=0;r<b.R;++r) {
                const double v=input[mu]*phi[r];
                double* row=q+(k*b.R+r)*M;
                for (int m=0;m<M;++m) row[m]+=v*y[m];
            }
        }
    }
}

template<int M>
void ForceFixed(const Block& b, const double* adj, double* edge_force) {
    const int l=(M-1)/2, H=b.L*b.L, channels=b.K*b.L;
    for (int j=0;j<b.count;++j) {
        const double* rb=b.radial+static_cast<size_t>(j)*b.radial_stride;
        const double* dr=b.radial_dr+static_cast<size_t>(j)*b.radial_stride;
        const double* y=b.sh+static_cast<size_t>(j)*H+l*l;
        const double* dy=b.sh_dr+static_cast<size_t>(j)*3*H+3*l*l;
        const double* input=b.input+static_cast<size_t>(j)*channels;
        const double* factor=b.gate_factor+static_cast<size_t>(j)*channels;
        double* gate_adj=b.gate_adjoint==nullptr ? nullptr
            : b.gate_adjoint+static_cast<size_t>(b.neighbors[j])*channels;
        double radial_force=edge_force[4*j];
        double angular_force[3]={edge_force[4*j+1],edge_force[4*j+2],edge_force[4*j+3]};
        for (int k=0;k<b.K;++k) {
            const int mu=k*b.L+l, base=b.mu_to_eval[mu]*b.R;
            // Fixed-width register accumulators, no reductions across b are
            // reassociated. Both radial and gate chains use the same bar-Q.
            double phi_adj[M]={}, radial_adj[M]={};
            for (int r=0;r<b.R;++r) {
                const double* row=adj+(k*b.R+r)*M;
                for (int m=0;m<M;++m) {
                    phi_adj[m]+=rb[base+r]*row[m];
                    radial_adj[m]+=dr[base+r]*row[m];
                }
            }
            double input_adj=0.0, phi_der=0.0, angular[3]={};
            for (int m=0;m<M;++m) {
                input_adj+=phi_adj[m]*y[m];
                phi_der+=radial_adj[m]*y[m];
                for (int a=0;a<3;++a) angular[a]+=phi_adj[m]*dy[3*m+a];
            }
            radial_force+=input[mu]*phi_der;
            for (int a=0;a<3;++a) angular_force[a]+=input[mu]*angular[a];
            if (gate_adj!=nullptr) gate_adj[mu]+=input_adj*factor[mu];
        }
        edge_force[4*j]=radial_force;
        for (int a=0;a<3;++a) edge_force[4*j+1+a]=angular_force[a];
    }
}
}

void Aggregate(int l, const Block& block, double* q) {
    switch (l) {
    case 0: AggregateFixed<1>(block,q); break;
    case 1: AggregateFixed<3>(block,q); break;
    case 2: AggregateFixed<5>(block,q); break;
    case 3: AggregateFixed<7>(block,q); break;
    case 4: AggregateFixed<9>(block,q); break;
    case 5: AggregateFixed<11>(block,q); break;
    case 6: AggregateFixed<13>(block,q); break;
    default: ERROR("Pure-W basis angular width is out of range");
    }
}

void Force(int l, const Block& block, const double* adjoint, double* edge_force) {
    switch (l) {
    case 0: ForceFixed<1>(block,adjoint,edge_force); break;
    case 1: ForceFixed<3>(block,adjoint,edge_force); break;
    case 2: ForceFixed<5>(block,adjoint,edge_force); break;
    case 3: ForceFixed<7>(block,adjoint,edge_force); break;
    case 4: ForceFixed<9>(block,adjoint,edge_force); break;
    case 5: ForceFixed<11>(block,adjoint,edge_force); break;
    case 6: ForceFixed<13>(block,adjoint,edge_force); break;
    default: ERROR("Pure-W basis angular width is out of range");
    }
}

void EdgeForce(int O, const Block& b, const double* adj, const double* w,
               const double* dw, double* edge_force) {
    const int H=b.L*b.L, channels=b.K*b.L;
    const size_t stride=static_cast<size_t>(b.L)*O*b.K;
    for (int j=0;j<b.count;++j) {
        const double* input=b.input+static_cast<size_t>(j)*channels;
        const double* factor=b.gate_factor+static_cast<size_t>(j)*channels;
        const double* y=b.sh+static_cast<size_t>(j)*H;
        const double* dy=b.sh_dr+static_cast<size_t>(j)*3*H;
        double* gate_adj=b.gate_adjoint==nullptr ? nullptr
            : b.gate_adjoint+static_cast<size_t>(b.neighbors[j])*channels;
        double radial=0.0, angular[3]={};
        for (int l=0;l<b.L;++l) for (int o=0;o<O;++o) {
            const double* row=w+j*stride+(l*O+o)*b.K;
            const double* drow=dw+j*stride+(l*O+o)*b.K;
            double value=0.0, derivative=0.0;
            for (int k=0;k<b.K;++k) {
                value+=row[k]*input[k*b.L+l];
                derivative+=drow[k]*input[k*b.L+l];
            }
            double adj_y=0.0, adj_dy[3]={};
            for (int m=0;m<2*l+1;++m) {
                const int h=l*l+m;
                const double a=adj[o*H+h];
                adj_y+=a*y[h];
                for (int axis=0;axis<3;++axis) adj_dy[axis]+=a*dy[3*h+axis];
            }
            radial+=derivative*adj_y;
            for (int axis=0;axis<3;++axis) angular[axis]+=value*adj_dy[axis];
            if (gate_adj!=nullptr) for (int k=0;k<b.K;++k) {
                const int mu=k*b.L+l;
                gate_adj[mu]+=adj_y*row[k]*factor[mu];
            }
        }
        edge_force[4*j]+=radial;
        for (int a=0;a<3;++a) edge_force[4*j+1+a]+=angular[a];
    }
}
}
