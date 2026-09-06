#ifndef MLIP_PURE_W_BASIS_KERNELS_H
#define MLIP_PURE_W_BASIS_KERNELS_H
#include <cstddef>

namespace PureWBasisKernels {
// One center's contiguous edge slices. All k/b/m dimensions are retained.
// Geometry/radial buffers are read-only; force scratch is only 4*neighbors.
struct Block {
    int count=0, L=0, K=0, R=0;
    size_t radial_stride=0;
    const int* mu_to_eval=nullptr;
    const int* neighbors=nullptr;
    const double* radial=nullptr;
    const double* radial_dr=nullptr;
    const double* sh=nullptr;
    const double* sh_dr=nullptr;
    const double* input=nullptr;
    const double* gate_factor=nullptr;
    double* gate_adjoint=nullptr;
};

// q/adjoint are one l block in the existing [k,b,m] layout. Dispatch angular
// width once per center/l, so short dynamic m loops leave the edge hot path.
void Aggregate(int l, const Block& block, double* q);
// Add radial,x,y,z to edge_force[4*j:]. Preserve ascending l calls, k order,
// and each gate-adjoint channel's edge order, including repeated neighbors.
void Force(int l, const Block& block, const double* adjoint, double* edge_force);
// Cached edge-first counterpart. W is [edge,l,o,k]; adjoint is [o,l,m].
// Contract K before rotating the radial derivative into Cartesian axes.
void EdgeForce(int outputs, const Block& block, const double* adjoint,
    const double* w, const double* dw, double* edge_force);
}
#endif
