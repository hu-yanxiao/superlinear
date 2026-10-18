# ML-SUS2 LAMMPS Install Standard

> Historical paths, branch labels, and evidence identifiers in this document are anonymized aliases, not current installation locations. Use the installation entry points in the repository README.

## Scope

This note records the current standard for maintaining the `ML-SUS2` LAMMPS interface on the phy-weigw server.

Use `ML-SUS2` naming as the active convention. Do not treat `ML-SUS2-MTP` as the active package name in new work, except when cleaning up historical build residue.

## Canonical Remote Paths

- Shared source tree:

```bash
/work/phy-weigw/apps/lammps-10Dec2025
```

- CPU install location:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

- GPU install location:

```bash
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

- Interface tarball reference:

```bash
<historical-path>
```

## CPU Standard

CPU LAMMPS uses the old `make` workflow.

Requirements:

1. Use the Intel compiler toolchain.
2. Build CPU-specific installed binaries for the main CPU instruction-set families instead of relying on `-xHost`.
3. Keep the canonical installed executable path as an auto-dispatch wrapper:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

Current CPU installed binaries:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi_avx2
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi_avx512_skx
```

Current CPU `make` targets:

```bash
make -j8 ml_sus2_avx2
make -j8 ml_sus2_avx512_skx
```

The wrapper accepts an override for A/B testing:

```bash
SUS2_LAMMPS_CPU_TARGET=auto
SUS2_LAMMPS_CPU_TARGET=avx2
SUS2_LAMMPS_CPU_TARGET=avx512_skx
```

CPU target guidance:

- Use `avx2` for Intel E5, AMD EPYC 7352, and unknown nodes.
- Use `avx2` for Xeon Gold 6133/6138 and Platinum 8175M-class Skylake-SP nodes; SUS2-SL benchmarked faster with AVX2 than AVX512 on Gold 6133 because AVX512 downclock outweighed vector-width gains.
- Use `avx512_skx` only for newer AVX512 nodes after direct validation, such as Ice Lake 8375C-class nodes.
- Consider a future Sapphire Rapids target only after direct SUS2 benchmarking on 8488C, 8457C, or 8581C nodes shows a benefit.

Operational rule:

- CPU `make` modifies the shared source tree directly. Assume it leaves residue that must be cleaned before any later GPU CMake rebuild.

## GPU Standard

GPU LAMMPS uses CMake from the same shared source tree.

Use the cluster CMake module for GPU CMake work:

```bash
module load cmake/3.25.2
```

Requirements:

1. Use the same source tree:

```bash
/work/phy-weigw/apps/lammps-10Dec2025
```

2. Before configuring GPU CMake, first clear the residue left by the CPU `make` route:

```bash
make -C /work/phy-weigw/apps/lammps-10Dec2025/src no-all purge
```

3. Also remove stale root-level SUS2 interface files from `src/` if they exist:

```bash
pair_sus2_mtp.cpp
pair_sus2_mtp.h
sus2_mtp_*.cpp
sus2_mtp_*.h
```

4. Use a clean CMake build directory instead of trying to reuse a polluted one.
5. Submit the GPU build to a GPU node through LSF instead of relying on the login node for the full rebuild.
6. GPU build scripts should load `cmake/3.25.2` explicitly before running `cmake --build` or configuring a build directory.

Preferred GPU compiler/MPI route:

Use NVHPC's CUDA-aware OpenMPI, but use GCC as the host C/C++ compiler and NVCC for CUDA code:

```bash
module purge
module load cuda/12.4 nvhpc/22.11 gcc/11.2.0 cmake/3.25.2
export OMPI_CC=gcc
export OMPI_CXX=g++
```

Configure with:

```bash
-D CMAKE_C_COMPILER=$(command -v gcc)
-D CMAKE_CXX_COMPILER=$(command -v mpicxx)
-D CMAKE_CUDA_COMPILER=$(command -v nvcc)
-D CMAKE_CUDA_HOST_COMPILER=$(command -v g++)
-D CMAKE_CXX_FLAGS_RELEASE="-O3 -march=x86-64 -mtune=generic -DNDEBUG"
-D CMAKE_C_FLAGS_RELEASE="-O3 -march=x86-64 -mtune=generic -DNDEBUG"
-D Kokkos_ARCH_AMPERE80=on
```

Rationale:

- `nvhpc/22.11` provides an OpenMPI build with CUDA support enabled.
- `OMPI_CXX=g++` keeps the CUDA-aware MPI wrapper while avoiding NVHPC host-code `-fast`/`-tp=host` cross-node illegal-instruction failures.
- `-march=x86-64 -mtune=generic` makes the host side portable across the GPU nodes.
- `Kokkos_ARCH_AMPERE80=on` targets the A100 GPUs directly and avoids the old compute-capability 7.0 on 8.0 warning.

Runtime convention for CUDA-aware testing:

```bash
module purge
module load cuda/12.4 nvhpc/22.11 gcc/11.2.0
export OMPI_CC=gcc
export OMPI_CXX=g++
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_btl=self,vader,smcuda,tcp
export OMPI_MCA_btl_smcuda_use_cuda_ipc=1
mpirun --bind-to none -np ${NGPU} \
  ./run_rankenv.sh ${LMP_BIN} -k on g ${NGPU} -sf kk \
  -pk kokkos neigh half newton on comm device gpu/aware on -in in.gpu
```

Validated experimental binary, kept outside the canonical install path until explicitly promoted:

```bash
<historical-path>
```

Validation on 2026-04-26 for 995,328 atoms, 2,000 NPT steps, SUS2-SL 1.1 model:

```text
1 A100: loop 952.553 s, 2.090 Matom-step/s
2 A100: loop 508.424 s, 3.915 Matom-step/s
3 A100: loop 352.852 s, 5.642 Matom-step/s
4 A100: loop 263.094 s, 7.566 Matom-step/s
```

The 4 GPU CUDA-aware route reduced average communication time from 12.776 s to 3.376 s versus the previous Intel-MPI canonical binary, while final thermodynamic output matched exactly at step 2000.

Installed result:

```bash
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

## Recommended Rebuild Order

1. Sync the interface source updates into the shared LAMMPS source tree.
2. Rebuild the CPU binary with `make` if the CPU interface changed.
3. Before any GPU rebuild, purge the CPU `make` residue from the shared tree.
4. Reconfigure GPU CMake in a clean build directory.
5. Submit the GPU compile on a GPU queue.
6. Replace the canonical installed GPU binary in place.
7. Run smoke tests from:

```bash
/work/phy-weigw/hyx/ma/l3k3/jacobi/lmp
```

## Operational Intent

The purpose of this standard is to keep future interface edits smooth:

- one shared source tree
- one canonical CPU install path
- one canonical GPU install path
- explicit purge between `make` and CMake
- GPU rebuilds performed on GPU nodes

Avoid ad hoc binary copies and mixed build states unless the user explicitly asks for a temporary experiment.

## SUS2-SH Interface Note

For SUS2-SH LAMMPS work, start from the developer-branch interface archive rather than older local interface copies:

```bash
<historical-path>
```

The current remote SUS2-SH LAMMPS worktree is:

```bash
<historical-path>
```

On 2026-05-15, the Kokkos `_lmp` radial-table path was changed to store value and derivative in one packed table entry:

```cpp
struct SUS2MTPKokkosRadialTableEntry {
  double value;
  double deriv;
};
```

This removes the separate force-path loads from `d_radial_list` and `d_radial_der_list`. Keep the packed entry type at namespace scope, not as a private nested class type, because CUDA/Kokkos uses the view value type in global-kernel template instantiations.

Validation on the 10,240-atom Cu-Zr SH case, 2,000 steps, one A100, `RBChebyshev_sss_lmp`:

```text
direct radial GPU: loop 29.7055 s, pair 29.074 s
packed _lmp GPU:  loop 29.0039 s, pair 28.371 s
old split _lmp GPU reference: loop 40.0261 s
```

The final step-2000 thermodynamic line matched the direct-radial run to printed precision apart from last-digit pressure noise.

The same source line also has a rebuilt CPU AVX2 binary:

```bash
<historical-path>
```

Build settings:

```text
make target: ml_sus2_avx2
core flags: -xCORE-AVX2 -O3 -ipo
```

Validation on the 10,240-atom Cu-Zr SH 3322 case, 2,000 steps, 40 MPI ranks on Xeon Gold 6133:

```text
model: sh_body_l_max = {3, 3, 2, 2}, alpha_moments_count = 525
direct radial CPU AVX2: loop 135.279 s, pair avg 129.68 s
_lmp radial CPU AVX2:  loop 41.4773 s, pair avg 39.188 s
```

The direct and `_lmp` final step-2000 thermodynamic lines matched to printed precision apart from last-digit pressure noise.
