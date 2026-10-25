# Qualification-only Kokkos role launch evidence

This observer is not used by throughput runs. It makes no CUDA, MPI or fence
calls. It explicitly implements `kokkosp_request_tool_settings` and writes
`requires_global_fencing=false`; Kokkos otherwise defaults that setting to true.
Every rank must record actual interface 20240906 init, settings version 1 with
the actual false value, matched begin/end callbacks, and finalize. No record or
no relevant label is incomplete evidence, not an inferred PASS.

The observer records the existing runtime kernel labels, not a guessed NVCC
demangling. Kokkos 4.7.1 encodes the execution-space type in the top eight bits
of the 32-bit device ID: Cuda is 2. Actual role labels with type 2, same-launch
Pair CUDA/UUID evidence and completed native E/F/site/global-virial comparisons
together establish CUDA role dispatch. A begin callback alone is not completion
or mathematical correctness. No-tools paired EFS controls remain mandatory.

Primary local source inspected, from the frozen LAMMPS/Kokkos 4.7.1 source at
`.codex_tmp/superlinear_kokkos_20261024/kokkos-reference/lib/kokkos/core/src/impl/`:

- `Kokkos_Profiling_C_Interface.h`: SHA256
  `5c7276a527609c2d62292fe6a602addb9e945f84ab828244a4e30c287235277d`;
  interface version, init/begin/end signatures and ToolSettings bool+padding ABI.
- `Kokkos_Profiling_Interface.hpp`: SHA256
  `a606c6789f9ceae0a860501dc73a643185bfb4857f32930b23648b442ace73c2`;
  type/device/instance widths 8/7/17 and Cuda enum value 2.
- `Kokkos_Profiling.cpp`: SHA256
  `0669dde5397f7ffb4aeeace7d458670c12a031959d8cfb702c3005ad7135f51b`;
  dynamic callback lookup and init-before-settings ordering, default fencing.

The ABI subset is isolated in `kokkos_role_trace_abi.h`. The shared observer is
compiled only inside the dedicated focused GPU allocation; source/header/library
SHA256, compiler command, compiler output and return status belong to the attempt.
No candidate rebuild or math/source modification is involved.
