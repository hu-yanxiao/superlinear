# Grouped outer cross-channel graph

`init-sh --outer-k-rank=O --outer-channel-block-size=b` writes version 1.9.0,
`outer_channel_mode = grouped-cross`, and `outer_channel_block_size = b`.
The positive block width must divide O. Blocks need not be powers of two.
Adaptive retains its existing O <= K radial-layout bound. Legacy dense mixers
allow O > K. The inner graph and dense K-to-O projection are unchanged.

The complete native primitive prefix retains its ordering. A q-total width-b
local graph is generated once. Groups are bound in ascending order; products,
nonbasic nodes and readout scalars are group-major. Each primitive maps local
channel k to global channel g*b+k. The reader validates dependency provenance,
uniform template identity, group-major scalar mappings and metadata uniqueness.
Old/default models retain their existing format and graph ordering.

Forward and adjoint product-row execution reuse the first group's terms and
map their nodes over all groups. Derivative/design-column paths retain the
exact general graph route. No independent group least-squares solves are used:
the ordinary full design matrix and cross-group Gram terms remain intact.
The serialized file retains explicit products for compatibility with native
model tools; template reuse here concerns product execution, not file size.
No throughput claim is made without a benchmark.

`native_tests/grouped_cross_check.cpp`, linked against ordinary serial model
objects, checks energy/force/virial design columns and their mixed head finite
differences, adjoint products, coordinate finite differences, cross-group
column inner products and save/load EFS. It accepts MODEL CFG [ROUNDTRIP].
`native_tests/grouped_cross_cli_check.py BINARY CHECKER` exercises b=1,
G=1, b=2, non-power b=3, Legacy O>K, malformed block metadata and exact G=1
agreement with the old cross-channel model.

Local build uses the serial developer make command with Clang and Accelerate.
Use the system vecLib CBLAS headers (the bundled header lacks the symmetric
BLAS declarations needed by the existing trainer). Tests also run against
cross-generated Effect and Adaptive PT fixtures.

An existing diagnostic limitation was exposed by fresh native initializers:
ordinary shared Static/Effect heads with nonunit species prefactors were
checked against raw TrainLinear columns, omitting that fixed prefactor. The
explicit `fixed_center_readout_columns=true` chart now includes the center in
all shared outer E/F/virial columns and gate-chain terms. The default false
raw TrainLinear chart and solver remain unchanged. The diagnostic requests the
explicit chart for shared heads; species heads keep their existing chart.
`fixed_center_columns_check.cpp` covers zero, negative and unequal centers,
nonunit global scaling, signed linear multipliers, inner on/off, species mode,
and bitwise unchanged raw E/F/virial columns after explicit-chart evaluation.
