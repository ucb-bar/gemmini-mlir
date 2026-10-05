# Exact physical layout propagation

`fused_mixed_catalog.merlin_callbacks(..., propagate_layout=True)` and the
pooled-stem wrapper opt into Merlin's static-axis permutation propagation.
The default stays off. Layout runs after direct convolution binding and before
remaining dense catalog binding, so every compiled call matches its source.
Attributed external declarations retain their generic argument properties.

Only proven tensor permutations, scalar arithmetic, static slices and padding
are propagated. Scalar operation order and reduction-axis order stay intact.
Unknown producers, external calls and unsupported reshapes retain explicit
boundary transposes. No residual ABI is implicitly declared layout-polymorphic.

## Measurement and gates

Original exact ResNet, pooled stem + banded3x3 +27 unary fusions:

| Metric | Baseline | Propagation |
|---|---:|---:|
| Return-reachable transpose count |213|76|
| Transpose tensor output bytes |91,200,620|39,037,312|
| Spike retired-instruction proxy |783,481,732|781,440,649|

Native scalar-device oracle and actual Gemmini Spike both preserve all1,000
original output bits, with zero descriptor-rank mismatches. Final ELF has no
FSM instructions. The0.26% instruction improvement is small: logical copy
elimination alone does not establish a hardware speedup. No FireSim claim.

After the parent's first weight-hoist pass, the same census is159copies /
65,697,708bytes before propagation and58copies /25,662,656bytes after.
Of the remainder,53copies /23,454,912bytes are static **second weight
permutations** after reshape chains. Parent is extending upstream weight
hoisting for those. Four late f32 activation copies and one input conversion
account for2,207,744bytes. These counts exclude dead abandoned gathers via
return-reachable SSA; they describe tensor outputs, not actual DRAM traffic.

Receipt: `docs/perf_records/resnet_exact_layout_probe.json`.
Regression tests cover inverse layout cancellation through a scalar epilogue,
unknown call boundaries, and attributed external-declaration serialization.

## Closed-recipe exact50 + CPU LUT census

The same pass reaches89copies /36,500,672bytes from159copies /
44,096,940bytes on the newer exact50/LUT/hoisted prepared graph. Residual
calls are explicit layout barriers; they are not treated as generic scalar ops.

Only25 live generic arithmetic regions remain. Four quantization sites cover
227,840elements: input150,528; late256channel50,176; late512channel25,088;
post-mean2,048. Two exact integer-readout fallback kernels can address the late
75,264elements; these are separate work. One float reduction remains.

The16 exact CPU LUT calls process5,519,360elements. The compiled RV64GC adapter
uses13instructions per element (including indexing, pointer increments and
branch), approximately71,751,680instructions before checks/copies. This is
15.5% of the current463,973,868 whole-model Spike count. Reindexing the table by
raw uint8 bits and unrolling its loop are exact, general opportunities; neither
optimization is implemented by the layout pass.

Full weighted source census: `docs/perf_records/resnet_exact50_host_census.json`.
Per-operation cardinalities are output-element counts; reduction work requires
its reduction dimension and is not included in those simple scalar counts.

The complete upstream static weight-hoist implementation (`c16584354`) plus
layout propagation passes exact50 +CPU LUT +pooled stem whole-model native and
actual Spike validation: all1,000 fresh-capture golden bits exact, descriptor
ranks correct, final ELF no-FSM. Spike records341,424,822instructions vs the
prior463,973,868 with first-only hoisting and no propagation. This26.41% gain
combines both changes and must not be attributed solely to layout propagation.

After full weight hoisting, the prepared activation census is106copies /
20,642,028bytes before propagation and36copies /13,045,760bytes after. Remaining
35tensor copies are16residual outputs and19contraction outputs feeding residual
calls; the other copy is the input conversion. Moving a **shared proven
permutation** across the flat elementwise residual ABI can address this seam.
Receipt: `docs/perf_records/resnet_exact50_nested_hoist_layout.json`.

## Shared residual permutation

`--residual-shared-permutation` (bundle `shared_permutation=True`) is an
additional opt-in for the exact CPU lookup implementation. Each operand must
come from an explicit `linalg.transpose` with identical permutation and
positive, static, unencoded i8 shapes. The proof checks both source-to-result
shape maps and the logical residual shape. Different permutations, missing
transposes, unknown axes and unsupported types retain the original layout;
the route records the refusal reason.

Because the proved lookup is uniform and elementwise, the common permutation
commutes with it. Consume both physical operands, flatten to the unchanged
`[M,64]` ABI, reshape its result to the physical shape, and apply the original
permutation at the output. Subsequent matched residuals repeat this proof;
the general layout pass cancels the intervening inverse transposes. The
manifest and typed external declaration retain the explicit layout proof.

On exact50, all 16 residuals are admitted. The live transpose census falls
from 36 copies / 13,045,760 output bytes to 2 copies / 702,464 bytes. Remaining
copies are the input float layout and the final i8 layout before mean reduction.
The full native model matches all 1,000 fresh-capture golden bits exactly.

Actual Gemmini Spike also passes all 1,000 golden bits, rank mismatch 0, final
ELF no-FSM. With otherwise identical raw-byte lookup scheduling, the shared
permutation reduces retired instructions from 316,587,699 to 254,701,066
(19.55%). Combined with complete weight hoisting and lookup scheduling this
is 45.10% below the earlier exact50 463,973,868-instruction baseline. These
are Spike functional counters; the new combination awaits FireSim measurement.
Receipt: `docs/perf_records/resnet_exact_shared_residual_layout.json`.
