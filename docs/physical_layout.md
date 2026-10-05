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
