# Complete-input and full-reduction weight residency

`GoldenResidentStripeConv` derives a primitive xDSL Gemmini schedule from an
unpadded NHWC/HWIO stride-one 3×3 convolution with aligned input channels.
It loads each padded channel plane once, reserves every reduction panel for
the current output-channel block, and partitions output rows into stripes
whose accumulators fit. Repeated tap gathers and repeated per-stripe weight
transfers disappear. Source K order, integer accumulation, output dtype,
scale and ReLU remain unchanged.

Activation and weight intervals must be disjoint and fit the scratchpad.
Every input DMA partition covers at most16 rows and64 channels; partitions
cover the complete resident extent exactly once. Every dynamic compute has
a proved maximum address and complete reserved extent. Independent M/N tails,
multiple N blocks, stripe tails and both output dtypes are qualified. Unsupported
stride, input-channel tails, explicit halos and insufficient resources refuse.
The existing direct and narrow resident schedules remain available.

The archived Jack executable motivated this general residency family. No model
name, source region or benchmark value selects its behavior. The target-specific
layout, command generation and resource checks belong in the OOT backend.

## Measured capsule result

For H56/W56/Cin64/Cout64 with the existing proven scale and ReLU, full signed
input-range fixtures measure619,364→520,943 fenced GSIM kernel cycles,
98,421 (15.89%) lower. Both verify all200,704 outputs and2,048 guards, pass
strict RV64GC Spike and the final linked zero-FSM audit. Fixtures are byte
identical. The new schedule uses13,456 activation rows and2,304 weight rows,
15,760 of16,384 scratchpad rows. Weight loads drop504→36 and activation
loads2,348→344. Compute commands increase28,224→32,256 because each output
row has its own tail; this is a measured tradeoff, not a compute-count win.

Two additional full-range capsules cover an odd width and partial output
channels with i32 readout, and multiple N blocks plus a final short stripe
with scaled/ReLU i8 readout. For H28/C128, the control630,670→509,443 kernel
cycles saves121,227 (19.22%) with BN4; all100,352 outputs and guards pass.
BN2 takes576,438 cycles and H56/BN1 takes539,920, so fewer real weight
preloads alone did not select the faster output-transfer geometry.

The explicit `resident_stripes` compiler policy retains the existing legal BN,
admits wider source-proven virtual padding only, and compares padded DIM issue
geometry plus requested DMA bytes at16 bytes/cycle. This serialized search
estimate is not a measured timing prediction or mandatory lower bound. It
records both costs and every refusal; narrow resident planes, unsupported
semantics, insufficient resources or a non-improving score retain the control.
The option is available through `captured_requant_bundle.build` and the CLI,
with default off. Existing narrow resident experiment selections are retained.
Forty focused tests pass, including policy decisions, numeric-field preservation,
source-proof refusal and unchanged direct/requant paths.
[Complete receipt](perf_records/resident_stripe_conv_gsim.json).

Whole-model source binding and stock FireSim qualification are still required.
This explicit schedule is not enabled as a universal default.
