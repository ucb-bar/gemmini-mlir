# Jack q1013 executable versus the current exact compiler plan

The owned archive contains the ELF, disassembly, FireSim bitstream bundle and
manifest. It contains no C source or per-layer hardware UART. The manifest gives
22,387,449 whole-model cycles. No Jack directory was opened. This study uses
the owned executable and its existing actual Spike PC histogram, and verifies
that all52 non-stem convolution geometries match our current source plan.
[Pinned per-layer command/shape comparison](perf_records/q1013_current_strategy_parity.json).

Reference command counts below come from the executed reference path. Counts for
shared identical-shape functions are divided by their number of calls; divisibility
is checked. Reference per-layer hardware cycles remain unknown. The current
device times are the measured1849/1850 control profile, and the receipt separately
records1853's selected strategies. There is no per-layer1853 timing claim.

| Device class | Jack executed computes | Current1849 computes | Current1850 device cycles |
|---|---:|---:|---:|
| Pointwise |555,840|555,840|13,150,309|
| Direct3×3 |524,736|499,392|11,969,037|
| Residual |22,208|324,576|6,769,409|
| Packed stem and pool |43,904|49,000|1,446,279|
| Classifier |8,064|8,064|479,009|
| Mean |512|Host|Included in host intervals|

Multiplying every compute by padded DIM gives18,484,224 reference geometry and
22,989,952 current geometry. The4,505,728 difference consists of4,837,888 more
residual geometry cycles,405,504 fewer direct-convolution geometry cycles,
81,536 more stem geometry cycles and8,192 fewer device mean geometry cycles.
These are arithmetic/tiling counts, not mandatory hardware cycle floors:
execute waves can be shorter and transfer/execute work can overlap.

## Techniques already implemented

The old October1 comparison identified accelerator FC, packed stem K, hardware
pool, wide stores, independent load streams, banked A/C buffering, weight reuse
and removal of host im2col. Those are represented in the current compiler plan.
They cannot be counted as new savings on the current40,479,548-cycle1853 result.
The source-bound full52 receipt preserves the exact original1000-word output.

The direct-convolution compute count is already below Jack's; further reducing
compute count alone does not address its repeated operand transfers. The wider
resident stripe schedule therefore makes a measured tradeoff: more row-tail
computes, fewer activation and weight transfers.

## Numeric and interface work differs

Our source entry accepts tensor<1×3×224×224×f32> and returns1000f32 logits.
Jack's measured default path starts with an int8 activation and uses a narrowed
classifier output. The source entry quantizer, source layout work and exact full
logit conversion must be included when qualifying this compiler workload.
The measured pre-stem host interval is4,017,289 cycles; that interval includes
all intervening CPU work and does not isolate quantization by itself.

The source residual scales also differ. Jack uses one identity-weight compute
per output tile after input narrowing. The exact source-bound residual here
requires a different arithmetic implementation; its first residual has39 chunks.
The exhaustive coefficient certificate proves39 minimal for the existing positive
integer coefficients/one binary32 scaled-store family, including accumulator bias.
It does not prove every exact algorithm requires39 chunks. Changing scales,
weights, input/output interfaces or goldens would not qualify this source model.
[Numeric family certificate](residual_coefficient_minimality.md).

## Concrete remaining priorities

1. **Remove redundant host work while retaining its required computation.**
   The measured control contains8,489,121 host cycles. The4.02M pre-stem interval
   and the1.96M/0.96M intervals before two exact integer readouts are the largest
   targets. Source-bound fusion, ownership/layout propagation, packing and exact
   requantization belong in Merlin. The readout intervals include other work.
2. **Keep complete activations and complete reduction weights resident when
   resources allow it.** The new general stripe convolution fits H56/C64 and
   H28/C128. H56/C64's paired full-range capsule measures619,364→520,943 kernel
   cycles (15.89% lower), with all outputs, guards, strict RV64GC Spike and final
   zero-FSM audit passing. Narrower BN1 measures539,920, slower than BN4 despite
   fewer real-B reloads; output transfer granularity matters. This target schedule
   belongs in OOT. [Implementation and measured tradeoff](resident_stripe_conv.md).
3. **Reduce exact residual scheduling overhead and investigate other exact
   arithmetic/readout families.** Existing39-chunk coefficient minimality is a
   family-specific constraint. Banked M prefetch and shorter legal execute waves
   are general schedule improvements; alternative numeric derivations need their
   own full-domain proofs and measured DMA/readout costs.
4. **Recover actual reference per-layer hardware costs.** The archive ELF prints
   them, but their UART is absent. A diagnostic replay needs an explicit changed
   artifact ledger: six dynamically dead library LOOP words are present, and
   `.diag` mixes code and a table under an executable section flag. The original
   strict audit fails. This work must not be conflated with our source qualification.

Whole-source native, actual target and stock FireSim gates decide whether the
locally faster resident convolution should join the best recipe. Capsule gains
and independent whole-model gains are not assumed additive.
