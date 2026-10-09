# Bounded weight prefetch for compact resident convolution

The owned `resnet50_nofsm_q1013.zip` executable guided a comparison of
convolution operand reuse and transfer order. Its comparable 14×14, 256 input
channel, 256 output channel function executes 32,256 computes and uses output
channel blocks of 32. Known decoded B addresses span several reduction panels.
Unresolved reference addresses remain unknown. Its weights, inputs and
quantization differ from our original capture.

We tested that geometry on the original ABI operands from stock job 1886. The
control is its exact device object, not a regenerated approximation. Native
operand capture retains all 1,000 original output words exactly. Both arms use
the same A, B, expected output and guarded output addresses.

| Schedule | Complete device GSIM cycles | Change versus control | Decision |
| --- | ---: | ---: | --- |
| Current compact resident A, one B panel, BN4 | 535,839 | — | Control |
| Complete reduction B residency, BN4 | 558,978 | +4.32% | Reject |
| Compact resident A, BN2 | 572,850 | +6.91% | Reject |
| Complete reduction B residency, BN2 | 628,003 | +17.20% | Reject |
| Compact resident A, next K B prefetch, BN4 | 499,034 | −6.87% | Admit to whole model test |

Every row passes all 50,176 original output values, 4,096 guard bytes, strict
RV64GC Spike execution, and the final ELF audit for zero FSM instructions.
These are GSIM capsule measurements. They do not establish whole model gains
or predict reference FireSim layer cycles.

## Compiler rule

When compact resident A is legal, an explicit schedule can reserve two
nonoverlapping weight slots in separate scratchpad banks. Load panel zero;
then issue the next K panel DMA into the other bank before the current K panel
computes. A slot is reused only after its earlier panel's commands. The
ordinary command stream preserves increasing HWIO reduction order.

The generator proves complete A and weight slot extents, accumulator capacity,
adjacent row group spans and tail output channels. In the measured fixture, A
uses bank 0; the two 64 row weight slots begin at rows 8,192 and 12,288 in
banks 2 and 3. Scratchpad placement and ISA facts belong to the OOT target.

Complete static CFG execution proves the BN4 control and prefetch have identical
32,256 computes/preloads, 2,304 real weight preloads, 752 loads, 56 stores, and
configuration/fence counts. Logical A accesses, source weight transfers,
source K order, accumulator destinations and output stores match. The emitted
delta is B placement and one panel transfer lookahead. The overlap mechanism
is a hypothesis supported by the measured gain; static command order does not
prove command queue dispatch timing.

## Interface and gates

`ResidentConvOptions.prefetch_b` defaults to false. The
`compact_channel_planes_prefetch_b` policy is an explicit source bound compiler
option. It derives adjacent row groups and checks target resources without
model names or provenance ID strategy lookup. Its reduction commands currently
use static emission; requesting the separate channel loop together with
prefetch refuses. The existing default regenerates the original device object
byte for byte.

Three generic static trace tests cover different spatial widths, grouped rows,
partial output channels and output types. A separate 5×5, Cin32, Cout19 i32
capsule checks all 475 outputs and 2,048 guard bytes with full signed i8 input
range, virtual halo padding, and grouped rows. It passes GSIM, strict Spike and
zero FSM audit.

The next whole model comparison uses immutable stock job 1897 as control,
38,468,933 cycles. Its existing compact resident source bindings delimit the
isolated arm; the chooser receives their shape, resource and numeric contracts.
All other kernel, adapter, host, runtime and parameter objects must remain
identical. Selection stays explicit until the full original numeric gates and
stock hardware comparison close.

The complete candidate now passes native and strict RV64GC Spike checks against
all 1,000 original output words, with zero FSM instructions. Five existing
compact convolution kernels change; all other 47 unary kernels, 52 adapters,
source proofs and other device implementations retain their control bytes.
The controlled final link preserves every host, runtime, weight and harness
object from 1897. Its ELF SHA is
`5b1d3f1641e9aab4e1b899ddea7aee7d10352daaa5d5e5fb5f6727c39fe5cc45`.
Its retained baseline harness marker identifies the frozen harness only.

The fresh normal compiler build also passes the complete numeric gate and emits
the same selected device aggregate. Its host bufferization order differs despite
identical captured, prepared and catalog source bytes. The cause is unknown;
that fresh ELF is retained separately. The hardware comparison uses the frozen
1897 host object to isolate the device change. Complete pins and scope are in
[compact_weight_prefetch_whole_qualification.json](perf_records/compact_weight_prefetch_whole_qualification.json).

## Current stock profile

Job 1899 profiles the exact 1874 implementation. Its 70 primitive intervals and
host gaps conserve 39,235,729 forward cycles. Instrumented whole timing is
39,236,274 cycles, 34,995 above the uninstrumented 1874 control. These intervals
do not describe the newer 1886 or 1897 implementations.

| Interval class | Actual 1899 cycles | Reference 1876 cycles |
| --- | ---: | ---: |
| Pointwise | 11,344,741 | 9,907,412 |
| Direct convolution | 11,971,291 | 8,723,089 |
| Residual | 5,471,537 | 2,192,393 |
| Pooled stem | 1,447,500 | 1,083,057 |
| Classifier | 478,845 | 429,781 |
| Host gaps and tail | 8,521,815 | 51,717 |

The largest host gaps precede the stem (4,016,144 cycles), matmul 26
(1,992,494), matmul 49 (962,283) and matmul 14 (615,348). They include every
intervening CPU operation and timer overhead. They do not identify an individual
packing or readout operation's cost. Reference inputs, weights, quantization and
timing boundaries differ, so these comparisons locate work to investigate
without asserting causal savings. All 54 geometry pairs, 70 actual calls and
hardware/source pins are in
[q1013_1874_current_profile_alignment.json](perf_records/q1013_1874_current_profile_alignment.json).

## Reusable optimization lessons

Complete residency and fewer transfer commands can serialize useful overlap.
Matching a reference block width can lose under a different numeric and command
contract. Search should retain distinct placement, lifetime and lookahead
choices and measure their complete cost. This rule is usable by future shared
schedule selection; no shared whole model solver or timeline prediction is
claimed here.

Exact byte pins, complete primitive traces, raw receipts and rejected arms are
in [compact_weight_prefetch_capsule.json](perf_records/compact_weight_prefetch_capsule.json).
Parent token accounting is recorded separately; these receipts do not claim
exclusive per optimization token attribution.
