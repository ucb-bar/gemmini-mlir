# Reuse resident input through the source spatial stride

The owned reference ZIP's `fx_conv_26` configures mesh A stride 2 and a 900 row
channel plane pitch. Known commands compute 14 output rows from that resident
layout. Some reference addresses remain unresolved. The reference uses different
inputs, weights, quantization and output readout; its 555,284 stock layer cycles
are separate evidence from the matched experiment below.

The current stride 2 convolution at 28×28/Cin256/Cout256 uses a flat spatial
schedule that gathers its input again for each tap and output channel block.
Actual source-bound command tracing counts 1,721,344 non-null input bytes. The
original input occupies 200,704 bytes. A padded resident input needs 14,400
scratchpad rows; a 64 row weight panel fits immediately after it, with 896
accumulator rows. These bounds derive from source shape and target resources.

The OOT lowering previously discarded `ConfigExOp.a_stride`, although the ISA
encoder and RTL support that field. The dialect now verifies a positive 16 bit
stride and the lowering retains it. The default value remains 1. The resident
generator has an explicit `source_stride` option and an optional proved weight
base. It partitions wide rows into legal DMA commands, preserves increasing
HWIO reduction order, and checks every strided read against its input plane.

| Same original device capsule | GSIM cycles | Correctness |
| --- | ---: | --- |
| Exact 1897 flat control object | 727,070 | 50,176 i32 outputs and 4,096 guards pass |
| Source stride resident input | 577,376 | Same complete values and guards pass |

The measured gain is **149,694 cycles (20.59%)**. Both arms use common operand,
expected output and output addresses. Strict RV64GC Spike and final ELF audits
pass with zero FSM instructions. The final opt-in generator reproduces the
tested object byte for byte; the existing compact default also regenerates its
original object byte for byte.

The new schedule increases computes from 29,952 to 32,256 because each physical
output row gets its own mesh tile. It removes 1,520,640 requested input bytes;
weight bytes remain 589,824 and output bytes remain 200,704. Complete command
traces prove output coverage and resource extents. Counts alone do not establish
hardware timing or command dispatch overlap.

An independent 5×21/Cin32/Cout19 stride 2 case validates all 627 i32 outputs and
2,048 guards over full signed int8 inputs, including partial spatial and channel
tiles. It uses the shape-derived remaining-row B placement, and passes GSIM and
strict Spike. Separate command tests compare decoded target behavior with an
independent scalar convolution, including grouped rows, tails, explicit refusal
cases and source K order.

`--source-stride-resident` is opt-in. Selection requires source-proved virtual
padding, spatial scheduling and a legal source/resource layout. A serialized
padded DIM issue plus requested 16 byte transfer score ranks this admitted
family; its 58,176 point improvement is an estimate, not a hardware cycle
prediction. Failed legality or ranking retains the existing schedule. No model
names, provenance IDs or observed runtime values select the strategy. ISA,
scratchpad layout and target schedule rules remain in the OOT backend.

The whole model still requires the original 1,000 output words to match exactly.
No new whole model FireSim gain is established by this capsule. Exact receipts,
reference scope, source bindings, emitted deltas, hardware pins and independent
checks are recorded in
[source_stride_resident_conv_capsule.json](perf_records/source_stride_resident_conv_capsule.json).
