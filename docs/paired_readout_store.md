# Exact paired readout epilogue (optional)

`GoldenFlatConv(..., store_plan=PairedReadoutPlan(...))` consumes an unscaled,
unactivated, zero-seeded signed-i8 convolution producer and emits two rounded i8
outputs from each completed accumulator block. Its raw ABI is
`(A:i8*, B:i8*, first:i8*, second:i8*)`. Both output buffers contain OH*OW*Cout
bytes and must be disjoint from each other and the inputs. The provider/caller
must preserve both until the decoder finishes; no public routing is changed.

The Merlin certificate partitions the complete declared i32 domain at every
source and store transition. Every reachable pair must identify exactly one
source output. An enclosing pair alone is insufficient and is refused when
ambiguous. The target constructor additionally checks conservative signed-i8
bounds (including -128) for all 9*Cin products and prefixes, and requires their
containment in that certificate. Source scale provenance must still be bound by
the calling compiler; the primitive constructor cannot infer upstream semantics.

Each block performs all first-scale wide stores, then all second-scale wide
stores, before any subsequent accumulator overwrite. The existing final fence
completes both outputs. Primitive RoCC commands and ordinary CPU loops are used;
no FSM instructions are introduced. `loop_spatial` remains independent.

Default `store_plan=None` preserves six independent original xDSL programs byte
for byte. Structural tests preserve all reduction/load commands, reject unsafe
contracts, and verify the four-pointer signature. Two independent non-square,
stride/tail cases pass actual strict Gemmini Spike against original ordered f32
readout, all outputs and 2048-byte guards, with final executable zero-FSM audit.
These are functional gates, not a whole-model performance result. The matched
original-accumulator readout-only GSIM screen is tracked separately.
