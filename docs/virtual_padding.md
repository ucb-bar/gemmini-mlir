# Exact virtual padding for direct convolution

The opt-in captured requantization bundle option `--virtual-padding` (also
requires `--flat-spatial`) removes only explicitly proven i8 zero pad-one
shells. The proof requires static NCHW extents, exact insert offsets/sizes/unit
strides, and a literal i8 zero splat initializer. Unknown or nonzero padding
retains the original materialized path and records the refusal. No model or
region names participate in the decision.

The typed adapter consumes unpadded NHWC dimensions. Banked spatial bands
partition every tap's spatial run into valid DMA segments and zero-fill lanes.
Invalid lanes use Gemmini's address-zero MVIN primitive; no invalid source
pointer is issued. CPU loop groups have identical vertical validity for every
relative row/tap, proved before code generation. Existing channel tails, band
tails, i32 scratch readouts, reduction order and i8 scale/ReLU are preserved.
The full padded activation matrix is never allocated or copied for a matched
3x3 convolution. The 7x7 stem is separate and retains its existing padding.

## Why this was investigated

The exact shared-NHWC prepared graph had 17 live padding insertions: 16 direct
3x3 and one stem, totaling 2,352,876 output bytes. Lowered LLVM emitted both
constant-zero copies and generic `memrefCopy` calls for the interiors. The old
runtime recomputed four-dimensional offsets and called byte memcpy per element.
A same-shape strict Spike probe measured **165,257,880 instructions** for those
interior copies alone. The parent's general contiguous-suffix runtime fix
reduces the same probe to **1,985,883**, with every interior value checked.
Thus any additional virtual-padding benefit must be compared against that
improved runtime, not credited with the entire old runtime overhead.

## Gates

All kernel outputs and 2,048-byte guards pass GSIM and strict RV64GC Spike:

| Shape H×W / Cin / Cout | Mode | GSIM kernel cycles |
|---|---|---:|
| 3×19 / 20 / 19 | i32, spatial and channel tails | 6,225 |
| 5×19 / 65 / 35 | stride2, band2, i8 scale/ReLU | 19,757 |
| 1×1 / 20 / 19 | i32, all corner halos | 2,725 |
| 56×56 / 64 / 64 | band4, full-range i8, captured scale | 619,364 |

The 56×56 schedule retains 28,224 computes and a 451,584-cycle padded array
issue floor. Border splitting changes A commands from 2,016 to 2,348 (+332);
the removed host pad allocation was 215,296 bytes. The older explicit-halo
probe took 617,005 GSIM cycles; this is not a whole-model speed comparison.

The exact52 bundle admits all 16 direct pad proofs with zero refusals, retaining
both caller-owned i32 scratch/readout routes. Full native and actual Spike
execution preserve all 1,000 fresh-capture golden bits, rank mismatch 0, and
final ELF no-FSM. With new runtime, shared residual layout and raw-byte CPU
LUTs, the complete model records 65,232,457 Spike instructions. The otherwise identical materialized-padding variant with the same fast runtime
records 67,090,426 instructions. Virtual padding alone removes 1,857,969
instructions (2.77%) while preserving every output bit.

Evidence: `docs/perf_records/virtual_padding_gates.json`. No new FireSim claim.
