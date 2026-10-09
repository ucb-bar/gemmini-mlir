# Exact independent host arithmetic lanes

The opt-in Merlin `scalar_pointwise_packet.py` pass interleaves two or four
independent scalar lanes of a static, pure all-parallel tensor body. Selection
uses affine maps and existing scalar FMA/division cost. It preserves source
operation order inside each lane and grants no floating reassociation. Upstream
bufferization retains tensor alias semantics. Static tails use separate bounded
packets. Source output-initialization reads, unknown arithmetic, nontrivial affine
expressions, fastmath and constrained floating scopes remain unsupported.

This shared host scheduling implementation belongs in Merlin. Gemmini's OOT
repository supplies its actual Rocket verification and performance records.
Both feature choices are default off; model names and provenance IDs never
select the transform.

## Measured capsule

The independent 2,049-element capsule reproduces the current dequantization,
source FMA exponential approximation, reciprocal division, gating and RNE
arithmetic. Every output byte and 128 guard bytes pass native execution, strict
RV64GC Spike and actual Rocket GSIM. The final ELF contains no FSM instructions.
The timing includes the helper call, defensive result copy and arena allocation.

| Schedule | Cold GSIM cycles | Warm GSIM cycles | Warm change |
| --- | ---: | ---: | ---: |
| Scalar control | 334,650 | 328,373 | — |
| Two independent lanes | 258,440 | 255,845 | −22.09% |
| Four independent lanes | 269,976 | 268,247 | −18.31% |

The functional Spike counts rise from 102,514 to 108,797/109,321 instructions.
Actual Rocket latency improves despite that increase. Neither counter predicts
whole-model FireSim speedup. Full details and engine/compiler/input/output pins
are in `perf_records/tiny_pointwise_packet_gsim.json`.

The two-lane whole Tiny candidate preserves all original 256,000 compiled
output words and the unchanged Torch gate in native execution. Device kernel and
shim bytes are identical to qualified stock job1846. Full strict Gemmini Spike
qualification also retains every original bit, with166,605,674 functional
instructions versus1846's168,567,482. Runtime, device, shim, weights, startup
and console objects are byte-identical to1846; only the host model and
marker-bearing harness object differ. Stock FireSim 1880 independently verified531,072,370 whole-model cycles,
38,078,697 cycles (6.6904%) below 1846 while preserving all 256,000 words
and the immutable Torch gate. See
`perf_records/tiny_pointwise_packet_spike.json`.

Stock FireSim job 1880 is admitted with the qualified ELF, independent
qualification/reference checks and detached collector 2863471. The original
qualification receipt remains immutable. Actual whole-model cycles closed at 531,072,370; see
`perf_records/tiny_pointwise_packet_firesim1880.json`. Admission is recorded in
`perf_records/tiny_pointwise_packet_firesim1880_admission.json`.
