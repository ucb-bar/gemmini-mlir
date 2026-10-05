# Prefetch B across K for cached-A dense kernels

`Shape.prefetch_b` is an explicit, default-off schedule choice. The source-bound
entrypoints accept `prefetch_b=True`, including
`merlin_builder(llvm_bin, large_n=True, prefetch_b=True)`. The integer contraction
matcher still requires the original dense signed i8 × i8 → i32 semantics with
zero initialization. Unsupported shapes retain the existing schedule.

## Schedule and resource proof

The complete A operand remains resident in the lower two scratchpad banks. Each
B panel alternates between banks 2 and 3. Before computing a current panel, the
CPU issues the next panel's loads to the other bank. The first load is outside
the reduction loop. A paired ordinary CPU loop fixes both bank addresses at
compile time; a static drain handles the final panels and exact K tails. The
kernel retains the original increasing-K accumulation, output stores and
optional bias initialization.

The option requires cached A, multiple K panels, no competing bank placement,
A fitting the lower two banks, and a B panel fitting one bank. In the stock
configuration, M8/K2048 and M8/K5632 reserve 2,048 and 5,632 A rows respectively;
BN64 reserves 1,024 rows per B panel in each separate 4,096-row bank. The
existing bounded dynamic A address and target IR verification remain. No new
dynamic B operation, hardware loop or FSM instruction is needed.

The analytical primitive count, requested transfer volume and padded array
issue floor are unchanged. This schedule exposes overlap between DMA and mesh
work. Instruction count alone does not predict its benefit.

## Measured standalone evidence

Both representative projection capsules originate from the same exact upstream
integer contraction. They check all 16,384 i32 outputs and 2,048 guard bytes
with independent host integer expected values and signed operand amplitude 21.
The candidate's mixed N173/K129 tail case checks all 1,384 outputs plus guards.
All final ELFs pass stock GSIM, strict RV64GC Gemmini Spike and no-FSM audits.

| M8/N2048/K2048, BM1/BN64 | Existing cached A | B prefetch |
| --- | ---: | ---: |
| GSIM kernel cycles | 927,138 | 700,215 |
| Spike retired instructions | 74,832 | 102,003 |
| Padded array issue floor | 262,144 | 262,144 |

GSIM kernel cycles decrease 24.475%. This is a standalone RTL measurement;
whole-model and FireSim cycle results require their own original reference
gates and hardware runs. The default-off model-scale device object was also
compiled from the previous source and remains byte-identical after the helper
refactor. Default catalog kernel symbols retain their previous identities.

Receipt: [b_prefetch_gsim.json](perf_records/b_prefetch_gsim.json).

## Full-model qualification

The full 22-layer, eight-token TinyLlama candidate enables prefetch on 111 of
155 contractions, across four of five device kernels. The N256/BN16 kernel
retains its existing schedule. The four-output/K4 host object and all selected
host LLVM bytes match the measured job 1832 control exactly. Fresh native
execution passes the original Torch gate (atol 0.03125, rtol 0.02); final strict
Spike matches all 256,000 f32 output words and reports 172,684,059 retired
instructions. The final ELF has no FSM instructions. These checks qualify a
device-only hardware comparison against job 1832's 612,076,811 forward cycles;
they do not establish a whole-model cycle improvement.

A further M8/N2048/K5632 capsule exercises resident A across both lower banks.
It passes all 16,384 outputs and 2,048 guard bytes on strict Spike and GSIM.
GSIM reports 1,987,584 kernel cycles. The original 180-second GSIM attempt
finished the kernel but timed out during harness verification; its failed
receipt is retained. Retrying the identical ELF with a 300-second wall limit
completes verification. The full-model receipt pins both records.

Receipt: [tiny_four_outputs_unroll4_b_prefetch_spike.json](perf_records/tiny_four_outputs_unroll4_b_prefetch_spike.json).

The eight-output/K2 composition also passes the original native gate and all
256,000 Spike output words. Its host object matches measured job 1835 exactly,
and its device object matches the isolated four/K4 prefetch variant exactly.
Final strict Spike reports 171,353,332 retired instructions. Hardware submission
is held until the isolated comparison establishes a device cycle improvement.

Receipt: [tiny_eight_outputs_unroll2_b_prefetch_spike.json](perf_records/tiny_eight_outputs_unroll2_b_prefetch_spike.json).

## Compiler and infrastructure follow-up

The tuning abstraction is an operand residency decision plus a two-slot K
pipeline. Automatic search must couple prefetch distance and bank allocation
to capacity facts, preserve data dependencies, and measure hardware overlap.
The current typed schedule is a concrete example for that future transform.

Complete-model compilation retains the immutable pre-offload source snapshot,
source-bound catalog coverage, supplemental object bytes and link order in
artifact identity. The selected prefetch policy remains explicit in the catalog.
The GSIM numeric harness currently uses the legacy target backend provider;
migrated core lowering compiles the actual xDSL device objects. The probe reads
`mcycle` in its existing M-mode runtime so strict RV64GC Spike can execute it
without adding Zicntr. This changes the numeric harness counter spelling and
leaves device instructions and arithmetic unchanged.
