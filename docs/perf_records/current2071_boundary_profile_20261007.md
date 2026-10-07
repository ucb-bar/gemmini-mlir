# Current ResNet boundary profile

Stock FireSim 2074 completed in 250.14 seconds with all 1,000 original float32
words bit exact (`atol=0`, `rtol=0`). The final ELF has zero LOOP/FSM
instructions. The original 2071 link reproduces byte for byte; every host,
runtime, weights and device object remains unchanged. Timers wrap the actual
declared external ABI: 68 four-pointer adapters, two five-pointer adapters and
one three-pointer classifier primitive. The source call order, symbol counts
and measured interval conservation close all 71 boundaries.

The forward window is 29,738,792 cycles, 40,445 above the accepted unprofiled
2071 observation of 29,698,347 cycles. The main metric is 29,739,307 cycles.
This is a diagnostic profile, not a new optimized champion. The stock archive
and staged bitstream hashes are separate pins; both were verified, along with
the actual simulator-staged ELF before teardown.

| Interval | Cycles |
| --- | ---: |
| Before pooled stem | 2,026,898 |
| All work outside wrapped callbacks, including that interval | 2,178,965 |
| Stem and pooling callback | 1,443,831 |
| 52 convolution/requantization callbacks | 20,934,226 |
| 16 residual callbacks | 4,638,842 |
| Global mean callback | 63,613 |
| Classifier primitive callback | 479,315 |

Callbacks include CPU descriptor checks, readout, instruction issue, transfers
and waits. Their 93% share is **not** accelerator utilization. Outside callbacks
also includes profiler overhead. Pure CPU/device shares remain unknown.

The permitted ZIP reference 1876 remains 22,387,449 cycles. Matching declared
layer geometry locates callback differences of 1,030,293 cycles in pointwise
convolutions, 1,273,432 in spatial convolutions, 2,446,449 in residuals,
360,774 in stem/pooling and 49,534 in classifier. The source numerics and timer
boundaries differ; these are locations of extra work, not independently
attainable savings. Global mean and host intervals also have different scopes.

The largest between-call interval directly motivates the pending generic
Merlin scheduling change for bounded-RNE packet helpers: the current pre-stem
map calls a 24-argument helper 18,816 times, with stack argument traffic. Target
schedule/layout work remains necessary after host preparation is improved.

The failed initial ABI assumption was retained in the work directory: two
actual LLVM declarations have five pointer arguments. The profiler now accepts
that explicitly supplied ABI; unsupported names and arities still refuse.
Eight existing profiler tests pass, and actual Spike and FireSim validate all
five arguments through the unchanged model computation. The initial preflight
attempt also retained its failure: the expected unpacked `.bit` hash was supplied
to a check of the compressed archive. Correct archive and `.bit` identities
were then independently closed, with no hardware configuration edits.

Phase 0/1/2 tooling should preserve this pattern: derive observation boundaries
from declared ABI and selected source operations; close complete callback
coverage; conserve one whole execution window; retain original accuracy gates;
price host preparation and dispatch as well as target primitives. Keep mixed
callback timings, instruction counts, analytical floors and whole hardware
cycles distinct. General observation/provenance infrastructure belongs in
Merlin; target timer/ISA support belongs in this OOT repository.

Task-specific token consumption is unavailable. The active goal counter is a
historical aggregate and is not attributed to this profile or Gemmini work.

Evidence: [terminal receipt](root_resnet2071_stock2074_boundary_profile_terminal_20261007.json),
[raw UART](stock2074_resnet2071_boundary_profile_uart.txt),
[source/ABI manifest](root_resnet2071_boundary_profile_manifest_20261007.json),
[diagnostic driver snapshots](source_snapshots/current2071_profile_20261007).
