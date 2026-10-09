# Gemmini operational service calibration

Owner: `reference_parity`; model fitting belongs to root's shared Merlin tools.
Stock queue ownership remains `firesim_recovery`.

The final v5 ELF passes strict RV64GC Spike with all 21 counter rows, full exact
i32 outputs, every loaded signed byte, guards, tails and immutable inputs. Its
checksum is `827e6e5c9d2b1b85`. The GSIM replay is active; partial measurements
are excluded. There is no timing fit or stock result in this release.

| Stream | Training counts | Held out | Timed work |
| --- | --- | --- | --- |
| Resident GEMM, N=K=64 | M=32,128 | M=64 | increasing-K PRELOAD/COMPUTE, 32/64/128 of each |
| Requested i8 DMA loads | 16,256 panels | 64 panels | disjoint 16×16 panels, 256 requested bytes each |
| Raw i32 readback | 16,256 stores | 64 stores | immutable 16×16 ACC tile, 1,024 requested bytes each |

Every stream has two repetitions plus three separate empty controls. The same
fenced mcycle/minstret call window contains command issue and completion stalls.
Actual input DMA and ACC preparation finish before the window; verification
readback finishes afterward. These are operational costs, not pure rates.

The target header statically verifies DIM=16, four 4,096-row SPAD banks and
1,024 ACC rows. The complete resident operands use at most 512 A rows plus
256 B rows at bank 2, and 512 ACC rows. Requested-load panels occupy at most
bank 0; identity B remains disjoint in bank 2. Readback writes disjoint private
host panels. ELF object ranges prove 64-byte alignment, full extents and input,
output and context disjointness. This does not establish physical DDR traffic
or cache coherence granularity.

Executed primitive totals agree with every source-declared ROI count, without
dividing varied-size function aliases. The final ELF contains legal primitive
commands and no FSM instructions. A failed direct SPAD validation attempt on
installed Spike row 1,024 is preserved: all load panels are now independently
verified using exact identity GEMM and raw i32 stores outside the load timer.
Compile failures and earlier source forms remain archived.

Current Rocket/Gemmini sources show fence stalls while RoCC commands and busy
work remain. Their source pins and the actual engine receipt are retained.
The historical GSIM FIR path is unavailable, so source-to-that-FIR identity is
unknown; the pinned actual engine and measured operational protocol are kept
distinct from any inferred service rate.

`gemmini_primitive_service_battery_spike_release.json` pins the final source,
manifest, expected values, compile/link argv, compiler stages, dependencies,
ELF, parser, strict console/histogram, resource witness and negative evidence.
CPU21 and gather released source files remain unchanged.
