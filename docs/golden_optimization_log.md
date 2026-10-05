# Golden Gemmini optimization log

This log records changes that the later automatic lowering must reproduce. A result is called **measured** only when a numeric probe or model run completed on the named engine. GSIM kernel cycles, FireSim full-model cycles, analytical floors and inferred class gains are different quantities. The device target is `FireSimGemminiRocketConfig`; every final linked ELF must contain zero Gemmini hardware `LOOP_*` instructions, including unused linked code.

## Reference and comparison contract

Jack's supplied `resnet50_nofsm_q1013.zip` records **22,387,449 FireSim whole-forward cycles** for ResNet-50. We read the ZIP, never Jack's folders. Its executing path uses 53 handwritten Exo convolution kernels with no host im2col and no dynamic hardware LOOP commands. The linked vendor library does contain six unused `LOOP_WS*` instructions, so that ELF fails our stricter static zero-FSM rule. The reference bitstream matches our pinned stock `FireSimGemminiRocketConfig` tar byte for byte. The archive has no source or per-layer FireSim UART log; its disassembly, Spike dynamic instruction histogram and whole-model cycle number are the available comparison evidence. The [reference analysis](/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-studies/exo-comparison/q1013_analysis.md) separates direct counts from per-class cycle estimates.

For each optimization, collect: source and compiler SHA, source operation or capsule identity, tensor layout, schedule flags, final ELF hash and no-FSM audit, numerical oracle, engine hash, timing window, run status, cycle count, and analytic command/byte counts. Compare only like for like. For FireSim group measurements, use the same stock bitstream and the same input, quantization semantics, cold/warm policy and `rdcycle` window as the reference or record the difference.

The reference's measured dynamic compute-command ledger is the section-level target for the handwritten schedule. The ZIP has no section-level FireSim cycle log, so only the whole-model 22.387M cycles and these Spike command counts are direct reference observations:

| ResNet section | Jack dynamic computes | Current xDSL golden status |
| --- | ---: | --- |
| Stem convolution and pool | 43,904 | Packed stem and pool-on-store not yet generated |
| 1×1 ReLU | 241,984 | Dense kernel numerically tested; whole graph layout/binding pending |
| 1×1 no ReLU | 216,832 | Dense kernel available; whole graph pending |
| 1×1 downsample | 97,024 | Dense kernel available; stride/layout binding pending |
| 3×3 stride one | 423,360 | Whole-output-row direct gather not yet generated |
| 3×3 stride two | 101,376 | Whole-output-row direct gather not yet generated |
| Residual identity matmul | 22,208 | Isolated primitive kernel numerically tested |
| Global average pool | 512 | No model-bound golden kernel yet |
| FC | 8,064 | Source matmul has a compiled device specialization |
| **Total** | **1,155,264** | No whole-model golden run yet |

## Measured device changes

| Change | Shape and engine | Before → after | Evidence and automatic-lowering lesson |
| --- | --- | ---: | --- |
| Keep B stationary across output-row tiles | 512×64×64 i8, pinned GSIM | 25,360 → 17,727 kernel cycles | Numeric and linked-ELF audit pass. The schedule must represent reuse scope, not infer it from a generic matmul opcode. |
| Cache a small whole B in scratchpad | 512×64×64 i8, pinned GSIM | 17,727 → 16,575 | Numeric pass. Scratchpad capacity and output-channel block count are legality constraints. On 17×19×20 it regressed 776 → 802, so select by shape and measurement. |
| Widen A/B MVIN panel commands to 16×64 | 512×64×64 i8, pinned GSIM | 16,575 → 15,607 | Same panel payload, fewer RoCC load commands. Typed Gemmini MVIN verifier now admits up to 64 scratchpad columns; accumulator MVIN remains DIM-wide. |
| Widen MVOUT to 16×64 for full i8 tiles | 144×64×64 i8, pinned GSIM | 8,414 → 8,142 | The 16×64 accumulator store is bit exact. At 16×64×16 it regressed 613 → 658, so it is a schedule choice, not a global rule. |
| Stock real-D residual identity matmul | 16×16, 17×19, 17×64 i8, pinned GSIM | 431, 3,192, 1,837 kernel cycles | Full numeric/guard/audit pass. The stock config admits real D; lean forces D garbage. Handle edge staging and output pitch explicitly. |
| Hoist Gemmini configuration outside batch loop | TinyLlama `matmul_194`, B32 M8 N64 K8 i32, pinned GSIM | 11,736 → **9,282** kernel cycles | Full numeric/guard/audit pass; [machine-readable record](perf_records/tinyllama_b32_config_once.json) gives both ELF/object/engine hashes. One config, flush and final fence for all 32 batches; ordinary CPU loop repeats primitive tile commands. The B2 M17 N19 K20 case changed 1,492 → 1,506, so small batches can regress. |
| Hoist Gemmini configuration outside batch loop | SmolVLA `matmul_384`, B15 M50 N64 K113 i32, pinned GSIM | 83,398 → **81,607** kernel cycles | Exact upstream object passed full numeric/guard/audit before and after; [machine-readable record](perf_records/smolvla_b15_config_once.json). The 2.2% gain is smaller because GEMM work dominates setup. |
| Cache a short A tile across N blocks | Synthetic TinyLlama-like M8 N512 K256 i32, pinned GSIM | 31,947 → **26,941** kernel cycles | Both final ELFs pass numeric, guard and no-FSM audit; [record](perf_records/tinyllama_like_cache_a_8x512x256.json). A MVIN commands fall from 128 to 16 across eight N blocks; compute count is unchanged. The model selector applies this schedule only when at least four N blocks reuse one legal M≤16 A tile. TinyLlama's 8×5632×2048 and 8×32000×2048 projections meet the geometric rule, but their own cycle gains remain unmeasured. |
| Keep cached-A K repetition in a CPU loop | Same synthetic M8 N512 K256 i32, pinned GSIM | 26,941 unrolled → **26,948** looped kernel cycles | Full numeric, output guard and final-ELF no-FSM checks pass; [looped versus baseline record](perf_records/tinyllama_like_cache_a_dynamic_8x512x256.json) is 31,947 → 26,948 (1.186×). The seven-cycle difference from unrolled is negligible at this shape. The TinyLlama eight-kernel catalog object shrinks from 953,672 to **31,856 bytes**; the baseline object was 28,432 bytes. All 200 contractions still bind through Merlin. No full TinyLlama model cycles have been measured. |
| Alternate two accumulator/A slots per M block | 512×64×64 i8, pinned GSIM | 16,674 → 16,732 | Numeric pass but a regression. This simple alternation does not reproduce Jack's interleaving of next MVIN and previous MVOUT inside the current compute burst; leave the flag off by default. |

The `M=3136,N=64,K=64` biased/scaled/ReLU 1×1 probe passes at 104,931 GSIM kernel cycles with 3,136 compute commands and a 50,176-cycle peak-array floor. The source-bound synthetic upstream 1×1 capsule at 512×64×64 passes at 17,822 cycles. These are isolated kernels. Neither is a FireSim ResNet layer result.

The upstream ResNet 1×1 im2col copy now has an exact view path. Merlin already had a structural `im2col_identity_view` pass, but it searched only `linalg.generic` contractions; QDQ preparation had converted these to named `linalg.matmul`, so the pass silently found zero. The matcher now inspects both forms. On the exact ResNet capsule, it proves **33 identity 1×1 windows**, declines **17 nonidentity windows**, and eliminates materialization of **7,200,256 int8 gathered elements**. This is a semantic IR rewrite, not a cycle measurement: the nominal copied tensor volume is 7.2 MB, while actual memory traffic depends on later allocation and cache behavior. The OOT `upstream_quant.py` preparation path enables the proven view rewrite by default after integer contraction preparation; it re-parses and verifies the emitted mixed-precision IR. A fresh catalog from that exact prepared source still covers **54/54 contractions → 21 kernels**, passes its object no-FSM audit, and Merlin's real rewrite/shim binding routes all **54** calls to those **21** symbols with exact source SHA. The remaining 3×3/strided windows need direct gather scheduling or a separate proven transform before Jack-level performance is plausible.

For TinyLlama, the exact prepared catalog's eight shapes and occurrence counts imply **64,741,408 padded array issue cycles** across all 200 contractions. The 44 occurrences of 8×5632×2048 account for 31,719,424 of those cycles; 22 occurrences of 8×2048×5632 account for 15,859,712; and 44 occurrences of 8×2048×2048 account for 11,534,336. Together these three projection shapes are **91.3% of that floor**. This is an array issue bound only; it excludes DMA, CPU dispatch, host graph work and FireSim stalls. For the real 8×5632×2048 cached-A candidate, A MVIN commands fall 768→128 but all primitive commands fall only **136,288→135,648 (0.47%)**. For 8×32000×2048 they fall 4,096→128, but all commands fall **774,096→770,128 (0.51%)**. The synthetic 8×512×256 case saved 6.6% of commands and 15.7% of GSIM cycles. Its measured speedup must not be projected onto the exact wide TinyLlama layers without a comparable run. The next TinyLlama performance work should target the repeated projection shapes and their row underutilization/weight movement, with correctness-preserving token or independent-batch packing only where the model dataflow proves independence.

## Device compilation and model coverage

The golden xDSL `gemmini.*` module is the sole source of device commands. `golden_device_lower.py` translates verified primitive operations to RV64 RoCC inline asm and rejects unsupported ones. `golden_device_compile.py` records target IR, LLVM dialect IR, LLVM IR, object and compiler hashes. `no_fsm_audit.py` scans all executable RISC-V ELF sections after the final link because a library can add forbidden instructions. This division is essential for automatically generated code: legality belongs at typed IR, command encoding at one lowering boundary, and the zero-FSM claim at the final binary.

`contraction_patterns.py` recognizes exact 2D or batched 3D i8×i8→i32 contraction semantics from maps, iterators, integer body, dimensions and zero accumulator. `golden_device_catalog.py` compiles one specialization per distinct legal schedule with stable symbols and source operation bindings. Current prepared models yield **54/54 ResNet → 21 unique kernels**, **367/367 SmolVLA → 26**, and **200/200 TinyLlama → 8**. All three objects pass the no-FSM audit. A selected TinyLlama symbol from the full catalog was linked into an ELF and passed the complete numeric oracle, output guard and ELF audit at 11,683 GSIM cycles with the original per-batch setup path. Catalog coverage means recognized contractions have code; it does **not** mean the graph's memory, layouts, epilogues and host operations are compiled.

The isolated Merlin integration branch `golden/device-catalog-integration` adds a generic external-catalog bridge to its whole-model offload path. The catalog now declares dense row-major `(lhs, rhs, out)` pointers, edge handling inside the kernel and one call for a whole batch. Merlin validates the catalog object hash, exact prepared-model source hash, unique `prov.region_id`, tensor types, shapes and dtypes before compiling a memref adapter. A compiled host ABI test passes for rank-2 and rank-3 with nonzero offsets and numerical output; stale source, object and operation bindings are refused. A cross-parser experiment found that Merlin and OOT operation ordinals diverge after the first few contractions on byte-identical IR because they parse nested operations differently; ordinals remain diagnostics, not link identities. On the exact prepared integer sources, Merlin's real rewrite and RV64 shim compilation routed and bound **54 ResNet contractions to 21 catalog kernels, 367 SmolVLA to 26, and 200 TinyLlama to 8**. Merlin's broader observer sees 24 additional SmolVLA candidates that the exact xDSL matcher rejects; source-region selection leaves those 24 on the host. The new `catalog_builder` callback runs after Merlin preparation and before device rewrite, so the device compiler receives the final prepared bytes; a TinyLlama callback smoke run built and bound all 200 calls to 8 kernels. The shim symbol check exposed a compiler-emitted `memset` from its all-zero error return. Explicit field initialization plus the bare-metal `-ffreestanding -fno-builtin` flags removed that unresolved dependency, and all three RV64 catalog/shim object pairs pass symbol binding and object-level no-FSM checks. Merlin now also accepts a target-supplied final-ELF audit callback; the OOT callback passes an audited probe ELF and rejects the archived reference ELF's six unused LOOP instructions. This is an integration seam, not a whole-model correctness result. The host path still needs a full-model memory, arithmetic and accuracy gate followed by a final linked-ELF no-FSM audit on an actual model image.

The old whole-model emitter still expands host tensor operations into straight-line SSA and declines the prepared ResNet at about 470M hypothetical scalar element evaluations versus a 400K code-size budget. Tensor views and allocations account for a large share of that estimate but have little or no runtime cost. Automatic lowering needs looped/fused host code and view/alias propagation rather than increasing the unroll budget. It also needs model memory planning, weight binding, quantized epilogue validation and a whole-model correctness oracle before any device catalog can be called a full model.

## Generalizable performance abstractions to add

1. **Physical tensor layout and alias analysis.** Record logical axes, DRAM strides, packing, padding and view aliases separately. Prepared ResNet `conv_0` is currently represented as `64×147 · 147×12544` from OIHW weights and a materialized NCHW im2col tensor. Jack's fast path uses spatial output rows and channel columns with direct resident-input gather. A generic matmul matcher alone preserves the wrong memory flow. The exact 1×1 alias proof now removes 33 identity gathers; next propagate those aliases through allocation and choose the spatial-row layout plus 3×3 direct gathers.
2. **Convolution window schedule.** Represent stem K packing, whole-output-row 3×3 tiling, shifted full-row MVIN, input residency, pool-on-store, B stationarity, accumulator ownership, and overlapped next-load/previous-store as typed schedule choices. The reference uses 43,904 stem computes versus a 31,360 ideal and 423,360 3×3 stride-one computes versus a 392,832 ideal. Those dynamic counts are stronger targets for our own command stream than an uncalibrated cycle formula.
3. **Multi-lane graph compiler.** Keep exact integer contractions on Gemmini and generate scalable CPU/RVV loops for QDQ, softmax, normalization, indexing and casts. Fuse adjacent layout-only or elementwise operations and avoid materializing tensors that are aliases. Preserve quantization and model accuracy; the optional ResNet pre-gather quantization changes integer arithmetic and has not passed a whole-model oracle.
4. **Device schedule selection with measured feedback.** The current tuner enforces scratchpad/accumulator legality and counts compute, RoCC and DMA requests. Its byte floors are geometric; they are not calibrated FireSim predictions. Store per-shape, per-layout FireSim or GSIM measurements in capsules keyed by object/bitstream/engine hash, and choose among legal schedules using actual observations. Include negative results such as small-shape cache and wide-store regressions.
   The typed `gemmini.compute` operation now accepts a dynamic i64 scratchpad-row operand with a declared maximum and reserved scratchpad extent. Lowering packs that row with the fixed tile dimensions into the primitive RoCC command, so the cached-A K sweep uses an ordinary CPU loop. A same-shape numerical GSIM run preserved the speedup within seven cycles and reduced the TinyLlama catalog object from **953,672 to 31,856 bytes** (LLVM IR 11.27 MB to about 0.45 MB). The handwritten builder constructs this operand from a fixed-bound loop, but the current verifier checks the declared range and operand type, **not a general SSA range proof**. Automatic scheduling must prove the loop bounds and expression range before emitting this form; a false range declaration can make an invalid scratchpad access. Exact 8×5632×2048 and 8×32000×2048 TinyLlama layer cycle gains are still unmeasured.
5. **Model-wide AOT compilation and link audit.** Deduplicate specializations, assign stable symbols, compile once per target/config/toolchain, bind all pointer arguments through the model allocator, then audit the linked ELF. A device object audit alone is insufficient. The catalog and Merlin bridge now emit object/compiler and source-to-entry binding receipts; whole-model execution and final-image grading remain open.
   The ABI must be a declared catalog property: dense row-major `(lhs, rhs, out)` with internal edge handling and one whole-batch call differs from the older padded `(weight, lhs, out)` capsule contract. Bind by final prepared source hash, unique provenance region and exact tensor types, not shape or parser-dependent operation ordinal alone. Treat the final preparation output as the device compiler's input artifact so a later model rewrite cannot silently invalidate the catalog.
6. **Section performance ledger.** Report exact dynamic compute/RoCC counts and comparable FireSim cycles for stem, 1×1 ReLU, 1×1 no-ReLU, downsample, 3×3 stride-one, 3×3 stride-two, residual, GAP, FC, attention GEMM, and scalar nonlinearities. The ZIP lacks per-layer FireSim cycles, so ResNet class cycle targets in the analysis are estimates; do not present them as measured reference numbers.

## Infrastructure observations

FireSim queue job 1670 failed before simulation because its requested Chipyard hardware key was absent. Job 1673 used our own stock worktree and the existing stock hardware key but timed out in `INFRASETUP` after its leading kill exceeded 90 seconds; the ELF never booted. A nearby unrelated job 1674 also failed during setup. The useful infrastructure change is an explicit queue preflight that validates the stock bitstream/hwdb key and simulator slot readiness before accepting a performance job, and records setup failures separately from model timeouts. Until an ELF boots, these jobs provide no cycle evidence. We should retry the pinned stock queue once setup is healthy, first with a small auditable probe, then per-group and whole-model payloads.

The queue later showed successful jobs 1692–1708. We submitted low-priority job **1709** with the audited configuration-once TinyLlama attention ELF on our own stock Chipyard worktree; it was queued when this log entry was written. `golden_firesim_preflight.py` now verifies the linked ELF's no-FSM audit, workload/bootbinary, HWDB key and exact stock bitstream SHA, then emits a one-entry HWDB artifact for future queue jobs. The test preflight reproduced the archived stock bitstream SHA `a9a190b9…d4eca1` and the TinyLlama ELF SHA `eaf8d350…648d3d`. A queue admission still needs slot readiness; this tool cannot repair a hung FireSim setup process.

After preflight, we submitted stock-config low-priority **job 1710** for the numerically checked ResNet-like 1×1 M3136 N64 K64 ELF and **job 1711** for the exact upstream SmolVLA `matmul_384` B15 M50 N64 K113 ELF. Both submissions use the immutable one-entry HWDB artifact (`sha256 5946d30d…7c126`) and remain probes, not full-model measurements. Queue status and UART output must be checked before recording cycles.

Jobs **1709, 1710 and 1711 all timed out after about 1080 seconds in `INFRASETUP`**. Their queue logs show staging, leading kill, `infrasetup` stopping after `[localhost] Checking if host instance is up...`, then teardown; none reached workload execution. The host-up path calls Fabric `run("uname -a")`. A direct FireSim-venv Fabric call as `agustin` completed normally. The active queue daemon is owned by `jack` (UID 2630), while these jobs are owned by `agustin` (UID 2621). The queue forwards `HOME=/home/agustin`, and FireSim selects `~/firesim.pem` as its SSH key; that file is mode 0400 and owned by `agustin`, so the daemon's `jack` process cannot read it. FireSim's 10 connection attempts at 100 seconds each explain the setup stall. This is a concrete cross-user queue execution bug, not evidence of a slow kernel or FPGA. The queue service must execute each job under the submitter's OS identity, or deliberately use a service-owned SSH identity and matching HOME/user for its remote work; forwarding a submitter HOME into another UID is insufficient. A fast admission check for effective UID/key readability would save the 18-minute timeout. Do not change private key permissions to work around this.

`golden_firesim_preflight.py --check-queue-daemon` now refuses the current UID mismatch before preparing a job. Its check reports `daemon UID 2630 differs from submitter UID 2621` on this host. This is a local admission guard; the shared `/opt/firesim-queue` service was not modified, and its proper cross-user credential handling remains an infrastructure task.

`firesim-queue tail 1709 --uartlog` misleadingly returned an older q1451 UART path (`2026-10-01--22-52-03-…q1451`) with unrelated whole-model measurements. Do not attribute that UART to 1709. Queue tooling should associate UART/result paths with the requested job ID and return “no UART” while the job has not produced one; setup timeout should preserve the last subprocess and host-slot diagnostics. Earlier jobs 1692–1708 did complete while the queue daemon had a usable identity; their success does not validate the current cross-user daemon setup.

## Nonlinear host pass experiment

The existing Merlin registry's `softmax_int`, `gelu_int`, `silu_int`, and `rsqrt_int` passes can now be requested through `upstream_quant.py --integer-nonlinears`. On the raw SmolVLA capsule, a complete parse/verify produced 44 softmax, 12 GELU, 33 SiLU and 91 reciprocal-square-root rewrites, with the same 367 exact mesh contractions. The transformed IR contains zero `math.exp` and zero `math.erf` operations versus 77 and 12 before. It still contains 169 textual `f64` occurrences in separate small scalar/time-encoding regions. The pass changes arithmetic and has **no full-policy accuracy or FireSim result**. Treat it as a candidate for host-lane optimization, not as a golden schedule choice. The automatic lowering needs a per-op semantic/accuracy gate and must measure total policy cycles, since removing libm calls alone does not prove a net model speedup.

The same optional path on TinyLlama parsed/verified 22 softmax, 22 SiLU and 45 reciprocal-square-root rewrites while retaining all 200 exact contractions. `math.exp` fell from 44 textual occurrences to zero; neither IR had f64. Accuracy and model-cycle effects remain unmeasured.

## 2026-10-05: upstream squares, host compilation, queue recovery

- Upstreamed model2MLIR scalar-square lowering to `main` commit `3d5acd3`: `pow(x,2)` emits `arith.mulf`, avoiding retained `powf` under freestanding compiler flags. Five tests pass, including float32/64 signed zero, infinity and NaN semantics. Existing direct/grouped/padded-convolution frontend changes are already present on upstream main. No full-model speedup measured yet.
- Merlin integration now preserves the chosen ISA/ABI for the GCC runtime harness as well as the LLVM model/device objects (`0db0e468e`); otherwise an RV64GC model could link RVV runtime code. Attributed `linalg.yield` uses generic syntax (`16b471d68`) because its custom xDSL syntax failed the upstream parser on the complete ResNet capture. A real upstream parsing regression passes. Prepared transforms run before device catalog source binding (`936fee86c`), and named integer matmuls retain the exact identity-im2col view optimization (`83fe42778`).
- Complete ResNet host build is running from the owned `resnet50-current-capture` bundle, with 54 contractions routed to21 signatures. TinyLlama and SmolVLA full host LLVM lowering completed in the language-model worktree; native object compilation remains in progress. These are compilation milestones, not whole-model correctness or performance evidence.
- Queue SSH setup recovers when submissions omit HOME/USER/LOGNAME overrides and our stock checkout uses default Fabric authentication. Job1717 booted a stale ELF because FireSim changed cwd away from the queue overlay; canceled and rejected as evidence. Our checkout now honors the overlay and the submission checks resolved workload hashes. Corrected jobs1721/1722/1723/1724 are ResNet GEMM, real direct convolution, Tiny attention and Smol attention respectively. Measurements require the actual simulator ELF hash and job-private UART; no cycles accepted yet.

### First verified FireSim section and full ResNet execution

Job1725 completed on the exact stock bitstream with105,258kernel cycles forM3136N64K64 and numericPASS. Submitted/stagedELF hashes, HWDB, bitstream and job-privateUART are bound in `perf_records/resnet_pointwise_3136x64x64_firesim.json`. The padded array issue floor is50,176cycles, so this schedule still has substantial overlap/memory overhead to remove; it does not establish full-model22M performance.

The whole ResNet ELF SHA18823d02… additionally passed actual Gemmini Spike execution: all1000outputs bit-exact, rank mismatch0. Its2,401,323,546Spike counter is retired-instruction proxy, not FireSim cycles. Whole-model FireSim baseline job1730 is queued. This coherent capture uses random-initialized weights/synthetic input; it proves compiler semantics for that capture, not pretrained model quality. Its frontend uses concat-of-tap-slices and NHWC GEMM orientation, unlike the older capsule's gather/reshape form; identity-im2col view count0 is expected because1x1already uses views after an actual NCHW→NHWC transpose.

Uniform-input scalarization initially crashed native `linalg-specialize-generic-ops` (SIGSEGV) with bare scalar operands. Retaining rank-zero tensors fixes native compatibility and still eliminates giant constant splats. Corrected ResNet builds successfully; LLVM text shrank575MiB→29MiB. Native full-output validation of this candidate is running; no runtime performance gain claimed yet.

### Exact whole-model fusion candidate

With corrected rank-zero uniform tensors plus existing `MERLIN_GENERALIZE_BEFORE_FUSE=1` and `MERLIN_FUSE_POST=1`, the full ResNet host reference and actual Gemmini Spike execution both remain bit-exact on all1000 outputs. Final ELF no-FSM audit passes (SHAa83f52ec…). Spike's retired-instruction counter falls2,401,323,546→1,908,630,558 (20.52% lower), with rank mismatches0. This is not a FireSim cycle result. The remaining host work is still orders above the desired full-model overhead; persistent layouts and device epilogue fusion remain essential. The separate uniform-only candidate also passes full native-host output equality; fused candidate queued for hardware A/B through queue agent.

### Mixed whole-model direct convolution

The current ResNet source now compiles with16proven direct3x3 convolutions and38dense GEMMs. Native scalar-reference execution through the identical ABI and actual Gemmini Spike execution both reproduce all1000outputs bit-exactly; final ELF SHA4ab852cd… passes static zero-FSM audit. Mixedcatalog callbacks compile the direct bundle before exact-source dense binding, relocatably link both objects, and bind component/object hashes. Generic external function declarations preserve bufferization access attributes across parser/printer boundaries, avoiding accidental defensive copies.

Spike retired instructions fall1,908,630,558→1,193,349,750 compared with the preceding fused variant (37.48% fewer); still not a FireSim cycle result. The target is not met. Kernel1727 is verified at961,350FireSim cycles; Tiny attention1728 at9,592cycles. Full-model hardware A/B remains queued.

model2MLIR main now includes `fc4c4a3`, an explicit optional static-W8A8 `extra_args.weight_granularity` choice: per_channel(default) or per_tensor, with fresh calibration and recorded policy. Four focused tests pass including both captured qparam forms and invalid-policy refusal. This enables target-compatible capture policies; it does not retroactively change the current captures or prove a model accuracy improvement. Current retained ResNet immediate dequant multipliers are scalar already; sequential float rounding, bias/residual handling and layout propagation remain the next exact-fusion challenges.

## 2026-10-05: explicit RV64 round-to-even experiment

Added optional target-owned scalar f32 rounding with explicit RNE, signed-zero
preservation and preserved finite-input exception flags. All 24,199 probe inputs
pass across five rounding modes; final ELF passes the zero-FSM audit. Whole
ResNet execution on Spike matches all 1,000 captured outputs bitwise. Its retired
instruction proxy is 1,195,079,840 versus 1,193,349,750 without this rewrite
(+0.145%). This is not a FireSim cycle measurement. Leave the option disabled;
calling a helper before graph fusion can alter optimization opportunities.
No additional queue run is justified yet. Evidence: perf_records/resnet_roundeven_spike.json.

Full original ResNet baseline FireSim job1730 now reports 5,680,463,426 forward
cycles and all 1,000 outputs match exactly. This is far above the 22M target;
compiled contraction coverage alone does not establish good model performance.
Queued fused and direct variants will measure their full graph effects. Next
priority is exact device epilogues and eliminating host layout materialization.

Validated model2MLIR changes pushed to upstream main: 3d5acd3 (scalar square
lowering), fc4c4a3 (explicit per-tensor weight quantization policy, default remains
per-channel). Verified remote main at fc4c4a3 after push.

## 2026-10-05: verified hardware and complete language capture gates

FireSim baseline1730 measures5,680,463,426 whole-forward cycles. Rank-zero uniform
inputs plus native elementwise fusion1731 measures4,516,405,461cycles (-20.49%);
both reproduce all1,000 ResNet outputs exactly. Hardware config, bitstream,
staged/source ELF hashes and job-private UART are sealed in the corresponding
`perf_records/resnet_{baseline,fused}_firesim.json`. These remain far above22M.

Exact27-layer epilogue fusion plus4remaining direct convolutions and23dense
GEMMs preserves every ResNet output in native and actual Gemmini Spike. Reusing
all external buffer-access declarations is essential: dropping their arg_attrs
introduced27 defensive activation copies. Corrected ELF43697418… has20 LLVM
memcpy sites versus47 in the defective composition. Actual Spike1,018,739,739
retired instructions (-14.6% against1,193,349,750); FireSim1743 queued. This
instruction counter is not a hardware timing. Source-bound proof refuses the
five unrepresentable unary chains and retains residual/pool/float-return paths.

Pointwise banked prefetch bm8 hardware1733 now measures92,101kernel cycles versus
105,258baseline (-12.50%), sameM3136N64K64 i8 bias/scale/ReLU program and stock
bitstream. The evidence applies to this fused int8 store contract; current i32
catalog timings cannot inherit it without measurement. Promote schedule choices
only after source applicability, full numerical validation and hardware A/B.

The optional permutation-propagation pass preserves scalar instruction order,
broadcast affine maps, static slices and padding. Current full-model candidate
passes native and actual Gemmini output equality but regresses Spike retired
instructions1,193,349,750→1,340,175,241 (+12.3%). It remains disabled. Permutation
cancellation alone is not a sufficient cost rule when producer fanout, reduction
layout and native fusion interact. Default-on promotion needs model-level timing.

Explicit RV64 roundeven selected AFTER model optimization by final-link wrapping
retains identical model.o and passes every ResNet output on actual Gemmini Spike.
It lowers retired instructions1,193,349,750→1,059,531,659 (-11.21%), unlike the
earlier before-fusion rewrite. `golden_host_math.relink_roundeven` reproduces exact
ELF b945ed41… from bound base objects and records compiler/linker commands plus
final no-FSM audit. Hardware effect remains unmeasured. Device compilation should
expose target-owned host runtime support at linking or after optimization, so it
does not insert opaque calls before linalg fusion.

Fresh atomic TinyLlama capture (full22-layer cached checkpoint, exact retained
8input IDs, same-instance static-int8 reference) now passes all256,000 logits
through native integer stand-ins and ACTUAL Gemmini Spike. Target and native
outputs are bit-identical; Torch golden relativeL2 is2.1918433e-7, maxabs9.536743e-6.
No-FSM ELFdb16fcb3…; Spike6,838,149,469retired instructions is not FireSim timing.
Older mixed bundles and floating-dequantized contraction substitutes failed and
are excluded. Fresh pretrained Smol capture with6retained inputs and same-instance
1,600-output reference is compiling; no full-policy numeric pass yet.

Merlin6c181eb39 precomputes exact stored-weight transposes after integer lowering:
155 Tiny permutations /1,034,420,224int8 elements disappear from inference, all
256,000 outputs remain bit-identical to the native baseline. Static constant
malloc-site sum drops1,080,659,050→48,261,314bytes; this is static accounting, not
an allocation trace. The existing weight-hoist ABI records packed tensors; source
weight identity, axes, static shape and dtype prove each substitution. Optimized
Tiny target is compiling. Dead weight argument pruning and activation lifetime
reuse remain useful to reduce image/arena sizes further.

model2MLIR8893a37 is now on upstream main: registered-buffer exports retain int64,
int8, bool and float64 values/dtypes instead of casting everything tofloat32.
Eight capture-receipt tests pass, including large integer/float64 precision and
bf16 fallback. This addresses correctness and inflated extra.npz artifacts.

Large language output validation must cover every logit. Merlin output_dump_cap
is now configurable (default4096); Tiny full-output Spike uses256000. Full decimal
HTIF dumps are costly for the shared hardware queue. A compact optional SHA256
after the timing window can compare every raw output byte to the same-artifact
validated target/native reference, while retaining the separate Torch tolerance
gate. Its implementation/target verification is pending.

### Latest complete gates and queue admissions

Packed7×7 RGB stem uses14 K blocks (seven21-byte kw×Cin rows, each16+5) rather
than49 Cin-only blocks. It preserves the original147×64 weight matrix and exact
quantized input halo. Whole ResNet stem1/fused27/direct4/dense22 passes native and
actual Gemmini Spike all1,000 outputs exactly; finalnoFSM ELFf3a432e5….
Spike908,532,797retired instructions versus1,018,739,739 without packedstem.
Hardwarepool remains disabled until contiguous spatial accumulator rows and
pooled-store bounds are proven; all64 scalar source-scale proofs already accept.

Final-link RNE on the corrected fused27 model also passes every actual Gemmini
output:922,268,279retired instructions versus1,018,739,739. The resulting
ELF55f4c0cc… reuses unchanged optimized model.o; no FireSim result claimed.

Pointwise1734 improves further to53,624kernel cycles, all200,704outputs+guard PASS.
This is49.05% below105,258baseline and6.87% above50,176 compute floor. Exact winner
uses bankedprefetch bm1 and reuse_b=False. Captured-scale/no-bias pointwise A/B
1745/1746 is queued; the earlier .125-scale/bias benchmark cannot directly establish
performance for our current exact-fusion contract.

Tiny precomputed-weight ACTUAL Gemmini gate now passes all256,000 logits, with
same Torch error as the baseline. Spike instruction proxy falls6,838,149,469→
881,448,291 (7.76×). Full-dump ELF932a8ab5… passes noFSM. Compact-output ELF
419d1881… also passed actual Gemmini: all1,024,000 output bytes have canonical
SHA256 ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3.
The byte hash is checked against the same artifact's validated native/Spike
outputs; the separate Torch tolerance result remains recorded. FireSim1747 is
queued on the pinned stock bitstream. No Tiny hardware timing yet.

Fresh pretrained Smol full host output fails both captured-scope (.09656 relative
L2) and broader integer candidate (.09876). Neither is released to hardware.
A minimal dynamic-int8 Linear capture matches exact integer host execution;
therefore the full mismatch needs region/operator localization, currently the
vision/prefix boundary. Atomic capture identity and preserved buffer dtypes
exclude the previously observed mixed-bundle defect. Targetcompilation is stopped
on failed numeric gates so queue/compiler work focuses on admissible programs.

### Half-precision frontend arithmetic fixed upstream

Vision-prefix localization exposed two model2MLIR bugs: bf16/f16 LayerNorm
and softmax reduced and normalized in the input dtype. PyTorch uses f32
intermediates. Wide bf16 reductions can stop accumulating while inputs remain
nonzero, so relaxing whole-model tolerances cannot repair this defect.

Upstream main commits `a472fef` (LayerNorm) and `d82f4aa` (softmax) widen
intermediates and narrow only the final output. The compiled 2×768 LayerNorm
reproducer changes relative L2 error1.227→0, bit-exact against Torch. The
compiled uniform1024-way softmax changes row sum4→1, likewise bit-exact.
The earlier broad absolute tolerance admitted the wrong small probabilities;
the operator gate now requires exact output or its explicit strict tolerance.
Fifty focused and related tests pass, including mixed f32 affine parameters,
multiple normalized axes, nonfinal softmax axes and named-op expansion.
Evidence: `perf_records/model2mlir_half_precision_fixes.json`.

Fresh full SmolVLA capture is being revalidated with these fixes. Its checkpoint,
input, buffer and framework-golden hashes match the failing fresh capture;
the numerical comparison therefore isolates frontend lowering changes. No
full-policy numerical or hardware-performance result is claimed yet.

Full ResNet direct16/dense38 FireSim job1737 now passes all1000 outputs exactly
at3,529,465,283forward cycles,21.85% below4,516,405,461 for1731. Pinned staged
ELF/bitstream identity is verified. This remains far above22M; boundary-profile
job1741 and later fused/stem variants are required to attribute the remaining
cost. Evidence: `perf_records/resnet_direct_firesim1737.json`.

### Closed recipe capture and complete residual-domain proof

Merlin7c072660a fixes container-owned functional quantization. The module
inventory lists leaves, so the old ownership check refused functional sums in
Bottleneck containers even when the operation planner admitted both operands.
The regression reproduced this for nested ordinary and in-place adds; root adds
passed. Known containers now use the existing per-operation plan. Explicitly
unsupported leaves, missing ownership metadata and float-returned open sums
remain refused. Five ownership and thirteen related quantization tests pass.

Recapturing the same random checkpoint39215a3f… and inputf6659bac… with the
unchanged recipe6497b3dd… annotates54 contractions,16 sums and1 mean, with no
placement refusals. Its framework golden changes because the originally selected
recipe now actually quantizes the sums/mean. This is a fresh same-instance
reference, not a bit-identical replacement for the earlier capture. Forty-six
unary epilogues admit exact readout fusion, including12 direct convolutions;
six fail the full float-transition proof. Pooled-stem whole native and actual
Gemmini Spike outputs match all1,000 fresh goldens exactly. Final no-FSM
ELFf25a8a17…;821,610,403 retired instructions, not hardware cycles. This random,
synthetic-input gate establishes compiler semantics rather than pretrained
accuracy. Evidence: `perf_records/resnet_closed_recipe_capture.json`.

The new residual proof compares every65,536 signed-i8 operand pair using the
original f32 arithmetic order against two hardware scaled-load rounds followed
by integer addition and scaled readout. All16 fresh residual sites fail exact
matching; each candidate has a maximum error of1 output step and retains a
counterexample. Seven tests cover exact identity, double-rounding differences,
a real structural Q/DQ fixture and malformed shape/clamp/body refusals. No
approximate rewrite is enabled. This finite domain is a reusable proof boundary
for the automatic optimizer; a bounded candidate needs an explicit numerical
policy and a separate full-model quality gate.

### Further upstream half arithmetic fixes

model2MLIR main e9aa908 preserves GELU approximate=none/tanh and computes half
GELU with f32 intermediates before narrowing. Previously all twelve Smol vision
GELUs used the erf form regardless of the selected tanh mode. Compiled BF16 and
FP16 none/tanh probes now match Torch exactly. Mainb0979bb similarly keeps
half convolution reduction and bias addition in f32 until one final narrowing.
The bf16 biased-convolution reproducer changes maxabs0.0078125→0; direct BF16
and FP16 probes also match Torch exactly. Seventy-four focused and related
frontend tests pass. Same-checkpoint/input/buffer/golden fresh Smol captures
isolate these lowering fixes; the three-fix full Smol outputs still fail their
numerical gate, and the four-fix gate is running. Operator correctness alone is
not a full-model release. Updated evidence:
`perf_records/model2mlir_half_precision_fixes.json`.

### Infrastructure and abstractions still required

Verified FireSim1741 attributes98.25% of ResNet forward cycles to host gaps.
The current graph inserts physical tensor permutations, float readout work and
residual conversions around device calls. Device compilation needs to carry
physical layout and numerical policy across operation boundaries, preserve
stored-weight transforms outside inference and account for every remaining host
region. An exact source graph can refuse a representationally limited readout;
a bounded transformation must declare its error in machine-readable metadata
and validate full-model quality independently. Bias initialization also needs
its own accumulator contract: Gemmini can preload D even when its store/readout
facet has no bias-add stage. Inferring bias capability from the readout alone
leaves unnecessary host arithmetic.

Spatial-flat direct convolution now derives output/channel blocks from actual
accumulator capacity and gathers across pixel rows for small feature maps. Its
late7×7 C512 wide-A kernel passes every25,088 output and guard at939,677 GSIM
kernel cycles; cross-simulator comparisons to FireSim are provisional. The
whole narrow-flat candidate passes native/Spike. Hardware job1757 measures the
best standalone late-conv variant before any blanket schedule promotion.

### Explicit bounded readout experiment

An opt-in `captured_requant_bundle --max-output-lsb=1` now admits the six
otherwise-refused zero-bias unary epilogues. Exact mode remains the default.
`prove_scale_bound` compares the union of every source/target monotone integer
output transition and the domain endpoints; the step functions are constant
between these points. This computes the exact worst error across all possible
accumulators rather than sampling calibration outputs. Each of the six has
maximum1 output step, with a reproducible accumulator witness. Source shapes,
qparams, bias bytes, original arithmetic order, rewritten bytes and compiled
objects remain bound. The complete new bundle has52 routes/16 direct and a
zero-FSM object audit. Twenty-four proof/rewrite regression tests pass.

The selected local limit and proven error are typed declaration attributes and
manifest fields, verified again after preparation. Derived capture receipts
state the selected numerical policy and preserve the original framework
reference. This candidate is not exact source lowering and has no full-model
quality or hardware release yet. Its next gate composes the explicit residual
candidate and compares to the unchanged golden; actual target outputs must
match the candidate's native implementation independently. Evidence:
`perf_records/resnet_bounded_unary_candidate.json`.
