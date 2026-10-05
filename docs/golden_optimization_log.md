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

### Scalar precision and measured model results

model2MLIR mainba77e6e preserves Python scalar precision for half mul/div.
Smol language embeddings multiply bysqrt(960); rounding this coefficient tobf16
before multiplication caused relativeL2.00182. The executed corrected probe
matches Torch exactly. Expanded testing rejected an initially broader opmath
change: Torch add/sub first narrow their scalar, whereas mul/div retain a f32
coefficient. The published change is limited accordingly. Actual compiled
left/right add/sub/mul/div forms match all8192 outputs for each ofbf16 andfp16.
Ninety-six related frontend tests pass; rejected all-operator promotion was not
pushed. The full Smol gate still needs correction of fused attention precision.

Tiny FireSim1747 now measures1,800,267,524 forward cycles for the full22-layer
prepacked model. Its complete1,024,000-byte output digest matches independently
validated native and actual Spike output, and Torch comparison retains relative
L2 2.1918433e-7/maxabs9.536743e-6. Identity and timing are recorded in
`perf_records/tiny_prepacked_firesim1747.json`. This is an absolute whole-model
hardware result, not a speedup versus an unmeasured baseline or a maximum claim.

Pooled-stem ResNet FireSim1750 measures2,663,212,150 forward cycles, all1000
outputs bit-exact the earlier original reference;15.35% below fused27 job1743,
53.12% below initial job1730. Record:
`perf_records/resnet_pooled_stem_firesim1750.json`. The corrected closed-recipe
and explicit bounded candidates have separate source/golden identities and are
not conflated with this result.

Complete-row convolution bands derive retained pixel blocks from accumulator
capacity. H56 holds4 rows/band;H28 holds8. All16 ResNet3×3 convolutions now need
487,872 mesh compute commands/7,805,952 padded issue-floor cycles. Measured
GSIM full-range kernel gates617,005(C64),628,112(C128),674,159(C256) and
769,976(C512) include every output and guard; their sum is not a whole-model
FireSim prediction. Entire exact original ResNet with banded/flat kernels passes
native and actual Spike all1000 outputs, noFSM. These reusable physical-layout,
capacity and bank-placement rules are the device schedules to compare with
later automatic search, while host-region closure remains separate work.


### Upstream attention precision and new exact closures

model2MLIR main `050009e` retains f32 intermediates throughout half SDPA in
both direct xDSL and Torch-export decomposition, including QK scaling, mask,
softmax and PV reduction. Executed bf16/f16 models exactly match Torch's math
attention backend; its fused CPU backend has different rounding. All 108
focused and related tests pass. The unchanged full Smol golden still fails its
quality gate, and no full Smol hardware candidate is admitted.

The exact unary readout solver searches every positive finite f32 scale by
intersecting monotone integer-output constraints, then independently rechecks
all transitions. Four formerly refused regions accept adjacent store scales:
matmul4 (-1 ULP), matmul29 (+1), matmul43 (-1), matmul51 (-1). The exact closed
recipe bundle covers 50 readouts, 14 direct convolutions. Regions matmul25 and
matmul48 have contradictory bit intervals, proving that no zero-bias positive
f32 store scale alone represents their original arithmetic. The stem and final
FC are distinct boundaries. Source qparams and original golden remain unchanged.

The whole-model harness now binds the same trailing hoisted weight arguments
as the bare-metal runtime. The original ResNet removes 54 inference-time weight
transposes /25,502,912 i8 elements and passes native plus actual Gemmini Spike
all1,000 original output bits, rank0, final zero-FSM ELF audit. Spike retires
771,156,560 instructions; this is not a FireSim cycle measurement. A separate
closed-recipe 50-readout + exact residual CPU LUT + hoisted-weight candidate
passes native all1,000 original golden bits; target validation is still running.
Each residual lookup table contains all65,536 signed-i8 pairs and preserves the
source's ordered f32 QDQ semantics. Sixteen tables consume1MiB; this exact CPU
fallback is a measured optimization candidate, not a Gemmini residual add.

Large-N FireSim1763/1764 confirms wide B groups and cached A improve the
representative M8,N2048,K2048 GEMM from1,445,879 to925,141 cycles (36.02% fewer).
All16,384 outputs and guards pass, with staged ELF and stock bitstream identities
checked. The model-level impact remains to be measured. The late C512
convolution FireSim1758 passes at793,557 cycles for all25,088 outputs and guard.


### Exact wider integer residual arithmetic

Searching a ratio neighborhood with integer coefficients up to32767 finds an
exact representative for every one of the16 closed-recipe residuals. Independent
NumPy comparison verifies all65,536 signed-i8 pairs for each against the original
ordered f32 source arithmetic. The source scales and golden are unchanged.
Largest coefficients are2609/2180; smallest217/179. The earlier p,q≤127 refusal
remains valid for its restricted form. No optimality or broader infeasibility
claim follows from this neighborhood search.

A prospective primitive schedule decomposes each coefficient into signed-i8
127 chunks plus a remainder, loads the two source panels once, and accumulates
diagonal products before one scaled i8 readout. This avoids CPU lookup work.
The naive all16 residual mesh issue floor is5,193,216cycles/324,576commands;
actual scheduling, correctness and FireSim timing are still to be established.
Evidence and reproduction command: `perf_records/residual_wide_integer_candidates.json`.

Exact shared permutation proofs now commute through all16 elementwise residual
lookups. The whole graph drops36 live layout copies to2 (input and final mean),
retains all1,000 original golden bits in native/actualSpike, and passes final
noFSM. Raw-byte four-way lookup unrolling and shared-layout propagation bring
wholeSpike instructions from463,973,868 to254,701,066. These are retired
instruction results, not hardware cycles. FireSim1767 measures2,461,226,911 for
the earlier original-golden hoisted/banded variant; FireSim1769 is a separately
identified closed-recipe exact52-readout/LUT/nested-hoist baseline.


### Runtime copy closure and complete exact device residuals

Merlin5586d188d coalesces common contiguous memref suffixes after proving positive
address bounds and disjoint source/destination ranges. Overlap and negative
strides retain the original scalar order. Merlin96b9b8440 copies/clears aligned
bytes through alias-safe native words, with byte fallback for misalignment and
tails. The39 runtime tests cover strided copies, overlap, full halos/guards,
singletonaxes, every pointeralignment and length0..129. Structure, no-target-name
and no-regex gates pass. The exact17-padding microbenchmark falls165,257,880 to
1,985,883 actualSpikeinstructions (98.80% fewer); this isolates coalescing and
uses the standalone benchmark's existing libc.

The wider primitive residual kernel passes all65,536 source operand pairs plus
guards at170,847GSIMcycles and170,994stockFireSimcycles (job1773), about7.04%
over its159,744array issue floor. Complete exact52 +wide16 +shared permutations
+propagated layouts +nested hoist +new runtime matches all1,000 original
closed-recipe golden bits in native and actualSpike. FinalELF02d029dc…54700c
passes zeroFSM; Spike retires21,144,570instructions. FullFireSim1774 is queued;
21.1M is not a whole-model hardware cycle claim. Earlier exact52 CPU-LUT
baselineFireSim1769 is771,357,461cycles with the same closed-recipe golden.

A full22-layer Tiny candidate enables the verified large-N cached-A/wide-B
schedule and these runtime fixes, keeping its original155 contractions/5kernels
and155hoisted arrays. Its prepared hostLLVM changed under exact host fusion,
so object reuse was refused and the host model was freshly compiled. Full native
quality passes unchanged versusTorch (relativeL2 2.1918433e-7, maxabs9.536743e-6).
Complete actualSpike output-digest validation is running before queue submission.

### Verified complete ResNet acceleration and Tiny large-N admission

Stock FireSim1774 verifies68,385,997 forward cycles for the complete exact52
readout/wide16 residual/shared-layout/runtime build. All1,000 closed-recipe
golden outputs are bit-exact, with pinned stagedELF and stock bitstream. This is
91.13% fewer cycles (11.28x faster) than same-capture1769's771,357,461. The
22M objective remains unmet by3.11x; final-link device/host boundary profiling
will determine the next priorities. These gains combine several changes and
cannot be attributed to one pass alone.

Exact virtual padding removes16 direct-convolution CPU padding buffers using
source-proven zero-border primitive gathers. Stem padding remains. Combined
with wide16 residuals and the same runtime it matches all1,000 golden bits and
retires19,286,601Spike instructions versus21,144,570 padded (8.79% fewer).
Its separately pinned ELF288af46a... is queued as1775 for a hardware comparison.

The full Tiny large-N candidate finishes actualSpike and passes full256,000
output SHA256 verification against the independently Torch-validated native
output. It is bit-exact to the earlier full Tiny compiled baseline. Final
ELF436efef0...742ad5b has no forbidden or unknown Gemmini instructions. Spike
retires678,959,918 instructions; this is not a hardware timing claim. Original
Torch tolerance remainsatol0.03125/rtol0.02, relativeL22.1918433e-7 and
maximumabsoluteerror9.536743e-6. Evidence, host/compiler/runtime identities and
scope: `perf_records/tiny_large_n_runtime_spike.json`. Stock FireSim submission
is authorized after these gates.

### Verified virtual padding, Tiny hardware gain, and ResNet attribution

FireSim1775 verifies62,441,162 ResNet forward cycles,8.69% fewer than1774.
Every1,000 closed-recipe output words remains exact. FireSim1776 verifies the
full22-layer Tiny at1,402,210,517 forward cycles,22.11% fewer than1747's
1,800,267,524. Its full256,000-output digest matches the independently
Torch-gated native output and the prior compiled baseline. These are pinned
stock FireSimGemminiRocketConfig runs with final-ELF zeroFSM, not Spike timing.

ResNet final-link leaf profile1777 conserves all70 calls and its61,466,934
interior cycles:37,187,546 device and24,279,388 host. The largest host intervals
are16,825,464 before the stem and3,210,641 before the classifier. The profiled
61,467,502 harness cycles differ from unprofiled1775;1775 remains the baseline.
The selected mesh issue geometry is22,805,632 cycles before CPU/DMA work.
A source-proven restricted skip-domain experiment would lower this to
22,466,944, but has not been promoted. Matching22M requires additional schedule
and exact arithmetic improvements beyond removing host overhead.

### Exact host quantization and explicit scalar rounding

Merlin36d43b7ba follows proven uniform zero points and reciprocal scales through
captured quantizers. The existing default-off fusion now reaches the remaining
two ResNet quantizers.19 focused tests and the full native/actualSpike golden
gate pass; Spike retires14,254,689 instructions versus19,286,601 before fusion.
FireSim1781 is queued for an independent hardware comparison.

Target-owned late LLVM legalization recognizes the complete typed SSA chain,
refuses strict/constrained floating arithmetic, and replaces bounded RNE with
explicit RISC-V rounding. RNE-only retires12,271,325 instructions; combined
clamp+RNE retires11,355,701. Each matches all1,000 outputs natively and on Spike,
passes107,415 boundary/random comparisons across all five frm modes, and passes
final-ELF zeroFSM. NaNs have no invented output contract: the original fptosi
is poison for NaNs. FireSim1786 is queued for the combined candidate.

The source-bound dense bank schedule passes the same whole golden. A controlled
49x2048x512 GSIM comparison improves447,124 to367,749 cycles (17.75% fewer).
This is a contraction measurement; the whole-model hardware result is pending.
Receipts preserve the capture, prepared source, compiler, linked objects and ELF
identities for every candidate.

### Verified host gains and compilation identity (2026-10-05)

ResNet1781 verifies55,239,221 cycles with quantization fusion. Combined clamp/RNE
1786 verifies49,673,153 cycles,10.08% fewer. The channel-block64 final mean keeps
each channel's original H,W sum order and verifies48,780,534 in1789,1.80% fewer.
All1,000 source output bits match on native, Spike and stock hardware. Its higher
Spike instruction count confirms that instruction counts alone cannot predict
cache behavior. These are individual hardware observations, not repeated-run
distributions. The zero-FSM final-ELF policy remains enforced.

Tiny1782's hardware profile records219,195,608 device and1,180,631,315 host
interior cycles across155 calls. ISA-based host scheduling therefore targets
the measured dominant cost: Merlin12b15afab disables the default RVV host schedule
when the final ISA cannot execute floating vectors. Explicit RVV targets retain
their prior behavior. The full Tiny scalar host preserves every output and
verifies1,030,207,906 cycles in1788,26.53% fewer than1776 and42.77% fewer than1747.
This changes the host schedule; the five device kernels remain identical after
declared symbol renaming.

The scalar host plus89 exact quantizer fusions retires310,716,114 Spike
instructions. Adding89 proved combined clamp/RNE routes reaches271,019,239,
60.08% below the original678,959,918. Full native and actual Gemmini Spike match
all256,000 prior output bits; the original Torch gate is unchanged. Strongest
candidate1792 is queued. These instruction counts are not hardware timing.

Merlin9592b212a adds an explicit host LLVM transform between ordinary lowering
and object compilation. Target support owns typed source recognition and target
legalization; the generic backend verifies source immutability, output location,
and compilation, then includes the actual selected model.o in its normal build
identity before harness generation. OOT supplies the bounded RNE callback and
portable native oracle. This removes the attribution limitation of historical
late-relink candidates, whose base harness hashes are explicitly documented.

Merlin0c6259bf7 also offers explicit scalar/vector f32/f64 FMA intrinsics after
linalg lowering. Native cancellation, overflow, subnormal and signed-zero probes
match libm bitwise. The option is default off and does not change expf or other
math calls. It is not yet a full-model performance claim.

Future full ResNet queue builds use the existing full-output SHA256 validation
with a one-element dump cap. The complete1,000-output digest stays authoritative;
avoiding UART printing shortens post-timing queue occupancy. Profiling1777 spent
2.239B emulated cycles overall versus61.5M in the measured forward window.

### Frontend upstream and remaining SmolVLA numerical work

model2MLIR main includes050009e's six precision fixes and69c0370's explicit SDPA
scale/causal/options semantics. The latter passes28 focused tests with actual
native execution. Nonzero dropout, unsupported GQA and ambiguous dynamic options
refuse instead of silently changing semantics. Both pushes were reviewed against
their fetched main and performed without rewriting published history.

SmolVLA's compatible first vision layer passes the original tolerance on trusted
input, but full12-layer replacement plus balanced normalization still FAILS the
unchanged full-model gate (relativeL2 .02038219, maxabs .1113653). No whole-model
hardware candidate is admitted. Actual PyTorch profiling identifies BF16 patch
convolution as THNN slow convolution with a four-partial CPUBlas fallback on AVX2.
An executed C replay matches all786,432 original convolution BF16 values; serial
accumulation explains all30 initial embedding differences. Perturbing only those
30 values in the original Torch chain produces substantial later drift. An
explicit backend-compatible xDSL schedule is being tested; frontend default
f32-opmath correctness is preserved. Attention BF16-to-f32 accumulation uses a
different dispatch and cannot inherit this convolution schedule blindly.

### Strongest composed hardware and explicit activation option

ResNet1795 verifies47,020,321 forward cycles,3.61% fewer than1789, with the dense
separate-B-bank schedule composed with blocked64 mean and combined clamp/RNE.
All1,000 original output bytes are exact; normal build identity includes the
selected host object. Explicit scalar versus historical vector host selection
produces byte-identical LLVM, model.o and final ELF for this transformed graph,
so that duplicate was not queued. Compact full-output SHA reduces total emulated
cycles to364,867,232 versus~1.897B in1789; this queue occupancy change is separate
from timed forward arithmetic. Fresh profile1801 measures47,057,935 interior
cycles:35,450,518 device and11,607,417 host, with all70 primitive calls conserved.
The47,058,439 whole-forward profile adds38,118 observed cycles (0.081%) over1795;
instrumentation and placement effects keep1795 as the unprofiled control.
Dense kernels account for14,897,959, direct conv12,333,730, residual6,777,154,
and pooled stem1,441,675. Host intervals are largest beforestem5,238,180,
beforeclassifier2,159,097 and aftermatmul25/48 readouts1,957,501/963,250.
The selected issue floor is22,805,632; it does not predict achievable total time.
See `perf_records/resnet_current_leaf_profile_firesim1801.json` and
`perf_records/resnet_current_issue_geometry.json`.

Tiny1792 verifies866,822,103 cycles,15.86% fewer than1788 and51.85% fewer than1747.
All256,000 output bytes remain exact to the prior compiled model and pass the
original Torch gate. The explicit scalar activation polynomial replaces22 f32
activation expf chains, retaining every normalization expf call. Native and
actual Spike still reproduce every output bit on this capture; the arithmetic
approximation is not universally bit-exact. Spike218,423,000 instructions are
19.41% fewer than1792. FireSim1800 verifies791,638,514 forward cycles,8.67% fewer
than1792 and56.03% fewer than1747. Full final-ELF zeroFSM, complete output digest
and normal object-bound build identity pass. Receipts:
`perf_records/tiny_activation_poly_firesim1800.json` and
`perf_records/tiny_scalar_quant_rne_act_poly_spike.json`.

The needed generic abstraction is numerical rewrite selection independent of
host scheduling. Merlin0ef811692 exposes `approximate_transcendental_activation`
as a default-off scalar/vector f32 arithmetic option; the existing vectorized
option implies it and separately selects vectorization. Unsupported bf16/f64
types and normalization keep their existing lowering.48 focused tests pass,
including actual scalar native arithmetic and unchanged normalization/default
behavior. The fresh explicit-option build is byte-identical in finalELF,
model.o, target/native LLVM and device.o to the fully executed trial. Every
future model/input still requires its original accuracy gate.

Two concrete device hypotheses were rejected after actual GSIM execution:
transposed Tiny M8/N512/K2048 costs368,318 kernel cycles versus180,722 before
online transpose costs; direct-conv alternating B banks costs777,710 versus
775,788. Full outputs and guards match and finalELFs are zeroFSM in both arms.
They were not promoted or queued. These are tested schedules, not proofs that
all orientation or overlap alternatives lose. Receipts:
`perf_records/tiny_short_m_orientation_gsim.json` and
`perf_records/resnet_direct_b_bank_overlap_gsim.json`.

SmolVLA's exact four-partial convolution fixes every initial embedding bit, but
the full conv4/flash/old-balanced-normalization gate still FAILS:84/1,600 outputs
outside the original tolerance, relativeL2 .019814495, maxabs .128245711. The
explicit pinned Welford LayerNorm builder matches all786,432 original first-norm
values and4.72M independent replay values. Full conv4/flash/Welford stillFAILS:
78/1,600 outputs outside tolerance, relativeL2 .018923141, maxabs .114600658. Default
frontend/pipeline behavior is unchanged. Core integrates the explicit flash,
masked attention, conv4 and LayerNorm builders. Complete source-bound gates
determine whole-model admission.

### Fused activation evaluation and precise CPU attention panels

Merlin0f6271072 exposes default-off `fuse_activation_polynomial_fma`, implying
the independent activation approximation and narrow exact FMA intrinsic lowering.
It changes polynomial arithmetic explicitly; normalization still calls expf.
All63 relevant scalar/vector/native/Spike/compiler tests pass. Initial six vector
test failures were missing llvm-objdump onPATH; with the declared toolchainPATH,
all six execute and pass. No gate or tool requirement was weakened.

The full Tiny fused-polynomial+ClangO3 arm retains every captured output bit and
the original Torch gate in native and actual Spike. Its176 activation FMAs and
all22 normalization expf calls are recorded.209,494,756 Spike instructions are
4.09% fewer than the current1800 build. ClangO3 alone changes the model object
and retires217,424,612 (0.46% fewer); its native oracle/device/runtime artifacts
are byte-identical toO2. The small isolated optimization screen is not separately
queued. Neither proxy result establishes hardware performance. Receipts:
`perf_records/tiny_fused_activation_poly_o3_spike.json` and
`perf_records/tiny_activation_poly_host_o3_spike.json`.

Subsequent FireSim1806 verifies764,493,870 forward cycles for the fused polynomial
withClangO3:3.43% fewer than1800 and57.53% fewer than1747. All256,000 captured
output bits, unchanged Torch tolerance, final zeroFSM audit, staged ELF and stock
bitstream identities pass. The activation approximation still requires original
accuracy gates for other captures. See
`perf_records/tiny_fused_activation_poly_o3_firesim1806.json`.

The independently compiled allocator declaration screen adds only LLVM malloc
return-noalias under the pinned monotonic arena contract. It produces byte-identical
model.o and final ELF to1806; no duplicate native, Spike or FireSim run is needed.
No permanent compiler option is promoted from this emission-neutral experiment.
The fresh1806 profile passes full Spike output/count/conservation checks and is
verified by FireSim1809:765,151,605 interior cycles =219,056,004 device +546,095,601
host. All155 calls, complete output digest and conservation pass. Observed
whole-forward profile overhead is658,156 cycles (0.0861%);1806 remains the
unprofiled control. Large host gaps precede attention output projections and
down projections and include all intervening CPU operations. Spike instructions
are not hardware cycles.
See `perf_records/tiny_malloc_noalias_emission_neutral.json` and
`perf_records/tiny_fused_poly_current_profile_spike.json` and
`perf_records/tiny_fused_current_profile_firesim1809.json`.

Captured ResNet classifier M1/N1000/K2048 GSIM execution measures594,580 to456,259
cycles (23.26% fewer) forwideB/cacheA. All1,000 i32outputs and2,048 guard bytes
match. Native full-model capture supplying its activation/hoisted weights matches
all original outputs; the schedule itself has not been promoted into a full model.
See `perf_records/resnet_classifier_captured_tail_gsim.json`.

SmolVLA CPU PV accumulation is now traced to actual oneMKL2024.0 Update2 DEF Zen
SGEMM dispatch, independently of ATenAVX2 softmax. It computes zero-seeded serial
FMA panels192+192+128, then adds the prior destination aftereachpanel. Explicit
policy `mkl_2024_0_u2_def_zen_k192` reproduces all12 heads and every inspected
intermediate. The compatible conv4/Welford/PV192 trusted-input vision chain
matches9,437,184 original outputs acrossall12layers bitexact. The unchanged full
model gate stillFAILS (relativeL2 .016156828, maxabs .096718788). Its original
Torch replay also reproduces all1,600 retainedgolden values; full-source boundary
tracing now localizes remaining language/actor error. No full Smol hardware is
admitted. Exact arithmetic compatibility remains optional and source/backend
bound; no default frontend rewrite or reference replacement was made.

### Upstream half-precision contraction correction

An executed long-reduction witness exposed BF16 accumulation in24 SmolVLA
language/actor PV results:K4096 ones produced256 instead of4096. model2MLIR main
now includes46851eaf, which widens BF16/f16 operands to f32, accumulates in f32,
and narrows once across mm, matmul, bmm, batched and broadcast paths. Rank-2
contractions also begin from zero rather than uninitialized tensor.empty.
All160 focused/related tests pass, including45 native long-reduction,
product-rounding and empty-K cases plus10 export-path gates. Remote main was
independently verified. The fresh whole SmolVLA source has zero opaque operations;
original checkpoint weights, extras, six inputs,1,600 reference outputs,
manifest and input-order hashes remain unchanged. Its full gate is pending.
Backend-compatible arithmetic policies remain explicit and independent.

### Exact destination reuse and rejected readout correction

Default-off `reuse_tensor_destination` clones a pure pointwise body unchanged
into a proved static interior slice of a fresh filled pad. Complete ResNet native
and actual Spike outputs remain exact; retired instructions fall11,497,420 to
11,001,021 (4.32%). A second full pad allocation/copy remains because upstream
bodyless tensor results conservatively alias all tensor arguments. A wrapper
returning original C after writable to_buffer is incorrect when bufferization
copies C; a live-input regression and the whole original gate both exposed stale
results. It was removed and never queued. The needed result-alias abstraction
must model copy-on-write and descriptor identity correctly. See
`perf_records/prestem_destination_alias_diagnosis.md`.

The branchless fixed-point readout correction prototype adds padded low/high
sentinels and loads both adjacent thresholds. Twelve arithmetic/adapter tests
pass. Actual50176-value ResNet scratch comes from an otherwise unchanged1795
native replay with all1000 original output bits exact. Four completed alternating
GSIM calls check every50176 value and2048 guard bytes: conditional1546507/1519331
cycles versusbranchless1912991/1902986. Median increases24.47%. The600-second
probe timed out before the smaller25088-value readout and full pass marker;
that incomplete run is not a successful hardware qualification. The prototype
was removed from active source, retained as a reproducible rejected patch, and
neither promoted nor queued. See
`perf_records/resnet_branchless_readout_rejected_gsim.json`.

### Verified composition and exact scalar contraction accumulators

ResNet1812 composes exact pre-stem destination reuse and the captured classifier wide-B/resident-A schedule with the existing strongest recipe. Stock FireSim verifies46,316,907 cycles,703,414 (1.50%) fewer than1795, with all1,000 original output words exact. Its normal marker includes linked device identity. The banked one-row stem experiment is rejected: qualified Spike outputs pass, but observed GSIM timing increases11.05%; its total-cycle cap ends before a complete numeric GSIM marker. No whole-model queue submission follows that rejected screen.

Merlin3db98325f adds default-off `scalar_contraction_accumulator` before elementwise fusion. Static f32 contractions with one final reduction dimension preserve separate multiply/add and increasing-K order with a scalar iter_arg, then insert each output once. Upstream bufferization owns tensor alias legality. Complete Tiny native and actualSpike retain all256,000 original output bits and the originalTorch gate;177,453,518 instructions are15.30% below1806. FireSim1816 verifies648,210,569 cycles,15.21% fewer than1806 and64.0% fewer than1747. Device objects are unchanged. A follow-up Clang `-funroll-loops` experiment produces a byte-identical model.o; no duplicate correctness or queue run, and no permanent option. See `perf_records/tiny_scalar_accumulator_firesim1816.json` and `perf_records/tiny_unroll_flag_emission_neutral.json`.

The explicit full-K wide-A matmul11 screen consolidates each256-column panel into64-column DMA commands without changing25,088 compute commands or the401,408 issue floor. All401,408 captured outputs,2048 guard bytes, tail geometry and final zeroFSM pass. GSIM measures582,086 to555,498 cycles (4.57% fewer);26,588 cycles alone do not justify a whole-model queue arm. Resource checks and the default selector remain unchanged. See `perf_records/resnet_matmul11_full_k_wide_a_gsim.json`.

### Guarded exact integer mean and device compilation identity

The original49-value serial f32 DQ/mean/Q reduction now has a universal rational error certificate indexed by exact integer sum. Every sum from-6,272 to6,223 is considered. A closed interval with one quantized value uses a readonly int16 table; the eight ambiguous totals replay original separate f32 multiply/add, divide and quantization in increasing reduction order. No sample bounds or reassociation justify this fast path.17 focused tests pass, including all65,536 two-element arrays and all12,49649-element totals in adversarial/reversed orders plus guard, proof tampering and source refusal cases. Full original native and actualSpike pass all1,000 bits;10,816,839 instructions versus1812's10,985,615. The finalELF passes zeroFSM. FireSim timing is pending. A volatile fallback product prevents accidental native FMA contraction; target compilation also uses `-ffp-contract=off`.

Merlin0068b4b79 binds the normal harness marker to actual device/matrix bytes in link order, using a domain-separated length-prefixed digest.31 identity/runtime/offload tests pass, including relocated paths, altered objects and reversed link order. Previously omitted device objects made old markers incomplete; historical final and staged ELF hashes remain valid authority. The guarded-mean CPU object participates in this same closure. The result descriptor returned by an external adapter is the actual passed destination; tensor bufferization may copy its logical init. Returning the original tensor in a wrapper is invalid.

### SmolVLA retained accuracy and trig dispatch correction

The user explicitly retains atol=0.03125 and rtol=0.02 for every element. Fresh model2MLIR46851 source preserves all checkpoint/input/golden identities but still fails47/1600 outputs, with identical prior output bits. Merlin already widened these half contractions, explaining why the necessary upstream fix did not alter this full result. Actual boundary recordings localized three BF16 language-query differences to source sine/cosine; projection and power-based timescales match. SLEEF u35 scalar reproduction matches SLEEF symbols but not Torch.sin/cos. Actual GDB dispatch is MKL VML (`vmsSin` to `mkl_vml_kernel_sSin_EXHAynn`) with high accuracy and FTZ/DAZ off. SLEEF is therefore not the selected backend. A pinned source-backend table for structurally bounded integer RoPE positions and constant frequencies is being qualified as an explicit policy. Arbitrary float angles and unrelated timestep trig are excluded. No full-model hardware admission yet.
