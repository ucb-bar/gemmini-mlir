# Primitive-only Gemmini golden: current evidence

The requested targets are ResNet-50 at or below 22,387,449 FireSim model cycles, full `SY_model_smolvla` near 5 billion, and TinyLlama as low as possible, all on `FireSimGemminiRocketConfig` with **zero** Gemmini `LOOP_*` instructions in the final linked ELF. Ordinary RISC-V branch loops repeat the xDSL Gemmini primitive tile schedule.

## Latest verified whole-model results (2026-10-05)

| Model/capture | Stock FireSim forward cycles | Correctness evidence | Receipt |
|---|---:|---|---|
| ResNet exact52/wide16, virtual padding/layout, general banked dense command-cost policy, five static resident convolutions, fresh output ownership, packed guarded mean and clamp/RNE | 42,269,808 (1849) | All 1,000 original output words exact | [1849](perf_records/resnet_banked_command_policy_firesim.json) |
| Full 22-layer pretrained TinyLlama, 8 tokens, eight exact scalar outputs/K-unroll2 plus explicit cached-A B-prefetch and fresh writer ownership | 569,151,067 (1846) | All 256,000 compiled output bits unchanged; original Torch gate passes | [1846](perf_records/tiny_expanded_writer_prefetch_firesim1846.json) |
| Full SmolVLA, all source numeric policies including64 ordered source sums and16 ordered actor f32 products | No qualified whole-model hardware result yet | Full native output now **bitexact all1,600** to immutable original golden, using scalar device stand-ins; original atol=0.03125/rtol=0.02 unchanged. Actual RV64GC target qualification in progress | [Native full gate](/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-source-fma-20261005/full_gate_receipt.json) |

Every listed hardware result pins its final zero-FSM ELF, actual staged ELF,
stock bitstream and job-owned output. ResNet is a random-weight semantic capture;
Tiny uses the full pretrained checkpoint. These results do not establish the
remaining performance targets or pretrained ResNet accuracy.

## Current compiler work and ownership

The OOT dialect is a general compiler backend. Production decisions use input semantics,
shapes, layouts, numeric contracts and hardware capabilities; model names, captured IDs and
benchmark constants are confined to experiment selection/receipts. See [AGENTS.md](../AGENTS.md).

Merlin now owns complete requantization transition proofs, CPU integer readout and guarded
contiguous/packed mean code generation. OOT delegates to those owners and retains device
schedules, instruction lowering, resource facts and target ABI wrappers. The extraction
preserves all proof and generated C bytes for five independent cases and both actual source
readouts; generic numeric tests moved to core. [Extraction receipt](perf_records/generic_numeric_owner_extraction.json).

Structural bounded host RNE legalization and common tensor permutation proofs now also live
in Merlin; the OOT provider delegates with unchanged qualified object/C bytes. The general
resident compiler policy chooses the same seven kernels from shape/padding/resource facts,
without a resident source-ID selection list, and reproduces the qualified device object and
rewritten IR byte for byte. [Policy equivalence](perf_records/resnet_resident_compiler_policy_equivalence.json).

Generic source-bound post-offload callbacks and explicit full-write/result-identity contracts
now let the normal model builder expose fresh output ownership. All155 Tiny calls are covered,
with native and actual Spike outputs unchanged. Isolated hardware arm1842 measures607,616,796 cycles versus610,246,484 in1835 (0.431% lower); device bytes are unchanged and prefetch is off. The separately gated ownership+prefetch composition1846 verifies569,151,067 cycles,1,946,440 (0.341%) below1841, with identical device bytes. This is a single-run small improvement, without a variance-adjusted claim.
Eight-output/K2 plus device prefetch arm1841 now verifies571,097,507 hardware cycles,0.300% below1839; this small difference is one run, not a variance-adjusted claim.
The current1837 four-output/K4 hardware profile conserves155 calls and attributes611,478,982
interior cycles to219,073,155 device and392,405,827 host; intervals include all intervening
operations after fusion, not one source operation. [Profile](perf_records/tiny_current_k4_profile_firesim1837.json).

The next ResNet resident channel-loop/grouped-row composition passes all1,000 outputs in native
and actual final-ELF Spike, with an identical1836 host object and only seven device kernel
objects changed. Stock FireSim1844 verifies43,514,726 cycles,1.582% slower than1836;42,837,088 remains the best. It retires9,467,083 instructions versus9,946,365, demonstrating that fewer Spike instructions did not imply a FireSim improvement. [Qualification](perf_records/resnet_resident_channel_loop_spike.json).

Smol's native numerical blocker is closed. The first optimized RV64GC host compile timed out
at900seconds on a large monolithic function. Generic loop extraction retains the full native original gate bitexact all1,600 outputs. An initial77.4-second prototype also altered original function attributes; it is not final policy timing. The corrected generic helper-only inlining policy preserves original attributes, compiles through the normal pipeline in111.254seconds and passes all1,600 original native outputs exactly; actual device qualification remains pending. A separately labeledO0 correctness ELF also builds,
passes the final no-FSM audit and runs in Spike. Reusable host fusion/compilation
scalability and cheaper explicitly gated device attention remain work; the exact scalar vision
attention has19.33B MACs and cannot by itself establish the5B target.

The following narrative retains the earlier experiment sequence; the table above is current.

The exact uniform-zero/scale-aware quantize-round pass now applies to the two
remaining ResNet quantizers. All1,000 outputs remain exact in native and actual
Spike, whose retired instructions fall19,286,601 to14,254,689 (26.09% fewer).
FireSim1781 verifies 55,239,221 cycles; instruction counts are not hardware cycles. The separately
proved explicit-RNE/clamp legalization reduces the same whole model further to
11,355,701 Spike instructions, retaining all1,000 golden words. All107,415
boundary checks across the five rounding modes pass. FireSim1786 measures this
candidate at 49,673,153 cycles. Blocked64 mean reduces this to 48,780,534 in1789;
both remain explicit host lowering options. The source-
bound dense group/bank schedule variants also pass complete golden gates, with
representative GSIM367,749 versus447,124cycles (17.75% fewer); whole-model bank
timing with the strongest host path remains pending.

Tiny's combined scalar host, quantization fusion and bounded clamp/RNE candidate
passes the full native and actual Spike output gate, with all 256,000 outputs
unchanged. It retires 271,019,239 instructions, 60.08% fewer than the earlier
678,959,918 build. FireSim1792 verifies 866,822,103 cycles, 15.86% fewer than1788 and51.85% fewer than1747.
Its actual legalized model object participates in the normal build identity.
See [target gate](perf_records/tiny_scalar_host_quant_rne_spike.json).
The explicit scalar activation polynomial replaces22 activation expf chains while
retaining every normalization expf call. All256,000 output bits remain unchanged
in native and actual Spike on this capture; the approximation makes no universal
bit-exact promise. It retires218,423,000 instructions (19.41% fewer than1792).
FireSim1800 verifies791,638,514 cycles,8.67% fewer than1792 and56.03% fewer than1747.
[Gate](perf_records/tiny_scalar_quant_rne_act_poly_spike.json).
Explicit fused polynomial evaluation with ClangO3 retains all captured output bits
and the original Torch gate, retiring209,494,756 Spike instructions (4.09% fewer
than1800's build). FireSim1806 verifies764,493,870 forward cycles,3.43% fewer
than1800 and57.53% fewer than1747. This remains an explicitly selected activation
approximation with capture-specific accuracy evidence.
[Gate](perf_records/tiny_fused_activation_poly_o3_spike.json),
[hardware](perf_records/tiny_fused_activation_poly_o3_firesim1806.json).
The fresh profile for this compiled model preserves all existing objects and
passes full output/count/conservation checks in Spike and FireSim1809. Hardware
interior765,151,605 cycles divides into219,056,004 device and546,095,601 host;
all155 calls are conserved. Its765,152,026 whole-forward timing adds658,156
observed cycles (0.0861%) to the unprofiled1806 control. This profile predates the1816 scalar accumulator improvement. Host intervals before
attention output projections and down projections include all intervening CPU
operations. See [fresh profile](perf_records/tiny_fused_current_profile_firesim1809.json).

Fresh ResNet profile1801 attributes47,057,935 interior cycles to35,450,518 device
and11,607,417 host cycles, with all70 primitive calls conserved. Instrumentation
and placement add38,118 observed whole-forward cycles (0.081%);47,020,321 remains
the unprofiled profile control. The captured classifier tail's wide-B/cached-A GSIM screen
is23.26% faster with exact outputs; it is now composed with destination reuse in1812.
See [current profile](perf_records/resnet_current_leaf_profile_firesim1801.json)
and [selected geometry](perf_records/resnet_current_issue_geometry.json).
Exact pointwise destination reuse for the padded pre-stem passes complete native
and Spike gates, reducing retired instructions4.32%. Composed with the classifier schedule,1812 verifies46,316,907 cycles,1.50% fewer than1795. A proposed external result-alias wrapper failed a live-alias regression
and was removed. A branchless readout correction also remains rejected: four
completed GSIM passes on50,176 captured values are24.47% slower by median; the
600-second probe timed out before its full pass marker and smaller readout.
See [pre-stem gate](perf_records/resnet_prestem_destination_spike.json),
[alias diagnosis](perf_records/prestem_destination_alias_diagnosis.md) and
[rejected screen](perf_records/resnet_branchless_readout_rejected_gsim.json).

Confirmed frontend fixes are upstream on model2MLIR main: precision fixes at
050009e, SDPA scale/causal/options semantics at69c0370, and half-precision
matmul f32 accumulation plus initialized rank-2 outputs at46851eaf. The latest
commit passes160 focused and related tests, including45 native execution cases.
Fresh SmolVLA capture retains the original weights, inputs, golden and manifest
identities. The fresh complete gate still fails47/1600 outputs, with relativeL2 .016156828 and maxabs .096718788. Explicit backend arithmetic
compatibility experiments remain separate from default frontend semantics.

The selected ResNet schedule's mesh issue geometry totals22,805,632 cycles,
already above22M before DMA and CPU work. A source-proven nonnegative skip-domain
experiment reduces this to22,466,944; it has not been promoted into a graph.
This floor describes the selected schedule, not all possible exact algorithms.
See [mesh geometry](perf_records/resnet_exact52_mesh_geometry.json) and
[domain evidence](perf_records/residual_source_domains.json).

## Reference and schedule

The sole 22,387,449-cycle evidence is Jack's `resnet50_nofsm_q1013.zip` manifest. Its ZIP contains an ELF, disassembly, FireSim bundle and manifest, but no source. We inspected the ZIP and extracted the ELF/disassembly into `out/reference_q1013/`; no Jack folder was opened. `python -B -m mlir_oot.reference_profile ELF DIS -o static_schedule.json` gives static instruction counts for the function symbols. It finds 72 model-related symbols, 42 four-byte aliases and 30 larger bodies. Among those bodies there are 13,764 static `PRELOAD_CMD`, 8,359 `COMPUTE_AND_STAY_CMD`, 5,405 `COMPUTE_AND_FLIP_CMD`, 2,178 `LOAD_CMD`, 1,537 `LOAD2_CMD`, and 909 `LOAD3_CMD` instructions. Static counts are not dynamic issue counts or cycle attribution. The archive's ELF includes six `LOOP_WS*` commands in a linked library routine; the ZIP therefore cannot meet the stricter final-ELF zero-FSM criterion even though the measured path may not execute that routine.

Our `no_fsm_audit.py` scans all executable RISC-V ELF sections after linking, decodes variable-length instructions, and refuses every `LOOP_*` funct, unknown Gemmini funct, and invalid Gemmini funct3. It must pass on every candidate submitted for a zero-FSM claim. The GSIM probe ELFs pass it; the archived reference fails it.

The prior Claude session's own [q1013 analysis](/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-studies/exo-comparison/q1013_analysis.md) resolves the reference control flow. On the FireSim `argc=0` path, **all 53 convolutions run as Exo Gemmini kernels**, with zero host im2col calls and zero executed hardware-loop commands. The alternate host im2col/vendor path is reached only with `argc>=3`; its linked `sp_tiled_matmul_ws` contains the six static loop commands. The q1013 stock bitstream matches our pinned stock `FireSimGemminiRocketConfig` tar byte for byte. Its full-forward timing window covers FC, argmax and dequantization and is comparable to the prior lean-board whole-model window within the report's approximate 2% hardware uncertainty. q1013 issues 1,155,264 computes against an ideal 1,083,136, or 1.067×, at about 19.4 cycles per compute.

The report attributes the prior 38.33M-to-22.39M gap to FC on Gemmini (~3.5M estimated), packed stem K with pooled readout (~3.3M), residual add as identity matmul with a real D input (~2.7–3.3M), overlapped double-buffered load/compute and 16×64 stores (~3.5–3.8M), and whole-output-row 3×3 tiling (~2.3–2.6M). These are **estimates on the old program**, not measured gains in this branch. The lean configuration forces D to garbage, so the residual form requires stock.

## Implemented path

`mlir_oot/golden_gemm.py` emits verified xDSL `gemmini.*` target IR for a dense int8 GEMM schedule. It supports i8 or i32 output, edge tiles, i32 bias, store scaling and ReLU. `mlir_oot/golden_tuning.py` searches scratchpad- and accumulator-legal tile blocks using a primitive-command and DMA-request model. This is a geometric model, not a fitted FireSim cycle predictor. `tests/gsim_gemm_probe.py` links the final ELF, audits it, and compares against a CPU reference; `--embed-expected` computes large references on the host to avoid spending GSIM time on scalar oracle loops.

Passing GSIM checks include 16×16×16, 16×16×32, 17×19×20, 80×80×17, 144×16×16, 16×144×16, and 17×19×20 with i8, bias, ReLU, and 0.25 scale variants. For 16×144×16 i32, analytical tuning selected `bm=1,bn=9` and measured **1,248 kernel cycles** versus **1,388** for the initial `bm=4,bn=4` probe, on the same pinned GSIM engine. This is a kernel probe, not a ResNet cycle result.

The optional 16×64 accumulator store is bit exact on the simulator. For 144×64×64 with a tuned 9×4 output block it measured **8,142 kernel cycles** versus **8,414** with 16×16 stores. For 16×64×16 it was slower (658 versus 613), so it remains a shape-selectable choice. All four final ELFs pass the no-FSM scan.

Holding B stationary across output-row tiles (`reuse_b`) passes numerical and final-ELF checks. At 144×64×64 it reduces GSIM kernel time from 8,142 to **6,273 cycles**; at 512×64×64 it reduces **25,360 to 17,727**. Preloading a whole small B matrix into scratchpad once per call (`cache_b`) reduces the latter again to **16,575**. The 17×19×20 edge/bias/scale/ReLU probe passes with both options, but `cache_b` raises its time from 776 to 802 cycles, so caching remains optional. The analytical model now counts one B load per cached panel, rather than per output block. These are simulator kernel measurements; there is still no full ResNet FireSim result for this implementation.

The q1013 trace's 16×64 MVIN/MVIN2 panel form is now supported for compatible channel extents. The dialect verifier admits up to four DIM-wide blocks for scratchpad MVIN, with the accumulator MVIN bound kept at DIM. `wide_a` and `wide_b` preserve the same panel payload but reduce RoCC load commands; the tuner counts commands separately from panel bytes. With a 16×4 output block, wide stores, stationary/cache B and both wide loads, 512×64×64 passes the numeric and linked-ELF checks at **15,607 GSIM kernel cycles**, compared with 16,575 using narrow loads. The edge 17×64×64 biased/scaled/ReLU case also passes, including a 2048-byte output guard. Alternating two accumulator/A slots (`pipeline_m`) passes but measured 16,732 versus 16,674 at 8×4 on this shape, so the upstream selector leaves it off. Jack's interleaving occurs inside compute bursts; this simple alternation does not reproduce that schedule.

`golden_resadd.py` implements the stock-only identity-matmul form with a real D input and wide stores. Its 16×16, 17×19 and 17×64 cases pass GSIM, the final linked-ELF audit, and a 2048-byte output guard check (431, 3,192 and 1,837 kernel cycles respectively). A tile-grouping bug had treated a final partial row as full when `bm=1`; it is fixed and covered by `test_golden_groups.py`. The kernel stages edge and non-64-byte-pitch stores in 1024-byte aligned scratch before copying valid elements into dense output. The odd-width edge path is correct but much slower than the full-tile path, which is acceptable for the ResNet channel dimensions that are multiples of 64.

### Device compilation

`golden_device_lower.py` is the only translation of golden `gemmini.*` operations to RoCC inline asm. It verifies the target module, encodes only exact primitive operations, and refuses an unsupported Gemmini op. CPU repetition remains ordinary LLVM CFG. `golden_device_compile.py` writes the target xDSL module, lowered LLVM dialect module, LLVM IR and RV64GC object, records their hashes and compiler arguments, and audits the object. The probe harnesses then link the object with their runtime and audit **every executable section of the final ELF** again, since a library could introduce forbidden LOOP commands after object compilation. The 32×16×16 target-to-LLVM run produced the same linked ELF hash and 409 GSIM kernel cycles as the former direct emitter; 17×64 residual likewise retained its original linked ELF hash and 1,837 cycles. This is a real compile path from the xDSL target module, rather than separately authored target and LLVM schedules.

For a supported single-layer capsule, run `python -m mlir_oot.golden_upstream capsule.interface.mlir --llvm-bin LLVM_BIN --workdir OUT`. The output includes `kernel.gemmini.mlir`, `kernel.llvm.mlir`, `kernel.ll`, `kernel.o`, `device_compile.json`, and `upstream_binding.json`. The receipt records the SHA-256 of both compiler binaries and all generated code artifacts. After linking the model runtime, run `python -m mlir_oot.no_fsm_audit final.elf`; the object audit alone cannot certify a linked program.

FireSim queue job **1670** failed before simulation because its Chipyard config lacked the requested hardware key. Job **1673** used our own stock Chipyard worktree and the existing `alveo_u250_firesim_gemmini_rocket_stock` entry, but timed out in `INFRASETUP` after its leading kill exceeded 90 seconds; it never booted the ELF or produced a cycle measurement. The queue records it as `TIMEOUT` at 1080 seconds. A nearby unrelated queue job also failed, so more submissions on the same slot are deferred until the infrastructure can complete setup.

## Upstream whole-model lowering

Running this OOT `gemmini-opt --convert-iface-to-gemmini --emit-command-buffer` on each shipped capsule yields zero commands and an explicit decline:

| Capsule | First blocking result |
| --- | --- |
| `SY_model_resnet50` | Estimated 65,404,026,705 CPU-lane straight-line evaluations exceed the 400,000 budget. Its conv matmuls remain f32 after dequantization despite `prov.quantization = "int8_static_act_int8_weight"`. |
| `SY_model_smolvla` | Estimated 9,395,352,591 CPU-lane straight-line evaluations exceed the same budget. An xDSL multi-result `linalg.generic` printer/parser mismatch was normalized first so the full 11 MB file parses. |
| `SY_model_tiny_llama` | Estimated 3,031,111,489,682 CPU-lane straight-line evaluations exceed the same budget. |

These estimates count element evaluations in a hypothetical unrolled scalar program; they are **not** predicted hardware cycles. The looped GEMM removes one code-size bottleneck for device contractions, but whole-model lowering still needs quantized conv recognition and direct input gathering, tensor layout propagation, scalable host-operation loops, residual add/reduction kernels, ABI and weight binding, and a correctness oracle before any full-model FireSim run can be called golden. The current ResNet capsule's f32 path is a different program identity from Jack's int8 static recapture, so equality of cycle numbers alone would be misleading.

One upstream geometry optimization is now implemented in `lowering/plan.py`: a stride-one, unpadded NHWC 1×1 `merlin_iface.conv2d` uses its original input as the GEMM lhs, because the physical C row pitch is identical to the matrix K row pitch. The real GQ1 capsule still emits a valid command buffer and xDSL target artifact, now with no `im2col_recipes`; GQ2's padded 3×3 path retains its recipe. This eliminates an unnecessary derived tensor for this upstream case. It does not convert the shipped f32 ResNet graph to the int8 q1013 program or route an entire model through the golden kernel yet.

`golden_upstream.py` is a strict single-layer bridge from typed `merlin_iface.conv2d` to this compiled golden kernel. It derives dimensions, dtype, bias, scale and ReLU through `Builder`, refuses a gather or unsupported epilogue, records its source tensor to device pointer binding, tunes legal tiles, and compiles through `golden_device_compile.py`. The real GQ1 capsule generated `M=36,N=16,K=16`, `bm=3,bn=1`, and the object that this bridge compiled passed a numerical GSIM probe at **535 kernel cycles**, including a final linked-ELF no-FSM audit. The bridge is not yet the whole-model runtime or a general 3×3 convolution lowering.

A synthetic 1×1 NHWC capsule with `M=512,N=64,K=64`, bias, 0.125 store scale and ReLU selected 16×4, stationary/cached B, wide A/B loads and 16×64 stores through the upstream bridge. The **exact object compiled from that capsule** passed the linked-ELF audit, numerical GSIM and the output guard at **17,822 kernel cycles**. The 15,607-cycle figure above omits this bias/scale/ReLU work and is a different kernel; neither number is a full ResNet layer timing in FireSim.

A larger `M=3136,N=64,K=64` probe with bias, 0.125 store scale and ReLU passes at **104,931 GSIM kernel cycles**, with 3,136 mesh compute commands and a 50,176-cycle peak array floor. The analytical model also reports unique tensor bytes and a distinct upper request volume; dividing repeated request bytes by DRAM bandwidth is not a lower bound because cache and stride-zero bias broadcast can satisfy repeated requests. The model remains uncalibrated to FireSim.

`upstream_quant.py` now invokes Merlin's existing QDQ-aware integer contraction rewrite, writes the transformed IR and a source/content-bound receipt, and validates that this OOT backend parses the result. On the shipped `SY_model_resnet50` capsule it rewrites **54** contractions (53 convs and FC), makes all **54** named i8×i8→i32 matmuls mesh eligible, and hoists activation quantization ahead of **50** pure gathers; three strided-hole gathers are refused and one calibrated static activation is reused. The pre-gather option changes numeric granularity, so its receipt marks model accuracy verification as required. `frontend/mixed_matmul.py` fixes xDSL's synthesized body for i8×i8→i32 named matmul by widening both inputs before the multiply; `linalg_reader.py` places rewritten contractions by their current operand dtype even when `prov.orig_dtype` remains f32. The full transformed ResNet now parses/verifies and gives an explicit mixed-lane decline: its remaining host ops cost about **470,139,986** scalar element evaluations in the current unrolled emitter, above the 400,000 budget. That count includes zero-runtime tensor views and allocations, so it is a code-size estimate, not a cycle forecast. The top counted families are `tensor.empty` 107.8M, generic 80.2M, reshape views 135M combined, and dequantization 63.5M. Whole-model looped/fused host lowering and weight binding remain unimplemented.

## Generalized integer contractions and model arithmetic

`contraction_patterns.py` now checks typed indexing maps, iterator order, i8-to-i32 signed extension, multiply and add wiring, shapes, and a zero accumulator before accepting a contraction. The same matcher handles ordinary 2D GEMM and dense batched 3D attention GEMM. On the prepared model IR it finds **54/54 ResNet**, **367/367 SmolVLA** (303 2D, 64 batched), and **200/200 TinyLlama** (155 2D, 45 batched) integer contractions. This is contraction recognition, not whole-model code coverage. The 302 SmolVLA 2D `linalg.generic` contractions, previously routed to the host because only named `linalg.matmul` was recognized, are now eligible for 2D mesh scheduling. Rank-3 contractions have a separate primitive-only batched kernel with ordinary CPU batch repetition.

The inventory now separately counts every top-level operation in each prepared function, making the model coverage gap explicit:

| Prepared model | Top-level graph ops | Device contraction ops | Other graph ops still needing graph lowering |
| --- | ---: | ---: | ---: |
| ResNet-50 | 3,619 | 54 | 3,565 |
| SmolVLA | 32,386 | 367 | 32,019 |
| TinyLlama | 12,988 | 200 | 12,788 |

Many remaining ops are constants, `tensor.empty`, or reshape views and should collapse under allocation and alias planning; the rest include quantization, gathers, reductions, softmax, transposes and elementwise computation. This is an operation coverage ledger, not an estimate of required runtime commands. The whole-model coverage flag remains false for every model.

`golden_model_inventory.py` writes an auditable arithmetic census from those exact shapes. The numbers below count integer contractions only and assume a 16×16 array with one fully pipelined output row per cycle; they exclude loads, stores, host work, and all model overhead:

| Prepared model | Exact contractions | Integer MACs | MAC/256 floor | Padded compute-command issue floor |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 54 | 4.089B | 15.973M | 17.330M |
| SmolVLA | 367 | 130.873B | 511.221M | 526.368M |
| TinyLlama | 200 | 8.281B | 32.348M | 64.741M |

Jack's 22.387M ResNet run is 1.29× the padded issue floor. The requested 5B SmolVLA target is about 9.5× its integer padded issue floor, but the old full-policy runs spent very large time in scalar softmax and other host operations. Arithmetic alone cannot establish feasibility. TinyLlama's short rows waste half or more of a 16-row array in many layers, making its padded issue floor about twice its pure MAC floor.

There is a structural ResNet layout mismatch to solve before model performance can approach Jack's schedule: the prepared `conv_0` is `64×147 · 147×12544`, and `conv_2` is `64×576 · 576×3136`. The left operand comes from OIHW weights and the right from a materialized NCHW im2col gather. Jack's high-performance kernel instead consumes spatial output rows and output-channel columns, and gathers each convolution from resident input without a host im2col. Merely compiling the current prepared matmul would preserve the expensive gather and wrong physical output layout. A convolution-specific tensor layout and direct-gather lowering is required.

### Model-wide device object compilation

`golden_contraction_upstream.py` selects one exact rewritten operation, chooses a legal schedule from its shape, and compiles it through the xDSL Gemmini to LLVM to RV64GC path. The small 2×17×19×20 batched probe passed complete numerical GSIM, output guard, and linked-ELF audit at **1,492 kernel cycles**. The **exact upstream SmolVLA `matmul_384` object** (batch 15, M50, N64, K113) passed the complete numeric oracle and linked-ELF scan at **83,398 GSIM kernel cycles**. Its first harness run timed out at 180 seconds before readback; the same ELF passed with a 600-second timeout. The exact upstream TinyLlama `matmul_194` (batch 32, M8, N64, K8) passed at **11,736 GSIM kernel cycles**. These kernel timings are on pinned GSIM, not FireSim model cycles.

`golden_device_catalog.py` now builds one model-scoped object from all recognized contractions, deduplicating identical schedules and assigning content-derived unique symbols. Both ordinary and batched specializations receive stable distinct symbols, preventing duplicate definitions when hundreds of operations are linked. The source SHA, every operation-to-symbol binding, per-kernel schedule, compiler hashes, and object audit are recorded. Complete catalogs compiled with **54/54 ResNet contractions → 21 kernels (133 KB)**, **367/367 SmolVLA → 26 kernels (117 KB)**, and **200/200 TinyLlama → 8 kernels (29 KB)**; all three object audits passed. The TinyLlama catalog object itself was linked into a final ELF and its `matmul_194` symbol passed the full numerical GSIM oracle, output guard and final-ELF no-FSM audit at **11,683 kernel cycles** before the configuration-once refactor. This validates the deduplicated multi-kernel object's symbol linkage as well as its arithmetic. The catalog currently covers the integer contractions only. Tensor allocation, layouts, quantization epilogues, host nonlinearities, complete model execution, correctness verification, and FireSim per-group timing remain open.

The batched implementation now configures and flushes Gemmini once per batched call, then repeats primitive tile work through an ordinary CPU loop and fences at the end. On the real TinyLlama `matmul_194` shape, complete numerical GSIM and the linked-ELF audit pass at **9,282 cycles** versus 11,736 before, a 1.264× speedup. The full TinyLlama catalog was rebuilt, linked, audited and numerically checked after the change at **9,338 cycles**. The real SmolVLA `matmul_384` improved **83,398 → 81,607** GSIM kernel cycles with complete numerical and ELF audit checks. The two-batch edge case was 1,506 versus 1,492 and remains a candidate for size-based selection. The [optimization log](golden_optimization_log.md) records comparable observations and the abstractions needed to automate them.

An optional short-row A-resident schedule now preloads one M≤16 A tile before sweeping several N blocks. On the same 8×512×256 i32 synthetic projection and pinned GSIM, full numerical and final-ELF checks passed at **26,941 cycles versus 31,947** (1.186×). The analytical count reduces A MVIN commands from 128 to 16 without changing compute count. The model rule selects it only with at least four N blocks; among the prepared model catalogs this currently applies to TinyLlama's 8×5632×2048 and 8×32000×2048 projections. Those exact layers have not been timed in FireSim or GSIM, so the 1.186× figure is a shape-specific probe result.
The updated TinyLlama catalog still covers 200/200 contractions with eight distinct kernels and binds all eight through Merlin. A typed dynamic A scratchpad address keeps cached-A K repetition in an ordinary CPU loop. The catalog object is now **31,856 bytes**, versus **953,672 bytes** with K unrolled and **28,432 bytes** before cached A. On the same pinned synthetic GSIM probe, the looped kernel passes full numeric, guard and final-ELF audits at **26,948 cycles**, seven cycles above the unrolled form and **1.186× faster** than the 31,947-cycle baseline. The typed verifier checks a declared address range; a future automatic lowering pass must establish that range from loop bounds before using the dynamic form. These are contraction-level results, not a TinyLlama model cycle result.

The isolated Merlin integration worktree now has an external catalog route in its existing whole-model offload compiler. The route checks exact prepared source bytes and provenance region/type bindings, adapts dense rank-2 and whole-batch rank-3 pointers, and compiles an RV64 shim. On the exact prepared sources, rewrite plus shim compilation bound **54/54 ResNet contractions (21 symbols), 367/367 SmolVLA (26 symbols), and 200/200 TinyLlama (8 symbols)**. Set `DeviceRouting.catalog_builder` to `mlir_oot.golden_device_catalog.merlin_builder(LLVM_BIN)` to compile from Merlin's final prepared file before offload; its TinyLlama smoke run built and bound all 200 calls. Set `DeviceRouting.final_elf_audit` to `mlir_oot.golden_device_catalog.final_elf_audit` to require the no-FSM policy on the final image. This is device-code binding, not whole-model lowering or execution. FireSim jobs 1709–1711 are probe jobs, not model performance results.

The upstream ResNet preparation now enables Merlin's exact 1×1 im2col identity-view pass after QDQ contraction rewriting. Its matcher was extended to named `linalg.matmul`, the form QDQ actually emits. On the exact capsule it replaced **33 copies** spanning **7,200,256 int8 elements**, refused **17 nonidentity windows**, and retained **54/54** mesh contractions. A newly compiled 21-kernel catalog passed its no-FSM object audit and its exact-source Merlin rewrite/shim binding routed all 54 calls. The copy elimination has no FireSim cycle measurement yet; 3×3 and strided windows still materialize or require a direct-gather schedule.

For an actual Merlin model build, set `DeviceRouting.prepared_transform` to `mlir_oot.golden_device_catalog.merlin_identity_view_transform` alongside the catalog-builder and final-ELF-audit callbacks. The preparation hook runs before catalog compilation and source hashing. Its ResNet callback smoke proved 33 views and compiled all 54 contractions to 21 audited kernels; a runtime regression test checks that catalog selection and offload bind the transformed bytes.

The complete ResNet capture now compiles to an RV64GC ELF with all54 contractions linked and a passing final no-FSM audit (SHA18823d02831d0b98…). The same LLVM host graph plus identical device ABI shim, using scalar int8 reference kernels, reproduces all1000 captured integer-reference outputs exactly: relative L2=0, max error=0, argmax713 matches. This proves host lowering/layout/ABI for the captured random-initialized model, not hardware execution, pretrained quality, or22M performance. Actual Gemmini Spike execution is now running before a complete-model FireSim measurement.

## Current exact candidates and compilation contract

Tiny1816's default-off scalar accumulator schedule preserves each f32 contraction's increasing-K multiply/add order, including nonzero initial values. FireSim verifies648,210,569 cycles,15.21% fewer than1806 and64.0% fewer than1747. Four independent output accumulators improve this to615,651,105 in1821,5.02% fewer than1816; every output bit is unchanged. Partial K-unroll2 improves this further to612,282,210 in1828,0.547% fewer than1821. Two-output1825 measures622,639,906 and is rejected in favor of four outputs. All155 device contractions and the previously selected activation approximation remain unchanged. Partial K-unroll4 is queued as1832. An additional Clang loop-unroll flag emits a byte-identical model object and was rejected without duplicate simulation.

The guarded quantized mean replaces a proved canonical Q/DQ serial mean with an integer sum and an exhaustive sum certificate. For the original49-value ResNet reduction, eight of12,496 totals require exact floating-point replay; the certificate covers every signed-i8 input sequence. Full native and actualSpike retain all1,000 original output bits at10,816,839 retired instructions,1.54% fewer than1812. FinalELF zeroFSM passes. FireSim1819 measured46,680,853cycles,0.79% above1812; this arm is not the best recipe and its driver omitted explicit host scheduling. See [proof and binding](guarded_quantized_mean.md) and [whole-model gate](perf_records/resnet_guarded_quantized_mean_spike.json).

Normal build markers now include the actual linked device/matrix object bytes in link order, with length and domain separation. Object paths do not affect identity. Final and staged ELF SHA256 remain authoritative for both new and historical runs. The prepared source, ABI, target object, native standin, source numeric policy and final ISA audit must close over the same compilation. Default-off compiler transforms remain independently selectable.

SmolVLA retains the existing elementwise gate by explicit user direction. Original Torch MKL VML high-accuracy sine/cosine dispatch differs from scalar libm at three BF16 query entries, which propagate into attention. SLEEF was investigated but is not the selected Torch backend. A bounded integer-position RoPE lookup policy is under qualification; unrelated timestep trig remains separate, and the original weights, inputs and golden remain immutable.

The packed NHWC mean variant proves the existing transpose and consumes the
physical layout directly. Exact unsigned16-lane sums process eight channels per
word with no interlane carry, keeping source f32 fallback for ambiguous sums.
Full native and actualSpike preserve all1,000 original words at10,214,792
instructions,7.02% below1812.23 focused tests pass. Stock1824 verifies44,507,889
cycles,3.91% below1812. The bankedmatmul11 arm1823 independently retains
a host object byte-identical1812 and all52 source/numeric proofs, measuring
46,127,051cycles. The combined packed-mean plus bankedmatmul5/8/11 candidate
passes all original native and actualSpike outputs and zeroFSM; its host object
is byte-identical1824. Stock1829 verifies43,969,384cycles,1.21% below1824. See
[combined gate](perf_records/resnet_three_banked_packed_spike.json).

SmolVLA source-compatible softmax initially returned NaNs in full preparation:
the pinned xDSL parser numerically interprets unquoted dense float hex literals,
turning the maximum initializer's negative infinity into4286578688.0. Merlin's
portable printer now emits quoted raw bytes for dense floating attributes and
recognizes splats by byte equality. Repeated xDSL/upstream parsing and native
tests retain infinities, signed zeros, NaN payloads and subnormals;44 related
tests pass. The trusted191,535-value softmax fixture is exact through full
preparation. The fresh whole-model gate remains pending; this does not admit
SmolVLA to hardware or relax its original elementwise criterion.

The explicit fresh-output descriptor interface passes all1,000 original native
and actualSpike outputs, zeroFSM and7 focused ownership/refusal tests. Caller
contracts reproduce the qualified host MLIR and compiled bridge byte-identically;
stock1831 is pending. See [ownership contract](descriptor_writer_contract.md).

Resident-input direct convolution reduces the matched H14/W14/C256 capsule from
675,375 to533,649 GSIM cycles with all50,176 outputs and guards exact. A DMA
block-stride field had been silently dropped by lowering; its corrected encoding
now has a regression gate. Five source-bound whole-model routes are qualifying
with separate preserved scale proofs. This is a capsule gain, not yet a whole
FireSim result. See [resident study](resident_convolution_study.md).


The general dense banked command-cost candidate passes all1,000 original native
and final-ELF Spike values. Its host object/LLVM match1836 byte for byte; only
four pointwise device objects change. Independent176x256x64 GSIM numeric/guard
checks measure29.00% fewer kernel cycles, and95x48x80 tails pass. Whole-model
stock1849 is queued; its slightly higher Spike instruction count is not a cycle
claim. [Candidate gate](perf_records/resnet_banked_command_policy_spike.json).

An explicit OOT support provider now exposes the curated harness/GSIM command to
current Merlin orchestration, removing the legacy Python registration dependency.
A real numeric/guard capsule retains byte-identical final ELF and1,919 GSIM kernel
cycles. [Infrastructure evidence](perf_records/current_core_gsim_provider.json).


Fresh1849 profile1850 verifies42,303,592 forward cycles (33,784 above unprofiled
1849). Conserved interior42,303,164 comprises33,814,043 device and8,489,121 host
cycles; all70 calls and original1,000 output words pass. The pre-stem host gap
is4,017,289; gaps before matmul26/49 are1,955,979/962,942 and include the two
integer-readout epilogues plus any other intervening CPU work. These are host
intervals, not isolated operation costs.
