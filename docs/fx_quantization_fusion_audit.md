# FX quantization, fusion and current lowering audit

## 2026-10-08 findings

These findings distinguish retained frontend graphs from current compiler IR and
emitted host work. An FX floating-point operation does not by itself establish
host execution. Graph counts do not establish cycles or accelerator coverage.
All candidates use the same compiler infrastructure; model names, provenance
IDs and benchmark ordinals do not select production transformations.

| Workload | Already established | Remaining evidence and action |
|---|---|---|
| TinyLlama | All155 dense products integerized;155 current calls use89 distinct i8 activation SSA values. The66 duplicate FX dequantization routes are already shared. All22 MLP observers already fuse the two integer producers' dequantization, SiLU polynomial, finishing operations and i8 quantizer. | All45 prepared BMMs bind to scalar host loops:22 QK,22 PV and one rotary outer product. Derive legal contraction/consumer representation candidates and integrate the current-source scalar family at the normal compiler seam. |
| SmolVLA |302 captured Linear products are INT8. Attention remains floating in prepared FX; the explicit vision provider already uses integer product families with host representation, bounds and replay. |104 duplicate activation-expression candidates exist in prepared FX. Current emitted Q/K/V code actually reruns three quantization producers over the same source and reciprocal, overwriting the same i8 buffer. Preserve source joins while eliminating equivalent quantization work; separately audit the intermediate exactness required by the selected numerical policy. |
| ResNet50 | The quantized graph has already folded53 BatchNorm operations. Fresh current device compilation recovered all five champion compact convolution schedules. | Trace quantization/layout/materialization and residual observations into current code. Fresh native and Spike output gates pass all1,000 original words exactly, with zero FSM instructions; whole stock timing remains unmeasured. |

### Provenance and CSE

An independent exact integer tensor fixture reproduces native CSE merging two
identical operations when untagged or identically tagged, while retaining both
with distinct diagnostic region IDs. On retained captured MLIR, native
canonicalize/CSE leaves10,015 Smol generics with provenance versus8,286 without;
Tiny leaves800 versus775. These are captured-IR comparisons, not current loop
counts or performance claims.

Merlin already has the opt-in `cse_through_provenance` feature. It strips all
provenance late in preparation and loses source/per-operation joins. It is absent
from the inspected current language-model recipes. Simply stripping attributes
earlier can also break source binding and preparation. The concrete generic
compiler improvement is to preserve every source join through native CSE, keep
semantic attributes and effects authoritative, and bind any sharing to current
SSA and ownership before bufferization. Existing default emission remains the
control until the new route is qualified.

### Numerical and host preparation boundaries

The retained Tiny capture's19 declared source files have been recovered from
their exact original Git commit, with every declared SHA matching. Immutable
archives and explicit original-path aliases restore the loader and tool-source
paths without changing the original receipt. This is declared source recovery;
it does not recreate an executable checkout or establish the full transitive
framework closure.

The original whole output gates remain ResNet0/0 and language-model
atol0.03125/rtol0.02. A source word equality claim requires raw word equality.
An explicit approximate policy does not imply exact intermediate source values;
its permission, fallback and original final gate must remain visible.

xDSL0.68.0 float construction/printing/parsing depends on the compiler host
rounding mode in the reproduced case. Rendering the f32 word0x3c1b7a9b under
upward rounding and parsing its decimal under RNE can produce0x3c1b7a9a. A
4-by-4 rounding matrix reproduces the issue. Fresh ResNet stages admit actual
host RNE through a native capability before IR construction and serialization.
That admission is not a general dependency fix or full environment restoration.
Target runtime FCSR capability is a separate obligation.

### ResNet entry work and reference timer

Current source accounting covers54 integerized contractions and all49 ReLU
uses. BatchNorm folding is already present. The final forward entry retains164
descriptor groups and1,256 expanded arguments; only57 groups are used by final
LLVM SSA. The unused groups represent53 obsolete bias tensors and54 unpacked
weights. Current packaging retains25,609,152 unnecessary weight bytes beyond
the25,506,912 live bytes. LLVM liveness does not alone prove those arguments
can be removed from prepared IR. The generic candidate must perform admitted
pure DCE and project the compiler-owned entry ABI, wrappers, descriptor table
and packed spans through one checked original-to-prepared argument mapping.

The owned ZIP's timed implementation takes a readonly150,528-byte INT8 image.
Its main calls the stem with that image; it does not time an FP32 input
quantizer. Our original whole timer includes converting the FP32 NCHW image
and setting up runtime descriptors. The original whole metric and gate stay
unchanged; phase accounting must expose this difference before attributing
the entire gap to device scheduling. A retirement prefix is not a measured
hardware saving across those different entry paths.

Fresh current normal compilation selects all52 convolution leaves with bytes
matching the qualified controls, including the five champion compact leaves.
The final611f3040 ELF passes native and Spike all1,000 original output words
exactly and contains no FSM instructions. The single remaining linked-body
attribution refusal is the stem's unsupported relocation kind; it is explicit.
Static stock bitstream/HWDB and HTIF console preparation pass. Hardware cycles
remain unmeasured. An additive preflight successor corrects an earlier read-only
queue observation that mistakenly counted historical TIMEOUT jobs as active;
the actual QUEUED/RUNNING filter reports an empty queue. Write access is absent.

### Current executable sharing and normal compiler insertion

On the inspected Smol first Q/K/V calls, three alpha-equivalent producers
separately overwrite the same i8 buffer. The first three exclusive producer
functions retire198,223,935 instructions, excluding their reductions. This is
actual repeated executable work; those instructions are not hardware cycles.
The exact CSE prototype retains every original source tree while native CSE
controls equivalence and effects. Its normal preparation integration and
current whole coverage remain pending.

Tiny's current-source scalar insertion topic uses ordinary lowering and model
builder APIs after normal tensor fusion and before bufferization. It parses
only new helper fragments into the live native session and replaces proved
private integer leaves, preserving producer/resource handles and source joins.
Runtime incoming-RNE predicates are provider owned and read per point.
Compiler-host RNE admission separately restores the full environment. The
28 source checks pass; independent package/installation qualification and the
original256,000 whole output gate are pending. No timing result is promoted.

### Rejected inference from a cache microbenchmark

The generic endpoint narrowing topic3ed806918 passes502 source and502 outside
installed checks, with no skips. Source/wheel/sdist/installation agree for1,029
Python modules,157 public resources and62 runtime files; four default LLVM
cases remain exact. The new owner preserves range facts only through admitted
subset refinement and keeps original checked fallback.

Its constructed nine-lookup component improves1.403%. In the real original
12-head,16-row slice, the provider uses three observer passes with quantizer row
counts16/9/2, batches66 partial refinements and lazily replays only two
denominators. All12,288 public words and source decisions remain exact. Complete
native medians242.300191ms to241.463285ms improve only0.3454%, with201,408
additional metadata bytes and84 additional allocations. This is not a material
or hardware gain; the constructed frequency must not drive a whole forecast.

## Ownership and automatic-loop requirements

- Merlin: provenance-preserving sharing, typed observation and numerical policy,
  host fusion/packing/requantization, ownership and compilation orchestration.
- model2MLIR: original/quantized/prepared correspondence, quantization schemas,
  axis/layout preservation and faithful importer decomposition.
- OOT: device operations, instructions, resources, ABI, target schedules and
  execution capability. The backend remains a general compiler.
- Phase0: bind each frontend candidate to current IR and executable work; show
  when a duplicate is already eliminated. Preserve source joins through fusion.
- Phase1: let the agent change generic middle-end representation/fusion and
  target lowering independently, with explicit numerical/effect/ownership facts.
- Phase2: price complete producer/consumer work, actual refinement frequency,
  allocation/storage, transfers and fallback. Calibrate analytical estimates to
  hardware; retained graph counts and native time do not replace stock cycles.

Best whole stock model cycles remain28,649,233 /258,621,872,969 /378,946,263
for ResNet/Smol/Tiny. Queue write access and GitHub DNS are still unavailable.
The22M/5B/300M objective remains active. Separate goal-meter observation at
03:10:17UTC is118,738,124 tokensUsed; exact billed topic/OOT allocation is UNKNOWN.

[Structural FX candidates](perf_records/fx_shared_quantization_candidates_20261008.json).
[Independent native CSE reproducer](perf_records/fx_provenance_cse_reproducer_20261008.json).
[Captured native CSE comparison](perf_records/fx_captured_native_cse_20261008.json).
[Tiny current source bindings](perf_records/tiny_fx_current_ir_20261008.json).
[Qualified generic narrowing](perf_records/endpoint_narrowing_installed_20261008.json).
[Real Smol refinement protocol](perf_records/smol_real_frontier_narrowing_20261008.json).
[Compiler-host float reproducer](perf_records/compiler_host_float_rounding_20261008.json).
[Recovered declared Tiny sources](perf_records/tiny_capture_declared_source_recovery_20261008.json).
[Smol current producer work](perf_records/smol_current_repeated_quantization_20261008.json).
[ResNet FX, timer and ABI audit](perf_records/resnet_fx_roi_abi_findings_20261008.json).
[Fresh exact whole gate](perf_records/resnet_fresh_whole_exact_gate_20261008.json).
[Current stock preparation successor](perf_records/resnet_current_stock_preflight_successor_20261008.json).
