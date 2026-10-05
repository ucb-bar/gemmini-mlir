# Exact original-attention signed-prefix certificate journey

## Scope and ownership

This is a diagnostic capsule, **not a production attention transform**. Merlin owns
`ordered_fma_bounds.h` / `fma_product_norms.h`, generic numerical derivation and host
metadata/replay. OOT owns the reusable xDSL `GoldenGemm` device lowering, target
ABI, device layouts/resources, final executable ISA audit and actual execution.
The experiment binds pinned original operands; neither model identity, golden
output nor captured provenance selects a production strategy. Golden data is
used only after source-derived decisions to audit results.

Scope: original first vision attention, up to all twelve heads, 1024 queries,
1024 keys and64 features. All12 **heads of one block** are not12 attention layers.
Nothing here closes the separate whole-model SmolVLA accuracy issue or measures
whole-model performance. Original gate remains atol=.03125, rtol=.02; we also
require every original BF16 output bit. Native Torch/BLAS digit emulation is
functional evidence only. Spike `mcycle`/stdout `METRIC cycles` and stage fields
are **retired instructions**, never hardware cycles. Actual stock FireSim is
collected separately through Recovery, sole queue owner.

## Original source semantics

QK is zero-seeded, increasing-K f32 FMA over64BF16 products, then source .125
scaling. Softmax uses the pinned `fexp_u20` polynomial, the original eight-lane
ordered denominator sum/tree (4,2,1), and two512-key online tiles with source
max/`expf` alpha updates. PV uses original zero-seeded192/192/128 partials within
each512tile, sequential f32 adds and source old-accumulator-times-alpha.
Final reciprocal/multiply and BF16 RNE bits are unchanged. Ambiguous outputs use
exact source replay; selective denominator replay uses source intervals, never
golden values. GCC/newlib and native glibc may differ in a few replay/cache
counts; every verified target final BF16 bit still equals the original.

## Numerical proof and independent checks

An exact signed chunk totalT and absolute product sumA enclose all chunkprefixes
between(T-A)/2 and(T+A)/2. For incoming exact prefixS, their magnitude is bounded
by abs(S+T/2)+A/2; representation error is added separately. A conservative gamma
recurrence encloses every original f32 FMA input. Its binade supplies a tighter
per-round half-ulp radius. Every operation used to form intervals is outward.
The cheaper gamma-only variant omits the binade refinement, preserving proof
with more replay. All APIs fail closed on nonfinite/overflow/invalid summaries,
nonRNE, fast-math, wrong IEEE format or disabled gradual underflow.

35 independent native tests cover exact-rational RNE/source `fmaf`, signs,
cancellation, BF16 boundaries, source reduction tails, lossy reconstruction,
underflow and invalid summaries/compilation modes. On all12 pinned heads, the
CPU study independently audited every12,582,912 source QK value and4,718,592 PV
source partials against their intervals, plus source probabilities, alpha and
denominators; all786432 final BF16 words match. Source/fixture SHA receipts live
in the companion Merlin artifact directory. These are proof/functional gates,
not performance projections.

## Complete emitted path and cost boundary

The target uses nine signed radix128 cross-plane GEMMs for3digit reconstruction;
control additionally executes nine absolute-digit GEMMs. Device outputs are
int32; cross-plane recombination is exact binary64 under explicit representable
integer/dyadic range proof. Holder L1/Linf/Cauchy bounds need only row/column
metadata. Cauchy sqrt is factored peroperand row/column, O(M+N), and all norm
reductions are O(MK+NK), not an accidental host matrix contraction.

QK target primitive is1024x1024x64; PV primitives1024x64x512/192/128 derive
resource legality/schedule through the ordinary shape-based generic tuner.
All use stock RV64GC; no RVV and no FSM/LOOP instructions. Every executable
section of finalELF is audited, including dead paths. Main numeric object uses
-O2 -fno-fast-math -ffp-contract=off; startup/syscalls are separately compiled.

Timed ROI includes input packing/metadata, actual device operations, all partial
readbacks, reconstruction, certificates, softmax, source replay, selective
denominator refinement and output stores. SHA/UART/golden auditing are after
ROI. Physical outputs are BF16, losslessly widened to f32le for the standard
digest. Actual complete primitive invocations perhead: control54; candidate63.
CPU `logical_plane_gemms` counts smaller256row/512key conceptual panels; it is
**not the target call count**. Extra sourcepartial device readout is real cost.

## Completed one-head target qualification

Every row below passed all65536 original bits, the immutable gate, independent
native digest, rank0/DONE/process rc0 and fresh all-section finalELF audit.
Receipts preserve exact compiler flags, numeric C closure, primitive object/IR
hashes and original input/golden SHAs. Historical first builds recorded a
mutable builder SHA without freezing the build-time script; this limitation is
explicit in each receipt. Their executed numeric sources/device objects and
ELFs are frozen. Future builds snapshot builder before building and refuse to
overwrite an experiment directory. The v2 historical build JSON mislabeled
its strategy; the strict collector derives/correctly labels it from executed
-DVARIANT=2, pinned driver and stdout, without altering the original receipt.

| Case | Strict Spike instructions | Exact ELF SHA256 |
|---|---:|---|
| v0_h1 | 2,098,108,464 | b9f450f248d91154a328b0e7f367a23637e09ddc24353c3ec10adfcb7f14c1a1 |
| v1_h1 | 3,531,451,594 | 879d989486335d9048c95ca7ac34fc64e689bf4c741de55a943819a7e1926f4e |
| v0_h1_inline | 1,204,116,510 | e41141e3746c1b420a10c27c125b88a764031e1891b315bb64d9b9b042fd0e10 |
| v1_h1_inline | 1,668,502,479 | 1c442fbbda03e3227d0578543dfae3d8553fc68bae078600a3388be09380d734 |
| v0_h1_factored | 1,185,308,895 | e85be94420007b990d87986d017b76f518df81643ab72ec1464cae6622675886 |
| v1_h1_factored | 1,483,561,398 | 4e82885451515ea277cdabd13a09aee4d19e53f1abd3a9866b713847a0938c1b |
| v2_h1_gamma | 1,430,369,073 | f0350a5364892e4fb07fc1d1bd100ddb9ccf94a022a55534940bd769922fa053 |

### Interpretation and promotion

* Strong certificate + libm adjacency:3.531B versus2.098B control instructions;
  **rejected as an instruction improvement**, despite lower device traffic.
* Generic IEEE bit adjacency: candidate1.669B versus1.204B fair control. Generic
  helper improvement cuts original control42.61% and candidate52.75%; value
  equivalence independently tested; fenv/errno diagnostic side effects are
  explicitly outside this certificate contract.
* Factored norms/initial zero summary:1.484B versus1.185B fair control,
  candidate25.16% instruction negative. Preserve the negative rather than
  promote a mathematically tighter bound by replay fraction alone.
* Cheaper gamma:1.430B,3.59% below strong factored candidate,20.67% above control.
  Target QK replay60703/PV10608 versus strong49408/8187 andcontrol58353/4971.
  Candidate int32readback51,904,512bytes versuscontrol84,934,656 (-38.89%).
  Actual hardware cycles can trade compute/traffic differently; qualified fair
  control/candidate pair was sent to queue owner for measurement. **No actual
  FireSim result or winning strategy is claimed here.**
* All twelve heads of the original first block passed native and strict target
  qualification: all 786,432 BF16 bits, original gate0, rank0, DONE and process
  rc0. Exact ELF3fd1427f…3132, marker7d440bea5b6e; complete target
  17,319,092,950 instructions. Stage costs:2.085B packing/metadata,2.387B QK
  device/readback/recombination/norms,7.277B QK certificate/softmax/replay,
  .898B PV device/readback/recombination/norms,1.606B PV certificates,3.064B
  final replay/denominator/output. Actual756 calls and622,854,144 readback
  bytes. This establishes target source accuracy and total cost; it is not a
  fair all-head baseline comparison or hardware cycle result.
* Recovery independently bound exact original output and submitted the fair
  one-head pair to stock FireSim:1894 control /1895 cheaper gamma candidate.
  Durable collectors3383623/3383935, currently queued; no winner claimed.

## Tokens and continuation

`token_usage_available=false`: child goal tools expose no usage counter. Root
records shared campaign goal snapshots (campaign checkpoint19,047,829), which
cannot be assigned exclusively to this optimization or treated as billing.
Input/cached/output/per-agent breakdown is unknown. Each above stage records
hypothesis, ownership, emitted change, actual metric scope and gate/promotion
result. Generic numeric APIs are default-off; there is no production attention
lowering or whole-model integration for this diagnostic yet. A general pass
would need semantic recognition, original reduction/order/numeric contracts,
integer/dyadic range proofs, guarded replay and real target cost selection.
