# Independent adjacent CPU rounding packets

Verified stock TinyLlama job 1880 completes all 22 layers/eight tokens in
**531,072,370 cycles**, preserving all 256,000 original compiled words and the
original Torch elementwise gate. Four-lane pointwise arithmetic preserves that
gate in native and strict Spike, but its original source capsule is slower than
two lanes. It remains held. Instructions do not decide Rocket latency.

## Generic compiler change

Merlin commit `94053801f` introduces an explicit CPU legalization of two through
four adjacent bounded-RNE results. Every lane must retain the full original
typed clamp/truncation/fraction/parity/sign proof. Both raw inputs must be
defined before insertion in the same basic block. Adjacency forbids crossing
any memory operation, unknown call, label or intervening statement. Existing
source arithmetic and other users remain intact. Strict/constrained floating
point refuses; exception flags remain unobserved under the explicit host policy.

An RV64GC packet emits independent scratch lanes: all maxima, all minima, then
all ties-even conversions. Normal Merlin compilation owns object and build
identity. The paired portable LLVM is independently executed against the
original source and original tensors. No accelerator/workload/provenance
identifier selects production behavior; applicability follows SSA and numeric
contracts. Device instruction policies and actual target qualification stay OOT.

The actual winning LLVM contains 22 eligible pairs, with 44 converted scalar
sites, among 111 total bounded-RNE sites. The source before the late callback is
byte-identical to 1880; all device/runtime/weight objects remain identical.

## Scoped evidence

| Scope | Before | After | Gate and decision |
| --- | ---: | ---: | --- |
| 2049-element source arithmetic, warm actual serial-clock Rocket GSIM | 256,637 cycles | 247,568 cycles | All 2049 bytes + 128 guard bytes exact; focused 3.5338% gain |
| Same capsule, functional Spike | 108,797 instructions | 108,797 instructions | Instruction counts unchanged; no cycle projection |
| Whole pointwise width 4 screen, functional Spike | 166,605,674 instructions | 166,863,672 instructions | All 256,000 words/Torch exact; held, no stock run |
| Current 1880 attribution | 531,072,370 stock cycles | Pending stock 1901 | Exact object reuse, all 155 calls and interval conservation |
| Whole adjacent-RNE candidate | 531,072,370 stock cycles | Stock timing pending | Full native and strict Spike all 256,000 words/Torch exact; identical 166,605,674 instructions |

Independent boundary tests preserve the original source, scalar legalization
and four-lane packet for all 792 words under all five actual RISC-V rounding
modes. Host source/portable parity also passes all four supported host modes.
The 42 focused compiler tests plus no-target-name/no-regex/structure gates pass.
No whole-model or stock-cycle improvement is established for this new packet yet.

Records: `perf_records/tiny_rne_basic_block_capsule.json`,
`tiny_pointwise_packet4_whole_screen.json`, `tiny_current1880_profile_spike.json`
`tiny_rne_basic_block_whole_spike.json` and `tiny_next_host_optimization_journey.json`.
Source-qualified artifacts remain
under `out/artifacts/probes/tiny-pointwise-packet` and
`out/artifacts/probes/tiny-rne-basic-block`. Recovery owns stock queue admission.

Token accounting is unavailable to this child agent. Per-optimization exclusive
tokens, input/cached/output splits and billing cannot be reconstructed. Root
maintains shared campaign snapshots; the journey records this limitation.

## Build infrastructure evidence

A fresh isolated Merlin worktree correctly refused exact source placement when
its cold RTL capability cache could not derive the datapath. The qualified
build explicitly bound the same facts artifact used by 1880 through
`MERLIN_RTL_FACTS`, SHA256
`e1b48268e2c72886032b51249b78c33e958ed6fb5279e7db5ad46b8c299eadb2`.
The source LLVM and all unchanged object hashes subsequently matched 1880.
The refusal is retained in the experiment log; no legality check was relaxed.

Reproducible device compilation needs the selected provider's capability facts
and execution support in its build closure. A cache's incidental presence in
another worktree must not supply an implicit capability. Generic build
orchestration and receipt pinning belong Merlin; provider capability derivation
and target execution support belong OOT. The new optimization itself changes
only the host callback and grants no new device legality or accuracy policy.
