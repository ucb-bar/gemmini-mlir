# Audited shared compiler changes

Conservative inventory of directly identified implemented change families in the golden compiler integration; related commits counted once. Lower bound, not all repository history, not every detected bug, not a count of commits or performance wins.

**58 implemented families:29 Merlin fixes,21 Merlin reusable improvements,8 model2MLIR fixes.**

The original55-family inventory is published on main. The three additional
generic families are reviewed implementations in open PR41/42/43, with54 tests
independently passed by root; they are not merged main changes or performance
wins. Source-wide interval-table and prepared-owner prototypes remain outside
this conservative inventory until separately audited.

Merlin owned changes were published as13 squashed topic commits to main at7fee5cfdac. Historical source commit hashes below remain provenance; they are not claimed as ancestors after squashing. The [publication manifest](merlin_main_upstream_topics_20261006.json) records actual published topic commits and files. model2MLIR's eight listed fixes are verified ancestors of main3a5acb8fd4c. Fresh whole-model hardware qualification of the new Merlin head remains unknown. Many improvements are explicit alternatives; passing tests do not establish automatic enablement or a measured whole-model gain.

Related exact integer mean/readout work remains grouped with reusable exact readout/guarded decoding family46; rejected numerical prototypes and target-specific changes are excluded.

The scalar LLVM AND/OR/XOR tracer fix is a follow-up to existing family39,
so it does not add another family. It is now published for review in
[PR40](https://github.com/ucb-bar/merlin/pull/40), clean topic `dafe64a4f`
based on main7fee5cfdac. Declared widths1/8/17/64/129 and disjoint/vector
refusals pass27core tests. Main remains unchanged pending review. Source-wide
interval tables and shared prepared-view ownership remain separately qualified
prototypes; this inventory does not silently count them as upstream wins.

| # | Owner | Kind | Change family | Source commits |
| --- | --- | --- | --- | --- |
| 1 | Merlin | bug_fix | Packaged runtime dependencies and paths | 014dd7fdf, b807bdefd |
| 2 | Merlin | bug_fix | Conflicting numerical provider build/workspace identities | 90fd4b66c, ff6f87455 |
| 3 | Merlin | bug_fix | Translation-unit and imported-object filename collisions | fd3069220 |
| 4 | Merlin | bug_fix | Helper and dependent-library link order | 6162c3f6e, 61e2628dc |
| 5 | Merlin | bug_fix | Equal shapes collapsing different device implementations | fd4816a60 |
| 6 | Merlin | bug_fix | Flush-to-zero detection from computed IEEE bits | 53e97343b |
| 7 | Merlin | bug_fix | Unsupported quantization-certificate rounding modes | 9005fc13e |
| 8 | Merlin | bug_fix | Signed-zero endpoint bins | 05f756c77 |
| 9 | Merlin | bug_fix | Conflicting inline/noinline LLVM attributes | 24b1e233c |
| 10 | Merlin | bug_fix | Scalar helper compilation ABI mismatch | 28c801298 |
| 11 | Merlin | bug_fix | Missing linked/compiler/helper bytes in build identity | 795d5378e, 0068b4b79 |
| 12 | Merlin | bug_fix | Bare-metal MLIR assertion runtime support | e5961684c |
| 13 | Merlin | bug_fix | Lost strict floating-point function scopes | a146cbe53 |
| 14 | Merlin | bug_fix | Outlining dropping declared catalog calls | 71bba42b0 |
| 15 | Merlin | bug_fix | Compiler policy lost between benchmark compile and link | 1bc88bf28 |
| 16 | Merlin | bug_fix | Stale success after a refused lowering recipe | e531045ca |
| 17 | Merlin | bug_fix | Poison inputs in conditional clamp guards | 8ead5e46b |
| 18 | Merlin | bug_fix | Numeric LLVM slots versus quoted numeric identifiers | 0829f35cb |
| 19 | Merlin | bug_fix | Uniform matrix-view row-segment coalescing | d43298ff5 |
| 20 | Merlin | bug_fix | Profiling missing typed effects or generic entry interfaces | b5b3664ef |
| 21 | Merlin | bug_fix | Fresh writer symbol collisions | 1d55b7a73 |
| 22 | Merlin | bug_fix | Standard bufferized writer parsing | b6b8a4495 |
| 23 | Merlin | bug_fix | Enclosing numerical and ownership context binding | 098697635 |
| 24 | Merlin | bug_fix | Invalid frontier storage and workspace byte-count overflow | 8948aab92, 39d6db1fd |
| 25 | Merlin | bug_fix | Source helper symbols leaking across providers | be1a6d564 |
| 26 | Merlin | bug_fix | Combined workspace and retained fallback descriptor ownership | ba02bae65, 93cad1c32 |
| 27 | Merlin | bug_fix | Incomplete live-consumer witness admitted for writer binding | 4db3183c2 |
| 28 | Merlin | bug_fix | CCA analyzer facets lost in MLIR serialization | 150f2af09 |
| 29 | Merlin | reusable_improvement | Scalable LLVM loop outlining and exact helper merging | 1287f7b50, f93e22c04 |
| 30 | Merlin | reusable_improvement | Fresh output contracts and final lowered-owner witnesses | 3455418f9, 8c0f2cc8a |
| 31 | Merlin | reusable_improvement | Ordered FMA contraction scheduling without reassociation | b1a4e9001, efc2ca151 |
| 32 | Merlin | reusable_improvement | Independent scalar pointwise packets and broadcast sharing | a22c9d27f, 83677314f |
| 33 | Merlin | reusable_improvement | Proved bounded RNE packets and exact host readout legalization | 27be364d6, 09fae6ebb, 37e0f5031 |
| 34 | Merlin | reusable_improvement | Source broadcast reciprocal-square-root hoisting | 0a5fcec61 |
| 35 | Merlin | reusable_improvement | Borrowed segmented tensor inputs with ownership proof | a8b20d95b, f9e72262d |
| 36 | Merlin | reusable_improvement | Buffer identity exposed before output conversion | efbf3374b |
| 37 | Merlin | reusable_improvement | Contiguous and strided byte-copy specialization | 5586d188d, 281f6ec73 |
| 38 | Merlin | reusable_improvement | Proved uniform fill-copy folding and exterior initialization | c743ceeaa, 615323b85 |
| 39 | Merlin | reusable_improvement | Statically resolvable LLVM control-flow tracing | d27dca7cd |
| 40 | Merlin | reusable_improvement | Byte-identical debug companions for PC attribution | e8efe0a6f |
| 41 | Merlin | reusable_improvement | Effective lowering and compilation recipe recording | e39059379, 7fb0fec7d |
| 42 | Merlin | reusable_improvement | Closed source groups and complete consumer observation DAGs | 9fb517dec, 8d73df233, 119229b8f, db70580b6 |
| 43 | Merlin | reusable_improvement | Caller-owned reusable and pooled provider workspaces | fe3c0751a, 219bcfd41 |
| 44 | Merlin | reusable_improvement | Exact signed BF16 radix packing | d9980988b |
| 45 | Merlin | reusable_improvement | Exact signed-i64 radix reconstruction and initialization | 0b477b892, d24a2c64a |
| 46 | Merlin | reusable_improvement | Portable paired readout enclosures and guarded exact decoding | 4ef27178b, cd6a46eda, 9cd9c7048 |
| 47 | Merlin | reusable_improvement | Source-proved preparation metadata, probability reuse and encoder fusion | f19d3cfc1, 10df0ef11, 2bc211070, a93dda580 |
| 48 | model2MLIR | bug_fix | Half SDPA f32 opmath intermediates | 050009e02 |
| 49 | model2MLIR | bug_fix | SDPA scale, causal options and unsupported-mode refusal | 69c037052 |
| 50 | model2MLIR | bug_fix | Half matmul f32 accumulation and initialized outputs | 46851eaf8 |
| 51 | model2MLIR | bug_fix | Half layer normalization f32 intermediates | a472fef68 |
| 52 | model2MLIR | bug_fix | Half softmax f32 intermediates | d82f4aa9d |
| 53 | model2MLIR | bug_fix | GELU approximation mode and half opmath precision | e9aa9085d |
| 54 | model2MLIR | bug_fix | Half convolution and bias f32 accumulation | b0979bbb7 |
| 55 | model2MLIR | bug_fix | Python scalar precision in half multiplication and division | ba77e6ece |
| 56 | Merlin | reusable_improvement | Exact sequential logical address-region recurrence census with explicit budgets | fb82cb94e, [PR41 pending review](https://github.com/ucb-bar/merlin/pull/41) |
| 57 | Merlin | bug_fix | Emulator output destination validated before expensive execution, including nonregular/ELF alias refusals | 6cb907bc3, [PR42 pending review](https://github.com/ucb-bar/merlin/pull/42) |
| 58 | Merlin | reusable_improvement | Indexed RAW-edge duplicate detection preserves complete graph order while accelerating dependency analysis | f46a5ba66, [PR43 pending review](https://github.com/ucb-bar/merlin/pull/43) |
| 59 | Merlin | reusable_improvement | Explicit immutable base binding preserves public ABI and source operations with frame/context refusals | aa484d509, [PR44 pending review](https://github.com/ucb-bar/merlin/pull/44) |

Latest conservative count:59 implemented families, comprising29 Merlin bug
families,22 Merlin reusable improvements and8 model2MLIR bug families. The55
historical main families and four pending main-based topic reviews are recorded
separately. Unpublished schedule/continuation prototypes are excluded.
[Followup inventory](non_oot_change_inventory_followup_20261006.json).
