# Paired readout with source-stride resident planes

Owner: `reference_parity`. Source change: `101fd4c`. All target changes are in
the OOT backend; the shared complete numeric certificate and decoder are unchanged.

The existing source-stride schedule keeps the complete padded input plane resident
and derives `A_stride` from the typed convolution stride. The paired readout already
stores two byte predictions before reusing accumulator storage. This change permits
their composition when the existing conservative i32 prefix bound fits the complete
paired certificate and both output lifetimes remain private and disjoint.

The source stride, scratch layout, weight panel, accumulator bound, and original
HWIO reduction order remain verified. Plain channel-plane paired schedules continue
to refuse. The default generated kernels are unchanged. The analytical ranking is
an issue/transfer estimate, with no hardware cycle claim.

## Complete matched capsules

| Scope | Control GSIM cycles | Source-stride cycles | Change |
| --- | ---: | ---: | ---: |
| Original H28/W28/C256, stride2, paired producer and decoder | 1,038,114 | 893,198 | −13.96% |
| Independent H5/W7/C32/N67, stride2, spatial/channel tails | 13,343 | 14,913 | +11.77% |

The timed interval includes the producer, both stores, final fence, and exact
decoder. It excludes ranked descriptor checks, allocation, and dispatch. Common
linked operand/output addresses were checked. The original capsule verifies every
100,352 output bytes, 8,192 exterior guard bytes, and 790,528 unchanged input bytes.
Strict Spike and final zero-FSM audits pass for both capsules. The independent
negative remains part of the evidence; there is no automatic profitability promotion.

The original fixture is a retained, qualified 1897 capture whose source numeric
contract matches the unchanged current producer. Captured values do not establish
eligibility or replace conservative source bounds. The actual target CFG proof
checks source cell addresses, increasing K, storage ownership, both output passes,
and final fence. It first requires byte identity with the reconstructed typed target
module. Initial parser representation and test expectation failures are retained.

## Current whole-source qualification

The stock control is job2013, **32,553,639 cycles**, ELF
`2757df664036fd2c9c4dc86951ac69c1b042af62f12cbba66984cff8c540411e`.
The six-stage partial-link graph reproduces this ELF byte for byte before substituting
one source-qualified producer. The selected source-stride kernel is byte-identical
to the measured original capsule. Other 51 kernels and all 52 adapters remain
unchanged, as do all three segmented-A consumers, paired-flat routes, five resident
weight packet routes, old39 residuals, stem, CPU mean, original host, weights, and
controlled runtime.

Both the fresh normal route and the controlled link pass native and strict Spike
checks against all 1,000 original binary32 words at `atol=rtol=0`; both final ELFs
have zero FSM instructions. The normal host object and selected device aggregate
are byte-identical to the controlled objects. The fresh normal runtime is preserved
as a separately qualified artifact; the controlled arm uses the actual 2013 runtime.

The candidate ELF is
`04044f0bde47ab2a358bc77bd8de7c00e333fe3a640a2d0e06885a40ba3b66ab`.
Its inherited build marker is nonunique; the final ELF hash is authoritative.
The 7,306 source/artifact pins reclose, and 43 focused checks pass. Whole-model
hardware performance remains unknown pending parent review and the sole queue
owner's admission. No capsule gain is added to or projected onto whole-model cycles.

Receipts: `paired_source_stride_capsules.json` and
`resnet_paired_source_stride_current2013_qualified.json`. Reclose with
`out/paired_source_stride/normal_source/reclose.sh` in the owning worktree.
Subagent goal token counters are unavailable and are explicitly recorded as such.
