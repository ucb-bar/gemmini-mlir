# Resident convolution weight packet issue

The permitted q1013 ZIP's repeated H14/C256 spatial function executes 32,256
preload/compute pairs, with 2,304 FLIP and 29,952 STAY commands. Its B loads are
1,152 packets of 16×32 bytes. Our frozen 1903 implementation performs the same
compute work with 576 packets of 16×64 bytes. It requests fewer total load bytes
(640,000 versus 1,257,472) and retires fewer instructions, yet its historical
stock wrapper interval is approximately 720–724k cycles versus the reference's
485–489k. The reference has different numeric/input contracts and timing scopes;
these intervals motivate a schedule investigation and do not prove causality.

The new explicit `resident_weight_issue_tiles` choice retains the complete
activation and the existing full output-channel panel. A packet contains adjacent
N tiles at one K position. The next packet writes the other weight bank while
the current packet is consumed across all spatial tiles. Its successor can be
another N packet at the same K or the first N packet at the next K. Every output
still follows increasing HWIO K, with the original scale, activation, accumulator
seed, output addresses, and stores. Ordinary bounded K/spatial loops reduce the
emitted command body; no Gemmini FSM instruction is introduced.

The normal capture command exposes `--resident-weight-issue-tiles 2`. Admission
requires an already proved complete resident-input family and disjoint weight
slots. Unsupported layouts or placements retain their existing generator. It
never chooses by workload, source ID, captured values, or expected output. The
option defaults to absent; no shared solver or automatic profitability claim is
attached to it.

## Matched complete capsules

| Same original H14/C256 fixture | GSIM cycles |
|---|---:|
| Frozen original control | 536,180 |
| Existing full-panel next-K prefetch | 499,263 |
| Two-tile packet lookahead, static commands | 492,736 |
| Two-tile packet lookahead, retained K/row loops | 468,824 |

The retained packet arm improves this kernel by 12.56% against its original
control and 6.10% against the existing prefetch family. All 50,176 original byte
outputs, 4,096 exterior guards, and 640,000 immutable operand bytes pass at
common addresses in GSIM and strict RV64GC Spike. These measurements use the
pinned GSIM memory regime and are not whole-model FireSim predictions.

An independent H5/W5/C32/N67 i32 case, with output-channel and grouped-row tails,
passes every output, guard, and input check but regresses from 6,359 to 7,858
cycles. This negative result prohibits a universal performance claim. Complete
source-cell/K-order/lifetime proofs also cover all one- through four-tile packets,
odd/even K loops, and odd/even N packet groups. Normal whole-model accuracy and
controlled hardware qualification are separate gates.

The capsule driver previously hashed its source file at completion. Two early
experimental runs overlapped a constructor-keyword refactor, so their late file
hash is not their loaded Python identity. Their loaded source copies and actual
C/MLIR/object/ELF artifacts are retained in an explicit ledger. The retained
packet arm uses the corrected driver, which records its source digest before
compilation or execution. Previous qualified artifacts remain immutable.
