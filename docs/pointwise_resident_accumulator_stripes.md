# Complete input and bounded output stripes

This screen reuses the previously archived `GoldenResidentStripeGemm` prototype. It keeps complete A in K-major scratchpad planes, loads one output channel group's full reduction B, and computes bounded M stripes in the accumulator. Every output consumes the same increasing K order and exact original readout. The prototype is an explicit alternative; the ordinary source compiler remains unchanged.

The typed M784/K128/N512 source control stores one output tile per command because complete A requires BM49 and accumulator capacity allows BN1. BM16/BN4 stripes reserve 6,272 A rows in banks 0/1, 512 full-K weight rows in bank 3 and 1,024 accumulator rows. There is no overlapping live storage. Complete A survives every channel group; B survives all M stripes for its current channel group and is overwritten after their readouts.

## Actual emitted work

The complete source-bound CFG census proves stores decrease from 1,568 to 392 and input loads from 648 to 162, using width 64 rather than width 16. Requested load payload remains 165,888 bytes and requested store payload remains 401,408 bytes. Compute/preload pairs remain 12,544. Real mesh weight preloads increase from 256 to 1,024. Counts describe emitted work; physical DRAM traffic, bank service, overlap and performance are unknown until measurement.

Fifteen checks close resource partitions, exact DMA endpoint bounds, accumulator capacity, source reduction order for every output tile, independent nonsquare/row/column/K tails and invalid seed/resource/stripe refusals. Six existing dense kernel objects byte reproduce, so ordinary defaults are unchanged.

The independent M521/N73/K65 i32 amplitude-21 pair closes all 38,033 outputs, 4,096 dirty guard bytes and 38,610 immutable input bytes on strict Spike and pinned GSIM. Control takes 50,144 GSIM cycles; stripes take 34,148. The original source-bound M784/K128/N512 pair now closes every401,408i8outputs,4,096guards and165,888immutableinputbytes:298,600 to251,981GSIMcycles (15.6125% lower). StrictSpike andfinalzeroFSMaudits pass. A second independentM37/N69/K33 signed-i8tail pair closes all2,553outputs and improves2,898 to1,977cycles. No whole-model, automatic-policy orstockperformanceclaim follows fromthese matchedcapsules.

The earlier M3136/N256/K64 experiment remains rejected: its control already used wide stores and the stripe alternatives increased cycles by 48–51%. This distinct hypothesis starts with actual narrow stores. It does not erase that negative evidence or infer profitability from residency alone.

[The capsule receipt](perf_records/pointwise_resident_acc_stripe_capsules.json) closes all artifacts. Exact scoped Spike opcode histograms independently confirm executed load/store/compute command multiplicities. Original kernel retired instructions decrease from61,441 to46,349 and touched instruction bytes from10,552 to9,510; these are observed functional/code witnesses, not a targetcycle model.
