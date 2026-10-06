# Bound the resident output panel, then measure it

The source-stride resident layout for a 14×14/Cin512/Cout512 stride 2
convolution needs 8,192 input rows and seven output row groups. The previous
16 tile channel panel exceeds the 1,024 row accumulator. Complete resource
bounds allow nine tiles; eight minimize wide weight/output transfer commands.
An independent 17×13/Cin32/Cout271 case derives seven tiles and validates
17,073 int8 outputs plus 2,048 guards in GSIM and strict RV64GC Spike.

The matched original-input test rejects the simple smaller-panel schedule:

| Kernel capsule | GSIM cycles | Original outputs and guards |
| --- | ---: | --- |
| Exact 1897 flat control | 774,520 | All 25,088 int8 and 4,096 guards pass |
| Resident input, bounded panel 8 | 1,034,092 | Same complete values pass |

This is **259,572 extra cycles (+33.52%)**. Both arms use common addresses,
retain the source reduction and store arithmetic, pass strict RV64GC Spike and
have zero FSM instructions. The existing serialized issue/traffic estimate
also rejects this family, so the ordinary selector keeps the flat control.
There is no whole-model build or FireSim submission for this negative branch.

The resource helper records the maximum legal panel, full A/B reservations,
accumulator footprint and transfer command ranking. It retains the supplied
panel when legal; a retile changes only the channel panel field. Legality and
fewer transferred bytes do not imply faster execution.

The naive resident schedule issues 64,512 computes, versus 36,864 in the flat
control. The owned reference executable also issues 36,864 computes for this
geometry; it uses mesh stride 2, plane stride 320 and known 16 row computes.
Its unresolved addresses and different numeric contract prevent a full
address or semantic equivalence claim. The next hypothesis is to deinterleave
input rows by source-stride residue, allowing multiple output rows in one mesh
tile while retaining the source HWIO reduction order.

Raw fixtures, exact command traces, failed diagnostic wrapper scope and complete
results are recorded in
[source_stride_bounded_bn_rejected.json](perf_records/source_stride_bounded_bn_rejected.json).
