# Resident input convolution screen

ZIP-derived reference only: our owned `exo_q1013` disassembly analysis,
`q1013_analysis.md` section3.3, describes `fx_conv_30` keeping the complete
padded activation in scratchpad and reusing shifted rows for every tap.
No reference source/private folders were read; no reference layer-cycle
number exists to claim a same-layer comparison.

Current source matmul29 (H14/W14/Cin256/Cout256, stride1) uses flattened
spatial tiles and regathers A for every tap/channel block. The new explicit
`GoldenResidentConv` places each16-channel tile in a complete padded spatial
plane. `CONFIG_LD.block_stride=256` scatters each64-channel input DMA into
four planes. Top/bottom and side zero loads partition scratch memory without
overlapping valid writes. The complete input occupies rows[0,4096); B starts
at8192. Four output-channel tiles across14 rows occupy896 accumulator rows.
The source HWIO reduction order and exact captured scale/ReLU are unchanged.

| Quantity | Control | Resident input |
|---|---:|---:|
| A MVIN commands | 4,880 | 176 |
| Valid activation DRAM bytes | 1,638,400 | 50,176 |
| B MVIN commands | 576 | 576 |
| Compute commands | 29,952 | 32,256 |
| Padded issue cycles | 479,232 | 516,096 |
| GSIM kernel cycles | 675,375 | 533,649 |

This is a **20.98% kernel-cycle reduction** despite7.69% more padded issue
work. The candidate is3.40% above its padded issue floor. Both full capsules
pass all50,176 source-scale output values plus a2,048-byte guard in actual
GSIM and independently in Gemmini Spike; final ELFs contain zero FSM words.
Inputs are deterministic signed-i8 synthetic data, not the captured image.
Whole-model source binding, original-output gates and FireSim still precede
any schedule selection. No selector/default changed.

## Compilation requirement found by the screen

The first small probe failed numerically because the xDSL IR carried
`block_stride=35` but device lowering silently encoded the default16.
`5bdad83` forwards load block stride, pixel repeats and shrink mode and
verifies field ranges; the legacy emitter now forwards its missing layout
fields too. Absent fields preserve existing defaults. Bitfield regressions
and actual small/full target output checks cover the fix. A semantic device
artifact hash is only useful if every selected scheduling field reaches the
instruction encoder; silent attribute omission must fail compiler gates.

The current encoder requires static local addresses, so the resident schedule
unrolls its channel/tap address selection. `kernel.o` grows from82,144 to
154,216 bytes (file sizes, not a dynamic instruction count). This is a concrete
instruction-footprint tradeoff for the whole-model hardware gate. A future
schedule/compiler abstraction for loop-varying local addresses could compact
the body without changing this scratch residency; it needs an explicit typed
operand and correct encoding rather than an ignored scheduling attribute.

## Explicit source-bound composition

`captured_requant_bundle --resident-input-region REGION` is repeatable and
empty by default. It requires an existing exact direct-convolution binding
and a proved removable zero-padding shell. Each selected region retains its
original source/weights hashes, complete scalar transition proof and numeric
contract. Missing/non-direct/refused requested regions fail the bundle build.
The generator then validates resident scratch/accumulator capacity.

The measured-class composition selects matmul29/32/35/38/41 only. Each is
H14/W14/Cin256/Cout256, stride1, i8/ReLU, with its own unchanged proved scale.
It preserves the three already gated banked pointwise bindings. No other
convolution shape is selected merely because it fits in scratch memory.

## Grouped output rows for narrow resident planes

The final-stage width7 source convolution needs a different resident mapping.
One mesh tile per output row would issue7 tiles instead of the flat schedule's4.
The explicit `rows_per_tile=2` mapping uses16 consecutive scratch rows: two
width7 output rows separated by two halo lanes at padded pitch9. Only the valid
lanes0–6 and9–15 are stored to their original NHWC destinations. The final tile
contains one row. Channel/tap reduction order, source scale and zero padding
remain unchanged. Static resource checks admit4×16×16=1024 accumulator rows.

On source matmul51 geometry/scale (H7/W7/Cin512/Cout512), both standalone
capsules pass all25,088 signed-i8 outputs and2,048 guard bytes in actual stock
GSIM and strict RV64GC Gemmini Spike, with zero FSM instructions. GSIM decreases
759,715 to743,898 cycles (2.08%). The padded mesh issue floor stays589,824.
Inputs are deterministic synthetic data; this is not a whole-model measurement.
The candidate's text grows68,980 to310,354 bytes because channel selection is
unrolled. A bounded ordinary CPU channel loop is the next controlled screen.
The default remains one row per tile, preserving already queued device bytes.

Receipt: [grouped resident screen](perf_records/resnet_grouped_resident7_gsim.json).

## Bounded channel loops and device code size

`loop_channels=True` keeps the original increasing HWIO reduction order in
ordinary CPU loops. The already typed dynamic A operand computes the address
`channel_tile*plane + shifted_spatial_row`. Its range is proved from the loop
bounds and encoded row extent. The verifier now uses `a_rows` rather than DIM
for that extent: the stock ExecuteController masks reads beyond the requested
short row count. The width5 tail ends exactly at its98-row resident allocation;
requesting one extra row is refused. Neither instruction encoding nor defaults
change, and replayed H14 and H7 static objects are byte-identical.

| Capsule | Static resident GSIM | Looped resident GSIM | Looped text bytes |
| --- | ---: | ---: | ---: |
| H7/W7/Cin512/Cout512, two output rows per tile | 743,898 | 708,664 | 11,102 |
| H14/W14/Cin256/Cout256, one output row per tile | 533,649 | 521,496 | 9,998 |

Both complete output/guard checks pass in GSIM and strict RV64GC Spike. A
H5/W5/Cin32/Cout19 i32 capsule additionally passes channel, output-channel and
spatial tails. The H7 flat control is759,715cycles, so the combined resident
mapping and channel loop save6.72% on that capsule. H7 text falls310,354 to11,102
bytes. These are synthetic source-shape measurements, not whole-model cycles.

Source-bound bundle callers provide typed `ResidentConvOptions` per explicitly
selected region. Missing selection or untyped options fail before output is
created; row/resource admission still comes from each bound source shape.
Default options keep one row and static channels. Complete original native,
Spike and FireSim gates precede enabling a new composition.

Receipt: [bounded channel loops](perf_records/resnet_resident_channel_loop_gsim.json).
