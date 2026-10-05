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
