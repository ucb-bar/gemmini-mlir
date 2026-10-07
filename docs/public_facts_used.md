# Facts this backend was built on, and where each came from

Every ISA / mesh / datapath / encoding decision below is derived from a granted, in-bundle source.
Nothing in this table was guessed, and nothing was read from a golden, a hidden capsule or a prior
backend.

## RTL-derived facts (CIRCT discovery bundle)

Obtained by CALLING the granted APIs, not by reading files:

    python -c "from rtl.facts import load_facts; load_facts('gemmini')"
    python -c "import rtl_backend as R; R.derived_levers(R.target_profile('gemmini'))"

| fact | value | where it is used |
|---|---|---|
| mesh (`arrays[0]`) | 16 x 16 `Tile`, 256 instances, corroborated | `target/facts.py:DIM` -- the tile edge of every contraction |
| operand store (`memories.scratchpad`) | 262144 B | `facts.SP_ROWS` = 16384 rows of DIM i8 |
| accumulator (`memories.accumulator`) | 65536 B | `facts.ACC_ROWS` = 1024 rows of DIM i32 |
| datapaths | input `i8` (scratchpad `UInt<8>`), accumulator `i32` (`AccumulatorMem SInt<32>`) | the i8 x i8 -> i32 contraction and the two readout widths |
| legal functs (`decoder_icmp_fanout`) | `{0..24, 126}` | `facts.LEGAL_FUNCTS`; `isa.Instr.funct` refuses anything else |
| derived levers | `spatial.dataflow`, `spatial.accumulator_resident`, `memory.capacity_fit` | the three `optimization_surfaces` in `manifest.yaml` |

`action_catalog.escalation_ladder("spatial.dataflow", "gemmini")` returns one rung, HEURISTIC, whose
seam is "the generated OOT backend's command/tile-program emitter". That is exactly
`mlir_oot/lowering/schedule.py` + `mlir_oot/lowering/contraction.py`, so the lever was built there.
`cca_contract.check_bijection("gemmini")` reports `clean: true` with no orphan fields and no orphan
routes, so no additional axis was owed and none was invented.

## ISA facts (shipped headers + RTL)

| fact | source | use |
|---|---|---|
| funct table (`k_CONFIG 0` .. `k_COUNTER 126`) | `isa_include/gemmini.h` | generated into `mlir_oot/tables/funct_table.py` by `mlir_oot/tables/gen_funct_table.py`; never typed in |
| CONFIG subtype selector (`CONFIG_EX 0`, `CONFIG_LD 1`, `CONFIG_ST 2`) | same header | the low 2 bits of a CONFIG's rs1 |
| rs1/rs2 field packing of every instruction | the `gemmini_*` macros in `gemmini.h` | one packer per class in `mlir_oot/target/isa.py`, each naming its macro |
| RoCC envelope | `XCUSTOM_ACC 3` -> opcode `0x7b`; xd=0/xs1=1/xs2=1 -> func3 `0x3` | `facts.ROCC_OPCODE` / `ROCC_FUNC3`; cross-checked against `isa_tools.py asm` output |
| local address layout | `rtl/gemmini/LocalAddr.scala` (`is_acc_addr`, `accumulate`, `read_full_acc_row` from the MSB down) | `facts.acc_addr()` |
| DMA burst limit | `gemmini_params.h`: `MAX_BYTES 64`, `MAX_BLOCK_LEN = MAX_BYTES/(DIM*1)` = 4, `MAX_BLOCK_LEN_ACC = MAX_BYTES/(DIM*4)` = 1 | the `cols <= DIM*MAX_BLOCK_LEN` verifier on every move, and the bound on one coalesced block transfer |
| Block-transfer destination stride | `LoadController.scala` (`block_strides` is a CONFIG_LD rs1 field, bits 31:16, held in a `Reg` with no reset) + `DMA.scala` (`reserve.entry.addr := req.spaddr + req.block_stride * block`) | block `b` of a transfer lands at `spad + DIM*b`, which is the stride `isa.config_ld` declares and the pitch `schedule.Blocking.a_row`/`b_row` already use |
| Accumulator block width | `DMA.scala`: `has_acc_bitwidth` selects `accWidthBytes` over `spadWidthBytes` when dividing `bytesRequested` | an accumulator destination carries a 4-byte element, so `coalesce` never merges into one |
| which readout carries an epilogue | `rtl/gemmini/AccumulatorScale.scala` (activation, then f32 scale, then clip to the operand width) and `full_data` bypassing both | `lowering/epilogue.py`: an i32 readout cannot carry relu/acc_scale, and says so by name |
| fused pooling geometry | `rtl/gemmini/StoreController.scala` (`pool_row_addr = localaddr + orow*ocols + ocol`, `pool_vaddr = vaddr + (porow*pool_out_dim + pocol)*stride`, out-of-range windows made garbage) | the pooled store in `lowering/contraction.py` |
| load-unit state | `rtl/gemmini/LoadController.scala`: LOAD/LOAD2/LOAD3 select register set 0/1/2 of one controller; `has_acc_bitwidth = is_acc_addr && !shrink` | the A/B unit split, and widening an identity move through the accumulator |
| activation placement | `ExecuteController.scala` applies `config_ex`'s activation only on the path to the SCRATCHPAD | a weight-stationary result lands in the accumulator, so relu is a readout stage, not a mesh stage |
| the weight-stationary instruction sequence | the explicit `sp_tiled_matmul_ws` body in `gemmini.h` | the PRELOAD / COMPUTE_PRELOADED / COMPUTE_ACCUMULATE order, the accumulate-bit rule, and the partial-tile row/col fields |

## Contract facts

| fact | source |
|---|---|
| the interface grammar and every op's attributes | `contract/interface_grammar.md`, `contract/merlin_iface.irdl.mlir` |
| the command-buffer opcode/epilogue/role vocabularies and the admission partition | `contract/command_buffer_abi.yaml`, `contract/schemas/command_buffer.schema.json` |
| the four CLI entrypoints, the manifest shape, and the kernel pointer ABI per command shape | `contract/mlir_oot_backend_contract.yaml`, `contract/schemas/manifest.schema.json` |
| the oracle ladder and the per-capsule numeric policy | `contract/oracle_runner_contract.yaml` |

## Tool cross-checks

`python isa_tools.py asm` was used once, on a hand-written listing, to confirm the canonical
`.insn r 0x7b, 0x3, <funct>, x0, $0, $1` form and the `llvm.mlir.constant` operand shape this
package emits. `disasm` and `lint` are run on the emitted artifacts before every self-check: the
decoded operand fields (DRAM argbase + offset, row pitch, spad/acc address, readout width, acc
scale) are read back against what the command buffer declares for the same command.

## Round 9 additions

| fact used | where it came from |
|---|---|
| the mesh transposes its STATIONARY operand on read, selected by `b_transpose` at rs1 bit 9 of the EX config | `gemmini.h` line 267 (`RS1: … [9] b_transpose | [8] a_transpose …`) and its `gemmini_extended3_config_ex` packing; the same bit `sp_tiled_matmul_ws` uses when it addresses `B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) : …` |
| a transposed stationary operand is stored row-per-output-column (`[N, K]`), so `X @ X^T` needs no staged copy of X | derived from the addressing above; already relied on by `attention_qk` via `contraction.transposed_b` |
| the mesh reads i8 operands into an i32 accumulator, so an i8 sum of products is exact | `rtl.facts.load_facts('gemmini')` → datapaths `input i8 (scratchpad smem UInt<8>)`, `accumulator i32 (AccumulatorMem SInt<32>)` |
| the band edge the reduction is emitted at is the array edge, 16 | `rtl.facts.load_facts('gemmini')` → mesh `DIM = 16`; reaches the pass as `facts.DIM`, never a literal |
| f32 represents every integer below `2**24` exactly, which bounds when a mesh i32 reduction and the f32 one it replaces must agree | IEEE-754 binary32 significand width (24 bits); combined with the declared i8 container's range (`|x| <= 128`) to give the `k <= 1024` guard in `sumsq.plan` |
| `AccumulatorScale` instantiates no normalizer in this elaborated design, so the SCALE has no store-path encoding | the RTL's `has_normalizations` parameter left at its default — the same fact `lanes.NON_READOUT_FAMILIES` is derived from |
