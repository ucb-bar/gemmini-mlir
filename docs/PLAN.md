# PLAN — gemmini out-of-tree MLIR target backend (xDSL / Python arm)

Design contract with myself. Written before any code. Refined, not rewritten, on later rounds.

## 1. Corpus (what must pass)

Discovered from `isa/`, `layers/`, `model/`, `model_slices/` (121 public capsule dirs on disk;
the launch block declares 103 required). Two *input grammars*, and they need different machinery:

| grammar | count | families | numeric policy |
|---|---|---|---|
| `merlin_iface` v0.1 | 90 | isa/, layers/, 9 of model_slices/ | `exact_int` |
| `linalg-on-tensors` | 31 | model/ (10 whole models), 21 model_slices/ | `tolerance_float` |

Distinct cases in the `merlin_iface` half (the exact-integer half, and the bulk):
- `matmul` on a `resident_pack`ed weight + `commit` (59), K/M/N from 2x4 up to 8192x32x32.
- `conv2d` whole-op, weight pre-im2col'd `[Kh*Kw*Ci, Co]`, NHWC ifm, stride 1/2/4/8/16,
  padding 0 and 1, dilation 1 and 2, kernel 1x1..16x16 (20).
- `movement` identity round-trip i8->i8 / i8->i32 (6).
- `matmul_batched` rank-3 (2), `attention_qk` (1).
- Epilogues actually present: `[]`, `[relu]`, `[acc_scale]`, `[acc_scale,relu]`,
  `[bias_add]`, `[bias_add,acc_scale,relu]`, `[maxpool]`. `requant` is in the ABI vocabulary but
  in no public capsule — implement it generally anyway (`requant_shift` is per-capsule, never 4).
- `output_dtype` is `i32` (full accumulator readout) or `i8` (scaled/clamped readout). Every
  non-empty epilogue in the corpus commits `i8`; that is a consequence of the datapath, not an
  assumption I may bake in.

`linalg-on-tensors` half: float (bf16/f32) softmax / layernorm / gelu / geglu / pooling /
depthwise / attention plus 10 whole models. These contract a `tolerance_float` golden against the
**oracle only** — the command-buffer self-consistency cross-check reports `not_applicable`. They are
therefore a whole-program codegen problem, not a command-buffer problem.

## 2. Input ingestion (`parse`)

**Structural, via xDSL. No regex anywhere in the package** (checked on the submission).

- `merlin_iface`: I define the input dialect as real xDSL IRDL ops + types
  (`mlir_oot/frontend/iface_dialect.py`) — `tensor`, `resident_pack`, `matmul`, `commit`, `evict`,
  `conv2d`, `movement`, `matmul_batched`, `bias_add`, `attention_qk`, `attention_pv`, `k_chain`,
  `residual_add`, `depthwise_conv2d` — with operand/result constraints and verifiers, and parse with
  `xdsl.parser.Parser`. A broken graph raises at parse, which is the whole point of using a dialect.
- `linalg-on-tensors`: parse with xDSL's builtin/func/linalg/tensor/arith/math dialects and
  `allow_unregistered_dialect` for the `quant_ext.*`/`prov.*` producer extensions.
- `parse --verify-diagnostics` = parse + `module.verify()` + my own contract checks
  (version must be `0.1`; attribute arity/type for geometry lists; epilogue-stage parameters
  present). Nonzero exit on any diagnostic.

Everything downstream reads a target-agnostic `Workload` model (`mlir_oot/ir/workload.py`) built
from the verified IR — never from the text.

## 3. Target dialect + lowering

`mlir_oot/target/dialect.py` defines a real xDSL `gemmini` dialect, one op per ISA instruction
class, each with a verifier that range-checks its fields against the RTL-derived facts:
`gemmini.flush`, `.config_ex`, `.config_ld`, `.config_st`, `.mvin` / `.mvin2` / `.mvin3`,
`.preload`, `.compute_preloaded`, `.compute_accumulated`, `.mvout`, `.fence`.

`mlir_oot/lowering/` is the xDSL rewrite-pass pipeline:
1. `iface_to_gemmini` — pattern rewrites, one per interface op, producing gemmini-dialect ops.
2. `schedule.py` — the tiling/residency *heuristic*: chooses the M/N/K tile walk, which operand
   stays resident, and the scratchpad/accumulator assignment, from the derived capacities
   (operand 262144 B, accumulator 65536 B, DIM 16). This is the declared optimization surface.
3. `epilogue.py` — maps the commit's epilogue list onto the readout path (`config_st`
   activation + acc_scale + pooling geometry, and the `read_full_acc_row` bit).
4. `conv.py` — im2col: the generic conv→matmul reduction, contracted **one kernel tap at a time**
   (K-tile width = Ci) so that each in-bounds run of output pixels is a plain strided `mvin` and
   out-of-bounds taps simply contribute nothing. No host gather, no scratch DRAM tensor.

## 4. Encoding

Derived, never invented: funct values come from the shipped `gemmini.h` `#define k_*` table, field
packing from the `gemmini_*` macros in the same header, and the mesh/capacity numbers from
`rtl.facts.load_facts("gemmini")`. `mlir_oot/target/isa.py` holds ONE packer per instruction class,
each a transcription of the header macro it names in its docstring.

Word format (derived from this target's decoder): `funct[31:25] rs2[24:20] rs1[19:15] xd[14]
xs1[13] xs2[12] rd[11:7] opcode[6:0]`, emitted as the canonical
`.insn r 0x7b, 0x3, <funct>, x0, $0, $1` with `"r,r"` and **two SSA operands**, every immediate a
`llvm.mlir.constant(... : i64)` and every DRAM address `llvm.ptrtoint` of the matching kernel
pointer argument (+ a constant tile offset via `llvm.add`). Never an inline integer literal.

Check before grading: `python isa_tools.py lint` + `disasm` on the emitted `.mlir`, and reconcile
each decoded operand field against what the command buffer declares for the same command.

## 5. Addressing + termination

- On-chip: scratchpad row addresses are plain constants; accumulator addresses set bit 31
  (`is_acc_addr`), bit 30 (`accumulate`) and bit 29 (`read_full_acc_row`), per `LocalAddr.scala`.
- Off-chip: **every** DRAM address is `llvm.ptrtoint` of the kernel pointer argument for that
  tensor plus a constant byte offset. No literal DRAM address anywhere.
- Kernel signature follows `kernel_abi.arg_order_by_command_shape`, matched top-down:
  whole_program → movement → native_whole_op → resident_matmul.
- Termination: `gemmini.fence` (a bare `fence`) then `llvm.return`.

## 6. Verification loop (cheapest first)

1. `python -m mlir_oot.selftest` style local run: the 4 entrypoints over every capsule, command
   buffer validated against `command_buffer.schema.json` — seconds, no simulator.
2. `python isa_tools.py lint` / `disasm` on the emitted `.mlir` — instant, catches encoding errors.
3. `python3 agent_selfcheck.py --submission submission --capsules <the one I changed>`.
4. `--shape-coverage` and `--offload-census` (emit-path only, so run them often).
5. `--capsules all` at convergence milestones only. `python await_verdict.py` to wait for a grade;
   never a poll loop.

## 7. Route discipline

Every program lands in exactly one of A (accelerated), H (declared host lane, with a reason the
target's own readout facet does not contradict) or D (explicit `declined` with a reason). Never an
empty command list with no statement.

### 7a. Refinement (round 1e) — the plan the rounds actually forced

Two things the original plan got wrong, both corrected rather than re-planned:

* **Route H is a placement, not a refusal to compute.** A declared host lane still has to write the
  tensor the program commits, so both grammars now COMPILE their host regions. The
  `linalg-on-tensors` half unrolls (`codegen/host_lane.py`); the `merlin_iface` half emits a loop
  nest (`codegen/scalar_lane.py`), because a contraction's element count is the payload and an
  emitted program must not grow with it. The route is chosen by `lowering/lanes.mesh_refusal`,
  asked *before* the rewrite so a region arrives at the lane as the interface wrote it.
* **The routing plan is an output, not bookkeeping.** `params.lane_placement` now names the lane of
  every region, accepted or refused. Where the work went is a statement only the compiler can make.

Verification gained a third rung, below the two that already existed: `devtools/simlane.py`
interprets the emitted scalar-lane kernel over harness-laid-out buffers and checks it elementwise
against numpy. The order is now simlane / simulate (seconds, no simulator) → `isa_tools lint` +
`disasm` → `agent_selfcheck` on the changed capsule → the official verdict.

### 7b. Refinement (round 4) — route X, and what a "declared intermediate" actually means

The plan's route table had four rows (A mesh / Q quantise / H host lane / D decline). It needed a
fifth, because a whole class of capsule fits none of them: a program that INTERLEAVES a region this
readout cannot execute with a contraction that belongs on the array. Route H computes it while the
array issues nothing; route A has no encoding for the off-mesh region; route D refuses work the
hardware can carry. All three are the wrong answer.

* **Route X — one kernel, two lanes.** `lowering/hybrid.py` splits the program into consecutive
  same-lane stages read off its own operation order. The mesh stages lower as they always did; each
  off-mesh stage becomes a `gemmini.lane_stage` marker holding its POSITION in the command stream,
  which the code generator expands into a call to a stage function it compiles alongside the kernel.
  Memory between the lanes is ordered by the same DRAM dependency pass every route now runs.

* **Route W — a contraction whose operand is wider than the operand port.** `lowering/widen.py`
  splits it into balanced radix-256 digits the port DOES encode, issues one contraction per digit,
  and recombines with the implied shifts. Exact, because the residual is zero in the output's own
  container. This is the general form of "the array can only multiply bytes" and is not specific to
  any capsule.

**The plan's biggest wrong assumption, now corrected.** Section 1 treated each interface operation
as a separately-defined function composed by the interface. For a FUSED operation that is false: the
capsule's own definition computes the whole chain and puts it in a container ONCE, at the end. The
tensors the interface declares between the stages describe BUFFERS, not roundings. Measured, not
assumed: rounding rmsnorm's intermediate into its declared i32 container before the contraction
disagrees with the reference on 250 of 256 elements by up to 13.

So the compiler must reach the unrounded definition without a float operand port. Two reassociations
do it, and both are the same idea — a per-ROW scalar commutes with a contraction that sums over
COLUMNS:

* `lowering/rowscale.py` — hoist rmsnorm's row RMS past the contraction, leaving an exact integer
  contraction of two declared i8 tensors and one division per output element.
* `lowering/softmaxfuse.py` — the softmax denominator is the same kind of row scalar, but its
  numerator is an exponential and never integer, so that contraction is DECLARED on the scalar lane
  while the query-key product it feeds from stays on the array.

Verification gained one more rung at the top, and it is the one that found the real bug twice:
reproduce the capsule's DECLARED stimulus with `Tensor.deterministic`, compute the candidate
definitions, and compare them TO EACH OTHER against the redacted verdict's own
(`mismatch_count`, `max_abs_diff`) pair. No golden is read; the disagreement pattern identifies the
definition.

## Refinement, round 9 — the reduction is separable from the scale

The plan above treated a normalisation as one indivisible off-mesh region, because the RTL says the
store path has no reciprocal square root. That inference is sound for the SCALE and was over-applied
to the whole region. A normalisation has two parts with different hardware needs:

* the **reduction** `sum_k x[i,k]**2` — a sum of products of two declared i8 operands, which is
  exactly what the weight-stationary mesh contracts, and
* the **scale** `x / sqrt(mean + eps) * gamma` — the part that actually needs arithmetic this
  readout does not implement.

So the routing question is per-STAGE, not per-region. `lowering/sumsq.py` (route S) issues the
reduction on the array as the diagonal of `X @ X^T` — whose stationary operand under the mesh's
`b_transpose` bit is X's own row-major layout, so both operand ports read the same declared tensor
and nothing is staged first — and leaves only the scale on the scalar lane, declared there. It is
emitted one `DIM`-row band at a time, so the intermediate is linear in the row extent.

This generalises the principle already in `rowscale.py` and `softmaxfuse.py` (a per-row scalar
commutes with a contraction that sums over columns): there, the row scale was hoisted past a
contraction the capsule already contained; here, the reduction that PRODUCES the row scalar is itself
recognised as a contraction. The same reading applies to any row-reduction-then-row-scale family, and
is the documented route for the scalar-lane weighted-sum loop in the attention_mx family.

Route S is a new rung of the driver's route ladder rather than an edit to a shared emitter, so a
refusal falls through to the host path the affected capsules already passed with. That is the
verification strategy for any future stage-level routing: add a rung, diff every emitted artifact
byte-for-byte to prove the blast radius, then screen only what moved.
