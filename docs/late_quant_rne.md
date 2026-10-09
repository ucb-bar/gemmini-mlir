# Late bounded integer rounding for RV64GC

Reusable recognition, CPU instruction emission, portable native oracle and the
normal pre-object callback now live in `merlin.llvmlower.late_quant_rne`. Its
default policy leaves source bytes unchanged; explicit `host_isa="rv64gc"`
selects CPU legalization independently of the accelerator. The OOT API delegates
to this owner and retains historical SSA printer identity. The legacy
`python -m mlir_oot.late_quant_rne` relink/audit tool remains target-owned.

The matcher requires this complete binary32/i8 SSA contract:

1. `llvm.maximum.f32(raw, -128)` then `llvm.minimum.f32(...,127)`.
2. Truncating `fptosi i8`, exact `sitofp`, subtraction and absolute fraction.
3. Fraction greater than one-half, or exactly one-half with odd truncated int.
4. Signed ±1 bump according to the clamped value's sign; integer result add.

Only that final result changes to `fcvt.w.s ..., rne` followed by i8 truncation.
Intermediate source values remain available to every other user. LLVM's
optimizer removes dead computations. Each function has its own SSA scope;
LLVM assembles/verifies both complete source and rewritten modules before
compilation. Changed bounds, types, predicates, connections or parity refuse.
Strict/constrained-FP modules refuse conservatively. Source/object/ELF hashes
and the actual compiler/linker commands are recorded.

## Numerical proof

On every source-defined input, the clamp is finite in [-128,127]. i8-to-f32
conversion is exact. The fractional subtraction is exact: when magnitude is
at least1, truncation and its input satisfy Sterbenz's bound; below1 subtraction
is by zero. Thus source parity/half comparisons implement mathematical nearest,
ties-even rounding without dependence on the current floating rounding mode.
Explicit instruction `rne` implements that same integer result and does not
modify `frm`. The rounded value fits i8, so the final truncation is exact.

±Infinity clamp to the corresponding finite endpoint. NaN propagates through
`llvm.maximum/minimum` and the source `fptosi` produces poison: no assumption
that runtime inputs are finite is introduced, and no NaN output contract is
invented. The source LLVM operations are unconstrained and expose no floating
exception-flag contract; this optimization does not promise fflags preservation.
The separate strict `roundevenf_rv64gc.S` helper retains its existing contract.

The native oracle replaces the same proven chain with `llvm.roundeven.f32`
and `fptosi i8`; it preserves surrounding floating operation order and layout.

Boundary probe: 21,483 binary32 values, including every half-step and both
adjacent representable neighbors through the clamp interval, signed zero,
subnormals, infinities and random finite bit patterns. All107,415 results
across the five legal RISC-V frm modes match independently computed integers;
frm stays unchanged. Final RV64GC ELF passes the no-FSM audit.

## Whole-model gate

Two source-bound quantizer chains are selected. Native oracle and actual
Gemmini Spike reproduce all1000 closed-capture golden outputs bitexact, with
zero rank mismatches and final zero-FSM. Spike retired instructions improve
14,254,689 →12,271,325 (13.91%). These are functional instruction counts,
not hardware cycles. All other linked model/device/runtime/weight objects
retain their pinned identity. Receipt: `docs/perf_records/late_bounded_quant_rne.json`.
`tests/probe_late_quant_whole.py` validates an existing late-RNE build using
its generated native oracle and the existing complete-model comparison gates.

## Optional combined clamp and RNE

`--combine-clamp` consumes the proven raw input in one inline assembly block:
`fmax.s` with -128, `fmin.s` with127, then `fcvt.w.s ..., rne`. The temporary
floating register is explicitly clobbered; integer conversion remains explicit
RNE. Existing clamp SSA values remain available when other uses need them.

For all source-defined finite/infinite inputs, RISC-V min/max produce the exact
same clamped float (zero signs cannot affect the integer result). RISC-V's
numeric-operand preference on NaN differs from LLVM minimum/maximum, but the
original NaN-to-i8 conversion was poison. The target may define that previously
undefined result; this creates no promise about NaN outputs. Strict FP still
refuses, and unconstrained fflags behavior remains outside the contract.

The combined variant passes107,415 independently computed numeric comparisons
across all five frm modes. It additionally executes positive/negative quiet and
signaling NaNs under all five modes, checking that frm remains unchanged while
deliberately making no assertion about their undefined source output values.
Fourteen structural/refusal tests pass.

Whole native + actual Spike remain bitexact on all1000 outputs, rank mismatch0,
final no-FSM. Spike instructions12,271,325 →11,355,701 (7.46% additional savings;
20.34% below the14,254,689 base). Stock FireSim1786 verifies49,673,153 cycles,
10.08% fewer than1781's55,239,221. Receipt:
`docs/perf_records/late_combined_clamp_quant_rne.json`.

## Normal model compilation

Use `merlin_host_llvm_transform(LLVM_BIN, combine_clamp=True)` as the explicit
`host_llvm_transform` argument to Merlin's Spike model build. The callback runs
after ordinary lowering and before compiling `model.o`. It verifies complete
source, selected target LLVM and portable native oracle modules with LLVM's
assembler, then records the proven routes and tool/content identities. It
refuses if no chain is proven and never edits the original lowered source.

Merlin owns typed lexical/SSA recognition, explicit CPU ISA legalization,
normal compiler flags, object generation, harness construction and
final linking. The selected object's bytes enter the existing build hash before
the harness is generated. Historical CLI late-relink builds retain their base
harness hash; their final ELF/object/transform receipts identify those variants.

The complete 22-layer Tiny build proves89 chains, passes its original Torch
tolerance and reproduces all256,000 prior output bits in native and actual
Gemmini Spike. It retires271,019,239 Spike instructions. Hardware timing is
pending in1792. Receipt: `perf_records/tiny_scalar_host_quant_rne_spike.json`.

## Emission-neutral core extraction

The structural core matcher reaches all 89 routes on immutable eight-output/K2
TinyLlama source. The OOT adapter preserves the selected target and portable
LLVM bytes exactly. Fresh RV64GC O3 and native O0 objects match the qualified
objects exactly. Both standalone RNE and clamp/RNE objects also match their
original 107,415-check, five-rounding-mode boundary capsules; no new hardware
arm is needed for this extraction. Independent i8/i16 native boundary tests,
quoted/multiline SSA forms, comments and precise refusal tests qualify the
generic recognizer. Strict/constrained or unproved forms remain unchanged.
