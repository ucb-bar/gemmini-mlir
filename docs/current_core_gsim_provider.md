# Explicit current-core GSIM harness support

`support/gemmini_gsim` is an opt-in Merlin support package. Select it with
`MERLIN_TARGET_PATH`. It registers only the Gemmini harness build recipe and
GSIM command protocol. It does not provide command-buffer compilation, device
routing, a functional model, or inferred RTL capability facts.

Merlin owns source/object compilation and linking, emulator receipt validation,
bounded subprocess execution, and console parsing. The OOT provider owns the
RV64GC curated-harness ABI, GSIM arguments, backdoor environment, and generated
engine stack requirement. No legacy Python backend is imported. No target-name
map or accelerator instruction implementation was added to Merlin.

## Required caller configuration

Set all paths explicitly; the provider never searches other worktrees:

```sh
export MERLIN_TARGET_PATH=/path/to/mlir-oot/support/gemmini_gsim
export MERLIN_RISCV_GCC=/path/to/riscv64-unknown-elf-gcc
export MERLIN_GEMMINI_HARNESS_DIR=/path/to/curated/gemmini-rocc-tests
export MERLIN_GEMMINI_GSIM_EMU=/path/to/pinned/gsim/emulator
export MERLIN_GEMMINI_LOAD_ADDRESS=0x80000000
```

The last value is the selected platform's load address, supplied by its owner;
it is validated as a positive page-aligned unsigned64 value. The example is the
stock platform used by the retained numeric gate. The selected harness provides
`riscv-tests/benchmarks/common/{crt.S,syscalls.c,test.ld}` and headers. It is a
caller-owned external source dependency, not imported Python implementation.
The engine must have an available, byte-bound build receipt. The provider does
not invent an RTL lineage when the receipt declares adopted FIRRTL or weaker
provenance; shared citation retains those qualifications.

Use normal current-core `merlin.perf.layer_bench.build_program` and
`run_on_gsim`, passing `target='gemmini'`. The runner must supply both simulator
`max_cycles` and its wall deadline. The provider command pins the ELF, engine,
receipt, interpreter, provider source, contract, provider manifest, and shared receipt validator;
pre/post execution revalidation refuses changes. No unsupported stack-frame
entry symbol or arbitrary kernel ABI is asserted for prebuilt objects.

## Verification and limits

Fifteen tests cover explicit discovery, missing configuration, cycle-bound
validation, byte/receipt/argv/environment mutation, and unbound engine refusal.
A real current-core run rebuilt a harness around the retained 17×64×64 device
object and numeric harness object. All1,088 output values and its existing guard
passed; final ELF has zero FSM instructions. Kernel metric1,919 GSIM cycles
matches the retained legacy-harness run. This validates the harness seam, not
FireSim performance or whole-model support. Exact engine/input/output hashes
and provenance are in [the receipt](perf_records/current_core_gsim_provider.json).

All prior FireSim artifacts remain unchanged. Default provider discovery and
routing remain unchanged; only explicit package selection enables this path.
