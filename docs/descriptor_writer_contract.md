# Fresh output ownership for device adapters

`mlir_oot.descriptor_writer` accepts explicit `DescriptorWriterContract` records
for ranked 64-bit C-interface adapters. The caller proves the adapter fully
writes each designated argument and returns the actual passed result descriptor.
The rewrite verifies static tensor types, complete read/write attributes, result
argument identity and sole-use `tensor.empty` writers before changing the module.
An unknown adapter or mismatching contract refuses the selection.

The tensor wrapper converts only readonly inputs to buffers and allocates each
writer freshly. A borrowed **void** memref bridge invokes the existing adapter,
discarding its returned descriptor. The wrapper materializes the fresh result
allocation as a writable tensor; upstream bufferization and deallocation own its
lifetime. The fresh allocation justifies the single `restrict` conversion; this
does not assert no-alias relationships among input pointers.

Returning the original writable tensor was previously incorrect under
copy-on-write. Calling a result-bearing memref external and discarding its
returned alias also caused double free: ownership assumed that descriptor owned
another allocation. The borrowed void bridge resolves this without disabling
deallocation. Actual native tests at O0/O2 keep a live input unchanged, return
both that input and a new result, and invoke repeatedly with normal deallocation.
Five refusal cases and two native cases pass.

## Whole ResNet qualification

The owned `out/fresh_device_result/whole_v5` candidate retains the1812 host
policy, original mean and device schedules.69 handwritten ranked adapters use
the new ownership interface; the raw expanded-memref classifier adapter retains
its ABI. All1,000 original f32 output words are exact in native execution and
actual Gemmini Spike. FinalELF zeroFSM passes.10,703,997 retired instructions
are2.56% below1812's10,985,615; hardware timing is pending.

Explicit contracts re-derive a host MLIR file byte-identical to the qualified
prototype. Compiled RV64GC bridge objects are also byte-identical. Ordinary
requant, residual and pooled stem adapters return argument2; the two scratch
readout requant adapters return argument3 and fully write arguments2/3. These
are current emitter facts, not a tensor-type inference rule. Provider integration
must supply these records directly from its adapter ABI description.

The generic textual lowering route does not synthesize a public C interface
from generic `func.func` syntax, so this derivation explicitly preserves the
public entry's `llvm.emit_c_interface`. It leaves the raw external classifier
unmarked: synthesizing a ranked C wrapper around that expanded ABI would call
a nonexistent symbol. This is why adapter ABI and output identity must be
explicit compiler inputs.

Automatic lowering needs a provider-owned result-identity/full-write contract,
a fresh-output wrapper option and one compilation receipt tying prepared source,
bridge, selected host LLVM, native oracle and final device objects together.
The current helper is manually selected; no compiler default changed.
