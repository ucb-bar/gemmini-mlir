# Closed integer-observation prefix experiment

Owner: `/root/reference_parity`. Created 2026-10-07. Merlin topic
`d974a679e3aef522c38cd1cf21249593c333f412`, based on `eb15a85ce`.
No production policy is enabled by this experiment.

## Source and hypothesis

The current exact ResNet control is stock2101, 28,728,702 whole cycles.
The separate stock2104 diagnostic reports a 2,027,253-cycle interval before
the stem callback. That interval motivates a complete source-map experiment;
it is not credited to this standalone capsule or treated as pure CPU busy time.

The experiment extracts the actual live binary32-to-i8 map and its transpose,
zero initialization and padding consumer from the unchanged typed source.
An ordinary upstream pass folds the original reciprocal. Generic Merlin
closure proves the original saturated ties-even scalar graph, invariant
literals, complete integer observation, uses and declared floating effects.
The fixed partition is 18 raw-word prefix bits, with 524,288 requested bytes.
Conservative binary32 intervals certify each admitted cell over every member
word. Zero and saturated observations are admitted because no floating carrier
escapes. Unsupported cells, nonfinite/subnormal prefixes and non-RNE modes
retain the original source callback. Flags must remain unobservable under the
existing effect policy; observed equality does not strengthen admission.

The table certifies 254,725 of 262,144 cells. The original input uses 145,224
certified cells and 5,304 fallbacks across 150,528 values. Coverage is not a
cycle prediction. Existing service models do not price this mixed table
gather, branch, allocation, layout and scalar-fallback stream.

## Complete qualification

Primary immutable receipt:
`out/current_quant_prefix_seal_v2/qualification.json`, SHA256
`1bf3258b7ff9f3fc19080e1c15f8f8b828125bacf23b4f0b3822440546d9b88d`.
The adjacent tracked JSON is an identical copy. All 195 flat pins reclose.

- The original source extraction and interval table rederive byte identically.
- `llvm-link` plus `always-inline` removes every hot lookup call. `llvm-diff`
  compares the retained original fallback before and after linking.
- Native checks compare all 158,700 original padded bytes in each arm and
  4,177,920 finite executor values across four native rounding modes.
- Strict RV64GC Spike checks both complete arms, 602,112 immutable input bytes,
  16,384 guard bytes, and five FRM modes with two sticky-flag initializations.
- Both final ELF audits cover every executable section and report zero custom
  instructions, forbidden instructions or unknown instructions.
- The actual ELFs contain identical input, saved input, expected output and
  512KiB table bytes. They differ at one readonly selector byte only and use
  identical timed input/output addresses.

Control ELF: `out/current_quant_prefix_capsule_v4/control.elf`, SHA256
`26bb5ea6daed81570a0ff8f1d05db4b23e57514061900ff3970d90075f1f1351`.
Candidate ELF: `out/current_quant_prefix_capsule_v4/candidate.elf`, SHA256
`3bc841789b7884e1d6ad8ff053e744b7c288ced1a237c84c167e613707503d0d`.
Selector file offset: 2,411,500.

The strict parser is `mlir_oot.quant_prefix_capsule.parse`; the receipt supplies
its complete keyword contract. Two separately timed repetitions include all
source quantization, transpose, zero padding, allocation and ranked output
publication. Numerical verification and UART are outside these windows.
The extracted map's allocation/copy graph may differ from whole-model
bufferization, so this does not grant a whole-model timing forecast.

Spike's functional counter reports 2,208,532 versus 3,062,919 retired
instructions in the first windows, a 38.6857% increase. Its cycle counter is
a functional proxy. GSIM has not run; FireSim and whole-model timing remain
unmeasured. The candidate is held for one complete hardware cost decision.

## Retained refusals and delivery scope

The first capsule pulled unused libm symbols through an unscoped runtime.
The following broad section-GC attempts changed legacy CRT/TLS placement and
corrupted UART. Those outputs are retained and are not qualified. The final
capsule applies function/data sections only to its fresh runtime object,
leaving the original CRT, syscalls and allocator source untouched.

The delivery emitter adds compile-only signed-i8 and binary32 ABI assertions;
the frozen timed C and ELF bytes remain unchanged. The lookup executor after
its declarations reproduces the frozen implementation. The experiment
producer files retain the exact bytes recorded by their original receipts.
The generic proof's 54 focused tests, default table/executor byte identity,
format, structure, target-neutral ownership and regex gates pass.

This experiment does not change source constants, original accuracy gates,
model inputs, weights, golden values, default target schedules or publication
authorization. No hardware claim or whole-model speedup is inferred.
