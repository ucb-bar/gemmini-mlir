# Closed integer observer experiment reproduction

This packages the source and controlled-link route used for TinyLlama stock job
2070. Model names and historical identifiers identify this experiment's evidence;
they do not select a production compiler policy.

The generic expression, finite interval table, complete integer observation proof,
LLVM rewrite and source continuation are implemented in Merlin. This experiment
loads published Merlin main `95e8142d9cd1ae2432527c005e37f77a5af6976d` and checks the
actual loaded helper bytes against the qualified source re-emission receipt.
Target compilation, controlled link replay and final instruction audit use the
existing OOT interfaces. No device scheduling policy is added here.

## Inputs and command

`inputs.json` names and hashes the external typed source, original LLVM, source
binding, accepted table and helper source, prepared control/candidate LLVM,
compiler/linker, original output/golden, and prior qualification receipts. Large
model data and artifacts remain external. To relocate them, preserve their bytes
and update their explicit paths. Receipt pins use the original paths; any changed
location must also be declared in `path_overrides` and must have the same expected
hash. The included override preserves the original documentation bytes from the
accepted core commit after its historical `AGENT.md` changed. It does not waive a
source or documentation pin.

From the OOT checkout, with its normal compiler dependencies available:

```sh
PYTHONPATH="$PWD" python experiments/tiny_closed_observer/build.py \
  --inputs experiments/tiny_closed_observer/inputs.json \
  --output out/fresh-tiny-observer-reproduction
```

The output must not exist. The command authenticates all inputs before executing
compilers, regenerates the 22 typed proofs and 44 integer routes, and compares all
accepted helper bodies. It regenerates the 512 KiB table, lookup C and source
continuation exactly. It then recompiles the already prepared LLVM, reproduces
both model objects and complete control/candidate ELFs byte for byte, and audits
the final executables for prohibited FSM instructions.

`--emit-only` performs source re-emission without target compilation. It does not
grant the previous whole-model numerical result by executable identity.

## Scope and retained drivers

The frozen driver snapshots in `drivers/` are retained byte for byte:

- `source_continuation_whole_build.py`: original source continuation/table route.
- `source_continuation_whole_qualify.py`: required capture/native/runtime context.
- `closed_i8_interval_normal_prepare.py`: opt-in host object preparation and
  original native accuracy gate; it does not construct the final ELF.
- `whole_target.py`: actual stock-cost-qualified controlled final link and strict
  original whole-model validation.

These snapshots document the original source recipe and fixed historical paths.
`build.py` provides the explicit-input reproduction interface. Its final-link
stage consumes authenticated prepared LLVM; it does not redo capture, complete
upstream lowering, native execution, or stock timing. The historical full-model
gate is bound only when the implementation and all eleven unchanged nonmodel
objects reproduce the qualified executable exactly. Source re-emission and that
target identity are separate checks.

The prior gate retains all 256,000 original output words and the original Torch
`atol=0.03125, rtol=0.02`. The source policy retains the floating fallback and the
declared RNE, gradual underflow, nontrapping and unobservable flags/zero policy.
No new accuracy tolerance, approximate policy or performance claim is made.
