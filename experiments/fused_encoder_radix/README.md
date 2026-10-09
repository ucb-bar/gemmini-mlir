# Fused encoder and completed radix experiment reproduction

This driver packages the source adaptation used for stock group 2072. It delegates
all encoder mathematics and proof metadata to Merlin's
`prepare_fused_encoded_witness`; no numeric transformation is duplicated here.
The target compiler commands, instruction policy and link closure belong to this
OOT experiment. Shapes and captured inputs belong to the explicit experiment,
not a production selector.

Required inputs are an existing, sealed 2069 completed-radix baseline and the
2072 qualification receipt. The receipt authenticates source, compiler, native
and target objects, retained source fallback, and the original capture/golden.
These large artifacts are external; this topic does not make a standalone model
redistribution or a new ordinary approximate-provider binding.

Load Merlin source `63578e4b7372fcb7772962781f7cd46c08f3c7f9` (including the
`12db3c126` fused encoded witness mechanism), then run from this repository:

```sh
PYTHONPATH="$MERLIN_SOURCE/src:." python experiments/fused_encoder_radix/build.py \
  --baseline "$RADIX_EXPERIMENT/out/fused_reconstruction" \
  --qualification "$ENCODER_EXPERIMENT/docs/perf_records/fused_encoder_radix_current2069_qualification.json" \
  --output out/fresh-encoder-reproduction
```

The output must not exist. Before executing the original compiler commands, the
driver rehashes the qualification and compiler dependency pins and checks the
loaded generic helper. It reproduces both native/target provider sources,
objects/libraries, and complete-group ELFs exactly, audits the final executable,
and queries the actual workspace ABI. `--emit-only` closes source identity only.
The existing compiled whole-native validation is bound through exact provider
identity; this command does not rerun a whole model or its retained fallback.

The historical native driver `out/encoder_compose/validate_native_v2.py`, fallback
check `check_retained_fallback.py`, source snapshots and all validation inputs are
named and hashed by the sealed qualification. They remain immutable external
experiment evidence. The native result used the existing exact normal pool with
an experimental RMS4 numeric override, 48 calls, 23,040 products, no fallback,
and all original 1,600 output words. It is not a new normal RMS4 source seal.
The stock result is a complete group, not whole-model timing.
