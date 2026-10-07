# Explicit normal ResNet preparation recipe

This archived experiment removes the original process-global layout and shared
preparation replacements. It uses the ordinary target catalog with explicit
`reduction_channel_block=64` and the generic invocation-local
`prepared_model_transform` API (local Merlin commit edf0e0ca4). Source numerical
contracts, 70 borrowed writers and four-lane exact host quantization remain.
These capture-bound selections do not define a production routing policy.

The fresh same-core legacy control and explicit route produce identical model
LLVM, host objects and final ELF; full target execution preserves all 1,000
original output words at 0/0 and has zero FSM instructions. This establishes
the interface refactor. It does not transfer stock2109 timing to the new ELF.

Original outputs, all build artifacts and qualification collector remain at
`/scratch/agustin/tmp/merlin-latest-20261004/out/artifacts/probes/resnet-shared-preparation-20261007`.
`build_whole.py` retains explicit owned source/provider paths and requires those
source-bound manifests. It writes only the requested private work directory.
`seal.py` is an exact archive of the collector with its original artifact-root
assumption; run the retained original collector, not this archive from another
directory. The capability input is explicitly pinned at `capabilities/facts.json`.
Both initial omissions of that input and their fail-closed refusals remain.

The compiler hook is source/package qualified locally. Publication is pending
GitHub DNS access; no new PR or hardware result is claimed.
