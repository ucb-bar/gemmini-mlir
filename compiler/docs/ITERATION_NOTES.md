# Compiler development evidence

Initial incumbent: source-typed xDSL frontend, shared checked dense counted schedules and convolution window/retention schedules, one terminal fact-derived encoder, shared host Linalg arithmetic. No benchmark-name dispatch.

## Bring-up
- Request 2_1793174460631466643_15f679867d444a7cbb5a8f71e053849d: five small dense/conv/movement/residual cases; L0/L1 succeeded where packaged facts resolved; L2 blocked by kernel_entry preflight.
- Request 2_1793174754578115944_a9297cf6bc7982dccabd47d12cf59ed4, snapshot fe23766a886c: five L0/L1 passed. L2 link failed missing kernel_complete; architectural fence string also unresolved structurally. Fixed native entry/completion symbols and canonical fence in same encoder. Numerical CB checks do not count as device execution. All cycles UNKNOWN.
- Public reference jack.fx_conv_1 lifted by normal reference-analysis owner. Source disassembly SHA 4e953c8a5c58aefeb6bdb11a92a0fc06a471d2da636d0a7354ad92c596fd5e55; semantic compatibility and dynamic timing UNKNOWN. Static instruction counts provide no performance score.

## Alternatives pending native comparisons
- Dense compact_mn incumbent; compact_nm, retained, prefetch, epochs use same effect projection and terminal encoder. Epoch laws bind selected RTL ordering hash. Knobs in TUNING_KNOBS.json. Independent schedule products and LLVM verification passed aligned, ragged, long-K geometries. No latency promotion yet.
- Conv sync incumbent; issue, epoch, readout use same schedule callbacks and encoder. Integer adapter tests pass tails/bias/dilation; native comparisons pending.
- QK cell incumbent; packed uses shared host transpose and bounded private stack panel (4352-byte measured frame). No public execution evidence yet.

## Ready work
1. Collect native baseline, fix exact ABI/trace/numerical failures.
2. Compare dense epochs and convolution readout complete-call cycles on large and connected regions.
3. Validate source-host counted lowering, ownership ledger and all source families.
4. Fuse conv pooling to bounded intermediate; retain typed scratch fallback.
5. Expand correctness roster and required gSIM evidence while independent implementation proceeds.
