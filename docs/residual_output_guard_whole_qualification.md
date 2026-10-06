# Exact residual output guard: whole compiler qualification

The opt-in normal catalog route and the controlled 1903 link both pass native and strict Gemmini Spike checks against all 1,000 original binary32 output words. The original criterion remains bitwise equality with `atol=rtol=0`. The source, capture inputs, weights, scales, and golden were not replaced.

The selected relation uses five diagonal coefficient chunks, one target prediction/store, an existential eight-byte output guard, and original ordered binary32 replay for certificate-marked operand pairs. The public ABI remains three arguments with fresh, disjoint C0. Eligibility comes from a complete 65,536 signed-byte proof and the source scale relation. Default selection remains disabled.

The real normal build exposed a callback ordering bug: the base catalog sealed its source before the residual callback changed the selected call. The opt-in callback now validates and selects residual calls before the base preparation seals the selected source. The existing SHA guard remains active. The ordinary route compiles the sealed selected source and its actual selected object. The stateful regression and signed native checks pass; the prior stub-only receipt remains historical.

## Qualified artifacts

| Arm | Final ELF SHA256 | Native and strict output | Spike retired instructions |
|---|---|---|---:|
| Fresh ordinary compiler route | `79548bb553de8325f831a20f163644f57afe18e1051daf8ed04ec1c7b963e09c` | All 1,000 original words exact | 10,664,230 |
| Frozen 1903 controlled link | `11c224650e91080b8fc28e15c7b5bf1747cec74797678492fe783e03c8790a59` | All 1,000 original words exact | 9,843,652 |

Spike's counter here is retired instructions, not hardware cycles. Neither arm has a measured whole-model performance result.

The controlled link first byte-reproduces the actual 1903 control ELF, then replaces exactly two leaves: one source-derived residual kernel and its adapter. It retains 30 other residual leaves, every other target component, and all 11 actual host/runtime/startup/weights/shim objects. An `llvm-objcopy` public symbol alias preserves the source-qualified adapter's allocatable section bytes and relocation payloads; independent decoding checks all 41 symbol entries and permits only the declared public name change.

The fresh normal route has its own full source/catalog closure and uses different unselected objects from the 1903 control. It is preserved separately so an eventual controlled hardware result cannot be attributed to unrelated host or kernel changes. The controlled ELF retains a nonunique inherited build marker; the complete ELF SHA is authoritative.

## Admission state

Hardware release is held until the explicit-Clang original 802,816-output capsule reaches terminal output, raw-predictor, immutable-input, and guard checks. Positive smaller capsules establish feasibility, not whole-model timing. The earlier GCC original capsule was complete and slower, and remains rejected in its separate receipt.

The qualification receipt pins 278 artifacts. Reclose it with:

```sh
PYTHONPATH=/scratch/agustin/tmp/gemmini-residual-output-compose-20261006:/scratch/agustin/tmp/merlin-residual-output-word-20261005/src \
  /scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  tests/residual_output_whole_closure_probe.py \
  docs/perf_records/residual_output_guard_whole_qualification.json
```

The portable guard/certificate/replay lives in Merlin; diagonal decomposition, device schedule, ranked ABI, and catalog binding live in the OOT dialect. Token attribution is unavailable to this child; the root retains session accounting.
