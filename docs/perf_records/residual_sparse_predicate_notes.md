# Exact sparse residual correction

The shared Merlin implementation derives a small byte-pair predicate from the
complete 65,536-pair source/predictor certificate. It replaces the existing
8,192-byte mismatch bitmap only when the caller explicitly supplies a bound of
1–16 pairs. The default remains the original bitmap and emits identical C.
The original ordered binary32 replay, packed output guard and ranked ABI stay
unchanged. Empty, single and multiple relations are proved independently.

## Complete adapter cost

| Fixture | Original bitmap | Sparse predicate | Change | Scope |
| --- | ---: | ---: | ---: | --- |
| Independent signed-source policy, all 65,536 pairs | 1,443,055 | 1,213,531 | −15.9054% | Pinned GSIM, complete ranked adapter |
| Original 1947 ReLU policy, 802,816 captured elements | Pending | Pending | Unknown | Same actual 1947 adapter and predictor |

The independent signed source has two ambiguous pairs. Its existing output
guard admits more values than the original ReLU source, which has one ambiguous
pair. These are distinct policies and distributions; the independent gain does
not predict the original operation or whole model.

Both arms use the same predictor object and common addresses. Their ELFs differ
by one selector byte. The measured interval includes descriptor validation,
device loads/compute/stores/fence, exact CPU correction and result descriptor
update. Identical warmup, dirty poisoning, all-byte output checks, input checks
and 4,096 sentinel bytes are outside the interval. Both strict RV64GC executions
and final executable no-FSM audits pass.

## Original adapter extraction repair

GNU `objcopy --dump-section` without an explicit output object rewrote the
original adapter object's metadata. Its `.text` stayed equal, but the whole-file
hash changed. The rewritten copy was retained, and a fresh recompilation using
the exact original source and recorded compiler arguments reproduced the
expected original object SHA byte for byte. That exact object was restored
before any candidate release. The source, predictor and controlled ELF were
unaffected.

The probe now supplies a fresh output object, rejects aliases and existing
outputs, and checks the input hash before and after extraction. A direct smoke
on the restored historical object passes. The repair ledger and all hashes are
included in `residual_sparse_predicate_initial_checkpoint.json`.

No normal model route or hardware candidate has been enabled. Original model
accuracy remains bitwise equality of all 1,000 binary32 outputs, with zero
absolute and relative tolerance. Physical DRAM traffic, overlap and the current
1988 model's section costs remain unknown.
