# Generic IEEE constant clamp screen, 2026-10-05

Current TinyLlama stock profile 1901 independently closed all 155 calls and all
256,000 original output words. Its forward interval is 530,599,828 cycles:
179,768,482 inside device-call regions, including their CPU adapters, and
350,831,346 outside. The 22 host intervals preceding down projections total
127,539,158 cycles. These include activation/gating, polynomial evaluation,
quantization and other work; they are not isolated clamp durations. The outer
profile measurement being 472,135 cycles below unprofiled 1880 is layout/cache/run
variation, not negative instrumentation overhead or an optimization result.

## Mechanism and ownership

Merlin commit `06e5bd45b` provides an explicit `constant_float_clamp.rewrite`
API. It uses the existing typed LLVM tokenizer and SSA spans, without regex or
target/model/source-label selection. Typed finite nonzero ordered binary32
endpoints are decoded through integer round-to-nearest-even, including gradual
underflow, without host floating conversions or FTZ/DAZ dependence.

Eligible source pairs are `minimum(maximum(x, lo), hi)` within a contiguous
same-basic-block group containing only independent typed maximum/minimum calls.
For independent lanes, `max0,max1,min0,min1` is accepted. Arithmetic, memory,
unknown calls and basic blocks stop the group. Dependent intervening calls refuse.
Other consumers retain the original maximum definition, and all minimum SSA
uses retain their definitions. No load/store or general arithmetic moves.

A shared helper tests the original input for NaN once. For non-NaN input,
`maxnum/minnum` with finite nonzero ordered bounds preserves exact values and
signed zeros. The NaN path executes the original maximum/minimum pair. Zero,
infinite, NaN, reversed or unproved bounds refuse. Strict/constrained floating
scopes and explicit floating environment observations refuse. This API changes
no numeric policy and grants no exception-flag or trapping permission. It is
opt-in and does not add an automatic pipeline/ISA policy. Target harnesses,
ISA execution and final executable audits remain in OOT.

## Rejected and positive screens

Both arms use the unchanged complete 2,049-element packet2 pointwise source,
immutable input/output fixture, common static buffer addresses and one common
ELF containing both objects. Matched O3/RV64GC compilation and four alternating
AB/BA rounds include tensor lane loads/stores, defensive result copy, arena
allocation and ABI/dispatch in the measured interval. All original 2,049 output
bytes and 128 guard bytes are independently checked outside it.

| Matcher | Warm control GSIM cycles | Warm candidate GSIM cycles | Change | Decision |
| --- | ---: | ---: | ---: | --- |
| Adjacent calls only | 253,567 | 258,966 | +2.129% | Rejected |
| Independent intrinsic groups | 253,765.33 | 249,014 | -1.872% | Qualify whole model |

Round zero is retained as a cold measurement. Warm means use rounds one through
three; every revised warm round saves 1.79–2.03%. The first arm reached only
67 current-source quantization pairs and missed all 44 braided activation
pairs. The revised matcher reaches 111 quantization plus 44 activation pairs
in the actual original 1880 host LLVM. At this screen checkpoint the selected
whole LLVM assembled; whole numeric/target/performance gates were still pending.

One immutable synthetic capsule was measured. It is not captured model operands,
addresses or cache/full-program context. Full-model savings remain **UNKNOWN**;
the profile stage total is not an expected gain and the capsule percentage is
not multiplied into a whole-model prediction.

## Independent numeric proofs and retained pins

Six endpoint families cover positive, negative, mixed, equal and minimum-subnormal
bounds. Inputs contain 38 directed IEEE patterns and 4,096 random raw words:
positive/negative zeros, subnormal/normal boundaries, infinities, quiet/signaling
NaNs with both signs and distinct payloads, and endpoint neighbors. Two further
functions exercise independent interleaving and retained other maximum users.
The revised frontend passes 57,876 native raw-output comparisons, 165,360 strict
RV64GC Spike function cases across all five rounding modes, and 1,520 directed
actual GSIM cases. Target raw bits **and fflags** agree. These comparisons test
behavior; production acquires no new flag-observation permission. Every linked
executable section passes zero-FSM audits. Spike counters are instructions,
not hardware cycles.

The structural frontend emits byte-identical measured numeric and both complete
capsule native/target LLVM modules, and the previously assembled 155-route whole
LLVM/proof. Immutable prototype sources and timing artifacts remain retained.
Twelve focused Merlin legality/parser tests pass, as does the staged no-regex
gate. No new allowlist entries or duplicate lexer/IR were introduced.

[The journey receipt](perf_records/tiny_constant_clamp_journey.json) records both
arms, every timing round, numeric gates, source/input/output/object/ELF/engine/log
pins and structural replay. Raw generated artifacts remain under
`out/artifacts/probes/tiny-constant-clamp{,-interleaved}-20261005`.

Root authorized the positive arm's normal `host_llvm_transform` composition
after unchanged original 1880 pointwise/RNE/expanded-writer preparation. Whole
qualification will reproduce baseline linking byte-exactly and freeze original
runtime/device/startup/weights/main objects while changing the selected host
object. Original Torch atol=0.03125/rtol=0.02 and all 256,000 compiled words remain
mandatory before Recovery may submit a stock comparison. At the initial
screen checkpoint no whole candidate was admitted. No composition with
adjacent-RNE1902 or family1911/1912 is claimed.

## Final generic legality and whole qualification

Two general source-semantics corrections followed the initial screen. Commit
`4e704627e` freezes the helper input before its guard and uses that value on
both paths. Defined float values are unchanged; undef/poison are lawfully
refined, and a source pair that merely propagates poison cannot become a branch
on poison. Commit `4abe2c9b0` distinguishes LLVM unnamed numeric slots from
quoted numeric names, such as `%0` and `%"0"`. The adversarial pair is conforming
IR, independently accepted by `llvm-as`. Ordinary quoted/unquoted named aliases
still match. Fourteen focused generic tests pass.

The final frontend reproduces all frozen numeric/capsule/whole LLVM/proofs byte
for byte. Poison-safe IEEE target/native executables differed, so all 57,876
native raw outputs, 165,360 strict cases across five rounding modes/fflags, and
1,520 directed actual GSIM cases were freshly requalified. Complete capsule
target/native binaries and whole target/native model objects are byte-identical
to the prior candidate. Those unchanged executable gates and the measured
capsule timings are reused through explicit object hashes. Earlier generation
frontends are preserved as immutable snapshots with their original hashes;
the current snapshot hash is
`9361646412fb8cb7bad8b80eca11ffa1c380235c57972ada6c519cabba546b65`.

The original stock1880 ELF was actually relinked and reproduced byte-exactly,
then retained as a hardlink to avoid duplicating 2.34GB of immutable weights.
The initial O2 control reconstruction failed object equality and was retained
as a refused diagnostic. Historic `device/catalog_binding.json` supplies O3
flags; recompilation with those flags reproduces the original target `model.o`
byte-exactly. The selected normal `_transform_host_ir` hook then runs on the
immutable original composed upstream/pointwise/RNE/expanded-writer LLVM and
native companion. This is a normal hook followed by a controlled whole link,
**not a fresh whole upstream pipeline run**. Original raw catalog source,
upstream-prepared IR and expanded-writer ABI IR are distinct pinned stages.
All 155 original source/type/ordinal bindings and five physical kernels remain.

Only `model.o` changes at final linking. All original1880 runtime, startup,
main, weights, kernel and shim objects remain frozen. Original runtime compiler
consistency fixes remain enabled in normal builds; this experiment introduces
no default rollback. Fresh native full-model execution passes all 256,000
compiled words and the original Torch gate. Final full strict RV64GC Spike
passes the same digest, rank0/DONE/exit0 and every executable-section no-FSM
audit. It retires 164,623,233 instructions; that is not a hardware cycle result.

The final ELF is
`f7f898565bf5e1df12c93b1e07c6021a043cb268b8140b46964ba7fff8cbba42`.
Marker `37bdf9be0856` is intentionally inherited from frozen1880 main and is
nonunique. Exact final ELF and component hashes identify this candidate.
[The full qualification receipt](perf_records/tiny_constant_clamp_whole_qualification.json)
pins the source/ABI closure, current frontend and snapshots, normal hook/actual
argv, object equality, rejected control flags, fresh IEEE and whole-model gates.
All 119 artifacts from the first screen archive were independently rechecked
unchanged. Full generated qualification is retained under
`out/artifacts/probes/tiny-constant-clamp-whole1880-20261005`.

Recovery independently reclosed full original reference/digest/Torch/console
and final-audit pins and admitted one stock FireSim job **1920** against 1880,
with detached collector PID1576977. It is queued behind the correct SmolVLA
baseline1906. [The admission record](perf_records/tiny_constant_clamp_stock1920_admission.json)
preserves this transition separately from immutable qualification receipts.
Actual whole-model cycles and performance promotion remain pending; no gain or
composition with adjacent-RNE1902 or device-family1911/1912 is inferred.

`token_usage_available=false`: child exact token allocation is unavailable.
Root records shared campaign checkpoints; exclusive per-optimization billing
and input/cached/output splits are unknown.
