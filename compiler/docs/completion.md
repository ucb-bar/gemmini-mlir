# Completion-only schedule composition

`COMPLETION_ADAPTER.json` exposes `completion_adapter.apply_completion(original,
context)`, which production `dense.plan_matmul` invokes after its inherited
route has built and checked the actual commands. The returned result is the
object lowered by `lower_matmul`. This entry is importable independently.

Context fields: `problem`, `block`, `resources`, `storage`, `adapter`, `admission`
and plain Boolean `premises`; optional `mode`, `completion_plan`, `static_budget`
and `timeout_ms`. Mode defaults to `output_block` at the public entry. Production
reads the declared `dense_completion` knob. `original` keeps the inherited
program; `period` groups adjacent singleton compute epochs; `output_block`
groups each complete K stream. No name/shape whitelist is involved.

A caller can use the original execution admission (no implicit enlargement), or
`make_completion_issue_budget` for an exact counted program-scoped budget. The
latter binds the input kernel/mode/scopes and retains all original admission
and physical pending limits. It proves counts, not target execution authority.

The existing scheduling dialect binds a post-construction `schedule.apply_policy`
to this exact kernel and context. Every operation is consumed or refused. The
original blocking/layout/pipeline plan remains unchanged. The result binds the
plan, context, actual kernel, adapter source and completion proof owners, and
revalidates before/after the ordinary counted emitter. Only explicit retained
schedule drains invoke the unchanged full completion callback.

The core checks exact symbolic data-tree conservation and counts deletions over
all loop multiplicities. The original route verifier proves source/coverage,
physical storage and original epoch hazards. All explicit target laws must be
true. A separate symbolic proof orders overlapping load writes through an
actual intervening read, using local RAW then WAR; it assumes no load-WAW law.
Stores, initialization, output-block/final and pre-readout completion stay.

`CompletionRefusal` means no safe optimization was established. The production
wrapper keeps its original checked result and records that refusal. Missing
closing completion or an invalid original source proof is separately rejected
by the original verifier. Neither a conditional source proof nor successful
SCF emission is physical execution or timing qualification.
