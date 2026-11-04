"""Completion-only rewriting of counted, privately owned reduction streams.

The target supplies effects and independently selected ordering facts. This
consumer never moves data calls, changes an expression, or allocates storage.
Its bounded proof interprets local effects in one complete output-block domain;
the emitted IR remains counted. Outer domains are partitioned exactly at Select
predicates, rather than tested at a few chosen dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import prod

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.expr import evaluate
from merlin.sched.ir.integer_domain import check_integer_domain


class CompletionRefused(ValueError):
    """No proof within the selected contract/budget: retain the original IR."""


@dataclass(frozen=True)
class Access:
    backing: str
    row: object
    rows: object
    lower: int
    upper: int
    writes: bool


@dataclass(frozen=True)
class Effect:
    # load writes local operands; execute reads them and optionally writes ACC.
    kind: str
    local: tuple[Access, ...] = ()
    accumulator: tuple[Access, ...] = ()


@dataclass(frozen=True)
class OrderingLaw:
    identity: str
    local_read_after_write: bool | None
    local_write_after_read: bool | None
    ordered_accumulator_writes: bool | None
    ordered_configuration_capture: bool | None
    finite_queue_progress: bool | None
    pending_load_commands: int
    pending_execute_commands: int
    complete_device_drain: bool | None
    exclusive_private_state: bool | None

    def require(self):
        if not self.identity or not all(value is True for value in (
            self.local_read_after_write, self.local_write_after_read,
            self.ordered_accumulator_writes, self.ordered_configuration_capture,
            self.finite_queue_progress, self.complete_device_drain,
            self.exclusive_private_state,
        )) or any(type(n) is not int or n < 1 for n in (
            self.pending_load_commands, self.pending_execute_commands)):
            raise CompletionRefused("missing selected ordering, progress, ownership or full-drain premise")


def _sites(body, path=(), domains=()):
    for i, op in enumerate(body):
        here = (*path, i)
        if type(op) is Loop:
            if op.iter_args or op.yields or op.results:
                raise CompletionRefused("event-carrying loops require a separate completion consumer")
            yield from _sites(op.body, here, (*domains, (op.var, op.extent)))
        elif type(op) is Call:
            if op.unit or op.produces or op.awaits or op.stages or op.sets or op.assumes:
                raise CompletionRefused("unmapped asynchronous or persistent-state annotations")
            yield here, domains, op
        else:
            raise CompletionRefused("unknown counted statement")


def _counts(body, drain):
    result = 0
    for op in body:
        if type(op) is Loop:
            result += op.extent * _counts(op.body, drain)
        elif op.instr == drain:
            result += 1
    return result


def _delete(body, removed, path=()):
    result = []
    for i, op in enumerate(body):
        here = (*path, i)
        if here in removed:
            continue
        result.append(replace(op, body=_delete(op.body, removed, here)) if type(op) is Loop else op)
    return tuple(result)


def _outer_partition(expressions, domains, max_contexts):
    """Exhaust all distinct outer effects of this piecewise-affine program.

    A direct outer-variable use requires enumerating its entire domain. A use
    solely as a Select discriminator needs its equality values and one member
    of the complement. Unknown expression forms refuse, rather than sample.
    """
    values = []
    for name, extent in domains:
        critical, direct = set(), False

        def visit(e):
            nonlocal direct
            if type(e) is Const:
                return
            if type(e) is Var:
                direct |= e.name == name
            elif type(e) is Sum:
                for child in e.terms:
                    visit(child)
            elif type(e) is Scaled:
                visit(e.expr)
            elif type(e) is Select:
                if e.var == name and 0 <= e.equals < extent:
                    critical.add(e.equals)
                visit(e.then)
                visit(e.other)
            else:
                raise CompletionRefused("unknown local-address expression")

        for expression in expressions:
            visit(expression)
        if direct:
            if extent > max_contexts:
                raise CompletionRefused("outer-dependent local address exceeds exhaustive proof budget")
            selected = tuple(range(extent))
        else:
            other = next((v for v in range(len(critical) + 1) if v < extent and v not in critical), None)
            selected = tuple(sorted(critical | ({other} if other is not None else set())))
        values.append(selected)
    if prod(map(len, values)) > max_contexts:
        raise CompletionRefused("outer effect partition exceeds exhaustive proof budget")
    return tuple(dict(zip((n for n, _ in domains), point)) for point in product(*values))


def _instances(body, env, path=()):
    for i, op in enumerate(body):
        here = (*path, i)
        if type(op) is Loop:
            for value in range(op.extent):
                yield from _instances(op.body, dict(env, **{op.var: value}), here)
        else:
            yield here, op, env


def _prove_stream(body, effects, removed, env, law, max_events):
    """Check RAW/WAR and the read-mediated WAW chain for every local row.

    Load completion may be out of order. For each overlapping pair of writes,
    an actual intervening execute read must connect old write -> read -> new
    write. Accumulator read/modify/write completion is separately ordered. All
    edges go from older to younger commands; finite queues use backpressure.
    A retained full drain closes the pending state, including host visibility.
    """
    writers, readers, acc_writers = {}, {}, {}
    footprint, pending_footprint = {}, {}
    peak_footprint = {}
    counts = {"load": 0, "preload": 0, "compute": 0}
    max_counts = dict(counts)
    raw = war = waw = acc_waw = events = 0
    last = None

    def rows(a, bindings):
        low, size = evaluate(a.row, bindings), evaluate(a.rows, bindings)
        if (type(low) is not int or type(size) is not int or size < 1
                or low < a.lower or low + size > a.upper):
            raise CompletionRefused("actual local access exceeds its original assigned storage view")
        return range(low, low + size)

    for path, call, bindings in _instances(body, env):
        events += 1
        if events > max_events:
            raise CompletionRefused("complete output-block effect domain exceeds proof budget")
        effect = effects[path]
        if effect.kind == "drain":
            if path in removed:
                continue
            for kind, count in counts.items():
                max_counts[kind] = max(max_counts[kind], count)
                counts[kind] = 0
            for memory, touched in pending_footprint.items():
                peak_footprint[memory] = max(peak_footprint.get(memory, 0), len(touched))
            writers.clear()
            readers.clear()
            acc_writers.clear()
            pending_footprint.clear()
            last = "drain"
            continue
        if effect.kind not in counts:
            raise CompletionRefused("an observer, readout or configuration occurs in the K stream")
        counts[effect.kind] += 1
        last = effect.kind
        for access in (*effect.local, *effect.accumulator):
            locations = tuple((access.backing, row) for row in rows(access, bindings))
            footprint.setdefault(access.backing, set()).update(row for _, row in locations)
            pending_footprint.setdefault(access.backing, set()).update(row for _, row in locations)
            if access in effect.accumulator:
                if not access.writes:
                    raise CompletionRefused("unexpected accumulator observer inside a K stream")
                for location in locations:
                    acc_waw += location in acc_writers
                    acc_writers[location] = events
            elif access.writes:
                if effect.kind != "load":
                    raise CompletionRefused("non-load local writer lacks selected ordering semantics")
                for location in locations:
                    if location in writers:
                        if readers.get(location, -1) <= writers[location]:
                            raise CompletionRefused("local load WAW lacks an intervening actual execute read")
                        waw += 1
                    war += location in readers
                    writers[location] = events
                    readers.pop(location, None)
            else:
                if effect.kind not in ("preload", "compute"):
                    raise CompletionRefused("local reader lacks selected execute ordering semantics")
                for location in locations:
                    raw += location in writers
                    readers[location] = events
    if last != "drain" or writers or readers or acc_writers:
        raise CompletionRefused("stream does not end with an original real full drain")
    if max_counts["preload"] != max_counts["compute"]:
        raise CompletionRefused("issued scope splits a preload/compute pair")
    return dict(
        proof_events=events, issued_between_retained_drains=max_counts,
        touched_local_rows={m: len(s) for m, s in footprint.items()},
        conservative_live_rows_between_drains=peak_footprint,
        local_raw_row_edges=raw, local_war_row_edges=war,
        read_mediated_load_waw_row_edges=waw, ordered_accumulator_row_edges=acc_waw,
        physical_pending_commands=dict(load=law.pending_load_commands, execute=law.pending_execute_commands),
        scope="one complete output block, including every K iteration; issued counts are not queue capacities",
    )


def coalesce_k_completions(kernel, project, law, *, drain, ordered_initial=False, max_events=250000, max_contexts=32):
    """Delete only internal drains before a separately drained readout subtree.

    The last compute completion, every readout and output-block completion
    remain byte-for-byte. Initial operand completion also remains unless
    ordered_initial admits its visibility through actual RAW in a multi-step
    stream. Other load-only prefixes remain protected. A
    source verifier must already have established the original ordered products
    and physical ownership. Missing effects/premises or exhausted proof budgets
    raise CompletionRefused, for the caller's unchanged fallback.
    """
    if type(kernel) is not Kernel or check_structure(kernel) or type(law) is not OrderingLaw:
        raise CompletionRefused("invalid original counted schedule or selected law")
    law.require()
    if type(ordered_initial) is not bool:
        raise CompletionRefused("initial completion policy must be Boolean")
    if (type(max_events) is not int or not 1 <= max_events <= 1000000
            or type(max_contexts) is not int or not 1 <= max_contexts <= 128):
        raise CompletionRefused("invalid exhaustive proof limits")
    check_integer_domain(kernel, 64)
    effects = {}
    for path, domains, call in _sites(kernel.body):
        effect = project(call)
        if type(effect) is not Effect or effect.kind not in ("drain", "load", "preload", "compute", "readout"):
            raise CompletionRefused("unknown or incomplete actual-call effect projection")
        if effect.kind == "drain" and call != Call(drain, ()):
            raise CompletionRefused("completion must be the original unannotated full-drain call")
        effects[path] = effect

    removed, reports = set(), []

    def kinds(path):
        return {effects[p].kind for p in effects if p[:len(path)] == path}

    def visit(body, path=(), domains=()):
        for i, op in enumerate(body):
            if type(op) is not Loop:
                continue
            here = (*path, i)
            children = [kinds((*here, j)) for j in range(len(op.body))]
            # The stream and its readout must be siblings in this output block.
            readouts = [j for j, k in enumerate(children) if "readout" in k]
            if readouts and all(children[j] <= {"readout", "drain"} for j in readouts):
                first = min(readouts)
                if first < 2 or readouts != list(range(first, len(op.body))):
                    raise CompletionRefused("noncanonical output-block/readout boundary")
                if not children[0] <= {"load", "drain"} or "load" not in children[0]:
                    raise CompletionRefused("missing separately completed initial operand prefix")
                stream = op.body[:first]
                prefix = [p for p in effects if p[:len(here)+1] == (*here, 0)
                          and effects[p].kind == "drain"]
                drains = [(p, ds) for p, ds, c in _sites(stream, here) if c.instr == drain]
                if len(prefix) != 1 or not drains or drains[0][0] != prefix[0]:
                    raise CompletionRefused("initial completion is not one original unconditional drain")
                if prod(extent for _, extent in drains[0][1]) != 1:
                    raise CompletionRefused("initial completion is inside a repeated loop")
                # The final dynamic effect is an unconditional original drain.
                last_path, last_domains = drains[-1]
                if prod(extent for _, extent in last_domains) != 1:
                    raise CompletionRefused("closing drain is inside a repeated loop")
                retained = {prefix[0], last_path}
                # Retained-operand programs can have more than one initial
                # load-only prefix. Completion deletion is restricted to
                # actual compute epochs, preserving every such load boundary.
                for p, _ in drains:
                    parent = p[:-1]
                    if not any(e.kind == "compute" for q, e in effects.items()
                               if q[:len(parent)] == parent):
                        retained.add(p)
                proposal = {p for p, _ in drains} - retained
                if not proposal:
                    continue
                if ordered_initial:
                    # Only a multi-step K stream already admitted above can
                    # order its first operand visibility through actual RAW.
                    # The original final-K and all readout/block drains remain.
                    retained.remove(prefix[0])
                    proposal.add(prefix[0])
                local_effects = {p: e for p, e in effects.items()
                                 if p[:len(here)] == here and p[len(here)] < first}
                expressions = [e for effect in local_effects.values()
                               for a in (*effect.local, *effect.accumulator) for e in (a.row, a.rows)]
                contexts = _outer_partition(expressions, (*domains, (op.var, op.extent)), max_contexts)
                # Local paths start at the block body, and preserve all inner loops.
                relative_effects = {p[len(here):]: e for p, e in local_effects.items()}
                relative_removed = {p[len(here):] for p in proposal}
                samples = [_prove_stream(stream, relative_effects, relative_removed,
                    env, law, max_events) for env in contexts]
                # A distinct post-readout drain is also mandatory and unchanged.
                for j in readouts:
                    calls = list(_sites((op.body[j],)))
                    if not calls or calls[-1][2] != Call(drain, ()):
                        raise CompletionRefused("readout subtree lacks its original closing completion")
                removed.update(proposal)
                reports.append(dict(block_path=here, outer_effect_partition=contexts,
                    removed_static_drains=sorted(proposal), retained_stream_drains=sorted(retained),
                    proof=samples))
            else:
                visit(op.body, here, (*domains, (op.var, op.extent)))

    visit(kernel.body)
    if not removed:
        raise CompletionRefused("no internal K-step completion can be removed")
    actual = replace(kernel, body=_delete(kernel.body, removed))
    if check_structure(actual):
        raise CompletionRefused("completion removal changed structural validity")
    # Independent structural conservation: all loop bounds/fields and every
    # non-completion Call are identical. No address/config/command permutation.
    def conserved(before, after, path=()):
        expected = []
        for i, op in enumerate(before):
            here = (*path, i)
            if here in removed:
                if op != Call(drain, ()):
                    raise CompletionRefused("attempted to delete a data effect")
                continue
            expected.append((here, op))
        if len(expected) != len(after):
            raise CompletionRefused("completion-only tree changed length unexpectedly")
        for (here, old), new in zip(expected, after):
            if type(old) is Loop:
                if replace(old, body=new.body) != new:
                    raise CompletionRefused("completion rewrite changed a loop bound or event binding")
                conserved(old.body, new.body, here)
            elif old != new:
                raise CompletionRefused("completion rewrite changed a data command or operand")
    conserved(kernel.body, actual.body)
    before, after = _counts(kernel.body, drain), _counts(actual.body, drain)
    return actual, dict(schema="fresh6.ordered_k_completion.v1",
        source_kernel_sha256=kernel.digest(), actual_kernel_sha256=actual.digest(),
        source_dynamic_drains=before, actual_dynamic_drains=after,
        removed_dynamic_drains=before-after, removed_static_paths=sorted(removed),
        law_identity=law.identity, ordered_initial=ordered_initial, blocks=reports,
        preserved="all data calls, issue order, loop domains, expressions, final-K/readout/block completions",
        qualification="selected ordering semantics plus exhaustive program proof; no RTL timing/certification")
