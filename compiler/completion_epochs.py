"""Consume selected ordering premises for completion-only dense transformations.

The scope is one synchronous, bias-free two-window invocation. Retained LHS,
resident RHS and unfamiliar kernels keep their incumbent route. Hardware
ordering is selected evidence, not RTL certification. The shared consumer proves
every removed load-reader boundary; these checks additionally bind the private
storage, issued scope and readout/block/final boundaries of this backend.
"""
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.dense_panel_epochs import (
    DensePanelEpochLaw, DensePanelEpochResult, STREAM_FLAGS, schedule_dense_panel_epochs,
)
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN, DensePanelPipelineResult, _windows,
)
from merlin.sched.ir import Call, Loop


@dataclass(frozen=True)
class BackendCompletionResult(DensePanelEpochResult):
    """Bind the checked epoch law to this backend's real completion emitter.

    The shared result establishes the abstract program. This consumer also
    freezes the implementation evidence and refuses a no-op completion callback
    or a callback bound to a different builder. The inherited source ABI owns
    the invocation; lower_matmul emits this program without another device
    schedule interleaved, and closes it before the caller can reuse the views.
    """
    backend_source_pins: tuple

    def revalidate(self):
        super().revalidate()
        root = Path(__file__).resolve().parent
        for relative, digest in self.backend_source_pins:
            if sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('completion backend implementation evidence changed')

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
        from target import Builder
        if (type(control) is not FnBuilderControlAdapter or type(control.builder) is not Builder
                or getattr(complete_epoch, '__self__', None) is not control.builder
                or getattr(complete_epoch, '__func__', None) is not Builder.fence):
            raise ValueError('completion law requires the selected target.Builder.fence on its own builder')
        return super().emit_index_program(control, lower_effect, complete_epoch=complete_epoch)

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
        return self.emit_index_program(FnBuilderControlAdapter(builder), lower_call,
                                       complete_epoch=builder.fence)


def stream_admission(problem, block, original, scope='output_block'):
    """Bound issued work over ONE output block, without inventing storage.

    K primitive slices each load at most bm+bn tiles and issue bm*bn pairs.
    The windows are recycled, so live storage remains two bm/bn panels, plus
    one bm*bn accumulator block. These are issued bounds, not queue capacities;
    finite queues backpressure issue. The latter bounds remain unchanged.
    """
    if scope not in ('period', 'output_block'):
        raise ValueError('unsupported completion epoch scope')
    kt = min(2, problem.tiles[1]) if scope == 'period' else problem.tiles[1]
    return replace(original,
        max_issued_compute_pairs=kt * block.m * block.n,
        max_issued_loads=max(2, kt) * (block.m + block.n))


def selected_stream_law(adapter, admission):
    """Missing, stale or unknown selected hardware premises mean no change."""
    root = Path(__file__).resolve().parent
    try:
        premises = json.loads((root / 'COMPLETION_PREMISES.json').read_text())
        ordering_bytes = (root / 'COMMAND_ORDERING_SELECTION.json').read_bytes()
        ordering = json.loads(ordering_bytes)
        for name, relative in (('facts', 'rtl_facts/facts.json'),
                               ('ordering', 'COMMAND_ORDERING_SELECTION.json')):
            if sha256((root / relative).read_bytes()).hexdigest() != premises['inputs'][name]['sha256']:
                return None
        if (premises['schema'] != 'merlin.public_completion_premises.v1'
                or premises['target'] != 'gemmini'
                or premises['ordering_identity'] != ordering['ordering_sha256']):
            return None
        # These records are preserved in the evidence report. An incomplete
        # table cannot authorize a transformation and then fail during reporting.
        if (type(premises['coverage_caveats']) is not list
                or any(type(item) is not str for item in premises['coverage_caveats'])):
            return None
        visibility = premises['premises']['accumulator_result_visibility']
        if (type(visibility) is not dict or type(visibility['status']) is not str
                or type(visibility['basis']) is not list
                or (visibility['value'] is not None and type(visibility['value']) is not bool)):
            return None
        flags = {}
        for flag in STREAM_FLAGS:
            if flag in ('complete_device_drain', 'exclusive_private_state'):
                continue  # Established below from this backend's construction.
            entry = premises['premises'][flag]
            if entry['status'] != 'fact' or entry['value'] is not True:
                return None
            flags[flag] = True
    except (OSError, ValueError, KeyError, TypeError):
        return None
    law = DensePanelEpochLaw(adapter.configuration_identity,
        adapter.numerical_policy_identity, premises['ordering_identity'],
        **flags, complete_device_drain=True, exclusive_private_state=True,
        # Hardware visibility is TRUE in the selected facts. This policy flag
        # is deliberately withheld because this study preserves readout drains.
        accumulator_result_visibility=False)
    law.validate_for(adapter, admission)
    return law, premises


def _closed(body):
    if not body:
        return False
    last = body[-1]
    return last == Call(DRAIN, ()) or type(last) is Loop and last.extent > 0 and _closed(last.body)


def _check_frontiers(kernel, adapter):
    """Check all counted paths without expanding their iteration domains.

    A store must follow a full completion after operand work. An operand call
    must follow a completion after any store, protecting output-block reuse.
    Loops transfer a finite three-state set; repeated states establish a fixed
    point, so large trip counts need no expansion. The invocation closes with
    its original full completion.
    """
    def walk(body, state):
        for statement in body:
            if type(statement) is Loop:
                seen, iteration = {}, 0
                while iteration < statement.extent:
                    if state in seen:
                        period = iteration - seen[state]
                        skip = (statement.extent - iteration) // period
                        if skip:
                            iteration += skip * period
                            continue
                    seen[state] = iteration
                    state = walk(statement.body, state)
                    iteration += 1
            elif statement == Call(DRAIN, ()):
                state = 'completed'
            else:
                op = adapter.project(statement)
                current = ('readout' if type(op) is Store else
                           'operand' if type(op) in (Load, Preload, Compute) else 'unknown')
                if current == 'unknown' or {state, current} == {'operand', 'readout'}:
                    raise ValueError('completion transform crossed a readout/output-block boundary')
                state = current
        return state
    if not _closed(kernel.body) or walk(kernel.body, 'completed') != 'completed':
        raise ValueError('completion transform lacks a real closing drain')


def _completion_difference(before, after, adapter):
    """Prove the actual tree edit consists solely of operand-epoch tail drains.

    This independent readback checks retained loop scopes/metadata and every
    data call. It also forbids removing a drain from an epoch containing a
    store, even if another later drain could satisfy a weaker visibility test.
    """
    removed = []

    def calls(body):
        for s in body:
            if type(s) is Loop:
                yield from calls(s.body)
            elif s != Call(DRAIN, ()):
                yield s

    def walk(old, new, path, multiplicity):
        if len(old) == len(new) + 1 and old[-1] == Call(DRAIN, ()):
            kinds = {type(adapter.project(c)) for c in calls(old[:-1])}
            if not kinds or not kinds <= {Load, Preload, Compute}:
                raise ValueError('removed completion is not an operand-only K-stream boundary')
            removed.append({'path': path, 'dynamic_instances': multiplicity})
            old = old[:-1]
        if len(old) != len(new):
            raise ValueError('completion transform inserted or reordered schedule statements')
        for left, right in zip(old, new, strict=True):
            if type(left) is Loop and type(right) is Loop:
                if replace(left,body=right.body) != right:
                    raise ValueError('completion transform changed a counted loop scope')
                walk(left.body,right.body,(*path,left.var),multiplicity*left.extent)
            elif left != right:
                raise ValueError('completion transform changed an actual data call')

    if replace(before,body=after.body) != after:
        raise ValueError('completion transform changed the kernel contract')
    walk(before.body,after.body,(),1)
    return removed


def _unchanged(original, reason):
    return replace(original, realization={**original.realization,
        'backend_completion_choice': {'emitted_change': False, 'fallback_reason': reason}})


def _merge_completion_streams(original, problem, block, resources, storage, adapter, scope):
    """Return the checked completion alternative, or the unchanged safe route.

    Called only from dense.lower_matmul, whose emitter exclusively owns the
    device windows until returning: it issues no other schedule inside this
    region and supplies target.Builder.fence to every completion. That method
    emits the same volatile memory-clobbering RISC-V fence as gemmini_fence;
    dense.lower_matmul also emits an unconditional final fence. Resident
    handles are excluded by the caller, and different invocations never share
    unfinished device work. External concurrent invocation is not admitted by
    the inherited backend ABI.
    """
    if (type(original) is not DensePanelPipelineResult or original.bias_source is not None
            or (original.realization.get('schedule_ir') or {}).get('pipeline_kind') != 'two_window'
            or problem.tiles[1] < 2):
        return _unchanged(original, 'requires a bias-free checked two-window stream with at least two K slices')
    selected = selected_stream_law(adapter, original.admission)
    if selected is None:
        return _unchanged(original, 'selected hardware inputs are missing, stale, false or unknown')
    original.revalidate()
    # The inherited checker proves all physical byte intervals, source cover,
    # original buffer disjointness and tails. Recheck the actual live windows;
    # a larger issued bound does not supply this proof.
    if (storage.accumulator.backing in (storage.lhs.backing, storage.rhs.backing)
            or not adapter.synchronous_completion):
        return _unchanged(original, 'separate private accumulator backing and full completion are required')
    try:
        slots = _windows(problem, block, storage, resources)
        _check_frontiers(original.kernel, adapter)
        law, premises = selected
        result = schedule_dense_panel_epochs(original, problem, block, resources,
            storage, adapter, original.admission, law,
            static_budget=128, timeout_ms=15000)
        _check_frontiers(result.kernel, adapter)
        removed = _completion_difference(original.kernel, result.kernel, adapter)
        if sum(site['dynamic_instances'] for site in removed) != result.realization['panel_epochs']['removed_dynamic_drains']:
            raise ValueError('completion owner and actual tree readback disagree')
    except ValueError as error:
        # Unsupported program structure or an inconclusive consumption proof
        # supplies no permission to remove completions.
        return _unchanged(original, 'program proof refused: ' + str(error))
    result.realization['backend_completion_choice'] = {
        'emitted_change': bool(removed), 'scope': scope,
        'fallback_reason': None if removed else 'no stream fits the checked issued scope',
    }
    result.realization['backend_completion_proof'] = {
        'schema': 'fresh6.completion_program_obligations.v1',
        'issued_scope': ('one output block, all primitive K slices' if scope == 'output_block'
                         else 'at most two adjacent primitive K slices within one output block'),
        'k_slices': problem.tiles[1],
        'removed_completion_sites': removed,
        'block_tiles': [block.m, block.k, block.n],
        'issued_compute_pair_bound': original.admission.max_issued_compute_pairs,
        'issued_load_bound': original.admission.max_issued_loads,
        'pending_compute_pairs': original.admission.pending_compute_pairs,
        'pending_loads': original.admission.pending_loads,
        'live_operand_bytes': sum(v.capacity * v.unit_bytes for slot in slots
                                  for v in (slot.lhs, slot.rhs)),
        'live_accumulator_bytes': block.m * block.n * problem.tile_m * storage.accumulator.unit_bytes,
        'assigned_storage': asdict(storage),
        'exclusive_ownership': 'synchronous invocation; no resident state or nested/interleaved device emitter',
        'completion_callback': 'target.Builder.fence; unchanged full fence plus unconditional lower_matmul final fence',
        'readout_policy': 'preserve all original readout boundaries although selected visibility is true',
        'hardware_visibility_fact': premises['premises']['accumulator_result_visibility'],
        'coverage_caveats': premises['coverage_caveats'],
        'ordering_identity': law.qualification_identity,
        'tracking_note': 'tracked:null selected dependencies remain uncertain for RTL certification',
        'source_pins': {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ('completion_epochs.py', 'dense.py', 'target.py', 'backend.py',
                                     'COMPLETION_PREMISES.json', 'COMMAND_ORDERING_SELECTION.json',
                                     'rtl_facts/facts.json', 'tuning_settings.json',
                                     'TUNING_KNOBS.json')},
    }
    result = BackendCompletionResult(
        **{field.name:getattr(result,field.name) for field in fields(DensePanelEpochResult)},
        backend_source_pins=tuple(sorted(result.realization['backend_completion_proof']['source_pins'].items())))
    result.revalidate()
    return result


def merge_completion_streams(original, problem, block, resources, storage, adapter,
                             *, enabled=True, scope='output_block'):
    """Registered completion alternatives on counted schedule IR.

    The selected result is the object consumed by the normal target emitter.
    Explicit input/check, completion and final-result passes prevent a stale
    precursor or unchecked proposal from becoming the emitted plan. A disabled
    choice or unsupported result has an unconditional unchanged alternative.
    """
    from merlin.sched.ir import check_structure
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
    from merlin.transforms.pipeline_policy import (
        RegisteredPass, PipelineVariant, PipelinePolicy, PolicyFacts,
        PropertyPredicate, execute_policy,
    )
    if type(enabled) is not bool or scope not in ('period', 'output_block'):
        raise ValueError('completion policy requires declared enabled/scope settings')
    if type(original) is not DensePanelPipelineResult:
        return original

    def verify(kernel):
        if check_structure(kernel):
            raise ValueError('invalid completion schedule IR')

    def checked(session):
        original.revalidate()
        if session.module != original.kernel:
            raise ValueError('completion input differs from its checked source')
        return {'checked_panel': original}

    def keep(session):
        return {'checked_completion': _unchanged(session.analysis('checked_panel'),
            'disabled by declared merge_k_completion_streams setting')}

    def merge(session):
        result = _merge_completion_streams(session.analysis('checked_panel'), problem,
            block, resources, storage, adapter, scope)
        # No partial proposal enters the session; the owner checked it first.
        session.module = result.kernel
        return {'checked_completion': result}

    def final(session):
        result = session.analysis('checked_completion')
        result.revalidate()
        if result.kernel != session.module:
            raise ValueError('completion emitter would consume a stale schedule')
        return {'completion_program': result}

    level = 'dense_counted_schedule'
    registry = {
        'completion_checked_input': RegisteredPass(PassSpec('completion_checked_input', 3,
            produces=('checked_panel',), mutates_ir=False, ir_level=level), checked),
        'completion_unchanged': RegisteredPass(PassSpec('completion_unchanged', 4,
            requires=('checked_panel',), produces=('checked_completion',),
            mutates_ir=False, ir_level=level), keep),
        'completion_merge_stream': RegisteredPass(PassSpec('completion_merge_stream', 4,
            requires=('checked_panel',), produces=('checked_completion',), ir_level=level), merge),
        'completion_final_result': RegisteredPass(PassSpec('completion_final_result', 6,
            requires=('checked_completion',), produces=('completion_program',),
            mutates_ir=False, ir_level=level), final),
    }
    policy = PipelinePolicy((
        PipelineVariant('merged_k_stream', ('completion_checked_input',
            'completion_merge_stream', 'completion_final_result'),
            (PropertyPredicate('capability', 'epoch_opt_in'),)),
        PipelineVariant('unchanged_completion', ('completion_checked_input',
            'completion_unchanged', 'completion_final_result')),
    ), 'unchanged_completion')
    session = PipelineSession(original.kernel,
        context=PipelineContext(hardware=adapter.configuration_identity,
            numerical_policy=adapter.numerical_policy_identity,
            resources=sha256(json.dumps(asdict(resources),sort_keys=True).encode()).hexdigest(),
            execution=original.admission.qualification_identity),
        ir_level=level, max_repairs=0, fingerprint=lambda k:k.digest(), verify_ir=verify)
    facts = PolicyFacts(session.stamp(), shapes=((problem.m,problem.k),(problem.k,problem.n)),
        element_types=('i8','i8','i32'), operations=frozenset(('matmul',)),
        capabilities=(('epoch_opt_in',enabled),))
    selected = execute_policy(session,policy,registry,facts,output='completion_program')
    result = replace(selected.artifact, realization={**selected.artifact.realization,
        'completion_pipeline': selected.receipt})
    result.revalidate()
    return result
