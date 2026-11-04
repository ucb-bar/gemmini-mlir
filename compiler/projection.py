"""Projection schedule composition using selected ordering facts and checked owners.

No shape names or expert machine programs participate in admission. Every
removed completion is consumed by dense_panel_epochs on the actual source-
checked counted schedule. Unsupported composition retains its input owner.
"""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


@lru_cache(maxsize=8)
def projection_epoch_law(configuration, numerics):
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
    root = Path(__file__).resolve().parent
    facts_bytes = (root / 'rtl_facts/facts.json').read_bytes()
    selection_bytes = (root / 'projection_ordering.json').read_bytes()
    selected = json.loads(selection_bytes)
    record = selected['ordering']
    valid = (selected['schema'] == 'merlin.rtl_command_ordering_selection.v1'
             and selected['target'] == 'gemmini'
             and selected['base_facts']['sha256'] == sha256(facts_bytes).hexdigest()
             and record['status'] == 'derived')
    cross = {(x['older'], x['younger'], x['space']): x for x in record['cross_queue']}
    loads = ('MVIN.spad_addr', 'MVIN2.spad_addr', 'MVIN3.spad_addr')
    readers = ('PRELOAD.weight_spad', 'COMPUTE_PRELOADED.a_spad', 'COMPUTE_ACCUMULATE.a_spad')
    def waits(a, b, space):
        return valid and cross.get((a, b, space), {}).get('overlap') == 'completion'
    classes = json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']
    def finite(role, minimum):
        entries = classes[role]['entries']
        queue = classes[role]['command_queue']
        return (entries['status'] == 'derived' and entries['value'] >= minimum
                and queue['status'] == 'derived' and queue['entries_total'] >= minimum)
    configs = next((x['by_code'] for x in record['funct_queue'] if x['funct'] == 0), {})
    flags = dict(
        local_read_after_write=all(waits(w, r, 'scratchpad') for w in loads for r in readers),
        local_write_after_read=all(waits(r, w, 'scratchpad') for w in loads for r in readers),
        ordered_accumulator_writes=valid and record['completion_order']['execute']['order'] == 'in_order',
        accumulator_result_visibility=waits('PRELOAD.c_addr', 'MVOUT.acc_addr', 'accumulator'),
        ordered_configuration_capture=valid and configs == {'0':'execute','1':'load','2':'store','3':'store'},
        finite_queue_progress=valid and finite('load', 1) and finite('execute', 2),
        complete_device_drain=True,  # The unchanged target fence callback.
        exclusive_private_state=True,  # Original checked operand/ACC views.
    )
    identity = 'selected-projection-ordering:' + sha256(selection_bytes).hexdigest()
    return DensePanelEpochLaw(configuration, numerics, identity, **flags)


def merge_projection_epochs(original, problem, block, resources, storage, adapter):
    from merlin.compile.scheduling.dense_panel_epochs import schedule_dense_panel_epochs
    law = projection_epoch_law(adapter.configuration_identity, adapter.numerical_policy_identity)
    # A retained wrapper exposes its real checked panel owner. The library
    # verifies the equality and reopens that owner's schedule plan and proof.
    source = getattr(original, 'checked', original)
    try:
        result = schedule_dense_panel_epochs(original, problem, block, resources, storage,
            adapter, source.admission, law, static_budget=128, timeout_ms=15000,
            checked_source=source if source is not original else None)
    except ValueError as exc:
        # The consumer records its refused premise in the normal admission log.
        # Original commands, ownership and completion are retained unchanged.
        original.realization['projection_epoch_refusal'] = str(exc)
        return original
    return result


# Final physical wide commands remain distinct from their tile-band checking
# expansion. This composes two existing checked consumers by an exact commuting
# square, following the library's wide-readout batching composition pattern.
from dataclasses import dataclass, replace


def _completion_paths(kernel):
    from merlin.sched.ir import Loop
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    paths = set()
    def visit(body, scope=()):
        for op in body:
            if type(op) is Loop:
                visit(op.body, (*scope, op.var))
            elif op.instr == DRAIN:
                if scope in paths:
                    raise ValueError('projection epochs require unique completion scopes')
                paths.add(scope)
    visit(kernel.body)
    return frozenset(paths)


def _remove_completions(kernel, removed):
    from merlin.sched.ir import Loop
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    def visit(body, scope=()):
        return tuple(replace(op, body=visit(op.body, (*scope, op.var))) if type(op) is Loop else op
                     for op in body if type(op) is Loop or op.instr != DRAIN or scope not in removed)
    return replace(kernel, body=visit(kernel.body))


@dataclass(frozen=True)
class WideProjectionEpochResult:
    original: object
    wide: object
    epochs: object
    kernel: object
    realization: dict
    removed: frozenset
    source_sha256: str

    def revalidate(self):
        from merlin.compile.scheduling.panel_tile_loads import expand_panel_tile_loads
        self.original.revalidate()
        self.wide.revalidate()
        self.epochs.revalidate()
        if (sha256(Path(__file__).read_bytes()).hexdigest() != self.source_sha256
                or _remove_completions(self.wide.kernel, self.removed) != self.kernel):
            raise ValueError('wide projection source, commands or completion selection changed')
        expanded, _ = expand_panel_tile_loads(self.kernel, project=self.wide.project,
            encode_band=self.wide.encode_band, max_static_nodes=self.wide.max_static_nodes)
        if expanded != self.epochs.kernel:
            raise ValueError('physical wide epochs differ from their complete checked band stream')

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        report = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return report


def widen_projection_epochs(original, problem, block, resources, storage, adapter):
    """Check final wide loads, then prove epochs on every actual tile band.

    The physical proposal changes only pure innermost load loops, so no effect
    or completion crosses a load. The checked wide owner verifies all bands in
    the original epochs. The epoch consumer then proves the FULL expansion.
    Removing the same original completion scopes from the physical kernel must
    commute exactly with expansion. No narrow first-band proxy is emitted or
    used as a temporal proof for a multi-band request.
    """
    from merlin.sched.ir import Const, Call
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing)
    from merlin.compile.scheduling.panel_tile_loads import (
        admit_panel_tile_loads, expand_panel_tile_loads, _offset_form)
    root = Path(__file__).resolve().parent
    selected = json.loads((root / 'projection_ordering.json').read_bytes())
    facts_bytes = (root / 'rtl_facts/facts.json').read_bytes()
    facts = json.loads(facts_bytes)['facts']
    if selected['base_facts']['sha256'] != sha256(facts_bytes).hexdigest():
        return merge_projection_epochs(original, problem, block, resources, storage, adapter)
    extents = selected['ordering']['extents']
    clients = facts['timing_structure']['memory_clients']['clients']
    # Conservative full request-width bands, bounded again by decoded MVIN
    # columns. Source pitch and local block stride are different quantities.
    max_columns = min(clients['dma_read']['max_request_bytes'],
        (1 << extents['MVIN2.spad_addr']['blocks']['cols']['width']) - 1)
    edge = problem.tile_n
    max_columns = max_columns // edge * edge
    if max_columns < 2 * edge:
        return merge_projection_epochs(original, problem, block, resources, storage, adapter)
    law_identity = 'gemmini-i8-load-bands-stride16:' + sha256(facts_bytes).hexdigest()
    def project(call):
        if call.instr not in ('gemmini_lhs_load', 'gemmini_weight_load'):
            return None
        a = dict(call.args)
        left = call.instr == 'gemmini_lhs_load'
        pitch = problem.k if left else problem.n
        extent = extents['MVIN.spad_addr' if left else 'MVIN2.spad_addr']
        if extent['blocks']['block'] != edge or extent['blocks']['stride']['offset'] != 16:
            raise ValueError('wide load addressing does not match selected block-stride field')
        law = TileLoadAddressLaw(edge, min(edge, (1 << extent['rows']['width']) - 1),
            max_columns, 1, pitch, edge, law_identity)
        return TileLoadMapping('lhs' if left else 'weight', storage.lhs.backing,
            a['row'] * pitch + a['col'], a['dst'], a['rows'], a['cols'], law)
    def encode(call, columns):
        a = dict(call.args)
        a['cols'] = Const(columns)
        return replace(call, args=tuple(a.items()))
    def encode_band(call, band):
        a = dict(call.args)
        delta, variables = _offset_form(band.local_row + a['dst'] * -1)
        if variables or delta % edge:
            raise ValueError('actual wide checking band has nonconstant tile displacement')
        index = delta // edge
        a['tile1'] = a['tile1'] + index
        a['col'] = a['col'] + index * edge
        a['dst'], a['rows'], a['cols'] = band.local_row, band.rows, band.columns
        return replace(call, args=tuple(a.items()))
    source = getattr(original, 'checked', original)
    try:
        proposal, coalescing = propose_tile_load_coalescing(source.kernel,
            project=project, encode=encode, max_tiles=max_columns // edge, coalesce_tail=True)
        if not coalescing['changes']:
            return merge_projection_epochs(original, problem, block, resources, storage, adapter)
        wide = admit_panel_tile_loads(source, proposal, problem, block, resources,
            storage, adapter, project=project, encode_band=encode_band,
            static_budget=128, timeout_ms=15000)
        expanded, projection = expand_panel_tile_loads(wide.kernel,
            project=wide.project, encode_band=wide.encode_band, max_static_nodes=wide.max_static_nodes)
        # The wide consumer has just verified this exact full source and epoch
        # expansion. Preserve its real original construction owner and law.
        logical = replace(source, kernel=expanded,
            realization=dict(wide.realization, kernel_sha256=expanded.digest()),
            checked_kernel_sha256=expanded.digest())
        epochs = merge_projection_epochs(logical, problem, block, resources, storage, adapter)
        before, after = _completion_paths(logical.kernel), _completion_paths(epochs.kernel)
        if not after <= before or _completion_paths(wide.kernel) != before:
            raise ValueError('wide composition changed original completion scopes')
        removed = before - after
        actual = _remove_completions(wide.kernel, removed)
        report = dict(epochs.realization, kernel_sha256=actual.digest(),
            wide_projection=dict(coalescing=coalescing, tile_load_projection=projection,
                physical_kernel_sha256=actual.digest(), expanded_kernel_sha256=epochs.kernel.digest(),
                completion_proof='full band expansion commutes exactly with checked completion removal'))
        result = WideProjectionEpochResult(original, wide, epochs, actual, report, removed,
            sha256(Path(__file__).read_bytes()).hexdigest())
        result.revalidate()
        return result
    except ValueError as exc:
        original.realization['projection_wide_refusal'] = str(exc)
        return merge_projection_epochs(original, problem, block, resources, storage, adapter)
