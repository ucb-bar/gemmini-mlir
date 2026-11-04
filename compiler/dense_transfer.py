"""Address-only wide MVIN proposals over original checked dense panel programs.

Every physical command is decoded into ALL local tile bands.  Exact counted
trace equivalence retains the original source, lifetime and completion proof;
no synchronous-load premise or new completion permission is introduced.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing, _substitute,
)
from merlin.compile.scheduling.panel_tile_loads import (
    expand_dense_panel_tile_loads, _canonical_load_runs, _map_call, _same_band_call, _offset_form,
)
from merlin.sched.ir import Call, Const, Loop, Var, add


def _trace_equal(left, right, substitutions=()):
    """Exact structural induction over counted traces, modulo bound names.

    Shared normalization only collapses mixed-radix one-load loops and joins
    contiguous runs. No compute, drain or other effect is crossed.
    """
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if type(a) is Loop and type(b) is Loop:
            if (replace(a, var=b.var, body=()) != replace(b, body=()) or
                not _trace_equal(a.body, b.body, (*substitutions, (a.var, b.var)))):
                return False
        elif type(a) is Call and type(b) is Call:
            for old, new in substitutions:
                if old != new:
                    a = _map_call(a, lambda value: _substitute(value, old, Var(new)))
            if not _same_band_call(a, b):
                return False
        else:
            return False
    return True


def _singleton_load_nests(kernel, project):
    """Specialize a one-trip pure-load wrapper before mixed-radix joining.

    A coalesced remainder is represented by an unused one-trip outer index
    around its decoded band loop. Its domain is exactly {0}; substitute that
    value and retain every child in issue order. Completion/effectful loops and
    loops carrying tokens are never changed. This is checking representation
    only: the actual wide commands remain the emitted program.
    """
    changes = []

    def loads_only(body):
        return bool(body) and all(
            (not x.iter_args and not x.yields and not x.results and loads_only(x.body))
            if type(x) is Loop else type(x) is Call and project(x) is not None
            for x in body)

    def specialize(body, variable):
        return tuple(replace(x, body=specialize(x.body, variable)) if type(x) is Loop else
            _map_call(x, lambda value: _substitute(value, variable, Const(0))) for x in body)

    def walk(body):
        out = []
        for x in body:
            if type(x) is Loop:
                x = replace(x, body=walk(x.body))
                if (x.extent == 1 and not x.iter_args and not x.yields and not x.results
                        and any(type(child) is Loop for child in x.body) and loads_only(x.body)):
                    out.extend(specialize(x.body, x.var))
                    changes.append(dict(variable=x.var, value=0, rule='exact_singleton_load_domain'))
                    continue
            out.append(x)
        return tuple(out)

    return replace(kernel, body=walk(kernel.body)), changes


@dataclass(frozen=True)
class DenseTransferResult:
    original: object
    kernel: object
    realization: dict
    check: object

    def revalidate(self):
        self.original.revalidate()
        self.check(self.kernel)

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt


def coalesce_dense_epochs(original, problem, storage, adapter, *, request_bound=False):
    """Consume a checked panel or epoch program; preserve its complete trace."""
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochResult
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelPipelineResult
    from dense_readout import DenseEpochReadyResult
    from merlin.compile.scheduling.dense_ready_readout import DenseReadyReadoutResult
    typed_owner = (type(original) is DensePanelPipelineResult or
        type(original) in (DenseEpochReadyResult, DenseReadyReadoutResult) and original._storage == storage and original._adapter == adapter or
        type(original) is DensePanelEpochResult and original._storage == storage and original._adapter == adapter)
    if not typed_owner or type(request_bound) is not bool:
        raise ValueError('wide dense loads require their original typed panel owner and storage')
    facts_path = Path(__file__).parent / 'data/command_ordering.json'
    facts = json.loads(facts_path.read_text())
    extents = facts['ordering']['extents']
    laws = {}
    for role, name, pitch, tile in (
        ('lhs', 'MVIN.spad_addr', problem.k, problem.tile_k),
        ('weight', 'MVIN2.spad_addr', problem.n, problem.tile_n),
    ):
        extent = extents[name]
        blocks = extent['blocks']
        stream = 0 if role == 'lhs' else 1
        if (extent['status'] != 'derived' or blocks['block'] != tile or
            blocks['stride']['code'] != 1 or blocks['stride']['state']['value'] != stream):
            raise ValueError('selected MVIN multi-block geometry is unavailable')
        max_columns = ((1 << blocks['cols']['width']) - 1) // tile * tile
        laws[role] = TileLoadAddressLaw(tile, min(tile, (1 << extent['rows']['width'])-1),
            max_columns, 1, pitch, problem.tile_m if role == 'lhs' else problem.tile_k,
            'selected-mvin-multiblock:' + hashlib.sha256(facts_path.read_bytes()).hexdigest())

    def project(call):
        if call.instr not in ('lhs_load', 'weight_load'):
            return None
        op = adapter.project(call)
        if type(op) is not Load or op.step != Const(1):
            raise ValueError('wide dense movement requires contiguous source rows')
        view = storage.lhs if op.role == 'lhs' else storage.rhs
        pitch = problem.k if op.role == 'lhs' else problem.n
        return TileLoadMapping(op.role, view.backing, op.dram_row*pitch+op.dram_col,
            op.row, op.rows, op.cols, laws[op.role])

    def encode(call, columns):
        return replace(call, args=tuple((key, Const(columns) if key == 'cols' else value)
                                       for key, value in call.args))

    def encode_band(call, band):
        mapping = project(call)
        delta = band.source_byte + mapping.source_byte * -1
        constant, terms = _offset_form(delta)
        tile = mapping.law.tile_columns
        if constant % tile or any(coefficient % tile for coefficient in terms.values()):
            raise ValueError('wide band source is not primitive-tile aligned')
        tiles = add(Const(constant//tile), *(atom*(coefficient//tile) for atom,coefficient in terms.items()))
        fields = dict(call.args)
        fields.update(src_col=fields['src_col']+delta, dst=band.local_row, cols=band.columns,
                      b=fields['b']+tiles)
        return replace(call, args=tuple(fields.items()))

    original.revalidate()
    max_tiles = min(law.max_columns//law.tile_columns for law in laws.values())
    if request_bound:
        selected = json.loads((Path(__file__).parent / 'data/facts.json').read_text())
        request_bytes = selected['facts']['timing_structure']['memory_clients']['clients']['dma_read']['max_request_bytes']
        max_tiles = min(max_tiles, request_bytes // max(law.tile_columns*law.element_bytes for law in laws.values()))
        if max_tiles <= 0:
            raise ValueError('selected DMA request cannot hold one primitive band')
    proposal, changes = propose_tile_load_coalescing(original.kernel, project=project, encode=encode,
        max_tiles=max_tiles, coalesce_tail=True)
    original_checking, _ = _singleton_load_nests(original.kernel, project)
    normalized_original, _ = _canonical_load_runs(original_checking, project)

    def check(kernel):
        expanded, projection = expand_dense_panel_tile_loads(kernel, problem, storage, adapter,
            project=project, encode_band=encode_band)
        expanded, singleton_changes = _singleton_load_nests(expanded, project)
        normalized, _ = _canonical_load_runs(expanded, project)
        projection["singleton_load_loop_normalization"] = singleton_changes
        if (replace(normalized, body=()) != replace(normalized_original, body=()) or
            not _trace_equal(normalized_original.body, normalized.body)):
            raise ValueError('wide MVIN bands differ from the original checked epoch trace')
        return projection

    projection = check(proposal)
    realization = dict(original.realization, kernel_sha256=proposal.digest(),
        transfer_proposal=changes, tile_load_projection=projection,
        transfer_original_kernel_sha256=original.kernel.digest(),
        transfer_proof='exact counted band trace including original completion boundaries')
    result = DenseTransferResult(original, proposal, realization, check)
    result.revalidate()
    return result
