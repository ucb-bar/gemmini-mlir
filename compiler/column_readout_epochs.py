"""Disjoint accumulator row epochs for exact per-column readout.

The producer is complete on entry. Every reload in an epoch owns different
whole accumulator rows, and both load and store phases end in a real drain.
No completion-order or configuration-snapshot assumption replaces those drains.
"""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from isa import DIM
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select, concretize
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter, StorageEffects, TemporaryObject, verify_temporary_storage)
from merlin.compile.scheduling.program_limits import EncodedAccess
from merlin.semantic_compiler.allocation_types import StorageBank


@lru_cache(maxsize=64)
def check_epoch(slots, row_bytes, capacity):
    """Check decoded private accesses and actual completion of a full epoch."""
    if not 1 <= slots <= capacity // (DIM * row_bytes):
        raise ValueError('column epoch exceeds exclusive accumulator capacity')
    loads = tuple(Call('reload_i32', (('slot', Const(s)), ('acc_row', Const(s*DIM))))
                  for s in range(slots))
    stores = tuple(Call('store_i8', c.args) for c in loads)
    kernel = Kernel('column_epoch_storage', (), (*loads, Call('loads_complete', ()),
                                                *stores, Call('stores_complete', ())))
    def project(call):
        if call.instr.endswith('_complete'):
            prefix = 'L' if call.instr == 'loads_complete' else 'S'
            return StorageEffects(call.instr, completion='wait',
                                  waits=tuple(prefix+str(s) for s in range(slots)))
        fields = concretize(call, {})
        write = call.instr == 'reload_i32'
        return StorageEffects(call.instr, (EncodedAccess('acc', fields['acc_row']*row_bytes,
            DIM*row_bytes, write),), completion='asynchronous',
            token=('L' if write else 'S')+str(fields['slot']))
    facts = Path(__file__).with_name('facts.json').read_bytes()
    adapter = StorageAdapter(project, lambda call, accesses: call,
        'gemmini:'+sha256(facts).hexdigest(), 'exact-i32-to-i8-per-column-readout',
        'disjoint-whole-row-DMA-with-full-phase-completion')
    objects = tuple(TemporaryObject('slot'+str(s), 'acc', s*DIM*row_bytes,
                                    DIM*row_bytes, row_bytes) for s in range(slots))
    return verify_temporary_storage(kernel, kernel, objects,
        (StorageBank('acc', 'physical_accumulator', capacity, 'bytes', row_bytes),),
        adapter, private_ownership=True, max_calls=256, max_objects=128)


def epoch_kernel(m, n, rows, scale_bits, group_runs, *, max_slots=None):
    """Construct counted source byte mappings, then check shared local lifetimes.

    Configuration remains outside the row epochs, as in the incumbent. Peeling
    the final partial row block keeps every load/store inside the original
    matrix, including M smaller than one complete accumulator phase.
    """
    from column_readout import _geometry, _readout_groups
    edge, row_bytes, capacity = _geometry()
    if rows != edge:
        raise ValueError('column epochs require original primitive row extent')
    slots = min((m+rows-1)//rows, capacity//(rows*row_bytes))
    if max_slots is not None:
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError('column epoch slot limit must be positive')
        slots = min(slots, max_slots)
    if slots < 2:
        raise ValueError('column epoch requires two disjoint row blocks')
    proof = check_epoch(slots, row_bytes, capacity)
    groups = _readout_groups(scale_bits, group_runs)
    body = []
    first = 0
    full, last = divmod(m, rows)
    epochs, remaining = divmod(full, slots)
    while first < len(groups):
        width = groups[first][1]
        end = first + 1
        while end < len(groups) and groups[end][1] == width:
            end += 1
        group = Var('readout_group_'+str(first))
        column = group*width + groups[first][0]
        def phase(start, count, partial):
            slot = Var('epoch_slot')
            logical = (start+slot*rows)*n + column
            height = select(slot, count-1, partial, rows) if partial else Const(rows)
            fields = (('rows', height), ('columns', Const(width)), ('acc_row', slot*rows))
            return (Loop(slot.name, count, (Call('reload_i32',
                        (('byte_offset', logical*4), *fields)),)),
                    Call('complete', ()),
                    Loop(slot.name, count, (Call('store_i8',
                        (('byte_offset', logical), *fields)),)),
                    Call('complete', ()))
        region = [Call('configure_scale', (('index', group+first),))]
        if epochs:
            region.append(Loop('row_epoch', epochs, phase(Var('row_epoch')*slots*rows, slots, 0)))
        if remaining or last:
            region.extend(phase(Const(epochs*slots*rows), remaining+bool(last), last))
        body.append(Loop(group.name, end-first, tuple(region)))
        first = end
    kernel = Kernel('gemmini_column_readout_epochs', (
        TensorArg('raw', (m,n), 'i32', 'read'),
        TensorArg('output', (m,n), 'i8', 'write')), tuple(body))
    return kernel, {'slots': slots, 'accumulator_bytes': slots*rows*row_bytes,
        'phases_per_column_group': epochs+bool(remaining or last),
        'completion': 'all disjoint loads; full drain; all stores; full drain',
        'configuration': 'same original per-group scale, fixed until all stores complete',
        'storage_check': proof, 'kernel_sha256': kernel.digest(),
        'source_partition': 'row epochs, slot rows, lanes; original column groups unchanged',
        'activity': 'modeled capacity reservation only; measured occupancy UNKNOWN'}
