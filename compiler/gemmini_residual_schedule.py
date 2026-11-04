"""Source-checked loaded residual epochs bound to selected Gemmini facts.

Only independent accumulator rectangles are pending within a phase. A real
whole-device fence separates initialization, addition and readout, including
before any private row is reused. No same-cell DMA completion order is assumed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct

from merlin.compile.scheduling.compact_loaded_sum import LoadedSumLayout, LoadedSumContract, LoadedSumEffect
from merlin.compile.scheduling.loaded_sum_epochs import LoadedSumEpochLaw, plan_loaded_sum_epochs, emit_loaded_sum_epochs
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
import gemmini_encode as E


def digest(v):
    if not isinstance(v, bytes):
        v = json.dumps(v, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(v).hexdigest()


def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def selected_epoch_evidence():
    """Check original selected resources and the actual load/store classes.

    Configuration is made immutable and fully drained by this adapter before
    phase issue, rather than relying on capture across a configuration change.
    The existing complete fence is the common encoder's target primitive.
    """
    root = Path(__file__).resolve().parent / 'data'
    raw = (root / 'rtl_facts.json').read_bytes()
    selection = json.loads((root / 'COMMAND_ORDERING_SELECTION.json').read_text())
    if digest(raw) != selection['base_facts']['sha256']:
        raise ValueError('loaded-sum selected facts are stale')
    contract_sha = digest((root / 'target_contract.yaml').read_bytes())
    if contract_sha != selection['base_contract']['sha256']:
        raise ValueError('loaded-sum selected target contract is stale')
    order = selection['ordering']
    if hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest() != selection['ordering_sha256']:
        raise ValueError('loaded-sum command ordering identity changed')
    facts = json.loads(raw)['facts']
    edge = min(facts['arrays'][0]['rows'], facts['arrays'][0]['cols'])
    memory = next(m for m in facts['memories'] if m['name'] == 'accumulator')
    if edge != 16 or memory['row_bytes'] != edge * 4 or memory['elem_bits'] != 32:
        raise ValueError('loaded residual encoder requires selected sixteen-lane i32 rows')
    classes = facts['timing_structure']['command_classes']['classes']
    entries = {}
    for queue in ('load', 'store'):
        item = classes[queue]['entries']
        if item['status'] != 'derived' or item['value'] < 1:
            raise ValueError('loaded sum needs selected finite load/store queues')
        entries[queue] = item['value']
    for name, queue, funct in (('MVIN.spad_addr', 'load', 2),
                              ('MVIN2.spad_addr', 'load', 1),
                              ('MVOUT.acc_addr', 'store', 3)):
        access = order['accesses'][name]
        if (access['queue'], access['funct'], access['width']) != (queue, funct, 32):
            raise ValueError('loaded sum encoder opcode differs from selected operand owner')
        if order['extents'][name]['status'] != 'derived':
            raise ValueError('loaded sum needs selected address extents')
    if order['same_queue']['load']['release'] not in ('issue', 'completion'):
        raise ValueError('loaded sum lacks finite load issue premise')
    if order['same_queue']['store']['release'] not in ('issue', 'completion'):
        raise ValueError('loaded sum lacks finite readout issue premise')
    if order['completion_order']['store']['order'] != 'in_order':
        raise ValueError('loaded sum readout path is unqualified')
    rows = memory['banks'] * memory['depth']
    resource = dict(edge=edge, accumulator_rows=rows, row_bytes=memory['row_bytes'],
        capacity_bytes=memory['bytes'], queue_entries=entries)
    if resource['capacity_bytes'] != rows * memory['row_bytes']:
        raise ValueError('loaded sum accumulator capacity views differ')
    return dict(facts_sha256=digest(raw), ordering_sha256=selection['ordering_sha256'],
        target_contract_sha256=contract_sha,
        common_encoder_sha256=digest((root.parent / 'gemmini_encode.py').read_bytes()),
        resources=resource, complete_device_drain='common encoder fence; full iorw/iorw',
        configuration='configured then full fence; immutable during all phases',
        memory_order='disjoint private rectangles within phase; full fence between phases')


def residual_epoch_plan(elements, scalar_plan, *, output_dtype='i8', relu=False,
                        external_buffers_disjoint=False, source_numeric=None):
    if external_buffers_disjoint is not True:
        raise ValueError('loaded residual epochs need original distinct external tensor bindings')
    if type(elements) is not int or elements < 256:
        raise ValueError('loaded residual epochs need one complete sixteen-row tile')
    prefix = elements // 256 * 256
    evidence = selected_epoch_evidence()
    resources = evidence['resources']
    proof = dict(schema='gemmini.residual_scalar_source.v1', exhaustive=scalar_plan,
                 source_numeric=dict(source_numeric or {}),
                 output_dtype=output_dtype, relu=bool(relu), checked_pairs=65536)
    proof_sha = digest(proof)
    params = (
        (('scale_bits', bits(scalar_plan['lhs_load_scale'])), ('stream', 0), ('accumulate', 0)),
        (('scale_bits', bits(scalar_plan['rhs_load_scale'])), ('stream', 1), ('accumulate', 1)),
        (('scale_bits', bits(scalar_plan['store_scale'])), ('relu', int(bool(relu))),
         ('full_width', int(output_dtype == 'i32'))),
    )
    implementation = digest(evidence)
    layout = LoadedSumLayout(prefix // 16, 16, 16, 16, 0, resources['row_bytes'],
        resources['capacity_bytes'], output_dtype=output_dtype)
    contract = LoadedSumContract(proof_sha, params, implementation, True, True)
    row, column = Var('row') * 16, Var('column') * 16
    height = select('row', layout.rows // 16 - 1, 16, 16)
    width = select('column', 0, 16, 16)
    calls = []
    for index, kind in enumerate(('initialize', 'accumulate', 'store')):
        args = [('row', row), ('column', column), ('rows', height), ('columns', width),
                ('local_byte', Const(0))]
        args += [(key, Const(value)) for key, value in params[index]]
        calls.append(Call('gemmini_residual_' + kind + '_sync', tuple(args)))
    shape = (layout.rows, layout.columns)
    source = Kernel('gemmini_residual_complete_prefix', tuple(
        TensorArg(name, shape, dtype, access) for name, dtype, access in
        (('lhs', 'i8', 'read'), ('rhs', 'i8', 'read'), ('output', output_dtype, 'write'))),
        (Loop('row', layout.rows // 16, (Loop('column', 1, tuple(calls)),)),))

    def project(call):
        # Decode the actual authored command's operands, not a remembered effect.
        kinds = {'gemmini_residual_' + k + '_sync': (k, t, i) for i, (k, t) in enumerate(
            (('initialize', 'lhs'), ('accumulate', 'rhs'), ('store', 'output')))}
        if call.instr not in kinds:
            raise ValueError('unknown residual source effect')
        kind, tensor, index = kinds[call.instr]
        values = dict(call.args)
        keys = ('row', 'column', 'rows', 'columns', 'local_byte') + tuple(k for k, _ in params[index])
        if set(values) != set(keys) or len(values) != len(call.args):
            raise ValueError('residual effect has unknown or duplicate operands')
        actual_params = tuple((key, values[key].value) for key, _ in params[index]
                              if type(values[key]) is Const)
        if len(actual_params) != len(params[index]):
            raise ValueError('residual effect parameters must be actual integer constants')
        return LoadedSumEffect(kind, tensor, values['row'], values['column'], values['rows'],
            values['columns'], values['local_byte'], layout.accumulator_row_bytes,
            proof_sha, actual_params, 'synchronous')

    law = LoadedSumEpochLaw(implementation, proof_sha,
        digest(dict(selected=evidence, ownership='exclusive ACC between entry/final full fences',
                    external_tensors='distinct original logical bindings')),
        True, True, True, True, True, True)
    maximum = resources['capacity_bytes'] // (16 * resources['row_bytes'])
    plan = plan_loaded_sum_epochs(source, layout, contract, project,
        scalar_proof=proof, target_law=law, max_live_tiles=maximum, flatten_rows=True)
    checked = plan.revalidate(source, layout, contract, project, scalar_proof=proof)
    return plan, dict(source=source, project=project, proof=proof,
        selected_evidence=evidence, compact_check=checked,
        original_elements=elements, prefix_elements=prefix, serial_tail_elements=elements-prefix)


def emit_residual_epochs(fn, a, b, dst, plan, admission, *, output_dtype='i8'):
    """Emit every effect of the checked shared plan with real phase drains."""
    control = FnBuilderControlAdapter(fn)
    width = 4 if output_dtype == 'i32' else 1
    E.fence(fn)  # Complete configuration and acquire the exclusive private rows.
    def lower(kind, fields):
        if kind == 'complete_phase':
            E.fence(fn)
            return
        offset = control.op('add', control.op('mul', fields['row'], 16), fields['column'])
        local = control.op('udiv', fields['local_byte'], fields['accumulator_row_bytes'])
        if kind in ('initialize', 'accumulate'):
            stream = int(kind == 'accumulate')
            ptr = E.pointer_offset(fn, b if stream else a, offset)
            address = control.op('add', E.ACC_ADD if stream else E.ACC, local)
            E.mvin(fn, ptr, address, fields['rows'], fields['columns'], stream=stream)
        elif kind == 'store':
            ptr = E.pointer_offset(fn, dst, control.op('mul', offset, width))
            address = control.op('add', E.ACC_FULL if width == 4 else E.ACC, local)
            E.mvout(fn, ptr, address, fields['rows'], fields['columns'])
        else:
            raise ValueError('unknown checked residual epoch effect ' + kind)
    receipt = emit_loaded_sum_epochs(plan, control, lower)
    plan.revalidate(admission['source'], plan.layout, plan.contract, admission['project'],
                    scalar_proof=admission['proof'])
    return {**receipt, 'selected_evidence': admission['selected_evidence'],
        'compact_check': admission['compact_check'],
        'original_elements': admission['original_elements'],
        'prefix_elements': admission['prefix_elements'],
        'serial_tail_elements': admission['serial_tail_elements'],
        'host_payload_arithmetic': False}
