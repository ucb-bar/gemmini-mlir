"""Bounded row-transfer epochs over the checked retained bundle projection.

Input epochs aggregate only one original tile's proven disjoint row writes.
Readout epochs reuse the existing store-only ACC ownership proof and selected
uniform-configuration/backpressure contract. Original compute epochs and data
calls stay unchanged. Every ownership transition retains a real full drain.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Const, Loop, check_structure
from . import isa
from .rhs_row_bundle_retained import _prepare as _prepare_retained
from .dense_readout import _pack
from .dense_store_epochs import _transform as _store_epochs, BEGIN, END, STORE
from .column_batches import _admission

DRAIN = 'merlin_dense_panel_drain'
COMPLETE = Call(DRAIN, ())


def _strip_completion(body):
    result = []
    for item in body:
        if type(item) is Loop:
            nested = _strip_completion(item.body)
            if nested: result.append(replace(item,body=nested))
        elif item.instr != DRAIN:
            result.append(item)
    return tuple(result)


def _load_epochs(kernel, count, limit):
    """Consume complete row-bijection groups, retaining both ownership edges."""
    groups = 0
    def walk(body):
        nonlocal groups
        output = []; position = 0
        while position < len(body):
            item = body[position]
            if type(item) is Loop:
                output.append(replace(item,body=walk(item.body)))
                position += 1; continue
            if item.instr == 'row_bundle_lhs_0':
                # The incumbent's source-bound mapper proved that these exact
                # row i writes touch disjoint rows within one original K tile.
                sequence = body[position:position+2*count]
                if len(sequence) != 2*count:
                    raise ValueError('input epoch lacks a complete original row group')
                first = dict(item.args)
                for index in range(count):
                    call, completion = sequence[2*index:2*index+2]
                    args = dict(first,addr=first['addr']+index)
                    if (type(call) is not Call or call.instr != 'row_bundle_lhs_'+str(index)
                            or dict(call.args) != args or completion != COMPLETE):
                        raise ValueError('input epoch row partition or original readiness changed')
                if not output or output[-1] != COMPLETE:
                    raise ValueError('input epoch lacks its original opening ownership drain')
                for index in range(count):
                    output.append(sequence[2*index])
                    if (index+1)%limit == 0 or index+1 == count:
                        output.append(COMPLETE)
                groups += 1
                position += 2*count; continue
            if item.instr.startswith('row_bundle_lhs_'):
                raise ValueError('unmatched input row outside a checked complete group')
            output.append(item); position += 1
        return tuple(output)
    actual = replace(kernel,body=walk(kernel.body))
    return actual, dict(static_original_tile_groups=groups,max_issued_loads=limit,
        disjointness='incumbent symbolic row bijection within one aligned original LHS tile',
        readiness='opening full drain; each bounded group fully drains before consumers or row reuse',
        ordering='load completion may be out of order; each pending destination row is disjoint')


def _readout_epochs(kernel, count):
    """Reuse the common store-only-region proof on the actual split readouts."""
    def normalize(items):
        result=[]
        for item in items:
            if type(item) is Loop: result.append(replace(item,body=normalize(item.body)))
            elif item.instr.startswith('row_bundle_out_'):
                member=int(item.instr.removeprefix('row_bundle_out_'))
                if not 0 <= member < count: raise ValueError('unknown output row binding')
                result.append(replace(item,instr='gemmini_mvout',
                    args=(*item.args,('member',Const(member)))))
            else: result.append(item)
        return tuple(result)
    normalized = replace(kernel,body=normalize(kernel.body))
    actual, regions = _store_epochs(normalized,prove=True)
    def restore(items):
        result=[]
        for item in items:
            if type(item) is Loop: result.append(replace(item,body=restore(item.body)))
            elif item.instr in (BEGIN,END):
                if item.args: raise ValueError('unexpected readout epoch payload')
                result.append(COMPLETE)
            elif item.instr in (STORE,'gemmini_mvout'):
                args=dict(item.args);member=args.pop('member')
                if type(member) is not Const or not 0 <= member.value < count:
                    raise ValueError('readout epoch changed an original pointer row binding')
                result.append(replace(item,instr='row_bundle_out_'+str(member.value),
                                      args=tuple(args.items())))
            else: result.append(item)
        return tuple(result)
    return replace(actual,body=restore(actual.body)), regions


def _transform(bundle, physical):
    bundle.revalidate()
    physical = dict(physical, store_configuration_snapshot=
        'not assumed; one immutable raw CONFIG_ST for complete invocation')
    admission = bundle.schedule.realization['admission']
    values = (admission['max_issued_loads'],admission['pending_loads'],physical['finite_load_entries'])
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError('row epochs require actual positive selected load issue/queue budgets')
    load_limit = min(values)
    loaded, loads = _load_epochs(bundle.kernel,len(bundle.members),load_limit)
    actual, stores = _readout_epochs(loaded,len(bundle.members))
    if check_structure(actual): raise ValueError('row epoch aggregation changed counted index scope')
    if _strip_completion(actual.body) != _strip_completion(bundle.kernel.body):
        raise ValueError('row epoch aggregation changed an original data call, order or operand')
    return actual, dict(pass_order=('bounded_disjoint_row_load_epochs','uniform_disjoint_row_store_epochs'),
        original_member_bindings_sha256=sha256(json.dumps(bundle.members,
            sort_keys=True,allow_nan=False).encode()).hexdigest(),
        input_kernel_sha256=bundle.kernel.digest(),output_kernel_sha256=actual.digest(),
        load_epochs=loads,store_regions=stores,
        data_projection='every original load/preload/MAC/readout and its pointer binding unchanged',
        store_readiness='real beginning drain after original producers; closing drain before any ownership reuse',
        store_configuration='single exact raw-i32 CONFIG_ST unchanged for the complete invocation',
        readout_issue='selected store queue completion release and finite backpressure; no overlap with producers',
        physical_admission=physical)


@dataclass(frozen=True)
class _EpochBundle:
    original: object
    kernel: object
    receipt: dict

    def revalidate(self):
        kernel, receipt = _transform(self.original,_admission())
        if kernel != self.kernel or receipt != self.receipt:
            raise ValueError('row epoch source, ownership proof or selected contract changed')


def _prepare(ctx,members):
    prepared = _prepare_retained(ctx,members)
    if prepared is None: return None
    bundle, law = prepared
    physical = _admission()
    kernel, proof = _transform(bundle,physical)
    return _EpochBundle(bundle,kernel,proof), law


def lower_rhs_row_bundle_epochs(ctx,members):
    """Return receipt, or None before mutation when any source/law is refused."""
    try:
        prepared = _prepare(ctx,members)
        if prepared is None: return None
        epoch,law = prepared
        epoch.revalidate()
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    bundle = epoch.original
    fb=ctx.fb
    ctx.fence()
    ctx.current_source_index=bundle.members[0].get('attributes',{}).get('merlin.source_op_index')
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(bundle.k,channel=0))
    ctx.emit(*isa.config_ld(bundle.n,channel=1))
    ctx.emit(*isa.config_st(bundle.n*4))
    descriptor=[None]
    def emit(instruction,arguments):
        a={key:bridge.value(value) for key,value in arguments.items()}
        if instruction == DRAIN:ctx.fence()
        elif instruction.startswith('row_bundle_lhs_'):
            row=bundle.members[int(instruction.removeprefix('row_bundle_lhs_'))]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            ctx.emit(*isa.mvin(ctx.ptr(row['lhs'],a['dram_col']),a['addr'],a['rows'],a['cols'],channel=0,fb=fb))
        elif instruction == 'gemmini_mvin_weight':
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(bundle.n)),a['dram_col'])
            ctx.emit(*isa.mvin(ctx.ptr(bundle.members[0]['rhs'],offset),a['addr'],a['rows'],a['cols'],channel=1,fb=fb))
        elif instruction == 'gemmini_preload':
            descriptor[0]=a['cols']
            acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif instruction == 'gemmini_compute_preloaded':
            if descriptor[0] is None:raise ValueError('row epoch compute lacks descriptor')
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
        elif instruction.startswith('row_bundle_out_'):
            row=bundle.members[int(instruction.removeprefix('row_bundle_out_'))]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            ctx.emit(*isa.mvout(ctx.ptr(row['dst'],a['dram_col']),isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
        else:raise ValueError('unknown row epoch terminal instruction '+instruction)
    bridge=FnBuilderIndexAdapter(fb)
    emission=emit_index_program(epoch.kernel,bridge,emit)
    ctx.fence()
    receipt=dict(mechanism='shared_rhs_row_bundle_transfer_epochs',products=len(bundle.members),
        bundles=[bundle.receipt],emitted_kernel_sha256=[epoch.kernel.digest()],retained_plan=law,
        transfer_epochs=epoch.receipt,emission=emission,
        qualification='source-checked bounded transfer ownership epochs; independent execution required')
    ctx.metadata.setdefault('rhs_row_bundles_epochs',[]).append(receipt)
    return receipt
