"""Reuse private source tensor storage under explicit materialization effects.

Structural liveness alone cannot bound lazy host tensors or asynchronous device
uses. Callers supply actual complete materializations and pointer aliases; this
adapter extends liveness through lazy results and uses the shared arena packer.
No source arithmetic, tensor identity or public output is changed.
"""
from dataclasses import dataclass, field
from math import prod

from xdsl.dialects.builtin import IntegerType, AnyFloat, NoneAttr, TensorType
from merlin.transforms.analyses import analyze_ir, validate_analysis
from merlin.transforms.analysis_types import IRIndex, ancestor_in
from merlin.xdsl_dialects.lowering.arena_plan import pack_disjoint
from .fresh_tensor_writer import PrivateWorkspaceContract, validate_private_workspaces


@dataclass(frozen=True)
class StorageRequest:
    value: object
    workspace: PrivateWorkspaceContract
    full_tensor_write: bool


@dataclass(frozen=True)
class StorageReusePlan:
    source_sha256: str
    aliases: tuple
    records: tuple
    refusals: tuple
    analysis: object = field(repr=False)

    @property
    def value_aliases(self):
        return dict(self.aliases)

    def record(self):
        return dict(schema='merlin.source_storage_reuse.v1', source_sha256=self.source_sha256,
                    structural_ir_sha256=self.analysis.source_sha256,
                    reuse=list(self.records), refusals=list(self.refusals),
                    numerical_execution='not_established', cycles='UNKNOWN')


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def plan_source_storage(module, *, source_sha256, requests, materialized_values,
                        completed_operations, storage_aliases, source_positions):
    """Plan exact-type private buffers in the original forward source order.

    Completed operations map actual operations to implementation effect hashes;
    requests carry the existing synchronous/noescape/private workspace contract
    plus a full logical tensor write. Neither fact is inferred from an opcode.
    The caller must bind every actual pointer alias, and every tensor whose
    source evaluation is replaced by an independent complete materialization.
    Grouped/reordered emission requires a separate actual scheduling proof.
    """
    if not _pin(source_sha256):
        raise ValueError('storage reuse requires the original source SHA256')
    functions = [o for o in module.body.block.ops if o.name == 'func.func']
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError('storage reuse requires one closed sequential source block')
    block = functions[0].body.block
    operations = tuple(o for o in block.ops if o.name != 'func.return')
    if set(source_positions) != set(operations):
        raise ValueError('storage source positions must name exactly the live operations')
    indices = tuple(source_positions[o] for o in operations)
    if any(type(i) is not int or i < 0 for i in indices) or any(a >= b for a,b in zip(indices,indices[1:])):
        raise ValueError('storage source positions must retain original forward order')
    if any(o not in source_positions or not _pin(pin) for o,pin in completed_operations.items()):
        raise ValueError('storage completion effects require actual source operations and hashes')
    analysis = analyze_ir(module)
    index = IRIndex(module)
    facts = {v.id:v for v in analysis.values}
    shared_block = next(b for b in analysis.blocks if b.block == index.block_ids[block])
    intervals = {v.value:v for v in shared_block.live_intervals}
    materialized = frozenset(materialized_values)
    all_values = set(index.value_ids)
    if not materialized <= all_values or any(v not in all_values or base not in all_values for v,base in storage_aliases.items()):
        raise ValueError('storage materialization/alias value absent from current source')
    for value in storage_aliases:
        seen=set(); cursor=value
        while cursor in storage_aliases:
            if cursor in seen: raise ValueError('storage pointer alias cycle')
            seen.add(cursor); cursor=storage_aliases[cursor]
    reverse={}
    for alias,base in storage_aliases.items(): reverse.setdefault(base,[]).append(alias)
    returned=set(block.last_op.operands)
    op_offset={op:i for i,op in enumerate(block.ops)}
    selected=[]; refusals=[]
    requests=tuple(requests)
    if len({r.value for r in requests}) != len(requests):
        raise ValueError('duplicate storage request')
    for request in requests:
        value=request.value; reason=None
        if value not in all_values: raise ValueError('storage request value absent from current source')
        owner=value.owner; typ=value.type
        source_index=source_positions.get(owner)
        if (source_index is None or value in returned or value in storage_aliases
                or value not in materialized): reason='not_private_independent_materialization'
        elif not isinstance(typ,TensorType) or not isinstance(typ.encoding,NoneAttr) or any(d<=0 for d in typ.get_shape()):
            reason='not_positive_static_packed_tensor'
        elif not isinstance(typ.element_type,(IntegerType,AnyFloat)) or typ.element_type.bitwidth % 8:
            reason='unknown_byte_addressable_dtype'
        elif request.full_tensor_write is not True: reason='full_tensor_write_unproved'
        else:
            validate_private_workspaces((request.workspace,))
            size=prod(typ.get_shape())*(typ.element_type.bitwidth//8)
            if size != request.workspace.bytes or request.workspace.alignment != 64:
                reason='exact_logical_size_or_ABI_alignment_unproved'
        if reason:
            refusals.append(dict(source_op_index=source_index,reason=reason)); continue
        pending=[value]; visited=set(); end=op_offset[owner]; touches={owner}; escaped=False
        while pending:
            current=pending.pop()
            if current in visited: continue
            visited.add(current)
            interval=intervals.get(index.id(current))
            if interval is None or current in returned or facts[index.id(current)].escapes:
                escaped=True; break
            end=max(end,interval.end)
            pending.extend(reverse.get(current,()))
            for use in current.uses:
                operation=ancestor_in(use.operation,block)
                if operation not in source_positions: escaped=True; break
                touches.add(operation)
                # Lazy scalar/tensor recipes can retain current storage until
                # a downstream independently materialized result is complete.
                for result in operation.results:
                    if isinstance(result.type,TensorType) and (result not in materialized or result in storage_aliases):
                        pending.append(result)
        if escaped or not touches <= set(completed_operations):
            refusals.append(dict(source_op_index=source_index,reason='escaping_lazy_or_incomplete_use')); continue
        selected.append((value,op_offset[owner],end,request))
    aliases=[]; records=[]
    buckets={}
    for item in selected: buckets.setdefault(item[0].type,[]).append(item)
    for items in buckets.values():
        items.sort(key=lambda x:x[1]); ids={v:index.id(v) for v,_,_,_ in items}
        conflicts={ids[v]:{ids[w] for w,c,d,_ in items if v is not w and not (b<c or d<a)}
                   for v,a,b,_ in items}
        offsets,_=pack_disjoint([(ids[v],r.workspace.bytes) for v,_,_,r in items],conflicts)
        allocations={}
        for value,start,end,request in items:
            slot=offsets[ids[value]]
            if slot not in allocations: allocations[slot]=value; continue
            prior=allocations[slot]; aliases.append((value,prior))
            records.append(dict(source_op_index=source_positions[value.owner],
                result_index=value.index, allocation_source_op_index=source_positions[prior.owner],
                allocation_result_index=prior.index, live_interval=[start,end],
                bytes=request.workspace.bytes,alignment=64,
                full_writer_effect_sha256=request.workspace.effect_witness_sha256,
                completed_use_effects=[completed_operations[o] for o in operations
                    if start<=op_offset[o]<=end and o in completed_operations]))
    validate_analysis(module,analysis)
    return StorageReusePlan(source_sha256,tuple(aliases),tuple(records),tuple(refusals),analysis)


@dataclass(frozen=True)
class EmittedStorageRequest:
    value: object
    argument: object
    source_op_index: int
    workspace: PrivateWorkspaceContract
    full_tensor_write: bool
    begin: object
    complete: object


def plan_emitted_storage(function, *, source_sha256, requests, abi_bindings, terminal_effects,
                         terminal_completions=None):
    """Plan actual emitted pointer lifetimes, including loop address uses.

    ``abi_bindings`` names every actual ABI argument's source SSA value. Every
    non-load/store terminal pointer use needs a caller-supplied noescape effect
    hash; completion anchors separately witness completion of asynchronous uses.
    Pointer forwarding via CFG arguments/aggregates and unknown escapes refuse.
    Anchors must be outside cycles, and reuse requires old completion to dominate
    new beginning. Source order is attribution only, never execution evidence.
    With ``terminal_completions``, every terminal must name its actual enclosing
    emission end; the resulting lifetime extends through all those boundaries.
    Without it, ``request.complete`` must already cover every terminal.
    """
    from xdsl.irdl.dominance import DominanceInfo
    from .arena_bind import _cyclic_blocks, _reachable
    from .source_storage_emission import emitted_pointer_uses
    from xdsl.dialects.llvm import LLVMPointerType
    from dataclasses import replace

    if not _pin(source_sha256): raise ValueError('emitted storage requires original source SHA256')
    region=function.body
    blocks=tuple(region.blocks)
    if not blocks: raise ValueError('emitted storage requires a function body')
    arguments=set(blocks[0].args)
    if set(abi_bindings)!=arguments: raise ValueError('every actual ABI argument requires a source binding')
    if any(not isinstance(argument.type,LLVMPointerType) for argument in arguments):
        raise ValueError('storage adapter requires the actual pointer-only tensor ABI')
    if len(set(abi_bindings.values()))!=len(abi_bindings):
        raise ValueError('duplicate source ABI bindings require an explicit alias proof')
    block_ids={b:i for i,b in enumerate(blocks)}
    successors=[[block_ids[b] for b in block.last_op.successors] for block in blocks]
    reachable=_reachable(successors)
    if len(reachable)!=len(blocks): raise ValueError('unreachable emitted storage blocks require separate proof')
    cyclic=_cyclic_blocks(successors)
    dominance=DominanceInfo(region)
    operations=tuple(o for b in blocks for o in b.ops)
    op_ids={o:i for i,o in enumerate(operations)}
    local={o:i for b in blocks for i,o in enumerate(b.ops)}
    if any(o not in op_ids or not _pin(pin) for o,pin in terminal_effects.items()):
        raise ValueError('terminal effects require actual emitted operations and hashes')

    terminal_completions = {} if terminal_completions is None else dict(terminal_completions)
    if any(op not in op_ids or end not in op_ids for op,end in terminal_completions.items()):
        raise ValueError('terminal completion boundaries must name actual emitted operations')

    def precedes(left,right):
        if left.parent is right.parent: return local[left]<local[right]
        return dominance.dominates(left.parent,right.parent)

    requests=tuple(requests)
    if len({r.argument for r in requests})!=len(requests): raise ValueError('duplicate emitted storage request')
    selected=[]; refusals=[]; pointer_uses={}
    for request in requests:
        value=request.value; argument=request.argument; typ=value.type; reason=None
        if argument not in arguments or abi_bindings[argument] is not value:
            raise ValueError('storage request differs from actual ABI source binding')
        if type(request.source_op_index) is not int or request.source_op_index<0:
            raise ValueError('emitted storage requires original source operation index')
        if request.begin not in op_ids or request.complete not in op_ids:
            raise ValueError('storage lifetime anchors absent from emitted function')
        if (not hasattr(value.owner,'results') or any(use.operation.name=='func.return' for use in value.uses)):
            reason='public_input_or_output'
        elif (not isinstance(typ,TensorType) or not isinstance(typ.encoding,NoneAttr)
              or any(d<=0 for d in typ.get_shape())
              or not isinstance(typ.element_type,(IntegerType,AnyFloat)) or typ.element_type.bitwidth%8):
            reason='not_positive_static_packed_machine_tensor'
        elif request.full_tensor_write is not True: reason='full_tensor_write_unproved'
        else:
            validate_private_workspaces((request.workspace,))
            if request.workspace.bytes!=prod(typ.get_shape())*(typ.element_type.bitwidth//8) or request.workspace.alignment!=64:
                reason='logical_size_or_ABI_alignment_unproved'
        if not reason and (block_ids[request.begin.parent] in cyclic or block_ids[request.complete.parent] in cyclic
                           or not precedes(request.begin,request.complete)):
            reason='noncyclic_ordered_lifetime_boundary_unproved'
        closure=emitted_pointer_uses(function,argument,terminal_effects,operation_indices=op_ids)
        pointer_uses[argument]=closure.terminals
        terminals=closure.terminals
        if not reason: reason=closure.refusal
        if not reason and terminal_completions:
            # Caller supplies the end of the actual enclosing emission. This
            # handles loops and asynchronous device commands without assuming
            # that the last instruction using an integer address is a fence.
            if any(op not in terminal_completions for op in terminals):
                reason='terminal_completion_boundary_missing'
            else:
                for op in terminals:
                    end=terminal_completions[op]
                    after=set()
                    for successor in successors[block_ids[end.parent]]:
                        after.update(_reachable(successors,successor))
                    if (block_ids[end.parent] in cyclic
                        or (op.parent is end.parent and local[op]>=local[end])
                        or block_ids[op.parent] in after):
                        reason='terminal_completion_precedes_access_or_cycles'; break
                ends=[request.complete,*(terminal_completions[op] for op in terminals)]
                candidates=[end for end in ends
                    if block_ids[end.parent] not in cyclic
                    and all(other is end or precedes(other,end) for other in ends)]
                if not reason and not candidates:
                    reason='terminal_completions_not_totally_ordered'
                elif not reason:
                    request=replace(request,complete=candidates[0])
        if not reason:
            after_completion=set()
            for successor in successors[block_ids[request.complete.parent]]:
                after_completion.update(_reachable(successors,successor))
            for op in terminals:
                if not (op is request.begin or precedes(request.begin,op)):
                    reason='access_before_full_writer_begin'; break
                if (op.parent is request.complete.parent and local[op]>local[request.complete]
                    or block_ids[op.parent] in after_completion):
                    reason='access_after_completion_boundary'; break
        if reason:
            refusals.append(dict(source_op_index=request.source_op_index,reason=reason));continue
        selected.append(request)
    aliases=[]; records=[]
    buckets={}
    for request in selected: buckets.setdefault(request.value.type,[]).append(request)
    for items in buckets.values():
        ids={r.argument:str(op_ids[r.begin])+':'+str(r.source_op_index)+':'+str(r.argument.index) for r in items}
        if len(set(ids.values()))!=len(items): raise ValueError('ambiguous emitted storage identities')
        conflicts={ids[a.argument]:{ids[b.argument] for b in items if a is not b and not (
            precedes(a.complete,b.begin) or precedes(b.complete,a.begin))} for a in items}
        # Dominator order is not necessarily block text order; only bind a slot
        # to a prior completed allocation, independently checked below.
        offsets,_=pack_disjoint([(ids[r.argument],r.workspace.bytes) for r in items],conflicts)
        slots={}
        for request in items:
            slot=offsets[ids[request.argument]]
            if slot not in slots: slots[slot]=request; continue
            owner=slots[slot]
            if not precedes(owner.complete,request.begin):
                refusals.append(dict(source_op_index=request.source_op_index,reason='allocation_owner_not_prior_completed'));continue
            aliases.append((request.argument,owner.argument))
            records.append(dict(source_op_index=request.source_op_index,
                allocation_source_op_index=owner.source_op_index,
                allocation_completion_instruction_index=op_ids[owner.complete],
                begin_instruction_index=op_ids[request.begin], completion_instruction_index=op_ids[request.complete],
                bytes=request.workspace.bytes,alignment=64,
                full_writer_effect_sha256=request.workspace.effect_witness_sha256,
                terminal_instruction_indices=[op_ids[op] for op in pointer_uses[request.argument]],
                terminal_effect_sha256s=sorted({terminal_effects[op] for op in
                    pointer_uses[request.argument] if op in terminal_effects})))
    # Pin the actual emitted IR; emitted aliases are LLVM argument identities.
    from hashlib import sha256
    from merlin.xdsl_dialects._common import text
    @dataclass(frozen=True)
    class EmissionIdentity:
        source_sha256: str
    identity=EmissionIdentity(sha256(text(function).encode()).hexdigest())
    return StorageReusePlan(source_sha256,tuple(aliases),tuple(records),tuple(refusals),identity)
