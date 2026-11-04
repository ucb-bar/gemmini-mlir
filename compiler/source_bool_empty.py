"""Prove storage-only private SSA empty destinations; never introduce i1 storage.

The existing shared host lowering remains responsible for scalar SSA placeholders
and actual full writers. An empty is classified only when no source observation
can reach its undefined elements before an independent complete writer.
"""
from hashlib import sha256
from math import prod
from pathlib import Path
from xdsl.dialects.builtin import TensorType
from xdsl.ir.affine import AffineDimExpr


def prove_private_empty(op, *, pointers, source_positions):
    if (op.name!='tensor.empty' or op.operands or len(op.results)!=1
            or not isinstance(op.results[0].type,TensorType)
            or str(op.results[0].type.element_type)!='i1'):
        return None
    value=op.results[0]
    if value in pointers or op not in source_positions:return None
    shape=tuple(value.type.get_shape())
    if any(type(d)is not int or d<=0 for d in shape):return None
    uses=tuple(value.uses)
    if not uses:return None
    writers=[]
    for use in uses:
        writer=use.operation
        if (writer.name!='linalg.generic' or writer not in source_positions
                or source_positions[writer]<=source_positions[op]
                or len(writer.results)!=1 or writer.results[0].type!=value.type
                or use.index!=len(writer.operands)-1
                or len(writer.regions)!=1 or len(writer.regions[0].blocks)!=1):return None
        counts=writer.properties.get('operandSegmentSizes')
        if counts is None or tuple(counts.get_values())!=(len(writer.operands)-1,1):return None
        maps=tuple(x.data for x in writer.properties['indexing_maps'])
        iterators=tuple(writer.properties['iterator_types'])
        if (len(iterators)!=len(shape) or any(x.data.value!='parallel' for x in iterators)
                or len(maps)!=len(writer.operands)):return None
        # Equal-shaped identity domains make every destination coordinate appear
        # exactly once; no projection, reduction, partial slice or alias assumed.
        for operand,mapping in zip(writer.operands,maps):
            if (not isinstance(operand.type,TensorType) or tuple(operand.type.get_shape())!=shape
                    or mapping.num_dims!=len(shape) or mapping.num_symbols
                    or len(mapping.results)!=len(shape)
                    or any(not isinstance(e,AffineDimExpr) or e.position!=i
                           for i,e in enumerate(mapping.results))):return None
        body=writer.regions[0].block
        if (len(body.args)!=len(writer.operands) or body.args[-1].uses
                or body.last_op.name!='linalg.yield' or len(body.last_op.operands)!=1
                or body.last_op.operands[0].type!=value.type.element_type):return None
        writers.append({'source_op_index':source_positions[writer],
            'operation':writer.name,'destination_operand_index':use.index,
            'destination_block_argument_unused':True,'iteration_domain':list(shape),
            'output_map':'identity','iteration_kind':'all_parallel',
            'writer_sha256':sha256(str(writer).encode()).hexdigest()})
    return {'schema':'authored.private_ssa_empty_storage.v1',
        'source_op_index':source_positions[op],'source_operation_sha256':sha256(str(op).encode()).hexdigest(),
        'shape':list(shape),'dtype':'i1','element_count':prod(shape),
        'representation':'private host scalar SSA; absent from physical tensor ABI',
        'physical_allocation':False,'source_element_initialization_required':False,
        'all_uses_are_independent_full_writers':True,'writers':writers,
        'rule':'Every use is the sole identity output of an all-parallel equal-domain generic whose destination scalar argument is unused; each undefined element is overwritten before observation.',
        'device_coverage_credit':False}


def classify_private_empty(host,op):
    """Classify after unchanged shared lowering; no executable IR is mutated."""
    proof=prove_private_empty(op,pointers=host.ptrs,source_positions=host.source_positions)
    if proof is None:return False
    from merlin.llvmlower.host_tensor_codegen import TensorVal
    value=host.vals.get(op.results[0])
    if not isinstance(value,TensorVal) or len(value.elems)!=proof['element_count']:return False
    root=Path(__file__).parent
    proof['implementation_owners']={p:sha256((root/p).read_bytes()).hexdigest()
        for p in ('source_bool_empty.py','linalg_backend.py',
                  'compiler_library/merlin/llvmlower/host_tensor_codegen.py')}
    if not hasattr(host,'source_empty_storage_proofs'):host.source_empty_storage_proofs={}
    host.source_empty_storage_proofs[op]=proof
    return True
