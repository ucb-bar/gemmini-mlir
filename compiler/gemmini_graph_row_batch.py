"""Opt-in checked row assembly for independent thin-M common-RHS products.

Original graph grouping owns source independence and original operation lists.
This bridge explicitly assembles separate ABI input buffers into fresh private
rows, applies the shared tensor row-batching pass to their exact source views,
consumes its actual candidate geometry, emits one ordinary dense contraction,
and scatters all original i32 outputs after device completion. Packing and
scatter are ordinary shared host-builder copies, included in emitted latency.
No installation, global patch, CPU contraction or production policy lives here.
"""
from dataclasses import dataclass
import hashlib
import json


def _root_state(roots):
    return tuple((root,tuple(root.operands),tuple(root.results),str(root)) for root in roots)


@dataclass(frozen=True)
class RowBatchPlan:
    roots: tuple
    source_identity: str
    source_geometry: tuple
    rows: tuple
    bridge: object
    batching: object
    candidate_shape: tuple
    original_state: tuple

    def revalidate(self):
        if _root_state(self.roots)!=self.original_state:
            raise ValueError('row batch original source changed')
        self.batching.revalidate()
        if _candidate_shape(self.batching.candidate,self.rows)!=self.candidate_shape:
            raise ValueError('row batch actual candidate/scatter coordinates changed')

    def record(self):
        self.revalidate()
        m,k,n=self.source_geometry
        return dict(schema='gemmini.graph_row_assembly.v1',source_identity=self.source_identity,
            original_members=len(self.roots),original_geometry=self.source_geometry,
            candidate_shape=self.candidate_shape,assembly_rows=self.rows,
            assembly='each original ABI lhs occupies its own complete contiguous row interval; no source byte changed',
            scatter='each returned candidate slice maps to its original liveout; no numerical operation',
            private_bytes=self.candidate_shape[0]*(k+4*n),
            library=self.batching.record(),terminal='ordinary emit_dense -> dense_terminal',timing='UNKNOWN')


def _candidate_shape(candidate, rows):
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.dialects import tensor
    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
    functions=[op for op in candidate.body.block.ops if op.name=='func.func']
    if len(functions)!=1:raise ValueError('row batch requires one actual candidate function')
    block=functions[0].regions[0].block
    products=[op for op in block.ops if type(op) is GenericOp]
    if len(products)!=1:raise ValueError('shared row pass must realize exactly one product')
    shape=static_zero_integer_matmul_geometry(products[0])
    returned=block.last_op
    if len(returned.operands)!=len(rows):raise ValueError('row batch lost an original returned result')
    for value,(offset,count) in zip(returned.operands,rows):
        view=value.owner
        if (type(view) is not tensor.ExtractSliceOp or view.source is not products[0].results[0]
                or tuple(view.static_offsets.get_values())!=(offset,0)
                or tuple(view.static_sizes.get_values())!=(count,shape[2])
                or tuple(view.static_strides.get_values())!=(1,1)):
            raise ValueError('row batch candidate scatter does not preserve an original liveout')
    return shape


def plan_row_batch(group, *, tile_edge, max_members=8, max_private_bytes=16<<20):
    """Build actual source-view assembly then apply the shared row-batching pass.

    The library's original route requires one common LHS allocation root; this
    source has separate original ABI inputs. The explicit copy intervals create
    precisely that missing common root. Its immutable row views clone every
    original scalar body and map; the shared independent inverse check verifies
    their replacement by one product and the original result slices.
    """
    from xdsl.dialects import arith,builtin,func,tensor
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import Block,Region
    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
    from merlin.transforms.dense_row_batching import plan_source_dense_row_batching,DenseRowBatchingBudgets
    from merlin.transforms.pipeline_state import PipelineContext
    roots=tuple(group['roots']);m,k,n=group['source_geometry']
    if (type(tile_edge) is not int or tile_edge<=0 or type(max_members) is not int
            or not 2<=len(roots)<=max_members<=8 or m<=0 or len(roots)*m>tile_edge
            or type(max_private_bytes) is not int or max_private_bytes<=0
            or len(roots)*m*(k+4*n)>max_private_bytes):return None
    source_identity=group['source_identity']
    if (type(source_identity) is not str or len(source_identity)!=64
            or any(c not in '0123456789abcdef' for c in source_identity)):
        raise ValueError('row assembly requires original source SHA256')
    if any(type(root) is not GenericOp or static_zero_integer_matmul_geometry(root)!=(m,k,n)
            or root.operands[1] is not roots[0].operands[1] for root in roots):return None
    total=len(roots)*m;rows=tuple((member*m,m) for member in range(len(roots)))
    lhs_type=builtin.TensorType(builtin.i8,(total,k));rhs_type=roots[0].operands[1].type
    out_types=[root.results[0].type for root in roots]
    block=Block(arg_types=[lhs_type,rhs_type]);returned=[]
    for root,(row,count) in zip(roots,rows):
        view=tensor.ExtractSliceOp.from_static_parameters(block.args[0],(row,0),(count,k))
        zero=arith.ConstantOp(builtin.IntegerAttr(0,builtin.i32))
        seed=tensor.SplatOp(zero.result,[],root.results[0].type)
        product=GenericOp.create(operands=[view.result,block.args[1],seed.result],
            result_types=[root.results[0].type],properties=dict(root.properties),
            attributes=dict(root.attributes),regions=[root.regions[0].clone()])
        block.add_ops([view,zero,seed,product]);returned.append(product.results[0])
    block.add_op(func.ReturnOp(*returned))
    function=func.FuncOp.from_region('source_rows',[lhs_type,rhs_type],out_types,Region(block))
    bridge=builtin.ModuleOp([function]);bridge.verify()
    context=PipelineContext(numerical_policy='original-source-ordered-i8-i8-i32',
        execution='private-complete-row-assembly:'+source_identity)
    batching=plan_source_dense_row_batching(bridge,context,DenseRowBatchingBudgets(
        max_members=max_members,max_batches=1,max_source_elements=max_private_bytes))
    if batching is None:return None
    shape=_candidate_shape(batching.candidate,rows)
    if shape!=(total,k,n):raise ValueError('shared batching candidate changes source dimensions')
    result=RowBatchPlan(roots,source_identity,(m,k,n),rows,bridge,batching,shape,_root_state(roots))
    result.revalidate();return result


def copy_bytes(fn,source,destination,count):
    """Bit-preserving counted copies using shared LLVM builder, with byte tails.

    Word accesses explicitly use alignment one; no ABI alignment is assumed.
    The private destination/source intervals are disjoint. There is no numeric
    conversion or reduction. Compiler lowering owns word access legalization.
    """
    from xdsl.dialects import llvm,builtin
    from gemmini_encode import pointer_offset
    if type(count) is not int or count<0:raise ValueError('row copy requires bounded byte count')
    words=count//8
    def word(iv):
        src=fn.gep(source,iv,builtin.i64);dst=fn.gep(destination,iv,builtin.i64)
        value=fn.add(llvm.LoadOp(src,builtin.i64,alignment=1)).results[0]
        fn.add(llvm.StoreOp(value,dst,alignment=1))
    fn.for_loop(0,words,1,word,retain_loop=True)
    for byte in range(words*8,count):
        src=pointer_offset(fn,source,fn.const(byte));dst=pointer_offset(fn,destination,fn.const(byte))
        value=fn.add(llvm.LoadOp(src,builtin.i8,alignment=1)).results[0]
        fn.add(llvm.StoreOp(value,dst,alignment=1))


def emit_group(fb, group, ptrs, names, boundary_name, *, emit_resident_group,
               emit_dot, fence, emit_dense=None, scratch=None, copier=copy_bytes,
               tile_edge=None, max_members=8, max_private_bytes=16<<20):
    """Drop-in pure parent hook; ordinary RHS groups remain the safe fallback."""
    from gemmini_graph_rhs_groups import emit_group as incumbent
    from gemmini_dense import geometry
    if emit_dense is None:
        from gemmini_dense import emit_dense
    if scratch is None:
        from gemmini_support import scratch_buffer as scratch
    if tile_edge is None:tile_edge=geometry()[0]
    plan=plan_row_batch(group,tile_edge=tile_edge,max_members=max_members,max_private_bytes=max_private_bytes)
    if plan is None:
        return incumbent(fb,group,ptrs,names,boundary_name,emit_resident_group=emit_resident_group,emit_dot=emit_dot,fence=fence)
    plan.revalidate();total,k,n=plan.candidate_shape;m=plan.source_geometry[0]
    packed=scratch(fb,total*k,'row_batch_lhs');output=scratch(fb,total*n*4,'row_batch_output')
    for root,(row,count) in zip(plan.roots,plan.rows):
        from gemmini_encode import pointer_offset
        copier(fb,ptrs[root.operands[0]],pointer_offset(fb,packed,fb.const(row*k)),count*k)
    # Complete CPU input stores before device loads; complete device stores
    # before CPU scatter. The ordinary dense owner retains its own epoch laws.
    fence(fb)
    emit_dense(fb,packed,ptrs[plan.roots[0].operands[1]],output,total,k,n,output_dtype='i32')
    fence(fb)
    for root,(row,count) in zip(plan.roots,plan.rows):
        from gemmini_encode import pointer_offset
        copier(fb,pointer_offset(fb,output,fb.const(row*n*4)),ptrs[root.results[0]],count*n*4)
    receipt=plan.record()
    receipt['copy_implementation']='shared FnBuilder counted i64 bit copies with alignment1 and exact byte tails'
    reads=list(dict.fromkeys(boundary_name(v) for root in plan.roots for v in root.operands[:2]))
    return reads,[names[root.results[0]] for root in plan.roots],receipt
