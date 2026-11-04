"""Original-SSA adapters for projected signed integer contractions.

No provenance or model identity participates in matching. Tensor maps and the
scalar SSA expression determine the product; the shared dense adapter owns its
accelerator schedule. Host work copies layouts only.
"""
from __future__ import annotations
from math import prod
from xdsl.ir.affine import AffineDimExpr
from xdsl.dialects.builtin import IntegerAttr, DenseIntOrFPElementsAttr
try:
    from .products import _contract
    from . import target
except ImportError:
    from products import _contract
    import target


def _attr(op, name):
    return op.properties.get(name, op.attributes.get(name))


def _refuse(graph, reason):
    graph.last_product_refusal = reason
    return False


def _is_product(op):
    block = op.body.block
    if [str(v.type) for v in block.args] != ['i8', 'i8', 'i32']:
        return False
    env = dict(zip(block.args, [('a',), ('b',), ('old',)]))
    for node in block.ops:
        args = [env.get(v) for v in node.operands]
        if node.name == 'linalg.yield':
            return args == [('sum',)] and node is block.last_op
        if len(node.results) != 1 or str(node.results[0].type) != 'i32' or node.regions:
            return False
        if node.name == 'arith.extsi' and args in ([('a',)], [('b',)]):
            result = ('wide', args[0])
        elif node.name == 'arith.muli' and len(args)==2 and set(args)=={('wide',('a',)),('wide',('b',))}:
            result = ('product',)
        elif node.name == 'arith.addi' and len(args)==2 and set(args)=={('old',),('product',)}:
            result = ('sum',)
        else:
            return False
        env[node.results[0]] = result
    return False


def _zero(value):
    owner = value.owner
    if getattr(owner,'name','') in ('linalg.fill','tensor.splat'):
        scalar = owner.operands[0].owner
        if getattr(scalar,'name','') == 'arith.constant':
            attr = _attr(scalar,'value')
            return isinstance(attr,IntegerAttr) and attr.value.data == 0
    if getattr(owner,'name','') == 'arith.constant':
        attr = _attr(owner,'value')
        return isinstance(attr,DenseIntOrFPElementsAttr) and all(v==0 for v in attr.get_values())
    return False


def _add_seed(graph, seed, out):
    """Add the source i32 seed through accumulator DMA, after product stores."""
    try:
        from .elementwise import _tiles, _address
    except ImportError:
        from elementwise import _tiles, _address
    shape = tuple(out.type.get_shape())
    rows, cols = prod(shape[:-1]), shape[-1] if shape else 1
    b = graph.b
    target.cfg_ld(b,cols*4,stream=0,shrunk=False)
    target.cfg_ld(b,cols*4,stream=1,shrunk=False)
    target.cfg_st(b,cols*4)
    def emit(row,col,nr,nc):
        target.mvin(b,_address(b,graph.ptrs[out],row,col,cols,4),target.ACC,nr,nc)
        b.fence()
        target.mvin(b,_address(b,graph.ptrs[seed],row,col,cols,4),target.ACC|target.ACCUMULATE,nr,nc,stream=1)
        b.fence()
        target.mvout(b,_address(b,graph.ptrs[out],row,col,cols,4),target.ACC|target.FULL,nr,nc)
        b.fence()
    _tiles(b,rows,cols,emit)


def lower_graph_product(graph, op):
    """Lower noncanonical projected i8 products, returning False for other ops.

    ``graph.last_product_refusal`` records why an apparent integer contraction
    was refused. The caller retains responsibility for other operator forms.
    Plain rank-two matmul stays with the graph's existing dense implementation.
    """
    name = op.name
    named = name in ('linalg.batch_matmul','linalg.batch_matmul_transpose_a',
                     'linalg.batch_matmul_transpose_b','linalg.matmul_transpose_a',
                     'linalg.matmul_transpose_b')
    if name != 'linalg.generic' and not named:
        return False
    ins = tuple(op.inputs); seeds = tuple(op.outputs); outs = tuple(op.results)
    if len(ins)!=2 or len(seeds)!=1 or len(outs)!=1:
        return False
    if [str(v.type.element_type) for v in (*ins,*seeds,*outs)] != ['i8','i8','i32','i32']:
        return False
    if name == 'linalg.generic':
        if not _is_product(op):
            return _refuse(graph,'scalar SSA is not signed i8 extension, i32 multiplication and accumulator addition')
        amaps = [a.data for a in op.indexing_maps]
        if len(amaps)!=3 or len({a.num_dims for a in amaps})!=1 or any(a.num_symbols for a in amaps):
            return _refuse(graph,'product requires three symbol-free maps with a common domain')
        if any(not isinstance(e,AffineDimExpr) for a in amaps for e in a.results):
            return _refuse(graph,'product map contains affine arithmetic rather than projected dimensions')
        maps = tuple(tuple(e.position for e in a.results) for a in amaps)
        rank = amaps[0].num_dims
        reductions = tuple(i for i,t in enumerate(op.iterator_types) if 'reduction' in str(t))
        if maps == ((0,2),(2,1),(0,1)) and rank==3 and reductions==(2,):
            return False
    else:
        lhs_shape = tuple(ins[0].type.get_shape())
        operand_rank = len(lhs_shape)
        if operand_rank < 2 or any(len(v.type.get_shape()) != operand_rank for v in (*ins,*seeds,*outs)):
            return _refuse(graph,'named product ranks must agree')
        batch = tuple(range(operand_rank-2))
        m,k,n = operand_rank-2,operand_rank-1,operand_rank
        maps = (batch+((k,m) if name.endswith('transpose_a') else (m,k)),
                batch+((n,k) if name.endswith('transpose_b') else (k,n)),batch+(m,n))
        rank = operand_rank+1
        reductions = (k,)
    if any(len(set(dims))!=len(dims) for dims in maps):
        return _refuse(graph,'product map repeats a dimension')
    left,right,result = map(set,maps)
    if set(reductions) != left & right - result:
        return _refuse(graph,'product reduction iterator dimensions disagree with maps')
    seed = seeds[0]; out = outs[0]
    if tuple(seed.type.get_shape()) != tuple(out.type.get_shape()):
        return _refuse(graph,'product initializer and result shapes differ')
    if getattr(seed.owner,'name','') == 'tensor.empty':
        return _refuse(graph,'product reads an uninitialized tensor.empty accumulator')
    tensors = {str(i):{'shape':list(v.type.get_shape()),'dtype':str(v.type.element_type)} for i,v in enumerate((*ins,out))}
    pointers = {str(i):graph.ptrs[v] for i,v in enumerate((*ins,out))}
    # All shape/map legality checks are performed by the shared normalization
    # before it emits copies or accelerator calls. Nonzero source seeds are
    # added in their original output layout after the product is scattered.
    _contract(graph.b,{'tensors':tensors},('0','1','2'),pointers,maps,rank,{})
    if not _zero(seed):
        _add_seed(graph,seed,out)
    graph.device_count += 1
    graph._last_device = True
    return True
