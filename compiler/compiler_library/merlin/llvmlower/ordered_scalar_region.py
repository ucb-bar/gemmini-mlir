"""Exact scalar-region cloning for static trailing tensor reductions.

No arithmetic rewrite or domain assumption: retain every source operation,
attribute, operand and lexicographic reduction step, including NaN/tie behavior.
Tensor SSA and ordinary bufferization retain all alias/lifetime authority.
"""

SOURCE = r'''
def _ordered_region_spec(op):
    from torch_mlir import ir
    import ast
    if (op.operation.name != 'linalg.generic' or len(op.results) != 1
        or len(op.operands) < 2 or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1):
        return None
    allowed = {'indexing_maps','iterator_types','operandSegmentSizes'}
    if any(n not in allowed and not n.startswith('prov.') for n in op.attributes):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any('strictfp' in n or 'strictfp' in str(ancestor.attributes[n]) for n in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    tensors = []
    for v in op.operands:
        if not isinstance(v.type, ir.RankedTensorType): return None
        t = ir.RankedTensorType(v.type)
        if any(x < 0 or x >= 2**63 for x in t.shape): return None
        if not (isinstance(t.element_type, ir.F32Type) or
                isinstance(t.element_type, ir.IntegerType) and ir.IntegerType(t.element_type).width in (1,8,16,32,64)):
            return None
        # Encoded layouts have independent physical contracts.
        if t.encoding is not None: return None
        tensors.append(t)
    if op.results[0].type != op.operands[-1].type: return None
    its = [str(x) for x in op.attributes['iterator_types']]
    np = sum(x == '#linalg.iterator_type<parallel>' for x in its)
    nr = len(its)-np
    if nr == 0 or its != ['#linalg.iterator_type<parallel>']*np + ['#linalg.iterator_type<reduction>']*nr:
        return None
    maps = [ir.AffineMapAttr(a).value for a in op.attributes['indexing_maps']]
    if len(maps) != len(tensors): return None
    nd = len(its)
    extents = [None]*nd
    for m,t in zip(maps,tensors):
        if m.n_dims != nd or m.n_symbols or len(m.results) != len(t.shape): return None
        for e,size in zip(m.results,t.shape):
            if isinstance(e,ir.AffineDimExpr):
                d=ir.AffineDimExpr(e).position
                if extents[d] is not None and extents[d] != size: return None
                extents[d]=size
    if any(x is None for x in extents): return None
    positions=[]
    for e in maps[-1].results:
        if not isinstance(e,ir.AffineDimExpr): return None
        positions.append(ir.AffineDimExpr(e).position)
    if positions != list(range(np)): return None
    # Closed nonnegative linear maps: prove every tensor read in bounds for
    # every iteration. Reject symbols, floors, modulus and unknown geometry.
    def polynomial(node):
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value>=0:
            return [0]*nd+[node.value]
        if isinstance(node,ast.Name) and node.id in ['d'+str(i) for i in range(nd)]:
            c=[0]*(nd+1); c[int(node.id[1:])]=1; return c
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
            a,b=polynomial(node.left),polynomial(node.right)
            return None if a is None or b is None else [x+y for x,y in zip(a,b)]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
            a,b=polynomial(node.left),polynomial(node.right)
            if a is None or b is None: return None
            if not any(a[:-1]): return [a[-1]*x for x in b]
            if not any(b[:-1]): return [b[-1]*x for x in a]
        return None
    for m,t in zip(maps,tensors):
        for e,size in zip(m.results,t.shape):
            try: c=polynomial(ast.parse(str(e),mode='eval').body)
            except (SyntaxError,ValueError): return None
            if c is None: return None
            if all(extents) and c[-1]+sum(x*(n-1) for x,n in zip(c[:-1],extents))>=size: return None
    block=op.regions[0].blocks[0]
    body=list(block.operations)
    if len(block.arguments)!=len(op.operands) or not body or body[-1].operation.name!='linalg.yield': return None
    pure={'arith.constant','arith.addf','arith.subf','arith.mulf','arith.divf',
          'arith.cmpf','arith.cmpi','arith.select','arith.ori','arith.andi','arith.xori',
          'arith.addi','arith.subi','arith.muli','arith.extsi','arith.extui','arith.trunci',
          'arith.sitofp','arith.uitofp','arith.maximumf','arith.maxnumf','arith.maxsi','arith.minsi'}
    known=set(block.arguments)
    for child in body[:-1]:
        if child.operation.name not in pure or child.regions or len(child.results)!=1: return None
        if any(v not in known for v in child.operands): return None
        if any(n not in {'value','predicate','fastmath'} and not n.startswith('prov.') for n in child.attributes): return None
        if 'fastmath' in child.attributes and str(child.attributes['fastmath'])!='#arith.fastmath<none>': return None
        known.update(child.results)
    if len(body[-1].operands)!=1 or body[-1].operands[0] not in known: return None
    return maps,extents,np


def _scalarize_ordered_regions(ctx,module):
    from torch_mlir import ir
    todo=[]
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec=_ordered_region_spec(child)
                    if spec is not None: todo.append((child,spec))
                    else: walk(child.operation)
    walk(module.operation)
    with ctx:
        for old,(maps,extents,np) in todo:
            with old.location,ir.InsertionPoint(old):
                index=ir.IndexType.get(); tensor=old.results[0].type
                scalar=ir.RankedTensorType(tensor).element_type
                def create(name,operands=(),results=(),attributes=None,regions=0):
                    return ir.Operation.create(name,operands=list(operands),results=list(results),attributes=attributes or {},regions=regions)
                def constant(n):
                    return create('arith.constant',results=[index],attributes={'value':ir.IntegerAttr.get(index,n)}).results[0]
                zero,one=constant(0),constant(1)
                limits=[constant(n) for n in extents]
                def extract(v,m,coords):
                    indices=[]
                    for expr in m.results:
                        amap=ir.AffineMap.get(len(extents),0,[expr])
                        indices.append(create('affine.apply',coords,[index],{'map':ir.AffineMapAttr.get(amap)}).results[0])
                    return create('tensor.extract',[v,*indices],[ir.RankedTensorType(v.type).element_type]).results[0]
                def recurrence(depth,coords,acc):
                    if depth<len(extents):
                        loop=create('scf.for',[zero,limits[depth],one,acc],[scalar],regions=1)
                        block=ir.Block.create_at_start(loop.regions[0],[index,scalar])
                        with ir.InsertionPoint(block):
                            value=recurrence(depth+1,[*coords,block.arguments[0]],block.arguments[1])
                            create('scf.yield',[value])
                        return loop.results[0]
                    source=old.regions[0].blocks[0]
                    bindings={a:extract(v,m,coords) for a,v,m in zip(list(source.arguments)[:-1],list(old.operands)[:-1],maps[:-1]) if list(a.uses)}
                    bindings[source.arguments[-1]]=acc
                    for child in list(source.operations)[:-1]:
                        clone=create(child.operation.name,[bindings[v] for v in child.operands],[v.type for v in child.results],{n:child.attributes[n] for n in child.attributes})
                        bindings.update(zip(child.results,clone.results))
                    return bindings[list(source.operations)[-1].operands[0]]
                def outputs(depth,current,coords):
                    if depth<np:
                        loop=create('scf.for',[zero,limits[depth],one,current],[tensor],regions=1)
                        block=ir.Block.create_at_start(loop.regions[0],[index,tensor])
                        with ir.InsertionPoint(block):
                            updated=outputs(depth+1,block.arguments[1],[*coords,block.arguments[0]])
                            create('scf.yield',[updated])
                        return loop.results[0]
                    seed=create('tensor.extract',[current,*coords],[scalar]).results[0]
                    value=recurrence(np,coords,seed)
                    return create('tensor.insert',[value,current,*coords],[tensor]).results[0]
                result=outputs(0,old.operands[-1],[])
                for n in old.attributes:
                    if n.startswith('prov.'): result.owner.attributes[n]=old.attributes[n]
                previous = ir.StringAttr(old.attributes['prov.transforms']).value + ',' if 'prov.transforms' in old.attributes else ''
                result.owner.attributes['prov.transforms']=ir.StringAttr.get(previous + 'scalar_ordered_tensor_regions')
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)
'''
