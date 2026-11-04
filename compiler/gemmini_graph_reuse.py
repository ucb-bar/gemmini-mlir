"""Optional checked shared-input products with independent original readouts.

The interface-to-linalg bridge states only the source MATMUL recurrences. The
original command identity and readout attributes remain bound separately.
"""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input, SharedInputEffect
from merlin.compile.scheduling.shared_input_verification import verify_shared_input
from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseResources
import gemmini_encode as E
from gemmini_dense import geometry
from gemmini_readout import emit_readout, source_scales, _relu, _scale_bits


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _bridge(m, k, widths):
    args = [f'%x: tensor<{m}x{k}xi8>'] + [f'%w{i}: tensor<{k}x{n}xi8>' for i,n in enumerate(widths)]
    types = [f'tensor<{m}x{n}xi32>' for n in widths]
    lines = ['builtin.module {', 'func.func @products('+', '.join(args)+') -> ('+', '.join(types)+') {']
    for i,n in enumerate(widths):
        lines += [f'%z{i} = arith.constant 0 : i32', f'%s{i} = tensor.splat %z{i} : tensor<{m}x{n}xi32>',
            f'''%r{i} = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%x, %w{i} : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%s{i} : tensor<{m}x{n}xi32>) {{
^bb0(%a: i8, %b: i8, %c: i32):
%aa = arith.extsi %a : i8 to i32
%bb = arith.extsi %b : i8 to i32
%p = arith.muli %aa, %bb : i32
%v = arith.addi %c, %p : i32
linalg.yield %v : i32
}} -> tensor<{m}x{n}xi32>''']
    lines += ['func.return '+', '.join('%r'+str(i) for i in range(len(widths)))+' : '+', '.join(types), '}', '}']
    module = parse_mlir_text('\n'.join(lines)); module.verify()
    return module


def _pack(effects, widths, k, tile):
    packets=[]
    for effect in effects:
        member,op=effect.member,effect.operation
        if isinstance(op,Load):
            stride=k if op.role=='lhs' else widths[member]
            packets.append(('load',member,0 if op.role=='lhs' else 1,op.dram_row*stride+op.dram_col,stride,op.row,op.rows,op.cols))
        elif isinstance(op,Preload):
            packets.append(('preload',member,E.GARBAGE if op.weight_row is None else op.weight_row,
                op.accumulator_row|E.ACC|(E.ACCUMULATE if op.accumulate else 0),op.weight_rows,op.cols,op.rows))
        elif isinstance(op,Compute):
            packets.append(('compute',member,op.input_row,E.GARBAGE,op.rows,op.cols,tile,tile,not op.fresh_weights))
        elif isinstance(op,Store):
            packets.append(('store',member,op.dram_row*widths[member]+op.dram_col,op.accumulator_row,op.rows,op.cols))
        else:
            raise ValueError('unsupported shared product effect')
    return tuple(packets)


def _project(packets, widths, k, tile):
    """Decode actual target arguments and DRAM offsets, not planned effects."""
    effects=[]; local={}; pending_fresh=None
    for packet in packets:
        kind,member,*a=packet
        if kind=='load':
            stream,offset,stride,row,rows,cols=a
            role='lhs' if stream==0 else 'weight'
            if stream not in (0,1) or stride!=(k if stream==0 else widths[member]):raise ValueError('changed source stride')
            dr,dc=divmod(offset,stride)
            for r in range(rows):local[role,row+r]=(dr+r,dc)
            block=(dr//tile,dc//tile)
            op=Load(role,block,dr,dc,rows,cols,row,dc//tile if stream==0 else dr//tile)
        elif kind=='preload':
            weight,acc,depth,cols,rows=a
            fresh=weight!=E.GARBAGE
            if not (acc&E.ACC) or acc&~(E.ACC|E.ACCUMULATE|((1<<29)-1)):raise ValueError('changed accumulator flags')
            block=tuple(x//tile for x in local['weight',weight]) if fresh else None
            op=Preload(weight if fresh else None,block,acc&((1<<29)-1),bool(acc&E.ACCUMULATE),rows,cols,depth)
            pending_fresh=fresh
        elif kind=='compute':
            lhs,bd,rows,depth,br,bc,stay=a
            if bd!=E.GARBAGE or (br,bc)!=(tile,tile) or pending_fresh is None or stay==pending_fresh:raise ValueError('changed compute primitive')
            block=tuple(x//tile for x in local['lhs',lhs])
            op=Compute(lhs,block,rows,depth,not stay);pending_fresh=None
        elif kind=='store':
            offset,acc,rows,cols=a; dr,dc=divmod(offset,widths[member]);op=Store(dr,dc,rows,cols,acc)
        else:raise ValueError('unknown shared target packet')
        effects.append(SharedInputEffect(member,op))
    return tuple(effects)


@dataclass(frozen=True)
class SharedProductsPlan:
    products: tuple
    widths: tuple
    m: int
    k: int
    module: object
    schedule: object
    resources: object
    storage: object
    packets: tuple
    source_identity: str
    binding_sha256: str

    def verify(self):
        if _digest(self.products)!=self.binding_sha256:raise ValueError('shared source bindings changed')
        return verify_shared_input(_project(self.packets,self.widths,self.k,self.schedule.primitive_tile[0]),
            module=self.module,group=self.schedule.group,primitive_tile=self.schedule.primitive_tile,
            resources=self.resources,storage=self.storage,max_commands=8192)


def plan_shared_products(tensors, products, *, source_identity, source_module=None, source_group=None):
    """Plan 2+ adjacent pure zero-seeded source products, or return None.

    ``products`` contains resolved ``lhs``, ``rhs``, ``dst``, and exact commit
    ``attrs``. The caller owns adjacency/effect guards and supplies the digest of
    the original command buffer; this function never chooses products by name.
    No original output can alias an input or another output.
    """
    if not isinstance(source_identity,str) or not source_identity or not 2<=len(products)<=16:return None
    products=tuple(copy.deepcopy(products))
    try:
        lhs=products[0]['lhs'];shape=tensors[lhs]['shape']
        if len(shape)!=2 or tensors[lhs]['dtype']!='i8':return None
        m,k=shape
        if any(type(v) is not int or v<=0 for v in (m,k)):return None
        widths=[]; outputs=[]
        for product in products:
            if product['lhs']!=lhs or set(product)-{'lhs','rhs','dst','attrs','source_command_indices'}:return None
            rhs,dst=product['rhs'],product['dst']; attrs=product.get('attrs',{})
            wt=tensors[rhs];out=tensors[dst]
            if wt['dtype']!='i8' or len(wt['shape'])!=2 or wt['shape'][0]!=k:return None
            n=wt['shape'][1];dtype=attrs.get('output_dtype',out['dtype'])
            if type(n) is not int or n<=0 or out['shape']!=[m,n] or out['dtype']!=dtype or dtype not in ('i8','i32'):return None
            if any(key not in {'name','output_dtype','epilogue','acc_scale','scale','relu','activation','bound_lsb'} and not key.startswith('prov.') for key in attrs):return None
            stages=attrs.get('epilogue',[]) or []
            if isinstance(stages,str):stages=[stages]
            if any(stage not in ('acc_scale','relu') for stage in stages):return None
            scale=source_scales(attrs)
            if isinstance(scale,(tuple,list)) or not isinstance(scale,(int,float)):return None
            import math
            if not math.isfinite(scale):return None
            relu=_relu(attrs)
            if dtype=='i8':_scale_bits(scale)
            if dtype=='i32' and (scale!=1 or relu):return None
            widths.append(n);outputs.append(dst)
        if len(set(outputs))!=len(outputs) or set(outputs)&({lhs}|{p['rhs'] for p in products}):return None
        module=_bridge(m,k,widths) if source_module is None else source_module
        groups=find_shared_input_groups(module)
        if source_group is None:
            if len(groups)!=1:return None
            group=groups[0]
        else:
            source_group.verify_current(module)
            group=source_group
        if (group.m,group.k,tuple(group.widths))!=(m,k,tuple(widths)):return None
        tile,operand,oprows,accumulator,accrows=geometry();split=oprows//2
        storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
            DenseMemoryView('rhs',operand.name,operand.row_bytes,split,oprows-split),
            DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accrows))
        resources=DenseResources(oprows*operand.row_bytes,accrows*accumulator.row_bytes,
            max_m_blocks=8,max_k_blocks=1,max_n_blocks=8,partial_output_tiles=True,partial_output_blocks=True)
        schedule=schedule_shared_input(module,group,primitive_tile=(tile,tile,tile),resources=resources,storage=storage,max_commands=8192)
        if schedule is None:return None
        packets=_pack(schedule.operations,widths,k,tile)
        plan=SharedProductsPlan(products,tuple(widths),m,k,module,schedule,resources,storage,packets,source_identity,_digest(products))
        plan.verify()
        return plan
    except (ValueError,KeyError,TypeError,IndexError):
        return None


def emit_shared_products(fn, plan, pointers):
    """Emit the checked transformed product; every original commit is retained."""
    realization=plan.verify();tile=plan.schedule.primitive_tile[0]
    E.fence(fn);E.config_ex(fn,dataflow='ws')
    for packet in plan.packets:
        kind,member,*a=packet
        if kind=='load':
            stream,offset,stride,row,rows,cols=a
            name=plan.products[0]['lhs'] if stream==0 else plan.products[member]['rhs']
            E.config_load(fn,stride,stream=stream)
            E.mvin(fn,E.pointer_offset(fn,pointers[name],offset),row,rows,cols,stream=stream)
            E.fence(fn)
        elif kind=='preload':
            weight,acc,depth,cols,rows=a;E.preload(fn,weight,acc,depth,cols,rows,cols)
        elif kind=='compute':
            lhs,bd,rows,depth,br,bc,stay=a
            E.compute(fn,lhs,bd,rows,depth,br,bc,accumulate=stay);E.fence(fn)
        elif kind=='store':
            offset,acc,rows,cols=a;product=plan.products[member];attrs=product.get('attrs',{})
            dtype=attrs.get('output_dtype','i32');width=1 if dtype=='i8' else 4
            emit_readout(fn,E.pointer_offset(fn,pointers[product['dst']],offset*width),acc,rows,cols,
                output_stride=plan.widths[member]*width,attrs=attrs,output_dtype=dtype)
            E.fence(fn)
    E.fence(fn)
    return {'library':'schedule_shared_input','source_identity':plan.source_identity,
        'source_bindings_sha256':plan.binding_sha256,'emitted_effects_sha256':_digest(plan.packets),
        'typed_product_proof_sha256':plan.schedule.group.source_ir_sha256,
        'source_shapes':[(plan.m,plan.k,n) for n in plan.widths],
        'source_command_indices':[p.get('source_command_indices',[]) for p in plan.products],
        'realization':realization,'independent_readouts':[p['dst'] for p in plan.products]}


def verify_numerics(plan, *, max_products=2_000_000):
    """Explicit development check: execute decoded effects on signed integers.

    This routine is never invoked during ordinary compilation or selection.
    It compares complete original outputs, including each original readout.
    """
    from merlin.llvmlower.requantization import quantized
    plan.verify()
    if plan.m*plan.k*sum(plan.widths)>max_products:raise ValueError('finite numeric check budget exceeded')
    def i32(x):return ((int(x)+(1<<31))%(1<<32))-(1<<31)
    lhs=[[((r*37+c*19+11)%255)-128 for c in range(plan.k)] for r in range(plan.m)]
    rhs=[[[((member*43+r*13+c*29+7)%255)-128 for c in range(n)] for r in range(plan.k)] for member,n in enumerate(plan.widths)]
    outputs=[[[None]*n for _ in range(plan.m)] for n in plan.widths]
    memory={};acc={};weights=None;descriptor=None
    def readout(member,x):
        attrs=plan.products[member].get('attrs',{});dtype=attrs.get('output_dtype','i32')
        stages=attrs.get('epilogue',[]) or []
        relu=_relu(attrs)
        return int(quantized(x,(float(source_scales(attrs)),),relu)) if dtype=='i8' else x
    for effect in _project(plan.packets,plan.widths,plan.k,plan.schedule.primitive_tile[0]):
        member,op=effect.member,effect.operation
        if isinstance(op,Load):
            source=lhs if op.role=='lhs' else rhs[member]
            for row in range(op.rows):memory[op.role,op.row+row]=list(source[op.dram_row+row][op.dram_col:op.dram_col+op.cols])
        elif isinstance(op,Preload):
            if op.weight_row is not None:weights=[memory['weight',op.weight_row+r][:op.cols] for r in range(op.weight_rows)]
            descriptor=op
        elif isinstance(op,Compute):
            for r in range(op.rows):
                vector=memory['lhs',op.input_row+r]
                previous=acc.get(descriptor.accumulator_row+r,[0]*descriptor.cols) if descriptor.accumulate else [0]*descriptor.cols
                acc[descriptor.accumulator_row+r]=[i32(previous[c]+sum(vector[z]*weights[z][c] for z in range(op.cols))) for c in range(descriptor.cols)]
        elif isinstance(op,Store):
            for r in range(op.rows):
                values=acc.pop(op.accumulator_row+r)
                for c in range(op.cols):outputs[member][op.dram_row+r][op.dram_col+c]=readout(member,values[c])
    for member,n in enumerate(plan.widths):
        for r in range(plan.m):
            for c in range(n):
                expected=readout(member,i32(sum(lhs[r][z]*rhs[member][z][c] for z in range(plan.k))))
                if outputs[member][r][c]!=expected:raise ValueError(f'changed shared output member{member}[{r},{c}]')
    return {'status':'complete_integer_effect_execution_pass','products':plan.m*plan.k*sum(plan.widths),
            'output_elements':plan.m*sum(plan.widths),'outputs':[p['dst'] for p in plan.products],
            'emitted_effects_sha256':_digest(plan.packets)}


def plan_boundary_view_reuse(module, needed, returned):
    """Reuse only exact duplicate immutable views at private graph boundaries.

    The shared transform runs on a proof clone. Original operations, source
    indices, provenance and published outputs remain untouched. The graph owns
    each retained allocation until this synchronous invocation returns; device
    consumers only read it. No representation or storage survives invocation.
    """
    from merlin.transforms.tensor_views import canonicalize_tensor_views
    from merlin.transforms.certificates import ir_digest
    from xdsl.dialects import builtin
    original = tuple(module.walk())
    functions = [op for op in original if op.name == 'func.func']
    if len(functions) != 1:
        return {}, None
    block = functions[0].body.block
    top = {op: i for i, op in enumerate(block.ops) if op.name != 'func.return'}
    before = ir_digest(module)
    try:
        witnesses = canonicalize_tensor_views(module.clone())
    except ValueError:
        # A refused lineage/equivalence proof retains ordinary materialization.
        return {}, None
    aliases, records = {}, []
    public = set(returned)
    for witness in witnesses:
        operation, retained = original[witness.operation_index], original[witness.retained_index]
        if operation not in top or retained not in top or len(operation.results) != 1 or len(retained.results) != 1:
            continue
        value, target = operation.results[0], retained.results[0]
        while target in aliases:
            target = aliases[target]
        if (value not in needed or value in public or target in public
                or value.type != target.type or not isinstance(value.type, builtin.TensorType)):
            continue
        aliases[value] = target
        records.append({'source_op_index': top[operation], 'retained_op_index': top[target.owner],
                        'operation': operation.name, 'result_type': str(value.type),
                        'shared_witness_sha256': witness.before_sha256})
    if ir_digest(module) != before:
        raise ValueError('view preparation analysis changed the original source')
    if not aliases:
        return {}, None
    return aliases, {'mechanism': 'merlin.transforms.tensor_views.canonicalize_tensor_views',
                     'original_ir_sha256': before, 'rewrites': records,
                     'lifetime': 'private synchronous invocation; immutable read-only device inputs'}

