"""Source-directed orchestration of shared schedules and target emission."""
from __future__ import annotations

from math import prod
from hashlib import sha256
import json
import os

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, i8, i32

from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.compile.logical_abi import kernel_arg_order, logical_abi, logical_interface, logical_outputs
from pathlib import Path
from merlin.llvmlower.host_builder import FnBuilder, PTR

from frontend import parse_source
from encoding import Encoder, kernel_module
from dense import emit_dense


def compile_source(source):
    module = parse_source(source)
    if 'merlin_iface.version' not in module.attributes:
        from linalg_backend import compile_linalg
        return compile_linalg(source)
    cb = parse_interface_mlir(source)
    cb['backend'] = 'authored-structured-gemmini'
    # Read every result's own logical type, including commits (whose grammar
    # inventory intentionally does not duplicate the result declaration).
    for op in module.body.block.ops:
        name = op.attributes.get('name')
        if isinstance(name, StringAttr) and op.results and hasattr(op.results[0].type,'get_shape'):
            if op.name != 'merlin_iface.tensor':
                ty = op.results[0].type
                cb['tensors'][name.data] = dict(shape=list(ty.get_shape()),
                    dtype=str(ty.element_type),role='output')
    if any('requant' in cmd.get('attributes',{}).get('epilogue',())
           for cmd in cb['commands']):
        cb['commands']=[]
        cb['declined']={'op':'requant', 'reason':
            'Selected readout rounds half-even; source requant requires half-up. '
            'No declared contraction-scoped hardware route preserves that boundary.'}
        return cb,str(ModuleOp([]))
    abi=logical_abi(Path(__file__).parent/'data/mlir_oot_backend_contract.yaml')
    # Publish declarations in the same order as source results. The interface
    # parser materializes direct results before deferred accumulator commits;
    # keep the serialized boundary consistent with the logical pointer ABI.
    outputs = logical_outputs(cb, abi.published_by)
    cb['tensors'] = {**{n:t for n,t in cb['tensors'].items() if n not in outputs},
                     **{n:cb['tensors'][n] for n in outputs}}
    # Readout pooling has a distinct rounded boundary. Keep the full-width
    # intermediate in a typed caller-owned scratch buffer, never on the stack.
    pool_buffers={}
    direct_pools=set()
    pre_residents,pre_products={},{}
    for i,command in enumerate(cb['commands']):
        code=command['opcode']; op=command.get('operands',{}); attrs=command.get('attributes',{})
        if code=='RES_PACK': pre_residents[op['dst']]=op['src']
        elif code in ('MATMUL','MATMUL_RESIDENT'): pre_products[op['dst']]=(op['lhs'],pre_residents.get(op['rhs'],op['rhs']))
        if 'maxpool' not in attrs.get('epilogue',[]): continue
        if code=='COMMIT':
            a,w=pre_products[op['src']]
            dims=[cb['tensors'][a]['shape'][0],cb['tensors'][w]['shape'][-1]]
            direct_pools.add(id(command))
            continue
        elif code=='CONV2D':
            from conv_pool import supports_conv_pool
            pool_attrs={k:v for k,v in attrs.items() if k!='kernel'}
            if supports_conv_pool(cb['tensors'][op['ifm']]['shape'],attrs['kernel'],**pool_attrs):
                direct_pools.add(id(command))
                continue
            b,h,w,c=cb['tensors'][op['ifm']]['shape']; kh,kw,ci,co=attrs['kernel']
            sh,sw=attrs.get('stride',[1,1]); dh,dw=attrs.get('dilation',[1,1]); pt,pl,pb,pr=attrs.get('padding',[0,0,0,0])
            oh=(h+pt+pb-dh*(kh-1)-1)//sh+1; ow=(w+pl+pr-dw*(kw-1)-1)//sw+1
            dims=[b,oh,ow,co]
        else: continue
        name=f'__pool_acc_{i}'
        pool_buffers[id(command)]=(name,dims)
    # A staged column readout is still accelerator work. One statically bounded
    # primitive tile is private to this invocation and synchronously reused.
    # It does not change the source pointer ABI or allocate an activation image.
    readout_buffers = {}
    settings_path = Path(__file__).with_name('DENSE_SCHEDULE.json')
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    packed_input_buffers = {}
    pool_mode = os.environ.get('CONV_POOL_SCHEDULE', settings.get('conv_pool_schedule', 'sync'))
    for command in cb['commands']:
        attrs = command.get('attributes', {})
        if (command['opcode'] != 'CONV2D' or id(command) not in direct_pools
                or attrs.get('pool_schedule', pool_mode) not in ('packed_raw', 'packed_shared')):
            continue
        from conv_pool import packed_input_workspace_bytes
        packed_attrs = {k: v for k, v in attrs.items() if k != 'kernel'}
        required = packed_input_workspace_bytes(
            cb['tensors'][command['operands']['ifm']]['shape'], attrs['kernel'], **packed_attrs)
        if required:
            packed_input_buffers[id(command)] = required
    readout_mode = os.environ.get('CONV_READOUT', settings.get('conv_readout', 'direct'))
    if readout_mode in ('staged_columns', 'regional_staged', 'staged_epochs'):
        from convolution import _resources
        edge, _, _, _ = _resources(str(Path(__file__).parent/'data/facts.json'))
        for command in cb['commands']:
            attrs = command.get('attributes', {})
            scales = attrs.get('acc_scale_by_column', ())
            if (command['opcode'] != 'CONV2D' or id(command) in direct_pools
                or id(command) in pool_buffers or attrs.get('output_dtype', 'i32') != 'i8'
                or not isinstance(scales, (list, tuple)) or len(set(scales)) < 2):
                continue
            if readout_mode == 'regional_staged':
                from conv_readout import regional_staged_candidate
                ci = attrs['kernel'][2]
                co = attrs['kernel'][3]
                if not regional_staged_candidate(ci, tuple(attrs['kernel'][:2]), co, edge, edge)['geometry_selected']:
                    continue
            # The schedule may use fewer rows at a narrow image edge; allocating
            # the selected primitive square also covers every legal tail.
            dims = [edge, edge]
            if readout_mode == 'staged_epochs':
                from conv_readout import staged_epoch_workspace
                workspace = staged_epoch_workspace(
                    cb['tensors'][command['operands']['ifm']]['shape'], attrs['kernel'],
                    strides=attrs.get('stride', (1, 1)),
                    dilation=attrs.get('dilation', (1, 1)),
                    padding=attrs.get('padding', (0, 0, 0, 0)))
                dims = workspace['shape']
                if (not dims or any(type(v) is not int or v <= 0 for v in dims)
                        or prod(dims)*4 != workspace['bytes'] or workspace['bytes'] > 4096):
                    raise ValueError('staged readout epoch exceeds bounded private scratch')
            readout_buffers[id(command)] = ('__conv_column_readout', dims)
    if pool_buffers:
        original=logical_interface(cb,abi)
        cb['kernel_abi']={'kind':'whole_program','args':[{'tensor':b.name,'access':'read' if b.kind=='input' else 'write'} for b in original],
                          'outputs':[b.name for b in original if b.kind=='output']}
        for name,dims in pool_buffers.values():
            cb['tensors'][name]={'shape':dims,'dtype':'i32','role':'intermediate'}
            cb['kernel_abi']['args'].append({'tensor':name,'access':'write'})
    fn, pointers = FnBuilder.for_logical_kernel(cb,abi=abi)
    packed_input_pointer = None
    if packed_input_buffers:
        packed_bytes = max(packed_input_buffers.values())
        packed_input_pointer = fn.add(llvm.AllocaOp(fn.const(packed_bytes), i8, alignment=64)).res
        cb.setdefault('params', {})['private_packed_input_workspace'] = dict(
            bytes=packed_bytes, alignment=64, lifetime='invocation', ownership='private',
            initialization='shared NHWC window-panel preparation writes each byte before device read',
            completion='consumer completion before panel overwrite and return')
    readout_pointer = None
    if readout_buffers:
        # The selected target permits an 8192-byte static entry frame. This
        # <=4096-byte allocation is constant, aligned, outside all loops,
        # never escapes, and is fully written before a device reads it. Object
        # stack admission remains part of ordinary native checks.
        elements = max(prod(dims) for _,dims in readout_buffers.values())
        readout_pointer = fn.add(llvm.AllocaOp(fn.const(elements), i32, alignment=64)).res
        cb.setdefault('params', {})['private_readout_workspace'] = dict(
            bytes=elements*4, alignment=64, lifetime='invocation',
            ownership='private', initialization='full primitive tile before read',
            completion='synchronous device fence before reuse and return')
    enc = Encoder(fn)
    enc.configure_execute()
    residents, products = {}, {}
    receipts = []

    def resolve(name):
        return residents.get(name,name)

    def shape(name):
        return cb['tensors'][resolve(name)]['shape']

    def opts(attrs):
        stages = attrs.get('epilogue',[])
        unknown = set(stages) - {'acc_scale','relu','bias','bias_add','maxpool'}
        if unknown:
            raise ValueError(f'epilogue lowering not implemented yet: {sorted(unknown)}')
        scale = attrs.get('acc_scale',1.0)
        if 'acc_scale_by_column' in attrs:
            scale = attrs['acc_scale_by_column']
        bias = attrs.get('bias') or attrs.get('bias_tensor')
        return dict(out_dtype=attrs.get('output_dtype','i32'),scale=scale,
            relu='relu' in stages,bias=pointers[bias] if bias else None)

    def destination(command,attrs):
        if id(command) not in pool_buffers: return command['operands']['dst'],attrs
        name,dims=pool_buffers[id(command)]
        raw=dict(attrs,output_dtype='i32',epilogue=[s for s in attrs.get('epilogue',[]) if s in ('bias','bias_add')])
        raw.pop('acc_scale',None);raw.pop('acc_scale_by_column',None)
        return name,raw

    def finish_pool(command,attrs):
        if id(command) in pool_buffers:
            from elementwise import emit_pool_from_i32
            name,dims=pool_buffers[id(command)]
            emit_pool_from_i32(enc,pointers[name],pointers[command['operands']['dst']],dims,**attrs)

    def matmul(lhs,rhs,dst,attrs):
        s,t = shape(lhs),shape(rhs)
        if len(s)!=2 or len(t)!=2 or s[1]!=t[0]:
            raise ValueError('dense lowering requires compatible rank-two operands')
        receipts.append(emit_dense(enc,pointers[resolve(lhs)],pointers[resolve(rhs)],pointers[dst],
            s[0],s[1],t[1],**opts(attrs)))

    completed_commits = set()
    commands = cb['commands']
    for command_index, command in enumerate(commands):
        if id(command) in completed_commits:
            continue
        code = command['opcode']
        operands, attrs = command.get('operands',{}), command.get('attributes',{})
        if code == 'RES_PACK':
            residents[operands['dst']] = resolve(operands['src'])
        elif code in ('MATMUL','MATMUL_RESIDENT'):
            products[operands['dst']] = (operands['lhs'],operands['rhs'])
        elif code == 'COMMIT':
            lhs,rhs = products[operands['src']]
            # An explicit immutable resident handle owns one local-storage
            # epoch across adjacent independent products. Metadata declarations
            # do not issue device effects; any other source command ends it.
            if rhs in residents and id(command) not in direct_pools and id(command) not in pool_buffers:
                local_products = dict(products)
                candidates = []
                member_commands = []
                for member in commands[command_index:]:
                    member_code = member['opcode']
                    mo, ma = member.get('operands', {}), member.get('attributes', {})
                    if member_code in ('MATMUL', 'MATMUL_RESIDENT'):
                        local_products[mo['dst']] = (mo['lhs'], mo['rhs'])
                        continue
                    if member_code != 'COMMIT' or id(member) in direct_pools or id(member) in pool_buffers:
                        break
                    ml, mr = local_products[mo['src']]
                    if mr != rhs:
                        break
                    left_shape, right_shape = shape(ml), shape(mr)
                    if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
                        break
                    candidates.append(dict(lhs=pointers[resolve(ml)], out=pointers[mo['dst']],
                        m=left_shape[0], k=left_shape[1], n=right_shape[1],
                        lhs_name=resolve(ml), out_name=mo['dst'], **opts(ma)))
                    member_commands.append(member)
                    if len(candidates) == 128:
                        break
                outputs = [p['out_name'] for p in candidates]
                inputs = {p['lhs_name'] for p in candidates} | {resolve(rhs)}
                independent = (len(outputs) == len(set(outputs)) and not inputs.intersection(outputs))
                if len(candidates) >= 2 and independent:
                    from resident_chain import emit_resident_products
                    receipt = emit_resident_products(enc, pointers[resolve(rhs)], tuple(candidates),
                        source_identity=sha256(source.encode()).hexdigest(),
                        external_buffers_disjoint=True)
                    if receipt is not None:
                        receipt['source_resident_handle'] = rhs
                        receipt['source_outputs'] = outputs
                        receipt['original_source_binding'] = 'adjacent independent products of exact source resident handle'
                        receipts.append(receipt)
                        completed_commits.update(id(m) for m in member_commands)
                        continue
            if id(command) in direct_pools:
                from dense_pool import emit_dense_pool
                s,t=shape(lhs),shape(rhs)
                pool_opts=dict(attrs);pool_opts['bias']=opts(attrs)['bias']
                receipts.append(emit_dense_pool(enc,pointers[resolve(lhs)],pointers[resolve(rhs)],pointers[operands['dst']],s[0],s[1],t[1],**pool_opts))
                continue
            dst,raw=destination(command,attrs)
            matmul(lhs,rhs,dst,raw)
            finish_pool(command,attrs)
        elif code == 'CONV2D':
            from convolution import emit_conv
            inp,weight = resolve(operands['ifm']),resolve(operands['weight'])
            if id(command) in direct_pools:
                from conv_pool import emit_conv_pool
                pool_opts=dict(attrs);pool_opts['bias']=opts(attrs)['bias']
                pool_opts.pop('kernel',None)
                if id(command) in packed_input_buffers:
                    pool_opts.update(packed_input_workspace=packed_input_pointer,
                        packed_input_workspace_bytes=packed_input_buffers[id(command)])
                receipts.append(emit_conv_pool(enc,pointers[inp],pointers[weight],pointers[operands['dst']],shape(inp),attrs['kernel'],**pool_opts))
                continue
            dst,raw=destination(command,attrs)
            readout_options = {}
            if id(command) in readout_buffers:
                scratch_name, scratch_dims = readout_buffers[id(command)]
                readout_options = dict(readout_mode=readout_mode,
                    readout_scratch=readout_pointer,
                    readout_scratch_bytes=prod(scratch_dims)*4)
            receipts.append(emit_conv(enc,pointers[inp],pointers[weight],pointers[dst],
                shape(inp),attrs['kernel'],strides=attrs.get('stride',(1,1)),
                padding=attrs.get('padding',(0,0,0,0)),dilation=attrs.get('dilation',(1,1)),
                **opts(raw), **readout_options))
            finish_pool(command,attrs)
        elif code in ('BATCHED_MATMUL','ATTENTION_QK','ATTENTION_PV'):
            from attention import emit_batched,emit_qk,emit_pv
            emitter={'BATCHED_MATMUL':emit_batched,'ATTENTION_QK':emit_qk,'ATTENTION_PV':emit_pv}[code]
            left=operands.get('lhs',operands.get('a',operands.get('q',operands.get('p'))))
            right=operands.get('rhs',operands.get('w',operands.get('k',operands.get('v'))))
            receipts.append(emitter(enc,pointers[resolve(left)],pointers[resolve(right)],pointers[operands['dst']],shape(left),shape(right),**opts(attrs)))
        elif code == 'MOVEMENT':
            from elementwise import emit_movement
            emit_movement(enc,pointers[operands['src']],pointers[operands['dst']],
                shape(operands['src']),**attrs)
        elif code == 'RESIDUAL_ADD':
            from elementwise import emit_residual
            proof=emit_residual(enc,pointers[operands['lhs']],pointers[operands['rhs']],
                pointers[operands['dst']],shape(operands['lhs']),**attrs)
            receipts.append(dict(kind='residual_add',operands=dict(operands),
                source_shape=list(shape(operands['lhs'])),numeric_proof=proof))
        elif code == 'BIAS_ADD':
            from elementwise import emit_bias_add
            emit_bias_add(enc,pointers[operands['src']],pointers[operands['bias']],
                pointers[operands['dst']],shape(operands['src']),**attrs)
        elif code == 'MAXPOOL2D':
            from elementwise import emit_maxpool
            emit_maxpool(enc,pointers[operands['src']],pointers[operands['dst']],
                shape(operands['src']),**attrs)
        elif code in ('EVICT','ROW_MAJOR_VIEW'):
            pass
        else:
            raise ValueError(f'compiler lowering not implemented: {code}')
    enc.fence()
    function = llvm.FuncOp('gemmini_kernel',llvm.LLVMFunctionType([PTR]*len(pointers),llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr('external'),body=fn.finish())
    emitted = kernel_module(function)
    cb.setdefault('params', {})['source_schedule_emissions'] = receipts
    # Describe the actual accepted interface computations. Residency handles,
    # views and lifetime markers are not additional computation regions.
    # This is a source-bound placement declaration, not execution or coverage
    # evidence; ordinary command and native checks still establish those.
    families = {
        'MATMUL': 'contraction', 'MATMUL_RESIDENT': 'contraction', 'CONV2D': 'convolution',
        'BATCHED_MATMUL': 'contraction', 'ATTENTION_QK': 'contraction',
        'ATTENTION_PV': 'contraction', 'MOVEMENT': 'movement',
        'RESIDUAL_ADD': 'elementwise_map', 'BIAS_ADD': 'elementwise_map',
        'MAXPOOL2D': 'pooling',
    }
    source_identity = sha256(source.encode()).hexdigest()
    cb['params']['lane_placement'] = [
        dict(lane='gemmini', family=families[command['opcode']],
             source_sha256=source_identity, command_index=index,
             operation=command['opcode'], operands=dict(command['operands']),
             reason='accepted typed interface computation lowered through target encoder',
             evidence='source-bound emitted placement; execution checked separately')
        for index, command in enumerate(cb['commands'])
        if command['opcode'] in families
    ]
    return cb,str(emitted)
