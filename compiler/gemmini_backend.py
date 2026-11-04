"""Source command dataflow to checked, source-shaped target schedules."""
from __future__ import annotations

import copy
from math import prod

from merlin.llvmlower.host_builder import FnBuilder, PTR
from gemmini_frontend import normalize_source
from gemmini_support import scratch_buffer, finish_module


def compile_source(source, *, evidence_sink=None):
    model = normalize_source(source)
    if model['grammar'] == 'linalg':
        from gemmini_graph import compile_linalg
        cb, lowered = compile_linalg(source)
        if evidence_sink is not None:
            evidence_sink([], cb, lowered)
        return cb, lowered
    cb = copy.deepcopy(model['cb'])
    cb['backend'] = 'authored-gemmini-structured'
    from gemmini_requant import source_admission
    refusal = source_admission(cb)
    if refusal:
        cb['commands'] = []
        cb['declined'] = {'reason': refusal}
        if evidence_sink is not None:
            evidence_sink([], cb, 'module {}')
        return cb, 'module {}'
    tensors = cb.setdefault('tensors', {})
    args = model['logical_args']
    for a in args:
        tensors.setdefault(a['name'], {'shape':a['shape'], 'dtype':a['dtype'],
                                      'role':'input' if a['access']=='read' else 'output'})
    cb['kernel_abi'] = {'kind':'whole_program',
        'args':[{'tensor':a['name'],'access':a['access']} for a in args],
        'outputs':[a['name'] for a in args if a['access']=='write']}
    fn = FnBuilder([PTR] * len(args))
    pointers = {a['name']:p for a,p in zip(args,fn.entry.args)}
    handles = {}
    products = {}
    receipts = []
    private_tensors = []

    def resolve(name):
        seen = set()
        while name in handles:
            if name in seen:
                raise ValueError('cyclic source resident handle')
            seen.add(name)
            name = handles[name]
        return name

    def pointer(name):
        name = resolve(name)
        if name not in pointers:
            spec = tensors.get(name)
            if not spec:
                raise ValueError('unbound source tensor '+str(name))
            width = {'i8':1,'i16':2,'i32':4,'i64':8,'f32':4}[spec['dtype']]
            pointers[name] = scratch_buffer(fn,prod(spec['shape'])*width,'tensor_'+str(len(pointers)))
        return pointers[name]

    def emit_product(opcode, operands, attrs):
        if 'bias' not in operands and isinstance(attrs.get('bias'), str):
            operands = {**operands, 'bias':attrs['bias']}
        dst = operands['dst']
        dtype = attrs.get('output_dtype',tensors.get(dst,{}).get('dtype','i32'))
        if 'maxpool' in attrs.get('epilogue', ()):
            from gemmini_pool import emit_pool
            from merlin.compile.tuning_knobs import knob
            pool_route = knob('pool_route',default='spill',values=('spill','producer','producer_striped','readout_cache','auto','strip_epochs','auto_epochs'))
            if pool_route in ('auto','auto_epochs'):
                use_pool_epochs = pool_route == 'auto_epochs'
                from gemmini_pool_store import select_pool_store_route
                source_left = resolve(operands.get('lhs',operands.get('a'))) if opcode in ('MATMUL','MATMUL_RESIDENT') else resolve(operands.get('input',operands.get('ifm',operands.get('x',operands.get('lhs')))))
                source_right = resolve(operands.get('rhs',operands.get('w',operands.get('weight'))))
                pool_route = select_pool_store_route(kind=opcode,
                    input_shape=tensors[source_left]['shape'],weight_shape=tensors[source_right]['shape'],
                    attrs=attrs,bias_present='bias' in operands)['route']
                if use_pool_epochs and pool_route == 'readout_cache':
                    pool_route = 'strip_epochs'
            if pool_route in ('producer','producer_striped','readout_cache','strip_epochs'):
                from gemmini_composition import emit_producer_pool, CompositionUnavailable
                left=resolve(operands.get('lhs',operands.get('a'))) if opcode in ('MATMUL','MATMUL_RESIDENT') else resolve(operands.get('input',operands.get('ifm',operands.get('x',operands.get('lhs')))))
                right=resolve(operands.get('rhs',operands.get('w',operands.get('weight'))))
                try:
                    return emit_producer_pool(fn,pointer(left),pointer(right),pointer(dst),kind=opcode,
                        input_shape=tensors[left]['shape'],weight_shape=tensors[right]['shape'],attrs=attrs,
                        bias=pointer(operands['bias']) if 'bias' in operands else None,
                        geometry=('strip_epochs' if pool_route == 'strip_epochs' else
                                  'readout_cache' if pool_route == 'readout_cache' else
                                  'striped' if pool_route == 'producer_striped' else 'halo'))
                except CompositionUnavailable:
                    pass
            if opcode in ('MATMUL', 'MATMUL_RESIDENT'):
                left = resolve(operands.get('lhs', operands.get('a')))
                right = resolve(operands.get('rhs', operands.get('w')))
                raw_shape = (tensors[left]['shape'][0], tensors[right]['shape'][-1])
            elif opcode == 'CONV2D':
                from gemmini_conv import _geometry
                left = resolve(operands.get('input', operands.get('ifm', operands.get('x', operands.get('lhs')))))
                right = resolve(operands.get('weight', operands.get('w', operands.get('rhs'))))
                window, channels = _geometry(tensors[left]['shape'], tensors[right]['shape'], attrs)
                raw_shape = (window.input_shape[0], *window.output_spatial, channels)
            else:
                raise ValueError('spatial pooling requires an original spatial product')
            temp = '__producer_raw_' + str(len(pointers))
            tensors[temp] = {'shape': list(raw_shape), 'dtype': 'i32', 'role': 'intermediate'}
            private_tensors.append(temp)
            producer_attrs = {k:v for k,v in attrs.items() if not k.startswith('pool') and k not in ('acc_scale','acc_scale_by_column','scale','output_scale','relu','activation')}
            producer_attrs['epilogue'] = [s for s in attrs.get('epilogue', ()) if s == 'bias_add']
            producer_attrs['output_dtype'] = 'i32'
            producer = emit_product(opcode, {**operands, 'dst':temp}, producer_attrs)
            pool = emit_pool(fn,pointer(temp),pointer(dst),shape=raw_shape,attrs=attrs,dtype='i32')
            return {'producer':producer,'pool':pool}
        bias = pointer(operands['bias']) if 'bias' in operands else None
        if opcode in ('MATMUL','MATMUL_RESIDENT'):
            from gemmini_dense import emit_dense
            left = resolve(operands.get('lhs',operands.get('a')))
            right = resolve(operands.get('rhs',operands.get('w')))
            ashape,wshape = tensors[left]['shape'],tensors[right]['shape']
            if len(ashape)!=2 or len(wshape)!=2 or ashape[1]!=wshape[0]:
                raise ValueError('rank-two dense source dimensions disagree')
            return emit_dense(fn,pointer(left),pointer(right),pointer(dst),
                ashape[0],ashape[1],wshape[1],attrs=attrs,bias=bias,
                lhs_dtype=tensors[left]['dtype'],rhs_dtype=tensors[right]['dtype'],output_dtype=dtype)
        if opcode=='CONV2D':
            from gemmini_conv import emit_conv
            left=resolve(operands.get('input',operands.get('ifm',operands.get('x',operands.get('lhs')))))
            right=resolve(operands.get('weight',operands.get('w',operands.get('rhs'))))
            return emit_conv(fn,pointer(left),pointer(right),pointer(dst),input_shape=tensors[left]['shape'],
                weight_shape=tensors[right]['shape'],attrs=attrs,bias=bias,output_dtype=dtype)
        from gemmini_attention import emit_attention
        bound = {name:pointer(name) for name in tensors if name in pointers or tensors[name].get('role') in ('input','weight','output')}
        bound.update({name:pointer(name) for name in operands.values() if resolve(name) in tensors})
        return emit_attention(fn,opcode,bound,tensors,operands,attrs)

    def shared_products_at(start):
        """Collect adjacent source product/readout pairs without moving effects."""
        from merlin.compile.tuning_knobs import knob
        shared_mode = knob('graph_shared_inputs', default='off', values=('off','on','counted'))
        if shared_mode == 'off':
            return None
        import hashlib, json
        if shared_mode == 'counted':
            from gemmini_shared_counted import plan_counted_shared_products as plan_shared_products
        else:
            from gemmini_graph_reuse import plan_shared_products
        candidates=[]
        commands=cb['commands']
        index=start
        while index+1<len(commands) and len(candidates)<16:
            product,commit=commands[index:index+2]
            if product['opcode'] not in ('MATMUL','MATMUL_RESIDENT') or commit['opcode']!='COMMIT':
                break
            operands=product.get('operands') or {}
            readout=commit.get('operands') or {}
            if readout.get('src') != operands.get('dst'):
                break
            attrs={**(product.get('attributes') or {}),**(commit.get('attributes') or {})}
            if 'bias' in operands or 'bias' in readout or 'bias' in attrs:
                break
            lhs=resolve(operands.get('lhs',operands.get('a')))
            if candidates and lhs != candidates[0]['lhs']:
                break
            candidates.append({'lhs':lhs,'rhs':resolve(operands.get('rhs',operands.get('w'))),
                'dst':readout['dst'],'attrs':attrs,'source_command_indices':[index,index+1]})
            index+=2
        if len(candidates)<2:
            return None
        identity=hashlib.sha256(json.dumps(cb,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        plan=plan_shared_products(tensors,candidates,source_identity=identity)
        return (plan,index) if plan is not None else None

    def resident_products_at(start):
        """Keep a packed, read-only RHS live across adjacent original products."""
        import hashlib, json
        commands=cb['commands']
        candidates=[]
        index=start
        while index+1<len(commands) and len(candidates)<128:
            product,commit=commands[index:index+2]
            if product['opcode']!='MATMUL_RESIDENT' or commit['opcode']!='COMMIT':
                break
            operands=product.get('operands') or {}
            readout=commit.get('operands') or {}
            rhs_handle=operands.get('rhs',operands.get('w'))
            if rhs_handle not in handles or readout.get('src')!=operands.get('dst'):
                break
            attrs={**(product.get('attributes') or {}),**(commit.get('attributes') or {})}
            if 'bias' in operands or 'bias' in readout or 'bias' in attrs:
                break
            attrs={k:v for k,v in attrs.items() if k!='name' and not k.startswith('prov.')}
            left=resolve(operands.get('lhs',operands.get('a')))
            right=resolve(rhs_handle)
            dst=resolve(readout['dst'])
            ashape,wshape=tensors[left]['shape'],tensors[right]['shape']
            if len(ashape)!=2 or len(wshape)!=2 or ashape[1]!=wshape[0]:
                break
            geometry=(ashape[0],ashape[1],wshape[1])
            attrs.setdefault('output_dtype',tensors[dst]['dtype'])
            if candidates and (right,geometry,attrs)!=(candidates[0]['rhs'],candidates[0]['geometry'],candidates[0]['attrs']):
                break
            candidates.append(dict(lhs=left,rhs=right,dst=dst,geometry=geometry,attrs=attrs))
            index+=2
        if len(candidates)<2:
            return None
        # The logical ABI supplies independent external buffers. Reject explicit
        # source aliases and any member that can overwrite a later operand.
        names=[candidates[0]['rhs'],*(c['lhs'] for c in candidates),*(c['dst'] for c in candidates)]
        if len(set(names))!=len(names):
            return None
        from gemmini_dense import emit_resident_group
        identity=hashlib.sha256(json.dumps(cb,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        first=candidates[0]
        receipt=emit_resident_group(fn,tuple(pointer(c['lhs']) for c in candidates),
            pointer(first['rhs']),tuple(pointer(c['dst']) for c in candidates),
            *first['geometry'],attrs=first['attrs'],source_identity=identity,
            external_buffers_disjoint=True)
        if receipt is None:
            return None
        return receipt,index

    skipped_commands=set()
    for i, command in enumerate(cb['commands']):
        if i in skipped_commands:
            continue
        if command['opcode'] in ('MATMUL','MATMUL_RESIDENT'):
            resident=resident_products_at(i)
            if resident is not None:
                receipt,end=resident
                receipts.append(receipt)
                skipped_commands.update(range(i,end))
                continue
            shared=shared_products_at(i)
            if shared is not None:
                from gemmini_graph_reuse import emit_shared_products
                from gemmini_shared_counted import CountedSharedProducts, emit_counted_shared_products
                plan,end=shared
                bound={name:pointer(name) for product in plan.products for name in (product['lhs'],product['rhs'],product['dst'])}
                emitter = emit_counted_shared_products if isinstance(plan,CountedSharedProducts) else emit_shared_products
                receipts.append(emitter(fn,plan,bound))
                skipped_commands.update(range(i,end))
                continue
        opcode=command['opcode']
        operands=dict(command.get('operands') or {})
        attrs=dict(command.get('attributes') or {})
        if opcode=='RES_PACK':
            handles[operands['dst']]=resolve(operands['src'])
        elif opcode=='EVICT':
            pass
        elif opcode in ('MATMUL','MATMUL_RESIDENT'):
            products[operands['dst']]=(opcode,operands,attrs)
        elif opcode=='COMMIT':
            if operands['src'] not in products:
                raise ValueError('commit requires an implemented original product')
            pop,poperands,pattrs=products[operands['src']]
            combined={**poperands,'dst':operands['dst']}
            if 'bias' in operands:
                combined['bias']=operands['bias']
            receipts.append(emit_product(pop,combined,{**pattrs,**attrs}))
        elif opcode in ('CONV2D','ATTENTION_QK','ATTENTION_PV','BATCHED_MATMUL'):
            receipts.append(emit_product(opcode,operands,attrs))
        elif opcode=='RESIDUAL_ADD':
            from gemmini_pointwise import emit_residual
            left,right=resolve(operands['lhs']),resolve(operands['rhs'])
            shape=tensors[left]['shape']
            receipts.append(emit_residual(fn,pointer(left),pointer(right),pointer(operands['dst']),
                prod(shape[:-1]),shape[-1],attrs=attrs,output_dtype=attrs.get('output_dtype','i8'),
                external_buffers_disjoint=len({left,right,resolve(operands['dst'])})==3))
        elif opcode=='MOVEMENT':
            from gemmini_pointwise import emit_movement
            left=resolve(operands['src'])
            receipts.append(emit_movement(fn,pointer(left),pointer(operands['dst']),tensors[left]['shape'],
                dtype=tensors[left]['dtype'],attrs=attrs))
        elif opcode=='BIAS_ADD':
            from gemmini_pointwise import emit_bias_add
            left=resolve(operands.get('src',operands.get('lhs')))
            bias=resolve(operands.get('bias',operands.get('rhs')))
            shape=tensors[left]['shape']
            receipts.append(emit_bias_add(fn,pointer(left),pointer(bias),pointer(operands['dst']),
                prod(shape[:-1]),shape[-1],attrs=attrs,output_dtype=attrs.get('output_dtype','i32')))
        elif opcode=='MAXPOOL2D':
            from gemmini_pool import emit_pool
            left=resolve(operands.get('src',operands.get('input')))
            receipts.append(emit_pool(fn,pointer(left),pointer(operands['dst']),
                shape=tensors[left]['shape'],attrs=attrs,dtype=tensors[left]['dtype']))
        elif opcode=='ROW_MAJOR_VIEW':
            if operands['src']!=operands['dst']:
                raise ValueError('row major view requires the same output buffer')
        else:
            raise ValueError('compiler lowering not implemented for '+opcode)
    import gemmini_encode as enc
    enc.fence(fn)
    for name in private_tensors:
        tensors.pop(name)
    lowered = finish_module(fn)
    if evidence_sink is not None:
        evidence_sink(receipts, cb, lowered)
    return cb,lowered
