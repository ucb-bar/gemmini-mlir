"""Compose semantic operations into one logical-ABI accelerator kernel."""
from __future__ import annotations

from pathlib import Path
from math import prod
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'compiler_library'))
from merlin.compile.logical_abi import logical_abi, kernel_arg_order
from target import Builder


def _resolve(handles, name):
    while name in handles:
        name = handles[name]
    return name



def _resident_lifetime_uses(commands, tensors):
    """Count materialized explicit consumers within each lexical handle lifetime."""
    lifetimes, products, counts = {}, {}, {}
    for index, command in enumerate(commands):
        code = command['opcode']
        args = command.get('operands', {})
        if code == 'RES_PACK':
            lifetimes[args['dst']] = index
            counts[index] = 0
        elif code == 'EVICT':
            lifetimes.pop(args['handle'], None)
        elif code in ('MATMUL_RESIDENT', 'MATMUL'):
            products[args['dst']] = (code, args)
            if code == 'MATMUL_RESIDENT' and args['dst'] in tensors:
                owner = lifetimes.get(args['rhs'])
                if owner is not None: counts[owner] += 1
        elif code == 'COMMIT' and args['src'] in products:
            product_code, pargs = products[args['src']]
            if product_code == 'MATMUL_RESIDENT':
                owner = lifetimes.get(pargs['rhs'])
                if owner is not None: counts[owner] += 1
    return counts


def lower_commands(b, cb, pointers):
    from dense import lower_matmul
    b.fence()
    # Standalone interface kernels establish the same accelerator entry state
    # as original-SSA graph tasks. Flush translation state before the first
    # device access; configuration and payload transfers follow below.
    b.rocc(7, 0, 0)
    tensors = cb['tensors']
    handles = {}
    products = {}
    uses = _resident_lifetime_uses(cb['commands'], tensors)
    resident_states={}
    active=None
    def invalidate():
        nonlocal active
        for state in resident_states.values():state['live']=False
        active=None
    def select_resident(pargs,epilogue):
        nonlocal active
        handle=pargs.get('rhs');state=resident_states.get(handle)
        if state is None or epilogue.get('bias') or epilogue.get('bias_ptr') or 'maxpool' in epilogue.get('epilogue',[]):
            invalidate();return None
        from dense import output_intervals
        source=tensors[state['source']]
        output=tensors[epilogue['_output']]
        e=dict(epilogue,out_dtype=output['dtype'])
        if len(output_intervals(e,source['shape'][1]))!=1:
            invalidate();return None
        if active!=handle:
            invalidate();active=handle
        return state
    for command_index, cmd in enumerate(cb['commands']):
        code = cmd['opcode']
        args, attrs = cmd.get('operands', {}), cmd.get('attributes', {})
        if code == 'RES_PACK':
            handles[args['dst']] = _resolve(handles, args['src'])
            handle=args['dst'];source=handles[handle]
            if active==handle:invalidate()
            resident_states.pop(handle,None)
            if uses.get(command_index,0)>1 and tensors[source]['dtype']=='i8' and len(tensors[source]['shape'])==2:
                from dense import _geometry
                from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
                edge,rows,accrows=_geometry();k,n=tensors[source]['shape']
                split=(rows//(2*edge))*edge
                fit=plan_dense_rhs_residency(1,k,n,block=edge,operand_rows=rows,
                    accumulator_rows=accrows,rhs_base=split,max_static_rhs_tiles=(rows-split)//edge)
                if fit is not None:resident_states[handle]={'source':source,'live':False}

        elif code == 'EVICT':
            handle=args['handle']
            if active==handle:invalidate()
            resident_states.pop(handle,None)
            handles.pop(handle,None)
            continue
        elif code in ('MATMUL_RESIDENT', 'MATMUL'):
            if code == 'MATMUL' and attrs.get('transpose_rhs'):
                invalidate()
                from products import lower_product
                if lower_product(b, cb, cmd, pointers):
                    continue
            products[args['dst']] = (args, attrs)
            # Interface matmul accumulators are materialized by COMMIT. Linalg
            # products own a typed result and are emitted immediately.
            if args['dst'] in tensors:
                e=dict(attrs,_output=args['dst'])
                state=select_resident(args,e)
                _matmul(b, cb, pointers, handles, args, attrs, args['dst'], attrs,resident_state=state)
                if any(s['source']==args['dst'] for s in resident_states.values()):invalidate()
        elif code == 'COMMIT':
            if args['src'] not in products:
                raise ValueError('COMMIT source is not a lowered product')
            pargs, pattrs = products[args['src']]
            epilogue = dict(attrs)
            bias_name = args.get('bias', attrs.get('bias'))
            if bias_name:
                if bias_name not in pointers:
                    raise ValueError('COMMIT bias is not a declared tensor')
                epilogue['bias_ptr'] = pointers[bias_name]
            e=dict(epilogue,_output=args['dst'])
            state=select_resident(pargs,e)
            _matmul(b, cb, pointers, handles, pargs, pattrs, args['dst'], epilogue,resident_state=state)
            # Any logical write to a packed source invalidates its residency.
            if any(s['source']==args['dst'] for s in resident_states.values()):invalidate()
        elif code == 'CONV2D':
            invalidate()
            from conv import lower_conv
            realized = dict(cmd, operands={k: _resolve(handles, v) if isinstance(v, str) else v for k, v in args.items()})
            lower_conv(b, cb, realized, pointers)
        elif code in ('MOVEMENT', 'RESIDUAL_ADD', 'BIAS_ADD'):
            invalidate()
            from elementwise import lower_elementwise
            lower_elementwise(b, cb, cmd, pointers)
        elif code in ('ATTENTION_QK', 'ATTENTION_PV', 'BATCHED_MATMUL', 'VREDUCE'):
            invalidate()
            from products import lower_product
            lower_product(b, cb, cmd, pointers)
        elif code == 'ROW_MAJOR_VIEW':
            invalidate()
            if args['src'] != args['dst']:
                src, dst = tensors[args['src']], tensors[args['dst']]
                if src['dtype'] != dst['dtype'] or prod(src['shape']) != prod(dst['shape']):
                    raise ValueError('view changes physical representation')
                from elementwise import lower_elementwise
                lower_elementwise(b, cb, {'opcode': 'MOVEMENT', 'operands': args, 'attributes': {'output_dtype': dst['dtype']}}, pointers)
        else:
            raise NotImplementedError(f'compiler lowering is missing for {code}')
    b.fence()


def _matmul(b, cb, pointers, handles, args, source_attrs, dst, attrs, *, resident_state=None):
    from dense import lower_matmul
    lhs = _resolve(handles, args['lhs'])
    rhs = _resolve(handles, args['rhs'])
    ls, rs, os = (cb['tensors'][x] for x in (lhs, rhs, dst))
    if ls['dtype'] != 'i8' or rs['dtype'] != 'i8':
        raise ValueError('dense target requires signed i8 operands')
    if len(ls['shape']) != 2 or len(rs['shape']) != 2:
        raise ValueError('ranked contraction must be normalized before dense lowering')
    m, k = ls['shape']; rk, n = rs['shape']
    if k != rk:
        raise ValueError('dense reduction dimension mismatch')
    if source_attrs.get('transpose_rhs'):
        raise NotImplementedError('transposed product requires layout normalization')
    epilogue = dict(attrs)
    epilogue['output_dtype'] = os['dtype']
    epilogue['out_dtype'] = os['dtype']
    epilogue['relu'] = 'relu' in attrs.get('epilogue', [])
    if 'maxpool' in attrs.get('epilogue', []):
        if os['dtype'] != 'i8':
            raise ValueError('pooled readout requires i8 storage')
        from conv import lower_maxpool
        temporary = b.alloca(m * n, 'i8')
        epilogue['epilogue'] = [s for s in attrs['epilogue'] if s != 'maxpool']
        lower_matmul(b, pointers[lhs], pointers[rhs], temporary, m, k, n, epilogue=epilogue)
        h, w = attrs['pool_in_dims']
        lower_maxpool(b, temporary, pointers[dst], n, attrs, batches=m // (h * w))
    else:
        dense_options = {}
        if resident_state is not None:
            dense_options['resident_state'] = resident_state
        elif (args['rhs'] in handles and os['dtype'] == 'i32'
              and not attrs.get('epilogue') and not attrs.get('bias')
              and not attrs.get('bias_ptr')):
            # An explicit packed handle binds the unchanged logical row-major
            # source. Scheduling does not reinterpret it as an expert layout.
            dense_options['projection'] = True
        lower_matmul(b, pointers[lhs], pointers[rhs], pointers[dst], m, k, n, epilogue=epilogue,
            options=dense_options)


def compile_buffer(cb):
    abi = logical_abi(Path(__file__).resolve().parent / 'logical_contract.yaml')
    b = Builder(kernel_arg_order(cb, abi))
    pointers = dict(b.args)
    for name, spec in cb['tensors'].items():
        if name not in pointers:
            pointers[name] = b.alloca(prod(spec['shape']), spec['dtype'])
    lower_commands(b, cb, pointers)
    return b.module_text()
