"""Reusable command graph to primitive accelerator C lowering."""
from __future__ import annotations
from math import prod
import itertools
import json
from pathlib import Path

def generate_c(cb):
    settings_path = Path(__file__).resolve().parent / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if settings.get('matmul_schedule') == 'resident112':
        from device_resident_wide import runtime_header
    elif settings.get('matmul_schedule') == 'panel112':
        from device_panel112 import runtime_header
    elif settings.get('matmul_schedule','tiles') == 'resident':
        from device_resident import runtime_header
    elif settings.get('matmul_schedule','tiles') in ('panel','static'):
        from device_optimized import runtime_header
    else:
        from device import runtime_header
    from conv import emit_conv, emit_attention, emit_pool, _product
    from merlin.compile.logical_abi import logical_abi, kernel_arg_order
    contract = Path(__file__).resolve().parent / 'contracts' / 'mlir_oot_backend_contract.yaml'
    order = kernel_arg_order(cb, logical_abi(contract))
    names = {name: f'arg_{i}' for i, name in enumerate(order)}
    tensors = cb['tensors']
    globals_ = []
    seq = itertools.count()
    def unique(): return next(seq)
    for name, spec in tensors.items():
        if name not in names:
            tag = f'tmp_{unique()}'
            width = {'i8':1,'i16':2,'i32':4,'i64':8,'f16':2,'bf16':2,'f32':4,'f64':8}.get(spec['dtype'])
            if width is None: raise ValueError('unsupported tensor storage dtype '+spec['dtype'])
            globals_.append(f'static unsigned char {tag}[{prod(spec["shape"])*width}] __attribute__((aligned(64)));')
            names[name] = tag
    handles = {}
    contractions = {}
    fence_pending = bool(cb.get('params',{}).get('global_program_plan'))
    body = [] if fence_pending else ['gm_fence();', 'gm_flush();']
    active_task = None
    def spec(name):
        while name in handles: name = handles[name]
        return tensors[name]
    def leaf(name):
        while name in handles: name = handles[name]
        return name
    for ci, cmd in enumerate(cb['commands']):
        op = cmd['opcode']; operands = cmd.get('operands', {}); attrs = cmd.get('attributes', {})
        extra, lines = [], []
        if op == 'RES_PACK':
            handles[operands['dst']] = leaf(operands['src'])
            names[operands['dst']] = names[leaf(operands['src'])]
        elif op in ('MATMUL_RESIDENT','MATMUL'):
            contractions[operands['dst']] = (leaf(operands['lhs']), leaf(operands['rhs']))
        elif op == 'COMMIT':
            a,b = contractions[operands['src']]
            m,k = tensors[a]['shape']; kb,n = tensors[b]['shape']
            if kb != k: raise ValueError('matmul reduction mismatch')
            if tensors[a]['dtype'] != 'i8' or tensors[b]['dtype'] != 'i8':
                raise ValueError('device contraction requires signed i8 operands')
            from copy import deepcopy
            attr = deepcopy(attrs)
            if operands.get('bias'): attr['bias'] = operands['bias']
            y = names[operands['dst']]
            seed = attrs.get('seed_tensor')
            target = y
            pool = 'maxpool' in attrs.get('epilogue', [])
            if pool:
                if attrs.get('output_dtype','i32') != 'i8':
                    raise ValueError('device maxpool requires i8 contraction output')
                target = f'prepool_mm_{unique()}'
                globals_.append(f'static signed char {target}[{m*n}] __attribute__((aligned(64)));')
            if seed:
                if attrs.get('output_dtype','i32') != 'i32' or attrs.get('epilogue',[]):
                    raise ValueError('seeded contraction needs i32 output without epilogue')
                target = f'seeded_mm_{unique()}'
                globals_.append(f'static int {target}[{m*n}] __attribute__((aligned(64)));')
            extra,lines = _product(names[a],names[b],target,m,k,n,attr,cb,names,f'mm_{unique()}')
            if seed:
                lines.append(f'gm_add_i32_flat((const int*){target},(const int*){names[seed]},(int*){y},{m*n});')
            if pool:
                pool_globals, pool_body = emit_pool(target,y,m,n,attr,unique,dst_shape=tensors[operands['dst']]['shape'])
                extra.extend(pool_globals)
                lines.extend(pool_body)
        elif op == 'CONV2D':
            extra,lines = emit_conv(cmd,cb,names,unique)
        elif op in ('ATTENTION_QK','ATTENTION_PV'):
            seed = attrs.get('seed_tensor')
            dst = operands['dst']
            attention_names = names
            if seed:
                if spec(seed)['dtype'] != 'i32' or spec(seed)['shape'] != spec(dst)['shape']:
                    raise ValueError('attention seed must match the i32 result shape')
                target = f'seeded_attention_{unique()}'
                globals_.append(f'static int {target}[{prod(spec(dst)["shape"])}] __attribute__((aligned(64)));')
                attention_names = dict(names, **{dst:target})
            extra,lines = emit_attention(cmd,cb,attention_names,unique)
            if seed:
                lines.append(f'gm_add_i32_flat((const int*){target},(const int*){names[seed]},(int*){names[dst]},{prod(spec(dst)["shape"])});')
        elif op == 'BATCHED_MATMUL':
            a,b,y = operands['a'],operands['w'],operands['dst']
            shape = spec(a)['shape']; m,k = shape[-2:]; n = spec(b)['shape'][-1]
            batches = prod(shape[:-2])
            if spec(a)['dtype'] != 'i8' or spec(b)['dtype'] != 'i8': raise ValueError('batched device operands must be i8')
            target = names[y]
            seed = attrs.get('seed_tensor')
            if seed:
                if spec(seed)['dtype'] != 'i32' or spec(seed)['shape'] != spec(y)['shape']:
                    raise ValueError('batched matmul seed must match the i32 result shape')
                target = f'seeded_batched_{unique()}'
                globals_.append(f'static int {target}[{batches*m*n}] __attribute__((aligned(64)));')
            batch_tag = f'batch_{unique()}'
            extra, product = _product(
                f'(const signed char*){names[a]}+{batch_tag}*{m*k}',
                f'(const signed char*){names[b]}+{batch_tag}*{k*n}',
                f'(int*){target}+{batch_tag}*{m*n}',
                m,k,n,{'output_dtype':'i32'},cb,names,f'batched_mm_{unique()}')
            lines = [f'for (int {batch_tag}=0;{batch_tag}<{batches};++{batch_tag}) {{'] + product + ['}']
            if seed:
                lines.append(f'gm_add_i32_flat((const int*){target},(const int*){names[seed]},(int*){names[y]},{batches*m*n});')
        elif op == 'MOVEMENT':
            x,y = operands['src'],operands['dst']; shape=spec(x)['shape']; n=shape[-1]; m=prod(shape[:-1])
            iw=int(spec(x)['dtype'][1:])//8; ow=int(spec(y)['dtype'][1:])//8
            from elementwise import validate_movement
            validate_movement(iw,ow)
            lines = [f'gm_movement({names[x]},{names[y]},{m},{n},{iw},{ow},1.0f,0);']
        elif op == 'RESIDUAL_ADD':
            a,b,y = operands['lhs'],operands['rhs'],operands['dst']; shape=spec(a)['shape']; n=shape[-1];m=prod(shape[:-1])
            ow=int(spec(y)['dtype'][1:])//8
            from elementwise import validate_residual
            validate_residual(attrs['lhs_scale'],attrs['rhs_scale'],ow,'relu' in attrs.get('epilogue',[]))
            lines = [f'gm_residual((const signed char*){names[a]},(const signed char*){names[b]},{names[y]},{m},{n},{ow},{float(attrs["lhs_scale"])}f,{float(attrs["rhs_scale"])}f,{int("relu" in attrs.get("epilogue",[]))});']
        elif op == 'BIAS_ADD':
            a,b,y = operands['src'],operands['bias'],operands['dst'];shape=spec(a)['shape'];n=shape[-1];m=prod(shape[:-1])
            lines = [f'gm_bias_add((const int*){names[a]},(const int*){names[b]},(int*){names[y]},{m},{n});']
        elif op in ('VECTOR_MAP','VREDUCE'):
            from vector import emit_vector
            extra,lines = emit_vector(cmd,cb,names,unique)
        elif op in ('EVICT','ROW_MAJOR_VIEW'):
            pass
        else:
            raise ValueError('no lowering for opcode '+op)
        globals_.extend(extra)
        if lines and 'merlin.global_task' in attrs:
            task = int(attrs['merlin.global_task'])
            if active_task != task:
                if active_task is not None:
                    body.append(f'__asm__ volatile("# merlin_task_end {active_task}" ::: "memory");')
                body.append(f'__asm__ volatile("# merlin_task_start {task}" ::: "memory");')
                if fence_pending:
                    body.extend(['gm_fence();', 'gm_flush();'])
                    fence_pending = False
                active_task = task
        body.extend(lines)
    if active_task is not None:
        body.append(f'__asm__ volatile("# merlin_task_end {active_task}" ::: "memory");')
    headers = [runtime_header()]
    if any(c['opcode'] in ('MOVEMENT','RESIDUAL_ADD','BIAS_ADD','VECTOR_MAP','VREDUCE') or c.get('attributes',{}).get('seed_tensor') for c in cb['commands']):
        from elementwise import runtime_header as ew_header
        headers.append(ew_header(settings.get('ew_tile_cols',16)))
    args = ','.join('void *'+names[n] for n in order)
    return '\n'.join(headers + globals_ + [f'void kernel_entry({args}) {{'] + body + ['}', 'void kernel_complete(void) { __asm__ volatile("fence" ::: "memory"); }'])
