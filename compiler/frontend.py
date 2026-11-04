"""Source readers and semantic lowering into the public command-buffer ABI."""
from __future__ import annotations

import sys
from pathlib import Path
from math import prod

_LIBRARY = Path(__file__).resolve().parent / 'compiler_library'
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))
from merlin.targetgen.oot_starterkit.iface import parse_interface, parse_linalg, is_linalg_on_tensors
from merlin.targetgen.contract import interface_emit as _ie


def _shared(text):
    cb = parse_interface(text)
    module = _ie._parse_attr_block(_ie._module_attrs(text))
    if module.get('merlin_iface.version') != '0.1':
        raise ValueError('unsupported merlin_iface.version')
    if cb['abi_version'] != '0.1':
        raise ValueError('unsupported merlin_iface.abi_version')
    aliases = {}
    command_index = 0
    output_order = []
    for line in text.splitlines():
        rec = _ie._op_line(line.strip())
        if not rec or rec['mnemonic'] not in _ie.defined_mnemonics():
            continue
        attrs = _ie._parse_attr_block(rec['attrs_body'])
        kind, result = rec['mnemonic'], rec['result']
        if kind == 'tensor':
            aliases[result] = attrs.get('name', result)
            continue
        cmd = cb['commands'][command_index]
        command_index += 1
        # The canonical reader exposes SSA operand identifiers. Resolve them to
        # source-declared logical names for tensors and previously named results.
        cmd['operands'] = {k: aliases.get(v, v) if k != 'dst' else v
                           for k, v in cmd.get('operands', {}).items()}
        if kind == 'commit' or kind in _ie._NAMED_OP_OPERAND_KEYS:
            name = attrs.get('name', result)
            aliases[result] = name
            shape, dtype = _ie._shape_dtype(_ie._last_type(rec['tail']))
            cb['tensors'][name] = {'shape': shape, 'dtype': dtype, 'role': 'output'}
            if name not in output_order:
                output_order.append(name)
    # The pointer ABI publishes tensor results in source/command order.  The
    # canonical reader declares whole-op results eagerly but COMMIT results
    # are synthesized above, so dictionary insertion order alone can reverse
    # them.  Keep every leaf in its declaration order, then publish outputs
    # in the same order used by the logical ABI and source result bindings.
    tensors = cb['tensors']
    cb['tensors'] = {name: spec for name, spec in tensors.items()
                     if spec.get('role') != 'output'}
    cb['tensors'].update((name, tensors[name]) for name in output_order)
    cb['tensors'].update((name, spec) for name, spec in tensors.items()
                         if spec.get('role') == 'output' and name not in output_order)
    cb.setdefault('params', {})
    return cb


def _linalg_module(text):
    from xdsl.parser import Parser
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    mod = Parser(make_linalg_context(), text).parse_module()
    fns = [op for op in mod.walk() if op.name == 'func.func' and op.body.blocks]
    forwards = [op for op in fns if op.sym_name.data == 'forward']
    fn = forwards[0] if forwards else fns[0]
    if len(fn.body.blocks) != 1:
        raise ValueError('linalg entry must have a single block')
    return fn


def _scalar_expr(body):
    """Recognize arithmetic by SSA edges, rather than operation-name bags."""
    env = {f'arg:{i}': ('arg', i) for i in range(len(body['arguments']))}
    for i, cap in enumerate(body['captures']):
        if 'const_value' not in cap:
            raise ValueError('nonconstant scalar capture')
        env[f'capture:{i}'] = ('const', cap['const_value'])
    for i, op in enumerate(body['operations']):
        name = op['op']
        args = [env[v] for v in op['operands']]
        if op['regions'] or len(op['results']) != 1:
            raise ValueError('unsupported scalar region or multiple results')
        if name == 'arith.constant':
            # Constants parsed from MLIR's canonical typed attribute spelling.
            value = op['attributes'].get('value', '').split(':')[0].strip()
            try:
                expr = ('const', float(value) if any(c in value.lower() for c in '.e') else int(value))
            except ValueError as exc:
                raise ValueError('unsupported scalar constant') from exc
        elif name in ('arith.extsi', 'arith.extf'):
            expr = ('cast', op['results'][0], args[0])
        elif name in ('arith.addi', 'arith.addf', 'arith.muli', 'arith.mulf'):
            expr = ('add' if 'add' in name else 'mul', *args)
        else:
            raise ValueError(f'unsupported scalar arithmetic {name}')
        env[f'op:{i}:0'] = expr
    return [env[v] for v in body['yields']]


def _strip_cast(expr):
    return _strip_cast(expr[2]) if expr[0] == 'cast' else expr


def _map_results(s):
    """Canonical dimension-only affine map result tuples."""
    from xdsl.parser import Parser
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from xdsl.ir.affine import AffineDimExpr
    attr = Parser(make_linalg_context(), s).parse_attribute()
    amap = attr.data
    if amap.num_symbols or not all(isinstance(x, AffineDimExpr) for x in amap.results):
        raise ValueError('non-permutation/broadcast affine map')
    return tuple(x.position for x in amap.results)


def _generic_contraction_layout(op, maps):
    """Prove a dense integer contraction independent of affine dimension names.

    Physical output dimensions determine batch, row and column axes.  The sole
    remaining loop must be a reduction.  Commutativity permits exchanging the
    two signed integer multiplicands, without changing reduction or seed order.
    """
    if maps is None or len(maps) != 3:
        raise ValueError('unsupported contraction indexing maps')
    output = maps[2]
    rank = len(output)
    if rank not in (2, 3) or len(set(output)) != rank:
        raise ValueError('contraction requires a dense matrix output')
    loops = op['iterator_types']
    if len(loops) != rank + 1 or set(output) - set(range(len(loops))):
        raise ValueError('contraction loop domain does not match output')
    reductions = [i for i, kind in enumerate(loops) if kind == 'reduction']
    if len(reductions) != 1:
        raise ValueError('contraction requires exactly one reduction iterator')
    k = reductions[0]
    if k in output or any(loops[i] != 'parallel' for i in output):
        raise ValueError('contraction output axes must be parallel')
    m, n = output[-2:]
    batch = output[:-2]
    # Every indexing-map dimension must have one consistent, positive extent.
    extents = {}
    for dims, tensor in zip(maps, [*op['ins'], op['results'][0]]):
        shape = tensor['shape']
        if len(dims) != len(shape) or len(set(dims)) != len(dims):
            raise ValueError('contraction operand is not a dense dimension map')
        for dim, extent in zip(dims, shape):
            if dim not in range(len(loops)) or extent <= 0 or (dim in extents and extents[dim] != extent):
                raise ValueError('contraction operand extents disagree')
            extents[dim] = extent
    if set(extents) != set(range(len(loops))):
        raise ValueError('contraction has an unbound loop dimension')
    mk = (*batch, m, k)
    kn = (*batch, k, n)
    nk = (*batch, n, k)
    for left, right in ((0, 1), (1, 0)):
        if maps[left] == mk and maps[right] == kn:
            return ('BATCHED_MATMUL' if batch else 'MATMUL'), left, right
        if not batch and maps[left] == mk and maps[right] == nk:
            return 'ATTENTION_QK', left, right
    raise ValueError('unsupported contraction indexing maps')


def _int_array(value):
    from xdsl.parser import Parser
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    attr = Parser(make_linalg_context(), value).parse_attribute()
    return [int(x) for x in attr.get_values()]


def _identity_body(dtype):
    return {'arguments':[dtype,dtype], 'captures':[], 'operations':[],
            'yields':['arg:0'],'effects':[]}


def _quant_attrs(op, inputs):
    """Expand per-tensor affine quantization into its typed scalar arithmetic."""
    if len(inputs) != 3 or op['ins'][1]['shape'] or op['ins'][2]['shape']:
        raise ValueError('per-tensor quantization requires scalar scale and zero point')
    dtype = op['results'][0]['dtype']
    rank = len(op['results'][0]['shape'])
    dimensions = ', '.join(f'd{i}' for i in range(rank))
    body = {'arguments':[x['dtype'] for x in op['ins']]+[dtype],
            'captures':[], 'operations':[], 'yields':[], 'effects':[]}
    def emit(name, args, result, **attrs):
        index = len(body['operations'])
        body['operations'].append({'op':name,'operands':args,'results':[result],
                                   'attributes':attrs,'effects':[],'regions':0})
        return f'op:{index}:0'
    if op['kind'] == 'quant_ext.quantize_per_tensor':
        q = emit('arith.divf',['arg:0','arg:1'],'f32')
        q = emit('math.roundeven',[q],'f32')
        q = emit('arith.fptosi',[q],'i64')
        q = emit('arith.addi',[q,'arg:2'],'i64')
        for attr, mnemonic in [('quant_min','arith.maxsi'),('quant_max','arith.minsi')]:
            value = int(op['attributes'][attr].split(':')[0].strip())
            constant = emit('arith.constant',[],'i64',value=f'{value} : i64')
            q = emit(mnemonic,[q,constant],'i64')
        if dtype != 'i64':
            q = emit('arith.trunci',[q],dtype)
    else:
        q = emit('arith.extsi',['arg:0'],'i64') if op['ins'][0]['dtype'] != 'i64' else 'arg:0'
        q = emit('arith.subi',[q,'arg:2'],'i64')
        q = emit('arith.sitofp',[q],'f32')
        q = emit('arith.mulf',[q,'arg:1'],'f32')
        if dtype != 'f32':
            raise ValueError('dequantization requires f32 output')
    body['yields'] = [q]
    return {'inputs':inputs,'scalar_body':body,
            'indexing_maps':[f'affine_map<({dimensions}) -> ({dimensions})>',
                             f'affine_map<({dimensions}) -> ()>',
                             f'affine_map<({dimensions}) -> ()>',
                             f'affine_map<({dimensions}) -> ({dimensions})>'],
            'iterator_types':['parallel']*rank,'reduction_dims':[],
            'initializers':[{'value':None,'kind':'tensor.empty'}],
            'prov':op['prov'],'output_dtype':dtype}


def _linalg(text):
    inv = parse_linalg(text)
    fn = _linalg_module(text)
    from xdsl.dialects.builtin import StringAttr
    attrs = fn.attributes
    def names(key):
        val = attrs.get(key)
        return [x.data for x in val] if val is not None else None
    inputs, outputs = names('merlin.logical_inputs'), names('merlin.logical_outputs')
    roles = names('merlin.logical_input_roles')
    if inputs is None:
        inputs = [a.name_hint or f'arg_{i}' for i,a in enumerate(fn.body.block.args)]
    if outputs is None:
        outputs = [v.name_hint or f'result_{i}' for i,v in enumerate(fn.body.block.ops.last.operands)]
    if len(inputs) != len(inv['args']) or len(outputs) != len(inv['results']):
        raise ValueError('logical ABI metadata arity differs from function signature')
    if len(set(inputs + outputs)) != len(inputs + outputs):
        raise ValueError('logical ABI tensor names must be distinct')
    cb = {'abi_version':'0.1', 'target':'gemmini', 'tensors':{}, 'commands':[], 'params':{}}
    tensors, commands = cb['tensors'], cb['commands']
    for i, a in enumerate(inv['args']):
        tensors[inputs[i]] = {'shape':a['shape'], 'dtype':a['dtype'], 'role':roles[i] if roles else 'input'}
    if inv['returns'] is None:
        raise ValueError('entry has no complete ordered return join')
    result_names = {}
    for name, ret in zip(outputs, inv['returns']):
        key = (*ret['source'], ret.get('result_index', 0))
        if key in result_names:
            raise ValueError('aliased or duplicate linalg return requires an explicit copy')
        result_names[key] = name
    values = {('arg',i,0): n for i,n in enumerate(inputs)}
    def resolve(rec):
        key = (*rec['source'], rec.get('result_index', 0))
        if key not in values:
            raise ValueError(f'unmaterialized linalg operand {key}')
        return values[key]
    def emit(opcode, operands, attributes=None):
        cmd = {'opcode':opcode, 'operands':operands}
        if attributes:
            cmd['attributes'] = attributes
            if attributes.get('seed_tensor'):
                cmd['operands']['seed'] = attributes['seed_tensor']
        commands.append(cmd)
    for op in inv['ops']:
        oid = op['id']
        if len(op['results']) != 1:
            raise ValueError(f"unsupported result arity for {op['kind']}")
        spec = op['results'][0]
        key = ('op', oid, 0)
        name = result_names.get(key, f'__value_{oid}')
        if name in tensors:
            raise ValueError('source result name collides with another logical tensor')
        tensors[name] = {**spec, 'role':'output' if key in result_names else 'intermediate'}
        values[key] = name
        input_records = []
        for i,x in enumerate(op['ins']):
            if x['source'][0] == 'const' and 'const_value' in x:
                input_records.append({'value':x['const_value'],'dtype':x['dtype'],'shape':x['shape']})
            elif x['source'][0] == 'init':
                body = op['scalar_body']
                ref = f'arg:{i}'
                used = body is None or ref in body['yields'] or any(ref in inner['operands'] for inner in body['operations'])
                if used:
                    raise ValueError('undefined initialization-only tensor is read as input')
                input_records.append({'value':0,'dtype':x['dtype'],'shape':x['shape']})
            else:
                input_records.append({'tensor':resolve(x)})
        ins = [x.get('tensor') for x in input_records]
        dtype = spec['dtype']
        kind = op['kind']
        try:
            maps = [_map_results(x) for x in op['indexing_maps']] if op['indexing_maps'] else None
        except ValueError:
            maps = None
        try:
            exprs = _scalar_expr(op['scalar_body']) if op['scalar_body'] else None
        except ValueError:
            exprs = None
        contraction = kind in ('linalg.matmul','linalg.batch_matmul','linalg.matmul_transpose_b')
        if kind == 'linalg.generic' and len(ins) == 2 and len(op['outs']) == 1 and exprs:
            e = exprs[0]
            if e[0] == 'add':
                a,b = e[1:]
                if a == ('arg',2):
                    a,b = b,a
                if b == ('arg',2) and a[0] == 'mul':
                    contraction = {_strip_cast(a[1]),_strip_cast(a[2])} == {('arg',0),('arg',1)}
        if contraction and (op['ins'][0]['dtype'] != 'i8' or op['ins'][1]['dtype'] != 'i8' or dtype != 'i32'):
            # Float contractions retain their scalar reduction order instead of
            # entering the integer accelerator contraction command.
            if kind == 'linalg.generic':
                contraction = False
            else:
                raise ValueError('named non-i8 contraction requires scalar region lowering')
        if contraction:
            # Replacing a contraction by zero-seeded MATMUL requires a proven
            # zero initializer; inspect its typed SSA producer rather than assuming fill=0.
            payload = [x for x in fn.body.block.ops if x.name not in ('func.return',)]
            # Inventory op order excludes initialization-only producers; match via result SSA
            # ownership using the same structural parser predicates.
            from merlin.targetgen.contract.linalg_iface import _initialization_only, _ins_outs, _scalar_const
            real = [x for x in payload if not _initialization_only(x)][oid]
            _, real_outs = _ins_outs(real)
            seed = None
            for init, init_rec in zip(real_outs, op['outs']):
                owner = init.owner
                if init_rec['source'][0] in ('op','arg'):
                    seed = resolve(init_rec)
                elif getattr(owner, 'name', '') not in ('linalg.fill','tensor.splat') or not owner.operands or _scalar_const(owner.operands[0]) != 0:
                    raise ValueError('matmul requires a materialized or explicitly zero-filled initializer')
            seed_attrs = {'seed_tensor':seed} if seed is not None else {}
            if op['ins'][0]['dtype'] != 'i8' or op['ins'][1]['dtype'] != 'i8' or dtype != 'i32':
                raise ValueError('matmul command requires i8 operands and i32 result')
            if kind == 'linalg.generic':
                layout, left, right = _generic_contraction_layout(op, maps)
                product_ins = [ins[left], ins[right]]
            else:
                layout = {'linalg.matmul':'MATMUL',
                          'linalg.matmul_transpose_b':'ATTENTION_QK',
                          'linalg.batch_matmul':'BATCHED_MATMUL'}[kind]
                product_ins = ins
            if layout == 'MATMUL':
                acc = f'__acc_{oid}'
                emit('MATMUL', {'lhs':product_ins[0], 'rhs':product_ins[1], 'dst':acc})
                emit('COMMIT', {'src':acc,'dst':name}, {'epilogue':[], 'output_dtype':dtype, **seed_attrs})
            elif layout == 'ATTENTION_QK':
                emit('ATTENTION_QK', {'q':product_ins[0],'k':product_ins[1],'dst':name}, {'output_dtype':dtype, 'epilogue':[], **seed_attrs})
            elif layout == 'BATCHED_MATMUL':
                emit('BATCHED_MATMUL', {'a':product_ins[0],'w':product_ins[1],'dst':name}, {'output_dtype':dtype, **seed_attrs})
            else:
                raise ValueError('unsupported contraction indexing maps')
        elif kind in ('quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor'):
            emit('VECTOR_MAP', {**{f'src{i}':n for i,n in enumerate(ins) if n is not None},'dst':name},
                 _quant_attrs(op,input_records))
        elif kind == 'tensor.concat':
            axis = int(op['attributes']['dim'].split(':')[0].strip())
            if not 0 <= axis < len(spec['shape']) or any(n is None for n in ins):
                raise ValueError('invalid concat axis or input')
            emit('VECTOR_MAP', {**{f'src{i}':n for i,n in enumerate(ins)},'dst':name},
                 {'op':'concat','axis':axis,'input_shapes':[x['shape'] for x in op['ins']],
                  'shape':spec['shape'],'output_dtype':dtype,'prov':op['prov']})
        elif kind in ('linalg.transpose','tensor.extract_slice'):
            if len(ins) != 1 or ins[0] is None or op['ins'][0]['dtype'] != dtype:
                raise ValueError('layout operation requires one typed tensor')
            rank = len(spec['shape'])
            dims = ', '.join(f'd{i}' for i in range(rank))
            if kind == 'linalg.transpose':
                perm = _int_array(op['attributes']['permutation'])
                if sorted(perm) != list(range(rank)):
                    raise ValueError('invalid transpose permutation')
                coordinates = [f'd{perm.index(i)}' for i in range(rank)]
            else:
                offsets = _int_array(op['attributes']['static_offsets'])
                sizes = _int_array(op['attributes']['static_sizes'])
                strides = _int_array(op['attributes']['static_strides'])
                if any(x < 0 for x in offsets) or any(x <= 0 for x in sizes+strides):
                    raise ValueError('dynamic or negative slice parameters unsupported')
                # Rank reduction removes only unit extents in source order.
                coordinates = []
                out_axis = 0
                for off,size,stride in zip(offsets,sizes,strides):
                    if out_axis < rank and size == spec['shape'][out_axis]:
                        coordinates.append(f'd{out_axis} * {stride} + {off}')
                        out_axis += 1
                    elif size == 1:
                        coordinates.append(str(off))
                    else:
                        raise ValueError('invalid rank-reducing slice shape')
                if out_axis != rank:
                    raise ValueError('slice result shape mismatch')
            emit('VECTOR_MAP', {'src0':ins[0],'dst':name},
                 {'inputs':input_records,'scalar_body':_identity_body(dtype),
                  'indexing_maps':[f'affine_map<({dims}) -> ({", ".join(coordinates)})>',
                                   f'affine_map<({dims}) -> ({dims})>'],
                  'iterator_types':['parallel']*rank,'reduction_dims':[],
                  'initializers':[{'value':None,'kind':'tensor.empty'}],
                  'prov':op['prov'],'output_dtype':dtype})
        elif kind == 'tensor.insert_slice':
            if len(ins) != 2 or any(n is None for n in ins):
                raise ValueError('insert_slice requires materialized source and destination')
            emit('VECTOR_MAP', {'src0':ins[0],'src1':ins[1],'dst':name},
                 {'op':'insert_slice','offsets':_int_array(op['attributes']['static_offsets']),
                  'sizes':_int_array(op['attributes']['static_sizes']),
                  'strides':_int_array(op['attributes']['static_strides']),
                  'source_shape':op['ins'][0]['shape'],'dest_shape':spec['shape'],
                  'output_dtype':dtype,'prov':op['prov']})
        elif kind in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
            if len(ins) != 1 or ins[0] is None or op['ins'][0]['dtype'] != dtype or prod(op['ins'][0]['shape']) != prod(spec['shape']):
                raise ValueError('row-major view requires equal dtype and element count')
            emit('VECTOR_MAP', {'src0':ins[0],'dst':name},
                 {'op':'row_major_copy','source_shape':op['ins'][0]['shape'],
                  'shape':spec['shape'],'output_dtype':dtype,'prov':op['prov']})
        elif kind in ('tensor.splat','linalg.fill'):
            if len(input_records) != 1 or op['ins'][0]['shape']:
                raise ValueError('splat/fill requires one scalar input')
            rank = len(spec['shape'])
            dims = ', '.join(f'd{i}' for i in range(rank))
            emit('VECTOR_MAP', {**({'src0':ins[0]} if ins[0] else {}),'dst':name},
                 {'inputs':input_records,'scalar_body':{'arguments':[dtype,dtype], 'captures':[],
                  'operations':[],'yields':['arg:0'],'effects':[]},
                  'indexing_maps':[f'affine_map<({dims}) -> ()>',f'affine_map<({dims}) -> ({dims})>'],
                  'iterator_types':['parallel']*rank,'reduction_dims':[],
                  'initializers':[{'value':None,'kind':'tensor.empty'}],
                  'prov':op['prov'],'output_dtype':dtype})
        elif kind in ('linalg.generic','linalg.map','linalg.reduce') and op['scalar_body']:
            from merlin.targetgen.contract.linalg_iface import _initialization_only, _ins_outs, _scalar_const
            real = [x for x in fn.body.block.ops if x.name != 'func.return' and not _initialization_only(x)][oid]
            _, real_outs = _ins_outs(real)
            initializers = []
            for rec, init in zip(op['outs'], real_outs):
                if rec['source'][0] in ('op','arg'):
                    initializers.append({'tensor':resolve(rec)})
                else:
                    owner = init.owner
                    cv = _scalar_const(init)
                    if owner.name == 'linalg.fill' and owner.operands:
                        cv = _scalar_const(owner.operands[0])
                    initializers.append({'value':cv, 'kind':owner.name})
            import copy
            body = copy.deepcopy(op['scalar_body'])
            for capture in body['captures']:
                if capture['source'][0] in ('arg','op'):
                    capture['tensor'] = resolve(capture)
            # Reject unmodeled regions or side effects. The known arithmetic
            # family has pure scalar semantics even where xDSL omits a trait.
            for inner in body['operations']:
                if inner['regions'] or (inner['effects'] not in ([],None)) or not inner['op'].startswith(('arith.','math.','linalg.index','tensor.extract')):
                    raise ValueError('unsupported effectful scalar body')
            emit('VREDUCE' if op['reduction_dims'] else 'VECTOR_MAP',
                 {**{f'src{i}':n for i,n in enumerate(ins) if n is not None},'dst':name},
                 {'inputs':input_records,'scalar_body':body,'indexing_maps':op['indexing_maps'],
                  'iterator_types':op['iterator_types'],'reduction_dims':op['reduction_dims'],
                  'initializers':initializers,'prov':op['prov'],'output_dtype':dtype})
        else:
            raise ValueError(f"unsupported linalg operation {kind}: {op['op']}")
    for name, ret in zip(outputs, inv['returns']):
        if resolve(ret) != name:
            raise ValueError('returned argument requires an explicit output copy')
    cb['kernel_abi'] = {'kind':'whole_program','args':[{'tensor':n,'access':'read' if s['role'] in ('input','weight','bias','scale') else 'write'} for n,s in tensors.items()], 'outputs':outputs}
    cb['params']['source_inventory'] = inv
    return cb


def load_program(text: str) -> dict:
    """Load source text, preserving complete declared tensors and arithmetic."""
    if is_linalg_on_tensors(text):
        return _linalg(text)
    return _shared(text)
