"""Bind the exact guarded mean to a canonical captured Q/DQ reduction.

The explicit bundle preserves the original reference. Its CPU adapter uses the
ordinary ranked-memref interface and returns the actual destination descriptor.
Unsupported arithmetic, axes, layouts or ABI strides refuse.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, NoneAttr, StringAttr, UnitAttr
from xdsl.ir import Operation, Region

from .captured_requant import scalar
from .captured_residual_bundle import reshape
from .direct_conv_binding import serialize
from .frontend.parse import parse_module
from .golden_resadd_proof import _qdq, op_name
from .guarded_quantized_mean import derive, emit_kernel, emit_packed_nhwc_kernel
from .no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(quantize):
    if op_name(quantize) != 'quant_ext.quantize_per_tensor':
        raise ValueError('not a scalar quantizer')
    output_scale = _qdq(quantize, input_elem='f32', output_elem='i8')
    value = quantize.operands[0]
    while isinstance(value.owner, Operation) and value.owner.name in ('tensor.collapse_shape', 'tensor.expand_shape'):
        view = value.owner
        if len(view.operands) != 1 or math.prod(view.operands[0].type.get_shape()) != math.prod(value.type.get_shape()):
            raise ValueError('dynamic or non-bijective mean view')
        value = view.operands[0]
    division = value.owner
    if not isinstance(division, Operation) or division.name != 'linalg.generic':
        raise ValueError('not mean division')
    if len(division.inputs) != 2 or len(division.outputs) != 1:
        raise ValueError('noncanonical mean division arity')
    block = division.body.block
    ops = list(block.ops)
    from xdsl.ir.affine import AffineMap
    rank = len(value.type.get_shape())
    if (rank != 2 or any(x.data != AffineMap.identity(rank) for x in division.indexing_maps.data)
            or any(x.data.value != 'parallel' for x in division.iterator_types.data)
            or [x.name for x in ops] != ['arith.divf', 'linalg.yield']
            or list(ops[0].operands) != list(block.args[:2])
            or list(ops[1].operands) != list(ops[0].results)):
        raise ValueError('noncanonical mean scalar division or indexing')
    reduction = division.inputs[0].owner
    if not isinstance(reduction, Operation) or reduction.name != 'linalg.reduce':
        raise ValueError('not a serial reduction')
    if (len(reduction.operands) != 2 or tuple(reduction.dimensions.get_values()) != (2, 3)
            or scalar(reduction.operands[1]) != 0):
        raise ValueError('mean requires zero init and trailing two reduction axes')
    source = reduction.operands[0]
    shape = source.type.get_shape()
    if len(shape) != 4 or any(d <= 0 for d in shape):
        raise ValueError('static BCHW mean required')
    reduced_shape = tuple(shape[:2])
    if value.type.get_shape() != reduced_shape or quantize.results[0].type.get_shape() != reduced_shape:
        raise ValueError('mean output geometry changed')
    count = shape[2] * shape[3]
    if scalar(division.inputs[1]) != count:
        raise ValueError('mean divisor differs from reduction extent')
    body = reduction.regions[0].block
    body_ops = list(body.ops)
    if ([o.name for o in body_ops] != ['arith.addf', 'linalg.yield']
            or set(body_ops[0].operands) != set(body.args)
            or list(body_ops[1].operands) != list(body_ops[0].results)):
        raise ValueError('noncanonical sum body')
    for op in (ops[0], body_ops[0]):
        if str(op.properties.get('fastmath', '#arith.fastmath<none>')) != '#arith.fastmath<none>':
            raise ValueError('fastmath mean unsupported')
    dq = source.owner
    if not isinstance(dq, Operation) or op_name(dq) != 'quant_ext.dequantize_per_tensor':
        raise ValueError('not scalar dequantized input')
    input_scale = _qdq(dq, input_elem='i8', output_elem='f32')
    proof = derive(count, input_scale, output_scale)
    if any(not isinstance(t.encoding, NoneAttr) for t in (dq.operands[0].type, quantize.results[0].type)):
        raise ValueError('encoded mean tensors unsupported')
    return dict(input=dq.operands[0], shape=list(shape), output_shape=list(reduced_shape),
                channels=math.prod(reduced_shape), proof=proof,
                source_region=getattr(quantize.attributes.get('prov.region_id'), 'data', ''))


def rewrite(module, quantize, route, symbol, source_sha):
    if any(o.name == 'func.func' and o.sym_name.data == symbol for o in module.body.block.ops):
        raise ValueError('mean symbol already declared')
    views, value = reshape(route['input'], route.get('input_matrix_shape', [route['channels'], route['proof']['count']]))
    outtype = quantize.results[0].type
    empty = tensor.EmptyOp([], outtype)
    call = func.CallOp(symbol, [value, empty.tensor], [outtype])
    quantize.parent.insert_ops_before([*views, empty, call], quantize)
    quantize.results[0].replace_all_uses_with(call.results[0])
    quantize.parent.erase_op(quantize)
    proof_sha = hashlib.sha256(json.dumps(route['proof'], sort_keys=True).encode()).hexdigest()
    declaration = func.FuncOp(symbol, ([value.type, outtype], [outtype]), Region(), visibility='private',
        arg_attrs=ArrayAttr([DictionaryAttr({'bufferization.access': StringAttr(x)}) for x in ('read', 'write')]))
    declaration.attributes.update({'llvm.emit_c_interface': UnitAttr(),
        'merlin.guarded_mean_proof_sha256': StringAttr(proof_sha),
        'merlin.guarded_mean_source_sha256': StringAttr(source_sha)})
    module.body.block.add_op(declaration)


def adapter(route, symbol):
    count, channels = route['proof']['count'], route['channels']
    b, c = route['output_shape']
    if route.get('physical_layout') == 'NHWC':
        kernel = emit_packed_nhwc_kernel(route['proof'], symbol+'_kernel', b, c)
    else:
        kernel = emit_kernel(route['proof'], symbol+'_kernel', channels)
    rows, cols = route.get('input_matrix_shape', [channels, count])
    return kernel + f'''
typedef struct {{void *allocated,*aligned; int64_t offset,size[2],stride[2];}} {symbol}_memref2;
void _mlir_ciface_{symbol}({symbol}_memref2 *result,{symbol}_memref2 *a,{symbol}_memref2 *out) {{
 if(a->size[0]!={rows} || a->size[1]!={cols} || a->stride[0]!={cols} || a->stride[1]!=1 ||
    out->size[0]!={b} || out->size[1]!={c} || out->stride[0]!={c} || out->stride[1]!=1) __builtin_trap();
 {symbol}_kernel((const int8_t*)a->aligned+a->offset,(int8_t*)out->aligned+out->offset);
 *result=*out;
}}
'''


def packed_nhwc_route(route):
    """Prove the named transpose and preserve each channel's H/W order."""
    view = route['input'].owner
    b, c, h, w = route['shape']
    if (not isinstance(view, Operation) or view.name != 'linalg.transpose'
            or tuple(view.permutation.get_values()) != (0, 3, 1, 2)
            or tuple(view.inputs[0].type.get_shape()) != (b, h, w, c)
            or c % 8 or not isinstance(view.inputs[0].type.encoding, NoneAttr)):
        raise ValueError('packed mean requires an exact NHWC to BCHW transpose and channels divisible by8')
    view.verify()
    return route | dict(input=view.inputs[0], input_matrix_shape=[b*h*w, c], physical_layout='NHWC',
                        transpose_permutation=[0, 3, 1, 2])


def build_and_apply(capture, llvm_bin, directory, *, packed_nhwc=False):
    capture, llvm_bin, directory = map(Path, (capture, llvm_bin, directory))
    directory.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((capture/'capture_receipt.json').read_text())
    for name in ('model.mlir', 'weights.safetensors', 'weights.safetensors.manifest.json'):
        if sha(capture/name) != receipt['artifacts'][name]['sha256']:
            raise ValueError('capture identity changed: '+name)
    original = sha(capture/'model.mlir'); golden = sha(capture/'golden.npy')
    module = parse_module((capture/'model.mlir').read_text())
    routes = []
    for op in list(module.walk()):
        if op_name(op) != 'quant_ext.quantize_per_tensor': continue
        try:
            route = inspect(op)
            if packed_nhwc: route = packed_nhwc_route(route)
        except ValueError: continue
        symbol = 'merlin_guarded_quant_mean_'+str(len(routes))
        rewrite(module, op, route, symbol, original)
        routes.append({k:v for k,v in route.items() if k != 'input'} | {'symbol': symbol})
    if not routes: raise ValueError('no canonical Q/DQ mean matched')
    module.verify()
    rewritten = directory/'rewritten.mlir'; rewritten.write_text(serialize(module, []))
    parse_module(rewritten.read_text()).verify()
    code = directory/'adapter.c'; code.write_text('\n'.join(adapter(r, r['symbol']) for r in routes))
    obj = directory/'adapter.o'
    command = [str(llvm_bin/'clang'), '--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d',
               '-mcmodel=medany', '-O2', '-ffp-contract=off', '-ffreestanding', '-fno-builtin', '-c', str(code), '-o', str(obj)]
    subprocess.run(command, check=True, capture_output=True)
    audit = audit_elf(obj.read_bytes())
    if audit['status'] != 'pass': raise ValueError('mean object audit failed')
    record = dict(schema='guarded_mean_bundle_v1', source_sha256=original,
                  rewritten_sha256=sha(rewritten), original_golden_sha256=golden,
                  routes=routes, object_sha256=sha(obj), source_code_sha256=sha(code),
                  compiler_argv=command, compiler_sha256=sha(llvm_bin/'clang'), object_audit=audit)
    (directory/'mean.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copyfile(rewritten, capture/'model.mlir')
    receipt['artifacts']['model.mlir'] = dict(bytes=(capture/'model.mlir').stat().st_size, sha256=record['rewritten_sha256'])
    receipt['guarded_mean_derivation'] = dict(source_sha256=original, bundle_sha256=sha(directory/'mean.json'), original_golden_sha256=golden)
    (capture/'capture_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert sha(capture/'golden.npy') == golden
    return record


def merlin_callbacks(llvm_bin, directory, base_callbacks):
    llvm_bin, directory = map(Path, (llvm_bin, directory))
    prepare, compile_base = base_callbacks
    record = json.loads((directory/'mean.json').read_text())
    pin = sha(directory/'mean.json')
    symbols = {r['symbol']: r for r in record['routes']}
    def check(source):
        if sha(directory/'mean.json') != pin:
            raise ValueError('mean manifest identity changed')
        module = parse_module(Path(source).read_text())
        calls = [op.callee.root_reference.data for op in module.walk() if op.name == 'func.call' and op.callee.root_reference.data in symbols]
        declarations = {op.sym_name.data: op for op in module.walk() if op.name == 'func.func' and op.sym_name.data in symbols}
        if sorted(calls) != sorted(symbols) or set(declarations) != set(symbols):
            raise ValueError('mean call/declaration coverage changed')
        for symbol, op in declarations.items():
            expected = hashlib.sha256(json.dumps(symbols[symbol]['proof'], sort_keys=True).encode()).hexdigest()
            if (getattr(op.attributes.get('merlin.guarded_mean_proof_sha256'), 'data', None) != expected
                    or getattr(op.attributes.get('merlin.guarded_mean_source_sha256'), 'data', None) != record['source_sha256']):
                raise ValueError('mean source/certificate binding changed')
            attrs = op.properties.get('arg_attrs')
            if attrs is None or [getattr(x.data.get('bufferization.access'), 'data', None) for x in attrs] != ['read', 'write']:
                raise ValueError('mean bufferization access changed')
    def prepare_bound(source, work):
        check(source)
        prepared = prepare(source, work)
        check(prepared)
        return prepared
    def compile(source, work):
        check(source)
        if sha(directory/'adapter.o') != record['object_sha256'] or sha(directory/'adapter.c') != record['source_code_sha256']:
            raise ValueError('mean artifact identity changed')
        manifest_path, obj = compile_base(source, work)
        merged = Path(work)/'guarded_mean_mixed.o'
        linker = llvm_bin/'ld.lld'
        if not linker.is_file():
            selected = shutil.which('ld.lld')
            if selected is None: raise ValueError('ld.lld required')
            linker = Path(selected)
        command = [str(linker), '-r', str(obj), str(directory/'adapter.o'), '-o', str(merged)]
        subprocess.run(command, check=True, capture_output=True)
        if audit_elf(merged.read_bytes())['status'] != 'pass': raise ValueError('mixed mean audit failed')
        manifest = json.loads(manifest_path.read_text())
        manifest['compilation'] = dict(schema='guarded_mean_mixed_compile_v1', object_sha256=sha(merged),
            object_nofsm_status='pass', remaining_compilation=manifest['compilation'],
            mean_manifest_sha256=sha(directory/'mean.json'), linker_argv=command, linker_sha256=sha(linker))
        manifest['guarded_mean_additions'] = record['routes']
        manifest['native_oracle_sources'].append(str(directory/'adapter.c'))
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest_path, merged
    return prepare_bound, compile


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derived-capture', type=Path, required=True,
                        help='Owned derived capture; model and receipt are rewritten in place')
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packed-nhwc', action='store_true')
    args = parser.parse_args()
    record = build_and_apply(args.derived_capture, args.llvm_bin, args.output, packed_nhwc=args.packed_nhwc)
    print(json.dumps({'routes': len(record['routes']),
                      'manifest': str(args.output / 'mean.json'),
                      'object_sha256': record['object_sha256']}))


if __name__ == '__main__':
    main()
