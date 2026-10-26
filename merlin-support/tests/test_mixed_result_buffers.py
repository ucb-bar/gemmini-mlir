"""Independent typed destinations and the actual MLIR multiple-result C ABI."""
import copy
import subprocess
from pathlib import Path

import pytest

from mlir_oot.core_aten.catalog import authenticate_catalog, build
from mlir_oot.core_aten.offload import rewrite, shim_source
from test_shared_input_fusion import options, source


def mixed():
    text, _, _ = source(3, 5, (7, 9), constant=False, raw=(False, True), relu=(False,))
    _, manifest, matches = build(text, readout_options=options())
    assert len(matches) == 1 and matches[0].result_buffers, manifest['declines']
    return text, manifest, matches


@pytest.mark.parametrize('field', ['source', 'argument', 'shape', 'type', 'byte_offset', 'byte_length',
                                   'source_operation_ordinal', 'remove', 'add', 'order', 'output_identity'])
def test_result_buffer_binding_rejects_ordinal_type_shape_or_argument_mutation(field):
    _, manifest, matches = mixed()
    authenticate_catalog(manifest, matches)
    changed = copy.deepcopy(manifest)
    rows = changed['bindings'][0]['result_buffers']
    if field == 'source':
        rows[0][field]['op_ordinal'] += 1
    elif field == 'argument':
        rows[0][field] = 'out1'
    elif field == 'shape':
        rows[0][field][0] += 1
    elif field == 'type':
        rows[0][field] = 'tensor<3x7xi32>'
    elif field in ('byte_offset', 'byte_length', 'source_operation_ordinal'):
        rows[0][field] += 1
    elif field == 'remove':
        rows.pop()
    elif field == 'add':
        rows.append(copy.deepcopy(rows[0]))
    elif field == 'order':
        rows.reverse()
    else:
        outs = [a for a in changed['kernels'][0]['args'] if a['role'] == 'output']
        outs[0]['name'], outs[1]['name'] = outs[1]['name'], outs[0]['name']
    with pytest.raises(ValueError):
        authenticate_catalog(changed, matches)


def test_multi_result_offload_uses_real_mlir_ciface_aggregate(tmp_path):
    from merlin.llvmlower import toolchain
    from mlir_oot.frontend.parse import parse_module

    # Different ranks make accidental one-descriptor-per-result conventions observable.
    text = '''module {
      func.func private @pair(tensor<3xi8>, tensor<2x3xi32>) -> (tensor<3xi8>, tensor<2x3xi32>)
      func.func @forward(%a:tensor<3xi8>, %b:tensor<2x3xi32>) -> (tensor<3xi8>,tensor<2x3xi32>) {
        %x,%y=func.call @pair(%a,%b):(tensor<3xi8>,tensor<2x3xi32>)->(tensor<3xi8>,tensor<2x3xi32>)
        func.return %x,%y:tensor<3xi8>,tensor<2x3xi32>
      }
    }'''
    module = parse_module(text)
    ops = list(module.walk())
    call = next(o for o in ops if o.name == 'func.call')
    ordinal = ops.index(call)
    args = [dict(name='a',dtype='i8',shape=[3],role='input'),
            dict(name='b',dtype='i32',shape=[2,3],role='input'),
            dict(name='out0',dtype='i8',shape=[3],role='output'),
            dict(name='out1',dtype='i32',shape=[2,3],role='output')]
    binding = dict(symbol='probe', root_ordinal=ordinal, replaced_ordinals=[ordinal],
        operands=dict(a=dict(func_arg=0),b=dict(func_arg=1)), result_buffers=[
            dict(source=dict(op_ordinal=ordinal,result=i),argument=a['name'],shape=a['shape'],
                 type=str(v.type),byte_offset=0,byte_length=(3 if i==0 else 24),
                 source_operation_ordinal=ordinal) for i,(a,v) in enumerate(zip(args[2:],call.results))])
    manifest = dict(kernels=[dict(symbol='probe', args=args)], bindings=[binding])
    rewritten = rewrite(text, manifest)
    parse_module(rewritten).verify()
    assert rewritten.count('func.call') == 1 and 'ca_buffers_probe' in rewritten
    assert 'tensor.extract_slice' not in rewritten
    # External pair is now dead but remains a declaration, harmless to lowering.
    src = tmp_path/'probe.mlir'
    src.write_text(rewritten)
    llvm = toolchain.llvm_install()/'bin'
    lowered = tmp_path/'probe.llvm.mlir'
    ir = tmp_path/'probe.ll'
    subprocess.run([str(llvm/'mlir-opt'),str(src),
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
        '--convert-bufferization-to-memref','--convert-func-to-llvm','--convert-arith-to-llvm',
        '--finalize-memref-to-llvm','--reconcile-unrealized-casts','-o',str(lowered)],check=True,capture_output=True)
    subprocess.run([str(llvm/'mlir-translate'),'--mlir-to-llvmir',str(lowered),'-o',str(ir)],check=True,capture_output=True)
    signature = next(line for line in ir.read_text().splitlines() if line.startswith('declare void @_mlir_ciface_ca_buffers_probe'))
    assert signature.count('ptr') == 5  # aggregate, two inputs, two destinations
    assert '{ ptr, ptr, i64, [1 x i64], [1 x i64] }, { ptr, ptr, i64, [2 x i64], [2 x i64] }' in ir.read_text()
    c = tmp_path/'shim.c'
    c.write_text(shim_source(manifest) + r'''
#include <stdlib.h>
#include <assert.h>
void ca_abort(const char *why) { fprintf(stderr,"%s\n",why); abort(); }
void probe(void *a,void *b,void *x,void *y) {
  memcpy(x,a,3); memcpy(y,b,24);
}
extern void _mlir_ciface_forward(ca_results_probe*,ca_memref_1*,ca_memref_2*);
int main(void) {
  int8_t a[]={-128,17,127}; int32_t b[]={100003,-200004,3,4,5,2147483647};
  ca_memref_1 x={a,a,0,{3},{1}}; ca_memref_2 y={b,b,0,{2,3},{3,1}};
  ca_results_probe result;
  _mlir_ciface_forward(&result,&x,&y);
  assert(!memcmp(result.result0.aligned,a,3));
  assert(!memcmp(result.result1.aligned,b,24));
  free(result.result0.allocated); free(result.result1.allocated);
  return 0;
}
''')
    exe=tmp_path/'probe'
    subprocess.run([str(llvm/'clang'),str(ir),str(c),'-o',str(exe)],check=True,capture_output=True)
    subprocess.run([str(exe)],check=True,capture_output=True)


def test_homogeneous_flat_result_abi_still_works():
    text,_,_=source(3,5,(7,9),constant=False,raw=False)
    _,manifest,matches=build(text,readout_options=options())
    assert matches[0].outputs and not matches[0].result_buffers
    assert 'result_slices' in manifest['bindings'][0]
    rewritten=rewrite(text,manifest)
    assert rewritten.count('tensor.extract_slice')==2
    assert 'ca_buffers_' not in rewritten
    assert '*result = *out;' in shim_source(manifest)


def test_buffer_shim_rejects_misalignment_aliasing_and_nonfinite_scale(tmp_path):
    from merlin.llvmlower import toolchain
    args=[dict(name='scale',dtype='f32',shape=[1],role='input'),
          dict(name='out0',dtype='i8',shape=[3],role='output'),
          dict(name='out1',dtype='i32',shape=[2],role='output')]
    manifest=dict(kernels=[dict(symbol='checked',args=args,runtime_scale_arguments=['scale'])],bindings=[dict(symbol='checked',result_buffers=[])])
    c=tmp_path/'checked.c'
    c.write_text(shim_source(manifest)+r'''
#include <stdlib.h>
void ca_abort(const char *why) { (void)why; exit(37); }
void checked(void *s,void *x,void *y) { (void)s; (void)x; (void)y; }
int main(int argc,char **argv) {
  _Alignas(4) uint8_t scale[8]={0,0,128,62}, x[16]={0}, y[16]={0};
  ca_memref_1 s={scale,scale,0,{1},{1}}, a={x,x,0,{3},{1}}, b={y,y,0,{2},{1}};
  ca_results_checked result;
  switch (argc>1 ? argv[1][0] : 'v') {
    case 'i': s.aligned=scale+1; break;
    case 'o': b.aligned=y+1; break;
    case 'a': a.aligned=scale; break;
    case 'b': b.aligned=x; break;
    case 'n': scale[2]=192; scale[3]=127; break;
    case 'f': scale[2]=128; scale[3]=127; break;
    case 's': b.sizes[0]=1; break;
    case 'p': b.strides[0]=2; break;
  }
  _mlir_ciface_ca_buffers_checked(&result,&s,&a,&b);
  return 0;
}
''')
    exe=tmp_path/'checked'
    subprocess.run([str(toolchain.llvm_install()/'bin/clang'),str(c),'-o',str(exe)],check=True,capture_output=True)
    for mutation in ['v','i','o','a','b','n','f','s','p']:
        result=subprocess.run([str(exe),mutation],capture_output=True)
        assert result.returncode==(0 if mutation=='v' else 37),mutation


def test_opaque_f32_inputs_are_not_runtime_scale_operands():
    from test_shared_input_fusion import provider, PACKAGE, POLICY

    text = """builtin.module {
      func.func @forward(%a:tensor<3xf32>,%b:tensor<2xf32>) -> tensor<5xf32> {
        %out = tensor.concat dim(0) %a,%b : (tensor<3xf32>,tensor<2xf32>) -> tensor<5xf32>
        func.return %out : tensor<5xf32>
      }
    }"""
    _, manifest, matches = build(text)
    authenticate_catalog(manifest, matches)
    assert all('runtime_scale_arguments' not in k for k in manifest['kernels'])
    shim = shim_source(manifest)
    assert 'nonfinite runtime scale' not in shim
    assert 'ca_results_' not in shim
