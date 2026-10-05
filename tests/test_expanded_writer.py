"""The provider adapts the real expanded return ABI to borrowed full writers."""
import subprocess
import numpy as np
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from mlir_oot.frontend.parse import parse_module
from mlir_oot.expanded_writer import ExpandedWriterContract, transform, shim


def native_case(tmp_path, m, n, k, a, b, *, adapter_source=None, symbol='raw', kernel='scalar_kernel'):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.device_shim import emit_dense_translation_unit
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text
    module = parse_module(f'''module {{
func.func @forward(%a:tensor<{m}x{k}xi8>,%b:tensor<{k}x{n}xi8>) -> (tensor<{m}x{k}xi8>,tensor<{m}x{n}xi32>) attributes {{llvm.emit_c_interface}} {{
 %c=tensor.empty():tensor<{m}x{n}xi32>
 %r=func.call @{symbol}(%a,%b,%c):(tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>,tensor<{m}x{n}xi32>)->tensor<{m}x{n}xi32>
 return %a,%r:tensor<{m}x{k}xi8>,tensor<{m}x{n}xi32>
}}
func.func private @{symbol}(tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>,tensor<{m}x{n}xi32>)->tensor<{m}x{n}xi32>
}}''')
    module.body.block.last_op.properties['arg_attrs'] = ArrayAttr([
        DictionaryAttr({'bufferization.access': StringAttr(x)}) for x in ('read','read','write')])
    routes = transform(module, [ExpandedWriterContract(symbol,'borrowed',2,(2,),32)])
    source = tmp_path/'model.mlir';source.write_text(text(module,generic=True))
    llvm = lower_to_llvm_ir(source.read_text(),workdir=tmp_path/'lower',vectorize=False)
    (tmp_path/'model.ll').write_text(llvm)
    (tmp_path/'borrowed.c').write_text(shim(routes))
    if adapter_source is None:
        adapter_source=emit_dense_translation_unit('fixture',{symbol:(m,n,k)},{symbol:('i8','i8','i32')},kernel_symbol_for=lambda _:kernel).text
    (tmp_path/'adapter.c').write_text(adapter_source)
    (tmp_path/'kernel.c').write_text(f'''#include <stdint.h>
void {kernel}(void*aa,void*bb,void*cc) {{
 int8_t*a=aa,*b=bb;int32_t*c=cc;
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{int32_t v=0;for(int z=0;z<{k};z++)v+=(int32_t)a[i*{k}+z]*b[z*{n}+j];c[i*{n}+j]=v;}}
}}
''')
    obj=tmp_path/'model.o';lib=tmp_path/'model.so'
    subprocess.run([str(clang()),'-O2','-fPIC','-c',str(tmp_path/'model.ll'),'-o',str(obj)],check=True)
    subprocess.run(['cc','-O2','-fPIC','-shared',str(obj),str(tmp_path/'adapter.c'),str(tmp_path/'borrowed.c'),str(tmp_path/'kernel.c'),str(mlir_runtime_c()),'-o',str(lib)],check=True)
    raw=np.full(m*n+32,0x12345678,dtype=np.int32);out=raw[16:-16].reshape(m,n)
    live=np.empty_like(a);saved_a=a.copy();saved_b=b.copy();expected=a.astype(np.int32)@b.astype(np.int32)
    invoke=HostModel.load(str(lib))
    for _ in range(3):
        invoke([(x.ctypes.data,x.shape) for x in (a,b,live,out)])
        np.testing.assert_array_equal(a,saved_a);np.testing.assert_array_equal(b,saved_b)
        np.testing.assert_array_equal(live,a);np.testing.assert_array_equal(out,expected)
        assert np.all(raw[:16]==0x12345678) and np.all(raw[-16:]==0x12345678)
    return routes


def test_real_expanded_adapter_fresh_output_and_live_input(tmp_path):
    a=np.array([[-128,127,2,3],[9,-8,5,6]],dtype=np.int8)
    b=np.arange(12,dtype=np.int8).reshape(4,3)-6
    native_case(tmp_path,2,3,4,a,b)


def test_provider_callback_requires_source_bound_complete_write_metadata(tmp_path):
    import hashlib,json
    from mlir_oot.expanded_writer import merlin_callbacks
    source=tmp_path/'source.mlir'
    source.write_text('''module {
func.func @forward(%a:tensor<2x4xi8>,%b:tensor<4x3xi8>)->tensor<2x3xi32> {
 %e=tensor.empty():tensor<2x3xi32>
 %r=func.call @raw(%a,%b,%e):(tensor<2x4xi8>,tensor<4x3xi8>,tensor<2x3xi32>)->tensor<2x3xi32>
 return %r:tensor<2x3xi32>
}
func.func private @raw(tensor<2x4xi8> {bufferization.access="read"},tensor<4x3xi8> {bufferization.access="read"},tensor<2x3xi32> {bufferization.access="write"})->tensor<2x3xi32>
}''')
    from merlin.xdsl_dialects._common import text
    module=parse_module(source.read_text())
    module.body.block.last_op.properties['arg_attrs']=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(v)}) for v in ('read','read','write')])
    source.write_text(text(module,generic=True))
    sha=hashlib.sha256(source.read_bytes()).hexdigest();obj=tmp_path/'kernel.o';obj.write_bytes(b'owned object')
    types=['tensor<2x4xi8>','tensor<4x3xi8>','tensor<2x3xi32>']
    catalog={'abi':{},'source_sha256':sha,'compilation':{'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest()},'bindings':[{'region':'region','symbol':'kernel','tensor_types':types}]}
    manifest=tmp_path/'catalog.json';manifest.write_text(json.dumps(catalog))
    sidecar=tmp_path/'routing.json';sidecar.write_text(json.dumps({'device':'fixture','catalog_manifest':str(manifest),'catalog_object':str(obj),'model_sha256':sha,'signatures':{'raw':[2,3,4]},'routed':[{'source_region':'region','symbol':'raw','tensor_types':types,'dtypes':['i8','i8','i32']}]}))
    prepare,_=merlin_callbacks(tmp_path,lambda p,w:p,allocation_alignment=32,target_cflags=())
    work=tmp_path/'abi';work.mkdir()
    import pytest
    with pytest.raises(ValueError,match='explicit complete-write'):
        prepare(source,sidecar,work)
    assert not (work/'model.mlir').exists()
    catalog['abi']['writer_effects']={'schema':'complete_output_writer_v1','fully_written_arguments':[2],'retains_arguments':False,'frees_arguments':False};manifest.write_text(json.dumps(catalog))
    result=prepare(source,sidecar,work)
    assert '__fresh_tensor_result' in result.read_text()
    assert json.loads((work/'writer_contracts.json').read_text())['contracts'][0]['result_argument']==2
    obj.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity changed'):
        prepare(source,sidecar,work)
