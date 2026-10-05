import ctypes
from pathlib import Path
import subprocess
import numpy as np
import pytest
from xdsl.dialects.builtin import TensorType, i8
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_resadd_proof import op_name
from mlir_oot.captured_residual_bundle import inspect, rewrite, serialize, adapter, native_oracle, policy


def fixture():
    module=parse_module(Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text())
    q=next(x for x in module.walk() if op_name(x)=='quant_ext.quantize_per_tensor')
    return module,q


def test_exact_default_rewrite_keeps_typed_binding_and_accesses():
    module,q=fixture();route=inspect(q)
    declaration=rewrite(q,route,'residual_test','a'*64,'b'*64)
    module.verify();parsed=parse_module(serialize(module,[declaration]));parsed.verify()
    d=next(x for x in parsed.walk() if x.name=='func.func' and x.sym_name.data=='residual_test')
    assert d.attributes['gemmini.source_sha256'].data=='a'*64
    assert d.attributes['gemmini.numeric_policy'].data['max_output_lsb'].value.data==0
    assert [x.data['bufferization.access'].data for x in d.arg_attrs]==['read','read','write']
    assert not any(op_name(x).startswith('quant_ext.') for x in parsed.walk())


def test_nonmultiple_static_shape_refused():
    module,q=fixture()
    # Reparse a consistently shaped graph, avoiding a separate shape-match failure.
    module=parse_module(str(module).replace('16x64','15x63'))
    q=next(x for x in module.walk() if op_name(x)=='quant_ext.quantize_per_tensor')
    with pytest.raises(ValueError,match='divisible by 64'):inspect(q)


def test_policy_is_explicit_and_bounded():
    for value in (True,-1,2,1.0):
        with pytest.raises(ValueError):policy(value)
    assert policy(0)['kind']=='exact_source'
    assert policy(1)['kind']=='bounded_residual_output_lsb'


def test_rank_four_flatten_rewrite():
    source=Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    # The actual capture exercises rank4; construct reshaped operands around the fixture.
    module,q=fixture();route=inspect(q)
    from mlir_oot.captured_residual_bundle import reshape
    ops,value=reshape(route['inputs'][0],[1,4,4,64])
    for op in ops:op.verify()
    back,result=reshape(value,[16,64])
    for op in back:op.verify()
    assert result.type==route['inputs'][0].type


def test_native_oracle_matches_two_load_rounds_for_all_pairs(tmp_path):
    route={'m':1024,'proof':{'source':{'relu':True},'primitive':{'lhs_load':.5,'rhs_load':.5,'readout':1.}}}
    source='#include <stdint.h>\n'+native_oracle(route,'kernel')
    (tmp_path/'oracle.c').write_text(source)
    subprocess.run(['cc','-O2','-shared','-fPIC','-ffp-contract=off',str(tmp_path/'oracle.c'),'-lm','-o',str(tmp_path/'oracle.so')],check=True)
    kernel=ctypes.CDLL(str(tmp_path/'oracle.so')).kernel
    kernel.argtypes=[ctypes.c_void_p]*5
    a=np.repeat(np.arange(-128,128,dtype=np.int8),256)
    b=np.tile(np.arange(-128,128,dtype=np.int8),256);out=np.empty_like(a)
    kernel(a.ctypes.data,b.ctypes.data,out.ctypes.data,None,None)
    expected=np.clip(np.rint(a.astype(np.float32)*.5)+np.rint(b.astype(np.float32)*.5),0,127).astype(np.int8)
    source_sum=np.clip(np.rint((a.astype(np.float32)+b.astype(np.float32))*.5),0,127).astype(np.int8)
    assert np.array_equal(out,expected)
    assert np.any(out!=source_sum)


def test_one_lsb_route_requires_opt_in():
    source=Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    source=source.replace('%scale = tensor.splat %one : tensor<f32>', '%scale = tensor.splat %one : tensor<f32>\n    %half = arith.constant 0.5 : f32\n    %input_scale = tensor.splat %half : tensor<f32>')
    source=source.replace('(%a, %scale, %zp)', '(%a, %input_scale, %zp)').replace('(%b, %scale, %zp)', '(%b, %input_scale, %zp)')
    module=parse_module(source);q=next(x for x in module.walk() if op_name(x)=='quant_ext.quantize_per_tensor')
    with pytest.raises(ValueError,match='explicitly requested'):inspect(q)
    route=inspect(q,1)
    assert route['proof']['max_output_lsb_error']==1
    declaration=rewrite(q,route,'residual_bound','a'*64,'b'*64)
    parsed=parse_module(serialize(module,[declaration]));parsed.verify()
    declaration=next(x for x in parsed.walk() if x.name=='func.func' and x.sym_name.data=='residual_bound')
    assert declaration.attributes['gemmini.numeric_policy'].data['max_output_lsb'].value.data==1


def test_adapter_owned_table_links_at_baremetal_address(tmp_path):
    import shutil
    from mlir_oot.captured_residual_bundle import compile_adapter
    from mlir_oot.no_fsm_audit import audit_elf
    llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
    linker=llvm/'ld.lld'
    if not linker.is_file():linker=Path(shutil.which('ld.lld') or '/nonexistent')
    if not (llvm/'clang').is_file() or not linker.is_file():pytest.skip('target toolchain unavailable')
    source=tmp_path/'adapter.c';source.write_text('static const char table[256]={1}; const char *identity(void){return table;}')
    compilation=compile_adapter(source,tmp_path/'adapter.o',llvm)
    script=tmp_path/'link.ld';script.write_text('SECTIONS { . = 0x80000000; .text : { *(.text*) } .rodata : { *(.rodata*) } }')
    subprocess.run([str(linker),'-T',str(script),'-e','identity',str(tmp_path/'adapter.o'),'-o',str(tmp_path/'adapter.elf')],check=True,capture_output=True)
    assert audit_elf((tmp_path/'adapter.elf').read_bytes())['status']=='pass'
    assert compilation['object_sha256'] and compilation['source_sha256']
