import subprocess

import numpy as np
import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

from mlir_oot.descriptor_writer import DescriptorWriterContract, shim, transform
from mlir_oot.frontend.parse import parse_module


SOURCE = '''module {
func.func @forward(%a:tensor<4xi8>) -> (tensor<4xi8>,tensor<4xi8>) attributes {llvm.emit_c_interface} {
 %out=tensor.empty():tensor<4xi8>
 %r=func.call @device(%a,%out):(tensor<4xi8>,tensor<4xi8>)->tensor<4xi8>
 return %a,%r:tensor<4xi8>,tensor<4xi8>
}
func.func private @device(tensor<4xi8>,tensor<4xi8>) -> tensor<4xi8> attributes {llvm.emit_c_interface}
}'''


def fixture(source=SOURCE):
    module = parse_module(source)
    declaration = module.body.block.last_op
    declaration.properties['arg_attrs'] = ArrayAttr([
        DictionaryAttr({'bufferization.access': StringAttr(a)}) for a in ('read', 'write')
    ])
    return module


@pytest.mark.parametrize('contract', [
    DescriptorWriterContract('device', 0, (1,)),
    DescriptorWriterContract('device', 1, (0, 1)),
    DescriptorWriterContract('absent', 1, (1,)),
])
def test_bad_contract_refuses_before_mutation(contract):
    module = fixture()
    before = str(module)
    with pytest.raises(ValueError):
        transform(module, [contract])
    assert str(module) == before


def test_live_writable_tensor_refuses():
    module = fixture(SOURCE.replace('return %a,%r', 'return %out,%r'))
    with pytest.raises(ValueError, match='sole-use'):
        transform(module, [DescriptorWriterContract('device', 1, (1,))])


def test_expanded_adapter_without_ranked_c_interface_refuses():
    module = fixture()
    del module.body.block.last_op.attributes['llvm.emit_c_interface']
    with pytest.raises(ValueError, match='ranked C-interface'):
        transform(module, [DescriptorWriterContract('device', 1, (1,))])


@pytest.mark.parametrize('optimization', ['-O0', '-O2'])
def test_actual_upstream_ownership_live_input_and_fresh_result(tmp_path, optimization):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    module = fixture()
    routes = transform(module, [DescriptorWriterContract('device', 1, (1,))])
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path/'lower', vectorize=False)
    source = tmp_path/'model.ll'
    source.write_text(llvm)
    adapter = tmp_path/'adapter.c'
    adapter.write_text('''#include <stdint.h>
typedef struct {void *allocated,*aligned;int64_t offset,size[1],stride[1];} M;
void _mlir_ciface_device(M*r,M*a,M*out) {
 for(int i=0;i<4;i++)((int8_t*)out->aligned)[out->offset+i*out->stride[0]]=(int8_t)(((int8_t*)a->aligned)[a->offset+i*a->stride[0]]+1);
 *r=*out;
}
''')
    bridge = tmp_path/'bridge.c'
    bridge.write_text(shim(routes))
    obj = tmp_path/'model.o'
    lib = tmp_path/f'model{optimization}.so'
    subprocess.run([str(clang()), optimization, '-fPIC', '-c', str(source), '-o', str(obj)], check=True)
    subprocess.run(['cc', optimization, '-fPIC', '-shared', str(obj), str(adapter), str(bridge),
                    str(mlir_runtime_c()), '-o', str(lib)], check=True)
    inp = np.array([-128, -1, 0, 125], dtype=np.int8)
    first, second = np.full(4, 77, dtype=np.int8), np.full(4, 77, dtype=np.int8)
    invoke = HostModel.load(str(lib))
    for _ in range(3):
        invoke([(v.ctypes.data, v.shape) for v in (inp, first, second)])
        np.testing.assert_array_equal(inp, [-128, -1, 0, 125])
        np.testing.assert_array_equal(first, inp)
        np.testing.assert_array_equal(second, [-127, 0, 1, 126])
