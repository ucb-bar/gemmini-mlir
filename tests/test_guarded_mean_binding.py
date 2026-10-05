from pathlib import Path
import pytest
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_resadd_proof import op_name
from mlir_oot.guarded_mean_bundle import inspect, packed_nhwc_route, rewrite
from mlir_oot.direct_conv_binding import serialize


def source():
    return Path(__file__).with_name('guarded_mean_fixture.mlir').read_text()


def quantizer(module):
    return next(o for o in module.walk() if op_name(o)=='quant_ext.quantize_per_tensor')


def test_bound_input_geometry_and_writer_survive_serialization():
    module=parse_module(source());q=quantizer(module);route=inspect(q)
    assert route['shape']==[1,2,1,2] and route['proof']['count']==2
    original_input=route['input']
    rewrite(module,q,route,'mean_test','a'*64)
    module.verify();reparsed=parse_module(serialize(module,[]));reparsed.verify()
    declaration=next(o for o in reparsed.body.block.ops if o.name=='func.func' and o.sym_name.data=='mean_test')
    assert [x.data['bufferization.access'].data for x in declaration.arg_attrs]==['read','write']
    call=next(o for o in module.walk() if o.name=='func.call')
    value=call.arguments[0]
    while value is not original_input and value.owner.name.startswith('tensor.'):
        value=value.owner.operands[0]
    assert value is original_input
    assert call.arguments[1].owner.name=='tensor.empty'


@pytest.mark.parametrize('old,new', [
    ('dimensions=[2,3]','dimensions=[3,2]'),
    ('%fzero = arith.constant 0.0','%fzero = arith.constant 1.0'),
    ('%n = arith.constant 2.0','%n = arith.constant 3.0'),
    ('%z = arith.constant 0','%z = arith.constant 1'),
    ('arith.addf %x,%acc','arith.mulf %x,%acc'),
    ('arith.divf %v,%d','arith.divf %d,%v'),
])
def test_unproved_source_variants_refuse(old,new):
    with pytest.raises(ValueError):inspect(quantizer(parse_module(source().replace(old,new))))


def packed_source():
    text=source().replace('tensor<1x2x1x2','tensor<1x8x1x2').replace('tensor<1x2x','tensor<1x8x')
    text=text.replace('%a:tensor<1x8x1x2xi8>', '%physical:tensor<1x1x2x8xi8>')
    return text.replace(' %dq =', ''' %init_transpose = tensor.empty() : tensor<1x8x1x2xi8>
 %a = linalg.transpose ins(%physical:tensor<1x1x2x8xi8>) outs(%init_transpose:tensor<1x8x1x2xi8>) permutation=[0,3,1,2]
 %dq =''')


def test_packed_binding_uses_original_nhwc_input_and_preserves_output_shape():
    module=parse_module(packed_source());q=quantizer(module)
    route=packed_nhwc_route(inspect(q));physical=route['input']
    assert route['input_matrix_shape']==[2,8] and route['output_shape']==[1,8]
    rewrite(module,q,route,'packed_test','b'*64)
    module.verify();parse_module(serialize(module,[])).verify()
    call=next(o for o in module.walk() if o.name=='func.call')
    value=call.arguments[0]
    while value is not physical:value=value.owner.operands[0]
    assert value is physical


def test_packed_binding_refuses_missing_transpose_and_partial_byte_group():
    with pytest.raises(ValueError):packed_nhwc_route(inspect(quantizer(parse_module(source()))))
    text=packed_source().replace('1x8x','1x4x').replace('1x1x2x8','1x1x2x4')
    with pytest.raises(ValueError):packed_nhwc_route(inspect(quantizer(parse_module(text))))
