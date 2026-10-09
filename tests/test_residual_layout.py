from pathlib import Path
import pytest
from mlir_oot.frontend.parse import parse_module
from mlir_oot.captured_residual_bundle import inspect,rewrite,serialize,op_name
from mlir_oot.residual_layout import shared_transpose


def fixture():
    source=Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    source=source.replace('%a: tensor<16x64xi8>, %b: tensor<16x64xi8>', '%pa: tensor<64x16xi8>, %pb: tensor<64x16xi8>')
    source=source.replace('    %one =', '''    %init = tensor.empty() : tensor<16x64xi8>
    %a = linalg.transpose ins(%pa : tensor<64x16xi8>) outs(%init : tensor<16x64xi8>) permutation = [1, 0]
    %b = linalg.transpose ins(%pb : tensor<64x16xi8>) outs(%init : tensor<16x64xi8>) permutation = [1, 0]
    %one =''')
    module=parse_module(source);q=next(o for o in module.walk() if op_name(o)=='quant_ext.quantize_per_tensor')
    return module,q


def test_common_permutation_moves_to_result_preserving_abi_and_proof():
    m,q=fixture();route=inspect(q,implementation='cpu_lut')
    route['physical_layout']=shared_transpose(route['inputs'],route['shape'])
    decl=rewrite(q,route,'permuted_residual','a'*64,'b'*64)
    m.verify();parsed=parse_module(serialize(m,[decl]));parsed.verify()
    result=next(o for o in parsed.walk() if o.name=='func.return').operands[0]
    assert result.owner.name=='linalg.transpose'
    assert tuple(result.owner.permutation.get_values())==(1,0)
    call=next(o for o in parsed.walk() if o.name=='func.call')
    assert str(call.results[0].type)=='tensor<16x64xi8>'
    # Both adapter inputs trace to the physical function arguments, never their transpose.
    for value in call.operands[:2]:
        while getattr(value.owner,'name','') in ('tensor.expand_shape','tensor.collapse_shape'):
            value=value.owner.operands[0]
        assert value in next(o for o in parsed.walk() if o.name=='func.func' and o.body.blocks).body.block.args
    declaration=next(o for o in parsed.walk() if o.name=='func.func' and o.sym_name.data=='permuted_residual')
    assert 'gemmini.residual_layout' in declaration.attributes
    assert [x.data['bufferization.access'].data for x in declaration.arg_attrs]==['read','read','write']


def test_mismatched_permutation_refuses_even_for_equal_dimensions():
    m,q=fixture();text=str(m).replace('64x16','16x16').replace('16x64','16x16')
    # Change only the second input permutation, with shapes still valid.
    first=text.find('permutation = [1, 0]');second=text.find('permutation = [1, 0]',first+1)
    text=text[:second]+text[second:].replace('permutation = [1, 0]','permutation = [0, 1]',1)
    m=parse_module(text);m.verify();q=next(o for o in m.walk() if op_name(o)=='quant_ext.quantize_per_tensor');route=inspect(q,implementation='cpu_lut')
    with pytest.raises(ValueError,match='disagree'):shared_transpose(route['inputs'],route['shape'])


def test_unknown_operand_and_changed_shape_refuse():
    m,q=fixture();route=inspect(q,implementation='cpu_lut')
    with pytest.raises(ValueError,match='shape proof'):shared_transpose(route['inputs'],[64,16])
    unknown=[route['inputs'][0].owner.operands[0],route['inputs'][1]]
    with pytest.raises(ValueError,match='explicit transpose'):shared_transpose(unknown,route['shape'])
