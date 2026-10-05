from pathlib import Path
import numpy as np
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_resadd_proof import op_name
from mlir_oot.golden_residual_domains import domain


def fixture(relu=True, scale='1.0', zero='0.0'):
    text=Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    text=text.replace('1.0 : f32',f'{scale} : f32')
    if relu:
        start=text.index('    %q =')
        body='''    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum : tensor<16x64xf32>) outs(%empty : tensor<16x64xf32>) {
      ^bb0(%x: f32, %out: f32):
        %zero_f = arith.constant ZERO : f32
        %r = arith.maximumf %x, %zero_f : f32
        linalg.yield %r : f32
    } -> tensor<16x64xf32>
'''.replace('ZERO',zero)
        text=text[:start]+body+text[start:].replace('(%sum,','(%relu,')
    module=parse_module(text);module.verify()
    q=next(o for o in module.walk() if op_name(o)=='quant_ext.quantize_per_tensor')
    return module,q


def test_complete_finite_source_relu_establishes_nonnegative_domain():
    _,q=fixture()
    proof=domain(q.results[0])
    assert (proof['minimum'],proof['maximum'])==(0,127)
    assert proof['path'][0]['qparams']['relu']
    # Independent exhaustive source oracle establishes that bound.
    a=np.arange(-128,128,dtype=np.float32)
    assert np.all(np.clip(np.rint(np.maximum(a[:,None]+a[None,:],0)),-128,127)>=0)


def test_no_relu_unknown_input_and_nonzero_threshold_keep_full_domain():
    for options in ({'relu':False},{'zero':'-1.0'},{'scale':'3.0e+38'}):
        _,q=fixture(**options)
        assert domain(q.results[0])['minimum']==-128
    module,_=fixture()
    fn=next(o for o in module.walk() if o.name=='func.func')
    assert domain(fn.body.block.args[0])['minimum']==-128


def test_domain_survives_verified_static_reshape_and_transpose():
    from mlir_oot.captured_residual_bundle import reshape
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.ops import TransposeOp
    from xdsl.dialects.builtin import TensorType,i8,i64,DenseArrayBase
    _,q=fixture()
    ops,value=reshape(q.results[0],[32,32])
    assert len(ops)==2
    proof=domain(value)
    assert proof['minimum']==0
    assert [p['operation'] for p in proof['path'][-2:]]==['tensor.collapse_shape','tensor.expand_shape']
    init=tensor.EmptyOp([],TensorType(i8,[64,16]))
    tr=TransposeOp(q.results[0],init.tensor, DenseArrayBase.from_list(i64,[1,0]))
    assert domain(tr.results[0])['minimum']==0
