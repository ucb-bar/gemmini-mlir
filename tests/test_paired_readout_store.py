import subprocess
import pytest
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_flat_conv import GoldenFlatConv
from mlir_oot.readout_store_plan import PairedReadoutPlan
from mlir_oot.tables import isa


@pytest.mark.parametrize('loop',[False,True])
@pytest.mark.parametrize('shape',[ConvShape(5,7,16,19,output_dtype='i32',explicit_halo=False),ConvShape(7,7,32,64,output_dtype='i8',scale=.125,explicit_halo=False),ConvShape(14,14,64,64,output_dtype='i32',explicit_halo=False)])
def test_default_program_byte_identity(loop,shape):
    baseline=subprocess.check_output(['git','show','eefe844:mlir_oot/golden_flat_conv.py'],text=True)
    namespace=dict(__name__='mlir_oot._paired_store_baseline',__package__='mlir_oot')
    exec(compile(baseline,'baseline.py','exec'),namespace)
    opts=dict(virtual_padding=True,loop_spatial=loop)
    assert str(GoldenFlatConv(shape,**opts).build())==str(namespace['GoldenFlatConv'](shape,**opts).build())


@pytest.mark.parametrize('loop',[False,True])
def test_two_passes_preserve_all_reduction_commands(loop):
    shape=ConvShape(5,7,16,19,output_dtype='i32',explicit_halo=False)
    plan=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1)
    baseline=GoldenFlatConv(shape,virtual_padding=True,loop_spatial=loop).build()
    candidate=GoldenFlatConv(shape,virtual_padding=True,loop_spatial=loop,store_plan=plan).build()
    candidate.verify()
    names={'gemmini.preload','gemmini.compute','gemmini.mvin'}
    assert [(op.name,op.attributes) for op in baseline.walk() if op.name in names]==[(op.name,op.attributes) for op in candidate.walk() if op.name in names]
    assert len(candidate.body.block.first_op.body.blocks.first.args)==4
    ops=[op for op in candidate.walk() if op.name.startswith('gemmini.')]
    stores=[op for op in ops if op.name=='gemmini.mvout']
    assert all(not op.a('local')&isa.ACC_FULL_ROW_BIT for op in stores)
    assert len(stores)==2*((shape.oh*shape.ow+15)//16)
    # Once a completed block is stored, its entire second pass follows without
    # any intervening compute or load overwriting accumulator state.
    first=next(i for i,op in enumerate(ops) if op.name=='gemmini.mvout')
    assert not any(op.name in names for op in ops[first:])


def test_refuse_unproved_or_changed_producer_contract():
    shape=ConvShape(5,7,16,19,output_dtype='i32',explicit_halo=False)
    with pytest.raises(ValueError):GoldenFlatConv(shape,virtual_padding=True,store_plan=True)
    with pytest.raises(ValueError):GoldenFlatConv(shape,virtual_padding=True,store_plan=PairedReadoutPlan((.25,),(.1,.4),-(1<<31),(1<<31)-1))
    with pytest.raises(ValueError):GoldenFlatConv(shape,virtual_padding=True,store_plan=PairedReadoutPlan((.125,),(.125,.125),-100,100))
    with pytest.raises(ValueError):GoldenFlatConv(ConvShape(5,7,16,19,output_dtype='i8',explicit_halo=False),virtual_padding=True,store_plan=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1))
