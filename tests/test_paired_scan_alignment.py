from pathlib import Path
import pytest
from mlir_oot.captured_requant_bundle import build
from mlir_oot.golden_conv import ConvShape
from mlir_oot.paired_readout_binding import adapter
from mlir_oot.readout_store_plan import PairedReadoutPlan


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_normal_api_invalid_types_refuse_before_capture_io(value):
    with pytest.raises(ValueError,match='boolean'):
        build(Path('missing'),Path('missing'),Path('missing'),pair_scan_checked_alignment=value)


@pytest.mark.parametrize('policy,exact',[(None,True),('source_proven',False)])
def test_normal_api_requires_original_pair_and_exact_selection(policy,exact):
    with pytest.raises(ValueError,match='source-proven exact'):
        build(Path('missing'),Path('missing'),Path('missing'),pair_scan_checked_alignment=True,readout_pair_policy=policy,exact_integer_readout=exact)


def test_adapter_delegates_to_generic_emitter_without_default_change():
    from merlin.llvmlower.enclosed_readout import emit_pair_scan
    shape=ConvShape(3,3,16,16,output_dtype='i32',explicit_halo=True)
    plan=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1)
    before=adapter(shape,'boundary','kernel',plan)
    assert before==adapter(shape,'boundary','kernel',plan,checked_alignment=False)
    selected=adapter(shape,'boundary','kernel',plan,checked_alignment=True)
    a=emit_pair_scan(plan.certificate(),'boundary_pair_decode',copy_policy='compiler_builtin')
    b=emit_pair_scan(plan.certificate(),'boundary_pair_decode',copy_policy='compiler_builtin',checked_alignment=True)
    assert before.startswith(a) and selected==b+before[len(a):]


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_adapter_invalid_selection_refuses_before_producer_access(value):
    with pytest.raises(ValueError,match='boolean'):
        adapter(None,'boundary','kernel',None,checked_alignment=value)


def test_unselected_adapter_keeps_the_existing_shared_emitter_api(monkeypatch):
    import mlir_oot.paired_readout_binding as binding
    from merlin.llvmlower.enclosed_readout import emit_pair_scan
    shape=ConvShape(3,3,16,16,output_dtype='i32',explicit_halo=True)
    plan=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1)
    seen=[]
    def existing_emitter(certificate,symbol,*,copy_policy):
        seen.append(copy_policy)
        return emit_pair_scan(certificate,symbol,copy_policy=copy_policy)
    monkeypatch.setattr(binding,'emit_pair_scan',existing_emitter)
    binding.adapter(shape,'boundary','kernel',plan)
    assert seen==['compiler_builtin']
