"""Distinct exact source uses may share only identical generated primitive code."""
from pathlib import Path
import runpy
import pytest
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_compiler_export import (
    SourceContractionEmitter, export_contraction, select_contraction_export,
)

source=runpy.run_path(str(Path(__file__).with_name('test_golden_compiler_export.py')))['source']
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')


def test_exact_source_identity_is_separate_from_physical_code(tmp_path):
    left=tmp_path/'left.mlir';right=tmp_path/'right.mlir'
    left.write_text(source(7,35,69))
    right.write_text(source(7,35,69).replace('"operation"','"other_region"'))
    a=select_contraction_export(left,'operation')
    b=select_contraction_export(right,'other_region')
    assert a['source_sha256']!=b['source_sha256'] and a['binding']!=b['binding']
    assert a['alternative'].id!=b['alternative'].id
    assert a['kernel_symbol']==b['kernel_symbol']
    assert a['implementation_ir_sha256']==b['implementation_ir_sha256']
    overlap=select_contraction_export(left,'operation',dense_input_policy='resident_a_prefetch')
    assert overlap['kernel_symbol']!=a['kernel_symbol']
    different=tmp_path/'different.mlir';different.write_text(source(7,35,70))
    c=select_contraction_export(different,'operation')
    assert c['kernel_symbol']!=a['kernel_symbol']


@pytest.mark.skipif(not (LLVM/'clang').is_file(),reason='real target compiler unavailable')
def test_physical_aliases_compile_to_exact_same_object(tmp_path):
    left=tmp_path/'left.mlir';right=tmp_path/'right.mlir'
    left.write_text(source(7,35,69))
    right.write_text(source(7,35,69).replace('"operation"','"other_region"'))
    a=export_contraction(left,'operation',LLVM,tmp_path/'a',dense_input_policy='resident_a_prefetch')
    b=export_contraction(right,'other_region',LLVM,tmp_path/'b',dense_input_policy='resident_a_prefetch')
    assert a['kernel_symbol']==b['kernel_symbol']
    assert a['compilation']['object_sha256']==b['compilation']['object_sha256']
    assert (tmp_path/'a/kernel.o').read_bytes()==(tmp_path/'b/kernel.o').read_bytes()
    assert a['compilation']['object_nofsm_status']=='pass'


def test_changed_primitive_refuses_even_when_schedule_fields_match(tmp_path):
    from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan
    path=tmp_path/'source.mlir';path.write_text(source(7,35,69))
    selected=select_contraction_export(path,'operation')
    alternative=selected['alternative'];generator=selected['generator']
    emitter=SourceContractionEmitter(path,selected['logical'],alternative,generator,LLVM,tmp_path/'emit',
        primitive_module=selected['primitive_module'])
    emitter.primitive_module.attributes['test_mutation']=StringAttr('different primitive program')
    plan=GlobalPlan((alternative,),(),alternative.cycles)
    with pytest.raises(ValueError,match='primitive implementation changed'):
        emitter.emit_global_plan(selected['logical'],plan)
    assert not (tmp_path/'emit/kernel.o').exists()
