"""Distinct exact source uses may share only identical generated primitive code."""
from pathlib import Path
import runpy
import hashlib
import json
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


@pytest.mark.skipif(not (LLVM/'clang').is_file(),reason='real target compiler unavailable')
def test_normal_catalog_links_one_physical_body_for_two_exact_selections(tmp_path,monkeypatch):
    from mlir_oot.golden_device_catalog import compile_catalog
    text=source(7,35,69)
    start=text.index('%result = linalg.generic')
    stop=text.index('func.return %result')
    second=text[start:stop].replace('%result','%other').replace('"operation"','"other_region"')
    text=text[:stop]+second+text[stop:]
    path=tmp_path/'model.mlir';path.write_text(text)
    def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    def price(selection,export_path,timing_path,engine_sha):
        exported=json.loads(export_path.read_text())
        assert exported['source_sha256']==selection['source_sha256']
        return dict(object_sha256=exported['compilation']['object_sha256'],
            timing_sha256=pin(timing_path)['sha256'],fixture={'mock':'unit-test-only fixture'},
            cycles=json.loads(timing_path.read_text())['test_only_cycles'])
    monkeypatch.setattr('mlir_oot.golden_calibrated_plan._measurement',price)
    regions=[]
    for region in ('operation','other_region'):
        candidates=[]
        for label,options,cost in [('control',{},31),
            ('resident',dict(dense_input_policy='resident_a_prefetch'),23)]:
            directory=tmp_path/region/label
            export_contraction(path,region,LLVM,directory,**options)
            timing=directory/'test_only_timing.json'
            timing.write_text(json.dumps(dict(test_only_cycles=cost)))
            candidates.append(dict(export=pin(directory/'golden_export.json'),timing=pin(timing)))
        calibration=tmp_path/region/'calibration.json'
        calibration.write_text(json.dumps(dict(schema='golden_contraction_calibrations_v1',
            gsim_engine=pin(LLVM/'clang'),candidates=candidates)))
        regions.append(dict(region=region,calibration=pin(calibration)))
    packet=tmp_path/'model_calibrations.json'
    packet.write_text(json.dumps(dict(schema='golden_model_contraction_calibrations_v1',
        source_sha256=pin(path)['sha256'],regions=regions)))
    result=compile_catalog(path,LLVM,tmp_path/'catalog',contraction_calibrations=packet)
    assert result['covered_contractions']==2 and result['unique_kernels']==1
    assert len(result['calibrated_contraction_selections'])==2
    assert result['bindings'][0]['symbol']==result['bindings'][1]['symbol']
    assert result['compilation']['uncalibrated_compilation'] is None
    assert len(result['compilation']['linker_argv'])==5  # linker, -r, one object, -o, output
