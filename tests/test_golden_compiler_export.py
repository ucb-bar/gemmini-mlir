"""Actual command ownership and selected-plan emission, without invented costs."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mlir_oot.gemmini_opt import main
from mlir_oot.golden_compiler_export import export_contraction,export_inventory,export_capture


PACKAGE=Path(__file__).resolve().parents[1]
LLVM_BIN=Path(os.environ.get('GEMMINI_EXPORT_TEST_LLVM_BIN',
    '/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin'))


def source(m=17,n=73,k=65):
    return f'''module {{
  func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<{m}x{n}xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %result = linalg.generic {{indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
      outs(%init : tensor<{m}x{n}xi32>) attrs = {{prov.region_id = "operation"}} {{
      ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
        %la = arith.extsi %lhs : i8 to i32
        %rb = arith.extsi %rhs : i8 to i32
        %product = arith.muli %la, %rb : i32
        %sum = arith.addi %acc, %product : i32
        linalg.yield %sum : i32
    }} -> tensor<{m}x{n}xi32>
    func.return %result : tensor<{m}x{n}xi32>
  }}
}}'''


def test_inventory_uses_actual_new_commands_and_existing_contract_checks():
    from merlin.targetgen.contract.schemas import validate
    from merlin.perf.phase2_edit_contract import validate_against_package
    manifest=yaml.safe_load((PACKAGE/'manifest.yaml').read_text())
    validate(manifest,'manifest')
    result=export_inventory(PACKAGE)
    inventory=result['package_inventory']
    assert not inventory['missing']
    assert len(inventory['surfaces'])==6
    symbol=next(row for row in inventory['symbols']
        if row['path']=='mlir_oot/golden_gemm.py' and row['symbol']=='GoldenGemm._output_block')
    assert set(symbol['commands'])=={'export_golden_capture','export_golden_contraction'}
    assert 'emit_command_buffer' not in symbol['commands']
    contract=result['compiler_edit_contract']
    assert len(contract['required_decisions'])==len(inventory['surfaces'])
    validate_against_package(contract,PACKAGE)
    assert not result['shared_solver_selected']


def test_inventory_command_runs_from_manifest(tmp_path):
    manifest=yaml.safe_load((PACKAGE/'manifest.yaml').read_text())
    receipt=tmp_path/'inventory.json'
    argv=[part.format(tool=str(PACKAGE/'gemmini-opt'),output_json=receipt)
          for part in manifest['commands']['export_golden_inventory']['argv']]
    completed=subprocess.run([sys.executable,*argv],capture_output=True,text=True,check=True)
    assert json.loads(completed.stdout)['surfaces']==6
    assert json.loads(receipt.read_text())['compiler_edit_contract']['sha256']


@pytest.mark.parametrize('args',[
    ['--dense-b-slot-policy','remaining_rows'],
    ['--export-golden-contraction','--resident-stripes'],
    ['--export-golden-capture','--region','ignored'],
    ['--emit-golden-inventory','out.json','--prefetch-b'],
    ['--export-golden-contraction','--emit-target-artifact'],
])
def test_incompatible_or_inert_cli_options_refuse(args):
    with pytest.raises(SystemExit) as failure:main(args)
    assert failure.value.code==2


@pytest.mark.skipif(not (LLVM_BIN/'mlir-translate').is_file(),reason='real LLVM compiler unavailable')
def test_manifest_compile_command_emits_selected_symbol_and_boundaries(tmp_path):
    manifest=yaml.safe_load((PACKAGE/'manifest.yaml').read_text())
    input_path=tmp_path/'source.mlir';input_path.write_text(source())
    before=input_path.read_bytes()
    workdir=tmp_path/'compiled'
    values=dict(tool=str(PACKAGE/'gemmini-opt'),input_mlir=input_path,
                region_id='operation',llvm_bin=LLVM_BIN,output_dir=workdir)
    argv=[part.format(**values) for part in manifest['commands']['export_golden_contraction']['argv']]
    completed=subprocess.run([sys.executable,*argv],capture_output=True,text=True,check=True)
    result=json.loads((workdir/'golden_export.json').read_text())
    assert json.loads(completed.stdout)['selected_plan_controls_emitted_code']
    assert result['source_sha256']==hashlib.sha256(before).hexdigest()
    assert input_path.read_bytes()==before
    assert result['dimensions']==dict(batch=1,m=17,n=73,k=65)
    symbol=result['kernel_symbol']
    assert result['global_plan']['selected'][0]['implementation']==symbol
    assert result['emitted_dispatch']['nodes'][0]['op']==symbol
    assert f'@{symbol}(' in (workdir/'kernel.ll').read_text()
    assert result['compilation']['object_nofsm_status']=='pass'
    assert result['compilation']['object_sha256']==hashlib.sha256((workdir/'kernel.o').read_bytes()).hexdigest()
    assert len(result['global_plan_emission']['boundaries'])==3
    assert result['occupancy_status']=='unknown' and result['activity'] is None
    assert result['global_plan']['cycles']['lo'] is None
    assert not result['shared_solver_selected']
    assert not result['whole_model_memory_bound'] and not result['whole_model_correctness_verified']


@pytest.mark.skipif(not (LLVM_BIN/'mlir-translate').is_file(),reason='real LLVM compiler unavailable')
def test_selected_schedule_changes_object_and_shared_emission_rejects_mutation(tmp_path,monkeypatch):
    import mlir_oot.golden_compiler_export as export
    from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
    input_path=tmp_path/'source.mlir';input_path.write_text(source(123,73,1041))
    control=export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'control')
    captured={}
    # Capture the real adapter at its compiler boundary, without replacing compilation.
    original=export.SourceContractionEmitter.emit_global_plan
    def record(self,program,plan):
        captured.update(emitter=self,program=program,plan=plan)
        return original(self,program,plan)
    monkeypatch.setattr(export.SourceContractionEmitter,'emit_global_plan',record)
    selected=export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'selected',
        dense_input_policy='resident_a_command_cost',dense_b_slot_policy='remaining_rows')
    assert selected['b_slot_decision']['applied']
    assert selected['kernel_symbol']!=control['kernel_symbol']
    assert selected['compilation']['object_sha256']!=control['compilation']['object_sha256']
    emitter,program,plan=(captured[key]for key in ('emitter','program','plan'))
    changed=replace(plan,selected=(replace(plan.selected[0],implementation='other_kernel'),))
    with pytest.raises(ValueError,match='exact bound singleton'):
        emitter.emit_global_plan(program,changed)
    emitter.generator.shape=replace(emitter.generator.shape,n=74)
    with pytest.raises(ValueError,match='schedule changed'):
        emitter.emit_global_plan(program,plan)
    input_path.write_text(source(123,73,1042))
    with pytest.raises(ValueError,match='binding changed'):
        emit_global_plan(program,plan,emitter)


@pytest.mark.skipif(not (LLVM_BIN/'mlir-translate').is_file(),reason='real LLVM compiler unavailable')
def test_activity_uses_shared_timeline_and_requires_exact_object_identity(tmp_path):
    from merlin.perf.activity_schedule import ActivityEvent,schedule_activity
    input_path=tmp_path/'source.mlir';input_path.write_text(source())
    base=export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'base')
    timeline=schedule_activity([ActivityEvent('compute','mesh','compute',100,
                               provenance='independent test duration, not hardware calibration')])
    with pytest.raises(ValueError,match='not bound'):
        export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'wrong',
            activity_timeline=timeline,activity_artifact_sha256='0'*64)
    bound=export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'bound',
        activity_timeline=timeline,activity_artifact_sha256=base['compilation']['object_sha256'])
    assert bound['occupancy_status']=='caller_supplied'
    assert bound['activity']['timeline']['total_cycles']==100
    assert bound['global_plan']['cycles']['lo'] is None
    assert not bound['shared_solver_selected']


def test_capture_export_delegates_existing_binder_without_whole_plan_claim(tmp_path,monkeypatch):
    import mlir_oot.captured_requant_bundle as binder
    called={}
    def build(capture,llvm,output,**options):
        called.update(capture=capture,llvm=llvm,output=output,options=options)
        return dict(routes=['source-proven-route'])
    monkeypatch.setattr(binder,'build',build)
    result=export_capture(tmp_path/'capture/model.mlir',LLVM_BIN,tmp_path/'output',resident_stripes=True)
    assert called['capture']==tmp_path/'capture' and called['options']['resident_stripes']
    assert not result['whole_model_plan_emitted'] and not result['shared_solver_selected']
    with pytest.raises(ValueError,match='model.mlir'):
        export_capture(tmp_path/'different.mlir',LLVM_BIN,tmp_path/'output')


def test_false_integer_semantics_refuse_before_compilation(tmp_path):
    input_path=tmp_path/'source.mlir'
    input_path.write_text(source().replace('arith.constant 0','arith.constant 1'))
    with pytest.raises(ValueError,match='found 0'):
        export_contraction(input_path,'operation',LLVM_BIN,tmp_path/'result')
    assert not (tmp_path/'result').exists()
