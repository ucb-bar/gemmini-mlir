"""Complete prepared-model preservation and actual catalog closure checks."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml

from merlin.llvmlower.device_build import DeviceRouting
from mlir_oot.golden_device_catalog import merlin_builder
from mlir_oot.golden_model_plan import bind_device_routing


ROOT=Path(__file__).resolve().parents[1]
LLVM=Path(os.environ.get('GEMMINI_EXPORT_TEST_LLVM_BIN',
    '/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin'))
source=runpy.run_path(str(Path(__file__).with_name('test_golden_compiler_export.py')))['source']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_normal_model_command_reaches_enabled_plan_binding(tmp_path):
    manifest=yaml.safe_load((ROOT/'manifest.yaml').read_text())
    assert 'mlir_oot/golden_model_plan.py' in manifest['components']['build_golden_model']
    assert 'tests/fused_whole_model_probe.py' in manifest['components']['build_golden_model']
    assert 'mlir_oot/golden_model_plan.py' not in manifest['components']['emit_command_buffer']
    completed=subprocess.run([sys.executable,str(ROOT/'gemmini-model-build'),'--help'],
                             capture_output=True,text=True,check=True)
    assert '--work' in completed.stdout and '--validate-existing' in completed.stdout


@pytest.mark.skipif(not (LLVM/'clang').is_file(),reason='real target compiler unavailable')
def test_complete_integer_source_plan_drives_unchanged_normal_catalog_path(tmp_path):
    input_path=tmp_path/'model.mlir';input_path.write_text(source(17,73,65))
    before=input_path.read_bytes()
    original=DeviceRouting('gemmini',ROOT,'i8','i32',catalog_builder=merlin_builder(LLVM))
    routing,binding=bind_device_routing(original,original_source=input_path)
    prepared=routing.prepared_transform(input_path,tmp_path/'prepared')
    assert prepared==input_path and input_path.read_bytes()==before
    state=binding.state
    assert len(state['logical_dispatch']['nodes'])>1
    assert len(state['global_plan']['selected'])==len(state['logical_dispatch']['nodes'])
    assert state['prepared_ir_operation_cover_complete']
    assert state['identity_plan_admission_gate']
    assert not state['selected_plan_controls_emission']
    assert state['outlined_preservation_proof']['computation']=='expanded_driver_structurally_equivalent'
    manifest,obj=routing.catalog_builder(prepared,tmp_path/'catalog')
    assert obj.is_file() and state['catalog']['object_sha256']==sha(obj)
    assert state['catalog']['dense_bindings'][0]['dimensions']==dict(batch=1,m=17,n=73,k=65)
    assert input_path.read_bytes()==before and state['catalog']['binary_symbol_closure_complete']
    assert not state['shared_solver_selected'] and state['cycles_status']=='UNKNOWN'
    assert not state['whole_model_correctness_verified']
    assert 'UNKNOWN' in state['task_to_machine_instruction_accounting']
    assert original.catalog_builder is not routing.catalog_builder
    input_path.write_text(source(17,73,66))
    with pytest.raises(ValueError,match='verified prepared graph'):
        routing.catalog_builder(input_path,tmp_path/'different')


EXTERNAL='''module {
  func.func private @external(%a: tensor<4xi8>) -> tensor<4xi8> attributes {llvm.emit_c_interface}
  func.func @forward(%a: tensor<4xi8>) -> tensor<4xi8> {
    %zero = arith.constant 0 : i32
    %result = func.call @external(%a) : (tensor<4xi8>) -> tensor<4xi8>
    func.return %result : tensor<4xi8>
  }
}'''


def external_catalog(tmp_path,*,symbol='_mlir_ciface_external',routes=True):
    input_path=tmp_path/'model.mlir';input_path.write_text(EXTERNAL)
    c=tmp_path/'opaque.c';c.write_text(f'void {symbol}(void) {{}}\n')
    obj=tmp_path/'opaque.o'
    subprocess.run([str(LLVM/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(c),'-o',str(obj)],
                   check=True,capture_output=True)
    manifest=tmp_path/'catalog.json'
    # Structural closure fixture only; no kernel arithmetic or ABI execution claim.
    record=dict(source_sha256=sha(input_path),compilation=dict(object_sha256=sha(obj)),
        coverage_complete=True,matched_contractions=0,covered_contractions=0,kernels=[],bindings=[],
        fused_requantizations=[dict(symbol='external',proof='opaque fixture; numeric equivalence unproven')] if routes else [])
    manifest.write_text(json.dumps(record))
    routing,binding=bind_device_routing(DeviceRouting('gemmini',ROOT,'i8','i32',
        catalog_builder=lambda source,work:(manifest,obj)))
    routing.prepared_transform(input_path,tmp_path/'prepared')
    return input_path,routing,binding


@pytest.mark.skipif(not (LLVM/'clang').is_file(),reason='real target compiler unavailable')
def test_declared_external_abi_is_preserved_and_actual_ciface_symbol_closed(tmp_path):
    input_path,routing,binding=external_catalog(tmp_path)
    assert binding.state['called_external_symbols']=={'external':1}
    assert binding.state['declared_external_entrypoints']=={'external':'_mlir_ciface_external'}
    routing.catalog_builder(input_path,tmp_path/'catalog')
    row=binding.state['catalog']['external_routes']['external']
    assert row['binary_entrypoint']=='_mlir_ciface_external'
    assert binding.state['catalog']['binary_symbol_closure_complete']
    assert not binding.state['whole_model_correctness_verified']


@pytest.mark.parametrize('kwargs,message',[
    ({'symbol':'unrelated'},'unique declared definitions'),
    ({'routes':False},'external catalog coverage'),
])
@pytest.mark.skipif(not (LLVM/'clang').is_file(),reason='real target compiler unavailable')
def test_missing_external_binding_or_actual_binary_definition_refuses(tmp_path,kwargs,message):
    input_path,routing,binding=external_catalog(tmp_path,**kwargs)
    with pytest.raises(ValueError,match=message):
        routing.catalog_builder(input_path,tmp_path/'catalog')
    assert 'catalog' not in binding.state


def test_no_catalog_cannot_claim_whole_model_plan_or_artifact_closure():
    with pytest.raises(ValueError,match='source-bound catalog builder'):
        bind_device_routing(DeviceRouting('gemmini',ROOT,'i8','i32'))
