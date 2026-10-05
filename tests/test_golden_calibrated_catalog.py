"""Measured alternatives must enter the ordinary source-bound model catalog."""

import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import pytest

from merlin.llvmlower.device_build import DeviceRouting
from mlir_oot.golden_device_catalog import compile_catalog, merlin_builder
from mlir_oot.golden_model_plan import bind_device_routing


LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
ROOT = Path(__file__).resolve().parents[1]
source = runpy.run_path(str(Path(__file__).with_name('test_golden_compiler_export.py')))['source']
# Reuse real compiled alternatives and explicitly test-only mocked fixture prices.
choices = runpy.run_path(str(Path(__file__).with_name('test_golden_calibrated_plan.py')))['choices']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet(path, input_path, calibration, regions=('operation',)):
    path.write_text(json.dumps(dict(schema='golden_model_contraction_calibrations_v1',
        source_sha256=sha(input_path), regions=[dict(region=region,
            calibration=dict(path=str(calibration), sha256=sha(calibration))) for region in regions])))
    return path


@pytest.mark.skipif(not (LLVM/'clang').is_file(), reason='real target compiler unavailable')
def test_default_catalog_object_is_unchanged_when_opt_in_absent(tmp_path):
    input_path = tmp_path/'model.mlir';input_path.write_text(source())
    left = compile_catalog(input_path, LLVM, tmp_path/'left')
    right = compile_catalog(input_path, LLVM, tmp_path/'right', contraction_calibrations=None)
    assert left['compilation']['object_sha256']==right['compilation']['object_sha256']
    assert left['bindings']==right['bindings'] and left['kernels']==right['kernels']
    assert 'calibrated_contraction_selections' not in left


def test_shared_solver_winner_is_linked_and_closed_by_normal_routing(choices,tmp_path):
    input_path, calibration, exports, timings, record = choices
    calibration_packet = packet(tmp_path/'packet.json',input_path,calibration)
    routing, binding = bind_device_routing(DeviceRouting('gemmini',ROOT,'i8','i32',
        catalog_builder=merlin_builder(LLVM,contraction_calibrations=calibration_packet)))
    before = input_path.read_bytes()
    prepared = routing.prepared_transform(input_path,tmp_path/'prepare')
    manifest_path,obj = routing.catalog_builder(prepared,tmp_path/'catalog')
    catalog = json.loads(manifest_path.read_text())
    resident = json.loads(exports[1].read_text())
    assert catalog['bindings'][0]['symbol']==resident['kernel_symbol']
    assert catalog['kernels'][0]['schedule']==resident['schedule']
    assert catalog['compilation']['uncalibrated_compilation'] is None
    assert catalog['calibrated_contraction_selections'][0]['object_sha256']==resident['compilation']['object_sha256']
    assert catalog['selection_controls_emitted_device_code']
    assert catalog['whole_model_cycles_status']=='UNKNOWN' and not catalog['whole_model_shared_solver_selected']
    assert binding.state['catalog']['binary_symbol_closure_complete']
    assert binding.state['catalog']['selection_controls_emitted_device_code']
    assert not binding.state['selected_plan_controls_emission'] and not binding.state['shared_solver_selected']
    assert input_path.read_bytes()==before and (tmp_path/'catalog/catalog_source.mlir').read_bytes()==before
    # Change only explicit test prices: normal catalog's actual linked symbol and object change.
    timings[0].write_text(json.dumps(dict(test_only_cycles=19)))
    record['candidates'][0]['timing']['sha256']=sha(timings[0]);calibration.write_text(json.dumps(record))
    packet(calibration_packet,input_path,calibration)
    other = compile_catalog(input_path,LLVM,tmp_path/'other',contraction_calibrations=calibration_packet)
    assert other['bindings'][0]['symbol']==json.loads(exports[0].read_text())['kernel_symbol']
    assert sha(obj)!=other['compilation']['object_sha256']


def test_model_calibration_refuses_inert_changed_ambiguous_and_missing_regions(choices,tmp_path):
    input_path, calibration, *_ = choices
    path = packet(tmp_path/'packet.json',input_path,calibration,regions=('missing',))
    with pytest.raises(ValueError,match='exactly one covered'):
        compile_catalog(input_path,LLVM,tmp_path/'missing',contraction_calibrations=path)
    packet(path,input_path,calibration,regions=('operation','operation'))
    with pytest.raises(ValueError,match='duplicate calibrated model region'):
        compile_catalog(input_path,LLVM,tmp_path/'duplicate',contraction_calibrations=path)
    packet(path,input_path,calibration)
    calibration.write_text('changed')
    with pytest.raises(ValueError,match='hash disagrees'):
        compile_catalog(input_path,LLVM,tmp_path/'changedcalibration',contraction_calibrations=path)
    input_path.write_text(source(17,73,66))
    with pytest.raises(ValueError,match='exact prepared source'):
        compile_catalog(input_path,LLVM,tmp_path/'changedsource',contraction_calibrations=path)


def test_shared_default_kernel_retained_for_unselected_source_use(choices,tmp_path):
    input_path, calibration, *_ = choices
    # The changed entire source needs its own exports; choices' prices remain explicitly test-only.
    from mlir_oot.golden_compiler_export import export_contraction
    text=input_path.read_text().replace('func.return %result', '''%other = linalg.matmul
      ins(%a, %b : tensor<17x65xi8>, tensor<65x73xi8>)
      outs(%init : tensor<17x73xi32>) -> tensor<17x73xi32>
    func.return %result''')
    input_path.write_text(text)
    record=json.loads(calibration.read_text())
    for index,options in enumerate(({},dict(dense_input_policy='resident_a_command_cost'))):
        directory=tmp_path/('two_'+str(index))
        export_contraction(input_path,'operation',LLVM,directory,**options)
        exported=directory/'golden_export.json'
        record['candidates'][index]['export']=dict(path=str(exported),sha256=sha(exported))
    calibration.write_text(json.dumps(record))
    path=packet(tmp_path/'packet.json',input_path,calibration)
    control=compile_catalog(input_path,LLVM,tmp_path/'control')
    assert control['unique_kernels']==1 and control['covered_contractions']==2
    selected=compile_catalog(input_path,LLVM,tmp_path/'selected',contraction_calibrations=path)
    assert selected['coverage_complete'] and selected['covered_contractions']==2 and selected['unique_kernels']==2
    assert selected['bindings'][1]==control['bindings'][1]
    assert selected['compilation']['uncalibrated_compilation'] is not None
    defined=subprocess.run([str(LLVM/'llvm-nm'),'--defined-only',str(tmp_path/'selected/kernel.o')],
                           check=True,capture_output=True,text=True).stdout
    for row in selected['kernels']:
        assert row['symbol'] in defined


def test_normal_model_command_exposes_calibration_and_refuses_validation_only():
    import sys
    result=subprocess.run([sys.executable,str(ROOT/'gemmini-model-build'),'--help'],capture_output=True,text=True,check=True)
    assert '--contraction-calibrations' in result.stdout
    result=subprocess.run([sys.executable,str(ROOT/'gemmini-model-build'),'missing','missing',
        '--work','/tmp/gemmini_calibration_cli_refusal','--llvm-bin',str(LLVM),'--spike','missing',
        '--validate-existing','--contraction-calibrations','missing'],capture_output=True,text=True)
    assert result.returncode==2 and 'require a fresh catalog build' in result.stderr
