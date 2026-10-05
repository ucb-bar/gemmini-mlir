"""Calibrated choice must select emitted code; unknown roofline stays unknown."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest

from mlir_oot.golden_compiler_export import export_contraction, select_contraction_export
from mlir_oot.golden_calibrated_plan import optimize_contraction, _measurement


ROOT = Path(__file__).resolve().parents[1]
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
source = runpy.run_path(str(Path(__file__).with_name('test_golden_compiler_export.py')))['source']


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def choices(tmp_path, monkeypatch):
    """Real compiler alternatives; test prices are explicit mock evidence only."""
    if not (LLVM/'clang').is_file():
        pytest.skip('real target compiler unavailable')
    input_path = tmp_path/'source.mlir';input_path.write_text(source(17,73,65))
    exports, timings, candidates = [], [], []
    for label, options, price in [('control',{},31),
            ('resident',dict(dense_input_policy='resident_a_command_cost'),23)]:
        export_contraction(input_path,'operation',LLVM,tmp_path/label,**options)
        exported = tmp_path/label/'golden_export.json';timing = tmp_path/(label+'.json')
        timing.write_text(json.dumps(dict(test_only_cycles=price)))
        exports.append(exported);timings.append(timing)
        candidates.append(dict(export=pin(exported),timing=pin(timing)))
    calibration = tmp_path/'calibration.json'
    record = dict(schema='golden_contraction_calibrations_v1',
                  gsim_engine=pin(LLVM/'clang'),candidates=candidates)
    calibration.write_text(json.dumps(record))

    def test_price(selection, export_path, timing_path, engine_sha):
        exported = json.loads(export_path.read_text())
        assert exported['source_sha256']==selection['source_sha256']
        return dict(object_sha256=exported['compilation']['object_sha256'],
            timing_sha256=pin(timing_path)['sha256'],fixture={'mock':'not hardware evidence'},
            cycles=json.loads(timing_path.read_text())['test_only_cycles'])

    monkeypatch.setattr('mlir_oot.golden_calibrated_plan._measurement',test_price)
    return input_path, calibration, exports, timings, record


def test_solver_choice_reaches_real_emitted_object_and_cost_scope(choices,tmp_path):
    input_path, calibration, exports, timings, record = choices
    chosen = optimize_contraction(input_path,'operation',LLVM,tmp_path/'chosen',calibration)
    resident = json.loads(exports[1].read_text())
    assert chosen['shared_solver_selected'] and chosen['selected_plan_controls_emitted_code']
    assert chosen['kernel_symbol']==resident['kernel_symbol']
    assert chosen['compilation']['object_sha256']==resident['compilation']['object_sha256']
    assert chosen['solver']['complete_plans']==2 and chosen['calibrated_enumeration_complete']
    assert not chosen['roofline_attainment_resolved']
    assert chosen['solver']['physical_floor']['cycles'] is None
    assert chosen['solver']['legal_floor']['cycles'] is None
    assert chosen['solver']['occupancy']['compute_resources']==[]
    assert 'UNKNOWN' in chosen['occupancy_scope']
    assert not chosen['whole_model_correctness_verified'] and not chosen['full_model_hardware_measured']
    assert chosen['emitted_dispatch']['nodes'][0]['op']==resident['kernel_symbol']
    timings[0].write_text(json.dumps(dict(test_only_cycles=19)))
    record['candidates'][0]['timing']=pin(timings[0]);calibration.write_text(json.dumps(record))
    other = optimize_contraction(input_path,'operation',LLVM,tmp_path/'other',calibration)
    assert other['kernel_symbol']==json.loads(exports[0].read_text())['kernel_symbol']
    assert other['compilation']['object_sha256']!=chosen['compilation']['object_sha256']


def test_calibration_pin_duplicate_and_source_mutation_refuse(choices,tmp_path):
    input_path, calibration, exports, timings, record = choices
    timings[0].write_text('changed')
    with pytest.raises(ValueError,match='hash disagrees'):
        optimize_contraction(input_path,'operation',LLVM,tmp_path/'badpin',calibration)
    timings[0].write_text(json.dumps(dict(test_only_cycles=31)))
    record['candidates']=[record['candidates'][0]]*2;calibration.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='duplicate compiler alternative'):
        optimize_contraction(input_path,'operation',LLVM,tmp_path/'duplicate',calibration)
    input_path.write_text(source(17,73,66))
    with pytest.raises(ValueError,match='current source bytes'):
        optimize_contraction(input_path,'operation',LLVM,tmp_path/'changedsource',calibration)


def test_unpriced_candidate_set_refuses_before_source_or_compiler_access(tmp_path):
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(schema='golden_contraction_calibrations_v1',candidates=[])))
    with pytest.raises(ValueError,match='at least two explicit calibrated'):
        optimize_contraction(tmp_path/'missing','operation',LLVM,tmp_path/'output',calibration)
    assert not (tmp_path/'output').exists()


@pytest.mark.skipif(not (ROOT/'out/calibrated_source_plan/calibration.json').is_file(),
                    reason='owned actual GSIM qualification fixture unavailable')
def test_actual_measurement_rejects_wrong_source_geometry_and_duration(tmp_path):
    # Replay existing completed measured artifacts; do not claim new test prices.
    directory = ROOT/'out/calibrated_source_plan'
    selection = select_contraction_export(directory/'source.mlir','operation')
    export_path = directory/'control/golden_export.json'
    timing_path = directory/'control_probe/result.json'
    engine_sha = json.loads(timing_path.read_text())['gsim_engine_sha256']
    result = _measurement(selection,export_path,timing_path,engine_sha)
    assert result['cycles']==3234
    wrong = dict(selection,dimensions=dict(batch=1,m=18,n=73,k=65))
    with pytest.raises(ValueError,match='current source selection: dimensions'):
        _measurement(wrong,export_path,timing_path,engine_sha)
    record = json.loads(timing_path.read_text());record['kernel_cycles']=float('nan')
    changed = tmp_path/'result.json';changed.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='finite positive'):
        _measurement(selection,export_path,changed,engine_sha)
