"""Bind measured target alternatives to the existing shared plan selector.

The complete optimization unit here is one source contraction. Prices describe
one pinned GSIM fixture, not a whole model or accelerator engine occupancy.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re

from .golden_compiler_export import SourceContractionEmitter, select_contraction_export
from .no_fsm_audit import audit_elf


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pinned(base, record):
    if set(record) != {'path', 'sha256'}:
        raise ValueError('calibration artifact requires exact path and sha256 fields')
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    if _sha(path) != record['sha256']:
        raise ValueError('calibration artifact hash disagrees with its pin')
    return path.resolve()


def _measurement(selection, export_path, timing_path, engine_sha):
    exported = json.loads(export_path.read_text())
    measured = json.loads(timing_path.read_text())
    generator = selection['generator']
    shape = asdict(generator.shape)
    for name in ('source_sha256', 'dimensions', 'binding', 'schedule_kind', 'kernel_symbol'):
        if exported[name] != selection[name]:
            raise ValueError(f'calibration export disagrees with current source selection: {name}')
    if exported['schedule'] != shape or exported['prefetch_b_rows'] != (
            list(generator.prefetch_b_rows) if generator.prefetch_b_rows is not None else None):
        raise ValueError('calibration export schedule or slot placement differs')
    compilation = exported['compilation']
    obj = export_path.parent / 'kernel.o'
    if _sha(obj) != compilation['object_sha256']:
        raise ValueError('calibrated target object differs from its export')
    if measured['schema'] != 'gemmini_golden_gemm_gsim_probe_v1' or measured['status'] != 'pass':
        raise ValueError('calibration requires a successful full-output GSIM probe')
    if not measured['embedded_expected'] or not measured['static_inputs']:
        raise ValueError('calibration requires pinned static inputs and all expected outputs')
    if (measured['shape'] != [shape['m'], shape['n'], shape['k']]
            or measured['block'] != [shape['bm'], shape['bn']]
            or measured['kernel_symbol'] != selection['kernel_symbol']):
        raise ValueError('timing probe shape, block or symbol differs from source export')
    for name in shape.keys() - {'m', 'n', 'k', 'bm', 'bn'}:
        if measured[name] != shape[name]:
            raise ValueError(f'timing probe schedule differs from source export: {name}')
    if measured['prefetch_b_rows'] != exported['prefetch_b_rows']:
        raise ValueError('timing probe B slots differ from source export')
    if (measured['target_ir_sha256'] != compilation['target_ir_sha256']
            or measured['llvm_ir_sha256'] != compilation['llvm_mlir_sha256']
            or measured['gsim_engine_sha256'] != engine_sha):
        raise ValueError('timing probe compiler/engine identity differs')
    cycles = measured['kernel_cycles']
    if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or not math.isfinite(cycles) or cycles <= 0:
        raise ValueError('calibration requires a finite positive measured duration')
    elf = timing_path.parent / 'layer.elf'
    if _sha(elf) != measured['elf_sha256'] or audit_elf(elf.read_bytes())['status'] != 'pass':
        raise ValueError('calibrated linked ELF identity or no-FSM audit differs')
    stdout = timing_path.parent / 'gsim.stdout'
    output = stdout.read_text()
    if (re.findall(r'^GOLDEN_GEMM_CYCLES (\d+)$', output, re.MULTILINE) != [str(cycles)]
            or f"GOLDEN_GEMM PASS M={shape['m']} N={shape['n']} K={shape['k']}" not in output):
        raise ValueError('calibrated raw UART lacks its exact duration and numeric marker')
    fixture = {name: _sha(timing_path.parent / name)
               for name in ('a.bin', 'b.bin', 'gemm_expected.inc')}
    return dict(export_path=str(export_path), export_sha256=_sha(export_path),
        timing_path=str(timing_path), timing_sha256=_sha(timing_path),
        object_sha256=compilation['object_sha256'], elf_sha256=measured['elf_sha256'],
        raw_stdout_sha256=_sha(stdout), fixture=fixture, cycles=cycles)


def optimize_contraction(source_path, region_id, llvm_bin, workdir, calibration_path):
    """Select calibrated legal alternatives and compile the actual solver winner."""
    from merlin.perf.activity_schedule import ActivityEvent, schedule_activity
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.explicit_plan_adapter import ExplicitPlanningAdapter
    from merlin.perf.global_planner import Bound, GlobalPlanPolicy, optimize_program
    from merlin.perf.headroom import Composition
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, ResourceOccupancy
    from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan, dispatch_digest

    source_path, workdir, calibration_path = map(Path, (source_path, workdir, calibration_path))
    calibration_sha = _sha(calibration_path)
    record = json.loads(calibration_path.read_text())
    if record['schema'] != 'golden_contraction_calibrations_v1' or len(record['candidates']) < 2:
        raise ValueError('selection requires at least two explicit calibrated contraction alternatives')
    engine = _pinned(calibration_path.parent, record['gsim_engine'])
    alternatives, selections, evidence = [], {}, {}
    logical, fixture = None, None
    for candidate in record['candidates']:
        exported_path = _pinned(calibration_path.parent, candidate['export'])
        timing_path = _pinned(calibration_path.parent, candidate['timing'])
        exported = json.loads(exported_path.read_text())
        options = exported['compiler_options']
        if (set(options) != {'large_n', 'prefetch_b', 'dense_input_policy', 'dense_b_slot_policy'}
                or type(options['large_n']) is not bool or type(options['prefetch_b']) is not bool):
            raise ValueError('calibration requires explicit supported compiler options')
        selection = select_contraction_export(source_path, region_id, **options)
        if exported['source_sha256'] != selection['source_sha256']:
            raise ValueError('calibration does not bind the current source bytes')
        measured = _measurement(selection, exported_path, timing_path, _sha(engine))
        if logical is None:
            logical, fixture = selection['logical'], measured['fixture']
        if dispatch_digest(selection['logical']) != dispatch_digest(logical) or measured['fixture'] != fixture:
            raise ValueError('alternatives must share one logical source and exact numeric fixture')
        alternative = selection['alternative']
        if alternative.id in selections:
            raise ValueError('duplicate compiler alternative in calibration set')
        provenance = f"gsim_fixture:{measured['timing_sha256']}:object:{measured['object_sha256']}"
        cycles = CycleInterval.point(measured['cycles'], provenance)
        alternative = replace(alternative, cycles=cycles,
            occupancy=(ResourceOccupancy('opaque_invocation_elapsed', cycles, provenance),))
        selections[alternative.id], evidence[alternative.id] = selection, measured
        alternatives.append(alternative)
    representations = {('lhs', 'input'): alternatives[0].inputs[0].representation,
        ('rhs', 'input'): alternatives[0].inputs[1].representation,
        ('output', 'output'): alternatives[0].outputs[0].representation}

    def events(program, selected, transitions, endpoint):
        if len(selected) != 1 or transitions:
            raise ValueError('calibrated singleton has no inter-region composition evidence')
        alternative = selected[0]
        return schedule_activity([ActivityEvent(alternative.id, 'opaque_invocation_elapsed',
            'opaque_invocation', getattr(alternative.cycles, endpoint),
            provenance=alternative.cycles.provenance[0])])

    adapter = ExplicitPlanningAdapter(tuple(alternatives), representations, (),
        {'opaque_invocation_elapsed': ResourceKind.OTHER}, Composition.SUM, 0,
        Bound.unknown('No full-source physical floor is derived from fixture duration'),
        event_builder=events, composition_provenance='Exactly one opaque measured invocation; no internal engine composition inferred')
    result = optimize_program(logical, adapter, policy=GlobalPlanPolicy(timeout_s=10, max_expanded_states=1000))
    if (result.plan is None or not result.exhausted or len(result.plan.selected) != 1
            or result.complete_plans != len(alternatives)):
        raise ValueError(f'calibrated source plan selection refused: {result.refusals}')
    winner = result.plan.selected[0]
    selection, selected_evidence = selections[winner.id], evidence[winner.id]
    emitter = SourceContractionEmitter(source_path, logical, winner, selection['generator'], llvm_bin, workdir)
    emission = emit_global_plan(logical, result.plan, emitter)
    if emitter.compilation['object_sha256'] != selected_evidence['object_sha256']:
        raise ValueError('solver-selected emitted object differs from its calibrated implementation')
    if _sha(calibration_path) != calibration_sha:
        raise ValueError('calibration changed during selection/emission')
    receipt = dict(schema='golden_calibrated_contraction_plan_v1', source_sha256=selection['source_sha256'],
        dimensions=selection['dimensions'], binding=selection['binding'],
        schedule=asdict(selection['generator'].shape), schedule_kind=selection['schedule_kind'],
        prefetch_b_rows=selection['generator'].prefetch_b_rows,
        calibration_path=str(calibration_path.resolve()), calibration_sha256=calibration_sha,
        gsim_engine_sha256=_sha(engine), alternatives=evidence,
        shared_solver_selected=True, selected_plan_controls_emitted_code=True,
        kernel_symbol=winner.implementation, compilation=emitter.compilation,
        solver=result.to_dict(), global_plan_emission=emission.receipt(),
        calibrated_enumeration_complete=True, roofline_attainment_resolved=result.resolved,
        emitted_dispatch=emission.dispatch.to_dict(),
        pricing_scope='One exact source contraction and pinned full-output GSIM fixture; no transfer to other source/input/model costs',
        occupancy_scope='Opaque invocation timeline only; hardware compute/DMA/issue occupancy UNKNOWN',
        physical_floor_status='UNKNOWN', whole_model_memory_bound=False,
        whole_model_correctness_verified=False, full_model_hardware_measured=False)
    (workdir / 'golden_optimized_plan.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
