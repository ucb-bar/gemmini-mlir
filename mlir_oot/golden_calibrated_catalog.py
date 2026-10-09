"""Compile measured source alternatives into the ordinary target catalog.

The shared planner owns measured ranking. This bridge owns source bindings and
target object composition; fixture prices are never summed into model costs.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from xdsl.dialects.builtin import StringAttr

from .golden_calibrated_plan import _pinned, optimize_contraction
from .golden_device_compile import compile_module
from .no_fsm_audit import audit_elf


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_calibrated_catalog(source_path, module, manifest, llvm_bin, workdir, packet_path):
    """Replace explicit exact bindings, retaining every unselected implementation."""
    source_path, llvm_bin, workdir, packet_path = map(Path,
        (source_path, llvm_bin, workdir, packet_path))
    source_sha, packet_sha = _sha(source_path), _sha(packet_path)
    packet = json.loads(packet_path.read_text())
    if (set(packet) != {'schema', 'source_sha256', 'regions'}
            or packet['schema'] != 'golden_model_contraction_calibrations_v1'
            or not isinstance(packet['regions'], list) or not packet['regions']):
        raise ValueError('model calibration requires explicit source pin and nonempty region list')
    if packet['source_sha256'] != source_sha or manifest['source_sha256'] != source_sha:
        raise ValueError('model calibration does not bind the exact prepared source bytes')
    if not manifest['coverage_complete']:
        raise ValueError('calibrated model catalog requires complete contraction coverage')
    records, regions = [], set()
    for row in packet['regions']:
        if set(row) != {'region', 'calibration'} or not isinstance(row['region'], str) or not row['region']:
            raise ValueError('model calibration requires exact region and calibration fields')
        if row['region'] in regions:
            raise ValueError('duplicate calibrated model region')
        regions.add(row['region'])
        bindings = [b for b in manifest['bindings'] if b['region'] == row['region']]
        if len(bindings) != 1:
            raise ValueError('calibrated region must bind exactly one covered source contraction')
        kernel = next(k for k in manifest['kernels'] if k['symbol'] == bindings[0]['symbol'])
        if kernel['batched']:
            raise ValueError('model calibration currently admits unbatched integer GEMM only')
        records.append((bindings[0], kernel, _pinned(packet_path.parent, row['calibration'])))

    workdir.mkdir(parents=True, exist_ok=True)
    selections, objects, replacements = [], {}, {}
    for binding, kernel, calibration in records:
        directory = workdir / ('calibrated_' + str(binding['source_operation_ordinal']))
        selection = optimize_contraction(source_path, binding['region'], llvm_bin, directory, calibration)
        exact = selection['binding']
        if (selection['source_sha256'] != source_sha
                or exact['source_operation_ordinal'] != binding['source_operation_ordinal']
                or exact['tensor_types'] != binding['tensor_types']
                or selection['dimensions'] != kernel['dimensions']):
            raise ValueError('selected object binding differs from the actual catalog operation')
        symbol, obj = selection['kernel_symbol'], directory / 'kernel.o'
        if _sha(obj) != selection['compilation']['object_sha256']:
            raise ValueError('selected catalog object changed after source-bound emission')
        if symbol in objects and _sha(objects[symbol]) != _sha(obj):
            raise ValueError('selected catalog symbol has conflicting object definitions')
        objects[symbol] = obj
        replacements[binding['source_operation_ordinal']] = symbol
        selections.append(dict(region=binding['region'],
            source_operation_ordinal=binding['source_operation_ordinal'],
            previous_symbol=binding['symbol'], selected_symbol=symbol,
            object_path=str(obj.resolve()), object_sha256=_sha(obj),
            selection_receipt=str((directory / 'golden_optimized_plan.json').resolve()),
            selection_receipt_sha256=_sha(directory / 'golden_optimized_plan.json'),
            selection=selection))

    bindings = [{**b, 'symbol': replacements.get(b['source_operation_ordinal'], b['symbol'])}
                for b in manifest['bindings']]
    retained = {b['symbol'] for b in bindings} - objects.keys()
    # A shared default kernel remains when any unselected source use needs it.
    for fn in list(module.body.block.ops):
        name = fn.sym_name.data
        if name not in retained and not any(name == symbol + '_core' for symbol in retained):
            module.body.block.erase_op(fn)
    kernels = [k for k in manifest['kernels'] if k['symbol'] in retained]
    for row in selections:
        selected = row['selection']
        if not any(k['symbol'] == row['selected_symbol'] for k in kernels):
            kernels.append(dict(symbol=row['selected_symbol'], dimensions=selected['dimensions'],
                schedule=selected['schedule'], batched=False,
                schedule_kind=selected['schedule_kind'], prefetch_b_rows=selected['prefetch_b_rows']))
    module.attributes['gemmini.golden_catalog'] = StringAttr(
        json.dumps(dict(kernels=len(retained), bindings=sum(b['symbol'] in retained for b in bindings)), sort_keys=True))
    module.verify()
    unselected = compile_module(module, llvm_bin, workdir / 'uncalibrated') if retained else None
    inputs = ([workdir / 'uncalibrated/kernel.o'] if retained else []) + list(objects.values())
    linker = llvm_bin.resolve(strict=True) / 'ld.lld'
    if not linker.is_file():
        found = shutil.which('ld.lld')
        if found is None:
            raise ValueError('ld.lld required for calibrated catalog composition')
        linker = Path(found)
    command = [str(linker), '-r', *map(str, inputs), '-o', str(workdir / 'kernel.o')]
    subprocess.run(command, check=True, capture_output=True)
    audit = audit_elf((workdir / 'kernel.o').read_bytes())
    if audit['status'] != 'pass':
        raise ValueError('calibrated model catalog contains forbidden Gemmini instructions')
    if _sha(packet_path) != packet_sha or _sha(source_path) != source_sha:
        raise ValueError('source or calibration packet changed during catalog emission')
    for row in selections:
        if (_sha(row['object_path']) != row['object_sha256']
                or _sha(row['selection_receipt']) != row['selection_receipt_sha256']):
            raise ValueError('selected implementation changed during catalog composition')
    for row in packet['regions']:
        _pinned(packet_path.parent, row['calibration'])
    if unselected is not None and _sha(workdir / 'uncalibrated/kernel.o') != unselected['object_sha256']:
        raise ValueError('unselected implementation changed during catalog composition')
    receipt = dict(schema='gemmini_calibrated_catalog_compile_v1',
        object_sha256=_sha(workdir / 'kernel.o'), object_nofsm_status=audit['status'],
        uncalibrated_compilation=unselected,
        linker_argv=command, linker_sha256=_sha(linker))
    (workdir / 'object_nofsm_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (workdir / 'device_compile.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {**manifest, 'bindings': bindings, 'kernels': kernels, 'unique_kernels': len(kernels),
        'prefetch_b_kernels': sum(k['schedule']['prefetch_b'] for k in kernels),
        'calibrated_contraction_selections': selections,
        'model_calibration_packet': dict(path=str(packet_path.resolve()), sha256=packet_sha),
        'selection_controls_emitted_device_code': True,
        'selection_scope': 'Only explicit exact source contractions; shared singleton selector per region',
        'whole_model_shared_solver_selected': False,
        'whole_model_cycles_status': 'UNKNOWN', 'whole_model_alias_status': 'UNKNOWN',
        'whole_model_roofline_status': 'UNKNOWN', 'compilation': receipt}
