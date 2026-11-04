"""Optional store epochs for the existing complete-private horizontal panels.

The shared horizontal emitter retains every original effect. Only the parent's
post-store completion is suppressed; its compute-ready fence, all load/bias
fences, group-last-use completion and invocation exit completion remain real.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.horizontal_convolution import (
    HorizontalConvolutionPanels, emit_horizontal_convolution_panels)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionComputeEpochLaw, ConvolutionReadoutEpochLaw, ConvolutionOutputTile)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _emitter_identity():
    from merlin.compile.scheduling import horizontal_convolution
    return _sha(Path(horizontal_convolution.__file__).read_bytes())


def _readout_library_identity():
    from merlin.compile.scheduling import window_transfer
    return _sha(Path(window_transfer.__file__).read_bytes())


@dataclass(frozen=True)
class HorizontalReadoutAdmission:
    plan: HorizontalConvolutionPanels
    law: ConvolutionReadoutEpochLaw
    source_identity: str
    program_identity: str
    selection_json: str
    facts_sha256: str
    ordering_sha256: str
    emitter_sha256: str
    readout_library_sha256: str
    checked_sha256: str

    def _binding(self):
        return _sha(repr((self.plan, self.law, self.source_identity,
            self.program_identity, self.selection_json, self.facts_sha256,
            self.ordering_sha256, self.emitter_sha256,
            self.readout_library_sha256)).encode())

    def revalidate(self):
        if type(self.plan) is not HorizontalConvolutionPanels:
            raise ValueError('horizontal readout requires its original typed private panel plan')
        self.plan.__post_init__()
        if self.checked_sha256 != self._binding():
            raise ValueError('horizontal readout checked source or law binding changed')
        selection = json.loads(self.selection_json)
        if (self.program_identity != _sha(repr(self.plan).encode())
                or selection['program_identity'] != self.program_identity
                or selection['source_identity'] != self.source_identity
                or self.law.source_identity != self.program_identity
                or self.law.hardware_identity != self.ordering_sha256
                or self.emitter_sha256 != _emitter_identity()
                or self.readout_library_sha256 != _readout_library_identity()):
            raise ValueError('horizontal readout source, private plan or shared emitter changed')
        p, d = self.plan, self.plan.source.primitive
        if self.law.max_issued_readouts != p.row_blocks * len(p.column_intervals):
            raise ValueError('horizontal readout issued budget differs from original private group')
        for mi in range(p.row_blocks):
            self.law.require(p.source, d, tuple(ConvolutionOutputTile(column, columns,
                (ni*p.row_blocks+mi)*d) for ni, (column, columns) in enumerate(p.column_intervals)))

    def receipt(self):
        self.revalidate()
        return dict(selected=True, mechanism='original horizontal stores with group-last-use drain',
            source_identity=self.source_identity, program_identity=self.program_identity,
            admission_sha256=self.checked_sha256,
            selection_sha256=_sha(self.selection_json.encode()), facts_sha256=self.facts_sha256,
            ordering_sha256=self.ordering_sha256, shared_emitter_sha256=self.emitter_sha256,
            readout_library_sha256=self.readout_library_sha256,
            readout_law=asdict(self.law), accumulator_rows=[0,self.plan.accumulator_rows],
            stores_per_complete_group=self.law.max_issued_readouts,
            configuration='original CONFIG_ST and MVOUT share completion-serialized in-order store queue',
            preserved_completions=['each RHS/input/bias load', 'horizontal_rhs_ready',
                'horizontal_group_ready', 'compute-ready before first store',
                'horizontal_group_last_use before ACC/input reuse', 'invocation exit'],
            omitted_completions='only parent-restored fence immediately after each terminal store',
            arithmetic_and_readout='unchanged original terminal encoder and original scalar configurations',
            compute_readout_overlap=False, target_execution_verified=False, timing=None)


def admit_horizontal_readout(plan, selection, *, facts_bytes, ordering_selection):
    """Pure optional admission; missing selected premises return None before IR.

    Existing selected compute readiness/private ownership are prerequisites.
    The software issued-store bound counts original M/N cells, not queue slots;
    finite selected queues backpressure until the final real group drain.
    """
    if type(plan) is not HorizontalConvolutionPanels or type(selection) is not dict:
        raise ValueError('horizontal readout requires original typed plan and selection')
    plan.__post_init__()
    if type(facts_bytes) is not bytes or type(ordering_selection) is not dict:
        raise ValueError('horizontal readout requires explicit selected fact bytes and ordering')
    try:
        pin = ordering_selection
        record = pin['ordering']
        digest = _sha(json.dumps(record, sort_keys=True).encode())
        facts_sha = _sha(facts_bytes)
        if pin['base_facts']['sha256'] != facts_sha or pin['ordering_sha256'] != digest:
            return None
        program = _sha(repr(plan).encode())
        epoch_record = selection['compute_epoch']
        epoch = ConvolutionComputeEpochLaw(**epoch_record['epoch_law'])
        epoch.require(epoch_record['compute_pairs'])
        if (selection.get('selected') is not True or epoch_record.get('selected') is not True
                or selection['program_identity'] != program
                or epoch_record['horizontal_program_identity'] != program
                or epoch.source_identity != program or epoch.hardware_identity != digest
                or not isinstance(selection['source_identity'], str) or not selection['source_identity']
                or selection['transfer_law']['facts_sha256'] != facts_sha
                or selection['transfer_law']['ordering_sha256'] != digest
                or selection['private_storage']['accumulator_rows'] != [0,plan.accumulator_rows]
                or selection['private_storage']['operand_rows'] != [0,plan.operand_rows]):
            return None
        if plan.row_blocks > 1:
            array = selection.get('array_retention', {})
            if (array.get('selected') is not True
                    or array.get('horizontal_program_identity') != program
                    or array.get('array_law', {}).get('hardware_identity') != digest):
                return None
        # Configuration is mutable per scalar interval. Its capture and the
        # corresponding MVOUT must complete in order on the SAME store queue.
        routes = {int(c):q for item in record.get('funct_queue', ())
                  if item.get('funct') == 0 for c,q in item.get('by_code', {}).items()}
        access = record.get('accesses', {}).get('MVOUT.acc_addr', {})
        if (routes.get(2) != 'store' or access.get('funct') != 3 or access.get('queue') != 'store'
                or record.get('extents', {}).get('MVOUT.acc_addr', {}).get('status') != 'derived'
                or record.get('same_queue', {}).get('store', {}).get('release') != 'completion'
                or record.get('completion_order', {}).get('store', {}).get('order') != 'in_order'):
            return None
        entries = json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']['store']['entries']
        if entries.get('status') != 'derived' or type(entries.get('value')) is not int or entries['value'] < 1:
            return None
        law = ConvolutionReadoutEpochLaw(digest, epoch.numerical_policy_identity, program,
            'original-horizontal-complete-private-MN-stores-group-last-use-drain-v1',
            plan.row_blocks*len(plan.column_intervals), entries['value'],
            epoch.local_read_after_write, epoch.accumulator_result_visibility,
            epoch.ordered_configuration_capture, epoch.finite_queue_pair_progress,
            epoch.complete_device_drain, epoch.exclusive_private_state)
        result = HorizontalReadoutAdmission(plan, law, selection['source_identity'], program,
            json.dumps(selection, sort_keys=True), facts_sha, digest, _emitter_identity(),
            _readout_library_identity(), '')
        result = replace(result, checked_sha256=result._binding())
        result.revalidate()
        return result
    except (KeyError, TypeError, ValueError):
        return None


def emit_horizontal_readout_epochs(admission, builder, lower, complete):
    """Delegate actual effects to the SAME horizontal emitter and terminal.

    ``lower(kind, fields, *, complete_after_effect=True)`` must preserve the
    incumbent pre-store compute drain, and use False only to skip its restored
    post-store fence. ``complete`` remains the original real completion callback.
    The caller retains its entry and exit fences. There is no default change.
    """
    if type(admission) is not HorizontalReadoutAdmission or not callable(lower) or not callable(complete):
        raise ValueError('horizontal readout emission requires its typed admission and original callbacks')
    admission.revalidate()  # Before any effect callback or builder mutation.
    p, d = admission.plan, admission.plan.source.primitive
    ready = False
    stores = 0
    computed = {}
    static_groups = 0

    def issue(kind, fields):
        nonlocal stores
        if kind == 'compute':
            if not ready or stores:
                raise ValueError('horizontal readout cannot overlap stores and later producers')
            key = (fields['output_column'], fields['accumulator_row'])
            computed[key] = (fields['output_row'], fields['rows'], fields['columns'])
        elif kind == 'store':
            if not ready or not computed or stores >= len(computed):
                raise ValueError('horizontal readout lost complete original group producers')
            if len(computed) % len(p.column_intervals):
                raise ValueError('horizontal readout incomplete original M/N group')
            live_m = len(computed)//len(p.column_intervals)
            ni, mi = divmod(stores, live_m)
            column, columns = p.column_intervals[ni]
            row = (ni*p.row_blocks+mi)*d
            if (fields['output_column'] != column or fields['columns'] != columns
                    or fields['accumulator_row'] != row
                    or computed.get((column,row)) != (fields['output_row'],fields['rows'],columns)):
                raise ValueError('horizontal readout changed original store ranges or private ACC cells')
            admission.law.require(p.source, fields['rows'], tuple(ConvolutionOutputTile(c,n,
                (index*p.row_blocks+mi)*d) for index,(c,n) in enumerate(p.column_intervals)))
            if len(computed) > admission.law.max_issued_readouts:
                raise ValueError('horizontal readout exceeds original group issued budget')
            stores += 1
        elif ready or stores:
            raise ValueError('horizontal readout must drain before any input, RHS or bias reuse')
        lower(kind, fields, complete_after_effect=(kind != 'store'))

    def finish(kind, fields):
        nonlocal ready, stores, computed, static_groups
        if kind == 'horizontal_group_ready':
            if ready or stores or computed:
                raise ValueError('horizontal readout entered a group before prior last-use drain')
            ready = True
        elif kind == 'horizontal_group_last_use':
            if not ready or not stores or stores != len(computed):
                raise ValueError('horizontal readout final drain requires all original stores')
            # Complete is the original callback: publication and private-state
            # retirement happen before resetting ownership for another group.
            complete(kind, fields)
            ready, stores, computed = False, 0, {}
            static_groups += 1
            return
        elif kind != 'horizontal_rhs_ready':
            raise ValueError('horizontal readout received an unknown completion boundary')
        complete(kind, fields)

    receipt = emit_horizontal_convolution_panels(p, builder, issue, finish)
    if ready or stores or computed or not static_groups:
        raise ValueError('horizontal readout did not close every original private group')
    admission.revalidate()
    return dict(receipt, horizontal_readout_epoch=admission.receipt())
