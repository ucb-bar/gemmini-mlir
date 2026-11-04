"""Checked final-K frontiers composed with counted dense operand epochs.

The shared ready pass first transforms the original drained pipeline and the
shared temporal/source verifier proves its actual producer/store order. Only
then does the existing epoch pass remove operand-stream drains. Its consumption
proof runs again on that exact program; no data effects cross that rewrite.
Wide transfers are subsequently proved by their existing all-band adapter.
"""
from dataclasses import dataclass, replace, asdict
import hashlib
import json
from pathlib import Path
import time

from merlin.compile.scheduling.dense_panel_epochs import (
    DensePanelEpochResult, _plan, _prove_stream, _drains, _record,
)
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseReadyReadoutAdmission, _verify_dense_epochs,
)
from merlin.compile.scheduling.dense_ready_readout import (
    propose_dense_ready_readout, DenseReadyReadoutCheckingProjection,
)


def _owner():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DenseEpochReadyResult:
    original: object
    kernel: object
    realization: dict
    bound: object
    admission: object
    problem: object
    _adapter: object
    _storage: object
    owner_sha256: str
    checked_identity: str

    def revalidate(self):
        self.original.revalidate()
        ready, _ = propose_dense_ready_readout(self.original.original.kernel,
                                               self.problem, self._adapter)
        expected, _ = _plan(ready, self._adapter, self.original.law, self.bound)
        self.admission.validate_for(self._adapter, self.bound)
        if (expected != self.kernel or self.kernel.digest() != self.realization['kernel_sha256']
                or self.owner_sha256 != _owner()
                or self.checked_identity != hashlib.sha256(repr((self.original.kernel.digest(),
                    self.kernel.digest(), asdict(self.bound), asdict(self.admission),
                    asdict(self.problem), repr(self._adapter), asdict(self._storage))).encode()).hexdigest()):
            raise ValueError('dense ready epoch source, frontier, law or owner changed')

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt


def schedule_epoch_ready_readout(original, problem, resources, storage, adapter,
                                  *, timeout_ms=20000, static_budget=256):
    if (type(original) is not DensePanelEpochResult or original._storage != storage
            or original._adapter != adapter):
        raise ValueError('ready epochs need their original checked two-window owner')
    original.revalidate()
    law = original.law
    if law.unknown():
        raise ValueError('ready epochs require all selected ordering and completion premises')
    root = Path(__file__).parent / 'data'
    evidence = json.loads((root / 'command_ordering.json').read_text())
    ordering = evidence['ordering']
    expected_identity = 'selected-ordering:' + evidence['ordering_sha256']
    if law.qualification_identity != expected_identity:
        raise ValueError('ready epochs require the selected command ordering identity')
    # The dependency checker stalls an overlapping read until the older ACC
    # producer retires; store configuration is captured by its ordered queue.
    raw = next((r for r in ordering['cross_queue']
                if (r['older'], r['younger'], r['space']) ==
                   ('PRELOAD.c_addr', 'MVOUT.acc_addr', 'accumulator')), None)
    codes = next(r['by_code'] for r in ordering['funct_queue'] if r.get('funct') == 0)
    facts = json.loads((root / 'facts.json').read_text())
    stores = facts['facts']['timing_structure']['command_classes']['classes']['store']
    pending = stores['entries']['value']
    if (raw is None or raw['overlap'] != 'completion' or codes.get('2') != 'store'
            or pending <= 0 or stores['entries']['status'] != 'derived'):
        raise ValueError('selected producer retirement/store snapshot/backpressure is unavailable')
    raw_owner = original.original
    ready, changes = propose_dense_ready_readout(raw_owner.kernel, problem, adapter)
    if not changes:
        raise ValueError('ready readout did not transform an original final-K frontier')
    issued = max(c['output_tiles'] for c in changes)
    bound = replace(raw_owner.admission, max_issued_readouts=issued,
                    pending_readouts=pending, readout_configuration='ordered_snapshot_before_store_issue')
    admission = DenseReadyReadoutAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, expected_identity, expected_identity,
        issued, pending)
    # This is the original drained two-window program, with the actual ready
    # frontier. The shared checker proves all hazards before using its exact
    # typed source projection. Earlier operand epochs retain their drains here.
    report = _verify_dense_epochs(ready, problem, resources, storage, adapter, bound,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=raw_owner.output_readout, bias_source=raw_owner.bias_source,
        ordered_accumulator_commits=True, source_phase_fission=True,
        readout_compute_admission=admission,
        readout_source_projection=DenseReadyReadoutCheckingProjection(raw_owner.kernel, problem))
    actual, streams = _plan(ready, adapter, law, bound)
    deadline = time.monotonic() + timeout_ms / 1000
    proofs = sum(_prove_stream(s, adapter, storage, deadline, 4096) for s in streams)
    from merlin.compile.scheduling.dense_panel_pipeline import _serial_projection
    if _serial_projection(actual) != _serial_projection(ready):
        raise ValueError('ready operand epoch composition changed a data effect')
    realization = dict(original.realization, schema='gemmini.dense_epoch_ready.v1',
        kernel_sha256=actual.digest(), ready_frontier=changes,
        ready_frontier_proof=report, ready_execution_bound=asdict(bound),
        ready_readout_admission=asdict(admission),
        ready_parent_kernel_sha256=original.kernel.digest(),
        ready_drained_kernel_sha256=ready.digest(),
        dynamic_epoch_drains=_drains(actual.body),
        ready_operand_epochs=dict(streams=[_record(s) for s in streams], consumption_proofs=proofs),
        ready_pass_order=['propose_dense_ready_readout', 'verify_actual_ready_epochs',
                          'propose_dense_panel_epochs', 'prove_operand_consumption',
                          'coalesce_dense_epochs_all_bands'])
    identity = hashlib.sha256(repr((original.kernel.digest(), actual.digest(),
        asdict(bound), asdict(admission), asdict(problem), repr(adapter), asdict(storage))).encode()).hexdigest()
    result = DenseEpochReadyResult(original, actual, realization, bound, admission,
                                    problem, adapter, storage, _owner(), identity)
    result.revalidate()
    return result


def resident_readout_laws(adapter, block, reduction, epoch, *, input_pipeline=False):
    """Selected RAW, snapshot, backpressure and optional disjoint LD conjunction.

    The shared full-K ready owner checks actual source effects and both physical
    LHS slots. These declarations supply ordering only, never capacity or timing.
    """
    root = Path(__file__).parent / 'data'
    evidence = json.loads((root / 'command_ordering.json').read_text())
    identity = 'selected-ordering:' + evidence['ordering_sha256']
    facts = json.loads((root / 'facts.json').read_text())
    stores = facts['facts']['timing_structure']['command_classes']['classes']['store']['entries']
    if (epoch.unknown() or reduction.qualification_identity != identity
            or stores['status'] != 'derived' or stores['value'] <= 0):
        raise ValueError('resident readout needs complete selected ordering and store backpressure')
    pending = stores['value']
    bound = replace(reduction, max_issued_readouts=block.m*block.n,
                    pending_readouts=pending,
                    readout_configuration='ordered_snapshot_before_store_issue')
    admission = DenseReadyReadoutAdmission(adapter.configuration_identity,
        adapter.numerical_policy_identity, identity, identity, block.m*block.n, pending,
        disjoint_load_qualification_identity=identity if input_pipeline else None)
    admission.validate_for(adapter, bound)
    return bound, admission
