"""Explicit spatial epoch alternative using the shared original-K consumer.

This pure adapter accepts already checked retention/ordering products. Software
issued work bounds never replace physical queue capacities. No terminal command,
source permutation, configuration, or private-storage allocation is changed.
"""
from dataclasses import replace, asdict
import hashlib
import json
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhases, plan_convolution_schedule)
from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw


def plan_spatial_epochs(schedule, rhs, input_rows, emission_options, *,
                        facts_bytes, ordering_selection, compute_pairs=32,
                        readout_epoch=False):
    """Return schedule, rebound RHS, terminal options, and explicit diagnostics.

    Scope is rolling 3x3 spatial rows wider than two primitive tiles, with a
    complete resident RHS and no competing spatial-group owner. Unsupported
    contracts retain the exact incoming objects and report a refusal.
    """
    diagnostic = dict(selected=False, compute_pairs=compute_pairs,
                      readout_epoch=readout_epoch, source_identity=schedule.source.identity)
    def refuse(reason):
        return schedule, rhs, emission_options, dict(diagnostic, reason=reason)
    p=schedule.source.plan
    if (p.window.kernel != (3,3) or p.window.output_spatial[1] <= 2*p.primitive
            or input_rows is None or rhs is None or type(rhs) is ConvolutionRhsPhases
            or emission_options.get('output_group') is not None
            or emission_options.get('readout_law') is not None):
        return refuse('requires ungrouped wide-row 3x3 with rolling input and complete resident RHS')
    if type(compute_pairs) is not int or not 1 < compute_pairs < 2**31:
        raise ValueError('software compute epoch must be a bounded positive integer greater than one')
    if type(readout_epoch) is not bool:
        raise ValueError('readout_epoch must be bool')
    schedule.verify_current()
    epoch=schedule.epoch_law
    if epoch is None or schedule.issue_law is None or not schedule.choice.load_issue:
        return refuse('requires existing checked compute/load epoch laws')
    pin=ordering_selection
    if pin['base_facts']['sha256'] != hashlib.sha256(facts_bytes).hexdigest():
        return refuse('selected RTL facts digest mismatch')
    digest=hashlib.sha256(json.dumps(pin['ordering'],sort_keys=True).encode()).hexdigest()
    if digest != pin['ordering_sha256'] or digest != epoch.hardware_identity:
        return refuse('checked epoch and selected ordering identities differ')
    if readout_epoch:
        record=pin['ordering']
        if (record.get('same_queue',{}).get('store',{}).get('release') != 'completion'
                or record.get('completion_order',{}).get('store',{}).get('order') != 'in_order'):
            return refuse('requires completion-serialized in-order store queue configuration capture')
        routes={int(c):q for item in record.get('funct_queue',())
                if item.get('funct')==0 for c,q in item.get('by_code',{}).items()}
        if routes.get(2) != 'store' or record.get('accesses',{}).get('MVOUT.acc_addr',{}).get('queue') != 'store':
            return refuse('selected CONFIG_ST and MVOUT must use the same store queue')
    updated=plan_convolution_schedule(schedule.source,
        replace(schedule.choice,compute_pairs=compute_pairs),
        retention_law=schedule.retention_law, issue_law=schedule.issue_law,
        epoch_law=replace(epoch,max_compute_pairs=compute_pairs))
    options=dict(emission_options)
    if readout_epoch:
        entries=json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']['store']['entries']
        if entries.get('status')!='derived' or entries.get('value',0)<1:
            return refuse('selected finite store progress unavailable')
        law=ConvolutionReadoutEpochLaw(epoch.hardware_identity,epoch.numerical_policy_identity,
            schedule.source.identity,'original-column-block-readout-final-drain-v1',
            len(schedule.source.column_intervals),entries['value'],
            epoch.local_read_after_write,epoch.accumulator_result_visibility,
            epoch.ordered_configuration_capture,epoch.finite_queue_pair_progress,
            epoch.complete_device_drain,epoch.exclusive_private_state)
        options['readout_law']=law
        diagnostic['readout_law']=asdict(law)
    rebound=replace(rhs,schedule=updated)
    diagnostic.update(selected=True,previous_compute_pairs=schedule.choice.compute_pairs,
        schedule_identity=updated.identity,epoch_law=asdict(updated.epoch_law),
        selected_ordering_sha256=digest,
        pass_order=['original-window','regional-retention','resident-rhs','spatial-epochs','shared-convolution-emission','original-terminal'],
        target_execution_verified=False,cycles='UNKNOWN')
    return updated,rebound,options,diagnostic
