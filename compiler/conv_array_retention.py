"""Optional binding of checked rolling convolution groups to WS fresh/STAY.

This binds the library's existing grouped source/RHS/ACC ownership checks to
actual terminal words. It supplies no performance qualification.
"""
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission


def retained_ordering_admission(facts_path,ordering_path):
    """Reopen selected funct5 addressing and every load/stay RAW/WAR edge."""
    try:
        raw=Path(facts_path).read_bytes(); facts=json.loads(raw)
        selected=json.loads(Path(ordering_path).read_bytes());order=selected['ordering']
        if selected['base_facts']['sha256']!=hashlib.sha256(raw).hexdigest():
            raise ValueError('retained ordering has different hardware facts')
        if selected['ordering_sha256']!=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest():
            raise ValueError('retained ordering hash changed')
        a=order['accesses']['COMPUTE_ACCUMULATE.a_spad']
        if (a['funct']!=5 or a['queue']!='execute' or a['direction']!='read'
                or order['extents']['COMPUTE_ACCUMULATE.a_spad']!=order['extents']['COMPUTE_PRELOADED.a_spad']):
            raise ValueError('selected retained A addressing differs from original fresh addressing')
        edges={(r['older'],r['younger'],r['space']):r for r in order['cross_queue']}
        for load in ('MVIN.spad_addr','MVIN2.spad_addr','MVIN3.spad_addr'):
            for x,y in ((load,'COMPUTE_ACCUMULATE.a_spad'),('COMPUTE_ACCUMULATE.a_spad',load)):
                if edges.get((x,y,'scratchpad'),{}).get('overlap')!='completion':
                    raise ValueError('retained A load/consume dependency is unresolved')
        dataflow=facts.get('facts',facts)['timing_structure']['dataflow']
        if dataflow['status']!='derived' or dataflow['modes'].get('1')!='weight_stationary':
            raise ValueError('selected execute dataflow does not bind WS retention')
        return dict(admitted=True,ordering_sha256=selected['ordering_sha256'],hardware_sha256=hashlib.sha256(raw).hexdigest(),
            retained_funct=5,descriptor_funct=6,A_mapping='exact selected fresh/stay extent equality',
            persistence='first real RHS PRELOAD/FLIP, later garbage-weight PRELOAD/STAY; independently native checked',
            output='each PRELOAD preserves original ACC destination, continuation and active extents')
    except (ValueError,KeyError,TypeError,OSError)as e:
        return dict(admitted=False,reason=str(e))


def plan_array_retention(schedule,rhs,input_rows,readout,ordering,*,facts_path,ordering_path,max_row_blocks=4,max_output_rows=1):
    record=dict(requested='groups',selected='incumbent',timing='UNKNOWN')
    retained=retained_ordering_admission(facts_path,ordering_path)
    record['retained_ordering']=retained
    if not retained['admitted']:
        return None,None,dict(record,reason=retained['reason'])
    if not ordering.get('admitted') or schedule.epoch_law is None or readout is None:
        return None,None,dict(record,reason='requires original admitted compute/readout epochs')
    if not schedule.retention_law.independent_output_rows or not schedule.retention_law.no_intermediate_observers:
        return None,None,dict(record,reason='requires original independent output rows and unobserved partial reductions')
    group=plan_rolling_convolution_output_group(schedule,rhs,input_rows,max_row_blocks=max_row_blocks,max_output_rows=max_output_rows)
    if group is None:
        return None,None,dict(record,reason='checked original rolling input, primitive partitions, resident RHS and multiple M tiles unavailable')
    law=DenseArrayRetentionAdmission(schedule.epoch_law.hardware_identity,
        schedule.epoch_law.numerical_policy_identity,
        'selected-WS-funct4-flip/funct5-stay-funct6-descriptor:'+retained['ordering_sha256'])
    def describe(g):
        return None if g is None else dict(row_blocks=g.row_blocks,output_rows=g.output_rows,
            rhs_base=g.panel.rhs_base,accumulator_rows=g.row_blocks*g.panel.accumulator_rows)
    record.update(selected='groups',groups=[describe(g)for g in group]if isinstance(group,tuple)else[describe(group)],
        source_identity=schedule.source.identity,array_admission='selected WS and exact fresh/stay A mapping; native-checked original RHS persistence and output descriptors',
        law_qualification=law.qualification_identity)
    return group,law,record


def emit_product(enc,fields,accumulate):
    """Encode actual paired output descriptor + fresh or retained WS compute.

    A retained descriptor has garbage weight address; its ACC address and
    all active extents remain original. STAY selects the captured array bank.
    """
    f=fields
    target=enc.add(f['accumulator_row'],(1<<30)if accumulate else 0)
    fresh=f.get('fresh_weights',True)
    enc.preload(f['rhs_row']if fresh else (1<<32)-1,target,
        f['depth'],f['columns'],f['rows'])
    if fresh:
        enc.compute(f['lhs_row'],f['depth'],f['rows'])
    else:
        enc.issue('COMPUTE_AND_STAY_CMD',
            enc.bundle('ComputeRs',local_addr=f['lhs_row'],num_rows=f['rows'],num_cols=f['depth']),
            enc.bundle('ComputeRs',local_addr=(1<<32)-1,num_rows=16,num_cols=16))
