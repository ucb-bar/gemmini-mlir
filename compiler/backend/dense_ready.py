"""Uniform causal dense readout, bound to selected ACC dependency facts.

The shared ready consumer independently checks each actual final-K producer and
store rectangle. Original operand and final ownership drains remain. Uniform
CONFIG_ST is issued after the initial full drain and stays unchanged throughout
the call, so no per-store numerical-state snapshot across changing words is used.
"""
from pathlib import Path
import hashlib,json
from merlin.compile.tuning_knobs import knob
from . import isa


def select_ready_readout(attributes,dtype,per_column):
    selection=attributes.get('dense_ready_readout',knob('DENSE_READY_READOUT',
        default='off',values=('off','ready'),root=Path(__file__).resolve().parents[1]))
    if selection not in ('off','ready'):raise ValueError('invalid dense ready readout selection')
    ep=attributes.get('epilogue',())
    eligible=(not per_column and dtype in ('i8','i32') and
        not any('pool' in str(x).lower() for x in ep) and
        not ('requant' in ep and attributes.get('requant_shift') is not None))
    if dtype=='i32':
        eligible=eligible and attributes.get('acc_scale_by_column') is None and float(attributes.get('acc_scale',1.0))==1.0 and not attributes.get('relu',False) and 'relu' not in ep
    return selection=='ready' and eligible,dict(selection=selection,
        uniform_configuration_eligible=eligible,scope='retained full-K; separately selected segmented adapter requires its original source and ownership proofs')


def uniform_store_command(attributes,n,dtype):
    scales=attributes.get('acc_scale_by_column')
    scale=scales[0] if scales else attributes.get('acc_scale',1.0)
    activation=int(bool(attributes.get('relu',False) or 'relu' in attributes.get('epilogue',())))
    return isa.config_st(n*(4 if dtype=='i32' else 1),scale=scale,activation=activation)


def causal_readout_admission(block,adapter):
    """Bind the selected causal law independently of the schedule family."""
    from merlin.compile.scheduling.dense_panel_pipeline import DenseReadyReadoutAdmission
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=hashlib.sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
        or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
        or ordering['status']!='derived'):
        raise ValueError('causal dense readout lacks selected ordering facts')
    cross={(x['older'],x['younger'],x['space']):x for x in ordering['cross_queue']}
    waits=[]
    for older,younger in [('PRELOAD.c_addr','MVOUT.acc_addr'),('MVOUT.acc_addr','PRELOAD.c_addr')]:
        row=cross.get((older,younger,'accumulator'),{})
        if row.get('overlap')!='completion' or row.get('tracked') is not True:
            raise ValueError('causal dense readout lacks tracked ACC completion/reuse wait')
        waits.append(row)
    if (ordering['completion_order']['execute']['order']!='in_order'
        or ordering['completion_order']['store']['order']!='in_order'
        or ordering['same_queue']['store']['release']!='completion'):
        raise ValueError('causal dense readout lacks selected ordered consumer progress')
    for lane in ('execute','store'):
        entry=record['finite_entries'][lane]
        if entry.get('status')!='derived' or type(entry.get('value')) is not int or entry['value']<1:
            raise ValueError('causal dense readout lacks finite queue progress')
    extents=ordering['extents']
    if any(extents.get(name,{}).get('status')!='derived' for name in ('PRELOAD.c_addr','MVOUT.acc_addr')):
        raise ValueError('causal dense readout lacks actual ACC descriptor extents')
    identity='selected-causal-acc-readout:'+digest
    admission=DenseReadyReadoutAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity,identity+':uniform-config-through-final-drain',block.m*block.n,1)
    binding=dict(ordering_sha256=digest,
        actual_accumulator_waits=waits,extents={name:extents[name] for name in ('PRELOAD.c_addr','MVOUT.acc_addr')},
        numerical_state='one uniform CONFIG_ST after initial full drain; unchanged until final call drain',
        producer_visibility='actual final K and ACC footprints checked by causal ready consumer',
        ownership='original operand and call completion drains retained',pending_readouts=1)
    return admission,binding


def admit_causal_readout(original,problem,block,resources,storage,adapter):
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_ready_readout
    admission,binding=causal_readout_admission(block,adapter)
    result=schedule_dense_ready_readout(original,problem,block,resources,storage,adapter,
        admission,static_budget=128,timeout_ms=30000)
    result.realization['ready_readout_selected']=True
    result.realization['ready_readout_target_binding']=binding
    return result
