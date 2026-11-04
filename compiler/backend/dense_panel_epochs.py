"""Fact-backed retained full-K load/compute/readout epochs.

Only a pure RHS-load epoch followed by a pure compute epoch immediately
before a checked uniform store epoch is eligible. The closing full drain is
unchanged, so neither operand nor accumulator ownership crosses panel reuse.
"""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import hashlib,json
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
from merlin.sched.ir import Loop,Call
from .dense_store_epochs import DenseStoreEpochResult,BEGIN,END,STORE,DRAIN

READY_BEGIN='gemmini_dense_ready_store_epoch_begin'

def _fact_law(adapter):
    from .column_batches import _admission
    _admission() # selected facts digest, CONFIG_ST queue/order, finite queues
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    cross={(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    def wait(a,b,space):
        row=cross.get((a,b,space),{})
        return row.get('overlap')=='completion' and (row.get('tracked') is True or row.get('disjoint')=='completion')
    loads=('MVIN.spad_addr','MVIN2.spad_addr')
    readers=('PRELOAD.weight_spad','COMPUTE_PRELOADED.a_spad')
    raw=all(wait(a,b,'scratchpad') for a in loads for b in readers)
    war=all(wait(b,a,'scratchpad') for a in loads for b in readers)
    acc=ordering['completion_order']['execute']['order']=='in_order'
    visible=wait('PRELOAD.c_addr','MVOUT.acc_addr','accumulator')
    configs={int(k):v for r in ordering['funct_queue'] for k,v in r.get('by_code',{}).items()}
    config=configs.get(0)=='execute' and configs.get(1)=='load' and configs.get(2)=='store' and ordering['same_queue']['store']['release']=='completion'
    finite=all(record['finite_entries'][q]['status']=='derived' and record['finite_entries'][q]['value']>=v
               for q,v in [('load',1),('execute',2),('store',1)])
    return DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-retained-full-k:'+record['canonical_ordering_sha256'],raw,war,acc,visible,config,finite,True,True)

def _calls(node):
    if type(node)is Loop:
        return tuple(c for child in node.body for c in _calls(child))
    return (node,)

def _strip(node):
    if type(node)is Loop:
        return replace(node,body=tuple(y for x in node.body if (y:=_strip(x)) is not None))
    return None if node.instr==DRAIN else node

def _plan(kernel):
    regions=[]
    def walk(body,mult=1):
        out=[]
        ready_at=set() # local sibling indices, never inherited across recursive walks
        for i,node in enumerate(body):
            if (type(node)is Loop and node.extent==1 and len(node.body)==2
                    and i+1<len(body) and type(body[i+1])is Call and body[i+1].instr==BEGIN):
                loads,computes=node.body
                a,b=_calls(loads),_calls(computes)
                if (type(loads)is Loop and loads.extent==1 and type(computes)is Loop and computes.extent==1
                    and a and b and a[-1].instr==DRAIN and b[-1].instr==DRAIN
                    and all(c.instr=='gemmini_mvin_weight' for c in a[:-1])
                    and all(c.instr in ('gemmini_preload','gemmini_compute_preloaded') for c in b[:-1])):
                    ready_at.add(i+1)
                    out.append(_strip(node));regions.append({'panel':node.var,'instances':mult,'removed_drains':3*mult})
                    continue
            if i in ready_at:
                if type(node)is not Call or node.instr!=BEGIN:
                    raise ValueError('ready store epoch lost its immediate producer')
                out.append(replace(node,instr=READY_BEGIN));continue
            out.append(replace(node,body=walk(node.body,mult*node.extent)) if type(node)is Loop else node)
        return tuple(out)
    actual=replace(kernel,body=walk(kernel.body))
    def serial(body):
        result=[]
        for node in body:
            if type(node)is Loop:result.append((node.var,node.extent,serial(node.body)))
            elif node.instr!=DRAIN:result.append(replace(node,instr=BEGIN) if node.instr==READY_BEGIN else node)
        return tuple(result)
    if serial(actual.body)!=serial(kernel.body):raise ValueError('panel epochs changed data operands, scopes or order')
    return actual,regions

@dataclass(frozen=True)
class DenseRetainedPanelEpochResult:
    original: object
    kernel: object
    realization: dict
    law: object
    checked_command: tuple
    def revalidate(self):
        from .dense import _adapter
        self.original.revalidate()
        if (tuple(self.original.command)!=self.checked_command or
                tuple(self.original.realization['store_epochs']['uniform_config'])!=self.checked_command):
            raise ValueError('retained panel epochs uniform CONFIG_ST word changed')
        actual,regions=_plan(self.original.kernel)
        if self.law!=_fact_law(_adapter()) or self.law.unknown() or actual!=self.kernel or self.kernel.digest()!=self.realization['kernel_sha256']:
            raise ValueError('retained panel epochs source, law or actual command stream changed')
    @property
    def command(self):return self.original.command
    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate();receipt=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call);self.revalidate();return receipt

def select_panel_epochs(schedule):
    selection=knob('DENSE_PANEL_EPOCHS',default='off',values=('off','on'),root=Path(__file__).resolve().parents[1])
    if selection=='off':return schedule,None
    refusal=None
    if type(schedule)is not DenseStoreEpochResult:refusal='requires an independently checked uniform disjoint-ACC store epoch'
    elif schedule.original.realization.get('schedule_ir',{}).get('pipeline_kind')!='retained_full_k':refusal='requires retained_full_k; segmented/two-window families retain their completion'
    elif getattr(schedule.original,'bias_source',None) is not None:refusal='initializer epochs are not admitted'
    if refusal:return schedule,dict(selection='on',applied=False,refusal=refusal)
    from .dense import _adapter
    schedule.revalidate();law=_fact_law(_adapter())
    if law.unknown():return schedule,dict(selection='on',applied=False,refusal='unknown selected ordering premises',unknown=list(law.unknown()))
    actual,regions=_plan(schedule.kernel)
    if not regions:return schedule,dict(selection='on',applied=False,refusal='no pure RHS-load/compute/uniform-readout panel chain')
    report=dict(selection='on',applied=True,source_kernel_sha256=schedule.kernel.digest(),kernel_sha256=actual.digest(),
        epoch_law=asdict(law),regions=regions,removed_dynamic_drains=sum(r['removed_drains'] for r in regions),
        source_proof='unchanged original full-K source/initializer/address/ownership certificate; original data operands and order retained',
        storage_proof='original RHS-only load epoch followed by read-only operands; no later load or ACC writer before closing store drain',
        readout_proof='original store-epoch exact disjoint ACC row proof and uniform CONFIG_ST; selected producer-to-readout completion wait',
        closing_completion='original store END full drain retained before every loop backedge and ownership reuse')
    result=DenseRetainedPanelEpochResult(schedule,actual,dict(schedule.realization,kernel_sha256=actual.digest(),panel_epochs=report),law,tuple(schedule.command))
    result.revalidate();return result,report
