"""Qualified invariant raw readout setup on the actual shared counted stream.

Every raw-i32 store has the same complete CONFIG_ST words: no pool, identity
scale, no activation, and the member's common output pitch. Configuration has
unknown state at invocation entry. The existing counted configuration passes
only move pure setters out of their store loops and remove established exact
repetitions. No memory, descriptor, completion, or loop domain is changed.

The selected public gemmini.h configures raw matmul readout once (line 727)
before its complete i0/j0/k0 loop and closes it with gemmini_fence. Its complete
CONFIG_ST encoding is at lines 275-282. That independent software contract
supplies persistent ordinary readout state; the selected ordered store queue
and unchanged completion epochs supply the source ordering premise here.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import json

from merlin.sched.ir import Call, Const, Loop
from merlin.compile.scheduling.configuration import ConfigurationAdapter, ConfigurationEffects
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration, verify_loop_configuration_rewrite
from merlin.compile.scheduling.configuration_loop_cleanup import eliminate_counted_configuration, verify_counted_configuration_rewrite
from . import isa
from .dense import _adapter
from . import dense_readout
from .dense_rhs_transfers import REFERENCE_FILES

CONFIG = 'shared_raw_readout_config'


def _with_configuration(kernel,word):
    setter=Call(CONFIG,tuple(zip(('funct','rs1','rs2'),map(Const,word))))
    def body(items):
        result=[]
        for x in items:
            if type(x) is Loop:result.append(replace(x,body=body(x.body)))
            else:
                if x.instr.endswith('gemmini_mvout'):result.append(setter)
                result.append(x)
        return tuple(result)
    return replace(kernel,body=body(kernel.body))


def _strip(kernel):
    def body(items):
        return tuple(replace(x,body=body(x.body)) if type(x) is Loop else x
            for x in items if type(x) is Loop or x.instr != CONFIG)
    return replace(kernel,body=body(kernel.body))


def _qualification():
    paths=(Path(__file__),Path(isa.__file__),Path(dense_readout.__file__),
        Path(__file__).with_name('dense_ordering.json'),
        Path(__file__).with_name('shared_rhs.py'))
    return {p.name:sha256(p.read_bytes()).hexdigest() for p in paths}


def _adapter_for(word):
    order=json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if (order['completion_order']['store']['order']!='in_order'
            or order['same_queue']['store']['release']!='completion'):
        raise ValueError('shared raw readout needs the selected ordered store completion queue')
    fields=tuple(zip(('raw_store_rs1','raw_store_rs2'),word[1:]))
    dense=_adapter()
    def project(call):
        if call.instr==CONFIG:
            args=dict(call.args)
            actual=tuple(args[k].value if type(args[k]) is Const else None for k in ('funct','rs1','rs2'))
            if actual!=word or actual[0]!=0 or actual[1]&3!=2:
                raise ValueError('shared raw readout setter differs from actual complete CONFIG_ST')
            return ConfigurationEffects(writes=fields,ordered=True,pure_write=True)
        if call.instr=='shared_epoch_drain':
            return ConfigurationEffects(ordered=True)
        if call.instr.startswith('shared_config_rhs_'):
            # That actual callback only emits CONFIG_LD(channel=1).
            return ConfigurationEffects(ordered=True)
        if call.instr.startswith('shared_common_'):
            name=call.instr[len('shared_common_'):]
        elif call.instr.startswith('shared_member'):
            member,name=call.instr[len('shared_member'):].split('_',1)
            int(member)
        else:
            return None
        if name not in ('gemmini_mvin_lhs','gemmini_mvin_weight','gemmini_preload',
                'gemmini_compute_preloaded','gemmini_mvout'):
            return None
        # Decode the actual data operands through the selected common adapter;
        # no unknown or instruction-name-only callback receives preservation.
        dense.project(replace(call,instr=name))
        return ConfigurationEffects(reads=fields if name=='gemmini_mvout' else (),ordered=True)
    pin=sha256(repr(sorted(_qualification().items())).encode()).hexdigest()
    return ConfigurationAdapter(project,'selected-full-raw-CONFIG_ST:'+repr(word),
        'same-complete-register-write; ordered store queue; unchanged checked epochs:'+pin)


@dataclass(frozen=True)
class SharedReadoutConfiguration:
    original: object
    kernel: object
    word: tuple
    stages: tuple
    pins: dict
    report: dict

    def revalidate(self):
        if self.pins!=_qualification():
            raise ValueError('shared raw readout terminal or selected qualification changed')
        adapter=_adapter_for(self.word)
        current=_with_configuration(self.original,self.word)
        for kind,before,after in self.stages:
            if before!=current:
                raise ValueError('shared raw configuration proof stages do not consume the actual source')
            if kind=='hoist':verify_loop_configuration_rewrite(before,after,adapter)
            else:verify_counted_configuration_rewrite(before,after,adapter)
            current=after
        if (current!=self.kernel or _strip(self.kernel)!=self.original
                or self.kernel.digest()!=self.report['kernel_sha256']):
            raise ValueError('shared readout cleanup changed actual data, loops or completion')


class _ExactStoreTerminal:
    """Let the original terminal encode stores, checking every hidden setup."""
    def __init__(self,ctx,word):
        self.ctx,self.word,self.configs,self.stores=ctx,word,0,0
    def __getattr__(self,name):return getattr(self.ctx,name)
    def emit(self,funct,rs1,rs2):
        if funct==isa.FUNCT['config']:
            if (funct,rs1,rs2)!=self.word or self.configs or self.stores:
                raise ValueError('raw shared terminal contains varying or additional configuration')
            self.configs+=1
        else:
            if funct!=isa.FUNCT['mvout'] or self.configs!=1 or self.stores:
                raise ValueError('raw shared terminal no longer encodes exactly one configured store')
            self.stores+=1
            self.ctx.emit(funct,rs1,rs2)
    def finish(self):
        if (self.configs,self.stores)!=(1,1):
            raise ValueError('raw shared terminal did not consume its checked configuration')


def store_tile(prepared,ctx,*args,**kwargs):
    terminal=_ExactStoreTerminal(ctx,prepared.word)
    dense_readout.store_tile(terminal,*args,**kwargs)
    terminal.finish()


def select_raw_readout_configuration(kernel,n):
    word=isa.config_st(n*4,scale=1.0,activation=0)
    expanded=_with_configuration(kernel,word)
    adapter=_adapter_for(word);current=expanded;stages=[];reports=[]
    # A prefix can rise at most one level per existing pass invocation. The
    # static loop depth bounds compilation; no runtime trip is expanded.
    def depth(items):return max((1+depth(x.body) for x in items if type(x) is Loop),default=0)
    for _ in range(depth(expanded.body)+1):
        selected=hoist_loop_configuration(current,adapter)
        if selected.kernel==current:break
        stages.append(('hoist',current,selected.kernel));reports.append(selected.report)
        current=selected.kernel
    selected=eliminate_counted_configuration(current,adapter)
    stages.append(('cleanup',current,selected.kernel));reports.append(selected.report)
    current=selected.kernel
    report={'applied':True,'source_kernel_sha256':kernel.digest(),'terminal_source_sha256':expanded.digest(),
        'kernel_sha256':current.digest(),'actual_config_st_words':word,'passes':reports,
        'data_and_completion_unchanged':True,'invocation_entry':'unknown; no cross-call state',
        'scope':'raw-i32 common-pitch shared products; exact persistent complete CONFIG_ST only',
        'reference_files':REFERENCE_FILES,
        'qualification':_qualification()}
    prepared=SharedReadoutConfiguration(kernel,current,word,tuple(stages),_qualification(),report)
    prepared.revalidate()
    return prepared
