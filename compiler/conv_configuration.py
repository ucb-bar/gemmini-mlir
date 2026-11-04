"""Default-off source-bound scope for invariant convolution RHS LD1 setup.

Install after the original CountedConvolutionPlan/prologue, before zero-row and
counted-loop emission. No LD0 configuration is moved. The scope owns exactly
full LD1 register words and never grants array, memory or completion overlap.
"""
import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.configuration import ConfigurationAdapter,ConfigurationEffects,verify_configuration_state
from merlin.sched.ir import Call,Const,Kernel
from isa import encode_config_ld

VALUES=('off','rhs_scope')
DECLARATION=dict(values=list(VALUES),mechanism='source-bound complete persistent LD1 configuration once before original counted convolution loops',note='Default off. Exact full RHS configuration only; all LD0/LD2 configurations, transfers, products, readouts and completion preserved. No timing claim.')

def selected():return knob('conv_configuration',default='off',values=VALUES,root=Path(__file__).parent)

def _footprint(root):
    """Read the actual callback, not a claimed selected-effect annotation."""
    source=(root/'convolution.py').read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lower_conv')
    allowed={'add','branch','compute','config_ex','config_ld','config_st','constant_i32_dispatch','constant_i32_table','fence','loop','mul','mvin','mvout','or_','preload','ptr_offset'}
    if sum(isinstance(n,ast.Name) and n.id=='co' and isinstance(n.ctx,ast.Store) for n in ast.walk(function))!=1:raise ValueError('original CO is not assigned once outside counted callbacks')
    rhs=[]
    for node in ast.walk(function):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or not isinstance(node.func.value,ast.Name) or node.func.value.id!='fb':continue
        name=node.func.attr
        if name not in allowed:raise ValueError('unqualified convolution builder effect: '+name)
        if name=='config_ld':
            keywords={k.arg:k.value for k in node.keywords}
            state=keywords.get('id',ast.Constant(0))
            if not isinstance(state,ast.Constant) or state.value not in (0,1,2):raise ValueError('dynamic/unqualified LD state selector')
            if state.value==1:
                if len(node.args)!=1 or not isinstance(node.args[0],ast.Name) or node.args[0].id!='co' or set(keywords)!={'id'}:raise ValueError('RHS setup is not exact full default config_ld(co,id=1)')
                rhs.append(node.lineno)
    if len(rhs)!=1:raise ValueError('scope requires exactly one invariant original RHS setter site')
    return dict(convolution_sha256=sha256(source.encode()).hexdigest(),rhs_setter_line=rhs[0],builder_effects=sorted(allowed),rhs_setter='config_ld(co,id=1), complete default scale/shrink/block_stride/pixel_repeats')

def qualify(root=None):
    root=Path(root) if root is not None else Path(__file__).parent
    facts=(root/'facts.json').read_bytes();selection=json.loads((root/'command_ordering.json').read_text());r=selection['ordering'];digest=sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
    if r['status']!='derived' or digest!=selection['ordering_sha256'] or sha256(facts).hexdigest()!=selection['base_facts']['sha256']:raise ValueError('RHS scope requires intact selected facts/order')
    queues={int(code):q for row in r['funct_queue'] for code,q in row.get('by_code',{}).items()}
    load2=[row.get('queue') for row in r['funct_queue'] if row['funct']==1]
    if queues.get(1)!='load' or load2!=['load'] or r['same_queue']['load']['release'] not in ('issue','completion'):raise ValueError('RHS scope lacks ordered CONFIG_LD/MVIN2 capture')
    bundles=json.loads(facts)['facts']['interfaces'][3]['bundles'];state=bundles['ConfigMvinRs1']['fields']['state_id']
    if state!={'offset':3,'width':2}:raise ValueError('RHS scope lacks separate selected LD state IDs')
    # Reuse the existing target's complete persistent-register qualification.
    # Its block-CSE policy is narrower than this effect contract; it does not
    # establish the new source-scope dominance proof, supplied separately here.
    from isa_configuration import _project,_FENCE_ADAPTER
    actual_adapter=Path(inspect.getsourcefile(_project)).read_bytes()
    if sha256(actual_adapter).digest()!=sha256((root/'isa_configuration.py').read_bytes()).digest():raise ValueError('loaded persistent-state adapter differs from qualified source')
    setup=encode_config_ld(37,id=1)
    def terminal(funct,a,b):return Call('terminal',tuple((k,Const(v)) for k,v in [('funct',funct),('static_mask',3),('rs1',a),('rs2',b)]))
    effect=_project(terminal(0,*setup))
    if effect is None or not effect.ordered or not effect.pure_write or dict(effect.writes)!={'ld1.rs1':setup[0],'ld1.rs2':37}:raise ValueError('selected persistent LD1 contract unavailable')
    for event in [terminal(0,*encode_config_ld(11,id=0)),terminal(0,*encode_config_ld(0,id=2)),*(terminal(f,0,0) for f in (1,2,3,4,5,6,14)),Call('cpu_fence',())]:
        e=_project(event)
        if e is None or not e.ordered or set(e.clobbers)&{'ld1.rs1','ld1.rs2'} or any(k.startswith('ld1.') for k,v in e.writes):raise ValueError('selected intervening effect may clobber LD1')
    def projected(call):
        e=_project(call)
        args={k:v.value for k,v in call.args}
        if call.instr=='terminal' and args['funct']==1:
            return ConfigurationEffects(reads=(('ld1.rs1',setup[0]),('ld1.rs2',setup[1])),ordered=True)
        return e
    events=(terminal(0,*setup),terminal(0,*encode_config_ld(11,id=0)),terminal(2,0,0),Call('cpu_fence',()),terminal(1,0,0),terminal(4,0,0),Call('cpu_fence',()),terminal(1,0,0))
    scope_adapter=ConfigurationAdapter(projected,_FENCE_ADAPTER.configuration_identity,_FENCE_ADAPTER.qualification_identity)
    state_proof=verify_configuration_state(Kernel('rhs_scope_effect_footprint',(),events),scope_adapter)
    return dict(shared_state_proof=state_proof,facts_sha256=sha256(facts).hexdigest(),ordering_sha256=digest,configuration_adapter=_FENCE_ADAPTER.record(),configuration_adapter_sha256=sha256((root/'isa_configuration.py').read_bytes()).hexdigest(),scope='full LD1 persistent words across only known preserving commands and plain CPU fences; selected load queue captures setup before consumers',footprint=_footprint(root))

class RhsConfigurationScope:
    def __init__(self,builder,plan,command,co,law):
        self.builder=builder;self.co=co;self.words=encode_config_ld(co,id=1);self.owner=self._owner()
        self.receipt=dict(schema='gemmini.convolution_rhs_configuration_scope.v1',mode='rhs_scope',plan_sha256=plan.identity,source_attributes=command.get('attributes',{}).copy(),full_ld1_words=list(self.words),qualification=law,hoisted_setters=1,suppressed_static_sites=0,static_rhs_consumers=0,ld0='unchanged',completion='all original fences unchanged',timing='UNKNOWN')
        # Dominates every counted loop/branch. Initial state is never assumed.
        builder.config_ld(co,id=1)
        owner=builder
        while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
        if not hasattr(owner,'conv_configuration_receipts'):owner.conv_configuration_receipts=[]
        owner.conv_configuration_receipts.append(self.receipt)
    def _owner(self):return tuple(getattr(self.builder,k,None) for k in ('task_index','source_op_index','region_id'))
    def _check(self):
        if self._owner()!=self.owner:raise ValueError('RHS configuration scope crossed source owner')
    def __getattr__(self,name):
        if name in ('insn','emit','flush','finish','config_norm'):raise ValueError('unqualified terminal access in RHS scope: '+name)
        return getattr(self.builder,name)
    def config_ld(self,stride,**kwargs):
        self._check();state=kwargs.get('id',0)
        if type(state) is not int or state not in (0,1,2):raise ValueError('unqualified dynamic LD context')
        if state!=1:return self.builder.config_ld(stride,**kwargs)
        if any(type(v) not in (int,float,bool) for v in (stride,*kwargs.values())) or encode_config_ld(stride,**kwargs)!=self.words:raise ValueError('conflicting/partial RHS configuration inside invariant scope')
        self.receipt['suppressed_static_sites']+=1
    def mvin(self,ptr,addr,cols=16,rows=16,id=0):
        self._check()
        if id==1:self.receipt['static_rhs_consumers']+=1
        elif id not in (0,2):raise ValueError('unqualified load context consumer')
        return self.builder.mvin(ptr,addr,cols=cols,rows=rows,id=id)

def wrap(builder,plan,command,co,*,mode=None,root=None):
    """Off returns the same object; refusal returns it before any new effects.

    Hook after original prologue config_st and before zero_row construction.
    A scope conflict after admission is a compilation error, never a partially
    emitted fallback. Pool/alternate producers remain outside this binding.
    """
    mode=selected() if mode is None else mode
    if mode=='off':return builder
    if mode not in VALUES:raise ValueError('unknown convolution configuration scope')
    try:
        from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
        if type(plan) is not CountedConvolutionPlan:raise ValueError('requires original typed counted convolution plan')
        plan.__post_init__()
        if type(co) is not int or co!=plan.output_channels:raise ValueError('CO differs from original plan')
        if 'maxpool' in command.get('attributes',{}).get('epilogue',()):raise ValueError('pool helper configuration scope not admitted')
        law=qualify(root)
    except (ValueError,KeyError,TypeError,IndexError,StopIteration,OSError) as e:
        owner=builder
        while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
        if not hasattr(owner,'conv_configuration_refusals'):owner.conv_configuration_refusals=[]
        owner.conv_configuration_refusals.append(str(e));return builder
    return RhsConfigurationScope(builder,plan,command,co,law)


def verify_streams(before,after,co):
    """Full concrete terminal-stream projection, no sampled loops or commands.

    Events are (funct,rs1,rs2), or ('fence',). Actual LLVM CFG execution supplies
    these words. Qualify target state independently before calling this helper.
    Every non-LD1 command/completion is compared exactly, each consumer sees
    exactly the original LD1 words, and final LD1 state is unchanged.
    """
    expected=encode_config_ld(co,id=1)
    def project(stream):
        state=None;events=[];consumers=[];setters=0
        for event in stream:
            if event[0]==0 and event[1]&3==1 and (event[1]>>3)&3==1:
                if tuple(event[1:])!=expected:raise ValueError('stream contains conflicting complete RHS words')
                state=tuple(event[1:]);setters+=1
            else:
                events.append(event)
                if event[0]==1:
                    if state!=expected:raise ValueError('RHS consumer lacks original complete state')
                    consumers.append(state)
                elif event[0]==7 and state is None:pass  # Original kernel prefix FLUSH, before this source scope.
                elif event[0]!='fence' and event[0] not in (0,2,3,4,5,6,14):raise ValueError('unknown terminal effect may invalidate persistence')
                elif event[0]==0 and (event[1]&3 not in (0,1,2) or event[1]&3==1 and (event[1]>>3)&3==3):raise ValueError('unknown CONFIG effect')
        return events,consumers,state,setters
    b=project(before);a=project(after)
    if b[:3]!=a[:3] or a[3]!=1:raise ValueError('RHS scope changed consumer configuration, nonconfiguration effects/completion, or final state')
    return dict(schema='gemmini.convolution_rhs_configuration_stream.v1',non_rhs_config_events=len(a[0]),rhs_consumers=len(a[1]),before_setters=b[3],after_setters=a[3],non_rhs_effects_sha256=sha256(repr(a[0]).encode()).hexdigest(),consumer_configuration_sha256=sha256(repr(a[1]).encode()).hexdigest(),scope='complete actual terminal streams and every RHS consumer; no timing conclusion')
