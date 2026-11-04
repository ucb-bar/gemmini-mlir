"""Default-off complete LD0 scope for proven uniform, valid NHWC windows.

The zero-source branch remains in emitted IR with its original setup. Exact
window/partition bounds prove that branch unreachable, including every tail.
This scope grants no new memory, array, configuration partial-write or fence law.
"""
import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
from merlin.compile.scheduling.configuration import ConfigurationAdapter, ConfigurationEffects, verify_configuration_state
from merlin.sched.ir import Call, Const, Kernel
from isa import encode_config_ld

WINDOW_TRANSFER_SHA256='3ff4d8843e775d2ee3297c44652d4a2a381b43bbaa507be169a822155e13abdc'
VALUES=('off','uniform_valid_scope')
DECLARATION=dict(values=list(VALUES),mechanism='source-bound complete invariant LD0 setup before original counted convolution loops',note='Default off. No padding or pooling; exact valid-window and channel-tail proof. Original unreachable zero-source setup remains in IR. Every transfer, other configuration and completion is preserved; no timing claim.')

def selected():return knob('conv_lhs_configuration',default='off',values=VALUES,root=Path(__file__).parent)

def source_proof(plan,command):
    if type(plan) is not CountedConvolutionPlan:raise ValueError('requires original typed counted convolution plan')
    plan.__post_init__();w=plan.window;a=command.get('attributes',{})
    if command.get('opcode') not in ('CONV','CONV2D'):raise ValueError('requires original convolution source')
    if any(w.padding):raise ValueError('padded windows have reachable zero-source LD0 writes')
    if 'maxpool' in a.get('epilogue',()) or any(k.startswith('pool_') for k in a):raise ValueError('pool helper effects not admitted')
    kernel=tuple(a.get('kernel',()))
    if (kernel[:2]!=w.kernel or len(kernel) not in (2,4) or len(kernel)==4 and kernel[2:]!=(w.input_shape[-1],plan.output_channels)
        or str(a.get('layout','nhwc')).lower()!='nhwc' or tuple(a.get('stride',(1,1)))!=w.stride
        or tuple(a.get('dilation',(1,1)))!=w.dilation or tuple(a.get('padding',(0,0,0,0)))!=w.padding):raise ValueError('attributes differ from exact original window')
    b,h,width,ci=w.input_shape;oh,ow=w.output_spatial;kh,kw=w.kernel;sy,sx=w.stride;dy,dx=w.dilation
    ymax=(oh-1)*sy+(kh-1)*dy;xmax=(ow-1)*sx+(kw-1)*dx
    if not (0<=ymax<h and 0<=xmax<width):raise ValueError('source window may reach halo')
    # Every full/tail output_x partition stays in [0,OW); each reduction
    # channel partition satisfies c+depth<=CI. The original library emits
    # exactly these partitions; therefore BOTH run and one-row predicates hold.
    return dict(plan_sha256=plan.identity,window=repr(w),row_stride_elements=ci*sx,
        reachable_source_ranges=dict(batch=[0,b-1],input_y=[0,ymax],input_x=[0,xmax],input_channel=[0,ci-1]),
        partitions=plan.coverage['partitions'],proof='valid-window extrema plus exact full/tail partitions imply every run and individual row predicate; zero-source branch unreachable',zero_source_branch='retained unchanged in emitted IR, unreachable for all admitted runtime iterations')

def footprint(root):
    source=(root/'convolution.py').read_text();tree=ast.parse(source)
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lower_conv')
    sites=[]
    for n in ast.walk(fun):
        if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute) or not isinstance(n.func.value,ast.Name) or n.func.value.id!='fb' or n.func.attr!='config_ld':continue
        kw={k.arg:k.value for k in n.keywords};state=kw.get('id',ast.Constant(0))
        if not isinstance(state,ast.Constant) or state.value not in (0,1,2):raise ValueError('dynamic LD context in source callback')
        if state.value==0:
            if len(n.args)!=1 or ast.dump(n.args[0])!=ast.dump(ast.parse("f['row_stride_elements']",mode='eval').body) or set(kw)!={'id'}:raise ValueError('unexpected source LD0 setter')
            sites.append(n.lineno)
    input_branches=[n for n in ast.walk(fun) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(ast.parse("kind=='load_input'",mode='eval').body)]
    if len(input_branches)!=1:raise ValueError('original input-effect branch missing/ambiguous')
    original="fb.config_ld(f['row_stride_elements'],id=0)\nptr=zero_row if f['zero'] else fb.ptr_offset(pointers[src],f['first_element'])\nfb.mvin(ptr,f['local_row'],cols=f['columns'],rows=f['rows'],id=0)"
    actual=ast.dump(ast.Module(body=input_branches[0].body,type_ignores=[]))
    expected=[ast.dump(ast.parse(text)) for text in (original,original.replace("fb.config_ld(f['row_stride_elements'],id=0)",'configure_input(fb,f)'))]
    if actual not in expected:raise ValueError('original input address/consumer effects changed')
    # The authored hook replaces this one exact call with configure_input(fb,f).
    hooked=[n for n in ast.walk(fun) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='configure_input']
    if any(len(n.args)!=2 or n.keywords or ast.dump(n.args[0])!=ast.dump(ast.Name(id='fb',ctx=ast.Load())) or ast.dump(n.args[1])!=ast.dump(ast.Name(id='f',ctx=ast.Load())) for n in hooked):raise ValueError('input configuration hook differs from original fields')
    if len(sites)+len(hooked)!=1:raise ValueError('requires one checked original input configuration site')
    return dict(convolution_sha256=sha256(source.encode()).hexdigest(),input_setter_lines=sites,hooked_input_sites=len(hooked))

def qualify(root=None):
    root=Path(root) if root is not None else Path(__file__).parent
    # Existing selected law checks facts digests, queue/state bank separation,
    # full persistent register writes, preserving commands and source inventory.
    from conv_configuration import qualify as rhs_qualify
    law=rhs_qualify(root)
    order=json.loads((root/'command_ordering.json').read_text())['ordering']
    if [row.get('queue') for row in order['funct_queue'] if row['funct']==2]!=['load']:raise ValueError('LD0 scope lacks selected ordered CONFIG_LD/MVIN capture')
    from isa_configuration import _project,_FENCE_ADAPTER
    if sha256(Path(inspect.getsourcefile(_project)).read_bytes()).digest()!=sha256((root/'isa_configuration.py').read_bytes()).digest():raise ValueError('loaded configuration law differs from selected source')
    def terminal(f,a,b):return Call('terminal',tuple((k,Const(v)) for k,v in [('funct',f),('static_mask',3),('rs1',a),('rs2',b)]))
    words=encode_config_ld(37,id=0);setup=terminal(0,*words);e=_project(setup)
    if e is None or not e.ordered or not e.pure_write or dict(e.writes)!={'ld0.rs1':words[0],'ld0.rs2':words[1]}:raise ValueError('complete persistent LD0 law missing')
    events=[setup]
    for call in [terminal(0,*encode_config_ld(11,id=1)),terminal(0,*encode_config_ld(0,id=2)),*(terminal(f,0,0) for f in (1,2,3,4,5,6,14)),Call('cpu_fence',())]:
        e=_project(call)
        if e is None or not e.ordered or set(e.clobbers)&{'ld0.rs1','ld0.rs2'} or any(k.startswith('ld0.') for k,v in e.writes):raise ValueError('intervening effect may clobber LD0')
        events.append(call)
    def project(call):
        args={k:v.value for k,v in call.args}
        if call.instr=='terminal' and args['funct']==2:return ConfigurationEffects(reads=(('ld0.rs1',words[0]),('ld0.rs2',words[1])),ordered=True)
        return _project(call)
    adapter=ConfigurationAdapter(project,_FENCE_ADAPTER.configuration_identity,_FENCE_ADAPTER.qualification_identity)
    proof=verify_configuration_state(Kernel('lhs_scope_reachable_effect_footprint',(),tuple(events+[terminal(2,0,0)])),adapter)
    import merlin.compile.scheduling.window_transfer as module
    if sha256(Path(module.__file__).read_bytes()).hexdigest()!=WINDOW_TRANSFER_SHA256:raise ValueError('window transfer source changed; valid-partition proof requires review')
    return dict(selected_persistence=law,shared_ld0_state_proof=proof,footprint=footprint(root),window_transfer_sha256=sha256(Path(module.__file__).read_bytes()).hexdigest(),scope='complete LD0 words across selected preserving commands; no partial writes, no interior flush, no completion removed')

class LhsConfigurationScope:
    def __init__(self,builder,plan,command,proof,law):
        self.builder=builder;self.stride=proof['row_stride_elements'];self.words=encode_config_ld(self.stride,id=0);self.owner=self._owner();self.pending=None
        self.receipt=dict(schema='gemmini.convolution_lhs_configuration_scope.v1',mode='uniform_valid_scope',source=proof,source_attributes=command.get('attributes',{}).copy(),qualification=law,full_ld0_words=list(self.words),hoisted_setters=1,suppressed_static_sites=0,retained_unreachable_zero_sites=0,static_input_consumers=0,completion='all original fences unchanged',timing='UNKNOWN')
        builder.config_ld(self.stride,id=0)
        owner=builder
        while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
        if not hasattr(owner,'conv_lhs_configuration_receipts'):owner.conv_lhs_configuration_receipts=[]
        owner.conv_lhs_configuration_receipts.append(self.receipt)
    def _owner(self):return tuple(getattr(self.builder,k,None) for k in ('task_index','source_op_index','region_id'))
    def _check(self):
        if self._owner()!=self.owner:raise ValueError('LD0 scope crossed source owner')
    def __getattr__(self,name):
        if name in ('insn','emit','flush','finish','config_norm'):raise ValueError('unqualified terminal access in LD0 scope: '+name)
        return getattr(self.builder,name)
    def configure_input(self,f):
        self._check()
        if self.pending is not None:raise ValueError('previous source input lacks its original consumer')
        if type(f.get('zero')) is not bool or type(f.get('row_stride_elements')) is not int:raise ValueError('input source/stride must be exact original fields')
        zero=f['zero'];stride=f['row_stride_elements']
        if zero:
            if stride!=0 or f.get('rows')!=1 or f.get('first_element')!=0:raise ValueError('malformed unreachable zero-source effect')
            self.builder.config_ld(0,id=0);self.receipt['retained_unreachable_zero_sites']+=1
        else:
            if stride!=self.stride:raise ValueError('actual input fragment has varying stride')
            self.receipt['suppressed_static_sites']+=1
        self.pending=zero
    def config_ld(self,stride,**kwargs):
        self._check();state=kwargs.get('id',0)
        if type(state) is not int or state not in (1,2):raise ValueError('unqualified direct LD0 write/context in invariant scope')
        return self.builder.config_ld(stride,**kwargs)
    def mvin(self,ptr,addr,cols=16,rows=16,id=0):
        self._check()
        if type(id) is not int or id not in (0,1,2):raise ValueError('unqualified input load context')
        if id==0:
            if self.pending is None:raise ValueError('input consumer lacks checked source fragment')
            self.pending=None;self.receipt['static_input_consumers']+=1
        return self.builder.mvin(ptr,addr,cols=cols,rows=rows,id=id)

def configure_input(builder,fields):
    if isinstance(builder,LhsConfigurationScope):return builder.configure_input(fields)
    return builder.config_ld(fields['row_stride_elements'],id=0)

def wrap(builder,plan,command,*,mode=None,root=None):
    mode=selected() if mode is None else mode
    if mode=='off':return builder
    if mode not in VALUES:raise ValueError('unknown LD0 scope mode')
    try:proof=source_proof(plan,command);law=qualify(root)
    except (ValueError,KeyError,TypeError,IndexError,StopIteration,OSError) as e:
        owner=builder
        while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
        if not hasattr(owner,'conv_lhs_configuration_refusals'):owner.conv_lhs_configuration_refusals=[]
        owner.conv_lhs_configuration_refusals.append(str(e));return builder
    return LhsConfigurationScope(builder,plan,command,proof,law)

def verify_streams(before,after,stride):
    expected=encode_config_ld(stride,id=0)
    def project(stream):
        state=None;events=[];consumers=[];setters=0
        for event in stream:
            if event[0]==0 and event[1]&3==1 and (event[1]>>3)&3==0:
                state=tuple(event[1:])
                if state==expected:setters+=1
                else:events.append(event)
            else:
                events.append(event)
                if event[0]==2:
                    if state is None:raise ValueError('input consumer lacks complete LD0 state')
                    consumers.append(state)
                elif event[0]==7 and state is None:pass
                elif event[0]!='fence' and event[0] not in (0,1,3,4,5,6,14):raise ValueError('unknown effect may invalidate LD0 persistence')
                elif event[0]==0 and (event[1]&3 not in (0,1,2) or event[1]&3==1 and (event[1]>>3)&3==3):raise ValueError('unknown CONFIG effect')
        return events,consumers,state,setters
    b=project(before);a=project(after)
    if b[:3]!=a[:3] or a[3]!=1:raise ValueError('changed input consumer state, nonconfiguration effects/completion, or final state')
    return dict(non_ld0_events=len(a[0]),input_consumers=len(a[1]),before_setters=b[3],after_setters=a[3],non_ld0_effects_sha256=sha256(repr(a[0]).encode()).hexdigest(),consumer_configuration_sha256=sha256(repr(a[1]).encode()).hexdigest(),scope='complete actual emitted LLVM execution; no timing conclusion')
