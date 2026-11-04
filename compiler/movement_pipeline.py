"""Optional disjoint-slot movement epochs with real global completions.

Shared copy_pipeline is not applicable: selected software provides no qualified
pure-token wait/concurrent-copy law. This route instead uses shared temporary
storage verification for two private slot lifetimes. It batches existing loads,
drains globally, issues original stores in their original order, then drains.
It does not claim load/store overlap or a performance result.
"""
from dataclasses import replace
from pathlib import Path
from functools import lru_cache
from hashlib import sha256
import json
from merlin.sched.ir import Call,Loop,Kernel,Const,Var,Select,Sum,Scaled,concretize,instances,check_structure
from merlin.compile.scheduling.tile_load_coalescing import _replace_call,TileLoadCoalescing
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
from merlin.compile.scheduling.program_limits import EncodedAccess
from merlin.semantic_compiler.allocation_types import StorageBank


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('movement_completion_epochs',default='off',values=('off','slots2'),root=Path(__file__).resolve().parent)


def _selection():
    from elementwise_tiles import _selected
    root=Path(__file__).resolve().parent
    spaces,ordering,facts=_selected(str(root))
    return spaces['scratchpad']['bytes'],ordering,facts


@lru_cache(maxsize=1)
def _slot_proof():
    capacity,ordering,facts=_selection()
    if capacity<2048:raise ValueError('two slots exceed scratchpad capacity')
    loads=tuple(Call('reserve_load_slot',(('slot',Const(s)),('local_row',Const(s*64)))) for s in range(2))
    stores=tuple(Call('consume_store_slot',c.args) for c in loads)
    kernel=Kernel('movement_epoch_slot_ownership',(),(*loads,Call('loads_complete',()),*stores,Call('stores_complete',())))
    def project(call):
        if call.instr.endswith('_complete'):
            prefix='L' if call.instr=='loads_complete' else 'S'
            return StorageEffects(call.instr,completion='wait',waits=tuple(prefix+str(s) for s in range(2)))
        a=concretize(call,{});load=call.instr=='reserve_load_slot'
        return StorageEffects(call.instr,(EncodedAccess('spad',a['local_row']*16,1024,load),),completion='asynchronous',token=('L' if load else 'S')+str(a['slot']))
    adapter=StorageAdapter(project,lambda call,accesses:call,'gemmini:'+facts,'byte-copy-no-arithmetic','disjoint-full-slot-ownership-with-global-phase-completion')
    proof=verify_temporary_storage(kernel,kernel,tuple(TemporaryObject('slot'+str(s),'spad',s*1024,1024,16) for s in range(2)),(StorageBank('spad','physical_scratchpad',capacity,'bytes',16),),adapter,private_ownership=True,max_calls=16,max_objects=2)
    return dict(storage=proof,ownership_kernel_sha256=kernel.digest(),ordering_sha256=ordering,facts_sha256=facts,
        interpretation='abstract whole-slot ownership; actual DMA cells are independently checked subsets',
        copy_pipeline={'status':'not_applicable','consumed':False,'reason':'missing selected pure-token wait and original-owner concurrent/crossing-copy qualification'})


def _has_select(expr,var):
    if isinstance(expr,Select):return expr.var==var or _has_select(expr.then,var) or _has_select(expr.other,var)
    if isinstance(expr,Sum):return any(_has_select(x,var) for x in expr.terms)
    if isinstance(expr,Scaled):return _has_select(expr.expr,var)
    return False


def _clear_last_select(expr,var,prefix):
    if isinstance(expr,Select):
        if expr.var==var:
            if expr.equals<prefix:raise ValueError('cannot remove an interior row selector')
            return _clear_last_select(expr.other,var,prefix)
        return replace(expr,then=_clear_last_select(expr.then,var,prefix),other=_clear_last_select(expr.other,var,prefix))
    if isinstance(expr,Sum):return replace(expr,terms=tuple(_clear_last_select(x,var,prefix) for x in expr.terms))
    if isinstance(expr,Scaled):return replace(expr,expr=_clear_last_select(expr.expr,var,prefix))
    return expr


def epoch_kernel(source):
    """Pair adjacent existing rectangles, retaining all original store order."""
    _slot_proof();changed=0
    def walk(nodes):
        nonlocal changed
        output=[]
        for node in nodes:
            if not isinstance(node,Loop):output.append(node);continue
            body=node.body
            eligible=(len(body)==4 and all(isinstance(c,Call) for c in body)
                and tuple(c.instr for c in body)==('load_bytes','complete','store_bytes','complete') and node.extent>=2
                and not(node.iter_args or node.yields or node.results))
            if not eligible:
                output.append(replace(node,body=walk(body)));continue
            selected=any(_has_select(v,node.var) for c in body for _,v in c.args)
            prefix=((node.extent-1)//2*2) if selected else node.extent//2*2
            if not prefix:output.append(node);continue
            clean=tuple(replace(c,args=tuple((k,_clear_last_select(v,node.var,prefix)) for k,v in c.args)) for c in body)
            group=Var(node.var+'_epoch')
            def effect(call,index,slot):
                c=_replace_call(call,node.var,index)
                return replace(c,args=(*c.args,('local_row',Const(slot*64))))
            loads=tuple(effect(clean[0],group*2+s,s) for s in range(2))
            stores=tuple(effect(clean[2],group*2+s,s) for s in range(2))
            output.append(Loop(group.name,prefix//2,(*loads,Call('complete',()),*stores,Call('complete',()))))
            for index in range(prefix,node.extent):output.extend(_replace_call(c,node.var,Const(index)) for c in body)
            changed+=1
        return tuple(output)
    kernel=replace(source,name=source.name+'_slots2',body=walk(source.body))
    if check_structure(kernel):raise ValueError('invalid movement epoch Kernel')
    verify_epoch_mapping(source,kernel)
    return kernel,changed


class _Control:
    def __init__(self,b):self.b=b
    def op(self,opcode,a,c):return getattr(self.b,opcode)(a,c)
    def cmp(self,p,a,c):return self.b.icmp(p,a,c)
    def select(self,c,a,d):return self.b.select(c,a,d)
    def loop(self,start,stop,step,callback):
        with self.b.counted_loop(start,stop,step) as i:callback(i)


def emit_if_enabled(builder,plan,ptrs):
    if setting()=='off':return None
    plan.revalidate()
    if ptrs[plan.source]==ptrs[plan.destination]:return None
    kernel,changed=epoch_kernel(plan.kernel)
    if not changed:return None
    mapping_proof=verify_epoch_mapping(plan.kernel,kernel)
    rowwise=hasattr(plan,'source_pitch')
    sp=plan.source_pitch if rowwise else 64;dp=plan.destination_pitch if rowwise else 64
    builder.fence();builder.config_ex();builder.config_ld(sp,block_stride=16,id=0);builder.config_st(dp)
    def effect(name,a):
        if name=='complete':builder.fence();return
        local=a.get('local_row',0)
        if rowwise:offset=a['source_byte'] if name=='load_bytes' else a['destination_byte']
        else:offset=builder.add(plan.source_byte if name=='load_bytes' else plan.destination_byte,a['byte_offset'])
        if name=='load_bytes':builder.mvin(builder.ptr_offset(ptrs[plan.source],offset),local,a['columns'],a['rows'],id=0)
        elif name=='store_bytes':builder.mvout(builder.ptr_offset(ptrs[plan.destination],offset),local,a['columns'],a['rows'])
        else:raise ValueError('unknown movement epoch effect')
    receipt=emit_index_program(kernel,_Control(builder),effect)
    builder.fence();plan.revalidate()
    return dict(plan.report(),epoch_mechanism='disjoint_slots_global_completion',epoch_kernel_sha256=kernel.digest(),
        paired_loop_sites=changed,private_scratchpad_rows=128,shared_slot_ownership=_slot_proof(),actual_mapping_check=mapping_proof,
        emission=receipt,performance='UNKNOWN; no matched complete-call timing')


def execute_epoch(plan,source_bytes,destination_bytes):
    """Development interpretation with actual pending-slot hazard rejection."""
    plan.revalidate();kernel,_=epoch_kernel(plan.kernel);out=bytearray(destination_bytes)
    rowwise=hasattr(plan,'source_pitch');sp=plan.source_pitch if rowwise else 64;dp=plan.destination_pitch if rowwise else 64
    local={};pending=[];writes=set();stores=[]
    for call,env in instances(kernel):
        a=concretize(call,env)
        if call.instr=='complete':pending=[];continue
        slot=a.get('local_row',0);mapping=TileLoadCoalescing(a['rows'],a['columns'],16,64,slot,16,128)
        cells={mapping.location(r,c) for r in range(a['rows']) for c in range(a['columns'])}
        load=call.instr=='load_bytes'
        if any(cells&old and (load or writing) for old,writing in pending):raise ValueError('movement epoch slot hazard')
        pending.append((cells,load))
        offset=(a['source_byte'] if load else a['destination_byte']) if rowwise else (plan.source_byte if load else plan.destination_byte)+a['byte_offset']
        for r in range(a['rows']):
            for c in range(a['columns']):
                cell=mapping.location(r,c);byte=offset+r*(sp if load else dp)+c
                if load:local[cell]=source_bytes[byte]
                else:
                    if byte in writes:raise ValueError('movement epoch writes output byte twice')
                    writes.add(byte);out[byte]=local[cell]
        if not load:stores.append({k:v for k,v in a.items() if k!='local_row'})
    originalstores=[concretize(c,e) for c,e in instances(plan.kernel) if c.instr=='store_bytes']
    if pending or stores!=originalstores:raise ValueError('movement epoch changed store order or final completion')
    return bytes(out)


def _poly_add(left,right,factor=1):
    terms=dict(left[1])
    for atom,coefficient in right[1]:terms[atom]=terms.get(atom,0)+factor*coefficient
    return (left[0]+factor*right[0],tuple(sorted(((a,c) for a,c in terms.items() if c),key=lambda x:repr(x[0]))))


def _poly_scale(poly,factor):
    return (poly[0]*factor,tuple((a,c*factor) for a,c in poly[1] if c*factor))


def _poly_range(poly,domains):
    low=high=poly[0]
    for atom,coefficient in poly[1]:
        if atom[0]=='var':
            if atom[1] not in domains:raise ValueError('unbound movement index '+atom[1])
            lo,hi=domains[atom[1]]
        elif atom[0]=='select':
            a,b=_poly_range(atom[3],domains),_poly_range(atom[4],domains)
            lo,hi=min(a[0],b[0]),max(a[1],b[1])
        else:raise ValueError('unsupported movement polynomial atom')
        low+=coefficient*(lo if coefficient>=0 else hi)
        high+=coefficient*(hi if coefficient>=0 else lo)
    return low,high


def _normal(expr,substitutions,domains):
    """Independent affine/selector normal form, bounded by counted domains."""
    if type(expr) is int:return expr,()
    if type(expr) is Const:return expr.value,()
    if type(expr) is Var:
        return substitutions.get(expr.name,(0,((('var',expr.name),1),)))
    if type(expr) is Sum:
        value=(0,())
        for term in expr.terms:value=_poly_add(value,_normal(term,substitutions,domains))
        return value
    if type(expr) is Scaled:return _poly_scale(_normal(expr.expr,substitutions,domains),expr.factor)
    if type(expr) is Select:
        condition=_normal(Var(expr.var),substitutions,domains);lo,hi=_poly_range(condition,domains)
        if lo==hi==expr.equals:return _normal(expr.then,substitutions,domains)
        if expr.equals<lo or expr.equals>hi:return _normal(expr.other,substitutions,domains)
        yes=_normal(expr.then,substitutions,domains);no=_normal(expr.other,substitutions,domains)
        if yes==no:return yes
        return 0,((('select',condition,expr.equals,yes,no),1),)
    raise ValueError('unsupported movement index expression '+type(expr).__name__)


def _check_dma_domains(kernel,*,original=False):
    """Check all static call domains; no dynamic instance expansion occurs."""
    capacity,_,_=_selection();calls=0
    def walk(nodes,domains):
        nonlocal calls
        for node in nodes:
            if type(node) is Loop:
                if type(node.extent) is not int or node.extent<=0 or node.var in domains or node.iter_args or node.yields or node.results:
                    raise ValueError('unsupported movement counted domain')
                walk(node.body,dict(domains,**{node.var:(0,node.extent-1)}));continue
            if type(node) is not Call:raise ValueError('unsupported movement schedule node')
            if node.instr=='complete':
                if node.args:raise ValueError('movement completion acquired operands')
                continue
            if node.instr not in ('load_bytes','store_bytes'):raise ValueError('unknown movement DMA effect')
            fields=dict(node.args)
            if len(fields)!=len(node.args):raise ValueError('duplicate movement DMA operands')
            if original and 'local_row' in fields:raise ValueError('original movement has unsupported local relocation')
            required={'rows','columns'}
            mapping={'byte_offset'} if 'byte_offset' in fields else {'source_byte','destination_byte'}
            if set(fields) not in (required|mapping,required|mapping|{'local_row'}):
                raise ValueError('unsupported movement DMA mapping operands')
            ranges={key:_poly_range(_normal(value,{},domains),domains) for key,value in fields.items()}
            r,c=ranges['rows'],ranges['columns'];local=ranges.get('local_row',(0,0))
            if not (1<=r[0]<=r[1]<=16 and 1<=c[0]<=c[1]<=64):raise ValueError('movement DMA extent exceeds16rows/64columns')
            if local[0]!=local[1] or local[0] not in (0,64):raise ValueError('movement DMA slot must be exactly row0 or64')
            high=local[0]+((c[1]-1)//16)*16+r[1]
            if high>128 or high*16>capacity:raise ValueError('movement DMA leaves128-row physical ownership')
            TileLoadCoalescing(r[1],c[1],16,64,local[0],16,128)
            for key in mapping:
                if not 0<=ranges[key][0]<=ranges[key][1]<(1<<63):raise ValueError('movement address domain outside signed i64')
            calls+=1
    walk(kernel.body,{})
    return calls


def verify_epoch_mapping(source,candidate):
    """Finite symbolic source-to-emitted coverage and store-order checker.

    Each paired loop covers indices2e and2e+1 exactly once in ascending store
    order. Remaining one/two source iterations are explicitly compared under
    constant substitutions, which independently resolve peeled selectors.
    All original byte expressions are compared in a separate normal form.
    """
    if replace(candidate,name=source.name,body=source.body)!=source:
        raise ValueError('movement epoch changed source boundary or kernel metadata')
    original_calls=_check_dma_domains(source,original=True)
    actual_calls=_check_dma_domains(candidate)
    paired=0
    def same_call(original,actual,substitutions,domains,slot=0):
        if type(actual) is not Call or replace(actual,args=original.args)!=original:
            raise ValueError('movement epoch changed an original effect or completion')
        expected=dict(original.args);fields=dict(actual.args)
        local=fields.pop('local_row',Const(0))
        if _normal(local,{},domains)!=(slot,()):raise ValueError('movement epoch selected wrong private slot')
        if fields.keys()!=expected.keys():raise ValueError('movement epoch changed byte mapping operands')
        for key,value in expected.items():
            if _normal(value,substitutions,domains)!=_normal(fields[key],{},domains):
                raise ValueError('movement epoch changed source byte/extent expression '+key)
    def walk(before,after,domains):
        nonlocal paired
        position=0
        for old in before:
            if position>=len(after):raise ValueError('movement epoch dropped original effects')
            new=after[position];position+=1
            if type(old) is Call:
                same_call(old,new,{},domains);continue
            if type(old) is not Loop or type(new) is not Loop:raise ValueError('movement epoch changed counted structure')
            if new.var==old.var:
                if replace(new,body=old.body)!=old:raise ValueError('movement epoch changed original loop domain')
                walk(old.body,new.body,dict(domains,**{old.var:(0,old.extent-1)}));continue
            if (new.var!=old.var+'_epoch' or new.iter_args or new.yields or new.results
                or len(old.body)!=4 or tuple(c.instr for c in old.body if type(c) is Call)!=('load_bytes','complete','store_bytes','complete')
                or len(new.body)!=6 or new.extent<=0):raise ValueError('unsupported movement epoch loop rewrite')
            prefix=new.extent*2
            if not 0<prefix<=old.extent or old.extent-prefix>2:raise ValueError('movement epoch counted coverage has gap or overlap')
            domain=dict(domains,**{new.var:(0,new.extent-1)})
            for slot in range(2):
                substitution={old.var:(slot,((('var',new.var),2),))}
                same_call(old.body[0],new.body[slot],substitution,domain,slot*64)
                same_call(old.body[2],new.body[3+slot],substitution,domain,slot*64)
            same_call(old.body[1],new.body[2],{},domain)
            same_call(old.body[3],new.body[5],{},domain)
            for index in range(prefix,old.extent):
                for call in old.body:
                    if position>=len(after):raise ValueError('movement epoch omitted peeled tail')
                    same_call(call,after[position],{old.var:(index,())},domains);position+=1
            paired+=1
        if position!=len(after):raise ValueError('movement epoch inserted unowned effects')
    walk(source.body,candidate.body,{})
    return dict(schema='authored.checked_movement_epoch_mapping.v1',source_kernel_sha256=source.digest(),
        candidate_kernel_sha256=candidate.digest(),paired_loop_sites=paired,
        original_static_dma_sites=original_calls,actual_static_dma_sites=actual_calls,
        coverage='symbolic2e/2e+1 partition plus all peeled constant indices; original store order',
        local_subset='every actual call rows1..16,cols1..64,localrow0/64; block-mapped highwater<=128rows',
        checking='finite per-static-call interval/selector normal forms; no dynamic instance expansion')
