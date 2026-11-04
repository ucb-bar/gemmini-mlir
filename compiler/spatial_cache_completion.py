"""Default-off selected-fact completion binding for original Q1 cache reloads."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
from dataclasses import asdict
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.spatial_cache_epochs import SpatialCacheEpochLaw,emit_spatial_cache_epochs
from merlin.compile.scheduling.spatial_readout_cache import emit_cached_spatial_max
from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
from merlin.compile.scheduling.program_limits import EncodedAccess,AddressWindow
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.sched.ir import Kernel,Call,Const,concretize

VALUES=('off','reload_epochs')
DECLARATION={'values':list(VALUES),'mechanism':'checked spatial_cache_epochs original Q1 cache effects with disjoint ACC reloads and real completion before each consumer', 'note':'Default off. Original producer primitives remain synchronous. No timing claim.'}
def selected():return knob('spatial_cache_completion',default='off',values=VALUES,root=Path(__file__).parent)

class Integers:
    def op(self,n,x,y,typ='i64'):
        return {'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'udiv':lambda:(x%(1<<64))//y,'urem':lambda:(x%(1<<64))%y}[n]()
    def cmp(self,n,x,y):return {'eq':lambda:x==y,'slt':lambda:x<y}[n]()
    def select(self,c,x,y):return x if c else y
    def branch(self,c,yes,no):return yes() if c else no()
    def loop(self,start,stop,step,body):
        for i in range(start,stop,step):body(i)

@lru_cache(maxsize=4)
def _selected():
    root=Path(__file__).parent
    fact=(root/'facts.json').read_bytes();s=json.loads((root/'command_ordering.json').read_text());r=s['ordering']
    h=sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
    if s['base_facts']['sha256']!=sha256(fact).hexdigest() or h!=s['ordering_sha256'] or r['status']!='derived':
        raise ValueError('cache epochs require intact selected facts and ordering')
    queues={int(code):queue for item in r['funct_queue'] for code,queue in item.get('by_code',{}).items()}
    load=[item.get('queue') for item in r['funct_queue'] if item['funct']==14]
    if queues.get(1)!='load' or load!=['load'] or r['same_queue']['load']['release'] not in ('issue','completion'):
        raise ValueError('cache epochs lack selected CONFIG_LD/MVIN3 ordered load lane')
    f=json.loads(fact)['facts'];entry=f['timing_structure']['command_classes']['classes']['load']['entries']
    spaces={v['name']:v for v in f['memory_address_spaces']}
    if entry['status']!='derived' or type(entry['value']) is not int or entry['value']<=0 or spaces['accumulator']['row_bytes']!=64:
        raise ValueError('cache epochs lack selected finite load queue or exact ACC row geometry')
    return sha256(fact).hexdigest(),h,entry['value'],spaces['accumulator']['bytes']

def selected_law(plan):
    plan.revalidate();fact,ordering,pending,capacity=_selected()
    if plan.cache.producer.start+plan.cache.producer.size>plan.cache.cache.start or plan.cache.cache.start+plan.cache.cache.size>capacity:
        raise ValueError('actual producer/cache windows overlap or exceed selected capacity')
    return SpatialCacheEpochLaw(fact,ordering,sha256(json.dumps(plan.proof,sort_keys=True).encode()).hexdigest(),plan.cache.identity,
        'invocation-private-Q1-ring-and-disjoint-ACC-until-closing-fence',pending,
        fixed_load_configuration=True,finite_load_queue_progress=True,
        complete_reload_drain=True,complete_store_drain=True,
        immutable_q1_until_drain=True,exclusive_producer_cache_windows=True)

@lru_cache(maxsize=32)
def _check(cache,law):
    law.require(cache);p=cache.source.packed.source.producer;s=cache.source.packed.source.spatial
    oh,ow=s.output_hw;tile=cache.source.packed.source.output_tile_width
    reports=[];comparisons=0;maximum=0
    for output_x in range(0,ow,tile):
        pixels=min(tile,ow-output_x);original=[];actual=[]
        for py in range(oh):
            kw=dict(image=0,pool_y=py,output_x=output_x,output_pixels=pixels)
            emit_cached_spatial_max(cache,Integers(),lambda k,f:original.append((k,f)),synchronous_completion=True,**kw)
            emit_spatial_cache_epochs(cache,Integers(),lambda k,f:actual.append((k,f)),law=law,**kw)
        payload=[(k,f) for k,f in actual if k in ('reload_intermediate','store_spatial_max')]
        if original!=payload:raise ValueError('cache epoch effects differ from original complete shared cache emitter')
        comparisons+=len(payload)
        # Decode the actual emitted effects, not the proposed cache intervals.
        body=[];effects={};objects={};pending=[];serial=0
        def add(kind,accesses=(),waits=()):
            nonlocal serial,maximum
            name='e'+str(serial);serial+=1
            if waits:effect=StorageEffects(kind,completion='wait',waits=tuple(waits))
            else:
                effect=StorageEffects(kind,tuple(accesses),completion='asynchronous',token=name)
                pending.append(name);maximum=max(maximum,len(pending))
            body.append(Call(name,()));effects[name]=effect
        for kind,f in actual:
            if kind=='begin_reload_epoch':
                if pending:raise ValueError('reload epoch entered with outstanding accesses')
            elif kind=='reload_intermediate':
                accesses=[]
                for row in range(f['rows']):
                    local=(f['accumulator_row']+row)*64;size=f['columns']*4
                    accesses.extend((EncodedAccess('ring',f['scratch_offset']+row*p.output_channels,f['columns'],False),EncodedAccess('acc',local,size,True)))
                    if not cache.cache.start<=local<local+size<=cache.cache.start+cache.cache.size:
                        raise ValueError('decoded cache write exceeds actual owned ACC window')
                    objects.setdefault(local,TemporaryObject('row'+str(local),'acc',local,size,4))
                add(kind,accesses)
            elif kind=='store_spatial_max':
                reads=[EncodedAccess('acc',(f['accumulator_row']+row)*64,f['columns']*4,False) for row in range(3*f['local_width'])]
                writes=[EncodedAccess('output',(f['output_row']+x)*p.output_channels+f['output_column'],f['columns'],True) for x in range(f['output_pixels'])]
                add(kind,reads+writes)
            else:
                if kind not in ('complete_reload_epoch','complete_store_epoch'):raise ValueError('unknown epoch effect')
                if not pending:raise ValueError('empty or duplicate epoch completion')
                add(kind,waits=pending);pending=[]
        if pending:raise ValueError('cache epoch returns with outstanding accesses')
        kernel=Kernel('spatial_cache_reload_epochs',(),tuple(body))
        adapter=StorageAdapter(lambda call:effects[call.instr],lambda call,accesses:call,
            law.hardware_identity,law.numerical_identity,law.ordering_identity)
        report=verify_temporary_storage(kernel,kernel,tuple(objects.values()),
            (StorageBank('acc','physical_accumulator',p.accumulator_rows*64,'bytes',4),),adapter,
            external_windows=(AddressWindow('ring',cache.source.packed.workspace.start,cache.source.required_bytes),
                AddressWindow('output',0,p.window.input_shape[0]*oh*ow*p.output_channels)),
            private_ownership=True,max_calls=16384,max_objects=2048)
        reports.append(dict(output_x=output_x,pixels=pixels,kernel_sha256=kernel.digest(),storage=report))
    import sys
    library=Path(sys.modules[emit_spatial_cache_epochs.__module__].__file__)
    return dict(shared_epoch_module_sha256=sha256(library.read_bytes()).hexdigest(),
        original_effects_compared=comparisons,maximum_issued_loads_or_stores=maximum,
        pending_load_capacity=law.pending_load_capacity,storage_checks=reports,
        source_ring_lifetime='producer/Q1 stores complete before entry; no producer callback inside consumer; full store completion before return',
        completion='all disjoint reloads; real full drain; original hardware max store; real full drain',
        ordering='No load completion order is assumed; decoded pending writes are disjoint and every token is closed before each consumer. Stores are individually completed.',
        cycles='UNKNOWN; no issue-count performance inference')

def prepare(plan):
    law=selected_law(plan)
    return law,_check(plan.cache,law)

def emit(plan,builder,lower,**fields):
    law,proof=prepare(plan)
    emit_spatial_cache_epochs(plan.cache,builder,lower,law=law,**fields)
    return proof
