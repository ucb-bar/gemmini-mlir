"""Bounded original-pointer lane DMA groups; no schedule/source/storage rewrite.

Only interior per-lane drains are removed. A closing drain precedes every next
abstract effect. Out-of-order load responses are explicitly permitted, never
used as a readiness signal. The frozen original emitter supplies all commands.
"""
from hashlib import sha256
import json
from pathlib import Path


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('original_row_transport', default='off', values=('off','bounded_lanes'), root=Path(__file__).parent)


def qualification(root=None):
    root=Path(__file__).parent if root is None else Path(root)
    selection=json.loads((root/'command_ordering.json').read_text())
    facts_bytes=(root/'facts.json').read_bytes()
    record=selection['ordering']
    digest=sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    if (record.get('status')!='derived' or digest!=selection['ordering_sha256']
            or sha256(facts_bytes).hexdigest()!=selection['base_facts']['sha256']):
        raise ValueError('lane transport selected ordering/facts identity mismatch')
    classes=json.loads(facts_bytes)['facts']['timing_structure']['command_classes']
    if classes.get('status')!='derived':return None
    limits={}
    for queue,release,order in [('load','issue','out_of_order'),('store','completion','in_order')]:
        if record['same_queue'][queue]['release']!=release or record['completion_order'][queue]['order']!=order:return None
        row=classes['classes'][queue];entries=row['entries'];buffer=row['command_queue']
        if entries.get('status')!='derived' or buffer.get('status')!='derived':return None
        bounds=[entries['value'],buffer['entries_total']]
        if any(type(x) is not int or x<=0 for x in bounds):return None
        # No more commands than either selected intact-command buffer holds.
        # Stores retire at completion; loads may retire in arbitrary order.
        limits[queue]=min(bounds)
    return {'ordering_sha256':digest,'facts_sha256':selection['base_facts']['sha256'],
            'limits':limits,'completion':'existing FunctionBuilder.fence after every bounded lane subgroup and final lane'}


class _Lanes:
    """Project emitted pointer/row operands before changing their completions."""
    def __init__(self,builder,group,pointers,law):
        self._builder=builder;self._group=group;self._pointers=pointers;self._law=law
        self._expr={};self._lane=0;self._kind=None;self._base=None;self._offset=None;self._cols=None
        self._needs_fence=False;self._subgroup=0
    def __setattr__(self,name,value):
        if name.startswith('_'):object.__setattr__(self,name,value)
        else:setattr(self._builder,name,value)
    def __getattr__(self,name):
        value=getattr(self._builder,name)
        if not callable(value):return value
        def call(*args,**kwargs):
            self._idle('intervening effect/control '+name)
            return value(*args,**kwargs)
        return call
    def _idle(self,reason):
        if self._kind is not None or self._needs_fence:raise ValueError('incomplete original lane group: '+reason)
    def _op(self,name,*args):
        value=getattr(self._builder,name)(*args)
        self._expr[(name,value)]=args
        return value
    def add(self,a,b):return self._op('add',a,b)
    def mul(self,a,b):return self._op('mul',a,b)
    def or_(self,a,b):return self._op('or_',a,b)
    def ptr_offset(self,p,o):return self._op('ptr_offset',p,o)
    def _transfer(self,kind,ptr,addr,cols,rows):
        if self._needs_fence:raise ValueError('lane lacks original closing completion')
        if self._kind not in (None,kind):raise ValueError('mixed lane DMA effects')
        if type(rows) is not int or rows!=1:raise ValueError('lane transport requires exact one-row footprint')
        ptr_expr=self._expr.get(('ptr_offset',ptr))
        if ptr_expr is None:raise ValueError('lane pointer lacks emitted original offset projection')
        pointer,offset=ptr_expr
        roster=self._group['lhs' if kind=='load' else 'outputs']
        if pointer!=self._pointers[roster[self._lane]]:raise ValueError('lane original pointer identity/order changed')
        if kind=='store':
            byte_offset=self._expr.get(('mul',offset))
            if byte_offset is None or byte_offset[1]!=4:raise ValueError('lane store lacks exact i32 column byte transport')
            offset=byte_offset[0]
            encoded=self._expr.get(('or_',addr))
            if encoded is None or encoded[1]!=0xa0000000:raise ValueError('lane store lacks exact checked accumulator encoding')
            addr=encoded[0]
        local=self._expr.get(('add',addr))
        if local is None or local[1]!=self._lane:raise ValueError('lane local rows overlap or order changed')
        base=local[0]
        if self._kind is None:
            self._kind=kind;self._base=base;self._offset=offset;self._cols=cols
        elif (base,offset,cols)!=(self._base,self._offset,self._cols):raise ValueError('lane footprint/column extent changed')
        self._lane+=1;self._subgroup+=1;self._needs_fence=True
    def mvin(self,ptr,addr,*,cols,rows,id):
        if id!=0:
            self._idle('RHS load before lane completion')
            return self._builder.mvin(ptr,addr,cols=cols,rows=rows,id=id)
        self._transfer('load',ptr,addr,cols,rows)
        return self._builder.mvin(ptr,addr,cols=cols,rows=rows,id=id)
    def mvout(self,ptr,addr,*,cols,rows):
        self._transfer('store',ptr,addr,cols,rows)
        return self._builder.mvout(ptr,addr,cols=cols,rows=rows)
    def fence(self):
        if self._kind is None:return self._builder.fence()
        if not self._needs_fence:raise ValueError('unexpected interior lane completion')
        final=self._lane==len(self._group['lhs'])
        if final or self._subgroup==self._law['limits'][self._kind]:
            self._builder.fence();self._subgroup=0
        self._needs_fence=False
        if final:self._lane=0;self._kind=None
    def finish(self):self._idle('omitted lane/final completion')


def emit(builder,owner,group,pointers,*,mode=None,root=None):
    from original_row_batching import emit as incumbent
    mode=setting() if mode is None else mode
    if mode not in ('off','bounded_lanes'):raise ValueError('unknown original row transport mode')
    if mode=='off':return incumbent(builder,owner,group,pointers)
    law=qualification(root)
    if law is None:return incumbent(builder,owner,group,pointers)
    owner.revalidate()
    proxy=_Lanes(builder,group,pointers,law)
    result=incumbent(proxy,owner,group,pointers)
    proxy.finish();owner.revalidate()
    return result
