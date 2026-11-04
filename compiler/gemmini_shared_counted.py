"""Counted, source-bound column views of checked shared-input products.

Only a single primitive M stripe is admitted. Its full K input remains private
and immutable until every original member's columns have completed. Checked
bounded full-K views supply all effects; counted loops only translate disjoint
original columns. No checker budget or completion rule is changed.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import copy
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, as_expr, instances, concretize, check_structure
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.compile.scheduling.projection_views import plan_shared_projection_views
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input
from merlin.compile.scheduling.shared_input_verification import verify_shared_input
from gemmini_graph_reuse import (_bridge, _digest, _pack, _project, plan_shared_products)
from gemmini_dense import geometry
from gemmini_readout import emit_readout
import gemmini_encode as E

_FIELDS={'load':('stream','offset','stride','local','rows','cols'),
         'preload':('weight','acc','depth','cols','rows'),
         'compute':('lhs','bd','rows','depth','br','bc','stay'),
         'store':('offset','acc','rows','cols')}

def _call(packet):
    kind,member,*fields=packet
    tag=kind+('_lhs' if member is None else '_'+str(member))
    return Call(tag,tuple((name,as_expr(int(value))) for name,value in zip(_FIELDS[kind],fields,strict=True)))

def _compress(packets, name):
    """Retain exact affine packet sequences as counted loops, checking expansion."""
    if not packets:return ()
    if len(packets)<3:return tuple(_call(p) for iteration in packets for p in iteration)
    # First and last iterations retain zero initialization and ragged K depths.
    first,middle,last=packets[0],packets[1:-1],packets[-1]
    body=[*(_call(p) for p in first)]
    if middle:
        base=middle[0];delta=None
        if len(middle)>1:
            delta=[[int(b)-int(a) for a,b in zip(x[2:],y[2:],strict=True)] for x,y in zip(base,middle[1],strict=True)]
        candidate=[]
        for pos,packet in enumerate(base):
            call=_call(packet);values=[]
            for index,(field,value) in enumerate(call.args):
                step=0 if delta is None else delta[pos][index]
                values.append((field,value+Var(name)*step))
            candidate.append(Call(call.instr,tuple(values)))
        loop=Loop(name,len(middle),tuple(candidate))
        expanded=tuple((c.instr.split('_',1)[0],None if c.instr.endswith('_lhs') else int(c.instr.split('_',1)[1]),*concretize(c,env).values()) for c,env in instances(Kernel('check',(),(loop,))))
        expected=tuple(p for iteration in middle for p in iteration)
        if expanded!=expected:raise ValueError('shared view effects are not affine counted iterations')
        body.append(loop)
    body.extend(_call(p) for p in last)
    return tuple(body)

@dataclass(frozen=True)
class CountedSharedProducts:
    products: tuple
    module: object
    group: object
    views: object
    kernel: Kernel
    templates: tuple
    source_identity: str
    binding_sha256: str
    proof: dict

    def verify(self):
        if _digest(self.products)!=self.binding_sha256:raise ValueError('changed original shared products')
        self.group.verify_current(self.module)
        self.views.verify_current(self.module)
        if check_structure(self.kernel):raise ValueError('invalid counted shared program')
        def static_weight_mode(body):
            for statement in body:
                if isinstance(statement,Loop):static_weight_mode(statement.body)
                elif statement.instr.startswith('compute_'):
                    stay=dict(statement.args)['stay']
                    if not isinstance(stay,Const) or stay.value not in (0,1):
                        raise ValueError('counted compute mode must be a compile-time boolean')
        static_weight_mode(self.kernel.body)
        for plan,reordered in self.templates:
            plan.verify()
            verify_shared_input(_project(reordered,plan.widths,plan.k,plan.schedule.primitive_tile[0]),
                module=plan.module,group=plan.schedule.group,primitive_tile=plan.schedule.primitive_tile,
                resources=plan.resources,storage=plan.storage,max_commands=8192)
        if self.kernel.digest()!=self.proof['counted_program_sha256']:raise ValueError('changed counted program')
        return copy.deepcopy(self.proof)


def plan_counted_shared_products(tensors,products,*,source_identity,source_module=None,source_group=None):
    """Return an actual counted plan, or None for unproved shapes/attributes.

    Admission is geometry/source based. Existing product admission checks all
    scalar readouts and physical aliases on bounded typed views. No capsule or
    tensor name participates in selection.
    """
    try:
        products=tuple(copy.deepcopy(products));lhs=products[0]['lhs'];m,k=tensors[lhs]['shape']
        tile=geometry()[0]
        if (not 2<=len(products)<=8 or type(m) is not int or type(k) is not int
                or not 0<m<=tile or k<=0):return None
        widths=tuple(tensors[p['rhs']]['shape'][1] for p in products)
        if tensors[lhs]['dtype']!='i8':return None
        for product,n in zip(products,widths,strict=True):
            rhs,out=tensors[product['rhs']],tensors[product['dst']]
            if (product['lhs']!=lhs or type(n) is not int or n<=0
                    or rhs['shape']!=[k,n] or rhs['dtype']!='i8'
                    or out['shape']!=[m,n] or out['dtype'] not in ('i8','i32')):return None
            product.setdefault('attrs',{}).setdefault('output_dtype',out['dtype'])
        module=_bridge(m,k,widths) if source_module is None else source_module
        groups=find_shared_input_groups(module)
        group=(groups[0] if len(groups)==1 else None) if source_group is None else source_group
        if group is None or (group.m,group.k,group.widths)!=(m,k,widths):return None
        group.verify_current(module)
        units=tuple(1 if p.get('attrs',{}).get('output_dtype',tensors[p['dst']]['dtype'])=='i8' else 4 for p in products)
        # Selected load/store configuration carries a 32-bit byte stride.
        if any(n >= 1<<32 or n*unit >= 1<<32 for n,unit in zip(widths,units,strict=True)):return None
        views=plan_shared_projection_views(module,group,weight_element_bytes=1,output_element_bytes=units,tile_columns=1)
        # Every actual full/tail view has a bounded checked template. A template
        # contains all members, retaining independent ACC windows and results.
        variants={tuple(min(tile,n) for n in widths)}
        variants.update(tuple((n%tile or tile) if i==member else min(tile,n) for i,n in enumerate(widths))
                        for member in range(len(widths)))
        templates=[];by_width={}
        for sizes in sorted(variants):
            vt=copy.deepcopy(tensors)
            for i,p in enumerate(products):
                vt[p['rhs']]['shape']=[k,sizes[i]];vt[p['dst']]['shape']=[m,sizes[i]]
            plan=plan_shared_products(vt,products,source_identity=source_identity)
            if plan is None or len(products)>8:return None
            schedule=schedule_shared_input(plan.module,plan.schedule.group,primitive_tile=(tile,tile,tile),
                resources=plan.resources,storage=plan.storage,selected_block=(1,1,len(products)),max_commands=8192)
            if schedule is None:return None
            plan=replace(plan,schedule=schedule,packets=_pack(schedule.operations,sizes,k,tile))
            plan.verify()
            lhs_packets=tuple(p for p in plan.packets if p[1] is None)
            members=[tuple(p for p in plan.packets if p[1]==i) for i in range(len(products))]
            ordered=lhs_packets+tuple(p for member in members for p in member)
            verify_shared_input(_project(ordered,sizes,k,tile),module=plan.module,group=plan.schedule.group,
                primitive_tile=(tile,tile,tile),resources=plan.resources,storage=plan.storage,max_commands=8192)
            templates.append((plan,ordered));by_width[sizes]=members
        base=templates[0][0];lhs_packets=tuple(p for p in base.packets if p[1] is None)
        body=list(_compress(tuple((p,) for p in lhs_packets),'lk'))
        covered=[]
        for member,n in enumerate(widths):
            full,tail=divmod(n,tile)
            for columns,count,start in ((tile,full,0),(tail,1,full*tile)):
                if not columns or not count:continue
                sizes=next(s for s in by_width if s[member]==columns)
                raw=by_width[sizes][member];triples=[];current=[];stores=[]
                for p in raw:
                    if p[0]=='store':stores.append(p)
                    else:
                        current.append(p)
                        if p[0]=='compute':triples.append(tuple(current));current=[]
                if current:raise ValueError('unfinished source view preload')
                statements=list(_compress(tuple(triples),f'k{member}_{columns}'))+[_call(p) for p in stores]
                nv=f'n{member}_{columns}';column=Var(nv)*tile+start
                def translated(statement):
                    if isinstance(statement,Loop):return Loop(statement.var,statement.extent,tuple(translated(x) for x in statement.body))
                    kind,_=statement.instr.split('_',1);fields=dict(statement.args)
                    if kind=='load':
                        # The template RHS is exactly one output-column tile.
                        # Its load offset is K-row * template-width + zero.
                        fields['stride']=as_expr(n)
                    if kind in ('load','store'):
                        # Rebuild offset expressions using exact coefficient
                        # division; all template column starts are zero.
                        fields['offset']=_rescale_offset(dict(statement.args)['offset'],columns,n)+column
                    return Call(statement.instr,tuple(fields.items()))
                translated_body=tuple(translated(x) for x in statements)
                body.append(Loop(nv,count,translated_body))
                # Source projection endpoints establish the affine interior:
                # contiguous tiles partition [start,start+count*columns).
                global_start=views.boundaries[member]+start
                for c in (global_start,global_start+(count-1)*tile):
                    views.rectangle('weight',0,c,k,columns)
                    views.rectangle('output',0,c,m,columns)
                covered.append({'member':member,'start':start,'count':count,'step':tile,'columns':columns,'width':n})
        kernel=Kernel('counted_shared_products',(),tuple(body))
        proof={'mechanism':'bounded shared_input_schedule + SharedProjectionViews + counted Kernel',
            'counted_program_sha256':kernel.digest(),'source_identity':source_identity,
            'typed_product_proof_sha256':group.source_ir_sha256,'binding_sha256':_digest(products),
            'source_shapes':[(m,k,n) for n in widths],'column_partitions':covered,
            'lhs_load_tiles':(k+tile-1)//tile,'original_readouts':[p['dst'] for p in products],
            'completion':'synchronous fence after each original load, compute and readout; final fence',
            'max_checked_view_commands':max(len(p.packets) for p,_ in templates)}
        result=CountedSharedProducts(products,module,group,views,kernel,tuple(templates),source_identity,_digest(products),proof)
        result.verify();return result
    except (ValueError,KeyError,TypeError,IndexError,OverflowError):return None


def _rescale_offset(expression,old_stride,new_stride):
    from merlin.sched.ir import Const,Scaled,Sum
    if isinstance(expression,Const):
        if expression.value%old_stride:raise ValueError('column view offset is not a row start')
        return as_expr(expression.value//old_stride*new_stride)
    if isinstance(expression,Scaled):
        if expression.factor%old_stride:raise ValueError('column view coefficient is not a row stride')
        return expression.expr*(expression.factor//old_stride*new_stride)
    if isinstance(expression,Sum):return sum((_rescale_offset(x,old_stride,new_stride) for x in expression.terms),as_expr(0))
    raise ValueError('unknown source view offset expression')


def emit_counted_shared_products(fn,plan,pointers):
    proof=plan.verify();E.fence(fn);E.config_ex(fn,dataflow='ws')
    def lower(tag,fields):
        kind,member=tag.split('_',1);member=None if member=='lhs' else int(member)
        a=fields
        if kind=='load':
            stream=0 if member is None else 1;name=plan.products[0]['lhs'] if member is None else plan.products[member]['rhs']
            E.config_load(fn,a['stride'],stream=stream)
            E.mvin(fn,E.pointer_offset(fn,pointers[name],a['offset']),a['local'],a['rows'],a['cols'],stream=stream);E.fence(fn)
        elif kind=='preload':E.preload(fn,a['weight'],a['acc'],a['depth'],a['cols'],a['rows'],a['cols'])
        elif kind=='compute':
            if type(a['stay']) is not int or a['stay'] not in (0,1):
                raise ValueError('compute mode cannot depend on an SSA index')
            E.compute(fn,a['lhs'],a['bd'],a['rows'],a['depth'],a['br'],a['bc'],accumulate=a['stay']==1)
            E.fence(fn)
        elif kind=='store':
            product=plan.products[member];attrs=product.get('attrs',{});dtype=attrs.get('output_dtype','i32');width=1 if dtype=='i8' else 4
            Eoff=a['offset']*width if type(a['offset']) is int else fn.mul_i(E._i64(fn,a['offset']),fn.const(width))
            emit_readout(fn,E.pointer_offset(fn,pointers[product['dst']],Eoff),a['acc'],a['rows'],a['cols'],output_stride=plan.group.widths[member]*width,attrs=attrs,output_dtype=dtype);E.fence(fn)
        else:raise ValueError('unknown counted shared instruction')
    emitted=emit_index_program(plan.kernel,FnBuilderIndexAdapter(fn),lower)
    E.fence(fn)
    return {**proof,'emission':emitted}
