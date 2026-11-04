"""Opt-in checked shared-input products, bound to separate original pointers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace


def enabled():
    value=os.environ.get('SOURCE_SHARED_INPUT')
    if value is not None:
        return value=='1'
    path=Path(__file__).with_name('DENSE_SCHEDULE.json')
    return path.exists() and json.loads(path.read_text()).get('source_shared_input',False) is True


def epochs_enabled():
    value=os.environ.get('SOURCE_SHARED_INPUT_EPOCHS')
    if value is not None:return value=='1'
    path=Path(__file__).with_name('DENSE_SCHEDULE.json')
    return path.exists() and json.loads(path.read_text()).get('source_shared_input_epochs',False) is True


def _source(m,k,widths):
    """Canonical exact signed products; no source packing or epilogue changes."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    args=[f'%a:tensor<{m}x{k}xi8>']+[f'%w{i}:tensor<{k}x{n}xi8>' for i,n in enumerate(widths)]
    body=[]
    for i,n in enumerate(widths):
        out=f'tensor<{m}x{n}xi32>'
        body.append(f'''%z{i}=arith.constant 0:i32
%s{i}="tensor.splat"(%z{i}):(i32)->{out}
%r{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}} ins(%a,%w{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%s{i}:{out}) {{
^bb0(%x:i8,%w:i8,%s:i32):
%xw=arith.extsi %x:i8 to i32
%ww=arith.extsi %w:i8 to i32
%p=arith.muli %xw,%ww:i32
%v=arith.addi %s,%p:i32
linalg.yield %v:i32
}}->{out}''')
    types=','.join(f'tensor<{m}x{n}xi32>' for n in widths)
    return parse_mlir_text('module {func.func @siblings('+','.join(args)+')->('+types+'){\n'+'\n'.join(body)+'\nreturn '+','.join(f'%r{i}' for i in range(len(widths)))+':'+types+'\n}}')


def _plan(m,k,widths):
    from dense import _capacities
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input
    from merlin.transforms.dense_planning import DenseResources
    module=_source(m,k,widths)
    groups=find_shared_input_groups(module)
    if len(groups)!=1:
        raise ValueError('canonical shared input products did not form one checked group')
    kr,edge,operand,rows,rowbytes,acc,accrows,accbytes=_capacities()
    half=rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand,rowbytes,0,half),
        DenseMemoryView('rhs',operand,rowbytes,half,rows-half),
        DenseMemoryView('acc',acc,accbytes,0,accrows))
    resources=DenseResources(rows*rowbytes,accrows*accbytes,max_m_blocks=1,
        max_n_blocks=1,max_k_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    schedule=schedule_shared_input(module,groups[0],primitive_tile=(edge,kr,edge),
        resources=resources,storage=storage,selected_block=(1,1,1),max_commands=200000)
    return module,groups[0],storage,resources,schedule


def emit_shared_input(enc,lhs,rhs_outputs,m,k,widths,*,source_indices=None,completion="synchronous"):
    """Emit checked zero-seeded rank-two i8 products, or return None before emission.

    Caller proves immutable common LHS and separate RHS/results, exact modular
    i32 arithmetic and zero seeds. All pointers retain original row-major shapes.
    No input/output alias is admitted. All commands complete synchronously.
    """
    from dense import dense_adapter
    from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect
    from merlin.compile.scheduling.shared_input_verification import verify_shared_input
    if (type(m) is not int or type(k) is not int or min(m,k)<=0 or len(widths)<2
        or len(widths)!=len(rhs_outputs) or any(type(n) is not int or n<=0 for n in widths)):
        return None
    try:module,group,storage,resources,schedule=_plan(m,k,tuple(widths))
    except ValueError:return None
    if schedule is None:return None
    if completion not in ("synchronous","epochs"):raise ValueError("unknown shared-input completion")
    adapter=dense_adapter()
    rows=[];projected=[]
    for effect in schedule.operations:
        from dataclasses import replace
        from merlin.compile.scheduling.block_schedule import Preload,Load
        encoded=effect.operation
        if isinstance(encoded,Preload):encoded=replace(encoded,accumulate=int(encoded.accumulate))
        # The source scheduler records K-stage ordinals in Load.step; this
        # target field denotes physical row-stride multiplication instead.
        # Bind the actual contiguous source rectangle to one original row.
        if isinstance(encoded,Load):encoded=replace(encoded,step=1)
        call=adapter.encode(encoded)
        # Decode actual encoded fields; the independent checker consumes these.
        actual=adapter.project(call)
        from dataclasses import replace
        for key,value in vars(actual).items():
            if hasattr(value,'value'): actual=replace(actual,**{key:value.value})
            elif isinstance(value,tuple):
                actual=replace(actual,**{key:tuple(v.value if hasattr(v,'value') else v for v in value)})
        if isinstance(actual,Preload):actual=replace(actual,accumulate=bool(actual.accumulate))
        projected.append(SharedInputEffect(effect.member,actual))
        rows.append((effect.member,call.instr,{key:value.value for key,value in call.args}))
    checked=verify_shared_input(tuple(projected),module=module,group=group,
        primitive_tile=schedule.primitive_tile,resources=resources,storage=storage,max_commands=200000)
    epoch_receipt=_admit_epochs(projected,adapter) if completion=="epochs" else None
    use_epochs=epoch_receipt is not None
    tm,tk,tn=schedule.primitive_tile
    kt=(k+tk-1)//tk
    # Compress only affine rectangles of the checked actual schedule. Every
    # numeric field is compared over every original effect before any command.
    grids=[];position=0
    for mi in range((m+tm-1)//tm):
        loads=rows[position:position+kt];position+=kt
        if kt>1 and k%tk:
            grids.append(([[[[x] for x in loads[:-1]]]],None))
            grids.append(([[[[loads[-1]]]]],None))
        else:grids.append(([[[[x] for x in loads]]],None))
        for member,n in enumerate(widths):
            nt=(n+tn-1)//tn;chunks=[]
            for ni in range(nt):
                chunks.append(rows[position:position+3*kt+1]);position+=3*kt+1
            # Tails remain separate, so every field in each grid is affine.
            nsplits=[(0,nt-1),(nt-1,nt)] if nt>1 else [(0,1)]
            ksplits=[(0,1),(1,kt-1),(kt-1,kt)] if kt>2 else [(i,i+1) for i in range(kt)]
            for nb,ne in nsplits:
                parts=[]
                for kb,ke in ksplits:
                    if kb==ke:continue
                    grid=[[chunk[3*ki:3*ki+3] for ki in range(kb,ke)] for chunk in chunks[nb:ne]]
                    parts.append(grid)
                parts.append([[chunk[-1:]] for chunk in chunks[nb:ne]])
                grids.append((parts,member))
    if position!=len(rows):raise ValueError('shared input schedule layout changed')
    def normalize(grid):
        h,w=len(grid),len(grid[0]);templates=[]
        for slot,(member,instruction,first) in enumerate(grid[0][0]):
            fields={}
            for name,v in first.items():
                dy=grid[1][0][slot][2][name]-v if h>1 else 0
                dx=grid[0][1][slot][2][name]-v if w>1 else 0
                for y,line in enumerate(grid):
                    for x,cell in enumerate(line):
                        row=cell[slot]
                        if row[:2]!=(member,instruction) or row[2][name]!=v+y*dy+x*dx:
                            raise ValueError('shared schedule rectangle is not affine')
                fields[name]=(v,dy,dx)
            templates.append((member,instruction,fields))
        return h,w,templates
    prepared=[([normalize(grid) for grid in parts],member) for parts,member in grids]
    previous=getattr(enc.fn,'current_source_index',None)
    def owner(member):
        if source_indices is not None:
            enc.fn.current_source_index=source_indices[0 if member is None else member]
    def issue(member,instruction,fields):
        owner(member)
        selected=0 if member is None else member
        rhs,out=rhs_outputs[selected]
        enc.dense_effect(instruction,fields,dict(lhs=lhs,rhs=rhs,out=out,out_dtype='i32'),
            SimpleNamespace(k=k,n=widths[selected]))
        if instruction=='store' or (not use_epochs and instruction!='preload'):enc.fence()
    for parts,member in prepared:
        owner(member)
        h=parts[0][0]
        if any(part[0]!=h for part in parts):raise ValueError('shared input outer extents differ')
        def row(y):
            for _,w,templates in parts:
                def cell(x):
                    for actual_member,instruction,template in templates:
                        fields={name:enc.add(enc.add(v,enc.mul(y,dy)),enc.mul(x,dx))
                            for name,(v,dy,dx) in template.items()}
                        issue(actual_member,instruction,fields)
                if w==1:cell(0)
                else:enc.fn.for_loop(0,w,1,cell,retain_loop=True)
        if h==1:row(0)
        else:enc.fn.for_loop(0,h,1,row,retain_loop=True)
    enc.fence();enc.fn.current_source_index=previous
    return {'schedule':'shared_input_products','source_group':group.record(),
        'realization':checked,'actual_effects':len(rows),'counted_rectangles':len(prepared),
        'completion':'epochs' if use_epochs else 'synchronous','epoch_admission':epoch_receipt}


def _admit_epochs(effects,adapter):
    """Check scratch overwrite chains before consuming selected ordering laws.

    Every replaced drain orders either load->reader, reader->overwrite or an
    ordered accumulator continuation. Store boundaries still drain, so no new
    accumulator write may overtake a previous readout. All initial operand loads
    use disjoint rows, or follow an actual read of the previous row contents.
    """
    from dataclasses import asdict
    from dense import _execution_laws
    from merlin.compile.scheduling.block_schedule import Load,Preload,Compute,Store
    _,admission,law=_execution_laws(adapter)
    law.validate_for(adapter,admission)
    if law.unknown():return None
    loaded={};reads=set();loads=pairs=0;boundaries=0;pending=False
    for effect in effects:
        op=effect.operation
        if isinstance(op,Load):
            if pending:return None
            for row in range(op.row,op.row+op.rows):
                key=(op.role,row)
                if key in loaded and key not in reads:return None
                loaded[key]=True;reads.discard(key)
            loads+=1
        elif isinstance(op,Preload):
            if pending or op.weight_row is None:return None
            for row in range(op.weight_row,op.weight_row+op.weight_rows):
                key=('weight',row)
                if key not in loaded:return None
                reads.add(key)
            pending=True
        elif isinstance(op,Compute):
            if not pending:return None
            for row in range(op.input_row,op.input_row+op.rows):
                key=('lhs',row)
                if key not in loaded:return None
                reads.add(key)
            pending=False;pairs+=1
        elif isinstance(op,Store):
            if pending:return None
            boundaries+=1;loads=pairs=0
        else:return None
        if loads>admission.max_issued_loads or pairs>admission.max_issued_compute_pairs:return None
    if pending:return None
    return {'law':asdict(law),'ordering_proof':'every overwritten operand row follows its actual reader; each output readout closes its epoch',
        'output_epoch_drains':boundaries,'closing_drain':True}


def find_source_groups(module):
    """Find adjacent independent strict products through pure zero preparation.

    This adapter admits fill or splat seeds and strips provenance only for the
    shared exact scalar recognizer. It never moves across arbitrary host work.
    All weights must be defined before the first product being co-emitted.
    """
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.ops import GenericOp
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul
    functions=[o for o in module.body.block.ops if o.name=='func.func' and len(o.body.blocks)==1]
    if len(functions)!=1:return ()
    block=functions[0].body.block;ops=list(block.ops);position={o:i for i,o in enumerate(ops)}
    def product(op):
        if op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1:return False
        if any(not isinstance(v.type,TensorType) or len(v.type.get_shape())!=2 for v in (*op.operands,*op.results)):return False
        clone=GenericOp.create(operands=op.operands,result_types=[r.type for r in op.results],
            attributes=dict(op.attributes),properties=dict(op.properties),regions=[r.clone() for r in op.regions])
        try:
            for inner in clone.walk():
                for key in tuple(inner.attributes):
                    if key.startswith('prov.'):del inner.attributes[key]
            recognize_signed_integer_matmul(clone,('i8','i8','i32'))
        except ValueError:return False
        finally:clone.drop_all_references()
        seed=op.operands[2];owner=seed.owner
        if owner.name not in ('linalg.fill','tensor.splat') or not owner.operands:return False
        constant=owner.operands[0].owner
        return (getattr(constant,'name',None)=='arith.constant'
            and getattr(constant.properties.get('value'),'value',None) is not None
            and constant.properties['value'].value.data==0)
    groups=[];run=[];between=[]
    for op in ops:
        if product(op):
            if run:
                first=run[0]
                compatible=(op.operands[0] is first.operands[0]
                    and all(o.name in ('tensor.empty','arith.constant','tensor.splat','linalg.fill') for o in between)
                    and position.get(op.operands[1].owner,-1)<position[first])
                if not compatible:
                    if len(run)>1:groups.append(tuple(run))
                    run=[]
            run.append(op);between=[]
        else:between.append(op)
    if len(run)>1:groups.append(tuple(run))
    return tuple(groups)
