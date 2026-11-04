"""Join independent row products sharing an immutable source RHS.

The ordinary source group discovery owns independence and reordering. This
consumer validates the actual members again, gathers only their original LHS
bytes with accelerator movement, computes a dense row concatenation, and
scatters every original result with exact integer accelerator movement.
"""
from __future__ import annotations

from . import isa


def lower_shared_rhs_rows(ctx,members):
    """Return a receipt, or None before mutation if row joining is inapplicable."""
    from .dense import lower_dense
    from .elementwise import lower_movement
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    selection=knob('ATTENTION_RHS_REUSE',default='off',values=('off','on','all_rows'),
                   root=Path(__file__).resolve().parent.parent)
    all_rows=selection=='all_rows'
    members=tuple(members)
    if not 2 <= len(members) <= 32:
        return None
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if aliases:
        return None
    try:
        rhs=members[0]['rhs']
        rhs_spec=ctx.spec(rhs)
        if rhs_spec['dtype']!='i8' or len(rhs_spec['shape'])!=2:
            return None
        k,n=rhs_spec['shape']
        shapes=[];attributes=[]
        outputs={row['dst'] for row in members}
        inputs={row[key] for row in members for key in ('lhs','rhs')}
        if len(outputs)!=len(members) or outputs&inputs:
            return None
        for row in members:
            a,c=ctx.spec(row['lhs']),ctx.spec(row['dst'])
            if (row['rhs']!=rhs or a['dtype']!='i8' or c['dtype'] not in ('i8','i32')
                    or len(a['shape'])!=2 or a['shape'][1]!=k
                    or c['shape']!=[a['shape'][0],n]):
                return None
            attrs=dict(row.get('attributes',{}))
            # Auxiliaries introduce shared/repeated row layouts, each needing
            # its own source proof rather than implicit broadcast changes.
            if attrs.get('bias') or attrs.get('accumulator_init'):
                return None
            if (any(e not in ('acc_scale','relu') for e in attrs.get('epilogue',()))
                    or attrs.get('requant_shift') is not None
                    or any(key.startswith('pool') for key in attrs)):
                return None
            if attrs.get('output_dtype',c['dtype'])!=c['dtype']:
                return None
            for key in ('merlin.source_op_index','merlin.global_task','prov.region_id'):
                attrs.pop(key,None)
            shapes.append((a['shape'][0],k,n,c['dtype']))
            attributes.append(attrs)
        if (any(s[1:]!=shapes[0][1:] for s in shapes)
                or any(a!=attributes[0] for a in attributes)):
            return None
        m,k,n,dtype=shapes[0]
        row_counts=[s[0] for s in shapes]
        if min(row_counts)<=0 or min(k,n)<=0:
            return None
        joined_m=sum(row_counts)
        old_tiles=sum((rows+isa.DIM-1)//isa.DIM for rows in row_counts)
        new_tiles=(joined_m+isa.DIM-1)//isa.DIM
        # The incumbent only joins equal short rows with fewer padded tiles.
        # The explicit all_rows alternative also explores shared scheduling
        # without a padding reduction, including unequal original row counts.
        if not all_rows and (any(s!=shapes[0] for s in shapes) or new_tiles>=old_tiles or m>=isa.DIM):
            return None
        lhs_bytes=joined_m*k
        result_bytes=joined_m*n*(1 if dtype=='i8' else 4)
        # Materialization is private DRAM, not a claim that the joined product
        # fits local memory. Bound the optional larger materialization by the
        # selected target's aggregate storage footprint; the ordinary dense
        # planner separately checks every actual local tile/window capacity.
        workspace_limit=isa.SPAD_ROWS*isa.DIM+isa.ACC_ROWS*isa.DIM*4
        if max(lhs_bytes,result_bytes)>=1<<32:
            return None
        if all_rows and lhs_bytes+result_bytes>workspace_limit:
            return None
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    scattered=None
    scatter_selection=knob('SHARED_RHS_SCATTERED_READOUT',default='off',values=('off','on'),
        root=Path(__file__).resolve().parent.parent)
    if scatter_selection=='on':
        from .shared_rhs_scattered_readout import make_readout
        try:
            scattered=make_readout(ctx,members,m=joined_m,k=k,n=n,attributes=dict(members[0].get('attributes',{})))
        except ValueError:
            # The optional readout mapping has a narrower proof domain than
            # the checked shared-RHS incumbent. Preserve that incumbent.
            scattered=None
    prefix=f'_shared_rhs_rows_{len(ctx.tensors)}'
    lhs_joined=prefix+'_lhs';dst_joined=prefix+'_dst'
    ctx.alloc(lhs_joined,(joined_m,k),'i8')
    if scattered is None:
        ctx.alloc(dst_joined,(joined_m,n),dtype)
    else:
        dst_joined=members[0]['dst']
    source_index=ctx.current_source_index
    def view(name,offset,shape):
        label=f'{prefix}_view_{len(ctx.tensors)}'
        ctx.register(label,ctx.ptr(name,offset),shape,ctx.spec(name)['dtype'])
        return label
    ctx.fence()
    row_start=0
    for row,rows in zip(members,row_counts):
        ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',source_index)
        part=view(lhs_joined,row_start*k,(rows,k))
        lower_movement(ctx,{'src':row['lhs'],'dst':part},{'semantic':'copy'})
        row_start+=rows
    ctx.current_source_index=source_index
    from .shared_rhs_scattered_readout import MappedReadoutRefusal
    try:
        leaf=lower_dense(ctx,lhs_joined,rhs,dst_joined,m=joined_m,k=k,n=n,
                         attributes=dict(members[0].get('attributes',{})),readout=scattered)
    except MappedReadoutRefusal:
        # LHS preparation is already complete, but no dense command has been
        # emitted. Reuse it and restore the ordinary joined result plus scatter.
        scattered=None
        dst_joined=prefix+'_dst'
        ctx.alloc(dst_joined,(joined_m,n),dtype)
        leaf=lower_dense(ctx,lhs_joined,rhs,dst_joined,m=joined_m,k=k,n=n,
                         attributes=dict(members[0].get('attributes',{})))
    ctx.fence()
    if scattered is None:
        row_start=0
        for row,rows in zip(members,row_counts):
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',source_index)
            part=view(dst_joined,row_start*n,(rows,n))
            lower_movement(ctx,{'src':part,'dst':row['dst']},{'semantic':'copy'})
            row_start+=rows
        ctx.current_source_index=source_index
    cursor=0
    partitions=[]
    for row,rows in zip(members,row_counts):
        partitions.append({'lhs':row['lhs'],'dst':row['dst'],
            'joined_rows':[cursor,cursor+rows],
            'lhs_element_interval':[cursor*k,(cursor+rows)*k],
            'result_element_interval':[cursor*n,(cursor+rows)*n]})
        cursor+=rows
    receipt=dict(mechanism='shared_rhs_gathered_rows',applied=True,rhs=rhs,
        products=len(members),source_shape=[m,k,n],source_shapes=[list(s[:3]) for s in shapes],joined_shape=[joined_m,k,n],
        selection=selection,workspace_bytes=lhs_bytes+(result_bytes if scattered is None else 0),
        workspace_limit_bytes=workspace_limit if all_rows else None,
        row_partitions=partitions,
        source_members=[{key:row.get(key) for key in ('command_index','lhs','rhs','dst')} for row in members],
        source_op_indices=[row.get('attributes',{}).get('merlin.source_op_index') for row in members],
        dense_leaf_schedule=leaf,baseline_m_tiles=old_tiles,joined_m_tiles=new_tiles,
        preparation_bytes=lhs_bytes,
        scatter_bytes=result_bytes if scattered is None else 0,
        rhs_replication_bytes=0,
        row_mapping='joined_row = sum(previous member row counts) + original_row; RHS source identity unchanged',
        coverage='all original source LHS and result intervals partition joined storage exactly once',
        numeric='exact signed-i8 LHS copy; unchanged dense contraction/readout; identity i8 or raw i32 result copy',
        completion='full accelerator completion after movement tiles and dense result before buffer reuse')
    if scattered is not None:
        receipt['scattered_readout']=scattered.certificate
    ctx.metadata.setdefault('shared_rhs_rows',[]).append(receipt)
    return receipt
