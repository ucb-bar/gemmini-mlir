"""Bounded N-panel alternative for the checked original-pointer row owner.

No source rewrite or new physical encoder: existing compact dense storage/order
verification certifies retention, and original_row_batching.emit splits exactly
those checked effects onto the same original ABI lanes with unchanged fences.
"""
from pathlib import Path


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('original_row_n_reuse',default='off',values=('off','n2','n4'),root=Path(__file__).parent)


def refine(owner, *, n_tiles):
    """Keep the checked incumbent or request at most four adjacent N tiles.

    This is an explicit alternative, not a timing policy. No extra K retention,
    ordering permission, buffer, source observer rule or completion change.
    """
    from original_row_batching import OriginalRows,_group_guard
    from dense import _geometry,_adapter
    from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    if type(owner) is not OriginalRows:raise ValueError('N reuse requires the original checked row owner')
    owner.revalidate()
    if type(n_tiles) is not int or n_tiles not in (1,2,4):raise ValueError('N reuse is bounded to incumbent, two or four tiles')
    if n_tiles==1:return owner
    edge,bank,sp,accbytes=_geometry()
    storage=DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank),DenseMemoryView('b','spad',edge,bank,bank),DenseMemoryView('c','acc',edge*4,0,accbytes//(edge*4)))
    resources=DenseResources(sp,accbytes,max_m_blocks=1,max_k_blocks=1,max_n_blocks=n_tiles,partial_output_tiles=True,partial_output_blocks=True)
    source=owner.cb['params']['source_sha256'];groups=[];cache={};refusals={}
    for original in owner.groups:
        g=dict(original);m=len(g['lhs']);k=g['k'];n=g['n']
        p=DenseProblem(m,k,n,edge,edge,edge,1,1,4)
        width=min(n_tiles,p.tiles[2])
        if width==1:groups.append(g);continue
        legal={block.n for block in dense_blocks(p,resources) if storage.admits(p,block,resources)}
        if width not in legal:
            g['n_reuse_refusal']='NOT_APPLICABLE: original private capacity does not admit requested N panel'
            groups.append(g);continue
        key=(m,k,n,width)
        if key not in cache:
            proposal=make_dense_schedule_plan(p,resources,storage,_adapter(),source_identity=source,dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(1,1,width),name='original_rows')
            try:
                cache[key]=schedule_compact_dense(p,resources,storage,_adapter(),name='original_rows',dtypes=('i8','i8','i32'),static_budget=128,
                    external_buffers_disjoint=True,verification_timeout_ms=5000,schedule_plan=proposal,source_identity=source)
            except DenseVerificationUnavailable as error:
                cache[key]=None;refusals[key]='UNKNOWN: '+str(error)
        candidate=cache[key]
        # A bounded proof UNKNOWN leaves the independently checked incumbent.
        # Concrete illegal source/storage/ordering proposals propagate failures.
        if candidate is not None:
            candidate.revalidate();g['schedule']=candidate
        else:g['n_reuse_refusal']=refusals.get(key,'UNKNOWN: no checked candidate')
        groups.append(g)
    owner.revalidate()
    result=OriginalRows(owner.source_owner,owner.cb,owner.guard,tuple(groups),_group_guard(groups))
    return result.revalidate()


def prepare(source,cb,*,max_members=8):
    """Minimal hook replacement: preserve existing source/ABI admission first."""
    from original_row_batching import prepare as original_prepare
    prepared=original_prepare(source,cb,max_members=max_members)
    if prepared is None:return None
    out,owner=prepared;mode=setting()
    if mode=='off':return prepared
    refined=refine(owner,n_tiles={'n2':2,'n4':4}[mode])
    out['params']['original_row_n_reuse']={'requested':mode,'groups':[
        {'shape':[len(g['lhs']),g['k'],g['n']],
         'block':[g['schedule'].choice.block.m,g['schedule'].choice.block.k,g['schedule'].choice.block.n],
         'kernel_sha256':g['schedule'].kernel.digest(),
         'schedule_plan_sha256':g['schedule'].realization.get('schedule_ir',{}).get('sha256'),
         'refusal':g.get('n_reuse_refusal')}
        for g in refined.groups],
        'scope':'same original source and pointer lanes; checked bounded private N panel; unchanged completion fences'}
    return out,refined
