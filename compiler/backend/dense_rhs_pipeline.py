"""Optional full-K RHS N-panel lookahead through the shared checked service."""
from __future__ import annotations
import ast,hashlib,json
from functools import lru_cache
from pathlib import Path
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView
from merlin.compile.scheduling.dense_rhs_panel_pipeline import DenseRhsPanelAdmission,schedule_dense_rhs_panels
from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from . import isa


def _binding(source,block):
    here=Path(__file__).resolve().parent
    record=json.loads((here/'conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=hashlib.sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
        or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
        or ordering.get('status')!='derived'):
        raise ValueError('RHS panel ordering differs from selected global public facts')
    cross={(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    for reader in ('PRELOAD.weight_spad','COMPUTE_PRELOADED.a_spad'):
        for older,younger in ((reader,'MVIN2.spad_addr'),('MVIN2.spad_addr',reader)):
            row=cross.get((older,younger,'scratchpad'),{})
            if row.get('overlap')!='completion' or row.get('disjoint') not in ('issue','none','completion'):
                raise ValueError('RHS panel lacks selected cross-engine operand dependency law')
    if ordering['completion_order']['execute']['order']!='in_order':
        raise ValueError('RHS panel needs actual ordered accumulator commits')
    same=ordering['same_queue']
    if any(same[lane]['release'] not in ('issue','completion') for lane in ('execute','load')):
        raise ValueError('RHS panel ordered lane progress is unavailable')
    queues={}
    for row in ordering['funct_queue']:
        if 'by_code' in row:
            queues.update({('config',int(k)):v for k,v in row['by_code'].items()})
        else:
            queues[row['funct']]=row['queue']
    if any(queues.get(f)!='execute' for f in (4,6)) or queues.get(('config',1))!='load':
        raise ValueError('RHS panel descriptors/configuration enter unexpected lanes')
    finite=record['finite_entries']
    for lane,minimum in [('execute',2),('load',1)]:
        if finite[lane].get('status')!='derived' or finite[lane].get('value',0)<minimum:
            raise ValueError('RHS panel lacks selected finite queue pair progress')
    # Establish the narrower configuration premise from the actual common
    # terminal emitter. Channels 0/1 each have one setup call, outside its
    # per-command callback; every later CONFIG_LD explicitly selects channel2.
    tree=ast.parse((here/'dense.py').read_text())
    setup=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='config_ld':
            channel=next((k.value for k in node.keywords if k.arg=='channel'),None)
            if not isinstance(channel,ast.Constant) or channel.value not in (0,1,2):
                raise ValueError('common dense emitter has a dynamic load channel')
            if channel.value in (0,1): setup.append((channel.value,node.lineno))
    callbacks=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='emit']
    if sorted(c for c,_ in setup)!=[0,1] or len(callbacks)!=1 or any(line>=callbacks[0].lineno for _,line in setup):
        raise ValueError('common dense emitter no longer has immutable operand load channels')
    from .dense import _adapter
    adapter=_adapter()
    paths=[here/'dense.py',here/'completion.py',here/'isa.py',here/'conv_ordering.json',Path(__file__).resolve()]
    pins=tuple((str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
    return DenseRhsPanelAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-global-ordering:'+digest+':immutable-dense-load-channels',source,
        block.m*block.k*block.n,max(block.k*block.m,block.k*block.n),
        finite['execute']['value']//2,finite['load']['value'],pins)


@lru_cache(maxsize=256)
def rhs_panel_schedule(m,k,n,bias=False,column_scales=False,*,n_tiles=1):
    """Return (checked product, refusal); n_tiles is an explicit trial option.

    No default hook, empirical selector or timing inference is installed.
    Two full-K RHS slots and one full-K LHS tile must fit selected real rows.
    Bias, accumulator initialization and mixed per-column readout are excluded.
    """
    if type(n_tiles) is not int or n_tiles<=0:
        raise ValueError('RHS panel width must be an explicit positive tile count')
    if bias or column_scales or not 1<=m<=isa.DIM:
        return None,'RHS panel lookahead currently requires one M tile, no seed and uniform readout'
    from .dense import _adapter
    dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,dim,1,1,4,bias=False)
    lhs_rows=p.tiles[1]*dim
    if lhs_rows>=isa.SPAD_ROWS:
        return None,'full-K LHS leaves no private RHS rows'
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,lhs_rows),
        DenseMemoryView('rhs','spad',dim,lhs_rows,isa.SPAD_ROWS-lhs_rows),
        DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,max_m_blocks=1,
        max_n_blocks=n_tiles,max_k_blocks=p.tiles[1],partial_output_tiles=True,partial_output_blocks=True)
    choices=[b for b in dense_blocks(p,r) if b.m==1 and b.k==p.tiles[1] and b.n==n_tiles
        and 2*b.k*b.n*dim<=s.rhs.capacity and s.admits(p,b,r)]
    if not choices or p.tiles[2]<=n_tiles:
        return None,'two full-K RHS slots or two output groups do not fit requested tile width'
    block=choices[0]
    source=hashlib.sha256(json.dumps([m,k,n,False,'i8-i8-i32'],sort_keys=True).encode()).hexdigest()
    admission=_binding(source,block)
    result=schedule_dense_rhs_panels(p,block,r,s,_adapter(),admission,name='dense_rhs_panels',
        source_identity=source,external_buffers_disjoint=True)
    result.realization['rhs_panel_storage']={'lhs_rows':lhs_rows,'rhs_rows':s.rhs.capacity,
        'rhs_slot_rows':block.k*block.n*dim,'operand_rows':isa.SPAD_ROWS,
        'accumulator_rows':isa.ACC_ROWS,'explicit_n_tiles':n_tiles}
    return result,None
