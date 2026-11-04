"""Source-bound RHS-only rectangular loads on checked original pointer rows.

Caller-owned refinement, not a ready/panel admission or a compact proof stamp.
The public actual-band projection must reconstruct the entire original checked
kernel exactly. The unchanged original terminal encoder emits the wider MVIN2
with its ordinary configured stride16 and real closing fence before consumers.
"""
from dataclasses import dataclass
from pathlib import Path


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('original_row_rhs_width',default='primitive',values=('primitive','coalesced4'),root=Path(__file__).parent)


def _codec(problem,storage,root):
    from dense import _adapter
    from dense_wide_loads import WideRHSAdapter
    return WideRHSAdapter(problem,storage,_adapter(),facts_root=root)


def _proposal(original,problem,storage,root):
    from dense import _adapter
    from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
    from merlin.compile.scheduling.panel_tile_loads import expand_dense_panel_tile_loads
    codec=_codec(problem,storage,root)
    actual,report=propose_tile_load_coalescing(original.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
    expanded,projection=expand_dense_panel_tile_loads(actual,problem,storage,_adapter(),project=codec.project,encode_band=codec.band)
    return actual,expanded,report,projection,codec.evidence


def _original_projection(original,problem,storage,root):
    from dense import _adapter
    from merlin.compile.scheduling.panel_tile_loads import expand_dense_panel_tile_loads
    codec=_codec(problem,storage,root)
    return expand_dense_panel_tile_loads(original.kernel,problem,storage,_adapter(),project=codec.project,encode_band=codec.band)[0]


@dataclass(frozen=True)
class _Wide:
    original: object
    kernel: object
    problem: object
    storage: object
    root: object
    before_sha256: str
    after_sha256: str
    realization: dict
    @property
    def choice(self):return self.original.choice
    def revalidate(self):
        self.original.revalidate()
        if self.original.kernel.digest()!=self.before_sha256:raise ValueError('wide RHS original checked owner changed')
        fresh,expanded,_,_,_=_proposal(self.original,self.problem,self.storage,self.root)
        if (self.kernel.digest()!=self.after_sha256 or fresh.digest()!=self.after_sha256
                or expanded.digest()!=_original_projection(self.original,self.problem,self.storage,self.root).digest()):
            raise ValueError('wide RHS actual commands do not reconstruct complete original effects/completions')
        return self


def refine(owner,*,mode='coalesced4',root=None):
    from original_row_batching import OriginalRows,_group_guard
    from merlin.compile.scheduling.compact_dense import CompactDenseResult,DenseLoopStorage,DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem
    from dense import _geometry
    if mode not in ('primitive','coalesced4'):raise ValueError('unknown original row RHS width')
    if type(owner) is not OriginalRows:raise ValueError('wide RHS requires original source row owner')
    owner.revalidate()
    if mode=='primitive':return owner
    root=Path(__file__).parent if root is None else Path(root)
    edge,bank,_,acc=_geometry()
    storage=DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank),DenseMemoryView('b','spad',edge,bank,bank),DenseMemoryView('c','acc',edge*4,0,acc//(edge*4)))
    groups=[];cache={}
    for g in owner.groups:
        original=g['schedule']
        if type(original) is not CompactDenseResult:raise ValueError('wide RHS requires exact unwrapped compact owner')
        if original.choice.block.m!=1 or original.choice.block.k!=1 or not 1<=original.choice.block.n<=4:
            raise ValueError('wide RHS requires original bounded single-M/single-K panel')
        p=DenseProblem(len(g['lhs']),g['k'],g['n'],edge,edge,edge,1,1,4)
        key=original.kernel.digest()
        if key not in cache:
            actual,expanded,report,projection,evidence=_proposal(original,p,storage,root)
            before_projection=_original_projection(original,p,storage,root)
            if not report['changes'] or expanded.digest()!=before_projection.digest():
                cache[key]=None
            else:
                receipt={'before_kernel_sha256':key,'actual_kernel_sha256':actual.digest(),
                    'expanded_kernel_sha256':expanded.digest(),'original_projection_sha256':before_projection.digest(),'changes':report,'projection':projection,
                    'target_addressing_evidence':evidence,'completion':'unchanged original encoder closes every actual RHS rectangle before next effect',
                    'admission':'caller-owned exact complete primitive expansion; original compact owner retained; no typed library stamp fabricated'}
                cache[key]=_Wide(original,actual,p,storage,root,key,actual.digest(),dict(original.realization,kernel_sha256=actual.digest(),original_row_rhs_width=receipt)).revalidate()
        candidate=cache[key]
        groups.append(g if candidate is None else dict(g,schedule=candidate))
    if all(a is b for a,b in zip(groups,owner.groups,strict=True)):return owner
    return OriginalRows(owner.source_owner,owner.cb,owner.guard,tuple(groups),_group_guard(groups)).revalidate()


def prepare(source,cb,*,max_members=8):
    from original_row_reuse import prepare as original_prepare
    prepared=original_prepare(source,cb,max_members=max_members)
    if prepared is None:return None
    out,owner=prepared;mode=setting()
    if mode=='primitive':return prepared
    candidate=refine(owner,mode=mode)
    out['params']['original_row_rhs_width']={'requested':mode,'groups':[
        {'shape':[len(g['lhs']),g['k'],g['n']],
         'refinement':g['schedule'].realization.get('original_row_rhs_width'),
         'actual_kernel_sha256':g['schedule'].kernel.digest()}
        for g in candidate.groups]}
    return out,candidate
