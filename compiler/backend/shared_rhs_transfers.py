"""Wide immutable RHS preparation, retaining the checked shared epochs.

The original shared scheduler owns the complete RHS row-major tile window.
The ordinary tile-load proposal consumes those actual calls. Its final wide
operands are independently expanded into *every* primitive band and compared
with the original counted effects, including all non-loads and full drains.
Only the wide kernel is emitted; the expansion is proof transport, not code.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Const, Loop, Ptr, Scaled, Select, Sum, Var
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing,
)
from merlin.compile.scheduling.panel_tile_loads import (
    expand_panel_tile_loads, _same_band_call,
)
from .dense_rhs_transfers import REFERENCE_FILES, _quotient
from . import isa

RHS = 'shared_common_gemmini_mvin_weight'


def _rename(value, names):
    if type(value) is Const:return value
    if type(value) is Var:return Var(names.get(value.name,value.name))
    if type(value) is Sum:return sum((_rename(x,names) for x in value.terms),Const(0))
    if type(value) is Scaled:return _rename(value.expr,names)*value.factor
    if type(value) is Select:
        return replace(value,var=names.get(value.var,value.var),
            then=_rename(value.then,names),other=_rename(value.other,names))
    if type(value) is Ptr:return replace(value,offset=_rename(value.offset,names))
    return value


def _unique_binders(kernel):
    # The ordinary coalescer intentionally uses the same local binder in its
    # sequential bulk/tail loops. Give duplicates fresh scoped names so the
    # unchanged bounded literal specializer can address each loop uniquely.
    used=set()
    def body(items,names):
        result=[]
        for item in items:
            if type(item) is Loop:
                if item.iter_args or item.yields or item.results:
                    raise ValueError('shared RHS transfer cannot rename carried completion')
                name=item.var
                while name in used:name+='_' 
                used.add(name)
                result.append(replace(item,var=name,body=body(item.body,{**names,item.var:name})))
            else:
                result.append(replace(item,args=tuple((k,_rename(v,names)) for k,v in item.args)))
        return tuple(result)
    result=replace(kernel,body=body(kernel.body,{}))
    if not _same_stream(kernel,result):
        raise ValueError('shared RHS binder transport changed counted effects')
    return result


def _same_stream(left, right):
    """Exact counted tree equality modulo binders and additive normalization."""
    if replace(left, body=()) != replace(right, body=()):
        return False
    def body(a,b,names):
        if len(a)!=len(b):return False
        for x,y in zip(a,b,strict=True):
            if type(x) is Loop and type(y) is Loop:
                if replace(x,var=y.var,body=())!=replace(y,body=()):return False
                if not body(x.body,y.body,{**names,x.var:y.var}):return False
            elif type(x) is Call and type(y) is Call:
                call=replace(x,args=tuple((k,_rename(v,names)) for k,v in x.args))
                if not _same_band_call(call,y):return False
            else:return False
        return True
    return body(left.body,right.body,{})


def _codecs(n):
    identity=sha256(repr(sorted(REFERENCE_FILES.items())).encode()).hexdigest()
    law=TileLoadAddressLaw(isa.DIM,isa.DIM,64,1,n,isa.DIM,
        'selected-gemmini-software-block-mvin:'+identity)
    def project(call):
        if call.instr!=RHS:return None
        a=dict(call.args)
        return TileLoadMapping('rhs','spad',a['dram_row']*n+a['dram_col'],
            a['addr'],a['rows'],a['cols'],law)
    def encode(call,columns):
        return replace(call,args=tuple((k,Const(columns) if k=='cols' else v)
            for k,v in call.args))
    def encode_band(call,band):
        a=dict(call.args);mapping=project(call)
        offset=band.source_byte+mapping.source_byte*(-1)
        a.update(block1=a['block1']+_quotient(offset,isa.DIM),
            dram_col=a['dram_col']+offset,addr=band.local_row,
            rows=band.rows,cols=band.columns)
        return replace(call,args=tuple((k,a[k]) for k,_ in call.args))
    return project,encode,encode_band


@dataclass(frozen=True)
class SharedRHSTransfers:
    original: object
    source_kernel: object
    kernel: object
    n: int
    report: dict
    implementation_sha256: str

    def revalidate(self):
        self.original.revalidate()
        if sha256(Path(__file__).read_bytes()).hexdigest()!=self.implementation_sha256:
            raise ValueError('shared RHS transfer implementation changed')
        project,_,encode_band=_codecs(self.n)
        before,_=expand_panel_tile_loads(self.source_kernel,project=project,encode_band=encode_band)
        after,receipt=expand_panel_tile_loads(self.kernel,project=project,encode_band=encode_band)
        if (not _same_stream(after,before)
                or self.kernel.digest()!=self.report['kernel_sha256']
                or receipt['actual_mapping_sha256']!=self.report['projection']['actual_mapping_sha256']):
            raise ValueError('wide shared RHS calls changed original bands or completion epochs')


def select_rhs_block_transfers(original, epoch_kernel, n):
    """Consume the actual pure shared staging epoch; retain exact tails/refusals.

    The original epoch's noninterference proof covers every primitive band.
    Equality after final-operand expansion transports that proof to the wide
    commands without relaxing footprints or enlarging finite issued bounds.
    A proposal supplies no load completion law. All original real drains stay.
    """
    original.revalidate()
    project,encode,encode_band=_codecs(n)
    proposal,changes=propose_tile_load_coalescing(epoch_kernel,
        project=project,encode=encode,max_tiles=4,coalesce_tail=True)
    if not changes['changes']:
        return None,{'applied':False,'reason':'no complete adjacent RHS tile load loop',
            'proposal':changes}
    proposal=_unique_binders(proposal)
    before,old=expand_panel_tile_loads(epoch_kernel,project=project,encode_band=encode_band)
    after,new=expand_panel_tile_loads(proposal,project=project,encode_band=encode_band)
    if not _same_stream(after,before):
        raise ValueError('expanded wide RHS changed original shared counted effects')
    report={'applied':True,'source_kernel_sha256':epoch_kernel.digest(),
        'kernel_sha256':proposal.digest(),'proposal':changes,'projection':new,
        'original_physical_rhs_commands':old['physical_load_commands'],
        'physical_rhs_commands':new['physical_load_commands'],
        'logical_rhs_bands':new['expanded_logical_tile_bands'],
        'all_data_and_completion_bands_equal':True,'reference_files':REFERENCE_FILES,
        'proof':'final actual operands expanded; exact original counted stream including all drains',
        'completion':'original proved staging/execute/readout epochs unchanged'}
    result=SharedRHSTransfers(original,epoch_kernel,proposal,n,report,
        sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate()
    return result,report
