"""Exact logical row concatenation over separately allocated source matrices.

The planner never assumes unrelated pointer arguments are adjacent. Its virtual
row axis must be split at every member boundary by the terminal DMA consumer.
Source discovery owns permission to reorder products; this plan establishes only
geometry, immutable common RHS identity, numeric agreement and pointer coverage.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import prod


class SourcePointerGroupError(ValueError):
    pass


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _numeric(attributes):
    # Source namespaces are annotations in this command-buffer adapter. Every
    # other attribute remains in the numeric equality check.
    return {key:deepcopy(value) for key,value in attributes.items()
            if not key.startswith(('prov.','merlin.'))}


def _products(cb):
    active,products,bound={},{},{}
    for index,command in enumerate(cb.get('commands',())):
        code=command.get('opcode');a=command.get('operands',{});attrs=command.get('attributes',{})
        if code=='RES_PACK':
            active[a.get('dst')]=a.get('src')
        elif code=='EVICT':
            active.pop(a.get('handle'),None)
        elif code in ('MATMUL','MATMUL_RESIDENT'):
            rhs=a.get('rhs') if code=='MATMUL' else active.get(a.get('rhs'))
            products[a.get('dst')]=(a.get('lhs'),rhs,dict(attrs))
        elif code=='COMMIT':
            item=products.get(a.get('src'))
            if item is not None:
                lhs,rhs,source_attrs=item
                bound[index]={'command_index':index,'lhs':lhs,'rhs':rhs,'dst':a.get('dst'),
                    'attributes':{**source_attrs,**attrs}}
    return bound


@dataclass(frozen=True)
class RowPointerView:
    start: int
    stop: int
    lhs: str
    rhs: str
    dst: str
    command_index: int
    source_index: int | None
    source_ids: dict=field(default_factory=dict)

    def record(self):
        return {'rows':[self.start,self.stop],'lhs':self.lhs,'rhs':self.rhs,'dst':self.dst,
            'command_index':self.command_index,'source_index':self.source_index,
            'source_ids':deepcopy(self.source_ids),'local_row':'virtual_row - rows[0]'}


@dataclass(frozen=True)
class RowPointerGroup:
    m: int
    k: int
    n: int
    rows_per_member: int
    members: tuple
    row_intervals: tuple[RowPointerView,...]
    attributes: dict
    output_dtype: str
    _binding: str=field(repr=False)

    def record(self):
        return {'schema':'submission.disjoint_source_row_group.v1',
            'geometry':{'m':self.m,'k':self.k,'n':self.n},
            'rows_per_member':self.rows_per_member,'members':[v.record() for v in self.row_intervals],
            'attributes':deepcopy(self.attributes),'output_dtype':self.output_dtype,
            'source_binding_sha256':self._binding,
            'rhs_identity':self.row_intervals[0].rhs,
            'lhs_address':'member.lhs + ((virtual_row-member.start)*K + column) i8 elements',
            'output_address':'member.dst + ((virtual_row-member.start)*N + column) output elements',
            'row_partition':'ordered disjoint intervals cover [0,M) exactly',
            'storage':'distinct source logical buffers; no physical adjacency inferred',
            'consumer_obligations':['split each DMA transfer at every member row boundary',
                'retain original source IDs for each member load/store',
                'prove device-local tile addresses and exact MAC/readout coverage',
                'establish independent product reordering before grouping'],
            'scope':'source pointer/numeric/row coverage proof; no scheduling, performance or reordering grant'}

    def revalidate(self,cb):
        fresh=plan_disjoint_row_groups(cb,self.members)
        if fresh._binding!=self._binding or fresh.record()!=self.record():
            raise SourcePointerGroupError('source command, pointer binding, geometry or numeric contract changed')
        return self.record()


def plan_disjoint_row_groups(cb,rows):
    """Bind equal-shape shared-RHS products to a checked virtual row partition.

    ``rows`` uses the resident_rhs discovery interface: command_index, lhs, rhs,
    dst, attributes, plus optional m/k/n. Returns RowPointerGroup with .m equal
    to the TOTAL virtual rows, .rows_per_member and .row_intervals. The caller
    must catch SourcePointerGroupError before emitting any target command.
    """
    rows=tuple(deepcopy(row) for row in rows)
    if not 2<=len(rows)<=32:
        raise SourcePointerGroupError('row groups need between 2 and 32 source products')
    bound=_products(cb);tensors=cb.get('tensors',{})
    aliases=cb.get('params',{}).get('tensor_aliases',{})
    alias_names=set(aliases)|{a.get('source') for a in aliases.values() if isinstance(a,dict)}
    prepared=[]
    def spec(name):
        item=tensors.get(name)
        if not isinstance(item,dict):raise SourcePointerGroupError('member has no typed source tensor')
        shape=item.get('shape',())
        if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):
            raise SourcePointerGroupError('source row views require positive static rank-two matrices')
        if any(key in item for key in ('strides','offset','layout','encoding')):
            raise SourcePointerGroupError('source row view has an unproved explicit layout or strided offset')
        width={'i8':1,'i32':4}.get(item.get('dtype'))
        if width is None or prod(shape)*width>=1<<63:
            raise SourcePointerGroupError('source row view has unsupported dtype or address extent')
        return item
    for row in rows:
        index=row.get('command_index')
        if type(index) is not int or index not in bound:
            raise SourcePointerGroupError('member has no original complete source COMMIT')
        original=bound[index]
        if any(row.get(key)!=original[key] for key in ('lhs','rhs','dst','attributes')):
            raise SourcePointerGroupError('member pointer or attributes differ from the original source product')
        a,b,c=(spec(row[key]) for key in ('lhs','rhs','dst'))
        m,k=a['shape'];bk,n=b['shape']
        if a['dtype']!='i8' or b['dtype']!='i8' or c['shape']!=[m,n] or bk!=k:
            raise SourcePointerGroupError('member matrix geometry or signed operand types disagree')
        if any(key in row and row[key]!=value for key,value in (('m',m),('k',k),('n',n))):
            raise SourcePointerGroupError('member geometry annotation differs from its source matrices')
        attrs=_numeric(row['attributes'])
        attrs.setdefault('epilogue',[]);attrs.setdefault('output_dtype',c['dtype'])
        if attrs['output_dtype']!=c['dtype']:
            raise SourcePointerGroupError('member output numeric contract differs from its typed result')
        auxiliary={attrs[key] for key in ('bias','accumulator_init') if attrs.get(key)}
        if auxiliary:
            raise SourcePointerGroupError('auxiliary bias/initializer pointers need an explicit grouped row projection')
        if ({row['lhs'],row['rhs'],row['dst']}|auxiliary)&alias_names:
            raise SourcePointerGroupError('source aliases have no disjoint contiguous member proof')
        prepared.append((m,k,n,c['dtype'],attrs,auxiliary))
    first=prepared[0]
    if any(item[:5]!=first[:5] for item in prepared[1:]):
        raise SourcePointerGroupError('ragged member geometry, dtype or numeric readout needs separate groups')
    if len({row['rhs'] for row in rows})!=1:
        raise SourcePointerGroupError('members do not share the identical original RHS value')
    outputs=[row['dst'] for row in rows]
    inputs={row[key] for row in rows for key in ('lhs','rhs')}
    inputs.update(name for item in prepared for name in item[5])
    indices=[row['command_index'] for row in rows]
    if len(set(outputs))!=len(outputs) or set(outputs)&inputs:
        raise SourcePointerGroupError('member output overlaps a live source input or another output')
    if indices!=sorted(set(indices)):
        raise SourcePointerGroupError('member source order or complete-product uniqueness changed')
    m,k,n,dtype,attrs,_=first
    if len(rows)*m*k>=1<<63 or len(rows)*m*n*({'i8':1,'i32':4}[dtype])>=1<<63:
        raise SourcePointerGroupError('virtual row extent exceeds signed address arithmetic')
    intervals=tuple(RowPointerView(i*m,(i+1)*m,row['lhs'],row['rhs'],row['dst'],row['command_index'],
        row['attributes'].get('merlin.source_op_index'),
        {key:deepcopy(value) for key,value in row['attributes'].items() if key.startswith(('prov.','merlin.source'))})
        for i,row in enumerate(rows))
    binding=_digest({'members':rows,'tensors':{name:tensors[name] for name in sorted(inputs|set(outputs))},
        'row_intervals':[view.record() for view in intervals]})
    return RowPointerGroup(len(rows)*m,k,n,m,rows,intervals,attrs,dtype,binding)
