"""Serial sibling output panels sharing one immutable complete LHS window.

Every member is an original checked compact dense schedule. The only data
rewrite suppresses identical LHS loads and projects their MAC addresses into
one capacity-checked retained window. RHS and accumulator windows have disjoint
lifetimes: completed readout closes each panel before either window is reused.
This is an opt-in lowering; no incumbent selector is changed by importing it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from functools import lru_cache
from math import isfinite
from pathlib import Path
import json
from typing import Mapping, Sequence, Any

from merlin.compile.scheduling.compact_dense import (
    DenseLoopStorage, DenseMemoryView, DenseBiasSource, schedule_compact_dense)
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.sched.ir import Call, Loop
from . import isa
from .dense import _adapter
from .resident_rhs import _retain_lhs, _prepare_lhs
from .shared_products import _epoch_index_kernel, _emit_shared_index_kernel


@dataclass(frozen=True)
class SharedPanelCapacity:
    """Physical windows, in DIM-element rows, shared only across closed epochs."""
    m: int
    k: int
    lhs_rows: int
    rhs_base: int
    rhs_rows: int
    accumulator_rows: int
    max_n_blocks: int


@lru_cache(maxsize=128)
def checked_member(m: int, k: int, n: int, seeded: bool = False):
    """Return an original library checked schedule and retained-LHS rewrite."""
    d = isa.DIM
    mt, kt, nt = ((int(x)+d-1)//d for x in (m,k,n))
    # All M and K tiles remain immutable throughout all sibling output panels.
    # Small-M is a resource class: its complete input must fit with a full-K
    # RHS primitive panel, rather than an empirical capsule-specific cutoff.
    lhs_rows = mt*kt*d
    rhs_rows = isa.SPAD_ROWS-lhs_rows
    if min(m,k,n) <= 0 or rhs_rows < kt*d or mt*d > isa.ACC_ROWS:
        raise ValueError('complete shared LHS and full-K RHS panel do not fit')
    bn = min(nt, rhs_rows//(kt*d), isa.ACC_ROWS//(mt*d))
    if bn < 1:
        raise ValueError('no disjoint output panel fits')
    capacity = SharedPanelCapacity(m,k,lhs_rows,lhs_rows,rhs_rows,isa.ACC_ROWS,bn)
    storage = DenseLoopStorage(
        DenseMemoryView('shared_lhs','spad',d,0,lhs_rows),
        DenseMemoryView('serial_rhs','spad',d,lhs_rows,rhs_rows),
        DenseMemoryView('serial_acc','acc',4*d,0,isa.ACC_ROWS))
    problem = DenseProblem(m,k,n,d,d,d,1,1,4,bias=seeded)
    resources = DenseResources(isa.SPAD_ROWS*d,isa.ACC_ROWS*d*4,
        max_m_blocks=mt,max_n_blocks=bn,max_k_blocks=kt,
        partial_output_tiles=True,partial_output_blocks=True)
    schedule = schedule_compact_dense(problem,resources,storage,_adapter(),
        name='shared_product_panels',dtypes=('i8','i8','i32'),static_budget=256,
        external_buffers_disjoint=True,verification_timeout_ms=5000,
        bias_source=DenseBiasSource(n,'i32') if seeded else None)
    schedule.revalidate()
    # Full K and M residency is explicit, not inferred from command count.
    if schedule.choice.block.k != kt or schedule.choice.block.m != mt:
        raise ValueError('library chose a different retained panel geometry')
    retained, proof = _retain_lhs(schedule.kernel,(m,k,n),0)
    return schedule, retained, proof, capacity


def _owned(kernel, member):
    def walk(item):
        if isinstance(item,Loop):
            return replace(item,body=tuple(walk(child) for child in item.body))
        if not isinstance(item,Call):
            raise ValueError('unknown checked dense operation')
        return replace(item,instr=f'shared_member{member}_'+item.instr)
    return replace(kernel,body=tuple(walk(item) for item in kernel.body))


def lower_shared_product_panels(ctx: Any, members: Sequence[Mapping[str,Any]]):
    """Emit independent sibling rows, or return None before context mutation.

    Rows use the existing lhs/rhs/dst/attributes/command_index interface. The
    caller consumes original COMMITs only after this returns a receipt. Unknown
    epilogues, mixed per-column scales, aliases and dependencies are refused;
    their exact existing lowering remains available. Bias, matrix initializer,
    scalar scale and ReLU use the common dense terminal consumer unchanged.
    """
    members = tuple(members)
    if not 2 <= len(members) <= 8 or any(r['lhs'] != members[0]['lhs'] for r in members):
        return None
    outputs = [r['dst'] for r in members]
    reads = {r[key] for r in members for key in ('lhs','rhs')}
    reads.update(r.get('attributes',{}).get(key) for r in members
                 for key in ('bias','accumulator_init') if r.get('attributes',{}).get(key) is not None)
    aliases = ctx.cb.get('params',{}).get('tensor_aliases',{})
    if len(set(outputs)) != len(outputs) or reads.intersection(outputs) or any(x in aliases for x in reads.union(outputs)):
        return None
    order = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if order.get('completion_order',{}).get('execute',{}).get('order') != 'in_order':
        return None
    prepared, specs = [], []
    try:
        for index,row in enumerate(members):
            attrs = row.get('attributes',{})
            lhs,rhs,dst = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
            a,b,c = lhs['shape'],rhs['shape'],dst['shape']
            if lhs['dtype'] != 'i8' or rhs['dtype'] != 'i8' or dst['dtype'] not in ('i8','i32'):
                return None
            if len(a) != 2 or len(b) != 2 or a[1] != b[0] or c != [a[0],b[1]]:
                return None
            m,k,n = int(a[0]),int(a[1]),int(b[1])
            if specs and (m,k) != specs[0][:2]: return None
            if any(key in row and row[key] != value for key,value in (('m',m),('k',k),('n',n))): return None
            if attrs.get('output_dtype',dst['dtype']) != dst['dtype']: return None
            if any(e not in ('bias_add','acc_scale','relu') for e in attrs.get('epilogue',())): return None
            if attrs.get('requant_shift') is not None or attrs.get('block_scale') is not None: return None
            scales = attrs.get('acc_scale_by_column')
            if scales is not None and (len(scales) != n or len(set(map(float,scales))) != 1): return None
            scale = float(scales[0] if scales else attrs.get('acc_scale',1.0))
            if not isfinite(scale) or scale < 0: return None
            if dst['dtype'] == 'i32' and (scale != 1 or scales is not None or attrs.get('relu') or 'relu' in attrs.get('epilogue',())): return None
            bias,init = attrs.get('bias'),attrs.get('accumulator_init')
            if bias and (ctx.spec(bias)['dtype'] != 'i32' or ctx.spec(bias)['shape'] != [n]): return None
            if init and (ctx.spec(init)['dtype'] != 'i32' or ctx.spec(init)['shape'] != [m,n]): return None
            schedule,kernel,proof,capacity = checked_member(m,k,n,bool(bias or init))
            kernel,epochs = _epoch_index_kernel(_owned(kernel,index))
            specs.append((m,k,n))
            prepared.append((schedule,kernel,proof,capacity,epochs))
    except (ValueError,TypeError,KeyError,OverflowError):
        return None
    # All admission and proof work has finished. Reuse is separated by the
    # common consumer's readout completions and a final member completion.
    ctx.fence()
    ctx.current_source_index = members[0].get('attributes',{}).get('merlin.source_op_index')
    ctx.emit(*isa.config_ld(specs[0][1],channel=0))
    _prepare_lhs(ctx,members[0]['lhs'],specs[0][0],specs[0][1],0)
    ctx.fence()
    receipts = []
    for index,(schedule,kernel,proof,capacity,epochs) in enumerate(prepared):
        emission = _emit_shared_index_kernel(ctx,members,specs,kernel)
        ctx.fence()
        schedule.revalidate()
        receipts.append({'capacity':asdict(capacity),'retained_lhs':proof,
            'source_schedule':schedule.realization,'completion_epochs':epochs,
            'emission':emission,'kernel_sha256':kernel.digest()})
    receipt = {'mechanism':'shared_lhs_serial_output_panels','members':receipts,
        'lhs_source_loads':prepared[0][3].lhs_rows//isa.DIM,
        'completion':'immutable disjoint LHS; full staging/execute/readout completions; serial RHS/ACC lifetimes',
        'scope':'checked source MAC/init/store projection and common terminal emission; independent execution required'}
    ctx.metadata.setdefault('shared_product_panels',[]).append(receipt)
    return receipt
