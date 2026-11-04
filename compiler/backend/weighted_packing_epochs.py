"""Exact source-once DMA packing with completed immutable readout epochs.

This changes only the packing schedule. The caller retains its source-domain
numeric proof, distinct packed allocation, identity configurations and dense
consumer. The existing counted LLVM emitter consumes these instruction names.
"""
from __future__ import annotations

from merlin.sched.ir import Kernel, TensorArg, Call, Loop, Const, Var, as_expr, check_structure
from . import isa


def _admission():
    from .column_batches import _admission as selected_ordering
    facts=selected_ordering()
    # The selected ordering digest binds funct 2 to the load queue; the
    # existing emitter uses the default-channel mvin rather than mvin3.
    if isa.FUNCT['mvin'] != 2:
        raise ValueError('weighted packing default load encoding changed')
    if isa.ACC_ROWS < isa.DIM:
        raise ValueError('weighted packing lacks one complete ACC row tile')
    # No changing-CONFIG_ST snapshot assumption is needed. Same-lane finite
    # backpressure may serialize the stores, and their real drain precedes
    # both the next ACC write and any later configuration change.
    return dict(facts, store_configuration_snapshot=
        'unchanged identity CONFIG_ST through all read-only stores and closing full drain')


def _geometry(m,lhs_columns,rhs_columns):
    if any(type(n) is not int or n<=0 for n in (m,lhs_columns,rhs_columns)):
        raise ValueError('weighted packing needs positive static matrix geometry')
    k=lhs_columns+rhs_columns
    if k>256 or m*k>=1<<63:
        raise ValueError('weighted packing exceeds original bounded geometry')
    return k


def _kernel(m,lhs_columns,rhs_columns):
    k=_geometry(m,lhs_columns,rhs_columns)
    def call(name,**args):
        return Call(name,tuple((key,as_expr(value)) for key,value in args.items()))
    def source(row,rows,name,start,columns):
        return (call(name,offset=row,addr=isa.acc_addr(0),rows=rows,cols=1),
            call('drain'),
            Loop('column',columns,(call('store',offset=row*k+start+Var('column'),
                addr=isa.acc_addr(0),rows=rows,cols=1),)),
            call('drain'))
    def tile(row,rows):
        return (*source(row,rows,'load_lhs',0,lhs_columns),
            *source(row,rows,'load_rhs',lhs_columns,rhs_columns))
    full,tail=divmod(m,isa.DIM)
    body=[call('drain'),call('config_load',stride=1),call('config_store',stride=k)]
    if full:body.append(Loop('row',full,tile(Var('row')*isa.DIM,isa.DIM)))
    if tail:body.extend(tile(Const(full*isa.DIM),tail))
    return Kernel('source_weighted_residual_pack_epochs',(
        TensorArg('lhs',(m,),'i8','read'),TensorArg('rhs',(m,),'i8','read'),
        TensorArg('packed',(m,k),'i8','write')),tuple(body))


def packing_kernel(m,lhs_columns,rhs_columns):
    """Return an admitted alternative; raises before caller emission on refusal."""
    _geometry(m,lhs_columns,rhs_columns)
    _admission()
    kernel=_kernel(m,lhs_columns,rhs_columns)
    if check_structure(kernel):
        raise ValueError('invalid weighted packing counted schedule')
    return kernel


def check_packing(kernel,m,lhs_columns,rhs_columns):
    """Revalidate actual operands, phase order, completion and complete coverage.

    Whole Kernel equality includes tensor effects, all loop domains, every
    source/destination address, both identity configurations and all drains.
    With the checked static geometry, row tiles partition [0,M), and column
    ranges [0,L) and [L,K) partition each destination row. Thus every packed
    element has exactly one store, reading the required original source row.
    This finite-affine proof does not depend on an optional solver or M-sized
    enumeration, and mutation of any counted operand or boundary is rejected.
    """
    k=_geometry(m,lhs_columns,rhs_columns)
    admission=_admission()
    if check_structure(kernel) or kernel!=_kernel(m,lhs_columns,rhs_columns):
        raise ValueError('weighted packing epoch stream or tensor effects changed')
    tiles=(m+isa.DIM-1)//isa.DIM
    return dict(kernel_sha256=kernel.digest(),shape=[m,k],
        mechanism='source_once_immutable_accumulator_readout_epochs',
        proof='whole counted Kernel equality, including typed effects, operands, loop domains, configuration and completion; adjacent full/tail row partition and disjoint lhs/rhs column partition',
        source_address_bounds=[0,m-1],destination_address_bounds=[0,m*k-1],
        writes=m*k,write_multiplicity=1,
        address_law='packed[r,c] = lhs[r] for c<lhs_columns, otherwise rhs[r]',
        source_load_descriptors=2*tiles,store_descriptors=k*tiles,
        requested_load_bytes=2*m,requested_store_bytes=m*k,
        full_drains=1+4*tiles,maximum_live_accumulator_rows=min(m,isa.DIM),
        source_partition='tiles [16*t,min(16*(t+1),M)); each source loaded once per tile',
        destination_partition='row*K+column, row stride K; columns [0,L) and [L,K), disjoint even across pending stores',
        completion='initial full drain; load -> full drain -> all same-config read-only stores -> full drain before next ACC overwrite, reconfiguration or dense use',
        alias_contract='caller must allocate packed private and distinct from both inputs; lhs/rhs may be identical',
        arithmetic='unchanged identity signed-i8 load and signed-i8 store; numeric weighted proof and dense leaf unchanged',
        admission=admission)
