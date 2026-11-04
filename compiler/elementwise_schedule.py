"""Checked optional loaded-sum epochs, emitted through the original Encoder.

Three runtime buffer intervals must be disjoint. No source ABI restrict premise
is inferred from pointer SSA identity. The caller retains its synchronous arm.
"""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.compact_loaded_sum import (
    LoadedSumContract, LoadedSumEffect, LoadedSumLayout)
from merlin.compile.scheduling.loaded_sum_epochs import (
    LoadedSumEpochLaw, plan_loaded_sum_epochs, emit_loaded_sum_epochs)
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from merlin.sched.ir.kernel import Ptr

ROOT=Path(__file__).resolve().parent
ACC=1<<31
ADD=1<<30


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def choice(attrs):
    selected=attrs.get('elementwise_schedule',os.environ.get('ELEMENTWISE_SCHEDULE'))
    if selected is None:
        selected=knob('elementwise_schedule',default='sync',
            values=('sync','epochs4','epochs16','epochs64'),root=ROOT)
    if selected not in ('sync','epochs4','epochs16','epochs64'):
        raise ValueError(f'invalid elementwise_schedule {selected!r}')
    return selected


def _bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


@lru_cache(maxsize=4)
def _target(facts_path,ordering_path):
    # This existing selected-target checker reopens source hashes, native
    # configuration issue lanes, finite queues and disjoint local dependencies.
    from convolution import _ordering_admission, _resources
    admission=_ordering_admission(facts_path,ordering_path)
    if not admission['admitted']:
        raise ValueError('loaded-sum epoch ordering refusal: '+admission['reason'])
    edge,_,rows,hardware=_resources(facts_path)
    if edge!=16:
        raise ValueError('loaded-sum Encoder primitive is specifically 16x16')
    return rows,hardware,admission


def plan_residual(shape,proof,relu,mode,attrs):
    """Bind scalar proof and actual native load/store operands to compact cover."""
    from math import prod
    rows,columns=prod(shape[:-1]),shape[-1]
    facts_path=str(attrs.get('facts_path',ROOT/'data/facts.json'))
    ordering_path=str(attrs.get('ordering_path',ROOT/'data/command_ordering.json'))
    acc_rows,hardware,admission=_target(facts_path,ordering_path)
    layout=LoadedSumLayout(rows,columns,16,16,0,16*4,acc_rows*16*4)
    scalar=dict(proof,activation=bool(relu),rounding='f32-half-even',output_dtype='i8')
    parameters=(tuple(sorted(dict(scale_bits=_bits(proof['lhs_load_f32']),stream=0,shrink=1,stride=columns).items())),
        tuple(sorted(dict(scale_bits=_bits(proof['rhs_load_f32']),stream=1,shrink=1,stride=columns).items())),
        tuple(sorted(dict(scale_bits=_bits(proof['readout_scale_f32']),relu=int(relu),stride=columns).items())))
    contract=LoadedSumContract(_sha(scalar),parameters,hardware,True,True)
    m,n=Var('m'),Var('n');row=m*16;column=n*16
    mt,nt=(rows+15)//16,(columns+15)//16
    height=select('m',mt-1,(rows-1)%16+1,16)
    width=select('n',nt-1,(columns-1)%16+1,16)
    calls=[]
    for index,(name,tensor) in enumerate((('MVIN','lhs'),('MVIN2','rhs'),('MVOUT','output'))):
        args=dict(dram=Ptr(tensor,row*columns+column),local_addr=Const(ACC|(ADD if index==1 else 0)),
            num_rows=height,num_cols=width,**{key:Const(value) for key,value in parameters[index]})
        calls.append(Call(name,tuple(args.items())))
    kernel=Kernel('residual_native_sync',tuple(TensorArg(name,(rows,columns),'i8',access)
        for name,access in (('lhs','read'),('rhs','read'),('output','write'))),
        (Loop('m',mt,(Loop('n',nt,tuple(calls)),)),))
    def project(call):
        # Decode actual target transfer descriptors. DRAM byte addressing,
        # local accumulator flags and all scalar config bits must match before
        # exposing any logical coordinates to the shared cover checker.
        index={'MVIN':0,'MVIN2':1,'MVOUT':2}.get(call.instr)
        if index is None:raise ValueError('unexpected loaded-sum native instruction')
        tensor=('lhs','rhs','output')[index]
        if (call.arg('dram')!=Ptr(tensor,row*columns+column)
            or call.arg('local_addr')!=Const(ACC|(ADD if index==1 else 0))
            or tuple((key,call.arg(key).value) for key,_ in parameters[index])!=parameters[index]):
            raise ValueError('native loaded-sum address or scalar parameters differ')
        return LoadedSumEffect(('initialize','accumulate','store')[index],tensor,
            row,column,call.arg('num_rows'),call.arg('num_cols'),Const(0),64,
            contract.scalar_proof_sha256,tuple((key,call.arg(key).value) for key,_ in parameters[index]),'synchronous')
    # Same selected native lane captures invariant configuration throughout
    # each phase. No configuration value changes until a full phase drain.
    qualification=dict(selected_ordering=admission,phase_configuration='invariant; changed only across complete drains',
        disjointness='runtime overflow-safe three-buffer interval guard',
        private_rows='distinct 16-row ACC slots, full phase completion before reuse',
        completion='original Encoder.fence native full-device completion')
    law=LoadedSumEpochLaw(hardware,contract.scalar_proof_sha256,_sha(qualification),
        disjoint_loads_pending=True,disjoint_stores_pending=True,
        ordered_configuration_capture=True,finite_queue_progress=True,
        complete_device_drain=True,exclusive_private_state=True)
    plan=plan_loaded_sum_epochs(kernel,layout,contract,project,scalar_proof=scalar,
        target_law=law,max_live_tiles=int(mode.removeprefix('epochs')),flatten_rows=True)
    plan.revalidate(kernel,layout,contract,project,scalar_proof=scalar)
    return plan,dict(source_kernel=kernel.text(),source_contract=asdict(contract),
        scalar_proof=scalar,target_qualification=qualification,pass_order=[
            'verify_compact_loaded_sum','plan_loaded_sum_epochs','emit_loaded_sum_epochs'],
        options=dict(max_live_tiles=plan.max_live_tiles,flatten_rows=True))


def disjoint_condition(enc,pointers,nbytes):
    """Unsigned interval separation, with explicit rejection of address wrap."""
    b=enc.index_builder
    starts=[enc.address(p) for p in pointers]
    ends=[b.op('add',p,nbytes) for p in starts]
    condition=b.cmp('ult',starts[0],ends[0])
    for start,end in zip(starts[1:],ends[1:]):
        condition=b.op('and',condition,b.cmp('ult',start,end),dtype='i1')
    for i in range(3):
        for j in range(i+1,3):
            from xdsl.dialects import llvm
            apart=enc.fn.add(llvm.OrOp(b.cmp('ule',ends[i],starts[j]),b.cmp('ule',ends[j],starts[i]))).results[0]
            condition=b.op('and',condition,apart,dtype='i1')
    return condition


def emit_residual_epochs(enc,pointers,plan,proof,relu):
    lhs,rhs,out=pointers
    columns=plan.layout.columns
    def lower(kind,fields):
        if kind=='complete_phase':enc.fence();return
        row=enc.fn.udiv_i(fields['local_byte'],enc.val(64))
        offset=enc.add(enc.mul(fields['row'],columns),fields['column'])
        h,w=fields['rows'],fields['columns']
        if kind=='initialize':
            enc.load(enc.ptr_offset(lhs,offset),enc.flag(row,ACC),h,w,columns,
                scale=proof['lhs_load_f32'],shrunk=True)
        elif kind=='accumulate':
            enc.load(enc.ptr_offset(rhs,offset),enc.flag(row,ACC|ADD),h,w,columns,
                stream=1,scale=proof['rhs_load_f32'],shrunk=True)
        elif kind=='store':
            enc.store(enc.ptr_offset(out,offset),row,h,w,columns,dtype='i8',
                scale=proof['readout_scale_f32'],relu=relu)
        else:raise ValueError('unknown loaded-sum epoch effect')
    enc.fence() # complete any preceding region before taking private ACC ownership
    return emit_loaded_sum_epochs(plan,enc.index_builder,lower)
