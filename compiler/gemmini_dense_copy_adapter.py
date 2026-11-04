"""Conservative byte-effect bridge from actual dense calls to private-copy IR.

The dense terminal issues RoCC commands, whose completion is asynchronous and
whose arithmetic/descriptor state cannot be represented as synchronous copies.
This bridge decodes actual addresses, including every row and wide-load band;
it deliberately leaves completion UNKNOWN when no actual token wait exists.
It grants no launch motion from queue capacity or disjointness alone.
"""
from dataclasses import asdict, replace
import hashlib
import json

from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store
from merlin.compile.scheduling.copy_pipeline import CopyPipelineAdmission
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.sched.ir import evaluate


def copy_storage_boundary(adapter, problem, geometry, facts_path):
    """Return actual byte windows and a no-motion selected admission.

    A dense load may be strided externally and banded locally. Treating its
    bounding box as one exact copy would authorize reading unrelated bytes.
    Global fence has arithmetic/readout completion effects too, so cannot be
    reclassified as the copy pass's pure token-completion wait.
    """
    edge, operand, rows, accumulator, accrows = geometry
    facts_bytes=facts_path.read_bytes();facts=json.loads(facts_bytes)
    queue=facts['facts']['timing_structure']['command_classes']['classes']['load']['entries']
    if queue.get('status')!='derived' or type(queue.get('value')) is not int or queue['value']<1:
        raise ValueError('copy bridge requires selected finite load queue capacity')
    private=operand.name;acc=accumulator.name
    def project(call):
        concrete=replace(call,args=tuple((key,evaluate(value,{})) for key,value in call.args))
        key=json.dumps(dict(instruction=call.instr,operands=dict(concrete.args)),sort_keys=True)
        accesses=[]
        try:op=adapter.project(concrete)
        except ValueError:
            return StorageEffects(key,completion='unknown')
        if isinstance(op,Load):
            pitch=problem.k if op.role=='lhs' else problem.n
            element=problem.lhs_bytes if op.role=='lhs' else problem.rhs_bytes
            for row in range(op.rows):
                accesses.append(EncodedAccess('external.'+op.role,
                    ((op.dram_row+row)*pitch+op.dram_col)*element,op.cols*element))
            for band in range(0,op.cols,edge):
                width=min(edge,op.cols-band)
                for row in range(op.rows):
                    accesses.append(EncodedAccess(private,
                        (op.row+(band//edge)*edge+row)*operand.row_bytes,width*element,True))
        elif isinstance(op,Preload):
            for row in range(op.weight_rows):
                accesses.append(EncodedAccess(private,(op.weight_row+row)*operand.row_bytes,
                    op.cols*problem.rhs_bytes))
            # PRELOAD writes a descriptor, not these accumulator result bytes.
            # Its hidden descriptor/compute state is intentionally UNKNOWN.
        elif isinstance(op,Compute):
            for row in range(op.rows):
                accesses.append(EncodedAccess(private,(op.input_row+row)*operand.row_bytes,
                    op.cols*problem.lhs_bytes))
        elif isinstance(op,Store):
            for row in range(op.rows):
                accesses.extend((EncodedAccess(acc,(op.accumulator_row+row)*accumulator.row_bytes,
                    op.cols*problem.accumulator_bytes),EncodedAccess('external.output',
                    ((op.dram_row+row)*problem.n+op.dram_col)*problem.accumulator_bytes,
                    op.cols*problem.accumulator_bytes,True)))
        return StorageEffects(key,tuple(accesses),completion='unknown')
    def rewrite(call,accesses):
        if accesses!=project(call).accesses:
            raise ValueError('dense copy bridge does not qualify address relocation')
        return call
    qualification='gemmini.dense-copy-byte-projection.v1:completion-unknown:no-motion'
    storage_adapter=StorageAdapter(project,rewrite,adapter.configuration_identity,
        adapter.numerical_policy_identity,qualification)
    banks=(StorageBank(private,private,rows*operand.row_bytes,'bytes',operand.row_bytes),
        StorageBank(acc,acc,accrows*accumulator.row_bytes,'bytes',accumulator.row_bytes))
    objects=tuple(TemporaryObject('dense.'+bank.name,bank.name,0,bank.capacity,bank.alignment) for bank in banks)
    windows=(AddressWindow('external.lhs',0,problem.m*problem.k*problem.lhs_bytes),
        AddressWindow('external.weight',0,problem.k*problem.n*problem.rhs_bytes),
        AddressWindow('external.output',0,problem.m*problem.n*problem.accumulator_bytes))
    declaration=dict(crossing_pairs=[],concurrent_pairs=[],max_in_flight=1,
        queue_capacity=queue['value'],completion='UNKNOWN; global drain is not a pure copy wait')
    admission=CopyPipelineAdmission((),(),1,queue['value'],adapter.configuration_identity,
        qualification,'gemmini.dense-copy-empty-motion.v1',hashlib.sha256(facts_bytes).hexdigest(),
        hashlib.sha256(json.dumps(declaration,sort_keys=True).encode()).hexdigest())
    return storage_adapter,objects,banks,windows,admission
