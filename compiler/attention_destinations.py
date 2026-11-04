"""Direct private integer head readouts into their exact concat destinations.

The shared scalar geometry reader proves each original zero-seeded product.
This adapter changes only physical placement of complete contiguous row runs;
it neither rewrites arithmetic nor changes source operation/observer order.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from xdsl.dialects import builtin, tensor
from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_row_batching_check import (
    geometry, boundary_attributes, RowBatchingUnavailable)

ENABLED = False


@dataclass(frozen=True)
class DestinationBinding:
    root: object
    value: object
    destination: object
    element_offset: int
    element_count: int
    source_op_index: int
    concat_source_op_index: int


@dataclass(frozen=True)
class DestinationPlan:
    module: object
    module_sha256: str
    source_sha256: str
    bindings: tuple[DestinationBinding, ...]
    concats: frozenset

    @property
    def private_values(self):
        return frozenset(binding.value for binding in self.bindings)

    def revalidate(self):
        if ir_digest(self.module) != self.module_sha256:
            raise ValueError('attention destination source changed; replan before allocation')
        for binding in self.bindings:
            shape,dtypes=geometry(binding.root,index_bits=64)
            if dtypes!=('i8','i8','i32') or shape[0]*shape[2]!=binding.element_count:
                raise ValueError('attention destination product proof changed')
            uses=tuple(binding.value.uses)
            if len(uses)!=1 or uses[0].operation not in self.concats:
                raise ValueError('attention destination private observer changed')

    def receipt(self):
        self.revalidate()
        return dict(schema='gemmini.private_concat_readout.v1',
            source_sha256=self.source_sha256,checked_module_sha256=self.module_sha256,
            owner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            source_arithmetic='shared geometry: original zero-seeded signed i8 product to i32',
            placement='complete disjoint contiguous dim0 row runs in original concat allocation',
            original_operations='unchanged; every scalar recurrence and seed retained',
            removed_copy_bytes_each_direction=4*sum(b.element_count for b in self.bindings),
            bindings=[dict(source_op_index=b.source_op_index,
                concat_source_op_index=b.concat_source_op_index,
                destination_element_offset=b.element_offset,elements=b.element_count)
                for b in self.bindings],
            concat_source_op_indices=sorted(set(b.concat_source_op_index for b in self.bindings)),
            timing='UNKNOWN until matched complete-region execution')


def plan_attention_destinations(module, *, source_sha256,
                                source_positions=None, excluded_roots=()):
    """Plan on immutable typed source before ABI/allocation; no partial writes.

    Caller allocates every destination independently from source inputs and
    binds each private value to destination+4*element_offset before its original
    producer. At the original concat site all producer completion obligations
    already hold; its result becomes materialized without copying. Unsupported
    observers/layouts/seeds or overlaps retain ordinary source lowering.
    """
    if (not isinstance(source_sha256,str) or len(source_sha256)!=64
        or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('attention destinations require original source SHA256')
    module.verify()
    functions=[op for op in module.body.block.ops if op.name=='func.func']
    if len(functions)!=1 or len(functions[0].regions[0].blocks)!=1:return None
    block=functions[0].regions[0].block
    operations=tuple(op for op in block.ops if op.name!='func.return')
    positions={op:i for i,op in enumerate(operations)}
    source_positions=positions if source_positions is None else source_positions
    excluded=set(excluded_roots);bindings=[];concats=set();claimed=set()
    for concat in operations:
        if (type(concat) is not tensor.ConcatOp or not boundary_attributes(concat)
            or len(concat.results)!=1 or len(concat.operands)<2
            or concat.dim.value.data!=0):continue
        result=concat.results[0];typ=result.type
        if (type(typ) is not builtin.TensorType or type(typ.encoding) is not builtin.NoneAttr
            or str(typ.element_type)!='i32' or len(typ.get_shape())!=2
            or any(d<=0 for d in typ.get_shape())):continue
        rows,columns=typ.get_shape();offset=0;candidate=[]
        for value in concat.operands:
            root=value.owner
            if (root in excluded or root in claimed or root not in positions
                or positions[root]>=positions[concat] or value in claimed
                or len(tuple(value.uses))!=1
                or next(iter(value.uses)).operation is not concat):break
            try:(m,k,n),dtypes=geometry(root,index_bits=64)
            except (RowBatchingUnavailable,AttributeError):break
            if (dtypes!=('i8','i8','i32') or len(root.results)!=1
                or value is not root.results[0] or n!=columns or m<=0
                or offset+m*n>rows*columns):break
            candidate.append(DestinationBinding(root,value,result,offset,m*n,
                source_positions[root],source_positions[concat]))
            offset+=m*n
        else:
            # Duplicate operands or incomplete coverage cannot share slices.
            if offset!=rows*columns or len({b.value for b in candidate})!=len(candidate):continue
            bindings.extend(candidate);concats.add(concat)
            claimed.update(b.root for b in candidate)
    if not bindings:return None
    plan=DestinationPlan(module,ir_digest(module),source_sha256,tuple(bindings),frozenset(concats))
    plan.revalidate()
    return plan
