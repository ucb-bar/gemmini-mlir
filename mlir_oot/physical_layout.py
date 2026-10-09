"""Opt-in exact permutation propagation, with a live tensor-copy census.

The existing Merlin pass proves static dimensions and affine maps. Unknown
producers (including external calls) keep an explicit transpose. No call ABI,
weight identity, scalar arithmetic, or numeric policy is changed here.
"""
from collections import Counter
from math import prod
from pathlib import Path
import hashlib
import json
from xdsl.ir import Operation
from .frontend.parse import parse_module
from .direct_conv_binding import serialize


def census(module):
    """Count return-reachable copies; abandoned source gathers are not runtime work.

Bytes are tensor output bytes, not measured DRAM traffic. The model assumes
pure tensor SSA, as required by this pre-bufferization callback. Calls are
included through their result and operands; no call is erased or modified.
"""
    from merlin.llvmlower.layout_propagation import _permutation
    seen=set();pending=[op for op in module.walk() if op.name=='func.return']
    while pending:
        op=pending.pop()
        if op in seen:continue
        seen.add(op)
        for value in op.operands:
            if isinstance(value.owner,Operation):pending.append(value.owner)
    barriers=Counter();copies=0;total=0
    for op in seen:
        if _permutation(op) is None:continue
        shape=op.results[0].type.get_shape()
        if any(x<=0 for x in shape):continue
        width=getattr(op.results[0].type.get_element_type(),'bitwidth',None)
        if width is None:continue
        copies+=1;total+=(prod(shape)*int(width)+7)//8
        owner=op.operands[0].owner
        barriers[owner.name if isinstance(owner,Operation) else 'argument']+=1
    return {'copies':copies,'output_bytes':total,'producer_boundaries':dict(sorted(barriers.items()))}


def rewrite(module, *, reduction_channel_block=0):
    from merlin.llvmlower.layout_propagation import rewrite_module
    before=census(module);report=rewrite_module(module,reduction_channel_block=reduction_channel_block)
    result={'schema':'exact_physical_layout_v1','before':before,'after':census(module),
            'rewrite':report.to_dict(),'call_abi_changed':False,'scalar_arithmetic_changed':False}
    if reduction_channel_block:result['reduction_channel_block']=reduction_channel_block
    return result


def rewrite_file(source,work, *, reduction_channel_block=0):
    source,work=Path(source),Path(work);work.mkdir(parents=True,exist_ok=True)
    data=source.read_bytes();module=parse_module(data.decode());report=rewrite(module,reduction_channel_block=reduction_channel_block)
    result=work/'layout.mlir';result.write_text(serialize(module,[]))
    reparsed=parse_module(result.read_text());reparsed.verify()
    if census(reparsed)!=report['after']:raise ValueError('layout census changed during serialization')
    report['source_sha256']=hashlib.sha256(data).hexdigest()
    report['rewritten_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
    (work/'layout_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return result,report
