"""Exact private integer source chains bound to the shared dense epilogue."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

from xdsl.dialects.builtin import TensorType
from xdsl.ir import BlockArgument
from xdsl.ir.affine import AffineDimExpr
from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.llvmlower.integer_epilogue import match_integer_epilogue, _setup, _constant, _plain
from merlin.verify.epilogue_applicability import ReadoutCapability, selectors_applying
from linalg_device import _identity


def enabled():
    path=Path(__file__).with_name('DENSE_SCHEDULE.json')
    selected=json.loads(path.read_text()).get('source_integer_epilogue_fusion',False) if path.exists() else False
    if type(selected) is not bool:
        raise ValueError('source_integer_epilogue_fusion must be Boolean')
    return selected


def _readout(relu):
    low=0 if relu else -128
    source=f'''module {{ func.func @readout(%x: i32) -> i8 {{
      %low = arith.constant {low} : i32
      %high = arith.constant 127 : i32
      %a = arith.maxsi %x, %low : i32
      %b = arith.minsi %a, %high : i32
      %y = arith.trunci %b : i32 to i8
      func.return %y : i8
    }} }}'''
    return DenseIntegerReadout(source,'i8','store',())


def _bias(root,positions):
    uses=tuple(root.results[0].uses)
    if len(uses)!=1:return None
    op=uses[0].operation
    if (op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1
        or op.parent_block() is not root.parent_block()):return None
    shape=tuple(root.results[0].type.get_shape())
    if len(shape)!=2 or op.results[0].type!=root.results[0].type:return None
    body=tuple(op.regions[0].block.ops);args=tuple(op.regions[0].block.args)
    if (len(args)!=3 or args[-1].uses or [x.name for x in body]!=['arith.addi','linalg.yield']
        or set(body[0].operands)!={args[0],args[1]}
        or str(body[0].results[0].type)!='i32' or tuple(body[1].operands)!=(body[0].results[0],)):
        return None
    if (not _plain(op) or any(not _plain(inner) or inner.regions or inner.successors for inner in body)
        or any(str(value)!='#arith.overflow<none>' for value in body[0].properties.values())):return None
    maps=[a.data for a in op.properties['indexing_maps']]
    if len(maps)!=3 or not _identity(maps[-1],2) or any('parallel' not in str(i) for i in op.properties['iterator_types']):return None
    for index in (0,1):
        other=op.operands[1-index];mapping=maps[1-index]
        if (op.operands[index] is root.results[0] and _identity(maps[index],2)
            and isinstance(other.type,TensorType) and str(other.type.element_type)=='i32'
            and tuple(other.type.get_shape())==(shape[1],)
            and mapping.num_dims==2 and mapping.num_symbols==0 and len(mapping.results)==1
            and isinstance(mapping.results[0],AffineDimExpr) and mapping.results[0].position==1
            and (isinstance(other,BlockArgument) or positions.get(other.owner,positions[root]+1)<positions[root])):
            cursor=root.next_op
            while cursor is not op:
                if cursor is None or not _setup(cursor):return None
                cursor=cursor.next_op
            return op,other
    return None


@dataclass(frozen=True)
class Epilogue:
    root: object
    operations: tuple
    output: object
    bias: object
    relu: bool
    narrowed: bool
    proof: object
    capability_sha256: str

    def receipt(self,positions):
        return dict(schema='authored.source_integer_epilogue.v1',
            source_op_indices=[positions[self.root],*[positions[op] for op in self.operations]],
            output_dtype='i8' if self.narrowed else 'i32',bias=self.bias is not None,
            scale=1.0,relu=self.relu,capability_sha256=self.capability_sha256,
            scalar_proof=None if self.proof is None else self.proof.correspondence.to_dict(),
            accumulator_law='signed i32 additions are equal modulo 2**32; bias before product preserves source addition',
            numerical_execution='not_established')


def plan(block,is_device):
    """No source mutation; only sole-use, zero-seed rank-two products qualify."""
    if not enabled():return {}
    declaration=Path(__file__).with_name('linalg_epilogue_contract.json').read_bytes()
    contract=json.loads(declaration)
    fields={key:contract[key] for key in ('output_path','readouts')}
    if sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()!=contract['fields_sha256']:
        raise ValueError('packaged epilogue declaration content hash mismatch')
    from merlin.targetgen.output_path import validate
    validate(contract['output_path'])
    readouts=tuple(ReadoutCapability(row['selector'],frozenset(row['applies']),row.get('evidence',''))
        for row in contract['readouts'])
    path=contract['output_path']
    raw=(path.get('accumulator',{}).get('dtype')=='i32'
        and path.get('exits',{}).get('accumulator') is True)
    preload=raw and any(s.get('kind')=='preload' and s.get('dtype')=='i32'
        and s.get('rounding')=='none' and s.get('range')==[-2147483648,2147483647]
        for s in path.get('accumulator_stages',()))
    # Unit-scale readout equals integer saturation over the complete i32 domain:
    # f32 conversion is exact around both clamp boundaries; larger magnitudes
    # saturate regardless of conversion rounding. Every required target stage
    # must be explicitly declared before admitting this lowering.
    readout=path.get('readout',())
    stages=[s for s in readout if s.get('kind')!='pool' and s.get('kind')!='activation']
    narrow=(raw and path.get('exits',{}).get('requant') is True and len(stages)==3
        and stages[0]==dict(kind='convert',to='f32',rounding='half_even')
        and stages[1].get('kind')=='scale' and stages[1].get('dtype')=='f32'
        and stages[1].get('rounding')=='half_even'
        and stages[1].get('granularity')=='per_command'
        and stages[2]==dict(kind='requant',to='i8',rounding='half_even',clamp=[-128,127]))
    rectifier=any(s==dict(kind='activation',function='relu',domain='accumulator')
        for s in readout)
    positions={op:index for index,op in enumerate(block.ops)};result={}
    for root in block.ops:
        if not is_device(root) or len(root.results[0].type.get_shape())!=2:continue
        if _constant(root.operands[2]) is None or _constant(root.operands[2]).value.data!=0:continue
        # The general float-chain fusion API intentionally refuses these integer
        # chains; its shared integer adapter proves actual scalar IR instead.
        from merlin.xdsl_dialects.lowering import output_fusion
        output_fusion.source_chain(root,output_fusion.Values())
        bias_match=_bias(root,positions) if preload else None
        producer=bias_match[0] if bias_match else root
        matched=None;relu=False
        if narrow and 'i8' in selectors_applying(readouts,['acc_scale']):
            for relu_choice in (False,True):
                if relu_choice and (not rectifier or 'i8' not in selectors_applying(readouts,['acc_scale','relu'])):continue
                matched=match_integer_epilogue(producer,_readout(relu_choice))
                if matched is not None:relu=relu_choice;break
        operations=(() if bias_match is None else (bias_match[0],))+(() if matched is None else matched.operations)
        if not operations:continue
        result[root]=Epilogue(root,operations,operations[-1].results[0],
            None if bias_match is None else bias_match[1],relu,matched is not None,matched,
            sha256(declaration).hexdigest())
    return result
