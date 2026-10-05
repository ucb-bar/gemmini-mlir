"""Exact scalar requantization proof and primitive GEMM/store fusion.

Positive f32 scaling, nearest-even rounding and saturation are monotone. Comparing
all 255 integer output transition points proves equivalence for the entire i32
accumulator range, without sampling or reassociating unproved float arithmetic.
"""
from dataclasses import dataclass, replace
import math
import struct
from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap
from .contraction_patterns import match_integer_gemm
from .golden_contraction_upstream import choose_shape


def f32(value):
    try:return struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError:return math.copysign(math.inf,value)


def quantized(acc,scales,relu=False):
    value=f32(acc)
    for scale in scales:value=f32(value*scale)
    low=0 if relu else -128
    if value>=127:return 127
    if value<=low:return low
    return max(low,min(127,round(value)))


def prove_scale(scales,lo=-(1<<31),hi=(1<<31)-1,relu=False):
    """Prove exact int8 equality at every representable accumulator in [lo,hi]."""
    scales=tuple(f32(x) for x in scales)
    if not scales or any(not math.isfinite(x) or x<=0 for x in scales):
        raise ValueError('all scales must be finite positive f32 constants')
    if not (-(1<<31)<=lo<=hi<(1<<31)):raise ValueError('invalid i32 proof domain')
    combined=1.0
    for scale in scales:combined=f32(combined*scale)
    if not math.isfinite(combined) or combined<=0:raise ValueError('combined scale is not finite positive f32')
    def first_at_least(sequence,q):
        left,right=lo,hi+1
        while left<right:
            mid=(left+right)//2
            if quantized(mid,sequence,relu)>=q:right=mid
            else:left=mid+1
        return left
    low=0 if relu else -128
    for q in range(low+1,128):
        source=first_at_least(scales,q);target=first_at_least((combined,),q)
        if source!=target:
            witness=min(source,target)
            raise ValueError(f'f32 reassociation changes output at accumulator {witness}: source={quantized(witness,scales,relu)}, store={quantized(witness,(combined,),relu)}')
    return dict(scale=combined,source_scales=list(scales),accumulator_min=lo,accumulator_max=hi,
                rounding='nearest_even',output_min=low,output_max=127,
                proof='exhaustive monotone integer-output transition comparison',transitions=127-low)


def prove_scale_bound(scales, lo=-(1 << 31), hi=(1 << 31)-1, relu=False):
    """Compute the exact worst output error over the complete accumulator domain.

    Both saturated int8 outputs are monotone step functions. Their values are
    constant between the union of their transition points, so comparing each
    transition and the domain endpoints covers every possible accumulator.
    This reports a bound; callers must explicitly select an error policy before
    replacing source arithmetic. It does not establish model-level quality.
    """
    scales = tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('all scales must be finite positive f32 constants')
    if not (-(1 << 31) <= lo <= hi < (1 << 31)):
        raise ValueError('invalid i32 proof domain')
    scale = 1.0
    for s in scales:
        scale = f32(scale*s)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('combined scale is not finite positive f32')
    low = 0 if relu else -128
    boundaries = {lo, hi}
    transitions = []
    for q in range(low+1, 128):
        thresholds = []
        for sequence in (scales, (scale,)):
            left, right = lo, hi+1
            while left < right:
                mid = (left+right)//2
                if quantized(mid, sequence, relu) >= q:
                    right = mid
                else:
                    left = mid+1
            thresholds.append(left)
            if left <= hi:
                boundaries.add(left)
        if thresholds[0] != thresholds[1]:
            transitions.append(dict(output=q, source=thresholds[0], target=thresholds[1]))
    error = 0
    witness = None
    for acc in sorted(boundaries):
        source = quantized(acc, scales, relu)
        target = quantized(acc, (scale,), relu)
        delta = abs(source-target)
        if delta > error:
            error = delta
            witness = dict(accumulator=acc, source=source, target=target)
    return dict(
        scale=scale, source_scales=list(scales), accumulator_min=lo, accumulator_max=hi,
        rounding='nearest_even', output_min=low, output_max=127,
        proof='complete union of monotone integer-output transition points and endpoints',
        transitions=127-low, differing_transitions=transitions,
        max_output_lsb_error=error, exact=error == 0, witness=witness,
        scope='Local readout error only; requires an explicit numerical policy and full-model quality gate.',
    )


def synthesize_store_scale(scales, lo=-(1 << 31), hi=(1 << 31)-1, relu=False):
    """Solve exact readout constraints over every positive finite f32 scale.

    Source output transitions constrain target(T)>=q and target(T-1)<q.
    Each constraint is monotone in the ordered positive-float bit pattern
    (reversed for a negative accumulator). Their intersection is therefore one
    exact interval, obtained by integer binary searches without real rounding
    relaxations. A contradictory interval proves this zero-bias form infeasible.
    """
    scales=tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s<=0 for s in scales):
        raise ValueError('all scales must be finite positive f32 constants')
    if not (-(1 << 31)<=lo<=hi<(1 << 31)):
        raise ValueError('invalid i32 proof domain')
    def from_bits(bits):
        return struct.unpack('<f',struct.pack('<I',bits))[0]
    first, last=1,0x7f7fffff
    lower, upper=first,last
    low=0 if relu else -128
    witnesses={}
    transitions=[]
    def constrain(acc,q,reached):
        nonlocal lower,upper
        def valid(bits):
            return (quantized(acc,(from_bits(bits),),relu)>=q)==reached
        if acc==0:
            if not valid(first):
                lower=last+1
                witnesses['constant']=dict(accumulator=acc,output=q,reached=reached)
            return
        increasing=(acc>0)==reached
        left,right=first,last+1
        while left<right:
            mid=(left+right)//2
            if valid(mid)==increasing:
                right=mid
            else:
                left=mid+1
        if increasing:
            if left>lower:
                lower=left
                witnesses['lower']=dict(accumulator=acc,output=q,reached=reached,scale_bits=left)
        else:
            if left-1<upper:
                upper=left-1
                witnesses['upper']=dict(accumulator=acc,output=q,reached=reached,scale_bits=left-1)
    for q in range(low+1,128):
        left,right=lo,hi+1
        while left<right:
            mid=(left+right)//2
            if quantized(mid,scales,relu)>=q:right=mid
            else:left=mid+1
        transitions.append((q,left))
        if left<=hi:constrain(left,q,True)
        if left>lo:constrain(left-1,q,False)
        if lower>upper:
            return dict(exact=False,source_scales=list(scales),accumulator_min=lo,
                accumulator_max=hi,relu=relu,scale_bits_min=lower,scale_bits_max=upper,
                proof='contradictory exact constraints across every positive finite f32 store scale',
                conflicting_output=q,witnesses=witnesses)
    combined=1.0
    for s in scales:combined=f32(combined*s)
    bits=struct.unpack('<I',struct.pack('<f',combined))[0]
    bits=max(lower,min(upper,bits))
    scale=from_bits(bits)
    # Independently compare all target transitions before publishing a solution.
    for q,threshold in transitions:
        left,right=lo,hi+1
        while left<right:
            mid=(left+right)//2
            if quantized(mid,(scale,),relu)>=q:right=mid
            else:left=mid+1
        if left!=threshold:
            raise AssertionError('synthesized scale transition validation failed')
    return dict(exact=True,scale=scale,source_scales=list(scales),accumulator_min=lo,
        accumulator_max=hi,relu=relu,scale_bits_min=lower,scale_bits_max=upper,
        rounding='nearest_even',output_min=low,output_max=127,transitions=127-low,
        proof='exact intersection of all source transition constraints over positive finite f32 scales; independently revalidated',
        witnesses=witnesses)


def _constant(value):
    op=value.owner
    if not isinstance(op,Operation) or op.name!='arith.constant' or str(value.type)!='f32':
        raise ValueError('epilogue needs uniform f32 constants')
    attr=op.properties.get('value',op.attributes.get('value'))
    data=getattr(getattr(attr,'value',None),'data',None)
    if data is None:raise ValueError('not a scalar float constant')
    return f32(data)


@dataclass
class FusedRequant:
    contraction: Operation
    epilogue: Operation
    shape: object
    proof: dict
    integer_bias: list[int] | None = None


def match(epilogue):
    """Match one canonical pointwise i32→i8 scale/round/saturate region.

    Bias additions, residuals, nonuniform scales, zero-point shifts, alternate
    rounding, fastmath and shared accumulator outputs are deliberately refused.
    """
    op=epilogue
    if op.name!='linalg.generic' or len(op.operands) not in (2,3) or len(op.results)!=1:
        raise ValueError('expected one-input pointwise requantization')
    has_bias=len(op.operands)==3
    contraction=op.operands[0].owner
    if not isinstance(contraction,Operation):raise ValueError('requantization must consume a contraction')
    dims=match_integer_gemm(contraction)
    if dims is None or dims.batch!=1:raise ValueError('expected exact signed integer GEMM')
    if len(list(contraction.results[0].uses))!=1:raise ValueError('accumulator has another consumer')
    ty=op.results[0].type
    if not isinstance(ty,TensorType) or str(ty.get_element_type())!='i8' or ty.get_shape()!=(dims.m,dims.n) or op.operands[-1].type!=ty:
        raise ValueError('requantization output geometry or dtype differs')
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    expected_maps=[AffineMap.identity(2)]
    if has_bias:
        from xdsl.ir.affine import AffineExpr
        expected_maps.append(AffineMap(2,0,(AffineExpr.dimension(1),)))
    expected_maps.append(AffineMap.identity(2))
    if maps is None or [x.data for x in maps.data]!=expected_maps:
        raise ValueError('requantization needs identity tensor indexing')
    if iters is None or [x.data.value for x in iters.data]!=['parallel','parallel']:
        raise ValueError('requantization is not pointwise')
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:raise ValueError('noncanonical epilogue region')
    block=op.regions[0].block
    for scalar_op in block.ops:
        flags=scalar_op.properties.get('fastmath')
        if flags is not None and str(flags)!='#arith.fastmath<none>':
            raise ValueError('fastmath epilogue contract unsupported')
    ops=[x for x in block.ops if x.name!='arith.constant']
    if len(block.args)!=(3 if has_bias else 2) or not ops or ops[0].name!='arith.sitofp' or list(ops[0].operands)!=[block.args[0]] or str(ops[0].results[0].type)!='f32':
        raise ValueError('expected signed accumulator to f32 conversion')
    value=ops.pop(0).results[0];scales=[]
    while ops and ops[0].name=='arith.mulf':
        mul=ops.pop(0)
        if value not in mul.operands:raise ValueError('scale chain disconnected')
        if str(mul.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>':raise ValueError('fastmath scale contract unsupported')
        scales.append(_constant(mul.operands[1] if mul.operands[0] is value else mul.operands[0]));value=mul.results[0]
    if not scales:raise ValueError('missing constant scale')
    biases=None;reciprocal=1.0
    if has_bias:
        biasop=op.operands[1].owner
        if not isinstance(biasop,Operation) or biasop.name!='arith.constant' or tuple(op.operands[1].type.get_shape())!=(dims.n,) or str(op.operands[1].type.get_element_type())!='f32':
            raise ValueError('bias must be an immutable dense f32 channel constant')
        attr=biasop.properties.get('value',biasop.attributes.get('value'))
        if not hasattr(attr,'get_values'):raise ValueError('dense bias literal required')
        biases=list(attr.get_values())
        if not ops or ops[0].name!='arith.addf' or set(ops[0].operands)!={value,block.args[1]}:
            raise ValueError('expected bias after source scale chain')
        value=ops.pop(0).results[0]
        if ops and ops[0].name=='arith.mulf':
            mul=ops.pop(0)
            if value not in mul.operands:raise ValueError('output scale disconnected')
            if str(mul.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>':raise ValueError('fastmath output scale unsupported')
            reciprocal=_constant(mul.operands[1] if mul.operands[0] is value else mul.operands[0]);value=mul.results[0]
    if not ops or ops[0].name!='math.roundeven' or list(ops[0].operands)!=[value]:raise ValueError('nearest-even rounding required')
    value=ops.pop(0).results[0]
    if len(ops)!=4 or [x.name for x in ops]!=['arith.maximumf','arith.minimumf','arith.fptosi','linalg.yield']:
        raise ValueError('expected exact signed i8 clamp; bias/residual/zero-point not supported')
    lower,upper,cast,yieldop=ops
    if value not in lower.operands:raise ValueError('clamp disconnected')
    lo=_constant(lower.operands[1] if lower.operands[0] is value else lower.operands[0]);value=lower.results[0]
    if lo not in (-128.0,0.0) or value not in upper.operands:raise ValueError('unsupported lower clamp')
    hi=_constant(upper.operands[1] if upper.operands[0] is value else upper.operands[0])
    if hi!=127.0 or list(cast.operands)!=[upper.results[0]] or str(cast.results[0].type)!='i8' or list(yieldop.operands)!=[cast.results[0]]:
        raise ValueError('unsupported upper clamp or conversion')
    bound=dims.k*128*128
    low,high=max(-(1<<31),-bound),min((1<<31)-1,bound)
    proof=prove_scale(scales,low,high,lo==0) if biases is None else synthesize_bias(scales,biases,reciprocal,low,high,lo==0)
    if biases is not None and proof['accepted_channels']!=dims.n:
        raise ValueError(f"integer bias cannot preserve all channels: {proof['accepted_channels']}/{dims.n}")
    shape=replace(choose_shape(dims),output_dtype='i8',scale=proof['scale'],relu=lo==0,wide_store=True,bias=has_bias)
    shape.validate()
    return FusedRequant(contraction,op,shape,proof,proof.get("integer_bias"))


def synthesize_bias(scales,biases,output_reciprocal,lo,hi,relu=False):
    """Solve all integer-bias transition constraints for a constant channel table.

    Source order is f32(acc), sequential f32 multiplications, f32 bias add,
    optional ReLU, f32 reciprocal multiply, nearest-even and signed saturation.
    Each target channel uses i32(acc+bias) then one Gemmini f32 scale/store.
    The accepted bias range excludes accumulator overflow.
    """
    import numpy as np
    scales=tuple(f32(x) for x in scales);reciprocal=f32(output_reciprocal)
    if not scales or any(not math.isfinite(x) or x<=0 for x in (*scales,reciprocal)):
        raise ValueError('positive finite scalar scales required')
    biases=np.asarray(biases,dtype=np.float32)
    if biases.ndim!=1 or not np.isfinite(biases).all():raise ValueError('finite channel bias vector required')
    if not (-(1<<31)<=lo<=hi<(1<<31)):raise ValueError('invalid accumulator range')
    scale=1.0
    for s in (*scales,reciprocal):scale=f32(scale*s)
    if not math.isfinite(scale) or scale<=0:raise ValueError('unrepresentable store scale')
    qlo=0 if relu else -128
    def source(xs):
        values=xs.astype(np.float32)
        for s in scales:values=np.multiply(values,np.float32(s),dtype=np.float32)
        values=np.add(values,biases,dtype=np.float32)
        if relu:values=np.maximum(values,np.float32(0))
        values=np.multiply(values,np.float32(reciprocal),dtype=np.float32)
        return np.clip(np.rint(values),qlo,127)
    # Every transition gives an equality or bound on the integer bias.
    lower=np.full(len(biases),max(-(1<<31),-(1<<31)-lo),dtype=np.int64)
    upper=np.full(len(biases),min((1<<31)-1,(1<<31)-1-hi),dtype=np.int64)
    first_failure=np.full(len(biases),-999,dtype=np.int32)
    for q in range(qlo+1,128):
        left=np.full(len(biases),lo,dtype=np.int64);right=np.full(len(biases),hi+1,dtype=np.int64)
        for _ in range((hi-lo+1).bit_length()):
            active=left<right;mid=(left+right)//2
            reached=source(mid)>=q
            right=np.where(active & reached,mid,right)
            left=np.where(active & ~reached,mid+1,left)
        source_threshold=left
        tl,tr=-(1<<31),(1<<31)
        while tl<tr:
            mid=(tl+tr)//2
            if quantized(mid,(scale,),relu)>=q:tr=mid
            else:tl=mid+1
        candidate=tl-source_threshold
        interior=(source_threshold>lo)&(source_threshold<=hi)
        lower=np.maximum(lower,np.where(source_threshold==lo,tl-lo,np.where(interior,candidate,lower)))
        upper=np.minimum(upper,np.where(source_threshold==hi+1,tl-hi-1,np.where(interior,candidate,upper)))
        first_failure=np.where((first_failure==-999)&(lower>upper),q,first_failure)
    accepted=lower<=upper
    return dict(scale=scale,source_scales=list(scales),output_reciprocal=reciprocal,
        accumulator_min=lo,accumulator_max=hi,relu=relu,
        proof='all monotone output transition constraints, including original f32 bias ordering',
        channels=len(biases),accepted_channels=int(accepted.sum()),
        integer_bias=[int(max(low,min(high,0))) if ok else None for low,high,ok in zip(lower,upper,accepted)],
        refused_channels=[{'channel':i,'conflicting_transition':int(first_failure[i]),'bias_lower':int(lower[i]),'bias_upper':int(upper[i])} for i in range(len(biases)) if not accepted[i]])


def compile_selected(source_path,llvm_bin,workdir):
    """Compile a proven pointwise epilogue; publish the required integer-bias table."""
    from dataclasses import asdict
    import hashlib,json
    from pathlib import Path
    from .frontend.parse import parse_module
    from .golden_device_compile import compile_module
    from .golden_gemm import GoldenGemm
    source=Path(source_path).read_text();module=parse_module(source);selected=[]
    for op in module.walk():
        if op.name!='linalg.generic':continue
        try:selected.append(match(op))
        except ValueError:pass
    if len(selected)!=1:raise ValueError(f'expected one provable requantized GEMM, got {len(selected)}')
    fused=selected[0]
    receipt=compile_module(GoldenGemm(fused.shape).build(),Path(llvm_bin),Path(workdir))
    result=dict(schema='gemmini_proven_requant_v1',source_sha256=hashlib.sha256(source.encode()).hexdigest(),shape=asdict(fused.shape),proof=fused.proof,integer_bias=fused.integer_bias,compilation=receipt,
                abi='A,B,C,i32_bias' if fused.integer_bias is not None else 'A,B,C',
                scope='Selected scalar epilogue proof and primitive kernel. Caller must bind published integer bias table, not original float bias.')
    (Path(workdir)/'upstream_requant_binding.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
