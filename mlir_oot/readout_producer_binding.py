"""Bind an exact readout interval to an immutable typed producer and kernel.

This adapter belongs to the provider: its signed-i8 convolution implementation
uses zero-initialized i32 dot products. Shared interval arithmetic stays Merlin.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np

from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange
from merlin.runtime.captured_constants import verify_capture_constant
from merlin.llvmlower.requantization import f32
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .captured_requant import inspect_chain
from .direct_conv_binding import match as match_conv


class ProducerBindingIntegrityError(ValueError):
    """Immutable source, numeric or executable closure no longer matches."""


def bind_readout_producer(capture, bundle, route):
    """Require source, constant and compiled-object closure before domain elision."""
    capture=Path(capture)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=capture/'model.mlir'
    if sha(source)!=bundle['source_sha256'] or route not in bundle['routes']:
        raise ProducerBindingIntegrityError('producer source/route binding changed')
    module=parse_module(source.read_text())
    ops=[op for op in module.walk() if getattr(op.attributes.get('prov.region_id'),'data',None)==route['region'] and match_integer_gemm(op) is not None]
    if len(ops)!=1:raise ValueError('unique typed integer producer required')
    op=ops[0];dims=match_integer_gemm(op);chain=inspect_chain(op);direct=match_conv(op)
    if not 0<dims.m*dims.n<=((1<<64)-1)//4:raise ValueError('scratch span exceeds descriptor address domain')
    schedule=route['schedule']
    for field in ('h','w','cin','cout','stride'):
        if schedule[field]!=getattr(direct.shape,field):raise ValueError('kernel/source convolution shape differs')
    if dims.m!=direct.shape.oh*direct.shape.ow or dims.n!=schedule['cout'] or dims.k!=9*schedule['cin'] or schedule['output_dtype']!='i32' or schedule['scale']!=1.0 or schedule['relu']:
        raise ValueError('unscaled exact i32 convolution producer required')
    constant=verify_capture_constant(manifest_path=capture/'weights.safetensors.manifest.json',manifest_sha256=bundle['manifest_sha256'],safetensors_path=capture/'weights.safetensors',safetensors_sha256=bundle['weights_sha256'],entry_argument_index=chain['bias'].index,source_shape=[dims.n],source_dtype='f32',max_payload_bytes=dims.n*4)
    if constant.payload_sha256!=route['bias_payload_sha256'] or np.any(np.frombuffer(constant.logical_payload,dtype='<f4')!=0):
        raise ProducerBindingIntegrityError('immutable zero-bias source proof required')
    proof=route['integer_readout']
    if proof['output_min']!=(0 if chain['relu'] else -128):raise ProducerBindingIntegrityError('source readout ReLU differs')
    if list(proof['source_scales'])!=[f32(x) for x in [*chain['scales'],chain['reciprocal']]]:raise ProducerBindingIntegrityError('source readout scale order differs')
    domain=IntegerSumProductsRange(dims.k,-128,127,-128,127)
    bounds=domain.require_contained(proof['accumulator_min'],proof['accumulator_max'])
    compilation=route['compilation'];argv=compilation['compiler_argv'][-1];obj=Path(argv[argv.index('-o')+1])
    if sha(obj)!=compilation['object_sha256'] or sha(obj.parent/'kernel.gemmini.mlir')!=compilation['target_ir_sha256']:
        raise ProducerBindingIntegrityError('compiled producer bytes changed')
    receipt=dict(schema='bound_integer_readout_producer_v1',source_sha256=sha(source),region_trace_only=route['region'],kernel_symbol=route['kernel'],kernel_object=str(obj),kernel_object_sha256=sha(obj),kernel_ir_sha256=compilation['target_ir_sha256'],bias_payload_sha256=constant.payload_sha256,integer_sum_products=asdict(domain),proven_interval=list(bounds),readout_domain=[proof['accumulator_min'],proof['accumulator_max']],storage_contract='Caller-owned exclusive i32 scratch; owner must prove/check disjoint readout output and stable scratch until last read; producer writes and fences before readout',numeric_contract='Exact signed-i8 operands including -128; zero accumulator seed; source-derived K; every product/prefix fits signed-i32; no scale, bias or ReLU before readout')
    return domain,receipt
