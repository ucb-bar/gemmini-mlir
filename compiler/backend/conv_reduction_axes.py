"""Bounded ordered channel-block unroll of the original convolution reduction.

The incumbent already exposes ky/kx/CI axes and LLVM hoists band arithmetic.
This separate alternative consumes an actual library unroll product to expose
consecutive complete input-gather/compute steps without reordering any effects.
"""
from dataclasses import dataclass
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, check_structure
from merlin.sched.ir.expr import Scaled, Sum, evaluate
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from . import isa


def _trace(body,env=None):
    env={} if env is None else env
    for node in body:
        if type(node)is Loop:
            for index in range(node.extent):
                yield from _trace(node.body,{**env,node.var:index})
        elif type(node)is Call and node.instr=='original_convolution_product':
            yield tuple(evaluate(node.arg(name),env) for name in ('kernel_y','kernel_x','channel_block','depth'))
        else:raise ValueError('unroll product must contain only original ordered convolution steps')


@dataclass(frozen=True)
class ConvChannelUnroll:
    source_identity: str
    window: object
    channel_blocks: int
    factor: int
    source: Kernel
    transformed: Kernel

    def validate(self):
        self.window.__post_init__()
        kh,kw=self.window.kernel;ci=self.window.input_shape[-1]
        if (self.factor not in (2,4) or ci%isa.DIM or self.channel_blocks!=ci//isa.DIM
                or self.channel_blocks<self.factor or check_structure(self.transformed)):
            raise ValueError('channel unroll requires exact original full channel blocks and bounded expansion')
        expected=tuple((y,x,c,isa.DIM)for y in range(kh)for x in range(kw)for c in range(self.channel_blocks))
        if tuple(_trace(self.source.body))!=expected or tuple(_trace(self.transformed.body))!=expected:
            raise ValueError('channel unroll changed original source reduction/effect order')

    def emit(self,builder,callback):
        """Interpret the transformed schedule, invoking the common effect emitter."""
        self.validate()
        def expression(value,env):
            if type(value)is Const:return value.value
            if type(value)is Var:return env[value.name]
            if type(value)is Scaled:return builder.op('mul',expression(value.expr,env),value.factor)
            if type(value)is Sum:
                out=0
                for term in value.terms:out=builder.op('add',out,expression(term,env))
                return out
            raise ValueError('unsupported transformed channel expression')
        def body(nodes,env):
            for node in nodes:
                if type(node)is Loop:
                    builder.loop(0,node.extent,1,lambda index,node=node:body(node.body,{**env,node.var:index}))
                else:
                    callback(*(expression(node.arg(name),env) for name in ('kernel_y','kernel_x','channel_block')),isa.DIM)
        body(self.transformed.body,{})

    def receipt(self):
        self.validate()
        return dict(mechanism='bounded_ordered_convolution_channel_unroll',factor=self.factor,
            original_source_sha256=self.source_identity,source_schedule=self.source.text(),
            transformed_schedule=self.transformed.text(),source_schedule_sha256=self.source.digest(),
            transformed_schedule_sha256=self.transformed.digest(),
            passes=['original.ConvolutionExplorationSource','loop_schedule.unroll','backend.isa'],
            effects='same ordered original input gather and all N compute consumers per channel step',
            completion='unchanged preparation, initializer, complete-K/readout and last-use boundaries',
            checked_original_steps=len(tuple(_trace(self.source.body))))


def plan_conv_channel_unroll(prepared,factor):
    w=prepared.source.plan.window;ci=w.input_shape[-1]
    if ci%isa.DIM or ci//isa.DIM<factor:return None
    # Factor2 is an explicitly selected regional alternative. Padded training
    # geometries regressed; original zero padding defines its supported scope.
    # Factor4 remains an unpromoted exploratory schedule.
    if factor==2 and any(w.padding):return None
    kh,kw=w.kernel;blocks=ci//isa.DIM
    source=Kernel('original_convolution_reduction',(),(Loop('kernel_y',kh,(
        Loop('kernel_x',kw,(Loop('channel_block',blocks,(Call('original_convolution_product',(
            ('kernel_y',Var('kernel_y')),('kernel_x',Var('kernel_x')),
            ('channel_block',Var('channel_block')),('depth',Const(isa.DIM)))),)),)),)),))
    transformed=propose_loop_transforms(source,(LoopTransform('unroll',
        ('kernel_y','kernel_x','channel_block'),factor=factor),),index_bits=64,max_static_operations=32)
    plan=ConvChannelUnroll(prepared.source.identity,w,blocks,factor,source,transformed)
    plan.validate()
    return plan
