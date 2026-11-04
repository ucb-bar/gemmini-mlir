"""Original-window interior guards consumed by the common convolution feed.

Window2D owns both endpoint predicates and the physical first-tap address.
Positive source stride/dilation make every intervening tap lie between those
endpoints. A failed guard retains the existing padded-coordinate gather.
This specialization changes no storage ownership or completion permission.
"""
from dataclasses import dataclass
from math import gcd

from . import isa


@dataclass(frozen=True)
class InteriorRuns:
    source_identity: str
    window: object
    interior_y: tuple
    interior_x: tuple
    full_interior_tiles: int
    total_spatial_tiles: int

    def guard(self, prepared, builder, first, rows):
        if (prepared.source.identity != self.source_identity
                or prepared.source.plan.window != self.window):
            raise ValueError('interior run source changed')
        w=self.window;ci=w.input_shape[-1];kh,kw=w.kernel
        # Both artifacts describe the complete original output-row span, not
        # a padded or clipped view. First/last tap bounds imply all tap bounds.
        head=w.emit_row_run(builder,first,rows,0,0,0,ci)
        tail=w.emit_row_run(builder,first,rows,kh-1,kw-1,0,ci)
        predicate=builder.op('and',head['predicate'],tail['predicate'],'i1')
        return predicate,head['first_element']

    def transfer(self, prepared, builder, base, rows, ky, kx, channel, depth):
        if prepared.source.identity != self.source_identity:
            raise ValueError('interior transfer source changed')
        w=self.window;_,_,width,ci=w.input_shape;dh,dw=w.dilation
        offset=builder.op('add',builder.op('mul',ky,dh*width*ci),
            builder.op('add',builder.op('mul',kx,dw*ci),channel))
        return dict(zero=False,first_element=builder.op('add',base,offset),
            local_row=prepared.group.lhs_row(0),rows=rows,columns=depth,
            row_stride_elements=w.stride[1]*ci)

    def receipt(self):
        return dict(source_sha256=self.source_identity,
            guard_api='merlin.compile.scheduling.window_transfer.Window2D.emit_row_run',
            endpoints=['original first kernel tap','original last kernel tap'],
            full_interior_tiles=self.full_interior_tiles,total_spatial_tiles=self.total_spatial_tiles,
            geometry_admission='at least one aligned full M tile inside original-row interior',
            interior_output_y=list(self.interior_y),interior_output_x=list(self.interior_x),
            transformation='one source endpoint guard outside K; direct affine run per tap/channel tile',
            fallback='unchanged shared padded-coordinate gather',
            ordering='original M/N/K order, storage, compute, terminal readout and fences retained')


def aligned_full_tiles(window, interior_y, interior_x):
    """Count original-row-safe aligned M tiles in at most DIM squared terms.

    Flattened tile starts are multiples of DIM. The original-row residue of a
    start repeats after DIM/gcd(OW,DIM) rows. Counting batch and Y residues
    avoids visiting the (possibly large) spatial domain during compilation.
    This is an applicability proof, not a profitability estimate.
    """
    batch,_,_,_=window.input_shape;oh,ow=window.output_spatial
    y0,y1=interior_y;x0,x1=interior_x;tile=isa.DIM
    period=tile//gcd(ow,tile);count=0
    for n in range(min(batch,period)):
        images=(batch-1-n)//period+1
        for residue in range(period):
            first_y=y0+(residue-y0)%period
            if first_y>=y1:continue
            rows=(y1-1-first_y)//period+1
            first=(n*oh+first_y)*ow+x0
            last=(n*oh+first_y)*ow+x1-tile
            tiles=max(0,last//tile-(first+tile-1)//tile+1)
            count+=images*rows*tiles
    return count


def plan(prepared):
    w=prepared.source.plan.window
    _,h,width,_=w.input_shape;kh,kw=w.kernel;sh,sw=w.stride;dh,dw=w.dilation
    pt,pl,_,_=w.padding;oh,ow=w.output_spatial
    y=(max(0,(pt+sh-1)//sh),min(oh,(h+pt-1-(kh-1)*dh)//sh+1))
    x=(max(0,(pl+sw-1)//sw),min(ow,(width+pl-1-(kw-1)*dw)//sw+1))
    if y[0]>=y[1] or x[1]-x[0]<isa.DIM:return None
    # These are exact source geometry bounds; legality still comes from the
    # actual shared endpoint predicates consumed by emitted control flow.
    count=aligned_full_tiles(w,y,x)
    if not count:return None
    total=(w.input_shape[0]*oh*ow+isa.DIM-1)//isa.DIM
    return InteriorRuns(prepared.source.identity,w,y,x,count,total)


def select_auto(prepared, candidate, *, complete_k):
    """Optional regional proposal through the existing typed pass selector.

    These source/resource guards bound the proposed performance region. They
    grant no extra legality: Window2D, admitted panels, the original epoch law,
    and the common numeric readout remain independently authoritative. No
    observations or case identifiers are consulted during compilation.
    """
    from dataclasses import asdict
    from merlin.transforms.pipeline_policy import (
        PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
        ResourcePredicate, ShapePredicate, execute_policy,
    )
    from merlin.transforms.pipeline_state import (
        AnalysisStamp, PassSpec, PipelineContext, PipelineSession,
    )
    p=prepared.source.plan;w=p.window
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources=prepared.epoch.hardware_identity)
    facts=PolicyFacts(AnalysisStamp(prepared.source.identity,context,'counted_convolution'),
        shapes=((*w.kernel,w.input_shape[-1],p.output_channels),w.stride,w.dilation),
        resources=(('complete_k_admitted_panel',int(complete_k)),
            ('aligned_full_interior_tiles',candidate.full_interior_tiles if candidate else 0),
            ('isotropic_stride',int(w.stride[0]==w.stride[1])),
            ('shared_numeric_readout',1),('primitive_rows',isa.DIM),
            ('rhs_rows',prepared.group.panel.rhs_rows),('spad_rows',isa.SPAD_ROWS),
            ('accumulator_rows',prepared.group.panel.accumulator_rows),('acc_rows',isa.ACC_ROWS)))
    predicates=(ShapePredicate(0,0,3,3),ShapePredicate(0,1,3,3),
        ShapePredicate(0,2,multiple_of=isa.DIM),ShapePredicate(0,3,multiple_of=isa.DIM),
        ShapePredicate(1,0,1,2),ShapePredicate(1,1,1,2),
        ShapePredicate(2,0,1,1),ShapePredicate(2,1,1,1),
        ResourcePredicate('isotropic_stride',1),
        ResourcePredicate('complete_k_admitted_panel',1),
        ResourcePredicate('aligned_full_interior_tiles',1),
        ResourcePredicate('shared_numeric_readout',1))
    output='conv_interior_feed';level=facts.stamp.ir_level
    def verify(source):
        source.__post_init__()
        if source.identity!=prepared.source.identity:raise ValueError('changed interior policy source')
    session=PipelineSession(prepared.source,context=context,ir_level=level,
        fingerprint=lambda source:source.identity,verify_ir=verify)
    def entry(name,artifact):
        return RegisteredPass(PassSpec(name,0,produces=(output,),mutates_ir=False,ir_level=level),
            lambda session:{output:artifact})
    policy=PipelinePolicy((PipelineVariant('interior',('interior',),predicates),
        PipelineVariant('incumbent',('incumbent',))),'incumbent')
    result=execute_policy(session,policy,
        {'interior':entry('interior',candidate),'incumbent':entry('incumbent',None)},facts,output=output)
    return result.artifact,dict(mode='optional source/resource regional proposal; default unchanged',
        rule='3x3; unit dilation; aligned CI/CO; isotropic stride 1 or 2; positive aligned interior; admitted complete-K panels',
        guards=[dict(predicate=asdict(predicate),matched=predicate.match(facts))for predicate in predicates],
        selection=result.receipt,
        legality='shared original endpoint guards, admitted panel ownership and epoch law remain authoritative',
        numerical_boundary='common original terminal readout and configuration callbacks retained',
        evidence='three paired public complete Spike gains motivate region; fresh geometry timing remains unknown')
