"""Sequential original-column views of capacity-bounded horizontal panels.

The shared phase planner retains the full original source and external pitches.
One Merlin input preparation serves every phase. Each phase consumes the same
horizontal effect encoder, with a checked projection back to original columns.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import hashlib
import json

from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
from merlin.compile.scheduling.horizontal_phases import plan_horizontal_convolution_phases
from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel


@dataclass(frozen=True)
class HorizontalPreparedInput:
    owner: object
    source: object
    pointer: object
    window: Window2D
    size: int
    transport: dict

    def revalidate(self, fn, source, plan):
        if (fn is not self.owner or source is not self.source
                or plan.source.window != self.window
                or self.size != prod(plan.padded_window.input_shape)
                or self.pointer is source):
            raise ValueError('horizontal shared input preparation changed original owner or coordinates')
        return self

    def receipt(self):
        return dict(kind='shared_original_padded_input', original_window=repr(self.window),
            padded_input_bytes=self.size, preparation=self.transport,
            completion='Merlin preparation before every phase; private immutable source until invocation exit')


def _intervals(attrs, co, edge, narrow, column_intervals=None):
    readout = None
    if narrow:
        scales = attrs.get('acc_scale_by_column', attrs.get('acc_scale',
            attrs.get('output_scale', attrs.get('scale', 1.0))))
        readout = DenseTiledIntegerReadout.from_source_scales(scales, columns=co,
            tile_columns=1, relu=bool(attrs.get('relu', False) or 'relu' in attrs.get('epilogue', ())))
    if column_intervals is not None:
        from gemmini_horizontal_conv import _validate_column_intervals
        return _validate_column_intervals(column_intervals, co, edge, readout)
    result=[]; column=0
    while column<co:
        end=min(column+edge,co)
        if readout is not None:
            end=column+1
            while end<min(column+edge,co) and readout.column_scale_bits[end]==readout.column_scale_bits[column]:
                end+=1
        result.append((column,end-column));column=end
    return tuple(result)


def _view_attrs(attrs, first, stop):
    result=dict(attrs)
    kernel=result.get('kernel')
    if kernel is not None and len(kernel)==4:
        result['kernel']=(*kernel[:3],stop-first)
    for key in ('acc_scale_by_column','acc_scale','output_scale','scale'):
        value=result.get(key)
        if isinstance(value,(list,tuple)):
            result[key]=tuple(value[first:stop])
    return result


def plan_horizontal_phases(*, input_shape, weight_shape, attrs=None, bias=False,
                           output_dtype='i32', max_row_blocks=8, compute_pairs=8,
                           column_intervals=None):
    """Return checked original-column views, or None without mutating code.

    Optional column_intervals preserves an exact source atomic partition. Every
    interval is independently checked against primitive width and source scales;
    phase-local planning consumes its projection without merging boundaries.
    """
    from gemmini_conv import _geometry, _resources
    from gemmini_horizontal_conv import plan_horizontal_conv, _selected_transfer_law
    attrs=dict(attrs or {})
    # Existing successful plans remain the incumbent, including their exact bytes.
    if plan_horizontal_conv(input_shape=input_shape,weight_shape=weight_shape,attrs=attrs,
            bias=bias,output_dtype=output_dtype,max_row_blocks=max_row_blocks,
            compute_pairs=compute_pairs,column_intervals=column_intervals) is not None:
        return None
    window,co=_geometry(input_shape,weight_shape,attrs)
    edge,operand_rows,accumulator_rows=_resources()
    kh,kw=window.kernel;ci=window.input_shape[-1]
    if (window.output_spatial[1]<edge or window.dilation[1]!=1 or kw<=1 or ci>=edge
            or (kw*ci+edge-1)//edge>=kw*((ci+edge-1)//edge)):
        return None
    resources,_=_selected_transfer_law(Path(__file__).resolve().parent,edge)
    source=CountedConvolutionPlan(window,co,edge,edge,edge,min(ci,edge),
        operand_rows,accumulator_rows,0,edge,0,bool(bias))
    intervals=_intervals(attrs,co,edge,str(output_dtype) in ('i8','int8'),column_intervals)
    phased=plan_horizontal_convolution_phases(source,intervals,max_row_blocks=max_row_blocks,
        maximum_transfer_columns=min(resources.max_columns,resources.max_request_bytes),
        private_rows_owned=True,row_block_choices=(1,2,4,8))
    if phased is None or len(phased.phases)<2:
        return None
    views=[]
    for phase in phased.phases:
        first,stop=phase.original_column_start,phase.original_column_end
        weights=(*weight_shape[:-1],stop-first)
        local_attrs=_view_attrs(attrs,first,stop)
        maximum=phase.inner.row_blocks
        if maximum not in (1,2,4,8):
            return None
        prepared=plan_horizontal_conv(input_shape=input_shape,weight_shape=weights,
            attrs=local_attrs,bias=bias,output_dtype=output_dtype,
            max_row_blocks=maximum,compute_pairs=compute_pairs,
            column_intervals=phase.inner.column_intervals)
        if prepared is None or prepared[0]!=phase.inner:
            return None
        views.append((phase,prepared,maximum))
    return phased,tuple(views)


def _outline_rhs_prefix(root, owner, block, boundary):
    """Extract the closed straight-line RHS epoch without changing its effects."""
    from merlin.llvmlower.host_builder import FnBuilder
    from xdsl.dialects import llvm
    from gemmini_support import register_internal_helper
    if block is not owner.entry:
        raise ValueError('horizontal RHS preparation must remain a straight-line prefix')
    prefix=[]
    for operation in block.ops:
        prefix.append(operation)
        if operation is boundary:break
    if not prefix or prefix[-1] is not boundary:
        raise ValueError('horizontal RHS completion boundary is missing')
    selected=set(prefix)
    retained=[]
    for operation in prefix:
        external=any(use.operation not in selected for result in operation.results for use in result.uses)
        if external:
            if operation.name != 'llvm.mlir.constant':
                raise ValueError('horizontal RHS epoch has nonconstant continuation captures')
            retained.append(operation)
        if operation.regions or operation.successors:
            raise ValueError('horizontal RHS prefix must have closed straight-line effects')
    helper=FnBuilder([argument.type for argument in owner.entry.args])
    mapping=dict(zip(owner.entry.args,helper.entry.args,strict=True))
    for operation in prefix:
        copied=operation.clone(value_mapper=mapping)
        helper.add(copied)
    symbol=register_internal_helper(root,helper,symbol_prefix='horizontal_rhs')
    call=llvm.CallOp(symbol,*owner.entry.args)
    block.insert_op_after(call,boundary)
    for operation in reversed(prefix):
        if operation not in retained:block.erase_op(operation)
    return symbol


def try_emit_horizontal_phases(fn,input_ptr,weight_ptr,dst_ptr,*,input_shape,
                               weight_shape,attrs=None,bias=None,output_dtype='i32',
                               max_row_blocks=8,compute_pairs=8,
                               readout_epochs=False,load_epochs=False,column_intervals=None,
                               input_transport='padded'):
    """Emit all exact original outputs; refusal precedes any builder mutation."""
    if input_transport not in ('padded', 'boundary', 'device'):
        raise ValueError('horizontal phase input transport is invalid')
    prepared=plan_horizontal_phases(input_shape=input_shape,weight_shape=weight_shape,
        attrs=attrs,bias=bias is not None,output_dtype=output_dtype,
        max_row_blocks=max_row_blocks,compute_pairs=compute_pairs,
        column_intervals=column_intervals)
    if prepared is None:return None
    phased,views=prepared
    # Establish every requested admission before allocating/preparing anything.
    root=Path(__file__).resolve().parent/'data'
    facts=(root/'rtl_facts.json').read_bytes()
    ordering=json.loads((root/'COMMAND_ORDERING_SELECTION.json').read_text())
    for phase,(plan,selection),maximum in views:
        if readout_epochs:
            from gemmini_horizontal_readout import admit_horizontal_readout
            if admit_horizontal_readout(plan,selection,facts_bytes=facts,ordering_selection=ordering) is None:
                return None
        if load_epochs:
            from gemmini_horizontal_loads import admit_horizontal_loads
            if admit_horizontal_loads(plan,selection,facts_bytes=facts,ordering_selection=ordering) is None:
                return None
    import gemmini_encode as enc
    from gemmini_horizontal_conv import _emit_horizontal_plan
    from gemmini_support import scratch_buffer
    device_transport = None
    if input_transport == 'device':
        from gemmini_horizontal_transport import make_device_span_transport
        device_transport = make_device_span_transport(fn)
    enc.fence(fn)
    first_plan=views[0][1][0];w=phased.original_source.window
    size=prod(first_plan.padded_window.input_shape)
    padded=scratch_buffer(fn,size,'horizontal_phased_input')
    transport=emit_nhwc_window_panel(fn,Window2D(w.input_shape,(1,1),(1,1),(1,1),w.padding),
        input_ptr,padded,fn.const(0),prod(first_plan.padded_window.input_shape[:-1]),
        source_destination_disjoint=True,maximum_panel_bytes=size,inline_memory_spans=True,
        span_transport=device_transport)
    if device_transport is not None:
        transport.update(kind='device_accumulator_window_spans', host_copy_bytes=0,
            signed_i8_exact=True, zero_source=0,
            storage_completion='all spans complete before any phase acquires ACC')
    enc.fence(fn)
    shared=HorizontalPreparedInput(fn,input_ptr,padded,w,size,transport)
    # Irregular narrow readout expands distinct per-column descriptors. Keep
    # complete phase constant lifetimes separate without changing source effects.
    scales = attrs.get('acc_scale_by_column', attrs.get('acc_scale',
        attrs.get('output_scale', attrs.get('scale', 1.0)))) if attrs else 1.0
    narrow = str(output_dtype) in ('i8', 'int8')
    outlined = (narrow and len(set(
        DenseTiledIntegerReadout.from_source_scales(scales,
            columns=weight_shape[-1], tile_columns=1, relu=False).column_scale_bits)) > 1)
    receipts=[]
    for phase,local,maximum in views:
        owner = fn
        original_input, original_weight, original_dst, original_bias = input_ptr, weight_ptr, dst_ptr, bias
        phase_shared = shared
        rhs_boundary=[]
        if outlined:
            from merlin.llvmlower.host_builder import FnBuilder, PTR
            owner = FnBuilder([PTR]*(5 if bias is not None else 4))
            original_input, original_weight, original_dst = owner.entry.args[:3]
            original_bias = owner.entry.args[3] if bias is not None else None
            phase_padded = owner.entry.args[-1]
            phase_shared = HorizontalPreparedInput(owner, original_input, phase_padded,
                w, size, transport)
        receipts.append(_emit_horizontal_plan(owner,original_input,original_weight,original_dst,
            input_shape=input_shape,weight_shape=weight_shape,attrs=attrs,bias=original_bias,
            output_dtype=output_dtype,max_row_blocks=maximum,compute_pairs=compute_pairs,
            source_destination_disjoint=False,split_input_blocks=False,
            input_transport='padded',readout_epochs=readout_epochs,load_epochs=load_epochs,
            prepared=local,prepared_input=phase_shared,original_column_start=phase.original_column_start,
            rhs_ready_boundary=(lambda block,op: rhs_boundary.append((block,op))) if outlined else None))
        if outlined:
            from gemmini_support import register_internal_helper
            from xdsl.dialects import llvm
            if len(rhs_boundary) != 1:
                raise ValueError('horizontal phase requires exactly one original RHS readiness boundary')
            _outline_rhs_prefix(fn,owner,*rhs_boundary[0])
            symbol = register_internal_helper(fn, owner, symbol_prefix='horizontal_phase')
            arguments = (input_ptr, weight_ptr, dst_ptr) + ((bias,) if bias is not None else ()) + (padded,)
            fn.add(llvm.CallOp(symbol, *arguments))
        enc.fence(fn)  # Full phase last-use before any private operand/ACC reuse.
    return dict(route='horizontal_original_column_phases',phase_count=len(views),
        original_source_identity=phased.original_source.identity,
        original_external_columns=weight_shape[-1],
        phase_program_sha256=hashlib.sha256(repr(phased).encode()).hexdigest(),
        phases=[dict(original_columns=[p.original_column_start,p.original_column_end],
            row_blocks=p.inner.row_blocks,receipt=r) for (p,_,_),r in zip(views,receipts)],
        shared_input=shared.receipt(),physical_mapping_verified=False,
        target_execution_verified=False,timing=None,
        **({'phase_function_boundaries':True} if outlined else {}))
