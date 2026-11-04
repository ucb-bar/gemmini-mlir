"""Source-view admission for bounded transposed-RHS contractions."""
import json
import os
from pathlib import Path

from xdsl.dialects.builtin import TensorType, NoneAttr

QK_WORKSPACE_BYTES=4096
SHARED_RHS_VIEWS_ENABLED = False


def selected_schedule(explicit=None):
    settings=Path(__file__).parent/'DENSE_SCHEDULE.json'
    selected=explicit or os.environ.get('QK_SCHEDULE') or (
        json.loads(settings.read_text()).get('qk_schedule','cell') if settings.exists() else 'cell')
    if selected not in ('cell','packed','tiled'):
        raise ValueError('QK_SCHEDULE must be cell, packed or tiled')
    return selected


def transposed_rhs_views(block, is_device):
    """Admit immutable rank-two swaps observed only by checked RHS products.

    The consumer still proves its complete scalar recurrence and affine maps.
    Returned or unknown-observer transposes retain ordinary materialization.
    Multiple product observers require the explicit default-off alternative.
    """
    if selected_schedule()!='tiled':return {}
    result={}
    for op in block.ops:
        if op.name!='linalg.transpose' or len(op.results)!=1 or not op.operands:continue
        source=op.operands[0];view=op.results[0]
        if any(not isinstance(v.type,TensorType) or len(v.type.get_shape())!=2
               or str(v.type.element_type)!='i8' or not isinstance(v.type.encoding,NoneAttr)
               or any(int(d)<=0 for d in v.type.get_shape()) for v in (source,view)):continue
        if int(source.type.get_shape()[1])>QK_WORKSPACE_BYTES:continue
        permutation=op.properties.get('permutation',op.attributes.get('permutation'))
        if permutation is None or tuple(permutation.get_values())!=(1,0):continue
        if tuple(source.type.get_shape())[::-1]!=tuple(view.type.get_shape()):continue
        uses=list(view.uses)
        if not uses or any(use.index!=1 or not is_device(use.operation) for use in uses):continue
        consumers=tuple(use.operation for use in uses)
        if any(len(consumer.operands[0].type.get_shape())!=2
               or len(consumer.results[0].type.get_shape())!=2 for consumer in consumers):continue
        if len(uses)>1:
            if not SHARED_RHS_VIEWS_ENABLED:continue
            # Multi-observer admission proves every original scalar product and
            # seed before omitting the single physical preparation allocation.
            # Each original consumer still executes at its original source site.
            from merlin.transforms.dense_row_batching_check import (
                geometry, boundary_attributes, RowBatchingUnavailable)
            if not boundary_attributes(op):continue
            try:
                contracts=[geometry(consumer,index_bits=64) for consumer in consumers]
            except (RowBatchingUnavailable,AttributeError):continue
            if any(dtypes!=('i8','i8','i32') for _,dtypes in contracts):continue
        result.update({consumer:op for consumer in consumers})
    return result


def complete_rhs_views(views):
    """Keep an elided transpose only while every original observer is selected.

    Run after consumer ownership filters (epilogue/preparation grouping), before
    allocation. A partial selection must retain the original materialized view
    for the excluded observer; all consumers then use ordinary lowering.
    """
    complete={transpose for transpose in set(views.values())
        if all(use.index==1 and views.get(use.operation) is transpose
               for use in transpose.results[0].uses)}
    return {consumer:transpose for consumer,transpose in views.items()
            if transpose in complete}
