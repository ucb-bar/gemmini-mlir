"""Source-view admission for bounded transposed-RHS contractions."""
import json
import os
from pathlib import Path

from xdsl.dialects.builtin import TensorType, NoneAttr

QK_WORKSPACE_BYTES=4096


def selected_schedule(explicit=None):
    settings=Path(__file__).parent/'DENSE_SCHEDULE.json'
    selected=explicit or os.environ.get('QK_SCHEDULE') or (
        json.loads(settings.read_text()).get('qk_schedule','cell') if settings.exists() else 'cell')
    if selected not in ('cell','packed','tiled'):
        raise ValueError('QK_SCHEDULE must be cell, packed or tiled')
    return selected


def transposed_rhs_views(block, is_device):
    """Admit only immutable rank-two swaps with one contraction RHS observer.

    The consumer still proves its complete scalar recurrence and affine maps.
    No returned, shared or unknown-observer transpose is elided here.
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
        if len(uses)!=1 or uses[0].index!=1 or not is_device(uses[0].operation):continue
        consumer=uses[0].operation
        if len(consumer.operands[0].type.get_shape())!=2 or len(consumer.results[0].type.get_shape())!=2:continue
        result[consumer]=op
    return result
