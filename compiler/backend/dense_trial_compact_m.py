"""Single-M-tile source checked packing with exact original descriptor rows."""
from .dense_streaming import streaming_schedule
from . import isa


def compact_m_segmented_schedule(m,k,n,bias,column_scales=False):
    if not 1<=m<=isa.DIM:
        return None, 'compact M panel requires one primitive M tile'
    return streaming_schedule(m,k,n,bias,column_scales,geometry=(8,8),
                              compact_m_rows=True)
