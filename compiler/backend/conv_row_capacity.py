"""Reserve simultaneously live retained input rows before shared RHS planning."""
from . import isa

def max_column_blocks(window, expansion_budget=32, *, packed=False):
    """Only resource geometry changes; original shared panel checks remain."""
    if packed:
        from .conv_packed_rows import column_capacity
        admitted=column_capacity(window,expansion_budget)
        if admitted is not None:return admitted
    if window.stride != (1,1) or window.dilation != (1,1):
        return expansion_budget
    kh,kw=window.kernel
    batch,h,width,ci=window.input_shape
    pt,pl,pb,pr=window.padding
    if min(pt,pl,pb,pr)<0:
        return expansion_budget
    chunks=(ci+isa.DIM-1)//isa.DIM
    ring_rows=kh*(width+pl+pr)*chunks
    reserved=((ring_rows+isa.DIM-1)//isa.DIM)*isa.DIM
    rhs_rows_per_column=kh*kw*chunks*isa.DIM
    capacity=(isa.SPAD_ROWS-reserved)//rhs_rows_per_column
    # No complete-K RHS column fits: keep the original common fallback.
    if capacity<1:
        return expansion_budget
    return min(expansion_budget,capacity)
