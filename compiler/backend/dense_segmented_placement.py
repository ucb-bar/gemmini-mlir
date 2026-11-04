"""Capacity-derived placement alternatives for two original RHS generations.

The original RHS view owns both windows. Separating them across its two halves
changes only local addresses; the shared consumer independently reconstructs
actual source rectangles, nonaliasing, epochs and all original lifetime bounds.
The name does not assert a physical bank map or permit more overlap.
"""
def choose_segmented_placement(problem,block,storage,selection):
    if selection not in ('compact','window_halves'):
        raise ValueError('unknown segmented operand placement')
    slot_rows=block.k*block.n*problem.tile_k
    offset=storage.rhs.capacity//2
    offset-=offset%storage.rhs.alignment
    legal=(offset>=slot_rows and offset+slot_rows<=storage.rhs.capacity)
    use=selection=='window_halves' and legal
    actual=offset if use else slot_rows
    return ((0,0),(0,actual)),dict(selection=selection,
        selected='window_halves' if use else 'compact',
        original_rhs_view=dict(base=storage.rhs.base,rows=storage.rhs.capacity,
            alignment=storage.rhs.alignment),rhs_slot_rows=slot_rows,
        second_rhs_row_offset=actual,half_view_placement_fits=legal,
        performance_basis='hypothesis only; paired complete-call cycles required',
        legality='original shared source/storage/epoch consumer; no new target law')
