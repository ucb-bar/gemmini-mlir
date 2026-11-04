"""Separate command encoding limits from the incumbent DMA-width policy.

Decoded operand capacity is necessary, but does not prove that the load engine
splits a command across transactions with the same complete copy effects. Keep
that missing premise explicit rather than granting authority from packet bits.
"""
from gemmini_dense_readout import ReadoutUnavailable


def select_transfer_tiles(selection, *, decoded_columns, tile_columns,
                          element_bytes, max_request_bytes):
    """Resolve a policy or explicit tile count without changing physical laws.

    The transaction policy is the existing checked candidate, not a claim that
    the DMA transaction maximum is also the hardware command maximum. The
    selected public evidence does not currently establish the latter.
    """
    if any(type(v) is not int or v <= 0 for v in
           (decoded_columns, tile_columns, element_bytes, max_request_bytes)):
        raise ReadoutUnavailable('dense transfer width needs positive selected physical bounds')
    decoded_tiles = decoded_columns // tile_columns
    incumbent_tiles = min(decoded_tiles, max_request_bytes // (tile_columns * element_bytes))
    if selection == 'transaction':
        return incumbent_tiles
    if selection == 'decoded':
        requested = decoded_tiles
    elif type(selection) is int and selection > 0:
        requested = selection
    else:
        raise ReadoutUnavailable('dense transfer width requires transaction, decoded, or a positive integer tile count')
    if requested > decoded_tiles:
        raise ReadoutUnavailable(
            f'dense transfer requests {requested} tiles but selected decoded MVIN2 columns '
            f'admit only {decoded_tiles} complete tiles ({decoded_columns} columns)')
    if requested > incumbent_tiles:
        raise ReadoutUnavailable(
            'dense transfer beyond one DMA transaction is unknown: selected decoded '
            'MVIN2 extent establishes address range, but no selected load-engine '
            'splitter/counter law establishes complete copy effects across requests')
    return requested
